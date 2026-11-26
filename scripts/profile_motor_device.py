"""Time the deployed checkpoint worker on matched training observations without motion."""

import argparse
from base64 import b64encode
from contextlib import closing
from hashlib import sha256
from pathlib import Path
from time import monotonic, monotonic_ns

import numpy as np

from export_lerobot import KEYS, load_episode, make_frame
from smart_swarm.cli import ROOT, settings
from smart_swarm.contract import ProtocolError
from smart_swarm.episode import atomic, json_bytes, load, sha
from smart_swarm.evaluate_motor import Worker, sha256_stream
from smart_swarm.motor_input import CLOCKED_PROFILE
from smart_swarm.research_jobs import admitted


PILOT = ROOT / "evidence/learning-pilot-v3"
INVENTORY = PILOT / "training-inventory-final.json"
JOBS = {"act": "act-clock-fit-cdfa3cdae4e0", "smolvla": "smolvla-pipeline-d9510148a574"}
WALL_LIMIT_S = 120


def original_material(model):
    job, = (row for row in load(INVENTORY)["jobs"] if row["job"] == JOBS[model])
    checkpoint = Path(job["checkpoint"])
    lineage = job["dataset"]["splits"]["train"]["lineage"]
    config = load(checkpoint / "config.json")
    binding = load(checkpoint / "smart-swarm-input.json")
    if (job["model"] != model or config["type"] != model or binding["input_profile"] != CLOCKED_PROFILE
            or Path(binding["dataset"]) / "train/smart-swarm-lineage.json" != Path(lineage["path"])
            or sha(Path(lineage["path"])) != lineage["sha256"] or any(
                sha(Path(row["path"])) != row["sha256"] for row in job["configuration_and_processors"].values())):
        raise ValueError("The timed checkpoint, processors or training lineage differs from its original identity")
    with (checkpoint / "model.safetensors").open("rb") as stream:
        if sha256_stream(stream) != job["weights"]["sha256"]:
            raise ValueError("The timed checkpoint weights differ from their archived digest")
    return job, checkpoint, Path(lineage["path"]), config


def timing_inputs(lineage_path, config):
    lineage = load(lineage_path)
    entry = lineage["episodes"][0]
    root = Path(entry["raw_path"])
    raw = load(root / "manifest.json")
    if (lineage["input_profile"] != CLOCKED_PROFILE or raw["sha256"] != entry["source_sha256"]
            or raw["fps"] != 30 or entry["dataset_episode_index"] != 0 or entry["source_split"] != "train"):
        raise ValueError("The archived timing input is outside the original clocked training split")
    arrays = load_episode(root, raw, aligned=True)
    amount = config["n_action_steps"]
    issues = (0, amount) if config["type"] == "smolvla" else tuple(range(amount + 1))
    frames = [make_frame(arrays, raw, issue, CLOCKED_PROFILE) for issue in issues]
    return raw, list(zip(issues, frames, strict=True))


def packet(issue, frame):
    images = [frame[key].tobytes() for key in KEYS.values()]
    payload = {"images": [b64encode(value).decode("ascii") for value in images],
               "state": frame["observation.state"].tolist(), "task": frame["task"]}
    return {"issue": issue, "dataset_index": issue, "episode_index": 0, "payload": payload,
            "ordered_rgb_sha256": [sha256(value).hexdigest() for value in images]}


def time_action(worker, issue, frame, profile, model):
    captured = monotonic_ns()
    request = packet(issue, frame)
    payload = request.pop("payload")
    kind, length = ("smolvla_chunk", profile["actions_per_chunk"]) if model == "smolvla" else ("action", 1)
    sent = monotonic_ns()
    result = worker.request({"type": kind, **payload})
    received = monotonic_ns()
    actions = result["actions"] if model == "smolvla" else [result["action"]]
    if np.shape(actions) != (length, 6) or not np.isfinite(actions).all():
        raise ValueError("The timing worker did not produce the native six-joint action format")
    age = received - captured
    deadline = round(length * 1e9 / 30)
    return {**request, "payload_sha256": sha256(json_bytes(payload)).hexdigest(),
            "fresh_chunk": issue % profile["actions_per_chunk"] == 0,
            "replay_release_monotonic_ns": captured, "worker_send_monotonic_ns": sent,
            "candidate_available_monotonic_ns": received, "available_observation_age_ns": age,
            "preprocessor_host_ns": result["worker_host_preprocessor_ns"], "worker_request_ns": result["worker_request_ns"],
            "candidate_actions": length, "nominal_coverage_ns": deadline, "within_coverage": age <= deadline,
            "first_candidate_native_action": actions[0]}


def probe(report, checkpoint, training, raw, frames, config, log, grant, heartbeat, deadline):
    begin = monotonic_ns()
    model, device = config["type"], report["requested_device"]
    with closing(Worker(checkpoint, training, log, deadline=deadline, device=device, reservation=grant,
                        smolvla_chunks=model == "smolvla", timing_startup=True)) as worker:
        report["worker_started"] = True
        profile = worker.recv(timeout=WALL_LIMIT_S)
        startup_ns = monotonic_ns() - begin
        if (profile["device"].partition(":")[0] != device or profile["policy"] != model
                or profile["actions_per_chunk"] != config["n_action_steps"] or profile["chunk_size"] != config["chunk_size"]
                or profile["diffusion_steps"] != config.get("num_steps")
                or profile["resize_with_padding"] != config.get("resize_imgs_with_padding")):
            raise ValueError("The policy worker changed the frozen action or image profile")
        proof = worker.request({"type": "timing_inputs", "windows": [packet(issue, frame) for issue, frame in frames]})
        if proof["verified_training_rows"] != len(frames) or not worker.request({"type": "reset", "seed": raw["seed"]})["reset"]:
            raise ValueError("The timing worker did not verify inputs and reset the original action queue")
        timings = []
        for issue, frame in frames:
            heartbeat()
            timings.append(time_action(worker, issue, frame, profile, model))
        memory = worker.request({"type": "timing_stats"})
    return {"status": "profiled", "runtime": profile, "worker_launch_to_startup_ns": startup_ns,
            "source_world": raw["world_episode_id"], "source_sha256": raw["sha256"],
            "verified_training_rows": proof["verified_training_rows"], "action_requests": timings, **memory}


def profile(model, device, destination):
    report = {"scope": "Read-only production policy pipe timing on a training recording", "model": model,
              "requested_device": device, "wall_limit_s": WALL_LIMIT_S, "optimizer_updates": 0,
              "worker_started": False, "robot_commands_sent": 0, "closed_loop_trials": 0}
    started = monotonic()
    admitted_here = False
    try:
        with admitted(settings(), f"smart-swarm-{model}-{device}-pipe-timing",
                      archive_only=device == "cpu", device=device) as (grant, heartbeat):
            admitted_here = True
            job, checkpoint, lineage, config = original_material(model)
            raw, frames = timing_inputs(lineage, config)
            log = destination.with_suffix(".log")
            report.update(checkpoint=job["weights"], training_lineage_sha256=sha(lineage),
                          input_profile=CLOCKED_PROFILE, camera_order=list(KEYS),
                          reservation=grant, worker_log=str(log))
            report.update(probe(report, checkpoint, lineage.parent, raw, frames, config, log,
                                grant, heartbeat, started + WALL_LIMIT_S))
    except (OSError, RuntimeError, ValueError, ProtocolError) as error:
        report.update(status="COMPUTE_LIMITED" if device == "cuda" and not admitted_here else "probe_failed",
                      reason=str(error))
    return {**report, "elapsed_s": monotonic() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=JOBS, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".log").exists():
        parser.error("Choose a new immutable timing receipt")
    report = profile(args.model, args.device, args.output)
    atomic(args.output, report)
    print(f"{report['status']} {args.output}")
    raise SystemExit(0 if report["status"] == "profiled" else 2)


if __name__ == "__main__":
    main()
