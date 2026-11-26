#!/usr/bin/env python3
"""Reopen the paired geometry with live MuJoCo RGB and the frozen ACT worker."""

import argparse
from base64 import b64decode
from hashlib import sha256
from pathlib import Path
from time import monotonic

import numpy as np

from check_precontact_teacher import FIT, GRASP, PILOT, original_command_path
from smart_swarm.arm_executor import ArmExecutor
from smart_swarm.cli import settings
from smart_swarm.episode import atomic, json_bytes, load, sha
from smart_swarm.evaluate_motor import Worker, action_request
from smart_swarm.motor_diagnostics import read_teacher, reset_world
from smart_swarm.motor_input import CLOCKED_PROFILE, conditioned_top
from smart_swarm.precontact_teacher import PRECONTACT, STOCK
from smart_swarm.rendering import configure_renderer
from smart_swarm.research_jobs import admitted

PHYSICS = PILOT / "act-precontact-teacher-physics.json"
CHECKPOINT = PILOT / "act-clock-closure-window.json"
WALL_CAP_S = 180
OBSERVATIONS = {"stock": (GRASP, 132), "raised_and_retreated": (GRASP,)}


def qualified_sources():
    if not PHYSICS.is_file():
        raise ValueError("The matched five-world contact-free geometry receipt is pending")
    receipt = load(PHYSICS)
    if not receipt["all_five_passed"]:
        raise ValueError("The same five worlds must pass the contact-free and placement geometry checks")
    fit = load(FIT)
    dataset, checkpoint = (Path(fit[name]) for name in ("dataset", "checkpoint"))
    lineage = dataset / "train/smart-swarm-lineage.json"
    if sha(lineage) != receipt["training_lineage_sha256"]:
        raise ValueError("The original teacher fitting cohort has changed")
    saved = load(CHECKPOINT)
    if any(sha(checkpoint / name) != checksum for name, checksum in saved["artifact_sha256"].items()):
        raise ValueError("The frozen ACT checkpoint or processor changed")
    sources = {case["seed"]: case for case in receipt["sources"]}
    return checkpoint, dataset, load(lineage)["episodes"], sources


def stock_input_parity(raw, marker, issue, request):
    with np.load(raw / "motor.npz", allow_pickle=False) as archive:
        expected = [archive["wrist"][issue],
                    conditioned_top(archive["top"][issue], marker, issue_index=issue)]
        if not np.allclose(request["state"], archive["state"][issue], atol=1e-5, rtol=0.):
            raise ValueError("Original ACT sampled joint state changed on replay")
    return [b64decode(text) == pixels.tobytes() for text, pixels in zip(request["images"], expected, strict=True)]


def captured_requests(world, commands, marker, indices):
    engine = ArmExecutor(world, "aligned-sample-v2")
    engine.prime()
    result = []
    for issue, action in enumerate(commands[:max(indices) + 1]):
        if issue in indices:
            if issue != engine.observation_index() or issue % 6:
                raise ValueError("The live motor sample is outside an ACT chunk refresh")
            request = action_request(world, CLOCKED_PROFILE, marker, issue_index=issue)
            result.append((issue, world.ns, request))
        engine.submit(action)
    return result


def one_variant(world, snapshot, pickup, marker, raw, name, reference, worker):
    world.restore(snapshot)
    pose = STOCK if name == "stock" else PRECONTACT
    trajectory = original_command_path(world, pickup, pose, issues=reference["matched_actuator_issues"])
    if sha256(json_bytes(trajectory.commands)).hexdigest() != reference["request_sha256"]:
        raise ValueError("The geometry does not reproduce the already checked trajectory")
    world.restore(snapshot)
    frames = captured_requests(world, trajectory.commands, marker, OBSERVATIONS[name])
    results = []
    for issue, at, request in frames:
        worker.request({"type": "reset"})
        prediction = worker.request(request)["action"]
        parity = stock_input_parity(raw, marker, issue, request) if name == "stock" else None
        if parity is not None and not all(parity):
            raise ValueError("Original wrist or top RGB bytes changed on the stock renderer")
        within_limits = world.arm_from_mj(world.arm_to_mj(prediction)).tolist()
        results.append({"issue": issue, "at_ns": at, "action_native": prediction, "state_native": request["state"],
                        "would_apply_after_executor_limits_native": within_limits,
                        "encoded_camera_sha256": [sha256(b64decode(text)).hexdigest() for text in request["images"]],
                        "original_stock_inputs_match": parity})
    return results


def one_source(assets, entry, geometry, worker):
    path = Path(entry["raw_path"])
    manifest, _, _ = read_teacher(path)
    if manifest["sha256"] != geometry["source_sha256"]:
        raise ValueError("The optical rehearsal lost its original matched raw source")
    world = reset_world(assets, manifest)
    world.deadline = min(world.deadline, worker.deadline)
    snapshot = world.snapshot()
    try:
        pickup = manifest["physics"]["pickup_xy_m"]
        marker = manifest.get("target_marker", "cyan")
        traces = {name: one_variant(world, snapshot, pickup, marker, path, name, reference, worker)
                  for name, reference in geometry["implementations"].items()}
    finally:
        world.close()
    return {"seed": manifest["seed"], "source_sha256": manifest["sha256"], "teacher_inputs": traces}


def run(config, output, qualified):
    checkpoint, dataset, entries, sources = qualified
    log = output.with_suffix(".worker.log")
    cases = []
    with admitted(config, "smart-swarm-frozen-act-precontact-rgb") as (grant, heartbeat):
        configure_renderer(config)
        worker = Worker(checkpoint, dataset / "validation", log, deadline=monotonic() + WALL_CAP_S)
        try:
            loaded = worker.recv()
            for entry in entries:
                heartbeat()
                seed = int(entry["source_world_episode_id"].rsplit("-", 1)[1])
                cases.append(one_source(config["asset_root"], entry, sources[seed], worker))
        finally:
            worker.close()
    return {"scope": "Frozen ACT first action on matched simulation teacher observations",
            "geometry_receipt_sha256": sha(PHYSICS), "checkpoint_sha256": sha(checkpoint / "model.safetensors"),
            "reservation": grant, "worker_reload": loaded, "cases": cases,
            "ordered_cameras": ["so101-1/wrist", "so101-1/top"], "input_profile": CLOCKED_PROFILE,
            "optimizer_updates": 0, "learned_actions_executed": 0, "wall_cap_seconds": WALL_CAP_S}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--output", type=Path, default=PILOT / "act-precontact-teacher-rgb.json")
    args = parser.parse_args()
    if args.output.exists() and not args.check_only:
        parser.error("Choose a new precontact RGB result path")
    try:
        qualified = qualified_sources()
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    if args.check_only:
        print("Five matched contact-valid geometries and the original frozen checkpoint are available.")
    else:
        atomic(args.output, run(settings(), args.output, qualified))
        print(args.output)


if __name__ == "__main__":
    main()
