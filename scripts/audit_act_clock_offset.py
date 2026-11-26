"""Vary only the issue clock on the archived, unmodified ACT teacher camera inputs."""

import argparse
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

import numpy as np

from probe_precontact_act import (
    CLOCK_OFFSETS, FIT, OPTICAL_FAILURE, PILOT, STOCK_MISMATCH, WALL_CAP_S, altered_clock, recorded_charges,
)
from smart_swarm.cli import ROOT, settings
from smart_swarm.episode import atomic, json_bytes, load, sha
from smart_swarm.motor_input import CLOCKED_PROFILE
from smart_swarm.research_jobs import admitted, execute
from smart_swarm.training import offline


CLOSURE = PILOT / "act-clock-closure-window.json"
ISSUES = (126, 132)


def original_material():
    fit, reference = load(FIT), load(CLOSURE)
    if sha(FIT) != reference["fit_receipt_sha256"] or reference["input_profile"] != CLOCKED_PROFILE:
        raise ValueError("The original ACT fitting receipt or archived closure scope changed")
    checkpoint, dataset = (Path(fit[name]) for name in ("checkpoint", "dataset"))
    if any(sha(checkpoint / name) != checksum for name, checksum in reference["artifact_sha256"].items()):
        raise ValueError("The frozen ACT checkpoint or processor changed")
    previous = {row["seed"]: row for row in reference["sources"]}
    if len(previous) != 7:
        raise ValueError("The original archive has exactly seven independent development starts")
    return checkpoint, dataset, reference, previous


def source_inputs(base, entry, official, archived):
    from export_lerobot import check_row, load_episode, make_frame

    raw = Path(entry["raw_path"])
    meta = load(raw / "manifest.json")
    if (entry["source_sha256"] != meta["sha256"] or archived["source_sha256"] != meta["sha256"]
            or archived["target_marker"] != entry["target_marker"]
            or archived["split"] != base.name or not meta["physics"]["released_stable_containment"]):
        raise ValueError("An archive source differs from the original contact-valid teacher trace")
    arrays = load_episode(raw, meta, aligned=True)
    origin = int(official.meta.episodes[entry["dataset_episode_index"]]["dataset_from_index"])
    frames = []
    for issue in ISSUES:
        frame = make_frame(arrays, meta, issue, CLOCKED_PROFILE)
        row = official[origin + issue]
        if row["frame_index"].item() != issue or row["episode_index"].item() != entry["dataset_episode_index"]:
            raise ValueError("The matched archive row changed its episode or issue index")
        check_row(row, frame)
        frames.append(frame)
    return meta, frames


def score_inputs(frames, archived, native):
    from audit_act_closure import CLOSE, MOVING_CLOSED, PARITY_LIMIT, WINDOW, decode, payload

    requests = [altered_clock(payload(frame), issue, offset)
                for issue, frame in zip(ISSUES, frames, strict=True) for offset in CLOCK_OFFSETS]
    chunks = decode(requests, *native).reshape(len(ISSUES), len(CLOCK_OFFSETS), 24, 6)
    scope = []
    for issue, choices in zip(ISSUES, chunks, strict=True):
        center = choices[CLOCK_OFFSETS.index(0), :6]
        original = archived["teacher_forced"]["native_gripper_targets"][issue - WINDOW.start:issue - WINDOW.start + 6]
        error = float(np.max(np.abs(center[:, -1] - original)))
        if error > PARITY_LIMIT:
            raise ValueError(f"The original ACT queue outputs changed at issue {issue}")
        cases = []
        for offset, result in zip(CLOCK_OFFSETS, choices, strict=True):
            first = result[:6]
            cases.append({"offset_issues": offset, "visible_issue": issue + offset,
                          "first_six_candidates_native": first.tolist(),
                          "first_gripper_percent": float(first[0, -1]),
                          "close_band_count": int((first[:, -1] <= CLOSE).sum()),
                          "below_motion_close_band_count": int((first[:, -1] <= MOVING_CLOSED).sum())})
        scope.append({"issue": issue, "archived_queue_gripper_residual_pp": error, "clock_conditions": cases})
    return scope


def score_split(dataset, split, reference, previous, native, expected):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    base = dataset / split
    binding = base / "smart-swarm-lineage.json"
    if sha(binding) != reference["teacher_lineage_sha256_by_split"][split]:
        raise ValueError("The archive split differs from the exact earlier ACT closure run")
    entries = load(binding)["episodes"]
    official = LeRobotDataset(f"smart-swarm/{split}", root=base, delta_timestamps={"action": [0.]})
    if len(entries) != expected or official.num_episodes != expected:
        raise ValueError("The fixed original archive split is incomplete")
    results = []
    for entry in entries:
        seed = int(entry["source_world_episode_id"].rsplit("-", 1)[1])
        meta, frames = source_inputs(base, entry, official, previous[seed])
        results.append({"seed": seed, "split": split, "source_sha256": meta["sha256"],
                        "samples": score_inputs(frames, previous[seed], native)})
    return results


def report():
    from importlib.metadata import version
    import torch
    from policy_worker import load_policy

    torch.set_num_threads(1)
    checkpoint, dataset, reference, previous = original_material()
    config, model, pre, post = load_policy(checkpoint)
    if (version("lerobot") != "0.6.1" or config.type != "act" or config.chunk_size != 24
            or config.n_action_steps != 6 or config.temporal_ensemble_coeff is not None):
        raise ValueError("The ACT runtime or original production chunk configuration changed")
    sources = [row for split, expected in (("train", 5), ("validation", 2))
               for row in score_split(dataset, split, reference, previous, (model, pre, post), expected)]
    return {"scope": "Counterfactual issue clock on matched archived training and development observations",
            "fit_receipt_sha256": sha(FIT), "previous_closure_sha256": sha(CLOSURE),
            "checkpoint_sha256": reference["artifact_sha256"]["model.safetensors"],
            "local_control_rate_hz": 30, "varied_clock_offsets_issues": list(CLOCK_OFFSETS),
            "physical_inputs": "Exact archived SO101 RGB, robot state and task with only the clock pixels varied.",
            "optimizer_updates": 0, "learned_actions_executed": 0, "simulator_steps": 0, "sources": sources}


def remaining_allowance():
    failure, probe = load(OPTICAL_FAILURE), load(STOCK_MISMATCH)
    if failure["original_exit_code"] != 1 or probe["learned_actions_executed"] != 0:
        raise ValueError("The preceding failures and renderer-only checkpoint must be present")
    charges = recorded_charges()
    return charges, WALL_CAP_S - sum(charges.values())


def bounded(output):
    charges, wall = remaining_allowance()
    if wall <= 0.:
        raise ValueError("The original cumulative diagnostic allowance is exhausted")
    exe = load(ROOT / "configs/training.json")["runtime"]
    with admitted(settings(), "smart-swarm-act-archived-clock-offset", archive_only=True) as (grant, heartbeat):
        with TemporaryDirectory(prefix="smart-swarm-clock-offset-") as directory:
            temporary = Path(directory) / "clock-offset.json"
            command = [exe, str(Path(__file__).resolve()), "--output", str(temporary), "--worker"]
            log = output.with_suffix(".log")
            start = monotonic()
            with log.open("x") as stream:
                status = execute(command, offline(), stream, start + wall, heartbeat)
            if status != 0:
                raise RuntimeError(f"The archived ACT clock replay failed. See {log}")
            elapsed = monotonic() - start
            return {**load(temporary), "reservation": grant, "previous_wall_charges_s": charges,
                    "original_limit_s": WALL_CAP_S, "current_replay_elapsed_s": elapsed,
                    "remaining_wall_seconds": wall - elapsed, "renderer_mismatch_receipt_sha256": sha(STOCK_MISMATCH)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".log").exists():
        parser.error("Choose a new immutable clock audit result")
    atomic(args.output, report() if args.worker else bounded(args.output))
    print(args.output)


if __name__ == "__main__":
    main()
