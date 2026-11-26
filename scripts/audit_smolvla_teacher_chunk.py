"""Use the selected SmolVLA checkpoint on archived validation teacher observations."""

import argparse
from base64 import b64encode
from hashlib import sha256
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import numpy as np

from export_lerobot import KEYS, check_row, load_episode, make_frame
from smart_swarm.cli import settings
from smart_swarm.contract import ProtocolError
from smart_swarm.episode import atomic, json_bytes, load, sha
from smart_swarm.motor_input import CLOCKED_PROFILE
from smart_swarm.motor_visual_profile import match_visual
from smart_swarm.research_jobs import admitted, execute
from smart_swarm.training import ROOT, offline

CHUNK = 50
REFRESHES = (0, CHUNK)
WALL_SECONDS = 150
TRACE_TOLERANCE = 1e-4


def source_inputs(dataset, lineage, evaluation):
    root = Path(lineage["raw_path"])
    manifest = load(root / "manifest.json")
    if (lineage["source_split"] != "validation" or lineage["source_sha256"] != manifest["sha256"]
            or manifest["seed"] != evaluation["seed"] or manifest["target_marker"] != lineage["target_marker"]):
        raise ValueError("The archived validation episode and checkpoint trace have different sources")
    raw = load_episode(root, manifest, aligned=True)
    episode_index = lineage["dataset_episode_index"]
    start = int(dataset.meta.episodes[episode_index]["dataset_from_index"])
    windows = []
    for issue in REFRESHES:
        frame = make_frame(raw, manifest, issue, CLOCKED_PROFILE)
        row = dataset[start + issue]
        if int(row["episode_index"]) != episode_index or int(row["frame_index"]) != issue:
            raise ValueError("The official validation row has a different episode or issue index")
        check_row(row, frame)
        views = [frame[key].tobytes() for key in KEYS.values()]
        payload = {"images": [b64encode(rgb).decode("ascii") for rgb in views],
                   "state": frame["observation.state"].tolist(), "task": frame["task"]}
        windows.append({"issue": issue, "dataset_index": start + issue, "ordered_rgb_sha256": [sha256(rgb).hexdigest() for rgb in views],
                        "payload_sha256": sha256(json_bytes(payload)).hexdigest(), "payload": payload})
    return {"seed": manifest["seed"], "marker": lineage["target_marker"], "teacher_source_sha256": manifest["sha256"],
            "windows": windows, "requested": raw["requested"], "feedback": evaluation["feedback"]}


def prepare_inputs(validation):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    job = validation.parents[2]
    weights = job / "training/checkpoints" / validation.parent.name / "pretrained_model"
    binding = load(weights / "smart-swarm-input.json")
    root = Path(binding["dataset"]) / "validation"
    lineage = load(root / "smart-swarm-lineage.json")
    match_visual(binding, lineage)
    evaluated = load(validation)
    if (lineage["input_profile"] != CLOCKED_PROFILE or binding["input_profile"] != CLOCKED_PROFILE
            or evaluated["policy"] != "smolvla" or evaluated["evaluation_role"] != "validation"
            or evaluated["action_transport"] != "verified-smolvla-chunk" or len(evaluated["resets"]) != 20):
        raise ValueError("Use the complete clocked SmolVLA validation receipt and matching checkpoint")
    chosen = lineage["episodes"][:2]
    if {row["target_marker"] for row in chosen} != {"cyan", "violet"}:
        raise ValueError("The two matched development inputs must cover both requested markers")
    dataset = LeRobotDataset("smart-swarm/validation", root=root, delta_timestamps={"action": [0.]})
    cases = [source_inputs(dataset, source, value) for source, value in zip(chosen, evaluated["resets"][:2], strict=True)]
    return weights, cases


def case_metadata(case):
    return {**{key: case[key] for key in ("seed", "marker", "teacher_source_sha256")},
            "windows": [{key: value for key, value in row.items() if key != "payload"} for row in case["windows"]]}


def compare_chunk(case, issue, predicted):
    expected = case["requested"][issue:issue + CHUNK]
    samples = [sample for sample in case["feedback"] if issue <= sample["tick"] - 1 < issue + CHUNK]
    if len(samples) != 3 or predicted.shape != (CHUNK, 6):
        raise ValueError("The selected saved validation chunk must have three aligned command samples")
    rows = []
    for sample in samples:
        index = sample["tick"] - 1
        slot = index - issue
        if not np.array_equal(expected[slot], np.asarray(sample["teacher_requested_at_index"], dtype=np.float32)):
            raise ValueError("A retained closed-loop sample is misaligned with the teacher command")
        rows.append({"issue": index, "slot": slot, "teacher_action": expected[slot].tolist(),
                     "policy_with_teacher_input": predicted[slot].tolist(), "policy_with_feedback": sample["requested"],
                     "max_abs_difference_from_closed_loop": float(np.max(np.abs(predicted[slot] - sample["requested"])))})
    return {"refresh_issue": issue, "native_mae_to_teacher": np.mean(np.abs(predicted - expected), axis=0).tolist(),
            "sampled_actions": rows}


def observed_chunks(weights, cases):
    import torch
    from policy_worker import example_batch, load_policy, smolvla_chunk

    config, model, pre, post = load_policy(weights)
    if config.type != "smolvla" or config.chunk_size != CHUNK or config.n_action_steps != CHUNK:
        raise ValueError("The archived checkpoint does not use the verified 50-action transport")
    report = []
    for case in cases:
        torch.manual_seed(case["seed"])
        model.reset()
        runs = []
        for window in case["windows"]:
            prediction = np.asarray(smolvla_chunk(example_batch(window["payload"]), model, pre, post))
            runs.append(compare_chunk(case, window["issue"], prediction))
        startup_residual = max(row["max_abs_difference_from_closed_loop"] for row in runs[0]["sampled_actions"])
        report.append({**case_metadata(case), "chunks": runs, "startup_max_abs_difference": startup_residual,
                       "matched_original_initial_chunk": startup_residual <= TRACE_TOLERANCE})
    return report


def draft(validation, inputs_only):
    started = time.monotonic()
    weights, cases = prepare_inputs(validation)
    rows = [case_metadata(case) for case in cases] if inputs_only else observed_chunks(weights, cases)
    return {"scope": "two matched archived development resets on teacher observations", "checkpoint": str(weights),
            "validation_sha256": sha(validation), "input_profile": CLOCKED_PROFILE, "inputs_only": inputs_only,
            "optimizer_updates": 0, "fresh_resets_read": 0, "teacher_windows": rows,
            "elapsed_s": time.monotonic() - started}


def bounded_replay(validation, target):
    config = settings()
    runtime = load(ROOT / "configs/training.json")["runtime"]
    with admitted(config, "smart-swarm-smolvla-teacher-chunks") as (grant, heartbeat):
        with TemporaryDirectory(prefix="smart-swarm-teacher-chunks-") as directory:
            receipt = Path(directory) / "worker.json"
            argv = [runtime, str(Path(__file__).resolve()), "--validation", str(validation), "--output", str(receipt), "--worker"]
            log = target.with_suffix(".log")
            with log.open("x") as stream:
                status = execute(argv, offline(), stream, time.monotonic() + WALL_SECONDS, heartbeat)
            if status:
                raise RuntimeError(f"Local policy worker exited {status}. See {log}")
            return {**load(receipt), "compute_reservation": grant, "wall_limit_s": WALL_SECONDS,
                    "startup_parity_tolerance_native_units": TRACE_TOLERANCE}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-inputs", action="store_true", help="Verify the archived inputs without loading the checkpoint")
    mode.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new immutable audit output")
    validation = args.validation.resolve()
    try:
        if args.worker or args.check_inputs:
            os.environ.update(offline())
            report = draft(validation, args.check_inputs)
        else:
            report = bounded_replay(validation, args.output)
    except (OSError, ValueError, RuntimeError, ProtocolError) as error:
        parser.exit(2, f"Teacher chunk audit stopped. {error}\n")
    atomic(args.output, report)
    print(args.output)


if __name__ == "__main__":
    main()
