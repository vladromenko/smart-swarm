#!/usr/bin/env python3
"""Replay the ACT closure window on preserved teacher inputs and perturb only gripper state."""

import argparse
from base64 import b64encode
from importlib.metadata import version
from itertools import product
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from export_lerobot import KEYS, check_row, load_episode, make_frame
from policy_worker import example_batch, load_policy, step
from smart_swarm.episode import atomic, load, sha
from smart_swarm.motor_input import CLOCKED_PROFILE, conditioned_top, instruction
from smart_swarm.motor_visual_profile import match_visual
from smart_swarm.training import ROOT

PILOT = ROOT / "evidence/learning-pilot-v3"
FIT = PILOT / "act-clock-fit.json"
REFERENCE = PILOT / "act-clock-grasp-behavior.json"
WINDOW = range(108, 204)
SWAP_ISSUES = (119, 134, 149, 164, 179)
CUE_ISSUES = (126, 132)
CUE_FIELDS = ("wrist", "top", "state", "clock")
CLOSE = 5.
MOVING_CLOSED = 25.
PARITY_LIMIT = 0.0002


def payload(frame, measured_grip=None):
    state = frame["observation.state"].copy()
    if measured_grip is not None:
        state[-1] = measured_grip
    return {"state": state.tolist(), "task": frame["task"],
            "images": [b64encode(frame[name].tobytes()).decode("ascii") for name in KEYS.values()]}


def prepare(requests, pre):
    rows = [example_batch(request) for request in requests]
    return pre({name: torch.stack([row[name] for row in rows]) if name != "task"
                else [row[name] for row in rows] for name in rows[0]})


@torch.inference_mode()
def decode(requests, model, pre, post):
    batches = []
    for offset in range(0, len(requests), 4):
        work = requests[offset:offset + 4]
        prepared = prepare(work, pre)
        raw = model.predict_action_chunk(prepared)
        chunks = torch.stack([post(raw[:, index]).cpu() for index in range(raw.shape[1])], dim=1)
        batches.extend(chunks.numpy())
    return np.stack(batches)


def queue_error(requests, emitted, model, pre, post, offset, steps):
    model.reset()
    actual = np.asarray([step(request, model, pre, post) for request in requests[offset:offset + steps]])
    model.reset()
    expected = emitted[offset:offset + steps]
    difference = float(np.max(np.abs(actual - expected)))
    if difference > PARITY_LIMIT:
        raise ValueError(f"Actual ACT select_action queue diverged from the compared slots by {difference}")
    return difference


def summarize_rows(teacher, emitted, chunks, refresh, steps):
    expected = teacher["requested"][WINDOW.start:WINDOW.stop, -1]
    grip = emitted[:, -1]
    states = teacher["state"][WINDOW.start:WINDOW.stop, -1]

    def first(values, limit):
        return next((issue for issue, value in zip(WINDOW, values, strict=True) if value <= limit), None)

    # Tail predictions are issued before the next refresh and refer to the same later teacher time.
    tail = {}
    for start, chunk in zip(refresh, chunks, strict=True):
        for slot, goal in enumerate(chunk[steps:, -1], steps):
            issue = start + slot
            if issue in WINDOW and goal <= CLOSE and expected[issue - WINDOW.start] <= CLOSE:
                tail.setdefault(issue, []).append(start)
    closed = expected <= CLOSE
    mismatched = [issue for issue, value in zip(WINDOW, grip, strict=True)
                  if issue in tail and value > CLOSE]
    return {"first_teacher_target_below_25": first(expected, MOVING_CLOSED), "first_act_target_below_25": first(grip, MOVING_CLOSED),
            "first_teacher_target_in_close_band": first(expected, CLOSE), "first_teacher_state_in_close_band": first(states, CLOSE),
            "first_act_target_in_close_band": first(grip, CLOSE), "teacher_close_issue_count": int(closed.sum()),
            "act_close_on_teacher_close_issues": int((grip[closed] <= CLOSE).sum()),
            "act_gripper_mae_on_teacher_close_issues_pp": float(np.abs(grip[closed] - expected[closed]).mean()),
            "discarded_close_at_issues_without_emitted_close": mismatched,
            "earlier_chunk_origins_for_close_in_discarded_tail": tail,
            "native_gripper_targets": grip.tolist(), "native_teacher_targets": expected.tolist(),
            "native_teacher_measured_gripper": states.tolist()}


def measured_gripper_intervention(frames, case, teacher, native):
    model, pre, post = native
    selected = {row["tick"] - 1: row for row in case["feedback"] if row["tick"] - 1 in SWAP_ISSUES}
    if set(selected) != set(SWAP_ISSUES):
        raise ValueError("An archived closed-loop gripper sample is missing")
    requests, report = [], []
    for index in SWAP_ISSUES:
        row = selected[index]
        if not np.allclose(row["teacher_requested_at_index"], teacher["requested"][index], atol=1e-5, rtol=0.):
            raise ValueError("The archived ACT measurement and teacher action are from different samples")
        observed = row["previous_observed_state"][-1]
        frame = frames[index - WINDOW.start]
        requests.extend((payload(frame), payload(frame, observed)))
        report.append({"issue": index, "teacher_measured_percent": float(teacher["state"][index, -1]),
                       "actual_closed_loop_measurement_percent": observed,
                       "saved_act_command_percent": row["requested"][-1],
                       "recorded_teacher_target_percent": float(teacher["requested"][index, -1])})
    predictions = decode(requests, model, pre, post)[:, 0, -1].reshape(-1, 2)
    for receipt, (original, changed) in zip(report, predictions, strict=True):
        receipt.update(act_with_teacher_measurement_percent=float(original),
                       act_with_closed_loop_gripper_measurement_percent=float(changed))
    return report


def cue_factorial(teacher, meta, marker, native):
    requests, records = [], []
    for wrist, top, state, clock in product(CUE_ISSUES, repeat=len(CUE_FIELDS)):
        rgb = {KEYS["wrist"]: teacher["wrist"][wrist],
               KEYS["top"]: conditioned_top(teacher["top"][top], marker, issue_index=clock)}
        frame = {"observation.state": teacher["state"][state], "task": instruction(meta["task"], marker), **rgb}
        requests.append(payload(frame))
        records.append(dict(zip(CUE_FIELDS, (wrist, top, state, clock), strict=True)))
    output = decode(requests, *native)
    return [{**record, "first_native_gripper_target_percent": float(chunk[0, -1]),
             "close_actions_among_first_six": int((chunk[:6, -1] <= CLOSE).sum())}
            for record, chunk in zip(records, output, strict=True)]


def source_report(split, official, entry, case, steps, native):
    model, pre, post = native
    path = Path(entry["raw_path"])
    meta = load(path / "manifest.json")
    marker = meta.get("target_marker", "cyan")
    if (meta["sha256"] != entry["source_sha256"] or marker != entry["target_marker"]
            or meta["world_episode_id"] != entry["source_world_episode_id"]
            or not meta["physics"]["released_stable_containment"]):
        raise ValueError(f"The aligned fitting export lost a successful teacher source: {path}")
    teacher = load_episode(path, meta, aligned=True)
    frames = [make_frame(teacher, meta, index, CLOCKED_PROFILE) for index in WINDOW]
    origin = entry["dataset_episode_index"] * meta["sample_count"]
    for index in (114, 134, 149, 164):
        check_row(official[origin + index], frames[index - WINDOW.start])
    requests = [payload(frame) for frame in frames]
    refresh = WINDOW[::steps]
    chunks = decode(requests[::steps], model, pre, post)
    emitted = chunks[:, :steps].reshape(len(WINDOW), -1)
    error = queue_error(requests, emitted, model, pre, post, 24, 2 * steps)
    scope = summarize_rows(teacher, emitted, chunks, refresh, steps)
    with np.load(path / "motor.npz", allow_pickle=False) as archive:
        times = archive["timestamps_ns"]
    if case and any(row["observed_at_ns"] != int(times[row["tick"] - 1]) for row in case["feedback"]):
        raise ValueError("Measured closed-loop gripper samples do not align with the raw teacher clocks")
    swaps = measured_gripper_intervention(frames, case, teacher, native) if case else []
    return {"seed": meta["seed"], "split": split, "target_marker": marker,
            "source_sha256": meta["sha256"], "sampled_actual_queue_native_max_abs_difference": error,
            "teacher_forced": scope, "single_requery_gripper_swaps": swaps,
            "teacher_input_component_swaps": cue_factorial(teacher, meta, marker, native)}


def checkpoint_visual_binding(checkpoint, dataset):
    profile = load(checkpoint / "smart-swarm-input.json")
    lineage = [load(dataset / split / "smart-swarm-lineage.json") for split in ("train", "validation")]
    if (profile["input_profile"] != CLOCKED_PROFILE or profile["dataset"] != str(dataset)
            or any(row["input_profile"] != CLOCKED_PROFILE or row["allowed_images"] != KEYS
                   or row["control_timing"] != "aligned-sample-v2" for row in lineage)):
        raise ValueError("The historical ACT checkpoint no longer identifies clock-conditioned inputs")
    return match_visual(profile, *lineage)


def audit():
    torch.set_num_threads(1)
    fit = load(FIT)
    reference = load(REFERENCE)
    checkpoint, dataset = (Path(fit[key]) for key in ("checkpoint", "dataset"))
    weights = sha(checkpoint / "model.safetensors")
    if (sha(FIT) != reference["fit_receipt_sha256"] or weights != reference["checkpoint_sha256"]
            or sha(dataset / "train/smart-swarm-lineage.json") != reference["training_lineage_sha256"]):
        raise ValueError("The historical clock-fit identity changed")
    visual = checkpoint_visual_binding(checkpoint, dataset)
    config, model, pre, post = load_policy(checkpoint)
    steps = config.n_action_steps
    if (version("lerobot") != "0.6.1" or config.type != "act" or steps != 6 or config.chunk_size != 24
            or config.temporal_ensemble_coeff is not None or WINDOW.start % steps or len(WINDOW) % steps):
        raise ValueError("The ACT runtime or historical execution queue no longer matches")
    closed_loop = {case["seed"]: case for case in fit["fitting_cases"]}
    sources, hashes = inspect_splits(dataset, closed_loop, steps, (model, pre, post))
    return {"scope": "Development archive on frozen ACT with recorded teacher observations", "checkpoint": str(checkpoint),
            "artifact_sha256": {file.name: sha(file) for file in sorted(checkpoint.iterdir()) if file.is_file()},
            "fit_receipt_sha256": sha(FIT), "input_profile": CLOCKED_PROFILE, **visual,
            "offline_issued_interval": [WINDOW.start, WINDOW.stop], "interval_is_half_open": True,
            "action_chunk_size": config.chunk_size, "issued_actions_per_refresh": steps,
            "teacher_lineage_sha256_by_split": hashes, "sources": sources, "new_optimizer_updates": 0,
            "teacher_input_component_swap_issue_indices": list(CUE_ISSUES),
            "component_swaps_scope": "Input interventions between two teacher frames, outside the physical rollout.",
            "closed_loop_replayed_or_mutated": False, "new_simulation_render_or_physical_trials": 0}


def inspect_splits(dataset, closed_loop, steps, native):
    sources = []
    hashes = {}
    for split, expected in (("train", 5), ("validation", 2)):
        base = dataset / split
        path = base / "smart-swarm-lineage.json"
        hashes[split] = sha(path)
        ancestry = load(path)
        official = LeRobotDataset(f"smart-swarm/{split}", root=base, delta_timestamps={"action": [0.]})
        if len(ancestry["episodes"]) != expected or official.num_episodes != expected:
            raise ValueError("The old ACT teacher corpus lost its independent split membership")
        for entry in ancestry["episodes"]:
            seed = int(entry["source_world_episode_id"].rsplit("-", 1)[1])
            case = closed_loop[seed] if split == "train" else None
            sources.append(source_report(split, official, entry, case, steps, native))
    return sources, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a new clocked ACT closure audit path")
    atomic(args.output, audit())
    print(args.output)


if __name__ == "__main__":
    main()
