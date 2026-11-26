#!/usr/bin/env python3
"""Compare saved closed-loop ACT commands with teacher targets at the same issue indices."""

import argparse
from pathlib import Path

import numpy as np

from smart_swarm.cli import ROOT
from smart_swarm.episode import atomic, load, sha

PILOT = ROOT / "evidence/learning-pilot-v3"
TEACHER_CLOSE_MAX_PERCENT = 5.0
STAGES = (("approach", 60), ("descend", 120), ("grasp", 180), ("lift", 240),
          ("transfer", 330), ("lower", 390), ("release", 450), ("withdraw", 510))


def action_samples(raw, case):
    manifest = load(raw / "manifest.json")
    archive_path = raw / "motor.npz"
    if (manifest["seed"] != case["seed"] or manifest["status"] != "accepted"
            or not manifest["physics"]["released_stable_containment"] or sha(archive_path) != manifest["sha256"]):
        raise ValueError(f"{raw} does not match the preserved successful teacher source")
    with np.load(archive_path, allow_pickle=False) as archive:
        targets = archive["requested"]
        times = archive["timestamps_ns"]
    feedback = case["feedback"]
    ticks = [row["tick"] for row in feedback]
    if ticks != list(range(15, len(targets) + 1, 15)) or targets.shape != (STAGES[-1][1], 6):
        raise ValueError("The archived evaluation does not have the declared feedback cadence")
    cases = []
    for tick, report in zip(ticks, feedback, strict=True):
        recorded = targets[tick - 1]
        if not np.allclose(report["teacher_requested_at_index"], recorded, rtol=0., atol=1e-5):
            raise ValueError("An archived command has a different teacher target")
        if report["observed_at_ns"] != int(times[tick - 1]):
            raise ValueError("An archived observation has a different issue timestamp")
        cases.append({"tick": tick, "teacher_grip": float(recorded[-1]), "act_grip": report["requested"][-1],
                      "clipped_grip": report["decoded_and_clipped"][-1], "bilateral_contact": len(report["gripper_contact_bodies"]) >= 2})
    return manifest["sha256"], targets, cases


def range_of(values):
    return [float(min(values)), float(max(values))]


def characterize(ledger):
    closed = [row for row in ledger if row["teacher_grip"] <= TEACHER_CLOSE_MAX_PERCENT]
    return {"logged_issue_count": len(ledger), "teacher_close_issue_count": len(closed),
            "act_close_issue_count": sum(row["act_grip"] <= TEACHER_CLOSE_MAX_PERCENT for row in ledger),
            "act_bilateral_contact_count": sum(row["bilateral_contact"] for row in ledger),
            "act_close_period_requested_percent": range_of([row["act_grip"] for row in closed]) if closed else None,
            "act_close_period_clipped_percent": range_of([row["clipped_grip"] for row in closed]) if closed else None,
            "teacher_close_period_requested_percent": range_of([row["teacher_grip"] for row in closed]) if closed else None}


def one_run(raw, case):
    digest, targets, ledger = action_samples(raw, case)
    phases = {}
    start = 0
    for name, end in STAGES:
        rows = [row for row in ledger if start < row["tick"] <= end]
        phases[name] = {**characterize(rows), "teacher_dense_action_rows": len(targets[start:end]),
                        "teacher_dense_close_rows": int(np.count_nonzero(targets[start:end, -1] <= TEACHER_CLOSE_MAX_PERCENT))}
        start = end
    return {"seed": case["seed"], "teacher_source_sha256": digest, "first_failed_stage": case["first_failed_stage"],
            "contained": case["stable_containment"], **characterize(ledger), "by_phase": phases}


def report(fit_path):
    fit = load(fit_path)
    lineage_path = Path(fit["dataset"]) / "train/smart-swarm-lineage.json"
    sources = {}
    for row in load(lineage_path)["episodes"]:
        raw = Path(row["raw_path"])
        manifest = load(raw / "manifest.json")
        if row["source_sha256"] != manifest["sha256"] or manifest["seed"] in sources:
            raise ValueError("Teacher source has changed or appears more than once")
        sources[manifest["seed"]] = raw
    trials = fit["fitting_cases"]
    if len(trials) != 5 or {case["seed"] for case in trials} != set(sources):
        raise ValueError("The fit archive does not cover the five matching teacher trajectories exactly once")
    cases = [one_run(sources[case["seed"]], case) for case in trials]
    ckpt = Path(fit["checkpoint"]) / "model.safetensors"
    offline = [{key: row[key] for key in ("stage", "index", "valid_chunk_actions")}
               | {"grip_one_step_mae_pp": row["one_step_mae"][-1], "grip_chunk_mae_pp": row["chunk_mae"][-1]}
               for row in fit["validation"]["offline_native_action_error"]]
    return {"scope": "Development archive only", "fit_receipt_sha256": sha(fit_path),
            "checkpoint_sha256": sha(ckpt), "training_lineage_sha256": sha(lineage_path),
            "logged_sampling_period_control_ticks": 15, "gripper_units": "percent_calibrated_range",
            "teacher_close_band_max_percent": TEACHER_CLOSE_MAX_PERCENT, "matching_resets": cases,
            "offline_teacher_input_examples": offline, "closed_loop_containments": sum(case["contained"] for case in cases),
            "new_optimizer_updates": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a new report path")
    atomic(args.output, report(PILOT / "act-clock-fit.json"))
    print(args.output)


if __name__ == "__main__":
    main()
