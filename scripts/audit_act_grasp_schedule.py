#!/usr/bin/env python3
"""Reissue the saved ACT grasp window on the original teacher's arm trajectory."""

import argparse
import os
from pathlib import Path
import sys
from time import monotonic

import numpy as np

from smart_swarm.cli import ROOT, settings
from smart_swarm.episode import atomic, load, sha
from smart_swarm.motor_diagnostics import contact_valid_placement, read_teacher, replay, reset_world
from smart_swarm.rendering import configure_renderer
from smart_swarm.research_jobs import admitted, execute

PILOT = ROOT / "evidence/learning-pilot-v3"
ARCHIVE = PILOT / "act-clock-closure-window.json"
FIT = PILOT / "act-clock-fit.json"
ARCHIVE_SHA = "7562a4112b80aa6f14a9b238346238faeb9ac4f3e5a48ca4fb777e47228e2339"
FIT_SHA = "1b318f30abe35c401624457b70ebc491116b210734eb85212ae4b2306e67eb72"
WINDOW = range(108, 204)
PROFILE = "aligned-sample-v2"
WALL_LIMIT_S = 180
NAMES = ("teacher", "archived_act_gripper", "first_act_close_held")
SOURCES = {"train": range(10000, 10005), "validation": range(20000, 20002)}


def bound_material():
    if sha(ARCHIVE) != ARCHIVE_SHA or sha(FIT) != FIT_SHA:
        raise ValueError("An original ACT fitting or command receipt changed")
    archive, fit = load(ARCHIVE), load(FIT)
    if (archive["fit_receipt_sha256"] != FIT_SHA or archive["checkpoint"] != fit["checkpoint"]
            or archive["offline_issued_interval"] != [WINDOW.start, WINDOW.stop]
            or archive["issued_actions_per_refresh"] != 6 or archive["action_chunk_size"] != 24
            or [(row["split"], row["seed"]) for row in archive["sources"]]
            != [(split, seed) for split, seeds in SOURCES.items() for seed in seeds]):
        raise ValueError("The archived ACT controls or development source order changed")
    paths = {}
    for split in SOURCES:
        lineage = Path(fit["dataset"]) / split / "smart-swarm-lineage.json"
        if sha(lineage) != archive["teacher_lineage_sha256_by_split"][split]:
            raise ValueError("The frozen ACT teacher lineage changed")
        paths[split] = [Path(row["raw_path"]) for row in load(lineage)["episodes"]]
    return archive, paths


def command_variants(arrays, archived):
    original = arrays["requested"]
    targets = np.asarray(archived["native_gripper_targets"], dtype=original.dtype)
    if (original.shape != (510, 6) or targets.shape != (len(WINDOW),) or not np.isfinite(targets).all()
            or not np.array_equal(original[WINDOW.start:WINDOW.stop, -1], archived["native_teacher_targets"])
            or not np.array_equal(arrays["state"][WINDOW.start:WINDOW.stop, -1], archived["native_teacher_measured_gripper"])):
        raise ValueError("The archived ACT commands do not match these exact teacher issues")
    first_close = archived["first_act_target_in_close_band"]
    closes = [issue for issue, value in zip(WINDOW, targets, strict=True) if value <= 5.]
    if not closes or closes[0] != first_close:
        raise ValueError("The archive's first full-close command changed")
    act = original.copy()
    act[WINDOW.start:WINDOW.stop, -1] = targets
    held = act.copy()
    held[first_close:WINDOW.stop, -1] = act[first_close, -1]
    return dict(zip(NAMES, (original, act, held), strict=True))


def outcome(result):
    feedback = result["feedback"]
    contact = next((row["tick"] - 1 for row in feedback if len(row["gripper_contact_bodies"]) >= 2), None)
    fields = ("first_failed_stage", "stable_containment", "held_lift_10cm", "carry_bilateral_samples", "carry_sample_count",
              "precontact_state_max_abs_residual", "sample_times_exact")
    return {**{key: result[key] for key in fields},
            "first_bilateral_contact_after_issue": contact, "first_bilateral_contact_during_patch": contact in WINDOW,
            "contact_valid_placement": contact_valid_placement(result),
            "grasp_observations": [{"after_issue": row["tick"] - 1, "bilateral": len(row["gripper_contact_bodies"]) >= 2,
                                    "cube_height_m": row["cube_height_above_table_m"]}
                                   for row in feedback if row["tick"] - 1 in WINDOW]}


def verify_case(assets, raw, archived):
    manifest, arrays, _ = read_teacher(raw)
    if (manifest["seed"] != archived["seed"] or manifest["sha256"] != archived["source_sha256"]
            or manifest.get("target_marker", "cyan") != archived["target_marker"]):
        raise ValueError("Teacher recording and original ACT output are from different resets")
    targets = command_variants(arrays, archived["teacher_forced"])
    world = reset_world(assets, manifest)
    results = {}
    try:
        snapshot = world.snapshot()
        for branch, requested in targets.items():
            world.restore(snapshot)
            trace = replay(world, {**arrays, "requested": requested}, PROFILE, contact_issues=WINDOW)
            results[branch] = outcome(trace)
            if not trace["sample_times_exact"] or trace["precontact_state_max_abs_residual"] > 1e-4:
                raise ValueError(f"Issue time or pre-intervention state changed on {branch}")
            if branch == "teacher" and (not results[branch]["contact_valid_placement"] or
                                        trace["carry_bilateral_samples"] != trace["carry_sample_count"]):
                raise ValueError("The original teacher failed the matched execution gate")
    finally:
        world.close()
    scope = archived["teacher_forced"]
    return {"seed": manifest["seed"], "split": archived["split"], "teacher_sha256": manifest["sha256"],
            "first_teacher_command_below_25_percent": scope["first_teacher_target_below_25"],
            "first_act_command_below_25_percent": scope["first_act_target_below_25"],
            "first_act_command_at_most_5_percent": scope["first_act_target_in_close_band"], "branches": results}


def summarize(results):
    total = {}
    for branch in NAMES:
        rows = [case["branches"][branch] for case in results]
        total[branch] = {"contact_during_patch": sum(row["first_bilateral_contact_during_patch"] for row in rows),
                         "contact_valid_placements": sum(row["contact_valid_placement"] for row in rows),
                         "first_bilateral_after_issue": [row["first_bilateral_contact_after_issue"] for row in rows],
                         "first_failed_stage": [row["first_failed_stage"] for row in rows]}
    return total


def worker(config, directory):
    configure_renderer(config)
    archive, paths = bound_material()
    cursors = {key: iter(items) for key, items in paths.items()}
    cases = []
    for row in archive["sources"]:
        raw = next(cursors[row["split"]])
        result = verify_case(config["asset_root"], raw, row)
        cases.append(result)
        atomic(directory / f"seed-{row['seed']}.json", result)
        print(f"Replayed the three command schedules for {row['seed']}", flush=True)
    atomic(directory / "report.json", {"cases": cases, "summary": summarize(cases)})


def run(config, output, prior_charge_s):
    directory = output.with_name(output.stem + "-job")
    directory.mkdir(exist_ok=False, parents=True)
    design = {"archive_sha256": ARCHIVE_SHA, "fit_sha256": FIT_SHA, "window_half_open": [WINDOW.start, WINDOW.stop],
              "branches": list(NAMES), "scope": "Seven previously used development resets. Only the gripper changes.",
              "wall_limit_s": WALL_LIMIT_S, "prior_charge_s": prior_charge_s, "optimizer_updates": 0,
              "model_inference": False, "fresh_resets_read": 0,
              "hypothesis": "The archived ACT grip sequence can close on time when the other targets remain the teacher's.",
              "limits": "ACT commands were inferred previously from teacher observations. The teacher controls the other joints "
                        "and all actions outside the archived window. No branch is a learned feedback rollout."}
    atomic(directory / "design.json", design)
    with admitted(config, "smart-swarm-act-grasp-schedule", archive_only=True) as (grant, heartbeat):
        start = monotonic()
        argv = [sys.executable, __file__, "--worker", str(directory)]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
        with (directory / "worker.log").open("w") as log:
            code = execute(argv, env, log, start + WALL_LIMIT_S - prior_charge_s, heartbeat)
        if code:
            raise RuntimeError(f"ACT schedule replay failed. See {directory / 'worker.log'}")
        result = load(directory / "report.json")
        elapsed = monotonic() - start
        atomic(output, {**design, **result, "reservation": grant, "elapsed_s": elapsed,
                        "cumulative_charged_wall_s": prior_charge_s + elapsed})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--worker", type=Path)
    modes.add_argument("--output", type=Path)
    parser.add_argument("--prior-charge-s", type=int, default=0, help="Time to deduct for an interrupted setup")
    args = parser.parse_args()
    if args.worker:
        worker(settings(), args.worker)
    elif args.output.exists():
        parser.error("Choose a new immutable audit result path")
    elif not 0 <= args.prior_charge_s < WALL_LIMIT_S:
        parser.error("The prior charge must leave time within the original allowance")
    else:
        run(settings(), args.output.resolve(), args.prior_charge_s)
        print(args.output)


if __name__ == "__main__":
    main()
