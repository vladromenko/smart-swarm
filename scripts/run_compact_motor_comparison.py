#!/usr/bin/env python3
"""Train or evaluate only the inherited bounded motor comparison on the published compact split."""

import argparse
from collections import Counter
from contextlib import closing
import json
from pathlib import Path
import shutil
import time

from complete_compact_motor_dataset import (LOCATION, TARGET, actual_inventory, budget_recipe, match_pilot_export,
                                             pilot_receipts, remaining)
from learning_pilot import EVIDENCE, ROOT, fresh_resets, verified_export
from smart_swarm.cli import settings
from smart_swarm.contract import ProtocolError
from smart_swarm.episode import atomic, load, sha
from smart_swarm.motor_input import CLOCKED_PROFILE
from smart_swarm.motor_visual_profile import COMPACT, match_visual, visual_binding
from smart_swarm.rendering import configure_renderer
from smart_swarm.research_jobs import admitted, execute
from smart_swarm.training import offline, verified_shared_path

FRESH = EVIDENCE / "fresh-resets.json"
COMPACT_DATA = LOCATION / "lerobot-full"
RELEASE = LOCATION / "motor-comparison-release.json"
PROCESSOR = LOCATION / "smolvla-compact-processor.json"
DEVELOPMENT_PROFILE = LOCATION / "smolvla-development-profile.json"
RENDER_RECOVERY = LOCATION / "smolvla-compact-render-recovery.json"
PROCESSOR_WALL_S = 300
MODELS = ("act", "smolvla")


def published_dataset():
    path = LOCATION / "full-capture.json"
    if not path.is_file():
        raise ValueError("The complete compact 80/20 dataset has not published its capture receipt")
    report = load(path)
    if (report["status"] != "dataset_ready" or report.get("active") or report["optimizer_updates"] != 0
            or report["source_counts"] != TARGET or report["accepted_sources"] != TARGET
            or Path(report["dataset"]).resolve() != COMPACT_DATA.resolve()):
        raise ValueError("Full compact capture/export did not publish the exact 80/20 cohort")
    pilot, digest = pilot_receipts(LOCATION)
    match_pilot_export(LOCATION, pilot)
    recipe = budget_recipe(LOCATION, pilot, digest)
    if report["recipe"] != recipe or any(remaining(report, stage) < 0 for stage in ("generation", "export")):
        raise ValueError("The compact corpus does not satisfy its original collection or export budget")
    raw = Path(recipe["raw"])
    if actual_inventory(raw) != TARGET or not verified_export(raw, COMPACT_DATA, "--clocked"):
        raise ValueError("The official export differs from the accepted raw 80/20 inventory")
    return raw, sha(path)


def source_plan(raw):
    saved = load(FRESH)
    if saved != fresh_resets():
        raise ValueError("The reserved fresh motor recipes changed")
    resets = saved["resets"]
    new_seeds = {row["seed"] for row in resets}
    occupied = {load(path)["seed"] for path in raw.glob("*/*/manifest.json")}
    if new_seeds & occupied or len(new_seeds) != 20:
        raise ValueError("A fresh motor recipe collides with an admitted training or validation source")
    return saved


def prepare(model):
    from smart_swarm.motor_full import require_fit

    raw, receipt = published_dataset()
    lineages = {split: COMPACT_DATA / split / "smart-swarm-lineage.json" for split in TARGET}
    train, validation = [load(path) for path in lineages.values()]
    match_visual(train, validation, visual_binding(COMPACT))
    if any(row["input_profile"] != CLOCKED_PROFILE or row["frames"] != TARGET[split] * 510
           or any(item["frame_count"] != 510 for item in row["episodes"])
           for split, row in (("train", train), ("validation", validation))):
        raise ValueError("The official train and validation export must retain identical clocked conditioning")
    original_fit = require_fit(EVIDENCE, model, CLOCKED_PROFILE)
    verified_shared_path()
    fresh = source_plan(raw)
    return {"dataset": str(COMPACT_DATA), "capture_receipt_sha256": receipt,
            "lineage_sha256": {split: sha(path) for split, path in lineages.items()},
            "original_act_diagnostic_fit_gate": original_fit["fit_gate"], "input_profile": CLOCKED_PROFILE,
            **visual_binding(COMPACT), "fresh_reset_sha256": sha(FRESH), "train_and_validation_counts": TARGET,
            "fresh_reset_count": len(fresh["resets"]), "fresh_seed_start": fresh["resets"][0]["seed"],
            "executor": "aligned-sample-v2", "action_schema_id": "sim-so101-absolute-v1",
            "camera_order": ["wrist", "top"], "episode_control_steps": 510, "training_compute": "CPU"}


def claim_release(release):
    if RELEASE.exists():
        if load(RELEASE) != release:
            raise ValueError("The existing compact comparison has a different dataset or held-out recipe")
    else:
        atomic(RELEASE, release)


def verify_processor(report, dataset):
    expected = {"observation.images.camera1", "observation.images.camera2", "observation.language.attention_mask",
                "observation.language.tokens", "observation.state"}
    if (set(report["parity"]) != expected or not all(report["parity"].values())
            or not report["target_only_changed_tokens"] or report["camera_order"] != ["wrist", "top"]
            or report["dataset"] != str(dataset) or report["input_profile"] != CLOCKED_PROFILE
            or report["optimizer_updates"] or report["closed_loop_trials"]):
        raise ValueError("The installed SmolVLA processors did not verify the actual compact split")


def processor_check(heartbeat, dataset, output):
    settings_file = load(ROOT / "configs/training.json")
    runtime = settings_file["runtime"]
    argv = [runtime, str(ROOT / "scripts/audit_smol_init.py"), "--dataset", str(dataset),
            "--output", str(output), "--timed-minibatch"]
    env = {**offline(), "PYTHONPATH": str(ROOT / "src")}
    started = time.monotonic()
    log_path = output.with_suffix(".log")
    with log_path.open("a") as log:
        if execute(argv, env, log, started + PROCESSOR_WALL_S, heartbeat) != 0:
            raise RuntimeError(f"The compact SmolVLA processor check failed. See {log_path}")
    report = load(output)
    verify_processor(report, dataset)
    if report["minibatch"]["optimizer_created"] or report["minibatch"]["unpadded_actions"] < 1:
        raise ValueError("The genuine compact minibatch and execution timing did not finish")
    return time.monotonic() - started, report["minibatch"]


def development_profile():
    from smart_swarm.motor_full import smolvla_schedule

    pilot, digest = pilot_receipts(LOCATION)
    match_pilot_export(LOCATION, pilot)
    dataset = Path(pilot["dataset"])
    output = DEVELOPMENT_PROFILE.with_name("smolvla-development-processor.json")
    config = settings()
    with admitted(config, "smart-swarm-smolvla-compact-development-profile") as (grant, heartbeat):
        if DEVELOPMENT_PROFILE.exists():
            raise ValueError("The bounded SmolVLA development profile was already attempted")
        started = time.monotonic()
        row = {"status": "running", "dataset": str(dataset), "pilot_receipt_sha256": digest,
               "preflight_wall_cap_s": PROCESSOR_WALL_S, "optimizer_updates": 0, "reservation": grant}
        atomic(DEVELOPMENT_PROFILE, row)
        try:
            _, probe = processor_check(heartbeat, dataset, output)
            free = shutil.disk_usage(config["data_root"]).free
            row.update(status="profiled", processor_receipt=str(output), processor_receipt_sha256=sha(output),
                       advisory_schedule=smolvla_schedule(probe, 3900, free))
        except (OSError, ValueError, RuntimeError, ProtocolError, KeyboardInterrupt) as error:
            row.update(status="failed", reason=str(error) or type(error).__name__)
            raise
        finally:
            row["charged_wall_s"] = time.monotonic() - started
            atomic(DEVELOPMENT_PROFILE, row)
    return row


def prior_profile_charge():
    if not DEVELOPMENT_PROFILE.exists():
        return 0.
    row = load(DEVELOPMENT_PROFILE)
    return row.get("charged_wall_s", float(row["preflight_wall_cap_s"]))


def recovered_processor():
    if not RENDER_RECOVERY.is_file():
        return 0., None
    row = load(RENDER_RECOVERY)
    failure = Path(row["failure_log"])
    if (row["status"] != "processor_complete_renderer_failed" or row["optimizer_updates"] != 0
            or row["release_sha256"] != sha(RELEASE) or row["processor_sha256"] != sha(PROCESSOR)
            or row["failure_sha256"] != sha(failure) or not 0 < row["charged_wall_s"] <= PROCESSOR_WALL_S):
        raise ValueError("The archived processor and charged renderer failure do not match the published run")
    report = load(PROCESSOR)
    verify_processor(report, COMPACT_DATA)
    return row["charged_wall_s"], report["minibatch"]


def verify_result(model, status):
    if status["dataset"] != str(COMPACT_DATA):
        raise ValueError("The existing motor job belongs to another dataset")
    if status["status"] != "evaluated":
        return False
    validation = status["validation"]
    if not validation or not all(row["offline_validation_sampled_sources"] == list(range(20)) and len(row["resets"]) == 20
                                 for row in validation):
        raise ValueError("The checkpoint selection did not score exactly the declared validation worlds")
    trials = status["fresh"]
    reserved = load(FRESH)["resets"]
    expected = [(row["seed"], row["target_marker"], row["sliding_friction"]) for row in reserved]
    actual = [(row["seed"], row["target_marker"], row["recipe"]["cube_sliding_friction"]) for row in trials["results"]]
    if (actual != expected or trials["evaluation_role"] != "fresh-test"
            or trials["executor_profile"] != "aligned-sample-v2" or trials["policy"] != model
            or trials["offline_validation_sampled_sources"] != list(range(20))
            or any(row["steps"] != 510 or row["seed"] != row["policy_reset_seed"] for row in trials["results"])):
        raise ValueError("The fresh score does not match the reserved motor execution protocol")
    if model == "smolvla" and any(row.get("action_transport") != "verified-smolvla-chunk" for row in (*validation, trials)):
        raise ValueError("The SmolVLA comparison did not use the verified native chunk transport")
    return True


def audited_trajectory(row, *, fresh):
    from smart_swarm.motor_diagnostics import contact_valid_placement, summarize

    observations = row["sampled_evaluator_truth" if fresh else "feedback"]
    if [sample["tick"] for sample in observations] != list(range(15, 511, 15)):
        raise ValueError(f"Reset {row['seed']} did not retain all 34 declared contact samples")
    summary = summarize(observations)
    successful = contact_valid_placement(summary)
    if summary["first_failed_stage"] != row["first_failed_stage"]:
        raise ValueError(f"Reset {row['seed']} failure stage differs from the original observations")
    if fresh:
        if (row["contact_valid_placement"] != successful
                or row["contained_and_released"] != summary["stable_containment"]):
            raise ValueError(f"Fresh reset {row['seed']} outcome differs from the original observations")
    elif any(summary[key] != row[key] for key in ("stable_containment", "held_lift_10cm", "carry_bilateral_samples")):
        raise ValueError(f"Validation reset {row['seed']} outcome differs from the original observations")
    return successful, summary["first_failed_stage"]


def audited_stage(rows, recorded_count, *, fresh):
    outcomes = [audited_trajectory(row, fresh=fresh) for row in rows]
    successful = sum(item[0] for item in outcomes)
    if successful != recorded_count:
        raise ValueError("Closed-loop success count differs from the recorded trajectory outcomes")
    return {"resets": len(rows), "contact_valid_placements": successful,
            "failed_lift_threshold_without_stage": sum(not ok and stage is None for ok, stage in outcomes),
            "first_failure_stages": dict(Counter(stage for _, stage in outcomes if stage is not None))}


def audit_completed_result(model, status):
    from smart_swarm.evaluate_motor import sha256_stream

    verify_result(model, status)
    lineage = load(COMPACT_DATA / "validation/smart-swarm-lineage.json")["episodes"]
    expected = [load(Path(row["raw_path"]) / "manifest.json")["seed"] for row in lineage]
    validation = {}
    for candidate in status["validation"]:
        trials = candidate["resets"]
        if candidate["evaluation_role"] != "validation" or [row["seed"] for row in trials] != expected:
            raise ValueError("Closed-loop validation resets differ from the actual published source order")
        validation[candidate["checkpoint"]] = audited_stage(trials, candidate["successful_resets"], fresh=False)
    fresh = status["fresh"]
    checkpoint = status["checkpoint"]
    if checkpoint != fresh["checkpoint"] or checkpoint not in validation:
        raise ValueError("The fresh score does not name a checkpoint selected from validation")
    with (Path(checkpoint) / "model.safetensors").open("rb") as stream:
        if sha256_stream(stream) != fresh["checkpoint_sha256"]:
            raise ValueError("The selected checkpoint no longer matches the original fresh evaluation weights")
    return {"status": "evaluated", "optimizer_updates": status["optimizer_updates"], "checkpoint": checkpoint,
            "checkpoint_sha256": fresh["checkpoint_sha256"], "validation": validation,
            "fresh": audited_stage(fresh["results"], fresh["successful_resets"], fresh=True)}


def read_comparison_result(model):
    path = EVIDENCE / f"{model}-full.json"
    if not path.is_file():
        return {"status": "awaiting_receipt"}
    result = load(path)
    if result["status"] == "evaluated":
        return audit_completed_result(model, result)
    return {"status": result["status"], "receipt": str(path)}


def time_motor_path(config, entry, spacing):
    from smart_swarm.arm_executor import ArmExecutor
    from smart_swarm.evaluate_motor import action_request
    from smart_swarm.motor_diagnostics import contact_valid_placement, observe, read_teacher, reset_world, summarize

    started = time.monotonic()
    meta, arrays, _ = read_teacher(Path(entry["raw_path"]))
    marker = meta["target_marker"]
    feedback, frames = [], 0
    with closing(reset_world(config["asset_root"], meta)) as world:
        executor = ArmExecutor(world, "aligned-sample-v2")
        executor.prime()
        for issue, command in enumerate(arrays["requested"]):
            if issue % spacing == 0:
                action_request(world, CLOCKED_PROFILE, marker, issue_index=executor.observation_index())
                frames += 1
            executor.submit(command)
            if issue % 15 == 14:
                feedback.append(observe(world, issue + 1))
    if frames != (510 + spacing - 1) // spacing or not contact_valid_placement(summarize(feedback)):
        raise ValueError("The timed motor path did not reproduce the archived contact-valid trajectory")
    return {"seed": meta["seed"], "marker": marker, "refreshes": frames,
            "wall_seconds": time.monotonic() - started, "source_sha256": meta["sha256"]}


def complete_smolvla_preflight(config, heartbeat, probe=None):
    started = time.monotonic()
    if probe is None:
        _, probe = processor_check(heartbeat, COMPACT_DATA, PROCESSOR)
    spacing = probe["queued_action_parity"]["action_steps_per_chunk"]
    rows = load(COMPACT_DATA / "validation/smart-swarm-lineage.json")["episodes"]
    cases = []
    for marker in ("cyan", "violet"):
        heartbeat()
        entry = next(row for row in rows if row["target_marker"] == marker)
        cases.append(time_motor_path(config, entry, spacing))
    output = PROCESSOR.with_name("smolvla-compact-motor-path-timing.json")
    atomic(output, {"simulated_teacher_paths": cases, "processor_receipt_sha256": sha(PROCESSOR),
                    "optimizer_updates": 0, "learned_policy_trials": 0})
    profile = {**probe, "motor_path_timing": cases, "motor_path_receipt_sha256": sha(output)}
    return time.monotonic() - started, profile


def start_comparison(config, model, destination, grant, heartbeat):
    from mujoco import FatalError
    from smart_swarm.motor_full import run

    started = time.monotonic()
    prior = prior_profile_charge() if model == "smolvla" else 0.
    try:
        recovery, previous = recovered_processor() if model == "smolvla" else (0., None)
        prior += recovery
        setup, probe = complete_smolvla_preflight(config, heartbeat, previous) if model == "smolvla" else (0., None)
        return run(config, model, COMPACT_DATA, EVIDENCE, grant, heartbeat, setup_wall_s=setup + prior, training_probe=probe)
    except (OSError, ValueError, RuntimeError, ProtocolError, FatalError) as error:
        if destination.exists():
            raise
        result = {"status": "inconclusive", "dataset": str(COMPACT_DATA), "optimizer_updates": 0,
                  "preflight_wall_s_charged": time.monotonic() - started + prior,
                  "evaluation_blocker": str(error), "provenance": grant}
        if model == "smolvla" and PROCESSOR.is_file():
            result["processor_receipt_sha256"] = sha(PROCESSOR)
        atomic(destination, result)
        return result


def execute_model(model, release):
    config = settings()
    configure_renderer(config)
    destination = EVIDENCE / f"{model}-full.json"
    archive_only = bool(config.get("native_cpu_admission"))
    with admitted(config, f"smart-swarm-compact-{model}-comparison", archive_only=archive_only) as (grant, heartbeat):
        if prepare(model) != release:
            raise ValueError("The approved motor dataset changed before native admission")
        claim_release(release)
        if destination.exists():
            result = load(destination)
        else:
            result = start_comparison(config, model, destination, grant, heartbeat)
        valid = verify_result(model, result)
        return {"job_status": str(destination), "fresh_result_verified": valid,
                "status": result["status"], "optimizer_updates": result.get("optimizer_updates", 0)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--profile-development", action="store_true", help="Time the verified three-source export without fitting")
    args = parser.parse_args()
    configure_renderer(settings())
    if args.profile_development:
        if args.model != "smolvla":
            parser.error("The development compute profile is available for SmolVLA")
        print(json.dumps(development_profile(), indent=2))
        return
    try:
        release = prepare(args.model)
    except (OSError, KeyError, ValueError) as error:
        parser.exit(2, f"Compact {args.model} comparison withheld. {error}\n")
    if args.check_only:
        print(json.dumps({"model": args.model, **release, "comparison": read_comparison_result(args.model)}, indent=2))
    else:
        result = execute_model(args.model, release)
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if result["fresh_result_verified"] else 2)


if __name__ == "__main__":
    main()
