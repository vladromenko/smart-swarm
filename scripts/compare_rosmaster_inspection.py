#!/usr/bin/env python3
"""Run the independent paired ROSMASTER camera intervention in MuJoCo."""

import argparse
from contextlib import closing
from hashlib import sha256
from pathlib import Path
import shutil
import time

import numpy as np

from probe_rosmaster_inspection import (CAMERA, SCAN, aim, capture_view, check_reset, count_visible_target,
                                       evaluator_state, hold, sealed_decision)
from smart_swarm.cli import settings
from smart_swarm.episode import atomic, load, sha
from smart_swarm.inspection_comparison import ARMS, MAX_DRIFT_M, MAX_TRACKING_ERROR_DEG, follow_up, score_pair, summarize
from smart_swarm.inspection_view import combine
from smart_swarm.primary_inputs import primary_policy_view
from smart_swarm.rendering import configure_renderer
from smart_swarm.research_jobs import admitted

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT.parent.parent / "work/active-inspection-comparison"
KINDS = ("events_only", "events_plus_reveal")
LIMITS = {"case_wall_seconds": 100, "cumulative_wall_seconds": 2400, "disk_bytes": 384 * 1024**2}
DISK_PER_CASE = 12 * 1024**2


def design(profiles):
    config = {"scope": "paired simulated optical intervention at N=3", "profiles_sha256": sha(profiles),
              "limits": LIMITS, "target_drift_max_m": MAX_DRIFT_M, "tracking_error_max_deg": MAX_TRACKING_ERROR_DEG,
              "arms": list(ARMS), "nominal_view_poses_deg": [[name, list(pose)] for name, pose in SCAN],
              "primary_endpoint": "correct location after a two-exposure scan versus the same two stationary exposures",
              "implementation_sha256": {name: sha(ROOT / name) for name in (
                  "scripts/compare_rosmaster_inspection.py", "scripts/probe_rosmaster_inspection.py",
                  "src/smart_swarm/inspection_comparison.py", "src/smart_swarm/inspection_view.py")},
              "model_kinds": list(KINDS), "independent_sources": 20, "optimizer_updates": 0}
    config["sources"] = [source(index, family) for index in range(10) for family in ("F1", "F6")]
    return config


def source(index, family):
    seed = (620_000 if family == "F1" else 621_000) + 8 * index + 4 * (index % 2) + index // 2 % 2
    concealed = ("hidden-contained", "hidden-surface")[(index + index // 2) % 2]
    variant = "visible-control" if family == "F1" else concealed
    return {"source_group": f"active-inspection-{family.lower()}-{index:02}", "seed": seed,
            "family": family, "variant": variant}


def profile_method(path, contexts):
    from score_primary_test import inference
    from score_primary_validation import IDENTITY, SELECTIVE

    decisions = {}
    with primary_policy_view(path) as scoped:
        for kind, context in contexts.items():
            predictions = inference(scoped, 3, context, False)
            choice = predictions[IDENTITY if context["inspect_markers"] else SELECTIVE]
            decisions[kind] = {key: choice[key] for key in ("decision", "belief", "serialized_bytes", "modeled_latency_ms")}
    return decisions


def inspect_branch(world, snapshot, directory, schedule=None):
    check_reset(world, snapshot)
    directory.mkdir()
    before = evaluator_state(world)
    start = world.ns
    samples, target_pixels = [], []
    for index, (label, goal) in enumerate(SCAN):
        motion = None
        if schedule is None:
            motion = aim(world, goal)
        else:
            hold(world, schedule[index] - world.ns)
        sample = capture_view(world, directory, label, CAMERA)
        samples.append({**sample, "motion": motion})
        target_pixels.append({"at_ns": world.ns, "pixels": count_visible_target(world, CAMERA)})
    after = evaluator_state(world)
    policy = {"start_ns": start, "stop_ns": world.ns, "samples": samples,
              "post_inspection_decision": combine([row["readout"] for row in samples])}
    truth = {"reset_exact": True, "target_pixels": target_pixels,
             "before": endpoint(before), "after": endpoint(after),
             "target_displacement_m": float(np.linalg.norm(np.subtract(before["cube_xyz_m"], after["cube_xyz_m"])))}
    return policy, truth


def endpoint(truth):
    return {"location": truth["terminal_location"], "relation": truth["cube_relation"]}


def capture_pair(config, recipe, path, pre_inspection):
    from smart_swarm.evidence_rollouts import prepare_world
    from smart_swarm.scaling import PassiveReceiverStory, describe_capture

    family, seed, variant = (recipe[key] for key in ("family", "seed", "variant"))
    world, variation, pickup = prepare_world(config["asset_root"], seed, family, variant=variant,
                                            seconds=LIMITS["case_wall_seconds"])
    with closing(world):
        story = PassiveReceiverStory(pre_inspection, family, world, seed, variant == "hidden-surface")
        describe_capture(story, variation, variant, 3, True, None)
        story.recorder.meta["scene_group_id"] = recipe["source_group"]
        snapshot, rule, expected = sealed_decision(story, pickup)
        scan, scan_truth = inspect_branch(world, snapshot, path / "scan")
        schedule = [sample["at_ns"] for sample in scan["samples"]]
        stationary, control_truth = inspect_branch(world, snapshot, path / "stationary", schedule)
    return {"scan": scan, "stationary": stationary, "initial_rule_decision": rule}, {
        "scan": scan_truth, "stationary": control_truth}, expected, sha256(snapshot.integration.tobytes()).hexdigest()


def attempt(config, recipe, path, contexts):
    path.mkdir()
    receipt = path / "receipt.json"
    started = time.monotonic()
    result = {**recipe, "status": "running", "start_wall_unix": time.time()}
    atomic(receipt, result)
    try:
        pre_inspection = path / f"{path.name}-pre-inspection"
        policy, truth, expected, snapshot_hash = capture_pair(config, recipe, path, pre_inspection)
        initial = profile_method(pre_inspection, contexts)
        policy["frozen_initial_predictions"] = initial
        policy["methods"] = {kind: {arm: follow_up(row["decision"], policy[arm]) for arm in ARMS}
                             for kind, row in initial.items()}
        if time.monotonic() - started > LIMITS["case_wall_seconds"]:
            raise TimeoutError("Paired case and inference exceeded the planned wall limit")
        prediction = path / "policy.json"
        atomic(prediction, policy)
        evaluation = score_pair(expected, policy, truth)
        (path / "evaluator").mkdir()
        atomic(path / "evaluator/result.json", {"truth": truth, "evaluation": evaluation,
                                               "sealed_policy_sha256": sha(prediction), "snapshot_sha256": snapshot_hash})
        result.update(status="sealed", evaluation=evaluation, policy_sha256=sha(prediction))
    except (OSError, ValueError, RuntimeError) as error:
        result.update(status="failed", reason=f"{type(error).__name__}: {error}")
    finally:
        result["wall_seconds"] = time.monotonic() - started
        atomic(receipt, result)
    return result


def release_output(root, plan):
    root.mkdir(exist_ok=True, parents=True)
    path = root / "design.json"
    if path.is_file() and load(path) != plan:
        raise ValueError("The inspection comparison design changed after allocation")
    if not path.is_file():
        atomic(path, plan)
    return path


def require_probe(path):
    probe = load(path)
    if not probe["complete"] or not all(probe["f6_scope"][name] for name in (
            "pre_inspection_permitted_input_match", "same_pre_inspection_rule_decision", "different_sealed_outcomes")):
        raise ValueError("The separate development optical and F6 scope checks need to complete")
    for case in probe["cases"]:
        scan, stationary = case["inspection_branch"], case["stationary_branch"]
        if (len(scan["active_scan"]) != 2 or not scan["reset_exact"] or not stationary["reset_exact"]
                or scan["stop_ns"] != stationary["stop_ns"]):
            raise ValueError("The development inspection probe did not exercise the matched optics")
        measured = [row["motion"]["requested_deg"] for row in scan["active_scan"]]
        if not np.allclose(measured, [pose for _, pose in SCAN], rtol=0., atol=.001):
            raise ValueError("The development probe used a different camera scan")
        if case["inspection_evaluation"]["correct_location_if_answered"] is not True:
            raise ValueError(f"The development scan did not localize {case['variant']}. Keep the independent cohort unopened")


def verify_sealed(path, receipt):
    prediction = path / "policy.json"
    policy = load(prediction)
    evaluated = load(path / "evaluator/result.json")
    if not receipt["policy_sha256"] == evaluated["sealed_policy_sha256"] == sha(prediction):
        raise ValueError("The sealed inspection policy payload changed")
    outcome = score_pair(evaluated["evaluation"]["location"], policy, evaluated["truth"])
    if not receipt["evaluation"] == evaluated["evaluation"] == outcome:
        raise ValueError("The saved inspection score could not be reproduced")
    for arm in ARMS:
        for sample in policy[arm]["samples"]:
            file = (path / arm / sample["frame"]).resolve()
            if not file.is_relative_to(path / arm) or sha(file) != sample["sha256"]:
                raise ValueError("A saved inspection RGB frame changed")


def resumption(root, plan):
    attempts, pending = [], []
    for recipe in plan["sources"]:
        path = root / recipe["source_group"]
        if path.exists():
            receipt = path / "receipt.json"
            previous = load(receipt) if receipt.is_file() else {**recipe, "status": "interrupted"}
            if any(previous[key] != value for key, value in recipe.items()):
                raise ValueError("An inspection case differs from its fixed recipe")
            if previous["status"] in ("running", "interrupted"):
                previous.update(status="interrupted", reason="An earlier capture stopped before sealing")
                previous["wall_seconds"] = LIMITS["case_wall_seconds"]
                atomic(receipt, previous)
            if previous["status"] == "sealed":
                verify_sealed(path, previous)
            attempts.append(previous)
        else:
            pending.append((recipe, path))
    return attempts, pending


def bounded_next(root, attempts):
    used = sum(row["wall_seconds"] for row in attempts)
    disk = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    return (used + LIMITS["case_wall_seconds"] <= LIMITS["cumulative_wall_seconds"]
            and disk + DISK_PER_CASE <= LIMITS["disk_bytes"] and shutil.disk_usage(root).free >= DISK_PER_CASE)


def run(root, plan, config, contexts, grant, heartbeat):
    attempts, pending = resumption(root, plan)
    for recipe, path in pending:
        if len(attempts) < plan["independent_sources"] and bounded_next(root, attempts):
            heartbeat()
            attempts.append(attempt(config, recipe, path, contexts))
            status_report(root, plan, attempts, grant)
    return status_report(root, plan, attempts, grant)


def status_report(root, plan, attempts, grant):
    report = root / "comparison.json"
    atomic(report, {**summarize(plan["independent_sources"], attempts), "attempts": attempts, "reservation": grant,
                    "design_sha256": sha(root / "design.json"), "endpoints": "simulated optical location only"})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--probe", type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    plan = design(args.profiles)
    root = args.output.resolve()
    if any(part in ("primary-sim-validation", "primary-sim-test") for part in root.parts):
        raise ValueError("The optical intervention uses a separate recording directory")
    if args.plan_only:
        print(release_output(root, plan))
        return
    if args.probe is None:
        raise ValueError("Supply the existing development optical probe receipt")
    require_probe(args.probe)
    from freeze_primary_profiles import require_profiles
    from score_primary_test import runtime
    import torch

    profile = require_profiles(args.profiles)
    config = settings()
    configure_renderer(config)
    with admitted(config, "smart-swarm-rosmaster-inspection-comparison") as (grant, heartbeat):
        release_output(root, plan)
        torch.set_num_threads(3)
        contexts = {kind: runtime(profile, kind) for kind in KINDS}
        print(run(root, plan, config, contexts, grant, heartbeat))


if __name__ == "__main__":
    main()
