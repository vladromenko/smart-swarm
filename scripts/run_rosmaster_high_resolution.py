#!/usr/bin/env python3
"""Replay the released high-resolution inspection protocol on its original independent allocation."""

import argparse
from contextlib import closing
from hashlib import sha256
from pathlib import Path
import shutil
import time

from PIL import Image

from compare_rosmaster_inspection import DISK_PER_CASE, KINDS, LIMITS, release_output, resumption
from probe_rosmaster_inspection_resolution import draw_without_offscreen_meshes
from probe_rosmaster_parallel_branches import ARMS, FRUSTUM, OLD, PILOT, WORK, run_pair
from smart_swarm.cli import ROOT, settings
from smart_swarm.contract import json_bytes
from smart_swarm.episode import atomic, load, sha
from smart_swarm.inspection_comparison import follow_up, score_pair, summarize
from smart_swarm.rendering import configure_renderer
from smart_swarm.research_jobs import admitted
from verify_mobile_frozen_head_reuse import shared_contexts, shared_profile_method

OUTPUT = WORK / "active-inspection-comparison-mobile-v3"
PROFILES = PILOT / "primary-profiles-frozen.json"
ENCODING = PILOT / "rosmaster-shared-frozen-encoding-verification.json"
WAYPOINT = PILOT / "rosmaster-inspection-west-waypoint.json"


def offsets():
    measured = set()
    for path in OLD.glob("mobile-rehearsal-*/policy.json"):
        scan = load(path)["scan"]
        measured.add(tuple(sample["at_ns"] - scan["start_ns"] for sample in scan["samples"]))
    if len(measured) != 1:
        raise ValueError("The fixed route must have one verified two-exposure schedule")
    clocks = next(iter(measured))
    if len(clocks) != 2 or not 0 < clocks[0] < clocks[1]:
        raise ValueError("Invalid measured route schedule")
    return list(clocks)


def design():
    from check_mobile_resolution_replays import audit as resolution_audit
    from compare_rosmaster_mobile import original_sources, stored_bytes
    from mobile_inspection_case import DECISION

    source = original_sources(PROFILES)
    if any(OLD.glob("active-inspection-f*")):
        raise ValueError("An independent source already exists under an earlier optical profile")
    qualified = resolution_audit()
    proof = load(ENCODING)
    if not qualified["ready"] or proof["status"] != "verified" or len(proof["verified_sources"]) != 3:
        raise ValueError("All optical and frozen-encoding development controls must pass")
    for row in proof["verified_sources"]:
        prior = OLD / f"mobile-rehearsal-{row['variant']}" / DECISION
        if row["decision_sha256"] != sha(prior) or json_bytes(row["outputs"]) != json_bytes(load(prior)["learned"]):
            raise ValueError("The exact encoding replay no longer matches an original frozen decision")
    files = ("scripts/run_rosmaster_high_resolution.py", "scripts/probe_rosmaster_parallel_branches.py",
             "scripts/verify_mobile_frozen_head_reuse.py", "scripts/probe_rosmaster_inspection_resolution.py",
             "scripts/rehearse_rosmaster_mobility.py", "scripts/mobile_inspection_case.py",
             "src/smart_swarm/physics.py", "src/smart_swarm/inspection_mobility.py",
             "src/smart_swarm/inspection_triangulation.py", "src/smart_swarm/inspection_comparison.py")
    return {"scope": "paired mobile optical intervention in simulation at N=3", "sources": source,
            "profiles_sha256": sha(PROFILES), "encoding_proof_sha256": sha(ENCODING),
            "frustum_proof_sha256": sha(FRUSTUM), "waypoint_sha256": sha(WAYPOINT),
            "development_sha256": {name: row["receipt_sha256"] for name, row in qualified["results"].items()},
            "exposure_offsets_ns": offsets(), "post_inspection_resolution": [640, 480],
            "limits": LIMITS, "charged_development_work": qualified["charged_development_work"],
            "charged_development_wall_seconds": qualified["previous_wall_seconds"],
            "charged_development_disk_bytes": sum(stored_bytes(Path(row["root"])) for row in qualified["charged_development_work"]),
            "implementation_sha256": {name: sha(ROOT / name) for name in files}, "optimizer_updates": 0}


def sealed_stock(config, recipe, directory, contexts, deadline):
    from audit_primary_scope import verify_projection
    from mobile_inspection_case import DECISION, record_decision
    from smart_swarm.evidence_rollouts import prepare_world

    seed, family, variant = (recipe[k] for k in ("seed", "family", "variant"))
    stock, variation, pickup = prepare_world(config["asset_root"], seed, family, variant=variant)
    pre = directory / "pre-inspection"
    with closing(stock):
        stock.deadline = deadline
        if family == "F1":
            stock.image = draw_without_offscreen_meshes(stock)
        fixed, rule, expected, teacher = record_decision(stock, variation, pickup, recipe, pre)
    signature, count = verify_projection(pre)
    prediction = {"permitted_input_sha256": sha256(json_bytes(signature)).hexdigest(), "permitted_rgb_frames": count,
                  "recording_manifest_sha256": sha(pre / "manifest.json"), "engineering_rule": rule,
                  "learned": shared_profile_method(pre, contexts)}
    atomic(directory / DECISION, prediction)
    return fixed, expected, teacher, prediction


def high_branch(path, branch):
    from audit_mobile_triangulation import prediction as optical_prediction
    from check_mobile_inspection_rehearsal import EXPOSURES

    frames = branch["policy_observations"]
    for frame, name in zip(frames, EXPOSURES, strict=True):
        if name != frame["frame"] or sha(path / name) != frame["sha256"]:
            raise ValueError("The optical branch changed a declared RGB exposure")
        with Image.open(path / name) as image:
            if image.mode != "RGB" or image.size != (640, 480):
                raise ValueError("The native optical policy expects 640 by 480 RGB")
    readout = optical_prediction(path, frames)
    motion = branch["motion"]
    error = {"max_tracking_error_deg": max(row["max_error_deg"] for row in motion["arm"])} if motion else None
    return {**{key: branch[key] for key in ("start_ns", "stop_ns", "motion", "checked_physics_steps",
                                             "external_contact_pairs", "self_contact_pairs")},
            "samples": [{**frame, "motion": error} for frame in frames], "post_inspection_decision": readout}


def paired_policy(directory, results, route, prediction):
    from check_mobile_inspection_rehearsal import verify_route
    from mobile_inspection_case import DECISION
    from rehearse_rosmaster_mobility import compare

    if len({row["reset_sha256"] for row in results.values()}) != 1:
        raise ValueError("The matched workers did not restore the same mobile state")
    branches = {arm: results[arm]["branch"] for arm in ARMS}
    matched = compare(branches["scan"], branches["stationary"])
    verify_route({"active": branches["scan"], "stationary": branches["stationary"], "comparison": matched}, route)
    public = {arm: high_branch(directory / arm, branch) for arm, branch in branches.items()}
    public["methods"] = {kind: {arm: follow_up(row["decision"], public[arm]) for arm in ARMS}
                         for kind, row in prediction["learned"].items()}
    public["pre_inspection_decision_sha256"] = sha(directory / DECISION)
    truth = {arm: {"reset_exact": True, **results[arm]["endpoints"],
                   "target_displacement_m": branch["evaluator"]["cube_displacement_m"],
                   "carrier_displacement_m": branch["evaluator"]["carrier_displacement_m"],
                   "visibility": branch["evaluator"]["visibility"]} for arm, branch in branches.items()}
    return public, truth


def development_replay():
    from check_mobile_inspection_rehearsal import STATION
    from mobile_inspection_case import DECISION
    from rehearse_rosmaster_mobility import read_waypoint

    directory = WORK / "rosmaster-parallel-visible-control-v2"
    proof = load(directory / "probe.json")
    evaluation = load(OLD / "mobile-rehearsal-visible-control/evaluator/result.json")
    forecast = load(directory / DECISION)
    if proof["status"] != "verified" or not proof["exact_rgb_and_contacts"]:
        raise ValueError("The high-resolution composition needs the verified CPU development run")
    results = {arm: load(directory / arm / "worker-result.json") for arm in ARMS}
    for arm, result in results.items():
        if result["reset_sha256"] != evaluation["snapshot_sha256"]:
            raise ValueError("The archived worker used a different confirmed development reset")
        # The replay predates the explicit endpoints in the worker receipt.
        original = evaluation["truth"][arm]
        result["endpoints"] = {key: original[key] for key in ("before", "after")}
    route = read_waypoint(WAYPOINT, STATION)
    public, truth = paired_policy(directory, results, route, forecast)
    score = score_pair(evaluation["evaluation"]["location"], public, truth)
    if (not score["paired_scoring_valid"] or not score["optical"]["scan"]["correct"]
            or score["optical"]["stationary"]["answered"]):
        raise ValueError("The actual archived high-resolution pair failed its composition check")
    return score


def seal_score(directory, public, truth, expected, teacher, reset):
    file = directory / "policy.json"
    atomic(file, public)
    result = score_pair(expected, public, truth)
    (directory / "evaluator").mkdir()
    digest = sha(file)
    atomic(directory / "evaluator/result.json", {"truth": truth, "evaluation": result, "teacher": teacher,
                                                "sealed_policy_sha256": digest, "snapshot_sha256": reset})
    return {"status": "sealed", "evaluation": result, "policy_sha256": digest}


def acquire(config, recipe, directory, contexts, route, plan, heartbeat, allowance):
    from check_mobile_inspection_rehearsal import verify_task
    from mobile_inspection_case import verify_case

    started = time.monotonic()
    directory.mkdir()
    file = directory / "receipt.json"
    row = {**recipe, "status": "running", "start_wall_unix": time.time()}
    atomic(file, row)
    request = {"assets": config["asset_root"], "recipe": recipe, "route": route,
               "exposure_offsets_ns": plan["exposure_offsets_ns"], "deadline": started + allowance}
    try:
        def seal():
            return sealed_stock(config, recipe, directory, contexts, request["deadline"])

        results, (_, expected, teacher, prediction) = run_pair(directory, request, heartbeat, seal, True)
        verify_task(teacher, recipe["variant"], expected)
        public, truth = paired_policy(directory, results, route, prediction)
        candidate = seal_score(directory, public, truth, expected, teacher, results["scan"]["reset_sha256"])
        verify_case(directory, candidate)
        if time.monotonic() > request["deadline"]:
            raise TimeoutError("The verified paired inspection reached its remaining wall allowance")
        row.update(candidate)
    except (AssertionError, KeyError, OSError, RuntimeError, ValueError) as error:
        row.update(status="failed", reason=f"{type(error).__name__}: {error}")
    finally:
        row["wall_seconds"] = time.monotonic() - started
        atomic(file, row)
    return row


def remaining(plan, attempts):
    return max(0., LIMITS["cumulative_wall_seconds"] - plan["charged_development_wall_seconds"]
               - sum(row["wall_seconds"] for row in attempts))


def publish(root, plan, attempts, grant):
    report = root / "comparison.json"
    summary = summarize(len(plan["sources"]), attempts)
    status = "complete" if summary["valid_pairs"] == len(plan["sources"]) else "bounded_incomplete"
    atomic(report, {**summary, "status": status, "attempts": attempts, "reservation": grant,
                    "design_sha256": sha(root / "design.json"), "remaining_wall_seconds": remaining(plan, attempts),
                    "endpoint": "simulated location after two matched high-resolution exposures",
                    "physical_motion_requested": False, "optimizer_updates": 0})
    return report


def run(config, plan, cases, grant, heartbeat):
    from check_mobile_inspection_rehearsal import STATION
    from compare_rosmaster_mobile import stored_bytes
    from freeze_primary_profiles import require_profiles
    from mobile_inspection_case import verify_case
    from rehearse_rosmaster_mobility import read_waypoint
    from score_primary_test import runtime
    import torch

    release_output(OUTPUT, plan)
    torch.set_num_threads(3)
    profile = require_profiles(PROFILES)
    contexts = shared_contexts({kind: runtime(profile, kind) for kind in KINDS})
    route = read_waypoint(WAYPOINT, STATION)
    attempts, pending = resumption(OUTPUT, plan)
    for row in attempts:
        if row["status"] == "sealed":
            verify_case(OUTPUT / row["source_group"], row)
    for recipe, path in pending[:cases]:
        allowance = min(float(LIMITS["case_wall_seconds"]), remaining(plan, attempts))
        total_disk = stored_bytes(OUTPUT) + plan["charged_development_disk_bytes"] + DISK_PER_CASE
        if allowance >= 1. and total_disk <= LIMITS["disk_bytes"] and shutil.disk_usage(OUTPUT).free >= DISK_PER_CASE:
            heartbeat()
            attempts.append(acquire(config, recipe, path, contexts, route, plan, heartbeat, allowance))
            publish(OUTPUT, plan, attempts, grant)
    return publish(OUTPUT, plan, attempts, grant)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--max-cases", type=int, choices=range(1, 21), default=20)
    args = parser.parse_args()
    configure_renderer(settings())
    plan = design()
    development = development_replay()
    if args.check_only:
        print(json_bytes({"planned": len(plan["sources"]), "remaining_wall_seconds": remaining(plan, []),
                          "resolution": plan["post_inspection_resolution"], "exposure_offsets_ns": plan["exposure_offsets_ns"],
                          "actual_development_replay": development["optical"]}).decode())
        return
    config = settings()
    with admitted(config, "smart-swarm-high-resolution-mobile-comparison") as (grant, heartbeat):
        print(run(config, plan, args.max_cases, grant, heartbeat))


if __name__ == "__main__":
    main()
