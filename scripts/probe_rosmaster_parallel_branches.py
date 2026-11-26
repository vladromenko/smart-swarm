#!/usr/bin/env python3
"""Time one full, exact matched inspection case with two fixed CPU branches in one native lease."""

import argparse
from contextlib import closing
from copy import copy, deepcopy
from hashlib import sha256
import multiprocessing
from multiprocessing.connection import wait
import os
from pathlib import Path
from time import monotonic

from smart_swarm.cli import ROOT, settings
from smart_swarm.contract import json_bytes
from smart_swarm.episode import atomic, load, sha
from smart_swarm.rendering import configure_renderer
from smart_swarm.research_jobs import admitted

WORK = ROOT.parent.parent / "work"
PILOT = ROOT / "evidence/learning-pilot-v3"
OLD = WORK / "active-inspection-comparison-mobile-v2"
QUALIFICATION = PILOT / "rosmaster-high-resolution-matched-development-v2.json"
FRUSTUM = WORK / "rosmaster-exact-frustum-visibility-v1/probe.json"
ARMS = ("scan", "stationary")
VARIANTS = ("visible-control", "hidden-contained", "hidden-surface")


def original_case(variant):
    root = OLD / f"mobile-rehearsal-{variant}"
    original = load(root / "receipt.json")
    release = load(QUALIFICATION)
    if not release["ready"] or original["status"] != "sealed" or sha(root / "policy.json") != original["policy_sha256"]:
        raise ValueError("The fixed performance case must already be qualified and sealed")
    high = Path(release["results"][variant]["receipt"])
    if sha(high) != release["results"][variant]["receipt_sha256"]:
        raise ValueError("The qualified high-resolution reference changed")
    return root, original, load(high)


def branch_models(assets, recipe):
    from smart_swarm.evidence_rollouts import prepare_world
    from smart_swarm.inspection_mobility import install, mobile_model

    seed, family, variant = (recipe[k] for k in ("seed", "family", "variant"))
    stock, _, _ = prepare_world(assets, seed, family, variant=variant)
    try:
        mobile = copy(stock)
        mobile.manifest = deepcopy(stock.manifest)
        mobile.rng = deepcopy(stock.rng)
        model, identity = mobile_model(assets, seed, family, variant)
        install(mobile, model, identity)
        if mobile.model is stock.model or mobile.data is stock.data:
            raise ValueError("The mobile branch must have its own compiled model and solver data")
        mobile.observed_inspection_steps = True
        mobile.frame_size = (640, 480)
    except Exception:
        stock.close()
        raise
    return stock, mobile


def worker(arm, directory, request, frozen_input):
    from compare_rosmaster_inspection import endpoint
    from rehearse_rosmaster_mobility import evaluator_state, mobile_scan, stationary
    from smart_swarm.mobile_decision_state import transplant

    started = monotonic()
    result = directory / "worker-result.json"
    try:
        stock, world = branch_models(request["assets"], request["recipe"])
        compiled = monotonic()
        with closing(stock), closing(world), closing(frozen_input):
            # Only compile static models before the parent seals its prediction.
            fixed = frozen_input.recv()
            released = monotonic()
            snapshot = transplant(stock, world, fixed)
            world.deadline = min(world.deadline, request["deadline"])
            reset_hash = sha256(snapshot.integration.tobytes()).hexdigest()
            if request.get("reset_sha256", reset_hash) != reset_hash:
                raise ValueError("The CPU worker did not reproduce the archived mobile reset")
            before = endpoint(evaluator_state(world))
            if arm == "scan":
                route = request["route"]
                branch = mobile_scan(world, snapshot, directory, route["station"], route)
            else:
                clocks = request.get("clocks")
                if clocks is None:
                    clocks = [world.ns + offset for offset in request["exposure_offsets_ns"]]
                branch = stationary(world, snapshot, clocks, directory)
            endpoints = {"before": before, "after": endpoint(evaluator_state(world))}
    except Exception as error:
        atomic(result, {"status": "failed", "error": str(error), "wall_seconds": monotonic() - started})
        raise
    atomic(result, {"status": "recorded", "branch": branch, "wall_seconds": monotonic() - started,
                    "model_build_seconds": compiled - started, "wait_for_frozen_input_seconds": released - compiled,
                    "reset_sha256": reset_hash, "endpoints": endpoints, "pid": os.getpid()})


def wait_for_pair(workers, deadline, heartbeat):
    while any(process.is_alive() for process in workers):
        remaining = deadline - monotonic()
        if remaining <= 0.:
            raise TimeoutError("The fixed inspection pair reached its original 100-second case cap")
        live = [process.sentinel for process in workers if process.is_alive()]
        if live:
            wait(live, timeout=min(remaining, 5.))
        heartbeat()
        if any(process.exitcode not in (0, None) for process in workers):
            raise RuntimeError("One of the owned inspection branches failed")
    if any(process.exitcode != 0 for process in workers):
        raise RuntimeError("The owned inspection branches did not both complete")


def run_pair(output, request, heartbeat, seal, prepare_early):
    if len(os.sched_getaffinity(0)) < 2:
        raise ValueError("This timing probe requires capacity for two CPU branches")
    context = multiprocessing.get_context("spawn")
    workers, senders = [], []
    frozen = None if prepare_early else seal()
    try:
        for arm in ARMS:
            directory = output / arm
            directory.mkdir()
            receive, send = context.Pipe(duplex=False)
            senders.append(send)
            with closing(receive):
                process = context.Process(target=worker, args=(arm, directory, request, receive))
                process.start()
            workers.append(process)
        if prepare_early:
            frozen = seal()
        for send in senders:
            send.send(frozen[0])
        wait_for_pair(workers, request["deadline"], heartbeat)
    finally:
        for send in senders:
            send.close()
        for process in workers:
            if process.is_alive():
                process.terminate()
            process.join(timeout=3.)
            if process.is_alive():
                process.kill()
                process.join()
            process.close()
    return {arm: load(output / arm / "worker-result.json") for arm in ARMS}, frozen


def verify_branch(directory, result, baseline, high):
    from check_mobile_resolution_replays import high_prediction

    branch = result["branch"]
    keys = ("start_ns", "stop_ns", "motion", "checked_physics_steps", "external_contact_pairs", "self_contact_pairs")
    if any(json_bytes(branch[key]) != json_bytes(baseline[key]) for key in keys):
        raise ValueError("A CPU branch changed the archived clock, route, or full-step contact result")
    rows = []
    for frame, expected, archived in zip(branch["policy_observations"], baseline["samples"], high, strict=True):
        if (frame["at_ns"] != expected["at_ns"] or frame["calibration"] != expected["calibration"]
                or frame["frame"] != expected["frame"] or frame["sha256"] != archived["high_sha256"]):
            raise ValueError("A CPU branch changed the archived camera pose, exposure, or native RGB values")
        rows.append({**archived, "high_sha256": frame["sha256"]})
    return high_prediction(directory, rows, baseline["samples"])


def sealed_stock(config, recipe, source, output, contexts):
    from mobile_inspection_case import freeze_prediction
    from probe_rosmaster_inspection_resolution import check_pre_inspection, record_stock
    from smart_swarm.evidence_rollouts import prepare_world

    stock, variation, pickup = prepare_world(config["asset_root"], recipe["seed"], recipe["family"], variant=recipe["variant"])
    pre = output / "pre-inspection"
    with closing(stock):
        proof = FRUSTUM if recipe["variant"] == "visible-control" else None
        fixed, rule, truth, _ = record_stock(stock, variation, pickup, recipe, source, pre, proof)
        scope = check_pre_inspection(source, pre)
        frozen = freeze_prediction(output, pre, contexts, rule)
    old = load(source / "pre-inspection-decision.json")
    if json_bytes(frozen["learned"]) != json_bytes(old["learned"]) or frozen["engineering_rule"] != old["engineering_rule"]:
        raise ValueError("The probe changed the pre-inspection policy on identical permitted inputs")
    return fixed, scope, truth


def timed_case(config, variant, output, contexts, heartbeat, prepare_early):
    from rehearse_rosmaster_mobility import compare, read_waypoint

    source, recipe, high = original_case(variant)
    policy = load(source / "policy.json")
    reference = load(source / "evaluator/result.json")
    started = monotonic()

    def seal():
        return (*sealed_stock(config, recipe, source, output, contexts), monotonic() - started)

    route = read_waypoint(PILOT / "rosmaster-inspection-west-waypoint.json", "west_surface")
    request = {"assets": config["asset_root"], "recipe": recipe, "route": route,
               "reset_sha256": reference["snapshot_sha256"], "deadline": started + 100.,
               "clocks": [sample["at_ns"] for sample in policy["scan"]["samples"]]}
    results, (_, scope, truth, stock_seconds) = run_pair(output, request, heartbeat, seal, prepare_early)
    clocks = compare(results["scan"]["branch"], results["stationary"]["branch"])
    if clocks != high["matched_branch_comparison"] or truth != clocks["same_location"]:
        raise ValueError("The parallel branches changed the original paired scene evidence")
    references = (high["two_native_resolutions"], high["stationary"]["two_native_resolutions"])
    choices = {arm: verify_branch(output / arm, results[arm], policy[arm], proof)
               for arm, proof in zip(ARMS, references, strict=True)}
    if monotonic() > request["deadline"]:
        raise TimeoutError("The complete inspection case reached the original 100-second cap")
    return {"status": "verified", "variant": variant, "build_and_pre_inspection_seconds": stock_seconds,
            "compiled_before_seal": prepare_early,
            "worker_preparation": {arm: {k: results[arm][k] for k in ("model_build_seconds", "wait_for_frozen_input_seconds")}
                                   for arm in ARMS},
            "case_wall_seconds": monotonic() - started, "branch_wall_seconds": {arm: results[arm]["wall_seconds"] for arm in ARMS},
            "worker_pids": [results[arm]["pid"] for arm in ARMS], "pre_inspection_scope": scope,
            "choices": choices, "matched_comparison": clocks, "exact_rgb_and_contacts": True,
            "independent_cases_opened": 0, "optimizer_updates": 0}


def run(config, args, grant, heartbeat):
    from compare_rosmaster_inspection import KINDS
    from freeze_primary_profiles import require_profiles
    from score_primary_test import runtime
    import torch

    torch.set_num_threads(3)
    profiler = require_profiles(PILOT / "primary-profiles-frozen.json")
    contexts = {kind: runtime(profiler, kind) for kind in KINDS}
    args.output.mkdir()
    report = args.output / "probe.json"
    started = monotonic()
    try:
        result = timed_case(config, args.variant, args.output, contexts, heartbeat, args.prepare_branches_early)
    except (OSError, KeyError, ValueError, RuntimeError, KeyboardInterrupt) as error:
        atomic(report, {"status": "failed", "reason": str(error), "wall_seconds": monotonic() - started,
                        "reservation": grant})
        raise
    atomic(report, {**result, "wall_seconds": monotonic() - started, "reservation": grant,
                    "available_cpu_count": len(os.sched_getaffinity(0))})
    print(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-branches-early", action="store_true", help="Compile both models before sealing the prediction")
    args = parser.parse_args()
    config = settings()
    configure_renderer(config)
    with admitted(config, "smart-swarm-two-branch-cpu-timing") as (grant, heartbeat):
        run(config, args, grant, heartbeat)


if __name__ == "__main__":
    main()
