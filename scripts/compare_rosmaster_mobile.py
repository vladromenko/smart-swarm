#!/usr/bin/env python3
"""Run the paired mobile inspection study after both development gates."""

import argparse
import json
import math
from pathlib import Path

from audit_mobile_triangulation import audit as optical_audit
from audit_primary_scope import paired
from check_mobile_inspection_rehearsal import STATION
from compare_rosmaster_inspection import (DISK_PER_CASE, KINDS, LIMITS, ROOT, bounded_next, design,
                                          release_output, resumption, summarize)
from mobile_inspection_case import capture_case, verify_case
from rehearse_rosmaster_mobility import FAMILY, SEED, read_waypoint
from smart_swarm.cli import settings
from smart_swarm.episode import atomic, load, sha
from smart_swarm.rendering import configure_renderer
from smart_swarm.research_jobs import admitted

WORK = ROOT.parent.parent / "work"
HISTORICAL = WORK / "active-inspection-comparison-v1-candidate/design.json"
REHEARSALS = WORK / "rosmaster-mobile-optical-v2"
PREVIOUS = WORK / "active-inspection-comparison-mobile-v1"
OUTPUT = WORK / "active-inspection-comparison-mobile-v2"
STEP_PARITY = WORK / "rosmaster-observed-step-parity-v2/verification.json"
EVIDENCE = ROOT / "evidence/learning-pilot-v3"
PROFILES = EVIDENCE / "primary-profiles-frozen.json"
WAYPOINT = EVIDENCE / "rosmaster-inspection-west-waypoint.json"
DEVELOPMENT_SEED = 416_039
DEVELOPMENT = tuple({"source_group": f"mobile-rehearsal-{variant}", "family": family,
                     "seed": DEVELOPMENT_SEED, "variant": variant}
                    for variant, family in FAMILY.items())


def original_sources(profiles):
    old = load(HISTORICAL)
    if old != design(profiles):
        raise ValueError("The original independent inspection allocation or implementation changed")
    for root in (HISTORICAL.parent, WORK / "active-inspection-comparison", PREVIOUS):
        if any(root.glob("active-inspection-f*")):
            raise ValueError("An independent inspection start already exists in the fixed-base allocation")
    return old["sources"]


def previous_charges():
    receipt = PREVIOUS / "development-gate.json"
    previous = load(receipt)
    if (previous["status"] != "failed" or {row["variant"] for row in previous["attempts"]} != set(FAMILY)
            or len(previous["attempts"]) != len(FAMILY) or (PREVIOUS / "comparison.json").exists()):
        raise ValueError("The timed-out development pass must remain available before recovery")
    for row in previous["attempts"]:
        actual = PREVIOUS / row["source_group"] / "receipt.json"
        if (row != load(actual) or row["seed"] != SEED or row["status"] != "failed"
                or not math.isfinite(row["wall_seconds"]) or row["wall_seconds"] < LIMITS["case_wall_seconds"]):
            raise ValueError("The prior case charges cannot be verified")
    return {"setup_receipt_sha256": sha(receipt), "disk_bytes": stored_bytes(PREVIOUS),
            "wall_seconds": sum(row["wall_seconds"] for row in previous["attempts"])}


def stored_bytes(root):
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def next_case(root, attempts, prior):
    return (bounded_next(root, attempts + [prior])
            and stored_bytes(root) + prior["disk_bytes"] + DISK_PER_CASE <= LIMITS["disk_bytes"])


def step_parity():
    report = load(STEP_PARITY)
    original = REHEARSALS / "hidden-surface/rehearsal.json"
    modules = ("src/smart_swarm/physics.py", "src/smart_swarm/inspection_mobility.py")
    if (report.get("status") != "matched" or report.get("original_route_exact") is not True
            or report.get("reference_sha256") != sha(original)
            or report.get("solver_sources_sha256") != {name: sha(ROOT / name) for name in modules}
            or report.get("steps_per_branch") != 54_180):
        raise ValueError("The per-tick observer must preserve the complete rendered route")
    for arm in ("active", "stationary"):
        for sample in report[arm]["policy_observations"]:
            frame = STEP_PARITY.parent / arm / sample["frame"]
            if frame.name not in ("carrier.png", "surface.png") or sha(frame) != sample["sha256"]:
                raise ValueError("An observed-step RGB verification frame changed")
    return sha(STEP_PARITY)


def mobile_design(profiles, optics):
    files = ("scripts/compare_rosmaster_mobile.py", "scripts/mobile_inspection_case.py",
             "scripts/check_mobile_inspection_rehearsal.py", "scripts/rehearse_rosmaster_mobility.py",
             "scripts/audit_mobile_triangulation.py", "src/smart_swarm/inspection_triangulation.py",
             "scripts/verify_rosmaster_observed_steps.py", "src/smart_swarm/physics.py",
             "scripts/compare_rosmaster_inspection.py", "scripts/audit_primary_scope.py",
             "src/smart_swarm/mobile_decision_state.py", "src/smart_swarm/inspection_mobility.py",
             "src/smart_swarm/inspection_comparison.py", "src/smart_swarm/inspection_view.py")
    return {"scope": "paired mobile optical intervention in simulation at N=3", "limits": LIMITS,
            "original_allocation_sha256": sha(HISTORICAL), "profiles_sha256": sha(profiles),
            "waypoint_sha256": sha(WAYPOINT), "development": list(DEVELOPMENT),
            "development_optical_sha256": {key: value["receipt_sha256"] for key, value in optics["results"].items()},
            "post_inspection_inputs": optics["prediction_inputs"],
            "matched_solver_route_sha256": step_parity(), "previous_development_charge": previous_charges(),
            "sources": original_sources(profiles), "frozen_models": list(KINDS),
            "endpoint": "correct location after a two-exposure scan versus the same two stationary exposures",
            "implementation_sha256": {name: sha(ROOT / name) for name in files}, "optimizer_updates": 0}


def dependency_status():
    sources = original_sources(PROFILES)
    primary = load(EVIDENCE / "primary-sim-source-allocation.json")
    seed_candidates = {seed for row in primary["slots"] for seed in row["seed_candidates"]}
    if DEVELOPMENT_SEED in seed_candidates | {SEED, *(row["seed"] for row in sources)}:
        raise ValueError("The new development seed overlaps a previous or held-out recipe")
    optics = optical_audit(REHEARSALS, WAYPOINT)
    try:
        parity = step_parity()
        issue = None
    except (OSError, KeyError, ValueError) as error:
        parity, issue = None, str(error)
    ready = bool(optics["ready"] and parity)
    return {**optics, "ready": ready, "optical_gate": optics["ready"], "route_parity_error": issue,
            "previous_development_charge": previous_charges(), "independent_sources_planned": len(sources),
            "status": "development_admissible" if ready else "development_admission_pending"}


def resume_attempts(root, cases):
    attempts, pending = resumption(root, {"sources": cases})
    for row in attempts:
        if row["status"] == "sealed":
            verify_case(root / row["source_group"], row)
    return attempts, pending


def check_development(root, reports):
    result = {}
    if {row["source_group"] for row in reports} != {row["source_group"] for row in DEVELOPMENT}:
        raise ValueError("The three stock-to-mobile development cases must complete")
    for row in reports:
        if row["status"] != "sealed" or not row["evaluation"]["paired_scoring_valid"]:
            raise ValueError(f"{row['variant']} has no valid matched intervention")
        evaluation = row["evaluation"]
        if not evaluation["optical"]["scan"]["correct"]:
            raise ValueError(f"The complete decision path did not resolve {row['variant']} with fresh RGB")
        directory = root / row["source_group"]
        result[row["variant"]] = {"path": directory / "pre-inspection", "location": evaluation["location"],
                                  "initial": verify_case(directory, row)}
    hidden = [result[variant] for variant in ("hidden-contained", "hidden-surface")]
    control = result["visible-control"]
    scope = paired({"actor_count": 3, "traces": [{"path": row["path"], "terminal_outcome": row["location"]} for row in hidden]})
    for kind in KINDS:
        a, b = [item["initial"]["learned"][kind] for item in hidden]
        direct = control["initial"]["learned"][kind]["decision"]
        if a != b or a["decision"] != "request_inspection" or direct != control["location"]:
            raise ValueError(f"{kind} failed the coupled F6 prediction or resolvable F1 control")
    return {"scope": scope, "learned": {key: item["initial"]["learned"] for key, item in result.items()},
            "same_f6_prediction": True, "separate_resolvable_control": True,
            "optical_resolved_all_controls": True, "independent_sources_used": 0}


def execute_development(root, config, contexts, route, heartbeat, prior):
    attempts, pending = resume_attempts(root, DEVELOPMENT)
    for recipe, path in pending:
        if all(row["status"] == "sealed" for row in attempts) and next_case(root, attempts, prior):
            heartbeat()
            attempts.append(capture_case(config, recipe, path, contexts, route))
    result = root / "development-gate.json"
    try:
        evidence = check_development(root, attempts)
    except (OSError, KeyError, ValueError) as error:
        atomic(result, {"status": "failed", "reason": str(error), "attempts": attempts})
        raise
    atomic(result, {"status": "passed", **evidence, "attempts": attempts})
    return attempts, sha(result)


def publish(root, design_file, cases, development, grant, gate_hash):
    plan = load(design_file)
    result = root / "comparison.json"
    report = summarize(len(plan["sources"]), cases)
    status = "bounded_incomplete"
    if len(cases) == len(plan["sources"]):
        status = "complete" if report["valid_pairs"] == len(cases) else "attempted_with_invalid_pairs"
    atomic(result, {**report, "status": status, "attempts": cases, "design_sha256": sha(design_file),
                    "development_gate_sha256": gate_hash, "reservation": grant,
                    "elapsed_case_wall_seconds": sum(row["wall_seconds"] for row in cases + development)
                    + plan["previous_development_charge"]["wall_seconds"],
                    "endpoint": plan["endpoint"], "physical_motion_requested": False, "optimizer_updates": 0})
    return result


def compare(config, contexts, route, grant, heartbeat, plan, *, development_only=False):
    design_file = release_output(OUTPUT, plan)
    prior = plan["previous_development_charge"]
    development, gate = execute_development(OUTPUT, config, contexts, route, heartbeat, prior)
    if development_only:
        return OUTPUT / "development-gate.json"
    cases, pending = resume_attempts(OUTPUT, plan["sources"])
    for recipe, path in pending:
        if next_case(OUTPUT, cases + development, prior):
            heartbeat()
            cases.append(capture_case(config, recipe, path, contexts, route))
            publish(OUTPUT, design_file, cases, development, grant, gate)
    return publish(OUTPUT, design_file, cases, development, grant, gate)


def run(optics, *, development_only=False):
    from freeze_primary_profiles import require_profiles
    from score_primary_test import runtime
    import torch

    profile = require_profiles(PROFILES)
    plan = mobile_design(PROFILES, optics)
    route = read_waypoint(WAYPOINT, STATION)
    config = settings()
    configure_renderer(config)
    with admitted(config, "smart-swarm-mobile-inspection-comparison") as (grant, heartbeat):
        torch.set_num_threads(3)
        contexts = {kind: runtime(profile, kind) for kind in KINDS}
        return compare(config, contexts, route, grant, heartbeat, plan, development_only=development_only)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--development-only", action="store_true")
    args = parser.parse_args()
    optics = dependency_status()
    if args.check_only or not optics["ready"]:
        print(json.dumps(optics, indent=2, sort_keys=True))
        raise SystemExit(0 if optics["ready"] else 2)
    result = run(optics, development_only=args.development_only)
    print(result)
    completed = "passed" if args.development_only else "complete"
    raise SystemExit(0 if load(result)["status"] == completed else 2)


if __name__ == "__main__":
    main()
