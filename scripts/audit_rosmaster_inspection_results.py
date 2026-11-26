"""Verify the bounded paired-inspection receipts and report source-level effects."""

import argparse
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path

from audit_primary_pairs import exact_binomial_interval
from mobile_inspection_case import verify_case
from run_rosmaster_high_resolution import OUTPUT, ROOT, remaining
from smart_swarm.episode import atomic, load, sha
from smart_swarm.inspection_comparison import ARMS, INSPECT, summarize


def verified_attempts(root, plan, report):
    entries = report["attempts"]
    sources = {recipe["source_group"]: recipe for recipe in plan["sources"]}
    if len(sources) != len(plan["sources"]):
        raise ValueError("The source allocation contains duplicate groups")
    visited = [row["source_group"] for row in entries]
    if visited != list(sources)[:len(entries)]:
        raise ValueError("The published attempts differ from the original source order")
    for row in entries:
        group = row["source_group"]
        path = root / group
        if load(path / "receipt.json") != row or any(row[key] != value for key, value in sources[group].items()):
            raise ValueError(f"Published receipt or source binding changed for {group}")
        if not isfinite(row["wall_seconds"]) or row["wall_seconds"] <= 0:
            raise ValueError(f"Invalid case wall charge for {group}")
        if row["status"] == "sealed":
            verify_case(path, row)
            if not row["evaluation"]["paired_scoring_valid"]:
                raise ValueError(f"Sealed pair failed its scoring gate for {group}")
        elif row["status"] not in ("failed", "interrupted") or "evaluation" in row:
            raise ValueError(f"Unexpected terminal receipt for {group}")
    for key, value in summarize(len(sources), entries).items():
        if report[key] != value:
            raise ValueError(f"The original comparison aggregate changed for {key}")
    if abs(report["remaining_wall_seconds"] - remaining(plan, entries)) > 1e-8:
        raise ValueError("The published remaining budget disagrees with the verified receipts")
    return entries


def paired_effect(rows, allocated):
    count = len(rows)
    wins = sum(row["scan"]["correct"] and not row["stationary"]["correct"] for row in rows)
    losses = sum(row["stationary"]["correct"] and not row["scan"]["correct"] for row in rows)
    missing = allocated - count
    net = wins - losses
    interval = None
    if count:
        win_low, win_high = exact_binomial_interval(wins, count)
        loss_low, loss_high = exact_binomial_interval(losses, count)
        interval = [win_low - loss_high, win_high - loss_low]
    return {"allocated": allocated, "scored_pairs": count, "scan_only_correct": wins,
            "stationary_only_correct": losses, "ties": count - wins - losses,
            "paired_difference": net / count if count else None, "conservative_paired_95pct": interval,
            "allocation_difference_bounds": [(net - missing) / allocated, (net + missing) / allocated],
            "branches": {arm: {field: sum(row[arm][field] for row in rows)
                               for field in ("answered", "correct", "confident_wrong")} for arm in ARMS}}


def summarize_scope(recipes, entries):
    sources = {row["source_group"] for row in recipes}
    attempted = [row for row in entries if row["source_group"] in sources]
    valid = [row["evaluation"] for row in attempted if row["status"] == "sealed"]
    methods = sorted({kind for evaluation in valid for kind in evaluation["methods"]})
    report = {"attempted": len(attempted), "valid_pairs": len(valid), "unattempted": len(sources) - len(attempted),
              "excluded": [{key: row[key] for key in ("source_group", "status", "reason", "wall_seconds")}
                           for row in attempted if row["status"] != "sealed"],
              "optical": paired_effect([row["optical"] for row in valid], len(sources)), "methods": {}}
    for kind in methods:
        rows = [row["methods"][kind] for row in valid]
        report["methods"][kind] = {**paired_effect(rows, len(sources)),
                                   "initial_inspections": sum(row["scan"]["initial_choice"] == INSPECT for row in rows),
                                   "initial_correct": sum(row["scan"]["initial_choice"] == evaluation["location"]
                                                          for row, evaluation in zip(rows, valid, strict=True))}
    return report


def optical_rows(root, entries):
    outcomes = []
    for row in entries:
        if row["status"] == "sealed":
            group = row["source_group"]
            path = root / group
            policy = load(path / "policy.json")
            decisions = load(path / "pre-inspection-decision.json")["learned"]
            readouts = {arm: {**policy[arm]["post_inspection_decision"],
                             "single_view_target_cues": [sample["readout"]["target_color_detected"]
                                                         for sample in policy[arm]["samples"]]} for arm in ARMS}
            scan = policy["scan"]
            drive = scan["motion"]["drive"]
            outcomes.append({"source_group": group, "family": row["family"], "variant": row["variant"],
                             "location": row["evaluation"]["location"], "readouts": readouts,
                             "pre_inspection": decisions, "policy_outcomes": row["evaluation"]["methods"],
                             "exposure_elapsed_ns": scan["stop_ns"] - scan["start_ns"],
                             "scan_base_offset_m": drive["measured_xy_m"],
                             "scan_peak_speed_m_s": drive["max_measured_speed_m_s"],
                             "scan_max_arm_error_deg": max(point["max_error_deg"] for point in scan["motion"]["arm"]),
                             "paired_case_wall_seconds": row["wall_seconds"]})
    return outcomes


def audit(root):
    design_file = root / "design.json"
    plan = load(design_file)
    data = (root / "comparison.json").read_bytes()
    report = json.loads(data)
    if report["design_sha256"] != sha(design_file) or any(
            sha(ROOT / name) != digest for name, digest in plan["implementation_sha256"].items()):
        raise ValueError("The comparison is not bound to the frozen design or implementation")
    if report["physical_motion_requested"] is not False or report["optimizer_updates"] != 0:
        raise ValueError("The source report disagrees with the simulation-only design")
    entries = verified_attempts(root, plan, report)
    recipes = plan["sources"]
    scopes = {"all": recipes, **{family: [row for row in recipes if row["family"] == family]
                                 for family in ("F1", "F6")}}
    charged = plan["charged_development_wall_seconds"] + sum(row["wall_seconds"] for row in entries)
    return {"scope": plan["scope"], "status": report["status"], "design_sha256": report["design_sha256"],
            "comparison_sha256": sha256(data).hexdigest(), "remaining_wall_seconds": remaining(plan, entries),
            "charged_wall_seconds": charged, "wall_limit_seconds": plan["limits"]["cumulative_wall_seconds"],
            "unattempted_sources": [row["source_group"] for row in recipes[len(entries):]],
            "post_inspection_resolution": plan["post_inspection_resolution"], "optimizer_updates": 0,
            "exposure_offsets_ns": plan["exposure_offsets_ns"], "hardware_motion": False,
            "scopes": {name: summarize_scope(source, entries) for name, source in scopes.items()},
            "optical_rows": optical_rows(root, entries),
            "receipt_sha256": {row["source_group"]: sha(root / row["source_group"] / "receipt.json") for row in entries},
            "interval_method": "Conservative simultaneous exact binomial bounds on paired wins and losses among sealed pairs.",
            "allocation_bounds": "Lower and upper possible complete-allocation differences with each unscored pair allowed any outcome."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=OUTPUT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    atomic(args.output, audit(args.root))
    print(args.output)


if __name__ == "__main__":
    main()
