#!/usr/bin/env python3
"""Verify paired high-resolution development replays and their cumulative study charge."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from audit_primary_scope import paired
from compare_rosmaster_inspection import KINDS, LIMITS
from mobile_inspection_case import verify_case
from smart_swarm.cli import ROOT
from smart_swarm.episode import atomic, load, sha
from smart_swarm.inspection_triangulation import locate

WORK = ROOT.parent.parent / "work"
VARIANTS = ("visible-control", "hidden-contained", "hidden-surface")
ORIGINAL = WORK / "active-inspection-comparison-mobile-v2"


def high_prediction(path, rows, baseline):
    images = []
    for row, original in zip(rows, baseline, strict=True):
        frame = row["frame"]
        if (frame not in ("carrier.png", "surface.png") or frame != original["frame"]
                or row["at_ns"] != original["at_ns"]
                or not all(row[key] for key in ("matches_archived_frame", "matches_archived_pose", "matches_archived_time"))):
            raise ValueError("A high-resolution image is outside its original matched exposure")
        filename = path / frame
        if sha(filename) != row["high_sha256"]:
            raise ValueError("A higher-resolution image changed after capture")
        with Image.open(filename) as image:
            if image.mode != "RGB" or image.size != (640, 480):
                raise ValueError("The high-resolution intervention must use 640 by 480 native RGB")
            images.append((np.asarray(image).copy(), original["calibration"]))
    return locate(images)


def check_variant(variant):
    root = WORK / f"rosmaster-native-resolution-{variant}-matched-v1"
    revised = WORK / f"rosmaster-native-resolution-{variant}-matched-v2"
    if (revised / "probe.json").is_file():
        root = revised
    receipt = root / "probe.json"
    probe = load(receipt)
    if (probe["status"] != "recorded" or not probe["archived_exposures_reproduced"]
            or not probe["archived_reset_state_reproduced"] or not probe["preserved_stationary_control"]
            or not probe["matched_branch_comparison"]["same_reset_and_exposure_times"]
            or probe["wall_seconds"] > LIMITS["case_wall_seconds"]):
        raise ValueError("The exact matched optical reproduction did not complete under its original cap")
    source = ORIGINAL / f"mobile-rehearsal-{variant}"
    prior = load(source / "receipt.json")
    frozen = verify_case(source, prior)
    scope = probe["pre_inspection_scope"]
    if (probe["reference_policy_sha256"] != prior["policy_sha256"]
            or scope["permitted_input_sha256"] != frozen["permitted_input_sha256"]
            or scope["permitted_rgb_frames"] != frozen["permitted_rgb_frames"]
            or scope["frozen_decision_sha256"] != sha(source / "pre-inspection-decision.json")):
        raise ValueError("The high-resolution replay changed its original pre-inspection decision")
    baseline = load(source / "policy.json")
    scan = high_prediction(root / "rgb-640", probe["two_native_resolutions"], baseline["scan"]["samples"])
    idle = high_prediction(root / "stationary/rgb-640", probe["stationary"]["two_native_resolutions"],
                           baseline["stationary"]["samples"])
    if scan != probe["high_resolution_prediction"] or idle != probe["stationary"]["high_resolution_prediction"]:
        raise ValueError("The saved optical result cannot be reproduced from its RGB inputs")
    return {"receipt": str(receipt), "receipt_sha256": sha(receipt), "wall_seconds": probe["wall_seconds"],
            "path": str(source / "pre-inspection"), "terminal_outcome": prior["evaluation"]["location"],
            "scan": scan, "stationary": idle, "learned": frozen["learned"]}


def previous_charges():
    rows = []
    for name in ("active-inspection-comparison-mobile-v1", "active-inspection-comparison-mobile-v2"):
        root = WORK / name
        receipts = load(root / "development-gate.json")["attempts"]
        if len(receipts) != 3 or any(row != load(root / row["source_group"] / "receipt.json") for row in receipts):
            raise ValueError("The previous development charges changed")
        rows.append({"root": str(root), "wall_seconds": sum(row["wall_seconds"] for row in receipts)})
    for probe in sorted(WORK.glob("rosmaster-native-resolution-*-matched-v*/probe.json")):
        rows.append({"root": str(probe.parent), "wall_seconds": load(probe)["wall_seconds"]})
    for probe in sorted(WORK.glob("rosmaster-parallel-*-v*/probe.json")):
        rows.append({"root": str(probe.parent), "wall_seconds": load(probe)["wall_seconds"]})
    return rows


def audit():
    results, refused = {}, {}
    for variant in VARIANTS:
        try:
            row = check_variant(variant)
            results[variant] = row
            if row["scan"]["answer"] != row["terminal_outcome"] or row["stationary"]["answer"] != "unresolved":
                refused[variant] = "The optical answer does not reproduce the required active and stationary contrast"
        except (OSError, KeyError, ValueError, RuntimeError) as error:
            refused[variant] = f"{type(error).__name__}: {error}"
    scope = None
    if not refused:
        shown, contained, surface = (results[name] for name in VARIANTS)
        scope = paired({"actor_count": 3, "traces": [contained, surface]})
        for kind in KINDS:
            first, second = (row["learned"][kind] for row in (contained, surface))
            if first != second or first["decision"] != "request_inspection" or shown["learned"][kind]["decision"] != shown["terminal_outcome"]:
                raise ValueError("The frozen F6 pair or resolving control did not survive the optical reproduction")
    charges = previous_charges()
    spent = sum(row["wall_seconds"] for row in charges)
    remaining = LIMITS["cumulative_wall_seconds"] - spent
    return {"scope": "exact high-resolution development reproduction", "ready": not refused, "refused": refused,
            "results": results, "permitted_f6_scope": scope, "charged_development_work": charges,
            "previous_wall_seconds": spent, "remaining_wall_seconds": remaining,
            "mean_wall_seconds_for_20_independent_cases": remaining / 20, "independent_cases_opened": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new high-resolution qualification report")
    result = audit()
    atomic(args.output, result)
    print(args.output)
    raise SystemExit(0 if result["ready"] else 2)


if __name__ == "__main__":
    main()
