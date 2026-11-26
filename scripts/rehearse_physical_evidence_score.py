#!/usr/bin/env python3
"""Verify physical development scoring with already-saved calibration forecasts."""

import argparse
from pathlib import Path

from smart_swarm.cli import ROOT, settings
from smart_swarm.episode import atomic, load, rows, sha
from smart_swarm.physical_evidence_dataset import DECISION_FILE
from smart_swarm.physical_evidence_score import matched_comparison, score_current, scoring_case

ARCHIVE = ROOT / "evidence/learning-pilot-v3/physical-learned-archive-rehearsal.json"
ARCHIVE_SHA256 = "3cb97403cac3204967db88173fa48c584397558705da77a898e945bbcd7191de"
CALIBRATION = ROOT.parent.parent / "work/primary-sim-validation"


def calibration_case(family, case):
    root = Path(case["source"]).resolve()
    if not root.is_relative_to(CALIBRATION.resolve()):
        raise ValueError("Only the archived simulation calibration sources may be used in this rehearsal")
    manifest = load(root / "manifest.json")
    path = root / DECISION_FILE
    if sha(path) != manifest["sidecar_sha256"][DECISION_FILE]:
        raise ValueError("The archived calibration rule prediction changed")
    saved, = rows(path)
    if saved["at"] != manifest["decision_boundary"]:
        raise ValueError("The archived calibration rule prediction lost its boundary")
    record = {"story_id": manifest["world_episode_id"], "scene_group_id": manifest["scene_group_id"],
              "split": "development", "family": family, "labels": {"terminal": case["terminal_truth_for_scoring"]}}
    decision = {"query_all": saved["query_all"], "learned_evidence": {"status": "scored", "receipt": case["prediction"]}}
    return scoring_case(record, decision)


def rehearsal(config):
    if sha(ARCHIVE) != ARCHIVE_SHA256:
        raise ValueError("The prior archive rehearsal must retain its original receipt")
    saved = load(ARCHIVE)
    cases = [calibration_case(family, case) for family, entries in saved["simulation"].items() for case in entries]
    comparison = matched_comparison(cases)
    if (comparison["labeled_matched_histories"] != 3 or comparison["labeled_matched_scene_groups"] != 2
            or any(method["correct_answers"] != 1 or method["inspections"] != 2
                   for method in comparison["metrics_by_method"].values())):
        raise ValueError("The frozen resolvable control and F6 pair must reproduce their saved behavior")
    native = saved["native"]
    if any(row["technical_only"] is not True or not row["prediction_saved_and_reopened_before_reveal"]
           for row in native.values()):
        raise ValueError("Native archived inputs must remain technical and pre-reveal")
    return {"scope": "physical evaluation software rehearsal on archived simulation calibration controls",
            "physical_development_status": score_current(config),
            "calibration_controls_only": comparison, "calibration_cases": cases,
            "technical_native_archives_excluded": list(native), "original_archive_sha256": ARCHIVE_SHA256,
            "optimizer_updates": 0, "native_motion_requested": False, "native_recording_started": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Choose a new immutable development scoring rehearsal receipt")
    result = rehearsal(settings())
    atomic(args.out, result)
    print(result["physical_development_status"]["status"])


if __name__ == "__main__":
    main()
