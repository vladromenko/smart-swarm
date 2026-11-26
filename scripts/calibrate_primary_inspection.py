#!/usr/bin/env python3
"""Audit decision thresholds using grouped validation outcomes only."""

import argparse
from pathlib import Path

from capture_memory_validation import CONFIG, validation_slots
from smart_swarm.episode import atomic, load, sha
from smart_swarm.evidence_comparison import THRESHOLDS

IDENTITY = "selective_with_observed_marker_support"
RESOLVABLE = ("F1", "F2", "F3", "F5")
INSPECT = "request_inspection"


def admitted_matrix(path):
    matrix = load(path)
    if (matrix["scope"] != "Seven-method repeated-source simulator validation rehearsal"
            or matrix["config_sha256"] != sha(CONFIG)):
        raise ValueError("Use admitted primary simulator validation outcomes")
    allowed = {slot["source_group"] for slot in validation_slots()}
    seen = set()
    for group in matrix["outcomes"]:
        key = (group["group"], group["team_size"])
        if key[0] not in allowed or key in seen:
            raise ValueError("Duplicate or unreserved source-team pair")
        for row in group["outcomes"]:
            manifest = Path(row["source"]) / "manifest.json"
            info = load(manifest)
            if (info["scene_group_id"] != key[0] or info["lineage"]["split"] != "validation"
                    or sha(manifest) != row["source_manifest_sha256"]):
                raise ValueError("A source escaped its validation manifest")
        seen.add(key)
    return matrix


def strongest(row, method):
    belief = row[method]["belief"]
    winner = max(belief, key=belief.get)
    return winner, belief[winner]


def threshold_score(groups, method, threshold, config):
    count = {"correct_resolvable": 0, "resolvable": 0, "false_decisions": 0, "f6_inspections": 0,
             "f6_variants": 0, "f4_inspections": 0, "f4_false_decisions": 0}
    total = 0.
    for group in groups:
        loss = 0.
        for row in group["outcomes"]:
            winner, probability = strongest(row, method)
            decision = winner if probability >= threshold else INSPECT
            inspect = decision == INSPECT
            wrong = not inspect and decision != row["truth"]
            resolvable = row["family"] in RESOLVABLE
            count["correct_resolvable"] += resolvable and decision == row["truth"]
            count["resolvable"] += resolvable
            count["false_decisions"] += wrong
            count["f6_variants"] += row["family"] == "F6"
            count["f6_inspections"] += row["family"] == "F6" and inspect
            count["f4_inspections"] += row["family"] == "F4" and inspect
            count["f4_false_decisions"] += row["family"] == "F4" and wrong
            loss += config["inspect_cost"] if inspect else config["wrong_answer_cost"] * wrong
        total += loss / len(group["outcomes"])
    return {**count, "group_mean_cost": total / len(groups)}


def feasibility(groups, method):
    lower, upper = [], []
    for group in groups:
        for row in group["outcomes"]:
            winner, confidence = strongest(row, method)
            if row["family"] == "F6" or winner != row["truth"]:
                lower.append(confidence)
            if row["family"] in RESOLVABLE and winner == row["truth"]:
                upper.append(confidence)
    floor = max(lower, default=None)
    ceiling = min(upper, default=None)
    return {"must_exceed_to_avoid_all_wrong_and_inspect_f6": floor,
            "at_most_to_keep_every_correct_resolvable_argmax": ceiling,
            "feasible_for_all_correct_controls": bool(ceiling is not None and (floor is None or floor < ceiling))}


def method_report(groups, method, config):
    scores = {value: threshold_score(groups, method, value, config) for value in THRESHOLDS}
    best = min(THRESHOLDS, key=lambda value: (scores[value]["group_mean_cost"], scores[value]["false_decisions"],
                                            -scores[value]["f6_inspections"], -scores[value]["correct_resolvable"],
                                            abs(value - .75)))
    return {"selected_validation_threshold": best, "scores": {str(value): result for value, result in scores.items()},
            "all_correct_control_threshold_range": feasibility(groups, method)}


def run(path):
    source = admitted_matrix(path)
    config = load(CONFIG)
    methods = list(config["primary_methods"])
    if "marker_support_diagnostic" in source:
        methods.append(IDENTITY)
    grouped = {size: [row for row in source["outcomes"] if row["team_size"] == size] for size in config["team_sizes"]}
    selected = {row["group"] for row in source["outcomes"]}
    if any({row["group"] for row in groups} != selected for groups in grouped.values()):
        raise ValueError("The validation threshold study requires matched group coverage across team sizes")
    return {"status": "validation_threshold_diagnostic", "matrix_sha256": sha(path),
            "checkpoint_kind": source["checkpoint_kind"], "independent_source_groups": len(selected),
            "allocated_validation_groups": len(validation_slots()),
            "scope": "Threshold selection and scoring use the same validation cohort; independent test is untouched.",
            "teams": {size: {name: method_report(groups, name, config) for name in methods}
                      for size, groups in grouped.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    atomic(args.output, run(args.matrix))
    print(args.output)


if __name__ == "__main__":
    main()
