#!/usr/bin/env python3
"""Rehearse uncertainty reporting on paired primary-study validation sources."""

import argparse
from pathlib import Path

from scipy.stats import beta

from calibrate_primary_inspection import admitted_matrix
from learn_evidence_v3 import paired_container_difference
from smart_swarm.episode import atomic, sha

SELECTIVE = "learned_selective_joint_memory"
COMPARATORS = ("central_compact_budget_matched", "calibrated_broadcast_joint_particle_filter")


def paired_rows(groups, method):
    results = []
    for group in groups:
        for row in group["outcomes"]:
            if row["family"] in ("F1", "F2", "F3", "F5") and row["truth"] in ("left", "right"):
                results.append({"world_episode_id": row["source"], "scene_group_id": group["group"],
                                "decision": row[method]["decision"], "terminal": row["truth"], "resolvable": True})
    return results


def exact_binomial_interval(successes, count):
    tail = .0125
    lower = 0. if successes == 0 else float(beta.ppf(tail, successes, count - successes + 1))
    upper = 1. if successes == count else float(beta.ppf(1. - tail, successes + 1, count - successes))
    return lower, upper


def conservative_interval(a, b):
    keys = [row["scene_group_id"] for row in a]
    if len(set(keys)) != len(a) or not keys:
        raise ValueError("This bound expects one resolvable container decision per independent group")
    paired = {row["world_episode_id"]: row for row in b}
    deltas = [int(row["decision"] == row["terminal"]) - int(paired[row["world_episode_id"]]["decision"] == row["terminal"])
              for row in a]
    wins, losses = deltas.count(1), deltas.count(-1)
    win_low, win_high = exact_binomial_interval(wins, len(keys))
    loss_low, loss_high = exact_binomial_interval(losses, len(keys))
    lower, upper = win_low - loss_high, win_high - loss_low
    return {"wins": wins, "ties": deltas.count(0), "losses": losses, "conservative_paired_95pct": [lower, upper],
            "interval_width": upper - lower}


def compare(matrix, methods):
    output = {}
    for size in (3, 4, 8):
        groups = [line for line in matrix["outcomes"] if line["team_size"] == size]
        a = paired_rows(groups, SELECTIVE)
        measurements = {}
        for method in methods:
            b = paired_rows(groups, method)
            described = paired_container_difference(a, b)
            described.pop("precision_target_met")
            low, high = described["bootstrap_95pct"]
            conservative = conservative_interval(a, b)
            measurements[method] = {"bootstrap_descriptive": described, **conservative,
                                    "bootstrap_degenerate": low == high,
                                    "conservative_precision_target_met":
                                        conservative["interval_width"] <= described["target_interval_width"]}
        output[size] = measurements
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    matrix = admitted_matrix(args.matrix)
    result = {"scope": "Validation-only methodological rehearsal. Independent test effect remains unmeasured.",
              "matrix_sha256": sha(args.matrix), "learner": matrix["checkpoint_kind"],
              "coverage": "Simultaneous exact binomial intervals for paired wins and losses, differenced conservatively.",
              "teams": compare(matrix, COMPARATORS)}
    atomic(args.output, result)
    print(args.output)


if __name__ == "__main__":
    main()
