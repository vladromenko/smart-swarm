"""Matched intervention scoring for the separate simulated inspection study."""

import numpy as np

INSPECT = "request_inspection"
LOCATIONS = ("left", "right", "surface")
ARMS = ("stationary", "scan")
MAX_DRIFT_M = .005
MAX_TRACKING_ERROR_DEG = 1.


def follow_up(decision, branch):
    if decision != INSPECT:
        return {"initial_choice": decision, "final_choice": decision, "extra_rgb_frames": 0, "elapsed_ns": 0}
    result = branch["post_inspection_decision"]
    choice = result["answer"] if result["answer"] in LOCATIONS else INSPECT
    return {"initial_choice": decision, "final_choice": choice, "extra_rgb_frames": len(branch["samples"]),
            "elapsed_ns": branch["stop_ns"] - branch["start_ns"]}


def score_pair(expected, policy, evaluator):
    scan, control = policy["scan"], policy["stationary"]
    visible_at = lambda branch: [row["at_ns"] for row in branch["samples"]]
    clock_match = (scan["start_ns"] == control["start_ns"] and scan["stop_ns"] == control["stop_ns"]
                   and visible_at(scan) == visible_at(control) and len(visible_at(scan)) == 2)
    restored = all(row["reset_exact"] for row in evaluator.values())
    endpoints = all(row["before"]["location"] == row["after"]["location"] == expected
                    and row["before"]["relation"] == row["after"]["relation"]
                    and row["target_displacement_m"] <= MAX_DRIFT_M for row in evaluator.values())
    tracking = all(row["motion"]["max_tracking_error_deg"] <= MAX_TRACKING_ERROR_DEG for row in scan["samples"])
    valid = expected in LOCATIONS and restored and clock_match and endpoints and tracking
    return {"paired_scoring_valid": valid, "reset_exact": restored, "exposure_clock_equal": clock_match,
            "target_relation_stable": endpoints, "tracking_passed": tracking, "location": expected,
            "optical": {arm: score_choice(policy[arm]["post_inspection_decision"]["answer"], expected, valid)
                        for arm in ARMS},
            "methods": {kind: {arm: {**row, **score_choice(row["final_choice"], expected, valid)}
                               for arm, row in branches.items()} for kind, branches in policy["methods"].items()},
            "retrieval_attempted": False, "actual_retrieval_success": None, "assistance": []}


def score_choice(choice, expected, valid):
    answered = choice in LOCATIONS
    return {"answered": answered, "correct": (choice == expected) if valid else None,
            "confident_wrong": (answered and choice != expected) if valid else None}


def paired_rate(rows):
    direct = [row["scan"]["correct"] for row in rows]
    stationary = [row["stationary"]["correct"] for row in rows]
    if not rows:
        return {"valid_pairs": 0, "scan_correct": 0, "stationary_correct": 0, "difference": None, "paired_95pct": None}
    difference = np.subtract(direct, stationary, dtype=float)
    rng = np.random.default_rng(20261126)
    draws = rng.choice(difference, size=(5_000, len(difference)), replace=True).mean(axis=1)
    return {"valid_pairs": len(rows), "scan_correct": sum(direct), "stationary_correct": sum(stationary),
            "difference": float(difference.mean()), "paired_95pct": np.quantile(draws, [.025, .975]).tolist(),
            "scan_only_correct": int(np.count_nonzero(difference == 1)),
            "stationary_only_correct": int(np.count_nonzero(difference == -1))}


def summarize(planned, attempts):
    if len({row["source_group"] for row in attempts}) != len(attempts):
        raise ValueError("An inspection source group may appear only once")
    valid = [row for row in attempts if row.get("evaluation", {}).get("paired_scoring_valid")]
    scopes = {"all": valid, **{family: [row for row in valid if row["family"] == family] for family in ("F1", "F6")}}
    methods = sorted({kind for row in valid for kind in row["evaluation"]["methods"]})
    return {"planned_independent_groups": planned, "attempted_groups": len(attempts), "valid_pairs": len(valid),
            "optical": {family: paired_rate([row["evaluation"]["optical"] for row in scope])
                        for family, scope in scopes.items()},
            "methods": {kind: {family: paired_rate([row["evaluation"]["methods"][kind] for row in scope])
                               for family, scope in scopes.items()} for kind in methods},
            "excluded_sources": [row["source_group"] for row in attempts if row not in valid]}
