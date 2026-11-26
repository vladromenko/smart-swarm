#!/usr/bin/env python3
"""Compare receiver-local current and temporal RGB cues on reserved validation."""

import argparse
from pathlib import Path

from calibrate_team_views import PILOT, admitted_groups, group_metrics
from audit_evidence_replay import receiver_rgb, verified_model
from capture_memory_validation import DEFAULT_OUTPUT
from smart_swarm.episode import Replay, atomic, load, sha
from smart_swarm.evidence_queries import RECEIVER
from smart_swarm.particle_memory import exact_filter

CUE_COUNTS = {"current": 2, "history": 3}


def cues(root):
    observed = receiver_rgb(Replay(root, RECEIVER))
    latest = observed["current_rgb"]
    earlier = observed["local_rgb_history"]
    if latest is None:
        return dict.fromkeys(CUE_COUNTS)
    return {"current": int(latest), "history": 2 if latest else int(earlier)}


def attach_local_cues(groups):
    for group in groups:
        records = [{**row, "local_rgb_cues": cues(Path(row["source"]))} for row in group["episodes"]]
        yield {**group, "episodes": records}


def fit_cue(groups, mode):
    counts = {category: [.5] * CUE_COUNTS[mode] for category in ("surface", "other")}
    for group in groups:
        for row in group["episodes"]:
            value = row["local_rgb_cues"][mode]
            if value is not None:
                category = "surface" if row["truth"] == "surface" else "other"
                counts[category][value] += 1. / len(group["episodes"])
    return {category: [value / sum(terms) for value in terms] for category, terms in counts.items()}


def predict(prior, weights, cue, threshold):
    posterior = dict(prior)
    if cue is not None:
        likelihood = {name: weights["surface" if name == "surface" else "other"][cue] for name in prior}
        unnormalized = {name: prior[name] * likelihood[name] for name in prior}
        total = sum(unnormalized.values())
        posterior = {name: value / total for name, value in unnormalized.items()}
    best = max(posterior, key=posterior.get)
    return {"belief": posterior, "decision": best if posterior[best] >= threshold else "request_inspection",
            "serialized_bytes": 0}


def crossfit(groups, prior, threshold):
    outcomes = []
    for group in groups:
        training = [item for item in groups if item["source_group"] != group["source_group"]]
        fits = {mode: fit_cue(training, mode) for mode in CUE_COUNTS}
        scored = []
        for row in group["episodes"]:
            predictions = {mode: predict(prior, fits[mode], row["local_rgb_cues"][mode], threshold)
                           for mode in CUE_COUNTS}
            scored.append({**predictions, "family": row["family"], "variant": row["variant"], "truth": row["truth"],
                           "local_rgb_cues": row["local_rgb_cues"]})
        if scored[0]["family"] == "F6":
            a, b = scored
            if any(a[field] != b[field] for field in ("current", "history", "local_rgb_cues")) or not group["f6_verified"]:
                raise ValueError("Receiver-only evidence differs inside a hidden pair")
        outcomes.append({"group": group["source_group"], "outcomes": scored, "fit_profiles": fits})
    return outcomes


def inspection_tradeoff(groups, mode):
    observed = [row for group in groups for row in group["outcomes"]]
    paired = [max(row[mode]["belief"].values()) for row in observed if row["family"] == "F6"]
    correct = []
    for row in observed:
        posterior = row[mode]["belief"]
        winner = max(posterior, key=posterior.get)
        if row["family"] in ("F1", "F2", "F3", "F5") and winner == row["truth"]:
            correct.append(posterior[winner])
    above = max(paired, default=None)
    at_most = max(correct, default=None)
    feasible = None if above is None else bool(at_most is not None and above < at_most)
    return {"inspection_threshold_must_exceed_for_all_f6": above,
            "threshold_at_most_for_one_correct_resolvable_answer": at_most,
            "single_threshold_can_do_both": feasible}


def run(path, validation):
    study_path = PILOT / "story-learning.json"
    replay = load(path)
    study = load(study_path)
    model = next(row for row in study["models"] if row["kind"] == "events_only")
    if replay["study_sha256"] != sha(study_path) or replay["scope"] != "Primary simulator validation only":
        raise ValueError("Use admitted primary simulator validation replay")
    frozen, _ = verified_model(model)
    prior = exact_filter(frozen["state_prior"], [])["belief"]
    groups = list(attach_local_cues(admitted_groups(validation, replay["cohorts"])))
    scores = {}
    for size in (3, 4, 8):
        examples = [row for row in groups if row["team_size"] == size]
        results = crossfit(examples, prior, frozen["threshold"])
        scores[size] = {"metrics": {mode: group_metrics(results, mode) for mode in CUE_COUNTS},
                        "inspection_tradeoff": {mode: inspection_tradeoff(results, mode) for mode in CUE_COUNTS},
                        "crossfit_outcomes": results}
    return {"scope": "Receiver-private RGB, same threshold and no messages, validation by independent source",
            "description": "Current cue uses visible orange at the final view. History separates earlier-only visibility.",
            "replay_sha256": sha(path), "decision_threshold": frozen["threshold"], "teams": scores}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, default=PILOT / "scaling-team-evidence-validation.json")
    parser.add_argument("--validation-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.replay, args.validation_root)
    atomic(args.output, result)
    print(args.output)


if __name__ == "__main__":
    main()
