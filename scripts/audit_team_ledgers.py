#!/usr/bin/env python3
"""Replay deterministic and calibrated particle ledgers on three team sizes."""

import argparse
from collections import defaultdict
from pathlib import Path

from audit_memory_validation import accepted_sources
from capture_memory_validation import DEFAULT_OUTPUT, ROOT, validation_slots
from calibrate_team_views import group_metrics
from smart_swarm.episode import LABELS, atomic, load, rows, sha
from smart_swarm.evidence_comparison import COUNTS, LABELS as EVENTS, MEAN_TV_TARGET
from smart_swarm.evidence_inputs import runtime_signature
from smart_swarm.evidence_queries import RECEIVER, WITNESSES
from smart_swarm.memory import admissible, predict, serialized_cost
from smart_swarm.particle_memory import FACTORS, exact_filter, particle_filter, total_variation
from smart_swarm.team_evidence import pool_likelihoods

PILOT = ROOT / "evidence/learning-pilot-v3"


def ledger_observations(root):
    scope = runtime_signature(root)
    delivered, bound = scope["delivered_messages"], scope["decision_ns"]
    evidence = admissible(delivered, bound, RECEIVER)
    reports = [(item["origin_agent"], item["factor"],
                max(item["hypotheses"], key=item["hypotheses"].get)) for item in evidence]
    return {"deterministic": predict(delivered, bound, RECEIVER), "reports": reports,
            "serialized_bytes": serialized_cost(scope["query"], delivered)["total_bytes"]}


def group_input(folder, only=None):
    allocation = {slot["source_group"]: slot for slot in validation_slots()}
    for data in accepted_sources(folder, allocation, only):
        for team in data["teams"]:
            cases = []
            for trace in team["traces"]:
                root = Path(trace["path"])
                observed = ledger_observations(root)
                labels = rows(root / LABELS / "settled-event.jsonl")[0]
                cases.append({**observed, "source": str(root), "truth": trace["terminal_outcome"], "family": trace["family"],
                              "variant": trace["variant"], "labels": labels})
            yield {"group": data["source_group"], "team_size": team["actor_count"], "outcomes": cases}


def learn_probabilities(groups):
    counts = defaultdict(lambda: defaultdict(lambda: [.5, .5]))
    support = defaultdict(lambda: defaultdict(set))
    for group in groups:
        for case in group["outcomes"]:
            for sender, factor, report in case["reports"]:
                label = case["labels"][EVENTS[factor]]
                if sender not in WITNESSES and factor == "association" and label in FACTORS[factor]:
                    counts[sender][label][FACTORS[factor].index(report)] += 1. / len(group["outcomes"])
                    support[sender][label].add(group["group"])
    result = {}
    for sender, values in counts.items():
        if all(len(support[sender][label]) >= 2 for label in FACTORS["association"]):
            classes = FACTORS["association"]
            result[sender] = {label: dict(zip(classes, (item / sum(bucket) for item in bucket), strict=True))
                              for label, bucket in values.items()}
    return result


def factors(reported, profile, mounted):
    evidence = defaultdict(list)
    for sender, factor, observed in reported:
        model = profile["report_likelihoods"][factor] if sender in WITNESSES else mounted.get(sender)
        if model is not None:
            evidence[factor].append({label: row[observed] ** profile["observation_power"]
                                     for label, row in model.items()})
    return pool_likelihoods(evidence)


def with_predictions(group, candidates, prior, profile, mounted):
    output = []
    for row in group["outcomes"]:
        evidence = factors(row["reports"], profile, mounted)
        exact = exact_filter(prior, evidence)
        estimates = {count: particle_filter(prior, evidence, count, seed=profile["seed"]) for count in candidates}
        errors = {count: total_variation(exact["joint"], sampled["joint"]) for count, sampled in estimates.items()}
        deterministic = {**row["deterministic"], "serialized_bytes": row["serialized_bytes"]}
        output.append({"source": row["source"], "truth": row["truth"], "family": row["family"], "variant": row["variant"],
                       "deterministic": deterministic, "estimates": estimates, "exact": exact,
                       "joint_tv": errors, "serialized_bytes": row["serialized_bytes"]})
    return {**group, "outcomes": output}


def mean_tv(groups, count):
    return sum(sum(row["joint_tv"][count] for row in item["outcomes"]) / len(item["outcomes"])
               for item in groups) / len(groups)


def particle_decision(row, count, threshold):
    state = row["estimates"][count]
    best = max(state["belief"], key=state["belief"].get)
    choice = best if state["belief"][best] >= threshold else "request_inspection"
    return {**state, "decision": choice, "serialized_bytes": row["serialized_bytes"]}


def family_report(groups, prior, profile):
    crossfit = []
    for group in groups:
        learning = [item for item in groups if item["group"] != group["group"]]
        mounted = learn_probabilities(learning)
        crossfit.append(with_predictions(group, COUNTS, prior, profile, mounted))
    tv = {count: mean_tv(crossfit, count) for count in COUNTS}
    count = next((count for count in COUNTS if tv[count] <= MEAN_TV_TARGET), COUNTS[-1])
    for group in crossfit:
        for row in group["outcomes"]:
            row["particle"] = particle_decision(row, count, profile["threshold"])
            row.pop("estimates")
        if group["outcomes"][0]["family"] == "F6":
            a, b = group["outcomes"]
            for method in ("deterministic", "particle"):
                if any(a[method][field] != b[method][field] for field in ("belief", "decision", "serialized_bytes")):
                    raise ValueError(f"{method} changed within a matched F6 pair")
    mounted = learn_probabilities(groups)
    return {"selected_particle_count": count, "validation_mean_joint_tv": tv,
            "metrics": {name: group_metrics(crossfit, name) for name in ("deterministic", "particle")},
            "additional_mounts_with_coverage": sorted(mounted), "mounted_report_likelihoods": mounted,
            "outcomes": crossfit}


def audit(folder, only=None):
    groups = list(group_input(folder, only))
    study = load(PILOT / "story-learning.json")
    path = Path(study["calibrated_particle_report"])
    profile = load(path)
    prior = next(value["state_prior"] for value in study["models"] if value["kind"] == "events_only")
    return {"scope": "Folded validation of recorded RGB rule messages only",
            "base_particle_report_sha256": sha(path), "modeled_transport": "Native simulator messages with opaque source IDs",
            "extra_mount_calibration": "Grouped confusion rows learned only from other validation sources",
            "teams": {size: family_report([row for row in groups if row["team_size"] == size], prior, profile)
                      for size in (3, 4, 8)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    atomic(args.output, audit(args.root))
    print(args.output)


if __name__ == "__main__":
    main()
