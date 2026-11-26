#!/usr/bin/env python3
"""Fit mount-specific likelihood scales and cross-check them by independent source."""

import argparse
from collections import defaultdict
import math
from pathlib import Path

from audit_evidence_replay import verified_model
from audit_memory_validation import accepted_sources
from capture_memory_validation import DEFAULT_OUTPUT, ROOT, validation_slots
from smart_swarm.episode import LABELS, atomic, load, rows, sha
from smart_swarm.evidence_comparison import learned_prior_map
from smart_swarm.memory import PERMUTATIONS
from smart_swarm.particle_memory import exact_filter
from smart_swarm.team_evidence import ANCHOR_TANK, pooled_factors
from smart_swarm.view_calibration import calibrated_replies, fit

PILOT = ROOT / "evidence/learning-pilot-v3"
BROADCAST = "broadcast_8192"


def admitted_groups(validation, replay):
    allocated = {slot["source_group"]: slot for slot in validation_slots()}
    admitted = {}
    for source in accepted_sources(validation, allocated):
        for team in source["teams"]:
            key = (source["seed"], team["actor_count"])
            paths = tuple(trace["path"] for trace in team["traces"])
            admitted[key] = (source["source_group"], paths)
    if len(replay) != len(admitted):
        raise ValueError("Frozen replay and admitted validation coverage differ")
    seen = set()
    for item in replay:
        key = (item["seed"], item["team_size"])
        if key not in admitted or key in seen:
            raise ValueError("Replay contains an unapproved or duplicate validation team")
        group, paths = admitted[key]
        if paths != tuple(row["source"] for row in item["episodes"]):
            raise ValueError("Replay used records outside its validation grant")
        seen.add(key)
        yield {**item, "source_group": group}


def association_samples(records):
    streams = defaultdict(lambda: defaultdict(dict))
    for record in records:
        key = record["source_group"]
        for episode in record["episodes"]:
            root = Path(episode["source"])
            label = rows(root / LABELS / "settled-event.jsonl")[0]["relocation_assignment"]
            if label != episode["scoring_association_label"]:
                raise ValueError("Replay association label differs from the accepted record")
            if label in PERMUTATIONS:
                for agent, posterior in episode["frozen_scores_by_witness"].items():
                    if set(posterior) == set(PERMUTATIONS):
                        groups = streams[record["team_size"]][agent]
                        value = (label, posterior)
                        if key in groups and groups[key] != value:
                            raise ValueError("A source repeated with inconsistent labels or predictions")
                        groups[key] = value
    return streams


def profiles(streams, reference, heldout=None):
    return {agent: fit({group: value for group, value in samples.items() if group != heldout}, reference)
            for agent, samples in sorted(streams.items())}


def prediction(row, models, reference, frozen):
    original = row["policies"][BROADCAST]
    sent = original["wire"]["replies"]
    usable = calibrated_replies(sent, models, reference["association"], ANCHOR_TANK)
    state = exact_filter(frozen["state_prior"], pooled_factors(usable, reference))
    best = max(state["belief"], key=state["belief"].get)
    choice = best if state["belief"][best] >= frozen["threshold"] else "request_inspection"
    return {**state, "decision": choice, "used_senders": [reply["sender"] for reply in usable],
            "serialized_bytes": original["serialized_bytes"]}


def crossfit(records, streams, reference, frozen):
    results = []
    for item in records:
        source, size = item["source_group"], item["team_size"]
        models = profiles(streams.get(size, {}), reference["association"], heldout=source)
        outcomes = []
        for episode in item["episodes"]:
            raw = episode["policies"][BROADCAST]
            calibrated = prediction(episode, models, reference, frozen)
            outcomes.append({"family": episode["family"], "variant": episode["variant"], "truth": episode["truth"],
                             "raw": {field: raw[field] for field in ("belief", "decision", "serialized_bytes")},
                             "calibrated": calibrated})
        if outcomes[0]["family"] == "F6":
            a, b = [row["calibrated"] for row in outcomes]
            if a != b or a["decision"] != "request_inspection" or not item["f6_verified"]:
                raise ValueError("View calibration changed matched F6 behavior")
        results.append({"group": source, "team_size": size, "fit_profiles": models, "outcomes": outcomes})
    return results


def group_metrics(grouped, policy):
    vals = [row for group in grouped for row in group["outcomes"]]
    controls = [row for row in vals if row["family"] in ("F1", "F2", "F3", "F5")]
    container = [row for row in controls if row["truth"] in ("left", "right")]
    f6 = [row for row in vals if row["family"] == "F6"]
    losses = [-sum(math.log(max(row[policy]["belief"][row["truth"]], 1e-12)) for row in group["outcomes"])
              / len(group["outcomes"]) for group in grouped]
    return {"independent_groups": len(grouped), "variant_count": len(vals), "resolvable_count": len(controls),
            "resolvable_correct": sum(row[policy]["decision"] == row["truth"] for row in controls),
            "container_count": len(container),
            "container_correct": sum(row[policy]["decision"] == row["truth"] for row in container),
            "f6_variants": len(f6), "f6_inspections": sum(row[policy]["decision"] == "request_inspection" for row in f6),
            "confident_wrong": sum(row[policy]["decision"] not in (row["truth"], "request_inspection") for row in vals),
            "group_mean_nll": sum(losses) / len(losses)}


def calibrate(replay_path, validation):
    study_path = PILOT / "story-learning.json"
    replay = load(replay_path)
    study = load(study_path)
    kind = replay.get("checkpoint_kind", "events_only")
    model = next(row for row in study["models"] if row["kind"] == kind)
    if (replay["study_sha256"] != sha(study_path) or replay["checkpoint_sha256"] != model["checkpoint_sha256"]
            or replay["scope"] != "Primary simulator validation only"):
        raise ValueError("Use a frozen replay of accepted simulator validation sources")
    frozen, _ = verified_model(model)
    reference = learned_prior_map(frozen["priors"])
    records = list(admitted_groups(validation, replay["cohorts"]))
    samples = association_samples(records)
    heldout = crossfit(records, samples, reference, frozen)
    by_n = {size: [row for row in heldout if row["team_size"] == size] for size in (3, 4, 8)}
    return {"scope": "Source-group cross-fit on prospective simulator validation, no primary test or optimizer",
            "replay_sha256": sha(replay_path), "base_checkpoint_kind": kind,
            "base_checkpoint_sha256": model["checkpoint_sha256"],
            "source_counts": len({row["source_group"] for row in records}),
            "full_validation_profiles": {size: profiles(samples[size], reference["association"]) for size in by_n},
            "crossfit_scores": {size: {name: group_metrics(grouped, name) for name in ("raw", "calibrated")}
                                for size, grouped in by_n.items()},
            "crossfit_outcomes": heldout}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, default=PILOT / "scaling-team-evidence-validation.json")
    parser.add_argument("--validation-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = calibrate(args.replay, args.validation_root)
    atomic(args.output, result)
    print(result["source_counts"], "independent groups", args.output)


if __name__ == "__main__":
    main()
