#!/usr/bin/env python3
"""Exercise distinct sender responses and fixed-budget fusion on archived team worlds."""

import argparse
from pathlib import Path
import time

import torch

from audit_central_history import load_backbone
from audit_evidence_replay import ROOT, verified_model
from audit_memory_validation import accepted_sources
from capture_memory_validation import validation_slots
from smart_swarm.cli import settings
from smart_swarm.episode import LABELS, atomic, load, rows, sha
from smart_swarm.evidence_comparison import learned_prior_map
from smart_swarm.evidence_inputs import clip, runtime_signature
from smart_swarm.memory import CUBE_SUPPORT, PERMUTATIONS
from smart_swarm.research_jobs import admitted
from smart_swarm.team_evidence import exchange, opportunities
from smart_swarm.temporal_evidence import EvidenceHead

PILOT = ROOT / "evidence" / "learning-pilot-v3"
REPORTS = ("scaling-f1-fixed-total.json", "scaling-f6-private-invariance.json",
           "scaling-receiver-history-f1.json", "scaling-receiver-history-f6.json")
SETTINGS = (("anchor", 3072), ("broadcast", 8192), *(("factor_coverage", cap) for cap in (2048, 3072, 4096)))


def sources(evidence, validation):
    if validation is not None:
        allocation = {slot["source_group"]: slot for slot in validation_slots()}
        for result in accepted_sources(validation, allocation):
            for team in result["teams"]:
                yield result["seed"], team
        return
    for name in REPORTS:
        report = load(evidence / name)
        if report["status"] != "verified":
            raise ValueError(f"{name} did not pass its acquisition audit")
        for team in report["teams"]:
            yield report["seed"], team


@torch.no_grad()
def decode_features(feature, factor, head, temperature):
    if factor == "containment":
        logits, _ = head(feature, None)
        labels = CUBE_SUPPORT
    elif factor == "association":
        _, logits = head(None, feature)
        labels = PERMUTATIONS
    else:
        raise ValueError("Unknown evidence factor")
    scores = (logits / temperature).softmax(-1).tolist()
    return dict(zip(labels, scores, strict=True))


def decode(root, agent, factor, network):
    encoder, head, temperature = network
    rgb = clip(root, agent)
    if rgb is None:
        raise ValueError("An advertised sender lost its private view")
    return decode_features(encoder(rgb), factor, head, temperature)


def episode_result(trace, network, frozen, gains):
    root = Path(trace["path"])
    window = opportunities(root)
    reports = {}
    reference = learned_prior_map(frozen["priors"])

    def posterior(agent, factor):
        key = (agent, factor)
        if key not in reports:
            reports[key] = decode(root, agent, factor, network)
        return reports[key]

    policies = {}
    for method, cap in SETTINGS:
        result = exchange(method, window, cap, gains, posterior, reference, frozen["state_prior"])
        belief = result["belief"]
        best = max(belief, key=belief.get)
        decision = best if belief[best] >= frozen["threshold"] else "request_inspection"
        policies[f"{method}_{cap}"] = {**result, "decision": decision}
    label = rows(root / LABELS / "settled-event.jsonl")[0]["relocation_assignment"]
    return {"source": str(root), "truth": trace["terminal_outcome"], "variant": trace["variant"],
            "family": trace["family"], "team_size": trace["team_size"],
            "scoring_association_label": label,
            "frozen_scores_by_witness": {agent: values for (agent, _), values in sorted(reports.items())},
            "policies": policies}


def matched_pair(items):
    if len(items) != 2 or len({row["truth"] for row in items}) != 2:
        raise ValueError("F6 replay expects two variants with different evaluator truths")
    first, second = items
    if runtime_signature(Path(first["source"])) != runtime_signature(Path(second["source"])):
        raise ValueError("F6 full permitted inputs differ")
    for policy, trace in first["policies"].items():
        if trace != second["policies"][policy] or trace["decision"] != "request_inspection":
            raise ValueError(f"{policy} violated matched F6 behavior")
    return True


def scored_associations(cohorts):
    results = []
    for group in cohorts:
        for row in group["episodes"]:
            label = row["scoring_association_label"]
            if label is not None:
                for agent, scores in row["frozen_scores_by_witness"].items():
                    if set(scores) == set(PERMUTATIONS):
                        best = max(scores, key=scores.get)
                        results.append({"seed": group["seed"], "team_size": row["team_size"], "agent": agent,
                                        "label": label, "top_class": best, "top_probability": scores[best],
                                        "correct": best == label})
    return results


def score(study, evidence, heartbeat, validation=None, *, kind="events_only"):
    started = time.monotonic()
    torch.set_num_threads(3)
    model = next(row for row in study["models"] if row["kind"] == kind)
    frozen, gains = verified_model(model)
    head = EvidenceHead().eval()
    head.load_state_dict(frozen["weights"], strict=True)
    network = (load_backbone(study), head, frozen["temperature"])
    cohorts = []
    for seed, team in sources(evidence, validation):
        heartbeat()
        episodes = [episode_result(trace, network, frozen, gains) for trace in team["traces"]]
        pair = matched_pair(episodes) if episodes[0]["family"] == "F6" else None
        cohorts.append({"seed": seed, "team_size": team["actor_count"], "f6_verified": pair, "episodes": episodes})
    if not cohorts:
        raise ValueError("No independently accepted source groups are available")
    scope = "Primary simulator validation only" if validation is not None else "Previously inspected development replay"
    return {"scope": scope,
            "seeds": sorted({row["seed"] for row in cohorts}), "checkpoint_kind": kind,
            "checkpoint_sha256": model["checkpoint_sha256"],
            "frozen_gain_nats": gains, "threshold": frozen["threshold"],
            "authority": "Each witness reads only its own pre-query camera grant. Truth joins after exchange.",
            "viewpoint_scope": "Association head and calibration come from the original Tank view. Extra viewpoints are unvalidated.",
            "pooling": "A geometric mean of likelihood ratios for distinct camera reports about the same event",
            "communication": "Common compact ads plus full serialized selected requests and replies, modeled original deadline",
            "elapsed_s": time.monotonic() - started, "cohorts": cohorts,
            "witness_association_diagnostics": scored_associations(cohorts)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validation-root", type=Path)
    parser.add_argument("--checkpoint-kind", choices=("events_only", "events_plus_reveal"), default="events_only")
    args = parser.parse_args()
    study = PILOT / "story-learning.json"
    name = "scaling-team-evidence-validation.json" if args.validation_root else "scaling-team-evidence-replay.json"
    if args.checkpoint_kind == "events_plus_reveal":
        name = name.replace(".json", "-reveal.json")
    output = args.output or PILOT / name
    with admitted(settings(), "smart-swarm-team-evidence-replay") as (grant, heartbeat):
        result = score(load(study), PILOT, heartbeat, args.validation_root, kind=args.checkpoint_kind)
    atomic(output, {**result, "reservation": grant, "study_sha256": sha(study)})
    print(output)


if __name__ == "__main__":
    main()
