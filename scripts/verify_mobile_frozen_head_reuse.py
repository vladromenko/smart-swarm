#!/usr/bin/env python3
"""Verify that two frozen decision heads can share encoding of the same scoped RGB."""

import argparse
from pathlib import Path
from time import monotonic

from smart_swarm.cli import ROOT, settings
from smart_swarm.contract import json_bytes
from smart_swarm.episode import Replay, atomic, load, sha
from smart_swarm.evidence_inputs import clip_packets
from smart_swarm.identity_support import witness_marker_support
from smart_swarm.primary_inputs import primary_policy_view
from smart_swarm.research_jobs import admitted
from smart_swarm.team_evidence import opportunities

WORK = ROOT.parent.parent / "work"
SOURCE = WORK / "active-inspection-comparison-mobile-v2"
PILOT = ROOT / "evidence/learning-pilot-v3"
VARIANTS = ("visible-control", "hidden-contained", "hidden-surface")
FIELDS = ("decision", "belief", "serialized_bytes", "modeled_latency_ms")


def shared_contexts(contexts):
    import torch

    encoder = next(iter(contexts.values()))["network"][0]
    reference = encoder.state_dict()
    result = {}
    for model, context in contexts.items():
        other, head, temperature = context["network"]
        candidate = other.state_dict()
        if (type(other) is not type(encoder) or encoder.model.training or other.model.training or head.training
                or reference.keys() != candidate.keys()
                or not all(torch.equal(value, candidate[name]) for name, value in reference.items())):
            raise ValueError("Both decision heads must use the same frozen visual encoder")
        result[model] = {**context, "network": (encoder, head, temperature)}
    return result


def clips(root, contexts):
    from audit_central_history import full_stream
    from score_primary_validation import frame_inventory

    window = opportunities(root)
    query, deadline, adverts = window
    encoder = next(iter(contexts.values()))["network"][0]
    if any(context["network"][0] is not encoder or deadline - query != context["horizon"] * 1_000_000
           for context in contexts.values()):
        raise ValueError("Shared encoding requires one verified backbone and the original decision clock")
    features, markers = {}, {}
    inspect = any(context["inspect_markers"] for context in contexts.values())
    for advert in adverts:
        sender, factor = advert["sender"], advert["factor"]
        client = Replay(root, sender)
        eligible, _, _ = frame_inventory(root, client)
        embeddings, _ = full_stream(encoder, client)
        if embeddings is None or len(embeddings) != len(eligible):
            raise ValueError("An advertised sender lost scoped visual frames")
        positions = {packet["sequence"]: index for index, packet in enumerate(eligible)}
        chosen = [positions[packet["sequence"]] for packet in clip_packets(client)]
        features[sender] = (factor, embeddings[chosen])
        if inspect and factor == "association":
            markers[sender] = witness_marker_support(client)
    return window, features, markers


def selective_decision(window, features, markers, context):
    from audit_team_evidence import decode_features
    from score_primary_validation import IDENTITY, PRIMARY_CAP, SELECTIVE, TRANSFER, choose, compact_result

    _, head, temperature = context["network"]
    posteriors = {sender: decode_features(feature, factor, head, temperature)
                  for sender, (factor, feature) in features.items()}
    row = {"window": window, "frozen_scores_by_witness": posteriors}
    fit = context["mounts"]["3"]
    args = (row, TRANSFER[SELECTIVE], PRIMARY_CAP, fit["association"]["compact"], context["common"])
    baseline = compact_result(*args)
    selected, method = baseline, SELECTIVE
    if context["inspect_markers"]:
        selected, method = compact_result(*args, markers=markers), IDENTITY
        if selected["serialized_bytes"] != baseline["serialized_bytes"]:
            raise ValueError("The marker check changed the frozen communication charge")
    result = choose(selected, fit["decision_thresholds"][method])
    return {key: result[key] for key in FIELDS}


def shared_profile_method(path, contexts):
    with primary_policy_view(path) as scoped:
        window, features, markers = clips(scoped, contexts)
        return {kind: selective_decision(window, features, markers, context)
                for kind, context in contexts.items()}


def timed(method, path, contexts):
    start = monotonic()
    result = method(path, contexts)
    return result, monotonic() - start


def compare_case(variant, contexts, shared, accelerated_first):
    from compare_rosmaster_inspection import profile_method

    root = SOURCE / f"mobile-rehearsal-{variant}"
    path = root / "pre-inspection"
    expected = load(root / "pre-inspection-decision.json")["learned"]
    if accelerated_first:
        result, new = timed(shared_profile_method, path, shared)
        baseline, old = timed(profile_method, path, contexts)
    else:
        baseline, old = timed(profile_method, path, contexts)
        result, new = timed(shared_profile_method, path, shared)
    if json_bytes(result) != json_bytes(expected) or json_bytes(result) != json_bytes(baseline):
        raise ValueError(f"{variant} changed the frozen prediction, belief or measured communication charge")
    return {"variant": variant, "decision_sha256": sha(root / "pre-inspection-decision.json"),
            "same_frozen_outputs": True, "separate_seconds": old, "shared_seconds": new, "outputs": result}


def run(heartbeat):
    from audit_primary_scope import paired
    from compare_rosmaster_inspection import KINDS
    from freeze_primary_profiles import require_profiles
    from score_primary_test import runtime
    import torch

    torch.set_num_threads(3)
    profile = require_profiles(PILOT / "primary-profiles-frozen.json")
    contexts = {kind: runtime(profile, kind) for kind in KINDS}
    shared = shared_contexts(contexts)
    cases = []
    for index, variant in enumerate(VARIANTS):
        heartbeat()
        cases.append(compare_case(variant, contexts, shared, bool(index % 2)))
    traces = [{"path": str(SOURCE / f"mobile-rehearsal-{variant}/pre-inspection"),
               "terminal_outcome": load(SOURCE / f"mobile-rehearsal-{variant}/receipt.json")["evaluation"]["location"]}
              for variant in VARIANTS[1:]]
    scope = paired({"actor_count": 3, "traces": traces})
    if cases[1]["outputs"] != cases[2]["outputs"]:
        raise ValueError("The shared frozen backbone changed the coupled F6 result")
    return {"status": "verified", "verified_sources": cases, "f6_scope": scope,
            "performance_scope": "offline inference on existing consumed development inputs",
            "independent_cases_opened": 0, "optimizer_updates": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new frozen head verification output")
    with admitted(settings(), "smart-swarm-frozen-head-reuse-verification") as (grant, heartbeat):
        started = monotonic()
        try:
            report = run(heartbeat)
        except (OSError, KeyError, ValueError, RuntimeError, KeyboardInterrupt) as error:
            atomic(args.output, {"status": "failed", "error": str(error), "wall_seconds": monotonic() - started})
            raise
        atomic(args.output, {**report, "wall_seconds": monotonic() - started, "reservation": grant})
    print(args.output)


if __name__ == "__main__":
    main()
