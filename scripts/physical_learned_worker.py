#!/usr/bin/env python3
"""Decode only role-granted RGB to run the two frozen heads on Homebrain."""

import argparse
from pathlib import Path

import numpy as np
import torch

from audit_central_history import load_backbone
from audit_evidence_replay import verified_model
from audit_team_evidence import decode, decode_features
from score_primary_validation import PRIMARY_CAP, choose, likelihoods_from_reports, on_time_compact
from smart_swarm.contract import json_bytes
from smart_swarm.episode import POLICY, Replay, load, sha
from smart_swarm.evidence_comparison import learned_prior_map
from smart_swarm.evidence_inputs import clip
from smart_swarm.identity_support import witness_marker_support
from smart_swarm.particle_memory import exact_filter
from smart_swarm.physical_learned import HEADS, METHOD, PROFILE, ROOT, SCHEMA, packets, window_digest
from smart_swarm.physical_slot_reference import SCOPE_FLAG, UNKNOWN, association_image, mapped_assignment, slot_reference
from smart_swarm.team_evidence import exchange, opportunities
from smart_swarm.temporal_evidence import EvidenceHead

STUDY = PROFILE.parent / "story-learning.json"


def decode_association(root, agent, backbone, head, temperature):
    camera = clip(root, agent)
    if camera is None:
        raise ValueError("The association witness lost its scoped RGB clip")
    prepared = np.stack([np.stack([association_image(frame) for frame in sample]) for sample in camera])
    return decode_features(backbone(prepared), "association", head, temperature)


def score(root, window, backbone, markers, model, fit, reference):
    frozen, gains = verified_model(model)
    head = EvidenceHead().eval()
    head.load_state_dict(frozen["weights"], strict=True)
    common = {"state_prior": frozen["state_prior"], "reference": learned_prior_map(frozen["priors"])}
    mounts = fit["mount_and_receiver_profiles"]["3"]
    threshold = mounts["decision_thresholds"][METHOD]
    mapping = reference["mapping"] if reference is not None else None
    overhead = reference["serialized_bytes"] if reference is not None else 0

    def report(agent, factor):
        if factor == "association" and not markers[agent]["supported"]:
            return common["reference"]["association"]
        if factor == "association" and reference is not None:
            return decode_association(root, agent, backbone, head, frozen["temperature"])
        return decode(root, agent, factor, (backbone, head, frozen["temperature"]))

    raw = exchange("factor_coverage", window, PRIMARY_CAP - overhead, gains, report, common["reference"], common["state_prior"])
    replies = on_time_compact(raw["wire"]["replies"], window[1])
    factors = likelihoods_from_reports(replies, mounts["association"]["compact"], common)
    oriented = [(kind, mapped_assignment(values, mapping) if kind == "association" else values)
                for kind, values in factors]
    state = exact_filter(common["state_prior"], oriented)
    return {**choose(state, threshold), "threshold": threshold, "checkpoint_sha256": model["checkpoint_sha256"],
            "serialized_bytes": raw["serialized_bytes"] + overhead, "modeled_latency_ms": raw["modeled_latency_ms"],
            "wire": raw["wire"]}


def run(root):
    if {path.name for path in root.iterdir()} != {POLICY, "manifest.json"}:
        raise ValueError("The physical inference worker accepts only a projected private window")
    manifest = load(root / "manifest.json")
    allowed = {"policy_cameras", "query_boundary", "decision_boundary", "world_episode_id"}
    if manifest.get(SCOPE_FLAG) is True:
        allowed.add(SCOPE_FLAG)
    if set(manifest) != allowed:
        raise ValueError("The physical window contains fields outside its declared policy metadata")
    torch.set_num_threads(1)
    study, profile = load(STUDY), load(PROFILE)
    models = {model["kind"]: model for model in study["models"]}
    if (profile["status"] != "frozen_for_primary_test" or profile["encoder_sha256"] != study["frozen_backbone_sha256"]
            or any(models[kind]["checkpoint_sha256"] != profile["fits"][kind]["checkpoint_sha256"] for kind in HEADS)):
        raise ValueError("The original validation profile and checkpoints disagree")
    reference = slot_reference(root, manifest)
    mapping = reference["mapping"] if reference is not None else None
    query, deadline, ads = opportunities(root)
    window = (query, deadline, [ad for ad in ads if ad["factor"] != "association" or mapping != UNKNOWN])
    markers = {ad["sender"]: witness_marker_support(Replay(root, ad["sender"]))
               for ad in window[2] if ad["factor"] == "association"}
    backbone = load_backbone(study)
    heads = {kind: score(root, window, backbone, markers, models[kind], profile["fits"][kind], reference) for kind in HEADS}
    digest = window_digest(manifest, packets(root, manifest), projected=True)
    report = {"schema": SCHEMA, "policy": METHOD, "permitted_input_sha256": digest,
              "profiles_sha256": sha(PROFILE), "encoder_sha256": profile["encoder_sha256"],
              "marker_support_by_witness": markers, "modeled_reply_window_ns": window[1] - window[0], "heads": heads}
    if reference is not None:
        report["slot_mapping"] = mapping
        report["slot_reference_wire"] = reference["wire"]
        report["slot_reference_bytes"] = reference["serialized_bytes"]
        report["association_landmark_mask"] = "neutral RGB in detected station-card regions"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("window", type=Path)
    args = parser.parse_args()
    print(json_bytes(run(args.window)).decode("utf-8"))


if __name__ == "__main__":
    main()
