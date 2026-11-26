#!/usr/bin/env python3
"""Verify phase-observation isolation on genuine native technical archives."""

import argparse
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from rehearse_native_event_annotations import ARCHIVE, NATIVE_FIXTURE, exact_projection, project
from smart_swarm.cli import settings
from smart_swarm.contract import json_bytes
from smart_swarm.episode import Replay, atomic, load
from smart_swarm.evidence_inputs import runtime_signature
from smart_swarm.physical_learned import scoped_window
from smart_swarm.physical_story_condition import PHASE_DIGEST, audit_condition, phase_digest
from smart_swarm.primary_inputs import primary_policy_view
from smart_swarm.studio import validate_story_episode


def hidden_phase(root, manifest, state, outcome):
    markers = {**state["markers"], "placement_closed": {**state["markers"]["placement_closed"],
               "observed_placement_outcome": outcome}}
    digest = phase_digest(markers)
    tagged = {**manifest, PHASE_DIGEST: digest}
    atomic(root / "manifest.json", tagged)
    record = {"family": "F6", "labels": {"terminal": None}, "attended_placement_outcomes": {"placement_closed": outcome}}
    result = audit_condition(record, markers, digest)
    with primary_policy_view(root) as scoped:
        if PHASE_DIGEST in load(scoped / "manifest.json"):
            raise ValueError("The primary policy projection opened an attended phase attestation")
        allowed = runtime_signature(scoped)
    with scoped_window(root, tagged) as (scoped, learned_window):
        if PHASE_DIGEST in load(scoped / "manifest.json"):
            raise ValueError("The learned witness window opened an attended phase attestation")
    forecast = Replay(root, "rosmaster-1").predict()
    return {"attested_phase_sha256": digest, "rule_scope_sha256": sha256(json_bytes(allowed)).hexdigest(),
            "learned_window_sha256": learned_window, "rule_forecast": forecast, "condition_audit": result}


def rehearse(config, root):
    source = config["studio_project"] / NATIVE_FIXTURE
    proof = validate_story_episode(config, source, technical=True)
    if not proof["technical_only"]:
        raise ValueError("The original Studio source must remain a technical archive")
    recorder, state, arm, tank, query = project(config, root, source, proof, "F6")
    exact_projection(recorder, state, arm, tank, query)
    manifest = load(root / "manifest.json")
    first, second = [hidden_phase(root, manifest, state, outcome) for outcome in ("placement_succeeded", "placement_failed")]
    stable = ("rule_scope_sha256", "learned_window_sha256", "rule_forecast", "condition_audit")
    if (first["attested_phase_sha256"] == second["attested_phase_sha256"]
            or any(first[name] != second[name] for name in stable)
            or first["condition_audit"]["status"] != "withholding_only"):
        raise ValueError("Changing hidden evaluator metadata affected a pre-inspection input or claim")
    return {"scope": "native technical F6-shaped replay with perturbed evaluator metadata", "studio_episode": proof["episode_id"],
            "explorer_capture": ARCHIVE, "outcome_labels": 0, "private_packets_by_role": recorder.counts,
            "rgb_packets_by_role": {agent: count if recorder.cameras[agent] else 0 for agent, count in recorder.counts.items()},
            "hidden_metadata_variants": [first, second], "new_physical_story_attempts": 0, "new_optimizer_updates": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Choose a new immutable phase-observation rehearsal receipt")
    with TemporaryDirectory(prefix="smart-swarm-attended-condition-") as directory:
        result = rehearse(settings(), Path(directory) / "technical-f6")
    atomic(args.out, result)
    print(args.out)


if __name__ == "__main__":
    main()
