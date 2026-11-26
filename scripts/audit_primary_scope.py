#!/usr/bin/env python3
"""Verify that primary F6 policy inputs survive removal of privileged sidecars."""

import argparse
import hashlib
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from audit_memory_validation import accepted_sources
from capture_memory_validation import DEFAULT_OUTPUT, validation_slots
from smart_swarm.contract import json_bytes
from smart_swarm.episode import LABELS, POLICY, TRUTH, atomic, listed_file, load, sha
from smart_swarm.evidence_inputs import packets_and_transcript, runtime_signature
from smart_swarm.primary_inputs import primary_policy_view

ALLOWED_MANIFEST = {"policy_cameras", "query_boundary", "decision_boundary", "world_episode_id", "sidecar_sha256"}


def verify_projection(path):
    with primary_policy_view(path) as scoped:
        if {item.name for item in scoped.iterdir()} != {POLICY, "communication", "manifest.json"}:
            raise ValueError("The materialized input has an unexpected namespace")
        if set(load(scoped / "manifest.json")) != ALLOWED_MANIFEST:
            raise ValueError("The materialized input has unexpected metadata")
        packets, _, _, _ = packets_and_transcript(scoped)
        frames = [camera for records in packets.values() for packet in records for camera in packet["cameras"]]
        if any(sha(listed_file(scoped, camera["frame_ref"], POLICY)) != camera["sha256"] for camera in frames):
            raise ValueError("A projected frame differs from its original recorded bytes")
        return runtime_signature(scoped), len(frames)


def substituted_sidecars(source):
    with TemporaryDirectory(prefix="smart-swarm-f6-scope-audit-") as directory:
        copy = Path(directory) / "episode"
        shutil.copytree(source, copy)
        for namespace in (LABELS, TRUTH):
            shutil.rmtree(copy / namespace)
            (copy / namespace).mkdir()
            atomic(copy / namespace / "substituted.json", {"private": "counterfactual sentinel"})
        manifest = load(copy / "manifest.json")
        manifest.update(family="unavailable", simulation_recipe_seed=-1, lineage={"split": "unavailable"})
        atomic(copy / "manifest.json", manifest)
        return verify_projection(copy)


def paired(team):
    left, right = [Path(row["path"]) for row in team["traces"]]
    original, frames = verify_projection(left)
    if (original != runtime_signature(left) or original != runtime_signature(right)
            or verify_projection(right) != (original, frames)
            or substituted_sidecars(left) != (original, frames) or substituted_sidecars(right) != (original, frames)):
        raise ValueError("Policy projection changed the complete F6 input scope")
    outcomes = [row["terminal_outcome"] for row in team["traces"]]
    if len(set(outcomes)) != 2:
        raise ValueError("An F6 pair must differ on isolated evaluator truth")
    return {"team_size": team["actor_count"], "outcomes": outcomes, "frame_count_per_variant": frames,
            "permitted_input_sha256": hashlib.sha256(json_bytes(original)).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    allocation = {slot["source_group"]: slot for slot in validation_slots()}
    relevant = {name for name, slot in allocation.items() if slot["family"] == "F6"}
    verified = {row["source_group"]: [paired(team) for team in row["teams"]]
                for row in accepted_sources(DEFAULT_OUTPUT, allocation, relevant)}
    if not verified:
        raise ValueError("The validation cohort has no admitted F6 pairs")
    result = {"scope": "sealed primary validation", "independent_source_groups": len(verified),
              "minimal_manifest_fields": sorted(ALLOWED_MANIFEST), "projected_input_and_rgb_equal": True,
              "substituted_evaluator_training_and_noncausal_metadata": True, "groups": verified}
    if args.output.exists() and load(args.output) != result:
        raise ValueError("Choose a new output name to preserve the earlier scope audit")
    atomic(args.output, result)
    print(len(verified), "independent F6 groups verified", args.output)


if __name__ == "__main__":
    main()
