#!/usr/bin/env python3
"""Reopen the actual native archives through the event review adapter without labeling them."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from rehearse_physical_story_recovery import NATIVE_FIXTURE, PILOT
from smart_swarm.cli import settings
from smart_swarm.contract import time_at
from smart_swarm.episode import LABELS, TRUTH, atomic, rows, sha
from smart_swarm.evidence_inputs import runtime_signature
from smart_swarm.physical_evidence_dataset import verify_studio_source
from smart_swarm.physical_fleet_timing import PhysicalStoryRecorder
from smart_swarm.physical_story_annotations import build_receipt, catalog
from smart_swarm.physical_story_replay import AGENTS, fleet_clips, studio_clips, write_decision
from smart_swarm.physical_story_source import fleet_manifest
from smart_swarm.primary_inputs import primary_policy_view
from smart_swarm.studio import validate_story_episode

ARCHIVE = "52f8e09287a4492f82d13198d7ebdb73"


def project(config, root, source, proof, family):
    fleet = fleet_manifest(config, ARCHIVE)
    if fleet["phase"] != "complete" or not fleet["readback"]["passed"]:
        raise ValueError("The original Explorer archive must retain its native readback")
    epoch = round(fleet["created_at"] * 1_000_000_000)
    native = [row["observation_timestamp_ns"] for row in rows(source / "frames.jsonl")]
    offset = epoch + 5_000_000_000 - max(native)

    def at(seconds):
        wall = epoch + seconds * 1_000_000_000
        return {"monotonic_ns": wall - offset, "unix_ns": wall}

    middle = native[len(native) // 2 - 1]
    markers = {"placement_closed": at(5), "relocation_closed": at(17)}
    if family == "F3":
        markers["insertion_closed"] = {"monotonic_ns": middle, "unix_ns": middle + offset}
    state = {"fleet_capture_id": ARCHIVE, "native_story_validation": proof, "markers": markers,
             "start": {"monotonic_ns": min(native), "unix_ns": min(native) + offset},
             "card": {"story_id": root.name, "family": family}, "status": "archive_technical_only"}
    clock = {"host_id": "archive-rehearsal", "boot_id": ARCHIVE, "kind": "monotonic"}
    cameras = {AGENTS[0]: [] if family == "F6" else ["so101-1/wrist", "so101-1/top"],
               **{agent: [agent + "/front"] for agent in AGENTS[1:]}}
    provenance = {"run_mode": "hardware", "studio_episode": proof["episode_id"], "fleet_capture": ARCHIVE}
    recorder = PhysicalStoryRecorder(root, root.name, "technical-replay", cameras, provenance, clock=clock)
    arm = studio_clips(recorder, source, state)
    query = at(21)
    tank, receiver = AGENTS[1:]
    audit = {}
    clips = fleet_clips(config, recorder, state, tank, query["monotonic_ns"], fleet, audit, current_anchor=query)
    fleet_clips(config, recorder, state, receiver, query["monotonic_ns"], fleet, audit, current_anchor=query)
    return recorder, state, arm, clips, query["monotonic_ns"]


def exact_projection(recorder, state, arm, tank, query):
    root = recorder.root
    decided = query + 1_000_000_000
    record = write_decision(recorder, arm, tank, query, decided, {"status": "unavailable", "reason": "archive rehearsal"})
    atomic(root / "manifest.json", {**recorder.meta, "sample_counts": recorder.counts,
                                   "query_boundary": time_at(query, recorder.clock), "decision_boundary": record["at"],
                                   "sidecar_sha256": {"communication/deliveries.jsonl": sha(root / "communication/deliveries.jsonl")}})
    before = runtime_signature(root)
    proposals = catalog(root, state)
    with primary_policy_view(root) as scoped:
        identical = runtime_signature(scoped) == before and not (scoped / TRUTH).exists() and not (scoped / LABELS).exists()
    if not identical:
        raise ValueError("Post-decision review changed the complete permitted input")
    return proposals, record


def admission_refusal(config, recorder, state, record, proposals):
    try:
        verify_studio_source(config, state, recorder.meta)
    except ValueError as error:
        production = str(error)
    else:
        raise ValueError("The technical Studio source was admitted as a physical story")
    try:
        build_receipt({**state, "decision": record}, sha(recorder.root / "manifest.json"),
                      {"settled_placement": {"value": "surface"}}, proposals)
    except ValueError as error:
        annotation = str(error)
    else:
        raise ValueError("An unsealed technical replay acquired an annotation")
    return {"physical_corpus_rejected": production, "confirmation_without_attended_seal_rejected": annotation}


def case(config, source, proof, root, family):
    recorder, state, arm, tank, query = project(config, root, source, proof, family)
    proposals, record = exact_projection(recorder, state, arm, tank, query)
    checked = admission_refusal(config, recorder, state, record, proposals)
    events = {name: {"frame_count": len(value["review_images"]), "candidate": value["candidate"],
                     "status": value["status"]} for name, value in proposals.items()}
    if family == "F3" and any(events[name]["frame_count"] != 3 for name in ("first_insertion", "settled_placement")):
        raise ValueError("Both original native phases must keep distinct scoped image samples")
    if family == "F6" and events["settled_placement"]["frame_count"]:
        raise ValueError("The technical F6 review opened a withheld arm image")
    return {**checked, "original_rgb_counts_by_role": recorder.counts, "events": events,
            "runtime_signature_unchanged": True, "confirmed_physical_labels": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a new technical annotation rehearsal receipt")
    config = settings()
    source = config["studio_project"] / NATIVE_FIXTURE
    proof = validate_story_episode(config, source, technical=True)
    with TemporaryDirectory(prefix="smart-swarm-native-event-review-") as directory:
        cases = {family: case(config, source, proof, Path(directory) / family, family) for family in ("F3", "F6")}
    receipt = {"scope": "Native technical playback at archive-relative times", "explorer_capture": ARCHIVE,
               "studio_episode": proof["episode_id"], "contexts": cases, "new_physical_story_attempts": 0,
               "new_optimizer_updates": 0}
    atomic(args.output, receipt)
    print(args.output)


if __name__ == "__main__":
    main()
