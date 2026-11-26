#!/usr/bin/env python3
"""Rehearse hardware slot normalization using original native and simulator archives."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from compare_primary_supervision import score_paths
from rehearse_native_event_annotations import ARCHIVE, PILOT, project
from rehearse_physical_story_recovery import native_fixture
from smart_swarm.cli import settings
from smart_swarm.contract import json_bytes, time_at
from smart_swarm.episode import POLICY, TRUTH, Recorder, Replay, atomic, load, rows, sha
from smart_swarm.evidence_inputs import AGENTS, clip_packets
from smart_swarm.physical_learned import run_worker, scoped_window, window_digest
from smart_swarm.physical_slot_reference import REVERSED, SAME, SCOPE_FLAG, UNKNOWN
from smart_swarm.physical_story_replay import validate_pending_decision, write_decision
from smart_swarm.studio import validate_story_episode


def native_case(config, root):
    source = native_fixture(config)
    proof = validate_story_episode(config, source, technical=True)
    recorder, state, arm, tank, query = project(config, root, source, proof, "F6")
    recorder.meta[SCOPE_FLAG] = True
    recorder.meta["native_recording_clock"] = recorder.clock
    atomic(root / "recording.json", {**recorder.meta, "recording_integrity": "recording"})
    manifest = {**recorder.meta, "sample_counts": recorder.counts, "query_boundary": time_at(query, recorder.clock)}
    with scoped_window(root, manifest) as (scoped, digest):
        prediction = run_worker(scoped, digest)
    decision = write_decision(recorder, arm, tank, query, query + 1_000_000_000, {"status": "scored", "receipt": prediction})
    checked, boundary = validate_pending_decision(Recorder.reopen_pending(root), root, state)
    if checked != decision or boundary != query or any((root / TRUTH).iterdir()):
        raise ValueError("The original native replay failed its pre-reveal readback")
    if prediction["slot_mapping"] != UNKNOWN or any(row["decision"] != "request_inspection" or row["wire"]["replies"]
                                                     for row in prediction["heads"].values()):
        raise ValueError("An unverified native setup produced a location or transmitted an association report")
    return {"scope": "Native technical archive with no physical outcome", "explorer_capture": ARCHIVE,
            "studio_episode": proof["episode_id"], "slot_check": decision["slot_reference"], "prediction": prediction}


def add_landmarks(root, agent, green_left):
    path = root / POLICY / f"{agent}.jsonl"
    records = rows(path)
    for row in records:
        for camera in row["cameras"]:
            frame = root / camera["frame_ref"]
            with Image.open(frame) as picture:
                pixels = np.asarray(picture.convert("RGB")).copy()
            width = pixels.shape[1]
            colors = ((0, 230, 0), (0, 0, 230)) if green_left else ((0, 0, 230), (0, 230, 0))
            pixels[4:13, 6:15] = colors[0]
            pixels[4:13, width - 15:width - 6] = colors[1]
            Image.fromarray(pixels).save(frame)
            camera["sha256"] = sha(frame)
    path.write_bytes(b"".join(json_bytes(row) + b"\n" for row in records))


def physical_packet_counts(root):
    tank, receiver = AGENTS[1:]
    windows = {tank: clip_packets(Replay(root, tank)), receiver: Replay(root, receiver).packets()[-1:]}
    for role, records in windows.items():
        path = root / POLICY / f"{role}.jsonl"
        path.write_bytes(b"".join(json_bytes(row) + b"\n" for row in records))


def counterfactual(source, mapping):
    original = load(source / "manifest.json")
    with scoped_window(source, {**original, SCOPE_FLAG: True}) as (scoped, _):
        physical_packet_counts(scoped)
        add_landmarks(scoped, AGENTS[1], mapping == SAME)
        if mapping != UNKNOWN:
            add_landmarks(scoped, AGENTS[2], True)
        manifest = load(scoped / "manifest.json")
        packets = {agent: rows(scoped / POLICY / f"{agent}.jsonl") for agent in AGENTS}
        digest = window_digest(manifest, packets)
        result = run_worker(scoped, digest)
    if result["slot_mapping"] != mapping:
        raise ValueError("The worker did not recover the configured fixture side relationship")
    return result


def prior_sources():
    outcomes = load(score_paths("validation")["events_only"])["outcomes"]
    return {family: next(group["outcomes"] for group in outcomes
                         if group["team_size"] == 3 and group["outcomes"][0]["family"] == family)
            for family in ("F1", "F6")}


def resolvable_control(control):
    source = Path(control["source"])
    result = {mapping: counterfactual(source, mapping) for mapping in (REVERSED, SAME, UNKNOWN)}
    terminal = control["truth"]
    opposing = "right" if terminal == "left" else "left"
    for mapping, choice in ((REVERSED, terminal), (SAME, opposing), (UNKNOWN, "request_inspection")):
        for row in result[mapping]["heads"].values():
            if row["decision"] != choice or row["serialized_bytes"] > 3072:
                raise ValueError(f"The resolvable fixture did not produce the expected receiver choice for {mapping}")
            has_association = any(packet["factor"] == "association" for packet in row["wire"]["replies"])
            if has_association == (mapping == UNKNOWN):
                raise ValueError("The association response disagrees with the verified side reference")
    for kind in result[SAME]["heads"]:
        values = [next(packet["probabilities"] for packet in result[mapping]["heads"][kind]["wire"]["replies"]
                       if packet["factor"] == "association") for mapping in (SAME, REVERSED)]
        if values[0] != values[1]:
            raise ValueError("Station-card identity changed the learned association encoder input")
    return {"source": str(source), "original_terminal": terminal, "fixture_mappings": result}


def hidden_pair(pair):
    results = [{"terminal_for_scoring": row["truth"], "prediction": counterfactual(Path(row["source"]), REVERSED)}
               for row in pair]
    first, second = results
    if first["terminal_for_scoring"] == second["terminal_for_scoring"] or first["prediction"] != second["prediction"]:
        raise ValueError("The complete projected hidden pair changed under its isolated evaluator outcome")
    if any(row["decision"] != "request_inspection" for row in first["prediction"]["heads"].values()):
        raise ValueError("The side reference alone answered a hidden placement")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a new slot-reference rehearsal receipt")
    original = prior_sources()
    with TemporaryDirectory(prefix="smart-swarm-slot-native-archive-") as directory:
        native = native_case(settings(), Path(directory) / "native")
    report = {"scope": "Native technical archive plus archived simulator frames with controlled landmark overlays",
              "native_archived": native, "resolvable_control": resolvable_control(original["F1"][0]),
              "hidden_pair": hidden_pair(original["F6"]), "new_physical_story_attempts": 0, "new_optimizer_updates": 0}
    atomic(args.output, report)
    print(args.output)


if __name__ == "__main__":
    main()
