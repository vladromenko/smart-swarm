#!/usr/bin/env python3
"""Verify the live read-only reveal path in an isolated technical story."""

import argparse
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import time

from smart_swarm.cli import settings
from smart_swarm.contract import json_bytes, time_at
from smart_swarm.episode import DELIVERIES, LABELS, TRUTH, Recorder, Replay, atomic, load, rows, sha
from smart_swarm.evidence_inputs import runtime_signature
from smart_swarm.memory import QUERY, predict
from smart_swarm.physical_evidence_dataset import verify_terminal
from smart_swarm.physical_reveal import FIELD, PROFILE
from smart_swarm.physical_story_annotations import terminal_catalog
from smart_swarm.physical_story_plan import prepare
from smart_swarm.physical_story_replay import AGENTS, boot_clock, terminal_reveal
from smart_swarm.physical_story_source import story_workspace
from smart_swarm.primary_inputs import primary_policy_view
from smart_swarm.studio import native_status


def pending_story(config):
    card = prepare(config)["cards"][0]
    trial = story_workspace(config) / card["story_id"]
    root = trial / "private-replay"
    clock = boot_clock()
    decision = time.monotonic_ns()
    query = decision - 1
    meta = {"run_mode": "hardware", FIELD: PROFILE, "native_recording_clock": clock,
            "source_type": "preview_only_technical_rehearsal"}
    recorder = Recorder(root, trial.name, card["source_group_id"], dict.fromkeys(AGENTS, []), meta, clock=clock)
    for agent in AGENTS:
        recorder.capture(agent, {}, query - 1, [], [])
    recorder.append("communication", "queries", {"issued_time": time_at(query, clock), "text": QUERY})
    (root / DELIVERIES).touch()
    record = {"at": time_at(decision, clock), "agent_id": AGENTS[-1], "query_all": predict([], decision)}
    recorder.append("decisions", "prediction", record)
    atomic(trial / "trial.json", {"status": "pending_reveal", "decision": record, "query_ns": query})
    return root, decision


def prove_scoped(root, frames):
    signature = runtime_signature(root)
    with primary_policy_view(root) as scoped:
        if (runtime_signature(scoped) != signature or any((scoped / namespace).exists() for namespace in (TRUTH, LABELS))
                or FIELD in load(scoped / "manifest.json")):
            raise ValueError("The live evaluator clip changed the permitted runtime input")
    client = Replay(root, AGENTS[-1])
    for sample in frames:
        try:
            client.image(sample["camera"]["frame_id"])
        except KeyError:
            pass
        else:
            raise ValueError("Replay exposed an evaluator clip frame")
    return sha256(json_bytes(signature)).hexdigest()


def rehearse(config):
    before = native_status(config)
    root, decision = pending_story(config)
    prediction = root / "decisions/prediction.jsonl"
    saved = sha(prediction)
    journal = terminal_reveal(config, root.parent.name, "unknown", preserved=True)
    manifest = load(root / "manifest.json")
    labels, count = verify_terminal(root, journal, manifest)
    row, = rows(root / TRUTH / "reveal.jsonl")
    frames = row["reveal_clip"]["frames"]
    review = terminal_catalog(root, journal)
    if ([image["frame_id"] for image in review["review_images"]] != [frame["camera"]["frame_id"] for frame in frames]
            or any(image["access_role"] != "evaluator_only" or image["conservative_capture_interval"] is None
                   for image in review["review_images"])
            or review["status"] != "needs_confirmation" or review["candidate"] is not None):
        raise ValueError("The terminal review lost an evaluator frame or answered without confirmation")
    digest = prove_scoped(root, frames)
    after = native_status(config)
    generation = before["native_stop_generation"]
    if (after["native_stop_generation"] != generation or sha(prediction) != saved or any(labels.values())
            or any(frame["native_preview_receipt"]["preview_status_after"]["native_stop_generation"] != generation
                   for frame in frames)):
        raise ValueError("The technical rehearsal changed native STOP, the pending prediction or outcome labels")
    return {"scope": "read-only native preview with an isolated technical pre-decision fixture", "profile": PROFILE,
            "frames": frames, "frame_count": count, "native_stop_generation": generation,
            "decision_ns": str(decision), "elapsed_to_last_receipt_s": (journal["recorded_at_ns"] - decision) / 1e9,
            "prediction_sha256_unchanged": saved, "permitted_input_sha256": digest,
            "reviewable_evaluator_samples": len(review["review_images"]), "terminal_review_status": review["status"],
            "evaluator_clip_excluded_from_all_runtime_inputs": True, "attended_outcome_confirmed": False,
            "native_recording_started": False, "robot_motion_requested": False, "physical_story_attempts_added": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with TemporaryDirectory(prefix="smart-swarm-native-reveal-") as directory:
        result = rehearse({**settings(), "data_root": Path(directory)})
    atomic(args.out, result)
    print(f"Verified {result['frame_count']} read-only native reveal samples in {result['elapsed_to_last_receipt_s']:.3f}s")


if __name__ == "__main__":
    main()
