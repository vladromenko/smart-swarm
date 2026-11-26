#!/usr/bin/env python3
"""Replay a centralized RGB reference with every eligible pre-query camera frame."""

import argparse
from pathlib import Path
import time

import numpy as np
import torch

from audit_evidence_replay import ROOT, compressed_rgb_size, evaluated, local_belief, receiver_rgb, verified_model
from learn_evidence import admitted_features
from learn_evidence_v3 import forecast
from smart_swarm.cli import settings
from smart_swarm.episode import Replay, atomic, load, sha
from smart_swarm.research_jobs import admitted
from smart_swarm.temporal_evidence import EvidenceHead, FrozenVision

ARM = "so101-1"
TANK = "tank-1"
RECEIVER = "rosmaster-1"
PILOT = ROOT / "evidence" / "learning-pilot-v3"


def full_stream(encoder, client):
    cameras = [packet["cameras"] for packet in client.packets() if packet["cameras"]]
    if not cameras:
        return None, 0
    inputs = []
    for frame_set in cameras:
        if tuple(camera["sensor_id"] for camera in frame_set) != client.allowed:
            raise ValueError(f"{client.agent} camera order does not match the grant")
        inputs.append(np.stack([np.asarray(client.image(camera["frame_id"])) for camera in frame_set]))
    return encoder(np.stack(inputs)), sum(len(frame_set) for frame_set in cameras)


def predict(model, frozen, key, features):
    return forecast(model, [{"world_episode_id": key}], {key: features}, frozen["priors"],
                    frozen["temperature"], frozen["state_prior"])[0]


def central_result(row, model, encoder, frozen, local_sensor):
    root = Path(row["path"])
    streams = [full_stream(encoder, Replay(root, agent)) for agent in (ARM, TANK)]
    features, counts = zip(*streams, strict=True)
    forecast_row = predict(model, frozen, row["world_episode_id"], features)
    receiver = receiver_rgb(Replay(root, RECEIVER))
    payload = {"receiver_rgb_cue": receiver["local_rgb_history"], "receiver_frame_hashes": receiver["rgb_sha256"],
               "frames_by_agent": dict(zip((ARM, TANK, RECEIVER), (*counts, receiver["frames"]), strict=True))}
    result = {**row, **forecast_row, "wire": payload, "serialized_bytes": compressed_rgb_size(root)}
    belief = local_belief(result["belief"], local_sensor, receiver["local_rgb_history"])
    return result, {**result, "belief": belief}


def load_backbone(study):
    config = load(ROOT / "configs" / "evidence-pilot.json")
    path = Path(config["frozen_weights"])
    if sha(path) != study["frozen_backbone_sha256"]:
        raise ValueError("Frozen backbone identity differs from the pilot")
    return FrozenVision(path)


def validate_clip(row, encoder, model, frozen):
    features = admitted_features(encoder, Path(row["path"]))
    replay = predict(model, frozen, row["world_episode_id"], features)["belief"]
    error = max(abs(value - row["belief"][name]) for name, value in replay.items())
    if error > 1e-6:
        raise ValueError("Clip control does not reproduce the archived forecast")
    return error


def run(study, local, heartbeat):
    started = time.monotonic()
    torch.set_num_threads(3)
    encoder = load_backbone(study)
    checkpoint = next(row for row in study["models"] if row["kind"] == "events_only")
    frozen, _ = verified_model(checkpoint)
    model = EvidenceHead()
    model.load_state_dict(frozen["weights"], strict=True)
    local_sensor = local["receiver_only"]["methods"]["local_rgb_history"]["train_likelihood_by_surface"]
    worlds = checkpoint["test_outcomes"]
    parity = validate_clip(worlds[0], encoder, model, frozen)
    records = []
    for index, row in enumerate(worlds):
        if index % 10 == 0:
            heartbeat()
        records.append(central_result(row, model, encoder, frozen, local_sensor))
    full, with_receiver = zip(*records, strict=True)
    return {"scope": "post-freeze centralized RGB replay on previously scored simulated sources",
            "checkpoint_sha256": checkpoint["checkpoint_sha256"], "threshold": frozen["threshold"],
            "permission_scope": "All own pre-query camera frames from the three grants. No evaluator, state or network delay.",
            "full_rgb_source_payload": local["all_private_rgb_payload"],
            "standard_clip_control_max_delta": parity, "elapsed_s": time.monotonic() - started,
            "all_witness_frames": evaluated(full, frozen["threshold"]),
            "all_witness_and_receiver_frames": evaluated(with_receiver, frozen["threshold"]),
            "outcomes": [{"world_episode_id": row["world_episode_id"], "frames": row["wire"]["frames_by_agent"],
                          "witness_belief": row["belief"], "central_belief": combined["belief"]}
                         for row, combined in records]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PILOT / "central-history-postfreeze.json")
    args = parser.parse_args()
    source = PILOT / "story-learning.json"
    local = load(PILOT / "evidence-budget-postfreeze.json")
    if local["frozen_study_sha256"] != sha(source):
        raise ValueError("Local likelihoods are from a different study")
    with admitted(settings(), "smart-swarm-central-history-readonly") as (grant, heartbeat):
        report = run(load(source), local, heartbeat)
    atomic(args.output, {**report, "reservation": grant})
    print(args.output)


if __name__ == "__main__":
    main()
