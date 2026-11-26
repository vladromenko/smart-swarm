#!/usr/bin/env python3
"""Use sealed paired observations to probe the frozen SO101 witness heads."""

import argparse
from pathlib import Path

import numpy as np
import torch

from audit_central_history import load_backbone
from audit_evidence_replay import verified_model
from audit_team_evidence import decode_features
from smart_swarm.cli import settings
from smart_swarm.episode import Replay, atomic, load, sha
from smart_swarm.evidence_inputs import clip, clip_packets
from smart_swarm.research_jobs import admitted
from smart_swarm.temporal_evidence import EvidenceHead

PILOT = Path(__file__).resolve().parents[1] / "evidence/learning-pilot-v3"
CAMERAS = ("so101-1/wrist", "so101-1/top")


def case_images(path):
    replay = Replay(path, "so101-1")
    if replay.allowed != CAMERAS:
        raise ValueError("This controlled input requires the original ordered SO101 camera grant")
    packets = clip_packets(replay)
    if not packets:
        raise ValueError("The SO101 witness did not supply a complete clip")
    clocks = [(row["sequence"], row["source_time"], row["measured_state"]) for row in packets]
    return clip(path, replay.agent), clocks


def inputs(path):
    study = load(path)
    groups = {}
    for name, mode in sorted(study["conditions"].items()):
        ref = Path(mode["receipt"])
        if sha(ref) != mode["receipt_sha256"] or sha(ref.with_name("predictions.json")) != mode["prediction_sha256"]:
            raise ValueError("An archived control receipt changed")
        a, b = [case_images(Path(row["raw_path"])) for row in mode["records"]]
        if a[1] != b[1] or a[0].shape != b[0].shape:
            raise ValueError("Matched encoder interventions require equal SO101 timing and measured state")
        groups[name] = (a[0], b[0])
    return study, groups


def positions():
    arms = {}
    for wrist in (0, 1):
        for top in (0, 1):
            name = f"wrist_{wrist}_top_{top}"
            arms[name] = [[(wrist, t, 0), (top, t, 1)] for t in range(8)]
    for index in (0, 1):
        blend = [[(index, t, 0), (index if t < 6 else 1 - index, t, 1)] for t in range(8)]
        arms[f"clip_{index}_other_settled_top"] = blend
        arms[f"clip_{index}_settled_repeat"] = [[(index, 7, 0), (index, 7, 1)]] * 8
    return arms


def concatenated(features, indices):
    frames = [torch.cat(tuple(features[story, step, camera] for story, step, camera in pair))
              for pair in indices]
    return torch.stack(frames)


def changed_pixels(pair):
    mask = np.any(pair[0] != pair[1], axis=-1)
    return {name: {"pixels_changed_per_frame": mask[:, index].sum(axis=(1, 2)).tolist(),
                   "pixels_per_frame": mask.shape[2] * mask.shape[3]}
            for index, name in enumerate(CAMERAS)}


def predict(model, temperature, features, matched_scores, kind):
    outputs = {}
    for name, indices in positions().items():
        probability = decode_features(concatenated(features, indices), "containment", model, temperature)
        outputs[name] = {"winner": max(probability, key=probability.get), "report": probability}
    for index in (0, 1):
        archived = matched_scores[kind][index]["so101_containment_report"]
        replay = outputs[f"wrist_{index}_top_{index}"]["report"]
        if any(abs(value - float(archived[label])) > 1e-6 for label, value in replay.items()):
            raise ValueError("The original camera pair no longer reproduces the sealed witness report")
    return outputs


def run(study, recorded, camera_pairs, heartbeat):
    torch.set_num_threads(3)
    backbone = load_backbone(study)
    staged = {}
    for name, pair in camera_pairs.items():
        heartbeat()
        joined = np.concatenate(pair, axis=0)
        embeddings = backbone(joined).reshape(2, 8, 2, 512)
        staged[name] = {"features": embeddings, "camera_changes": changed_pixels(pair)}
    for entry in study["models"]:
        heartbeat()
        model = EvidenceHead().eval()
        frozen, _ = verified_model(entry)
        model.load_state_dict(frozen["weights"], strict=True)
        kind = entry["kind"]
        for name, item in staged.items():
            saved = recorded["conditions"][name]["frozen_heads"]
            values = predict(model, frozen["temperature"], item["features"], saved, kind)
            item.setdefault("heads", {})[kind] = values
    return {name: {key: value for key, value in item.items() if key != "features"} for name, item in staged.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a new diagnostic report path")
    source = PILOT / "primary-matched-history-controls.json"
    study = load(PILOT / "story-learning.json")
    recorded, pairs = inputs(source)
    with admitted(settings(), "smart-swarm-frozen-containment-channel-probe") as (grant, heartbeat):
        scored = run(study, recorded, pairs, heartbeat)
    atomic(args.output, {"scope": "Development-only model-input interventions", "paired_source_sha256": sha(source),
                         "independent_source_groups": 1,
                         "mixed_channels_are_counterfactual_model_inputs": True,
                         "input_index_zero": "successful placement", "input_index_one": "failed placement",
                         "reservation": grant, "conditions": scored})
    print(args.output)


if __name__ == "__main__":
    main()
