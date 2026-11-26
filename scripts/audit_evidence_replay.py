#!/usr/bin/env python3
"""Compare receiver RGB and compact query schedules on the consumed v3 test split."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from learn_evidence_v3 import decide, metrics
from smart_swarm.episode import POLICY, Replay, atomic, listed_file, load, sha
from smart_swarm.evidence_budget_replay import exchange
from smart_swarm.evidence_comparison import OUTPUTS, choose_threshold, decision_cost, grouped_mean, learned_prior_map
from smart_swarm.evidence_queries import RECEIVER, WITNESSES, opportunities, paired_query_traces
from smart_swarm.particle_memory import exact_filter
from smart_swarm.vision import center

BYTE_CAPS = (2048, 3072)
HORIZONS_MS = (90, 120)
LOCAL_MODES = ("current_rgb", "local_rgb_history")
ROOT = Path(__file__).resolve().parents[1]


def receiver_rgb(client):
    camera = f"{RECEIVER}/front"
    frames = [image for packet in client.packets() for image in packet["cameras"] if image["sensor_id"] == camera]
    visible = [center(np.asarray(client.image(image["frame_id"])), "cube") is not None for image in frames]
    return {"frames": len(frames), "current_rgb": visible[-1] if visible else None,
            "local_rgb_history": any(visible) if visible else None,
            "rgb_sha256": [image["sha256"] for image in frames]}


def local_sensor_model(rows, rgb, mode):
    sizes = Counter(row["scene_group_id"] for row in rows)
    counts = {name: [0.5, 0.5] for name in ("surface", "other")}
    for row in rows:
        cue = rgb[row["world_episode_id"]][mode]
        if cue is not None:
            category = "surface" if row["terminal"] == "surface" else "other"
            counts[category][int(cue)] += 1 / sizes[row["scene_group_id"]]
    return {category: [count / sum(bucket) for count in bucket] for category, bucket in counts.items()}


def local_belief(prior, model, cue):
    if cue is None:
        return prior.copy()
    weighted = {name: q * model["surface" if name == "surface" else "other"][int(cue)]
                for name, q in prior.items()}
    mass = sum(weighted.values())
    return {name: q / mass for name, q in weighted.items()}


def local_score(rows, rgb, mode, model, prior):
    return [{"world_episode_id": row["world_episode_id"], "scene_group_id": row["scene_group_id"],
             "family": row["family"], "resolvable": row["resolvable"], "terminal": row["terminal"],
             "settled_placement": None, "relocation_assignment": None,
             "wire": {"local_rgb": rgb[row["world_episode_id"]]["rgb_sha256"]}, "serialized_bytes": 0,
             "belief": local_belief(prior, model, rgb[row["world_episode_id"]][mode])} for row in rows]


def evaluated(rows, threshold):
    scored = decide(rows, threshold)
    summary = metrics(scored)
    summary.pop("event_labels")
    return {**summary, "decision_cost": decision_cost(scored), "coupled_f6_pairs": paired_query_traces(scored)}


def local_report(records, prior):
    partitions = {split: [row for row in records if row["split"] == split] for split in ("train", "validation", "test")}
    rgb = {row["world_episode_id"]: receiver_rgb(Replay(Path(row["path"]), RECEIVER)) for row in records}
    approaches = {}
    for mode in LOCAL_MODES:
        sensor = local_sensor_model(partitions["train"], rgb, mode)
        validation = local_score(partitions["validation"], rgb, mode, sensor, prior)
        threshold, _, _ = choose_threshold(validation, decide, metrics)
        test = local_score(partitions["test"], rgb, mode, sensor, prior)
        approaches[mode] = {"threshold": threshold, "train_likelihood_by_surface": sensor,
                            "test": evaluated(test, threshold)}
    return {"scope": "Receiver front RGB only, no messages or proprioception. Color cue calibrated on train, threshold on validation.",
            "methods": approaches, "test_rgb": {row["world_episode_id"]: rgb[row["world_episode_id"]]
                                               for row in partitions["test"]}}


def score_schedule(rows, windows, settings, frozen, gains):
    strategy, cap, horizon = settings
    result = []
    reference = learned_prior_map(frozen["priors"])
    for row in rows:
        query, deadline, ads = windows[row["world_episode_id"]]
        if deadline - query != HORIZONS_MS[0] * 1_000_000:
            raise ValueError("Archived query horizon changed")
        window = (query, query + horizon * 1_000_000, ads)
        state = exchange(window, cap, strategy, gains, lambda factor: row[OUTPUTS[factor]],
                         reference, frozen["state_prior"], frozen["threshold"])
        visible = dict.fromkeys(OUTPUTS.values())
        for reply in state["wire"]["replies"]:
            visible[OUTPUTS[reply["factor"]]] = {name: float(score) for name, score in reply["probabilities"].items()}
        result.append({**row, **visible, **state})
    return {"schedule": strategy, "byte_cap": cap, "horizon_ms": horizon, "threshold": frozen["threshold"],
            "max_serialized_bytes": max(row["serialized_bytes"] for row in result),
            "max_modeled_latency_ms": max(row["modeled_latency_ms"] for row in result),
            **{"mean_" + key: grouped_mean(result, lambda row: row[key])
               for key in ("serialized_bytes", "responses", "modeled_latency_ms")},
            "two_reply_variants": sum(row["responses"] == 2 for row in result),
            "test": evaluated(result, frozen["threshold"]),
            "wire_by_world": {row["world_episode_id"]: row["wire"] for row in result}}


def verified_model(model):
    checkpoint = Path(model["checkpoint"])
    if sha(checkpoint) != model["checkpoint_sha256"]:
        raise ValueError("Frozen weights changed")
    frozen = torch.load(checkpoint, map_location="cpu", weights_only=True)
    gains = model["communication"]["validation_expected_information_nats"]
    if (frozen["query_expected_information_nats"] != gains or frozen["threshold"] != model["threshold"]
            or frozen["state_prior"] != model["state_prior"]):
        raise ValueError("Archived inference policy disagrees with frozen metadata")
    return frozen, gains


def compressed_rgb_size(root):
    size = 0
    for agent in (RECEIVER, *WITNESSES):
        client = Replay(root, agent)
        cameras = [cam for packet in client.packets() for cam in packet["cameras"]]
        paths = {listed_file(root, cam["frame_ref"], POLICY) for cam in cameras}
        size += sum(path.stat().st_size for path in paths)
    return size


def run(study):
    portfolio = Path(study["dataset"])
    if sha(portfolio) != study["input_portfolio_sha256"]:
        raise ValueError("Frozen source portfolio changed")
    records = load(portfolio)["records"]
    model = next(model for model in study["models"] if model["kind"] == "events_only")
    frozen, gains = verified_model(model)
    heldout = model["test_outcomes"]
    if {r["world_episode_id"] for r in records if r["split"] == "test"} != {r["world_episode_id"] for r in heldout}:
        raise ValueError("Frozen forecasts and portfolio test worlds differ")
    windows = {row["world_episode_id"]: opportunities(Path(row["path"])) for row in heldout}
    settings = [(strategy, cap, horizon) for strategy in ("parallel", "sequential")
                for cap in BYTE_CAPS for horizon in HORIZONS_MS]
    reports = {"_".join(map(str, setting)): score_schedule(heldout, windows, setting, frozen, gains)
               for setting in settings}
    return {"status": "post-freeze replay diagnostic on consumed simulation sources",
            "checkpoint_sha256": model["checkpoint_sha256"], "portfolio_sha256": study["input_portfolio_sha256"],
            "frozen_query_gains_nats": gains, "transport": reports,
            "transport_scope": "All candidates use frozen pre-query witness clips. Modeled delivery excludes compute and network overhead.",
            "centralized_compact_scope": "An offboard broker can schedule the parallel arm using only compact advertisements and replies.",
            "all_private_rgb_payload": {"mean_archived_png_bytes": grouped_mean(
                heldout, lambda row: compressed_rgb_size(Path(row["path"]))),
                "scope": "Pre-query camera files from all three agent grants without evaluator images or transport overhead."},
            "receiver_only": local_report(records, exact_filter(frozen["state_prior"], [])["belief"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    pilot = ROOT / "evidence" / "learning-pilot-v3"
    parser.add_argument("--study", type=Path, default=pilot / "story-learning.json")
    parser.add_argument("--output", type=Path, default=pilot / "evidence-budget-postfreeze.json")
    args = parser.parse_args()
    report = run(load(args.study))
    atomic(args.output, {**report, "frozen_study_sha256": sha(args.study)})
    print(args.output)


if __name__ == "__main__":
    main()
