#!/usr/bin/env python3
"""Rehearse a resolvable pair with matched receiver observations and different witness histories."""

import argparse
from contextlib import closing
from pathlib import Path

import numpy as np

from freeze_primary_profiles import KINDS, require_profiles
from score_primary_test import inference, runtime
from score_primary_validation import CURRENT, SELECTIVE, TEMPORAL
from smart_swarm.cli import settings
from smart_swarm.episode import LABELS, Replay, atomic, sha
from smart_swarm.evidence_inputs import clip_packets, runtime_signature
from smart_swarm.primary_inputs import primary_policy_view
from smart_swarm.rendering import configure_renderer
from smart_swarm.research_jobs import admitted
from smart_swarm.vision import center


def store_labels(story, terminal, slot):
    from smart_swarm.evidence_portfolio import labels_for

    labels = labels_for(story, terminal, slot)
    if not any(item.factor == "containment" for item in story.evidence):
        labels.update(settled_placement=None, placement_masked_reason="target absent in last granted station frame",
                      resolvable=False)
    story.recorder.append(LABELS, "settled-event", labels)


def execute_story(config, path, seed, family, variant, wide_station):
    from smart_swarm.evidence_rollouts import prepare_world, reveal
    from smart_swarm.scaling import SLOT_STEPS, ScalingStory, describe_capture
    from smart_swarm.story import AGENTS

    class WitnessedCounterfactual(ScalingStory):
        def relocate(self):
            self.capture(AGENTS[2])
            self.world.step(SLOT_STEPS)
            self.capture(AGENTS[2])
            super().relocate()

    world, variation, pickup = prepare_world(config["asset_root"], seed, "F6", variant=variant)
    with closing(world):
        if wide_station:
            world.model.cam_fovy[world.model.camera("station").id] = 70.
        story = WitnessedCounterfactual(path, family, world, seed, family == "F2")
        try:
            describe_capture(story, variation, variant, 3, False, None)
            story.recorder.meta.update(experiment_role="development_matched_current_scene", pair_seed=seed,
                                       expanded_station_view=wide_station,
                                       scaling_receiver_profile="front-after-placement-before-and-after-relocation")
            story.recorder.meta["source_file_sha256"][Path(__file__).name] = sha(Path(__file__))
            story.scene(pickup)
            query, decision, report = story.ask()
            finished, terminal, slot = reveal(story, decision)
            store_labels(story, terminal, slot)
            story.recorder.close(query, decision, finished)
            return {"raw_path": str(path), "outcome": terminal["ground_truth"],
                    "physical_checks": story.recorder.meta["physical_story_gate"],
                    "rule_decision": report["query_all"]["decision"]}
        except (Exception, KeyboardInterrupt) as error:
            story.recorder.quarantine(str(error) or type(error).__name__)
            raise


def matched_scopes(paths):
    left, right = [runtime_signature(path)["private_history"] for path in paths]
    same = {name: left[name] == right[name] for name in left}
    if same != {"so101-1": False, "tank-1": True, "rosmaster-1": True}:
        raise ValueError(f"Unexpected match across the complete private scopes: {same}")
    return same


def station_observation(path):
    client = Replay(path, "so101-1")
    points = []
    for packet in clip_packets(client):
        frame = next(row["frame_id"] for row in packet["cameras"] if row["sensor_id"] == "so101-1/top")
        points.append(center(np.asarray(client.image(frame)), "cube"))
    return {"visible_frames": sum(point is not None for point in points),
            "clip_frames": len(points), "last_position": points[-1]}


def score(paths, profile, heartbeat):
    import torch

    torch.set_num_threads(3)
    forecasts = {}
    for kind in KINDS:
        heartbeat()
        frozen = runtime(profile, kind)
        views = []
        for path in paths:
            with primary_policy_view(path) as scoped:
                views.append(inference(scoped, 3, frozen, False))
        if any(views[0][method] != views[1][method] for method in (CURRENT, TEMPORAL)):
            raise ValueError("A local receiver policy changed under exactly matching receiver inputs")
        forecasts[kind] = views
    return forecasts


def run(config, args, profile, grant, heartbeat):
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [args.output / name for name in ("insertion-witness", "failed-placement-witness")]
    receipt = args.output / "development-pair.json"
    records = []
    for path, family, variant in zip(paths, ("F1", "F2"), ("hidden-contained", "hidden-surface"), strict=True):
        heartbeat()
        record = execute_story(config, path, args.seed, family, variant, args.wide_station)
        records.append({**record, "station_observation": station_observation(path)})
        atomic(receipt, {"status": "captured", "seed": args.seed, "records": records, "reservation": grant})
    same = matched_scopes(paths)
    forecasts = score(paths, profile, heartbeat)
    atomic(args.output / "predictions.json", forecasts)
    outcomes = [record["outcome"] for record in records]
    if outcomes[0] not in ("left", "right") or outcomes[1] != "surface":
        raise ValueError(f"The matched world did not produce the reserved outcomes: {outcomes}")
    selections = {kind: [item[SELECTIVE]["decision"] for item in rows] for kind, rows in forecasts.items()}
    atomic(receipt, {"status": "scored", "seed": args.seed, "shared_private_scopes": same, "records": records,
                     "independent_source_groups": 1, "correct_search_by_head": {k: v == outcomes for k, v in selections.items()},
                     "selective_decisions": selections, "profile_sha256": sha(args.profiles),
                     "predictions_sha256": sha(args.output / "predictions.json"), "reservation": grant})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=46_013)
    parser.add_argument("--wide-station", action="store_true", help="Expose the occluded surface through SO101's 70-degree simulated top view")
    args = parser.parse_args()
    profile = require_profiles(args.profiles, rehearsal=True)
    config = settings()
    configure_renderer(config)
    with admitted(config, "smart-swarm-matched-current-history-control") as (grant, heartbeat):
        run(config, args, profile, grant, heartbeat)
    print(args.output / "development-pair.json")


if __name__ == "__main__":
    main()
