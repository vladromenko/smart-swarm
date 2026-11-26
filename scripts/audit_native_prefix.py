"""Reopen the original native technical export and check a bounded action prefix."""

import argparse
from pathlib import Path

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from consume_studio_export import verify
from smart_swarm.episode import atomic, load, rows
from smart_swarm.physical_motor_prefix import NativePlacement, official_prefix


def reject_physical_read(export, episode, segment):
    checks = (lambda: official_prefix(export, episode, 0, 2), lambda: NativePlacement(segment))
    for read in checks:
        try:
            read()
        except ValueError:
            pass
        else:
            raise ValueError("Technical recordings must be excluded from physical intake")


def audit(rehearsal, output):
    export = Path(load(rehearsal)["export"]["path"])
    receipt = load(export / "provenance/export.json")
    raw = Path(receipt["source_path"])
    verification = verify(raw, export, True)
    episode = raw / "episodes" / verification["episodes"][1]["episode_id"]
    original = rows(episode / "frames.jsonl")
    binding = official_prefix(export, episode, 0, 2, technical=True)
    segment = {"native_export": binding, "sample_count": 2,
               "frame_index_start_inclusive": original[0]["frame_id"],
               "frame_index_end_inclusive": original[1]["frame_id"]}
    reject_physical_read(export, episode, segment)
    view = NativePlacement(segment, technical=True)
    tail = view.action_chunk(1, 3)
    actions = tail["action"]
    mask = tail["action_is_pad"].tolist()
    data = LeRobotDataset(receipt["repo_id"], root=export, return_uint8=True)
    following = data[binding["dataset_row_stop_exclusive"]]["action"]
    if mask != [False, True, True] or not torch.equal(actions, view[1]["action"].repeat(3, 1)):
        raise ValueError("Tail must contain the last admissible native action and two masked pads")
    if torch.equal(actions[0], following):
        raise ValueError("The archived next action must be distinguishable to test prefix isolation")
    result = {"scope": "Direct technical Studio export readback", "native_export": binding,
              "official_verified_export_frames": verification["frames"], "native_episode": episode.name,
              "rows_in_test_view": len(view), "last_chunk_pad_mask": mask,
              "next_official_row_differs_but_is_excluded": True, "technical_source_rejected_for_physical_intake": True,
              "physical_prefixes_admitted": 0, "model_optimizer_updates": 0}
    atomic(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rehearsal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.rehearsal.resolve(), args.output.resolve())
