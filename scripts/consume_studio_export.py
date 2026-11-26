"""Check native Studio's unchanged export with LeRobot and its source sidecars."""

import argparse
from pathlib import Path
import json

import numpy as np

from consume_platform_export import JOINTS, as_rgb_bytes
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ORDER = ("wrist", "top")
CAMERAS = tuple(f"observation.images.{name}" for name in ORDER)


def read(path):
    return json.loads(path.read_text())


def verify(raw, output, test):
    meta = read(output / "provenance/export.json")
    source = read(raw / "dataset.json")
    if meta["status"] != "complete" or source["source"] != ("synthetic_test" if test else "hardware"):
        raise ValueError("Unexpected native export type or integrity status")
    if not test and (source.get("test_only") or source["camera_mode"] != "two_camera"):
        raise ValueError("Physical pilot requires the two-camera native recording mode")
    ds = LeRobotDataset(meta["repo_id"], root=output, return_uint8=True)
    images = {key for key in ds.features if key.startswith("observation.images.")}
    if images != set(CAMERAS) or tuple(ds.features["action"]["names"]) != JOINTS:
        raise ValueError("Expected exactly the two SO101 policy cameras and native joint ordering")
    files = sorted((output / "provenance").glob("episode_*.json"))
    if not files or len(files) != ds.num_episodes:
        raise ValueError("Native export must preserve episode provenance")
    total = 0
    episodes = []
    for index, path in enumerate(files):
        detail = read(path)
        frames = int(detail["frames"])
        episode = verify_episode(ds, detail, index, total, raw, path)
        episodes.append(episode)
        total += frames
    if total != len(ds):
        raise ValueError("Native episode frame counts differ from the official loader")
    return {"official_version": meta["lerobot_version"], "rgb_camera_order": list(ORDER),
            "episodes": episodes, "dataset_sha256": meta["dataset_sha256"],
            "source": source["source"], "frames": total, "technical_only": test,
            "action_semantics": meta["action_semantics"], "policy_training_eligible": False}


def verify_episode(ds, detail, index, start, raw, sidecar):
    from PIL import Image

    raw_episodes = [folder for folder in (raw / "episodes").iterdir()
                    if read(folder / "episode.json").get("frames_sha256") == detail.get("frames_sha256")]
    identifier = detail.get("episode_id") or (raw_episodes[0].name if len(raw_episodes) == 1 else None)
    if not identifier or not (raw / "episodes" / identifier / "frames.jsonl").is_file():
        raise ValueError("Could not match native episode to source")
    source = raw / "episodes" / identifier
    raw_events = [json.loads(row) for row in (source / "frames.jsonl").read_text().splitlines()]
    if len(raw_events) != int(detail["frames"]):
        raise ValueError("Native episode count changed")
    for offset in sorted({0, len(raw_events) - 1}):
        sample = ds[start + offset]
        row = raw_events[offset]
        if sample["episode_index"].item() != index or sample["task"] != detail["task"]:
            raise ValueError("Native episode boundaries or task changed")
        vectors = "observation.state", "action"
        for key in vectors:
            if not np.array_equal(sample[key].numpy(), np.asarray(row[key], dtype=np.float32)):
                raise ValueError(f"Native vector differs for {key}")
        verify_timing(sample, row)
        for camera in ORDER:
            relative = row["cameras"][camera]["path"]
            image_path = (source / relative).resolve()
            if not image_path.is_relative_to(source):
                raise ValueError("Camera path is outside its episode")
            with Image.open(image_path) as image:
                expected = np.asarray(image.convert("RGB"))
            if not np.array_equal(expected, as_rgb_bytes(sample[f"observation.images.{camera}"])):
                raise ValueError("Native frame changed during export")
    return {"episode_id": identifier, "catalogue_index": index, "frames": len(raw_events),
            "session_id": detail.get("session_id"), "sidecar": sidecar.name}


def verify_timing(sample, row):
    stamps = {"raw.timestamp_ns": row["timestamp_ns"],
              "raw.state_timestamp_ns": row["state_timestamp_ns"],
              "raw.action_timestamp_ns": row["action_timestamp_ns"]}
    stamps.update({f"raw.{camera}.timestamp_ns": row["cameras"][camera]["timestamp_ns"] for camera in ORDER})
    differences = [key for key, stamp in stamps.items() if int(sample[key].item()) != int(stamp)]
    if differences:
        raise ValueError(f"Native timestamps changed in {differences}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--allow-technical-rehearsal", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.raw.resolve(), args.output.resolve(), args.allow_technical_rehearsal)))
