"""Write and verify standalone LeRobot 0.6.1 splits using the installed API."""

import argparse
import hashlib
from importlib.metadata import version
from pathlib import Path

import numpy as np

from smart_swarm.datasets import audit_lineage
from smart_swarm.episode import atomic, load
from smart_swarm.motor_input import (CLOCKED_PROFILE, EXACT_REPLAY_ISSUES, FLOAT32_COMMAND_PROFILE, PROFILE,
                                     conditioned_top, instruction, verify_issue_time)
from smart_swarm.motor_visual_profile import match_visual

KEYS = {"wrist": "observation.images.camera1", "top": "observation.images.camera2"}
SCHEMA = "sim-so101-absolute-v1"
HISTORICAL = "historical-rgb"


def entries(root, split):
    return [(path, load(path / "manifest.json")) for path in sorted((root / split).glob("*"))
            if (path / "manifest.json").is_file()]


def features(names, image_shape):
    descriptions = {
        "observation.state": {"dtype": "float32", "shape": (6,), "names": names},
        "action": {"dtype": "float32", "shape": (6,), "names": names},
    }
    descriptions.update({key: {"dtype": "image", "shape": image_shape,
                               "names": ["height", "width", "channels"]} for key in KEYS.values()})
    return descriptions


def load_episode(path, meta, aligned=False):
    recording = path / "motor.npz"
    if (meta["status"] != "accepted" or meta["action_schema_id"] != SCHEMA
            or hashlib.sha256(recording.read_bytes()).hexdigest() != meta["sha256"]):
        raise ValueError(f"Unverified motor source {path.name}")
    array = np.load(recording, allow_pickle=False)
    times = array["timestamps_ns"]
    count = meta["sample_count"]
    delta = (times - times[0]) - np.arange(count, dtype=np.float64) * 1e9 / meta["fps"]
    if delta.size != count or np.max(np.abs(delta)) > 500_000:
        raise ValueError("Motor timestamps exceed simulation-clock quantization")
    if aligned:
        verify_issue_time(times, array["applied_interval_start_ns"])
    data = {key: array[key] for key in (*KEYS, "state", "requested")}
    if any(value.shape[0] != count or not np.isfinite(value).all() for value in data.values()):
        raise ValueError("Missing or non-finite motor frames")
    if any(data[key].dtype != np.uint8 or data[key].shape[-1] != 3 for key in KEYS):
        raise ValueError("Both admitted camera streams must be unannotated RGB images")
    return data


def write_one(split, episodes, root, input_profile, visual):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    aligned = input_profile != HISTORICAL
    first, info = episodes[0]
    shape = load_episode(first, info, aligned)["wrist"].shape[1:]
    destination = root / split
    repo = f"smart-swarm/{split}"
    ds = LeRobotDataset.create(repo, root=destination, fps=info["fps"], robot_type="so101",
                               features=features(info["names"], shape), use_videos=False)
    ancestry = []
    for idx, (path, meta) in enumerate(episodes):
        data = load_episode(path, meta, aligned)
        for i in range(meta["sample_count"]):
            ds.add_frame(make_frame(data, meta, i, input_profile))
        ds.save_episode()
        ancestry.append({"dataset_episode_index": idx, "source_world_episode_id": path.name,
                         "scene_group_id": meta["scene_group_id"], "source_sha256": meta["sha256"],
                         "source_split": split, "frame_count": meta["sample_count"],
                         "raw_path": str(path.resolve()), "target_marker": meta.get("target_marker", "cyan")})
    ds.finalize()
    validate_export(destination, repo, ancestry, ds.num_frames, input_profile, visual)


def make_frame(data, meta, i, input_profile):
    rgb = {name: data[key][i] for key, name in KEYS.items()}
    text = meta["task"]
    if input_profile != HISTORICAL:
        marker = meta.get("target_marker", "cyan")
        issue_index = i if input_profile == CLOCKED_PROFILE else None
        rgb[KEYS["top"]] = conditioned_top(rgb[KEYS["top"]], marker, issue_index=issue_index)
        text = instruction(text, marker)
    return {"observation.state": data["state"][i], "action": data["requested"][i], "task": text, **rgb}


def audit_rows(dataset, start, entry, input_profile):
    source = Path(entry["raw_path"])
    meta = load(source / "manifest.json")
    data = load_episode(source, meta, aligned=True)
    count = entry["frame_count"]
    if input_profile == CLOCKED_PROFILE and meta.get("command_execution_profile") == FLOAT32_COMMAND_PROFILE:
        indices = sorted({min(index, count - 1) for index in EXACT_REPLAY_ISSUES} | {count - 1})
    elif input_profile == CLOCKED_PROFILE:
        indices = sorted({0, min(41, count - 1), min(58, count - 1), count - 1})
    else:
        indices = [0]
    for index in indices:
        check_row(dataset[start + index], make_frame(data, meta, index, input_profile))


def check_row(row, expected):
    if row["task"] != expected["task"]:
        raise ValueError("Requested target text changed in export")
    for key in ("observation.state", "action"):
        saved = row[key] if key != "action" else row[key][0]
        if not np.array_equal(saved.numpy(), expected[key]):
            raise ValueError(f"{key} does not match the same issue-time raw sample")
    for key in KEYS.values():
        pixels = np.rint(row[key].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        if not np.array_equal(pixels, expected[key]):
            raise ValueError(f"Camera bytes changed in export for {key}")


def validate_export(destination, repo, ancestry, frame_count, input_profile, visual):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    offsets = {"action": [0., 1/30, 2/30]}
    ds = LeRobotDataset(repo, root=destination, delta_timestamps=offsets)
    end = 0
    for entry in ancestry:
        end += entry["frame_count"]
        final = ds[end - 1]
        if input_profile != HISTORICAL:
            audit_rows(ds, end - entry["frame_count"], entry, input_profile)
        if (final["episode_index"].item() != entry["dataset_episode_index"]
                or final["action_is_pad"].tolist() != [False, True, True]):
            raise ValueError("Action chunk must pad inside each source episode")
    expected = {*KEYS.values()}
    observed = {key for key in ds.features if key.startswith("observation.images.")}
    if observed != expected or ds.num_episodes != len(ancestry) or len(ds) != frame_count:
        raise ValueError("Reopened native split changed camera or episode membership")
    atomic(destination / "smart-swarm-lineage.json",
           {"lerobot": version("lerobot"), "allowed_images": KEYS, "frames": len(ds),
            "episodes": ancestry, "episode_padding_mask_checked": True,
            "input_profile": input_profile, **visual,
            "control_timing": "aligned-sample-v2" if input_profile != HISTORICAL else "historical-data"})


def export(source, output, input_profile=HISTORICAL):
    if version("lerobot") != "0.6.1":
        raise RuntimeError("Run export in the verified LeRobot 0.6.1 runtime")
    groups = {name: entries(source, name) for name in ("train", "validation", "test")}
    if not groups["train"] or not groups["validation"]:
        raise ValueError("Need separate captured train and validation worlds")
    manifests = [row for split in groups.values() for _, row in split]
    audit_lineage(manifests)
    visual = match_visual(*manifests)
    if output.exists():
        raise FileExistsError(output)
    for split, episodes in groups.items():
        if episodes:
            write_one(split, episodes, output, input_profile, visual)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    formats = parser.add_mutually_exclusive_group()
    formats.add_argument("--aligned", dest="input_profile", action="store_const", const=PROFILE)
    formats.add_argument("--clocked", dest="input_profile", action="store_const", const=CLOCKED_PROFILE)
    parser.set_defaults(input_profile=HISTORICAL)
    args = parser.parse_args()
    export(args.source, args.output, args.input_profile)
