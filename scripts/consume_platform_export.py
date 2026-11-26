"""Read A's unmodified LeRobot export and independently check the v1 fixture."""

import hashlib
import json
from pathlib import Path
import sys
import warnings

import numpy as np
from PIL import Image

CAMERAS = ("so101-1/wrist", "so101-1/top")
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
REPO = "robot-platform-fixture/two-camera-v1"


def read(path):
    return json.loads(path.read_text())


def stream(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def frame(recording, camera):
    rel = camera["frame_ref"]
    path = (recording / rel).resolve()
    if not path.is_relative_to(recording / "frames") or hashlib.sha256(path.read_bytes()).hexdigest() != camera["sha256"]:
        raise ValueError("Invalid admitted camera path or content hash")
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def verify(recording, export):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    manifest = read(recording / "manifest.json")
    lineage = read(export / "robot-platform-lineage.json")
    if manifest["run_mode"] != "fixture" or lineage["policy_training_eligibility"] != "fixture_only":
        raise ValueError("Native SO101 export schema has not been supplied by A")
    packets = stream(recording / "observations.jsonl")
    observations = packets[0]
    dataset = LeRobotDataset(REPO, root=export, return_uint8=True)
    features = tuple(key for key in dataset.features if key.startswith("observation.images."))
    expected = tuple("observation.images." + camera.replace("/", "_") for camera in CAMERAS)
    if features != expected or len(dataset) != len(packets) or len(dataset) != 1 or dataset.num_episodes != 1:
        raise ValueError("Export changed policy camera scope or the single-sample episode boundary")
    check_lineage(manifest, observations, lineage)
    sample = dataset[0]
    if sample["task"] != manifest["task_id"] or tuple(dataset.features["action"]["names"]) != JOINTS:
        raise ValueError("Export changed declared task or action order")
    for key, camera in zip(features, observations["cameras"], strict=True):
        if not np.array_equal(as_rgb_bytes(sample[key]), frame(recording, camera)):
            raise ValueError(f"Raw and exported policy RGB differ for {key}")
    action = lineage["applied_command"]
    if (action["native_applied_values"] is not None or not np.array_equal(sample["action"], action["adapter_values"])
            or not np.array_equal(sample["observation.state"], observations["measured_state"])):
        raise ValueError("Export changed measured state or fixture target semantics")
    return {"frames": len(dataset), "camera_order": list(CAMERAS), "rgb_bit_exact": True,
            "state_exact": True, "fixture_adapter_target_exact": True, "restricted_camera_excluded": True,
            "lineage_verified": True, "native_action_mapping_verified": False, "training_eligible": False}


def as_rgb_bytes(tensor):
    rgb = tensor.numpy().transpose(1, 2, 0)
    if np.issubdtype(rgb.dtype, np.floating):
        rounded = np.rint(rgb * 255)
        if np.any(np.abs(rounded / 255 - rgb) > 1e-6) or np.any((rgb < 0) | (rgb > 1)):
            raise ValueError("Float RGB is not a lossless byte scaling")
        return rounded.astype(np.uint8)
    if rgb.dtype != np.uint8:
        raise ValueError(f"Unsupported RGB dtype {rgb.dtype}")
    return rgb


def check_lineage(manifest, observations, lineage):
    if (lineage["source_world_episode_id"] != manifest["world_episode_id"]
            or lineage["scene_group_id"] != manifest["scene_group_id"]
            or lineage["lineage"]["source_world_ids"] != manifest["lineage"]["source_world_ids"]
            or lineage["lineage"]["source_scene_group_id"] != manifest["lineage"]["source_scene_group_id"]
            or lineage["lineage"]["split"] != manifest["lineage"]["split"]
            or tuple(lineage["policy_camera_ids"]) != CAMERAS
            or tuple(row["sensor_id"] for row in observations["cameras"]) != CAMERAS
            or lineage["source_time"] != observations["source_time"]
            or lineage["receive_time"] != observations["receive_time"]
            or lineage["clock_mapping"] != manifest["clock_mapping"]
            or lineage["applied_command"]["command_id"] != observations["applied_command_ref"]):
        raise ValueError("Export changed task lineage, camera order, clocks or action association")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
    print(json.dumps(verify(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())))
