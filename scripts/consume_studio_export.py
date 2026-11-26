"""Check native Studio's unchanged export with LeRobot and its source sidecars."""

import argparse
from hashlib import sha256
from pathlib import Path
import json

import numpy as np
from datasets.exceptions import DatasetGenerationError

from consume_platform_export import JOINTS, as_rgb_bytes
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ORDER = ("wrist", "top")
CAMERAS = tuple(f"observation.images.{name}" for name in ORDER)
VECTORS = ("observation.state", "action")
TIMESTAMPS = ("timestamp_ns", "state_timestamp_ns", "action_timestamp_ns")
FLAGS = ("state_valid", "action_valid", "action_acknowledged")
EXPORT_MISMATCH_EXIT = 3


def read(path):
    return json.loads(path.read_text())


def verify(raw, output, test):
    meta = read(output / "provenance/export.json")
    catalogue = (raw / "dataset.json").read_bytes()
    source = json.loads(catalogue)
    if meta["status"] != "complete" or source["source"] != ("synthetic_test" if test else "hardware"):
        raise ValueError("Unexpected native export type or integrity status")
    hashes = meta["source_hashes"]
    if hashes.get("dataset.json") != sha256(catalogue).hexdigest():
        raise ValueError("Native dataset catalogue differs from the official export")
    if not test and (source.get("test_only") or source["camera_mode"] != "two_camera"):
        raise ValueError("Physical pilot requires the two-camera native recording mode")
    ds = LeRobotDataset(meta["repo_id"], root=output, return_uint8=True)
    images = {key for key in ds.features if key.startswith("observation.images.")}
    if images != set(CAMERAS) or any(tuple(ds.features[key]["names"]) != JOINTS for key in VECTORS):
        raise ValueError("Expected exactly the two SO101 policy cameras and native joint ordering")
    if ds.meta.fps != meta["nominal_fps"]:
        raise ValueError("Native nominal export rate changed")
    files = sorted((output / "provenance").glob("episode_*.json"))
    catalogue_episodes = meta["episodes"]
    if not files or len(files) != ds.num_episodes or len(files) != len(catalogue_episodes):
        raise ValueError("Native export must preserve episode provenance")
    total = 0
    episodes = []
    for index, (path, entry) in enumerate(zip(files, catalogue_episodes)):
        episode = verify_episode(ds, entry, index, total, raw, path, hashes)
        episodes.append(episode)
        total += episode["frames"]
    if total != len(ds):
        raise ValueError("Native episode frame counts differ from the official loader")
    return {"official_version": meta["lerobot_version"], "rgb_camera_order": list(ORDER),
            "episodes": episodes, "dataset_sha256": meta["dataset_sha256"],
            "calibration": meta.get("calibration", {}),
            "source": source["source"], "frames": total, "numeric_rows_verified": total, "technical_only": test,
            "action_semantics": meta["action_semantics"], "policy_training_eligible": False}


def episode_source(raw, identifier):
    if not identifier or Path(identifier).name != identifier:
        raise ValueError("Could not match native episode to source")
    source = (raw / "episodes" / identifier).resolve()
    if source.parent != raw / "episodes" or not (source / "frames.jsonl").is_file():
        raise ValueError("Could not match native episode to source")
    return source


def verified_ledger(source, raw, detail, hashes):
    frame_path = source / "frames.jsonl"
    ledger = frame_path.read_bytes()
    frame_hash = sha256(ledger).hexdigest()
    sidecar_hash = detail.get("frames_sha256")
    if (frame_hash != hashes.get(frame_path.relative_to(raw).as_posix())
            or (sidecar_hash and frame_hash != sidecar_hash)):
        raise ValueError(f"Native frame ledger differs from the official export: {source.name}")
    manifest = source / "episode.json"
    manifest_hash = sha256(manifest.read_bytes()).hexdigest()
    if manifest_hash != hashes.get(manifest.relative_to(raw).as_posix()):
        raise ValueError(f"Native episode metadata differs from the official export: {source.name}")
    events = [json.loads(row) for row in ledger.splitlines() if row.strip()]
    return events, {"frames_sha256": frame_hash, "episode_manifest_sha256": manifest_hash}


def verify_episode(ds, entry, index, start, raw, sidecar, hashes):
    detail = read(sidecar)
    source = episode_source(raw, entry["source_episode_id"])
    raw_events, integrity = verified_ledger(source, raw, detail, hashes)
    if not raw_events or len(raw_events) != int(entry["frames"]):
        raise ValueError("Native episode count changed")
    count = len(raw_events)
    expected = scalar_columns(raw_events, detail.get("session_id", "unknown"), index, start, ORDER)
    frame_table = ds.hf_dataset.select_columns([*VECTORS, *expected, "task_index", "timestamp"]).with_format(None)
    verify_numeric_rows(frame_table[start:start + count], raw_events, expected)
    for offset in sorted({0, count - 1}):
        sample = ds[start + offset]
        if sample["episode_index"].item() != index or sample["task"] != entry["task"]:
            raise ValueError("Native episode boundaries or task changed")
        verify_rgb(sample, source, raw_events[offset])
    return {"episode_id": source.name, "catalogue_index": index, "frames": count, "numeric_rows_verified": count,
            "session_id": detail.get("session_id"), "sidecar": sidecar.name, **integrity, **recorded_timing(expected, ORDER)}


def scalar_columns(rows, session, episode, start, cameras):
    count = len(rows)
    fields = {"index": list(range(start, start + count)), "episode_index": [episode] * count,
              "frame_index": list(range(count)), "raw.frame_id": [str(row["frame_id"]) for row in rows],
              "raw.session_id": [str(session)] * count}
    fields.update({f"raw.{key}": [row[key] for row in rows] for key in TIMESTAMPS})
    fields.update({f"raw.{key}": [bool(row.get(key, False)) for row in rows] for key in FLAGS})
    fields.update({f"raw.{cam}.timestamp_ns": [row["cameras"][cam]["timestamp_ns"] for row in rows] for cam in cameras})
    return fields


def verify_numeric_rows(table, rows, expected):
    for key, values in expected.items():
        actual, original = np.asarray(table[key]), np.asarray(values)
        if actual.dtype.kind != original.dtype.kind or not np.array_equal(actual, original):
            raise ValueError(f"Native numeric row identity changed in {key}")
    for key in VECTORS:
        original = np.asarray([row[key] for row in rows], dtype=np.float32)
        if not np.array_equal(table[key], original):
            raise ValueError(f"Native vector differs for {key}")
    if len(set(table["task_index"])) != 1:
        raise ValueError("Native task conditioning changed inside the episode")
    observation = np.asarray(expected["raw.timestamp_ns"], dtype=np.int64)
    relative = (observation - observation[0]) / 1e9
    error = np.abs(np.asarray(table["timestamp"]) - relative)
    if not np.all(error <= np.maximum(2e-7, 2e-7 * relative)):
        raise ValueError("Native measured relative time changed during export")


def recorded_timing(expected, cameras):
    observation = np.asarray(expected["raw.timestamp_ns"], dtype=np.int64)
    sent = np.asarray(expected["raw.action_timestamp_ns"], dtype=np.int64)
    state = np.asarray(expected["raw.state_timestamp_ns"], dtype=np.int64)
    ages = {name: observation - np.asarray(expected[f"raw.{name}.timestamp_ns"], dtype=np.int64) for name in cameras}
    return {"max_observation_period_ms": float(np.diff(observation).max(initial=0) / 1e6),
            "max_state_to_transmit_ms": float((sent - state).max() / 1e6),
            "max_camera_age_at_observation_ms": {name: float(values.max() / 1e6) for name, values in ages.items()},
            "valid_measured_states": sum(expected["raw.state_valid"]),
            "valid_transmitted_actions": sum(expected["raw.action_valid"]),
            "per_motor_acknowledgement_rows": sum(expected["raw.action_acknowledged"])}


def verify_rgb(sample, source, row):
    from PIL import Image

    for camera in ORDER:
        relative = row["cameras"][camera]["path"]
        image_path = (source / relative).resolve()
        if not image_path.is_relative_to(source):
            raise ValueError("Camera path is outside its episode")
        with Image.open(image_path) as image:
            expected = np.asarray(image.convert("RGB"))
        if not np.array_equal(expected, as_rgb_bytes(sample[f"observation.images.{camera}"])):
            raise ValueError("Native frame changed during export")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--allow-technical-rehearsal", action="store_true")
    args = parser.parse_args()
    try:
        result = verify(args.raw.resolve(), args.output.resolve(), args.allow_technical_rehearsal)
    except DatasetGenerationError as error:
        cause = error.__cause__
        if not isinstance(cause, (OSError, ValueError, TypeError)):
            raise
        parser.exit(EXPORT_MISMATCH_EXIT, f"{type(cause).__name__}: {cause}\n")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(EXPORT_MISMATCH_EXIT, f"{type(error).__name__}: {error}\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
