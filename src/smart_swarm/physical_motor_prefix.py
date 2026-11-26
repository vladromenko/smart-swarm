"""Bind a witnessed placement prefix to the unchanged official Studio export."""

from hashlib import file_digest, sha256
import json
from pathlib import Path

from .contract import json_bytes
from .episode import atomic, load, rows, sha
from .studio import recorded_calibration

CAMERA_ORDER = ("wrist", "top")
MOTOR_FAMILIES = ("F1", "F3", "F4", "F5")
MINIMUM_PREFIX_ROWS = 30


def episode_directory(config, state):
    studio = config["studio_project"].resolve()
    directory = (studio / "data" / state["studio_dataset_id"] / "episodes" / state["native_episode_id"]).resolve()
    if not directory.is_relative_to(studio / "data"):
        raise ValueError("Motor prefix source is outside the native Studio data directory")
    return directory


def prefix_bounds(observations, first_ns, last_ns):
    indices = []
    for index, row in enumerate(observations):
        observed = row["observation_timestamp_ns"]
        sent = row["action_timestamp_ns"]
        if first_ns <= observed <= sent <= last_ns:
            if row["state_valid"] is not True or row["action_valid"] is not True:
                raise ValueError("The placement window contains invalid measured state or transmitted action")
            indices.append(index)
    if not indices:
        return 0, 0
    begin, stop = indices[0], indices[-1] + 1
    if len(indices) != stop - begin:
        raise ValueError("The placement prefix contains a gap in its source rows")
    return begin, stop


def segment_choice(config, state):
    card = state["card"]
    family = card["family"]
    if family not in MOTOR_FAMILIES:
        return None, "story family does not admit a useful placement prefix"
    boundary = "insertion_closed" if family == "F3" else "placement_closed"
    mark = state["markers"][boundary]
    if mark["observed_placement_outcome"] != "placement_succeeded":
        return None, "no observed successful placement at the prefix boundary"
    if mark["assistance"] not in ("none", "attended_native_motion"):
        return None, "placement required assistance outside native arm motion"
    path = episode_directory(config, state) / "frames.jsonl"
    observations = rows(path)
    begin, stop = prefix_bounds(observations, state["start"]["monotonic_ns"], mark["monotonic_ns"])
    if stop - begin < MINIMUM_PREFIX_ROWS:
        return None, "second camera and issued actions cover fewer than 30 eligible placement rows"
    return {"story": card["story_id"], "scene_group_id": card["source_group_id"], "split": card["split"],
            "native_episode": state["native_episode_id"], "native_frame_manifest_sha256": sha(path),
            "frame_index_start_inclusive": observations[begin]["frame_id"],
            "frame_index_end_inclusive": observations[stop - 1]["frame_id"], "sample_count": stop - begin,
            "episode_row_start": begin, "episode_row_stop_exclusive": stop, "camera_order": list(CAMERA_ORDER),
            "outcome": mark["observed_placement_outcome"], "outcome_source": "attended placement-end observation",
            "end_boundary": boundary, "end_monotonic_ns": mark["monotonic_ns"],
            "last_transmitted_action_ns": observations[stop - 1]["action_timestamp_ns"],
            "quality": "eligible_for_motor_dataset_review"}, None


def native_episode_rows(export, relative):
    with (export / "provenance/frames.jsonl").open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row["source_episode_relative"] == relative:
                yield row


def export_storage_digest(export):
    root = Path(export).resolve()
    files = sorted(path for directory in ("data", "meta") for path in (root / directory).rglob("*")
                   if path.is_file() and path.suffix in (".parquet", ".json"))
    if not any(path.relative_to(root).parts[0] == "data" for path in files):
        raise ValueError("The official Studio image-backed dataset has no data files")
    bindings = {}
    for path in files:
        if not path.resolve().is_relative_to(root):
            raise ValueError("Native export storage path leaves its original dataset")
        with path.open("rb") as stream:
            bindings[path.relative_to(root).as_posix()] = file_digest(stream, "sha256").hexdigest()
    return sha256(json_bytes(bindings)).hexdigest()


def official_prefix(export, episode, begin, stop, *, technical=False, storage_digest=None):
    export, episode = Path(export).resolve(), Path(episode).resolve()
    metadata = load(export / "provenance/export.json")
    root = Path(metadata["source_path"]).resolve()
    relative = episode.relative_to(root).as_posix()
    frame_path = episode / "frames.jsonl"
    observed_hash = sha(frame_path)
    admitted_source = "synthetic_test" if technical else "hardware"
    native_cameras = metadata["cameras"]
    if (metadata["status"] != "complete" or metadata["source"] != admitted_source
            or bool(metadata["test_only"]) != technical or sorted(native_cameras) != sorted(CAMERA_ORDER)
            or metadata["source_hashes"].get(f"{relative}/frames.jsonl") != observed_hash):
        raise ValueError("The official two-camera export is missing this exact native episode or has an ineligible source")
    calibration = recorded_calibration(load(episode / "episode.json"), metadata)
    if not technical and len(calibration) != 2:
        raise ValueError("Both original native live calibration readbacks must match the export snapshot")
    raw = rows(frame_path)
    approved = list(native_episode_rows(export, relative))
    if not 0 <= begin < stop <= len(raw) or len(raw) != len(approved):
        raise ValueError("The admitted native placement range does not map to the official export")
    first = approved[0]["lerobot_index"]
    episode_index = approved[0]["lerobot_episode_index"]
    for offset, (record, source) in enumerate(zip(approved, raw, strict=True)):
        if (record["lerobot_frame_index"] != offset or record["lerobot_index"] != first + offset
                or record["lerobot_episode_index"] != episode_index or record["raw_frame"] != source):
            raise ValueError("Native export row identity or boundaries changed")
    return {"path": str(export), "repo_id": metadata["repo_id"], "dataset_sha256": metadata["dataset_sha256"],
            "export_metadata_sha256": sha(export / "provenance/export.json"), "dataset_episode_index": episode_index,
            "export_storage_sha256": storage_digest or export_storage_digest(export),
            "dataset_row_start": first + begin, "dataset_row_stop_exclusive": first + stop,
            "source_frame_manifest_sha256": observed_hash, "nominal_fps": metadata["nominal_fps"],
            "native_export_camera_order": native_cameras, "live_calibration_snapshot_hashes": calibration,
            "camera_keys": [f"observation.images.{name}" for name in CAMERA_ORDER], "joint_units": metadata["joint_units"],
            "source_task": approved[begin]["task"], "action_semantics": metadata["action_semantics"],
            "source": admitted_source, "technical_only": technical, "future_actions_stop_exclusive": first + stop}


def save_prefix(config, trial, state, export, *, storage_digest=None):
    rejection = None
    try:
        choice, rejection = segment_choice(config, state)
        if choice is not None:
            proof = state.get("native_story_validation")
            episode = episode_directory(config, state)
            if proof and (proof["frames_sha256"] != choice["native_frame_manifest_sha256"]
                          or proof["episode_metadata_sha256"] != sha(episode / "episode.json")):
                raise ValueError("The native placement has changed since story validation")
            binding = official_prefix(export, episode, choice["episode_row_start"],
                                      choice["episode_row_stop_exclusive"], storage_digest=storage_digest)
            atomic(trial / "motor-segment.json", {**choice, "native_export": binding})
    except (OSError, ValueError, KeyError) as error:
        rejection = str(error)
    if rejection is not None:
        atomic(trial / "motor-intake.json", {"status": "excluded", "reason": rejection})
        return False
    atomic(trial / "motor-intake.json", {"status": "reviewable_prefix", "segment": "motor-segment.json"})
    return True


class NativePlacement:
    """View a single accepted prefix without copying frames or reading past its end."""

    def __init__(self, segment, *, technical=False):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        binding = segment["native_export"]
        root = Path(binding["path"])
        metadata_path = root / "provenance/export.json"
        metadata = load(metadata_path)
        expected = "synthetic_test" if technical else "hardware"
        if (binding["source"] != expected or metadata["source"] != expected
                or binding["technical_only"] != technical or metadata["test_only"] != technical
                or binding["repo_id"] != metadata["repo_id"] or sha(metadata_path) != binding["export_metadata_sha256"]):
            raise ValueError("The placement export binding changed or is not eligible for this reader")
        storage = binding.get("export_storage_sha256")
        if storage and export_storage_digest(root) != storage:
            raise ValueError("The native export data or metadata bytes changed")
        self._native = LeRobotDataset(binding["repo_id"], root=root, return_uint8=True)
        self._actions = self._native.hf_dataset.select_columns(["action"])
        self.start = binding["dataset_row_start"]
        self.stop = binding["future_actions_stop_exclusive"]
        if (not 0 <= self.start < self.stop <= len(self._native)
                or self.stop != binding["dataset_row_stop_exclusive"] or len(self) != segment["sample_count"]):
            raise ValueError("The native placement range no longer fits the verified dataset")
        expected_frame_ids = (segment["frame_index_start_inclusive"], segment["frame_index_end_inclusive"])
        for offset, frame in zip((0, len(self) - 1), expected_frame_ids, strict=True):
            row = self[offset]
            if row["episode_index"].item() != binding["dataset_episode_index"] or row["raw.frame_id"] != str(frame):
                raise ValueError("Native placement range no longer matches the original frame IDs")

    def __len__(self):
        return self.stop - self.start

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError("Index is outside the native placement prefix")
        return self._native[self.start + index]

    def action_chunk(self, index, length):
        import torch

        if not 0 <= index < len(self) or length < 1:
            raise ValueError("Action chunk starts outside the native placement prefix or is empty")
        valid = min(length, len(self) - index)
        start = self.start + index
        states = [torch.as_tensor(value) for value in self._actions[start:start + valid]["action"]]
        states.extend([states[-1]] * (length - valid))
        return {"action": torch.stack(states), "action_is_pad": torch.arange(length) >= valid}
