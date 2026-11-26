"""Publish witnessed native placement ranges as an episode-bounded motor corpus."""

from bisect import bisect_right
from collections import Counter
from hashlib import sha256
from itertools import accumulate
import json
from pathlib import Path
import re

from .contract import json_bytes
from .episode import SPLITS, atomic, load, sha
from .physical_motor_prefix import CAMERA_ORDER, NativePlacement, export_storage_digest, segment_choice
from .studio import DATASET_PATTERN, reject_export_storage, rejected_native_exports
from .studio_export_storage import checked_export_path

SCHEMA = "smart-swarm-native-motor-prefixes-v1"
CAMERA_KEYS = tuple(f"observation.images.{name}" for name in CAMERA_ORDER)


def admitted_segment(config, trial, card, intake, verified_storage):
    if intake.get("segment") != "motor-segment.json":
        raise ValueError("Unrecognized native placement receipt")
    candidate = load(trial / intake["segment"])
    journal = load(trial / "trial.json")
    if journal["card"] != card:
        raise ValueError("The recorded card differs from its prepared source groups")
    expected, reason = segment_choice(config, journal)
    if expected is None or any(candidate.get(key) != value for key, value in expected.items()):
        raise ValueError(reason or "Physical placement range no longer fits its observed boundary")
    binding = candidate["native_export"]
    original = Path(binding["path"])
    dataset = journal["studio_dataset_id"]
    if not re.fullmatch(DATASET_PATTERN, dataset):
        raise ValueError("Motor prefix review requires a native Studio dataset ID")
    cache = trial.parent / "_native-exports" / dataset
    refusal = rejected_native_exports(cache).get(str(original))
    if refusal is not None:
        raise ValueError(refusal)
    export = checked_export_path(config, original)
    if binding["source_frame_manifest_sha256"] != candidate["native_frame_manifest_sha256"]:
        raise ValueError("Native frame identity disagrees between raw and exported placement")
    metadata = verified_export_metadata(cache, original, export, binding["export_metadata_sha256"])
    if (metadata["status"] != "complete" or metadata["source"] != "hardware" or metadata["test_only"]
            or metadata["dataset_sha256"] != binding["dataset_sha256"] or metadata["repo_id"] != binding["repo_id"]):
        raise ValueError("The original Studio export changed or is ineligible for physical data")
    check_segments([candidate], technical=False)
    check_export_storage(verified_storage, cache, original, export, binding.get("export_storage_sha256"))
    return candidate


def verified_export_metadata(cache, original, export, expected):
    meta = export / "provenance/export.json"
    contents = meta.read_bytes()
    actual = sha256(contents).hexdigest()
    if actual != expected:
        reject_export_storage(cache, original, expected, actual, material="provenance/export.json")
        raise ValueError("Native export data or metadata bytes changed")
    return json.loads(contents)


def check_export_storage(verified, cache, original, export, expected):
    if expected is not None:
        if export not in verified:
            verified[export] = export_storage_digest(export)
        actual = verified[export]
        if actual != expected:
            reject_export_storage(cache, original, expected, actual)
            raise ValueError("Native export data or metadata bytes changed")


def check_segments(segments, *, technical):
    seen_sources = set()
    split_by_group = {}
    source = "synthetic_test" if technical else "hardware"
    for segment in segments:
        binding = segment["native_export"]
        split = segment["split"]
        if (split not in SPLITS or binding["source"] != source or binding["technical_only"] is not technical
                or binding["camera_keys"] != list(CAMERA_KEYS) or segment["sample_count"] <= 0):
            raise ValueError("Ineligible source, split, cameras or sample count for the native motor corpus")
        if not technical and (segment["outcome"] != "placement_succeeded" or
                              set(binding["live_calibration_snapshot_hashes"]) != {"leader", "follower"}):
            raise ValueError("Physical motor prefixes require a witnessed success and both native calibration readbacks")
        register_source(segment, seen_sources, split_by_group)


def register_source(segment, seen, splits):
    identity = segment["native_export"]["source_frame_manifest_sha256"]
    group, split = segment["scene_group_id"], segment["split"]
    if identity in seen or splits.get(group, split) != split:
        raise ValueError("A native episode is repeated or one physical source group crosses splits")
    seen.add(identity)
    splits[group] = split


def inventory(config, workspace, cards):
    segments, rejected = [], []
    seen = set()
    splits = {}
    verified_storage = {}
    for card in cards:
        trial = workspace / card["story_id"]
        if (trial / "trial.json").is_file():
            try:
                path = trial / "motor-intake.json"
                intake = load(path) if path.is_file() else {"status": "not_reviewed"}
                if intake["status"] == "reviewable_prefix":
                    segment = admitted_segment(config, trial, card, intake, verified_storage)
                    register_source(segment, seen, splits)
                    segments.append(segment)
                else:
                    rejected.append({"story": card["story_id"], "status": intake["status"],
                                     "reason": intake.get("reason", "No completed motor prefix review")})
            except (OSError, ValueError, KeyError) as error:
                rejected.append({"story": card["story_id"], "status": "rejected", "reason": str(error)})
    return segments, rejected


def corpus_record(segments, *, technical=False):
    check_segments(segments, technical=technical)
    frames = Counter()
    for row in segments:
        frames[row["split"]] += row["sample_count"]
    return {"schema": SCHEMA, "source": "synthetic_test" if technical else "hardware",
            "camera_order": list(CAMERA_ORDER), "prefixes": segments, "prefix_count": len(segments),
            "source_group_count": len({row["scene_group_id"] for row in segments}),
            "frames_by_split": dict(frames), "action_chunk_boundary": "native placement end",
            "technical_only": technical}


def publish(config):
    from .physical_story_plan import prepare

    root = config["data_root"] / "physical-stories"
    segments, rejected = inventory(config, root, prepare(config)["cards"])
    record = corpus_record(segments)
    manifest = None
    if segments:
        digest = sha256(json_bytes(record) + b"\n").hexdigest()
        directory = root / "motor-datasets"
        directory.mkdir(parents=True, exist_ok=True)
        manifest = directory / f"{digest}.json"
        if not manifest.exists():
            atomic(manifest, record)
        if sha(manifest) != digest:
            raise ValueError("Immutable motor corpus content no longer matches its file identity")
    return {**record, "manifest": str(manifest) if manifest else None, "corpus_rejections": rejected}


class PhysicalMotorDataset:
    """Read official native RGB/state/actions with chunks bounded to each witnessed prefix."""

    def __init__(self, record, split, chunk_size, *, technical=False):
        if record["schema"] != SCHEMA or record["technical_only"] is not technical or split not in SPLITS:
            raise ValueError("Native motor corpus has an incompatible source, version or split")
        if type(chunk_size) is not int or chunk_size <= 0:
            raise ValueError("Action chunk size must be a positive integer")
        rows = record["prefixes"]
        check_segments(rows, technical=technical)
        self._episodes = [NativePlacement(row, technical=technical) for row in rows if row["split"] == split]
        self._stops = list(accumulate(len(episode) for episode in self._episodes))
        self._chunk_size = chunk_size

    def __len__(self):
        return self._stops[-1] if self._stops else 0

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError("Index is outside the native motor corpus split")
        location = bisect_right(self._stops, index)
        begin = self._stops[location - 1] if location else 0
        prefix_index = index - begin
        episode = self._episodes[location]
        row = episode[prefix_index]
        pictures = {key: row[key].float() / 255. for key in CAMERA_KEYS}
        return {**pictures, "observation.state": row["observation.state"], "task": row["task"],
                **episode.action_chunk(prefix_index, self._chunk_size)}
