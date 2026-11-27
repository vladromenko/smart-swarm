"""Inventory calibrated legacy Studio demonstrations and their own checkpoint lineage."""

from collections import Counter
from pathlib import Path

from .episode import atomic, load, sha
from .studio import native_request, recorded_calibration

KNOWN_DATA = "ds_8ba060b36e1440fe"
HARDWARE_OUTCOMES = {"placement_succeeded", "placement_failed"}


def native_episodes(config, dataset):
    root = config["studio_project"] / "data" / dataset
    index = load(root / "dataset.json")
    rows = native_request(config, f"/api/episodes/{dataset}?include_frames=false")
    statuses = Counter(f"{row['storage']}:{row['metadata'].get('status')}" for row in rows)
    complete = [row for row in rows if row["storage"] == "episodes" and row["metadata"]["status"] == "complete"]
    results = [audit_episode(row, index, root) for row in complete]
    return {"dataset": dataset, "directory": str(root), "inventory": dict(statuses),
            "raw_manifest_hash_and_live_readback_verified": sum(row["native_structure"] for row in results),
            "wrist_only_candidate_demonstrations": sum(row["native_structure"] and row["cameras"] == ["wrist"]
                                                       for row in results),
            "native_two_camera_candidates": sum(row["native_structure"] and row["cameras"] == ["wrist", "top"]
                                                 for row in results),
            "outcomes_explicitly_observed": sum(row["observed_outcome"] in HARDWARE_OUTCOMES for row in results),
            "eligible_current_two_camera_motor_train": sum(row["eligible_current_motor"] for row in results),
            "accepted_camera_order": ["wrist", "top"], "episodes": results}


def audit_episode(row, index, root):
    metadata = row["metadata"]
    identifier = row["id"]
    raw = root / "episodes" / identifier / "frames.jsonl"
    native = (raw.is_file() and sha(raw) == metadata.get("frames_sha256") and metadata["frames"] > 0
              and metadata.get("source") == "hardware" and metadata.get("test_only") is False
              and metadata.get("last_timestamp_ns", 0) > metadata.get("first_timestamp_ns", 0))
    native = native and len(recorded_calibration(metadata, index)) == 2
    cameras = metadata["cameras"]
    outcome = metadata.get("operator_labels", {}).get("placement_outcome")
    return {"id": identifier, "session": metadata["session_id"], "cameras": cameras,
            "frames": metadata["frames"], "native_structure": bool(native), "observed_outcome": outcome,
            "eligible_current_motor": bool(native and set(cameras) == {"wrist", "top"} and outcome == "placement_succeeded"),
            "timing_protocol": metadata["recording_protocol"]}


def checkpoints(config, dataset):
    projects = config["studio_project"]
    root = projects / "models"
    results = []
    for summary in sorted(root.glob("*/result.json")):
        lineage = summary.with_name("lineage.json")
        if lineage.is_file():
            data = load(lineage)
            if Path(data.get("dataset_path", "/")).name == dataset and not data.get("technical_test"):
                results.append(checkpoint_summary(summary, data, root))
    return results


def checkpoint_summary(summary, lineage, root):
    result = load(summary)
    checkpoint = Path(result["checkpoint"]).resolve() / "pretrained_model"
    if not checkpoint.is_relative_to(root):
        raise ValueError("Native checkpoint escapes Studio's model directory")
    profile = load(checkpoint / "config.json")
    feature = lineage["feature_contract"]
    cameras = [key for key in profile["input_features"] if key.startswith("observation.images.")]
    return {"model": result["policy_type"], "checkpoint": str(checkpoint), "training_update": result["step"],
            "offline_loss": result["loss"], "native_offline_test_loss": result["test_loss"],
            "native_camera_mode": feature["camera_mode"], "feature_mode": feature["mode"],
            "image_inputs": cameras, "base_model": lineage["base_model"], "base_revision": lineage["base_revision"],
            "backbone_revision": lineage["backbone_revision"], "producer_versions": lineage["versions"],
            "weight_file_present": (checkpoint / "model.safetensors").is_file(),
            "preprocessing_sidecars": sorted(path.name for path in checkpoint.glob("policy*processor*.json")),
            "eligible_current_two_camera_comparison": False,
            "task_specific_closed_loop_result": None}


def run(config, output):
    data = native_episodes(config, KNOWN_DATA)
    result = {"recording audit": data, "archived native checkpoint audit": checkpoints(config, KNOWN_DATA),
              "purpose": "read-only inventory. Native structural quality and semantics are independently scored.",
              "operator labels": "A missing observed outcome remains unknown."}
    atomic(output, result)
    return result
