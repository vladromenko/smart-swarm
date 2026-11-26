"""Smart Swarm card, attempt provenance and A-export consumption."""

from collections import Counter
import os
from pathlib import Path
import subprocess
import uuid

from .contract import VERSION, ProtocolError
from .episode import atomic, load, sha

ROOT = Path(__file__).resolve().parents[2]
CAMERAS = ("so101-1/wrist", "so101-1/top")
FIXTURE = "robot-platform-fixture/two-camera-v1"


def prepare(config):
    root = config["data_root"] / "so101-pilot"
    root.mkdir(exist_ok=True, parents=True)
    path = root / "card.json"
    template = load(ROOT / "configs/so101-pilot.json")
    if not path.is_file():
        group = "so101-attended-" + uuid.uuid4().hex[:12]
        attempts = [{"attempt": i, "source_world_episode_id": f"{group}-{i:02}", "scene_group_id": group}
                    for i in range(1, template["attempt_limit"] + 1)]
        atomic(path, {**template, "scene_group_id": group, "attempts": attempts})
    return {**load(path), **template, "card_path": str(path), **pilot_status(config)}


def pilot_status(config):
    data = config["data_root"]
    state = load(data / "so101-pilot-status.json") if (data / "so101-pilot-status.json").is_file() else {}
    return {"project": "smart-swarm", "state": "blocked", "physical_preflight": "pending", **state}


def count_attempts(records):
    counts = Counter(record["classification"] for record in records)
    return {"attempts": len(records), "accepted_motor": 0, "eligible_for_motor_review": counts["eligible_for_motor_review"],
            "valid_story_failures": counts["story_failure"], "incomplete": counts["incomplete"],
            "fixture_only": counts["fixture_only"], "rejected": counts["rejected"],
            "unknown_outcome": counts["outcome_unknown"]}


def classify(manifest, validation):
    if manifest["recording_integrity"] != "complete":
        return "incomplete", [manifest.get("quarantine_reason", "recording_incomplete")]
    if manifest["validation_status"] != "valid" or not validation:
        return "rejected", ["export_validation_failed_or_missing"]
    outcome = manifest.get("task_outcome_label")
    if manifest["run_mode"] == "fixture":
        return "fixture_only", ["fixture_has_no_manipulation_evidence"]
    if outcome in ("placement_failed", "insertion_failed"):
        return "story_failure", [outcome]
    if outcome != "placement_succeeded":
        return "outcome_unknown", ["observed_terminal_outcome_required"]
    if not validation.get("native_action_mapping_verified"):
        return "rejected", ["native_action_mapping_unverified"]
    return "eligible_for_motor_review", []


def consume(config, recording, export, destination):
    manifest = load(recording / "manifest.json")
    full = manifest["recording_integrity"] == "complete"
    verified = validate_export(config, recording, export) if full and export else None
    classification, reasons = classify(manifest, verified)
    declared = manifest["lineage"].get("declared", {})
    result = {"protocol_version": VERSION, "recording_id": manifest["recording_id"],
              "source_world_episode_id": manifest["world_episode_id"],
              "scene_group_id": manifest["scene_group_id"], "split": manifest["lineage"]["split"],
              "source_lineage": manifest["lineage"], "task_id": manifest["task_id"],
              "recording_integrity": manifest["recording_integrity"], "platform_recording": str(recording),
              "platform_export": str(export) if export else None, "validation": verified,
              "outcome": manifest.get("task_outcome_label"), "classification": classification,
              "reasons": reasons, "declared": declared, "source_manifest_sha256": sha(recording / "manifest.json")}
    atomic(destination, result)
    return result


def validate_export(config, recording, export):
    child = [config["platform_python"], str(ROOT / "scripts/consume_platform_export.py"),
             str(recording), str(export)]
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1",
           "HF_DATASETS_OFFLINE": "1", "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"}
    result = subprocess.run(child, env=env, capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise ValueError(f"A export consumer failed: {result.stderr.strip()}")
    import json

    return json.loads(result.stdout)


def choose(config, args):
    card = prepare(config)
    if not args.plan_only:
        raise ProtocolError("OFFLINE", "Native SO101 capture and export await A's attended preflight")
    return card
