"""Minimal Smart Swarm adapter to native SO101 Studio capture and export."""

import json
import os
from pathlib import Path
import re
import subprocess
from urllib.request import ProxyHandler, Request, build_opener
import uuid

from .contract import ProtocolError
from .episode import atomic, load, sha
from .platform_client import NoRedirect
from .so101_pilot import CAMERAS, ROOT, prepare

STUDIO_TEST = "runtime/acceptance-final-verified/test_official_lerobot_cpu_load0/raw_moved"
DATASET_PATTERN = r"ds_[a-f0-9]{16}"
OUTCOMES = ("placement_succeeded", "placement_failed", "unknown")
STOP_UNCONFIRMED = "torque_disable_unconfirmed_operator_check_required"


def native_request(config, path, body=None):
    if config["studio_origin"] != "http://127.0.0.1:8020":
        raise ValueError("This adapter expects Homebrain SO101 Studio on loopback")
    request = Request(config["studio_origin"] + path, data=json.dumps(body).encode() if body is not None else None,
                      headers={"Content-Type": "application/json"} if body is not None else {})
    with build_opener(ProxyHandler({}), NoRedirect()).open(request, timeout=5) as response:
        return json.load(response)


def native_status(config):
    report = native_request(config, "/api/status")
    inventory = report["resources"]["device_inventory"]
    devices = {key: bool(inventory.get(key, {}).get("present")) for key in ("wrist", "top", "leader", "follower")}
    active = [job for job in report["jobs"] if job.get("state") in ("running", "queued", "waiting_resources")]
    teleop = report["teleoperation"]
    recording = report["recording"]
    readback = (teleop.get("provenance") or {}).get("live_calibration") or {}
    return {"source": config["studio_origin"], "inventory_present": devices,
            "recording_active": recording.get("active", False),
            "teleoperation_active": teleop.get("active", False),
            "teleoperation_phase": teleop.get("phase") if teleop.get("active") else None,
            "native_calibration_readback": {arm: readback.get(arm, {}).get("verified_live")
                                            for arm in ("leader", "follower")},
            "native_calibration_readback_monotonic_ns": {arm: readback.get(arm, {}).get("timestamp_ns")
                                                        for arm in ("leader", "follower")},
            "native_stop_state": {"teleoperation": teleop.get("hardware_stop_state"),
                                  "recording": recording.get("hardware_stop_state")},
            "native_stop_check_required": any(session.get("hardware_stop_state") == STOP_UNCONFIRMED
                                              for session in (teleop, recording)),
            "native_arm_hardware_open": any(session.get("hardware_open") is True
                                            for session in (teleop, recording)),
            "policy_active": any(job.get("kind") == "policy" for job in active),
            "learning_job_active": any(job.get("kind") in ("train", "annotate") for job in active),
            "native_stop_generation": report["hardware_stop_generation"],
            "native_camera_preview": preview_status(report["preview"]),
            "physical_preflight_verified": False}


def preview_status(preview):
    cameras = {}
    for sensor in CAMERAS:
        name = sensor.split("/")[-1]
        camera = preview.get("cameras", {}).get(name, {})
        age = camera.get("last_frame_age_ms")
        fresh = (preview.get("camera_only") is True and preview.get("torque_enabled") is False
                 and preview.get("active") is True and camera.get("connected") is True
                 and isinstance(age, (int, float)) and 0 <= age <= 500)
        cameras[name] = {"fresh": fresh, "frames_received": camera.get("frames_received"),
                         "last_frame_age_ms": age}
    return cameras


def studio_env(config):
    return {**os.environ, "PYTHONPATH": str(config["studio_project"] / "src"),
            "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
            "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}


def recorded_calibration(episode, dataset):
    readbacks = episode.get("provenance", {}).get("live_calibration", {})
    recorded = dataset.get("calibration", {}).get("files", {})
    matches = {}
    for arm in ("leader", "follower"):
        value = recorded.get(arm, {}).get("sha256")
        readback = readbacks.get(arm, {})
        if value and readback.get("verified_live") is True and readback.get("snapshot_sha256") == value:
            matches[arm] = value
    return matches


def validate_story_episode(config, episode, *, technical=False):
    script = ROOT / "scripts/validate_studio_story_episode.py"
    command = [config["studio_python"], str(script), str(episode)]
    if technical:
        command.append("--allow-technical-rehearsal")
    try:
        result = subprocess.run(command, env=studio_env(config), capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired as error:
        raise ProtocolError("OFFLINE", "Saved Studio episode validation timed out. The same card can be reopened") from error
    if result.returncode == 75:
        raise ProtocolError("RESOURCE_BUSY", "Studio is still using the saved episode. Reopen this card after native work ends")
    if result.returncode:
        raise ValueError(f"Native Studio episode rejected: {result.stderr.strip()}")
    return json.loads(result.stdout)


def export_native(config, raw, *, test=False):
    studio = config["studio_project"]
    raw = raw.resolve()
    if not raw.is_relative_to(studio):
        raise ValueError("Native Studio exporter accepts only its own raw dataset")
    tag = sha(raw / "dataset.json")[:12] + "-" + uuid.uuid4().hex[:8]
    output = studio / "runtime/smart-swarm-pilot-exports" / tag
    command = [config["studio_python"], str(studio / "scripts/export_lerobot.py"),
               str(raw), "--output", str(output)]
    if test:
        command.append("--allow-test-data")
    run = subprocess.run(command, env=studio_env(config), capture_output=True, text=True, timeout=180)
    if run.returncode:
        raise ValueError(f"Native exporter rejected input: {run.stderr.strip()}")
    report = json.loads(run.stdout)
    if Path(report["path"]).resolve() != output or report["status"] != "complete":
        raise ValueError("Native exporter did not return the expected completed dataset")
    return output, report


def consume_native(config, raw, export, *, test=False):
    script = ROOT / "scripts/consume_studio_export.py"
    args = [config["studio_python"], str(script), str(raw), str(export)]
    if test:
        args.append("--allow-technical-rehearsal")
    child = subprocess.run(args, env=studio_env(config), capture_output=True, text=True, timeout=160)
    if child.returncode:
        raise ValueError(f"Native export admission failed: {child.stderr.strip()}")
    return json.loads(child.stdout)


def replay_existing(config):
    raw = config["studio_project"] / STUDIO_TEST
    before = sha(raw / "dataset.json")
    output, service = export_native(config, raw, test=True)
    consume = consume_native(config, raw, output, test=True)
    if before != sha(raw / "dataset.json"):
        raise ValueError("Studio raw manifest changed during export")
    live = native_status(config)
    result = {"native_studio_capture_runtime": "existing recorded technical source",
              "export": service, "B_consumption": consume, "source_manifest_preserved": True,
              "native_hardware_preflight": live, "physical_capture_exercised": False}
    atomic(ROOT / "evidence/resume/native-studio-rehearsal.json", result)
    return result


def native_binding(config, *, create=False):
    card = prepare(config)
    root = Path(card["card_path"]).parent
    target = root / "studio-binding.json"
    if target.is_file():
        return load(target)
    if not create:
        raise ProtocolError("OFFLINE", "Create the attended pilot dataset in SO101 Studio first")
    status = native_status(config)
    if status["recording_active"] or status["teleoperation_active"] or status["policy_active"]:
        raise ProtocolError("RESOURCE_BUSY", "SO101 Studio has an active native owner")
    name = card["title"] + " " + card["scene_group_id"]
    existing = native_request(config, "/api/status")["datasets"]
    metadata = next((item for item in existing if item["name"] == name), None)
    if metadata is None:
        metadata = native_request(config, "/api/datasets", {"name": name, "task": card["placement_instruction"],
                                    "source": "hardware", "camera_mode": "two_camera", "target_episodes": 4})
    identifier = metadata["dataset_id"]
    if re.fullmatch(DATASET_PATTERN, identifier) is None or metadata["camera_mode"] != "two_camera":
        raise ValueError("Unexpected native dataset identity or camera mode")
    binding = {"dataset_id": identifier, "scene_group_id": card["scene_group_id"],
               "task_id": card["task_id"], "split": "development", "policy_cameras": list(CAMERAS)}
    atomic(target, binding)
    return binding


def label_attempt(config, binding, episode, outcome):
    if outcome not in OUTCOMES or re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", episode) is None:
        raise ValueError("Unknown attempt ID or outcome")
    existing = native_request(config, f"/api/episodes/{binding['dataset_id']}?include_frames=false")
    if not any(row["id"] == episode and row["storage"] == "episodes" for row in existing):
        raise ValueError("Outcome labels are for completed native recordings")
    path = config["data_root"] / "so101-pilot/observed-outcomes.json"
    records = load(path) if path.is_file() else {}
    records[episode] = {"outcome": outcome, "source": "explicit_operator_terminal_observation"}
    atomic(path, records)
    return records[episode]


def sync_pilot(config, binding):
    status = native_status(config)
    dataset = binding["dataset_id"]
    if status["recording_active"] or status["teleoperation_active"] or status["policy_active"]:
        raise ProtocolError("RESOURCE_BUSY", "Finish the native session before exporting")
    root = config["data_root"] / "so101-pilot"
    records = native_request(config, f"/api/episodes/{dataset}?include_frames=false")
    source = config["studio_project"] / "data" / dataset
    export, validation, rejection = export_for_pilot(config, source, root, records)
    labels = load(root / "observed-outcomes.json") if (root / "observed-outcomes.json").is_file() else {}
    card = prepare(config)
    mapping = assign_attempts(root, records)
    attempts = [status_for(row, labels.get(row["id"]), card, mapping[row["id"]], validation)
                for row in sorted(records, key=lambda row: mapping[row["id"]])]
    tally = pilot_counts(attempts)
    report = {"binding": binding, "attempts": attempts, "counts": tally,
              "native_export": str(export) if export else None, "validation": validation,
              "validation_rejection": rejection}
    atomic(root / "native-intake.json", report)
    atomic(config["data_root"] / "so101-pilot-status.json", {"project": "smart-swarm",
           "state": "blocked" if rejection or not attempts else "complete" if tally["attempts"] >= card["attempt_limit"] else "collecting",
           "physical_preflight": "operator-managed_native_console",
           "phase": "native_capture_import" if attempts else "dataset_prepared_waiting_for_attended_preflight",
           "attempt_limit": card["attempt_limit"], "counts": tally})
    return report


def assign_attempts(root, records):
    path = root / "source-world-index.json"
    values = load(path) if path.is_file() else {}
    for entry in sorted(records, key=lambda row: row["id"]):
        if entry["id"] not in values:
            values[entry["id"]] = len(values)
    atomic(path, values)
    return values


def pending_outcome(config, binding):
    root = config["data_root"] / "so101-pilot"
    card = prepare(config)
    records = native_request(config, f"/api/episodes/{binding['dataset_id']}?include_frames=false")
    mapping = assign_attempts(root, records)
    path = root / "observed-outcomes.json"
    observed = load(path) if path.is_file() else {}
    for row in sorted(records, key=lambda row: mapping[row["id"]]):
        episode = row["id"]
        index = mapping[episode]
        if (row["storage"] == "episodes" and row["metadata"].get("status") == "complete"
                and index < card["attempt_limit"] and episode not in observed):
            return {"episode_id": episode, "attempt": index + 1}
    return None


def export_for_pilot(config, source, root, records):
    if any(row["storage"] == "episodes" for row in records):
        try:
            path, validation = accepted_export(config, source, root)
            return path, validation, None
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            return None, None, str(error)
    return None, None, None


def accepted_export(config, source, root):
    pointer = root / "accepted-native-export.json"
    fingerprint = sha(source / "dataset.json")
    if pointer.is_file():
        ref = load(pointer)
        if ref["source_manifest_sha256"] == fingerprint:
            export = Path(ref["export"])
            return export, consume_native(config, source, export)
    candidate = root / "completed-native-export.json"
    completed = load(candidate) if candidate.is_file() else {}
    export = Path(completed["export"]) if completed.get("source_manifest_sha256") == fingerprint else None
    if export is None or not export.is_dir():
        export, _ = export_native(config, source)
        completed = {"source_manifest_sha256": fingerprint, "export": str(export)}
        atomic(candidate, completed)
    validation = consume_native(config, source, export)
    atomic(pointer, completed)
    return export, validation


def status_for(record, label, card, index, validation):
    outcome = label["outcome"] if label else "unknown"
    met = record["metadata"]
    category = "outcome_unknown"
    reasons = ["terminal_observation_needed"]
    if index >= card["attempt_limit"]:
        category, reasons = "rejected", ["attempt_limit_exceeded"]
    elif record["storage"] == "quarantine" or met.get("status") != "complete":
        category, reasons = "incomplete", [met.get("reason", "native_recording_incomplete")]
    elif validation is None:
        category, reasons = "rejected", ["native_export_validation_required"]
    elif outcome == "placement_failed":
        category, reasons = "story_failure", [outcome]
    elif outcome == "placement_succeeded":
        category, reasons = "eligible_for_motor_review", []
    world = card["attempts"][index]["source_world_episode_id"] if index < card["attempt_limit"] else None
    return {"native_episode_id": record["id"], "source_world_episode_id": world, "outcome": outcome,
            "scene_group_id": card["scene_group_id"], "split": "development",
            "classification": category, "reasons": reasons, "frames": met.get("frames", 0)}


def pilot_counts(attempts):
    from .so101_pilot import count_attempts

    return count_attempts(attempts)
