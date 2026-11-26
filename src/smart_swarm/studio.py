"""Minimal Smart Swarm adapter to native SO101 Studio capture and export."""

from hashlib import sha256
from fcntl import LOCK_EX, LOCK_NB, flock
import json
from math import ceil
import os
from pathlib import Path
import re
import subprocess
from urllib.request import ProxyHandler, Request, build_opener
import uuid

from .contract import ProtocolError, json_bytes
from .episode import atomic, load, sha
from .platform_client import NoRedirect
from .so101_pilot import CAMERAS, ROOT, prepare

STUDIO_TEST = "runtime/acceptance-final-verified/test_official_lerobot_cpu_load0/raw_moved"
DATASET_PATTERN = r"ds_[a-f0-9]{16}"
OUTCOMES = ("placement_succeeded", "placement_failed", "unknown")
ASSISTANCE = ("attended_native_motion", "manual_scene_assistance", "unknown")
STOP_UNCONFIRMED = "torque_disable_unconfirmed_operator_check_required"
EXPORT_MIN_SECONDS = 180
EXPORT_MAX_SECONDS = 1800
EXPORT_START_SECONDS = 60
EXPORT_PLANNING_IMAGES_PER_SECOND = 8
EXPORT_MISMATCH_EXIT = 3


class NativeExportMismatch(ValueError):
    """The official export could not be matched to the saved native recording."""


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
            "native_following_readback": following_readback(teleop, readback),
            "native_device_identities": {key: inventory.get(key, {}).get("resolved_path") for key in devices},
            "native_camera_preview": preview_status(report["preview"]),
            "native_resources": resource_readback(report["resources"]),
            "physical_preflight_verified": False}


def native_controls_idle(status):
    if any(status[key] for key in ("recording_active", "teleoperation_active", "policy_active")):
        raise ProtocolError("RESOURCE_BUSY", "A native recording, motion or policy job is active")


def following_readback(teleop, calibration):
    keys = ("state", "phase", "last_sample_ns", "samples", "error", "stop_errors")
    return {**{key: teleop.get(key) for key in keys}, "source": (teleop.get("provenance") or {}).get("source"),
            "calibration_sha256": {arm: calibration.get(arm, {}).get("snapshot_sha256")
                                   for arm in ("leader", "follower")}}


def resource_readback(resources):
    inventory = resources["device_inventory"]
    order = [(sensor, sensor.split("/")[-1]) for sensor in CAMERAS]
    cameras = [{"sensor_id": sensor, **inventory.get(key, {})} for sensor, key in order]
    return {"ordered_cameras": cameras, "hardware_owners": resources.get("hardware_owners", []),
            "disk_free_bytes": resources.get("disk_free_bytes"),
            "compute_conflicts": resources.get("compute_conflicts", []),
            "snapshot_monotonic_ns": resources.get("timestamp_ns")}


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


def require_recording_lease(config):
    code = ("from so101_lab.resources import ResourceBusy, ResourceLease\n"
            "try:\n    with ResourceLease('recording'):\n        pass\n"
            "except ResourceBusy:\n    raise SystemExit(75)")
    try:
        result = subprocess.run([config["studio_python"], "-c", code], env=studio_env(config),
                                capture_output=True, text=True, timeout=5)
    except subprocess.TimeoutExpired as error:
        raise ProtocolError("OFFLINE", "The native Studio recorder reservation did not answer") from error
    if result.returncode == 75:
        raise ProtocolError("RESOURCE_BUSY", "The native Studio recording reservation is held")
    if result.returncode:
        raise ProtocolError("OFFLINE", f"The native Studio recorder reservation check failed. {result.stderr.strip()}")


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


def native_export_timeout(raw):
    _, episodes = native_export_snapshot(raw)
    metadata = load(raw / "dataset.json")
    mode = metadata.get("camera_mode", metadata.get("settings", {}).get("camera_mode", "two_camera"))
    cameras = 1 if mode == "wrist_only" else len(CAMERAS)
    samples = cameras * sum(episode["frames"] for episode in episodes)
    allowance = EXPORT_START_SECONDS + ceil(samples / EXPORT_PLANNING_IMAGES_PER_SECOND)
    return min(EXPORT_MAX_SECONDS, max(EXPORT_MIN_SECONDS, allowance))


def completed_native_output(output, raw):
    path = output / "provenance/export.json"
    if not path.is_file():
        return None
    result = load(path)
    verification = result.get("official_verification", {})
    if (result.get("status") != "complete" or result.get("source_path") != str(raw)
            or result.get("source_hashes", {}).get("dataset.json") != sha(raw / "dataset.json")
            or verification.get("rgb_pixels_bit_exact") is not True):
        return None
    return {**result, "path": str(output), "recovered_after_native_timeout": True}


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
    timeout = native_export_timeout(raw)
    try:
        run = subprocess.run(command, env=studio_env(config), capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        result = completed_native_output(output, raw)
        if result is None:
            raise
        return output, result
    if run.returncode:
        detail = run.stderr.strip()
        summary = detail.rsplit("\n", 1)[-1]
        native_busy = "so101_lab.resources.ResourceBusy: "
        if summary.startswith(native_busy):
            raise ProtocolError("RESOURCE_BUSY", summary.removeprefix(native_busy), retryable=True)
        raise ValueError(f"Native exporter rejected input: {detail}")
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
    if child.returncode == EXPORT_MISMATCH_EXIT:
        raise NativeExportMismatch(f"Native export admission failed: {child.stderr.strip()}")
    if child.returncode:
        raise ProtocolError("OFFLINE", f"The native export consumer did not run successfully. {child.stderr.strip()}", retryable=True)
    try:
        receipt = json.loads(child.stdout)
    except ValueError as error:
        raise ProtocolError("OFFLINE", "The native export consumer returned an unreadable receipt", retryable=True) from error
    if not isinstance(receipt, dict) or not isinstance(receipt.get("episodes"), list):
        raise ProtocolError("OFFLINE", "The native export consumer returned an incomplete receipt", retryable=True)
    return receipt


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


def label_attempt(config, binding, observed_episode, outcome, *, assistance):
    dataset = binding["dataset_id"]
    episode = observed_episode["recording"]["episode_id"]
    if (outcome not in OUTCOMES or assistance not in ASSISTANCE or re.fullmatch(DATASET_PATTERN, dataset) is None
            or re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", episode) is None):
        raise ValueError("Unknown pilot dataset, attempt ID, outcome or assistance")
    path = config["data_root"] / "so101-pilot/observed-outcomes.json"
    with path.with_suffix(".lock").open("a") as lock:
        try:
            flock(lock, LOCK_EX | LOCK_NB)
        except BlockingIOError as error:
            raise ProtocolError("RESOURCE_BUSY", "Another Smart Swarm window is saving an outcome") from error
        pending = pending_outcome(config, binding)
        if pending != observed_episode:
            raise ValueError("The latest completed pilot attempt changed or already has an outcome")
        records = load(path) if path.is_file() else {}
        records[episode] = {"outcome": outcome, "assistance": assistance, "recording": pending["recording"],
                            "source": "explicit_operator_terminal_observation"}
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
    retryable = isinstance(rejection, ProtocolError) and rejection.retryable
    labels = load(root / "observed-outcomes.json") if (root / "observed-outcomes.json").is_file() else {}
    card = prepare(config)
    mapping = assign_attempts(root, records)
    attempts = [status_for(row, labels.get(row["id"]), card, mapping[row["id"]], validation, retry_pending=retryable)
                for row in sorted(records, key=lambda row: mapping[row["id"]])]
    tally = pilot_counts(attempts)
    report = {"binding": binding, "attempts": attempts, "counts": tally,
              "native_export": str(export) if export else None, "validation": validation,
              "validation_rejection": str(rejection) if rejection else None, "validation_retryable": retryable}
    atomic(root / "native-intake.json", report)
    atomic(config["data_root"] / "so101-pilot-status.json", {"project": "smart-swarm",
           "state": "blocked" if rejection or not attempts else "complete" if tally["attempts"] >= card["attempt_limit"] else "collecting",
           "physical_preflight": "operator-managed_native_console",
           "phase": "native_capture_import" if attempts else "dataset_prepared_waiting_for_attended_preflight",
           "attempt_limit": card["attempt_limit"], "counts": tally})
    return report


def attempt_time(row):
    return row.get("metadata", {}).get("finished_at") or "", row["id"]


def assign_attempts(root, records):
    path = root / "source-world-index.json"
    with path.with_suffix(".lock").open("a") as lock:
        flock(lock, LOCK_EX)
        existed = path.is_file()
        values = load(path) if existed else {}
        count = len(values)
        for entry in sorted(records, key=attempt_time):
            values.setdefault(entry["id"], len(values))
        if not existed or len(values) != count:
            atomic(path, values)
    return values


def pending_outcome(config, binding):
    root = config["data_root"] / "so101-pilot"
    card = prepare(config)
    dataset = binding["dataset_id"]
    if re.fullmatch(DATASET_PATTERN, dataset) is None:
        raise ValueError("Unknown pilot dataset identity")
    records = native_request(config, f"/api/episodes/{dataset}?include_frames=false")
    if not records:
        return None
    mapping = assign_attempts(root, records)
    path = root / "observed-outcomes.json"
    observed = load(path) if path.is_file() else {}
    row = max(records, key=attempt_time)
    episode, metadata = row["id"], row["metadata"]
    index = mapping[episode]
    if (row["storage"] != "episodes" or metadata.get("status") != "complete" or episode in observed
            or index >= card["attempt_limit"] or re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", episode) is None
            or re.fullmatch(r"[a-f0-9]{64}", metadata.get("frames_sha256") or "") is None):
        return None
    manifest = config["studio_project"] / "data" / dataset / "episodes" / episode / "episode.json"
    identity = native_episode_snapshot(manifest, expected_metadata=metadata)
    require_saved_native_scene(config, dataset, episode)
    return {"attempt": index + 1, "recording": identity}


def require_saved_native_scene(config, dataset, episode):
    recorder = native_request(config, "/api/status")["recording"]
    if recorder.get("active") is False:
        return
    if (recorder.get("state") != "ready" or recorder.get("dataset_id") != dataset
            or (recorder.get("last_episode") or {}).get("episode_id") != episode):
        raise ProtocolError("RESOURCE_BUSY", "Studio must finish saving before the terminal scene can be labelled")


def export_for_pilot(config, source, root, records):
    if any(row["storage"] == "episodes" for row in records):
        try:
            path, validation = accepted_export(config, source, root)
            return path, validation, None
        except subprocess.TimeoutExpired as error:
            return None, None, ProtocolError("OFFLINE", f"Native export or validation exceeded {error.timeout} seconds",
                                             retryable=True)
        except (ProtocolError, ValueError, OSError, subprocess.SubprocessError) as error:
            return None, None, error
    return None, None, None


def native_episode_snapshot(manifest, *, expected_metadata=None):
    metadata_bytes = manifest.read_bytes()
    metadata = json.loads(metadata_bytes)
    if expected_metadata is not None and metadata != expected_metadata:
        raise ValueError(f"Native episode metadata changed during the catalogue read: {manifest.parent.name}")
    ledger = (manifest.parent / "frames.jsonl").read_bytes()
    frames = sum(bool(line.strip()) for line in ledger.splitlines())
    digest = sha256(ledger).hexdigest()
    episode = manifest.parent.name
    if metadata.get("status") != "complete" or frames == 0:
        raise ValueError(f"Native episode is not complete: {episode}")
    if metadata.get("frames") is not None and int(metadata["frames"]) != frames:
        raise ValueError(f"Native episode frame count changed: {episode}")
    if metadata.get("frames_sha256") and metadata["frames_sha256"] != digest:
        raise ValueError(f"Native episode frame ledger changed: {episode}")
    return {"episode_id": episode, "frames": frames, "frames_sha256": digest,
            "episode_manifest_sha256": sha256(metadata_bytes).hexdigest()}


def native_export_snapshot(source):
    if any((source / ".partial").glob("*.partial")):
        raise ValueError("An unfinished native recording needs Stop or recovery before export")
    catalogue = sha(source / "dataset.json")
    episodes = [native_episode_snapshot(path) for path in sorted((source / "episodes").glob("*/episode.json"))]
    if not episodes:
        raise ValueError("No complete native episodes are available for export")
    return {"source_manifest_sha256": catalogue,
            "source_episode_set_sha256": sha256(json_bytes(episodes)).hexdigest()}, episodes


def cached_native_exports(paths, generation):
    legacy, current, seen = [], [], set()
    for path in paths:
        ref = load(path) if path.is_file() else {}
        if ref.get("source_manifest_sha256") == generation["source_manifest_sha256"]:
            export = Path(ref["export"])
            if export.is_dir() and export not in seen:
                membership = ref.get("source_episode_set_sha256")
                if membership == generation["source_episode_set_sha256"]:
                    current.append((export, False))
                    seen.add(export)
                elif membership is None:
                    legacy.append((export, True))
    return current + [item for item in legacy if item[0] not in seen]


def consume_matching_export(config, source, export, episodes, **options):
    validation = consume_native(config, source, export, **options)
    expected = {item["episode_id"]: item for item in episodes}
    observed = validation["episodes"]
    ids = {item["episode_id"] for item in observed}
    if ids != expected.keys() or len(observed) != len(expected):
        raise NativeExportMismatch("Native export does not contain every finalized episode exactly once")
    for item in observed:
        if any(item.get(key) != value for key, value in expected[item["episode_id"]].items()):
            raise NativeExportMismatch(f"Native export differs from the saved episode: {item['episode_id']}")
    return validation


def ensure_native_generation(source, generation, phase):
    if native_export_snapshot(source)[0] != generation:
        raise ProtocolError("RESOURCE_BUSY", f"Native recordings changed during {phase}", retryable=True)


def rejection_path(root, generation, export):
    identity = sha256(json_bytes([generation, str(export)])).hexdigest()
    return root / "rejected-native-exports" / f"{identity}.json"


def storage_rejection_path(root, export):
    identity = sha256(json_bytes(str(export))).hexdigest()
    return root / "rejected-native-exports" / f"storage-{identity}.json"


def rejected_native_exports(root):
    entries = [load(path) for path in sorted((root / "rejected-native-exports").glob("*.json"))]
    return {row["export"]: row["reason"] for row in entries if row["state"] in ("validation_rejected", "storage_rejected")}


def reject_export_storage(root, export, expected, actual, *, material="data_and_meta"):
    path = storage_rejection_path(root, export)
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic(path, {"export": str(export), "state": "storage_rejected", "material": material,
                      "expected_sha256": expected, "observed_sha256": actual,
                      "reason": "Native export data or metadata bytes changed"})


def reject_export(root, source, generation, export, error):
    ensure_native_generation(source, generation, "export validation")
    path = rejection_path(root, generation, export)
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic(path, {**generation, "export": str(export), "state": "validation_rejected", "reason": str(error)})


def reuse_native_export(root, source, generation, paths, inspect):
    for candidate, legacy in cached_native_exports(paths, generation):
        if not (rejection_path(root, generation, candidate).is_file() or storage_rejection_path(root, candidate).is_file()):
            try:
                return candidate, inspect(candidate)
            except ValueError as error:
                if not legacy and not isinstance(error, NativeExportMismatch):
                    raise
                reject_export(root, source, generation, candidate, error)
    return None, None


def accepted_export(config, source, root, *, test=False):
    from .studio_export_storage import archive_accepted_export

    accepted = root / "accepted-native-export.json"
    completed = root / "completed-native-export.json"
    generation, episodes = native_export_snapshot(source)
    options = {"test": True} if test else {}

    def inspect(target):
        return consume_matching_export(config, source, target, episodes, **options)

    export, validation = reuse_native_export(root, source, generation, (accepted, completed), inspect)
    if validation is None:
        export, _ = export_native(config, source, **options)
        ensure_native_generation(source, generation, "export")
        atomic(completed, {**generation, "export": str(export)})
        try:
            validation = inspect(export)
        except NativeExportMismatch as error:
            reject_export(root, source, generation, export, error)
            raise
    ensure_native_generation(source, generation, "validation")
    pointer = {**generation, "export": str(export)}
    atomic(completed, pointer)
    atomic(accepted, pointer)
    try:
        storage = archive_accepted_export(config, source, export, root, validation, lambda: inspect(export))
    except (OSError, ValueError) as error:
        storage = {"state": "pending", "reason": str(error)}
    if storage is not None:
        validation = {**validation, "archive_storage": storage}
    return export, validation


def verified_pilot_episode(record, validation):
    if validation.get("source") != "hardware" or validation.get("technical_only") is not False:
        return False
    metadata = record["metadata"]
    digest = metadata.get("frames_sha256")
    return any(item.get("episode_id") == record["id"] and item.get("frames") == metadata.get("frames")
               and (not digest or item.get("frames_sha256") == digest)
               for item in validation.get("episodes", []))


def pilot_outcome_category(record, label, validation):
    if label is None:
        return "unknown", "outcome_unknown", ["terminal_observation_needed"]
    outcome = label["outcome"]
    recording = label.get("recording")
    fields = "episode_id", "frames", "frames_sha256", "episode_manifest_sha256"
    if recording is not None and not any(recording == {key: row.get(key) for key in fields}
                                         for row in validation["episodes"] if row["episode_id"] == record["id"]):
        return "unknown", "outcome_unknown", ["terminal_observation_recording_changed"]
    if outcome == "placement_failed":
        return outcome, "story_failure", [outcome]
    if outcome != "placement_succeeded":
        return "unknown", "outcome_unknown", ["terminal_observation_needed"]
    if not recording or label.get("assistance") != "attended_native_motion":
        reason = "outcome_not_bound_to_native_recording" if not recording else "native_only_placement_not_confirmed"
        return outcome, "motor_review_excluded", [reason]
    if len(recorded_calibration(record["metadata"], validation)) != 2:
        return outcome, "motor_review_excluded", ["native_calibration_readbacks_unverified"]
    return outcome, "eligible_for_motor_review", []


def status_for(record, label, card, index, validation, *, retry_pending=False):
    outcome = label["outcome"] if label else "unknown"
    met = record["metadata"]
    if index >= card["attempt_limit"]:
        category, reasons = "rejected", ["attempt_limit_exceeded"]
    elif record["storage"] == "quarantine" or met.get("status") != "complete":
        outcome, category, reasons = "unknown", "incomplete", [met.get("reason", "native_recording_incomplete")]
    elif retry_pending:
        category, reasons = "pending_validation", ["native_export_validation_deferred"]
    elif validation is None:
        category, reasons = "rejected", ["native_export_validation_required"]
    elif not verified_pilot_episode(record, validation):
        category, reasons = "rejected", ["native_export_episode_not_verified"]
    else:
        outcome, category, reasons = pilot_outcome_category(record, label, validation)
    world = card["attempts"][index]["source_world_episode_id"] if index < card["attempt_limit"] else None
    return {"native_episode_id": record["id"], "source_world_episode_id": world, "outcome": outcome,
            "observed_outcome_receipt": label,
            "scene_group_id": card["scene_group_id"], "split": "development",
            "classification": category, "reasons": reasons, "frames": met.get("frames", 0)}


def pilot_counts(attempts):
    from .so101_pilot import count_attempts

    return {**count_attempts(attempts), "motor_review_excluded": sum(row["classification"] == "motor_review_excluded" for row in attempts),
            "pending_validation": sum(row["classification"] == "pending_validation" for row in attempts)}
