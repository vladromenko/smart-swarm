"""Bind an attended Smart Swarm card to the native Studio and Explorer sources."""

from collections import Counter
import os
from pathlib import Path
import re
import subprocess
import time

from .contract import ProtocolError
from .episode import atomic, load
from .fleet_adapter import data, native_team_status
from .physical_story_condition import CHANGE_VALUES, OBSERVED_CHANGE
from .physical_story_plan import ROOT, prepare
from .studio import native_request, studio_env

TITLE = "Smart Swarm / Distributed visual stories v1"
TASK = "Use the attended story card for the SO101 placement attempt and withdraw."
PHASES = ("placement_closed", "relocation_closed")
ACTIVE_STORY_STATES = ("preparing", "recording", "pending_reveal")
FLEET_TERMINAL = ("complete", "failed", "interrupted")


class StudioEpisodeUnlinked(ValueError):
    """The native source is absent or ambiguous within the card's interval."""


def dataset(config, workspace):
    binding = workspace / "studio-story-dataset.json"
    if binding.is_file():
        return load(binding)["dataset_id"]
    metadata = native_request(config, "/api/status")["datasets"]
    existing = next((row for row in metadata if row["name"] == TITLE), None)
    payload = {"name": TITLE, "task": TASK, "source": "hardware", "camera_mode": "two_camera", "target_episodes": 24}
    active = existing or native_request(config, "/api/datasets", payload)
    if active["camera_mode"] != "two_camera" or active["source"] != "hardware":
        raise ValueError("Studio source does not match the two-camera attended dataset")
    identifier = active["dataset_id"]
    atomic(binding, {"dataset_id": identifier, "camera_order": ["wrist", "top"], "operator_title": TITLE})
    return identifier


def story_workspace(config):
    root = config["data_root"] / "physical-stories"
    root.mkdir(exist_ok=True, parents=True)
    return root


def story_progress(config):
    root = story_workspace(config)
    plan = prepare(config)
    attempts, reasons = [], Counter()
    for row in plan["cards"]:
        trial = root / row["story_id"]
        journal = trial / "trial.json"
        if journal.is_file():
            state = load(journal)
            admission = trial / "motor-intake.json"
            motor = load(admission) if admission.is_file() else {"status": "not_reviewed"}
            timing = trial / "fleet-timing-audit.json"
            if motor.get("reason"):
                reasons[motor["reason"]] += 1
            learned = state.get("decision", {}).get("learned_evidence", {})
            attempts.append({"story_id": row["story_id"], "family": row["family"], "status": state["status"],
                             "source_group_id": row["source_group_id"], "split": row["split"],
                             "reason": state.get("reason"), "motor_intake": motor,
                             "learned_evidence_status": learned.get("status", "no_learned_forecast"),
                             "native_timing": load(timing) if timing.is_file() else {},
                             "observed_terminal_outcome": state.get("observed_terminal_outcome")})
    motors = Counter(item["motor_intake"]["status"] for item in attempts)
    motor_groups = {item["source_group_id"] for item in attempts if item["motor_intake"]["status"] == "reviewable_prefix"}
    return {"scope": "physical collection and motor prefix review", "planned": plan["planned_stories"],
            "attempted": len(attempts), "stories_by_status": dict(Counter(item["status"] for item in attempts)),
            "learned_evidence_by_status": dict(Counter(item["learned_evidence_status"] for item in attempts)),
            "motor_by_status": dict(motors), "reviewable_motor_source_groups": len(motor_groups),
            "motor_exclusion_reasons": dict(reasons), "attempts": attempts}


def card_for(config, identifier):
    plan = prepare(config)
    matches = [row for row in plan["cards"] if row["story_id"] == identifier]
    if not matches:
        raise ValueError("Use one of the prepared physical story IDs")
    return matches[0]


def anchor():
    return {"monotonic_ns": time.monotonic_ns(), "unix_ns": time.time_ns()}


def homebrain_boot():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def same_recording_boot(state):
    recorded = state.get("homebrain_boot_id")
    return recorded is None or recorded == homebrain_boot()


def save_incomplete(trial, state, reason):
    state.update(status="incomplete", reason=reason, observed_terminal_outcome="unknown")
    atomic(trial / "trial.json", state)
    return state


def fleet_manifest(config, identifier):
    if not re.fullmatch(r"[a-f0-9]{32}", identifier):
        raise ValueError("Invalid native fleet capture ID")
    manifest = data(config, f"/api/fleet/captures/{identifier}/manifest.json")
    if manifest["phase"] != "recording":
        return manifest
    status = data(config, "/api/fleet/captures")
    return next((row for row in status["records"] if row["id"] == identifier and row["phase"] in FLEET_TERMINAL),
                manifest)


def worker_identity(pid):
    try:
        info = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return None
    return {"pid": pid, "started_ticks": info[19]} if info[0] != "Z" else None


def restore_start(config, trial, state):
    if state["status"] != "preparing":
        return state
    receipt = trial / "fleet-start.json"
    if receipt.is_file():
        identifier = load(receipt)["id"]
        if not re.fullmatch(r"[a-f0-9]{32}", identifier):
            raise ValueError("Invalid ID in the native Explorer start receipt")
        state.update(status="recording", fleet_capture_id=identifier)
        state.pop("reason", None)
        atomic(trial / "trial.json", state)
        return state
    launch = trial / "fleet-launch.json"
    holder = load(launch) if launch.is_file() else None
    if holder and holder == worker_identity(holder["pid"]):
        return state
    if time.time_ns() - state["start"]["unix_ns"] < 15_000_000_000:
        return state
    if data(config, "/api/fleet/captures")["active"]:
        return state
    return save_incomplete(trial, state, state.get("reason") or "Native Explorer capture never acknowledged start")


def launch_fleet(config, directory):
    receipt = directory / "fleet-start.json"
    interpreter = config["explorer_python"]
    checkout = str(Path(config["explorer_project"]).resolve())
    env = {**os.environ, "PYTHONPATH": checkout, "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""}
    command = [interpreter, str(ROOT / "scripts/native_fleet_story_capture.py"), str(receipt)]
    with (directory / "fleet-capture.log").open("w") as log:
        worker = subprocess.Popen(command, cwd=checkout, env=env, stdout=log, stderr=subprocess.STDOUT,
                                  start_new_session=True)
    identity = worker_identity(worker.pid)
    if identity is not None:
        atomic(directory / "fleet-launch.json", identity)
    deadline = time.monotonic() + 12
    while not receipt.is_file() and worker.poll() is None and time.monotonic() < deadline:
        time.sleep(.1)
    if receipt.is_file():
        return load(receipt)["id"]
    if worker.poll() is not None:
        raise RuntimeError(f"Explorer native capture did not acknowledge start. See {directory / 'fleet-capture.log'}")
    return None


def story_readiness(config):
    from .research_jobs import native_idle

    team = native_team_status(config, probe_rosmaster=False)
    studio = team["so101_studio"]
    rgb = team["story_observation_readiness"]
    missing = [name for name, present in studio["inventory_present"].items() if not present]
    issues = []
    if studio["native_stop_check_required"]:
        issues.append(("OFFLINE", "Studio requires its hardware STOP check before another session"))
    if studio["native_arm_hardware_open"]:
        issues.append(("RESOURCE_BUSY", "The native Studio arm devices are still open"))
    if team["active_native_capture"]:
        issues.append(("RESOURCE_BUSY", "Explorer already has an active native recording"))
    try:
        native_idle(studio)
    except ProtocolError as error:
        issues.append((error.code, error.message))
    if missing:
        issues.append(("OFFLINE", f"Studio did not detect {', '.join(missing)}"))
    issues.extend(("OFFLINE", issue) for issue in rgb["issues"])
    return {"scope": "read-only distributed story readiness", "ready_for_native_lease_check": not issues,
            "blocking_issues": [{"code": code, "reason": reason} for code, reason in issues],
            "so101_studio": studio, "mobile_rgb_and_clocks": rgb,
            "active_native_capture": team["active_native_capture"],
            "pending_attended_checks": ["Scene and camera framing", "Native control and STOP verification"],
            "motion_requested": False, "recording_started": False}


def story_preflight(config):
    readiness = story_readiness(config)
    blockers = readiness["blocking_issues"]
    if blockers:
        raise ProtocolError(blockers[0]["code"], ". ".join(item["reason"] for item in blockers))
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


def begin(config, identifier):
    card = card_for(config, identifier)
    story_preflight(config)
    root = story_workspace(config)
    open_trials = [load(path) for path in root.glob("*/trial.json")]
    if any(row["status"] in ACTIVE_STORY_STATES for row in open_trials):
        raise ProtocolError("RESOURCE_BUSY", "Complete the active physical story before opening another card")
    native_id = dataset(config, root)
    before = episode_ids(config, native_id)
    trial = root / identifier
    trial.mkdir(exist_ok=False)
    record = {"card": card, "status": "preparing", "studio_dataset_id": native_id,
              "start": anchor(), "homebrain_boot_id": homebrain_boot(),
              "preexisting_native_episode_ids": sorted(before), "markers": {}}
    atomic(trial / "trial.json", record)
    try:
        capture = launch_fleet(config, trial)
        if capture is not None:
            record.update(fleet_capture_id=capture, status="recording")
    except (OSError, RuntimeError) as error:
        record["reason"] = str(error)
        raise
    finally:
        atomic(trial / "trial.json", record)
    return record


def episode_ids(config, identifier):
    rows = native_request(config, f"/api/episodes/{identifier}?include_frames=false")
    return {row["id"]: row for row in rows}


def story_phases(state):
    return ("insertion_closed", *PHASES) if state["card"]["family"] == "F3" else PHASES


def mark(config, identifier, phase, *, assistance=None, placement_outcome=None, observed_change=None):
    card_for(config, identifier)
    journal = story_workspace(config) / identifier / "trial.json"
    state = load(journal)
    completed = state["markers"]
    phases = story_phases(state)
    if not same_recording_boot(state):
        raise ValueError("Homebrain rebooted before this phase could be recorded")
    if state["status"] != "recording" or len(completed) >= len(phases) or phase != phases[len(completed)]:
        raise ValueError("Story markers must follow the card's placement and relocation order")
    if phase == "relocation_closed" and assistance not in ("human_relocation", "attended_native_motion", "none", "unknown"):
        raise ValueError("Record who relocated the carrier")
    if phase != "relocation_closed" and assistance not in (None, "human_hand_assistance", "attended_native_motion", "none"):
        raise ValueError("Record whether hands assisted the native arm phase")
    if placement_outcome not in (None, "placement_succeeded", "placement_failed", "unknown"):
        raise ValueError("Use a directly observed placement outcome or unknown")
    motion_check = phase == "relocation_closed" or (phase == "placement_closed" and state["card"]["family"] == "F3")
    if observed_change is not None and (not motion_check or observed_change not in CHANGE_VALUES):
        raise ValueError("Use a direct change observation only at the removal or carrier transition")
    if observed_change == "confirmed" and assistance not in ("human_relocation", "human_hand_assistance", "attended_native_motion"):
        raise ValueError("Confirm who moved the carrier or cube before attesting the transition")
    completed[phase] = {**anchor(), "assistance": assistance, "observed_placement_outcome": placement_outcome}
    if observed_change is not None:
        completed[phase][OBSERVED_CHANGE] = observed_change
    atomic(journal, state)
    return state


def corresponding_native_episode(config, state, *, boundary="placement_closed"):
    available = episode_ids(config, state["studio_dataset_id"])
    prior = set(state["preexisting_native_episode_ids"])
    lower = state["start"]["monotonic_ns"]
    upper = state["markers"][boundary]["monotonic_ns"]
    candidates = [row for key, row in available.items() if key not in prior and row["storage"] == "episodes"]
    matching = [row for row in candidates if lower <= row["metadata"]["first_timestamp_ns"] < upper
                <= row["metadata"]["last_timestamp_ns"] + 2_000_000_000]
    if len(matching) != 1:
        raise StudioEpisodeUnlinked("Exactly one newly saved native Studio episode must cover the declared placement window")
    return matching[0]
