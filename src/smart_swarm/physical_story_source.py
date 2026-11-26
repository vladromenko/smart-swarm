"""Bind an attended Smart Swarm card to the native Studio and Explorer sources."""

from collections import Counter
from fcntl import LOCK_EX, LOCK_NB, flock
from functools import wraps
import json
import os
from pathlib import Path
import re
from shutil import disk_usage
import subprocess
import time

from .attended_so101 import attended_status
from .contract import ProtocolError
from .episode import atomic, load
from .fleet_adapter import data, native_team_status
from .native_fleet_start import START_GATE
from .physical_story_condition import CHANGE_VALUES, OBSERVED_CHANGE
from .physical_story_plan import PLAN_VERSION, ROOT, prepare
from .studio import native_controls_idle, native_request, require_recording_lease

TITLE = "Smart Swarm / Distributed visual stories v1"
TASK = "Use the attended story card for the SO101 placement attempt and withdraw."
PHASES = ("placement_closed", "relocation_closed")
ACTIVE_STORY_STATES = ("placement_recording", "preparing", "recording", "pending_reveal")
FLEET_TERMINAL = ("complete", "failed", "interrupted")
FLEET_WINDOW_S = 60
STATIONARY_LEAD_IN_S = 12


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


def story_update(action):
    @wraps(action)
    def update(config, *args, **kwargs):
        path = story_workspace(config) / ".journal.lock"
        with path.open("a") as lock:
            try:
                flock(lock, LOCK_EX | LOCK_NB)
            except BlockingIOError as error:
                raise ProtocolError("RESOURCE_BUSY", "Another Smart Swarm action is updating the story journal") from error
            return action(config, *args, **kwargs)

    return update


def learned_story_progress(trial, state):
    from .physical_learned import DEFERRED
    from .physical_offline import read_replay

    learned = state.get("decision", {}).get("learned_evidence", {})
    result = {"learned_evidence_status": learned.get("status", "no_learned_forecast")}
    if learned.get("status") == "scored":
        result["learned_forecast_mode"] = "online"
    if learned.get("status") == DEFERRED and state["status"] == "complete":
        try:
            replay = read_replay(trial / "private-replay")
            if replay is not None:
                result.update(learned_evidence_status="scored", learned_forecast_mode="blind_offline")
        except (OSError, ValueError, TypeError, KeyError, ProtocolError) as error:
            result.update(learned_evidence_status="invalid_blind_offline_receipt", learned_forecast_reason=str(error))
    failure = learned.get("online_attempt")
    if isinstance(failure, dict) and isinstance(failure.get("reason"), str):
        result["online_admission_failure"] = failure["reason"]
    return result


def story_progress(config):
    from .physical_story_motor import rejected_prefix

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
            refusal = rejected_prefix(trial, state, motor) if motor["status"] == "reviewable_prefix" else None
            if refusal:
                motor = {"status": "pending_review", "reason": refusal}
            timing = trial / "fleet-timing-audit.json"
            storage = trial / "archive-storage.json"
            if motor.get("reason"):
                reasons[motor["reason"]] += 1
            attempts.append({"story_id": row["story_id"], "family": row["family"], "status": state["status"],
                             "source_group_id": row["source_group_id"], "split": row["split"],
                             "reason": state.get("reason"), "motor_intake": motor,
                             **learned_story_progress(trial, state),
                             "native_timing": load(timing) if timing.is_file() else {},
                             "native_archive_storage": load(storage) if storage.is_file() else None,
                             "observed_terminal_outcome": state.get("observed_terminal_outcome")})
    motors = Counter(item["motor_intake"]["status"] for item in attempts)
    motor_groups = {item["source_group_id"] for item in attempts if item["motor_intake"]["status"] == "reviewable_prefix"}
    return {"scope": "physical collection and motor prefix review", "planned": plan["planned_stories"],
            "attempted": len(attempts), "stories_by_status": dict(Counter(item["status"] for item in attempts)),
            "learned_evidence_by_status": dict(Counter(item["learned_evidence_status"] for item in attempts)),
            "motor_by_status": dict(motors), "reviewable_motor_source_groups": len(motor_groups),
            "motor_exclusion_reasons": dict(reasons), "attempts": attempts,
            "block_reviews": [str(path) for path in sorted((root / "block-reviews").glob("*.json"))]}


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
        return accept_fleet_start(trial, state, load(receipt))
    launch = trial / "fleet-launch.json"
    holder = load(launch) if launch.is_file() else None
    if holder and holder == worker_identity(holder["pid"]):
        return state
    requested = state.get("fleet_launch_requested", state["start"])
    if time.time_ns() - requested["unix_ns"] < 15_000_000_000:
        return state
    if data(config, "/api/fleet/captures")["active"]:
        return state
    reason = state.get("reason") or "Native Explorer capture never acknowledged start"
    if "placement_closed" in state.get("markers", {}):
        from .physical_story_replay import preserve_finished_prefix

        preserve_finished_prefix(config, trial, state, reason)
        return load(trial / "trial.json")
    return save_incomplete(trial, state, reason)


def accept_fleet_start(trial, state, receipt):
    identifier = receipt["id"]
    if not re.fullmatch(r"[a-f0-9]{32}", identifier):
        raise ValueError("Invalid ID in the native Explorer start receipt")
    acknowledged = receipt.get("acknowledged")
    if "fleet_launch_requested" in state:
        launch = state["fleet_launch_requested"]["monotonic_ns"]
        if (not acknowledged or acknowledged["monotonic_ns"] < launch
                or receipt.get("requested_seconds") != FLEET_WINDOW_S):
            raise ValueError("The staged Explorer launch lacks a matching native acknowledgement")
    if acknowledged is not None:
        state["fleet_acknowledgement"] = acknowledged
        state["fleet_window_s"] = receipt["requested_seconds"]
    gate = receipt.get("rgb_start_gate")
    if gate is not None:
        if gate != START_GATE or acknowledged is None:
            raise ValueError("The native Explorer acknowledgement has an invalid recorded-RGB gate")
        state["fleet_rgb_start_gate"] = gate
    state.update(status="recording", fleet_capture_id=identifier)
    state.pop("reason", None)
    atomic(trial / "trial.json", state)
    return state


def fleet_invocation(config, *options):
    interpreter = config["explorer_python"]
    checkout = str(Path(config["explorer_project"]).resolve())
    python_path = os.pathsep.join((checkout, str(ROOT / "src")))
    env = {**os.environ, "PYTHONPATH": python_path, "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""}
    return [interpreter, str(ROOT / "scripts/native_fleet_story_capture.py"), *options], {"cwd": checkout, "env": env}


def launch_fleet(config, directory):
    receipt = directory / "fleet-start.json"
    command, environment = fleet_invocation(config, str(receipt), "--seconds", str(FLEET_WINDOW_S))
    with (directory / "fleet-capture.log").open("w") as log:
        worker = subprocess.Popen(command, **environment, stdout=log, stderr=subprocess.STDOUT,
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
    team = native_team_status(config, probe_rosmaster=False)
    studio = team["so101_studio"]
    attended = attended_status(config, studio)
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
        native_controls_idle(studio)
    except ProtocolError as error:
        issues.append((error.code, error.message))
    if missing:
        issues.append(("OFFLINE", f"Studio did not detect {', '.join(missing)}"))
    preview = studio.get("native_camera_preview", {})
    issues.extend(("OFFLINE", f"Studio {key} has no fresh camera-only preview")
                  for key, row in preview.items() if not row["fresh"])
    issues.extend(("OFFLINE", issue) for issue in rgb["issues"])
    installation = recording_installation(config)
    storage = fleet_storage_blocker(installation["native_capture_storage"])
    if storage:
        issues.append(storage)
    return {"scope": "read-only distributed story readiness", "ready_for_native_lease_check": not issues,
            "status": "SOFTWARE_READY_HARDWARE_UNAVAILABLE" if issues else "SOFTWARE_READY_AWAITING_ATTENDED_CHECK",
            "blocking_issues": [{"code": code, "reason": reason} for code, reason in issues],
            "so101_studio": {**studio, "physical_preflight_verified": attended["current"]}, "mobile_rgb_and_clocks": rgb,
            "active_native_capture": team["active_native_capture"],
            "recording_installation": installation,
            "capture_volume_reference": team.get("capture_volume_reference"),
            "capture_budget": capture_budget(config, installation["native_capture_storage"],
                                             team.get("capture_volume_reference")),
            "attended_so101_check": attended, "pending_attended_checks": pending_attended_setup(attended),
            "motion_requested": False, "recording_started": False}


def pending_attended_setup(so101):
    mobile = ["Distributed scene and witness framing"]
    if so101["current"]:
        return mobile
    return ["SO101 scene and camera framing", "SO101 native control and STOP verification", *mobile]


def mobile_archive_pending(journal):
    if not journal.is_file():
        return True
    try:
        state = load(journal)
    except (OSError, ValueError):
        return True
    if not isinstance(state, dict):
        return True
    return state.get("status") in ACTIVE_STORY_STATES and not state.get("fleet_capture_id")


def projected_capture_space(count, estimates, storage):
    floor = storage["native_minimum_free_bytes"]
    required = [floor + count * sample for sample in estimates]
    return {"archives_remaining": count, "estimated_bytes_with_native_reserve": required,
            "free_bytes_beyond_upper_estimate": storage["free_bytes"] - required[-1]}


def capture_budget(config, storage, reference):
    root = config.get("data_root")
    if root is None or not storage or "readback_error" in storage or reference is None:
        return None
    prepared = root / "collection-plans" / f"{PLAN_VERSION}.json"
    if not prepared.is_file():
        return None
    workspace = root / "physical-stories"
    try:
        pending = [row for row in load(prepared)["cards"]
                   if mobile_archive_pending(workspace / row["story_id"] / "trial.json")]
    except (OSError, ValueError, TypeError, KeyError):
        return None
    earliest = min((row["block"] for row in pending), default=None)
    next_count = sum(row["block"] == earliest for row in pending)
    estimates = reference["estimated_60s_payload_bytes"]
    return {"scope": "Observed Explorer payload rates plus a retained native reserve",
            "excluded_costs": ["Studio recording and export", "derived story files", "capture overhead"],
            "next_block": {"block": earliest, **projected_capture_space(next_count, estimates, storage)},
            "remaining_deck": projected_capture_space(len(pending), estimates, storage)}


def fleet_storage_blocker(space):
    if space is None:
        return "OFFLINE", "The native Explorer recorder helper is not configured"
    if "readback_error" in space:
        return "OFFLINE", f"The native Explorer recorder helper did not report capture storage. {space['readback_error']}"
    if space["free_bytes"] < space["native_minimum_free_bytes"]:
        return "STORAGE_LIMIT", "Explorer requires at least 1 GiB free before starting native capture"
    return None


def recording_installation(config):
    root = config.get("data_root")
    studio = config.get("studio_project")
    location = config.get("physical_archive_volume")
    volume = Path(location) if location else None
    mounted = bool(volume and volume.is_mount())
    exporter = studio / "scripts/export_lerobot.py" if studio else None
    native_python = config.get("studio_python")
    export_present = bool(exporter and exporter.is_file() and native_python and os.access(native_python, os.X_OK))
    return {"data_root_free_bytes": disk_usage(root).free if root and root.is_dir() else None,
            "studio_exporter_path": str(exporter) if exporter else None, "studio_exporter_installed": export_present,
            "native_capture_storage": fleet_capacity(config),
            "secondary_archive_storage": {"volume": str(volume), "mounted": mounted,
                                          "free_bytes": disk_usage(volume).free if mounted else None} if volume else None,
            "recorder_reservation": "Checked at attended start by Studio's native reservation",
            "observation_endpoint": "/api/fleet/live", "capture_inventory_endpoint": "/api/fleet/captures"}


def fleet_capacity(config):
    if not (config.get("explorer_python") and config.get("explorer_project")):
        return None
    command, environment = fleet_invocation(config, "--capacity")
    try:
        done = subprocess.run(command, **environment, text=True, capture_output=True, check=True, timeout=5)
        space = json.loads(done.stdout)
        if (not isinstance(space, dict) or type(space.get("free_bytes")) is not int
                or type(space.get("native_minimum_free_bytes")) is not int):
            raise ValueError("Invalid native capture storage response")
        return space
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        return {"readback_error": str(error)}


def mobile_transport_action(role, routes, observation):
    states = {row["status"] for row in routes}
    if observation["rgb_fresh"] and observation["clock_within_limit"]:
        return "The live Explorer RGB and clock checks are ready."
    if "clock_sample" in states:
        return "Check this role's RGB and clock on the existing read-only Explorer observation page."
    if "native_admission_denied" in states:
        return "Check this role's admission in the existing native readiness settings."
    if "name_resolution_failed" in states:
        return "On the robot console, verify the configured hostname and its connection to Homebrain's lab network."
    if states & {"host_unreachable", "timeout"}:
        return f"On {role}'s console, verify the configured address and its Ethernet or Wi-Fi connection to the lab network."
    return "Check this role's configured read-only sensor service on the native console."


def story_readiness_with_transport(config):
    report = story_readiness(config)
    if not (config.get("explorer_python") and config.get("explorer_project")):
        report["mobile_native_transport"] = {"probe_error": "The native Explorer helper is not configured"}
        return report
    command, environment = fleet_invocation(config, "--probe-mobile")
    roles = report["mobile_rgb_and_clocks"]["roles"]
    try:
        done = subprocess.run(command, **environment, text=True, capture_output=True, check=True, timeout=8)
        check = json.loads(done.stdout)
        routes = check["routes"]
        expected = {(role, isolated) for role in roles for isolated in (False, True)}
        actual = {(row["role"], row["isolated"]) for row in routes
                  if isinstance(row["status"], str) and type(row["isolated"]) is bool and type(row["port"]) is int}
        if (len(routes) != len(expected) or actual != expected or check["motion_requested"] is not False
                or check["recording_started"] is not False):
            raise ValueError("Invalid native mobile probe receipt")
        check["next_check_by_role"] = {role: mobile_transport_action(role, [row for row in routes if row["role"] == role], observation)
                                      for role, observation in roles.items()}
        report["mobile_native_transport"] = check
    except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as error:
        report["mobile_native_transport"] = {"probe_error": str(error)}
    return report


def story_preflight(config):
    readiness = story_readiness(config)
    blockers = readiness["blocking_issues"]
    if blockers:
        raise ProtocolError(blockers[0]["code"], ". ".join(item["reason"] for item in blockers))
    require_recording_lease(config)


def prepare_trial_folder(trial):
    if trial.is_symlink() or (trial.exists() and not trial.is_dir()):
        raise ValueError("The prepared trial path is not a story directory")
    trial.mkdir(exist_ok=True)
    pending = r"trial\.json\.[a-f0-9]{32}\.tmp"
    if any(item.is_symlink() or not item.is_file() or not re.fullmatch(pending, item.name) for item in trial.iterdir()):
        raise ValueError("The prepared story already contains saved material")


@story_update
def begin(config, identifier, *, staged=False, evaluation_mode="online"):
    if evaluation_mode not in ("online", "blind_offline"):
        raise ValueError("Choose online or blind offline evaluation for this story")
    card = card_for(config, identifier)
    story_preflight(config)
    root = story_workspace(config)
    open_trials = [load(path) for path in root.glob("*/trial.json")]
    if any(row["status"] in ACTIVE_STORY_STATES for row in open_trials):
        raise ProtocolError("RESOURCE_BUSY", "Complete the active physical story before opening another card")
    native_id = dataset(config, root)
    before = episode_ids(config, native_id)
    trial = root / identifier
    prepare_trial_folder(trial)
    record = {"card": card, "status": "placement_recording" if staged else "preparing", "studio_dataset_id": native_id,
              "start": anchor(), "homebrain_boot_id": homebrain_boot(),
              "preexisting_native_episode_ids": sorted(before), "markers": {}, "evaluation_mode": evaluation_mode}
    atomic(trial / "trial.json", record)
    return record if staged else request_fleet_start(config, trial, record)


def request_fleet_start(config, trial, state):
    try:
        capture = launch_fleet(config, trial)
        if capture is not None:
            return accept_fleet_start(trial, state, load(trial / "fleet-start.json"))
    except (OSError, RuntimeError) as error:
        state["reason"] = str(error)
        atomic(trial / "trial.json", state)
        raise ProtocolError("OFFLINE", str(error)) from error
    return state


@story_update
def start_relocation(config, identifier):
    card_for(config, identifier)
    trial = story_workspace(config) / identifier
    state = load(trial / "trial.json")
    if state["status"] != "placement_recording" or "placement_closed" not in state["markers"]:
        raise ValueError("Finish the Studio placement phase before starting Explorer")
    if not same_recording_boot(state):
        raise ValueError("Homebrain rebooted before the mobile observation phase")
    story_preflight(config)
    episode = corresponding_native_episode(config, state)
    if episode["metadata"].get("status") != "complete":
        raise ValueError("Wait for Studio to finish saving the matching episode")
    state.update(status="preparing", fleet_launch_requested=anchor(), staged_studio_episode_id=episode["id"])
    atomic(trial / "trial.json", state)
    return request_fleet_start(config, trial, state)


def episode_ids(config, identifier):
    rows = native_request(config, f"/api/episodes/{identifier}?include_frames=false")
    return {row["id"]: row for row in rows}


def story_phases(state):
    return ("insertion_closed", *PHASES) if state["card"]["family"] == "F3" else PHASES


def mobile_phase_bounds(state, query_ns):
    begin = state["markers"]["placement_closed"]["monotonic_ns"]
    moved = state["markers"]["relocation_closed"]["monotonic_ns"]
    if "fleet_launch_requested" in state:
        launch = state["fleet_launch_requested"]["monotonic_ns"]
        acknowledged = state["fleet_acknowledgement"]["monotonic_ns"]
        if not begin <= launch <= acknowledged or state["fleet_window_s"] != FLEET_WINDOW_S:
            raise ValueError("The staged mobile capture does not follow the saved Studio phase")
        begin = acknowledged
    if not begin < moved < query_ns:
        raise ValueError("The native placement, relocation and query windows are out of order")
    return (begin, moved), (moved, query_ns)


def stage_allows_marker(config, state, phase):
    if state["status"] == "placement_recording":
        return phase != "relocation_closed"
    if state["status"] != "recording":
        return False
    if "fleet_launch_requested" in state:
        return phase == "relocation_closed" and fleet_manifest(config, state["fleet_capture_id"])["phase"] == "recording"
    return True


@story_update
def mark(config, identifier, phase, *, assistance=None, placement_outcome=None, observed_change=None):
    card_for(config, identifier)
    journal = story_workspace(config) / identifier / "trial.json"
    state = load(journal)
    completed = state["markers"]
    phases = story_phases(state)
    if not same_recording_boot(state):
        raise ValueError("Homebrain rebooted before this phase could be recorded")
    if len(completed) >= len(phases) or phase != phases[len(completed)] or not stage_allows_marker(config, state, phase):
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
    expected = state.get("staged_studio_episode_id")
    if expected and matching[0]["id"] != expected:
        raise StudioEpisodeUnlinked("The Studio episode changed after the mobile observation phase started")
    return matching[0]
