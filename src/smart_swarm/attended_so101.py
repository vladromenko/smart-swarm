"""Record an operator's SO101 setup check against fresh native follow and STOP evidence."""

import re
import time

from .contract import ProtocolError
from .episode import atomic, load, sha
from .studio import native_binding, native_status, require_recording_lease

SCHEMA = "smart-swarm-attended-so101-v1"
DEVICES = ("leader", "follower", "wrist", "top")
ARMS = DEVICES[:2]
CAMERAS = DEVICES[2:]
FLAGS = ("recording_active", "teleoperation_active", "policy_active", "native_arm_hardware_open",
         "native_stop_check_required")
SNAPSHOT_KEYS = (*FLAGS, "inventory_present", "native_device_identities", "native_camera_preview",
                 "native_stop_generation", "native_stop_state", "native_following_readback",
                 "native_calibration_readback", "native_calibration_readback_monotonic_ns")


def receipt_path(config):
    root = config.get("data_root")
    return root / "so101-pilot/attended-preflight.json" if root else None


def pilot_readiness(config):
    native = native_status(config)
    attended = attended_status(config, native)
    dataset, issues = None, []
    try:
        dataset = native_binding(config)["dataset_id"]
        if re.fullmatch(r"ds_[a-f0-9]{16}", dataset) is None or not (
                config["studio_project"] / "data" / dataset / "dataset.json").is_file():
            issues.append("The bound SO101 pilot dataset is missing from Studio")
    except ProtocolError as error:
        issues.append(error.message)
    exporter = (config["studio_project"] / "scripts/export_lerobot.py").is_file()
    if not exporter:
        issues.append("The installed Studio exporter is unavailable")
    lease = False
    try:
        idle_hardware_snapshot(native)
        require_recording_lease(config)
        lease = True
    except (OSError, ValueError, ProtocolError) as error:
        issues.append(str(error))
    state = "READY_FOR_SO101_PILOT" if attended["current"] else "SOFTWARE_READY_AWAITING_ATTENDED_CHECK"
    return {"scope": "SO101-only four-attempt pilot", "status": "SOFTWARE_READY_HARDWARE_UNAVAILABLE" if issues else state,
            "native_dataset_id": dataset, "inventory_present": native["inventory_present"],
            "camera_preview": native["native_camera_preview"], "studio_exporter_installed": exporter,
            "studio_free_bytes": native["native_resources"]["disk_free_bytes"], "recording_reservation_available": lease,
            "attended_so101_check": attended, "blocking_issues": issues, "motion_requested": False}


def idle_hardware_snapshot(native):
    if any(native[key] is not False for key in FLAGS):
        raise ProtocolError("RESOURCE_BUSY", "Studio must report idle controls and no outstanding STOP issue")
    identities = native["native_device_identities"]
    if (not all(native["inventory_present"][key] and identities[key] for key in DEVICES)
            or len(set(identities.values())) != len(DEVICES)):
        raise ProtocolError("OFFLINE", "Studio must identify both native arms and both distinct cameras")
    if not all(native["native_camera_preview"][role]["fresh"] for role in CAMERAS):
        raise ProtocolError("OFFLINE", "Connect the wrist and top camera-only previews in Studio")
    generation = native["native_stop_generation"]
    if type(generation) is not int or generation < 0:
        raise ValueError("Studio did not provide a valid STOP generation")
    return {key: native[key] for key in SNAPSHOT_KEYS}


def verify_following(native, started_ns, finished_ns):
    report = native["native_following_readback"]
    sample = report["last_sample_ns"]
    if (report["source"] != "hardware" or report["state"] != "stopped" or report["phase"] != "following"
            or report["error"] is not None or report["stop_errors"] != []
            or type(report["samples"]) is not int or report["samples"] < 1
            or type(sample) is not int or not started_ns <= sample <= finished_ns):
        raise ValueError("Studio did not report a new successful native following session in this check")
    stamps = native["native_calibration_readback_monotonic_ns"]
    digests = report["calibration_sha256"]
    if not all(native["native_calibration_readback"][arm] is True and type(stamps[arm]) is int
               and started_ns <= stamps[arm] <= sample and re.fullmatch(r"[a-f0-9]{64}", digests[arm] or "")
               for arm in ARMS):
        raise ValueError("Studio did not verify both saved calibrations during this following session")


def verify_window(before, after, started_ns, finished_ns):
    idle_hardware_snapshot(before)
    idle_hardware_snapshot(after)
    if before["native_device_identities"] != after["native_device_identities"]:
        raise ValueError("Native device identities changed during the attended check")
    if (after["native_stop_generation"] <= before["native_stop_generation"]
            or after["native_stop_state"].get("teleoperation") != "devices_closed"):
        raise ValueError("Studio did not acknowledge a new native STOP with the arm devices closed")
    verify_following(after, started_ns, finished_ns)


def record_attended(config):
    from .physical_story_source import homebrain_boot

    before = idle_hardware_snapshot(native_status(config))
    boot = homebrain_boot()
    started = time.monotonic_ns()
    print("Use the existing Studio controls for the attended check. Keep its saved calibration.", flush=True)
    print("Confirm leader and follower tracking, top coverage of the setup, and wrist visibility at pickup and release.")
    print("Use Studio STOP hands and confirm that following has stopped.")
    answer = input("Enter y after you observed all checks and native STOP, or Enter to return: ").strip().lower()
    if answer != "y":
        return {"recorded": False, "reason": "Attended observations were not confirmed"}
    after = idle_hardware_snapshot(native_status(config))
    finished = time.monotonic_ns()
    verify_window(before, after, started, finished)
    path = receipt_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"schema": SCHEMA, "operator_confirmed_framing_tracking_and_stop": True,
              "homebrain_boot_id": boot, "started_monotonic_ns": started, "finished_monotonic_ns": finished,
              "recorded_unix_s": time.time(),
              "before": before, "after": after, "motion_requested_by_smart_swarm": False}
    atomic(path, record)
    return {"recorded": True, "receipt": str(path), "sha256": sha(path),
            "native_stop_generation": after["native_stop_generation"]}


def attended_status(config, native):
    from .physical_story_source import homebrain_boot

    path = receipt_path(config)
    if path is None or not path.is_file():
        return {"current": False, "reason": "No attended SO101 setup and native STOP check has been recorded"}
    try:
        record = load(path)
        if (record["schema"] != SCHEMA or record["homebrain_boot_id"] != homebrain_boot()
                or record["operator_confirmed_framing_tracking_and_stop"] is not True):
            raise ValueError("The attended check must belong to this Homebrain boot and have operator confirmation")
        before, after = record["before"], record["after"]
        verify_window(before, after, record["started_monotonic_ns"], record["finished_monotonic_ns"])
        if native["native_device_identities"] != after["native_device_identities"]:
            raise ValueError("The connected Studio devices changed since the attended check")
        if native["native_following_readback"]["calibration_sha256"] != after["native_following_readback"]["calibration_sha256"]:
            raise ValueError("Studio calibration readback changed since the attended check")
        if native["native_stop_check_required"] or native["native_stop_generation"] < after["native_stop_generation"]:
            raise ValueError("Studio STOP evidence is no longer current")
        return {"current": True, "receipt": str(path), "sha256": sha(path),
                "verified_stop_generation": after["native_stop_generation"]}
    except (OSError, ValueError, TypeError, KeyError, ProtocolError) as error:
        return {"current": False, "receipt": str(path), "reason": str(error)}
