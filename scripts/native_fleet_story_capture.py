"""Keep the existing Explorer read-only capture alive for one attended story."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import errno
import json
from pathlib import Path
import re
from shutil import disk_usage
import socket
from struct import error as StructError
import time

def capacity():
    from explorer4.paths import DATA

    target = DATA / "fleet-captures"
    existing = next(path for path in (target, *target.parents) if path.is_dir())
    return {"directory": str(target), "checked_directory": str(existing),
            "free_bytes": disk_usage(existing).free, "native_minimum_free_bytes": 1024**3}


def transport_failure(error, *, timed_out=False):
    cause = error
    while cause.__cause__ is not None:
        cause = cause.__cause__
    detail = str(cause)
    code = getattr(cause, "errno", None)
    wrapped = re.match(r"^\[Errno (-?\d+)\]", detail)
    if code is None and wrapped is not None:
        code = int(wrapped[1])
    if isinstance(cause, socket.gaierror) or code in (socket.EAI_NONAME, socket.EAI_AGAIN):
        status = "name_resolution_failed"
    elif code in (errno.EHOSTUNREACH, errno.ENETUNREACH):
        status = "host_unreachable"
    elif code == errno.ECONNREFUSED:
        status = "connection_refused"
    elif timed_out:
        status = "timeout"
    else:
        status = "transport_error"
    return {"status": status, "errno": code, "detail": detail}


def probe_route(robot, isolated):
    import httpx

    from explorer4.sensors import SensorClient

    port = (8043 if robot == "tank" else 8053) + int(isolated)
    result = {"role": robot, "isolated": isolated, "port": port}
    try:
        with closing(SensorClient(robot=robot, isolated=isolated)) as client:
            result["host"] = client.http.base_url.host
            sample = client.clock_sample()
        remote = sample["remote"]
        pair = remote.get("clock_pair") or {}
        valid = bool(remote.get("boot_id") and pair.get("ros_time_ns") and not pair.get("use_sim_time"))
        return {**result, "status": "clock_sample" if valid else "source_clock_unusable",
                "round_trip_ms": (sample["after_ns"] - sample["before_ns"]) / 1e6}
    except PermissionError as error:
        return {**result, "status": "native_admission_denied", "detail": str(error)}
    except httpx.HTTPStatusError as error:
        return {**result, "status": "native_http_error", "http_status": error.response.status_code}
    except httpx.RequestError as error:
        return {**result, **transport_failure(error, timed_out=isinstance(error, httpx.TimeoutException))}
    except (OSError, ValueError, KeyError) as error:
        return {**result, "status": "native_clock_check_error", "detail": str(error)}


def probe_mobile():
    roles = ("tank", "rosmaster")
    with ThreadPoolExecutor(max_workers=4) as executor:
        requests = [executor.submit(probe_route, role, isolated) for role in roles for isolated in (False, True)]
        routes = [request.result() for request in requests]
    return {"scope": "read-only native sensor clock health", "routes": routes,
            "motion_requested": False, "recording_started": False}


def store_receipt(path, value):
    temporary = path.with_suffix(".pending")
    temporary.write_text(json.dumps(value))
    temporary.replace(path)


def observe_archive_start(archive, path, launch):
    from smart_swarm.native_fleet_start import first_recorded_rgb

    acknowledged = launch["acknowledged"]
    try:
        frames = first_recorded_rgb(archive, acknowledged["unix_ns"])
        if frames is None:
            return False
        store_receipt(path, {"capture_id": launch["id"], "gate": launch["rgb_start_gate"],
                             "acknowledged": acknowledged, "recorded_rgb": frames,
                             "seen_monotonic_ns": time.monotonic_ns(), "seen_unix_ns": time.time_ns()})
    except (OSError, ValueError, StructError) as error:
        print(f"Live Explorer RGB readback unavailable. {error}", flush=True)
    return True


def await_capture(fleet_capture, launch, ready_path):
    archive = fleet_capture.entry(launch["id"])
    deadline = time.monotonic() + 150
    observed = False
    while time.monotonic() < deadline and fleet_capture.state()["active"] == launch["id"]:
        if not observed:
            observed = observe_archive_start(archive, ready_path, launch)
        time.sleep(.5)
    outcome = json.loads((archive / "manifest.json").read_text())
    if outcome["phase"] != "complete":
        raise RuntimeError(outcome.get("error") or "Explorer native capture has no complete manifest")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path, nargs="?")
    probe = parser.add_mutually_exclusive_group()
    probe.add_argument("--capacity", action="store_true", help="Read the native capture filesystem without recording")
    probe.add_argument("--probe-mobile", action="store_true", help="Read configured mobile sensor clock endpoints")
    parser.add_argument("--seconds", type=int, choices=(15, 30, 60), default=60)
    args = parser.parse_args()
    if args.capacity:
        print(json.dumps(capacity()))
        return
    if args.probe_mobile:
        print(json.dumps(probe_mobile()))
        return
    if args.receipt is None:
        parser.error("The native start receipt path is required")
    from explorer4 import fleet_capture
    from smart_swarm.native_fleet_start import START_GATE, START_RECEIPT

    result = fleet_capture.start(args.seconds)
    result.update(requested_seconds=args.seconds, rgb_start_gate=START_GATE,
                  acknowledged={"monotonic_ns": time.monotonic_ns(), "unix_ns": time.time_ns()})
    store_receipt(args.receipt, result)
    await_capture(fleet_capture, result, args.receipt.with_name(START_RECEIPT))


if __name__ == "__main__":
    main()
