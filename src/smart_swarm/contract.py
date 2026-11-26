"""Small v1 client-side wire and observation checks."""

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import re
from pathlib import Path

VERSION = "robot-platform/1.0.0"
CONTRACT_SHA = "f079a477abcb8448c5e520de945f9c1b63e29fe39951bd378c8314171e9fce02"
CLOCK = {"host_id": "sim-host", "boot_id": "sim-boot-1", "kind": "monotonic"}
PACKET_KEYS = ("protocol_version", "run_mode", "world_episode_id", "episode_id", "agent_id",
               "sample_id", "sequence", "source_time", "receive_time", "clock_mapping_id",
               "clock_uncertainty_ms", "measured_state", "state_units", "validity",
               "applied_command_ref")
CAMERA_KEYS = ("sensor_id", "frame_id", "frame_ref", "sha256", "hash_valid", "hash_reason",
               "capture_interval", "exposure_time", "width", "height", "encoding", "access_role",
               "dropped_since_previous", "repeated", "frame_age_ms", "calibration_revision")


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")


def time_at(ns, clock=CLOCK):
    return {"clock_id": clock.copy(), "monotonic_ns": str(ns)}


def interval(start, end, clock=CLOCK):
    return {"clock_id": clock.copy(), "start_ns": str(start), "end_ns": str(end)}


def read_ns(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9]{1,19}", value) is None:
        raise ValueError("Nanosecond values must be unsigned decimal strings")
    if int(value) >= 2**63:
        raise ValueError("Nanosecond value exceeds signed 64-bit range")
    return int(value)


def clock_at(timestamp):
    clock = timestamp["clock_id"]
    keys = ("host_id", "boot_id", "kind")
    if any(not isinstance(clock[name], str) or not clock[name] for name in keys):
        raise ValueError("Invalid clock identity")
    return {key: clock[key] for key in keys}


def timestamp_at(row):
    return {**{"clock_id": clock_at(row)}, "monotonic_ns": str(read_ns(row["monotonic_ns"]))}


def capture_at(row):
    start, end = [read_ns(row[key]) for key in ("start_ns", "end_ns")]
    if start > end:
        raise ValueError("Reversed capture interval")
    return {"clock_id": clock_at(row), "start_ns": str(start), "end_ns": str(end)}


def project_camera(record):
    camera = {key: deepcopy(record[key]) for key in CAMERA_KEYS if key in record}
    camera["capture_interval"] = capture_at(camera["capture_interval"])
    if camera["exposure_time"] is not None:
        camera["exposure_time"] = timestamp_at(camera["exposure_time"])
    for key in ("sensor_id", "frame_id", "frame_ref", "encoding"):
        if not isinstance(camera[key], str) or not camera[key]:
            raise ValueError(f"Invalid camera {key}")
    if not camera["width"] > 0 or not camera["height"] > 0:
        raise ValueError("Invalid camera resolution")
    return camera


def project_packet(packet, agent, allowed):
    if packet["agent_id"] != agent or packet["protocol_version"] != VERSION:
        raise ValueError("Packet scope or protocol mismatch")
    result = {key: deepcopy(packet[key]) for key in PACKET_KEYS}
    result["source_time"] = timestamp_at(result["source_time"])
    result["receive_time"] = timestamp_at(result["receive_time"])
    result["validity"] = {key: value for key, value in result["validity"].items()
                          if key in ("state_measured", "camera_timestamp_is_exposure",
                                     "state_unavailable_reason")}
    for value in result["measured_state"]:
        if value is not None and type(value) not in (int, float):
            raise ValueError("Invalid measured state scalar")
    if len(result["state_units"]) != len(result["measured_state"]):
        raise ValueError("Measured state and units disagree")
    result["cameras"] = [
        project_camera(row) for row in packet["cameras"]
        if row["sensor_id"] in allowed and row["access_role"] == "agent_input"
    ]
    json_bytes(result)
    return result


@dataclass
class ProtocolError(Exception):
    code: str
    message: str
    retryable: bool = False
    details: dict = field(default_factory=dict)

    def __str__(self):
        return f"{self.code}: {self.message}"


def envelope(request_id, *, data=None, error=None):
    if (data is None) == (error is None):
        raise ValueError("An envelope has exactly one of data or error")
    return {"protocol_version": VERSION, "request_id": request_id, "data": data,
            "error": vars(error) if error is not None else None}


def decode_envelope(reply, request_id):
    if reply["protocol_version"] != VERSION or reply["request_id"] != request_id:
        raise ValueError("Response version or request ID mismatch")
    json_bytes(reply)
    if (reply["data"] is None) == (reply["error"] is None):
        raise ValueError("Invalid v1 response envelope")
    if reply["error"] is not None:
        raise ProtocolError(**reply["error"])
    return reply["data"]


def check_contract(root: Path):
    actual = hashlib.sha256((root / "robot-platform-1.0.0.txt").read_bytes()).hexdigest()
    if actual != CONTRACT_SHA:
        raise ValueError("Frozen contract checksum mismatch")
    wire = json.loads((root / "wire-examples.json").read_text())
    for case in wire["cases"]:
        reply = case["response"]
        request = case["request"]
        request_id = request["request_id"] if request else reply["request_id"]
        try:
            decode_envelope(reply, request_id)
        except ProtocolError as error:
            if case["name"] != "delayed_command_rejection" or error.code != "STALE_COMMAND":
                raise
    return {"protocol_version": VERSION, "contract_sha256": actual,
            "wire_examples": len(wire["cases"])}
