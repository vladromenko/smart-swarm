"""Atomic episode storage, separate namespaces and read-only scoped replay."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import uuid

from PIL import Image

from .contract import (CLOCK, CONTRACT_SHA, VERSION, ProtocolError, interval, json_bytes,
                       project_packet, read_ns, time_at)
from .memory import admissible, predict

POLICY = "policy_observations"
TRUTH = "evaluator_truth"
LABELS = "training_labels"
NAMESPACES = (POLICY, TRUTH, LABELS, "communication", "decisions")
SPLITS = ("development", "train", "validation", "test")
DELIVERIES = "communication/deliveries.jsonl"


def atomic(path, value):
    pending = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    pending.write_bytes(json_bytes(value) + b"\n")
    pending.replace(path)


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Recorder:
    def __init__(self, path, world_id, scene_group_id, cameras, metadata, *, quota_mb=112, clock=CLOCK):
        self.root = path
        self.world_id = world_id
        self.cameras = deepcopy(cameras)
        self.counts = dict.fromkeys(cameras, 0)
        self.quota_bytes = quota_mb * 1024**2
        self.saved_bytes = 0
        self.clock = deepcopy(clock)
        self.meta = {**metadata, "protocol_version": VERSION, "contract_sha256": CONTRACT_SHA,
                     "world_episode_id": world_id, "scene_group_id": scene_group_id,
                     "lineage": {"source_world_ids": [], "split": "development"},
                     "episodes": {agent: f"{world_id}-{idx}" for idx, agent in enumerate(cameras)},
                     "policy_cameras": cameras}
        path.mkdir(parents=True, exist_ok=False)
        for name in NAMESPACES:
            (path / name).mkdir()
        atomic(path / "recording.json", {**self.meta, "recording_integrity": "recording"})

    @classmethod
    def reopen_pending(cls, path):
        recorder = cls.__new__(cls)
        recorder.meta = load(path / "recording.json")
        recorder.root = path
        recorder.world_id = recorder.meta["world_episode_id"]
        recorder.cameras = recorder.meta["policy_cameras"]
        recorder.clock = recorder.meta["native_recording_clock"]
        if recorder.meta["recording_integrity"] != "recording":
            raise ValueError("Only a pending native recording can be reopened")
        recorder.counts = {agent: len(rows(path / POLICY / f"{agent}.jsonl")) for agent in recorder.cameras}
        recorder.saved_bytes = sum(item.stat().st_size for item in path.glob("*/*.png"))
        recorder.quota_bytes = 112 * 1024**2
        return recorder

    def append(self, namespace, name, row):
        path = self.root / namespace / f"{name}.jsonl"
        with path.open("ab") as stream:
            stream.write(json_bytes(row) + b"\n")

    def frame(self, namespace, sensor, sequence, image, timestamp):
        frame_id = f"{self.world_id}-{sensor.replace('/', '-')}-{sequence:04}"
        path = self.root / namespace / (frame_id + ".png")
        Image.fromarray(image).save(path)
        self.saved_bytes += path.stat().st_size
        if self.saved_bytes > self.quota_bytes:
            raise ProtocolError("STORAGE_LIMIT", "Episode frame cap reached")
        height, width = image.shape[:2]
        return {"sensor_id": sensor, "frame_id": frame_id, "frame_ref": str(path.relative_to(self.root)),
                "sha256": sha(path), "hash_valid": True, "hash_reason": None,
                "capture_interval": interval(timestamp, timestamp, self.clock),
                "exposure_time": time_at(timestamp, self.clock) if self.meta["run_mode"] == "mujoco" else None,
                "encoding": "png", "height": height, "width": width, "frame_age_ms": 0.0,
                "calibration_revision": self.meta.get("frame_calibration_revision", "synthetic-b1-v1"), "dropped_since_previous": 0,
                "repeated": False, "access_role": "agent_input" if namespace == POLICY else "evaluator_only"}

    def capture(self, agent, views, timestamp, state, units, applied_ref=None, *, exposures=None):
        self.counts[agent] += 1
        sequence = self.counts[agent]
        frames = exposures or {}
        cameras = [self.frame(POLICY, sensor, sequence, im, frames.get(sensor, timestamp)) for sensor, im in views.items()]
        for camera in cameras:
            reference = frames.get(camera["sensor_id"], timestamp)
            camera["frame_age_ms"] = (timestamp - reference) / 1e6
        packet = {"protocol_version": VERSION, "run_mode": self.meta["run_mode"],
                  "world_episode_id": self.world_id, "episode_id": self.meta["episodes"][agent],
                  "agent_id": agent, "sample_id": f"{self.meta['episodes'][agent]}-{sequence:04}",
                  "sequence": sequence, "source_time": time_at(timestamp, self.clock),
                  "receive_time": time_at(timestamp, self.clock),
                  "clock_mapping_id": self.meta.get("clock_mapping_id", "identity-sim-clock"),
                  "clock_uncertainty_ms": self.meta.get("clock_mapping_uncertainty_ms", 0.0),
                  "measured_state": state, "state_units": units,
                  "validity": {"state_measured": bool(state), "camera_timestamp_is_exposure": self.meta["run_mode"] == "mujoco"},
                  "applied_command_ref": applied_ref, "cameras": cameras}
        private = project_packet(packet, agent, self.cameras[agent])
        self.append(POLICY, agent, private)
        return private

    def close(self, query, decision, reveal):
        record = {**self.meta, "recording_integrity": "complete", "sample_counts": self.counts,
                  "query_boundary": time_at(query, self.clock), "decision_boundary": time_at(decision, self.clock),
                  "reveal_boundary": time_at(reveal, self.clock)}
        validate(self.root, record)
        record["sidecar_sha256"] = {str(path.relative_to(self.root)): sha(path)
                                   for path in sorted(self.root.glob("*/*.jsonl"))}
        atomic(self.root / "manifest.json", record)
        (self.root / "recording.json").unlink()

    def quarantine(self, reason):
        atomic(self.root / "recording.json", {**self.meta, "recording_integrity": "incomplete",
                                             "status": "quarantined", "reason": reason})


def rows(path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def listed_file(root, reference, namespace):
    home = (root / namespace).resolve()
    path = (root / reference).resolve()
    if not path.is_relative_to(home) or not path.is_file():
        raise ValueError(f"Frame is outside authorized {namespace} scope or missing")
    return path


def audited_frame(root, camera, namespace):
    path = listed_file(root, camera["frame_ref"], namespace)
    if sha(path) != camera["sha256"]:
        raise ValueError("Frame content does not match hash")
    with Image.open(path) as image:
        if image.size != (camera["width"], camera["height"]) or image.mode != "RGB":
            raise ValueError("Frame format disagrees with descriptor")
        image.verify()


def validate(root, manifest):
    if (manifest["contract_sha256"] != CONTRACT_SHA or manifest["protocol_version"] != VERSION
            or manifest["recording_integrity"] != "complete"
            or manifest["lineage"]["split"] not in SPLITS):
        raise ValueError("Invalid protocol, split or recording integrity")
    query, decision, reveal = [manifest[key] for key in
                               ("query_boundary", "decision_boundary", "reveal_boundary")]
    clock = query["clock_id"]
    expected = manifest.get("native_recording_clock") if manifest["run_mode"] == "hardware" else CLOCK
    if not clock == decision["clock_id"] == reveal["clock_id"] == expected:
        raise ValueError("Episode clock disagrees with the declared run clock")
    query, decision, reveal = [read_ns(row["monotonic_ns"]) for row in (query, decision, reveal)]
    if not query <= decision < reveal:
        raise ValueError("Invalid query, decision or reveal ordering")
    for agent, allowed in manifest["policy_cameras"].items():
        validate_agent(root, manifest, agent, allowed, query, clock)
    validate_reveal(root, decision, reveal, clock)
    for reference, checksum in manifest.get("sidecar_sha256", {}).items():
        namespace = Path(reference).parts[0]
        if namespace not in NAMESPACES or sha(listed_file(root, reference, namespace)) != checksum:
            raise ValueError("Invalid episode sidecar hash or path")


def validate_agent(root, manifest, agent, allowed, bound, clock):
    latest = -1
    packets = rows(root / POLICY / f"{agent}.jsonl")
    if len(packets) != manifest["sample_counts"][agent] or not packets:
        raise ValueError(f"Missing required camera packets for {agent}")
    for seq, row in enumerate(packets, 1):
        packet = project_packet(row, agent, allowed)
        stamp = read_ns(packet["source_time"]["monotonic_ns"])
        receive = read_ns(packet["receive_time"]["monotonic_ns"])
        if (row["sequence"] != seq or not latest < stamp <= receive <= bound
                or packet["source_time"]["clock_id"] != clock or packet["receive_time"]["clock_id"] != clock
                or packet["world_episode_id"] != manifest["world_episode_id"]
                or packet["episode_id"] != manifest["episodes"][agent]):
            raise ValueError("Invalid private packet order, clock, identity or time")
        if sorted(c["sensor_id"] for c in packet["cameras"]) != sorted(allowed):
            raise ValueError("Declared private camera missing from packet")
        for camera in packet["cameras"]:
            capture = camera["capture_interval"]
            if capture["clock_id"] != clock or read_ns(capture["end_ns"]) > stamp:
                raise ValueError("Camera exposure is inconsistent with source packet")
            audited_frame(root, camera, POLICY)
        latest = stamp


def validate_reveal(root, decision, reveal, clock):
    records = rows(root / TRUTH / "reveal.jsonl")
    if len(records) != 1:
        raise ValueError("Expected one terminal reveal")
    row = records[0]
    target = row["target_state_time_or_interval"]
    camera = row["camera"]
    capture = camera["capture_interval"]
    if (not row["relations_preserved_until_reveal"] or target != interval(decision, reveal, clock)
            or camera["access_role"] != "evaluator_only" or capture != interval(reveal, reveal, clock)):
        raise ValueError("Terminal reveal cannot label the scored decision")
    audited_frame(root, camera, TRUTH)


class Replay:
    """A read-only client with a fixed agent grant and no evaluator reader."""

    def __init__(self, root, agent):
        self._root = root
        manifest = load(root / "manifest.json")
        self.agent = agent
        self.world_id = manifest["world_episode_id"]
        self.allowed = tuple(manifest["policy_cameras"][agent])
        self.clock = manifest["query_boundary"]["clock_id"]
        self.query = read_ns(manifest["query_boundary"]["monotonic_ns"])
        self.decision = read_ns(manifest["decision_boundary"]["monotonic_ns"])
        self._delivery_hash = manifest.get("sidecar_sha256", {}).get(DELIVERIES)

    def acquire(self, run_mode, mode="observe"):
        if run_mode != "replay" or mode != "observe":
            raise ProtocolError("NOT_ARMED", "Replay grants observation only")
        return {"run_mode": "replay", "state": "OBSERVE", "agent_id": self.agent,
                "camera_ids": self.allowed}

    def packets(self, until_ns=None, after_sequence=0):
        cutoff = self.query if until_ns is None else min(until_ns, self.query)
        data = rows(self._root / POLICY / f"{self.agent}.jsonl")
        return [packet for row in data if row["sequence"] > after_sequence
                and available(packet := project_packet(row, self.agent, self.allowed), cutoff, self.clock)]

    def image(self, frame_id, until_ns=None):
        for packet in self.packets(until_ns):
            for camera in packet["cameras"]:
                if camera["frame_id"] == frame_id:
                    path = listed_file(self._root, camera["frame_ref"], POLICY)
                    if sha(path) != camera["sha256"]:
                        raise ValueError("Frame content does not match hash")
                    with Image.open(path) as image:
                        return image.copy()
        raise KeyError("Frame was not granted to this agent at the requested time")

    def predict(self, until_ns=None):
        cutoff = self.decision if until_ns is None else min(until_ns, self.decision)
        return predict(self.deliveries(cutoff), cutoff, self.agent)

    def evidence(self, until_ns=None):
        cutoff = self.decision if until_ns is None else min(until_ns, self.decision)
        return admissible(self.deliveries(cutoff), cutoff, self.agent)

    def deliveries(self, until_ns=None):
        cutoff = self.decision if until_ns is None else min(until_ns, self.decision)
        if self._delivery_hash is None:
            if (self._root / DELIVERIES).exists():
                raise ValueError("Delivery channel is absent from sealed provenance")
            return []
        path = listed_file(self._root, DELIVERIES, "communication")
        if sha(path) != self._delivery_hash:
            raise ValueError("Delivery channel differs from sealed provenance")
        return [row for row in rows(path) if row["recipient"] == self.agent and int(row["delivered_ns"]) <= cutoff]

def available(packet, cutoff, clock=CLOCK):
    source, receive = packet["source_time"], packet["receive_time"]
    stamp, receipt = read_ns(source["monotonic_ns"]), read_ns(receive["monotonic_ns"])
    frames = [camera["capture_interval"] for camera in packet["cameras"]]
    return (source["clock_id"] == receive["clock_id"] == clock and stamp <= receipt <= cutoff
            and all(frame["clock_id"] == clock and read_ns(frame["end_ns"]) <= stamp
                    for frame in frames))
