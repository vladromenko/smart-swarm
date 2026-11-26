"""Scope and run the frozen learned witness protocol before a physical reveal."""

from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from .contract import ProtocolError, json_bytes, time_at
from .episode import POLICY, atomic, load, rows, sha, validate_agent
from .evidence_inputs import AGENTS, canonical_packet, opaque_map
from .physical_slot_reference import SCOPE_FLAG
from .primary_inputs import copy_camera_stream, json_lines
from .research_jobs import admitted
from .training import ROOT, offline

PROFILE = ROOT / "evidence/learning-pilot-v3/primary-profiles-frozen.json"
WORKER = ROOT / "scripts/physical_learned_worker.py"
SCHEMA = "smart-swarm-frozen-physical-memory-v1"
METHOD = "selective_with_observed_marker_support"
HEADS = ("events_only", "events_plus_reveal")
TIMEOUT_S = 30


def packets(root, manifest):
    grants = manifest["policy_cameras"]
    if set(grants) != set(AGENTS):
        raise ValueError("The learned physical window requires the three declared roles")
    return {agent: rows(root / POLICY / f"{agent}.jsonl") for agent in grants}


def window_digest(manifest, stream):
    mapping = opaque_map(stream, manifest["world_episode_id"])
    clean = {agent: [canonical_packet(packet, mapping) for packet in frames] for agent, frames in stream.items()}
    fields = {key: manifest[key] for key in ("policy_cameras", "query_boundary")}
    if manifest.get(SCOPE_FLAG):
        fields[SCOPE_FLAG] = True
    return sha256(json_bytes({**fields, "private_history": clean})).hexdigest()


def validate_scope(root, manifest, stream):
    boundary = manifest["query_boundary"]
    query = int(boundary["monotonic_ns"])
    hardware = manifest.get("run_mode") == "hardware"
    for agent, allowed in manifest["policy_cameras"].items():
        validate_agent(root, manifest, agent, allowed, query, boundary["clock_id"])
        for packet in stream[agent]:
            if [camera["sensor_id"] for camera in packet["cameras"]] != list(allowed):
                raise ValueError(f"{agent} camera order differs from its grant")
            if hardware and not allowed and (packet["measured_state"] or packet["state_units"]
                                             or packet["applied_command_ref"] is not None):
                raise ValueError(f"{agent} retained state or commands in a withheld view")


@contextmanager
def scoped_window(root, manifest):
    stream = packets(root, manifest)
    boundary = manifest["query_boundary"]
    query = int(boundary["monotonic_ns"])
    validate_scope(root, manifest, stream)
    mapping = opaque_map(stream, manifest["world_episode_id"])
    scoped = {key: manifest[key] for key in ("policy_cameras", "query_boundary")}
    if manifest.get(SCOPE_FLAG):
        scoped[SCOPE_FLAG] = True
    deadline = query + load(ROOT / "configs/primary-memory-study.json")["communication_reply_window_ms"] * 1_000_000
    scoped.update(world_episode_id="world", decision_boundary=time_at(deadline, boundary["clock_id"]))
    with TemporaryDirectory(prefix="smart-swarm-native-learned-") as directory:
        target = Path(directory)
        (target / POLICY).mkdir()
        for agent, records in stream.items():
            projected = copy_camera_stream(root, target, records, mapping)
            json_lines(target / POLICY / f"{agent}.jsonl", projected)
        atomic(target / "manifest.json", scoped)
        digest = window_digest(manifest, stream)
        if window_digest(scoped, packets(target, scoped)) != digest:
            raise ValueError("The learned window changed the recorded private packets")
        yield target, digest


def verify_result(report, digest):
    if (report["schema"] != SCHEMA or report["policy"] != METHOD or report["permitted_input_sha256"] != digest
            or report["profiles_sha256"] != sha(PROFILE) or set(report["heads"]) != set(HEADS)):
        raise ValueError("The frozen physical inference receipt disagrees with its policy input or profile")
    profile = load(PROFILE)
    if report["encoder_sha256"] != profile["encoder_sha256"]:
        raise ValueError("The frozen physical inference receipt names a different RGB encoder")
    fits = profile["fits"]
    for kind, row in report["heads"].items():
        fit = fits[kind]
        threshold = fit["mount_and_receiver_profiles"]["3"]["decision_thresholds"][METHOD]
        if row["checkpoint_sha256"] != fit["checkpoint_sha256"] or row["threshold"] != threshold:
            raise ValueError("The frozen physical inference receipt names different checkpoint settings")


def run_worker(scoped, digest):
    env = {**offline(), "PYTHONPATH": str(ROOT / "src"), "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    argv = [sys.executable, "-B", str(WORKER), str(scoped)]
    result = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, text=True, timeout=TIMEOUT_S)
    if result.returncode:
        details = result.stderr.strip().splitlines()
        raise ValueError(f"Frozen learned physical inference exited {result.returncode}. {details[-1] if details else ''}")
    report = json.loads(result.stdout)
    verify_result(report, digest)
    return report


def inference(config, recorder, query):
    manifest = {**recorder.meta, "sample_counts": recorder.counts, "query_boundary": time_at(query, recorder.clock)}
    try:
        with admitted(config, "smart-swarm-physical-evidence-inference") as (grant, heartbeat):
            with scoped_window(recorder.root, manifest) as (scoped, digest):
                result = run_worker(scoped, digest)
            heartbeat()
        admission = {key: grant[key] for key in ("job_id", "device", "scope") if key in grant}
        return {"status": "scored", "receipt": result, "admission": admission}
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.TimeoutExpired, ProtocolError) as error:
        return {"status": "unavailable", "reason": str(error)}


def verify_optional(root, manifest, decision):
    learned = decision.get("learned_evidence")
    if learned is None:
        return {"status": "historical_rule_only"}
    if learned["status"] == "scored":
        verify_result(learned["receipt"], window_digest(manifest, packets(root, manifest)))
    elif learned["status"] != "unavailable" or not learned.get("reason"):
        raise ValueError("The physical inference attempt has neither a receipt nor an explicit failure")
    return learned
