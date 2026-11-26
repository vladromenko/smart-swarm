"""Scope and run the frozen learned witness protocol before a physical reveal."""

from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from .contract import ProtocolError, json_bytes, project_packet, time_at, timestamp_at
from .episode import POLICY, atomic, load, rows, sha, validate_agent
from .evidence_inputs import AGENTS, canonical_packet, opaque_map
from .physical_slot_reference import SCOPE_FLAG
from .primary_inputs import copy_camera_stream, json_lines
from .research_jobs import admitted
from .training import ROOT, offline

PROFILE = ROOT / "evidence/learning-pilot-v3/primary-profiles-frozen.json"
WORKER = ROOT / "scripts/physical_learned_worker.py"
LEGACY_SCHEMA = "smart-swarm-frozen-physical-memory-v1"
SCHEMA = "smart-swarm-frozen-physical-memory-v2"
METHOD = "selective_with_observed_marker_support"
HEADS = ("events_only", "events_plus_reveal")
TIMEOUT_S = 30
DEFERRED = "deferred_blind_offline"


def packets(root, manifest):
    grants = manifest["policy_cameras"]
    if set(grants) != set(AGENTS):
        raise ValueError("The learned physical window requires the three declared roles")
    return {agent: rows(root / POLICY / f"{agent}.jsonl") for agent in grants}


def policy_manifest(manifest):
    query = timestamp_at(manifest["query_boundary"])
    duration = load(ROOT / "configs/primary-memory-study.json")["communication_reply_window_ms"] * 1_000_000
    fields = {"policy_cameras": manifest["policy_cameras"], "query_boundary": query, "world_episode_id": "world",
              "decision_boundary": time_at(int(query["monotonic_ns"]) + duration, query["clock_id"])}
    if manifest.get(SCOPE_FLAG):
        fields[SCOPE_FLAG] = True
    return fields


def project_stream(manifest, stream):
    grants = manifest["policy_cameras"]
    return {agent: [project_packet(packet, agent, grants[agent]) for packet in frames] for agent, frames in stream.items()}


def window_digest(manifest, stream, *, projected=False, legacy=False):
    if legacy:
        fields = {key: manifest[key] for key in ("policy_cameras", "query_boundary")}
        if manifest.get(SCOPE_FLAG):
            fields[SCOPE_FLAG] = True
    else:
        fields = manifest if projected else policy_manifest(manifest)
        stream = project_stream(manifest, stream)
    mapping = opaque_map(stream, manifest["world_episode_id"])
    clean = {agent: [canonical_packet(packet, mapping) for packet in frames] for agent, frames in stream.items()}
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
    validate_scope(root, manifest, stream)
    stream = project_stream(manifest, stream)
    mapping = opaque_map(stream, manifest["world_episode_id"])
    scoped = policy_manifest(manifest)
    with TemporaryDirectory(prefix="smart-swarm-native-learned-") as directory:
        target = Path(directory)
        (target / POLICY).mkdir()
        for agent, records in stream.items():
            projected = copy_camera_stream(root, target, records, mapping)
            json_lines(target / POLICY / f"{agent}.jsonl", projected)
        atomic(target / "manifest.json", scoped)
        digest = window_digest(manifest, stream)
        if window_digest(scoped, packets(target, scoped), projected=True) != digest:
            raise ValueError("The learned window changed the recorded private packets")
        yield target, digest


def verify_result(report, digest, *, schema=SCHEMA):
    if (report["schema"] != schema or report["policy"] != METHOD or report["permitted_input_sha256"] != digest
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
    admitted_online = False
    try:
        with admitted(config, "smart-swarm-physical-evidence-inference") as (grant, heartbeat):
            admitted_online = True
            with scoped_window(recorder.root, manifest) as (scoped, digest):
                result = run_worker(scoped, digest)
            heartbeat()
        admission = {key: grant[key] for key in ("job_id", "device", "scope") if key in grant}
        return {"status": "scored", "receipt": result, "admission": admission}
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.TimeoutExpired, ProtocolError) as error:
        failure = {"status": "unavailable", "reason": str(error) or type(error).__name__}
        if admitted_online:
            return failure
        return defer_online_admission(recorder.root, manifest, failure)


def defer_online_admission(root, manifest, failure):
    try:
        seal = offline_seal(root, manifest)
    except (OSError, ValueError, KeyError, TypeError, ProtocolError) as error:
        return {**failure, "reason": f"{failure['reason']}. Blind offline input seal failed. {error}"}
    return {**seal, "online_attempt": {**failure, "stage": "compute_admission"}}


def offline_seal(root, manifest):
    stream = packets(root, manifest)
    validate_scope(root, manifest, stream)
    return {"status": DEFERRED, "profiles_sha256": sha(PROFILE),
            "permitted_input_sha256": window_digest(manifest, stream)}


def verify_offline_seal(root, manifest, seal):
    expected = offline_seal(root, manifest)
    if "online_attempt" in seal:
        attempt = seal["online_attempt"]
        if (not isinstance(attempt, dict) or set(attempt) != {"status", "reason", "stage"}
                or attempt["status"] != "unavailable" or attempt["stage"] != "compute_admission"
                or not isinstance(attempt["reason"], str) or not attempt["reason"].strip()):
            raise ValueError("The offline seal has an invalid online admission record")
        expected["online_attempt"] = attempt
    if expected != seal:
        raise ValueError("The archived private window differs from its pre-reveal seal")


def blind_inference(config, root, manifest, seal):
    verify_offline_seal(root, manifest, seal)
    with admitted(config, "smart-swarm-physical-blind-offline", archive_only=True) as (grant, heartbeat):
        with scoped_window(root, manifest) as (scoped, digest):
            if digest != seal["permitted_input_sha256"]:
                raise ValueError("The isolated reader received a different private window")
            result = run_worker(scoped, digest)
        heartbeat()
    return {"status": "scored", "mode": "blind_offline", "receipt": result,
            "admission": {key: grant[key] for key in ("job_id", "device", "scope")}}


def verify_optional(root, manifest, decision):
    learned = decision.get("learned_evidence")
    if learned is None:
        return {"status": "historical_rule_only"}
    if learned["status"] == "scored":
        receipt = learned["receipt"]
        legacy = receipt["schema"] == LEGACY_SCHEMA
        if not legacy:
            scope = policy_manifest(manifest)
            duration = int(scope["decision_boundary"]["monotonic_ns"]) - int(scope["query_boundary"]["monotonic_ns"])
            if receipt["modeled_reply_window_ns"] != duration:
                raise ValueError("The frozen physical inference receipt names a different reply window")
        digest = window_digest(manifest, packets(root, manifest), legacy=legacy)
        verify_result(receipt, digest, schema=LEGACY_SCHEMA if legacy else SCHEMA)
    elif learned["status"] == DEFERRED:
        verify_offline_seal(root, manifest, learned)
    elif learned["status"] != "unavailable" or not learned.get("reason"):
        raise ValueError("The physical inference attempt has neither a receipt nor an explicit failure")
    return learned
