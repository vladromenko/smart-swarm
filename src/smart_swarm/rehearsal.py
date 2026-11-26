"""Orchestrate the real A fixture capture and export adapters in an isolated run."""

from contextlib import contextmanager, suppress
import os
from pathlib import Path
import secrets
import subprocess
import time
from urllib.error import URLError
import uuid

from .contract import CONTRACT_SHA, VERSION, ProtocolError
from .episode import atomic, sha
from .platform_client import PlatformClient, reservation
from .so101_pilot import CAMERAS, consume, count_attempts, prepare

OUTPUT = Path(__file__).resolve().parents[2] / "evidence/resume"


def check_handover(config):
    root = config["platform_handover"].resolve()
    for line in (root / "SHA256SUMS").read_text().splitlines():
        checksum, name = line.split(maxsplit=1)
        path = (root / name).resolve()
        if not path.is_relative_to(root) or sha(path) != checksum:
            raise ValueError(f"A handover checksum failed for {name}")
    if sha(root / "03_SHARED_CONTRACT_V1_EN.txt") != CONTRACT_SHA:
        raise ValueError("A's frozen contract differs")
    return root


@contextmanager
def local_fixture(config, run):
    platform = config["platform_project"]
    port = config["platform_fixture_port"]
    argv = [config["platform_python"], "-m", "robot_platform.service", "--port", str(port),
            "--fixture", str(config["platform_handover"] / "fixtures/v1"),
            "--recording-root", str(run / "recordings")]
    token = secrets.token_hex(24)
    environment = {**os.environ, "PYTHONPATH": str(platform / "src"),
                   "PYTHONDONTWRITEBYTECODE": "1", "ROBOT_PLATFORM_RECORDER_TOKEN": token}
    client = PlatformClient(f"http://127.0.0.1:{port}", token)
    with (run / "platform.log").open("w") as log:
        proc = subprocess.Popen(argv, cwd=platform, env=environment, stdout=log, stderr=subprocess.STDOUT)
        try:
            wait_health(client, proc)
            yield client
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)


def wait_health(client, proc):
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline and proc.poll() is None:
        with suppress(URLError, ConnectionError):
            health = client.get("/v1/health")
            if health["contract_sha256"] != CONTRACT_SHA or health["backends"]["hardware"] != "unverified":
                raise ValueError("Unexpected local fixture health or contract")
            return
        time.sleep(.15)
    raise RuntimeError("A fixture service failed to start, see platform.log")


def observer(client):
    caps = client.get("/v1/capabilities")
    by_sensor = {row["sensor_id"]: row for row in caps["cameras"]}
    sensors = (*CAMERAS, "evaluator/overhead")
    grant = client.post("/v1/sessions", {"run_mode": "fixture", "mode": "observe",
                        "resource_ids": [by_sensor[name]["resource_id"] for name in sensors],
                        "requested_ttl_ms": 10000})
    streams = [{"agent_id": "so101-1" if name in CAMERAS else "evaluator", "sensor_id": name,
                "required": name in CAMERAS, "access_role": by_sensor[name]["access_role"]}
               for name in sensors]
    return grant["session_id"], streams


def capture(client, card, item, outcome):
    session, streams = observer(client)
    world = item["source_world_episode_id"]
    admit = {"run_mode": "fixture", "recording_session_id": session, "world_episode_id": world,
             "scene_group_id": item["scene_group_id"], "task_id": card["task_id"],
             "output_root_id": "robot-platform-fixtures", "streams": streams,
             "lineage": {"source_world_ids": ["fixture-world-1"], "split": "development",
                         "pilot_attempt": item["attempt"], "research_group": item["scene_group_id"]},
             "research_boundaries": {"placement_intent": card["placement_instruction"]}}
    try:
        begin = client.post("/v1/episodes", admit)
        recording = f"/v1/episodes/{begin['recording_id']}"
        if outcome is None:
            client.post(recording + "/abort", {"reason": "rehearsed_operator_interruption", "preserve": True})
        else:
            client.post(recording + "/finalize", {"world_episode_id": world, "task_outcome_label": outcome})
        return client.get(recording)
    finally:
        client.delete(f"/v1/sessions/{session}")


def export_command(config, source, output):
    platform = config["platform_project"]
    argv = [config["platform_python"], "-m", "robot_platform.recording.export", str(source), str(output)]
    for camera in CAMERAS:
        argv.extend(["--camera", camera])
    env = {**os.environ, "PYTHONPATH": str(platform / "src"), "PYTHONDONTWRITEBYTECODE": "1",
           "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "CUDA_VISIBLE_DEVICES": "",
           "OMP_NUM_THREADS": "1"}
    return subprocess.run(argv, env=env, text=True, capture_output=True, timeout=60)


def fixture_attempt(config, client, run, card, item, outcome):
    manifest = capture(client, card, item, outcome)
    identifier = manifest["recording_id"]
    complete = manifest["recording_integrity"] == "complete"
    bucket = "complete" if complete else "quarantine"
    recording = run / "recordings" / bucket / identifier
    export = run / "exports" / identifier if complete else None
    if export:
        result = export_command(config, recording, export)
        (run / f"export-{identifier}.log").write_text(result.stdout + result.stderr)
        result.check_returncode()
    return consume(config, recording, export, run / f"attempt-{item['attempt']}.json")


def compute_probe(client):
    job = "smart-swarm-compatibility"
    with reservation(client, job, host="fixture-host", ttl=30, memory_mb=512) as (grant, heartbeat):
        beat = heartbeat()
    try:
        with reservation(client, job, ttl=30, memory_mb=512):
            raise RuntimeError("Unexpected production entitlement from fixture")
    except ProtocolError as error:
        return {"fixture_grant": grant, "fixture_heartbeat": beat, "production_denial": str(error)}


def run_rehearsal(config):
    handover = check_handover(config)
    base = config["data_root"] / "so101-pilot"
    work = base / ("rehearsal-" + uuid.uuid4().hex[:12])
    work.mkdir(parents=True)
    attended = prepare(config)
    group = work.name
    items = [{"attempt": i, "source_world_episode_id": f"{group}-{i:02}", "scene_group_id": group}
             for i in range(1, 4)]
    results = []
    with local_fixture(config, work) as client:
        for item, label in zip(items, ("placement_failed", "unknown", None), strict=True):
            results.append(fixture_attempt(config, client, work, attended, item, label))
        compute = compute_probe(client)
    result = {"protocol_version": VERSION, "contract_sha256": CONTRACT_SHA, "A_handover": str(handover),
              "A_fixture_export_to_B": "passed", "physical_capture_and_export": "unverified",
              "source_group_count": 1, "counts": count_attempts(results), "attempts": results,
              "compute_compatibility": compute, "run_directory": str(work)}
    atomic(work / "report.json", result)
    OUTPUT.mkdir(exist_ok=True, parents=True)
    atomic(OUTPUT / "platform-rehearsal.json", result)
    atomic(config["data_root"] / "so101-pilot-status.json", {"project": "smart-swarm", "state": "blocked",
           "phase": "fixture_rehearsal_passed", "physical_preflight": "pending", "attempt_limit": 4,
           "counts": count_attempts([]), "fixture_counts": result["counts"]})
    atomic(OUTPUT / "so101-attended-card.json", prepare(config))
    return result
