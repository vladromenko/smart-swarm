"""Publish the precise ACT candidate for A's execution compatibility work."""

from pathlib import Path
import time
import uuid

from .contract import CONTRACT_SHA, VERSION, ProtocolError
from .episode import atomic, load, sha
from .rehearsal import OUTPUT, check_handover, local_fixture
from .so101_pilot import CAMERAS, ROOT


def manifest_for(checkpoint, trace):
    inputs = load(checkpoint / "config.json")["input_features"]
    artifact = {path.name: sha(path) for path in checkpoint.iterdir() if path.is_file()
                and path.suffix in (".json", ".safetensors")}
    camera_keys = ["observation.images.camera1", "observation.images.camera2"]
    return {"protocol_version": VERSION, "contract_sha256": CONTRACT_SHA, "checkpoint": str(checkpoint),
            "checkpoint_artifacts_sha256": artifact, "runtime": {"lerobot": "0.6.1", "device": "cpu"},
            "camera_ids": list(CAMERAS), "camera_input_keys": camera_keys, "camera_color": "RGB",
            "model_input_features": inputs, "action_schema_id": "sim-so101-absolute-v1",
            "joint_order": trace["native_joint_order"], "joint_units": trace["native_units"],
            "target_conditioning": {"kind": "fixed_single_bin", "instruction": trace["fixed_task"]},
            "processor_profile": "saved LeRobot pre/post, HWC RGB bytes to CHW float /255, no external crop",
            "executor_profile_id": "aligned-sample-v2", "sim_control_hz": 30, "max_steps": 510,
            "timeout_s": 150, "chunk_length": trace["chunk_length"], "reset_chunk_each_episode": True,
            "limits": checkpoint_limits(),
            "allowed_scope": "offline MuJoCo development only", "native_mapping_verified": False}


def checkpoint_limits():
    from .cli import settings
    from .physics import World

    world = World(settings()["asset_root"], legacy_reveal=True)
    try:
        low, high = world.arm_limits.T
        return {"native_lower": world.arm_from_mj(low).tolist(),
                "native_upper": world.arm_from_mj(high).tolist()}
    finally:
        world.close()


def payload(session, manifest):
    return {"session_id": session["session_id"], "lease_epoch": session["lease_epoch"], "agent_id": "so101-1",
            "run_mode": "fixture", "skill_id": None, "checkpoint_manifest": manifest,
            "action_schema_id": manifest["action_schema_id"], "camera_ids": list(CAMERAS),
            "target_conditioning": manifest["target_conditioning"], "executor_profile_id": manifest["executor_profile_id"],
            "timeout_s": manifest["timeout_s"], "max_steps": manifest["max_steps"]}


def attempt(client, manifest):
    client.get("/v1/capabilities")
    endpoint = "/v1/executions"
    session = client.post("/v1/sessions", {"run_mode": "fixture", "mode": "policy",
                          "resource_ids": ["fixture-host/so101-1/actuator"], "requested_ttl_ms": 10000})
    address = f"/v1/sessions/{session['session_id']}"
    try:
        client.post(address + "/arm", {"lease_epoch": session["lease_epoch"], "run_mode": "fixture",
                    "preflight_id": session["preflight_id"], "operator_intent_id": None})
        request = payload(session, manifest)
        demo = fixture_binding(client, request)
        try:
            result = client.post(endpoint, request)
            return {"request": request, "unexpected_checkpoint_result": result}
        except ProtocolError as error:
            return {"checkpoint_request": request, "checkpoint_error": {"code": error.code, "message": error.message},
                    "fixture_only_operator_intent": True, "fixture_camera_executor": demo}
    finally:
        client.delete(address)


def fixture_binding(client, request):
    body = {**request, "skill_id": "fixture-inspect", "checkpoint_manifest": None,
            "action_schema_id": "so101-six-joint-example-v1", "executor_profile_id": "fixture-executor-v1"}
    submitted = client.post("/v1/executions", body)
    address = f"/v1/executions/{submitted['execution_id']}"
    deadline = time.monotonic() + 2
    latest = client.get(address)
    while latest["status"] == "accepted" and time.monotonic() < deadline:
        time.sleep(.05)
        latest = client.get(address)
    if latest["status"] != "completed" or latest["motion_dispatched"] or tuple(latest["fixture_views_checked"]) != CAMERAS:
        raise ValueError("A's fixture camera-only execution did not pass")
    return {"request": body, "result": latest}


def publish(config):
    check_handover(config)
    diagnostic = load(OUTPUT / "act-development-diagnostic.json")
    if diagnostic["status"] != "complete":
        raise ValueError("Completed ACT development trace required for execution handoff")
    manifest = manifest_for(Path(diagnostic["checkpoint"]), diagnostic["sample_trace"])
    root = config["data_root"] / ("execution-compatibility-" + uuid.uuid4().hex[:10])
    root.mkdir()
    with local_fixture(config, root) as client:
        fixture = attempt(client, manifest)
    if fixture.get("checkpoint_error", {}).get("code") != "OFFLINE":
        raise ValueError("A's fixture recorded unexpected checkpoint compatibility")
    summary = [{"profile": case["profile"], "seed": case["seed"], "first_failed_stage": case["first_failed_stage"],
                "stable_containment": case["stable_containment"]} for case in diagnostic["cases"]]
    result = {"manifest": manifest, "A_fixture_checkpoint_request_and_result": fixture,
              "B_simulation": {"request": {"run_mode": "mujoco", "steps": 510,
                                            "camera_ids": list(CAMERAS), "action_schema_id": manifest["action_schema_id"]},
                               "result": summary, "sample_decode": diagnostic["sample_trace"]["decoded_exec"]},
              "hardware_authorization": "not established", "compatibility_work_owner": "A"}
    atomic(ROOT / "handover/act-execution-handoff.json", result)
    return result
