#!/usr/bin/env python3
"""Reopen the clock-conditioned ACT deployment evidence and check Studio compatibility."""

import argparse
from hashlib import file_digest, sha256
import json
from pathlib import Path
import subprocess

from smart_swarm.arm_executor import FPS
from smart_swarm.cli import ROOT, settings
from smart_swarm.contract import CONTRACT_SHA, VERSION
from smart_swarm.episode import atomic, load, sha
from smart_swarm.motor import SENSORS, TASK
from smart_swarm.motor_diagnostics import read_teacher, reset_world
from smart_swarm.motor_input import CLOCKED_PROFILE, ISSUE_CLOCK_BITS, TARGET_RGB, conditioned_top, instruction
from smart_swarm.physics import ARM_NAMES, ARM_UNITS
from smart_swarm.studio import studio_env

FIT = ROOT / "evidence/learning-pilot-v3/act-clock-fit.json"
FIT_SHA = "1b318f30abe35c401624457b70ebc491116b210734eb85212ae4b2306e67eb72"
WEIGHTS_SHA = "aa85afccb9e8385e202b194cda0a66e269c9b54f50feba8828c3c85cbcdd433f"
CAMERA_KEYS = ("observation.images.camera1", "observation.images.camera2")
EXECUTOR = "aligned-sample-v2"
NATIVE_PROBE = """
import json
from pathlib import Path
import sys
import numpy as np
from so101_lab.features import FeatureContract
from so101_lab.inference import load_checkpoint
from so101_lab.policies import image_tensor

checkpoint, raw = (Path(arg).resolve() for arg in sys.argv[1:])
seal = checkpoint.parent / 'checkpoint.json'
if seal.exists():
    raise ValueError('The expected early read-only refusal no longer applies')
try:
    load_checkpoint(checkpoint, device='cpu')
except FileNotFoundError as error:
    if Path(error.filename) != seal:
        raise
    refusal = {'error_type': type(error).__name__, 'missing_path': str(seal)}
else:
    raise ValueError('Studio accepted an unsealed simulation checkpoint')
contract = FeatureContract(mode='rgb_state', camera_mode='two_camera')
with np.load(raw, allow_pickle=False) as data:
    image = data['top'][0]
    rgb_shape = list(image.shape)
    tensor_shape = list(image_tensor(image, size=rgb_shape[1]).shape)
print(json.dumps({'loader_refusal': refusal, 'feature_contract_sha256': contract.digest,
      'expected_named_camera_keys': ['observation.images.' + cam for cam in contract.cameras],
      'native_state_dimension': contract.state_dim, 'native_state_layout': contract.to_dict()['state_layout'],
      'archived_input_rgb_shape': rgb_shape, 'native_act_tensor_shape': tensor_shape}))
"""


def artifacts(checkpoint):
    results = {}
    for path in sorted(checkpoint.iterdir()):
        if path.is_file() and path.suffix in (".json", ".safetensors"):
            with path.open("rb") as stream:
                results[path.name] = file_digest(stream, "sha256").hexdigest()
    if results["model.safetensors"] != WEIGHTS_SHA:
        raise ValueError("The archived clock-conditioned ACT weights changed")
    return results


def saved_setup():
    if sha(FIT) != FIT_SHA:
        raise ValueError("The completed clocked ACT fitting receipt changed")
    fit = load(FIT)
    checkpoint = Path(fit["checkpoint"])
    cfg = load(checkpoint / "config.json")
    binding = load(checkpoint / "smart-swarm-input.json")
    dataset = Path(fit["dataset"])
    index = load(dataset / "train/smart-swarm-lineage.json")
    expected_images = dict(zip(("wrist", "top"), CAMERA_KEYS, strict=True))
    if (binding != {"dataset": str(dataset), "input_profile": CLOCKED_PROFILE}
            or index["allowed_images"] != expected_images or index["control_timing"] != EXECUTOR
            or index["input_profile"] != CLOCKED_PROFILE or cfg["type"] != "act"):
        raise ValueError("The checkpoint does not point to the aligned clocked input path")
    row = index["episodes"][0]
    source = Path(row["raw_path"])
    meta, arrays, frames = read_teacher(source)
    if (meta["sha256"] != row["source_sha256"] or meta["sensor_ids"] != list(SENSORS)
            or meta["sample_count"] != row["frame_count"] or meta["fps"] != FPS):
        raise ValueError("The reference request disagrees with its accepted teacher source")
    return fit, cfg, checkpoint, source, row, meta, arrays, frames


def reference_request(fit, source, row, meta, arrays, frames):
    trace = fit["validation"]["matched_sample_trace"]
    stamp = int(arrays["timestamps_ns"][0])
    if (trace["pipeline_profile"] != CLOCKED_PROFILE or trace["recorded_at_ns"] != stamp
            or trace["measured_at_ns"] != stamp or trace["recorded_native_action"] != arrays["requested"][0].tolist()
            or set(trace["preprocessing_max_abs_residual"]) != {*CAMERA_KEYS, "observation.state"}
            or any(trace["preprocessing_max_abs_residual"].values()) or not trace["task_matches"]):
        raise ValueError("The saved deployment trace does not match the recorded first issue")
    marker = row["target_marker"]
    rgb = [frames["wrist"], conditioned_top(frames["top"], marker, issue_index=0)]
    return {"run_mode": "mujoco", "type": "action", "source": str(source), "source_sha256": meta["sha256"],
            "seed": meta["seed"], "issue_index": 0, "observed_at_ns": stamp, "state": arrays["state"][0].tolist(),
            "task": instruction(TASK, marker), "requested_marker": marker, "camera_ids": list(SENSORS),
            "images": [{"sensor": sensor, "key": key, "rgb_hwc_shape": list(image.shape),
                        "raw_rgb_bytes_sha256": sha256(image.tobytes()).hexdigest()}
                       for sensor, key, image in zip(SENSORS, CAMERA_KEYS, rgb, strict=True)]}


def native_readback(config, checkpoint, source, cfg):
    command = [config["studio_python"], "-c", NATIVE_PROBE, str(checkpoint), str(source / "motor.npz")]
    workdir = config["studio_project"]
    result = subprocess.run(command, cwd=workdir, env=studio_env(config), capture_output=True, text=True, timeout=30, check=True)
    receipt = json.loads(result.stdout)
    actual_keys = set(receipt["expected_named_camera_keys"])
    policy_keys = set(CAMERA_KEYS)
    model_shape = cfg["input_features"][CAMERA_KEYS[-1]]["shape"]
    if actual_keys & policy_keys or receipt["native_act_tensor_shape"] == model_shape:
        raise ValueError("Recheck this handoff against the changed Studio interface")
    return {**receipt, "model_input_camera_shape": model_shape, "sim_checkpoint_loads_in_studio": False,
            "hardware_mapping_verified": False, "registered_or_executed_in_studio": False}


def simulation_manifest(config, checkpoint, cfg, meta, trace):
    world = reset_world(config["asset_root"], meta)
    try:
        lower, upper = [world.arm_from_mj(values).tolist() for values in world.arm_limits.T]
    finally:
        world.close()
    if trace["chunk_length"] != cfg["chunk_size"] or cfg["n_action_steps"] != 6:
        raise ValueError("The archived and deployed action chunk execution differs")
    return {"protocol_version": VERSION, "contract_sha256": CONTRACT_SHA, "checkpoint": str(checkpoint),
            "checkpoint_artifacts_sha256": artifacts(checkpoint), "runtime": {"device": "cpu", "lerobot": "0.6.1"},
            "permitted_use": "offline MuJoCo development", "camera_ids": list(SENSORS), "camera_input_keys": list(CAMERA_KEYS),
            "input_features": cfg["input_features"], "output_features": cfg["output_features"], "input_profile": CLOCKED_PROFILE,
            "target_conditioning": {"requested_marker_rgb": TARGET_RGB, "camera": SENSORS[-1],
                                    "issue_clock_bits": ISSUE_CLOCK_BITS, "instruction": TASK},
            "processors": ["policy_preprocessor.json", "policy_postprocessor.json"],
            "rgb_conversion": "uint8 HWC to scaled float32 CHW; preserve 120×160 shape",
            "executor_profile": EXECUTOR, "action_schema": meta["action_schema_id"], "sim_model_sha256": meta["model_sha256"],
            "joint_order": list(ARM_NAMES), "joint_units": list(ARM_UNITS),
            "limits": {"native_lower": lower, "native_upper": upper}, "control_hz": FPS, "max_issues": meta["sample_count"],
            "chunk_length": cfg["chunk_size"], "issued_before_refresh": cfg["n_action_steps"],
            "reset_action_queue_each_episode": True, "rollout_timeout_s": 150}


def build_receipt(config):
    fit, cfg, checkpoint, source, row, meta, arrays, frames = saved_setup()
    request = reference_request(fit, source, row, meta, arrays, frames)
    trace = fit["validation"]["matched_sample_trace"]
    manifest = simulation_manifest(config, checkpoint, cfg, meta, trace)
    cases = [{field: case[field] for field in ("seed", "first_failed_stage", "bilateral_contact", "stable_containment")}
             for case in fit["fitting_cases"]]
    if len(cases) != 5 or any(row["first_failed_stage"] != "grasp" or row["stable_containment"] for row in cases):
        raise ValueError("The saved five-case failure diagnosis changed")
    return {"schema": "smart-swarm-clocked-act-development-handoff-v1", "manifest": manifest,
            "simulation": {"saved_diagnostic_sha256": FIT_SHA, "request": request,
                           "result": {"decoder": trace["decoded_executor"], "normalization_roundtrip_max_abs": trace["normalization_roundtrip_max_abs"],
                                      "preprocessing_max_abs_residual": trace["preprocessing_max_abs_residual"],
                                      "reset_residual": trace["reset_residual"], "matched_closed_loop": cases,
                                      "closed_loop_successes": fit["successful"]}},
            "studio_read_only_compatibility": native_readback(config, checkpoint, source, cfg),
            "training": {"optimizer_updates_this_handoff": 0, "checkpoint_fit_updates": fit["cumulative_fit_updates"]},
            "physical_verification": "pending attended development and compatible native checkpoint"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Choose a new clocked ACT handoff output")
    atomic(args.out, build_receipt(settings()))
    print(args.out)


if __name__ == "__main__":
    main()
