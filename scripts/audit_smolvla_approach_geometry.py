"""Replay validation teachers and isolate shoulder pan in archived command targets."""

import argparse
from hashlib import sha256
import math
import os
from pathlib import Path
from statistics import median
import sys
from time import monotonic

from audit_smolvla_approach import matched_arrays
from smart_swarm.cli import ROOT, settings
from smart_swarm.episode import atomic, json_bytes, load, sha
from smart_swarm.rendering import configure_renderer
from smart_swarm.research_jobs import admitted, execute

APPROACH = ROOT / "evidence/learning-pilot-v3/smolvla-validation-approach-complete.json"
APPROACH_SHA = "a214dc7c08eb601d58df1aca0cc1e650541057ff825ae12157e140ea5665e512"
VALIDATION_SHA = "a53682142201bc5994fef7ecf19d9a22e50e325c7c1929d442ab1d2b4e393c8a"
ISSUES = (14, 29, 44, 59)
PROFILE = "aligned-sample-v2"
STATE_TOLERANCE = 1e-4
WALL_LIMIT_S = 180


def bound_material():
    if sha(APPROACH) != APPROACH_SHA:
        raise ValueError("The archived validation command audit changed")
    evidence = load(APPROACH)
    checkpoint = Path(evidence["checkpoint"])
    dataset = Path(load(checkpoint / "smart-swarm-input.json")["dataset"])
    validation = Path(evidence["validation_receipt"])
    if sha(validation) != VALIDATION_SHA:
        raise ValueError("The original checkpoint validation receipt changed")
    receipt = load(validation)
    lineage = dataset / "validation/smart-swarm-lineage.json"
    sources, cases = load(lineage)["episodes"], receipt["resets"]
    indices = list(range(20))
    if (len(cases) != len(sources) or len(cases) != len(evidence["case_rows"]) or len(cases) != len(indices)
            or receipt["evaluation_role"] != "validation" or receipt["policy"] != "smolvla"
            or receipt["action_transport"] != "verified-smolvla-chunk"
            or validation != checkpoint.parents[3] / "validation" / checkpoint.parent.name / "validation.json"
            or sha(checkpoint / "config.json") != evidence["checkpoint_config_sha256"]
            or [row["dataset_episode_index"] for row in sources] != indices
            or len({row["scene_group_id"] for row in sources}) != len(indices)
            or any(row["source_split"] != "validation" or row["frame_count"] != 510 for row in sources)):
        raise ValueError("The approach audit requires the original held-out twenty-source validation split")
    bindings = {"approach_audit_sha256": APPROACH_SHA, "validation_sha256": VALIDATION_SHA,
                "dataset_lineage_sha256": sha(lineage), "checkpoint_config_sha256": evidence["checkpoint_config_sha256"]}
    return bindings, list(zip(cases, sources, evidence["case_rows"], strict=True))


def command_points(world, saved, teacher):
    import mujoco
    import numpy as np

    decoded = world.arm_from_mj(world.arm_to_mj(saved["requested"]))
    if not np.allclose(decoded, saved["decoded_and_clipped"], rtol=0, atol=1e-8):
        raise ValueError("The archived learned target no longer matches native conversion")
    scratch = mujoco.MjData(world.model)
    scratch.qpos[:] = world.data.qpos
    command = {"teacher": teacher, "policy": decoded, "policy_with_teacher_pan": [teacher[0], *decoded[1:]],
               "teacher_with_policy_pan": [decoded[0], *teacher[1:]]}
    xyz = {}
    for kind, targets in command.items():
        scratch.qpos[world.arm_joints] = world.arm_to_mj(targets)
        mujoco.mj_kinematics(world.model, scratch)
        xyz[kind] = scratch.site("so/gripperframe").xpos.tolist()
    return {"teacher_pan_deg": float(teacher[0]), "policy_pan_deg": float(decoded[0]),
            "target_xyz_m": xyz, "target_error_m": {kind: math.dist(point, xyz["teacher"]) for kind, point in xyz.items()}}


def verify_case(assets, validation, lineage, archived):
    from smart_swarm.motor_diagnostics import replay, reset_world
    from smart_swarm.physics import ARM_NAMES, ARM_UNITS

    digest = sha256(json_bytes(validation)).hexdigest()
    manifest, indices, vectors = matched_arrays(validation, lineage)
    if (digest != archived["feedback_row_sha256"] or lineage["target_marker"] != archived["marker"]
            or manifest["sha256"] != archived["teacher_artifact_sha256"] or tuple(indices[:4]) != ISSUES
            or tuple(manifest["names"]) != ARM_NAMES or tuple(manifest["units"]) != ARM_UNITS):
        raise ValueError("Saved validation feedback, teacher action order or source binding changed")
    world = reset_world(assets, manifest)
    try:
        replays = replay(world, vectors, PROFILE)
        errors = [{"issue": issue, **command_points(world, observation, vectors["requested"][issue])}
                  for issue, observation in zip(ISSUES, validation["feedback"][:4], strict=True)]
    finally:
        world.close()
    rows = ("first_failed_stage", "stable_containment", "bilateral_contact", "held_lift_10cm", "carry_bilateral_samples",
            "carry_sample_count", "precontact_state_max_abs_residual", "sample_times_exact", "sampled_state_max_abs_residual")
    return {"seed": validation["seed"], "marker": lineage["target_marker"], "teacher_sha256": manifest["sha256"],
            "feedback_row_sha256": digest, "learned_first_failed_stage": validation["first_failed_stage"],
            "teacher_replay": {key: replays[key] for key in rows}, "samples": errors}


def summarize(results):
    replays = [case["teacher_replay"] for case in results]
    aligned = [row for row in replays if row["sample_times_exact"]
               and row["precontact_state_max_abs_residual"] < STATE_TOLERANCE]
    valid = [row for row in replays if row["stable_containment"] and row["held_lift_10cm"]
             and row["first_failed_stage"] is None and row["carry_bilateral_samples"] == row["carry_sample_count"] == 7]
    samples = [row["target_error_m"] for case in results for row in case["samples"]]
    methods = ("policy", "policy_with_teacher_pan", "teacher_with_policy_pan")
    return {"teacher_replays": len(replays), "clock_and_precontact_state_aligned": len(aligned),
            "contact_valid_placements": len(valid), "learned_approach_failures": sum(
                row["learned_first_failed_stage"] == "approach" for row in results),
            "tested_static_commands": len(samples),
            "median_teacher_target_error_mm": {key: 1000 * median(row[key] for row in samples) for key in methods},
            "pan_replacement_reduced_target_error": sum(row["policy_with_teacher_pan"] < row["policy"] for row in samples),
            "maximum_precontact_teacher_state_error_native_units": max(row["precontact_state_max_abs_residual"] for row in replays)}


def worker(config, directory):
    configure_renderer(config)
    binding, cases = bound_material()
    results = []
    for validation, lineage, archived in cases:
        results.append(verify_case(config["asset_root"], validation, lineage, archived))
        atomic(directory / f"validation-{validation['seed']}.json", results[-1])
        print(f"Audited validation {validation['seed']}", flush=True)
    atomic(directory / "report.json", {"bindings": binding, "cases": results, "summary": summarize(results)})


def run(config, output):
    directory = output.with_name(output.stem + "-job")
    directory.mkdir(exist_ok=False)
    job = "smart-swarm-smolvla-validation-approach-geometry"
    design = {"scope": "Original twenty validation sources and their first four logged approach issue targets",
              "hypothesis": "Check that recorded teacher commands succeed and isolate shoulder pan from other joint errors",
              "static_interventions": ["policy_with_teacher_pan", "teacher_with_policy_pan"], "issues": list(ISSUES),
              "wall_limit_s": WALL_LIMIT_S, "optimizer_updates": 0, "model_inference": False}
    atomic(directory / "design.json", design)
    with admitted(config, job, archive_only=True) as (grant, heartbeat):
        start = monotonic()
        argv = [sys.executable, __file__, "--worker", str(directory)]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
        with (directory / "worker.log").open("w") as log:
            code = execute(argv, env, log, start + WALL_LIMIT_S, heartbeat)
        if code:
            raise RuntimeError(f"Validation replay failed. See {directory / 'worker.log'}")
        result = load(directory / "report.json")
        atomic(output, {**design, **result, "reservation": grant, "elapsed_s": monotonic() - start,
                        "limits": "Static end-effector targets omit motion, contact, feedback and learned rollouts. "
                                  "Teacher replay measures the existing complete action trajectories."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--worker", type=Path)
    mode.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = settings()
    if args.worker:
        worker(config, args.worker)
    else:
        if args.output.exists():
            parser.error("Choose a new immutable audit result path")
        run(config, args.output.resolve())
        print(args.output)


if __name__ == "__main__":
    main()
