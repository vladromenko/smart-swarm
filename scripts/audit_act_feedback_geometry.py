#!/usr/bin/env python3
"""Reconstruct wrist projection at sampled ACT grasp commands from archived state."""

import argparse
from math import dist
from pathlib import Path

from smart_swarm.cli import ROOT, settings
from smart_swarm.episode import atomic, load, sha
from smart_swarm.rendering import configure_renderer
from smart_swarm.research_jobs import admitted

PILOT = ROOT / "evidence/learning-pilot-v3"
INPUTS = {
    "act-clock-fit.json": "1b318f30abe35c401624457b70ebc491116b210734eb85212ae4b2306e67eb72",
    "act-wrist-contact-replay.json": "99dbf3b0ee0f3d5fb032c06d3bbaf693063954c874fa1661d3363a2d404df052",
}
TICKS = (120, 135, 150)
REFERENCE_ISSUES = (126, 132)


def load_inputs():
    if any(sha(PILOT / name) != digest for name, digest in INPUTS.items()):
        raise ValueError("An original ACT feedback or teacher geometry receipt changed")
    fit, teacher = (load(PILOT / name) for name in INPUTS)
    if fit["successful"] != 0 or fit["matching_starts"] != 5:
        raise ValueError("Expected the five original non-successful development starts")
    if (teacher["first_observation_issue"], teacher["second_observation_issue"]) != REFERENCE_ISSUES:
        raise ValueError("Unexpected original wrist comparison issues")
    return fit, teacher


def checked_geometry(world, camera, native, cube, issue):
    from audit_act_wrist_contact import observe
    from audit_motor_grasp_pose import set_pose

    set_pose(world, native, cube)
    return observe(world, issue, camera)


def reference_control(world, camera, arrays, references):
    residuals = []
    for row in references:
        issue = row["issue"]
        if int(arrays["timestamps_ns"][issue]) != row["at_ns"]:
            raise ValueError("Teacher wrist pose is from a different issue time")
        restored = checked_geometry(world, camera, arrays["state"][issue], row["cube_xyz_m"], issue)
        residuals.append(dist(row["cube_camera_center_px"], restored["cube_camera_center_px"]))
    if max(residuals) > .0001:
        raise ValueError("The archived teacher wrist projection did not round-trip")
    return max(residuals)


def compare_row(world, camera, arrays, row, preceding, reference, anchors):
    from audit_act_wrist_contact import projected_cube

    issue = row["tick"] - 1
    if (row["observed_at_ns"] != int(arrays["timestamps_ns"][issue]) or row["observed_at_ns"] != reference["at_ns"]
            or row["next_observed_at_ns"] != int(arrays["timestamps_ns"][issue + 1])
            or row["teacher_requested_at_index"] != arrays["requested"][issue].tolist()):
        raise ValueError("ACT and teacher do not match at the recorded action boundary")
    cube, alternate = row["cube_xyz_m"], preceding["cube_xyz_m"]
    drift = dist(cube, alternate)
    if any(item["gripper_contact_bodies"] or item["cube_speed_m_s"] > .01 for item in (row, preceding)) or drift > .001:
        raise ValueError("Learned samples do not support the stationary-cube reconstruction")
    pose = checked_geometry(world, camera, row["previous_observed_state"], cube, issue)
    center = pose["cube_camera_center_px"]
    teacher_center = reference["cube_camera_center_px"]
    return {"pre_action_issue": issue, "pre_action_ns": row["observed_at_ns"], "cube_readback_ns": row["next_observed_at_ns"],
            "earlier_cube_readback_ns": preceding["next_observed_at_ns"], "observed_cube_drift_mm": drift * 1000.,
            "archived_ACT_grip_target_percent": row["requested"][-1], "teacher_grip_target_percent": row["teacher_requested_at_index"][-1],
            "estimated_act_center_px": center, "teacher_same_issue_center_px": teacher_center,
            "act_camera_teacher_cube_center_px": projected_cube(reference["cube_xyz_m"], pose),
            "teacher_camera_act_cube_center_px": projected_cube(cube, reference),
            "prior_cube_center_sensitivity_px": dist(center, projected_cube(alternate, pose)),
            "distance_to_teacher_early_center_px": dist(center, anchors[0]["cube_camera_center_px"]),
            "distance_to_teacher_late_center_px": dist(center, anchors[1]["cube_camera_center_px"])}


def compare_case(config, case, entry, source):
    from smart_swarm.motor_diagnostics import read_teacher, reset_world

    manifest, arrays, _ = read_teacher(Path(entry["raw_path"]))
    if (case["seed"] != manifest["seed"] or source["seed"] != manifest["seed"] or source["split"] != "train"
            or source["raw_sha256"] != manifest["sha256"] or entry["source_sha256"] != manifest["sha256"]
            or case["profile"] != "aligned-sample-v2" or case["first_failed_stage"] != "grasp"):
        raise ValueError("The archived feedback is not paired with the exact admitted teacher")
    world = reset_world(config["asset_root"], manifest)
    reference = {row["issue"]: row for row in source["contact_window"]}
    feedback = {row["tick"]: row for row in case["feedback"]}
    anchors = [reference[issue] for issue in REFERENCE_ISSUES]
    try:
        camera = world.cameras["so101-1"]["so101-1/wrist"]
        control = reference_control(world, camera, arrays, anchors)
        rows = [compare_row(world, camera, arrays, feedback[tick], feedback[tick - 15], reference[tick - 1], anchors)
                for tick in TICKS]
        if world.renderer is not None:
            raise ValueError("The reconstruction must leave the renderer unopened")
    finally:
        world.close()
    return {"seed": case["seed"], "teacher_source_sha256": manifest["sha256"], "teacher_control_max_error_px": control,
            "teacher_reference_centers_px": [row["cube_camera_center_px"] for row in anchors], "samples": rows}


def audit(config, grant, heartbeat):
    fit, teacher = load_inputs()
    lineage_path = Path(fit["dataset"]) / "train/smart-swarm-lineage.json"
    lineage = {int(row["source_world_episode_id"].rsplit("-", 1)[-1]): row for row in load(lineage_path)["episodes"]}
    replay = {row["seed"]: row for row in teacher["sources"]}
    cases = []
    for case in fit["fitting_cases"]:
        heartbeat()
        seed = case["seed"]
        cases.append(compare_case(config, case, lineage[seed], replay[seed]))
    if len(cases) != len(lineage) or sorted(case["seed"] for case in cases) != sorted(lineage):
        raise ValueError("Incomplete or duplicated development source comparison")
    return {"scope": "Evaluator reconstruction of five original ACT development failures",
            "source_receipts_sha256": INPUTS, "training_lineage_sha256": sha(lineage_path), "cases": cases,
            "teacher_reference_issues": REFERENCE_ISSUES, "sampled_feedback_ticks": TICKS,
            "reservation": {"backend": grant["backend"], "locks": grant["locks"]},
            "limits": "The wrist mount uses the exact archived pre-command joints. The cube uses the following saved readback "
                      "with a sensitivity check against the preceding saved readback. The projection estimates a geometric center "
                      "and does not establish visibility, pixel identity or a causal policy response.",
            "new_rgb_frames": 0, "new_optimizer_updates": 0, "new_learned_rollouts": 0, "fresh_sources_read": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new immutable audit result path")
    config = settings()
    configure_renderer(config)
    with admitted(config, "smart-swarm-act-feedback-geometry", archive_only=True) as (grant, heartbeat):
        report = audit(config, grant, heartbeat)
    atomic(args.output, report)
    print(args.output)


if __name__ == "__main__":
    main()
