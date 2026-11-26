"""Capture paired contact-valid worlds with alternate informative camera mounts."""

import argparse
from contextlib import closing
from hashlib import sha256
from pathlib import Path
import time

from smart_swarm.cli import settings
from smart_swarm.episode import LABELS, TRUTH, Replay, atomic, load, rows, sha
from smart_swarm.rendering import configure_renderer
from smart_swarm.research_jobs import admitted

OWNERS = ("tank-1", "tank-2")
RECIPES = ((720008, .06, (.46, .10, .26)), (720013, .12, (.52, .12, .25)))
OCCLUSION_FIX = (720013, {"tank-1": .36, "tank-2": .12}, (.52, .12, .25))
CASE_LIMIT_S = 85
CASE_QUOTA_BYTES = 8 * 1024**2


def aim_mount(world, agent, station, height_shift, *, toward):
    import mujoco
    import numpy as np

    name = world.cameras[agent][agent + "/front"]
    camera = world.model.camera(name)
    original = camera.pos.copy()
    camera.pos[2] += height_shift
    mujoco.mj_forward(world.model, world.data)
    camera_id = camera.id
    location = world.data.cam_xpos[camera_id]
    forward = np.asarray(station) - location
    if not toward:
        forward[:2] *= -1
    backward = -forward / np.linalg.norm(forward)
    horizontal = np.cross((0., 0., 1.), backward)
    horizontal /= np.linalg.norm(horizontal)
    rotation = np.column_stack((horizontal, np.cross(backward, horizontal), backward))
    mount = world.data.xmat[int(world.model.cam_bodyid[camera_id])].reshape(3, 3)
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, (mount.T @ rotation).reshape(-1))
    camera.quat[:] = quaternion
    mujoco.mj_forward(world.model, world.data)
    return {"actor_camera": name, "local_position_before_m": original.tolist(), "local_position_m": camera.pos.tolist(),
            "local_quaternion_wxyz": quaternion.tolist(), "nominal_field_of_view_deg": float(camera.fovy[0])}


def capture(config, path, recipe, owner):
    from smart_swarm.evidence_portfolio import labels_for
    from smart_swarm.evidence_rollouts import prepare_world, reveal
    from smart_swarm.scaling import PassiveReceiverStory, audit_record, describe_capture

    seed, height, station = recipe
    world, variation, pickup = prepare_world(config["asset_root"], seed, "F1", variant="witnessed", seconds=CASE_LIMIT_S, team_size=4)
    with closing(world):
        started = time.monotonic()
        heights = height if isinstance(height, dict) else dict.fromkeys(OWNERS, height)
        mounts = {agent: aim_mount(world, agent, station, heights[agent], toward=agent == owner) for agent in OWNERS}
        initial = sha256(world.snapshot().integration.tobytes()).hexdigest()
        story = PassiveReceiverStory(path, "F1", world, seed, False)
        story.recorder.quota_bytes = CASE_QUOTA_BYTES
        try:
            lineage = {"source_group": f"witness-mount-development-{seed}", "split": "development"}
            describe_capture(story, variation, "rotated-witness-mounts", 4, True, lineage)
            story.scene(pickup)
            query, decision, report = story.ask()
            finish, terminal, slot = reveal(story, decision)
            story.recorder.append(LABELS, "settled-event", labels_for(story, terminal, slot))
            story.recorder.meta.update(status="valid", measured_capture_wall_s=time.monotonic() - started,
                                       experiment_role="witness-mount-development")
            story.recorder.close(query, decision, finish)
        except (Exception, KeyboardInterrupt) as error:
            story.recorder.quarantine(str(error) or type(error).__name__)
            raise
        final = sha256(world.snapshot().integration.tobytes()).hexdigest()
    return {"path": str(path), "owner": owner, "mounts": mounts, "initial_state_sha256": initial, "final_state_sha256": final,
            "world_sha256": story.recorder.meta["simulation_provenance"]["model_sha256"],
            "contact_trace_sha256": sha(path / TRUTH / "physics.jsonl"), "outcome": terminal["ground_truth"],
            "engineering_choice": report["query_all"]["decision"], "capture": audit_record(path)}


def witness_views(result):
    from smart_swarm.identity_support import marker_support
    from smart_swarm.primary_inputs import primary_policy_view

    root = Path(result["path"])
    phases = result["capture"]["observation_budget"]
    support = {}
    with primary_policy_view(root) as scoped:
        allowed = {"policy_cameras", "query_boundary", "decision_boundary", "world_episode_id", "sidecar_sha256"}
        if set(load(scoped / "manifest.json")) != allowed or (scoped / TRUTH).exists() or (scoped / LABELS).exists():
            raise ValueError("The scoped runtime received intervention metadata or evaluator observations")
        for owner in OWNERS:
            client = Replay(scoped, owner)
            pictures = {int(packet["source_time"]["monotonic_ns"]): client.image(packet["cameras"][0]["frame_id"])
                        for packet in client.packets()}
            support[owner] = {phase: marker_support(pictures[row["source_time_ns"]] for row in phases[phase]
                                                   if row["agent_id"] == owner)
                              for phase in ("before_relocation", "after_relocation")}
    local = root / "communication/local-evidence.jsonl"
    records = rows(local) if local.is_file() else []
    support["recorded_association_reports"] = [row for row in records if row["factor"] == "association"]
    return support


def verify_pair(cases):
    from smart_swarm.scaling import matching_budgets

    matching_budgets([case["capture"] for case in cases])
    fields = ("initial_state_sha256", "final_state_sha256", "contact_trace_sha256", "world_sha256", "outcome")
    if any(cases[0][field] != cases[1][field] for field in fields):
        raise ValueError("The camera intervention changed the modeled task or physical trajectory")
    for case in cases:
        views = case["rgb_support"]
        eligible = {owner for owner in OWNERS if all(row["supported"] for row in views[owner].values())}
        if eligible != {case["owner"]}:
            raise ValueError(f"The intended camera did not uniquely see both markers before and after the move: {eligible}")
        if case["engineering_choice"] != case["outcome"]:
            raise ValueError(f"The active camera did not support the correct search choice for {case['owner']}")
    return True


def previous_left_pair(source):
    from smart_swarm.scaling import audit_record

    old = load(source)
    left = old["groups"][0]
    right = old["groups"][1]
    if (old["status"] != "incomplete" or len(old["groups"]) != 2 or not left["verified"] or right["verified"]
            or left["seed"] != RECIPES[0][0] or right["seed"] != OCCLUSION_FIX[0]):
        raise ValueError("The parent receipt has no accepted left pair and rejected right-side camera case")
    for case in left["cases"]:
        path = Path(case["path"])
        if not path.resolve().is_relative_to(source.parent.resolve()) or (
                sha(path / TRUTH / "physics.jsonl") != case["contact_trace_sha256"]
                or audit_record(path) != case["capture"] or witness_views(case) != case["rgb_support"]):
            raise ValueError("The existing accepted left pair changed")
    verify_pair(left["cases"])
    return left


def run(config, root, grant, heartbeat, parent):
    prior = [previous_left_pair(parent)] if parent else []
    pending = (OCCLUSION_FIX,) if parent else RECIPES
    root.mkdir(exist_ok=False, parents=True)
    receipt = root / "mount-rehearsal.json"
    report = {"scope": "paired simulated design rehearsal", "case_limit_s": CASE_LIMIT_S, "status": "running",
              "allocated_recipes": pending, "groups": prior, "reservation": grant, "new_physical_attempts": 0}
    if parent:
        report["previous_mount_receipt_sha256"] = sha(parent)
    atomic(receipt, report)
    try:
        for recipe in pending:
            group = {"seed": recipe[0], "cases": [], "verified": False}
            report["groups"].append(group)
            for owner in OWNERS:
                heartbeat()
                case = capture(config, root / f"seed-{recipe[0]}-{owner}", recipe, owner)
                group["cases"].append({**case, "rgb_support": witness_views(case)})
                atomic(receipt, report)
            group["verified"] = verify_pair(group["cases"])
            atomic(receipt, report)
        report["status"] = "verified"
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="incomplete", reason=str(error) or type(error).__name__)
        raise
    finally:
        atomic(receipt, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revisit-right-from", type=Path, help="Reuse the verified left pair after the documented occlusion")
    args = parser.parse_args()
    config = settings()
    configure_renderer(config)
    with admitted(config, "smart-swarm-witness-mount-development") as (grant, heartbeat):
        run(config, args.output, grant, heartbeat, args.revisit_right_from)
    print(args.output / "mount-rehearsal.json")


if __name__ == "__main__":
    main()
