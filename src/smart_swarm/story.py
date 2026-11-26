"""Bounded canonical/F2/F6 rehearsal in the shared contact world."""

from collections import defaultdict
from pathlib import Path
import uuid

import numpy as np

from .contract import VERSION, interval, time_at
from .episode import LABELS, TRUTH, Recorder, sha
from .memory import QUERY, WITNESS_PROCESSING_NS, WITNESS_TRANSPORT_NS, deliver, predict, serialized_cost
from .physics import ARM_NAMES, ARM_UNITS, ROS_NAMES, World
from .scene import CAMERAS, EVALUATOR
from .teacher import Teacher, move_box, run_insertion, truth
from .vision import insertion, relocation

AGENTS = ("so101-1", "tank-1", "rosmaster-1")


class Story:
    def __init__(self, path, family, world, seed, miss):
        self.world = world
        self.family = family
        self.miss = miss
        self.path = path
        self.frames = defaultdict(list)
        self.measurements = defaultdict(list)
        self.evidence = []
        self.command_count = 0
        self.relocation_ticks = 0
        self.teacher = Teacher(world)
        sensors = {agent: list(grants) for agent, grants in world.cameras.items()}
        if family == "F6":
            sensors[AGENTS[0]] = []
        meta = {"run_mode": "mujoco", "family": family, "simulation_recipe_seed": seed,
                "source_type": "scripted_simulation", "operator_intervention": [],
                "controls": "SO101 guided IK; Tank planar approach; stationary ROSMASTER",
                "external_environment_event": "actuated pusher with physical contacts",
                "proprioception": "simulated, synthetic calibration, no hardware mappings",
                "simulation_provenance": world.manifest, "action_schemas": schemas(world),
                "source_file_sha256": source_identity(),
                "withheld_cameras": list(CAMERAS[AGENTS[0]]) if family == "F6" else []}
        self.recorder = Recorder(path, path.name, f"scene-development-{seed}", sensors, meta)

    def capture(self, agent):
        views = self.world.views(agent) if self.recorder.cameras[agent] else {}
        state, units = self.world.state(agent)
        ref = f"arm-application-{self.command_count:04}" if agent == AGENTS[0] and self.command_count else None
        packet = self.recorder.capture(agent, views, self.world.ns, state, list(units), ref)
        self.frames[agent].append(packet)
        return packet, views

    def approach(self):
        tank = self.world.tank_joints[0]
        origin = float(self.world.data.qpos[tank])
        self.world.tank_target[:] = [.10, 0]
        self.world.step(1000)
        self.world.tank_target[:] = 0
        self.world.step(1000)
        travel = float(self.world.data.qpos[tank]) - origin
        self.recorder.meta["tank_travel_m"] = travel

    def arm_sample(self, phase, index, start, requested, applied):
        self.command_count += 1
        self.recorder.append(LABELS, "arm-control", {
            "command_id": f"arm-{self.command_count:04}",
            "application_id": f"arm-application-{self.command_count:04}",
            "action_schema_id": "sim-so101-absolute-v1",
            "intent": "attempt_cube_placement",
            "requested_values": requested, "adapter_values": self.world.arm_from_mj(
                self.world.arm_to_mj(requested)).tolist(),
            "native_applied_values": applied, "native_applied_valid": True,
            "native_applied_reason": "MuJoCo filtered targets written during preceding control interval",
            "native_applied_command_id": f"arm-{self.command_count - 1:04}" if self.command_count > 1 else None,
            "hardware_acknowledgment": None, "task_outcome": "unknown",
            "next_requested_time": time_at(self.world.ns),
            "dispatch_interval": interval(start, self.world.ns),
            "current_measured_state": self.world.state(AGENTS[0])[0],
        })
        if index % 10 == 9:
            sample = truth(self.world)
            self.measurements[phase].append(sample)
            self.recorder.append(TRUTH, "physics", {"at": time_at(self.world.ns),
                                                     "teacher_phase": phase, **sample})
            if self.family != "F6":
                self.capture(AGENTS[0])

    def scene(self):
        self.approach()
        self.capture(AGENTS[0])
        run_insertion(self.teacher, self.arm_sample)
        self.world.step(200)
        after_insert = truth(self.world)
        if self.family != "F6":
            packet, pictures = self.capture(AGENTS[0])
            self.add_evidence(insertion(packet, pictures["so101-1/top"]))
        before, views = self.capture(AGENTS[1])
        move_box(self.world, self.motion_sample)
        after, final_views = self.capture(AGENTS[1])
        self.add_evidence(relocation([before, after], views["tank-1/front"],
                                     final_views["tank-1/front"]))
        self.capture(AGENTS[2])
        self.recorder.meta["physics_check"] = physics_check(self, after_insert)

    def motion_sample(self):
        self.relocation_ticks += 1
        if self.relocation_ticks % 7 == 0:
            self.capture(AGENTS[1])
            self.recorder.append(TRUTH, "physics", {"at": time_at(self.world.ns),
                                                     "teacher_phase": "relocation", **truth(self.world)})

    def add_evidence(self, item):
        if item is not None:
            self.evidence.append(item)
            self.recorder.append("communication", "local-evidence", item.as_record())

    def ask(self):
        query = self.world.ns
        request = {"origin_agent": AGENTS[2], "recipients": list(self.witnesses()), "text": QUERY,
                   "issued_time": time_at(query)}
        self.recorder.append("communication", "queries", request)
        self.world.queue.extend(deliver(item, query + WITNESS_PROCESSING_NS, self.transport_delay(item.origin_agent))
                                for item in self.evidence)
        self.world.step(180)
        decision = self.world.ns
        available = [msg for msg in self.world.queue if int(msg["delivered_ns"]) <= decision]
        for message in available:
            self.recorder.append("communication", "deliveries", message)
        result = predict(available, decision)
        baseline = predict([], decision)
        record = {"at": time_at(decision), "query": QUERY, "agent_id": AGENTS[2],
                  "current_observation_only": baseline, "query_all": result,
                  "cost": serialized_cost(request, available)}
        self.recorder.append("decisions", "prediction", record)
        return query, decision, record

    def witnesses(self):
        return AGENTS[:2]

    def transport_delay(self, agent):
        return WITNESS_TRANSPORT_NS[agent]

    def reveal(self, decision):
        prior = truth(self.world)
        self.world.step(200)
        reveal = self.world.ns
        after = truth(self.world)
        preserved = prior["cube_relation"] == after["cube_relation"]
        camera = self.recorder.frame(TRUTH, EVALUATOR, 1, self.world.image("reveal"), reveal)
        expected = "left" if prior["inside_carrier"] else prior["cube_relation"]
        row = {"camera": camera, "ground_truth": expected, "relation": after["cube_relation"],
               "target_state_time_or_interval": interval(decision, reveal),
               "relations_preserved_until_reveal": preserved, "observed_at": time_at(reveal)}
        self.recorder.append(TRUTH, "reveal", row)
        return reveal, row


def schemas(world):
    arm_bounds = np.stack([world.arm_from_mj(world.arm_limits[:, i]) for i in range(2)]).T
    return {
        "sim-so101-absolute-v1": schema(ARM_NAMES, ARM_UNITS, arm_bounds.tolist(),
                                      "simulated_joint_frame", "absolute_position", 1000/30),
        "sim-tank-base-v1": schema(["forward", "yaw_rate"], ["m/s", "rad/s"],
                                   [[-.2, .2], [-.5, .5]], "body", "velocity", 50),
        "sim-rosmaster-stationary-arm-v1": schema(ROS_NAMES, ["deg"]*6, np.rad2deg(
            world.model.actuator_ctrlrange[world.ros_actuators]).tolist(),
            "manufacturer_joint_coordinates_in_simulation", "absolute_position", 1000/30),
    }


def schema(names, units, limits, frame, semantics, period):
    return {"names": list(names), "units": list(units), "limits": limits, "frame": frame,
            "semantics": semantics, "control_period_ms": period}


def physics_check(story, inserted):
    world = story.world
    final = truth(world)
    lift = story.measurements["lift"]
    carry = story.measurements["transfer"] + story.measurements["lower"]
    displacement = final["carrier_xyz_m"][0] - inserted["carrier_xyz_m"][0]
    grasped = any(len(row["gripper_contact_bodies"]) >= 2 for row in lift)
    lifted = max(row["cube_height_above_table_m"] for row in lift)
    retained = all(len(row["gripper_contact_bodies"]) >= 2 for row in carry)
    motions_valid = displacement > .12 and story.recorder.meta["tank_travel_m"] > .02
    success = (grasped and lifted >= .10 and retained and inserted["released"]
               and inserted["inside_carrier"] and inserted["stationary"]
               and final["inside_carrier"] and final["stationary"])
    failure = not grasped and not inserted["inside_carrier"] and final["cube_relation"] == "surface"
    if not motions_valid or (story.miss and not failure) or (not story.miss and not success):
        raise ValueError(f"Unverified physics: {final}, grasped={grasped}, retained={retained}")
    return {"bilateral_grasp": grasped, "peak_sampled_lift_m": lifted, "carry_retained": retained,
            "gripper_open_and_clear": inserted["released"], "insertion_stationary": inserted["stationary"],
            "contained_before_relocation": inserted["inside_carrier"],
            "contained_after_relocation": final["inside_carrier"],
            "box_displacement_m": displacement, "open_bin": True, "physics_steps": world.physics_steps,
            "final_cube_relation": final["cube_relation"]}


def snapshot_check(world):
    saved = world.snapshot()
    uniform = world.rng.uniform()
    world.step(80)
    pose = world.data.qpos.copy()
    world.restore(saved)
    world.step(80)
    delta = float(np.max(np.abs(pose - world.data.qpos)))
    if delta > 1e-10 or world.rng.uniform() != uniform:
        raise ValueError("Matched snapshot branches diverged")
    world.restore(saved)
    return {"max_qpos_delta": delta, "matched_steps": 80, "rng_restored": True}


def run_story(assets: Path, parent: Path, family: str, seed: int, *, swap=False, f6_miss=False):
    if family not in ("F1", "F2", "F6") or f6_miss and family != "F6":
        raise ValueError("Unsupported B1 story recipe")
    miss = family == "F2" or f6_miss
    path = parent / ("world-" + uuid.uuid4().hex)
    target = ("surface_cube" if f6_miss else "cube") if family == "F6" else None
    world = World(assets, miss=family == "F2", seed=seed, swap_markers=swap, f6_target=target)
    story = Story(path, family, world, seed, miss)
    try:
        story.recorder.meta["snapshot_check"] = snapshot_check(world)
        story.scene()
        query, decision, result = story.ask()
        reveal, withheld = story.reveal(decision)
        expected = "request_inspection" if family == "F6" else withheld["ground_truth"]
        if result["query_all"]["decision"] != expected:
            raise ValueError(f"Unexpected image-grounded decision, see {path}/decisions")
        story.recorder.meta["status"] = "valid"
        story.recorder.close(query, decision, reveal)
    except (Exception, KeyboardInterrupt) as error:
        story.recorder.quarantine(str(error) or type(error).__name__)
        raise
    finally:
        world.close()
    return path

def source_identity():
    root = Path(__file__).parent
    names = ("scene", "physics", "teacher", "story", "vision", "memory", "episode", "contract")
    return {f"{name}.py": sha(root / f"{name}.py") for name in names}
