"""Privileged scripted arm and evaluator-only contact measurements."""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .physics import STEP, TABLE


class Teacher:
    def __init__(self, world, *, native_float32=False):
        self.world = world
        self.native_float32 = native_float32
        self.scratch = mujoco.MjData(world.model)
        self.site = world.model.site("so/gripperframe").id
        self.orientation = world.data.site(self.site).xmat.reshape(3, 3).copy()
        world.controls["teacher_command"] = world.arm_from_mj(world.pending).tolist()
        qpos = world.model.jnt_qposadr
        self.velocity = [int(world.model.jnt_dofadr[np.flatnonzero(qpos == q)[0]])
                         for q in world.arm_joints[:5]]

    def solve(self, point, grip):
        model, scratch = self.world.model, self.scratch
        scratch.qpos[:] = self.world.data.qpos
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        rotation = self.orientation
        indices = self.world.arm_joints
        limits = self.world.arm_limits[:5]
        for _ in range(80):
            mujoco.mj_forward(model, scratch)
            site = scratch.site(self.site)
            errp = point - site.xpos
            errr = Rotation.from_matrix(rotation @ site.xmat.reshape(3, 3).T).as_rotvec()
            mujoco.mj_jacSite(model, scratch, jacp, jacr, self.site)
            jac = np.concatenate([jacp[:, self.velocity], .08 * jacr[:, self.velocity]])
            error = np.concatenate([errp, .08 * errr])
            delta = jac.T @ np.linalg.solve(jac @ jac.T + np.eye(6) * 1e-5, error)
            scratch.qpos[indices[:5]] = (scratch.qpos[indices[:5]] +
                                         delta.clip(-.12, .12)).clip(*limits.T)
        result = scratch.qpos[indices].copy()
        result[-1] = grip
        return self.world.arm_from_mj(result)

    def advance(self, phase, point, grip, joint_speed, seconds, on_sample):
        goal = self.solve(np.asarray(point) + [0, 0, TABLE], grip)
        ticks = round(seconds * 30)
        step_limit = np.array([joint_speed] * 5 + [90]) / 30
        for index in range(ticks):
            leftover = self.world.controls["arm_leftover_steps"]
            count, remainder = divmod(leftover + round(1 / STEP), 30)
            self.world.controls["arm_leftover_steps"] = remainder
            command = np.asarray(self.world.controls["teacher_command"])
            asked = command + (goal - command).clip(-step_limit, step_limit)
            interval = self.world.ns
            self.world.step(count)
            applied = self.world.arm_from_mj(self.world.filtered).tolist()
            issued = asked.astype(np.float32).astype(float) if self.native_float32 else asked
            self.world.pending = self.world.arm_to_mj(issued)
            self.world.controls["teacher_command"] = asked.tolist()
            self.world.controls["teacher_chunk"] = {
                "phase": phase, "goal": goal.tolist(), "index": index + 1, "length": ticks,
                "command": asked.tolist(),
            }
            on_sample(phase, index, interval, issued.tolist(), applied)


def run_insertion(teacher, sample, *, pickup=(.24, -.06)):
    x, y = pickup
    phases = [
        ("approach", (x, y, .108), .9, 50, 2),
        ("descend", (x, y, .019), .9, 50, 2),
        ("grasp", (x, y, .019), -.12, 50, 2),
        ("lift", (x, y, .148), -.12, 50, 2),
        ("transfer", (.24, .10, .14), -.12, 20, 3),
        ("lower", (.24, .10, .069), -.12, 20, 2),
        ("release", (.24, .10, .069), .9, 50, 2),
        ("withdraw", (.24, .10, .168), .9, 50, 2),
    ]
    for name, point, grip, speed, seconds in phases:
        teacher.advance(name, point, grip, speed, seconds, sample)


def contacts(world):
    cube = world.model.geom(f"{world.truth_target}_geom").id
    names = set()
    for item in world.data.contact:
        pair = [int(item.geom1), int(item.geom2)]
        if cube in pair:
            geom_id = pair[1 - pair.index(cube)]
            body_id = int(world.model.geom_bodyid[geom_id])
            names.add(world.model.geom(geom_id).name if body_id == 0 else world.model.body(body_id).name)
    return sorted(names)


def truth(world):
    data = world.data
    cube = data.body(world.truth_target).xpos.copy()
    box = data.body("carrier")
    local = box.xmat.reshape(3, 3).T @ (cube - box.xpos)
    q = int(world.model.joint(f"{world.truth_target}_pose").dofadr[0])
    cube_speed = float(np.linalg.norm(data.qvel[q:q + 3]))
    engaged = [n for n in contacts(world) if n.startswith("so/")]
    open_gripper = world.arm_from_mj(data.qpos[world.arm_joints])[-1] > 28
    gap = float(np.linalg.norm(data.site("so/gripperframe").xpos - cube))
    inside = bool(abs(local[0]) < .034 and abs(local[1]) < .024 and .016 < local[2] < .035)
    return {"cube_xyz_m": cube.tolist(), "carrier_xyz_m": box.xpos.tolist(),
            "cube_height_above_table_m": float(cube[2] - TABLE), "cube_speed_m_s": cube_speed,
            "inside_carrier": inside, "stationary": cube_speed < .025,
            "released": bool(open_gripper and gap > .055 and not engaged),
            "gripper_contact_bodies": engaged, "contact_bodies": contacts(world),
            "cube_relation": "carrier" if inside else "surface" if "station_surface" in contacts(world)
            else "held" if engaged else "outside"}


def move_box(world, sample):
    actuator = world.model.actuator("external_drive").id
    for target, ticks in ((.233, 48), (-.025, 20)):
        for _ in range(ticks):
            world.data.ctrl[actuator] = target
            world.step(100)
            sample()
