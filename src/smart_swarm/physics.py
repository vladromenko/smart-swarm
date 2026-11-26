"""Trusted simulation and controller state. Policy clients receive only packets."""

from copy import deepcopy
from dataclasses import dataclass
import time

import mujoco
import numpy as np

from .scene import CAMERAS, STEP, TABLE, build

ARM_HOME = [-11.00079, -15.01149, 14.49583, 64.74423, 4.54356, 50.76047]
ARM_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
ARM_UNITS = ("deg",) * 5 + ("calibrated_percent",)
ROS_NAMES = ("arm1_Joint", "arm2_Joint", "arm3_Joint", "arm4_Joiint", "arm5_Joint", "rlink1_Joint")
ROS_ACTUATORS = tuple(f"motor{i}" for i in range(1, 7))
FULL_STATE = mujoco.mjtState.mjSTATE_INTEGRATION
FRAME_WIDTH, FRAME_HEIGHT = 320, 240


@dataclass
class Snapshot:
    integration: np.ndarray
    queue: list
    pending: np.ndarray
    filtered: np.ndarray
    tank_target: np.ndarray
    camera_pose: tuple
    rng: dict
    controls: dict
    physics_steps: int


class World:
    def __init__(self, assets, *, miss=False, seed=3, swap_markers=False, timeout_s=120,
                 frame_size=(FRAME_WIDTH, FRAME_HEIGHT), f6_target=None, legacy_reveal=False,
                 evidence_family=None, evidence_reverse_slots=False, team_size=3):
        self.model, self.manifest = build(assets, swap_markers, f6_target=f6_target,
                                          legacy_reveal=legacy_reveal, evidence_family=evidence_family,
                                          evidence_reverse_slots=evidence_reverse_slots, team_size=team_size)
        self.parked_observers = self.manifest.get("parked_observers", {})
        self.cameras = {**CAMERAS, **{agent: {agent + "/front": details["camera"]}
                                     for agent, details in self.parked_observers.items()}}
        self.truth_target = f6_target or "cube"
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self.queue = []
        self.controls = {"arm_leftover_steps": 0, "last_applied": None, "teacher_chunk": {}}
        self.physics_steps = 0
        self.frame_size = frame_size
        self.tank_dofs = self.joints("tank", ("x", "y", "yaw"), velocity=True)
        self.tank_mass = self.manifest["assets"]["tank"]["planar_mass_kg"]
        self.deadline = time.monotonic() + timeout_s
        self.arm_joints = self.joints("so", ARM_NAMES)
        self.ros_joints = self.joints("ros", ROS_NAMES)
        self.tank_joints = self.joints("tank", ("x", "y", "yaw"))
        self.arm_actuators = self.actuators("so", ARM_NAMES)
        self.ros_actuators = self.actuators("ros", ROS_ACTUATORS)
        self.tank_actuators = self.actuators("tank", ("x", "y", "yaw"))
        self.arm_limits = self.model.actuator_ctrlrange[self.arm_actuators]
        self.pending = self.arm_to_mj(ARM_HOME)
        self.filtered = self.pending.copy()
        self.tank_target = np.zeros(2)
        self.reset_pose(miss)
        self.renderer = None

    def reset_pose(self, miss):
        self.data.qpos[self.arm_joints] = self.pending
        self.data.ctrl[self.arm_actuators] = self.pending
        self.hold_ros_arm(self.ros_joints, self.ros_actuators)
        for observer in self.parked_observers.values():
            if observer["family"] == "rosmaster":
                namespace = observer["namespace"]
                self.hold_ros_arm(self.joints(namespace, ROS_NAMES), self.actuators(namespace, ROS_ACTUATORS))
        self.data.qpos[self.tank_joints[1]] = .055
        if miss:
            q = int(self.model.joint("cube_pose").qposadr[0])
            self.data.qpos[q:q + 3] = [.12, -.155, TABLE + .018]
        mujoco.mj_forward(self.model, self.data)

    def hold_ros_arm(self, joints, actuators):
        limits = self.model.actuator_ctrlrange[actuators]
        pose = np.deg2rad([0, 40, -85, -55, 0, 15]).clip(*limits.T)
        self.data.qpos[joints] = pose
        self.data.ctrl[actuators] = pose

    def joints(self, namespace, names, *, velocity=False):
        address = "dofadr" if velocity else "qposadr"
        return np.array([int(getattr(self.model.joint(f"{namespace}/{name}"), address)[0])
                         for name in names])

    def actuators(self, namespace, names):
        return np.array([self.model.actuator(f"{namespace}/{name}").id for name in names])

    def arm_from_mj(self, values):
        result = np.rad2deg(values)
        closed, opened = self.arm_limits[-1]
        result[-1] = (values[-1] - closed) * 100 / (opened - closed)
        return result

    def arm_to_mj(self, values):
        result = np.deg2rad(values)
        closed, opened = self.arm_limits[-1]
        result[-1] = closed + values[-1] / 100 * (opened - closed)
        return result.clip(*self.arm_limits.T)

    def apply_step_controls(self):
        self.filtered += STEP / .04 * (self.pending - self.filtered)
        self.data.ctrl[self.arm_actuators] = self.filtered
        self.tank_control()

    def check_finite_position(self):
        if not np.isfinite(self.data.qpos).all():
            raise RuntimeError("Non-finite MuJoCo dynamics")

    def step(self, count):
        if not 0 < count <= 2000 or time.monotonic() > self.deadline:
            raise TimeoutError("Simulated step or wall-time cap reached")
        for _ in range(count):
            self.apply_step_controls()
            mujoco.mj_step(self.model, self.data)
        self.physics_steps += count
        mujoco.mj_forward(self.model, self.data)
        self.check_finite_position()

    def step_observed(self, count, observe):
        if not 0 < count <= 2000:
            raise ValueError("An observed simulation block must contain one to 2,000 ticks")
        if (self.model.opt.integrator != mujoco.mjtIntegrator.mjINT_IMPLICITFAST
                or mujoco.get_mjcb_control() is not None):
            raise ValueError("Observed steps require the compiled implicit-fast solver and direct controls")
        for index in range(count):
            if time.monotonic() > self.deadline:
                raise TimeoutError("Simulated wall-time cap reached")
            # Step 1 exposes the current contact geometry before another integration.
            if index:
                self.check_finite_position()
            mujoco.mj_step1(self.model, self.data)
            if index:
                observe()
            self.apply_step_controls()
            mujoco.mj_step2(self.model, self.data)
            self.physics_steps += 1
        mujoco.mj_forward(self.model, self.data)
        self.check_finite_position()
        observe()

    def tank_control(self):
        yaw = self.data.qpos[self.tank_joints[-1]]
        forward, turn = self.tank_target
        target = np.array([np.cos(yaw)*forward, np.sin(yaw)*forward, turn]) * .85
        gain = self.tank_mass / .12
        force = np.array([gain, gain, 1.2]) * (target - self.data.qvel[self.tank_dofs])
        limits = self.model.actuator_ctrlrange[self.tank_actuators]
        self.data.ctrl[self.tank_actuators] = force.clip(*limits.T)

    def tank_velocity(self):
        vx, vy, turn = self.data.qvel[self.tank_dofs]
        yaw = self.data.qpos[self.tank_joints[-1]]
        return [float(vx*np.cos(yaw) + vy*np.sin(yaw)), float(turn)]

    def state(self, agent):
        if agent in self.parked_observers:
            return self.parked_state(agent)
        states = {
            "so101-1": (lambda: self.arm_from_mj(self.data.qpos[self.arm_joints]).tolist(), ARM_UNITS),
            "tank-1": (self.tank_velocity, ("m/s", "rad/s")),
            "rosmaster-1": (lambda: np.rad2deg(self.data.qpos[self.ros_joints]).tolist(), ("deg",)*6),
        }
        observe, units = states[agent]
        return observe(), units

    def parked_state(self, agent):
        details = self.parked_observers[agent]
        namespace = details["namespace"]
        if details["family"] == "rosmaster":
            return np.rad2deg(self.data.qpos[self.joints(namespace, ROS_NAMES)]).tolist(), ("deg",) * 6
        fields = ("x", "y", "yaw")
        vx, vy, turn = self.data.qvel[self.joints(namespace, fields, velocity=True)]
        yaw = self.data.qpos[self.joints(namespace, fields)[-1]]
        return [float(vx * np.cos(yaw) + vy * np.sin(yaw)), float(turn)], ("m/s", "rad/s")

    def image(self, camera):
        if self.renderer is None:
            width, height = self.frame_size
            self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render().copy()

    def views(self, agent):
        return {sensor: self.image(camera) for sensor, camera in self.cameras[agent].items()}

    def snapshot(self):
        state = np.empty(mujoco.mj_stateSize(self.model, FULL_STATE))
        mujoco.mj_getState(self.model, self.data, state, FULL_STATE)
        pose = tuple(getattr(self.model, field).copy()
                     for field in ("cam_pos", "cam_quat", "cam_fovy", "cam_intrinsic"))
        return Snapshot(state, deepcopy(self.queue), self.pending.copy(), self.filtered.copy(),
                        self.tank_target.copy(), pose, deepcopy(self.rng.bit_generator.state),
                        deepcopy(self.controls), self.physics_steps)

    def restore(self, snapshot):
        mujoco.mj_setState(self.model, self.data, snapshot.integration, FULL_STATE)
        for field, part in zip(("cam_pos", "cam_quat", "cam_fovy", "cam_intrinsic"),
                               snapshot.camera_pose, strict=True):
            getattr(self.model, field)[:] = part
        self.queue = deepcopy(snapshot.queue)
        self.pending = snapshot.pending.copy()
        self.filtered = snapshot.filtered.copy()
        self.tank_target = snapshot.tank_target.copy()
        self.rng.bit_generator.state = deepcopy(snapshot.rng)
        self.controls = deepcopy(snapshot.controls)
        self.physics_steps = snapshot.physics_steps
        mujoco.mj_forward(self.model, self.data)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    @property
    def ns(self):
        return round(float(self.data.time) * 1_000_000_000)
