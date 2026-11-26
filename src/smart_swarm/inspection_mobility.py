"""A separate, force-limited planar ROSMASTER model for inspection development."""

from hashlib import sha256
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from . import scene
from .evidence_rollouts import prepare_world
from .physics import ARM_NAMES, ROS_ACTUATORS, ROS_NAMES, STEP, TABLE

PLANAR_AXES = ("inspection_x", "inspection_y")
BASE_RANGE = ((-.69, .04), (-.04, .18))
BASE_SPEED_M_S = .10
ARM_SPEED_RAD_S = np.deg2rad(35.)
COMMAND_PERIOD_S = .02
STATIONS = {"carrier": (0., 0.), "surface": (-.48, .14), "west_surface": (-.64, .14)}


def add_base_drive(xml):
    base = xml.find("worldbody/body[@name='base_footprint']")
    base.insert(0, ET.Element("inertial", pos="0 0 .025", mass=".06", diaginertia=".0001 .0001 .0001"))
    actuators = xml.find("actuator")
    for name, direction, bounds in zip(PLANAR_AXES, ("1 0 0", "0 1 0"), BASE_RANGE, strict=True):
        limits = " ".join(str(value) for value in bounds)
        ET.SubElement(base, "joint", name=name, type="slide", axis=direction, range=limits, damping="3")
        ET.SubElement(actuators, "position", name=name, joint=name, kp="300", kv="38",
                      ctrlrange=limits, forcerange="-24 24")


def decouple_planar_ground(spec):
    floor, = [geom for geom in spec.geoms if geom.type == mujoco.mjtGeom.mjGEOM_PLANE]
    exclusive = 128
    if any((geom.contype | geom.conaffinity) & exclusive for geom in spec.geoms):
        raise ValueError("The reserved simulated floor collision bit is already in use")
    for geom in spec.geoms:
        original_match = (geom.contype & floor.conaffinity) or (geom.conaffinity & floor.contype)
        if geom.type != mujoco.mjtGeom.mjGEOM_PLANE and not geom.parent.name.startswith("ros/") and original_match:
            geom.conaffinity |= exclusive
    floor.contype = exclusive
    floor.conaffinity = 0


def mobile_model(assets, seed, family, variant, *, isolate_floor=True):
    root, fixture = scene.workspace(bool(seed % 2))
    scene.extend_evidence_scene(fixture, family, bool(seed // 4 % 2))
    if family == "F6":
        scene.add_matched_target(fixture, "surface_cube" if variant == "hidden-surface" else "cube")
    spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))
    components = (("so", "so101", (0., 0., TABLE)), ("ros", "rosmaster", (.44, -.62, 0.)),
                  ("tank", "tank", (.43, .73, 0.)))
    for namespace, robot, position in components:
        xml, _ = scene.load_component(assets, robot)
        scene.configure_component(xml, robot)
        if robot == "rosmaster":
            add_base_drive(xml)
        child = mujoco.MjSpec.from_string(ET.tostring(xml, encoding="unicode"))
        spec.attach(child, prefix=f"{namespace}/", frame=spec.worldbody.add_frame(pos=position))
    if isolate_floor:
        decouple_planar_ground(spec)
    return spec.compile(), sha256(spec.to_xml().encode()).hexdigest()


def verify_floor_mask(previous, extended):
    floor, = np.flatnonzero(previous.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
    if extended.geom_contype[floor] == previous.geom_contype[floor]:
        return
    robot = np.asarray([previous.body(int(body)).name.startswith("ros/") for body in previous.geom_bodyid])
    initial = (previous.geom_contype[floor] & previous.geom_conaffinity) | (
        previous.geom_conaffinity[floor] & previous.geom_contype)
    result = (extended.geom_contype[floor] & extended.geom_conaffinity) | (
        extended.geom_conaffinity[floor] & extended.geom_contype)
    expected = (initial != 0) & ~robot
    expected[floor] = False
    if not np.array_equal(expected, result != 0):
        raise ValueError("The alternate floor coupling changed another robot or object")


def initial_state(source, model):
    if source.physics_steps or source.renderer is not None:
        raise ValueError("The mobility model can only be installed at the unchanged initial state")
    old, known = source.data, source.model
    data = mujoco.MjData(model)
    if known.nbody != model.nbody or known.ngeom != model.ngeom or known.ncam != model.ncam:
        raise ValueError("The inspection extension changed the original scene inventory")
    verify_floor_mask(known, model)
    for index in range(known.njnt):
        key = known.joint(index).name
        before, after = int(known.jnt_qposadr[index]), int(model.joint(key).qposadr[0])
        stop = int(known.jnt_qposadr[index + 1]) if index + 1 < known.njnt else known.nq
        data.qpos[after:after + stop - before] = old.qpos[before:stop]
    for index in range(known.nu):
        data.ctrl[model.actuator(known.actuator(index).name).id] = old.ctrl[index]
    model.cam_fovy[:] = known.cam_fovy
    model.geom_friction[:] = known.geom_friction
    mujoco.mj_forward(model, data)
    for array in ("xpos", "xmat", "geom_xpos", "geom_xmat", "cam_xpos", "cam_xmat"):
        np.testing.assert_allclose(getattr(data, array), getattr(old, array), atol=1e-12, rtol=0.)
    return data


def install(world, model, identity):
    world.data = initial_state(world, model)
    world.model = model
    tank = ("x", "y", "yaw")
    world.arm_joints = world.joints("so", ARM_NAMES)
    world.ros_joints = world.joints("ros", ROS_NAMES)
    world.tank_joints = world.joints("tank", tank)
    world.tank_dofs = world.joints("tank", tank, velocity=True)
    world.arm_actuators = world.actuators("so", ARM_NAMES)
    world.ros_actuators = world.actuators("ros", ROS_ACTUATORS)
    world.tank_actuators = world.actuators("tank", tank)
    world.arm_limits = model.actuator_ctrlrange[world.arm_actuators]
    world.mobile_joints = world.joints("ros", PLANAR_AXES)
    world.mobile_dofs = world.joints("ros", PLANAR_AXES, velocity=True)
    world.mobile_actuators = world.actuators("ros", PLANAR_AXES)
    world.manifest.update(parent_model_sha256=world.manifest["model_sha256"], model_sha256=identity,
                          rosmaster_motion="force-limited planar drive surrogate; fixed yaw")


def prepare_mobile(assets, seed, family, variant, *, seconds=100, isolate_floor=True, observed_steps=False):
    if (family, variant) not in (("F1", "visible-control"), ("F6", "hidden-contained"), ("F6", "hidden-surface")):
        raise ValueError("Only the isolated F1 and F6 inspection development scenarios are supported")
    world, variation, pickup = prepare_world(assets, seed, family, variant=variant, seconds=seconds)
    model, identity = mobile_model(assets, seed, family, variant, isolate_floor=isolate_floor)
    install(world, model, identity)
    world.observed_inspection_steps = observed_steps
    return world, variation, pickup


def chassis_path(station):
    x, y = STATIONS[station]
    return ((x, 0.), (x, y))


class SceneContactWatch:
    """Record environmental contacts alongside the permitted motion command trace."""

    def __init__(self, world):
        self.world = world
        model = world.model
        ids = range(model.ngeom)
        body = [model.body(int(model.geom_bodyid[index])).name for index in ids]
        self.labels = [model.geom(index).name or f"{body[index]}/geom-{index}" for index in ids]
        self.robot = np.asarray([name.startswith("ros/") for name in body])
        self.floor = (model.geom_bodyid == 0) & (model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
        self.pairs = set()
        self.self_pairs = set()
        self.active_steps = 0
        self.peak_speed_m_s = 0.

    def sample(self):
        state = self.world.data
        for contact in state.contact:
            left, right = int(contact.geom1), int(contact.geom2)
            if self.robot[left] and self.robot[right]:
                self.self_pairs.add(tuple(sorted((self.labels[left], self.labels[right]))))
            if (self.robot[left] != self.robot[right]) and not (self.floor[left] or self.floor[right]):
                self.pairs.add(tuple(sorted((self.labels[left], self.labels[right]))))
        velocity = float(np.linalg.norm(state.qvel[self.world.mobile_dofs]))
        self.peak_speed_m_s = max(self.peak_speed_m_s, velocity)
        self.active_steps += 1

    def checked_sample(self):
        self.sample()
        if self.pairs:
            raise ValueError(f"Inspection motion contacted the scene: {sorted(self.pairs)}")

    def advance(self, steps):
        if self.world.observed_inspection_steps:
            for start in range(0, steps, 2000):
                self.world.step_observed(min(2000, steps - start), self.checked_sample)
        else:
            for _ in range(steps):
                self.world.step(1)
                self.checked_sample()


def slew(world, watch, indices, goal, rate, settle_s):
    value = world.data.ctrl[indices].copy()
    goal = np.asarray(goal, dtype=float)
    limits = world.model.actuator_ctrlrange[indices]
    if goal.shape != value.shape or not np.isfinite(goal).all() or np.any((goal < limits[:, 0]) | (goal > limits[:, 1])):
        raise ValueError("Inspection drive command is outside the simulated actuator limits")
    steps = round(COMMAND_PERIOD_S / STEP)
    max_change = rate * steps * STEP
    count = int(np.ceil(np.linalg.norm(goal - value) / max_change))
    started = world.ns
    for index in range(count):
        world.data.ctrl[indices] = value + ((index + 1) / count) * (goal - value)
        watch.advance(steps)
    watch.advance(round(settle_s / STEP))
    return world.ns - started


def drive_to(world, watch, station):
    paths = chassis_path(station)
    for delta in paths:
        slew(world, watch, world.mobile_actuators, delta, BASE_SPEED_M_S, .75)
    actual = world.data.qpos[world.mobile_joints].copy()
    error = float(np.linalg.norm(actual - paths[-1]))
    if error > .005 or watch.peak_speed_m_s > .13:
        raise ValueError(f"Inspection chassis tracking exceeded limits, position error {error:.4f} m")
    return {"requested_xy_m": list(paths[-1]), "measured_xy_m": actual.tolist(), "tracking_error_m": error,
            "max_measured_speed_m_s": watch.peak_speed_m_s, "external_contact_pairs": sorted(watch.pairs),
            "self_contact_pairs": sorted(watch.self_pairs), "checked_physics_steps": watch.active_steps}


def mounted_feedback(world, camera):
    model = world.model
    identity = model.camera(camera).id
    body = model.body(int(model.cam_bodyid[identity])).name
    if not body.startswith("ros/") or np.any(model.cam_sensorsize[identity]):
        raise ValueError("The mobile readout requires ROSMASTER's mounted fovy camera")
    scratch = mujoco.MjData(model)
    base = world.data.qpos[world.mobile_joints]
    arm = world.data.qpos[world.ros_joints]
    scratch.qpos[world.mobile_joints] = base
    scratch.qpos[world.ros_joints] = arm
    mujoco.mj_kinematics(model, scratch)
    mujoco.mj_camlight(model, scratch)
    origin = scratch.cam_xpos[identity].copy()
    pose = scratch.cam_xmat[identity].reshape(3, 3).copy()
    np.testing.assert_allclose(origin, world.data.cam_xpos[identity], atol=1e-10, rtol=0.)
    np.testing.assert_allclose(pose.ravel(), world.data.cam_xmat[identity], atol=1e-10, rtol=0.)
    return {"origin_m": origin.tolist(), "camera_to_station_rotation": pose.tolist(),
            "vertical_fov_deg": float(model.cam_fovy[identity]), "measured_base_xy_m": base.tolist(),
            "own_joints_deg": np.rad2deg(arm).tolist()}
