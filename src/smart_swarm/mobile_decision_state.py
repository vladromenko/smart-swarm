"""Move one sealed stock decision state into the isolated mobile inspection model."""

from copy import deepcopy

import mujoco
import numpy as np

SHARED_GEOMETRY = ("xpos", "xmat", "geom_xpos", "geom_xmat", "cam_xpos", "cam_xmat")
STATIC_STATE = ("mjSTATE_TIME", "mjSTATE_XFRC_APPLIED", "mjSTATE_EQ_ACTIVE")
CAMERA_FIELDS = ("cam_pos", "cam_quat", "cam_fovy", "cam_intrinsic")


def one_bit_states():
    for name, state in mujoco.mjtState.__members__.items():
        value = int(state)
        if value > 0 and value & (value - 1) == 0 and value & int(mujoco.mjtState.mjSTATE_INTEGRATION):
            yield name, state


def state_vector(world, state):
    values = np.empty(mujoco.mj_stateSize(world.model, state))
    mujoco.mj_getState(world.model, world.data, values, state)
    return values


def named_joint_lanes(stock, mobile, address, size):
    old = np.append(getattr(stock, address), getattr(stock, size))
    new = np.append(getattr(mobile, address), getattr(mobile, size))
    targets = []
    for index in range(stock.njnt):
        joint = stock.joint(index)
        there = mobile.joint(joint.name)
        start, stop = new[there.id:there.id + 2]
        if stop - start != old[index + 1] - old[index] or there.type[0] != joint.type[0]:
            raise ValueError(f"The mobile model changed the layout of the joint {joint.name}")
        targets.extend(range(start, stop))
    if len(set(targets)) != getattr(stock, size):
        raise ValueError("The mobile state has repeated or missing joint coordinates")
    return targets


def state_lanes(stock, mobile):
    actuator = [mobile.actuator(stock.actuator(index).name).id for index in range(stock.nu)]
    position = named_joint_lanes(stock, mobile, "jnt_qposadr", "nq")
    velocity = named_joint_lanes(stock, mobile, "jnt_dofadr", "nv")
    return {"mjSTATE_QPOS": position, "mjSTATE_QVEL": velocity, "mjSTATE_WARMSTART": velocity,
            "mjSTATE_QFRC_APPLIED": velocity, "mjSTATE_CTRL": actuator}


def verify_inventory(stock, mobile):
    for field in ("body", "geom", "camera", "equality"):
        kind = {"body": "nbody", "geom": "ngeom", "camera": "ncam", "equality": "neq"}[field]
        old, new = (tuple(getattr(model, field)(index).name for index in range(getattr(model, kind)))
                    for model in (stock, mobile))
        if old != new:
            raise ValueError(f"The mobile model changed the ordering of shared {field} state")


def copy_solver_state(stock, mobile):
    lanes = state_lanes(stock.model, mobile.model)
    copied = []
    for name, component in one_bit_states():
        original, extended = (state_vector(world, component) for world in (stock, mobile))
        mapping = lanes.get(name)
        if mapping is not None:
            if np.any(np.delete(extended, mapping) != 0.):
                raise ValueError(f"The additional mobile coordinates are not neutral in {name}")
            extended[mapping] = original
        elif name in STATIC_STATE and extended.shape == original.shape:
            extended[:] = original
        elif extended.size or original.size:
            raise ValueError(f"The mobile model introduced unmapped integration state {name}")
        mujoco.mj_setState(mobile.model, mobile.data, extended, component)
        copied.append((name, component, extended))
    return copied


def transplant(stock, mobile, snapshot):
    if mobile.physics_steps or mobile.renderer is not None:
        raise ValueError("Mobile inspection must start from a cold model")
    stock.restore(snapshot)
    verify_inventory(stock.model, mobile.model)
    for name, camera in zip(CAMERA_FIELDS, snapshot.camera_pose, strict=True):
        getattr(mobile.model, name)[:] = camera
    np.testing.assert_array_equal(stock.model.geom_friction, mobile.model.geom_friction)
    copied = copy_solver_state(stock, mobile)
    for field in ("pending", "filtered", "queue", "tank_target", "controls", "physics_steps"):
        setattr(mobile, field, deepcopy(getattr(snapshot, field)))
    mobile.rng.bit_generator.state = deepcopy(snapshot.rng)
    mobile.deadline = stock.deadline
    mujoco.mj_forward(mobile.model, mobile.data)
    for field in SHARED_GEOMETRY:
        np.testing.assert_allclose(getattr(stock.data, field), getattr(mobile.data, field), atol=1e-12, rtol=0.)
    for name, component, values in copied:
        np.testing.assert_array_equal(state_vector(mobile, component), values, err_msg=name)
    if mobile.ns != stock.ns or mobile.truth_target != stock.truth_target:
        raise ValueError("Mobile inspection changed the decision clock or target binding")
    return mobile.snapshot()
