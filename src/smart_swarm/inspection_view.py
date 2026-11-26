"""Read a fresh simulated wrist image from RGB and the inspecting robot's calibrated pose."""

import numpy as np

from .vision import center

CUBE_CENTER_HEIGHT_M = (.237, .263)
RIM_MARKER_HEIGHT_M = (.244, .270)
FRONT_SURFACE_MAX_Y_M = -.045
REAR_MARKER_OFFSET_Y_M = .048


def mounted_calibration(world, camera):
    import mujoco

    model = world.model
    identity = model.camera(camera).id
    body = model.body(int(model.cam_bodyid[identity])).name
    if not body.startswith("ros/") or np.any(model.cam_sensorsize[identity]):
        raise ValueError("This readout requires ROSMASTER's mounted fovy camera")
    scratch = mujoco.MjData(model)
    proprioception = world.data.qpos[world.ros_joints]
    scratch.qpos[world.ros_joints] = proprioception
    mujoco.mj_kinematics(model, scratch)
    mujoco.mj_camlight(model, scratch)
    origin = scratch.cam_xpos[identity].copy()
    orientation = scratch.cam_xmat[identity].reshape(3, 3).copy()
    if (not np.allclose(origin, world.data.cam_xpos[identity], rtol=0., atol=1e-10)
            or not np.allclose(orientation.ravel(), world.data.cam_xmat[identity], rtol=0., atol=1e-10)):
        raise ValueError("Own measured joints do not reproduce the mounted camera pose")
    return {"origin_m": origin.tolist(), "camera_to_station_rotation": orientation.tolist(),
            "vertical_fov_deg": float(model.cam_fovy[identity]), "own_joints_deg": np.rad2deg(proprioception).tolist()}


def tabletop_interval(pixel, camera, size, heights):
    if pixel is None:
        return None
    width, height = size
    focal = height / (2 * np.tan(np.deg2rad(camera["vertical_fov_deg"]) / 2))
    origin = np.asarray(camera["origin_m"])
    rotation = np.asarray(camera["camera_to_station_rotation"])
    x, y = pixel
    ray = rotation @ np.array([(x + .5 - width / 2) / focal, (height / 2 - y - .5) / focal, -1.])
    if abs(ray[2]) < 1e-9:
        return None
    distance = (np.asarray(heights) - origin[2]) / ray[2]
    if not np.isfinite(distance).all() or np.any(distance <= 0):
        return None
    points = origin + distance[:, None] * ray
    return points[:, :2]


def bin_choice(cube, cyan, violet):
    if cyan is None or violet is None:
        return None
    if cyan[:, 0].max() < violet[:, 0].min():
        ordered = (cyan, violet)
    elif violet[:, 0].max() < cyan[:, 0].min():
        ordered = (violet, cyan)
    else:
        return None
    possible = []
    for label, marker in zip(("left", "right"), ordered, strict=True):
        interior = marker - [0., REAR_MARKER_OFFSET_Y_M]
        differences = np.abs(cube[:, None, :] - interior[None, :, :])
        if np.all(differences <= [.035, .028]):
            possible.append(label)
    return possible[0] if len(possible) == 1 else None


def interpret(rgb, camera):
    height, width = rgb.shape[:2]
    cues = {key: center(rgb, key) for key in ("cube", "cyan", "violet")}
    cube = tabletop_interval(cues["cube"], camera, (width, height), CUBE_CENTER_HEIGHT_M)
    decision = None
    if cube is not None:
        if float(cube[:, 1].max()) <= FRONT_SURFACE_MAX_Y_M and np.all((0 <= cube[:, 0]) & (cube[:, 0] <= .73)):
            decision = "surface"
        else:
            marks = [tabletop_interval(cues[key], camera, (width, height), RIM_MARKER_HEIGHT_M)
                     for key in ("cyan", "violet")]
            decision = bin_choice(cube, *marks)
    return {"answer": decision or "unresolved", "target_color_detected": cues["cube"] is not None,
            "rgb_marker_centers_px": cues, "cube_xy_interval_m": None if cube is None else cube.tolist()}


def combine(readouts):
    answers = {item["answer"] for item in readouts if item["answer"] != "unresolved"}
    return {"answer": next(iter(answers)) if len(answers) == 1 else "unresolved",
            "visible_target_cue": any(item["target_color_detected"] for item in readouts),
            "conflicting_localizations": len(answers) > 1, "fresh_frames": len(readouts)}
