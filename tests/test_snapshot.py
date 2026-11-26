import os

import numpy as np
import pytest

from smart_swarm.cli import settings


def test_integration_snapshot_restores_pending_runtime_state():
    pytest.importorskip("mujoco")
    from smart_swarm.physics import World

    assets = settings()["asset_root"]
    if not (assets / "so101.xml").is_file():
        pytest.skip("Prepare the approved MuJoCo assets first")
    os.environ.setdefault("MUJOCO_GL", "egl")
    world = World(assets)
    try:
        world.queue.append({"recipient": "rosmaster-1", "delivered_ns": "35000000"})
        world.controls["teacher_chunk"] = {"phase": "transfer", "index": 3}
        world.tank_target[:] = [.10, .03]
        world.pending[0] += .2
        world.step(16)
        sample = world.snapshot()
        expected_rng = world.rng.uniform()
        world.step(43)
        expected_pose = world.data.qpos.copy()
        world.model.cam_fovy += 4
        world.queue.clear()
        world.controls["teacher_chunk"].clear()
        world.restore(sample)
        assert world.queue == sample.queue
        assert world.controls["teacher_chunk"]["index"] == 3
        assert np.array_equal(world.model.cam_fovy, sample.camera_pose[2])
        assert world.rng.uniform() == expected_rng
        world.step(43)
        assert np.allclose(world.data.qpos, expected_pose, rtol=0, atol=1e-10)
    finally:
        world.close()


def test_mobile_camera_pose_uses_own_feedback_after_exact_snapshot_restore():
    mujoco = pytest.importorskip("mujoco")
    from smart_swarm.inspection_mobility import mounted_feedback, prepare_mobile

    assets = settings()["asset_root"]
    if not (assets / "rosmaster.xml").is_file():
        pytest.skip("Prepare the approved MuJoCo assets first")
    world, _, _ = prepare_mobile(assets, 416031, "F6", "hidden-surface")
    try:
        camera = "ros/front_cam"
        initial = world.snapshot()
        nominal = mounted_feedback(world, camera)
        translation = [-.64, .14]
        world.data.qpos[world.mobile_joints] = translation
        mujoco.mj_forward(world.model, world.data)
        translated = mounted_feedback(world, camera)
        assert translated["measured_base_xy_m"] == translation
        assert np.allclose(np.subtract(translated["origin_m"], nominal["origin_m"]), [*translation, 0.])
        hidden = int(world.model.joint("surface_cube_pose").qposadr[0])
        world.data.qpos[hidden:hidden + 3] += [.015, .015, .03]
        world.truth_target = "cube"
        mujoco.mj_forward(world.model, world.data)
        assert mounted_feedback(world, camera) == translated
        world.restore(initial)
        assert mounted_feedback(world, camera) == nominal
        assert np.array_equal(world.snapshot().integration, initial.integration)
    finally:
        world.close()
