"""Clone pristine compiled compact geometry and instantiate independent motor states."""

from copy import copy, deepcopy
from time import monotonic

import mujoco
import numpy as np

from .motor_renderer import MotorWorld
from .motor_visual_profile import COMPACT

PROFILE = "compact-pristine-model-copy-v1"


class CompactModelCache:
    def __init__(self, assets, frame_size):
        self.assets = assets
        self.frame_size = frame_size
        self.templates = {}

    def world(self, seed, timeout_s):
        swapped = bool(seed % 2)
        reused = swapped in self.templates
        started = monotonic()
        if not reused:
            self.templates[swapped] = MotorWorld(self.assets, seed=seed, swap_markers=swapped,
                                                  visual_profile=COMPACT, frame_size=self.frame_size)
        original = self.templates[swapped]
        model = copy(original.model)
        data = mujoco.MjData(model)
        world = deepcopy(original, {id(original.model): model, id(original.data): data})
        world.rng = np.random.default_rng(seed)
        world.deadline = started + timeout_s
        world.reset_pose(False)
        if world.data.model is not world.model or world.renderer is not None:
            raise ValueError("A copied motor model must have fresh simulator data and no renderer")
        world.motor_model_cache_hit = reused
        return world

    def close(self):
        for world in self.templates.values():
            world.close()
        self.templates.clear()
