"""Simulated SO101 command conversion and deployment control timing."""

import numpy as np

from .physics import STEP

FPS = 30
PROFILES = ("historical-eval-v1", "aligned-sample-v2")


class ArmExecutor:
    def __init__(self, world, profile):
        if profile not in PROFILES:
            raise ValueError(f"Unknown SO101 executor profile {profile}")
        self.world = world
        self.profile = profile
        self.ticks = 0

    def prime(self):
        if self.profile == "aligned-sample-v2":
            self.advance()

    def advance(self):
        ticks_per_second = round(1 / STEP)
        quantize = round if self.profile == "historical-eval-v1" else int
        end = quantize((self.ticks + 1) * ticks_per_second / FPS)
        start = quantize(self.ticks * ticks_per_second / FPS)
        self.world.step(end - start)
        self.ticks += 1

    def observation_index(self):
        if self.profile != "aligned-sample-v2" or self.ticks < 1:
            raise ValueError("Issue-clock observations require a primed aligned executor")
        return self.ticks - 1

    def submit(self, values):
        request = np.asarray(values, dtype=float)
        if request.shape != (6,) or not np.isfinite(request).all():
            raise ValueError("Executor requires six finite SO101 targets")
        actuator = self.world.arm_to_mj(request)
        if self.profile == "historical-eval-v1":
            self.advance()
        self.world.pending = actuator
        if self.profile == "aligned-sample-v2":
            self.advance()
        return {"requested": request.tolist(), "decoded_and_clipped": self.world.arm_from_mj(actuator).tolist(),
                "actuator_radians": actuator.tolist()}
