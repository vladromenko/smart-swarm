"""Reuse compact draw contexts while rebuilding the complete simulation for each source."""

from .motor_visual_profile import COMPACT

PROFILE = "compact-draw-context-reuse-v1"


class CompactDrawCache:
    def __init__(self):
        self.ready = {}

    def key(self, world):
        if world.motor_visual["visual_profile"] != COMPACT:
            raise ValueError("Only the compact simulator drawing context can be reused")
        return world.manifest["model_sha256"], tuple(world.frame_size)

    def attach(self, world):
        if world.renderer is not None:
            raise ValueError("A new motor world already owns a drawing context")
        entry = self.ready.pop(self.key(world), None)
        if entry is not None:
            world.renderer, world.mesh_bounds = entry
        return entry is not None

    def release(self, world):
        key = self.key(world)
        if world.renderer is None or key in self.ready or len(self.ready) >= 2:
            raise ValueError("The bounded compact context cache cannot receive this renderer")
        self.ready[key] = world.renderer, world.mesh_bounds
        world.renderer = None

    def close(self):
        for renderer, _ in self.ready.values():
            renderer.close()
        self.ready.clear()
