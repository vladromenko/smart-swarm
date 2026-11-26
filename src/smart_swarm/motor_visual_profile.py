"""Bind captured RGB, official splits and checkpoint deployment to a drawing profile."""

from hashlib import sha256
from pathlib import Path

STOCK = "stock-rgb"
COMPACT = "compact-mesh-frustum-v1"
VISUAL_PROFILES = (STOCK, COMPACT)


def visual_binding(profile):
    if profile not in VISUAL_PROFILES:
        raise ValueError(f"Unknown motor visual profile {profile!r}")
    binding = {"visual_profile": profile}
    if profile == COMPACT:
        root = Path(__file__).parent
        names = ("motor_renderer.py", "motor_visual_mesh.py")
        binding["visual_implementation_sha256"] = {name: sha256((root / name).read_bytes()).hexdigest()
                                                   for name in names}
    return binding


def recorded_visual(record):
    value = visual_binding(record.get("visual_profile", STOCK))
    if value.get("visual_implementation_sha256") != record.get("visual_implementation_sha256"):
        raise ValueError("Motor visual implementation differs from the original capture binding")
    return value


def match_visual(*records):
    values = [recorded_visual(row) for row in records]
    if any(value != values[0] for value in values[1:]):
        raise ValueError("Motor raw sources, dataset splits and checkpoint must share one visual profile")
    return values[0]
