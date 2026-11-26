"""Untrained, local RGB evidence for the disclosed color-cue condition."""

import numpy as np

from .memory import CUBE_SUPPORT, PERMUTATIONS, TRACKS, Evidence


def center(image, color):
    r, g, b = image.astype(np.int16).transpose(2, 0, 1)
    if color == "cyan":
        mask = (g > 80) & (b > 80) & (g - r > 50) & (b - r > 50)
    elif color == "violet":
        mask = (r > 65) & (b > 85) & (r - g > 40) & (b - g > 60)
    else:
        mask = (r > 140) & (r - g > 77) & (g - b > 10)
    y, x = np.where(mask)
    return (float(np.median(x)), float(np.median(y))) if x.size >= 7 else None


def visual_record(factor, chosen, possibilities, packets, sensor):
    cameras = [next(cam for cam in packet["cameras"] if cam["sensor_id"] == sensor)
               for packet in packets]
    origin = packets[0]["agent_id"]
    frames = tuple(item["frame_id"] for item in cameras)
    return Evidence(f"{origin}:{factor}:{frames[-1]}", origin, frames,
                    cameras[0]["capture_interval"]["start_ns"],
                    cameras[-1]["capture_interval"]["end_ns"],
                    packets[-1]["receive_time"]["monotonic_ns"], factor,
                    {item: (.98 if item == chosen else .012) for item in possibilities})


def insertion(packet, image):
    cube = center(image, "cube")
    if cube is None:
        return None
    tags = {name: center(image, name) for name in TRACKS}
    distances = {name: np.linalg.norm(np.subtract(cube, tag)) for name, tag in tags.items()
                 if tag is not None}
    if not distances:
        return None
    closest = min(distances, key=distances.get)
    chosen = closest if distances[closest] <= 37 else "surface"
    return visual_record("containment", chosen, CUBE_SUPPORT, [packet], "so101-1/top")


def relocation(packets, before, after, *, sensor="tank-1/front"):
    early = {name: center(before, name) for name in TRACKS}
    final = {name: center(after, name) for name in TRACKS}
    if any(point is None for point in (*early.values(), *final.values())):
        return None
    travel = {name: abs(final[name][0] - early[name][0]) for name in TRACKS}
    if max(travel.values()) < 8 or min(travel.values()) > 5:
        return None
    # Tank looks south. Its pixel-right is north-facing ROSMASTER's pixel-left.
    chosen = max(final, key=lambda name: final[name][0]) + "_left"
    return visual_record("association", chosen, PERMUTATIONS, packets, sensor)
