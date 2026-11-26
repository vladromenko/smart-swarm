"""Use two calibrated post-inspection RGB views to resolve container depth."""

import numpy as np

from .inspection_view import CUBE_CENTER_HEIGHT_M, RIM_MARKER_HEIGHT_M, bin_choice, combine, interpret

MAX_RAY_SEPARATION_M = .006
MIN_VIEW_ANGLE_DEG = 10.
MIN_WEAK_CUBE_PIXELS = 4
POSITION_MARGIN_M = .004


def pixel_ray(pixel, camera, size):
    width, height = size
    x, y = pixel
    focal = height / (2 * np.tan(np.deg2rad(camera["vertical_fov_deg"]) / 2))
    direction = np.asarray(camera["camera_to_station_rotation"]) @ [
        (x + .5 - width / 2) / focal, (height / 2 - y - .5) / focal, -1.]
    return np.asarray(camera["origin_m"]), direction / np.linalg.norm(direction)


def intersect(first, second, heights):
    a, d = first
    b, e = second
    if abs(d @ e) >= np.cos(np.deg2rad(MIN_VIEW_ANGLE_DEG)):
        return None
    near, far = np.linalg.lstsq(np.stack((d, -e), axis=1), b - a, rcond=None)[0]
    points = np.stack((a + near * d, b + far * e))
    center = points.mean(axis=0)
    if (min(near, far) <= 0 or np.linalg.norm(points[0] - points[1]) > MAX_RAY_SEPARATION_M
            or not heights[0] - POSITION_MARGIN_M <= center[2] <= heights[1] + POSITION_MARGIN_M):
        return None
    return np.stack((center[:2] - POSITION_MARGIN_M, center[:2] + POSITION_MARGIN_M))


def weak_cube_center(rgb):
    # Keep the existing orange mask. Use scarce support only alongside a strong view and ray agreement.
    r, g, b = rgb.astype(np.int16).transpose(2, 0, 1)
    y, x = np.where((r > 140) & (r - g > 77) & (g - b > 10))
    return (float(np.median(x)), float(np.median(y))) if x.size >= MIN_WEAK_CUBE_PIXELS else None


def extent(samples, readouts, cue, heights):
    pixels = [row["rgb_marker_centers_px"][cue] for row in readouts]
    if cue == "cube" and any(pixel is not None for pixel in pixels):
        pixels = [pixel if pixel is not None else weak_cube_center(rgb)
                  for pixel, (rgb, _) in zip(pixels, samples, strict=True)]
    if any(pixel is None for pixel in pixels):
        return None
    rays = [pixel_ray(pixel, camera, rgb.shape[:2][::-1])
            for pixel, (rgb, camera) in zip(pixels, samples, strict=True)]
    return intersect(*rays, heights)


def locate(samples):
    if len(samples) != 2:
        raise ValueError("Two ordered post-inspection RGB exposures are required")
    readouts = [interpret(rgb, camera) for rgb, camera in samples]
    result = combine(readouts)
    if result["answer"] != "unresolved" or result["conflicting_localizations"]:
        return {**result, "used_triangulation": False}
    cube = extent(samples, readouts, "cube", CUBE_CENTER_HEIGHT_M)
    tags = [extent(samples, readouts, cue, RIM_MARKER_HEIGHT_M) for cue in ("cyan", "violet")]
    answer = bin_choice(cube, *tags) if cube is not None else None
    return {**result, "answer": answer or "unresolved", "used_triangulation": answer is not None}
