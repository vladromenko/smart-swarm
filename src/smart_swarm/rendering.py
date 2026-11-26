"""Choose the verified Homebrain EGL device before importing MuJoCo."""

import os
from pathlib import Path


def configure_renderer(config, *, grant=None):
    gpu_render = bool(grant and grant.get("gpu_render"))
    if gpu_render and ("gpu_compute" not in grant["locks"] or not grant.get("reserved")):
        raise ValueError("Hardware rendering requires active native GPU admission")
    key = "nvidia_egl_vendor" if gpu_render else "mesa_egl_vendor"
    vendor = config[key]
    if config["render_backend"] != "egl" or not Path(vendor).is_file():
        raise ValueError(f"Configured EGL vendor file is unavailable: {vendor}")
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = vendor
    if gpu_render:
        os.environ.pop("LIBGL_ALWAYS_SOFTWARE", None)
        os.environ.pop("MESA_LOADER_DRIVER_OVERRIDE", None)
        os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)
    else:
        os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
        os.environ["MESA_LOADER_DRIVER_OVERRIDE"] = "llvmpipe"
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(config["mesa_egl_device_index"])
