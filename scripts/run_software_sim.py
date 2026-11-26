#!/usr/bin/env python3
"""Initialize Homebrain software EGL before an RGB diagnostic imports MuJoCo."""

import argparse
from pathlib import Path
import runpy
import sys

from smart_swarm.cli import settings
from smart_swarm.rendering import configure_renderer

DIAGNOSTICS = (
    "rehearse_rosmaster_mobility",
    "compare_rosmaster_mobile",
    "probe_precontact_act",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("diagnostic", choices=DIAGNOSTICS)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if "mujoco" in sys.modules:
        raise RuntimeError("The simulation launcher must set EGL before MuJoCo loads")
    configure_renderer(settings())
    sys.argv[1:] = args.arguments
    runpy.run_path(str(Path(__file__).with_name(f"{args.diagnostic}.py")), run_name="__main__")


if __name__ == "__main__":
    main()
