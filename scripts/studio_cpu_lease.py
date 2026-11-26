"""Hold the existing Studio leases for one owned research job."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys

from so101_lab.resources import ResourceBusy, ResourceLease, assert_compute_available, owners


def check_graphics():
    devices = [Path("/dev/nvgpu/igpu0") / name for name in ("ctrl", "channel")]
    if not all(path.exists() for path in devices):
        raise ResourceBusy("Homebrain integrated GPU device paths are unavailable")
    conflicts = owners(devices)
    if conflicts:
        raise ResourceBusy(f"Existing graphics device owners: {json.dumps(conflicts)}")
    return assert_compute_available("cuda", minimum_available_gb=7)


def hold(gpu_render):
    if gpu_render:
        check_graphics()
    with ExitStack() as stack:
        locks = ["cpu_compute", "recording"]
        if gpu_render:
            locks.append("gpu_compute")
        for name in locks:
            stack.enter_context(ResourceLease(name))
        snapshot = check_graphics() if gpu_render else assert_compute_available("cpu", minimum_available_gb=3)
        print(json.dumps({"reserved": True, "backend": "so101_lab.resources.ResourceLease",
                          "locks": locks,
                          "memory_available_bytes": snapshot["memory_available_bytes"]}), flush=True)
        for line in sys.stdin:
            if line.strip():
                raise ValueError("Studio lease holder only accepts EOF")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-render", action="store_true")
    args = parser.parse_args()
    try:
        hold(args.gpu_render)
    except ResourceBusy as error:
        raise SystemExit(str(error))


if __name__ == "__main__":
    main()
