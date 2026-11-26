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


def hold(gpu_render, archive_only=False, *, cuda_compute=False):
    needs_gpu = gpu_render or cuda_compute
    if needs_gpu:
        check_graphics()
    with ExitStack() as stack:
        locks = ["cpu_compute"] if archive_only else ["cpu_compute", "recording"]
        if needs_gpu:
            locks.append("gpu_compute")
        for name in locks:
            stack.enter_context(ResourceLease(name))
        snapshot = check_graphics() if needs_gpu else assert_compute_available("cpu", minimum_available_gb=3)
        print(json.dumps({"reserved": True, "backend": "so101_lab.resources.ResourceLease",
                          "locks": locks,
                          "memory_available_bytes": snapshot["memory_available_bytes"]}), flush=True)
        for line in sys.stdin:
            if line.strip():
                raise ValueError("Studio lease holder only accepts EOF")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--gpu-render", action="store_true")
    scope.add_argument("--gpu-compute", action="store_true", help="Hold Studio's existing CUDA and recorder reservations")
    scope.add_argument("--archive-only", action="store_true", help="Reserve just CPU for already sealed inputs")
    args = parser.parse_args()
    try:
        hold(args.gpu_render, args.archive_only, cuda_compute=args.gpu_compute)
    except ResourceBusy as error:
        raise SystemExit(str(error))


if __name__ == "__main__":
    main()
