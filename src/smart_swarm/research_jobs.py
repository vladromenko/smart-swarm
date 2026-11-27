"""Use the existing native reservations around Smart Swarm research jobs."""

from contextlib import contextmanager
import json
from os import EX_TEMPFAIL
from pathlib import Path
import select
import subprocess
import time

from .platform_client import production_client, reservation


class NativeComputeBusy(RuntimeError):
    """The native lease holder declined compute for a temporary resource condition."""


@contextmanager
def admitted(config, job_id, *, gpu_render=False, archive_only=False, device="cpu"):
    native = config.get("native_cpu_admission")
    if device not in ("cpu", "cuda"):
        raise ValueError(f"Unknown research device {device}")
    if device == "cuda" and (gpu_render or archive_only or not native):
        raise ValueError("CUDA policy compute requires its own verified Studio reservation mode")
    if archive_only and (gpu_render or not native):
        raise ValueError("Archive-only work requires the verified native Studio CPU reservation")
    if native:
        with native_cpu(config, job_id, gpu_render=gpu_render, archive_only=archive_only, device=device) as grant:
            yield grant
        return
    if gpu_render:
        raise ValueError("NVIDIA rendering requires the verified native Studio graphics admission")
    client = production_client(config)
    with reservation(client, job_id) as (grant, heartbeat):
        heartbeat()
        yield grant, heartbeat


def studio_lease_mode(gpu_render, archive_only, device):
    if device == "cuda":
        return ["--gpu-compute"], "Studio CUDA policy compute"
    if gpu_render:
        return ["--gpu-render"], "Studio CPU/recorder and GPU for MuJoCo rendering"
    if archive_only:
        return ["--archive-only"], "Studio CPU only"
    return [], "Studio CPU/recorder"


def verify_compute_scope(device, grant):
    if device not in ("cpu", "cuda"):
        raise ValueError(f"Unknown policy device {device}")
    if device == "cuda" and (not grant or not grant.get("reserved") or grant.get("device") != "cuda"
            or grant.get("gpu_render") is not False or grant.get("backend") != "so101_lab.resources.ResourceLease"
            or set(grant.get("locks", ())) != {"cpu_compute", "recording", "gpu_compute"}):
        raise ValueError("CUDA policy compute requires the live native Studio compute grant")


def read_native_grant(holder):
    if not select.select([holder.stdout], [], [], 10)[0]:
        raise TimeoutError("Native Studio reservation was not acknowledged")
    line = holder.stdout.readline()
    if not line:
        reason = f"Native Studio reservation rejected: {holder.stderr.read().strip()}"
        if holder.wait(timeout=5) == EX_TEMPFAIL:
            raise NativeComputeBusy(reason)
        raise RuntimeError(reason)
    return json.loads(line)


@contextmanager
def native_cpu(config, job_id, *, gpu_render=False, archive_only=False, device="cpu"):
    from .studio import native_status, studio_env

    if not archive_only:
        native_idle(native_status(config))
    script = Path(__file__).resolve().parents[2] / "scripts/studio_cpu_lease.py"
    flags, scope = studio_lease_mode(gpu_render, archive_only, device)
    args = [config["studio_python"], str(script), *flags]
    with subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True, env=studio_env(config)) as holder:
        try:
            grant = {**read_native_grant(holder), "job_id": job_id, "device": device,
                     "scope": scope, "gpu_render": gpu_render}
            if archive_only and grant.get("locks") != ["cpu_compute"]:
                raise RuntimeError("Archive-only work was not granted the native CPU-only reservation")
            verify_compute_scope(device, grant)

            def heartbeat():
                if holder.poll() is not None:
                    raise RuntimeError("The native Studio admission process exited")
                if not archive_only:
                    native_idle(native_status(config))

            heartbeat()
            yield grant, heartbeat
        finally:
            holder.stdin.close()
            try:
                holder.wait(timeout=5)
            except subprocess.TimeoutExpired:
                holder.terminate()
                holder.wait(timeout=5)


def native_idle(status):
    from .contract import ProtocolError

    if any(status[key] for key in ("recording_active", "teleoperation_active", "policy_active", "learning_job_active")):
        raise ProtocolError("RESOURCE_BUSY", "A native recording, motion, policy or learning job is active")


def execute(argv, env, stream, deadline, heartbeat):
    """Wait for only this worker and keep its shared lease live until it exits."""
    with subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT, env=env) as worker:
        next_heartbeat = time.monotonic() + 10
        try:
            while worker.poll() is None and time.monotonic() < deadline:
                if time.monotonic() >= next_heartbeat:
                    heartbeat()
                    next_heartbeat = time.monotonic() + 10
                wait_until = max(.01, min(deadline, next_heartbeat) - time.monotonic())
                try:
                    worker.wait(timeout=wait_until)
                except subprocess.TimeoutExpired:
                    pass
            if worker.poll() is None:
                raise TimeoutError("Research job reached its wall cap")
            return worker.returncode
        finally:
            if worker.poll() is None:
                worker.terminate()
                try:
                    worker.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait(timeout=5)
