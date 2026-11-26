"""Use the existing native reservations around Smart Swarm research jobs."""

from contextlib import contextmanager
import json
from pathlib import Path
import select
import subprocess
import time

from .platform_client import production_client, reservation


@contextmanager
def admitted(config, job_id, *, gpu_render=False):
    if config.get("native_cpu_admission"):
        with native_cpu(config, job_id, gpu_render=gpu_render) as grant:
            yield grant
        return
    if gpu_render:
        raise ValueError("NVIDIA rendering requires the verified native Studio graphics admission")
    client = production_client(config)
    with reservation(client, job_id) as (grant, heartbeat):
        heartbeat()
        yield grant, heartbeat


@contextmanager
def native_cpu(config, job_id, *, gpu_render=False):
    from .studio import native_status, studio_env

    native_idle(native_status(config))
    script = Path(__file__).resolve().parents[2] / "scripts/studio_cpu_lease.py"
    args = [config["studio_python"], str(script)]
    if gpu_render:
        args.append("--gpu-render")
    with subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True, env=studio_env(config)) as holder:
        try:
            if not select.select([holder.stdout], [], [], 10)[0]:
                raise TimeoutError("Native Studio reservation was not acknowledged")
            line = holder.stdout.readline()
            if not line:
                raise RuntimeError(f"Native Studio reservation rejected: {holder.stderr.read().strip()}")
            grant = json.loads(line)

            def heartbeat():
                if holder.poll() is not None:
                    raise RuntimeError("The native Studio admission process exited")
                native_idle(native_status(config))

            heartbeat()
            scope = "Studio CPU/recorder and GPU for MuJoCo rendering" if gpu_render else "Studio CPU/recorder"
            yield {**grant, "job_id": job_id, "device": "cpu", "scope": scope, "gpu_render": gpu_render}, heartbeat
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
