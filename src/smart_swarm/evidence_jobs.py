"""Run the evidence audit and sequential bounded learners through research admission."""

import os
from pathlib import Path
import subprocess
import time
import uuid

from .episode import atomic, load, sha
from .rehearsal import OUTPUT
from .research_jobs import admitted, execute
from .training import ROOT, offline


def command(portfolio=None, output=OUTPUT):
    config = load(ROOT / "configs/training.json")
    argv = [config["runtime"], str(ROOT / "scripts/learn_evidence.py"),
            "--config", str(ROOT / "configs/evidence-pilot.json"), "--output", str(output)]
    if portfolio is not None:
        return argv + ["--portfolio", str(portfolio)]
    return argv + ["--b1", str(ROOT / "evidence"), "--scratch", str(ROOT.parent.parent / "work")]


def audit():
    env = {**offline(), "PYTHONPATH": str(ROOT / "src")}
    subprocess.run(command(), check=True, env=env, timeout=180)
    return load(OUTPUT / "b1-full-input-audit.json")


def launch(config, portfolio):
    source = load(portfolio)
    if source["recorded_variants"] != 25 or source["independent_scene_groups"] != 5:
        raise ValueError("Complete grouped pilot required before research admission")
    root = config["data_root"] / "training-runs" / ("temporal-evidence-" + uuid.uuid4().hex[:12])
    with admitted(config, root.name) as (grant, heartbeat):
        return run(root, portfolio, grant, heartbeat)


def run(root, portfolio, grant, heartbeat):
    root.mkdir(exist_ok=False, parents=True)
    spec = load(ROOT / "configs/evidence-pilot.json")
    env = {**offline(), "PYTHONPATH": str(ROOT / "src"),
           "SMART_SWARM_VERIFIED_RESEARCH_RESERVATION": grant["job_id"]}
    record = {"purpose": "bounded visual-event and terminal-reveal ablation",
              "status": "training", "compute_reservation": grant, "input_portfolio": str(portfolio),
              "input_sha256": sha(portfolio), "hypothesis": "Causal learned factors can resolve observed histories "
              "while keeping F6 prediction invariant and requesting inspection on hidden cases.",
              "update_cap_per_model": spec["training_step_cap_per_model"], "models": []}
    atomic(root / "run.json", record)
    try:
        with (root / "training.log").open("w") as stream:
            code = execute(command(portfolio, root), env, stream,
                           time.monotonic() + min(1800, spec["combined_wall_cap_s"]), heartbeat)
        record["status"] = "trained" if code == 0 else "failed"
    except (OSError, ValueError, RuntimeError) as error:
        record.update(status="partial", reason=str(error))
        raise
    finally:
        if (root / "models.json").is_file():
            record.update(load(root / "models.json"))
        atomic(root / "run.json", record)
        atomic(OUTPUT / "temporal-evidence-job.json", record)
    if record["status"] != "trained":
        raise RuntimeError(f"Evidence job failed, see {root / 'training.log'}")
    return record


def forecast(run, checkpoint, through_ns, query_policy=None):
    import json

    from .physical_slot_reference import SCOPE_FLAG

    if load(run / "manifest.json").get(SCOPE_FLAG):
        raise ValueError("Use story-recording predict for the saved physical forecast with its verified slot reference")
    argv = command()[:6] + ["--run", str(run.resolve()), "--checkpoint", str(checkpoint.resolve())]
    if through_ns is not None:
        argv.extend(["--through-ns", str(through_ns)])
    if query_policy is not None:
        argv.extend(["--query-policy", query_policy])
    env = {**offline(), "PYTHONPATH": str(ROOT / "src")}
    result = subprocess.run(argv, check=True, env=env, text=True, capture_output=True, timeout=90)
    return json.loads(result.stdout)
