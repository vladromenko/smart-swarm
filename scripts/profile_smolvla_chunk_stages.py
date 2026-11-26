"""Time the original CPU SmolVLA inference phases on two verified archive requests."""

import argparse
from collections import defaultdict
from contextlib import ExitStack
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic, monotonic_ns
from unittest.mock import patch

from smart_swarm.cli import ROOT, settings
from smart_swarm.contract import ProtocolError
from smart_swarm.episode import atomic, json_bytes, load, sha
from smart_swarm.research_jobs import admitted, execute
from smart_swarm.training import offline

PILOT = ROOT / "evidence/learning-pilot-v3"
BASELINE = PILOT / "smolvla-cpu-worker-timing.json"
BASELINE_SHA256 = "ecc78704f7d79b714ae948a30f095bc4c7041c2d0efcbd522ede916dd0bb2f71"
OUTPUT = PILOT / "smolvla-cpu-chunk-stage-timing.json"


def original_input(baseline):
    from profile_motor_device import original_material, packet, timing_inputs

    job, checkpoint, lineage, model_config = original_material("smolvla")
    raw, frames = timing_inputs(lineage, model_config)
    if (job["weights"] != baseline["checkpoint"] or raw["sha256"] != baseline["source_sha256"]
            or sha(lineage) != baseline["training_lineage_sha256"]):
        raise ValueError("The original action source, checkpoint or training split changed")
    requests = [packet(issue, frame) for issue, frame in frames]
    for saved, current in zip(baseline["action_requests"], requests, strict=True):
        if (saved["issue"] != current["issue"] or saved["ordered_rgb_sha256"] != current["ordered_rgb_sha256"]
                or saved["payload_sha256"] != sha256(json_bytes(current["payload"])).hexdigest()):
            raise ValueError("The original timed camera, state or task changed")
    return {"checkpoint": str(checkpoint), "training": str(lineage.parent), "reset_seed": raw["seed"], "requests": requests,
            "runtime": baseline["runtime"], "native_first_actions": [row["first_candidate_native_action"]
                                                                     for row in baseline["action_requests"]]}


def measured(function, durations):
    def wrapped(*args, **kwargs):
        started = monotonic_ns()
        result = function(*args, **kwargs)
        durations.append(monotonic_ns() - started)
        return result

    return wrapped


def breakdown(timings, result, steps):
    forward = timings["transformer_forward"]
    denoise = timings["expert_denoise"]
    if (len(forward) != steps + 1 or len(denoise) != steps or len(timings["embed_image"]) != 2
            or any(len(timings[name]) != 1 for name in ("prepare_images", "embed_prefix"))
            or len(timings["postprocess"]) != 50):
        raise ValueError("The measured phases differ from the original two-camera cached 50-action computation")
    phases = {"processor": result["worker_host_preprocessor_ns"], "image_preparation": timings["prepare_images"][0],
              "prefix_embeddings": timings["embed_prefix"][0], "prefix_cache_prefill": forward[0],
              "expert_denoising_total": sum(denoise), "action_postprocessing": sum(timings["postprocess"])}
    remainder = result["worker_request_ns"] - sum(phases.values())
    if remainder < 0 or sum(timings["embed_image"]) > phases["prefix_embeddings"]:
        raise ValueError("Exclusive phase accounting overlapped another recorded stage")
    prefix_total = sum(phases[name] for name in ("image_preparation", "prefix_embeddings", "prefix_cache_prefill"))
    return {"exclusive_stage_ns": {**phases, "other_worker_time": remainder},
            "complete_worker_ns": result["worker_request_ns"], "vision_encoding_each_image_ns": timings["embed_image"],
            "denoising_steps_ns": denoise, "denoising_transformer_only_ns": forward[1:],
            "prefix_before_first_denoising_step_ns": prefix_total}


def timed_chunk(model, pre, post, request):
    from policy_worker import timed_action

    flow = model.model
    transformer = flow.vlm_with_expert
    timings = defaultdict(list)
    hooks = ((model, "prepare_images", "prepare_images"), (flow, "embed_prefix", "embed_prefix"),
             (transformer, "embed_image", "embed_image"), (transformer, "forward", "transformer_forward"),
             (flow, "denoise_step", "expert_denoise"))
    with ExitStack() as stack:
        for owner, name, key in hooks:
            stack.enter_context(patch.object(owner, name, measured(getattr(owner, name), timings[key])))
        result = timed_action({"type": "smolvla_chunk", **request["payload"]}, model, pre,
                              measured(post, timings["postprocess"]))
    return {**breakdown(timings, result, model.config.num_steps), "issue": request["issue"],
            "payload_sha256": sha256(json_bytes(request["payload"])).hexdigest(),
            "decoded_actions_sha256": sha256(json_bytes(result["actions"])).hexdigest(),
            "first_candidate_native_action": result["actions"][0]}


def profile_inside_worker(index):
    import torch
    from policy_runtime_device import verify_runtime
    from policy_worker import check_timing_inputs, load_policy, memory_usage, startup_profile

    runtime = verify_runtime("cpu")
    start = monotonic_ns()
    config, model, pre, post = load_policy(index["checkpoint"])
    startup = startup_profile(config, runtime, monotonic_ns() - start)
    for key in ("policy", "device", "torch_version", "chunk_size", "actions_per_chunk", "diffusion_steps", "resize_with_padding"):
        if json_bytes(startup[key]) != json_bytes(index["runtime"][key]):
            raise ValueError(f"The measured checkpoint differs from the original runtime at {key}")
    if config.type != "smolvla" or not config.use_cache or config.num_steps != 10 or config.n_action_steps != 50:
        raise ValueError("The measured checkpoint must use the original cached ten-step inference")
    proof = check_timing_inputs(index["training"], index["requests"])
    torch.manual_seed(index["reset_seed"])
    model.reset()
    traces = [timed_chunk(model, pre, post, row) for row in index["requests"]]
    if [row["first_candidate_native_action"] for row in traces] != index["native_first_actions"]:
        raise ValueError("The instrumented action differs from the earlier timed output")
    return {"runtime": startup, "verified_original_native_actions": len(traces), "requests": traces,
            "checkpoint_prefix_cache_enabled": config.use_cache, "cpu_threads": torch.get_num_threads(),
            **proof, **memory_usage("cpu")}


def bounded(output):
    if sha(BASELINE) != BASELINE_SHA256:
        raise ValueError("The original production timing receipt changed")
    baseline = load(BASELINE)
    remaining = baseline["wall_limit_s"] - baseline["elapsed_s"]
    log = output.with_suffix(".log")
    summary = {"status": "profiled", "baseline_sha256": BASELINE_SHA256, "original_wall_limit_s": baseline["wall_limit_s"],
               "previous_timing_seconds": baseline["elapsed_s"], "additional_wall_limit_s": remaining,
               "checkpoint_sha256": baseline["checkpoint"]["sha256"], "training_lineage_sha256": baseline["training_lineage_sha256"],
               "worker_implementation_sha256": sha(ROOT / "scripts/policy_worker.py"),
               "scope": "CPU-only instrumentation of two saved production chunks with no evaluation or actuation",
               "chunk_coverage_ns_at_30hz": baseline["action_requests"][0]["nominal_coverage_ns"],
               "optimizer_updates": 0, "closed_loop_trials": 0, "robot_commands_sent": 0}
    started = monotonic()
    try:
        with admitted(settings(), "smart-swarm-smolvla-cpu-stage-timing", archive_only=True) as (grant, heartbeat):
            summary["reservation"] = grant
            with TemporaryDirectory(prefix="smart-swarm-smolvla-stage-timing-") as directory, log.open("x") as stream:
                inputs, result = (Path(directory) / file for file in ("input.json", "result.json"))
                atomic(inputs, original_input(baseline))
                runtime = load(ROOT / "configs/training.json")["runtime"]
                command = [runtime, str(Path(__file__).resolve()), "--worker", str(inputs), "--output", str(result)]
                code = execute(command, {**offline(), "PYTHONPATH": str(ROOT / "src")}, stream, started + remaining, heartbeat)
                if code != 0:
                    raise ValueError(f"Stage worker exited {code}. See {log}")
                summary.update(load(result))
    except (OSError, ValueError, RuntimeError, TimeoutError, ProtocolError) as error:
        summary.update(status="failed", reason=str(error))
    elapsed = monotonic() - started
    return {**summary, "additional_elapsed_s": elapsed, "remaining_original_seconds": remaining - elapsed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, default=OUTPUT, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker is None and args.output.resolve() != OUTPUT:
        parser.error("The original cumulative timing uses its single recorded receipt path")
    if args.output.exists() or args.output.with_suffix(".log").exists():
        parser.error("Choose an unused stage timing receipt")
    result = profile_inside_worker(load(args.worker)) if args.worker else bounded(args.output)
    atomic(args.output, result)
    print(args.output)
    raise SystemExit(2 if result.get("status") == "failed" else 0)


if __name__ == "__main__":
    main()
