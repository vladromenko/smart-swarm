#!/usr/bin/env python3
"""Exercise the normal capture, official export and live action-input path without training."""

import argparse
from base64 import b64decode
from contextlib import closing
from hashlib import sha256
import math
from pathlib import Path
import time

import numpy as np

from learning_pilot import EVIDENCE, MOTOR_RAW, WORK, official_export, verified_export
from smart_swarm.cli import settings
from smart_swarm.episode import atomic, load, sha
from smart_swarm.motor_input import EXACT_REPLAY_ISSUES as TRACE_ROWS, FLOAT32_COMMAND_PROFILE
from smart_swarm.motor_visual_profile import COMPACT, match_visual, visual_binding
from smart_swarm.rendering import configure_renderer
from smart_swarm.research_jobs import admitted

COUNTS = {"train": 1, "validation": 2}
GENERATION_CEILING_S = 210
EXPORT_CEILING_S = 180
PREVIOUS = WORK / "motor-compact-rehearsal-v1"
LOCATION = WORK / "motor-compact-rehearsal-v2"
MIB = 1024**2


def previous_spend(budget):
    receipt = PREVIOUS / "rehearsal.json"
    previous = load(receipt)
    if (previous["status"] != "incomplete" or previous["recipe"]["stock_capture_ledger_sha256"] != sha(budget)
            or previous["optimizer_updates"] != 0 or previous.get("capture_started_unix")
            or not verified_export(PREVIOUS / "raw", PREVIOUS / "lerobot", "--clocked")):
        raise ValueError("The failed compact capture and its official export must remain available")
    generated = (PREVIOUS / "raw/validation-generation.json").stat().st_mtime_ns
    finalized = receipt.stat().st_mtime_ns
    # Charge the whole interval through the failed verification, including the export.
    export_s = (finalized - generated) / 1e9
    generation_s = previous["generation_elapsed_s"]
    if not 0 < generation_s < GENERATION_CEILING_S or not 0 < export_s < EXPORT_CEILING_S:
        raise ValueError("The failed compact attempt has no confirmed budget left")
    preserved_bytes = sum(path.stat().st_size for root in (MOTOR_RAW, PREVIOUS / "raw")
                          for path in root.glob("*/*/motor.npz"))
    return {"receipt_sha256": sha(receipt), "failed_generation_s": generation_s,
            "failed_export_upper_bound_s": export_s, "preserved_raw_bytes": preserved_bytes}


def open_report(root):
    budget = EVIDENCE / "motor-full-capture.json"
    last = load(budget)
    remaining = int(last["wall_cap_s"] - last["generation_elapsed_s"])
    if remaining <= 0 or last.get("capture_in_progress"):
        raise ValueError("The existing motor capture ledger has no confirmed spare time")
    receipt = root / "rehearsal.json"
    previous = previous_spend(budget)
    available_mib = last["storage_cap_mib"] - math.ceil(previous["preserved_raw_bytes"] / MIB)
    recipe = {**visual_binding(COMPACT), "stock_capture_ledger_sha256": sha(budget), "source_counts": COUNTS,
              "generation_cap_s": min(remaining, GENERATION_CEILING_S), "export_cap_s": EXPORT_CEILING_S,
              "stock_elapsed_s": last["generation_elapsed_s"], "storage_cap_mib": available_mib,
              "command_execution_profile": FLOAT32_COMMAND_PROFILE, "previous_attempt": previous}
    report = (load(receipt) if receipt.is_file() else
              {"recipe": recipe, "generation_elapsed_s": previous["failed_generation_s"],
               "export_elapsed_s": previous["failed_export_upper_bound_s"], "splits": {}})
    if report["recipe"] != recipe:
        raise ValueError("The existing compact rehearsal has another capture budget or implementation")
    interrupted = report.pop("capture_started_unix", None)
    if interrupted is not None:
        report["generation_elapsed_s"] += max(0., time.time() - interrupted)
    interrupted_export = report.pop("export_started_unix", None)
    if interrupted_export is not None:
        report["export_elapsed_s"] += max(0., time.time() - interrupted_export)
    atomic(receipt, report)
    return receipt, report


def capture(config, raw, receipt, report, heartbeat):
    from smart_swarm.motor import generate

    for split, amount in COUNTS.items():
        if report["splits"].get(split, {}).get("status") != "complete":
            recipe = report["recipe"]
            remaining = int(recipe["generation_cap_s"] - report["generation_elapsed_s"])
            if remaining < 1:
                raise TimeoutError("The compact capture exhausted its portion of the original wall budget")
            started = time.monotonic()
            report["capture_started_unix"] = time.time()
            atomic(receipt, report)
            try:
                report["splits"][split] = generate(config["asset_root"], raw, split, amount, remaining,
                                                   recipe["storage_cap_mib"], heartbeat, visual_profile=COMPACT)
            finally:
                report["generation_elapsed_s"] += time.monotonic() - started
                report.pop("capture_started_unix")
                atomic(receipt, report)
            if report["splits"][split]["status"] != "complete":
                raise ValueError(f"The compact capture did not complete its {split} rehearsal cohort")


def export(raw, dataset, receipt, report, heartbeat):
    if verified_export(raw, dataset, "--clocked"):
        return
    remaining = int(report["recipe"]["export_cap_s"] - report["export_elapsed_s"])
    if remaining < 1:
        raise TimeoutError("The compact export exhausted its portion of the original wall budget")
    started = time.monotonic()
    report["export_started_unix"] = time.time()
    atomic(receipt, report)
    try:
        official_export(raw, dataset, "--clocked", receipt.with_name("export.log"), heartbeat, remaining)
    finally:
        report["export_elapsed_s"] += time.monotonic() - started
        report.pop("export_started_unix")
        atomic(receipt, report)


def compare_issue(world, engine, marker, index, arrays, expected):
    from export_lerobot import KEYS
    from smart_swarm.evaluate_motor import action_request
    from smart_swarm.motor_input import CLOCKED_PROFILE

    now = int(arrays["timestamps_ns"][index])
    actual = action_request(world, CLOCKED_PROFILE, marker, issue_index=engine.observation_index())
    pixels = dict(zip(KEYS, actual["images"], strict=True))
    if (world.ns != now or engine.observation_index() != index or actual["task"] != expected["task"]
            or not np.array_equal(np.asarray(actual["state"], dtype=np.float32), expected["observation.state"])
            or not all(b64decode(pixels[name]) == expected[key].tobytes() for name, key in KEYS.items())):
        raise ValueError(f"The compact observation and same-issue export disagree at {index}")
    return {"issue": index, "at_ns": now, "camera_order": list(KEYS), "task": actual["task"],
            "state_bit_exact": True, "rgb_bit_exact": True, "recorded_request_native": expected["action"].tolist()}


def replay_source(assets, entry):
    from export_lerobot import make_frame
    from smart_swarm.arm_executor import ArmExecutor
    from smart_swarm.motor import ARM
    from smart_swarm.motor_diagnostics import contact_valid_placement, observe, read_teacher, reset_world, summarize
    from smart_swarm.motor_input import CLOCKED_PROFILE

    path = Path(entry["raw_path"])
    meta, arrays, _ = read_teacher(path)
    match_visual(meta, visual_binding(COMPACT))
    if meta.get("command_execution_profile") != FLOAT32_COMMAND_PROFILE:
        raise ValueError(f"Exact issue-time replay requires the shared float32 command: {path}")
    traced, feedback = [], []
    with np.load(path / "motor.npz", allow_pickle=False) as archive, closing(reset_world(assets, meta)) as world:
        view = {key: archive[key] for key in ("wrist", "top", "state", "requested")}
        engine = ArmExecutor(world, "aligned-sample-v2")
        engine.prime()
        for index, command in enumerate(arrays["requested"]):
            state = np.asarray(world.state(ARM)[0], dtype=np.float32)
            applied = world.arm_from_mj(world.filtered).astype(np.float32)
            if (world.ns != arrays["timestamps_ns"][index] or not np.array_equal(state, arrays["state"][index])
                    or not np.array_equal(applied, arrays["applied_previous"][index])):
                raise ValueError(f"The compact deployment state, previous action or issue clock diverges at {index}")
            if index in TRACE_ROWS:
                expected = make_frame(view, meta, index, CLOCKED_PROFILE)
                traced.append(compare_issue(world, engine, meta["target_marker"], index, arrays, expected))
            engine.submit(command)
            if index % 15 == 14:
                feedback.append(observe(world, index + 1))
        result = summarize(feedback)
        if not contact_valid_placement(result) or len(traced) != len(TRACE_ROWS):
            raise ValueError("The compact replay did not preserve the complete contact-valid teacher trajectory")
    return {"source": str(path), "source_sha256": meta["sha256"], "marker": meta["target_marker"],
            "trial": result, "issue_time_trace": traced, "same_issue_state_rows": len(arrays["state"])}


def verify(config, dataset, heartbeat):
    splits = {name: load(dataset / name / "smart-swarm-lineage.json") for name in COUNTS}
    match_visual(*splits.values(), visual_binding(COMPACT))
    output = []
    for split, lineage in splits.items():
        if len(lineage["episodes"]) != COUNTS[split] or not lineage["episode_padding_mask_checked"]:
            raise ValueError("The official compact export changed the prescribed split membership")
        for row in lineage["episodes"]:
            heartbeat()
            output.append({"split": split, **replay_source(config["asset_root"], row)})
    return output


def run(config, root, grant, heartbeat):
    root.mkdir(exist_ok=True, parents=True)
    receipt, report = open_report(root)
    if report.get("status") == "complete":
        return receipt
    raw = root / "raw"
    dataset = root / "lerobot"
    report.update(status="running", reservation=grant, raw=str(raw), dataset=str(dataset), optimizer_updates=0)
    atomic(receipt, report)
    try:
        capture(config, raw, receipt, report, heartbeat)
        export(raw, dataset, receipt, report, heartbeat)
        report.update(status="complete", replays=verify(config, dataset, heartbeat))
    except (OSError, RuntimeError, ValueError, KeyboardInterrupt) as error:
        report.update(status="incomplete", reason=str(error) or type(error).__name__)
        raise
    finally:
        atomic(receipt, report)
    return receipt


def archived_smoke(config):
    from learning_pilot import MOTOR_RAW
    from smart_swarm.arm_executor import ArmExecutor
    from smart_swarm.motor import ARM
    from smart_swarm.motor_diagnostics import read_teacher, reset_world

    source = MOTOR_RAW / "train/so101-10000"
    benchmark = EVIDENCE / "combined-motor-render-dense.json"
    measured = load(benchmark)
    meta, arrays, _ = read_teacher(source)
    if measured["dataset_sha256"] != meta["sha256"]:
        raise ValueError("The archived dense renderer used a different teacher")
    groups = {idx: [row["sha256"] for row in measured["compared_rgb"] if row["control_issue"] == idx] for idx in (0, 60)}
    binding = visual_binding(COMPACT)
    verified = []
    with closing(reset_world(config["asset_root"], {**meta, **binding})) as world:
        issue = ArmExecutor(world, "aligned-sample-v2")
        issue.prime()
        for index, command in enumerate(arrays["requested"][:61]):
            if index in groups:
                before = world.snapshot().integration
                images = world.views(ARM)
                hashes = [sha256(frame.tobytes()).hexdigest() for frame in images.values()]
                if hashes != groups[index] or not np.array_equal(before, world.snapshot().integration):
                    raise ValueError("The production drawing differed from the archived probe or changed solver state")
                if world.renderer.model is world.model or world.ns != int(arrays["timestamps_ns"][index]):
                    raise ValueError("Compact drawing must preserve the separate physics model and issue clock")
                verified.append({"issue": index, "camera_order": list(images), "rgb_sha256": hashes})
            issue.submit(command)
    return {"scope": "two camera issues on one archived source; local software-render smoke check",
            "source_sha256": meta["sha256"], "prior_benchmark_sha256": sha(benchmark), "issues": verified,
            **binding, "original_source_unchanged": True, "new_training_sources": 0, "optimizer_updates": 0}


def compare_archived_commands(assets, path):
    from smart_swarm.arm_executor import ArmExecutor
    from smart_swarm.motor import ARM, record_insertion
    from smart_swarm.motor_diagnostics import contact_valid_placement, observe, read_teacher, reset_world, summarize

    meta, previous, _ = read_teacher(path.parent)
    issued, samples, feedback = [], [], []
    with closing(reset_world(assets, meta)) as world:
        reset = world.snapshot()

        def capture(_phase, _index, _start, command, applied):
            issued.append(command)
            samples.append((world.ns, world.state(ARM)[0], applied))
            if len(issued) % 15 == 0:
                feedback.append(observe(world, len(issued)))

        record_insertion(world, capture, meta["physics"]["pickup_xy_m"])
        if not np.array_equal(np.asarray(issued, dtype=np.float32), previous["requested"]):
            raise ValueError("This diagnostic requires the original recorded request values")
        world.restore(reset)
        executor = ArmExecutor(world, "aligned-sample-v2")
        executor.prime()
        observed = []
        for index, (command, (at, state, applied)) in enumerate(zip(previous["requested"], samples, strict=True)):
            if (world.ns != at or not np.array_equal(world.state(ARM)[0], state)
                    or not np.array_equal(world.arm_from_mj(world.filtered), applied)):
                raise ValueError(f"Exact capture and replay differ at issue {index} in {path.parent.name}")
            if index % 15 == 14:
                observed.append(observe(world, index + 1))
            executor.submit(command)
        if not all(contact_valid_placement(summarize(rows)) for rows in (feedback, observed)):
            raise ValueError(f"Contact-valid placement did not carry over in {path.parent.name}")
    return {"source": path.parent.name, "original_raw_sha256": meta["sha256"], "recorded_command_rows": len(issued),
            "issue_clocks_exact": True, "preceding_applied_exact": True, "proprioception_exact": True,
            "capture_and_replay_contact_valid": True}


def archived_command_replay(config):
    paths = sorted((PREVIOUS / "raw").glob("*/*/manifest.json"))
    if len(paths) != sum(COUNTS.values()):
        raise ValueError("The exact three original compact trajectories are required")
    return {"scope": "Command and physics replay with the exact archived float32 commands",
            "sources": [compare_archived_commands(config["asset_root"], path) for path in paths],
            "command_execution_profile": FLOAT32_COMMAND_PROFILE, "captured_rgb_frames": 0,
            "recording_started": False, "optimizer_updates": 0, "new_training_sources": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    checks = parser.add_mutually_exclusive_group()
    checks.add_argument("--archived-smoke", action="store_true", help="Compare two software-RGB issues with the old benchmark")
    checks.add_argument("--archived-command-replay", action="store_true", help="Compare all command issues without rendering")
    args = parser.parse_args()
    config = settings()
    configure_renderer(config)
    if args.archived_smoke or args.archived_command_replay:
        if args.output.exists():
            raise ValueError("Choose a new compact smoke report path")
        check = archived_smoke if args.archived_smoke else archived_command_replay
        atomic(args.output, check(config))
        print(args.output)
        return
    if args.output.resolve() != LOCATION:
        raise ValueError(f"Use the budgeted compact rehearsal directory {LOCATION}")
    with admitted(config, "smart-swarm-compact-motor-rehearsal") as (grant, heartbeat):
        print(run(config, args.output.resolve(), grant, heartbeat))


if __name__ == "__main__":
    main()
