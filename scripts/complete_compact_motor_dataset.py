#!/usr/bin/env python3
"""Complete the compact 80/20 source split after its real round-trip gate."""

import argparse
from collections import Counter
from contextlib import ExitStack, closing, contextmanager
import math
from pathlib import Path
import signal
import time

from learning_pilot import EVIDENCE, MOTOR_RAW, official_export, verified_export
from rehearse_compact_motor import COUNTS, LOCATION, TRACE_ROWS, previous_spend
from smart_swarm.cli import ROOT, settings
from smart_swarm.episode import atomic, load, sha
from smart_swarm import motor_draw_cache
from smart_swarm.motor_input import CLOCKED_PROFILE, FLOAT32_COMMAND_PROFILE
from smart_swarm.motor_visual_profile import COMPACT, match_visual, visual_binding
from smart_swarm.rendering import configure_renderer
from smart_swarm.research_jobs import admitted

TARGET = {"train": 80, "validation": 20}
ORIGINAL_EXPORT_CAP_S = 1_800
MIB = 1024**2
DRAW_CACHE_PROOF = EVIDENCE / "compact-draw-cache-verification.json"
MODEL_CACHE_PROOF = EVIDENCE / "compact-model-cache-verification.json"


def verified_draw_cache():
    report = load(DRAW_CACHE_PROOF)
    if (report["status"] != "verified" or report["cache_profile"] != motor_draw_cache.PROFILE
            or report["cache_source_sha256"] != sha(Path(motor_draw_cache.__file__))
            or not report["fresh_physics_for_each_source"]):
        raise ValueError("The compact draw cache needs exact admitted replay verification")
    match_visual(report["visual"], visual_binding(COMPACT))
    cases = report["cases"]
    if len(cases) != 2 or {row["checked"]["marker"] for row in cases} != {"cyan", "violet"}:
        raise ValueError("The draw cache verification must cover both requested markers")
    for row in cases:
        checked = row["checked"]
        if (checked["exact_camera_and_control_rows"] != 510 or not checked["contact_valid"]
                or checked["sha256"] != load(Path(checked["source"]) / "manifest.json")["sha256"]):
            raise ValueError("The verified render trace no longer matches an admitted source")
    return sha(DRAW_CACHE_PROOF)


def verified_model_cache():
    report = load(MODEL_CACHE_PROOF)
    if report["status"] != "verified" or report["implementation_sha256"] != sha(ROOT / "src/smart_swarm/motor_model_cache.py"):
        raise ValueError("A compiled model copy must match its verified implementation")
    match_visual(report["visual"], visual_binding(COMPACT))
    cases = report["cases"]
    if len(cases) != 2 or {row["marker"] for row in cases} != {"cyan", "violet"}:
        raise ValueError("Both original motor layouts must qualify for compiled model copies")
    for row in cases:
        if (row["exact_rgb_and_controls"] != 510 or not row["independent_physics"]
                or row["sha256"] != load(Path(row["source"]) / "manifest.json")["sha256"]):
            raise ValueError("The exact model verification no longer matches an admitted source")
    return sha(MODEL_CACHE_PROOF)


@contextmanager
def capture_caches(config, draw_proof, model_proof):
    with ExitStack() as stack:
        draw = stack.enter_context(closing(motor_draw_cache.CompactDrawCache())) if draw_proof else None
        models = None
        if model_proof:
            from smart_swarm.motor import IMAGE_SIZE
            from smart_swarm.motor_model_cache import CompactModelCache

            models = stack.enter_context(closing(CompactModelCache(config["asset_root"], IMAGE_SIZE)))
        yield {"draw_cache": draw, "model_cache": models}


class StudyBudgetExpired(RuntimeError):
    """The original capture budget was exhausted."""


def pilot_receipts(root):
    pilot = root / "rehearsal.json"
    if not pilot.is_file():
        raise ValueError("The reserved three-trajectory round-trip has not published a receipt")
    report = load(pilot)
    if report.get("status") != "complete" or report.get("optimizer_updates") != 0:
        raise ValueError("The three-trajectory round-trip must pass before expanding the compact split")
    if Path(report["raw"]).resolve() != (root / "raw").resolve() or Path(report["dataset"]).resolve() != (root / "lerobot").resolve():
        raise ValueError("The original raw and official pilot paths must remain in this root")
    match_visual(report["recipe"], visual_binding(COMPACT))
    if report["recipe"]["command_execution_profile"] != FLOAT32_COMMAND_PROFILE:
        raise ValueError("The compact rehearsal must use the exact exported command representation")
    rows = report["replays"]
    if dict(Counter(row["split"] for row in rows)) != COUNTS or {row["marker"] for row in rows} != {"cyan", "violet"}:
        raise ValueError("The pilot requires three verified sources across the declared splits and both target markers")
    for row in rows:
        issue_times = row["issue_time_trace"]
        if ([sample["issue"] for sample in issue_times] != list(TRACE_ROWS)
                or row["same_issue_state_rows"] != 510
                or not all(sample["rgb_bit_exact"] and sample["state_bit_exact"] for sample in issue_times)):
            raise ValueError("A compact teaching replay lacks its exact acquisition-to-policy input check")
        result = row["trial"]
        if not result["stable_containment"] or not result["held_lift_10cm"] or result["first_failed_stage"] is not None:
            raise ValueError("A compact teaching replay did not pass the contact-valid placement gate")
    return report, sha(pilot)


def match_pilot_export(root, report):
    seen = set()
    receipts = {str(Path(row["source"]).resolve()): row for row in report["replays"]}
    for split, expected in COUNTS.items():
        lineage = load(root / "lerobot" / split / "smart-swarm-lineage.json")
        match_visual(lineage, report["recipe"])
        if (len(lineage["episodes"]) != expected or lineage["input_profile"] != CLOCKED_PROFILE
                or lineage["episode_padding_mask_checked"] is not True):
            raise ValueError("The saved official rehearsal split changed after round-trip validation")
        for episode in lineage["episodes"]:
            source = Path(episode["raw_path"]).resolve()
            if not source.is_relative_to((root / "raw" / split).resolve()) or str(source) not in receipts:
                raise ValueError("The official pilot split names an unverified source")
            meta, proof = load(source / "manifest.json"), receipts[str(source)]
            match_visual(meta, lineage)
            digest = sha(source / "motor.npz")
            if (proof["split"] != split or meta["lineage"]["split"] != split or meta["status"] != "accepted"
                    or not digest == episode["source_sha256"] == proof["source_sha256"] == meta["sha256"]
                    or meta["scene_group_id"] in seen or proof["marker"] != meta["target_marker"]):
                raise ValueError("The official compact pilot source changed or crossed splits")
            seen.add(meta["scene_group_id"])
    return len(seen)


def budget_recipe(root, pilot, pilot_sha):
    old = EVIDENCE / "motor-full-capture.json"
    digest = sha(old)
    stock = load(old)
    if stock.get("capture_in_progress") or pilot["recipe"]["stock_capture_ledger_sha256"] != digest:
        raise ValueError("The original motor acquisition ledger changed after the compact round-trip")
    used = pilot["generation_elapsed_s"]
    previous = previous_spend(old)
    if (not math.isfinite(used) or not 0 <= used <= pilot["recipe"]["generation_cap_s"]
            or pilot["recipe"]["previous_attempt"] != previous or used < previous["failed_generation_s"]
            or not previous["failed_export_upper_bound_s"] <= pilot["export_elapsed_s"] <= pilot["recipe"]["export_cap_s"]
            or stock["rollout_timeout_s"] != 100 or stock["max_attempts_per_requested"] != 5):
        raise ValueError("The compact round-trip time or inherited limits differ from the original allocation")
    old_files = list(MOTOR_RAW.glob("*/*/motor.npz"))
    if len(old_files) < sum(row["accepted"] for row in stock["splits"].values()):
        raise ValueError("The original accepted motor archives are missing from the storage accounting")
    old_bytes = previous["preserved_raw_bytes"]
    storage = int(stock["storage_cap_mib"] - math.ceil(old_bytes / MIB))
    if storage < 64:
        raise ValueError("The original raw budget cannot fit another motor trajectory")
    return {**visual_binding(COMPACT), "stock_receipt_sha256": digest, "pilot_receipt_sha256": pilot_sha,
            "target_counts": TARGET, "raw": str(root / "raw"), "dataset": str(root / "lerobot-full"),
            "stock_elapsed_s": stock["generation_elapsed_s"], "round_trip_elapsed_s": used,
            "generation_cap_s": stock["wall_cap_s"], "raw_storage_cap_mib": stock["storage_cap_mib"],
            "preserved_raw_bytes": old_bytes, "compact_storage_cap_mib": storage,
            "original_export_cap_s": ORIGINAL_EXPORT_CAP_S, "round_trip_export_ceiling_s": pilot["recipe"]["export_cap_s"]}


def remaining(record, stage):
    recipe = record["recipe"]
    if stage == "generation":
        used = recipe["stock_elapsed_s"] + recipe["round_trip_elapsed_s"] + record["spent_s"][stage]
        return recipe["generation_cap_s"] - used
    used = recipe["round_trip_export_ceiling_s"] + record["spent_s"][stage]
    return recipe["original_export_cap_s"] - used


def open_record(path, recipe):
    record = (load(path) if path.is_file() else
              {"recipe": recipe, "spent_s": {"generation": 0., "export": 0.}, "splits": {}, "interruptions": []})
    if record["recipe"] != recipe:
        raise ValueError("The prepared full cohort has a different original gate or budget")
    active = record.pop("active", None)
    if active is not None:
        stage = active["stage"]
        duration = min(max(0., time.time() - active["wall_started_unix"]), active["available_s"])
        record["spent_s"][stage] += duration
        record["interruptions"].append({**active, "charged_s": duration})
    atomic(path, record)
    return record


def expired(_signal, _frame):
    raise StudyBudgetExpired("The compact study reached the original acquisition wall cap")


@contextmanager
def phase(path, record, stage, scope):
    available = remaining(record, stage)
    if available < 1:
        raise StudyBudgetExpired(f"The original {stage} allocation has no time remaining")
    started = time.monotonic()
    handler = None
    record["active"] = {"stage": stage, "scope": scope, "available_s": available, "wall_started_unix": time.time()}
    atomic(path, record)
    try:
        if stage == "generation":
            handler = signal.signal(signal.SIGALRM, expired)
            signal.setitimer(signal.ITIMER_REAL, available)
        yield int(available)
    finally:
        if handler is not None:
            signal.setitimer(signal.ITIMER_REAL, 0.)
            signal.signal(signal.SIGALRM, handler)
        record["spent_s"][stage] += time.monotonic() - started
        record.pop("active", None)
        atomic(path, record)


def actual_inventory(raw):
    from smart_swarm.datasets import audit_lineage

    records = []
    groups = Counter()
    binding = visual_binding(COMPACT)
    for path in raw.glob("*/*/manifest.json"):
        row = load(path)
        split = row["lineage"]["split"]
        if (split not in TARGET or row["status"] != "accepted" or path.parent.parent.name != split
                or row.get("command_execution_profile") != FLOAT32_COMMAND_PROFILE
                or path.parent.name != row["world_episode_id"] or not path.resolve().is_relative_to(raw.resolve())):
            raise ValueError("The compact study raw root contains an unexpected recording or split")
        match_visual(row, binding)
        records.append(row)
        groups[split] += 1
    keys = ("scene_group_id", "seed", "sha256")
    if any(len({row[key] for row in records}) != len(records) for key in keys):
        raise ValueError("The 80/20 motor cohort requires independent sources")
    if any(groups[split] > target for split, target in TARGET.items()):
        raise ValueError("The compact study raw inventory exceeds a declared split size")
    audit_lineage(records)
    return {split: groups[split] for split in TARGET}


def capture(config, raw, path, record, heartbeat, max_blocks, caches):
    from smart_swarm.motor import generate, interrupted_summary

    sizes = actual_inventory(raw)
    goals = ({"train": block * 8, "validation": block * 2} for block in range(1, 11))
    pending = [group for group in goals if any(sizes[split] < amount for split, amount in group.items())]
    for group in pending[:max_blocks]:
        for split, amount in group.items():
            if sizes[split] < amount:
                with phase(path, record, "generation", {"split": split, "goal": amount}) as budget:
                    try:
                        receipt = generate(config["asset_root"], raw, split, amount, budget,
                                           record["recipe"]["compact_storage_cap_mib"], heartbeat,
                                           visual_profile=COMPACT, **caches)
                    except (StudyBudgetExpired, KeyboardInterrupt):
                        record["splits"][split] = interrupted_summary(raw, split, amount)
                        raise
                    record["splits"][split] = receipt
                    sizes[split] = receipt["accepted"]
                    if sizes[split] != amount:
                        raise StudyBudgetExpired(f"Compact {split} reached {sizes[split]} of {amount} under its existing limits")
    if actual_inventory(raw) != sizes:
        raise ValueError("The compact source inventory differs from the recorded split counts")
    return sizes


def finish(config, root, path, record, grant, heartbeat, max_blocks, caches):
    raw, destination = root / "raw", root / "lerobot-full"
    record.update(status="running", reservation=grant, optimizer_updates=0)
    record.pop("reason", None)
    atomic(path, record)
    try:
        if capture(config, raw, path, record, heartbeat, max_blocks, caches) != TARGET:
            record.update(status="capture_pending")
            return path
        if not verified_export(raw, destination, "--clocked"):
            with phase(path, record, "export", "official-full-splits") as budget:
                official_export(raw, destination, "--clocked", root / "full-export.log", heartbeat, budget)
        record.update(status="dataset_ready", dataset=str(destination), source_counts=TARGET)
    except (OSError, ValueError, RuntimeError, KeyboardInterrupt) as error:
        record.update(status="incomplete", reason=str(error) or type(error).__name__)
        raise
    finally:
        save_inventory(raw, path, record)
    return path


def save_inventory(raw, path, record):
    try:
        record["accepted_sources"] = actual_inventory(raw)
    except (OSError, ValueError, KeyError) as error:
        record.update(status="incomplete", inventory_error=str(error))
        record.pop("dataset", None)
    atomic(path, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Verify the three-source gate and remaining budgets")
    parser.add_argument("--max-blocks", type=int, choices=range(1, 11), default=10,
                        help="Release the native reservation after this many pending split blocks")
    parser.add_argument("--reuse-draw-contexts", action="store_true", help="Use verified matching compact draw contexts")
    parser.add_argument("--reuse-compiled-models", action="store_true", help="Copy verified pristine compiled compact geometry")
    args = parser.parse_args()
    try:
        pilot, digest = pilot_receipts(LOCATION)
        match_pilot_export(LOCATION, pilot)
        recipe = budget_recipe(LOCATION, pilot, digest)
        draw_proof = verified_draw_cache() if args.reuse_draw_contexts else None
        model_proof = verified_model_cache() if args.reuse_compiled_models else None
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"Compact motor expansion withheld. {error}\n")
    proofs = {"draw_cache_verification_sha256": draw_proof, "model_cache_verification_sha256": model_proof}
    if args.check_only:
        print({**recipe, **proofs})
        return
    config = settings()
    configure_renderer(config)
    with admitted(config, "smart-swarm-compact-80-20-capture") as (grant, heartbeat):
        path = LOCATION / "full-capture.json"
        record = open_record(path, recipe)
        admission = {**grant, **proofs}
        record.setdefault("capture_admissions", []).append(admission)
        with capture_caches(config, draw_proof, model_proof) as caches:
            print(finish(config, LOCATION, path, record, admission, heartbeat, args.max_blocks, caches))


if __name__ == "__main__":
    main()
