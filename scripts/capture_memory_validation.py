#!/usr/bin/env python3
"""Acquire one reserved validation group using the fixed-total simulated team profile."""

import argparse
from pathlib import Path

from plan_primary_memory import CONFIG, ROOT, plan
from rehearse_scaling import one_team
from smart_swarm.cli import settings
from smart_swarm.episode import atomic, load, sha, validate
from smart_swarm.rendering import configure_renderer
from smart_swarm.research_jobs import admitted

ALLOCATION = ROOT / "evidence/learning-pilot-v3/primary-sim-source-allocation.json"
DEFAULT_OUTPUT = ROOT.parent.parent / "work/primary-sim-validation"


def allocated_slots(split):
    if split not in ("validation", "test"):
        raise ValueError(f"Unknown primary split {split}")
    stored = load(ALLOCATION)
    if stored != (plan(load(CONFIG)) | {"config_sha256": sha(CONFIG)}):
        raise ValueError("The prospective source allocation changed")
    return [slot for slot in stored["slots"] if slot["split"] == split]


def validation_slots():
    return allocated_slots("validation")


def load_slot(family, index, split="validation"):
    match = [slot for slot in allocated_slots(split) if slot["family"] == family and slot["group_index"] == index]
    if len(match) != 1:
        raise ValueError(f"Requested group is outside the reserved {split} allocation")
    return match[0]


def validate_variants(slot, seed, teams):
    seen = set()
    for team in teams:
        traces = team["traces"]
        if [row["variant"] for row in traces] != slot["capture_variants"]:
            raise ValueError("A source group needs all planned variants, in their reserved order")
        for row in traces:
            root = Path(row["path"])
            manifest = load(root / "manifest.json")
            prefix = f"{slot['source_group']}-{seed}-"
            identity = manifest["world_episode_id"]
            if (identity in seen or not identity.startswith(prefix)
                    or manifest["scene_group_id"] != slot["source_group"] or manifest["lineage"]["split"] != slot["split"]
                    or not row["family"] == manifest["family"] == slot["family"]
                    or row["variant"] != manifest["evidence_variant"]
                    or not row["team_size"] == manifest["scaling_team_size"] == team["actor_count"]
                    or manifest["simulation_recipe_seed"] != seed):
                raise ValueError(f"A recording escaped its reserved {slot['split']} lineage, team or variant")
            seen.add(identity)
            validate(root, manifest)


def validate_group(slot, seed, teams):
    from smart_swarm.scaling import matching_budgets

    if [team["actor_count"] for team in teams] != slot["paired_team_sizes"]:
        raise ValueError("A source group needs all planned team sizes")
    validate_variants(slot, seed, teams)
    if slot["family"] == "F6" and not all(team["f6_full_input_invariance"] is True for team in teams):
        raise ValueError("A paired source must pass its complete permitted-input audit")
    for matches in zip(*(team["traces"] for team in teams), strict=True):
        if len({row["terminal_outcome"] for row in matches}) != 1:
            raise ValueError("Team size changed the sealed physical outcome")
        if matches[0]["family"] == "F4":
            expected = "left" if seed // 4 % 2 else "right"
            if matches[0]["terminal_outcome"] != expected:
                raise ValueError("Screened F4 outcome disagrees with the preallocated geometry")
        matching_budgets(matches)


def capture_candidate(config, path, slot, seed, grant, heartbeat):
    receipt = path / "capture.json"
    status = {"status": "running", "source_group": slot["source_group"], "seed": seed,
              "reservation": grant, "teams": []}
    path.mkdir()
    atomic(receipt, status)
    try:
        lineage = {"split": slot["split"], "source_group": slot["source_group"]}
        for size in slot["paired_team_sizes"]:
            heartbeat()
            team = one_team(config["asset_root"], path, size, slot["family"], seed,
                            receiver_history=True, lineage=lineage)
            status["teams"].append(team)
            atomic(receipt, status)
        validate_group(slot, seed, status["teams"])
        status["status"] = "accepted"
    except KeyboardInterrupt:
        status.update(status="interrupted", reason="Operator or owning process interrupted simulation")
        raise
    except (TimeoutError, ValueError) as error:
        status.update(status="rejected", reason=str(error))
    finally:
        atomic(receipt, status)
    return status


def acquire(config, root, slot, grant, heartbeat):
    configure_renderer(config)
    group = root / slot["source_group"]
    group.mkdir(parents=True, exist_ok=True)
    for seed in slot["seed_candidates"]:
        attempt = group / str(seed)
        receipt = attempt / "capture.json"
        if not attempt.exists():
            trial = capture_candidate(config, attempt, slot, seed, grant, heartbeat)
            if trial["status"] == "accepted":
                return trial
        elif receipt.is_file():
            trial = load(receipt)
            if trial["status"] == "accepted":
                validate_group(slot, seed, trial["teams"])
                return trial
            if trial["status"] == "running":
                atomic(receipt, {**trial, "status": "interrupted", "reason": "Previous owner exited before commit"})
    raise ValueError("This source used all five reserved seed candidates")


def pending_slots(root, limit, split="validation"):
    allocated = allocated_slots(split)
    if not 1 <= limit <= len(allocated):
        raise ValueError(f"Choose between 1 and {len(allocated)} {split} groups")
    accepted = set()
    for path in root.glob("*/*/capture.json"):
        receipt = load(path)
        if receipt["status"] == "accepted":
            accepted.add(receipt["source_group"])
    ordered = sorted(allocated, key=lambda group: (group["group_index"], group["family"]))
    return [slot for slot in ordered if slot["source_group"] not in accepted][:limit]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=tuple(f"F{index}" for index in range(1, 7)))
    parser.add_argument("--index", type=int)
    parser.add_argument("--next", type=int, metavar="COUNT", help="Acquire the next unfilled reserved groups in order")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    explicit = args.family is not None and args.index is not None
    batch = args.next is not None and args.family is None and args.index is None
    if not (batch or explicit and args.next is None):
        parser.error("Choose --family with --index, or --next COUNT")
    slots = pending_slots(args.output, args.next) if batch else [load_slot(args.family, args.index)]
    config = settings()
    for slot in slots:
        with admitted(config, slot["source_group"]) as (grant, heartbeat):
            result = acquire(config, args.output, slot, grant, heartbeat)
        print(result["source_group"], result["status"], result["seed"], flush=True)
    if not slots:
        print("All reserved validation groups are filled")


if __name__ == "__main__":
    main()
