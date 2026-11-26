#!/usr/bin/env python3
"""Publish grouped source recipes before opening primary outcomes."""

import argparse
from pathlib import Path

from smart_swarm.episode import atomic, load, sha

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "primary-memory-study.json"
SUBSET_COUNTS = {"F1": 9, "F2": 8, "F3": 8, "F4": 8, "F5": 9, "F6": 8}


def group_slot(split, family, index, offset, config):
    seed = config["seed_bases"][split] + 1024 * offset + 48 * index
    phase = (index % 2) * 4 + ((index // 2) % 2)
    alternatives = [seed + phase + 8 * repeat for repeat in range(config["max_attempts_per_source"])]
    variants = {"F4": ["screened-association-v2"], "F6": ["hidden-contained", "hidden-surface"]}
    return {"source_group": f"primary-{split}-{family.lower()}-{index:02d}", "family": family,
            "group_index": index, "split": split, "seed_candidates": alternatives,
            "capture_variants": variants.get(family, ["witnessed"]),
            "paired_team_sizes": config["team_sizes"], "frozen_n3_ablation_subset": split == "test"
            and index < SUBSET_COUNTS[family]}


def plan(config):
    heldout = config["heldout_groups_per_team_by_family"]
    if sum(heldout.values()) != 100 or config["team_sizes"] != [3, 4, 8] or len(config["primary_methods"]) != 7:
        raise ValueError("The plan must retain the specified team sizes and primary matrix")
    if config["max_attempts_per_source"] != 5 or any(not 1 <= size <= 21 for size in heldout.values()):
        raise ValueError("Seed spacing no longer separates the configured source slots")
    partitions = {key: heldout if key == "test" else dict.fromkeys(heldout, config["validation_groups_per_family"])
                  for key in ("validation", "test")}
    slots = [group_slot(split, family, index, offset, config)
             for split, families in partitions.items() for offset, (family, size) in enumerate(sorted(families.items()))
             for index in range(size)]
    seeds = [seed for slot in slots for seed in slot["seed_candidates"]]
    if len(set(seeds)) != len(seeds) or sum(slot["frozen_n3_ablation_subset"] for slot in slots) != 50:
        raise ValueError("The allocation overlaps sources or changed the N=3 ablation subset")
    f4 = [slot for slot in slots if slot["family"] == "F4" and slot["split"] == "test"]
    if sum(slot["seed_candidates"][0] // 4 % 2 for slot in f4) * 2 != len(f4):
        raise ValueError("Screened F4 target layouts are imbalanced")
    return {"status": "allocated_unopened", "protocol": config["protocol"], "slots": slots,
            "source_counts": {split: sum(families.values()) for split, families in partitions.items()},
            "primary_world_runs": 100 * len(config["team_sizes"]),
            "primary_variant_runs": 3 * (100 + heldout["F6"]),
            "independence": "100 groups per team size. Across sizes the same 100 groups are paired, including F6 variants."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {**plan(load(CONFIG)), "config_sha256": sha(CONFIG)}
    if args.output.exists() and load(args.output) != result:
        raise ValueError("Existing source allocation differs. Choose a new versioned destination.")
    if not args.output.exists():
        atomic(args.output, result)
    print(args.output)


if __name__ == "__main__":
    main()
