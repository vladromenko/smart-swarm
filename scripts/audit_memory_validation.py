#!/usr/bin/env python3
"""Audit independent groups and per-camera supervision before viewpoint calibration."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from capture_memory_validation import DEFAULT_OUTPUT, validate_group, validation_slots
from smart_swarm.episode import LABELS, atomic, load, rows


def accepted_sources(folder, allocation, only=None, *, verify=True):
    seen = set()
    for path in sorted(folder.glob("*/*/capture.json")):
        report = load(path)
        if report["status"] == "accepted" and (only is None or report["source_group"] in only):
            group = report["source_group"]
            if group not in allocation or group in seen:
                raise ValueError(f"Unreserved or repeated recording source {group}")
            slot = allocation[group]
            if report["seed"] not in slot["seed_candidates"]:
                raise ValueError(f"Unreserved seed for {group}")
            if verify:
                validate_group(slot, report["seed"], report["teams"])
            seen.add(group)
            yield report


def count_camera_labels(report, counts, world_ids):
    group = report["source_group"]
    for team in report["teams"]:
        for trace in team["traces"]:
            root = Path(trace["path"])
            manifest = load(root / "manifest.json")
            world_id = manifest["world_episode_id"]
            if world_id in world_ids:
                raise ValueError(f"Duplicate validation world identity {world_id}")
            world_ids.add(world_id)
            label = rows(root / LABELS / "settled-event.jsonl")[0]
            for agent, cameras in manifest["policy_cameras"].items():
                if cameras and agent != "rosmaster-1":
                    field = "settled_placement" if agent == "so101-1" else "relocation_assignment"
                    key = f"N{team['actor_count']}/{agent}/{field}"
                    name = "masked" if label[field] is None else label[field]
                    counts[key][name].add(group)


def report(folder):
    allocation = {slot["source_group"]: slot for slot in validation_slots()}
    complete = list(accepted_sources(folder, allocation))
    world_ids = set()
    labels = defaultdict(lambda: defaultdict(set))
    for record in complete:
        count_camera_labels(record, labels, world_ids)
    groups = {record["source_group"] for record in complete}
    return {"scope": "sealed simulation validation only", "accepted_independent_groups": len(groups),
            "recorded_variants_across_teams": len(world_ids), "accepted_sources": sorted(groups),
            "reserved_groups": len(allocation), "allocation_filled": len(groups) == len(allocation),
            "family_counts": dict(Counter(allocation[name]["family"] for name in groups)),
            "groups_by_camera_event_label": {
                key: {label: len(items) for label, items in classes.items()} for key, classes in labels.items()},
            "unrecorded_sources": sorted(set(allocation) - groups)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = report(args.root)
    if args.output:
        atomic(args.output, result)
    print(result["accepted_independent_groups"], "accepted of", result["reserved_groups"], "reserved source groups")


if __name__ == "__main__":
    main()
