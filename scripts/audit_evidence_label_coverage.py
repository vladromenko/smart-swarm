"""Recount source and label coverage from the frozen temporal evidence portfolio."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path

from smart_swarm.contract import json_bytes
from smart_swarm.episode import LABELS, TRUTH, atomic, load, rows, sha

LABEL_PATH = f"{LABELS}/settled-event.jsonl"
REVEAL_PATH = f"{TRUTH}/reveal.jsonl"
SPLITS = ("train", "validation", "test")
EVENTS = ("settled_placement", "relocation_assignment")


def bound_labels(record):
    root = Path(record["path"])
    path = root / "manifest.json"
    manifest = load(path)
    if (any(manifest[key] != record[key] for key in ("world_episode_id", "scene_group_id", "family"))
            or manifest["lineage"]["split"] != record["split"] or manifest["evidence_variant"] != record["variant"]
            or manifest["recording_integrity"] != "complete" or manifest["run_mode"] != "mujoco"):
        raise ValueError(f"{root.name} differs from its accepted portfolio lineage")
    for relative in (LABEL_PATH, REVEAL_PATH):
        if sha(root / relative) != manifest["sidecar_sha256"][relative]:
            raise ValueError(f"{root.name} has an altered event or reveal sidecar")
    label, = rows(root / LABEL_PATH)
    reveal, = rows(root / REVEAL_PATH)
    scored_at = manifest["decision_boundary"]
    observed_at = reveal["observed_at"]
    if (label["terminal"] != reveal or label["scope"] != "training_or_evaluation_only"
            or reveal["camera"]["access_role"] != "evaluator_only" or reveal["ground_truth"] != record["terminal"]
            or observed_at["clock_id"] != scored_at["clock_id"]
            or int(observed_at["monotonic_ns"]) <= int(scored_at["monotonic_ns"])):
        raise ValueError(f"{root.name} has an ineligible terminal label")
    if any(label[key] != record[key] for key in EVENTS) or label["resolvable"] != record["resolvable"]:
        raise ValueError(f"{root.name} label values differ from the frozen training record")
    if (label["intermediate_events"] is not None or record["intermediate_event_labels"] != 0
            or any((label[key] is None) != bool(label[reason])
                   for key, reason in zip(EVENTS, ("placement_masked_reason", "relocation_masked_reason"), strict=True))):
        raise ValueError(f"{root.name} changed its unknown or masked events")
    if record["family"] == "F6" and (manifest["policy_cameras"]["so101-1"] or label["settled_placement"] is not None):
        raise ValueError(f"{root.name} discloses the withheld placement")
    return sha(path)


def verify_groups(records):
    groups = defaultdict(list)
    for row in records:
        groups[row["scene_group_id"]].append(row)
    for source, siblings in groups.items():
        if len({(row["split"], row["family"]) for row in siblings}) != 1:
            raise ValueError(f"Source group {source} crosses a split or scenario")
        if siblings[0]["family"] == "F6":
            choices = {row["terminal"] for row in siblings}
            if (len(siblings) != 2 or {row["variant"] for row in siblings} != {"hidden-contained", "hidden-surface"}
                    or len(choices) != 2 or "surface" not in choices or not choices <= {"surface", "left", "right"}
                    or any(row["resolvable"] for row in siblings)):
                raise ValueError(f"Source group {source} is missing the paired hidden alternatives")
        elif len(siblings) != 1:
            raise ValueError(f"Source group {source} repeats a non-counterfactual history")
    return groups


def counts(records):
    return {"independent_histories": len({row["scene_group_id"] for row in records}), "variants": len(records),
            "known_events": {name: sum(row[name] is not None for row in records) for name in EVENTS},
            "masked_events": {name: sum(row[name] is None for row in records) for name in EVENTS},
            "terminal_labels": dict(Counter(row["terminal"] for row in records)),
            "intermediate_labels": sum(row["intermediate_event_labels"] for row in records)}


def audit(fit_path):
    fit = load(fit_path)
    path = Path(fit["dataset"])
    if fit["status"] != "complete" or sha(path) != fit["input_portfolio_sha256"]:
        raise ValueError("Use the portfolio bound to the completed learned-evidence job")
    portfolio = load(path)
    records = portfolio["records"]
    if (not portfolio["complete"] or len(records) != portfolio["recorded_variants"]
            or len({row["world_episode_id"] for row in records}) != len(records)):
        raise ValueError("The frozen portfolio has missing or repeated variants")
    groups = verify_groups(records)
    if (len(groups) != portfolio["accepted_independent"]
            or Counter(group[0]["split"] for group in groups.values()) != portfolio["groups_by_split"]):
        raise ValueError("Recounted independent histories differ from the recorded split")
    manifests = {row["world_episode_id"]: bound_labels(row) for row in records}
    families = sorted({row["family"] for row in records})
    split_rows = {split: [row for row in records if row["split"] == split] for split in SPLITS}
    by_split = {split: {**counts(items), "families": {family: counts([row for row in items if row["family"] == family])
                                                      for family in families}} for split, items in split_rows.items()}
    return {"scope": "sealed simulation evidence corpus label coverage", "fit_receipt_sha256": sha(fit_path),
            "portfolio_sha256": fit["input_portfolio_sha256"],
            "source_manifest_index_sha256": sha256(json_bytes(manifests)).hexdigest(),
            "variants": len(records), "independent_histories": len(groups), "by_split": by_split,
            "runtime_inference_run": False, "image_bytes_opened": False, "optimizer_updates": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new immutable audit output")
    atomic(args.output, audit(args.fit.resolve()))
    print(args.output)


if __name__ == "__main__":
    main()
