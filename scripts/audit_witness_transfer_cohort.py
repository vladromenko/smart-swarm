"""Verify native-format capture receipts for the separate MuJoCo transfer splits."""

import argparse
from collections import Counter
from pathlib import Path

from capture_witness_transfer import SCHEMA, new_design, validate_group
from smart_swarm.episode import LABELS, TRUTH, atomic, load, rows, sha, validate


def bound_design(root):
    path = root / "design.json"
    plan = load(path)
    if plan["schema"] != SCHEMA or plan != new_design():
        raise ValueError("The transfer allocation or pinned acquisition code changed")
    return plan, sha(path)


def validated_episode(path, case, source):
    manifest_path = path / "manifest.json"
    digest = sha(manifest_path)
    if digest != case["manifest_sha256"] or sha(path / TRUTH / "physics.jsonl") != case["contact_trace_sha256"]:
        raise ValueError(f"The native simulation manifest or contact trace changed for {source['source_group']}")
    manifest = load(manifest_path)
    validate(path, manifest)
    if (manifest["scene_group_id"] != source["source_group"] or manifest["lineage"]["split"] != source["split"]
            or manifest["run_mode"] != "mujoco" or manifest["simulation_recipe_seed"] != source["seed"]):
        raise ValueError("A transfer episode changed source-group lineage or simulator seed")
    annotation, = rows(path / LABELS / "settled-event.jsonl")
    if annotation["terminal"]["ground_truth"] != case["terminal"] or annotation["relocation_assignment"] != case["association"]:
        raise ValueError("The acquisition receipt and isolated labels disagree")
    return {"path": str(path), "manifest_sha256": digest, "contact_trace_sha256": case["contact_trace_sha256"],
            "owner_condition": case["owner"], "sample_counts": manifest["sample_counts"]}


def validate_source(root, source):
    directory = root / source["source_group"]
    path = directory / "receipt.json"
    base = {key: source[key] for key in ("source_group", "seed", "split", "task")}
    if not path.is_file():
        return {**base, "status": "unopened"}
    result = load(path)
    if result["recipe"] != source:
        raise ValueError("The source receipt differs from the preallocated recipe")
    status = result["status"]
    if status != "verified":
        return {**base, "status": "unsealed" if status == "running" else status,
                "reason": result.get("reason"), "acquired_variants": len(result["cases"])}
    validate_group(source, result["cases"])
    audited = []
    for case in result["cases"]:
        episode = Path(case["path"]).resolve()
        if not episode.is_relative_to(directory.resolve()):
            raise ValueError("An episode escaped its independently allocated group")
        audited.append(validated_episode(episode, case, source))
    return {**base, "status": "verified", "receipt_sha256": sha(path), "terminal": source["expected_terminal"],
            "assignment": source["expected_assignment"], "variants": audited}


def audit(root, split):
    design, digest = bound_design(root)
    result = [validate_source(root, source) for source in design["sources"] if source["split"] == split]
    completed = [row for row in result if row["status"] == "verified"]
    identities = [case["manifest_sha256"] for row in completed for case in row["variants"]]
    if len(identities) != len(set(identities)):
        raise ValueError("A transfer view appeared twice in the recorded split")
    return {"schema": "smart-swarm-witness-transfer-intake-v1", "split": split,
            "status": "complete" if len(completed) == len(result) else "incomplete",
            "design_sha256": digest, "original_profiles_sha256": design["original_profiles_sha256"],
            "allocated_independent_groups": len(result), "scored_variants": 0, "verified_independent_groups": len(completed),
            "verified_variants": len(identities), "capture_counts": dict(Counter(row["status"] for row in result)),
            "terminal_counts": dict(Counter(row["terminal"] for row in completed)),
            "assignment_counts": dict(Counter(row["assignment"] for row in completed)),
            "task_counts": dict(Counter(row["task"] for row in completed)), "sources": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    atomic(args.output, audit(args.root, args.split))
    print(args.output)


if __name__ == "__main__":
    main()
