"""Audit exact visual-clip overlap in the three frozen evidence pilot splits."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path

from audit_evidence_label_coverage import verify_groups
from smart_swarm.contract import json_bytes
from smart_swarm.episode import POLICY, Replay, atomic, load, sha
from smart_swarm.evidence_inputs import AGENTS, clip_packets, runtime_signature

WITNESSES = AGENTS[:2]


def clips(record):
    root = Path(record["path"])
    manifest = load(root / "manifest.json")
    if manifest["world_episode_id"] != record["world_episode_id"] or manifest["lineage"]["split"] != record["split"]:
        raise ValueError(f"{root.name} differs from its frozen source lineage")
    selected = {}
    for role in WITNESSES:
        relative = f"{POLICY}/{role}.jsonl"
        if sha(root / relative) != manifest["sidecar_sha256"][relative]:
            raise ValueError(f"{root.name} has an altered witness history")
        packets = clip_packets(Replay(root, role))
        hashes = [[(frame["sensor_id"], frame["sha256"]) for frame in packet["cameras"]] for packet in packets]
        selected[role] = sha256(json_bytes(hashes)).hexdigest() if hashes else None
    return selected


def collisions(records):
    indexed = defaultdict(list)
    for row, signature in records:
        if signature is not None:
            indexed[signature].append({key: row[key] for key in ("scene_group_id", "split", "world_episode_id")})
    repeats = [group for group in indexed.values() if len(group) > 1]
    cross = [group for group in repeats if len({item["scene_group_id"] for item in group}) > 1]
    return {"signatures": len(indexed), "identical_within_source": len(repeats) - len(cross),
            "identical_across_independent_sources": cross,
            "identical_across_splits": [group for group in cross if len({item["split"] for item in group}) > 1]}


def pair_checks(groups, inputs):
    result = Counter()
    for rows in groups.values():
        if rows[0]["family"] == "F6":
            first, second = rows
            if inputs[first["world_episode_id"]] != inputs[second["world_episode_id"]]:
                raise ValueError("A matched hidden pair changed the admitted RGB clip")
            observed = [runtime_signature(Path(row["path"])) for row in rows]
            if observed[0] != observed[1]:
                raise ValueError("A matched hidden pair differs on its full permitted inputs")
            result[first["split"]] += 1
    return dict(result)


def audit(fit_path):
    fit = load(fit_path)
    portfolio = Path(fit["dataset"])
    if fit["status"] != "complete" or sha(portfolio) != fit["input_portfolio_sha256"]:
        raise ValueError("Use the portfolio bound to the fitted memory checkpoints")
    collection = load(portfolio)
    records = collection["records"]
    groups = verify_groups(records)
    if len(records) != collection["recorded_variants"] or len(groups) != collection["accepted_independent"]:
        raise ValueError("The frozen source counts changed")
    source = Path(__file__).resolve().parents[1] / "src/smart_swarm"
    if any(sha(source / name) != fit["study_source_sha256"][name]
           for name in ("contract.py", "episode.py", "evidence_inputs.py")):
        raise ValueError("The admitted projection differs from the original frozen reader")
    signatures = {row["world_episode_id"]: clips(row) for row in records}
    pairs = pair_checks(groups, signatures)
    scoped = [(row, signatures[row["world_episode_id"]]) for row in records]
    by_role = {role: collisions([(row, value[role]) for row, value in scoped]) for role in WITNESSES}
    combined = collisions([(row, sha256(json_bytes(value)).hexdigest()) for row, value in scoped])
    return {"scope": "exact stored PNG digests in the selected eight-packet windows per admitted witness",
            "independent_histories": len(groups), "variants": len(records), "portfolio_sha256": sha(portfolio),
            "f6_full_input_pairs_by_split": pairs, "joint_witness_overlap": combined, "individual_witness_overlap": by_role,
            "image_bytes_opened": False, "runtime_inference_run": False, "optimizer_updates": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new immutable audit output")
    atomic(args.output, audit(args.fit.resolve()))
    print(args.output)


if __name__ == "__main__":
    main()
