"""Evaluate the frozen acquisition policies on the separately reserved reframed sources."""

import argparse
from collections import Counter
from pathlib import Path

from audit_witness_transfer_cohort import audit, validate_source
from fit_witness_transfer_selector import ARMS, PROFILE, SCHEMA, code_binding, forecasts, score_summary, sessions
from reframe_witness_transfer import reframe_design
from smart_swarm.cli import settings
from smart_swarm.episode import atomic, load, sha
from smart_swarm.research_jobs import admitted


def evaluation_inputs(original, root, policy):
    release = load(policy)
    if (release["schema"] != SCHEMA or release["status"] != "frozen" or release["code_sha256"] != code_binding()
            or release["profile_sha256"] != sha(PROFILE) or release["held_out_likelihoods_read"] is not False):
        raise ValueError("The pre-evaluation policy or its original implementation changed")
    if release["validation_intake"] != audit(original, "validation"):
        raise ValueError("A source used in the original validation fit changed")
    allocation_path = root / "design.json"
    if not allocation_path.is_file():
        raise ValueError("The fresh held-out allocation has not been reserved")
    design = reframe_design(original, policy, root)
    inputs = []
    for source in design["sources"]:
        group = root / source["source_group"]
        journal = group / "receipt.json"
        receipt = load(journal) if journal.is_file() else {}
        if receipt.get("status") != "verified" or receipt["recipe"] != source:
            raise ValueError(f"The new physical capture gate is incomplete for {source['source_group']}")
        for variant in receipt["cases"]:
            episode = Path(variant["path"]).resolve()
            digest = sha(episode / "manifest.json")
            if not episode.is_relative_to(group.resolve()) or digest != variant["manifest_sha256"]:
                raise ValueError("The reserved evidence image source changed after capture")
            inputs.append({"path": str(episode), "source_group": source["source_group"], "manifest_sha256": digest})
    return release, design, inputs


def predictions(original, root, policy, destination, heartbeat):
    release, design, inputs = evaluation_inputs(original, root, policy)
    path = destination.with_name(destination.stem + "-sealed-predictions.json")
    binding = {"policy_sha256": sha(policy), "test_design_sha256": sha(root / "design.json"), "manifests": inputs}
    if path.is_file():
        prior = load(path)
        if prior["binding"] != binding:
            raise ValueError("Stored predictions do not match the reserved holdout and fixed policy")
        return design, path, prior["predictions"]
    contexts = sessions()
    records = []
    for row in inputs:
        heartbeat()
        records.append({**row, "methods": forecasts(Path(row["path"]), contexts, release)})
    atomic(path, {"binding": binding, "predictions": records})
    return design, path, records


def evaluate(original, root, policy, destination, heartbeat):
    design, path, sealed = predictions(original, root, policy, destination, heartbeat)
    verified = [validate_source(root, source) for source in design["sources"]]
    labels = {row["path"]: (group["terminal"], row["owner_condition"], group["assignment"], group["task"])
              for group in verified for row in group["variants"]}
    if len(labels) != len(sealed) or set(labels) != {row["path"] for row in sealed}:
        raise ValueError("Verified isolated evaluator labels cover a different set of predictions")
    results = []
    for row in sealed:
        terminal, owner, assignment, task = labels[row["path"]]
        results.append({**row, "truth": terminal, "evaluator_visibility_condition": owner,
                        "evaluator_assignment": assignment, "task": task})
    report = {"status": "evaluated", "scope": "separately reserved reframed MuJoCo viewpoints with the original frozen fit",
              "policy_sha256": sha(policy), "predictions_sha256": sha(path), "test_capture_design_sha256": sha(root / "design.json"),
              "independent_test_sources": len(verified), "test_variants": len(results), "new_optimizer_updates": 0,
              "terminal_groups": dict(Counter(row["terminal"] for row in verified)), "matched_methods": list(ARMS),
              "source_groups": verified, "case_results": results, "source_weighted_metrics": score_summary(results)}
    atomic(destination, report)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a new evaluation receipt path")
    with admitted(settings(), "smart-swarm-frozen-reframed-witness-scoring", archive_only=True) as (_, heartbeat):
        result = evaluate(args.original, args.root, args.policy, args.output, heartbeat)
    print(result)


if __name__ == "__main__":
    main()
