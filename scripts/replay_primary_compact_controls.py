"""Re-evaluate matched compact selection on the consumed primary simulation cohort."""

import argparse
from hashlib import sha256
from pathlib import Path
from time import monotonic

from audit_team_evidence import decode_features
from compare_compact_witness_controls import (
    ARMS, PRIMARY_CAP, SEED, cached_inputs, forecast, quality_advertisements, reliability,
)
from compare_primary_supervision import (
    FIELDS, KINDS, TEAMS, authority, evaluator_record, prediction_path, read_scores, score_paths, verify_scores,
)
from score_primary_test import check_prediction_receipt, runtime
from score_primary_validation import SELECTIVE
from smart_swarm.cli import ROOT, settings
from smart_swarm.episode import atomic, json_bytes, load, sha
from smart_swarm.evidence_inputs import runtime_signature
from smart_swarm.evidence_queries import receive_reply
from smart_swarm.primary_inputs import primary_policy_view
from smart_swarm.research_jobs import admitted
from verify_mobile_frozen_head_reuse import clips, shared_contexts


PILOT = ROOT / "evidence/learning-pilot-v3"
PROFILE = PILOT / "primary-profiles-frozen.json"
SCOPE = "exploratory matched transport replay on the consumed primary simulation cohort"
WALL_LIMIT_S = 1800
CONTROL_FILES = ("compare_compact_witness_controls.py", "verify_mobile_frozen_head_reuse.py",
                 "compare_primary_supervision.py", "replay_primary_compact_controls.py")


def validation_quality():
    identity, traces = authority(PROFILE, "validation")
    truths = {key: [evaluator_record(trace, identity["manifests"]) for trace in rows]
              for key, rows in traces.items()}
    paths = score_paths("validation")
    frozen = load(PROFILE)
    result = {}
    for kind in KINDS:
        admitted_scores = verify_scores(*read_scores(paths[kind], kind, identity), truths)
        indexed = {row["source"]: (group, {key: row[key] for key in FIELDS})
                   for (group, _), bundle in admitted_scores.items() for row in bundle["outcomes"]}
        rows, digest = cached_inputs(kind, frozen, identity, indexed)
        independent = sorted({row["scene_group_id"] for row in rows})
        if len(independent) != 6:
            raise ValueError("The matched controls use the established six-source validation cache")
        result[kind] = {"independent_source_groups": independent, "cached_scores_sha256": digest,
                        "frozen_validation_score_sha256": sha(paths[kind]), "bins": reliability(rows, None)}
    return result


def freeze(root):
    root.mkdir(exist_ok=True, parents=True)
    path = root / "release.json"
    recipe = {"scope": SCOPE, "primary_profile_sha256": sha(PROFILE), "wall_limit_s": WALL_LIMIT_S,
              "cap_bytes": PRIMARY_CAP, "coupled_seed": SEED, "team_sizes": list(TEAMS), "methods": list(ARMS),
              "optimizer_updates": 0, "implementation_sha256": {
                  name: sha(Path(__file__).with_name(name)) for name in CONTROL_FILES}}
    if path.exists():
        existing = load(path)
        if any(existing[key] != value for key, value in recipe.items()):
            raise ValueError("The established compact replay has different code or limits")
    else:
        atomic(path, {**recipe, "quality_fit": validation_quality()})
    return load(path)


def original_predictions(identity):
    seals = {}
    for kind, score in score_paths("test").items():
        path = prediction_path(score)
        receipt = load(path)
        seals[kind] = {"sha256": sha(path), "predictions": check_prediction_receipt(
            receipt, {**identity, "checkpoint_kind": kind}, identity["manifests"])}
    return seals


def verify_original_replies(window, posteriors, original):
    query, _, ads = window
    expected = {ad["sender"]: receive_reply(ad, query, lambda factor: posteriors[ad["sender"]]) for ad in ads}
    wires = [entry["wire"] for caps in original["budget_diagnostics"].values() for entry in caps.values()]
    for wire in wires:
        if any(reply != expected[reply["sender"]] for reply in wire["replies"]):
            raise ValueError("Shared scoped encoding changed a compact reply in the original prediction seal")
    return len({reply["sender"] for wire in wires for reply in wire["replies"]})


def replay_world(path, size, contexts, originals, quality):
    with primary_policy_view(path) as scoped:
        window, features, _ = clips(scoped, contexts)
    results = {}
    for kind, context in contexts.items():
        _, head, temperature = context["network"]
        posteriors = {sender: decode_features(feature, factor, head, temperature)
                      for sender, (factor, feature) in features.items()}
        replies = verify_original_replies(window, posteriors, originals[kind])
        fit = context["mounts"][str(size)]
        paid = quality_advertisements(window, posteriors)
        methods = forecast(paid, posteriors, fit, context["common"], quality[kind]["bins"])
        results[kind] = {"methods": methods, "original_compact_senders_reproduced": replies}
    return results


def public_inputs(traces, manifests):
    return [{"path": row["path"], "manifest_sha256": manifests[row["path"]]} for row in traces]


def predict_group(key, paths, contexts, originals, quality):
    group, size = key
    rows = []
    for entry in paths:
        path = entry["path"]
        if sha(Path(path) / "manifest.json") != entry["manifest_sha256"]:
            raise ValueError(f"Original policy manifest changed for {path}")
        references = {kind: originals[kind]["predictions"][path] for kind in KINDS}
        predictions = replay_world(Path(path), size, contexts, references, quality)
        signature = sha256(json_bytes(runtime_signature(Path(path)))).hexdigest() if len(paths) == 2 else None
        rows.append({**entry, "heads": predictions, "matched_private_input_sha256": signature})
    if len(paths) == 2:
        a, b = rows
        if a["matched_private_input_sha256"] != b["matched_private_input_sha256"] or a["heads"] != b["heads"]:
            raise ValueError("An F6 pair changes its permitted runtime input or compact prediction")
    return {"source_group": group, "team_size": size, "variants": rows}


def progress_path(root, key):
    group, size = key
    if not group.replace("-", "").isalnum() or size not in TEAMS:
        raise ValueError("Invalid source identity in the original cohort")
    return root / "groups" / f"{group}-n{size}.json"


def read_group(path, key, inputs, binding):
    row = load(path)
    identity = key == (row["source_group"], row["team_size"])
    listed = [{k: entry[k] for k in ("path", "manifest_sha256")} for entry in row["variants"]]
    if not identity or row["binding"] != binding or listed != inputs:
        raise ValueError(f"A persisted compact replay group changed at {path}")
    return row


def compute_groups(root, traces, identity, originals, release, heartbeat, deadline):
    binding = {"release_sha256": sha(root / "release.json"), "primary_test_release_sha256": identity[
               "primary_test_release_sha256"], "original_prediction_sha256": {k: v["sha256"] for k, v in originals.items()}}
    contexts = shared_contexts({kind: {**runtime(load(PROFILE), kind), "inspect_markers": False} for kind in KINDS})
    result, total = [], len(traces)
    order = sorted(traces, key=lambda x: (-x[1], x[0]))
    (root / "groups").mkdir(exist_ok=True)
    for index, key in enumerate(order, 1):
        source = progress_path(root, key)
        inputs = public_inputs(traces[key], identity["manifests"])
        if source.exists():
            row = read_group(source, key, inputs, binding)
        else:
            heartbeat()
            if monotonic() >= deadline:
                raise TimeoutError("The compact control wall cap was reached before a complete forecast seal")
            row = predict_group(key, inputs, contexts, originals, release["quality_fit"])
            row["binding"] = binding
            atomic(source, row)
        result.append(row)
        if index % 20 == 0 or index == total:
            print(f"{index}/{total} source-and-team forecast groups sealed", flush=True)
    return result


def begin_charge(root):
    path = root / "budget.json"
    previous = load(path) if path.is_file() else {"used_s": 0., "attempts": 0}
    if "active_cap_s" in previous:
        previous["used_s"] += previous.pop("active_cap_s")
    remaining = WALL_LIMIT_S - previous["used_s"]
    if remaining <= 0:
        raise TimeoutError("The original scalar replay budget is exhausted")
    previous.update(active_cap_s=remaining, attempts=previous["attempts"] + 1)
    atomic(path, previous)
    return path, previous, remaining


def predict(root, release, heartbeat):
    import torch

    scope, traces = authority(PROFILE, "test")
    originals = original_predictions(scope)
    if (len(scope["source_groups"]), len(traces), len(scope["manifests"])) != (100, 300, 348):
        raise ValueError("The original full F1–F6 primary cohort is required")
    torch.set_num_threads(3)
    charge_path, charge, remaining = begin_charge(root)
    start = monotonic()
    try:
        rows = compute_groups(root, traces, scope, originals, release, heartbeat, start + remaining)
        if monotonic() - start > remaining:
            raise TimeoutError("The completed forecasts crossed the declared replay wall cap")
        payload = {"scope": SCOPE, "source_groups": scope["source_groups"], "groups": rows,
                   "groups_payload_sha256": sha256(json_bytes(rows)).hexdigest(), "eligible_variants": len(scope["manifests"]),
                   "optimizer_updates": 0, "evaluator_inputs_read_by_predictor": False}
        atomic(root / "predictions.json", payload)
    finally:
        charge.update(used_s=charge["used_s"] + monotonic() - start)
        charge.pop("active_cap_s")
        atomic(charge_path, charge)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    with admitted(settings(), "smart-swarm-primary-matched-compact-replay", archive_only=True) as (_, heartbeat):
        if (args.root / "predictions.json").exists():
            parser.error("The full original cohort already has a prediction seal in this directory")
        release = freeze(args.root)
        predict(args.root, release, heartbeat)
    print(args.root / "predictions.json")


if __name__ == "__main__":
    main()
