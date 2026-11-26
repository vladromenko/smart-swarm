#!/usr/bin/env python3
"""Change the order of earlier witness frames while holding the final exposure fixed."""

import argparse
from collections import defaultdict
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import tarfile
import time

import torch

from archive_primary_release import INDEX, KINDS, read_member
from archive_primary_results import verify
from audit_central_history import full_stream, load_backbone
from audit_primary_ablations import reexchange
from audit_primary_delivery_delay import CALIBRATION, RESULTS, RESULTS_SHA256
from audit_primary_pairs import conservative_interval, paired_rows
from audit_team_evidence import decode_features
from compare_primary_supervision import describe
from export_primary_ablation_table import bound_inputs, publish, signed_groups
from export_primary_result_table import HEAD_NAMES
from score_primary_test import check_prediction_receipt
from score_primary_validation import SELECTIVE, source_hashes
from smart_swarm.cli import settings
from smart_swarm.contract import json_bytes
from smart_swarm.episode import Replay, load, sha
from smart_swarm.evidence_inputs import clip_packets, runtime_signature
from smart_swarm.primary_inputs import primary_policy_view
from smart_swarm.research_jobs import admitted
from smart_swarm.team_evidence import opportunities
from smart_swarm.temporal_evidence import EvidenceHead

CAP_SECONDS = 120
FORWARD = tuple(range(8))
EARLIER_REVERSED = (*reversed(FORWARD[:-1]), FORWARD[-1])
ALL_REVERSED = tuple(reversed(FORWARD))
FINAL_REPEAT = (FORWARD[-1],) * len(FORWARD)
ARMS = {
    "chronological": (FORWARD, FORWARD),
    "reverse_earlier_both": (EARLIER_REVERSED, EARLIER_REVERSED),
    "reverse_earlier_so101": (EARLIER_REVERSED, FORWARD),
    "reverse_earlier_tank": (FORWARD, EARLIER_REVERSED),
    "repeat_final": (FINAL_REPEAT, FINAL_REPEAT),
    "fully_reversed": (ALL_REVERSED, ALL_REVERSED),
}
LABELS = ("Chronological", "Reverse earlier frames, both witnesses", "Reverse earlier frames, SO101",
          "Reverse earlier frames, Tank", "Repeat final frame", "Full reversal, archival control")
FACTORS = ("containment", "association")
SCOPE = "Exploratory fixed-final-frame replay on the registered 50 consumed three-robot sources"


def archive_inputs(results, calibration):
    if verify(results)["archive_sha256"] != RESULTS_SHA256:
        raise ValueError("The replay requires the original independent result archive")
    with tarfile.open(results) as archive:
        identity, subset, profile, config, priors = bound_inputs(archive, calibration)
    return identity, subset, profile, config, priors


def source_inputs(archive, identity, subset):
    records = json.loads(read_member(archive, INDEX))["original_recording_manifests"]
    sources = {}
    for location, record in records.items():
        manifest_path = Path(location)
        manifest = json.loads(read_member(archive, record["archive_member"]))
        source = str(manifest_path.parent)
        name, size = manifest["scene_group_id"], manifest["scaling_team_size"]
        if size == 3 and name in subset:
            if (sha(manifest_path) != identity["manifests"][source] or manifest["lineage"]["split"] != "test"
                    or manifest["family"] != subset[name] or source in sources):
                raise ValueError("A source manifest differs from the original three-robot cohort")
            sources[source] = name
    counts = {name: list(sources.values()).count(name) for name in subset}
    if len(sources) != 58 or any(counts[name] != (2 if family == "F6" else 1) for name, family in subset.items()):
        raise ValueError("The registered cohort lacks a required history or hidden variant")
    receipts = {kind: check_prediction_receipt(json.loads(read_member(archive, f"scores/primary-test-{kind}-predictions.json")),
                                               {**identity, "checkpoint_kind": kind}, identity["manifests"])
                for kind in KINDS}
    return sources, receipts


def networks(calibration, profile, priors):
    if source_hashes() != profile["implementation_sha256"]:
        raise ValueError("Frozen model or policy implementation has changed")
    visual = load_backbone({"frozen_backbone_sha256": profile["encoder_sha256"]})
    models = {}
    with tarfile.open(calibration) as archive:
        for kind in KINDS:
            payload = read_member(archive, f"models/{kind}.pt")
            fit = profile["fits"][kind]
            if sha256(payload).hexdigest() != fit["checkpoint_sha256"]:
                raise ValueError("The archived network differs from the original checkpoint")
            saved = torch.load(BytesIO(payload), map_location="cpu", weights_only=True)
            model = EvidenceHead().eval().requires_grad_(False)
            model.load_state_dict(saved["weights"], strict=True)
            mounts = fit["mount_and_receiver_profiles"]["3"]
            models[kind] = {"head": model, "temp": saved["temperature"], "fit": mounts,
                            "common": {**priors[kind], "gains": saved["query_expected_information_nats"]}}
    return visual, models


def local_stream(path, advertisement, encoder):
    client = Replay(path, advertisement["sender"])
    eligible = [packet for packet in client.packets() if packet["cameras"]]
    positions = {packet["sequence"]: index for index, packet in enumerate(eligible)}
    picked = clip_packets(client)
    identities = [f"{camera['sensor_id']}:{packet['sequence']:04d}"
                  for packet in picked for camera in packet["cameras"]]
    if len(picked) != len(FORWARD) or identities != advertisement["frame_ids"]:
        raise ValueError("The frozen clip and the advertised frames disagree")
    observations, _ = full_stream(encoder, client)
    features = observations[[positions[packet["sequence"]] for packet in picked]]
    hashes = [[camera["sha256"] for camera in packet["cameras"]] for packet in picked]
    return {"agent": client.agent, "factor": advertisement["factor"], "features": features, "hashes": hashes}


def predict_head(window, streams, model, archived):
    common, fit = model["common"], model["fit"]
    threshold = fit["decision_thresholds"][SELECTIVE]
    estimates = {}
    for name, orders in ARMS.items():
        order_by_factor = dict(zip(FACTORS, orders, strict=True))
        posteriors = {stream["agent"]: decode_features(stream["features"][list(order_by_factor[stream["factor"]])],
                                                       stream["factor"], model["head"], model["temp"])
                      for stream in streams}
        estimates[name] = reexchange({"window": window}, fit["association"]["compact"], common, threshold,
                                      "factor_coverage", common["gains"], posteriors)
    controls = {"chronological": archived[SELECTIVE],
                "fully_reversed": archived["bounded_ablations"]["reversed_witness_time"]}
    check_fields = ("wire", "decision", "joint", "serialized_bytes", "modeled_latency_ms")
    if any(estimates[name][field] != saved[field] for name, saved in controls.items() for field in check_fields):
        raise ValueError("Chronological or full reversal control differs from the sealed policy")
    source_wire = estimates["chronological"]["wire"]
    signatures = {field: source_wire[field] for field in ("advertisements", "requests")}
    for result in estimates.values():
        observed = {field: result["wire"][field] for field in signatures}
        metadata = lambda wire: [{key: value for key, value in row.items() if key != "probabilities"} for row in wire["replies"]]
        if (observed != signatures or metadata(result["wire"]) != metadata(source_wire)
                or result["serialized_bytes"] != archived[SELECTIVE]["serialized_bytes"]):
            raise ValueError("A visual perturbation changed the witness, timestamps, or wire charge")
    return estimates


def predict_source(source, name, encoder, models, reference):
    with primary_policy_view(Path(source)) as projected:
        signature = sha256(json_bytes(runtime_signature(projected))).hexdigest()
        window = opportunities(projected)
        streams = [local_stream(projected, item, encoder) for item in window[2]]
        decisions = {kind: predict_head(window, streams, models[kind], reference[kind][source]) for kind in KINDS}
    clips = {stream["agent"]: stream["hashes"] for stream in streams}
    return {"group": name, "source": source, "private_scope_sha256": signature, "clips": clips, "decisions": decisions}


def coupled_pairs(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["group"]].append({key: value for key, value in row.items() if key != "source"})
    pairs = [group for group in groups.values() if len(group) > 1]
    if len(pairs) != 8 or any(len(pair) != 2 or pair[0] != pair[1] for pair in pairs):
        raise ValueError("The complete projected inputs or frozen predictions disagree in a hidden pair")
    return len(pairs)


def forecast(results, calibration):
    with admitted(settings(), "smart-swarm-fixed-final-witness-order", archive_only=True) as (grant, heartbeat):
        started = time.monotonic()
        torch.set_num_threads(3)
        identity, subset, profile, _, priors = archive_inputs(results, calibration)
        with tarfile.open(results) as original:
            sources, reference = source_inputs(original, identity, subset)
        encoder, models = networks(calibration, profile, priors)
        rows = []
        for source, name in sorted(sources.items()):
            heartbeat()
            if time.monotonic() - started >= CAP_SECONDS:
                raise TimeoutError("The archive replay reached its 120-second native CPU cap")
            rows.append(predict_source(source, name, encoder, models, reference))
        elapsed = time.monotonic() - started
        if elapsed > CAP_SECONDS:
            raise TimeoutError("The archive replay reached its 120-second native CPU cap")
    return {"schema": "smart-swarm-fixed-final-witness-order-forecasts-v1", "scope": SCOPE,
            "archive_sha256": RESULTS_SHA256, "profile_sha256": identity["profiles_sha256"],
            "labels_used_for_forecasts": False, "optimizer_updates": 0, "new_simulation_steps": 0,
            "replay_cap_s": CAP_SECONDS, "replay_wall_s": elapsed, "reservation": grant,
            "paired_equal_f6_groups": coupled_pairs(rows), "clip_index_orders": ARMS,
            "prediction_payload_sha256": sha256(json_bytes(rows)).hexdigest(), "forecasts": rows}


def joined_cases(groups, rows, kind):
    joined, matched = [], set()
    for group in groups:
        cases = []
        for row in group["outcomes"]:
            source = row["source"]
            output = rows[source]
            if output["group"] != group["group"] or source in matched:
                raise ValueError("A forecast has a different scored source or repeat assignment")
            matched.add(source)
            cases.append({**row, **output["decisions"][kind]})
        joined.append({**group, "outcomes": cases})
    if matched != set(rows):
        raise ValueError("The evaluator did not score every sealed forecast")
    return joined


def summaries(groups, config):
    reference = paired_rows(groups, "chronological")
    cases = [row for group in groups for row in group["outcomes"]]
    return {arm: {**describe(groups, arm, config),
                  "paired_first_choice_vs_chronological": conservative_interval(paired_rows(groups, arm), reference),
                  "changed_decisions": sum(row[arm]["decision"] != row["chronological"]["decision"] for row in cases)}
            for arm in ARMS}


def score(results, calibration, path):
    receipt = load(path)
    material = receipt["forecasts"]
    if (receipt["archive_sha256"] != RESULTS_SHA256 or receipt["labels_used_for_forecasts"] is not False
            or receipt["prediction_payload_sha256"] != sha256(json_bytes(material)).hexdigest()
            or receipt["clip_index_orders"] != json.loads(json_bytes(ARMS)) or coupled_pairs(material) != 8):
        raise ValueError("Score only the sealed, label-free, coupled visual predictions")
    identity, subset, _, config, _ = archive_inputs(results, calibration)
    if receipt["profile_sha256"] != identity["profiles_sha256"]:
        raise ValueError("The forecast has a different frozen policy profile")
    rows = {row["source"]: row for row in material}
    summaries_by_head = {}
    with tarfile.open(results) as archive:
        for kind in KINDS:
            groups, _, _ = signed_groups(archive, kind, identity, subset)
            summaries_by_head[kind] = summaries(joined_cases(groups, rows, kind), config)
    return {"schema": "smart-swarm-fixed-final-witness-order-results-v1", "scope": SCOPE,
            "original_archive_sha256": RESULTS_SHA256, "frozen_profile_sha256": receipt["profile_sha256"],
            "forecast_receipt_sha256": sha(path), "independent_source_groups": len(subset),
            "variants_per_head": len(rows), "complete_f6_pairs": receipt["paired_equal_f6_groups"],
            "metrics_by_head": summaries_by_head, "optimizer_updates": 0, "new_simulation_steps": 0}


def markdown(report):
    lines = ["# Witness order with fixed final frames", "", report["scope"] + ".", "",
             "Each perturbation uses the original frozen head, eight input positions, witness selection, modeled timing, "
             "message bytes and decision threshold. Frame hashes and index orders are in the sealed forecasts. "
             "The full-reversal control changes the final frame. The first four perturbed conditions leave the "
             "final frame unchanged and vary earlier input. Repeated final frames have no independent exposure value.", "",
             "| Objective | Witness clips | First container | Resolving correct | F6 pairs inspect | Wrong variants | NLL |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for kind in KINDS:
        for name, label in zip(ARMS, LABELS, strict=True):
            row = report["metrics_by_head"][kind][name]
            lines.append(f"| {HEAD_NAMES[kind]} | {label} | {row['container_correct']}/{row['container_count']} | "
                         f"{row['resolvable_correct']}/{row['resolvable_count']} | "
                         f"{row['f6_inspections'] // 2}/{row['f6_variants'] // 2} | {row['confident_wrong']} | "
                         f"{row['group_mean_nll']:.3f} |")
    return "\n".join([*lines, ""])


def figure(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(7.1, 3.45), layout="constrained")
    styles = ((-.13, "o", "#235a78"), (.13, "s", "#a24a24"))
    for kind, (offset, marker, color) in zip(KINDS, styles, strict=True):
        for index, name in enumerate(tuple(ARMS)[1:]):
            row = report["metrics_by_head"][kind][name]
            paired = row["paired_first_choice_vs_chronological"]
            score = 100 * (paired["wins"] - paired["losses"]) / row["container_count"]
            lower, upper = [100 * value for value in paired["conservative_paired_95pct"]]
            axis.hlines(index + offset, lower, upper, color=color, linewidth=1.35)
            label = HEAD_NAMES[kind] if index == 0 else None
            axis.scatter(score, index + offset, s=28, marker=marker, color=color, zorder=3, label=label)
    axis.set_yticks(range(len(LABELS) - 1), LABELS[1:])
    axis.invert_yaxis()
    axis.axvline(0, color="#444444", linewidth=.8, linestyle="--")
    axis.grid(axis="x", alpha=.18)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.set(xlabel="Difference from chronological clips (percentage points)", xlim=(-106, 28))
    fig.suptitle("First correct container choice · paired 95% intervals", fontsize=12)
    axis.legend(loc="lower right", frameon=False, handletextpad=.3, fontsize=8)
    buffer = BytesIO()
    fig.savefig(buffer, format="pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    if path.exists():
        if path.read_bytes() != buffer.getvalue():
            raise ValueError(f"Choose a new publication path to preserve {path}")
    else:
        with path.open("xb") as output:
            output.write(buffer.getvalue())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--calibration", type=Path, default=CALIBRATION)
    parser.add_argument("--forecasts", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--table", type=Path)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    if ((args.output is None) != (args.table is None)) or (args.figure is not None and args.output is None):
        parser.error("The scoring stage takes --output and --table, with an optional --figure")
    paths = [path for path in (args.forecasts, args.output, args.table, args.figure) if path is not None]
    if len({path.resolve() for path in paths}) != len(paths):
        parser.error("The prediction seal and score destinations must be distinct")
    if args.output is None and args.forecasts.exists():
        parser.error("Choose an unused path for the prediction seal")
    try:
        if args.output is None:
            result = forecast(args.results, args.calibration)
            with args.forecasts.open("xb") as output:
                output.write(json_bytes(result) + b"\n")
        else:
            report = score(args.results, args.calibration, args.forecasts)
            publish(report, args.output, args.table, render=markdown)
            if args.figure is not None:
                figure(report, args.figure)
    except (OSError, ValueError, KeyError, tarfile.TarError) as error:
        parser.exit(2, f"Fixed-final-frame report withheld. {error}\n")


if __name__ == "__main__":
    main()
