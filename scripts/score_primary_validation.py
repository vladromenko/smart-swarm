#!/usr/bin/env python3
"""Rehearse the full memory comparison on admitted simulator validation sources."""

import argparse
from pathlib import Path
import time

import numpy as np
import torch

from audit_central_history import full_stream, load_backbone
from audit_evidence_replay import verified_model
from audit_memory_validation import accepted_sources
from audit_receiver_history import cues, fit_cue, predict as local_predict
from audit_team_evidence import decode_features
from audit_team_ledgers import audit as ledger_audit
from calibrate_team_views import PILOT, association_samples, group_metrics, profiles
from capture_memory_validation import CONFIG, DEFAULT_OUTPUT, ROOT, validation_slots
from smart_swarm.cli import settings
from smart_swarm.episode import LABELS, POLICY, Replay, atomic, listed_file, load, rows, sha
from smart_swarm.evidence_comparison import learned_prior_map
from smart_swarm.evidence_inputs import clip_packets, runtime_signature
from smart_swarm.evidence_queries import RECEIVER
from smart_swarm.identity_support import witness_marker_support
from smart_swarm.memory import STATES
from smart_swarm.particle_memory import exact_filter, location, normalize, result as joint_result
from smart_swarm.research_jobs import admitted
from smart_swarm.team_evidence import ANCHOR_TANK, exchange, opportunities, pooled_factors
from smart_swarm.temporal_evidence import EvidenceHead
from smart_swarm.view_calibration import calibrated_replies

CURRENT = "receiver_current_rgb"
TEMPORAL = "receiver_private_temporal"
LEDGER = "deterministic_broadcast_ledger"
PARTICLE = "calibrated_broadcast_joint_particle_filter"
FULL = "central_full_rgb_joint_filter"
FULL_WITNESSES = "central_full_witnesses_without_receiver_cue"
COMPACT = "central_compact_budget_matched"
SELECTIVE = "learned_selective_joint_memory"
IDENTITY = "selective_with_observed_marker_support"
PRIMARY_CAP = 3072
TRANSFER = {COMPACT: "recent_per_factor", SELECTIVE: "factor_coverage"}
SOURCES = ("score_primary_validation.py", "audit_team_ledgers.py", "audit_receiver_history.py",
           "calibrate_primary_inspection.py", "calibrate_team_views.py", "audit_team_evidence.py",
           "capture_memory_validation.py", "audit_memory_validation.py", "audit_central_history.py",
           "audit_evidence_replay.py", "learn_evidence_v3.py", "plan_primary_memory.py", "rehearse_scaling.py",
           "capture_primary_test.py", "score_primary_test.py", "freeze_primary_profiles.py", "audit_primary_pairs.py",
           "audit_primary_ablations.py", "primary_test_release.py", "primary_test_budget.py")
MODULES = ("team_evidence.py", "view_calibration.py", "identity_support.py", "particle_memory.py",
           "evidence_inputs.py", "episode.py", "vision.py", "temporal_evidence.py", "memory.py",
           "evidence_queries.py", "evidence_budget_replay.py", "contract.py", "evidence_comparison.py", "scaling.py",
           "primary_inputs.py")


def source_hashes():
    paths = [ROOT / "scripts" / file for file in SOURCES]
    paths.extend(ROOT / "src/smart_swarm" / file for file in MODULES)
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def frame_inventory(root, client):
    eligible = [packet for packet in client.packets() if packet["cameras"]]
    paths = {listed_file(root, camera["frame_ref"], POLICY)
             for packet in eligible for camera in packet["cameras"]}
    identifiers = [f"{camera['sensor_id']}:{packet['sequence']:04d}"
                   for packet in eligible for camera in packet["cameras"]]
    return eligible, identifiers, sum(path.stat().st_size for path in paths)


@torch.no_grad()
def witness_stream(client, eligible, factor, network, reverse_clip):
    encoder, head, temperature = network
    embeddings, count = full_stream(encoder, client)
    if embeddings is None or len(embeddings) != len(eligible):
        raise ValueError("An advertised sender lost scoped visual frames")
    positions = {packet["sequence"]: index for index, packet in enumerate(eligible)}
    chosen = [positions[packet["sequence"]] for packet in clip_packets(client)]
    normal = decode_features(embeddings[chosen], factor, head, temperature)
    reversed_clip = decode_features(embeddings[chosen[::-1]], factor, head, temperature) if reverse_clip else None
    return normal, decode_features(embeddings, factor, head, temperature), count, reversed_clip


def scene_observations(root, network, horizon, *, inspect_markers=False, reverse_clip=False):
    window = opportunities(root)
    query, deadline, ads = window
    if deadline - query != horizon * 1_000_000:
        raise ValueError("Recorded decision window differs from the study configuration")
    compact, full, captured, payload = {}, [], {}, 0
    markers, reordered = {}, {}
    for ad in ads:
        name, factor = ad["sender"], ad["factor"]
        client = Replay(root, name)
        eligible, identifiers, size = frame_inventory(root, client)
        compact[name], posterior, count, reordered[name] = witness_stream(client, eligible, factor, network, reverse_clip)
        full.append({"sender": name, "factor": factor, "frame_ids": identifiers, "probabilities": posterior})
        captured[name] = count
        payload += size
        if inspect_markers and factor == "association":
            markers[name] = witness_marker_support(client)
    receiver = Replay(root, RECEIVER)
    _, identities, receiver_bytes = frame_inventory(root, receiver)
    captured[RECEIVER] = len(identities)
    return {"window": window, "local_rgb_cues": cues(root), "frozen_scores_by_witness": compact, "central_reports": full,
            "identity_support": markers if inspect_markers else None,
            "reversed_clip_scores": reordered if reverse_clip else None,
            "all_rgb_frame_counts_by_agent": captured, "all_rgb_png_bytes": payload + receiver_bytes}


def scene_input(trace, network, horizon, *, inspect_markers=False):
    root = Path(trace["path"])
    observed = scene_observations(root, network, horizon, inspect_markers=inspect_markers)
    label = rows(root / LABELS / "settled-event.jsonl")[0]["relocation_assignment"]
    return {**observed, "source": str(root), "source_manifest_sha256": sha(root / "manifest.json"),
            "truth": trace["terminal_outcome"], "family": trace["family"],
            "variant": trace["variant"], "scoring_association_label": label}


def read_groups(validation, network, horizon, heartbeat, *, only=None, inspect_markers=False):
    allocation = {slot["source_group"]: slot for slot in validation_slots()}
    for group in accepted_sources(validation, allocation, only):
        for team in group["teams"]:
            heartbeat()
            scenes = [scene_input(trace, network, horizon, inspect_markers=inspect_markers) for trace in team["traces"]]
            if scenes[0]["family"] == "F6":
                a, b = [runtime_signature(Path(row["source"])) for row in scenes]
                if a != b or len({row["truth"] for row in scenes}) != 2:
                    raise ValueError("F6 complete observed inputs fail the matched validation design")
            yield {"source_group": group["source_group"], "team_size": team["actor_count"], "episodes": scenes}


def full_samples(records):
    remapped = []
    for group in records:
        episodes = [{**row, "frozen_scores_by_witness": {
            entry["sender"]: entry["probabilities"] for entry in row["central_reports"]}}
            for row in group["episodes"]]
        remapped.append({**group, "episodes": episodes})
    return association_samples(remapped)


def choose(state, threshold):
    posterior = state["belief"]
    winner = max(posterior, key=posterior.get)
    return {**state, "decision": winner if posterior[winner] >= threshold else "request_inspection"}


def likelihoods_from_reports(replies, calibration, common):
    usable = calibrated_replies(replies, calibration, common["reference"]["association"], ANCHOR_TANK)
    return pooled_factors(usable, common["reference"])


def exact_from_reports(replies, calibration, common):
    return exact_filter(common["state_prior"], likelihoods_from_reports(replies, calibration, common))


def on_time_compact(replies, deadline):
    return sorted((row for row in replies if int(row["capture_end_ns"]) <= int(row["sent_ns"])
                   <= int(row["delivered_ns"]) <= deadline), key=lambda row: (int(row["delivered_ns"]), row["sender"]))


def central_full(row, association, receiver_fit, common):
    state = exact_from_reports(row["central_reports"], association, common)
    cost = {"serialized_bytes": row["all_rgb_png_bytes"], "modeled_latency_ms": None,
            "transmitted_frames_by_agent": row["all_rgb_frame_counts_by_agent"]}
    witness_result = choose({**state, **cost}, common["threshold"])
    cue = row["local_rgb_cues"]["history"]
    if cue is not None:
        factors = np.asarray([receiver_fit["surface" if location(s) == "surface" else "other"][cue] for s in STATES])
        state = joint_result(normalize(np.asarray(state["joint"]) * factors))
    return choose({**state, **cost}, common["threshold"]), witness_result


def compact_result(row, method, cap, association, common, *, markers=None):
    posteriors = row["frozen_scores_by_witness"]

    def report(agent, factor):
        if markers is not None and factor == "association" and not markers[agent]["supported"]:
            return common["reference"]["association"]
        return posteriors[agent]

    raw = exchange(method, row["window"], cap, common["gains"], report,
                   common["reference"], common["state_prior"])
    state = exact_from_reports(on_time_compact(raw["wire"]["replies"], row["window"][1]), association, common)
    return choose({**state, "serialized_bytes": raw["serialized_bytes"], "modeled_latency_ms": raw["modeled_latency_ms"],
                   "joint_posterior_storage_bytes_f64": 8 * len(STATES), "wire": raw["wire"]}, common["threshold"])


def method_predictions(row, ledger, association, fits, common):
    private = {name: local_predict(common["terminal_prior"], fits[camera], row["local_rgb_cues"][camera],
                                  common["threshold"])
               for name, camera in ((CURRENT, "current"), (TEMPORAL, "history"))}
    compact = {key: {cap: compact_result(row, policy, cap, association["compact"], common)
                     for cap in common["caps"]} for key, policy in TRANSFER.items()}
    full, witnesses = central_full(row, association["full"], fits["history"], common)
    markers = row["identity_support"]
    diagnostic = {}
    if markers is not None:
        gated = compact_result(row, TRANSFER[SELECTIVE], PRIMARY_CAP, association["compact"], common, markers=markers)
        if gated["serialized_bytes"] != compact[SELECTIVE][PRIMARY_CAP]["serialized_bytes"]:
            raise ValueError("An identity diagnostic changed the communication charge")
        diagnostic = {IDENTITY: gated, "observed_marker_support_by_witness": markers}
    return {**private, LEDGER: ledger["deterministic"], PARTICLE: ledger["particle"],
            FULL: full, FULL_WITNESSES: witnesses,
            **{name: options[PRIMARY_CAP] for name, options in compact.items()}, "budget_diagnostics": compact, **diagnostic}


def baselines(row, ledger, association, fits, common):
    prediction = method_predictions(row, ledger, association, fits, common)
    return {**prediction, **{key: row[key] for key in ("source", "source_manifest_sha256", "family", "variant", "truth")}}


def score_group(group, groups, ledgers, samples, common):
    source, size = group["source_group"], group["team_size"]
    others = [item for item in groups if item["team_size"] == size and item["source_group"] != source]
    fits = {kind: fit_cue(others, kind) for kind in ("current", "history")}
    association = {name: profiles(available[size], common["reference"]["association"], heldout=source)
                   for name, available in samples.items()}
    cases = []
    for row in group["episodes"]:
        ledger = ledgers[row["source"]]
        if (row["truth"], row["variant"], row["family"]) != (ledger["truth"], ledger["variant"], ledger["family"]):
            raise ValueError("Compared methods use different recorded truths or variants")
        cases.append(baselines(row, ledger, association, fits, common))
    if cases[0]["family"] == "F6":
        a, b = cases
        for name in (*common["methods"], FULL_WITNESSES):
            fields = ("belief", "decision", "serialized_bytes")
            if any(a[name][key] != b[name][key] for key in fields):
                raise ValueError(f"{name} changes under a matched hidden alternative")
        if a["budget_diagnostics"] != b["budget_diagnostics"]:
            raise ValueError("Budget diagnostics changed across the hidden pair")
        if IDENTITY in a and any(a[name] != b[name] for name in (IDENTITY, "observed_marker_support_by_witness")):
            raise ValueError("Marker support changed across the hidden pair")
    return {"group": source, "team_size": size, "outcomes": cases}


def team_metrics(groups, method, *, inspect_cost=.25, wrong_cost=1.):
    result = group_metrics(groups, method)
    brier, costs, data = [], [], []
    peak = 0
    for item in groups:
        squared, decision, byte_counts = [], [], []
        for row in item["outcomes"]:
            state = row[method]
            squared.append(sum((p - float(label == row["truth"])) ** 2 for label, p in state["belief"].items()))
            decision.append(inspect_cost if state["decision"] == "request_inspection"
                            else wrong_cost * float(state["decision"] != row["truth"]))
            byte_counts.append(state["serialized_bytes"])
            peak = max(peak, state["serialized_bytes"])
        brier.append(sum(squared) / len(squared))
        costs.append(sum(decision) / len(decision))
        data.append(sum(byte_counts) / len(byte_counts))
    return {**result, "group_mean_brier": sum(brier) / len(brier),
            "group_mean_decision_cost": sum(costs) / len(costs), "group_mean_bytes": sum(data) / len(data),
            "peak_bytes": peak}


def by_family(groups, method):
    results = {}
    for group in groups:
        family = group["outcomes"][0]["family"]
        counts = results.setdefault(family, {"groups": 0, "variants": 0, "exact_answers": 0, "inspections": 0})
        counts["groups"] += 1
        counts["variants"] += len(group["outcomes"])
        counts["exact_answers"] += sum(row[method]["decision"] == row["truth"] for row in group["outcomes"])
        counts["inspections"] += sum(row[method]["decision"] == "request_inspection" for row in group["outcomes"])
    return results


def summarize(groups, common):
    by_size = {size: [g for g in groups if g["team_size"] == size] for size in (3, 4, 8)}
    return {size: {name: {**team_metrics(items, name, inspect_cost=common["inspect_cost"],
                                        wrong_cost=common["wrong_cost"]), "families": by_family(items, name)}
                   for name in common["methods"]} for size, items in by_size.items()}


def compare_caps(groups, common):
    results = {}
    for cap in common["caps"]:
        budgets = [{**group, "outcomes": [{**row, **{name: options[cap]
                                                   for name, options in row["budget_diagnostics"].items()}}
                                          for row in group["outcomes"]]} for group in groups]
        results[cap] = summarize(budgets, {**common, "methods": list(TRANSFER)})
    return results


def cohort_matches(records, expected):
    actual = {scene["source"]: scene["source_manifest_sha256"]
              for group in records for scene in group["episodes"]}
    if not actual:
        raise ValueError("No admitted recordings were found for the validation cohort")
    groups = sorted({group["source_group"] for group in records})
    if expected and (expected["config_sha256"] != sha(CONFIG)
                     or actual != expected["manifests"] or groups != expected["source_groups"]):
        raise ValueError("Recorded source manifests differ from the selected validation cohort")
    return {"config_sha256": sha(CONFIG), "source_groups": groups, "manifests": actual}


def fitted_profiles(records, samples, common, ledger):
    output = {}
    for size, data in ledger["teams"].items():
        groups = [group for group in records if group["team_size"] == size]
        association = {name: profiles(available[size], common["reference"]["association"])
                       for name, available in samples.items()}
        output[size] = {"receiver": {mode: fit_cue(groups, mode) for mode in ("current", "history")},
                        "association": association, "particle_count": data["selected_particle_count"],
                        "rule_report_likelihoods_by_extra_mount": data["mounted_report_likelihoods"]}
    return output


def run(folder, heartbeat, kind, expected=None, *, inspect_markers=False):
    started = time.monotonic()
    torch.set_num_threads(3)
    config = load(CONFIG)
    study = load(PILOT / "story-learning.json")
    model = next(row for row in study["models"] if row["kind"] == kind)
    frozen, gains = verified_model(model)
    head = EvidenceHead().eval()
    head.load_state_dict(frozen["weights"], strict=True)
    only = set(expected["source_groups"]) if expected else None
    records = list(read_groups(folder, (load_backbone(study), head, frozen["temperature"]),
                               config["communication_reply_window_ms"], heartbeat,
                               only=only, inspect_markers=inspect_markers))
    cohort = cohort_matches(records, expected)
    receiver_prior = exact_filter(frozen["state_prior"], [])["belief"]
    common = {"reference": learned_prior_map(frozen["priors"]), "gains": gains,
              "state_prior": frozen["state_prior"], "threshold": frozen["threshold"],
              "terminal_prior": receiver_prior, "caps": config["communication_byte_caps"],
              "methods": config["primary_methods"], "inspect_cost": config["inspect_cost"],
              "wrong_cost": config["wrong_answer_cost"]}
    ledger = ledger_audit(folder, only)
    ledgers = {row["source"]: row for team in ledger["teams"].values()
               for group in team["outcomes"] for row in group["outcomes"]}
    samples = {"compact": association_samples(records), "full": full_samples(records)}
    result = [score_group(group, records, ledgers, samples, common) for group in records]
    report = describe_result(result, common, kind, model, study, cohort, started, inspect_markers)
    return {**report, "full_validation_fit": fitted_profiles(records, samples, common, ledger),
            "base_particle_report_sha256": ledger["base_particle_report_sha256"]}


def describe_result(result, common, kind, model, study, cohort, started, inspect_markers):
    optional = {"marker_support_diagnostic": summarize(result, {**common, "methods": [IDENTITY]})} if inspect_markers else {}
    return {"scope": "Seven-method repeated-source simulator validation rehearsal",
            "checkpoint_kind": kind, "checkpoint_sha256": model["checkpoint_sha256"], "config_sha256": sha(CONFIG),
            "encoder_sha256": study["frozen_backbone_sha256"], "cohort": cohort, "implementation_sha256": source_hashes(),
            "validation_source_groups": len({row["group"] for row in result}),
            "encoded_features": "Each witness's full private RGB history encoded once. Compact arms select the sealed eight positions.",
            "matched_compact": "Same candidate grants, classifier, fit, factor priorities, 3,072-byte ceiling and 90 ms window.",
            "central_full_bytes": "Sum of original compressed PNG payloads with no invented network latency.",
            "metrics_by_team_size": summarize(result, common), "budget_diagnostics_by_cap": compare_caps(result, common),
            "central_receiver_cue_ablation": summarize(result, {**common, "methods": [FULL_WITNESSES]}),
            "elapsed_s": time.monotonic() - started, **optional,
            "outcomes": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--kind", choices=("events_only", "events_plus_reveal"), default="events_only")
    parser.add_argument("--cohort", type=Path)
    parser.add_argument("--diagnose-identity", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with admitted(settings(), "smart-swarm-primary-validation-matrix") as (grant, heartbeat):
        output = run(args.root, heartbeat, args.kind, load(args.cohort) if args.cohort else None,
                     inspect_markers=args.diagnose_identity)
    atomic(args.output, {**output, "reservation": grant})
    print(args.output)


if __name__ == "__main__":
    main()
