"""Replay a modeled delivery delay on the sealed primary collective-memory cohort."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import tarfile

from archive_primary_release import INDEX, KINDS, read_member, verify as verify_calibration
from archive_primary_results import verify as verify_results
from audit_primary_pairs import conservative_interval, paired_rows
from compare_primary_supervision import TEAMS, describe
from freeze_primary_profiles import PILOT
from score_primary_test import check_prediction_receipt
from score_primary_validation import SELECTIVE, choose, exact_from_reports, on_time_compact
from smart_swarm.contract import json_bytes, read_ns
from smart_swarm.episode import atomic, sha
from smart_swarm.evidence_budget_replay import wire_size
from smart_swarm.evidence_comparison import learned_prior_map
from smart_swarm.evidence_queries import time_field

RESULTS = PILOT / "primary-independent-results-homebrain.tar.gz"
CALIBRATION = PILOT / "primary-frozen-calibration-homebrain.tar.gz"
RESULTS_SHA256 = "cb8e12354782d5bdbba787c3aceedc9c3bb06475e872e049896fe9d9c51ecf2f"
DELAYS = (0, 20_000_000, 20_000_001, 50_000_000, 50_000_001)
DELAY_NAMES = ("0 ms", "20 ms", "20 ms + 1 ns", "50 ms", "50 ms + 1 ns")
MANIFEST = "configuration/primary-profiles-frozen.json"
SCOPE = "Exploratory deadline stress replay on the consumed primary simulation sources"


def original_archives(results, calibration):
    proof = verify_results(results)
    if proof["archive_sha256"] != RESULTS_SHA256 or proof["scope"] != "test" or proof["independent_source_groups"] != 100:
        raise ValueError("The delay replay uses the original 100-source result archive")
    with tarfile.open(results, "r:gz") as archive:
        binding = json.loads(read_member(archive, INDEX))
    frozen = verify_calibration(calibration)
    if (frozen["archive_sha256"] != binding["external_calibration_archive"]["sha256"]
            or frozen["profile_sha256"] != binding["profiles_sha256"]):
        raise ValueError("The frozen calibration archive differs from the original test release")
    return binding, frozen["archive_sha256"]


def frozen_parameters(path, profile):
    import torch

    with tarfile.open(path, "r:gz") as archive:
        config = json.loads(read_member(archive, "configuration/primary-memory-study.json"))
        common = {}
        for kind in KINDS:
            data = read_member(archive, f"models/{kind}.pt")
            if sha256(data).hexdigest() != profile["fits"][kind]["checkpoint_sha256"]:
                raise ValueError("The delay replay uses different frozen fusion priors")
            state = torch.load(BytesIO(data), weights_only=True, map_location="cpu")
            common[kind] = {"state_prior": state["state_prior"], "reference": learned_prior_map(state["priors"])}
    if config["communication_reply_window_ms"] != 90:
        raise ValueError("The original decision cutoff must remain 90 ms")
    return config, common


def projected_manifest(payload, source):
    manifest = json.loads(payload)
    fields = ("scene_group_id", "scaling_team_size", "query_boundary", "decision_boundary")
    if manifest["lineage"]["split"] != "test" or manifest["world_episode_id"] != Path(source).name:
        raise ValueError("The archive contains a manifest outside the original test provenance")
    return {key: manifest[key] for key in fields}


def forecast_inputs(path, binding):
    mapping = {row["archive_member"]: str(Path(path).parent)
               for path, row in binding["original_recording_manifests"].items()}
    sources = {str(Path(path).parent): row["sha256"] for path, row in binding["original_recording_manifests"].items()}
    members = {f"scores/primary-test-{kind}-predictions.json": kind for kind in KINDS}
    projections, receipts = {}, {}
    with tarfile.open(path, "r:gz") as archive:
        profile = json.loads(read_member(archive, MANIFEST))
        for name in sorted(mapping):
            source = mapping[name]
            projections[source] = projected_manifest(read_member(archive, name), source)
        for name, kind in members.items():
            saved = json.loads(read_member(archive, name))
            identity = {"scope": "independent primary test", "checkpoint_kind": kind, "manifests": sources,
                        "profiles_sha256": binding["profiles_sha256"],
                        "primary_test_release_sha256": binding["primary_test_release_sha256"],
                        "source_groups": sorted({row["scene_group_id"] for row in projections.values()})}
            sealed = check_prediction_receipt(saved, identity, sources)
            receipts[kind] = {source: row[SELECTIVE] for source, row in sealed.items()}
    return profile, projections, receipts


def delayed_reply(reply, delay):
    return {**reply, "delivered_ns": time_field(read_ns(reply["delivered_ns"]) + delay)}


def replay_window(saved, query, deadline, delay, calibration, common, threshold):
    wire = saved["wire"]
    if any(read_ns(row["issued_ns"]) != query for row in wire["requests"]):
        raise ValueError("Saved messages do not use the sealed query origin")
    emitted = [delayed_reply(reply, delay) for reply in wire["replies"]]
    sent = wire_size(wire["requests"]) + wire_size(wire["advertisements"])
    if sent + wire_size(emitted) != saved["serialized_bytes"]:
        raise ValueError("The delay changed emitted bytes or the archived byte contract")
    usable = on_time_compact(emitted, deadline)
    replay = choose(exact_from_reports(usable, calibration, common), threshold)
    return {"belief": replay["belief"], "decision": replay["decision"], "serialized_bytes": saved["serialized_bytes"],
            "modeled_timely_bytes": sent + wire_size(usable),
            "reply_factors_before_deadline": dict(Counter(row["factor"] for row in usable)),
            "replies_after_deadline": len(emitted) - len(usable)}


def forecasts_for_source(manifest, saved, fit, common, source):
    size = manifest["scaling_team_size"]
    mount = fit["mount_and_receiver_profiles"][str(size)]
    query = read_ns(manifest["query_boundary"]["monotonic_ns"])
    deadline = read_ns(manifest["decision_boundary"]["monotonic_ns"])
    if deadline - query != 90_000_000:
        raise ValueError("A sealed forecast used a different deadline")
    rows = {str(delay): replay_window(saved, query, deadline, delay, mount["association"]["compact"],
                                       common, mount["decision_thresholds"][SELECTIVE]) for delay in DELAYS}
    original = rows["0"]
    difference = max(abs(original["belief"][slot] - prob) for slot, prob in saved["belief"].items())
    if difference > 1e-12 or original["decision"] != saved["decision"] or original["replies_after_deadline"]:
        raise ValueError("The unchanged delay control failed to reproduce the original model")
    return {"source": source, "group": manifest["scene_group_id"], "size": size,
            "baseline_max_probability_difference": difference, "delays": rows}


def paired_equal(rows):
    pairs = defaultdict(list)
    for row in rows:
        pairs[(row["group"], row["size"])].append(row)
    repeated = [group for group in pairs.values() if len(group) > 1]
    if len(pairs) != 300 or len(repeated) != 48 or any(
            len(pair) != 2 or pair[0]["delays"] != pair[1]["delays"] for pair in repeated):
        raise ValueError("The delay changed decisions for a pair of identical F6 policy inputs")
    return len(repeated)


def forecast(results, calibration, binding, calibration_sha):
    profile, inputs, receipts = forecast_inputs(results, binding)
    config, priors = frozen_parameters(calibration, profile)
    forecasts = {kind: [forecasts_for_source(manifest, receipts[kind][source], profile["fits"][kind], priors[kind], source)
                        for source, manifest in sorted(inputs.items())] for kind in KINDS}
    counts = {kind: paired_equal(rows) for kind, rows in forecasts.items()}
    return {"schema": "smart-swarm-frozen-delay-forecast-v1", "scope": SCOPE,
            "primary_archive_sha256": RESULTS_SHA256, "calibration_archive_sha256": calibration_sha,
            "profile_sha256": binding["profiles_sha256"], "delay_ns": list(DELAYS), "fixed_window_ms": 90,
            "fit_performed": False, "labels_used_for_forecasts": False, "paired_equal_by_head": counts,
            "costs": {key: config[key] for key in ("inspect_cost", "wrong_answer_cost")},
            "payload_sha256": sha256(json_bytes(forecasts)).hexdigest(), "predictions": forecasts}


def grouped_outcomes(archive, kind, rows):
    saved = json.loads(read_member(archive, f"scores/primary-test-{kind}.json"))
    predictions = {row["source"]: row for row in rows}
    used = set()
    result = []
    for group in saved["outcomes"]:
        cases = []
        for entry in group["outcomes"]:
            source = entry["source"]
            row = predictions[source]
            if source in used or (row["group"], row["size"]) != (group["group"], group["team_size"]):
                raise ValueError("Evaluation truth is attached to a different source or team")
            original = entry[SELECTIVE]
            if row["delays"]["0"]["decision"] != original["decision"]:
                raise ValueError("Original evaluation score disagrees with the verified forecast")
            used.add(source)
            cases.append({**{key: entry[key] for key in ("source", "family", "variant", "truth")}, **row["delays"]})
        if len(cases) == 2 and ({case["family"] for case in cases} != {"F6"} or len({case["truth"] for case in cases}) != 2):
            raise ValueError("The coupled delay test did not preserve the hidden evaluator assignments")
        result.append({**group, "outcomes": cases})
    if used != predictions.keys():
        raise ValueError("The score omitted a frozen forecast")
    return result


def summarize_delay(groups, delay, costs):
    name = str(delay)
    statistics = describe(groups, name, costs)
    baseline, shifted = [paired_rows(groups, str(value)) for value in (0, delay)]
    interval = conservative_interval(shifted, baseline)
    values = [row[name] for group in groups for row in group["outcomes"]]
    equal_mean = lambda field: sum(sum(row[name][field] for row in group["outcomes"]) / len(group["outcomes"])
                                   for group in groups) / len(groups)
    return {**statistics, "container_correct_paired_vs_original": interval,
            "mean_bytes_on_time": equal_mean("modeled_timely_bytes"),
            "late_reply_count": sum(row["replies_after_deadline"] for row in values),
            "on_time_replies_by_factor": dict(sum((Counter(row["reply_factors_before_deadline"]) for row in values), Counter()))}


def score(results, forecasts_path):
    saved = json.loads(forecasts_path.read_bytes())
    if (sha(results) != saved["primary_archive_sha256"]
            or sha256(json_bytes(saved["predictions"])).hexdigest() != saved["payload_sha256"]
            or saved["labels_used_for_forecasts"] is not False):
        raise ValueError("Score only the sealed, label-free delivery forecasts")
    grouped = {}
    with tarfile.open(results, "r:gz") as archive:
        for kind in KINDS:
            grouped[kind] = grouped_outcomes(archive, kind, saved["predictions"][kind])
    summary = {}
    for kind, groups in grouped.items():
        by_team = {str(size): [row for row in groups if row["team_size"] == size] for size in TEAMS}
        summary[kind] = {size: {str(delay): summarize_delay(rows, delay, saved["costs"]) for delay in DELAYS}
                         for size, rows in by_team.items()}
    return {"schema": "smart-swarm-frozen-delay-analysis-v1", "scope": SCOPE, "forecast_sha256": sha(forecasts_path),
            "primary_archive_sha256": saved["primary_archive_sha256"], "paired_equal_by_head": saved["paired_equal_by_head"],
            "independent_source_groups": 100, "team_sizes": list(TEAMS), "variants_across_teams_per_head": 348,
            "modeled_not_measured_transport": True, "fit_performed": False, "optimizer_updates": 0, "new_simulator_steps": 0,
            "metrics_by_head_and_team": summary}


def markdown(report):
    names = {"events_only": "Events only", "events_plus_reveal": "Events + reveal"}
    lines = ["# Delivery-delay sensitivity of collective memory", "",
             "Exploratory replay of 100 used simulator sources. The same source groups recur at each team size. "
             "Summary shows N=8. The machine-readable report includes N=3, 4 and 8.", "",
             "| Evidence head | Added delay | Container correct | All resolvable correct | F6 pairs inspect | Wrong variants | Mean emitted bytes | Mean on-time bytes |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for head in KINDS:
        for delay, label in zip(DELAYS, DELAY_NAMES, strict=True):
            row = report["metrics_by_head_and_team"][head]["8"][str(delay)]
            containers = f"{row['container_correct']}/{row['container_count']}"
            resolved = f"{row['resolvable_correct']}/{row['resolvable_count']}"
            inspections = f"{row['f6_inspections'] // 2}/{row['f6_variants'] // 2}"
            lines.append(f"| {names[head]} | {label} | {containers} | {resolved} | {inspections} | "
                         f"{row['confident_wrong']} | {row['group_mean_bytes']:.0f} | {row['mean_bytes_on_time']:.0f} |")
    lines.extend(["", "Each selected reply kept its original sender, content and byte cost. "
                  "The replay shifted delivery time and reapplied the frozen 90 ms cutoff. "
                  "An arrival exactly at the deadline is admitted. Timing is modeled. "
                  "F6 alternatives are paired and counted once per source.", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--calibration", type=Path, default=CALIBRATION)
    parser.add_argument("--forecasts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    args = parser.parse_args()
    outputs = (args.forecasts, args.output, args.table)
    if len({path.resolve() for path in outputs}) != 3 or any(path.exists() for path in outputs):
        parser.error("Choose three distinct unused report paths")
    binding, calibration_sha = original_archives(args.results, args.calibration)
    atomic(args.forecasts, forecast(args.results, args.calibration, binding, calibration_sha))
    result = score(args.results, args.forecasts)
    atomic(args.output, result)
    with args.table.open("x", encoding="utf-8") as handle:
        handle.write(markdown(result))
    print(args.output)


if __name__ == "__main__":
    main()
