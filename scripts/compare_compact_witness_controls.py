"""Compare compact witness choices on frozen, previously inspected validation sources."""

import argparse
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
import random
from statistics import mean

from audit_screened_association import PROFILE, STUDY, model_parameters
from capture_memory_validation import CONFIG
from compare_primary_supervision import KINDS, TEAMS, authority, evaluator_record, read_scores, score_paths, verify_scores
from freeze_primary_profiles import PILOT
from score_primary_validation import PRIMARY_CAP, SELECTIVE, choose, exact_from_reports, on_time_compact
from smart_swarm.cli import settings
from smart_swarm.episode import atomic, json_bytes, load, sha
from smart_swarm.evidence_budget_replay import wire_size
from smart_swarm.evidence_comparison import grouped_mean, nll
from smart_swarm.evidence_inputs import runtime_signature
from smart_swarm.evidence_queries import receive_reply, request_wire
from smart_swarm.research_jobs import admitted
from smart_swarm.team_evidence import announcement, opportunities, select

SEED = 260111
INSPECT = "request_inspection"
EVENT_LABEL = {"containment": "settled_placement", "association": "relocation_assignment"}
ARMS = ("factor_coverage", "central_identical_selection", "broadcast_capped", "fixed_original", "most_recent",
        "random_eligible", "validation_quality", "all_witness_reference")


def quality_advertisements(window, posteriors):
    query, deadline, ads = window
    updated = [{**ad, "quality": f"{max(posteriors[ad['sender']].values()):.2f}"} for ad in ads]
    return query, deadline, updated


def quality_bin(ad):
    return f"{ad['factor']}/{min(4, int(float(ad['quality']) * 5))}"


def reliability(cases, held_out):
    observations = defaultdict(lambda: defaultdict(list))
    for row in cases:
        if row["scene_group_id"] != held_out:
            for ad in row["window"][2]:
                truth = row["event_labels"][EVENT_LABEL[ad["factor"]]]
                scores = row["posteriors"][ad["sender"]]
                if truth is not None:
                    if truth not in scores:
                        raise ValueError("An event label is outside its frozen likelihood support")
                    hit = max(scores, key=scores.get) == truth
                    for key in (quality_bin(ad), ad["factor"]):
                        observations[key][row["scene_group_id"]].append(hit)
    return {key: {"probability": (1 + sum(mean(hits) for hits in grouped.values())) / (2 + len(grouped)),
                  "source_groups": sorted(grouped)} for key, grouped in sorted(observations.items())}


def custom_selection(window, ordered):
    query, deadline, ads = window
    metadata_bytes = wire_size([announcement(ad) for ad in ads])
    if metadata_bytes > PRIMARY_CAP:
        raise ValueError("Shared advertisements exceed the comparison envelope")
    selected, covered = [], set()
    for ad in ordered:
        trial = [*selected, ad]
        estimate = metadata_bytes + wire_size([request_wire(trial, query)]) + sum(row["reply_bytes"] for row in trial)
        if ad["factor"] not in covered and estimate <= PRIMARY_CAP and query + ad["delay_ns"] <= deadline:
            selected = trial
            covered.add(ad["factor"])
    return selected


def ranked(window, quality):
    query, _, ads = window
    keys = sorted((ad["sender"], ad["factor"], ad["capture_end_ns"]) for ad in ads)
    seed = sha256(json_bytes([SEED, query, keys])).digest()
    rng = random.Random(seed)
    order = {sender: rng.random() for sender, _, _ in keys}
    random_rank = sorted(ads, key=lambda ad: order[ad["sender"]])

    def estimate(ad):
        fallback = quality.get(ad["factor"], {"probability": .5})
        return quality.get(quality_bin(ad), fallback)["probability"]

    quality_rank = sorted(ads, key=lambda ad: (-estimate(ad), -int(ad["capture_end_ns"]), ad["sender"]))
    return random_rank, quality_rank


def transcript(window, chosen, posteriors):
    query, _, ads = window
    return {"advertisements": [announcement(ad) for ad in ads],
            "requests": [request_wire(chosen, query)] if chosen else [],
            "replies": [receive_reply(ad, query, lambda factor: posteriors[ad["sender"]]) for ad in chosen]}


def score_wire(wire, window, fit, common):
    query, deadline, _ = window
    replies = on_time_compact(wire["replies"], deadline)
    state = exact_from_reports(replies, fit["association"]["compact"], common)
    bytes_by_kind = {key: wire_size(entries) for key, entries in wire.items()}
    spent = sum(bytes_by_kind.values())
    latency = max((int(row["delivered_ns"]) for row in replies), default=query) - query
    return {**choose(state, fit["decision_thresholds"][SELECTIVE]), "wire": wire, "serialized_bytes": spent,
            "bytes_by_kind": bytes_by_kind, "modeled_latency_ms": latency / 1e6,
            "within_cap": spent <= PRIMARY_CAP and len(replies) == len(wire["replies"])}


def forecast(window, posteriors, fit, common, quality):
    query, deadline, ads = window
    gains = common["gains"]
    random_rank, quality_rank = ranked(window, quality)
    selected = {name: select(method, ads, query, deadline, PRIMARY_CAP, gains) for name, method in
                (("factor_coverage", "factor_coverage"), ("fixed_original", "anchor"),
                 ("most_recent", "recent_per_factor"), ("broadcast_capped", "broadcast"))}
    selected.update(random_eligible=custom_selection(window, random_rank), validation_quality=custom_selection(window, quality_rank),
                    all_witness_reference=ads)
    result = {name: score_wire(transcript(window, choices, posteriors), window, fit, common)
              for name, choices in selected.items()}
    decentralized = result["factor_coverage"]
    central = score_wire(decentralized["wire"], window, fit, common)
    if central != decentralized:
        raise ValueError("Identical compact inputs produced a different centralized forecast")
    result["central_identical_selection"] = central
    if any(not result[name]["within_cap"] for name in ARMS if name != "all_witness_reference"):
        raise ValueError("A matched compact arm exceeded its original cap or deadline")
    return result


def cached_inputs(kind, profile, identity, truths):
    source = PILOT / f"study-validation-six-{kind}.json"
    archive = load(source)
    if archive["checkpoint_sha256"] != profile["fits"][kind]["checkpoint_sha256"]:
        raise ValueError("Cached likelihoods differ from the frozen model")
    rows = []
    for group in archive["cohorts"]:
        size = group["team_size"]
        for case in group["episodes"]:
            path = Path(case["source"])
            group_id, verified = truths[str(path)]
            if identity["manifests"][str(path)] != sha(path / "manifest.json") or case["truth"] != verified["truth"]:
                raise ValueError("The cached input differs from its verified original manifest or outcome")
            raw = opportunities(path)
            posteriors = case["frozen_scores_by_witness"]
            if set(posteriors) != {ad["sender"] for ad in raw[2]}:
                raise ValueError("The archive lacks a likelihood for an eligible witness")
            rows.append({**verified, "team_size": size, "scene_group_id": group_id, "posteriors": posteriors,
                         "window": quality_advertisements(raw, posteriors)})
    return rows, sha(source)


def score_cases(rows, common, mounts):
    groups = sorted({row["scene_group_id"] for row in rows})
    qualities = {group: reliability(rows, group) for group in groups}
    scored = []
    f6 = defaultdict(list)
    for row in rows:
        group = row["scene_group_id"]
        fit = mounts[str(row["team_size"])]
        methods = forecast(row["window"], row["posteriors"], fit, common, qualities[group])
        item = {key: value for key, value in row.items() if key not in ("window", "posteriors", "event_labels")}
        scored.append({**item, "methods": methods})
        if row["family"] == "F6":
            f6[group, row["team_size"]].append((runtime_signature(Path(row["source"])), methods, row["truth"]))
    for matched in f6.values():
        first, second = matched
        if first[0] != second[0] or first[1] != second[1] or first[2] == second[2]:
            raise ValueError("The matched transport or selection changed under an F6 truth substitution")
    return scored, qualities, len(f6)


def summarize(scored, study):
    def measures(rows, arm):
        outcomes = [{**row["methods"][arm], "scene_group_id": row["scene_group_id"], "terminal": row["truth"]}
                    for row in rows]
        containers = [result for row, result in zip(rows, outcomes, strict=True)
                      if row["family"] in ("F1", "F2", "F3", "F5") and row["truth"] in ("left", "right")]
        cost = lambda x: study["inspect_cost"] if x["decision"] == INSPECT else study["wrong_answer_cost"] * (x["decision"] != x["terminal"])
        return {"independent_sources": len({row["scene_group_id"] for row in rows}), "variants": len(outcomes),
                "resolvable_container_variants": len(containers),
                "correct_first_container_choices": sum(row["decision"] == row["terminal"] for row in containers),
                "correct_answers": sum(row["decision"] == row["terminal"] for row in outcomes),
                "wrong_answers": sum(row["decision"] not in (row["terminal"], INSPECT) for row in outcomes),
                "inspections": sum(row["decision"] == INSPECT for row in outcomes),
                "cases_within_original_cap": sum(row["within_cap"] for row in outcomes),
                "mean_total_serialized_bytes": grouped_mean(outcomes, lambda x: x["serialized_bytes"]),
                "mean_modeled_latency_ms": grouped_mean(outcomes, lambda x: x["modeled_latency_ms"]),
                "source_weighted_decision_cost": grouped_mean(outcomes, cost), "source_weighted_nll": nll(outcomes)}

    return {str(size): {arm: measures([row for row in scored if row["team_size"] == size], arm) for arm in ARMS}
            for size in TEAMS}


def audit(grant, heartbeat):
    profile, study = load(PROFILE), load(STUDY)
    identity, traces = authority(PROFILE, "validation")
    groups = {key: [evaluator_record(trace, identity["manifests"]) for trace in rows] for key, rows in traces.items()}
    scores = score_paths("validation")
    config = load(CONFIG)
    report = {}
    for kind in KINDS:
        heartbeat()
        indexed = verify_scores(*read_scores(scores[kind], kind, identity), groups)
        truths = {row["source"]: (group, {key: row[key] for key in ("source", "source_manifest_sha256", "family", "variant",
                                                                  "event_labels", "truth")})
                  for (group, _), bundle in indexed.items() for row in bundle["outcomes"]}
        cached, digest = cached_inputs(kind, profile, identity, truths)
        common, fit = model_parameters(profile, study, kind)
        model = next(row for row in study["models"] if row["kind"] == kind)
        common["gains"] = model["communication"]["validation_expected_information_nats"]
        cases, calibration, verified_pairs = score_cases(cached, common, fit)
        report[kind] = {"checkpoint_sha256": model["checkpoint_sha256"], "cached_likelihoods_sha256": digest,
                        "sealed_validation_score_sha256": sha(scores[kind]), "cases": cases,
                        "quality_calibration_by_held_out_group": calibration, "f6_pairs_by_team": verified_pairs,
                        "metrics_by_team": summarize(cases, config)}
    return {"scope": "exploratory replay on six previously inspected validation source groups", "optimizer_updates": 0,
            "profiles_sha256": identity["profiles_sha256"], "budget_bytes": PRIMARY_CAP, "coupled_seed": SEED,
            "reservation": {key: grant[key] for key in ("scope", "locks")}, "heads": report,
            "metadata": "All arms pay for the same two-decimal locally computed confidence advertisements.",
            "broadcast_reference": "Unconstrained all-witness offer charged with the same format. Check cap counts before comparing."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with admitted(settings(), "smart-swarm-compact-witness-controls", archive_only=True) as (grant, heartbeat):
        result = audit(grant, heartbeat)
    atomic(args.output, result)
    print(args.output)


if __name__ == "__main__":
    main()
