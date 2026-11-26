"""Bounded N=3 controls on a preregistered source subset."""

from collections import defaultdict
import hashlib
from itertools import combinations

import numpy as np

from score_primary_validation import PRIMARY_CAP, SELECTIVE, choose, exact_from_reports, likelihoods_from_reports, on_time_compact, team_metrics
from smart_swarm.contract import json_bytes
from smart_swarm.evidence_inputs import runtime_signature
from smart_swarm.memory import STATES
from smart_swarm.particle_memory import LOCATIONS, exact_filter, particle_filter, result, total_variation
from smart_swarm.team_evidence import exchange


def reexchange(row, association, common, threshold, strategy, gains, scores, *, single=False):
    def posterior(agent, factor):
        return scores[agent]

    raw = exchange(strategy, row["window"], PRIMARY_CAP, gains, posterior, common["reference"], common["state_prior"],
                   max_replies=1 if single else None)
    replies = on_time_compact(raw["wire"]["replies"], row["window"][1])
    state = exact_from_reports(replies, association, common)
    return choose({**raw, **state}, threshold)


def delivery_controls(replies, deadline, association, common, reference):
    stale = [{**row, "delivered_ns": str(deadline + 1)} for row in replies]
    streams = {"out_of_order": replies[::-1], "retransmitted": [*replies, *replies], "post_deadline": [*stale, *replies]}
    distances = {}
    for name, stream in streams.items():
        kept = on_time_compact(stream, deadline)
        current = exact_from_reports(kept, association, common)
        distances[name] = total_variation(reference, current["joint"])
    if any(value > 1e-12 for value in distances.values()):
        raise ValueError("Compact fusion changed under an out-of-order, duplicate or late packet")
    return distances


def evidence_controls(root, row, selected, association, common, threshold, *, count, seed):
    replies = selected["wire"]["replies"]
    factors = likelihoods_from_reports(on_time_compact(replies, row["window"][1]), association, common)
    matched = particle_filter(common["state_prior"], factors, count, seed=seed)
    withheld = exact_filter(common["state_prior"], [item for item in factors if item[0] != "association"])
    neutral = dict.fromkeys(common["gains"], 0.)
    scores = row["frozen_scores_by_witness"]
    costs = {key: selected[key] for key in ("serialized_bytes", "modeled_latency_ms")}
    return {
        "learned_broadcast": reexchange(row, association, common, threshold, "broadcast", common["gains"], scores),
        "single_information_gain": reexchange(row, association, common, threshold, "recent_per_factor",
                                              common["gains"], scores, single=True),
        "single_latest_witness": reexchange(row, association, common, threshold, "recent_per_factor",
                                           neutral, scores, single=True),
        "reversed_witness_time": reexchange(row, association, common, threshold, "factor_coverage", common["gains"],
                                           row["reversed_clip_scores"]),
        "withheld_association": choose({**withheld, **costs}, threshold),
        "particle_matched_learned_factors": choose({**matched, **costs}, threshold),
        "matched_particle_joint_tv": total_variation(matched["joint"], selected["joint"]),
        "delivery_controls_max_joint_tv": delivery_controls(replies, row["window"][1], association, common,
                                                            selected["joint"]),
        "receiver_last_frame_and_history": history_identity(root),
    }


def history_identity(root):
    scoped = runtime_signature(root)
    frames = {name: [camera["sha256"] for packet in records for camera in packet["cameras"]]
              for name, records in scoped["private_history"].items()}
    return {"last_receiver_rgb_sha256": frames["rosmaster-1"][-1],
            "witness_history_sha256": hashlib.sha256(json_bytes({name: seq for name, seq in frames.items()
                                                                if name != "rosmaster-1"})).hexdigest()}


def oracles(row, common, threshold):
    labels = row["event_labels"]
    a, b = labels["settled_placement"], labels["relocation_assignment"]
    weights = [float((a is None or a == cube) and (b is None or b == assignment)) * prior
               for (cube, assignment), prior in zip(STATES, exact_filter(common["state_prior"], [])["joint"], strict=True)]
    belief = result(np.asarray(weights) / sum(weights))
    event = choose({**belief, "serialized_bytes": 0}, threshold)
    terminal = {"belief": {key: float(key == row["truth"]) for key in LOCATIONS},
                "decision": row["truth"], "serialized_bytes": 0}
    return {"isolated_event_oracle": event, "isolated_terminal_oracle": terminal}


def matched_final_views(groups):
    buckets = defaultdict(list)
    for group in groups:
        for row in group["outcomes"]:
            if row["family"] in ("F1", "F2", "F3", "F5"):
                history = row["receiver_last_frame_and_history"]
                key = history["last_receiver_rgb_sha256"]
                buckets[key].append((group["group"], history["witness_history_sha256"], row["truth"]))
    matched = []
    for entries in buckets.values():
        for (first, history_a, truth_a), (second, history_b, truth_b) in combinations(entries, 2):
            if first != second and history_a != history_b and truth_a != truth_b:
                matched.append(sorted((first, second)))
    return {"same_receiver_last_rgb_distinct_witness_histories_and_answers": matched,
            "unique_source_groups": len({group for pair in matched for group in pair}),
            "scope": "Resolvable archive matches on exact last receiver RGB with different histories and correct answers."}


def report(groups, common, threshold):
    material = [{**group, "outcomes": [{**row, **row["bounded_ablations"], **oracles(row, common, threshold)}
                                      for row in group["outcomes"]]} for group in groups
                if "bounded_ablations" in group["outcomes"][0]]
    measures = (SELECTIVE, "learned_broadcast", "single_information_gain", "single_latest_witness",
                "reversed_witness_time", "withheld_association", "particle_matched_learned_factors",
                "isolated_event_oracle", "isolated_terminal_oracle")
    rows = [row for group in material for row in group["outcomes"]]
    return {"source_groups": sorted(group["group"] for group in material), "team_size": 3,
            "metrics": {method: team_metrics(material, method, inspect_cost=common["inspect_cost"],
                                              wrong_cost=common["wrong_cost"]) for method in measures},
            "same_learner_decision_threshold": True, "single_witness_reply_limit": 1, "common_cap_bytes": PRIMARY_CAP,
            "max_matched_particle_joint_tv": max(row["matched_particle_joint_tv"] for row in rows),
            "max_delivery_control_joint_tv": max(value for row in rows
                                                  for value in row["delivery_controls_max_joint_tv"].values()),
            "same_final_receiver_observations": matched_final_views(material)}
