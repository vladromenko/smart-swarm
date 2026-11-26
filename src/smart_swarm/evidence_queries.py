"""Replay budgeted requests over the same learned witness opportunities."""

from collections import defaultdict
import math
from pathlib import Path

from .contract import json_bytes
from .episode import Replay
from .evidence_comparison import OUTPUTS, decision_cost, grouped_mean, learned_likelihood, learned_prior_map, observation_model
from .evidence_inputs import clip_packets
from .memory import QUERY, witness_delay_ns
from .particle_memory import FACTORS, exact_filter

RECEIVER = "rosmaster-1"
WITNESSES = {"so101-1": ("containment", witness_delay_ns("so101-1")), "tank-1": ("association", witness_delay_ns("tank-1"))}
POLICIES = {"broadcast": (2, 4096), "latest_witness": (1, 2048), "expected_information_gain": (1, 2048)}


def time_field(ns):
    return f"{ns:019d}"


def reply_wire(ad, sent_ns, probabilities):
    return {"sender": ad["sender"], "recipient": RECEIVER, "factor": ad["factor"], "frame_ids": ad["frame_ids"],
            "capture_end_ns": ad["capture_end_ns"], "sent_ns": time_field(sent_ns),
            "delivered_ns": time_field(sent_ns + ad["delay_ns"]), "probabilities": probabilities}


def advertise(client, factor, delay, query):
    packets = clip_packets(client)
    if not packets:
        return None
    references = [(packet["sequence"], camera) for packet in packets for camera in packet["cameras"]]
    frames = [f"{camera['sensor_id']}:{sequence:04d}" for sequence, camera in references]
    latest = max(int(camera["capture_interval"]["end_ns"]) for _, camera in references)
    if len(set(frames)) != len(frames) or latest > query:
        raise ValueError("Invalid query advertisement frame provenance")
    result = {"sender": client.agent, "recipient": RECEIVER, "factor": factor, "frame_ids": frames,
              "capture_end_ns": time_field(latest), "delay_ns": delay}
    dummy = dict.fromkeys(FACTORS[factor], "0.00000000")
    result["reply_bytes"] = len(json_bytes(reply_wire(result, query, dummy)))
    return result


def opportunities(root):
    receiver = Replay(root, RECEIVER)
    advertised = [advertise(Replay(root, agent), factor, delay, receiver.query)
                  for agent, (factor, delay) in WITNESSES.items()]
    return receiver.query, receiver.decision, [item for item in advertised if item is not None]


def entropy(probabilities):
    return -sum(value * math.log(value) for value in probabilities.values() if value > 0)


def expected_gain(factor, model, prior):
    classes = FACTORS[factor]
    mass = prior if factor == "containment" else dict.fromkeys(classes, 1. / len(classes))
    initial = entropy(exact_filter(prior, [])["belief"])
    expected = 0.
    for observed in classes:
        likelihood = {state: model[factor][state][observed] for state in classes}
        observation_mass = sum(mass[state] * likelihood[state] for state in classes)
        expected += observation_mass * entropy(exact_filter(prior, [(factor, likelihood)])["belief"])
    return max(0., initial - expected)


def request_wire(selected, query):
    return {"sender": RECEIVER, "recipients": [item["sender"] for item in selected], "text": QUERY,
            "issued_ns": time_field(query)}


def allocate(method, ads, query, deadline, gains):
    reply_cap, byte_cap = POLICIES[method]
    candidates = [item for item in ads if query + item["delay_ns"] <= deadline]
    if method == "latest_witness":
        candidates.sort(key=lambda item: (-int(item["capture_end_ns"]), item["sender"]))
    if method == "expected_information_gain":
        candidates.sort(key=lambda item: (-gains[item["factor"]], -int(item["capture_end_ns"]), item["sender"]))
    ad_bytes = sum(len(json_bytes(item)) for item in ads)
    if ad_bytes > byte_cap:
        raise ValueError("Common candidate metadata exceeds the configured communication budget")
    selected = []
    for candidate in candidates:
        trial = [*selected, candidate]
        total = ad_bytes + len(json_bytes(request_wire(trial, query))) + sum(item["reply_bytes"] for item in trial)
        if len(trial) <= reply_cap and total <= byte_cap:
            selected = trial
    request = request_wire(selected, query) if selected else None
    return selected, request, ad_bytes


def receive_reply(ad, query, posterior):
    factor = ad["factor"]
    probabilities = posterior(factor)
    if probabilities is None or set(probabilities) != set(FACTORS[factor]):
        raise ValueError("Solicited learner output does not match the advertised clip")
    if any(not math.isfinite(q) or not 0 <= q <= 1 for q in probabilities.values()):
        raise ValueError("Invalid probability in a solicited learner report")
    reply = reply_wire(ad, query, {name: format(value, ".8f") for name, value in probabilities.items()})
    if len(json_bytes(reply)) != ad["reply_bytes"]:
        raise ValueError("Serialized reply does not match its public budget estimate")
    return reply


def exchange(method, window, gains, posterior, priors, prior):
    query, deadline, ads = window
    selected, request, ad_bytes = allocate(method, ads, query, deadline, gains)
    received = [receive_reply(ad, query, posterior) for ad in selected]
    factors = [(item["factor"], learned_likelihood({name: float(q) for name, q in item["probabilities"].items()},
                                                  priors[item["factor"]])) for item in received]
    filtered = exact_filter(prior, factors)
    request_bytes = len(json_bytes(request)) if request is not None else 0
    reply_bytes = sum(len(json_bytes(item)) for item in received)
    latency = max((int(item["delivered_ns"]) - query for item in received), default=0)
    return {"belief": filtered["belief"], "wire": {"advertisements": ads, "request": request, "replies": received},
            "metadata_bytes": ad_bytes, "request_bytes": request_bytes, "reply_bytes": reply_bytes,
            "serialized_bytes": ad_bytes + request_bytes + reply_bytes, "responses": len(received),
            "modeled_latency_ms": latency / 1e6}


def validation_reports(rows):
    return {row["world_episode_id"]: {factor: max(row[key], key=row[key].get) for factor, key in OUTPUTS.items()
            if row[key] is not None} for row in rows}


def validation_query_gains(validation, prior):
    noise = observation_model(validation, validation_reports(validation))
    return {factor: expected_gain(factor, noise, prior) for factor in FACTORS}


def paired_query_traces(outcomes):
    groups = defaultdict(list)
    for row in outcomes:
        if row["family"] == "F6":
            groups[row["scene_group_id"]].append(row)
    for pair in groups.values():
        if len(pair) != 2 or any(pair[0][field] != pair[1][field] for field in ("belief", "decision", "wire", "serialized_bytes")):
            raise ValueError("Matched F6 inputs changed a query trace or prediction")
    return len(groups)


def query_outcome(row, method, window, gains, reference, prior):
    result = exchange(method, window, gains, lambda kind: row[OUTPUTS[kind]], reference, prior)
    visible = dict.fromkeys(OUTPUTS.values())
    for reply in result["wire"]["replies"]:
        visible[OUTPUTS[reply["factor"]]] = {name: float(q) for name, q in reply["probabilities"].items()}
    return {**row, **visible, **result}


def query_report(validation, test, priors, prior, threshold, decide, metrics):
    gains = validation_query_gains(validation, prior)
    windows = {row["world_episode_id"]: opportunities(Path(row["path"])) for row in test}
    reference = learned_prior_map(priors)
    reports = {}
    for method, (reply_cap, byte_cap) in POLICIES.items():
        rows = [query_outcome(row, method, windows[row["world_episode_id"]], gains, reference, prior) for row in test]
        scored = decide(rows, threshold)
        outcome_metrics = metrics(scored)
        outcome_metrics.pop("event_labels")
        report = {"reply_cap": reply_cap, "byte_cap": byte_cap, "max_serialized_bytes": max(row["serialized_bytes"] for row in scored),
                  "coupled_test_pairs": paired_query_traces(scored), "test": {**outcome_metrics, "decision_cost": decision_cost(scored)}}
        for field in ("serialized_bytes", "metadata_bytes", "request_bytes", "reply_bytes", "responses", "modeled_latency_ms"):
            report["mean_" + field] = grouped_mean(scored, lambda row: row[field])
        reports[method] = {**report, "outcomes": scored}
    return {"validation_expected_information_nats": gains, "selection_scope": "Pre-query clip metadata and validation reliability only",
            "transport_scope": "Common modeled query horizon, serialized JSON envelopes and fixed witness delays",
            "candidate_pool": "Same eight-frame clips and calibrated learned heads for all replay policies", "policies": reports}
