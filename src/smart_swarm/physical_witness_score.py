"""Use sealed witness replies for evaluation after physical event review."""

from collections import Counter
from hashlib import sha256
from math import isclose, log

from .contract import json_bytes, read_ns
from .evidence_comparison import grouped_mean
from .evidence_queries import RECEIVER, WITNESSES
from .memory import TRACKS
from .particle_memory import FACTORS
from .physical_learned import HEADS
from .physical_slot_reference import REVERSED, SAME, UNKNOWN, mapped_assignment

EVENTS = {"containment": "settled_placement", "association": "relocation_assignment"}
SENDERS = {factor: sender for sender, (factor, _) in WITNESSES.items()}
MODES = ("online", "blind_offline")
USABLE = "labeled_visual_reply"
INSPECT = "request_inspection"


def packets_by_factor(packets):
    indexed = {}
    for row in packets:
        factor = row["factor"]
        if (factor not in SENDERS or factor in indexed or SENDERS[factor] != row["sender"]
                or row["recipient"] != RECEIVER):
            raise ValueError("A physical witness packet has an unexpected factor, sender or receiver")
        indexed[factor] = row
    return indexed


def wire_index(wire, query):
    ads = packets_by_factor(wire["advertisements"])
    replies = packets_by_factor(wire["replies"])
    if any(read_ns(row["capture_end_ns"]) > query or type(row["frame_count"]) is not int or row["frame_count"] <= 0
           or type(row["delay_ns"]) is not int or row["delay_ns"] < 0 for row in ads.values()):
        raise ValueError("A physical witness advertisement has invalid frames or timing")
    requests = wire["requests"]
    if len(requests) > 1 or any(row["sender"] != RECEIVER or read_ns(row["issued_ns"]) != query for row in requests):
        raise ValueError("The physical witness request disagrees with the sealed query")
    recipients = requests[0]["recipients"] if requests else []
    if len(set(recipients)) != len(recipients) or not set(recipients) <= {row["sender"] for row in ads.values()}:
        raise ValueError("The physical witness request names an unsupplied or repeated sender")
    if any(row["sender"] not in recipients for row in replies.values()):
        raise ValueError("An unsolicited physical witness reply entered the receipt")
    return ads, set(recipients), replies


def normalized_probabilities(reply, factor, mapping):
    raw = {name: float(value) for name, value in reply["probabilities"].items()}
    total = sum(raw.values())
    if (set(raw) != set(FACTORS[factor]) or any(not 0 <= value <= 1 for value in raw.values())
            or not isclose(total, 1., abs_tol=1e-6, rel_tol=0.)):
        raise ValueError("A physical witness reply has an invalid probability distribution")
    probabilities = {name: value / total for name, value in raw.items()}
    return mapped_assignment(probabilities, mapping) if factor == "association" else probabilities


def received_factor(base, receipt, ad, reply, query):
    factor = base["factor"]
    capture, sent, delivered = (read_ns(reply[key]) for key in ("capture_end_ns", "sent_ns", "delivered_ns"))
    if (capture != read_ns(ad["capture_end_ns"]) or len(set(reply["frame_ids"])) != ad["frame_count"]
            or len(reply["frame_ids"]) != ad["frame_count"] or not capture <= sent == query <= delivered
            or delivered - sent != ad["delay_ns"]):
        raise ValueError("A saved witness reply disagrees with its advertised frames or modeled times")
    lag = delivered - query
    result = {**base, "reply_delay_ns": lag, "frame_count": ad["frame_count"]}
    if lag > receipt["modeled_reply_window_ns"]:
        return {**result, "status": "after_deadline"}
    if factor == "association":
        support = receipt.get("marker_support_by_witness", {}).get(SENDERS[factor], {}).get("supported")
        if support is not True:
            return {**result, "status": "prior_only" if support is False else "visual_support_unverified"}
    distribution = normalized_probabilities(reply, factor, receipt.get("slot_mapping"))
    status = USABLE if base["label"] is not None else "unknown_label"
    return {**result, "status": status, "probabilities": distribution}


def factor_row(factor, label, receipt, index, query):
    if label is not None and label not in FACTORS[factor]:
        raise ValueError("A reviewed event label is outside its original class vocabulary")
    base = {"factor": factor, "sender": SENDERS[factor], "label": label}
    ads, requested, replies = index
    if factor == "association" and receipt.get("slot_mapping") == UNKNOWN:
        if factor in ads or factor in replies:
            raise ValueError("An unverified side mapping cannot carry a physical association reply")
        return {**base, "status": "slot_reference_unverified"}
    if factor not in ads:
        return {**base, "status": "no_advertisement"}
    if base["sender"] not in requested:
        return {**base, "status": "not_requested"}
    if factor not in replies:
        return {**base, "status": "no_reply"}
    return received_factor(base, receipt, ads[factor], replies[factor], query)


def diagnose(receipt, labels, query):
    """Project a pre-reveal receipt onto evaluator-reviewed factor labels."""
    binding = {"receipt_sha256": sha256(json_bytes(receipt)).hexdigest()}
    window = receipt.get("modeled_reply_window_ns")
    if window is None or any("wire" not in receipt["heads"][name] for name in HEADS):
        return {**binding, "status": "wire_unavailable"}
    if type(window) is not int or window <= 0 or receipt.get("slot_mapping") not in (None, SAME, REVERSED, UNKNOWN):
        raise ValueError("The physical witness receipt names an invalid deadline or side mapping")
    rows = {}
    for head in HEADS:
        index = wire_index(receipt["heads"][head]["wire"], query)
        rows[head] = {factor: factor_row(factor, labels.get(event), receipt, index, query)
                      for factor, event in EVENTS.items()}
    return {**binding, "status": "available", "modeled_reply_window_ns": window,
            "slot_mapping": receipt.get("slot_mapping"), "heads": rows}


def top_class(row):
    probabilities = row["probabilities"]
    leader = max(probabilities, key=probabilities.get)
    tied = sum(value == probabilities[leader] for value in probabilities.values()) > 1
    return None if tied else leader


def matches(row):
    return top_class(row) == row["label"]


def factor_metrics(cases, head, factor):
    rows = [{**case["witness_diagnostics"]["heads"][head][factor], "scene_group_id": case["scene_group_id"]}
            for case in cases]
    measured = [row for row in rows if row["status"] == USABLE]
    return {"statuses": dict(Counter(row["status"] for row in rows)),
            "confirmed_event_labels": sum(row["label"] is not None for row in rows),
            "scored_visual_replies": len(measured), "scored_source_groups": len({row["scene_group_id"] for row in measured}),
            "correct_top_class": sum(matches(row) for row in measured), "tied_top_class": sum(top_class(row) is None for row in measured),
            "group_weighted_accuracy": grouped_mean(measured, matches) if measured else None,
            "group_weighted_nll": grouped_mean(measured, lambda row: -log(max(row["probabilities"][row["label"]], 1e-12)))
            if measured else None}


def terminal_pattern(case, head):
    decision = case["forecasts"][head]["decision"]
    expected = case["terminal"]
    if expected is None:
        return "terminal_unknown"
    if decision == expected or decision == INSPECT:
        return "correct_final_answer" if decision == expected else "inspection_requested"
    factors = case["witness_diagnostics"]["heads"][head]
    containment = factors["containment"]
    relevant = [containment, factors["association"]] if containment["label"] in TRACKS else [containment]
    if any(row["label"] is None for row in relevant):
        return "wrong_final_missing_event_labels"
    if any(row["status"] != USABLE for row in relevant):
        return "wrong_final_missing_visual_reply"
    if any(top_class(row) is None for row in relevant):
        return "wrong_final_with_ambiguous_witness"
    if not all(matches(row) for row in relevant):
        return "wrong_final_with_witness_error"
    return "wrong_final_despite_correct_witness_argmax"


def summarize(cases):
    rows = [case for case in cases if case.get("witness_diagnostics")]
    if not rows:
        return {}
    modes = {}
    for mode in MODES:
        measured = [case for case in rows if case["learned_forecast_mode"] == mode]
        available = [case for case in measured if case["witness_diagnostics"]["status"] == "available"]
        if measured:
            modes[mode] = {"receipt_counts": dict(Counter(row["witness_diagnostics"]["status"] for row in measured)),
                           "models": {head: {"factors": {factor: factor_metrics(available, head, factor) for factor in EVENTS},
                                             "terminal_patterns": dict(Counter(terminal_pattern(case, head) for case in available))}
                                      for head in HEADS}}
    return {"witness_evaluation": {"scope": "Evaluator-only scoring of saved, on-time visual witness replies",
                                   "reply_times_modeled": True, "attribution_is_descriptive": True, "by_mode": modes}}
