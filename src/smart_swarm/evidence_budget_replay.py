"""Replay compact witness traffic over a common frozen candidate pool."""

from .contract import json_bytes
from .evidence_comparison import learned_likelihood
from .evidence_queries import receive_reply, request_wire
from .particle_memory import exact_filter


def wire_size(items):
    return sum(len(json_bytes(item)) for item in items)


def ordered_candidates(ads, gains):
    return sorted(ads, key=lambda ad: (-gains[ad["factor"]], -int(ad["capture_end_ns"]), ad["sender"]))


def fuse(replies, references, prior):
    factors = []
    for reply in replies:
        factor = reply["factor"]
        probabilities = {name: float(score) for name, score in reply["probabilities"].items()}
        factors.append((factor, learned_likelihood(probabilities, references[factor])))
    return exact_filter(prior, factors)["belief"]


def fits(candidates, issued, deadline, spent, cap):
    request_bytes = len(json_bytes(request_wire(candidates, issued)))
    estimated = spent + request_bytes + sum(ad["reply_bytes"] for ad in candidates)
    return estimated <= cap and all(issued + ad["delay_ns"] <= deadline for ad in candidates)


def parallel_selection(ads, query, deadline, cap):
    selected = []
    spent = wire_size(ads)
    for ad in ads:
        trial = [*selected, ad]
        if fits(trial, query, deadline, spent, cap):
            selected = trial
    return selected


def sequential_exchange(ads, query, deadline, cap, posterior, references, prior, threshold):
    requests, replies = [], []
    issued = query
    metadata_bytes = wire_size(ads)
    for ad in ads:
        cost = metadata_bytes + wire_size(requests) + wire_size(replies)
        if max(fuse(replies, references, prior).values()) < threshold and fits([ad], issued, deadline, cost, cap):
            requests.append(request_wire([ad], issued))
            reply = receive_reply(ad, issued, posterior)
            replies.append(reply)
            issued = int(reply["delivered_ns"])
    return requests, replies


def exchange(window, cap, strategy, gains, posterior, references, prior, threshold):
    query, deadline, ads = window
    if wire_size(ads) > cap:
        raise ValueError("Common advertisements already exceed the byte cap")
    ranked = ordered_candidates(ads, gains)
    if strategy == "parallel":
        selected = parallel_selection(ranked, query, deadline, cap)
        requests = [request_wire(selected, query)] if selected else []
        replies = [receive_reply(ad, query, posterior) for ad in selected]
    elif strategy == "sequential":
        requests, replies = sequential_exchange(ranked, query, deadline, cap, posterior, references, prior, threshold)
    else:
        raise ValueError(f"Unknown strategy {strategy}")
    spent = wire_size(ads) + wire_size(requests) + wire_size(replies)
    latency = max((int(reply["delivered_ns"]) - query for reply in replies), default=0)
    if spent > cap or latency > deadline - query:
        raise ValueError("Serialized traffic exceeded the declared communication limits")
    return {"belief": fuse(replies, references, prior), "serialized_bytes": spent,
            "responses": len(replies), "modeled_latency_ms": latency / 1e6,
            "wire": {"advertisements": ads, "requests": requests, "replies": replies}}
