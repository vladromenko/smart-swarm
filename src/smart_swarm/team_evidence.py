"""Causal multi-witness exchanges for the fixed-total team study."""

from collections import defaultdict
import math

from .contract import json_bytes
from .episode import Replay, load
from .evidence_budget_replay import wire_size
from .evidence_comparison import learned_likelihood
from .evidence_queries import RECEIVER, WITNESSES, advertise, receive_reply, request_wire
from .particle_memory import exact_filter

ANCHOR_ARM, ANCHOR_TANK = WITNESSES
ANCHORS = frozenset(WITNESSES)
POLICIES = ("anchor", "broadcast", "factor_coverage", "recent_per_factor")
ONE_PER_FACTOR = ("factor_coverage", "recent_per_factor")


def opportunities(root):
    grants = load(root / "manifest.json")["policy_cameras"]
    receiver = Replay(root, RECEIVER)
    private_ads = []
    for agent in sorted(set(grants) - {RECEIVER}):
        factor, delay = WITNESSES.get(agent, WITNESSES[ANCHOR_TANK])
        ad = advertise(Replay(root, agent), factor, delay, receiver.query)
        if ad is not None:
            private_ads.append(ad)
    return receiver.query, receiver.decision, private_ads


def announcement(ad):
    return {key: value for key, value in ad.items() if key != "frame_ids"} | {"frame_count": len(ad["frame_ids"])}


def candidate_order(method, ads, gains):
    if method not in POLICIES:
        raise ValueError(f"Unknown team policy {method}")
    candidates = [ad for ad in ads if method != "anchor" or ad["sender"] in ANCHORS]
    return sorted(candidates, key=lambda ad: (-gains[ad["factor"]],
                                             method != "recent_per_factor" and ad["sender"] not in ANCHORS,
                                             -int(ad["capture_end_ns"]), ad["sender"]))


def select(method, ads, query, deadline, cap, gains, max_replies=None):
    remaining = wire_size([announcement(ad) for ad in ads])
    if remaining > cap:
        raise ValueError("The chosen envelope cannot fit the common metadata")
    selected, covered = [], set()
    for ad in candidate_order(method, ads, gains):
        trial = [*selected, ad]
        request = len(json_bytes(request_wire(trial, query)))
        cost = remaining + request + sum(item["reply_bytes"] for item in trial)
        novel = method not in ONE_PER_FACTOR or ad["factor"] not in covered
        within_limit = max_replies is None or len(trial) <= max_replies
        if novel and within_limit and cost <= cap and query + ad["delay_ns"] <= deadline:
            selected = trial
            covered.add(ad["factor"])
    return selected


def pooled_factors(replies, references):
    """Pool reports about the same event without counting each camera as another event."""
    by_event = defaultdict(list)
    used_frames = set()
    for reply in replies:
        frames = set(reply["frame_ids"])
        if frames and frames.isdisjoint(used_frames):
            factor = reply["factor"]
            probabilities = {name: float(score) for name, score in reply["probabilities"].items()}
            by_event[factor].append(learned_likelihood(probabilities, references[factor]))
            used_frames.update(frames)
    return pool_likelihoods(by_event)


def pool_likelihoods(by_event):
    return [(kind, {name: math.exp(sum(math.log(source[name]) for source in sources) / len(sources))
                    for name in sources[0]}) for kind, sources in by_event.items()]


def exchange(method, window, cap, gains, posterior, references, prior, *, max_replies=None):
    query, deadline, ads = window
    metadata = [announcement(ad) for ad in ads]
    selected = select(method, ads, query, deadline, cap, gains, max_replies)
    requests = [request_wire(selected, query)] if selected else []
    replies = [receive_reply(ad, query, lambda factor: posterior(ad["sender"], factor)) for ad in selected]
    spent = wire_size(metadata) + wire_size(requests) + wire_size(replies)
    latency = max((int(row["delivered_ns"]) for row in replies), default=query) - query
    if spent > cap or latency > deadline - query:
        raise ValueError("Actual team exchange exceeded its byte or latency budget")
    return {**exact_filter(prior, pooled_factors(replies, references)),
            "serialized_bytes": spent, "modeled_latency_ms": latency / 1e6,
            "wire": {"advertisements": metadata, "requests": requests, "replies": replies}}
