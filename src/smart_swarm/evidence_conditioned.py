"""Finite-horizon acquisition using calibrated observation channels and native packet budgets."""

from collections import defaultdict
from itertools import combinations

from .evidence_budget_replay import wire_size
from .evidence_queries import receive_reply, request_wire
from .particle_memory import FACTORS, likelihood_weights, normalize, prior_weights, result
from .team_evidence import announcement


def channel(ad):
    visibility = "visible" if ad["visual_support"] else "unsupported"
    return f"{ad['factor']}/{visibility}"


def fit_channels(labeled):
    grouped = defaultdict(list)
    for row in labeled:
        factor, observed = row["advertisement"]["factor"], row["reported_class"]
        truth = row["event_label"]
        if truth is not None:
            if truth not in FACTORS[factor] or observed not in FACTORS[factor]:
                raise ValueError("An observation falls outside the declared event vocabulary")
            key = channel(row["advertisement"]), truth, row["source_group"]
            grouped[key].append(observed)
    tables = {}
    for factor, labels in FACTORS.items():
        for visible in (False, True):
            key = channel({"factor": factor, "visual_support": visible})
            rows, support = {}, {}
            for truth in labels:
                cases = [reports for (name, state, _), reports in grouped.items() if name == key and state == truth]
                counts = {guess: .5 + sum(reports.count(guess) / len(reports) for reports in cases) for guess in labels}
                total = sum(counts.values())
                rows[truth] = {guess: value / total for guess, value in counts.items()}
                support[truth] = len(cases)
            tables[key] = {"probabilities": rows, "independent_group_support": support}
    return tables


def conditioned(joint, ad, tables, observed):
    factor = ad["factor"]
    probabilities = tables[channel(ad)]["probabilities"]
    vector = likelihood_weights(factor, {state: row[observed] for state, row in probabilities.items()})
    updated = joint * vector
    return float(updated.sum()), normalize(updated)


def terminal_risk(joint, inspection_cost):
    belief = result(joint)["belief"]
    return min(inspection_cost, 1. - max(belief.values()))


def expected_loss(joint, choices, tables, inspection_cost):
    if not choices:
        return terminal_risk(joint, inspection_cost)
    ad, *rest = choices
    possibilities = [conditioned(joint, ad, tables, answer) for answer in FACTORS[ad["factor"]]]
    return sum(weight * expected_loss(updated, rest, tables, inspection_cost) for weight, updated in possibilities)


def exchange_cost(choices, when):
    return wire_size([request_wire(choices, when)]) + sum(ad["reply_bytes"] for ad in choices)


def available(ads, covered, when, deadline, budget):
    return [ad for ad in ads if ad["factor"] not in covered and when + ad["delay_ns"] <= deadline
            and exchange_cost([ad], when) <= budget]


def one_more(joint, ads, when, tables, inspection_cost):
    options = [(terminal_risk(joint, inspection_cost), 0)]
    options.extend((expected_loss(joint, [ad], tables, inspection_cost), exchange_cost([ad], when)) for ad in ads)
    return min(options, key=lambda value: (round(value[0], 12), value[1]))


def lookahead(joint, ad, ads, clock, tables, inspection_cost):
    when, deadline, budget = clock
    first_cost = exchange_cost([ad], when)
    arrival = when + ad["delay_ns"]
    follow = available(ads, {ad["factor"]}, arrival, deadline, budget - first_cost)
    candidates = [conditioned(joint, ad, tables, answer) for answer in FACTORS[ad["factor"]]]
    outcomes = [(weight, one_more(updated, follow, arrival, tables, inspection_cost)) for weight, updated in candidates]
    return sum(weight * costs[0] for weight, costs in outcomes), first_cost + sum(weight * costs[1] for weight, costs in outcomes)


def next_queries(joint, ads, tables, clock, slots, inspection_cost):
    when, deadline, budget = clock
    eligible = available(ads, set(), when, deadline, budget)
    actions = [(terminal_risk(joint, inspection_cost), 0, [])]
    for ad in eligible:
        single = expected_loss(joint, [ad], tables, inspection_cost), exchange_cost([ad], when)
        costs = lookahead(joint, ad, eligible, clock, tables, inspection_cost) if slots == 2 else single
        actions.append((*costs, [ad]))
    if slots == 2:
        for pair in combinations(eligible, 2):
            cost = exchange_cost(list(pair), when)
            if pair[0]["factor"] != pair[1]["factor"] and cost <= budget:
                actions.append((expected_loss(joint, pair, tables, inspection_cost), cost, list(pair)))
    return min(actions, key=lambda row: (round(row[0], 12), row[1], [ad["sender"] for ad in row[2]]))[2]


def risk_exchange(window, cap, report, tables, prior, *, inspection_cost=.25, evidence_tables=None, fixed_choices=None):
    when, deadline, ads = window
    wire = {"advertisements": [announcement(ad) for ad in ads], "requests": [], "replies": []}
    spent = wire_size(wire["advertisements"])
    if spent > cap:
        raise ValueError("The common advertisements exceed this exchange budget")
    joint = prior_weights(prior)
    calibrated = tables if evidence_tables is None else evidence_tables
    covered, updates = set(), []
    pending = list(ads)
    for batch in range(2):
        available_ads = [ad for ad in pending if ad["factor"] not in covered]
        clock = when, deadline, cap - spent
        if fixed_choices is None:
            chosen = next_queries(joint, available_ads, tables, clock, 2 - len(covered), inspection_cost) if pending else []
        else:
            chosen = fixed_choices if batch == 0 else []
        if chosen:
            wire["requests"].append(request_wire(chosen, when))
            replies = [(receive_reply(ad, when, lambda factor: report(ad["sender"], factor)), ad) for ad in chosen]
            for reply, ad in sorted(replies, key=lambda pair: int(pair[0]["delivered_ns"])):
                probabilities = reply["probabilities"]
                observed = max(probabilities, key=lambda value: float(probabilities[value]))
                _, joint = conditioned(joint, ad, calibrated, observed)
                when = int(reply["delivered_ns"])
                covered.add(ad["factor"])
                wire["replies"].append(reply)
                updates.append({"sender": ad["sender"], "delivered_ns": when, "belief": result(joint)["belief"]})
            spent = sum(wire_size(values) for values in wire.values())
        pending = [ad for ad in available_ads if chosen and ad["factor"] not in covered]
    final = result(joint)
    winner = max(final["belief"], key=final["belief"].get)
    if spent > cap or when > deadline:
        raise ValueError("Actual adaptive exchange exceeded its byte or latency budget")
    return {**final, "decision": winner if 1. - final["belief"][winner] <= inspection_cost else "request_inspection",
            "serial_latency_ms": (when - window[0]) / 1e6, "serialized_bytes": spent, "updates": updates, "wire": wire}
