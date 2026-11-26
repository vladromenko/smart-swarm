"""Calibrate association likelihoods separately for each observed camera mount."""

import math

from .memory import PERMUTATIONS

SLOPES = (-1., -.5, -.25, -.1, 0., .1, .25, .5, 1.)
MIN_GROUPS_PER_CLASS = 2
REGULARIZATION = .05
FORWARD = PERMUTATIONS[0]
REVERSE = PERMUTATIONS[1]


def log_odds(posterior):
    a, b = [max(float(posterior[label]), 1e-8) for label in PERMUTATIONS]
    return math.log(a) - math.log(b)


def apply(posterior, reference, slope):
    offset = log_odds(reference)
    logit = offset + slope * (log_odds(posterior) - offset)
    p = 1. / (1. + math.exp(-logit))
    return {FORWARD: p, REVERSE: 1. - p}


def loss(rows, reference, slope):
    balanced = []
    for label in PERMUTATIONS:
        selected = [posterior for target, posterior in rows.values() if target == label]
        score = sum(-math.log(max(apply(p, reference, slope)[label], 1e-12)) for p in selected)
        balanced.append(score / len(selected))
    return sum(balanced) / len(balanced)


def fit(rows, reference):
    """Each entry is one independent source group for one mount."""
    counts = {label: sum(target == label for target, _ in rows.values()) for label in PERMUTATIONS}
    if min(counts.values()) < MIN_GROUPS_PER_CLASS:
        return {"support": counts, "slope": None, "status": "insufficient_support"}
    grid = {slope: loss(rows, reference, slope) for slope in SLOPES}
    selected = min(SLOPES, key=lambda slope: (grid[slope] + REGULARIZATION * slope**2, abs(slope), slope))
    return {"support": counts, "slope": selected, "status": "fitted", "balanced_nll": grid[selected],
            "balanced_nll_by_slope": {str(slope): score for slope, score in grid.items()}}


def calibrated_replies(replies, profiles, reference, anchor):
    adjusted = []
    for reply in replies:
        sender = reply["sender"]
        if reply["factor"] != "association":
            adjusted.append(reply)
        else:
            slope = profiles.get(sender, {}).get("slope")
            if slope is None and sender == anchor:
                adjusted.append(reply)
            elif slope is not None and slope != 0.:
                probabilities = apply(reply["probabilities"], reference, slope)
                adjusted.append({**reply, "probabilities": probabilities})
    return adjusted
