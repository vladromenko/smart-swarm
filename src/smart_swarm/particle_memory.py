"""Finite joint particle baseline over settled content and relocated identity."""

import numpy as np

from .memory import CUBE_SUPPORT, PERMUTATIONS, SLOTS, STATES, TRACKS

LOCATIONS = (*SLOTS, "surface", "held", "outside")
FACTORS = {"containment": CUBE_SUPPORT, "association": PERMUTATIONS}


def location(state):
    cube, assignment = state
    return ("left" if assignment.startswith(cube) else "right") if cube in TRACKS else cube


def normalize(weights):
    total = weights.sum()
    if np.any(weights < 0) or not np.isfinite(total) or total <= 0:
        raise ValueError("Invalid joint particle weights")
    return weights / total


def prior_weights(prior):
    return normalize(np.asarray([prior[cube] / len(PERMUTATIONS) for cube, _ in STATES], dtype=np.float64))


def likelihood_weights(kind, likelihood):
    expected = FACTORS.get(kind)
    if expected is None or set(likelihood) != set(expected):
        raise ValueError("Likelihood does not describe the complete evidence vocabulary")
    vector = np.asarray([likelihood[cube if kind == "containment" else assignment]
                         for cube, assignment in STATES], dtype=np.float64)
    if not np.isfinite(vector).all() or np.any(vector <= 0):
        raise ValueError("Likelihoods must be finite and positive on all supported states")
    return vector


def result(weights, *, ess=None, resamples=0):
    terminal = dict.fromkeys(LOCATIONS, 0.)
    for state, weight in zip(STATES, weights, strict=True):
        terminal[location(state)] += float(weight)
    return {"belief": terminal, "joint": weights.tolist(), "effective_particles": ess, "resamples": resamples}


def exact_filter(prior, factors):
    weights = prior_weights(prior)
    for kind, likelihood in factors:
        weights = normalize(weights * likelihood_weights(kind, likelihood))
    return result(weights)


def systematic(weights, size, rng):
    cumulative = weights.cumsum()
    cumulative[-1] = 1.
    positions = (np.arange(size) + rng.random()) / size
    return np.searchsorted(cumulative, positions, side="right")


def particle_filter(prior, factors, count, *, seed=17):
    support = len(STATES)
    if count <= support:
        raise ValueError("Particle count must exceed the finite joint support")
    rng = np.random.default_rng(seed)
    initial = prior_weights(prior)
    indices = np.r_[np.arange(support), systematic(initial, count - support, rng)]
    multiplicity = np.bincount(indices, minlength=support)
    weights = initial[indices] / multiplicity[indices]
    resamples = 0
    for order, (kind, likelihood) in enumerate(factors):
        weights = normalize(weights * likelihood_weights(kind, likelihood)[indices])
        if order + 1 < len(factors) and effective_count(weights) < count / 2:
            indices = indices[systematic(weights, count, rng)]
            weights = np.full(count, 1. / count)
            resamples += 1
    joint = np.bincount(indices, weights=weights, minlength=support)
    return result(normalize(joint), ess=effective_count(weights), resamples=resamples)


def effective_count(weights):
    return float(1. / np.square(weights).sum())


def total_variation(first, second):
    return float(np.abs(np.asarray(first) - np.asarray(second)).sum() / 2)
