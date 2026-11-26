"""Fit a small, sender-independent observation model for acquisition planning."""

from collections import Counter

import numpy as np

from .evidence_conditioned import channel
from .particle_memory import FACTORS

UPDATES = 200
LEARNING_RATE = .25
REGULARIZATION = .04


def features(index, count, visible):
    vector = np.zeros(2 * count + 2, dtype=np.float64)
    vector[0] = 1.
    vector[1 + index] = 1.
    vector[1 + count] = float(visible)
    vector[2 + count + index] = float(visible)
    return vector


def probabilities(x, weights):
    scores = x @ weights
    shifted = np.exp(scores - scores.max(axis=1, keepdims=True))
    return shifted / shifted.sum(axis=1, keepdims=True)


def training_arrays(examples, labels):
    n = len(labels)
    keys = [(row["source_group"], row["event_label"], row["advertisement"]["visual_support"]) for row in examples]
    repetitions = Counter(keys)
    x = np.stack([features(labels.index(label), n, visible) for _, label, visible in keys])
    targets = np.eye(n, dtype=np.float64)[[labels.index(row["reported_class"]) for row in examples]]
    weights = np.asarray([1. / repetitions[key] for key in keys], dtype=np.float64)
    return x, targets, weights / weights.sum()


def fit_factor(rows, labels):
    x, targets, importance = training_arrays(rows, labels)
    coefficients = np.zeros((x.shape[1], targets.shape[1]), dtype=np.float64)

    def nll():
        return float(-sum(importance * np.log((probabilities(x, coefficients) * targets).sum(axis=1))))

    initial = nll()
    for _ in range(UPDATES):
        residual = (probabilities(x, coefficients) - targets) * importance[:, None]
        gradient = x.T @ residual + REGULARIZATION * coefficients
        coefficients -= LEARNING_RATE * gradient
    return coefficients, {"updates": UPDATES, "parameters": coefficients.size, "initial_fit_nll": initial,
                          "final_fit_nll": nll(), "coefficients": coefficients.tolist(),
                          "independent_source_groups": len({row["source_group"] for row in rows})}


def fit_learned(labeled, empirical):
    table, details = {}, {}
    for factor, labels in FACTORS.items():
        observed = [row for row in labeled if row["advertisement"]["factor"] == factor and row["event_label"] is not None]
        coefficients, details[factor] = fit_factor(observed, labels)
        for visible in (False, True):
            key = channel({"factor": factor, "visual_support": visible})
            vectors = np.stack([features(index, len(labels), visible) for index in range(len(labels))])
            learned = {truth: dict(zip(labels, row.tolist(), strict=True))
                       for truth, row in zip(labels, probabilities(vectors, coefficients), strict=True)}
            support = empirical[key]["independent_group_support"]
            for truth, count in support.items():
                if count == 0:
                    learned[truth] = empirical[key]["probabilities"][truth]
            table[key] = {"probabilities": learned, "independent_group_support": support}
    return table, {"learning_rate": LEARNING_RATE, "l2": REGULARIZATION, "factors": details}
