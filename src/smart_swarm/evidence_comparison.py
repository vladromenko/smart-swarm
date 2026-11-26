"""Observation-matched particle comparators and validation-only decision calibration."""

from collections import Counter, defaultdict
import math
from pathlib import Path
import random

from .episode import Replay
from .particle_memory import FACTORS, exact_filter, particle_filter, total_variation

ERROR_COST = 1.
INSPECTION_COST = .25
THRESHOLDS = (.60, .65, .70, .75, .80, .85, .90, .95, 1.01)
COUNTS = (128, 512, 2048)
OBSERVATION_POWERS = (.5, .75, 1., 1.3, 1.7)
MEAN_TV_TARGET = .01
LABELS = {"containment": "settled_placement", "association": "relocation_assignment"}
OUTPUTS = {"containment": "visual_placement", "association": "visual_relocation"}


def grouped_mean(rows, value):
    groups = Counter(row["scene_group_id"] for row in rows)
    if not groups:
        raise ValueError("An independent group is required")
    return sum(value(row) / groups[row["scene_group_id"]] for row in rows) / len(groups)


def decision_cost(rows):
    return grouped_mean(rows, lambda row: INSPECTION_COST if row["decision"] == "request_inspection"
                        else ERROR_COST * (row["decision"] != row["terminal"]))


def choose_threshold(rows, decide, metrics):
    outcomes = {}
    for threshold in THRESHOLDS:
        predicted = decide(rows, threshold)
        outcomes[threshold] = {**metrics(predicted), "decision_cost": decision_cost(predicted)}
    selected = min(THRESHOLDS, key=lambda value: (outcomes[value]["decision_cost"], outcomes[value]["confident_wrong"],
                   -outcomes[value]["f6_inspections"], -outcomes[value]["resolvable_correct"], abs(value - .75)))
    return selected, outcomes[selected], outcomes


def rule_observations(rows):
    result = {}
    for row in rows:
        evidence = Replay(Path(row["path"]), "rosmaster-1").evidence()
        report = {}
        for item in evidence:
            kind = item["factor"]
            if kind not in FACTORS:
                raise ValueError("Unknown local rule factor")
            report[kind] = max(item["hypotheses"], key=item["hypotheses"].get)
        result[row["world_episode_id"]] = report
    return result


def observation_model(rows, observed):
    group_sizes = Counter(row["scene_group_id"] for row in rows)
    result = {kind: {value: dict.fromkeys(classes, .5) for value in classes} for kind, classes in FACTORS.items()}
    for row in rows:
        observation = observed[row["world_episode_id"]]
        for kind, detected in observation.items():
            actual = row[LABELS[kind]]
            if actual is not None:
                result[kind][actual][detected] += 1. / group_sizes[row["scene_group_id"]]
    return {kind: {value: {reported: weight / sum(counts.values()) for reported, weight in counts.items()}
                   for value, counts in variables.items()} for kind, variables in result.items()}


def rule_factors(rows, observed, model, power):
    return {row["world_episode_id"]: [(kind, {value: probs[reported]**power for value, probs in model[kind].items()})
            for kind, reported in observed[row["world_episode_id"]].items()] for row in rows}


def learned_factors(rows, priors):
    result = {}
    reference = learned_prior_map(priors)
    for row in rows:
        evidence = []
        for kind, key in OUTPUTS.items():
            probabilities = row.get(key)
            if probabilities is not None:
                evidence.append((kind, learned_likelihood(probabilities, reference[kind])))
        result[row["world_episode_id"]] = evidence
    return result


def learned_prior_map(priors):
    return {kind: dict(zip(classes, values, strict=True))
            for (kind, classes), values in zip(FACTORS.items(), priors, strict=True)}


def learned_likelihood(probabilities, prior):
    return {name: max(q, 1e-20) / prior[name] for name, q in probabilities.items()}


def run_filter(rows, factors, prior, count=None):
    outputs = []
    for row in rows:
        evidence = factors[row["world_episode_id"]]
        filtered = exact_filter(prior, evidence) if count is None else particle_filter(prior, evidence, count)
        outputs.append({**row, **filtered})
    return outputs


def nll(rows):
    return grouped_mean(rows, lambda row: -math.log(max(row["belief"][row["terminal"]], 1e-12)))


def fit_rule_pf(train, validation, prior):
    observed = rule_observations(train + validation)
    model = observation_model(train, observed)
    candidates = {power: rule_factors(validation, observed, model, power) for power in OBSERVATION_POWERS}
    losses = {power: nll(run_filter(validation, factors, prior)) for power, factors in candidates.items()}
    power = min(OBSERVATION_POWERS, key=losses.get)
    return model, power, losses, candidates[power]


def tune_particles(validation, factors, prior):
    exact = run_filter(validation, factors, prior)
    reference = {row["world_episode_id"]: row["joint"] for row in exact}
    errors = {}
    for count in COUNTS:
        sampled = run_filter(validation, factors, prior, count)
        errors[count] = grouped_mean(sampled, lambda row: total_variation(row["joint"], reference[row["world_episode_id"]]))
    candidates = [count for count, error in errors.items() if error <= MEAN_TV_TARGET]
    chosen = min(candidates) if candidates else max(COUNTS)
    return chosen, {"validation_mean_joint_total_variation": errors, "target": MEAN_TV_TARGET, "target_reached": bool(candidates)}


def maximum_terminal_delta(rows):
    return max(abs(row["belief"][key] - row["exact_belief"][key]) for row in rows for key in row["belief"])


def coupled_ambiguity(rows):
    groups = defaultdict(list)
    for row in rows:
        if row["family"] == "F6":
            groups[row["scene_group_id"]].append(row)
    for group in groups.values():
        if len(group) != 2 or any(group[0][field] != group[1][field] for field in ("joint", "belief", "decision")):
            raise ValueError("Joint particle filter violated coupled F6 inference")
    return len(groups)


def pf_report(validation, test, valid_factors, test_factors, prior, decide, metrics, *, threshold=None):
    count, audit = tune_particles(validation, valid_factors, prior)
    valid = run_filter(validation, valid_factors, prior, count)
    candidates = None
    if threshold is None:
        threshold, _, candidates = choose_threshold(valid, decide, metrics)
    exact = run_filter(test, test_factors, prior)
    approx = run_filter(test, test_factors, prior, count)
    paired = [{**row, "exact_belief": reference["belief"], "joint_tv": total_variation(row["joint"], reference["joint"])}
              for row, reference in zip(approx, exact, strict=True)]
    scored = decide(paired, threshold)
    return {"particles": count, "seed": 17, "threshold": threshold, "validation": audit,
            "validation_threshold_candidates": candidates, "test": {**metrics(scored), "decision_cost": decision_cost(scored)},
            "test_exact_reference": metrics(decide(exact, threshold)), "test_max_terminal_delta": maximum_terminal_delta(paired),
            "test_mean_joint_total_variation": grouped_mean(paired, lambda row: row["joint_tv"]),
            "coupled_test_pairs": coupled_ambiguity(scored), "test_outcomes": scored}


def calibrated_rule_pf(sets, prior, decide, metrics):
    model, power, losses, validation = fit_rule_pf(sets["train"], sets["validation"], prior)
    test = sets["test"]
    test_factors = rule_factors(test, rule_observations(test), model, power)
    report = pf_report(sets["validation"], test, validation, test_factors, prior, decide, metrics)
    for selected in (report["test"], report["test_exact_reference"], *report["validation_threshold_candidates"].values()):
        selected.pop("event_labels")
    return {**report, "report_likelihoods": model, "observation_power": power, "validation_power_nll": losses,
            "scope": "Recorded private RGB rule messages only, confusion likelihoods fitted on labelled training groups"}


def same_visual_pf(validation, test, priors, prior, threshold, decide, metrics):
    valid = learned_factors(validation, priors)
    heldout = learned_factors(test, priors)
    exact = run_filter(test, heldout, prior)
    mismatch = max(abs(row["belief"][key] - other["belief"][key]) for row, other in zip(test, exact, strict=True)
                   for key in row["belief"])
    if mismatch > 1e-5:
        raise ValueError("Learned likelihood conversion does not reproduce the causal exact filter")
    report = pf_report(validation, test, valid, heldout, prior, decide, metrics, threshold=threshold)
    return {**report, "exact_learned_terminal_delta": mismatch,
            "scope": "Same calibrated head probabilities, priors, delivered factors and inspection threshold as exact memory"}


def availability(row):
    return (row.get("visual_placement") is not None, row.get("visual_relocation") is not None)


def shuffled_association(rows, priors, prior, threshold, decide, metrics):
    groups = defaultdict(list)
    for row in rows:
        groups[row["scene_group_id"]].append(row)
    patterns = defaultdict(list)
    for group in groups.values():
        group.sort(key=lambda row: (availability(row), row["world_episode_id"]))
        patterns[tuple(availability(row) for row in group)].append(group)
    substitute = {}
    moved = set()
    generator = random.Random(59)
    for cohort in patterns.values():
        generator.shuffle(cohort)
        for original, donor in zip(cohort, [*cohort[1:], cohort[0]], strict=True):
            for row, other in zip(original, donor, strict=True):
                substitute[row["world_episode_id"]] = other["visual_relocation"]
                if row["scene_group_id"] != other["scene_group_id"] and other["visual_relocation"] is not None:
                    moved.add(row["scene_group_id"])
    shuffled = [{**row, "visual_relocation": substitute[row["world_episode_id"]]} for row in rows]
    scored = decide(run_filter(shuffled, learned_factors(shuffled, priors), prior), threshold)
    return {"exchanged_source_groups": len(moved), "coupled_pairs": coupled_ambiguity(scored),
            "test": {**metrics(scored), "decision_cost": decision_cost(scored)},
            "scope": "Diagnostic substitution across independent groups with matching report availability"}
