"""Bounded temporal evidence checks on the independent six-family pilot."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import random
import time

import torch

from learn_evidence import admitted_features
from smart_swarm.episode import Replay, atomic, load, sha
from smart_swarm.evidence_comparison import (
    calibrated_rule_pf, choose_threshold, decision_cost, grouped_mean, same_visual_pf, shuffled_association,
)
from smart_swarm.evidence_inputs import runtime_signature
from smart_swarm.evidence_queries import query_report, validation_query_gains
from smart_swarm.memory import CUBE_SUPPORT, PERMUTATIONS, predict, witness_delay_ns
from smart_swarm.temporal_evidence import (
    EvidenceHead, FrozenVision, LOCATIONS, joint_logbelief, loss_for, marginalized,
)


def digest(model):
    return {name: sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for name, value in model.named_parameters()}


def admission(portfolio, *, audit_holdout=True):
    records = portfolio["records"]
    owners, groups = {}, defaultdict(list)
    for row in records:
        group = row["scene_group_id"]
        if owners.setdefault(group, row["split"]) != row["split"]:
            raise ValueError("Scene lineage crosses research splits")
        groups[group].append(row)
    if Counter(owners.values()) != {"train": 300, "validation": 60, "test": 60}:
        raise ValueError("Full ablation requires 300, 60 and 60 independent groups")
    for items in groups.values():
        if items[0]["family"] == "F6":
            if audit_holdout or items[0]["split"] != "test":
                verify_pair(items)
            elif len(items) != 2:
                raise ValueError("The held-out F6 group must have two matched variants")
        elif len(items) != 1:
            raise ValueError("Unexpected duplicated independent history")
    result = {split: [row for row in records if row["split"] == split] for split in ("train", "validation", "test")}
    for split, expected in (("train", 50), ("validation", 10), ("test", 10)):
        items = result[split]
        families = {name: len({row["scene_group_id"] for row in items if row["family"] == name})
                    for name in ("F1", "F2", "F3", "F4", "F5", "F6")}
        if set(families.values()) != {expected}:
            raise ValueError(f"Incomplete family coverage in {split}")
        if audit_holdout or split != "test":
            support = {row["terminal"] for row in items if row["resolvable"]}
            if not {"left", "right", "surface"} <= support:
                raise ValueError(f"Incomplete resolvable outcome coverage in {split}")
    return {split: items for split, items in result.items() if audit_holdout or split != "test"}


def verify_pair(items):
    if len(items) != 2 or len({row["terminal"] for row in items}) != 2:
        raise ValueError("F6 requires exactly two matched variants with different hidden targets")
    a, b = [runtime_signature(Path(row["path"])) for row in items]
    if a != b:
        raise ValueError("F6 differs on a permitted pre-inspection input")


def fitting_rows(records):
    buckets = defaultdict(list)
    for row in records:
        if row["resolvable"]:
            buckets[(row["settled_placement"], row["relocation_assignment"])].append(row)
    if len(buckets) < 4 or any(len(items) < 2 for items in buckets.values()):
        raise ValueError("Clearly witnessed fitting data do not cover observed labels")
    return [row for key, items in sorted(buckets.items()) for row in items[:2]]


def priors_for(records):
    groups = Counter(row["scene_group_id"] for row in records)
    priors = []
    for label, vocabulary in (("settled_placement", CUBE_SUPPORT), ("relocation_assignment", PERMUTATIONS)):
        counts = dict.fromkeys(vocabulary, .5)
        for row in records:
            if row[label] is not None:
                counts[row[label]] += 1 / groups[row["scene_group_id"]]
        priors.append([counts[name] / sum(counts.values()) for name in vocabulary])
    return priors


def optimize(model, optimizer, rows, features, priors, cap, deadline, reveal, randomizer, state_prior, *, progress):
    model.train()
    by_group = defaultdict(list)
    for row in rows:
        by_group[row["scene_group_id"]].append(row)
    eligible = list(by_group.values())
    kind, draw_offset, updates = progress
    values = []
    while len(values) < cap and time.monotonic() < deadline:
        example = randomizer.choice(randomizer.choice(eligible))
        world = example["world_episode_id"]
        inputs = features[world]
        loss = loss_for(model(*inputs), example, priors, reveal_weight=reveal, state_prior=state_prior)
        masked = not reveal and example["settled_placement"] is None and example["relocation_assignment"] is None
        if not torch.isfinite(loss) or loss.requires_grad != (not masked):
            raise ValueError("Evidence objective disagrees with the declared masks or is non-finite")
        value = None
        if not masked:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            updates += 1
            value = float(loss.detach())
        values.append((world, value))
        print(json.dumps({"event": "masked_example" if masked else "optimizer_update", "model": kind,
                          "opportunity": draw_offset + len(values), "update": updates, "loss": value},
                         allow_nan=False), flush=True)
    return values


def optimizer_losses(trace):
    return [value for _, value in trace if value is not None]


@torch.no_grad()
def forecast(model, items, features, priors, temp, state_prior=None):
    model.eval()
    results = []
    for row in items:
        out = model(*features[row["world_episode_id"]])
        belief = marginalized(joint_logbelief(*out, priors, temp, state_prior=state_prior)).exp().tolist()
        forecasts = [dict(zip(classes, (logits / temp).softmax(-1).tolist(), strict=True)) if logits is not None else None
                     for logits, classes in zip(out, (CUBE_SUPPORT, PERMUTATIONS), strict=True)]
        results.append({**row, "visual_placement": forecasts[0], "visual_relocation": forecasts[1],
                        "belief": dict(zip(LOCATIONS, belief, strict=True))})
    return results


def decide(rows, threshold):
    predictions = []
    for row in rows:
        probabilities = row["belief"]
        best = max(probabilities, key=probabilities.get)
        choice = best if probabilities[best] >= threshold else "request_inspection"
        nll = -torch.tensor(max(probabilities[row["terminal"]], 1e-12)).log().item()
        brier = sum((score - (label == row["terminal"])) ** 2 for label, score in probabilities.items())
        predictions.append({**row, "argmax": best, "decision": choice, "nll": nll, "brier": brier})
    return predictions


def event_metrics(rows, predictions, label):
    labelled = [row for row in rows if row[label] is not None]
    reported = [row for row in labelled if row.get(predictions) is not None]
    values = []
    for row in reported:
        scores, truth = row[predictions], row[label]
        values.append({"scene_group_id": row["scene_group_id"], "correct": max(scores, key=scores.get) == truth,
                       "nll": -math.log(max(scores[truth], 1e-12)),
                       "brier": sum((score - (name == truth)) ** 2 for name, score in scores.items())})
    proper = {f"mean_{name}": grouped_mean(values, lambda row: row[name]) if values else None
              for name in ("nll", "brier")}
    return {"valid": len(labelled), "reported": len(reported), "correct": sum(row["correct"] for row in values),
            "independent_labelled_groups": len({row["scene_group_id"] for row in labelled}),
            "independent_reported_groups": len({row["scene_group_id"] for row in reported}), **proper}


def metrics(rows):
    controls = [row for row in rows if row["resolvable"]]
    containers = [row for row in controls if row["terminal"] in ("left", "right")]
    answered = [row for row in controls if row["decision"] != "request_inspection"]
    ambiguous = [row for row in rows if row["family"] == "F6"]
    groups = Counter(row["scene_group_id"] for row in rows)
    wrong = [row for row in rows if row["decision"] not in ("request_inspection", row["terminal"])]
    score = {field: sum(row[field] / groups[row["scene_group_id"]] for row in rows) / len(groups)
             for field in ("brier", "nll")}
    return {"independent_groups": len(groups), "resolvable_count": len(controls), "resolvable_answers": len(answered),
            "resolvable_correct": sum(row["decision"] == row["terminal"] for row in answered),
            "resolvable_container_count": len(containers),
            "resolvable_first_correct_container": sum(row["decision"] == row["terminal"] for row in containers),
            "resolvable_argmax_correct": sum(row["argmax"] == row["terminal"] for row in controls),
            "confident_wrong": len(wrong), "confident_wrong_groups": len({row["scene_group_id"] for row in wrong}),
            "f6_independent_pairs": len({row["scene_group_id"] for row in ambiguous}),
            "f6_inspections": sum(row["decision"] == "request_inspection" for row in ambiguous),
            "f6_variants": len(ambiguous), "mean_brier": score["brier"], "mean_nll": score["nll"],
            "event_labels": {"visual_placement": event_metrics(rows, "visual_placement", "settled_placement"),
                             "visual_relocation": event_metrics(rows, "visual_relocation", "relocation_assignment")}}


def calibrate(model, validation, features, priors, state_prior):
    candidates = (.7, 1., 1.4, 2.)
    scored = {temp: forecast(model, validation, features, priors, temp, state_prior) for temp in candidates}
    fitted = min(candidates, key=lambda temp: metrics(decide(scored[temp], .75))["mean_nll"])
    selected, validation_summary, candidates = choose_threshold(scored[fitted], decide, metrics)
    return fitted, selected, validation_summary, candidates


def stage_checkpoint(kind, model, optimize_stage, context, fit_draws, model_cap, deadline):
    sets, features, priors, output = context
    stages, trace = [], []
    state_prior = dict(zip(CUBE_SUPPORT, priors[0], strict=True))
    for boundary in (model_cap // 3, 2 * model_cap // 3, model_cap):
        remaining = boundary - len(fit_draws) - len(trace)
        elapsed = time.monotonic() >= deadline
        if remaining > 0 and not elapsed:
            previous = fit_draws + trace
            new = optimize_stage(remaining, len(previous), len(optimizer_losses(previous)))
            trace.extend(new)
            if optimizer_losses(new):
                updates = len(optimizer_losses(fit_draws + trace))
                result = forecast(model, sets["validation"], features, priors, 1., state_prior)
                validation = metrics(decide(result, .75))
                checkpoint = output / f"{kind}-{updates:04d}.pt"
                torch.save(model.state_dict(), checkpoint)
                stages.append({"update": updates, "opportunities": len(fit_draws) + len(trace),
                               "validation": validation, "checkpoint": str(checkpoint)})
                atomic(output / f"{kind}-stages.json", {"checkpoints": stages, "full_loss": optimizer_losses(trace)})
    if stages:
        best = min(stages, key=lambda row: row["validation"]["mean_nll"])
        model.load_state_dict(torch.load(best["checkpoint"], map_location="cpu", weights_only=True))
        return trace, stages, best["update"]
    return trace, stages, len(optimizer_losses(fit_draws))


def fit_model(kind, weight, context, stage_cap, model_cap, deadline):
    sets, features, priors, output = context
    torch.manual_seed(3)
    randomizer = random.Random(3)
    model = EvidenceHead()
    initial = digest(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    fit = fitting_rows(sets["train"])
    state_prior = dict(zip(CUBE_SUPPORT, priors[0], strict=True))
    before = metrics(decide(forecast(model, fit, features, priors, 1., state_prior), .75))
    fit_draws = optimize(model, optimizer, fit, features, priors, stage_cap, deadline, weight, randomizer, state_prior,
                         progress=(kind, 0, 0))
    after = metrics(decide(forecast(model, fit, features, priors, 1., state_prior), .75))
    choices_hit = after["resolvable_argmax_correct"] >= .75 * len(fit)
    def train_stage(count, offset, updates):
        return optimize(model, optimizer, sets["train"], features, priors, count, deadline,
                        weight, randomizer, state_prior, progress=(kind, offset, updates))

    trace, candidates, selected_update = stage_checkpoint(kind, model, train_stage, context, fit_draws, model_cap, deadline)
    fit_loss, full_loss = optimizer_losses(fit_draws), optimizer_losses(trace)
    report = {"kind": kind, "weight": weight, "updates": len(fit_loss) + len(full_loss),
              "optimizer_draw_order": [world for world, _ in fit_draws + trace],
              "fit_independent": len(fit), "fit_loss": fit_loss, "full_loss": full_loss, "fit_before": before,
              "fit_after": after, "fit_argmax_at_least_75_percent": choices_hit,
              "validation_candidates": candidates, "selected_update": selected_update,
              "changed_trainable_parameters": [name for name, value in digest(model).items() if initial[name] != value]}
    if not report["changed_trainable_parameters"]:
        raise ValueError("No trainable evidence parameters changed within the optimizer budget")
    return {**report, **freeze_model(kind, weight, model, context, state_prior), "wall_capped": time.monotonic() >= deadline}


def freeze_model(kind, weight, model, context, state_prior):
    sets, features, priors, output = context
    destination = output / f"{kind}.pt"
    temperature, threshold, validation, threshold_grid = calibrate(model, sets["validation"], features, priors, state_prior)
    valid_forecasts = forecast(model, sets["validation"], features, priors, temperature, state_prior)
    gains = validation_query_gains(valid_forecasts, state_prior)
    torch.save({"weights": model.state_dict(), "temperature": temperature, "priors": priors,
                "threshold": threshold, "training_seed": 3, "reveal_weight": weight, "state_prior": state_prior,
                "query_expected_information_nats": gains}, destination)
    return {"checkpoint": str(destination), "checkpoint_sha256": sha(destination), "temperature": temperature,
            "threshold": threshold, "validation": validation, "validation_thresholds": threshold_grid}


def score_model(record, sets, features):
    path = Path(record["checkpoint"])
    if sha(path) != record["checkpoint_sha256"]:
        raise ValueError("The selected evidence checkpoint changed before evaluation")
    frozen = torch.load(path, map_location="cpu", weights_only=True)
    model = EvidenceHead()
    model.load_state_dict(frozen["weights"], strict=True)
    priors, state_prior = frozen["priors"], frozen["state_prior"]
    temperature, threshold = frozen["temperature"], frozen["threshold"]
    results = decide(forecast(model, sets["test"], features, priors, temperature, state_prior), threshold)
    verify_forecasts(results)
    valid_forecasts = forecast(model, sets["validation"], features, priors, temperature, state_prior)
    pf = same_visual_pf(valid_forecasts, results, priors, state_prior, threshold, decide, metrics)
    communication = query_report(valid_forecasts, results, priors, state_prior, threshold, decide, metrics)
    gains = communication["validation_expected_information_nats"]
    if frozen.get("query_expected_information_nats", gains) != gains:
        raise ValueError("Replay query priorities differ from the frozen validation policy")
    methods = communication["policies"]
    communication["paired_one_reply_difference"] = paired_container_difference(
        methods["expected_information_gain"]["outcomes"], methods["latest_witness"]["outcomes"])
    families = {family: metrics([row for row in results if row["family"] == family])
                for family in {row["family"] for row in results}}
    return {**record, "state_prior": state_prior, "frozen_reload_pid": os.getpid(),
            "test": {**metrics(results), "decision_cost": decision_cost(results)}, "test_by_family": families,
            "test_group_bootstrap_95pct": grouped_intervals(results),
            "same_visual_without_association": association_ablation(results, threshold), "same_visual_particle_filter": pf,
            "shuffled_association_diagnostic": shuffled_association(results, priors, state_prior, threshold, decide, metrics),
            "communication": communication, "test_outcomes": results}


def verify_forecasts(rows):
    groups = defaultdict(list)
    for row in rows:
        if row["family"] == "F6":
            groups[row["scene_group_id"]].append(row)
    for first, second in groups.values():
        for field in ("visual_placement", "visual_relocation", "belief", "decision"):
            if first[field] != second[field]:
                raise ValueError(f"F6 coupling failed for {field}")


def grouped_intervals(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["scene_group_id"]].append(row)
    values = [[sum(item[field] for item in items) / len(items) for field in ("brier", "nll")]
              for items in groups.values()]
    generator = random.Random(31)
    draws = [generator.choices(values, k=len(values)) for _ in range(1000)]
    means = [sorted(sum(row[index] for row in draw) / len(draw) for draw in draws) for index in (0, 1)]
    return {field: [scores[24], scores[974]] for field, scores in zip(("brier", "nll"), means, strict=True)}


def paired_container_difference(model, comparator):
    reference = {row["world_episode_id"]: row for row in comparator}
    groups = defaultdict(list)
    for row in model:
        if row["resolvable"] and row["terminal"] in ("left", "right"):
            other = reference[row["world_episode_id"]]
            if other["terminal"] != row["terminal"] or other["scene_group_id"] != row["scene_group_id"]:
                raise ValueError("Paired comparison has different held-out group or evaluator truth")
            difference = int(row["decision"] == row["terminal"]) - int(other["decision"] == row["terminal"])
            groups[row["scene_group_id"]].append(difference)
    values = [sum(items) / len(items) for items in groups.values()]
    generator = random.Random(37)
    draws = sorted(sum(generator.choices(values, k=len(values))) / len(values) for _ in range(2000))
    return {"independent_paired_groups": len(groups), "difference": sum(values) / len(values),
            "bootstrap_95pct": [draws[49], draws[1949]], "meaningful_absolute_effect": .10,
            "target_interval_width": .20, "precision_target_met": draws[1949] - draws[49] <= .20}


def association_ablation(rows, threshold):
    independent = []
    for row in rows:
        # Share the same learned content belief while withholding identity-to-slot association.
        both = (row["belief"]["left"] + row["belief"]["right"]) / 2
        independent.append({**row, "belief": {**row["belief"], "left": both, "right": both}})
    return metrics(decide(independent, threshold))


def baselines(test):
    constant = predict([], 0)["belief"]
    prior = [{**row, "belief": constant} for row in test]
    visual = [{**row, "belief": Replay(Path(row["path"]), "rosmaster-1").predict()["belief"]} for row in test]
    oracle = [{**row, "belief": {name: float(name == row["terminal"]) for name in LOCATIONS}} for row in test]
    report = {name: metrics(decide(rows, .75)) for name, rows in (("constant_prior", prior),
              ("deterministic_rgb_reference", visual), ("privileged_terminal_oracle_ceiling", oracle))}
    for row in report.values():
        row.pop("event_labels")
    return {**report,
            "oracle_scope": "Evaluator outcome as a labelled ceiling; no deployable method receives it",
            "reference_input_audit": "SO101 final permitted top frame and Tank first/final permitted front frames; "
            "all are endpoints of the model's admitted clips. Only RGB rules, causal deliveries and fixed priors. "
            "No evaluator, annotations, privileged state or terminal reveal in the reference runtime."}


def encode_batch(encoder, rows):
    encoded = {}
    for row in rows:
        source = Path(row["path"])
        receiver = Replay(source, "rosmaster-1")
        features = admitted_features(encoder, source)
        encoded[row["world_episode_id"]] = tuple(
            feature if receiver.query + witness_delay_ns(agent) <= receiver.decision else None
            for agent, feature in zip(("so101-1", "tank-1"), features, strict=True))
    return encoded


def study_identity(args, config):
    script = Path(__file__).resolve()
    package = script.parents[1] / "src/smart_swarm"
    modules = ("contract", "episode", "evidence_inputs", "temporal_evidence", "memory", "vision",
               "particle_memory", "evidence_comparison", "evidence_queries")
    files = (script, script.with_name("learn_evidence.py"), *(package / f"{name}.py" for name in modules))
    return {"input_portfolio_sha256": sha(args.portfolio), "study_config_sha256": sha(args.config),
            "frozen_backbone_sha256": sha(Path(config["frozen_weights"])),
            "study_source_sha256": {path.name: sha(path) for path in files},
            "runtime_versions": {name: version(name) for name in ("torch", "torchvision", "numpy", "Pillow")}}


def matched_training_draws(results):
    orders = [row["optimizer_draw_order"] for row in results]
    common = min(map(len, orders))
    prefixes = [order[:common] for order in orders]
    if any(order != prefixes[0] for order in prefixes):
        raise ValueError("The evidence ablations changed the matched sample schedule")
    return {"verified_common_draws": common if len(results) > 1 else None,
            "draws_total_by_model": {row["kind"]: len(order) for row, order in zip(results, orders, strict=True)},
            "draws_without_update_by_model": {row["kind"]: len(order) - row["updates"]
                                              for row, order in zip(results, orders, strict=True)}}


def fit_study(args, sets, config, encoder):
    identity = study_identity(args, config)
    train_and_validation = sets["train"] + sets["validation"]
    print(f"Encoding {len(train_and_validation)} admitted training and validation recording variants", flush=True)
    features = encode_batch(encoder, train_and_validation)
    validation_cache = args.output / "validation-features.pt"
    torch.save({row["world_episode_id"]: features[row["world_episode_id"]] for row in sets["validation"]}, validation_cache)
    context = sets, features, priors_for(sets["train"]), args.output
    origin = {**identity, "fit_pid": os.getpid(), "validation_feature_sha256": sha(validation_cache),
              "per_model_cap_updates": 1500, "per_model_wall_cap_s": 1800}
    results = []
    for kind, weight in config["variants"].items():
        started = time.monotonic()
        outcome = fit_model(kind, weight, context, 200, 1500, started + 1740)
        results.append({**outcome, "actual_elapsed_s": time.monotonic() - started})
        atomic(args.output / "models-fit.json", {**origin, "models": results,
               "matched_training_draws": matched_training_draws(results)})


def evaluate_study(args, sets, config, encoder):
    fit = load(args.output / "models-fit.json")
    identity = study_identity(args, config)
    validation_cache = args.output / "validation-features.pt"
    complete_variants = {item["kind"] for item in fit["models"]} == set(config["variants"])
    if not complete_variants or any(fit[key] != value for key, value in identity.items()) or fit["fit_pid"] == os.getpid():
        raise ValueError("Held-out evaluation requires both selected checkpoints from a separate fitting process")
    if sha(validation_cache) != fit["validation_feature_sha256"]:
        raise ValueError("The cached validation feature set changed after fitting")
    features = torch.load(validation_cache, map_location="cpu", weights_only=True)
    if set(features) != {row["world_episode_id"] for row in sets["validation"]}:
        raise ValueError("Validation features cross the declared split")
    print(f"Encoding {len(sets['test'])} held-out recording variants for frozen scoring", flush=True)
    features.update(encode_batch(encoder, sets["test"]))
    prior = dict(zip(CUBE_SUPPORT, priors_for(sets["train"])[0], strict=True))
    comparison = calibrated_rule_pf(sets, prior, decide, metrics)
    atomic(args.output / "calibrated-rgb-particle-filter.json", comparison)
    controls = baselines(sets["test"])
    controls["calibrated_rgb_particle_filter"] = {key: comparison[key]
        for key in ("particles", "threshold", "scope", "validation", "test", "test_mean_joint_total_variation")}
    results = []
    for model in fit["models"]:
        score = score_model(model, sets, features)
        score["paired_primary_comparisons"] = {key: paired_container_difference(score["test_outcomes"], rows)
            for key, rows in (("calibrated_rgb_particle_filter", comparison["test_outcomes"]),
                             ("same_visual_particle_filter", score["same_visual_particle_filter"]["test_outcomes"]))}
        results.append(score)
        atomic(args.output / "models.json", {**fit, "models": results, "baselines": controls,
               "evaluation_pid": os.getpid(), "primary_metric": "first correct container choice on resolvable container histories"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--portfolio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", required=True, choices=("fit", "evaluate"))
    args = parser.parse_args()
    if not os.environ.get("SMART_SWARM_VERIFIED_RESEARCH_RESERVATION"):
        raise ValueError("A native research reservation is required")
    args.output.mkdir(exist_ok=True, parents=True)
    sets = admission(load(args.portfolio), audit_holdout=args.phase == "evaluate")
    config = load(args.config)
    torch.set_num_threads(3)
    encoder = FrozenVision(config["frozen_weights"])
    action = fit_study if args.phase == "fit" else evaluate_study
    action(args, sets, config, encoder)


if __name__ == "__main__":
    main()
