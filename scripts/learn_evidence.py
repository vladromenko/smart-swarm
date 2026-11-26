"""Fit and evaluate the bounded causal evidence pilot on grouped histories."""

import argparse
import json
from collections import Counter
import os
from pathlib import Path
import random

import torch

from smart_swarm.evidence_inputs import audit_b1, clip, runtime_signature
from smart_swarm.episode import atomic, load, sha
from smart_swarm.memory import CUBE_SUPPORT, PERMUTATIONS
from smart_swarm.temporal_evidence import (EvidenceHead, FrozenVision, LOCATIONS, joint_logbelief,
                                           loss_for, marginalized)


def admitted_features(encoder, root):
    arm = clip(root, "so101-1")
    tank = clip(root, "tank-1")
    if tank is None:
        raise ValueError("Pilot expects a permitted Tank clip")
    return (encoder(arm) if arm is not None else None, encoder(tank))


def priors_for(records):
    rows = [item for item in records if item["split"] == "train"]
    result = []
    for field, names in (("settled_placement", CUBE_SUPPORT), ("relocation_assignment", PERMUTATIONS)):
        counts = Counter(row[field] for row in rows)
        weights = [counts[name] + .5 for name in names]
        result.append([value / sum(weights) for value in weights])
    return result


def group_data(portfolio):
    records = portfolio["records"]
    owners = {}
    groups = {}
    for item in records:
        identifier = item["scene_group_id"]
        split = item["split"]
        if owners.setdefault(identifier, split) != split:
            raise ValueError("Leakage across source scene-group splits")
        groups.setdefault(identifier, []).append(item)
    expected = {"F1", "F2", "F6-contained", "F6-surface", "F6-resolvable"}
    if (len(groups) != 5 or any({row["variant"] for row in group} != expected for group in groups.values())
            or sorted(Counter(owners.values()).values()) != [1, 1, 3]):
        raise ValueError("Bounded five-group evidence pilot is incomplete")
    return records


def fit(rows, features, priors, iterations, weight, seed):
    torch.manual_seed(seed)
    generator = random.Random(seed)
    model = EvidenceHead()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    history = []
    for index in range(iterations):
        record = generator.choice(rows)
        outputs = model(*features[record["world_episode_id"]])
        loss = loss_for(outputs, record, priors, reveal_weight=weight)
        if not torch.isfinite(loss):
            raise ValueError("Non-finite evidence objective")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        history.append(float(loss.detach()))
    return model.eval(), history


@torch.no_grad()
def score(model, rows, features, priors, temperature):
    report = []
    for item in rows:
        belief = marginalized(joint_logbelief(*model(*features[item["world_episode_id"]]),
                                               priors, temperature))
        reference = item["terminal"]
        probabilities = dict(zip(LOCATIONS, belief.exp().tolist(), strict=True))
        pick = max(probabilities, key=probabilities.get)
        if probabilities[pick] < .75:
            pick = "request_inspection"
        report.append({"variant": item["variant"], "source_world_episode_id": item["world_episode_id"],
                       "scene_group_id": item["scene_group_id"], "terminal": reference,
                       "probabilities": probabilities, "decision": pick,
                       "terminal_nll": -float(belief[LOCATIONS.index(reference)]),
                       "b1_rule_reference": item["untrained_rule_decision"]})
    return report


def select_temperature(model, validation, features, priors):
    grid = (.7, 1., 1.4, 2.)
    nll = {temp: sum(row["terminal_nll"] for row in score(model, validation, features, priors, temp))
           for temp in grid}
    return min(grid, key=nll.get)


def run_model(name, weight, context):
    cfg, records, features, priors, output = context
    datasets = {split: [item for item in records if item["split"] == split]
                for split in ("train", "validation", "test")}
    model, trace = fit(datasets["train"], features, priors, cfg["training_step_cap_per_model"],
                       weight, cfg["seed"])
    temp = select_temperature(model, datasets["validation"], features, priors)
    outcomes = score(model, datasets["test"], features, priors, temp)
    counterfactual = [row for row in outcomes if row["variant"] in ("F6-contained", "F6-surface")]
    if len(counterfactual) != 2 or counterfactual[0]["probabilities"] != counterfactual[1]["probabilities"]:
        raise ValueError("Trained F6 counterfactual predictions diverged")
    target = output / f"{name}.pt"
    torch.save({"weights": model.state_dict(), "priors": priors, "temperature": temp,
                "training_seed": cfg["seed"], "reveal_weight": weight}, target)
    return {"model": name, "updates": len(trace), "checkpoint": str(target), "checkpoint_sha256": sha(target),
            "classification_temperature": temp, "validation_group_count": 1, "test_group_count": 1,
            "train_loss": trace, "test_outcomes": outcomes, "f6_trained_counterfactual_equal": True,
            "parameter_count": sum(param.numel() for param in model.parameters())}


def train(args, cfg, encoder):
    if not os.environ.get("SMART_SWARM_VERIFIED_RESEARCH_RESERVATION"):
        raise ValueError("Optimizer requires the existing-reservation job launcher")
    records = group_data(load(args.portfolio))
    features = {row["world_episode_id"]: admitted_features(encoder, Path(row["path"])) for row in records}
    context = cfg, records, features, priors_for(records), args.output
    results = []
    for name, weight in cfg["variants"].items():
        results.append(run_model(name, weight, context))
        atomic(args.output / "models.json", {"models": results, "frozen_backbone_sha256": sha(Path(cfg["frozen_weights"]))})


def audit(args, cfg, encoder):
    outcome = audit_b1(args.b1, args.scratch)
    pairs = [p.parent for p in (args.b1 / "worlds").glob("*/manifest.json")
             if load(p)["family"] == "F6"]
    torch.manual_seed(cfg["seed"])
    model = EvidenceHead().eval()
    priors = ([.2] * 5, [.5] * 2)
    with torch.inference_mode():
        signatures = [runtime_signature(p) for p in pairs]
        features = [admitted_features(encoder, p) for p in pairs]
        logits = [joint_logbelief(*model(*pair), priors) for pair in features]
    if signatures[0] != signatures[1] or not torch.equal(logits[0], logits[1]):
        raise ValueError("Untrained evidence model failed coupled F6 input invariance")
    outcome.update(visual_feature_and_untrained_head_invariant=True, optimizer_updates=0,
                   frozen_backbone_sha256=sha(Path(cfg["frozen_weights"])))
    atomic(args.output / "b1-full-input-audit.json", outcome)


def forecast(args, cfg, encoder):
    from smart_swarm.episode import Replay
    from smart_swarm.memory import witness_delay_ns

    receiver = Replay(args.run, "rosmaster-1")
    deadline = receiver.decision if args.through_ns is None else min(args.through_ns, receiver.decision)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = EvidenceHead().eval()
    model.load_state_dict(checkpoint["weights"], strict=True)
    arm, tank = admitted_features(encoder, args.run)
    roles = ("so101-1", "tank-1")
    delivered = {role: tensor is not None and receiver.query + witness_delay_ns(role) <= deadline
                 for role, tensor in zip(roles, (arm, tank), strict=True)}
    admitted = (arm if delivered["so101-1"] else None, tank if delivered["tank-1"] else None)
    with torch.inference_mode():
        belief = marginalized(joint_logbelief(*model(*admitted), checkpoint["priors"],
                                               checkpoint["temperature"], state_prior=checkpoint.get("state_prior"))).exp()
    probabilities = dict(zip(LOCATIONS, belief.tolist(), strict=True))
    choice = max(probabilities, key=probabilities.get)
    threshold = checkpoint.get("threshold", cfg["abstain_below"])
    selection = choice if probabilities[choice] >= threshold else "request_inspection"
    return {"belief": probabilities, "decision": selection, "query_ns": receiver.query, "through_ns": deadline,
            "delivered_sources": delivered, "transport": "fixed recorded-protocol witness delays",
            "policy_scope": "per-agent causal camera clips and scoped witness factors"}


def budgeted_forecast(args, cfg, encoder):
    import math
    from smart_swarm.evidence_comparison import learned_prior_map
    from smart_swarm.evidence_queries import exchange, opportunities
    from smart_swarm.memory import PRIOR
    from smart_swarm.particle_memory import FACTORS

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    gains = checkpoint.get("query_expected_information_nats", {})
    valid_gains = set(gains) == set(FACTORS) and all(math.isfinite(value) and value >= 0 for value in gains.values())
    if args.query_policy == "expected_information_gain" and not valid_gains:
        raise ValueError("Expected information gain requires validation priorities sealed in the checkpoint")
    model = EvidenceHead().eval()
    model.load_state_dict(checkpoint["weights"], strict=True)
    query, recorded_deadline, ads = opportunities(args.run)
    deadline = recorded_deadline if args.through_ns is None else min(args.through_ns, recorded_deadline)
    encoded = []

    def posterior(factor):
        arm = factor == "containment"
        role = "so101-1" if arm else "tank-1"
        encoded.append(role)
        features = encoder(clip(args.run, role))
        with torch.inference_mode():
            logits = model(features, None)[0] if arm else model(None, features)[1]
            probabilities = (logits / checkpoint["temperature"]).softmax(0).tolist()
        return dict(zip(FACTORS[factor], probabilities, strict=True))

    result = exchange(args.query_policy, (query, deadline, ads if deadline >= query else []), gains, posterior,
                      learned_prior_map(checkpoint["priors"]), checkpoint.get("state_prior", PRIOR))
    best = max(result["belief"], key=result["belief"].get)
    threshold = checkpoint.get("threshold", cfg["abstain_below"])
    return {**result, "decision": best if result["belief"][best] >= threshold else "request_inspection",
            "query_policy": args.query_policy, "query_ns": query, "through_ns": deadline, "encoded_witnesses": encoded,
            "policy_scope": "Read-only replay of only the requested witness image clips"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--b1", type=Path)
    parser.add_argument("--scratch", type=Path)
    parser.add_argument("--portfolio", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--through-ns", type=int)
    parser.add_argument("--query-policy", choices=("broadcast", "latest_witness", "expected_information_gain"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    cfg = load(args.config)
    encoder = FrozenVision(cfg["frozen_weights"])
    if args.run is not None:
        if args.checkpoint is None:
            raise ValueError("A trained temporal evidence checkpoint is required")
        action = budgeted_forecast if args.query_policy is not None else forecast
        print(json.dumps(action(args, cfg, encoder), allow_nan=False))
    elif args.portfolio is None:
        audit(args, cfg, encoder)
    else:
        train(args, cfg, encoder)


if __name__ == "__main__":
    main()
