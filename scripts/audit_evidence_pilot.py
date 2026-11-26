"""Preflight the installed learner on admitted training and validation observations."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from learn_evidence_v3 import digest, encode_batch, fitting_rows, priors_for, study_identity, verify_pair
from smart_swarm.contract import json_bytes
from smart_swarm.episode import Replay, atomic, load
from smart_swarm.evidence_inputs import clip_packets
from smart_swarm.memory import CUBE_SUPPORT
from smart_swarm.temporal_evidence import EvidenceHead, FrozenVision, loss_for

OBSERVERS = ("so101-1", "tank-1")
FIELDS = ("settled_placement", "relocation_assignment", "terminal")


def visual_digest(path):
    camera_trace = []
    for role in OBSERVERS:
        client = Replay(path, role)
        cameras = [[(frame["sensor_id"], frame["sha256"]) for frame in packet["cameras"]]
                   for packet in clip_packets(client)]
        camera_trace.append(cameras)
    return sha256(json_bytes(camera_trace)).hexdigest()


def inventory(records):
    scopes, pixels = {}, defaultdict(set)
    worlds = set()
    for row in records:
        path = Path(row["path"])
        seal = load(path / "manifest.json")
        group, split = row["scene_group_id"], row["split"]
        world = row["world_episode_id"]
        ancestry = (split, row["family"])
        if (seal["world_episode_id"] != world or seal["scene_group_id"] != group or world in worlds
                or seal["lineage"]["split"] != split or scopes.setdefault(group, ancestry) != ancestry):
            raise ValueError("An archived recording has duplicated identity or incorrect world ancestry")
        worlds.add(world)
        pixels[visual_digest(path)].add(group)
    counts = {}
    for split in ("train", "validation"):
        selected = [row for row in records if row["split"] == split]
        masks = {field: dict(Counter(str(row[field]) for row in selected)) for field in FIELDS}
        family_groups = {family: len({row["scene_group_id"] for row in selected if row["family"] == family})
                         for family in sorted({row["family"] for row in selected})}
        counts[split] = {"independent_histories": sum(family_groups.values()), "records": len(selected),
                         "groups_by_family": family_groups, "label_coverage_by_record": masks,
                         "intermediate_labels": sum(row["intermediate_event_labels"] for row in selected)}
    duplicates = sorted(sorted(groups) for groups in pixels.values() if len(groups) > 1)
    return counts, duplicates


def sampled_rows(records):
    buckets = defaultdict(list)
    for row in records:
        buckets[(row["split"], row["family"])].append(row)
    selected = []
    for (_, family), group in sorted(buckets.items()):
        first = group[0]
        if family == "F6":
            pair = [row for row in group if row["scene_group_id"] == first["scene_group_id"]]
            verify_pair(pair)
            selected.extend(pair)
        else:
            selected.append(first)
    return selected


def gradient_check(model, placement, relocation, row, priors, prior, reveal):
    inputs = model(placement, relocation)
    loss = loss_for(inputs, row, priors, reveal_weight=reveal, state_prior=prior)
    if not torch.isfinite(loss):
        raise ValueError("The archived sample produced a non-finite learner objective")
    if not loss.requires_grad:
        return {"loss": float(loss), "nonzero_finite_gradient_tensors": 0}
    gradients = torch.autograd.grad(loss, tuple(model.parameters()), allow_unused=True)
    active = [gradient for gradient in gradients if gradient is not None]
    if any(not torch.isfinite(gradient).all() for gradient in active):
        raise ValueError("A learner gradient is non-finite on an admitted sample")
    count = sum(bool(torch.count_nonzero(gradient)) for gradient in active)
    return {"loss": float(loss.detach()), "nonzero_finite_gradient_tensors": count}


def feature_checks(model, selected, features, training):
    counts = {}
    priors = priors_for(training)
    prior = dict(zip(CUBE_SUPPORT, priors[0], strict=True))
    for row in selected:
        world = row["world_episode_id"]
        arm, tank = features[world]
        if tank is None or tuple(tank.shape) != (8, 512) or not torch.isfinite(tank).all():
            raise ValueError("The permitted Tank clip did not produce eight finite native visual embeddings")
        if ((arm is None) != (row["family"] == "F6") or
                arm is not None and (tuple(arm.shape) != (8, 1024) or not torch.isfinite(arm).all())):
            raise ValueError("The arm clip shape or withholding differs from the declared evidence scope")
        counts[world] = {"family": row["family"], "split": row["split"], "arm_visible": arm is not None,
                         "event_only": gradient_check(model, arm, tank, row, priors, prior, 0.),
                         "event_plus_reveal": gradient_check(model, arm, tank, row, priors, prior, .35)}
    for split in ("train", "validation"):
        hidden = [row for row in selected if row["split"] == split and row["family"] == "F6"]
        first, second = [features[row["world_episode_id"]][1] for row in hidden]
        if not torch.equal(first, second):
            raise ValueError("Matched F6 sources produced different features from identical permitted inputs")
    return counts


def audit(portfolio_path, config_path, output):
    torch.manual_seed(3)
    torch.set_num_threads(1)
    payload = portfolio_path.read_bytes()
    portfolio, config = json.loads(payload), load(config_path)
    rows = [row for row in portfolio["records"] if row["split"] in ("train", "validation")]
    counts, duplicates = inventory(rows)
    training = [row for row in rows if row["split"] == "train"]
    cohort, checked = fitting_rows(training), sampled_rows(rows)
    encoder = FrozenVision(config["frozen_weights"])
    features = encode_batch(encoder, checked)
    model = EvidenceHead()
    initial = digest(model)
    gradients = feature_checks(model, checked, features, training)
    if digest(model) != initial or any(value.requires_grad for value in encoder.parameters()):
        raise ValueError("Read-only preflight changed weights or left the backbone unfrozen")
    identity = study_identity(SimpleNamespace(portfolio=portfolio_path, config=config_path), config)
    identity["input_portfolio_sha256"] = sha256(payload).hexdigest()
    result = {**identity, "scope": "Training and validation only, with no optimizer or held-out visual access",
              "optimizer_updates": 0, "snapshot_counts": counts,
              "same_admitted_visual_clip_across_different_groups": duplicates, "samples_checked": gradients,
              "balanced_fitting_independent_groups": len({row["scene_group_id"] for row in cohort}),
              "f6_feature_counterfactual_equal": True, "backbone_frozen": True}
    atomic(output, result)
    return result


if __name__ == "__main__":
    arguments = argparse.ArgumentParser(description=__doc__)
    arguments.add_argument("--portfolio", type=Path, required=True)
    arguments.add_argument("--config", type=Path, required=True)
    arguments.add_argument("--output", type=Path, required=True)
    args = arguments.parse_args()
    audit(args.portfolio.resolve(), args.config.resolve(), args.output.resolve())
