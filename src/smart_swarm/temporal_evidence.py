"""Small causal temporal heads and a differentiable ten-state belief update."""

import torch
from torch import nn
from torch.nn.functional import cross_entropy, log_softmax
from torchvision.models import resnet18

from .memory import CUBE_SUPPORT, PERMUTATIONS, PRIOR, SLOTS, STATES, TRACKS

LOCATIONS = (*SLOTS, "surface", "held", "outside")


class EvidenceHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.arm_rnn = nn.GRU(1024, 48, batch_first=True)
        self.tank_rnn = nn.GRU(512, 48, batch_first=True)
        self.insertion = nn.Linear(48, len(CUBE_SUPPORT))
        self.relocation = nn.Linear(48, len(PERMUTATIONS))

    def forward(self, arm, tank):
        transport = self.relocation(self.tank_rnn(tank.unsqueeze(0))[0][0, -1]) if tank is not None else None
        if arm is None:
            return None, transport
        hidden = self.arm_rnn(arm.unsqueeze(0))[0][0, -1]
        return self.insertion(hidden), transport


class FrozenVision(nn.Module):
    def __init__(self, path):
        super().__init__()
        model = resnet18(weights=None)
        weights = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(weights, strict=True)
        model.fc = nn.Identity()
        self.model = model.eval().requires_grad_(False)
        self.register_buffer("mean", torch.tensor((.485, .456, .406))[None, :, None, None])
        self.register_buffer("std", torch.tensor((.229, .224, .225))[None, :, None, None])

    @torch.no_grad()
    def forward(self, rgb):
        seq, cameras, height, width, channels = rgb.shape
        data = torch.from_numpy(rgb.reshape(seq*cameras, height, width, channels))
        data = data.permute(0, 3, 1, 2).float() / 255.
        data = nn.functional.interpolate(data, size=(128, 128), mode="bilinear",
                                         align_corners=False, antialias=True)
        return self.model((data - self.mean) / self.std).reshape(seq, cameras * 512)


def log_factors(logits, priors, temperature=1.):
    if logits is None:
        return torch.zeros(len(priors))
    return log_softmax(logits / temperature, dim=-1) - torch.tensor(priors).log()


def joint_logbelief(placement_logits, relocation_logits, class_priors, temperature=1., *, state_prior=None):
    placement_prior, transport_prior = class_priors
    support = PRIOR if state_prior is None else state_prior
    placement = log_factors(placement_logits, placement_prior, temperature)
    transport = log_factors(relocation_logits, transport_prior, temperature)
    path_logits = torch.stack([placement[CUBE_SUPPORT.index(location)] + transport[PERMUTATIONS.index(mapping)]
                               + torch.tensor(support[location] / len(PERMUTATIONS)).log()
                               for location, mapping in STATES])
    return log_softmax(path_logits, dim=0)


def target_index(location, permutation):
    slot = "left" if permutation.startswith(location) else "right"
    return LOCATIONS.index(slot if location in TRACKS else location)


def marginalized(log_belief):
    mask = [target_index(location, permutation) for location, permutation in STATES]
    return torch.stack([torch.logsumexp(log_belief[[idx for idx, name in enumerate(mask) if name == k]], dim=0)
                        for k in range(len(LOCATIONS))])


def loss_for(outputs, labels, priors, *, reveal_weight, state_prior=None):
    placement, transport = outputs
    events = torch.zeros(())
    for prediction, label, vocabulary in ((placement, labels["settled_placement"], CUBE_SUPPORT),
                                           (transport, labels["relocation_assignment"], PERMUTATIONS)):
        if prediction is not None and label is not None:
            events = events + cross_entropy(prediction.unsqueeze(0), torch.tensor([vocabulary.index(label)]))
    if reveal_weight:
        log_belief = joint_logbelief(placement, transport, priors, state_prior=state_prior)
        events = events - reveal_weight * marginalized(log_belief)[LOCATIONS.index(labels["terminal"])]
    return events
