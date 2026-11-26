"""Observed marker support for a witness's own causal RGB clip."""

import numpy as np

from .evidence_inputs import clip_packets
from .memory import TRACKS
from .vision import center


def marker_support(images):
    seen = dict.fromkeys(TRACKS, 0)
    examined, together = 0, 0
    for image in images:
        visible = {color: center(np.asarray(image), color) is not None for color in TRACKS}
        examined += 1
        together += all(visible.values())
        for color, observed in visible.items():
            seen[color] += observed
    return {"examined_frames": examined, "marker_frames": seen, "both_marker_frames": together,
            "supported": together > 0}


def witness_marker_support(client):
    images = (client.image(camera["frame_id"]) for packet in clip_packets(client) for camera in packet["cameras"])
    return marker_support(images)
