"""Materialize only the causal inputs available to the primary-study policies."""

from contextlib import contextmanager
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from .contract import json_bytes
from .episode import DELIVERIES, POLICY, atomic, listed_file, load, sha
from .evidence_inputs import canonical_delivery, canonical_packet, opaque_map, packets_and_transcript


def json_lines(path, items):
    path.write_bytes(b"".join(json_bytes(item) + b"\n" for item in items))


def copy_camera_stream(source, target, records, mapping):
    projected = []
    for packet in records:
        clean = canonical_packet(packet, mapping)
        for original, camera in zip(packet["cameras"], clean["cameras"], strict=True):
            image = listed_file(source, original["frame_ref"], POLICY)
            path = target / POLICY / f"{camera['frame_id']}{image.suffix}"
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(image, path)
            camera["frame_ref"] = str(path.relative_to(target))
        projected.append(clean)
    return projected


@contextmanager
def primary_policy_view(source):
    packets, messages, query, receiver = packets_and_transcript(source)
    mapping = opaque_map(packets, receiver.world_id)
    original = load(source / "manifest.json")
    keys = ("policy_cameras", "query_boundary", "decision_boundary")
    manifest = {key: original[key] for key in keys}
    manifest.update(world_episode_id="world", sidecar_sha256={})
    with TemporaryDirectory(prefix="smart-swarm-primary-policy-") as directory:
        path = Path(directory)
        (path / POLICY).mkdir()
        (path / "communication").mkdir()
        for agent, records in packets.items():
            projected = copy_camera_stream(source, path, records, mapping)
            json_lines(path / POLICY / f"{agent}.jsonl", projected)
        json_lines(path / "communication/queries.jsonl", [query])
        json_lines(path / DELIVERIES, (canonical_delivery(row, mapping) for row in messages))
        manifest["sidecar_sha256"][DELIVERIES] = sha(path / DELIVERIES)
        atomic(path / "manifest.json", manifest)
        yield path
