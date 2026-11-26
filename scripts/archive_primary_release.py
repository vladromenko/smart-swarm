#!/usr/bin/env python3
"""Archive the exact frozen calibration, weights and fixed-profile validation receipts."""

import argparse
from gzip import GzipFile
from hashlib import sha256
from importlib.metadata import version
from io import BytesIO
import json
from pathlib import Path
import sys
import tarfile
from tempfile import NamedTemporaryFile

from smart_swarm.contract import json_bytes
from smart_swarm.episode import load, sha, validate

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "evidence/learning-pilot-v3"
CONFIG = ROOT / "configs/primary-memory-study.json"
ALLOCATION = BASE / "primary-sim-source-allocation.json"
BACKBONE_CONFIG = ROOT / "configs/evidence-pilot.json"
VALIDATION = ROOT.parent.parent / "work/primary-sim-validation"
KINDS = ("events_only", "events_plus_reveal")
INDEX = "RELEASE.json"


def stash(files, member, path, expected=None):
    payload = path.read_bytes()
    digest = sha256(payload).hexdigest()
    if member in files or (expected is not None and digest != expected):
        raise ValueError(f"Frozen file identity failed for {path}")
    files[member] = payload
    return digest


def stash_code(files, profile):
    for relative, digest in profile["implementation_sha256"].items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or not relative.startswith(("src/", "scripts/")):
            raise ValueError("Frozen comparison lists an unexpected implementation path")
        stash(files, f"implementation/{relative}", path, digest)
    stash(files, f"configuration/{CONFIG.name}", CONFIG, profile["config_sha256"])
    stash(files, f"configuration/{ALLOCATION.name}", ALLOCATION, profile["source_allocation_sha256"])
    stash(files, f"configuration/{BACKBONE_CONFIG.name}", BACKBONE_CONFIG)


def stash_manifests(files, profile):
    references = {}
    for path, digest in profile["validation_cohort"]["manifests"].items():
        root = Path(path).resolve()
        if not root.is_relative_to(VALIDATION.resolve()):
            raise ValueError("Archived calibration references a non-validation recording")
        original = root / "manifest.json"
        manifest = load(original)
        if manifest["lineage"]["split"] != "validation" or root.name != manifest["world_episode_id"]:
            raise ValueError("Calibration recording identity and split disagree")
        relative = f"validation/{root.name}/manifest.json"
        stash(files, relative, original, digest)
        references[str(original)] = {"archive_member": relative, "sha256": digest}
    return references


def stash_rehearsal(files, kind, profile_digest, cohort):
    from score_primary_test import check_prediction_receipt

    stem = f"primary-frozen-rehearsal-forty-eight-{kind}"
    report_path = BASE / f"{stem}.json"
    predictions_path = BASE / f"{stem}-predictions.json"
    identity = {"scope": "validation replay using fixed profiles", "checkpoint_kind": kind,
                "profiles_sha256": profile_digest, "primary_test_release_sha256": None,
                "source_groups": cohort["source_groups"], "manifests": cohort["manifests"]}
    report = load(report_path)
    if (any(report.get(key) != value for key, value in identity.items()) or report["fit_on_inputs"]
            or report["training_labels_read_by_predictor"]):
        raise ValueError("Fixed-profile validation receipt has a different identity or runtime scope")
    check_prediction_receipt(load(predictions_path), identity, cohort["manifests"])
    stash(files, f"rehearsal/{predictions_path.name}", predictions_path, report["predictions_sha256"])
    stash(files, f"rehearsal/{report_path.name}", report_path)


def stash_models(files, profile, profile_digest):
    training = BASE / "story-learning.json"
    study = load(training)
    particle = Path(study["calibrated_particle_report"])
    for kind in KINDS:
        fit = profile["fits"][kind]
        stash(files, f"models/{kind}.pt", Path(fit["checkpoint"]), fit["checkpoint_sha256"])
        if sha(particle) != fit["particle_reference_sha256"]:
            raise ValueError("Frozen particle reference changed")
        matrix = BASE / f"primary-matrix-forty-eight-{kind}.json"
        stash(files, f"calibration/{matrix.name}", matrix, fit["fit_matrix_sha256"])
        stash_rehearsal(files, kind, profile_digest, profile["validation_cohort"])
    stash(files, f"calibration/{training.name}", training)
    stash(files, f"calibration/{particle.name}", particle, profile["fits"][KINDS[0]]["particle_reference_sha256"])
    fitted = Path(study["frozen_fit_report"])
    stash(files, f"calibration/{fitted.name}", fitted)
    encoder = Path(load(BACKBONE_CONFIG)["frozen_weights"]).resolve()
    if sha(encoder) != profile["encoder_sha256"]:
        raise ValueError("The installed frozen image encoder changed")
    return {str(encoder): profile["encoder_sha256"]}


def build(profile_path):
    from freeze_primary_profiles import require_profiles

    profile = require_profiles(profile_path)
    files = {}
    profile_member = "calibration/primary-profiles-frozen.json"
    digest = stash(files, profile_member, profile_path)
    if json.loads(files[profile_member]) != profile:
        raise ValueError("Frozen profile changed while collecting its provenance")
    stash_code(files, profile)
    references = stash_manifests(files, profile)
    external = stash_models(files, profile, digest)
    libraries = {name: version(name) for name in ("numpy", "scipy", "Pillow", "torch", "torchvision", "mujoco")}
    files["configuration/local-runtime.json"] = json_bytes({"python": sys.version.split()[0], "packages": libraries}) + b"\n"
    return files, {"schema": "smart-swarm-frozen-calibration-archive-v1", "profile_member": profile_member,
                   "profile_sha256": digest, "validation_source_groups": profile["validation_count"],
                   "validation_recording_manifests": references, "external_assets_sha256": external,
                   "independent_test_recordings_archived": 0,
                   "contents_sha256": {name: sha256(value).hexdigest() for name, value in files.items()}}


def put(archive, name, data):
    entry = tarfile.TarInfo(name)
    entry.size = len(data)
    entry.mode = 0o644
    archive.addfile(entry, BytesIO(data))


def read_member(archive, member):
    entry = archive.getmember(member)
    if not entry.isreg():
        raise ValueError(f"Restricted archive member {member}")
    handle = archive.extractfile(entry)
    if handle is None:
        raise ValueError(f"Missing regular archive member {member}")
    with handle:
        return handle.read()


def verify(path, *, live_sources=False):
    with tarfile.open(path, "r:gz") as archive:
        index = json.loads(read_member(archive, INDEX))
        hashes = index["contents_sha256"]
        names = archive.getnames()
        if (index["schema"] != "smart-swarm-frozen-calibration-archive-v1"
                or set(names) != {INDEX, *hashes} or len(names) != len(hashes) + 1):
            raise ValueError("The calibration archive membership changed")
        for member, digest in hashes.items():
            if sha256(read_member(archive, member)).hexdigest() != digest:
                raise ValueError(f"Archive member changed: {member}")
        if hashes[index["profile_member"]] != index["profile_sha256"]:
            raise ValueError("Archived profile identity differs")
    if live_sources:
        verify_originals(index)
    return {"archive": str(path), "archive_sha256": sha(path), "members_verified": len(hashes),
            "profile_sha256": index["profile_sha256"], "validation_source_groups": index["validation_source_groups"],
            "validation_manifests": len(index["validation_recording_manifests"]), "live_sources_checked": live_sources}


def verify_originals(index):
    root = VALIDATION.resolve()
    for location, row in index["validation_recording_manifests"].items():
        source = Path(location).resolve()
        if source.name != "manifest.json" or not source.is_relative_to(root) or sha(source) != row["sha256"]:
            raise ValueError(f"Original validation manifest changed: {location}")
        validate(source.parent, load(source))
    encoder = Path(load(BACKBONE_CONFIG)["frozen_weights"]).resolve()
    expected = index["external_assets_sha256"]
    if set(expected) != {str(encoder)} or sha(encoder) != expected[str(encoder)]:
        raise ValueError("Original frozen encoder changed")


def publish(path, files, index):
    if path.exists():
        raise ValueError("Choose a new immutable calibration archive path")
    with NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with temporary.open("wb") as stream, GzipFile(fileobj=stream, filename="", mode="wb", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                put(archive, INDEX, json_bytes(index) + b"\n")
                for name, data in sorted(files.items()):
                    put(archive, name, data)
        verify(temporary)
        path.hardlink_to(temporary)
    finally:
        temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--profiles", type=Path)
    parser.add_argument("--check-originals", action="store_true")
    args = parser.parse_args()
    if args.profiles is not None:
        publish(args.archive, *build(args.profiles))
    print(json.dumps(verify(args.archive, live_sources=args.check_originals), indent=2))


if __name__ == "__main__":
    main()
