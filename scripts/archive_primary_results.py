#!/usr/bin/env python3
"""Package sealed primary scores and their reproducible analysis together."""

import argparse
from gzip import GzipFile
from hashlib import sha256
import json
from pathlib import Path
import tarfile
from tempfile import NamedTemporaryFile, TemporaryDirectory

from archive_primary_release import INDEX, VALIDATION, put, read_member, stash, verify as verify_calibration
from audit_frozen_f6_results import collect as audit_f6
from capture_primary_test import TEST_OUTPUT
from compare_primary_supervision import authority, compare, prediction_path, read_scores, score_paths
from export_primary_result_table import collect as result_table, markdown
from freeze_primary_profiles import PILOT
from smart_swarm.contract import json_bytes
from smart_swarm.episode import load, sha, validate

SCHEMA = "smart-swarm-primary-results-archive-v1"
ANALYSIS_SOURCES = ("archive_primary_results.py", "archive_primary_release.py", "compare_primary_supervision.py", "study_figures.py",
                    "export_primary_result_table.py", "audit_frozen_f6_results.py")


def manifests(files, identity, scope):
    root = (TEST_OUTPUT if scope == "test" else VALIDATION).resolve()
    references = {}
    for location, digest in identity["manifests"].items():
        episode = Path(location).resolve()
        original = episode / "manifest.json"
        manifest = load(original)
        if not episode.is_relative_to(root) or episode.name != manifest["world_episode_id"] or manifest["lineage"]["split"] != scope:
            raise ValueError(f"Result manifest is outside its declared {scope} source")
        member = f"manifests/{episode.name}/manifest.json"
        stash(files, member, original, digest)
        references[str(original)] = {"archive_member": member, "sha256": digest}
    return references


def output_documents(files, profile, scope):
    table = result_table(profile, scope)
    audit = audit_f6(profile, scope)
    comparison = compare(profile, scope)
    reports = {"method-table": table, "f6-audit": audit, "paired-supervision": comparison}
    for name, result in reports.items():
        files[f"analysis/{name}.json"] = json_bytes(result) + b"\n"
    files["analysis/method-table.md"] = markdown(table).encode()
    with TemporaryDirectory(prefix="smart-swarm-results-figure-") as directory:
        from study_figures import paired_supervision

        data, output = (Path(directory) / name for name in ("paired.json", "paired.pdf"))
        data.write_bytes(files["analysis/paired-supervision.json"])
        paired_supervision(data, output)
        files["analysis/paired-supervision.pdf"] = output.read_bytes()


def build(profile, scope, calibration):
    paths = score_paths(scope)
    identity, traces = authority(profile, scope)
    for kind, path in paths.items():
        read_scores(path, kind, identity)
    calibration_info = verify_calibration(calibration)
    if calibration_info["profile_sha256"] != identity["profiles_sha256"]:
        raise ValueError("The calibration archive belongs to another result profile")
    files = {}
    stash(files, "configuration/primary-profiles-frozen.json", profile, identity["profiles_sha256"])
    release = identity["primary_test_release_sha256"]
    if release is not None:
        stash(files, "configuration/primary-test-release.json", TEST_OUTPUT / "primary-test-release.json", release)
    references = manifests(files, identity, scope)
    for kind, path in paths.items():
        stash(files, f"scores/{path.name}", path)
        predictions = prediction_path(path)
        stash(files, f"scores/{predictions.name}", predictions, load(path)["predictions_sha256"])
    captured = {Path(entries[0]["path"]).parent / "capture.json" for entries in traces.values()}
    for path in sorted(captured):
        stash(files, f"capture/{path.parent.parent.name}/capture.json", path)
    for name in ANALYSIS_SOURCES:
        stash(files, f"implementation/{name}", Path(__file__).with_name(name))
    output_documents(files, profile, scope)
    return files, {"schema": SCHEMA, "scope": scope, "profiles_sha256": identity["profiles_sha256"],
                   "primary_test_release_sha256": release, "independent_source_groups": len(identity["source_groups"]),
                   "external_calibration_archive": {"path": str(calibration.resolve()), "sha256": calibration_info["archive_sha256"]},
                   "original_recording_manifests": references, "contents_sha256": {
                       name: sha256(value).hexdigest() for name, value in files.items()}}


def check_originals(index):
    calibration = index["external_calibration_archive"]
    proof = verify_calibration(Path(calibration["path"]), live_sources=True)
    if proof["archive_sha256"] != calibration["sha256"] or proof["profile_sha256"] != index["profiles_sha256"]:
        raise ValueError("The original calibration archive changed")
    root = (TEST_OUTPUT if index["scope"] == "test" else VALIDATION).resolve()
    for location, record in index["original_recording_manifests"].items():
        path = Path(location).resolve()
        if path.name != "manifest.json" or not path.is_relative_to(root) or sha(path) != record["sha256"]:
            raise ValueError(f"The original result manifest changed at {path}")
        validate(path.parent, load(path))


def verify(path, *, originals=False):
    with tarfile.open(path, "r:gz") as archive:
        index = json.loads(read_member(archive, INDEX))
        hashes = index["contents_sha256"]
        names = archive.getnames()
        scope, release = index["scope"], index["primary_test_release_sha256"]
        if (index["schema"] != SCHEMA or scope not in ("validation", "test") or (scope == "test") != (release is not None)
                or set(names) != {INDEX, *hashes} or len(names) != len(hashes) + 1):
            raise ValueError("The results archive has mismatched members")
        for name, digest in hashes.items():
            if sha256(read_member(archive, name)).hexdigest() != digest:
                raise ValueError(f"The archived result member changed at {name}")
        if hashes["configuration/primary-profiles-frozen.json"] != index["profiles_sha256"]:
            raise ValueError("The archived profile differs from its result index")
        if hashes.get("configuration/primary-test-release.json") != index["primary_test_release_sha256"]:
            raise ValueError("The archived independent release differs from its result index")
        if any(hashes[row["archive_member"]] != row["sha256"] for row in index["original_recording_manifests"].values()):
            raise ValueError("An archived recording manifest differs from its lineage")
    if originals:
        check_originals(index)
    return {"scope": index["scope"], "archive_sha256": sha(path), "members_verified": len(hashes),
            "independent_source_groups": index["independent_source_groups"],
            "recording_manifests": len(index["original_recording_manifests"]), "originals_verified": originals}


def publish(path, files, index):
    if path.exists():
        raise ValueError("Choose a new immutable result archive path")
    with NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as handle:
        working = Path(handle.name)
    try:
        with working.open("wb") as stream, GzipFile(fileobj=stream, filename="", mode="wb", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                put(archive, INDEX, json_bytes(index) + b"\n")
                for name, data in sorted(files.items()):
                    put(archive, name, data)
        verify(working)
        path.hardlink_to(working)
    finally:
        working.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--scope", choices=("validation", "test"))
    parser.add_argument("--profiles", type=Path, default=PILOT / "primary-profiles-frozen.json")
    parser.add_argument("--calibration-archive", type=Path, default=PILOT / "primary-frozen-calibration-homebrain.tar.gz")
    parser.add_argument("--check-originals", action="store_true")
    args = parser.parse_args()
    try:
        if args.scope is not None:
            publish(args.archive, *build(args.profiles, args.scope, args.calibration_archive))
        print(json.dumps(verify(args.archive, originals=args.check_originals), indent=2))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"Primary results archive withheld. {error}\n")


if __name__ == "__main__":
    main()
