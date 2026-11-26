"""Package and rescore the matched compact replay without its original RGB files."""

import argparse
from gzip import GzipFile
from hashlib import sha256
import json
from pathlib import Path
import tarfile
from tempfile import NamedTemporaryFile, TemporaryDirectory

from archive_primary_release import INDEX, put, read_member, stash
from capture_primary_test import TEST_OUTPUT
from replay_primary_compact_controls import CONTROL_FILES, PILOT, PROFILE
from score_primary_compact_controls import CONFIG, figure, markdown, source_metrics
from smart_swarm.episode import json_bytes, load, sha


SCHEMA = "smart-swarm-matched-compact-replay-v1"
REPORT = PILOT / "primary-matched-compact-controls.json"
FIGURE = PILOT / "primary-matched-compact-controls.pdf"
TABLE = PILOT.parent.parent / "docs/PRIMARY_MATCHED_COMPACT_RESULTS.md"
ANALYSIS = ("score_primary_compact_controls.py", "study_figures.py", "calibrate_team_views.py",
            "score_primary_validation.py", "audit_primary_pairs.py", "archive_primary_compact_controls.py")


def collect(root):
    files = {}
    report, profile = load(REPORT), load(PROFILE)
    entries = {"controlled-replay/release.json": (root / "release.json", report["release_sha256"]),
               "controlled-replay/budget.json": (root / "budget.json", None),
               "controlled-replay/predictions.json": (root / "predictions.json", report["predictions_sha256"]),
               "scoring/evaluator-index.json": (root / "evaluator-index.json", report["evaluator_index_sha256"]),
               "scoring/study-config.json": (CONFIG, profile["config_sha256"]),
               "scoring/source-allocation.json": (PILOT / "primary-sim-source-allocation.json", profile["source_allocation_sha256"]),
               "scoring/primary-test-release.json": (TEST_OUTPUT / "primary-test-release.json", None),
               "scoring/primary-profiles-frozen.json": (PROFILE, None),
               "analysis/report.json": (REPORT, None), "analysis/table.md": (TABLE, None), "analysis/figure.pdf": (FIGURE, None)}
    for member, (path, expected) in entries.items():
        stash(files, member, path, expected)
    for name in sorted(set(CONTROL_FILES + ANALYSIS)):
        stash(files, f"implementation/{name}", Path(__file__).with_name(name))
    return files


def read_verified(path):
    with tarfile.open(path, "r:gz") as archive:
        index = json.loads(read_member(archive, INDEX))
        hashes = index["contents_sha256"]
        names = archive.getnames()
        if index["schema"] != SCHEMA or set(names) != {INDEX, *hashes} or len(names) != len(hashes) + 1:
            raise ValueError("Compact replay archive membership differs from its manifest")
        files = {name: read_member(archive, name) for name in hashes}
    if any(sha256(data).hexdigest() != hashes[name] for name, data in files.items()):
        raise ValueError("A compact replay archive member no longer matches its digest")
    return files, hashes


def verify_bindings(files, hashes, report, predictions):
    profile = json.loads(files["scoring/primary-profiles-frozen.json"])
    release = json.loads(files["controlled-replay/release.json"])
    digest = hashes["scoring/primary-test-release.json"]
    if (report["predictions_sha256"] != hashes["controlled-replay/predictions.json"]
            or report["evaluator_index_sha256"] != hashes["scoring/evaluator-index.json"]
            or report["release_sha256"] != hashes["controlled-replay/release.json"]
            or profile["config_sha256"] != hashes["scoring/study-config.json"]
            or profile["source_allocation_sha256"] != hashes["scoring/source-allocation.json"]
            or release["primary_profile_sha256"] != hashes["scoring/primary-profiles-frozen.json"]
            or report["score_implementation_sha256"] != hashes["implementation/score_primary_compact_controls.py"]):
        raise ValueError("The archived result is bound to another policy or source release")
    if any(hashes[f"implementation/{name}"] != value for name, value in release["implementation_sha256"].items()):
        raise ValueError("A frozen prediction implementation changed in the archive")
    groups = predictions["groups"]
    if (sha256(json_bytes(groups)).hexdigest() != predictions["groups_payload_sha256"] or any(
            row["binding"]["primary_test_release_sha256"] != digest or
            row["binding"]["release_sha256"] != report["release_sha256"] for row in groups)):
        raise ValueError("The archived forecast seal lost its exact source release")


def verify_source_roster(groups, labels, allocation):
    expected = {(row["source_group"], size): (row["family"], set(row["capture_variants"]))
                for row in allocation["slots"] if row["split"] == "test" for size in row["paired_team_sizes"]}
    observed = {}
    for row in groups:
        key = row["source_group"], row["team_size"]
        records = [labels[variant["path"]] for variant in row["variants"]]
        families = {label["family"] for label in records}
        variants = {label["variant"] for label in records}
        if key in observed or len(families) != 1 or len(variants) != len(records):
            raise ValueError("The archive repeats a group or alters the assigned family or variant order")
        observed[key] = (next(iter(families)), variants)
    if expected != observed:
        raise ValueError("The archive does not cover the reserved F1–F6 and team-size roster")


def verify(path):
    files, hashes = read_verified(path)
    report = json.loads(files["analysis/report.json"])
    predictions = json.loads(files["controlled-replay/predictions.json"])
    labels = json.loads(files["scoring/evaluator-index.json"])
    verify_bindings(files, hashes, report, predictions)
    verify_source_roster(predictions["groups"], labels, json.loads(files["scoring/source-allocation.json"]))
    count = len({row["source_group"] for row in predictions["groups"]})
    if (count != report["independent_sources"] or len(labels) != report["team_condition_variants"]
            or source_metrics(predictions["groups"], labels, json.loads(files["scoring/study-config.json"])) != report["results"]
            or markdown(report).encode() != files["analysis/table.md"]):
        raise ValueError("The packaged source data does not reproduce the published score and table")
    with TemporaryDirectory(prefix="smart-swarm-compact-figure-") as tmp:
        fresh = Path(tmp) / "comparison.pdf"
        figure(report, fresh)
        if fresh.read_bytes() != files["analysis/figure.pdf"]:
            raise ValueError("The packaged results do not reproduce the paper figure")
    return {"archive": str(path), "sha256": sha(path), "members_verified": len(hashes), "independent_sources": count,
            "arithmetic_reproduced": True, "table_reproduced": True, "figure_reproduced": True,
            "raw_recordings_required_for_verification": False}


def publish(path, root):
    files = collect(root)
    index = {"schema": SCHEMA, "contents_sha256": {name: sha256(value).hexdigest() for name, value in files.items()}}
    with NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as handle:
        temporary = Path(handle.name)
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
    parser.add_argument("--root", type=Path, help="Include the completed replay and editor-ready analysis")
    args = parser.parse_args()
    if args.root is not None:
        if args.archive.exists():
            parser.error("Choose a new immutable archive path")
        publish(args.archive, args.root)
    print(json.dumps(verify(args.archive), indent=2))


if __name__ == "__main__":
    main()
