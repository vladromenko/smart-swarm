"""Package and independently rescore the existing Smart Swarm paper evidence."""

import argparse
from gzip import GzipFile
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path, PurePosixPath
import tarfile
from tempfile import NamedTemporaryFile, TemporaryDirectory

from archive_motor_results import verify as verify_motor
from archive_primary_compact_controls import verify as verify_compact
from archive_primary_release import put, read_member, verify as verify_calibration
from archive_rosmaster_inspection import verify as verify_inspection
from audit_primary_inspection_thresholds import collect, figure, markdown
from report_witness_acquisition import markdown as witness_table, report as witness_report
from smart_swarm.contract import json_bytes
from smart_swarm.episode import load, sha
from verify_paper_diagnostics import MATERIAL, verify_diagnostics
from verify_paper_primary import verify_primary

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = Path("evidence/learning-pilot-v3")
STUDY_CONFIG = "configs/primary-memory-study.json"
INDEX = "PAPER-EVALUATION.json"
SCHEMA = "smart-swarm-paper-evaluation-v2"
LEGACY_SCHEMA = "smart-swarm-paper-evaluation-v1"
FROZEN = {
    "primary-calibration-results": "57b7037c771a06c553bda0aef40591ebe845a4a6456443956eaad6ba238ca792",
    "primary-frozen-calibration": "b17fec81d8281ee14eae9831f9f24ecd4363845a273f3fd2655c2e851f8500f4",
    "primary-independent-results": "cb8e12354782d5bdbba787c3aceedc9c3bb06475e872e049896fe9d9c51ecf2f",
    "primary-matched-compact-controls": "f247928eba2ac9218417a3cf91a65e7071d197bd6dfc389c8f289810255d47d7",
    "rosmaster-bounded-comparison": "56f1d0477a657c5f493c9b33fd48ca439f8f668cecf4c705d18a399d9517d7e2",
    "verified-motor-results": "86b0af6d341ab9ffb273275f122c432eb8190640f69dbf504f4adfbba4a922cb",
    "witness-transfer": "2d6f352144d4845fed60b0f89ad1a8097c7387ca05348c68befc2141bc241a10",
}
REPORTS = (
    "README.md", "pyproject.toml", STUDY_CONFIG, "docs/PAPER_DRAFT.md", "docs/PAPER_DRAFT.pdf", "docs/RESEARCH_PROTOCOL.md",
    "docs/CLAIMS_AND_LIMITATIONS.md", "docs/LITERATURE_MATRIX.md", "docs/PRIMARY_RESULTS_CALIBRATION.md",
    "docs/PRIMARY_RESULTS_INDEPENDENT.md", "docs/PRIMARY_MATCHED_COMPACT_RESULTS.md", "docs/WITNESS_TRANSFER_RESULTS.md",
    "docs/ROSMASTER_ACTIVE_INSPECTION_RESULTS.md", "docs/MOTOR_RESULTS_REPRODUCIBILITY.md", "docs/PRIMARY_INSPECTION_THRESHOLDS.md",
)
FIGURES = ("primary-example-f1-views.pdf", "primary-test-decision-cost.pdf", "primary-test-inspection-controls.pdf",
           "primary-inspection-thresholds.json", "primary-inspection-thresholds.pdf", "primary-fixed-final-witness-order.pdf",
           "primary-matched-compact-controls.pdf", "rosmaster-bounded-comparison-effect.pdf")


def archive_paths(root):
    return {name: root / EVIDENCE / f"{name}-homebrain.tar.gz" for name in FROZEN}


def verify_witness(path, working):
    stem = "results/witness-transfer-reframed-evaluation"
    with tarfile.open(path) as archive:
        index = json.loads(read_member(archive, "index.json"))
        hashes = index["sha256"]
        names = archive.getnames()
        if (index["schema"] != "smart-swarm-witness-transfer-bundle-v1"
                or set(names) != {"index.json", *hashes} or len(names) != len(hashes) + 1):
            raise ValueError("The witness transfer archive has unexpected members")
        files = {name: read_member(archive, name) for name in hashes}
    if any(sha256(data).hexdigest() != hashes[name] for name, data in files.items()):
        raise ValueError("A saved witness transfer member changed")
    score = working / "witness-score.json"
    score.write_bytes(files[f"{stem}.json"])
    labels = load(score)
    predictions = json.loads(files[f"{stem}-sealed-predictions.json"])["predictions"]
    keys = ("manifest_sha256", "methods", "path", "source_group")
    projected = [{key: row[key] for key in keys} for row in labels["case_results"]]
    if labels["predictions_sha256"] != hashes[f"{stem}-sealed-predictions.json"] or projected != predictions:
        raise ValueError("The viewpoint scores differ from sealed predictions")
    result = witness_report(score)
    if (files["results/witness-transfer-paired-results.json"] != json_bytes(result) + b"\n"
            or files["results/WITNESS_TRANSFER_RESULTS.md"] != witness_table(result).encode()):
        raise ValueError("The viewpoint table differs from the original per-source results")
    return {"independent_source_groups": result["independent_sources"], "paired_variants": result["paired_camera_variants"],
            "table_reproduced": True}


def verify_thresholds(root, test, validation, calibration, working):
    result = collect(test, validation, calibration)
    if (load(root / EVIDENCE / "primary-inspection-thresholds.json") != result
            or (root / "docs/PRIMARY_INSPECTION_THRESHOLDS.md").read_text() != markdown(result)):
        raise ValueError("Frozen threshold analysis differs from the original results")
    output = working / "thresholds.pdf"
    figure(result, output)
    if output.read_bytes() != (root / EVIDENCE / "primary-inspection-thresholds.pdf").read_bytes():
        raise ValueError("The regenerated inspection threshold figure differs from the original")
    return {"profiles_retained": True, "table_reproduced": True, "figure_reproduced": True}


def verify_components(root, working, *, diagnostics=False):
    archives = archive_paths(root)
    for name, path in archives.items():
        if sha(path) != FROZEN[name]:
            raise ValueError(f"The original frozen archive differs at {name}")
    calibration = archives["primary-frozen-calibration"]
    cal = verify_calibration(calibration)
    with tarfile.open(calibration) as archive:
        frozen_config = read_member(archive, "configuration/primary-memory-study.json")
    if (root / STUDY_CONFIG).read_bytes() != frozen_config:
        raise ValueError("The packaged study configuration differs from the frozen calibration")
    config = json.loads(frozen_config)
    validation, test = archives["primary-calibration-results"], archives["primary-independent-results"]
    primary = {split: verify_primary(path, split, config, working)
               for split, path in (("validation", validation), ("test", test))}
    first, second = primary["validation"].pop("source_groups"), primary["test"].pop("source_groups")
    if set(first) & set(second) or any(row["profile_sha256"] != cal["profile_sha256"] for row in primary.values()):
        raise ValueError("The primary splits overlap or do not use the same frozen profile")
    compact = verify_compact(archives["primary-matched-compact-controls"])
    inspection = verify_inspection(archives["rosmaster-bounded-comparison"])
    compact.pop("archive")
    inspection.pop("archive")
    result = {"scope": "Recorded simulation and historical evaluator results",
              "frozen_archives": FROZEN, "primary": primary, "compact": compact, "inspection": inspection,
              "witness": verify_witness(archives["witness-transfer"], working),
              "motor": verify_motor(archives["verified-motor-results"]),
              "thresholds": verify_thresholds(root, test, validation, calibration, working)}
    if diagnostics:
        result["diagnostics"] = verify_diagnostics(root, working, archives)
    return result


def bundle_files(root):
    paths = [root / name for name in REPORTS]
    paths.extend(root / EVIDENCE / name for name in FIGURES)
    paths.extend(root / EVIDENCE / name for name in MATERIAL)
    paths.extend(archive_paths(root).values())
    paths.extend(sorted((root / "src/smart_swarm").glob("*.py")))
    paths.extend(sorted((root / "scripts").glob("*.py")))
    files = {path.relative_to(root).as_posix(): path.read_bytes() for path in paths}
    packages = ("numpy", "scipy", "matplotlib", "pillow", "mujoco", "torch", "av")
    ver = {name: version(name) for name in packages}
    receipt = {"schema": SCHEMA, "versions": ver,
               "contents_sha256": {name: sha256(data).hexdigest() for name, data in files.items()}}
    return files, receipt


def verify_bundle(path):
    with TemporaryDirectory(prefix="smart-swarm-paper-bundle-") as temporary:
        root = Path(temporary)
        with tarfile.open(path) as archive:
            index = json.loads(read_member(archive, INDEX))
            hashes = index["contents_sha256"]
            names = archive.getnames()
            if index["schema"] not in (SCHEMA, LEGACY_SCHEMA) or set(names) != {INDEX, *hashes} or len(names) != len(hashes) + 1:
                raise ValueError("The paper bundle has unexpected or duplicate members")
            diagnostics = includes_diagnostics(index["schema"], hashes)
            for name, expected in hashes.items():
                relative = member_path(name)
                data = read_member(archive, name)
                if sha256(data).hexdigest() != expected:
                    raise ValueError(f"A packaged paper entry changed at {name}")
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
        result = verify_components(root, root, diagnostics=diagnostics)
    return {**result, "bundle_sha256": sha(path), "members_verified": len(hashes)}


def publish(path):
    if path.exists():
        raise ValueError("Choose a new immutable paper bundle path")
    files, receipt = bundle_files(ROOT)
    with NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as handle:
        scratch = Path(handle.name)
    try:
        with scratch.open("wb") as raw, GzipFile(fileobj=raw, filename="", mode="wb", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                put(archive, INDEX, json_bytes(receipt) + b"\n")
                for name, data in sorted(files.items()):
                    put(archive, name, data)
        result = verify_bundle(scratch)
        path.hardlink_to(scratch)
        return result
    finally:
        scratch.unlink()


def includes_diagnostics(schema, hashes):
    required = {(EVIDENCE / name).as_posix() for name in MATERIAL}
    expected = required if schema == SCHEMA else set()
    if required.intersection(hashes) != expected:
        raise ValueError("The paper diagnostics disagree with the package revision")
    return schema == SCHEMA


def member_path(name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != name or name == ".":
        raise ValueError("A paper member escapes its package layout")
    return relative


def verify_tree(root):
    path = root / INDEX
    index = load(path) if path.is_file() else None
    if index is None:
        diagnostics = all((root / EVIDENCE / name).is_file() for name in MATERIAL)
    else:
        if index["schema"] not in (SCHEMA, LEGACY_SCHEMA):
            raise ValueError("Unknown paper package revision")
        hashes = index["contents_sha256"]
        diagnostics = includes_diagnostics(index["schema"], hashes)
        for name, digest in hashes.items():
            target = root / member_path(name)
            if not target.resolve().is_relative_to(root.resolve()) or sha(target) != digest:
                raise ValueError(f"A packaged paper entry changed at {name}")
    with TemporaryDirectory(prefix="smart-swarm-paper-scoring-") as temporary:
        return verify_components(root, Path(temporary), diagnostics=diagnostics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, nargs="?")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--build", action="store_true")
    mode.add_argument("--verify-tree", action="store_true")
    args = parser.parse_args()
    if args.verify_tree != (args.archive is None):
        parser.error("Pass an archive, or use --verify-tree from an extracted package")
    if args.verify_tree:
        result = verify_tree(ROOT)
    else:
        result = publish(args.archive) if args.build else verify_bundle(args.archive)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
