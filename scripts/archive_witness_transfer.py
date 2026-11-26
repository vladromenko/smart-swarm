"""Package the frozen transfer evidence without duplicating raw recordings or weights."""

import argparse
from gzip import GzipFile
from hashlib import sha256
from pathlib import Path
import tarfile
from tempfile import NamedTemporaryFile

from archive_primary_release import put
from smart_swarm.cli import ROOT
from smart_swarm.episode import Replay, json_bytes, load, sha


def build(original, test, policy, score, analysis, table):
    result, frozen = load(score), load(policy)
    predictions = score.with_name(score.stem + "-sealed-predictions.json")
    if (result["status"] != "evaluated" or result["policy_sha256"] != sha(policy)
            or result["predictions_sha256"] != sha(predictions) or load(analysis)["score_sha256"] != sha(score)):
        raise ValueError("The score, fitted policy, prediction seal and paired analysis disagree")
    files = {f"design/{name}": path.read_bytes() for name, path in
             (("validation.json", original / "design.json"), ("test.json", test / "design.json"),
              ("clipping-rejections.json", original / "progress.json"), ("reframe-qualification.json", test / "qualification.json"))}
    if sha(test / "design.json") != result["test_capture_design_sha256"]:
        raise ValueError("The reserved evaluation design changed")
    files.update({f"results/{file.name}": file.read_bytes() for file in (policy, score, analysis, table, predictions)})
    for name, expected in frozen["code_sha256"].items():
        file = ROOT / name
        if sha(file) != expected:
            raise ValueError("The original fitted implementation changed")
        files["implementation/" + name] = file.read_bytes()
    source = result["source_groups"][0]["variants"][0]
    root = Path(source["path"])
    if sha(root / "manifest.json") != source["manifest_sha256"]:
        raise ValueError("The admitted camera example changed")
    view = Replay(root, "tank-1")
    scoped = view.packets()
    for label, packet in (("before", scoped[0]), ("after", scoped[-1])):
        camera, = packet["cameras"]
        view.image(camera["frame_id"])
        files[f"example/tank-1-{label}.png"] = (root / camera["frame_ref"]).read_bytes()
    files["README.md"] = (
        "# Smart Swarm witness transfer\n\nReproduce the table from the sealed score with "
        "`PYTHONPATH=src python scripts/report_witness_acquisition.py --score SCORE --output ANALYSIS --table TABLE`.\n\n"
        "The evaluation and simulator sources use the installed Homebrain Python, MuJoCo, NumPy, PyTorch and software EGL. "
        "Raw recordings, original encoder and checkpoint weights remain at the locations referenced by the receipts.\n").encode()
    return files


def package(path, files):
    if path.exists():
        raise ValueError("Choose a new immutable evidence bundle path")
    digests = {name: sha256(data).hexdigest() for name, data in files.items()}
    index = {"schema": "smart-swarm-witness-transfer-bundle-v1", "sha256": digests}
    with NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with temporary.open("wb") as raw, GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                put(archive, "index.json", json_bytes(index) + b"\n")
                for name, data in sorted(files.items()):
                    put(archive, name, data)
        with tarfile.open(temporary, "r:gz") as archive:
            for name, expected in digests.items():
                with archive.extractfile(name) as member:
                    if sha256(member.read()).hexdigest() != expected:
                        raise ValueError("A packed transfer artifact changed")
        path.hardlink_to(temporary)
    finally:
        temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original", "test", "policy", "score", "analysis", "table", "archive"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    package(args.archive, build(args.original, args.test, args.policy, args.score, args.analysis, args.table))
    print(args.archive)


if __name__ == "__main__":
    main()
