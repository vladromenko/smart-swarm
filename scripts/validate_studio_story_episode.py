"""Read one saved episode through SO101 Studio's native raw validator."""

import argparse
import json
from pathlib import Path

from so101_lab.hardware import JOINTS
from so101_lab.recording import SCHEMA, _hash_file, _validate_episode_raw, dataset_cameras
from so101_lab.resources import ROOT, ResourceBusy, dataset_lease


def validate(path, technical):
    episode = path.resolve()
    dataset = episode.parent.parent
    source = "synthetic_test" if technical else "hardware"
    admitted_root = ROOT.resolve() if technical else (ROOT / "data").resolve()
    if not dataset.is_relative_to(admitted_root) or episode.parent.name != "episodes":
        raise ValueError("Expected an original saved SO101 Studio episode")
    with dataset_lease(dataset):
        manifest = dataset / "dataset.json"
        metadata = json.loads(manifest.read_text())
        if (metadata.get("source") != source or metadata.get("schema_version") != SCHEMA
                or metadata.get("dataset_id") != dataset.name or metadata.get("joints") != list(JOINTS)
                or metadata.get("units") != ["degrees"] * 5 + ["percent_calibrated_range"]
                or dataset_cameras(metadata) != ("top", "wrist")):
            raise ValueError("Expected the two-camera native source and calibrated joint contract")
        metrics = _validate_episode_raw(episode, metadata, allow_test_data=technical)
        return {**metrics, "source": source, "technical_only": technical, "scope": "native raw episode",
                "dataset_id": dataset.name, "dataset_manifest_sha256": _hash_file(manifest),
                "episode_metadata_sha256": _hash_file(episode / "episode.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--allow-technical-rehearsal", action="store_true")
    args = parser.parse_args()
    try:
        result = validate(args.episode, args.allow_technical_rehearsal)
    except ResourceBusy as error:
        parser.exit(75, f"{error}\n")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"{type(error).__name__}: {error}\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
