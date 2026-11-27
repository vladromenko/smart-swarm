#!/usr/bin/env python3
"""Export an offline primary-score package from the unchanged frozen result archives."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import tarfile
from tempfile import NamedTemporaryFile, TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from archive_paper_evaluation import EVIDENCE, FROZEN, ROOT, STUDY_CONFIG
from archive_primary_release import read_member
from smart_swarm.episode import json_bytes, load, sha
from verify_paper_primary import archived_cohort
from verify_portable_primary import HEADS, SCOPE, TEAMS, verify

ARCHIVES = {"validation": "primary-calibration-results", "test": "primary-independent-results"}
README = """# Smart Swarm primary score replay

Extract this ZIP into a new directory and run the following command using Python 3.11 or later.

```bash
python verify_portable_primary.py --output verified-tables.json
```

The verifier uses the Python standard library. It recomputes the original score rows for the two frozen heads,
seven registered methods and three team sizes. It keeps both F6 alternatives inside their shared source group
and checks equality of the saved policy outputs for the pair. The calibration and independent evaluation cohorts
retain their original split. It weights each underlying scene equally within a team size.

The exported forecasts contain the stored beliefs, decisions and byte counts needed for scoring. Evaluator outcomes
live in a separate file. Reference identities bind this projection to the original frozen archives and score tables.
The full Homebrain study bundle retains the scoped image inputs, wire records, ablations and training details.
Reproducing model inference, full input isolation and physics requires that full study material.
"""


def load_split(split):
    name = ARCHIVES[split]
    path = ROOT / EVIDENCE / f"{name}-homebrain.tar.gz"
    if sha(path) != FROZEN[name]:
        raise ValueError(f"The original primary archive changed for {split}")
    with tarfile.open(path) as archive:
        identity, groups, scores = archived_cohort(archive, split)
        reference = json.loads(read_member(archive, "analysis/method-table.json"))
    if scores != reference["scored_receipts_sha256"] or identity["source_groups"] != sorted(
            group["group"] for group in groups[HEADS[0]][TEAMS[0]]):
        raise ValueError("The frozen score table does not match the cohort")
    keys = ("scope", "registered_method_order", "metrics_by_model_and_team", "scored_receipts_sha256",
            "independent_source_groups", "recorded_variants_across_teams", "primary_test_release_sha256", "profiles_sha256")
    return groups, {"original_archive_sha256": FROZEN[name], **{key: reference[key] for key in keys}}


def project_cases(groups, split, methods):
    forecasts, outcomes = [], []
    for team in TEAMS:
        for paired in zip(*(groups[head][team] for head in HEADS), strict=True):
            ids = {group["group"] for group in paired}
            if len(ids) != 1:
                raise ValueError("Frozen objectives use different source groups")
            for matched in zip(*(group["outcomes"] for group in paired), strict=True):
                first = matched[0]
                fields = ("source", "source_manifest_sha256", "family", "variant", "truth")
                if any(any(row[key] != first[key] for key in fields) for row in matched):
                    raise ValueError("Frozen objectives use different cases or evaluator outcomes")
                key = {"split": split, "group": paired[0]["group"], "team": team, "variant": first["variant"],
                       "manifest_sha256": first["source_manifest_sha256"]}
                heads = {head: {name: {field: row[name][field] for field in ("belief", "decision", "serialized_bytes")}
                                for name in methods} for head, row in zip(HEADS, matched, strict=True)}
                forecasts.append({**key, "family": first["family"], "heads": heads})
                outcomes.append({**key, "truth": first["truth"]})
    return forecasts, outcomes


def contents():
    config = load(ROOT / STUDY_CONFIG)
    methods = config["primary_methods"]
    forecasts, outcomes, references = [], [], {}
    for split in SCOPE:
        groups, references[split] = load_split(split)
        if references[split]["registered_method_order"] != methods:
            raise ValueError("The collection and reference table differ on the primary methods")
        prediction_rows, evaluator_rows = project_cases(groups, split, methods)
        forecasts.extend(prediction_rows)
        outcomes.extend(evaluator_rows)
    files = {"README.md": README.encode(), "verify_portable_primary.py": Path(__file__).with_name(
             "verify_portable_primary.py").read_bytes()}
    for key, value in (("study", config), ("forecasts", forecasts), ("outcomes", outcomes), ("reference", references)):
        files[f"{key}.json"] = json_bytes(value) + b"\n"
    files["SHA256.json"] = json_bytes({name: sha256(value).hexdigest() for name, value in files.items()}) + b"\n"
    return files


def publish(output):
    if output.exists():
        raise ValueError("Choose a new immutable replay package path")
    files = contents()
    with TemporaryDirectory(prefix="smart-swarm-portable-primary-") as directory:
        root = Path(directory)
        for name, payload in files.items():
            (root / name).write_bytes(payload)
        verify(root)
    with NamedTemporaryFile(dir=output.parent, suffix=".zip", delete=False) as pending:
        destination = Path(pending.name)
    try:
        with ZipFile(destination, "w", compression=ZIP_DEFLATED) as archive:
            for name in sorted(files):
                item = ZipInfo(name)
                item.compress_type = ZIP_DEFLATED
                archive.writestr(item, files[name], compresslevel=9)
        destination.replace(output)
    finally:
        destination.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    publish(args.output)
    print(args.output, sha(args.output))


if __name__ == "__main__":
    main()
