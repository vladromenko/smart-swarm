"""Package the retained motor scores and recompute outcomes from saved evaluator samples."""

import argparse
from collections import Counter
from gzip import GzipFile
from hashlib import sha256
import json
from pathlib import Path
import tarfile
from tempfile import NamedTemporaryFile

from archive_primary_release import put, read_member
from smart_swarm.cli import ROOT
from smart_swarm.contract import json_bytes
from smart_swarm.episode import load, sha
from smart_swarm.evaluate_motor import sha256_stream
from smart_swarm.motor_diagnostics import contact_valid_placement, summarize

PILOT = ROOT / "evidence/learning-pilot-v3"
INVENTORY = "training-inventory-final.json"
INVENTORY_SHA = "e106f96afe6ad3887ccf41a218df7080fd1842419747587b41b5c626a61a58f0"
TEACHER = "teacher-replays.json"
FRESH = "fresh-resets.json"
MOTOR_JOBS = (
    ("act-pilot-04ea2a31947b", "ACT historical", "historical"),
    ("smolvla-pilot-67fb3abbb312", "SmolVLA historical", "historical"),
    ("act-five-fit-e1205c1dfa3c", "ACT aligned fit", "matching"),
    ("act-clock-fit-cdfa3cdae4e0", "ACT clock-conditioned fit", "matching"),
    ("smolvla-pipeline-d9510148a574", "SmolVLA compact", "compact"),
)
SCHEMA = "smart-swarm-retained-motor-results-v1"


def index_jobs(artifact):
    if sha256(artifact).hexdigest() != INVENTORY_SHA:
        raise ValueError("The retained training inventory changed")
    saved = {row["job"]: row for row in json.loads(artifact)["jobs"]}
    return {name: saved[name] for name, _, _ in MOTOR_JOBS}


def bound_contents(contents, receipt):
    if sha256(contents).hexdigest() != receipt["sha256"]:
        raise ValueError(f"Original receipt changed at {receipt['path']}")
    return contents


def motor_artifacts(name, job, scope):
    base = f"jobs/{name}"
    artifacts = {f"{base}/{split}-lineage.json": record["lineage"] for split, record in job["dataset"]["splits"].items()}
    artifacts.update({f"{base}/processors/{name}": record for name, record in job["configuration_and_processors"].items()})
    artifacts.update({f"{base}/{field}.json": job[field] for field in ("training_receipt", "step_receipt")})
    artifacts[f"{base}/evaluation.json"] = job["evaluation"]["receipt"]
    if scope == "compact":
        artifacts.update({f"{base}/{split}-evaluation.json": job["evaluation"][split]["receipt"]
                          for split in ("validation", "fresh")})
    return artifacts


def feedback_counts(feedback, result):
    if [row["tick"] for row in feedback] != list(range(15, 511, 15)):
        raise ValueError("An aligned evaluation lacks one of its 34 declared contact samples")
    computed = summarize(feedback)
    if computed["first_failed_stage"] != result["first_failed_stage"]:
        raise ValueError("Saved stage differs from the original evaluator samples")
    saved = result.get("contact_valid_placement")
    if saved is not None and saved != contact_valid_placement(computed):
        raise ValueError("Saved outcome differs from the original contact and release evidence")
    if "held_lift_10cm" in result and any(result[key] != value for key, value in computed.items()):
        raise ValueError("Saved matching-run statistics differ from their evaluator samples")
    return contact_valid_placement(computed), computed["first_failed_stage"]


def evaluate_cases(rows, expected_count, expected_success, *, historical=False):
    seeds = [row["seed"] for row in rows]
    if len(seeds) != expected_count or len(set(seeds)) != len(seeds):
        raise ValueError("A saved evaluation is incomplete or repeats a reset")
    scored = []
    for row in rows:
        feedback = row.get("feedback", row.get("sampled_evaluator_truth"))
        if len(feedback) < 2:
            raise ValueError("An evaluation lacks its sampled feedback")
        if historical:
            placed = all(item["inside_carrier"] and item["released"] and item["stationary"] for item in feedback[-2:])
            if placed != row["contained_and_released"]:
                raise ValueError("The historical placement endpoint changed")
            scored.append((placed, "unrecorded"))
        else:
            scored.append(feedback_counts(feedback, row))
    successes = sum(placed for placed, _ in scored)
    if successes != expected_success:
        raise ValueError("Recomputed motor successes disagree with the original count")
    return {"resets": len(seeds), "successes": successes, "seeds": seeds,
            "first_failure_stages": dict(Counter(stage or "none" for placed, stage in scored if not placed))}


def dataset_counts(files, base, job):
    sets = {}
    counts = {}
    for split, original in job["dataset"]["splits"].items():
        raw = files[f"{base}/{split}-lineage.json"]
        if sha256(raw).hexdigest() != original["lineage"]["sha256"]:
            raise ValueError("An archived research split differs from its export")
        lineage = json.loads(raw)
        rows = lineage["episodes"]
        identities = {row["source_world_episode_id"] for row in rows}
        hashes = {row["source_sha256"] for row in rows}
        groups = {row["scene_group_id"] for row in rows}
        sizes = [row["frame_count"] for row in rows if "frame_count" in row]
        frames = sum(sizes) if sizes else lineage["frames"]
        if (len(groups) != original["groups"] or frames != original["frames"] or frames != lineage["frames"]
                or (sizes and len(sizes) != len(rows))):
            raise ValueError("Independent groups or frames differ from the original export")
        sets[split] = identities, hashes, groups
        counts[split] = {"groups": len(groups), "frames": frames}
    if any(left & right for left, right in zip(sets["train"], sets["validation"], strict=True)):
        raise ValueError("A source, byte identity or scene group crosses research splits")
    return counts


def compact_cases(files, base, original, job):
    validation, = json.loads(files[f"{base}/validation-evaluation.json"])
    fresh = json.loads(files[f"{base}/fresh-evaluation.json"])
    if (fresh["checkpoint_sha256"] != job["weights"]["sha256"] or validation["checkpoint"] != job["checkpoint"]
            or fresh["results"] != original["fresh"]["results"] or validation not in original["validation"]):
        raise ValueError("The compact evaluation variants differ from the original records")
    expected = [(row["seed"], row["target_marker"], row["sliding_friction"]) for row in json.loads(files[FRESH])["resets"]]
    actual = [(row["seed"], row["target_marker"], row["recipe"]["cube_sliding_friction"]) for row in fresh["results"]]
    training = json.loads(files[f"{base}/train-lineage.json"])["episodes"]
    validation_export = json.loads(files[f"{base}/validation-lineage.json"])["episodes"]
    seed = lambda row: int(row["source_world_episode_id"].rsplit("-", 1)[1])
    if (actual != expected or fresh["evaluation_role"] != "fresh-test" or validation["evaluation_role"] != "validation"
            or [row["seed"] for row in validation["resets"]] != [seed(row) for row in validation_export]
            or {row[0] for row in expected} & {seed(row) for row in (*training, *validation_export)}
            or any(row["steps"] != 510 or row["policy_reset_seed"] != row["seed"] for row in fresh["results"])):
        raise ValueError("A compact reset differs from its isolated reserved or validation world")
    return validation, fresh


def evaluations(files, base, job, scope):
    evaluation = job["evaluation"]
    original = json.loads(files[f"{base}/evaluation.json"])
    if original["checkpoint"] != job["checkpoint"]:
        raise ValueError("Motor evaluation belongs to a different checkpoint")
    if scope == "compact":
        validation, fresh = compact_cases(files, base, original, job)
        return [(name, evaluate_cases(rows, evaluation[part]["resets"], evaluation[part]["contact_valid_placements"]), loss)
                for part, name, rows, loss in (("validation", "matched validation", validation["resets"], validation["offline_validation_mean_loss"]),
                                               ("fresh", "reserved fresh", fresh["results"], None))]
    historic = scope == "historical"
    if historic and original["checkpoint_sha256"] != job["weights"]["sha256"]:
        raise ValueError("The original checkpoint digest differs from its historical score")
    rows = original["results" if historic else "fitting_cases"]
    scored = evaluate_cases(rows, evaluation["resets"], evaluation["successes"], historical=historic)
    return [("historical ten-world" if historic else "matched fitting", scored, evaluation["offline_validation_mean_loss"])]


def calculate(files):
    jobs = index_jobs(files[INVENTORY])
    teacher = json.loads(files[TEACHER])
    aligned = [{"seed": row["seed"], **row["aligned-sample-v2"]} for row in teacher["cases"]]
    teacher_control = evaluate_cases(aligned, 5, 5)
    results = []
    for name, label, scope in MOTOR_JOBS:
        job = jobs[name]
        base = f"jobs/{name}"
        for member, receipt in motor_artifacts(name, job, scope).items():
            bound_contents(files[member], receipt)
        counts = dataset_counts(files, base, job)
        for cohort, score, loss in evaluations(files, base, job, scope):
            results.append({"model": label, "cohort": cohort, **score, "dataset": counts, "job": name,
                            "offline_validation_mean_loss": loss, "updates_this_job": job["optimizer_updates"],
                            "checkpoint_sha256": job["weights"]["sha256"]})
    return {"schema": SCHEMA, "scope": "Preserved simulation evaluator samples and official export lineage",
            "teacher_contact_control": teacher_control, "evaluations": results,
            "optimizer_updates_performed": 0, "simulator_steps_performed": 0, "physical_attempts_performed": 0}


def markdown(report):
    lines = ["# Reproduced motor evaluation results", "", report["scope"] + ".", "",
             "The five matched teacher replays each satisfy verified contact, lift and released containment. "
             "The historical placement score keeps its original endpoint and has no recorded failure stage. "
             "ACT has no fresh result on the compact split. The SmolVLA offline loss comes from validation.", "",
             "| Model | Cohort | Placements | First failure | Updates in job | Offline validation loss |",
             "| --- | --- | ---: | --- | ---: | ---: |"]
    for row in report["evaluations"]:
        stages = ", ".join(f"{name} ({count})" for name, count in row["first_failure_stages"].items())
        loss = row["offline_validation_mean_loss"]
        shown = f"{loss:.6f}" if loss is not None else "—"
        lines.append(f"| {row['model']} | {row['cohort']} | {row['successes']}/{row['resets']} | "
                     f"{stages} | {row['updates_this_job']} | {shown} |")
    return "\n".join([*lines, ""])


def build():
    files = {INVENTORY: (PILOT / INVENTORY).read_bytes(), TEACHER: (ROOT / "evidence/resume" / TEACHER).read_bytes(),
             FRESH: (PILOT / FRESH).read_bytes()}
    jobs = index_jobs(files[INVENTORY])
    for name, _, scope in MOTOR_JOBS:
        for member, receipt in motor_artifacts(name, jobs[name], scope).items():
            files[member] = bound_contents(Path(receipt["path"]).read_bytes(), receipt)
    for filename in ("act-precontact-teacher-physics.json", "act-matched-render-precontact-control.json",
                     "act-stock-static-raster.json", "smolvla-validation-approach-complete.json",
                     "smolvla-cpu-chunk-stage-timing.json", "smolvla-compact-training-exposure.json"):
        files[f"diagnostics/{filename}"] = (PILOT / filename).read_bytes()
    report = calculate(files)
    files["results.json"] = json_bytes(report) + b"\n"
    files["results.md"] = markdown(report).encode()
    files["README.md"] = ("# Retained Smart Swarm motor runs\n\nUse the project simulation Python with PYTHONPATH=src to run "
                           "scripts/archive_motor_results.py --archive PATH. The verifier regenerates the table "
                           "from saved feedback and checks the archived members. Add --check-originals on Homebrain "
                           "to rehash the referenced weights and original receipts. Original RGB datasets and large "
                           "checkpoint weights remain in their recorded locations.\n").encode()
    return files


def original_artifacts(jobs):
    for job in jobs.values():
        weight = job["weights"]
        with Path(weight["path"]).open("rb") as stream:
            if sha256_stream(stream) != weight["sha256"]:
                raise ValueError("An original checkpoint weight object changed")
        stored = [job["training_receipt"], job["step_receipt"], job["evaluation"]["receipt"],
                  *job["configuration_and_processors"].values(), *(part["lineage"] for part in job["dataset"]["splits"].values())]
        stored.extend(job["evaluation"][key]["receipt"] for key in ("validation", "fresh") if key in job["evaluation"])
        if any(sha(Path(entry["path"])) != entry["sha256"] for entry in stored):
            raise ValueError("An original training, export, processor or evaluation receipt changed")


def verify(path, *, originals=False):
    with tarfile.open(path) as archive:
        binding = json.loads(read_member(archive, "index.json"))
        hashes = binding["sha256"]
        names = archive.getnames()
        if binding["schema"] != SCHEMA or len(names) != len(hashes) + 1 or set(names) != {"index.json", *hashes}:
            raise ValueError("The motor archive has unexpected or duplicate members")
        files = {name: read_member(archive, name) for name in hashes}
    if any(sha256(data).hexdigest() != hashes[name] for name, data in files.items()):
        raise ValueError("An archived motor artifact changed")
    report = calculate(files)
    if files["results.json"] != json_bytes(report) + b"\n" or files["results.md"] != markdown(report).encode():
        raise ValueError("An archived result table differs from the saved motor observations")
    if originals:
        original_artifacts(index_jobs(files[INVENTORY]))
    return {"archive_sha256": sha(path), "members_verified": len(files), "originals_verified": originals,
            "scored_learned_resets": sum(row["resets"] for row in report["evaluations"]), "table_reproduced": True}


def publish(path, files):
    if path.exists():
        raise ValueError("Choose a new immutable archive path")
    binding = {"schema": SCHEMA, "sha256": {name: sha256(value).hexdigest() for name, value in files.items()}}
    with NamedTemporaryFile(dir=path.parent, prefix=path.name + ".") as staged:
        with GzipFile(filename="", fileobj=staged, mode="wb", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                put(archive, "index.json", json_bytes(binding) + b"\n")
                for name, value in sorted(files.items()):
                    put(archive, name, value)
        staged.flush()
        verify(Path(staged.name))
        path.hardlink_to(staged.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--check-originals", action="store_true")
    parser.add_argument("--table", type=Path)
    args = parser.parse_args()
    try:
        if args.build:
            publish(args.archive, build())
        result = verify(args.archive, originals=args.check_originals)
        if args.table is not None:
            with tarfile.open(args.archive) as archive:
                table = read_member(archive, "results.md")
            if args.table.exists() and args.table.read_bytes() != table:
                raise ValueError("Choose a new table path to preserve the existing table")
            if not args.table.exists():
                with args.table.open("xb") as output:
                    output.write(table)
        print(json.dumps(result, indent=2))
    except (OSError, ValueError, KeyError, tarfile.TarError) as error:
        parser.exit(2, f"Motor result archive withheld. {error}\n")


if __name__ == "__main__":
    main()
