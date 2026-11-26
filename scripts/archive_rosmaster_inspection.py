"""Reproduce the bounded moving-camera comparison from its admitted RGB and receipts."""

import argparse
from gzip import GzipFile
from hashlib import sha256
from io import BytesIO
import json
from math import isfinite
from pathlib import Path
import tarfile
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
from PIL import Image

from archive_primary_release import INDEX, put, read_member, stash
from smart_swarm.cli import ROOT
from smart_swarm.episode import json_bytes, load, sha
from smart_swarm.inspection_comparison import ARMS, follow_up, score_pair, summarize
from smart_swarm.inspection_triangulation import locate
from smart_swarm.inspection_view import interpret


PILOT = ROOT / "evidence/learning-pilot-v3"
REPORT = PILOT / "rosmaster-bounded-comparison.json"
OUTPUT = ROOT.parent.parent / "work/active-inspection-comparison-mobile-v3"
SCHEMA = "smart-swarm-mobile-inspection-results-v1"
FRAMES = ("carrier.png", "surface.png")
HEADS = ("events_only", "events_plus_reveal")
LABELS = {"optical": "RGB localizer", "events_only": "Events only", "events_plus_reveal": "Events + reveal"}
BRANCH_FIELDS = ("start_ns", "stop_ns", "motion", "checked_physics_steps", "external_contact_pairs", "self_contact_pairs")


def stash_case(files, root, attempt):
    group = attempt["source_group"]
    if not group.replace("-", "").isalnum():
        raise ValueError("Invalid source identity in the original comparison")
    path = root / group
    member = f"cases/{group}"
    stash(files, f"{member}/receipt.json", path / "receipt.json")
    if attempt["status"] == "sealed":
        policy = load(path / "policy.json")
        for name in ("policy.json", "evaluator/result.json", "pre-inspection-decision.json", "pre-inspection/manifest.json"):
            stash(files, f"{member}/{name}", path / name)
        for arm in ARMS:
            stash(files, f"{member}/{arm}/worker-result.json", path / arm / "worker-result.json")
            for name, row in zip(FRAMES, policy[arm]["samples"], strict=True):
                if row["frame"] != name:
                    raise ValueError("The archived branch must contain the two registered images")
                stash(files, f"{member}/{arm}/{name}", path / arm / name, row["sha256"])


def collect(root):
    from audit_rosmaster_inspection_results import audit

    report = load(REPORT)
    if audit(root) != report:
        raise ValueError("The original ROSMASTER recording workspace no longer reproduces the saved result")
    files = {}
    plan = load(root / "design.json")
    stash(files, "configuration/design.json", root / "design.json", report["design_sha256"])
    comparison = root / "comparison.json"
    stash(files, "configuration/comparison.json", comparison, report["comparison_sha256"])
    for key, name in (("profiles_sha256", "primary-profiles-frozen.json"), ("waypoint_sha256", "rosmaster-inspection-west-waypoint.json"),
                      ("encoding_proof_sha256", "rosmaster-shared-frozen-encoding-verification.json")):
        stash(files, f"configuration/{name}", PILOT / name, plan[key])
    for name, digest in plan["implementation_sha256"].items():
        stash(files, f"implementation/{name}", ROOT / name, digest)
    for name in ("scripts/archive_rosmaster_inspection.py", "scripts/audit_rosmaster_inspection_results.py",
                 "scripts/audit_primary_pairs.py", "src/smart_swarm/inspection_view.py", "src/smart_swarm/vision.py"):
        stash(files, f"implementation/{name}", ROOT / name)
    for attempt in load(comparison)["attempts"]:
        stash_case(files, root, attempt)
    stash(files, "analysis/original-report.json", REPORT)
    return files, report


def recover_branch(files, base, policy, start, plan):
    views = []
    for name, sample, offset in zip(FRAMES, policy["samples"], plan["exposure_offsets_ns"], strict=True):
        blob = files[f"{base}/{name}"]
        if sample["frame"] != name or sha256(blob).hexdigest() != sample["sha256"] or sample["at_ns"] != start + offset:
            raise ValueError("A post-decision camera exposure, schedule or hash changed")
        with Image.open(BytesIO(blob)) as image:
            if image.mode != "RGB" or list(image.size) != plan["post_inspection_resolution"] or image.format != "PNG":
                raise ValueError("The replay needs its original 640 by 480 RGB exposure")
            pixels = np.asarray(image).copy()
        calibration = sample["calibration"]
        if json_bytes(interpret(pixels, calibration)) != json_bytes(sample["readout"]):
            raise ValueError("A saved single-frame cue readout no longer reproduces")
        views.append((pixels, calibration))
    result = locate(views)
    if result != policy["post_inspection_decision"] or policy["start_ns"] != start or policy["stop_ns"] != start + plan["exposure_offsets_ns"][-1]:
        raise ValueError("The matched optical readout or branch window changed")
    return result


def check_motion(worker, policy, evaluator, arm):
    branch, endpoint = worker["branch"], evaluator["truth"][arm]
    if (worker["status"] != "recorded" or worker["reset_sha256"] != evaluator["snapshot_sha256"]
            or any(branch[key] != policy[key] for key in BRANCH_FIELDS)
            or any(worker["endpoints"][key] != endpoint[key] for key in ("before", "after"))
            or branch["evaluator"]["cube_displacement_m"] != endpoint["target_displacement_m"]
            or branch["evaluator"]["carrier_displacement_m"] != endpoint["carrier_displacement_m"]
            or branch["external_contact_pairs"] or branch["self_contact_pairs"]):
        raise ValueError("A worker restore, motion receipt or evaluated physical endpoint changed")
    if any({k: v for k, v in actual.items() if k != "motion"} != expected
           for actual, expected in zip(policy["samples"], branch["policy_observations"], strict=True)):
        raise ValueError("A pose-bound camera sample differs from its original worker receipt")


def score_case(files, attempt, plan):
    prefix = f"cases/{attempt['source_group']}"
    policy_path = f"{prefix}/policy.json"
    policy = json.loads(files[policy_path])
    prediction = files[f"{prefix}/pre-inspection-decision.json"]
    early = json.loads(prediction)
    manifest = files[f"{prefix}/pre-inspection/manifest.json"]
    if (early["recording_manifest_sha256"] != sha256(manifest).hexdigest()
            or policy["pre_inspection_decision_sha256"] != sha256(prediction).hexdigest() or set(early["learned"]) != set(HEADS)):
        raise ValueError("The archived pre-inspection decision no longer matches its input commitment")
    boundary = int(json.loads(manifest)["decision_boundary"]["monotonic_ns"])
    for arm in ARMS:
        recover_branch(files, f"{prefix}/{arm}", policy[arm], boundary, plan)
    predictions = {head: {arm: follow_up(early["learned"][head]["decision"], policy[arm]) for arm in ARMS}
                   for head in HEADS}
    if predictions != policy["methods"]:
        raise ValueError("The branches do not follow the original learned inspection policy")
    evaluator = json.loads(files[f"{prefix}/evaluator/result.json"])
    if attempt["policy_sha256"] != sha256(files[policy_path]).hexdigest() or attempt["policy_sha256"] != evaluator["sealed_policy_sha256"]:
        raise ValueError("Evaluator labels belong to another sealed policy")
    for arm in ARMS:
        worker = json.loads(files[f"{prefix}/{arm}/worker-result.json"])
        check_motion(worker, policy[arm], evaluator, arm)
    score = score_pair(evaluator["evaluation"]["location"], policy, evaluator["truth"])
    if not score["paired_scoring_valid"] or score != attempt["evaluation"] or score != evaluator["evaluation"]:
        raise ValueError("The archived outcomes differ from the matched scored pair")
    return score


def verify_configuration(hashes, plan, comparison, report):
    design = hashes["configuration/design.json"]
    if (design != comparison["design_sha256"] or design != report["design_sha256"]
            or hashes["configuration/comparison.json"] != report["comparison_sha256"]
            or hashes["configuration/primary-profiles-frozen.json"] != plan["profiles_sha256"]
            or hashes["configuration/rosmaster-inspection-west-waypoint.json"] != plan["waypoint_sha256"]
            or hashes["configuration/rosmaster-shared-frozen-encoding-verification.json"] != plan["encoding_proof_sha256"]
            or comparison["physical_motion_requested"] or comparison["optimizer_updates"]
            or report["hardware_motion"] or report["optimizer_updates"] or report["status"] != comparison["status"]):
        raise ValueError("The archive does not bind the original simulation release")
    if any(hashes[f"implementation/{key}"] != value for key, value in plan["implementation_sha256"].items()):
        raise ValueError("A frozen implementation file changed in the archive")
    if sha(ROOT / "src/smart_swarm/inspection_triangulation.py") != plan["implementation_sha256"]["src/smart_swarm/inspection_triangulation.py"]:
        raise ValueError("Use the original triangulation implementation for optical replay")
    for name in ("inspection_view.py", "inspection_comparison.py", "vision.py"):
        relative = f"src/smart_swarm/{name}"
        if sha(ROOT / relative) != hashes[f"implementation/{relative}"]:
            raise ValueError(f"Use the original {name} implementation for optical replay")


def verify_attempts(files, hashes, plan, comparison, report):
    from audit_rosmaster_inspection_results import summarize_scope

    entries = comparison["attempts"]
    recipes = plan["sources"]
    if len({row["source_group"] for row in recipes}) != len(recipes):
        raise ValueError("The registered independent allocation contains repeated sources")
    for attempt, recipe in zip(entries, recipes[:len(entries)], strict=True):
        group = attempt["source_group"]
        name = f"cases/{group}/receipt.json"
        if (json.loads(files[name]) != attempt or hashes[name] != report["receipt_sha256"][group]
                or any(attempt[key] != value for key, value in recipe.items())
                or not isfinite(attempt["wall_seconds"]) or attempt["wall_seconds"] <= 0):
            raise ValueError("The original attempt order, provenance or charged time changed")
        if attempt["status"] == "sealed":
            score_case(files, attempt, plan)
        elif attempt["status"] not in ("interrupted", "failed") or "evaluation" in attempt:
            raise ValueError("The archive treats an unscored source as valid")
    scopes = {name: summarize_scope([row for row in recipes if name == "all" or row["family"] == name], entries)
              for name in ("all", "F1", "F6")}
    total_s = plan["charged_development_wall_seconds"] + sum(row["wall_seconds"] for row in entries)
    limit = plan["limits"]["cumulative_wall_seconds"]
    remaining = max(0., limit - total_s)
    if (scopes != report["scopes"] or total_s != report["charged_wall_seconds"]
            or report["wall_limit_seconds"] != limit or report["remaining_wall_seconds"] != remaining
            or comparison["remaining_wall_seconds"] != remaining
            or any(comparison[key] != value for key, value in summarize(len(recipes), entries).items())
            or [row["source_group"] for row in recipes[len(entries):]] != report["unattempted_sources"]):
        raise ValueError("The archive does not reproduce the original grouped result or allocation bounds")


def rows(report, family):
    scope = report["scopes"][family]
    return {"optical": scope["optical"], **scope["methods"]}


def paper_table(report):
    overall = report["scopes"]["all"]
    allocated = overall["optical"]["allocated"]
    attempted, valid = overall["attempted"], overall["valid_pairs"]
    lines = ["# Matched moving-camera inspection", "",
             "The moving and stationary branches used the same saved reset and the same two exposure times.",
             f"The original allocation had {allocated} independent scenes. Of {attempted} attempted scenes, {valid} were scored.",
             "", "| Scene set | Method | Moving correct | Stationary correct | Paired effect (pp) | 95% interval (pp) | All allocated bounds (pp) |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for family in ("all", "F1", "F6"):
        for method, row in rows(report, family).items():
            scan = row["branches"]["scan"]["correct"]
            control = row["branches"]["stationary"]["correct"]
            count = row["scored_pairs"]
            difference = 100 * row["paired_difference"]
            interval = " to ".join(f"{100 * value:+.1f}" for value in row["conservative_paired_95pct"])
            bounds = " to ".join(f"{100 * value:+.1f}" for value in row["allocation_difference_bounds"])
            lines.append(f"| {family.upper()} | {LABELS[method]} | {scan}/{count} | {control}/{count} | "
                         f"{difference:+.1f} | {interval} | {bounds} |")
    lines.extend(["", "The allocation bounds assign each missing pair any possible outcome.",
                  "The endpoint is simulated location. The learned methods include direct answers they issued before motion."])
    return "\n".join(lines) + "\n"


def paper_figure(report, output):
    from study_figures import plt

    families = ("all", "F1", "F6")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5), sharey=True, layout="constrained")
    series = ((axes[0], "optical", "#235a78", "o", 0.),
              (axes[1], "events_only", "#235a78", "o", -.11), (axes[1], "events_plus_reveal", "#a24a24", "s", .11))
    for axis, key, color, marker, offset in series:
        for index, family in enumerate(families):
            row = rows(report, family)[key]
            where = index + offset
            interval = [100 * value for value in row["conservative_paired_95pct"]]
            bounds = [100 * value for value in row["allocation_difference_bounds"]]
            axis.hlines(where, *interval, color=color, linewidth=1.1)
            axis.scatter(bounds, [where, where], facecolors="white", edgecolors=color, s=22, zorder=3)
            axis.scatter(100 * row["paired_difference"], where, s=25, color=color, marker=marker, zorder=4,
                         label=LABELS[key] if index == 0 else None)
    sizes = [rows(report, family)["optical"] for family in families]
    axes[0].set_yticks(range(3), [f"{family.upper()}\n{row['scored_pairs']} of {row['allocated']}"
                                 for family, row in zip(families, sizes, strict=True)])
    axes[0].set_ylim(2.4, -.4)
    axes[0].set_title("RGB localizer")
    axes[1].set_title("Learned inspection")
    axes[1].legend(loc="upper left", frameon=False, fontsize=8)
    for axis in axes:
        axis.set(xlim=(-105, 105), xlabel="Moving minus stationary correct (pp)")
        axis.axvline(0, color="#555555", linestyle="--", linewidth=.8)
        axis.grid(axis="x", alpha=.17)
        axis.spines[["top", "right", "left"]].set_visible(False)
    fig.suptitle("Matched simulated inspection\nLines show 95% intervals. Open points show full-allocation bounds.", fontsize=10)
    fig.savefig(output, dpi=220, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def read_verified(path):
    with tarfile.open(path, "r:gz") as archive:
        index = json.loads(read_member(archive, INDEX))
        hashes = index["contents_sha256"]
        names = archive.getnames()
        if index["schema"] != SCHEMA or set(names) != {INDEX, *hashes} or len(names) != len(hashes) + 1:
            raise ValueError("The inspection package has unexpected members")
        files = {key: read_member(archive, key) for key in hashes}
    if any(sha256(data).hexdigest() != hashes[name] for name, data in files.items()):
        raise ValueError("An inspection package member no longer matches its digest")
    return files, hashes


def verify(path):
    files, hashes = read_verified(path)
    plan = json.loads(files["configuration/design.json"])
    comparison = json.loads(files["configuration/comparison.json"])
    report = json.loads(files["analysis/original-report.json"])
    verify_configuration(hashes, plan, comparison, report)
    verify_attempts(files, hashes, plan, comparison, report)
    images = sum(name.endswith(".png") for name in files)
    if images != 4 * comparison["valid_pairs"] or paper_table(report).encode() != files["analysis/table.md"]:
        raise ValueError("The archived image count or generated table changed")
    with TemporaryDirectory(prefix="smart-swarm-inspection-figure-") as directory:
        figure = Path(directory) / "figure.pdf"
        paper_figure(report, figure)
        if figure.read_bytes() != files["analysis/figure.pdf"]:
            raise ValueError("The original report did not reproduce the paper figure")
    return {"archive": str(path), "sha256": sha(path), "members_verified": len(hashes),
            "allocated_sources": len(plan["sources"]), "attempted_sources": len(comparison["attempts"]),
            "paired_results_reproduced": comparison["valid_pairs"], "scope_rgb_exposures_replayed": images,
            "table_reproduced": True, "figure_reproduced": True, "original_worktree_recordings_required": False}


def publish(root, path):
    files, report = collect(root)
    files["analysis/table.md"] = paper_table(report).encode()
    with TemporaryDirectory(prefix="smart-swarm-inspection-figure-") as directory:
        figure = Path(directory) / "figure.pdf"
        paper_figure(report, figure)
        files["analysis/figure.pdf"] = figure.read_bytes()
    manifest = {"schema": SCHEMA, "contents_sha256": {name: sha256(value).hexdigest() for name, value in files.items()}}
    with NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with temporary.open("wb") as stream, GzipFile(fileobj=stream, filename="", mode="wb", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                put(archive, INDEX, json_bytes(manifest) + b"\n")
                for name, blob in sorted(files.items()):
                    put(archive, name, blob)
        verify(temporary)
        path.hardlink_to(temporary)
    finally:
        temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--root", type=Path, default=OUTPUT)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    if args.publish:
        if args.archive.exists():
            parser.error("Choose a new immutable inspection archive path")
        publish(args.root, args.archive)
    print(json.dumps(verify(args.archive), indent=2))


if __name__ == "__main__":
    main()
