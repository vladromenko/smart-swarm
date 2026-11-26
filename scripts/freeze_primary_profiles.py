#!/usr/bin/env python3
"""Package validation profiles before the first primary test read."""

import argparse
from pathlib import Path

from calibrate_primary_inspection import admitted_matrix, method_report
from capture_memory_validation import ALLOCATION, CONFIG, DEFAULT_OUTPUT, ROOT, validation_slots
from score_primary_validation import IDENTITY, source_hashes
from smart_swarm.episode import atomic, load, sha

PILOT = ROOT / "evidence/learning-pilot-v3"
KINDS = ("events_only", "events_plus_reveal")


def profiles(path, expected_kind, config, study):
    matrix = admitted_matrix(path)
    model = next(row for row in study["models"] if row["kind"] == expected_kind)
    if matrix.get("implementation_sha256") != source_hashes():
        raise ValueError(f"Comparison implementation changed after the validation replay for {expected_kind}")
    if (matrix["checkpoint_kind"] != expected_kind or matrix["checkpoint_sha256"] != model["checkpoint_sha256"]
            or model["checkpoint_sha256"] != sha(Path(model["checkpoint"]))
            or matrix["encoder_sha256"] != study["frozen_backbone_sha256"]):
        raise ValueError(f"Network identities changed for {expected_kind}")
    prior = Path(study["calibrated_particle_report"])
    if matrix["base_particle_report_sha256"] != sha(prior):
        raise ValueError("The particle reference training report changed")
    teams = config["team_sizes"]
    lookup = matrix["full_validation_fit"]
    if sorted(map(int, lookup)) != teams:
        raise ValueError("The validation fit omitted an allocated team size")
    marker_diagnostic = "marker_support_diagnostic" in matrix
    methods = [*config["primary_methods"], IDENTITY] if marker_diagnostic else config["primary_methods"]
    for size in teams:
        item = lookup[str(size)]
        sources = [group for group in matrix["outcomes"] if group["team_size"] == size]
        if sorted(group["group"] for group in sources) != matrix["cohort"]["source_groups"]:
            raise ValueError("Validation thresholds require identical source coverage at every team size")
        item["decision_thresholds"] = {method: method_report(sources, method, config)["selected_validation_threshold"]
                                       for method in methods}
    return {"fit_matrix_sha256": sha(path), "checkpoint_sha256": model["checkpoint_sha256"],
            "checkpoint": model["checkpoint"], "mount_and_receiver_profiles": lookup,
            "marker_support_diagnostic": marker_diagnostic,
            "particle_reference_sha256": matrix["base_particle_report_sha256"]}, matrix["cohort"]


def build(paths, *, draft):
    config = load(CONFIG)
    study = load(PILOT / "story-learning.json")
    fits = [profiles(path, kind, config, study) for path, kind in zip(paths, KINDS, strict=True)]
    cohort, second = [value[1] for value in fits]
    if cohort != second:
        raise ValueError("The two frozen heads were compared on different validation data")
    if fits[0][0]["marker_support_diagnostic"] != fits[1][0]["marker_support_diagnostic"]:
        raise ValueError("The two frozen heads require the same marker diagnostic protocol")
    allocated = {slot["source_group"] for slot in validation_slots()}
    fitted = set(cohort["source_groups"])
    if not draft and fitted != allocated:
        raise ValueError(f"Primary freeze requires the reserved validation allocation ({len(fitted)}/{len(allocated)})")
    return {"version": "smart-swarm-primary-frozen-profiles-v1", "status": "draft" if draft else "frozen_for_primary_test",
            "config_sha256": sha(CONFIG), "encoder_sha256": study["frozen_backbone_sha256"],
            "source_allocation_sha256": sha(ALLOCATION),
            "validation_cohort": cohort, "validation_count": len(fitted), "allocated_validation_count": len(allocated),
            "fits": {kind: result[0] for kind, result in zip(KINDS, fits, strict=True)}, "implementation_sha256": source_hashes()}


def verify_calibration_cohort(cohort, slots):
    allowed = {(group, size, variant) for group, slot in slots.items() if group in cohort["source_groups"]
               for size in slot["paired_team_sizes"] for variant in slot["capture_variants"]}
    seen = set()
    for location, digest in cohort["manifests"].items():
        root = Path(location).resolve()
        if not root.is_relative_to(DEFAULT_OUTPUT.resolve()):
            raise ValueError("A calibration manifest is outside the reserved validation directory")
        path = root / "manifest.json"
        if sha(path) != digest:
            raise ValueError("A frozen validation source manifest changed")
        manifest = load(path)
        name = manifest["scene_group_id"]
        record = name, len(manifest["policy_cameras"]), manifest["evidence_variant"]
        if (record not in allowed or record in seen or manifest["lineage"]["split"] != "validation"
                or manifest["family"] != slots[name]["family"]
                or manifest["simulation_recipe_seed"] not in slots[name]["seed_candidates"]):
            raise ValueError("Unexpected calibration team or variant in the lineage")
        seen.add(record)
    if seen != allowed:
        raise ValueError("The frozen validation cohort is missing a team or variant")


def require_profiles(path, *, rehearsal=False):
    profile = load(path)
    if profile["status"] != "frozen_for_primary_test" and not (rehearsal and profile["status"] == "draft"):
        raise ValueError("Primary test requires released profiles. Drafts are available only for validation rehearsal.")
    if profile["version"] != "smart-swarm-primary-frozen-profiles-v1":
        raise ValueError("Unknown frozen profile version")
    if (profile["config_sha256"] != sha(CONFIG) or profile["source_allocation_sha256"] != sha(ALLOCATION)
            or profile["implementation_sha256"] != source_hashes()):
        raise ValueError("The study configuration, allocation or implementation changed after profiling")
    cohort = profile["validation_cohort"]
    slots = {slot["source_group"]: slot for slot in validation_slots()}
    expected = set(slots)
    included = set(cohort["source_groups"])
    if (not included or not included <= expected or (not rehearsal and included != expected)
            or len(included) != profile["validation_count"]):
        raise ValueError("The frozen profile has invalid validation source coverage")
    verify_calibration_cohort(cohort, slots)
    if set(profile["fits"]) != set(KINDS):
        raise ValueError("Both original learned models must remain in the comparison")
    for fit in profile["fits"].values():
        if sha(Path(fit["checkpoint"])) != fit["checkpoint_sha256"]:
            raise ValueError("A frozen model checkpoint changed")
    return profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--reveal", type=Path, required=True)
    parser.add_argument("--draft", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build([args.events, args.reveal], draft=args.draft)
    if args.output.exists() and load(args.output) != result:
        raise ValueError("Existing calibration differs. Use a new versioned output path.")
    if not args.output.exists():
        atomic(args.output, result)
    print(result["status"], args.output)


if __name__ == "__main__":
    main()
