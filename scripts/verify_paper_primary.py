"""Recompute the original paper's primary scores from the frozen result archives."""

from hashlib import sha256
import json
import tarfile

from archive_primary_release import INDEX, KINDS, read_member
from audit_primary_inspection_thresholds import METADATA, index_sources, sealed_sources
from compare_primary_supervision import METHODS, TEAMS, describe, team_comparison
from export_primary_result_table import controls, markdown
from smart_swarm.contract import json_bytes
from study_figures import paired_supervision


def archived_cohort(archive, split):
    binding = json.loads(read_member(archive, INDEX))
    identity, sources = index_sources(archive, binding, split)
    groups, digests = {}, {}
    for kind in KINDS:
        scoped = sealed_sources(archive, identity, sources, split, kind)
        groups[kind] = {size: sorted(scoped[size], key=lambda row: row["group"]) for size in TEAMS}
        digests[kind] = sha256(read_member(archive, f"scores/{METADATA[split][2]}-{kind}.json")).hexdigest()
    return identity, groups, digests


def verify_table(archive, identity, groups, digests, config):
    methods = config["primary_methods"]
    result = {kind: {str(size): {method: describe(team, method, config) for method in methods}
                     for size, team in by_size.items()} for kind, by_size in groups.items()}
    checks = {kind: {str(size): controls(team, config) for size, team in by_size.items()}
              for kind, by_size in groups.items()}
    report = {"schema": "smart-swarm-primary-method-table-v1", "scope": identity["scope"],
              "independent_source_groups": len(identity["source_groups"]), "teams": list(TEAMS),
              "recorded_variants_across_teams": len(identity["manifests"]),
              "profiles_sha256": identity["profiles_sha256"],
              "primary_test_release_sha256": identity["primary_test_release_sha256"],
              "scored_receipts_sha256": digests, "registered_method_order": methods,
              "metrics_by_model_and_team": result, "recorded_control_checks_by_model_and_team": checks}
    if report != json.loads(read_member(archive, "analysis/method-table.json")):
        raise ValueError("The archived forecasts and scored labels did not reproduce the method table")
    if markdown(report).encode() != read_member(archive, "analysis/method-table.md"):
        raise ValueError("The regenerated primary table differs from the original rendered table")


def verify_figure(archive, identity, groups, digests, config, working):
    original = json.loads(read_member(archive, "analysis/paired-supervision.json"))
    recomputed = {method: {str(size): team_comparison({kind: groups[kind][size] for kind in KINDS}, method, config)
                           for size in TEAMS} for method in METHODS}
    if (recomputed != original["methods"] or digests != original["score_sha256"]
            or original["profiles_sha256"] != identity["profiles_sha256"]
            or original["primary_test_release_sha256"] != identity["primary_test_release_sha256"]):
        raise ValueError("The paired comparison differs from the frozen source-level results")
    source, figure = working / "paired.json", working / "paired.pdf"
    source.write_bytes(json_bytes({**original, "methods": recomputed}) + b"\n")
    paired_supervision(source, figure)
    if figure.read_bytes() != read_member(archive, "analysis/paired-supervision.pdf"):
        raise ValueError("The regenerated primary decision-cost figure differs from the original")


def verify_primary(path, split, config, working):
    with tarfile.open(path) as archive:
        identity, groups, digests = archived_cohort(archive, split)
        verify_table(archive, identity, groups, digests, config)
        verify_figure(archive, identity, groups, digests, config, working)
    return {"independent_source_groups": len(identity["source_groups"]),
            "recorded_variants_across_teams": len(identity["manifests"]),
            "table_reproduced": True, "figure_reproduced": True,
            "profile_sha256": identity["profiles_sha256"], "source_groups": identity["source_groups"]}
