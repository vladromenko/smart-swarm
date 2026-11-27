"""Compare saved pre-reveal forecasts on the native development corpus."""

from collections import Counter
import math
from pathlib import Path

from .contract import read_ns
from .episode import load, rows, sha
from .evidence_comparison import ERROR_COST, INSPECTION_COST, decision_cost, grouped_mean, nll
from .particle_memory import LOCATIONS
from .physical_evidence_dataset import DECISION_FILE, publish
from .physical_inspection_controls import control_case, inspection_controls
from .physical_learned import HEADS, METHOD, PROFILE, online_failure_fields
from .physical_witness_score import diagnose, summarize as summarize_witnesses

RULE = "rgb_rule"
METHODS = (RULE, *HEADS)
INSPECT = "request_inspection"
RULE_THRESHOLD = .75


def checked_forecast(raw, threshold):
    belief, decision = raw["belief"], raw["decision"]
    if (set(belief) != set(LOCATIONS)
            or any(type(value) not in (float, int) or not 0 <= value <= 1 for value in belief.values())
            or not math.isclose(sum(belief.values()), 1., rel_tol=0., abs_tol=1e-8)):
        raise ValueError("A saved physical forecast has an invalid terminal distribution")
    winner = max(belief, key=belief.get)
    if decision != (winner if belief[winner] >= threshold else INSPECT):
        raise ValueError("A saved physical choice differs from its fixed inspection threshold")
    return {"belief": belief, "decision": decision}


def saved_decision(record):
    root = Path(record["path"])
    manifest_path = root / "manifest.json"
    if sha(manifest_path) != record["manifest_sha256"]:
        raise ValueError("The physical manifest changed after corpus admission")
    manifest = load(manifest_path)
    path = root / DECISION_FILE
    if sha(path) != manifest["sidecar_sha256"][DECISION_FILE]:
        raise ValueError("The admitted pre-reveal prediction changed")
    decision, = rows(path)
    if (decision["at"] != manifest["decision_boundary"]
            or decision["query_all"]["decision"] != record["pre_inspection_decision"]
            or decision.get("learned_evidence") != record.get("learned_evidence")):
        raise ValueError("The saved physical prediction differs from its admitted record")
    return decision


def recorded_forecast(record, decision):
    replay = record.get("blind_offline_forecast")
    return replay if replay is not None else decision.get("learned_evidence") or {"status": "historical_rule_only"}


def scoring_case(record, decision):
    terminal = record["labels"]["terminal"]
    if record["split"] != "development" or (terminal is not None and terminal not in LOCATIONS):
        raise ValueError("The physical pilot comparison requires development stories and verified outcomes")
    learned = recorded_forecast(record, decision)
    forecasts = {RULE: checked_forecast(decision["query_all"], RULE_THRESHOLD)}
    if learned["status"] == "scored":
        receipt = learned["receipt"]
        if receipt["profiles_sha256"] != sha(PROFILE) or receipt["policy"] != METHOD:
            raise ValueError("Physical pilot forecasts must share the same frozen profile")
        forecasts.update({kind: checked_forecast(receipt["heads"][kind], receipt["heads"][kind]["threshold"])
                          for kind in HEADS})
    reason = record.get("blind_offline_forecast_error") or learned.get("reason")
    online_failure = online_failure_fields(decision.get("learned_evidence") or {})
    score = {**{key: record[key] for key in ("story_id", "scene_group_id", "family", "split")},
            "terminal": terminal, "learned_forecast_status": learned["status"],
            "learned_forecast_mode": learned.get("mode", "online") if learned["status"] == "scored" else None,
            "condition_status": record.get("condition_audit", {}).get("status", "unverified"),
            "learned_forecast_reason": reason, "forecasts": forecasts, **online_failure}
    candidate = control_case(record, score)
    return {**score, "inspection_control_case": candidate} if candidate is not None else score


def native_scoring_case(record):
    decision = saved_decision(record)
    result = scoring_case(record, decision)
    if result["learned_forecast_status"] == "scored":
        receipt = recorded_forecast(record, decision)["receipt"]
        try:
            manifest = load(Path(record["path"]) / "manifest.json")
            query = read_ns(manifest["query_boundary"]["monotonic_ns"])
            witness = diagnose(receipt, record["labels"], query)
        except (OSError, ValueError, KeyError, TypeError) as error:
            witness = {"status": "wire_unverified", "reason": str(error)}
        result["witness_diagnostics"] = witness
    return result


def metrics(rows):
    if not rows:
        return None
    answered = sum(row["decision"] != INSPECT for row in rows)
    correct = sum(row["decision"] == row["terminal"] for row in rows)
    return {"histories": len(rows), "scene_groups": len({row["scene_group_id"] for row in rows}),
            "answers": answered, "correct_answers": correct, "wrong_answers": answered - correct,
            "inspections": len(rows) - answered,
            "group_weighted_answer_coverage": grouped_mean(rows, lambda row: row["decision"] != INSPECT),
            "group_weighted_correct_answer_rate": grouped_mean(rows, lambda row: row["decision"] == row["terminal"]),
            "group_weighted_cost": decision_cost(rows), "group_weighted_nll": nll(rows),
            "group_weighted_brier": grouped_mean(rows, lambda row: sum(
                (prob - (slot == row["terminal"])) ** 2 for slot, prob in row["belief"].items()))}


def matched_comparison(cases):
    paired = [row for row in cases if set(row["forecasts"]) == set(METHODS)]
    labeled = [row for row in paired if row["terminal"] is not None]
    methods = {name: metrics([{**row, **row["forecasts"][name]} for row in labeled]) for name in METHODS}
    f6 = [row for row in paired if row["family"] == "F6"]
    observed = [row for row in labeled if row["condition_status"] == "observed"]
    outcomes = ("group_weighted_cost", "group_weighted_nll", "group_weighted_brier", "group_weighted_correct_answer_rate")
    delta = {key: methods[HEADS[1]][key] - methods[HEADS[0]][key] for key in outcomes} if labeled else None
    return {"labeled_matched_histories": len(labeled),
            "labeled_matched_scene_groups": len({row["scene_group_id"] for row in labeled}),
            "labeled_histories_by_family": dict(Counter(row["family"] for row in labeled)),
            "labeled_observed_conditions_by_family": dict(Counter(row["family"] for row in observed)),
            "metrics_by_method": methods, "events_plus_reveal_minus_events_only": delta,
            "f6_matched_histories": len(f6), "f6_matched_scene_groups": len({row["scene_group_id"] for row in f6}),
            "f6_inspection_counts": {name: sum(row["forecasts"][name]["decision"] == INSPECT for row in f6)
                                     for name in METHODS}}


def score_corpus(corpus):
    cases = [native_scoring_case(record) for record in corpus["records"]]
    labels = sum(row["terminal"] is not None for row in cases)
    comparison = matched_comparison(cases)
    by_mode = {mode: matched_comparison([row for row in cases if row["learned_forecast_mode"] == mode])
               for mode in ("online", "blind_offline")}
    if not cases:
        status = "no_admitted_histories"
    elif not labels:
        status = "no_verified_terminal_labels"
    elif not comparison["labeled_matched_histories"]:
        status = "no_labeled_matches"
    else:
        status = "scored_development"
    manifest = corpus["manifest"]
    controls = {"inspection_controls": inspection_controls(cases, METHODS)} if any(
        row.get("inspection_control_case") is not None for row in cases) else {}
    return {"schema": "smart-swarm-physical-evidence-development-score-v1", "status": status,
            "scope": "descriptive physical development scoring of online and blind offline forecasts",
            "uses_post_reveal_labels": True, "inference_recomputed": False, "thresholds_refit": False,
            "f6_scope": "inspection tally for physical withholding cards",
            "profile_sha256": sha(PROFILE), "corpus_manifest": manifest,
            "corpus_manifest_sha256": sha(Path(manifest)) if manifest is not None else None,
            "planned_histories": corpus["planned"], "attempted_histories": corpus["attempted"],
            "admitted_histories": len(cases), "admitted_scene_groups": corpus["independent_source_groups"],
            "terminal_outcomes_available": labels, "unknown_terminal_outcomes": len(cases) - labels,
            "learned_forecast_statuses": dict(Counter(row["learned_forecast_status"] for row in cases)),
            "condition_status_counts": corpus["condition_status_counts"],
            "capture_exclusion_counts": corpus["exclusion_counts"],
            "decision_costs": {"wrong_answer": ERROR_COST, "inspection": INSPECTION_COST},
            "matched_comparison": comparison, "matched_comparison_by_forecast_mode": by_mode, "cases": cases,
            **summarize_witnesses(cases), **controls}


def score_current(config):
    return score_corpus(publish(config))
