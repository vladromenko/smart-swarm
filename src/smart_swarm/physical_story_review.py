"""Summarize closed recording blocks from the verified development corpora."""

from collections import Counter
from pathlib import Path

from .episode import atomic, load
from .physical_evidence_dataset import describe, publish as evidence_corpus
from .physical_evidence_score import INSPECT, METHODS, RULE, native_scoring_case
from .physical_inspection_controls import inspection_controls
from .physical_motor_dataset import publish as motor_corpus
from .physical_paper import publish_report
from .physical_story_plan import prepare

TERMINAL = {"complete", "incomplete", "rejected"}


def closed_card(path):
    if not path.is_file():
        return False
    try:
        return load(path)["status"] in TERMINAL
    except (OSError, ValueError, KeyError, TypeError):
        return True  # The corpus inventory records this attempted card as rejected.


def ready_scopes(plan, workspace):
    blocks = range(1, plan["blocks"] + 1)
    by_block = {number: [card for card in plan["cards"] if card["block"] == number] for number in blocks}
    ready = {number: cards for number, cards in by_block.items() if cards and all(
        closed_card(workspace / card["story_id"] / "trial.json") for card in cards)}
    scopes = {f"block-{number:02}": cards for number, cards in ready.items()}
    if len(ready) == plan["blocks"]:
        scopes["all-24"] = plan["cards"]
    next_cards = next((cards for number, cards in by_block.items() if number not in ready), [])
    return scopes, next_cards


def index_records(stories, motors):
    return {"story": {row["story_id"]: row for row in stories["records"]},
            "exclusion": {row["story_id"]: row for row in stories["exclusions"]},
            "prefix": {row["story"]: row for row in motors["prefixes"]},
            "motor_exclusion": {row["story"]: row for row in motors["corpus_rejections"]}}


def motor_coverage(identifier, index):
    result = index["prefix"].get(identifier)
    if result is None:
        return index["motor_exclusion"].get(identifier, {"status": "not_reviewed", "reason": "No native motor intake"})
    return {"status": "accepted", "sample_count": result["sample_count"], "scene_group_id": result["scene_group_id"]}


def camera_coverage(record):
    manifest = load(Path(record["path"]) / "manifest.json")
    return {camera: record["camera_packets"][agent] for agent, cameras in manifest["policy_cameras"].items()
            for camera in cameras}


def review_card(card, index):
    identifier = card["story_id"]
    result = {"story_id": identifier, "assigned_family": card["family"], "scene_group_id": card["source_group_id"],
              "motor_prefix": motor_coverage(identifier, index)}
    record = index["story"].get(identifier)
    if record is None:
        missing = index["exclusion"].get(identifier, {"status": "unstarted", "reason": "No recorded attempt"})
        return {**result, "capture_status": missing["status"], "reason": missing["reason"]}
    score = native_scoring_case(record)
    witness = {"witness_diagnostics": score["witness_diagnostics"]} if "witness_diagnostics" in score else {}
    controls = {"inspection_control_case": score["inspection_control_case"]} if "inspection_control_case" in score else {}
    online = {"online_admission_failure": score["online_admission_failure"]} if "online_admission_failure" in score else {}
    return {**result, "capture_status": "complete", "condition": record["condition_audit"],
            "cameras": camera_coverage(record), "labels": record["labels"],
            "observed_placement": record["attended_placement_outcomes"], "assistance": record["assistance_by_phase"],
            "forecasts": {method: prediction["decision"] for method, prediction in score["forecasts"].items()},
            "learned_forecast_mode": score["learned_forecast_mode"], "learned_forecast_status": score["learned_forecast_status"],
            "learned_forecast_reason": score["learned_forecast_reason"], **witness, **controls, **online}


def decision_summary(rows):
    modes, misses = Counter(), Counter()
    for row in rows:
        observed = row["labels"]["terminal"]
        for method, prediction in row["forecasts"].items():
            mode = "online" if method == RULE else row["learned_forecast_mode"]
            category = "inspection" if prediction == INSPECT else "answer"
            modes[method, mode, category] += 1
            if observed is not None and category == "answer" and prediction != observed:
                misses[method, mode, observed, prediction] += 1
    return {"prediction_counts": [{"method": m, "mode": mode, "type": kind, "count": count}
                                   for (m, mode, kind), count in modes.items()],
            "wrong_answers": [{"method": m, "mode": mode, "observed": observed, "predicted": predicted, "count": count}
                              for (m, mode, observed, predicted), count in misses.most_common()]}


def next_checks(rows):
    checks = []
    gaps = [row["story_id"] for row in rows if row["capture_status"] in ("rejected", "incomplete")]
    unknown = [row["story_id"] for row in rows if "labels" in row and row["labels"]["terminal"] is None]
    deviations = [row["story_id"] for row in rows if row.get("condition", {}).get("status") == "deviation"]
    if gaps:
        checks.append({"action": "Use the recorded reasons to correct capture or timing before repeating these conditions.", "stories": gaps})
    if unknown:
        checks.append({"action": "Review the saved evaluator views if they can resolve the unknown terminal outcomes.", "stories": unknown})
    if deviations:
        checks.append({"action": "Use the observed condition checks to verify setup for these families in the next block.", "stories": deviations})
    return checks


def summarize(cards, index, stores, next_cards):
    rows = [review_card(card, index) for card in cards]
    complete = [row for row in rows if row["capture_status"] == "complete"]
    records = [index["story"][row["story_id"]] for row in complete]
    measured = describe(records)
    place = Counter(value for row in complete for value in row["observed_placement"].values())
    cameras = Counter(camera for row in complete for camera, count in row["cameras"].items() if count)
    return {"schema": "smart-swarm-physical-block-review-v1", "scope": "native physical development data",
            "planned_attempts": len(cards), "capture_counts": dict(Counter(row["capture_status"] for row in rows)),
            "admitted_source_groups": measured["independent_source_groups"], "verified_terminal_outcomes": measured["terminal_label_count"],
            "unknown_terminal_outcomes": len(complete) - measured["terminal_label_count"], "label_coverage": measured["label_coverage"],
            "usable_camera_histories": dict(cameras), "condition_status_counts": measured["condition_status_counts"],
            "observed_conditions_by_family": measured["observed_conditions_by_family"], "attended_placement_observations": dict(place),
            "learned_forecast_statuses": dict(Counter(row["learned_forecast_status"] for row in complete)),
            "motor_prefix_counts": dict(Counter(row["motor_prefix"]["status"] for row in rows)),
            "inspection_controls": inspection_controls(complete, METHODS),
            **decision_summary(complete), "cases": rows, "corpora": stores, "checks_for_next_session": next_checks(rows),
            "next_prepared_cards": [{key: card[key] for key in ("story_id", "family", "block")} for card in next_cards]}


def publish_ready(config, *, stories=None, motors=None, refresh=False):
    workspace = config["data_root"] / "physical-stories"
    scopes, next_cards = ready_scopes(prepare(config), workspace)
    directory = workspace / "block-reviews"
    pending = {name: cards for name, cards in scopes.items() if refresh or not (directory / f"{name}.json").is_file()}
    if not pending and stories is None and not refresh:
        return []
    stories = evidence_corpus(config) if stories is None else stories
    if not pending:
        if stories["attempted"]:
            publish_report(config, stories)
        return []
    motors = motor_corpus(config) if motors is None else motors
    stores = {"evidence_manifest": stories["manifest"], "motor_manifest": motors["manifest"]}
    index = index_records(stories, motors)
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for name, cards in pending.items():
        result = summarize(cards, index, stores, next_cards)
        if all(row["capture_status"] in TERMINAL for row in result["cases"]):
            path = directory / f"{name}.json"
            if not path.is_file() or load(path) != result:
                atomic(path, result)
                written.append({"path": str(path), **result})
    paper = publish_report(config, stories) if stories["attempted"] else None
    return [{**report, "paper_package": paper["package"] if paper else None} for report in written]


def announce(reviews):
    for report in reviews:
        counts = report["capture_counts"]
        print(f"Review {Path(report['path']).stem}: {counts.get('complete', 0)} usable, "
              f"{counts.get('incomplete', 0)} incomplete, {counts.get('rejected', 0)} rejected, "
              f"{report['unknown_terminal_outcomes']} unknown terminal outcomes, "
              f"{report['motor_prefix_counts'].get('accepted', 0)} accepted motor prefixes.")
        controls = report["inspection_controls"]["by_forecast_mode"]
        if any(row["eligible_f1_histories"] or row["eligible_f6_histories"] for row in controls.values()):
            pairs = sum(row["matched_source_blocks"] for row in controls.values())
            print(f"{pairs} source blocks have both F1 and F6 inspection controls in the same forecast mode.")
        for check in report["checks_for_next_session"]:
            print(check["action"])
        print(report["path"])
        if report["paper_package"]:
            print(f"Physical development results {report['paper_package']}")
