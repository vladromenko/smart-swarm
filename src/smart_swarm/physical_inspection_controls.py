"""Compare withholding and observed F1 controls within physical source blocks."""

from collections import defaultdict

INSPECT = "request_inspection"
MODES = ("online", "blind_offline")


def control_case(record, score):
    audit = record.get("condition_audit", {})
    if record["family"] not in ("F1", "F6") or not audit.get("attended_markers_sealed"):
        return None
    labels = record["labels"]
    return {"story_id": score["story_id"], "scene_group_id": score["scene_group_id"], "family": score["family"],
            "mode": score["learned_forecast_mode"], "terminal": score["terminal"],
            "condition_status": audit["status"], "carrier_transition": audit.get("checks", {}).get("carrier_transition"),
            "reviewed_placement_label": labels.get("settled_placement"),
            "reviewed_association_label": labels.get("relocation_assignment"),
            "choices": {key: value["decision"] for key, value in score["forecasts"].items()}}


def eligible(case):
    if case["carrier_transition"] != "confirmed":
        return False
    if case["family"] == "F1":
        return case["condition_status"] == "observed" and case["terminal"] in ("left", "right")
    return case["family"] == "F6" and case["condition_status"] == "withholding_only"


def summarize_pair(group, first, hidden, methods):
    outcomes = {}
    for name in methods:
        left, right = first["choices"][name], hidden["choices"][name]
        direct = left == first["terminal"]
        waits = right == INSPECT
        outcomes[name] = {"f1_correct_direct": direct, "f1_inspection": left == INSPECT,
                          "f6_inspection": waits, "both_controls": direct and waits}
    return {"source_group": group, "f1_story": first["story_id"], "f6_story": hidden["story_id"],
            "f6_terminal_available": hidden["terminal"] is not None,
            "f1_both_witness_labels_reviewed": first["reviewed_placement_label"] is not None
            and first["reviewed_association_label"] is not None, "methods": outcomes}


def one_mode(candidates, mode, methods):
    groups = defaultdict(dict)
    for row in candidates:
        if row["mode"] == mode and set(row["choices"]) == set(methods) and eligible(row):
            group, family = row["scene_group_id"], row["family"]
            if family in groups[group]:
                raise ValueError("A source block has multiple copies of the same physical inspection control")
            groups[group][family] = row
    pairs = [summarize_pair(group, rows["F1"], rows["F6"], methods) for group, rows in sorted(groups.items())
             if "F1" in rows and "F6" in rows]
    fields = ("f1_correct_direct", "f1_inspection", "f6_inspection", "both_controls")
    metrics = {name: {key: sum(row["methods"][name][key] for row in pairs) for key in fields} for name in methods}
    count = len(pairs)
    metrics["always_inspect"] = dict(zip(fields, (0, count, count, 0), strict=True))
    return {"eligible_f1_histories": sum("F1" in rows for rows in groups.values()),
            "eligible_f6_histories": sum("F6" in rows for rows in groups.values()),
            "matched_source_blocks": count,
            "matched_f1_with_both_witness_labels_reviewed": sum(row["f1_both_witness_labels_reviewed"] for row in pairs),
            "by_method": metrics, "blocks": pairs}


def bound_control(row):
    candidate = row["inspection_control_case"]
    from_block_review = "assigned_family" in row
    family = row["assigned_family"] if from_block_review else row["family"]
    terminal = row["labels"]["terminal"] if from_block_review else row["terminal"]
    condition = row["condition"]["status"] if from_block_review else row["condition_status"]
    choices = row["forecasts"] if from_block_review else {name: value["decision"] for name, value in row["forecasts"].items()}
    expected = {"story_id": row["story_id"], "scene_group_id": row["scene_group_id"], "family": family,
                "condition_status": condition, "mode": row["learned_forecast_mode"], "terminal": terminal, "choices": choices}
    if any(candidate[key] != value for key, value in expected.items()):
        raise ValueError("An inspection control differs from its saved story and pre-reveal forecasts")
    return candidate


def inspection_controls(cases, methods):
    candidates = [bound_control(row) for row in cases if row.get("inspection_control_case") is not None]
    return {"scope": "Separate attended F1 and F6 histories from the same development source block",
            "f6_terminal_label_required": False, "identical_input_counterfactual": False,
            "by_forecast_mode": {mode: one_mode(candidates, mode, methods) for mode in MODES}}
