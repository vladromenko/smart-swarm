"""Publish the physical development results from admitted, saved forecasts."""

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from .episode import json_bytes, load, sha
from .physical_evidence_dataset import publish
from .physical_evidence_score import METHODS, PROFILE, matched_comparison, score_corpus
from .physical_inspection_controls import inspection_controls
from .physical_story_plan import prepare
from .physical_witness_score import USABLE, summarize as summarize_witnesses

SCHEMA = "smart-swarm-physical-development-paper-v1"
MODES = {"online": "Online", "blind_offline": "Blind offline"}
LABELS = dict(zip(METHODS, ("RGB rule", "Events only", "Events + reveal"), strict=True))
OUTCOME_COLORS = {"Correct": "#257a78", "Wrong": "#a44c34", "Inspect": "#bbc5cc"}


def paired_results(score):
    cases = score["cases"]
    return {mode: matched_comparison([row for row in cases if row["learned_forecast_mode"] == mode])
            for mode in MODES}


def table_rows(score):
    lines = ["| Forecast mode | Method | Histories | Source blocks | Correct | Wrong | Inspect | Group-weighted cost |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for mode, name in MODES.items():
        paired = score["matched_comparison_by_forecast_mode"][mode]
        for method in METHODS:
            row = paired["metrics_by_method"][method]
            if row:
                lines.append(f"| {name} | {LABELS[method]} | {row['histories']} | {row['scene_groups']} | "
                             f"{row['correct_answers']} | {row['wrong_answers']} | {row['inspections']} | "
                             f"{row['group_weighted_cost']:.3f} |")
        if not paired["labeled_matched_histories"]:
            lines.append(f"| {name} | No verified matched outcomes | 0 | 0 | — | — | — | — |")
    return lines


def witness_table(score):
    results = score.get("witness_evaluation")
    if results is None:
        return []
    lines = ["", "## Witness event recognition", "",
             "| Mode | Model | Factor | Confirmed labels | Scored replies | Correct top class | Source blocks |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    notes = []
    for mode, name in MODES.items():
        if mode in results["by_mode"]:
            entry = results["by_mode"][mode]
            for head, report in entry["models"].items():
                for factor, row in report["factors"].items():
                    counts = (row["confirmed_event_labels"], row["scored_visual_replies"],
                              row["correct_top_class"], row["scored_source_groups"])
                    lines.append(f"| {name} | {LABELS[head]} | {factor} | " + " | ".join(map(str, counts)) + " |")
                    excluded = {key: count for key, count in row["statuses"].items() if key != USABLE}
                    if excluded:
                        reason = ", ".join(f"{key.replace('_', ' ')} ({count})" for key, count in sorted(excluded.items()))
                        notes.append(f"- {name}, {LABELS[head]}, {factor}. Excluded replies include {reason}.")
            missing = {key: count for key, count in entry["receipt_counts"].items() if key != "available"}
            if missing:
                details = ", ".join(f"{key.replace('_', ' ')} ({count})" for key, count in sorted(missing.items()))
                notes.append(f"- {name} receipt exclusions include {details}.")
    lines.extend(["", "Witness scores use confirmed event labels and on-time visual replies in the receiver's frame. "
                  "Terminal errors and witness errors are reported separately in the per-source data.", "", *notes])
    return lines


def inspection_control_table(score):
    comparison = score.get("inspection_controls")
    if comparison is None:
        return []
    lines = ["", "## Inspection and direct-answer controls", "",
             "Each block contributes one observed F1 and one withheld F6 recording in the same forecast mode. "
             "F1 uses the verified final side. F6 uses the request for inspection. The controls are separate recordings.", "",
             "| Mode | Method | F1 correct direct | F1 inspect | F6 inspect | Both controls |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    labels = {**LABELS, "always_inspect": "Always inspect"}
    for mode, name in MODES.items():
        row = comparison["by_forecast_mode"][mode]
        denom = row["matched_source_blocks"]
        for method, label in labels.items():
            fields = ("f1_correct_direct", "f1_inspection", "f6_inspection", "both_controls")
            counts = [f"{row['by_method'][method][key]}/{denom}" if denom else "—" for key in fields]
            lines.append("| " + " | ".join((name, label, *counts)) + " |")
    for mode, name in MODES.items():
        row = comparison["by_forecast_mode"][mode]
        lines.extend(["", f"{name} has {row['eligible_f1_histories']} eligible F1 and {row['eligible_f6_histories']} eligible F6 "
                      f"histories. Both event labels were reviewed for {row['matched_f1_with_both_witness_labels_reviewed']} matched F1 histories."])
    return lines


def table(score):
    status = score["status"].replace("_", " ")
    lines = ["# Physical development pilot", "", f"Status **{status}**.", "",
             f"Attempted {score['attempted_histories']} of {score['planned_histories']} planned cards. "
             f"Admitted {score['admitted_histories']} histories across {score['admitted_scene_groups']} source blocks. "
             f"Verified terminal outcomes for {score['terminal_outcomes_available']} admitted histories. "
             f"Unknown terminal outcomes for {score['unknown_terminal_outcomes']} admitted histories.", "",
             *table_rows(score), "", "Each scored comparison uses the same labeled histories across methods. "
             "The RGB rule always decides online and is subset to the matching learned forecast mode. "
             "Blind offline forecasts use pre-reveal sealed inputs. Each source block has equal weight in the cost.", "",
             "F6 inspection counts use all matched withholding cards, including unknown outcomes. "
             "They describe requests issued by the policies.", ""]
    for mode, name in MODES.items():
        paired = score["matched_comparison_by_forecast_mode"][mode]
        count = paired["f6_matched_histories"]
        if count:
            groups = paired["f6_matched_scene_groups"]
            unit = "source block" if groups == 1 else "source blocks"
            entries = [f"{LABELS[key]} {paired['f6_inspection_counts'][key]}/{count}" for key in METHODS]
            lines.append(f"- {name}, {groups} {unit}. " + ", ".join(entries) + ".")
    exclusions = score["capture_exclusion_counts"]
    if exclusions:
        counts = ", ".join(f"{count} {status.replace('_', ' ')}" for status, count in sorted(exclusions.items()))
        lines.extend(["", f"Capture and admission exclusions include {counts}."])
    lines.extend(["", "Attended development data. The results describe localization forecasts and inspection requests. "
                  "Autonomous physical retrieval and independent final-test performance require separate evaluation."])
    lines.extend(witness_table(score))
    lines.extend(inspection_control_table(score))
    return "\n".join(lines) + "\n"


def draw_panel(axis, name, paired):
    count = paired["labeled_matched_histories"]
    groups = paired["labeled_matched_scene_groups"]
    axis.set_title(f"{name}\nHistories {count} · Source blocks {groups}", fontsize=10)
    if not count:
        axis.text(.5, .5, "No verified matches", ha="center", va="center", transform=axis.transAxes)
        axis.set_axis_off()
        return
    position = range(len(METHODS))
    rows = [paired["metrics_by_method"][key] for key in METHODS]
    correct = [row["group_weighted_correct_answer_rate"] for row in rows]
    answered = [row["group_weighted_answer_coverage"] for row in rows]
    wrong = [max(0., total - hit) for total, hit in zip(answered, correct, strict=True)]
    axis.barh(position, correct, color=OUTCOME_COLORS["Correct"])
    axis.barh(position, wrong, left=correct, color=OUTCOME_COLORS["Wrong"])
    axis.barh(position, [max(0., 1 - total) for total in answered], left=answered, color=OUTCOME_COLORS["Inspect"])
    axis.set(yticks=list(position), yticklabels=[LABELS[key] for key in METHODS], xlim=(0, 1), ylim=(2.65, -.65))
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.set_xlabel("Source-block-weighted fraction")


def figure(score):
    try:
        from matplotlib.backends.backend_pdf import FigureCanvasPdf
        from matplotlib.figure import Figure
        from matplotlib.patches import Patch
    except ImportError as error:
        raise ValueError("The paper figure needs Matplotlib in the Smart Swarm interpreter") from error

    fig = Figure(figsize=(8, 3.3), layout="constrained")
    FigureCanvasPdf(fig)
    fig.suptitle("Development localization forecasts", fontsize=11)
    axes = fig.subplots(1, 2)
    for axis, (mode, name) in zip(axes, MODES.items(), strict=True):
        draw_panel(axis, name, score["matched_comparison_by_forecast_mode"][mode])
    handles = [Patch(facecolor=color, label=label) for label, color in OUTCOME_COLORS.items()]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, fontsize=8)
    stream = BytesIO()
    fig.savefig(stream, format="pdf", metadata={"CreationDate": None, "ModDate": None})
    return stream.getvalue()


def checked_analysis(score):
    modes = paired_results(score)
    if (matched_comparison(score["cases"]) != score["matched_comparison"]
            or modes != score["matched_comparison_by_forecast_mode"]):
        raise ValueError("The paper results differ from the saved per-source forecasts")
    witness = summarize_witnesses(score["cases"])
    if score.get("witness_evaluation") != witness.get("witness_evaluation"):
        raise ValueError("The paper witness audit differs from the saved per-source evidence")
    has_controls = any(row.get("inspection_control_case") is not None for row in score["cases"])
    expected_controls = inspection_controls(score["cases"], METHODS) if has_controls else None
    if expected_controls != score.get("inspection_controls"):
        raise ValueError("The physical inspection controls differ from the saved cases")
    result = {"table.md": table(score).encode()}
    if any(row["labeled_matched_histories"] for row in modes.values()):
        result["forecast-outcomes.pdf"] = figure(score)
    return result


def sources(config, corpus):
    score = json.loads(json_bytes(score_corpus(corpus)))
    score["matched_comparison"] = matched_comparison(score["cases"])
    score["matched_comparison_by_forecast_mode"] = paired_results(score)
    plan = Path(prepare(config)["plan_path"])
    if sha(plan) != corpus["plan_sha256"] or sha(PROFILE) != score["profile_sha256"]:
        raise ValueError("The physical plan or frozen model profile changed during reporting")
    score.pop("corpus_manifest")
    score["source_plan_sha256"] = corpus["plan_sha256"]
    score["admitted_histories_by_family"] = corpus["histories_by_family"]
    score["event_label_coverage"] = corpus["label_coverage"]
    score["excluded_attempts"] = corpus["exclusions"]
    fields = ("manifest_sha256", "permitted_input_sha256", "camera_packets", "assistance_by_phase", "attended_placement_outcomes")
    score["admission_receipts"] = {row["story_id"]: {key: row.get(key) for key in fields} for row in corpus["records"]}
    check_source_groups(load(plan), score)
    return {"score.json": json_bytes(score) + b"\n", "source-plan.json": plan.read_bytes(),
            "frozen-profile.json": PROFILE.read_bytes(), **checked_analysis(score)}


def write_package(directory, files):
    hashes = {name: sha256(value).hexdigest() for name, value in files.items()}
    manifest = json_bytes({"schema": SCHEMA, "members_sha256": hashes}) + b"\n"
    identity = sha256(manifest).hexdigest()
    path = directory / f"{identity}.zip"
    directory.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=directory, prefix="physical-paper-") as stream:
        with ZipFile(stream, "w") as archive:
            for name, value in sorted({"PACKAGE.json": manifest, **files}.items()):
                archive.writestr(ZipInfo(name), value, compress_type=ZIP_DEFLATED)
        stream.flush()
        result = verify(Path(stream.name))
        try:
            path.hardlink_to(stream.name)
        except FileExistsError:
            if path.read_bytes() != Path(stream.name).read_bytes():
                raise ValueError("The immutable physical paper package has changed")
    return {**result, "package": str(path)}


def check_source_groups(plan, score):
    cards = {card["story_id"]: card for card in plan["cards"]}
    cases = score["cases"]
    admitted = {row["story_id"] for row in cases}
    excluded = {row["story_id"] for row in score["excluded_attempts"]}
    labels = sum(row["terminal"] is not None for row in cases)
    if (admitted & excluded or not (admitted | excluded) <= cards.keys()
            or len(excluded) != len(score["excluded_attempts"]) or len(admitted) != len(cases)
            or len(cases) != score["admitted_histories"] or set(score["admission_receipts"]) != admitted
            or len(admitted | excluded) != score["attempted_histories"] or len(cards) != score["planned_histories"]
            or len({row["scene_group_id"] for row in cases}) != score["admitted_scene_groups"]
            or score["terminal_outcomes_available"] != labels or score["unknown_terminal_outcomes"] != len(cases) - labels):
        raise ValueError("The physical package source and label counts disagree with the saved plan")
    for row in cases:
        card = cards[row["story_id"]]
        if (row["scene_group_id"] != card["source_group_id"] or row["family"] != card["family"]
                or row["split"] != card["split"] or row["split"] != "development"):
            raise ValueError("A reported physical story changed its assigned source block or split")


def verify(path):
    with ZipFile(path) as archive:
        manifest = json.loads(archive.read("PACKAGE.json"))
        hashes = manifest["members_sha256"]
        files = {name: archive.read(name) for name in hashes}
        if (manifest["schema"] != SCHEMA or len(archive.namelist()) != len(hashes) + 1
                or set(archive.namelist()) != {*hashes, "PACKAGE.json"}
                or any(sha256(value).hexdigest() != hashes[name] for name, value in files.items())):
            raise ValueError("The physical paper package members or digests have changed")
    score = json.loads(files["score.json"])
    if hashes["source-plan.json"] != score["source_plan_sha256"] or hashes["frozen-profile.json"] != score["profile_sha256"]:
        raise ValueError("The physical plan or frozen profile no longer matches the score")
    check_source_groups(json.loads(files["source-plan.json"]), score)
    expected = checked_analysis(score)
    base = {"score.json", "source-plan.json", "frozen-profile.json"}
    if set(files) != base | set(expected) or any(files[name] != data for name, data in expected.items()):
        raise ValueError("The paper analysis differs from the saved forecasts")
    modes = [mode for mode, row in paired_results(score).items() if row["labeled_matched_histories"]]
    return {"package": str(path), "sha256": sha(path), "status": score["status"], "scored_modes": modes,
            "attempted_histories": score["attempted_histories"], "admitted_histories": score["admitted_histories"],
            "table_reproduced": True, "figure_reproduced": "forecast-outcomes.pdf" in expected}


def publish_report(config, corpus=None):
    corpus = publish(config) if corpus is None else corpus
    directory = config["data_root"] / "physical-stories/paper-results"
    return write_package(directory, sources(config, corpus))
