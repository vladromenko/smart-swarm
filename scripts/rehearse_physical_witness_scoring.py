"""Verify physical witness accounting with sealed simulation and native technical receipts."""

import argparse
from pathlib import Path

from rehearse_physical_evidence_score import CALIBRATION, calibration_case
from smart_swarm.cli import ROOT, settings
from smart_swarm.contract import read_ns
from smart_swarm.episode import atomic, load, rows, sha
from smart_swarm.physical_evidence_score import score_current
from smart_swarm.physical_witness_score import HEADS, USABLE, diagnose, summarize, top_class

PILOT = ROOT / "evidence/learning-pilot-v3"
EXPECTED = {
    "physical-learned-archive-rehearsal-v2.json": "5ecc969af81ec9fef1e7ad1d179ebccf6782ead5e8ece216031bbb6b25f1d03b",
    "physical-learned-blind-offline-archive.json": "bd025ca64452c3d0aecdb4604aa4c22ea5ca666cf5e754c5360a2d44e515e4d9",
    "physical-slot-reference-rehearsal.json": "3a5fd90593fdb510ddb88730d5e3e6e88a8f514141c2a2adf39f17c8887cb4cb",
}
LABELS = "training_labels/settled-event.jsonl"


def original(name):
    path = PILOT / name
    if sha(path) != EXPECTED[name]:
        raise ValueError(f"The original technical proof changed at {path}")
    return load(path)


def calibration(family, saved):
    root = Path(saved["source"]).resolve()
    if not root.is_relative_to(CALIBRATION.resolve()):
        raise ValueError("This scoring rehearsal uses only the already opened calibration sources")
    manifest = load(root / "manifest.json")
    if sha(root / LABELS) != manifest["sidecar_sha256"][LABELS]:
        raise ValueError("The simulation event-label sidecar changed")
    labels, = rows(root / LABELS)
    if labels["terminal"]["ground_truth"] != saved["terminal_truth_for_scoring"]:
        raise ValueError("The separate terminal labels disagree")
    query = read_ns(manifest["query_boundary"]["monotonic_ns"])
    result = calibration_case(family, saved)
    diagnostic = diagnose(saved["prediction"], labels, query)
    return {**result, "witness_diagnostics": diagnostic}


def technical_receipt(receipt):
    requests = [row for name in HEADS for row in receipt["heads"][name]["wire"]["requests"]]
    query, = {read_ns(row["issued_ns"]) for row in requests}
    return diagnose(receipt, {}, query)


def slot_controls():
    saved = original("physical-slot-reference-rehearsal.json")["resolvable_control"]["fixture_mappings"]
    result = {key: technical_receipt(row) for key, row in saved.items()}
    for head in HEADS:
        rows = {key: row["heads"][head]["association"] for key, row in result.items()}
        if (top_class(rows["reversed"]) != "cyan_left" or top_class(rows["same"]) != "violet_left"
                or rows["unverified"]["status"] != "slot_reference_unverified"):
            raise ValueError("The witness score no longer follows the measured receiver reference")
    return result


def native_controls(saved):
    native = saved["native"]
    if any(not row["technical_only"] or not row["prediction_saved_and_reopened_before_reveal"] for row in native.values()):
        raise ValueError("The native receipts must remain technical and sealed before reveal")
    blind = original("physical-learned-blind-offline-archive.json")
    if blind["new_physical_attempts"] or not blind["reopened_unchanged"] or blind["forecast"]["mode"] != "blind_offline":
        raise ValueError("The archived blind result is not the original technical replay")
    controls = {f"online_{family}": technical_receipt(row["prediction"]) for family, row in native.items()}
    controls["blind_offline_F6"] = technical_receipt(blind["forecast"]["receipt"])
    for result in controls.values():
        if any(factor["label"] is not None or factor["status"] == USABLE
               for head in result["heads"].values() for factor in head.values()):
            raise ValueError("A native technical archive must contribute no event label or score")
    return controls


def rehearsal(config):
    saved = original("physical-learned-archive-rehearsal-v2.json")
    cases = [calibration(family, row) for family, entries in saved["simulation"].items() for row in entries]
    score = summarize(cases)
    f6 = [case for case in cases if case["family"] == "F6"]
    if (len(cases) != 3 or len({row["scene_group_id"] for row in cases}) != 2 or len(f6) != 2
            or f6[0]["witness_diagnostics"] != f6[1]["witness_diagnostics"]):
        raise ValueError("The hidden-truth alternatives changed the separate witness diagnostics")
    status = score_current(config)
    return {"scope": "Software rehearsal on archived simulation calibration and native technical receipts",
            "input_sha256": EXPECTED, "calibration_cases": cases, "calibration_scores": score,
            "native_technical_receipts": native_controls(saved), "side_reference_controls": slot_controls(),
            "physical_development_attempted": status["attempted_histories"], "physical_development_admitted": status["admitted_histories"],
            "new_physical_recordings": 0, "optimizer_updates": 0, "new_inference_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Choose a new immutable witness scoring rehearsal receipt")
    atomic(args.out, rehearsal(settings()))
    print(args.out)


if __name__ == "__main__":
    main()
