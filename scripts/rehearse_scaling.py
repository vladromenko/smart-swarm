"""Exercise the shared-world, fixed-total team-scaling capture and replay."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smart_swarm.cli import settings
from smart_swarm.episode import atomic
from smart_swarm.rendering import configure_renderer
from smart_swarm.research_jobs import admitted

FAMILIES = tuple(f"F{index}" for index in range(1, 7))
VARIANTS = {"F4": ("screened-association-v2",), "F6": ("hidden-contained", "hidden-surface")}


def one_team(assets, root, team, family, seed, *, receiver_history=False, lineage=None):
    from smart_swarm.scaling import audit_record, record, verify_hidden_pair

    cases = VARIANTS.get(family, ("witnessed",))
    traces = []
    for variant in cases:
        prefix = f"{lineage['source_group']}-{seed}-" if lineage else ""
        path = root / f"{prefix}n{team}-{family.lower()}-{variant}"
        capture = record(assets, path, seed, family, variant, team, 100, receiver_history=receiver_history, lineage=lineage)
        traces.append({**capture, **audit_record(path)})
    invariant = verify_hidden_pair(*traces) if family == "F6" else None
    return {"actor_count": team, "traces": traces, "f6_full_input_invariance": invariant}


def rehearse(config, output, family, teams, seed, *, receiver_history=False, heartbeat=None, reservation=None):
    configure_renderer(config)
    from smart_swarm.scaling import matching_budgets

    output.mkdir(parents=True, exist_ok=False)
    report = {"mode": "mujoco", "scope": "matched development rehearsal before the study",
              "independent_source_groups": 1, "seed": seed, "family": family, "teams": [], "status": "running",
              "reservation": reservation}
    destination = output / "scaling-rehearsal.json"
    try:
        for team in teams:
            if heartbeat is not None:
                heartbeat()
            report["teams"].append(one_team(config["asset_root"], output, team, family, seed, receiver_history=receiver_history))
            atomic(destination, report)
        matching_budgets([state["traces"][0] for state in report["teams"]])
        report["status"] = "verified"
        report["matched_exposure_times_and_sample_counts"] = True
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="incomplete", reason=str(error) or type(error).__name__)
        raise
    finally:
        atomic(destination, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--family", choices=FAMILIES, default="F1")
    parser.add_argument("--team", type=int, choices=(3, 4, 8), nargs="+", default=(3, 4, 8))
    parser.add_argument("--seed", type=int, default=45_001)
    parser.add_argument("--receiver-history", action="store_true")
    arguments = parser.parse_args()
    config = settings()
    with admitted(config, "smart-swarm-scaling-rehearsal-" + arguments.family.lower()) as (grant, heartbeat):
        report = rehearse(config, arguments.output.resolve(), arguments.family, sorted(set(arguments.team)), arguments.seed,
                          receiver_history=arguments.receiver_history, heartbeat=heartbeat, reservation=grant)
    print(f"Verified {len(report['teams'])} simulated team sizes for one matched development group")


if __name__ == "__main__":
    main()
