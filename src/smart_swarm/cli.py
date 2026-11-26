"""Local Smart Swarm milestone commands."""

import argparse
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import shutil
from subprocess import SubprocessError
import sys
import tomllib

from .contract import ProtocolError, check_contract
from .datasets import audit_lineage
from .episode import Replay, atomic, load, validate
from .rendering import configure_renderer
from .research_jobs import admitted

ROOT = Path(__file__).resolve().parents[2]


def settings():
    path = Path(os.environ.get("SMART_SWARM_CONFIG", ROOT / "configs" / "homebrain.toml"))
    cfg = tomllib.loads(path.read_text())
    for key in ("data_root", "asset_root", "platform_handover", "platform_project", "studio_project"):
        cfg[key] = (ROOT / cfg[key]).resolve()
    return cfg


def doctor(config):
    contract = check_contract(ROOT / "contracts")
    assets = config["asset_root"]
    components = {robot: (assets / f"{robot}.xml").is_file()
                  for robot in ("so101", "rosmaster", "tank")}
    packages = {name: package_version(name) for name in ("mujoco", "numpy", "scipy", "Pillow", "av", "lerobot")}
    platform = config["platform_handover"]
    report = {**contract, "actuation": "none", "display_backend": config["render_backend"],
              "python": sys.executable, "versions": packages, "asset_dir": str(assets),
              "sim_model_files": components, "simulation_dependencies_present": (
                  all(components.values()) and all(packages[name] for name in ("mujoco", "scipy", "av"))),
              "replay_dependencies_present": bool(packages["Pillow"]),
              "platform_handover_dir": str(platform), "platform_handover_present": platform.is_dir(),
              "data_root": str(config["data_root"]), "free_disk_mib": shutil.disk_usage(ROOT).free // 2**20}
    return report


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def demo(config, args):
    environment = config["data_root"]
    if shutil.disk_usage(ROOT).free < 256 * 1024**2:
        raise ProtocolError("STORAGE_LIMIT", "At least 256 MiB free is required for this bounded demo")
    configure_renderer(config)
    from .story import run_story
    from .video import encode_replay

    parent = args.output.resolve() if args.output else environment / "worlds"
    path = run_story(config["asset_root"], parent, args.family, args.seed,
                     swap=args.swap_markers, f6_miss=args.f6_miss)
    film = path / "replay.mp4"
    encode_replay(path, film)
    result = report(path, "rosmaster-1")
    state = status(config)
    latest = {**state["latest_demos"], args.family: str(path)}
    atomic(environment / "status.json", {**state, "latest_demos": latest,
                                          "contract_version": "robot-platform/1.0.0"})
    return {**result, "replay_video": str(film)}


def status(config):
    path = config["data_root"] / "status.json"
    if path.is_file():
        return load(path)
    return {"project": "smart-swarm", "state": "development", "latest_demos": {},
            "hardware": "unverified", "training": "pending"}


def locate(config, identifier):
    path = Path(identifier)
    return path if path.is_dir() else config["data_root"] / "worlds" / identifier


def report(path, agent):
    client = Replay(path, agent)
    return {"read_mode": "replay", "source_world_episode_id": client.world_id,
            "source_path": str(path.resolve()), "agent_id": agent,
            "packets": len(client.packets()), **client.predict()}


def validate_path(path):
    paths = [path] if (path / "manifest.json").is_file() else sorted(
        p.parent for p in path.glob("*/manifest.json"))
    if not paths:
        raise ValueError(f"No complete Smart Swarm manifests in {path}")
    manifests = [load(p / "manifest.json") for p in paths]
    for folder, manifest in zip(paths, manifests, strict=True):
        validate(folder, manifest)
    return {"valid": True, **audit_lineage(manifests)}


def parser():
    commands = argparse.ArgumentParser(prog="smart-swarm", description=__doc__)
    sub = commands.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Check runtime, assets and contract without opening devices")
    sub.add_parser("status", help="Read current simulation, training and physical status receipts")
    run = sub.add_parser("demo", help="Run a bounded shared-world simulation and save an episode")
    run.add_argument("--mode", choices=["mujoco"], required=True)
    run.add_argument("--family", choices=["F1", "F2", "F6"], default="F1")
    run.add_argument("--seed", type=int, default=3)
    run.add_argument("--swap-markers", action="store_true")
    run.add_argument("--f6-target-on-surface", dest="f6_miss", action="store_true",
                     help="In F6, make the occluded surface cube the target")
    run.add_argument("--output", type=Path, help="Parent directory for the new world episode")
    audit = sub.add_parser("validate", help="Check recording integrity and split ancestry")
    audit.add_argument("--dataset", required=True)
    watch = sub.add_parser("report", help="Causally replay a saved episode from one agent's scope")
    watch.add_argument("--run", required=True)
    watch.add_argument("--agent", default="rosmaster-1")
    add_motor_commands(sub)
    collection = sub.add_parser("collect", help="Prepare operator cards and check the recording gate")
    collection.add_argument("--plan", choices=["pilot-stories", "motor-skills", "so101-pilot"], required=True)
    collection.add_argument("--plan-only", action="store_true", help="Create or reopen the operator queue")
    sub.add_parser("rehearse-so101", help="Use A's fixture capture and export adapters and consume the export")
    add_resume_commands(sub)
    add_operator_commands(sub)
    return commands


def add_resume_commands(sub):
    diag = sub.add_parser("diagnose-motor", help="Follow matched teacher and ACT development checks")
    diag.add_argument("--stage", choices=["teacher", "act", "smolvla"], required=True)
    diag.add_argument("--raw", type=Path, required=True)
    diag.add_argument("--dataset", type=Path)
    diag.add_argument("--checkpoint", type=Path)


def add_operator_commands(sub):
    pilot = sub.add_parser("pilot", help="Attended SO101 pilot workflow using native Studio")
    pilot.add_argument("--print-card", action="store_true")
    sub.add_parser("rehearse-native", help="Re-export existing Studio technical recordings and consume them")
    sub.add_parser("execution-handoff", help="Record ACT simulation and fixture execution eligibility")
    team = sub.add_parser("native-team", help="Check fleet readiness and optionally read private archives")
    team.add_argument("--capture-id")
    story = sub.add_parser("story-recording", help="Operate the attended distributed story workflow")
    story.add_argument("stage", choices=("plan", "status", "readiness", "evidence-dataset", "evidence-score",
                                         "motor-dataset", "review-motor", "begin",
                                         "insertion-closed", "placement-closed", "relocation-closed", "predict", "reveal",
                                         "annotation-proposals", "annotate"))
    story.add_argument("--card")
    story.add_argument("--assistance", choices=("none", "unknown", "human_relocation", "attended_native_motion", "human_hand_assistance"))
    story.add_argument("--placement-outcome", choices=("placement_succeeded", "placement_failed", "unknown"))
    story.add_argument("--observed-change", choices=("confirmed", "different", "unknown"))
    story.add_argument("--outcome", choices=("left", "right", "surface", "held", "outside", "unknown", "relation_unverified"),
                       default="unknown")
    story.add_argument("--relations-preserved", action="store_true")
    story.add_argument("--annotation-file", type=Path)
    infer = sub.add_parser("evidence-predict", help="Replay a trained evidence checkpoint through the private scopes")
    infer.add_argument("--run", type=Path, required=True)
    infer.add_argument("--checkpoint", type=Path, required=True)
    infer.add_argument("--through-ns", type=int)
    infer.add_argument("--query-policy", choices=("broadcast", "latest_witness", "expected_information_gain"))
    evidence = sub.add_parser("evidence-pilot", help="Generate, audit or train the bounded evidence pilot")
    evidence.add_argument("--stage", choices=("generate", "audit", "train"), required=True)
    evidence.add_argument("--root", type=Path)

def add_motor_commands(sub):
    from .motor_visual_profile import STOCK, VISUAL_PROFILES

    generation = sub.add_parser("generate-motor", help="Generate contact-valid synthetic SO101 data")
    generation.add_argument("--split", choices=["train", "validation", "test"], required=True)
    generation.add_argument("--count", type=int, required=True)
    generation.add_argument("--wall-seconds", type=int, default=120)
    generation.add_argument("--disk-mb", type=int, default=1024)
    generation.add_argument("--output", type=Path)
    generation.add_argument("--visual-profile", choices=VISUAL_PROFILES, default=STOCK)
    export = sub.add_parser("export-motor", help="Export validated raw motor sets as LeRobot splits")
    export.add_argument("--source", type=Path)
    export.add_argument("--output", type=Path)
    export.add_argument("--clocked", action="store_true", help="Use the aligned target cue and local issue index")
    train = sub.add_parser("train", help="Run the bounded CPU LeRobot profile")
    train.add_argument("--recipe", required=True, choices=["act-pilot", "smolvla-pilot",
                                                          "act-pipeline", "smolvla-pipeline"])
    train.add_argument("--dataset", type=Path)
    evaluation = sub.add_parser("evaluate", help="Run offline and closed-loop MuJoCo motor checks")
    evaluation.add_argument("--protocol", choices=["motor-pilot"], required=True)
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--dataset", type=Path)
    evaluation.add_argument("--episodes", type=int, default=1)
    evaluation.add_argument("--steps", type=int, default=510)
    evaluation.add_argument("--wall-seconds", type=int, default=360)
    evaluation.add_argument("--executor-profile", choices=("aligned-sample-v2", "historical-eval-v1"),
                            default="aligned-sample-v2")
    evaluation.add_argument("--seed-start", type=int, default=31_000)
    evaluation.add_argument("--gpu-render", action="store_true", help="Request the native graphics lease for MuJoCo only")

def dispatch(config, args):
    if args.command == "doctor":
        return doctor(config)
    if args.command == "demo":
        return demo(config, args)
    if args.command == "validate":
        return validate_path(locate(config, args.dataset))
    if args.command == "report":
        return report(locate(config, args.run), args.agent)
    if args.command in ("generate-motor", "export-motor", "train", "evaluate"):
        return motor_command(config, args)
    if args.command == "collect":
        return collect_command(config, args)
    if args.command in ("rehearse-so101", "diagnose-motor"):
        return resume_command(config, args)
    if args.command == "status":
        from .research_status import project_summary
        return project_summary(config, status(config))
    return operator_command(config, args)


def operator_command(config, args):
    if args.command == "pilot":
        from .operator import run

        return run(config, print_card=args.print_card)
    if args.command == "rehearse-native":
        from .studio import replay_existing

        return replay_existing(config)
    if args.command == "execution-handoff":
        from .execution_handoff import publish

        return publish(config)
    if args.command == "native-team":
        from .fleet_adapter import archive_rehearsal, native_team_status

        state = native_team_status(config)
        if args.capture_id:
            state["archive_rehearsal"] = archive_rehearsal(config, args.capture_id)
        return state
    if args.command == "story-recording":
        return story_recording(config, args)
    if args.command == "evidence-predict":
        from .evidence_jobs import forecast

        return forecast(args.run, args.checkpoint, args.through_ns, args.query_policy)
    return evidence_command(config, args)


def story_recording(config, args):
    from .physical_story_plan import prepare
    from .physical_story_source import begin, mark, story_progress, story_readiness
    if args.stage == "plan":
        return prepare(config)
    if args.stage == "status":
        return story_progress(config)
    if args.stage == "readiness":
        return story_readiness(config)
    if args.stage in ("evidence-dataset", "evidence-score"):
        from .physical_evidence_dataset import publish
        from .physical_evidence_score import score_current
        return score_current(config) if args.stage == "evidence-score" else publish(config)
    if args.stage == "motor-dataset":
        from .physical_motor_dataset import publish

        return publish(config)
    if args.stage == "review-motor":
        from .physical_motor_dataset import publish
        from .physical_story_motor import review_pending_motor

        return {"review": review_pending_motor(config), "corpus": publish(config)}
    if args.card is None:
        raise ValueError("Select a physical story card from story-recording plan")
    if args.stage in ("annotation-proposals", "annotate"):
        from .physical_story_annotations import annotation_action
        if args.stage == "annotation-proposals" and args.annotation_file is not None:
            raise ValueError("Use annotate to confirm scoped event labels")
        return annotation_action(config, args.card, args.annotation_file,
                                 interactive=args.stage == "annotate" and args.annotation_file is None)
    if args.stage == "begin":
        return begin(config, args.card)
    if args.stage in ("insertion-closed", "placement-closed", "relocation-closed"):
        return mark(config, args.card, args.stage.replace("-", "_"), assistance=args.assistance,
                    placement_outcome=args.placement_outcome, observed_change=args.observed_change)
    from .physical_story_replay import predict_selected, terminal_reveal

    if args.stage == "predict":
        return predict_selected(config, args.card)
    return terminal_reveal(config, args.card, args.outcome, args.relations_preserved)


def evidence_command(config, args):
    from .evidence_jobs import audit, launch

    if args.stage == "audit":
        return audit()
    root = args.root or config["data_root"] / "evidence-pilot"
    if args.stage == "train":
        return launch(config, root.resolve() / "portfolio.json")
    configure_renderer(config)
    from .evidence_data import generate

    return generate(config["asset_root"], root.resolve())


def resume_command(config, args):
    from .rehearsal import OUTPUT, run_rehearsal

    if args.command == "rehearse-so101":
        return run_rehearsal(config)
    configure_renderer(config)
    from .motor_diagnostics import act_development, smolvla_development, teachers

    OUTPUT.mkdir(exist_ok=True, parents=True)
    if args.stage == "teacher":
        return teachers(config["asset_root"], args.raw.resolve(), OUTPUT / "teacher-replays.json")
    if args.dataset is None or args.checkpoint is None:
        raise ValueError("ACT diagnostics require the original dataset and checkpoint")
    evaluator = act_development if args.stage == "act" else smolvla_development
    return evaluator(config["asset_root"], args.raw.resolve(), args.dataset.resolve(),
                     args.checkpoint.resolve(), OUTPUT)


def collect_command(config, args):
    if args.plan == "so101-pilot":
        from .so101_pilot import choose

        return choose(config, args)
    from .collection import collect

    return collect(config, args)

def motor_command(config, args):
    root = config["data_root"]
    if args.command == "generate-motor":
        configure_renderer(config)
        from .motor import generate

        with admitted(config, "smart-swarm-generate-motor") as (_, heartbeat):
            return generate(config["asset_root"], args.output or root / "motor",
                            args.split, args.count, args.wall_seconds, args.disk_mb, heartbeat,
                            visual_profile=args.visual_profile)
    if args.command == "evaluate":
        check_motor_graphics(args)
        with admitted(config, "smart-swarm-evaluate-motor", gpu_render=args.gpu_render) as (grant, heartbeat):
            configure_renderer(config, grant=grant)
            from .evaluate_motor import evaluate

            return evaluate(config["asset_root"], args.checkpoint.resolve(),
                            (args.dataset or root / "lerobot").resolve(), root / "evaluation-runs",
                            args.episodes, args.steps, args.wall_seconds, args.executor_profile, args.seed_start,
                            heartbeat=heartbeat, reservation=grant)
    from .training import export_motor, train

    if args.command == "export-motor":
        return export_motor(args.source or root / "motor", args.output or root / "lerobot", clocked=args.clocked)
    return train(args.recipe, args.dataset or root / "lerobot", root / "training-runs", config)


def check_motor_graphics(args):
    from .motor_visual_profile import COMPACT, recorded_visual

    path = args.checkpoint / "smart-swarm-input.json"
    if args.gpu_render and path.is_file() and recorded_visual(load(path))["visual_profile"] == COMPACT:
        raise ValueError("The compact RGB profile is bound to the verified software renderer")


def main():
    args = parser().parse_args()
    try:
        config = settings()
        config["data_root"].mkdir(parents=True, exist_ok=True)
        result = dispatch(config, args)
        if args.command == "doctor":
            atomic(config["data_root"] / "doctor.json", result)
    except (OSError, ValueError, ImportError, RuntimeError, ProtocolError,
            SubprocessError) as error:
        print(f"smart-swarm: {error}", file=sys.stderr)
        return 2
    if result is not None:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
