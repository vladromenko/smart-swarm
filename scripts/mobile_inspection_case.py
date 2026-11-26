"""Record ordinary private observations and intervene only after their sealed decision."""

from contextlib import closing, contextmanager
from hashlib import sha256
import signal
import time

from audit_primary_scope import verify_projection
from audit_mobile_triangulation import prediction as optical_prediction
from check_mobile_inspection_rehearsal import STATION, verify_route, verify_task, verify_views
from compare_rosmaster_inspection import ARMS, LIMITS, endpoint, profile_method, score_pair, verify_sealed
from probe_rosmaster_inspection import evaluator_state
from rehearse_rosmaster_mobility import compare, mobile_scan, stationary
from smart_swarm.evidence_rollouts import prepare_world, reveal
from smart_swarm.episode import atomic, json_bytes, load, sha
from smart_swarm.inspection_comparison import follow_up
from smart_swarm.inspection_mobility import prepare_mobile
from smart_swarm.mobile_decision_state import transplant
from smart_swarm.scaling import PassiveReceiverStory, describe_capture

DECISION = "pre-inspection-decision.json"


def record_decision(world, variation, pickup, recipe, path):
    family, seed, variant = (recipe[key] for key in ("family", "seed", "variant"))
    story = PassiveReceiverStory(path, family, world, seed, variant == "hidden-surface")
    try:
        split = "development" if recipe["source_group"].startswith("mobile-rehearsal-") else "test"
        lineage = {"source_group": recipe["source_group"], "split": split}
        describe_capture(story, variation, variant, 3, True, lineage)
        story.scene(pickup)
        query, decision, report = story.ask()
        snapshot = world.snapshot()
        finished, terminal, _ = reveal(story, decision)
        story.recorder.meta.update(status="valid", experiment_role=f"mobile_inspection_{split}")
        story.recorder.close(query, decision, finished)
        initial = report["query_all"]["decision"]
        return snapshot, initial, terminal["ground_truth"], story.recorder.meta["physical_story_gate"]
    except (Exception, KeyboardInterrupt) as error:
        story.recorder.quarantine(str(error) or type(error).__name__)
        raise


def freeze_prediction(directory, pre, contexts, engineering):
    signature, frame_count = verify_projection(pre)
    prediction = {"permitted_input_sha256": sha256(json_bytes(signature)).hexdigest(),
                  "permitted_rgb_frames": frame_count, "recording_manifest_sha256": sha(pre / "manifest.json"),
                  "engineering_rule": engineering, "learned": profile_method(pre, contexts)}
    atomic(directory / DECISION, prediction)
    return prediction


def record_branches(world, snapshot, directory, waypoint):
    before = endpoint(evaluator_state(world))
    scan_dir = directory / "scan"
    scan_dir.mkdir()
    scan = mobile_scan(world, snapshot, scan_dir, STATION, waypoint)
    scan_after = endpoint(evaluator_state(world))
    schedule = [row["at_ns"] for row in scan["policy_observations"]]
    idle_dir = directory / "stationary"
    idle_dir.mkdir()
    idle = stationary(world, snapshot, schedule, idle_dir)
    idle_after = endpoint(evaluator_state(world))
    matched = compare(scan, idle)
    verify_route({"active": scan, "stationary": idle, "comparison": matched}, waypoint)
    branches = {"scan": scan, "stationary": idle}
    public = {arm: public_branch(directory / arm, branch) for arm, branch in branches.items()}
    truth = {arm: {"reset_exact": True, "before": before, "after": after,
                   "target_displacement_m": branch["evaluator"]["cube_displacement_m"],
                   "carrier_displacement_m": branch["evaluator"]["carrier_displacement_m"],
                   "visibility": branch["evaluator"]["visibility"]}
             for (arm, branch), after in zip(branches.items(), (scan_after, idle_after), strict=True)}
    return public, truth


def public_branch(path, branch):
    verify_views(path, branch)
    decision = optical_prediction(path, branch["policy_observations"])
    motion = branch["motion"]
    error = {"max_tracking_error_deg": max(row["max_error_deg"] for row in motion["arm"])} if motion else None
    samples = [{**frame, "motion": error} for frame in branch["policy_observations"]]
    return {"start_ns": branch["start_ns"], "stop_ns": branch["stop_ns"], "samples": samples,
            "motion": motion, "checked_physics_steps": branch["checked_physics_steps"],
            "external_contact_pairs": branch["external_contact_pairs"], "self_contact_pairs": branch["self_contact_pairs"],
            "post_inspection_decision": decision}


def acquire(config, recipe, directory, contexts, waypoint):
    family, seed, variant = (recipe[key] for key in ("family", "seed", "variant"))
    assets = config["asset_root"]
    stock, variation, pickup = prepare_world(assets, seed, family, variant=variant, seconds=LIMITS["case_wall_seconds"])
    pre = directory / "pre-inspection"
    with closing(stock):
        fixed, rule, expected, teacher = record_decision(stock, variation, pickup, recipe, pre)
        prediction = freeze_prediction(directory, pre, contexts, rule)
        stock.close()
        mobile, _, _ = prepare_mobile(assets, seed, family, variant, seconds=LIMITS["case_wall_seconds"],
                                     observed_steps=True)
        with closing(mobile):
            snapshot = transplant(stock, mobile, fixed)
            mobile.frame_size = (320, 240)
            policy, truth = record_branches(mobile, snapshot, directory, waypoint)
            models = {field: mobile.manifest[field] for field in ("parent_model_sha256", "model_sha256")}
    verify_task(teacher, variant, expected)
    policy["pre_inspection_decision_sha256"] = sha(directory / DECISION)
    policy["methods"] = {kind: {arm: follow_up(row["decision"], policy[arm]) for arm in ARMS}
                         for kind, row in prediction["learned"].items()}
    public = directory / "policy.json"
    atomic(public, policy)
    result = score_pair(expected, policy, truth)
    (directory / "evaluator").mkdir()
    atomic(directory / "evaluator/result.json", {"truth": truth, "evaluation": result, "teacher": teacher,
           "sealed_policy_sha256": sha(public), "snapshot_sha256": sha256(snapshot.integration.tobytes()).hexdigest(),
           "stock_snapshot_sha256": sha256(fixed.integration.tobytes()).hexdigest(), "simulation_models": models})
    return {"status": "sealed", "evaluation": result, "policy_sha256": sha(public)}


def deadline_reached(_signal, _frame):
    raise TimeoutError("The paired mobile comparison reached its original per-case wall cap")


@contextmanager
def case_deadline():
    previous = signal.signal(signal.SIGALRM, deadline_reached)
    signal.setitimer(signal.ITIMER_REAL, LIMITS["case_wall_seconds"])
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def capture_case(config, recipe, directory, contexts, waypoint):
    directory.mkdir()
    receipt = directory / "receipt.json"
    started = time.monotonic()
    report = {**recipe, "status": "running", "start_wall_unix": time.time()}
    atomic(receipt, report)
    try:
        with case_deadline():
            result = acquire(config, recipe, directory, contexts, waypoint)
        report.update(result)
    except (AssertionError, KeyError, OSError, RuntimeError, ValueError) as error:
        report.update(status="failed", reason=f"{type(error).__name__}: {error}")
    finally:
        report["wall_seconds"] = time.monotonic() - started
        atomic(receipt, report)
    return report


def verify_case(directory, receipt):
    verify_sealed(directory, receipt)
    policy = load(directory / "policy.json")
    frozen = load(directory / DECISION)
    pre = directory / "pre-inspection"
    signature, frames = verify_projection(pre)
    if (policy["pre_inspection_decision_sha256"] != sha(directory / DECISION)
            or frozen["recording_manifest_sha256"] != sha(pre / "manifest.json")
            or frozen["permitted_input_sha256"] != sha256(json_bytes(signature)).hexdigest()
            or frozen["permitted_rgb_frames"] != frames):
        raise ValueError("The original pre-inspection prediction or admitted inputs changed")
    if any(policy["methods"][kind][arm]["initial_choice"] != row["decision"]
           for kind, row in frozen["learned"].items() for arm in ARMS):
        raise ValueError("The branch modified its frozen initial decision")
    for arm in ARMS:
        branch = policy[arm]
        if optical_prediction(directory / arm, branch["samples"]) != branch["post_inspection_decision"]:
            raise ValueError("The two-view RGB decision could not be reproduced")
    return frozen
