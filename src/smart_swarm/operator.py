"""Small attended workflow shell around A's existing native controller."""

from fcntl import LOCK_EX, LOCK_NB, flock
from hashlib import sha256
import json
from subprocess import TimeoutExpired
from threading import Event, Lock, Thread
import time
from urllib.error import URLError
import webbrowser

from .contract import ProtocolError, json_bytes
from .episode import atomic, load, sha
from .so101_pilot import prepare
from .studio import label_attempt, native_binding, native_request, native_status, pending_outcome, sync_pilot

SYNC_LOCK = Lock()
INTAKE_RETRY_DELAYS_S = (10, 30, 60)
ATTEMPTS = (
    "Establish the nominal placement from the taped start and observe the outcome.",
    "Reset fully. Repeat the placement and verify that the cube lifts clear of the surface.",
    "Reset fully. Repeat and check release. Keep a complete failure and label its observed outcome.",
    "Reset fully. Repeat, withdraw to neutral and hold the final scene until the outcome is checked.",
)

CHOICES = """
  d  Refresh live recording readiness
  v  Save both camera-only previews for framing review
  p  Record the attended SO101 setup, control and STOP check
  w  View the current distributed visual story block
  r  Start or resume the next story with an online learned forecast
  b  Start the next story with blind offline evaluation, or resume a saved mode
  x  Save an interrupted active story as incomplete
  s  View the four-attempt SO101 task and current counts
  c  Create or reopen the pilot dataset in SO101 Studio
  o  Open the native SO101 Studio console
  i  Validate and import completed native attempts
  l  Record the next completed attempt's terminal outcome
  a  Review an unclear story outcome from saved images
  q  Quit and review any saved motor prefixes
"""


def show(config):
    card = prepare(config)
    print(f"\n{card['title']}\n{card['placement_instruction']}")
    for title in ("preflight", "setup", "demonstrate", "reveal", "reset"):
        print(f"\n{title.capitalize()}")
        for text in card[title]:
            print(f"  {text}")
    for index, instruction in enumerate(ATTEMPTS, start=1):
        print(f"\nAttempt {index}. {instruction}")
    with_status = native_status(config)
    print("\nNative inventory", json.dumps(with_status["inventory_present"], sort_keys=True))
    print("Last native calibration readback", json.dumps(with_status["native_calibration_readback"], sort_keys=True))
    print("Native STOP", json.dumps(with_status["native_stop_state"], sort_keys=True))
    print("Native camera preview", json.dumps(with_status["native_camera_preview"], sort_keys=True))
    try:
        binding = native_binding(config)
        print(f"Native dataset {binding['dataset_id']} ({card['title']} {binding['scene_group_id']})")
        records = native_request(config, f"/api/episodes/{binding['dataset_id']}?include_frames=false")
        print(f"Native attempts saved {len(records)} of {card['attempt_limit']}")
    except ProtocolError:
        print("Use c to create the native dataset after checking inventory.")
    print("Open the native console with o. Keep this launcher open for automatic validation.")
    print("Save with the native recorder. Release teleoperation using native STOP to allow validation.")
    print("Pilot recordings", json.dumps(card.get("counts", {}), sort_keys=True))
    intake = config["data_root"] / "so101-pilot/native-intake.json"
    if intake.is_file():
        for attempt in load(intake)["attempts"]:
            reasons = ", ".join(attempt["reasons"])
            print(f"  {attempt['native_episode_id']} {attempt['outcome']} {attempt['classification']} {reasons}".rstrip())


def select(config, choice):
    if choice == "d":
        show_readiness(config, probe_mobile=True)
    if choice == "v":
        from .physical_framing import capture_framing

        result = capture_framing(config)
        print(f"Camera framing review {result['review_page']}")
    if choice == "p":
        from .attended_so101 import record_attended

        result = record_attended(config)
        print(result.get("receipt") or result["reason"])
    if choice == "s":
        show(config)
    if choice == "w":
        from .physical_story_operator import show as show_stories

        show_stories(config)
    if choice in ("r", "b", "x"):
        run_story_and_refresh(config, choice)
    if choice == "c":
        binding = native_binding(config, create=True)
        print(f"Native dataset {binding['dataset_id']}. Use the matching dataset in SO101 Studio.")
    if choice == "o":
        url = config["studio_origin"] + "/"
        print(url)
        webbrowser.open(url, new=2)
    if choice == "i":
        review_saved_motor(config)
        record = sync_observed_intake(config, native_binding(config), force=True)
        print(json.dumps(record, indent=2))
    if choice == "l":
        label_saved_attempt(config)
    if choice == "a":
        review_story(config)
    if choice == "q":
        report_archive_storage(config)
        review_saved_motor(config)


def review_saved_motor(config):
    from .physical_motor_dataset import publish
    from .physical_story_operator import report_motor_review
    from .physical_story_review import announce, publish_ready

    result = report_motor_review(config)
    motors = publish(config) if result["attempted"] else None
    announce(publish_ready(config, motors=motors, refresh=result["attempted"] > 0))


def show_mobile_transport(transport):
    if "probe_error" in transport:
        print(f"Native mobile observation probe failed. {transport['probe_error']}")
        return
    for row in transport.get("routes", []):
        address = f"{row['host']}:{row['port']}" if row.get("host") else f"port {row['port']}"
        result = row["status"].replace("_", " ")
        detail = row.get("detail") or row.get("http_status")
        rendered = f"{result} ({detail})" if detail is not None else result
        print(f"  {row['role']} {address} {rendered}")
    for role, action in transport.get("next_check_by_role", {}).items():
        print(f"  {role} next check. {action}")


def show_recording_storage(report):
    installation = report["recording_installation"]
    free = installation["data_root_free_bytes"]
    if free is not None:
        print(f"Smart Swarm storage {free / 1024**3:.2f} GiB free.")
    native_space = installation.get("native_capture_storage") or {}
    if "free_bytes" in native_space:
        print(f"Explorer capture storage {native_space['free_bytes'] / 1024**3:.2f} GiB free.")
    secondary = installation.get("secondary_archive_storage")
    if secondary:
        state = "mounted for closed archives" if secondary["mounted"] else "not mounted"
        print(f"Secondary archive volume {secondary['volume']}, {state}.")
    reference = report["capture_volume_reference"]
    if reference:
        lower, upper = [round(value / 1024**2) for value in reference["estimated_60s_payload_bytes"]]
        print(f"Explorer estimate {lower}–{upper} MiB per 60-second capture. Native startup requires 1 GiB free.")
    budget = report.get("capture_budget")
    if budget and budget["remaining_deck"]["archives_remaining"]:
        block = budget["next_block"]
        for label, estimate in ((f"Block {block['block']}", block), ("Remaining deck", budget["remaining_deck"])):
            low, high = [value / 1024**3 for value in estimate["estimated_bytes_with_native_reserve"]]
            print(f"{label}, {estimate['archives_remaining']} Explorer archives, estimate {low:.2f}–{high:.2f} GiB "
                  "with the native reserve. Studio and derived data need additional space.")
        margin = budget["remaining_deck"]["free_bytes_beyond_upper_estimate"] / 1024**3
        if margin < 0:
            print(f"Keeping the remaining mobile archives on the recording volume may need {-margin:.2f} GiB more.")


def show_readiness(config, *, probe_mobile=False):
    show_pilot_readiness(config)
    from .physical_story_source import story_readiness, story_readiness_with_transport
    try:
        read = story_readiness_with_transport if probe_mobile else story_readiness
        report = read(config)
    except (ProtocolError, OSError, ValueError, URLError) as error:
        print(f"Live readiness readback failed. {error}")
        return
    native = report["so101_studio"]
    devices = native["inventory_present"]
    resources = native["native_resources"]
    print(f"\nDistributed story readiness {report['status']}")
    print(f"SO101 leader {'present' if devices['leader'] else 'missing'}, "
          f"follower {'present' if devices['follower'] else 'missing'}.")
    for camera in resources["ordered_cameras"]:
        name = camera["sensor_id"].split("/")[-1]
        fresh = native["native_camera_preview"][name]["fresh"]
        print(f"  {name} {camera.get('resolved_path') or 'unavailable'}, {'fresh' if fresh else 'no fresh RGB'}")
    for role, status in report["mobile_rgb_and_clocks"]["roles"].items():
        rgb = "fresh RGB" if status["rgb_fresh"] else "no fresh RGB"
        clock = f"±{status['clock_uncertainty_ms']:.1f} ms" if status["clock_within_limit"] else "no usable clock"
        print(f"{role}: {rgb}, {clock}.")
    install = report["recording_installation"]
    show_recording_storage(report)
    exporter = "installed" if install["studio_exporter_installed"] else "unavailable"
    print(f"Studio exporter {exporter}. Hardware owners {len(resources['hardware_owners'])}.")
    for issue in report["blocking_issues"]:
        print(f"  {issue['reason']}")
    if probe_mobile:
        show_mobile_transport(report["mobile_native_transport"])
    check = report["attended_so101_check"]
    print(f"Current attended SO101 check {check['receipt']}" if check["current"] else check["reason"])
    for pending in report["pending_attended_checks"]:
        print(f"Pending attended check. {pending}.")


def show_pilot_readiness(config):
    from .attended_so101 import pilot_readiness

    try:
        report = pilot_readiness(config)
    except (ProtocolError, OSError, ValueError, URLError) as error:
        print(f"SO101-only readiness readback failed. {error}")
        return
    print(f"\nSO101-only pilot readiness {report['status']}")
    for reason in report["blocking_issues"]:
        print(f"  {reason}")
    check = report["attended_so101_check"]
    if not check["current"]:
        print(f"  {check['reason']}")


def run_story_and_refresh(config, choice):
    from .physical_story_operator import close_active, run

    if choice == "x":
        if not close_active(config):
            return
    else:
        run(config, evaluation_mode="blind_offline" if choice == "b" else "online")
    refresh_story_results(config)


def refresh_story_results(config):
    from .physical_evidence_dataset import publish as evidence_corpus
    from .physical_motor_dataset import publish as motor_corpus
    from .physical_story_review import announce, publish_ready

    report_archive_storage(config)
    stories = evidence_corpus(config)
    print(f"Native evidence corpus has {stories['admitted_histories']} admitted histories "
          f"across {stories['independent_source_groups']} source groups.")
    if stories["manifest"]:
        print(stories["manifest"])
    candidates = sum(item["candidate"] is not None and row["event_annotations"]["events"][name]["value"] is None
                     for row in stories["records"] for name, item in row["event_annotation_proposals"].items())
    if candidates:
        print(f"{candidates} visual event suggestions await optional review. Use a for the saved images.")
    ambiguous = [row["story_id"] for row in stories["records"]
                 if row["event_annotations"]["events"].get("terminal", {}).get("status") == "needs_confirmation"]
    if ambiguous:
        print(f"Use a to review unclear terminal outcomes for {', '.join(ambiguous)} from the saved overhead images.")
    motors = motor_corpus(config)
    if motors["prefix_count"]:
        print(f"Native motor corpus has {motors['prefix_count']} selected prefixes. {motors['manifest']}")
    announce(publish_ready(config, stories=stories, motors=motors, refresh=True))


def report_archive_storage(config):
    from .physical_archive_storage import retain_closed_captures

    try:
        results = retain_closed_captures(config)
    except (OSError, ValueError, KeyError, ProtocolError) as error:
        print(f"Explorer archive storage is pending. {error}", flush=True)
        return
    for result in results:
        identifier = result["capture_id"]
        if result["state"] == "retained":
            print(f"Explorer capture {identifier} preserved at {result['storage_path']}.", flush=True)
        else:
            print(f"Explorer capture {identifier} is awaiting archive storage. {result['reason']}", flush=True)


def review_story(config):
    from .physical_evidence_dataset import publish
    from .physical_story_annotations import annotation_action
    from .physical_story_review import announce, publish_ready

    pending = [row["story_id"] for row in publish(config)["records"]
               if any(item["status"] == "needs_confirmation" for item in row["event_annotations"]["events"].values())]
    if not pending:
        print("No unresolved outcomes have scoped review images.")
        return
    choices = {str(index): story for index, story in enumerate(pending, 1)}
    for key, story in choices.items():
        print(f"  {key}  {story}")
    selected = input("Enter the story number, or press Enter to return: ").strip()
    if selected:
        if selected not in choices:
            raise ValueError("Choose a listed story number")
        result = annotation_action(config, choices[selected], interactive=True)
        print(result.get("saved_receipt") or "Unconfirmed outcomes stayed unknown.")
        if result.get("saved_receipt"):
            announce(publish_ready(config, refresh=True))


def label_saved_attempt(config):
    binding = native_binding(config)
    with SYNC_LOCK:
        pending = pending_outcome(config, binding)
    if pending is None:
        print("The latest saved pilot attempt has no pending outcome label.")
        return
    print(f"Latest saved pilot attempt {pending['attempt']} ({pending['recording']['episode_id']}).")
    print("Label this attempt while its terminal scene is still visible, before resetting or starting the next attempt.")
    key = input("Enter s for native-arm success, h for manually assisted success, f for failure, or Enter for unknown: ").strip().lower()
    outcomes = {"s": ("placement_succeeded", "attended_native_motion"),
                "h": ("placement_succeeded", "manual_scene_assistance"),
                "f": ("placement_failed", "unknown"), "": ("unknown", "unknown")}
    if key not in outcomes:
        raise ValueError("Use s, h, f, or Enter for the observed outcome")
    outcome, assistance = outcomes[key]
    with SYNC_LOCK:
        result = label_attempt(config, binding, pending, outcome, assistance=assistance)
    print(result)


def intake_fingerprint(config, binding):
    dataset = binding["dataset_id"]
    records = native_request(config, f"/api/episodes/{dataset}?include_frames=false")
    if not records:
        return None
    labels = config["data_root"] / "so101-pilot/observed-outcomes.json"
    phases = [[item["id"], item["storage"], sha256(json_bytes(item["metadata"])).hexdigest()] for item in records]
    manifest = config["studio_project"] / "data" / dataset / "dataset.json"
    return {"recordings": sorted(phases), "labels": sha(labels) if labels.is_file() else None,
            "native_manifest": sha(manifest) if manifest.is_file() else None}


def intake_is_due(prior, fingerprint, now):
    if fingerprint is None:
        return False
    if prior.get("fingerprint", prior) != fingerprint:
        return True
    return prior.get("state") == "retry_pending" and now >= prior["next_retry_unix"]


def intake_after_check(prior, fingerprint, result, now):
    failures = list(prior.get("transient_failures", [])) if prior.get("fingerprint") == fingerprint else []
    rejection = result.get("validation_rejection")
    receipt = {"fingerprint": fingerprint, "transient_failures": failures,
               "state": "rejected" if rejection else "checked"}
    if result.get("validation_retryable"):
        failures.append({"unix_s": now, "reason": rejection})
        count = len(failures)
        if count <= len(INTAKE_RETRY_DELAYS_S):
            receipt.update(state="retry_pending", next_retry_unix=now + INTAKE_RETRY_DELAYS_S[count - 1])
        else:
            receipt["state"] = "retry_exhausted"
    return receipt


def sync_observed_intake(config, binding, *, force=False):
    with SYNC_LOCK:
        path = config["data_root"] / "so101-pilot/observed-input.json"
        fingerprint = intake_fingerprint(config, binding)
        if fingerprint is None and not force:
            return None
        with path.with_name("intake-receipt.lock").open("a") as guard:
            try:
                flock(guard, LOCK_EX | LOCK_NB)
            except BlockingIOError:
                if force:
                    raise ProtocolError("RESOURCE_BUSY", "Another Smart Swarm window is checking the saved recordings",
                                        retryable=True) from None
                return None
            prior = load(path) if path.is_file() else {}
            if not force and not intake_is_due(prior, fingerprint, time.time()):
                return None
            result = sync_pilot(config, binding)
            if fingerprint is not None:
                receipt = intake_after_check(prior, fingerprint, result, time.time())
                atomic(path, receipt)
                return {**result, "automatic_retry": receipt["state"]}
            return result


def observe_intake(config):
    binding = native_binding(config)
    status = native_status(config)
    if any(status[name] for name in ("recording_active", "teleoperation_active", "policy_active")):
        return
    result = sync_observed_intake(config, binding)
    if result is not None:
        print("\nNative intake", json.dumps(result["counts"], sort_keys=True), flush=True)
        storage = (result.get("validation") or {}).get("archive_storage", {})
        if storage.get("state") == "pending":
            print(f"Studio export is usable at its original path. Secondary storage is pending. {storage['reason']}", flush=True)
        if result.get("validation_rejection"):
            print(result["validation_rejection"], flush=True)
        if result.get("automatic_retry") == "retry_pending":
            print("Native export validation will retry automatically when Studio is idle.", flush=True)
        if result.get("automatic_retry") == "retry_exhausted":
            print("Native export validation is still pending. Use i for a manual retry.", flush=True)


def observe_motor_review(config, retry):
    from .physical_motor_dataset import publish
    from .physical_story_motor import motor_review_generation, review_pending_motor
    from .physical_story_operator import show_motor_review
    from .physical_story_review import announce, publish_ready

    generation = motor_review_generation(config)
    if not intake_is_due(retry, generation, time.time()):
        return
    result = review_pending_motor(config, closed_blocks_only=True)
    if not result["attempted"]:
        return
    receipt = motor_retry_receipt(retry, generation, result, time.time())
    report_changed = receipt.get("reason") != retry.get("reason") or result["reservation_acquired"]
    retry.update(receipt)
    if not report_changed:
        return
    show_motor_review(result)
    if receipt["state"] == "retry_exhausted":
        print("Motor review is still pending. It will resume after the native input changes. Use i to retry now.", flush=True)
    announce(publish_ready(config, motors=publish(config), refresh=True))


def motor_retry_receipt(prior, generation, result, now):
    if not result["reservation_acquired"]:
        return {"fingerprint": generation, "state": "retry_pending", "next_retry_unix": now + 60,
                "reason": result["errors"]["shared_reservation"], "transient_failures": []}
    pending = any(item["status"] == "pending_review" for item in result["states"].values())
    reason = ". ".join(result["errors"].values()) or "A saved Studio episode or export is still pending"
    return {**intake_after_check(prior, generation, {"validation_retryable": pending,
                                "validation_rejection": reason if pending else None}, now), "reason": None}


def watch_intake(config, stopping):
    from .physical_evidence_dataset import publish
    from .physical_offline import observe_pending
    from .physical_story_review import announce, publish_ready

    error_path = config["data_root"] / "so101-pilot/last-watch-error.json"
    reviewed = {}
    motor_retry = {}
    reopen_reports = True
    while not stopping.wait(5):
        try:
            observe_intake(config)
        except (ProtocolError, OSError, ValueError, URLError) as error:
            atomic(error_path, {"reason": str(error)})
        try:
            story = observe_pending(config, reviewed)
            reopen_reports = reopen_reports or story is not None
            stories = publish(config) if story else None
            if reopen_reports:
                report_archive_storage(config)
            if story:
                print(f"\nSaved blind offline forecast for {story}.", flush=True)
            announce(publish_ready(config, stories=stories, refresh=reopen_reports))
            reopen_reports = False
        except (ProtocolError, OSError, ValueError, RuntimeError, KeyError, TimeoutExpired) as error:
            atomic(error_path.with_name("last-blind-offline-error.json"), {"reason": str(error)})
        if not stopping.is_set():
            try:
                observe_motor_review(config, motor_retry)
            except (ProtocolError, OSError, ValueError, RuntimeError, KeyError, TimeoutExpired) as error:
                atomic(error_path.with_name("last-motor-review-error.json"), {"reason": str(error)})


def run(config, *, print_card=False):
    if print_card:
        show(config)
        return None
    print("Smart Swarm\nUse the native console for all hardware operations.")
    show_readiness(config)
    stopping = Event()
    worker = Thread(target=watch_intake, args=(config, stopping))
    worker.start()
    choice = ""
    try:
        while choice != "q":
            print(CHOICES)
            try:
                choice = input("Smart Swarm action: ").strip().lower()
                if choice == "q":
                    finish_intake(stopping, worker)
                select(config, choice)
            except (EOFError, KeyboardInterrupt):
                choice = "q"
            except (ProtocolError, OSError, ValueError, URLError) as error:
                print(f"Smart Swarm: {error}")
    finally:
        finish_intake(stopping, worker)
    return None


def finish_intake(stopping, worker):
    stopping.set()
    if worker.is_alive():
        print("Closing the background saved-data review loop.", flush=True)
        worker.join()
