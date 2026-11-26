"""Small attended workflow shell around A's existing native controller."""

import json
from threading import Event, Lock, Thread
from urllib.error import URLError
import webbrowser

from .contract import ProtocolError
from .episode import atomic, load, sha
from .so101_pilot import prepare
from .studio import label_attempt, native_binding, native_request, native_status, pending_outcome, sync_pilot

SYNC_LOCK = Lock()
ATTEMPTS = (
    "Establish the nominal placement from the taped start and observe the outcome.",
    "Reset fully. Repeat the placement and verify that the cube lifts clear of the surface.",
    "Reset fully. Repeat and check release. Keep a complete failure and label its observed outcome.",
    "Reset fully. Repeat, withdraw to neutral and hold the final scene until the outcome is checked.",
)

CHOICES = """
  s  View the four-attempt task and current counts
  w  View the first distributed visual story block
  r  Start or resume the next attended distributed story
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


def select(config, choice):
    if choice == "s":
        show(config)
    if choice == "w":
        from .physical_story_plan import show as show_stories

        show_stories(config)
    if choice == "r":
        run_story_and_refresh(config)
    if choice == "c":
        binding = native_binding(config, create=True)
        print(f"Native dataset {binding['dataset_id']}. Use the matching dataset in SO101 Studio.")
    if choice == "o":
        url = config["studio_origin"] + "/"
        print(url)
        webbrowser.open(url, new=2)
    if choice == "i":
        with SYNC_LOCK:
            record = sync_pilot(config, native_binding(config))
        print(json.dumps(record, indent=2))
    if choice == "l":
        label_saved_attempt(config)
    if choice == "a":
        review_story(config)
    if choice == "q":
        from .physical_motor_dataset import publish
        from .physical_story_operator import report_motor_review

        if report_motor_review(config)["attempted"]:
            publish(config)


def run_story_and_refresh(config):
    from .physical_evidence_dataset import publish as evidence_corpus
    from .physical_motor_dataset import publish as motor_corpus
    from .physical_story_operator import run

    run(config)
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


def review_story(config):
    from .physical_evidence_dataset import publish
    from .physical_story_annotations import annotation_action

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


def label_saved_attempt(config):
    binding = native_binding(config)
    with SYNC_LOCK:
        pending = pending_outcome(config, binding)
    if pending is None:
        print("No completed pilot attempt awaits an outcome label.")
        return
    print(f"Observed outcome for attempt {pending['attempt']}.")
    key = input("Enter s for success, f for failure, or Enter for unknown: ").strip().lower()
    outcomes = {"s": "placement_succeeded", "f": "placement_failed", "": "unknown"}
    if key not in outcomes:
        raise ValueError("Use s, f, or Enter for the observed outcome")
    with SYNC_LOCK:
        result = label_attempt(config, binding, pending["episode_id"], outcomes[key])
    print(result)


def intake_fingerprint(config, binding):
    dataset = binding["dataset_id"]
    records = native_request(config, f"/api/episodes/{dataset}?include_frames=false")
    if not records:
        return None
    labels = config["data_root"] / "so101-pilot/observed-outcomes.json"
    phases = [[item["id"], item["storage"], item["metadata"].get("status")] for item in records]
    manifest = config["studio_project"] / "data" / dataset / "dataset.json"
    return {"recordings": sorted(phases), "labels": sha(labels) if labels.is_file() else None,
            "native_manifest": sha(manifest) if manifest.is_file() else None}


def observe_intake(config):
    binding = native_binding(config)
    status = native_status(config)
    if any(status[name] for name in ("recording_active", "teleoperation_active", "policy_active")):
        return
    with SYNC_LOCK:
        path = config["data_root"] / "so101-pilot/observed-input.json"
        fingerprint = intake_fingerprint(config, binding)
        prior = load(path) if path.is_file() else None
        if fingerprint is not None and fingerprint != prior:
            result = sync_pilot(config, binding)
            atomic(path, fingerprint)
            print("\nNative intake", json.dumps(result["counts"], sort_keys=True), flush=True)


def watch_intake(config, stopping):
    error_path = config["data_root"] / "so101-pilot/last-watch-error.json"
    while not stopping.wait(5):
        try:
            observe_intake(config)
        except (ProtocolError, OSError, ValueError, URLError) as error:
            atomic(error_path, {"reason": str(error)})


def run(config, *, print_card=False):
    if print_card:
        show(config)
        return None
    print("Smart Swarm / SO101 pilot\nUse the native console for all hardware operations.")
    stopping = Event()
    Thread(target=watch_intake, args=(config, stopping), daemon=True).start()
    choice = ""
    while choice != "q":
        print(CHOICES)
        try:
            choice = input("Smart Swarm action: ").strip().lower()
            select(config, choice)
        except (EOFError, KeyboardInterrupt):
            choice = "q"
        except (ProtocolError, OSError, ValueError, URLError) as error:
            print(f"Smart Swarm: {error}")
    stopping.set()
    return None
