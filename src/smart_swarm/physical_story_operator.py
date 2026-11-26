"""Short attended prompts around the native controls and story journal."""

import time

from .episode import load
from .physical_story_condition import CARRIER_MOVE, RELOCATION_CHECKS, REMOVAL
from .physical_story_motor import review_pending_motor
from .physical_story_plan import prepare
from .physical_story_replay import (OUTCOMES, predict_selected, recover_pending_prediction, recover_sealed_reveal,
                                    terminal_reveal)
from .physical_story_source import (ACTIVE_STORY_STATES, FLEET_TERMINAL, TITLE, begin, fleet_manifest, mark,
                                    restore_start, same_recording_boot, story_preflight, story_workspace)


def next_story(config):
    root = story_workspace(config)
    plan = prepare(config)
    unstarted = None
    for row in plan["cards"]:
        trial = root / row["story_id"]
        journal = trial / "trial.json"
        if journal.is_file():
            state = restore_start(config, trial, load(journal))
            if state["status"] == "recording":
                recover_pending_prediction(trial, state)
            state = recover_sealed_reveal(trial, state)
            if state["status"] in ACTIVE_STORY_STATES:
                return row, state
        elif unstarted is None:
            unstarted = row
    if unstarted is not None:
        return unstarted, None
    raise ValueError("All 24 prepared stories already have a saved attempt")


def wait_for_capture(config, identifier):
    deadline = time.monotonic() + 150
    current = fleet_manifest(config, identifier)
    while current["phase"] == "recording" and time.monotonic() < deadline:
        time.sleep(2)
        current = fleet_manifest(config, identifier)
    if current["phase"] not in FLEET_TERMINAL:
        raise ValueError("The native fleet capture has not reached a terminal state")


def is_capturing(config, state):
    return fleet_manifest(config, state["fleet_capture_id"])["phase"] == "recording"


def physical_steps(config, identifier, state):
    if not same_recording_boot(state) or not is_capturing(config, state):
        return state
    if not state["markers"]:
        print(f"In SO101 Studio select {TITLE}. Start native recording and follow the placement card.")
    family = state["card"]["family"]
    f3 = family == "F3"
    if f3 and "insertion_closed" not in state["markers"]:
        outcome, assistance = observe_placement("After the first insertion and withdrawal")
        state = mark(config, identifier, "insertion_closed", assistance=assistance, placement_outcome=outcome)
    if "placement_closed" not in state["markers"] and is_capturing(config, state):
        observed = None
        if f3:
            print("Keep Studio recording for the removal and withdrawal.")
            assistance, observed = observe_motion("removal", "human_hand_assistance", REMOVAL)
            outcome = "unknown"
        else:
            outcome, assistance = observe_placement("After withdrawal")
        state = mark(config, identifier, "placement_closed", assistance=assistance, placement_outcome=outcome,
                     observed_change=observed)
        print("Save the native Studio episode now, then end its session before moving the carrier.")
    if "placement_closed" in state["markers"] and "relocation_closed" not in state["markers"] and is_capturing(config, state):
        print("Give Tank at least twelve seconds of footage around the stationary start, relocation and settled destination.")
        expected = RELOCATION_CHECKS.get(family, CARRIER_MOVE)
        assistance, observed = observe_motion("relocation", "human_relocation", expected, unknown_actor="unknown")
        state = mark(config, identifier, "relocation_closed", assistance=assistance, observed_change=observed)
    return state


def observe_placement(when):
    response = input(f"{when}, enter s/f for success/failure, hs/hf if hands assisted, or Enter if unclear: ")
    labels = {"s": ("placement_succeeded", "attended_native_motion"),
              "f": ("placement_failed", "attended_native_motion"),
              "hs": ("placement_succeeded", "human_hand_assistance"),
              "hf": ("placement_failed", "human_hand_assistance")}
    return labels.get(response.strip().lower(), ("unknown", None))


def observe_motion(phase, human, expected, *, unknown_actor=None):
    print(f"Confirm only if you saw {expected}.")
    response = input(f"After {phase}, enter h for hands or r for robot. Add d if different or u if unclear. "
                     "Use n for no movement, or Enter if both are unknown: ").strip().lower()
    if not response:
        return unknown_actor, "unknown"
    if response == "n":
        return "none", "different"
    source = {"h": human, "r": "attended_native_motion"}.get(response[0])
    observation = {"": "confirmed", "d": "different", "u": "unknown"}.get(response[1:])
    if source is None or observation is None:
        raise ValueError(f"Use h, r, hd, rd, hu, ru, n or Enter to record {phase}")
    return source, observation


def block_saved(config, block):
    root = story_workspace(config)
    journals = [root / item["story_id"] / "trial.json" for item in prepare(config)["cards"] if item["block"] == block]
    return bool(journals) and all(path.is_file() and load(path)["status"] in ("complete", "incomplete") for path in journals)


def report_motor_review(config):
    result = review_pending_motor(config)
    if result["attempted"]:
        eligible = sum(item["status"] == "reviewable_prefix" for item in result["states"].values())
        waiting = sum(item["status"] == "pending_review" for item in result["states"].values())
        print(f"Motor review saved {eligible} prefixes. {waiting} remain pending.")
    if result["errors"]:
        print("Native export will be retried on the next review. " + ". ".join(result["errors"].values()))
    return result


def close_block(config, card):
    if block_saved(config, card["block"]):
        print("This story block is saved. The motor review needs no scene or operator action.", flush=True)
        report_motor_review(config)


def show_prediction(decision):
    learned = decision.get("learned_evidence", {"status": "historical_rule_only"})
    reference = decision.get("slot_reference")
    if reference and reference["reason"]:
        print("Physical container sides are unresolved. " + reference["reason"])
    if learned["status"] == "scored":
        heads = learned["receipt"]["heads"]
        print("ROSMASTER pre-inspection choice", heads["events_only"]["decision"], "(frozen events-only policy)")
        print("Reveal-supervised comparison", heads["events_plus_reveal"]["decision"])
    else:
        print("Frozen learned prediction", learned.get("reason", learned["status"]))
    print("RGB rule comparison", decision["query_all"]["decision"], flush=True)


def run(config):
    card, state = next_story(config)
    identifier = card["story_id"]
    for field in ("task", "placement", "relocation", "withheld", "reset"):
        print(card[field])
    if state is None:
        print("Place a green station card on ROSMASTER's left and a blue one on its right. Both cameras should see both cards.")
        story_preflight(config)
        input("Press Enter when the scene, views and native controls are ready to start capture: ")
        state = begin(config, identifier)
    if state["status"] == "preparing":
        print("Waiting for the native Explorer start receipt. Reopen this card after the capture acknowledges.")
        return
    if state["status"] == "recording" and same_recording_boot(state):
        state = physical_steps(config, identifier, state)
        print("Keep the terminal relation unchanged. Waiting for the native recording readback.")
        wait_for_capture(config, state["fleet_capture_id"])
    decision = predict_selected(config, identifier)
    if decision.get("status") == "incomplete":
        print(f"Saved {identifier} as incomplete. {decision['reason']}")
        close_block(config, card)
        return
    show_prediction(decision)
    print("With motion stopped, reveal for the evaluator's overhead view only.")
    print("Use an outcome only if the object relation stayed unchanged until reveal. Enter x if you cannot confirm that.")
    print("Keep the relation fixed until the launcher confirms that it saved three overhead images.")
    choice = input("Enter left, right, surface, held, outside, x, or Enter for an unclear outcome: ").strip().lower()
    outcome = "relation_unverified" if choice == "x" else choice or "unknown"
    if outcome not in (*OUTCOMES, "relation_unverified"):
        raise ValueError("Unknown outcome label. The prediction is saved and can be reopened")
    result = terminal_reveal(config, identifier, outcome, preserved=outcome != "relation_unverified")
    print(f"Saved {identifier}, status {result['status']}, outcome {result['observed_terminal_outcome']}.")
    close_block(config, card)
