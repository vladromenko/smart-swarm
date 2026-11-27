"""Short attended prompts around the native controls and story journal."""

from math import ceil
import time

from .episode import load
from .native_fleet_start import START_GATE, START_RECEIPT, valid_start_receipt
from .physical_learned import DEFERRED
from .lab_domain import MIN_RELOCATION_S, relocation_readiness
from .physical_story_condition import CARRIER_MOVE, RELOCATION_CHECKS, REMOVAL
from .physical_story_motor import review_pending_motor
from .physical_story_plan import prepare
from .physical_story_replay import OUTCOMES, close_interrupted_story, predict_selected, recover_story, terminal_reveal
from .physical_story_source import (ACTIVE_STORY_STATES, FLEET_TERMINAL, SETTLED_TANK_S, STATIONARY_LEAD_IN_S, TITLE, begin,
                                    claim_relocation_prompt, fleet_manifest, mark, same_recording_boot, start_relocation,
                                    story_preflight, story_workspace)


class StoryDeckComplete(ValueError):
    """Every prepared card already has a saved attempt."""


def next_story(config, *, preview=False):
    root = story_workspace(config)
    plan = prepare(config)
    unstarted = None
    for row in plan["cards"]:
        trial = root / row["story_id"]
        journal = trial / "trial.json"
        if journal.is_file():
            state = load(journal)
            if not preview and state["status"] in ACTIVE_STORY_STATES:
                state = recover_story(config, trial)
            if state["status"] in ACTIVE_STORY_STATES:
                return row, state
        elif unstarted is None:
            unstarted = row
    if unstarted is not None:
        return unstarted, None
    raise StoryDeckComplete("All 24 prepared stories already have a saved attempt")


def carrier_instruction(card):
    return f"Assigned carrier station is {card['requested_carrier_destination']} as seen by ROSMASTER."


def show(config):
    plan = prepare(config)
    print(f"\n{plan['title']}\n{plan['plan_path']}")
    try:
        selected, state = next_story(config, preview=True)
    except StoryDeckComplete as done:
        print(done)
        return
    print(f"Block {selected['block']} of {plan['blocks']}.")
    print(f"{'Resume' if state else 'Next'} {selected['story_id']}. {carrier_instruction(selected)}")
    for row in plan["cards"]:
        if row["block"] == selected["block"]:
            print(f"\n{row['family']} · {row['task']}")
            for field in ("placement", "relocation", "withheld", "query_boundary", "reveal_boundary"):
                print(f"  {row[field]}")


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
    state = record_placement(config, identifier, state)
    if "placement_closed" in state["markers"] and "relocation_closed" not in state["markers"]:
        state = record_relocation(config, identifier, state)
    return state


def record_placement(config, identifier, state):
    if not state["markers"]:
        print(f"In SO101 Studio select {TITLE}. Start native recording and follow the placement card.")
    family = state["card"]["family"]
    f3 = family == "F3"
    if f3 and "insertion_closed" not in state["markers"]:
        outcome, assistance = observe_placement("After the first insertion and withdrawal")
        state = mark(config, identifier, "insertion_closed", assistance=assistance, placement_outcome=outcome)
    if "placement_closed" not in state["markers"] and (state["status"] == "placement_recording"
                                                       or is_capturing(config, state)):
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
    return state


def archived_rgb_started(config, state):
    gate = state.get("fleet_rgb_start_gate")
    if gate is None:
        return True
    path = story_workspace(config) / state["card"]["story_id"] / START_RECEIPT
    try:
        receipt = load(path)
        return (gate == START_GATE and receipt["seen_monotonic_ns"] <= time.monotonic_ns()
                and valid_start_receipt(receipt, state["fleet_capture_id"], state["fleet_acknowledgement"]))
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return False


def wait_stationary_lead_in(config, state):
    acknowledged = state.get("fleet_acknowledgement")
    if acknowledged is None:
        print("Give Tank at least twelve seconds around the start, relocation and settled destination.")
        return is_capturing(config, state)
    age = lambda: (time.monotonic_ns() - acknowledged["monotonic_ns"]) / 1e9
    hold = ceil(STATIONARY_LEAD_IN_S - age())
    if hold > 0:
        print(f"Explorer acknowledged recording. Keep the scene still for {hold} seconds.", flush=True)
    proof = lambda: relocation_readiness(state, time.monotonic_ns(), archived_rgb_started(config, state))["ready"]
    if state.get("fleet_rgb_start_gate"):
        print("Waiting for recorded RGB from both native streams before relocation.", flush=True)
    while age() < state["fleet_window_s"] - MIN_RELOCATION_S and not proof() and is_capturing(config, state):
        until_lead_in = STATIONARY_LEAD_IN_S - age()
        time.sleep(min(2, until_lead_in) if until_lead_in > 0 else 1.)
    remaining = state["fleet_window_s"] - age()
    if remaining < MIN_RELOCATION_S or not proof() or not is_capturing(config, state):
        print("There is no full relocation window left. Keep the scene still for archive readback.")
        return False
    print(f"Explorer recording. About {ceil(remaining)} seconds remain in the acknowledged window.")
    return True


def record_relocation(config, identifier, state):
    if "relocation_prompt_opened" in state or "relocation_prompt_withheld" in state:
        print("Keep the scene fixed. Waiting for the existing native capture to finish.")
        return state
    capture = state["fleet_capture_id"]
    if not wait_stationary_lead_in(config, state):
        if state.get("fleet_rgb_start_gate"):
            reason = "The live recording did not confirm a usable RGB start with time left for relocation"
            state, _ = claim_relocation_prompt(config, identifier, capture, withheld=reason)
        return state
    state, granted = claim_relocation_prompt(config, identifier, capture)
    if not granted:
        print("Keep the scene fixed. Waiting for the existing native capture to finish.")
        return state
    print(carrier_instruction(state["card"]))
    if state.get("fleet_settled_window_s"):
        print(f"Move the carrier now. Confirm once it settles. Tank records the next {SETTLED_TANK_S} seconds.")
    else:
        print("Move the carrier now. Hold the settled destination for three seconds so Tank can record it, then confirm.")
    print("Keep the final scene still for ROSMASTER until Explorer finishes.")
    expected = RELOCATION_CHECKS.get(state["card"]["family"], CARRIER_MOVE)
    assistance, observed = observe_motion("relocation", "human_relocation", expected, unknown_actor="unknown")
    if not is_capturing(config, state):
        print("Explorer ended before confirmation. Saving the available recording for review.")
        return state
    return mark(config, identifier, "relocation_closed", assistance=assistance, observed_change=observed)


def observe_placement(when):
    response = input(f"{when}, enter s/f for success/failure, hs/hf if hands assisted, or Enter if unclear: ")
    labels = {"s": ("placement_succeeded", "attended_native_motion"),
              "f": ("placement_failed", "attended_native_motion"),
              "hs": ("placement_succeeded", "human_hand_assistance"),
              "hf": ("placement_failed", "human_hand_assistance")}
    return labels.get(response.strip().lower(), ("unknown", None))


def observe_motion(phase, human, expected, *, unknown_actor=None):
    print(f"Confirm only if you saw {expected}.")
    prompt = (f"After {phase}, enter h for hands or r for robot. Add d if different or u if unclear. "
              "Use n for no movement, or Enter if both are unknown: ")
    result = motion_response(input(prompt), human, unknown_actor)
    while result is None:
        print("Use h, r, hd, rd, hu, ru, n or Enter to confirm the same observation.")
        result = motion_response(input(prompt), human, unknown_actor)
    return result


def motion_response(text, human, unknown_actor):
    response = text.strip().lower()
    if not response:
        return unknown_actor, "unknown"
    if response == "n":
        return "none", "different"
    source = {"h": human, "r": "attended_native_motion"}.get(response[0])
    observation = {"": "confirmed", "d": "different", "u": "unknown"}.get(response[1:])
    if source is None or observation is None:
        return None
    return source, observation


def block_saved(config, block):
    root = story_workspace(config)
    journals = [root / item["story_id"] / "trial.json" for item in prepare(config)["cards"] if item["block"] == block]
    return bool(journals) and all(path.is_file() and load(path)["status"] in ("complete", "incomplete") for path in journals)


def report_motor_review(config):
    result = review_pending_motor(config)
    show_motor_review(result)
    return result


def show_motor_review(result):
    if result["attempted"]:
        eligible = sum(item["status"] == "reviewable_prefix" for item in result["states"].values())
        waiting = sum(item["status"] == "pending_review" for item in result["states"].values())
        print(f"Motor review saved {eligible} prefixes. {waiting} remain pending.")
    if result["errors"]:
        print("Native export will be retried on the next review. " + ". ".join(result["errors"].values()))
    for storage in result.get("export_storage", {}).values():
        if storage["state"] == "pending":
            print(f"Studio export is usable at its original path. Secondary storage is pending. {storage['reason']}")


def close_block(config, card):
    if block_saved(config, card["block"]):
        print("This story block is saved. Pending motor reviews run between stories while the launcher is open.", flush=True)


def close_active(config):
    card, state = next_story(config, preview=True)
    if state is None:
        print("There is no active story to close.")
        return False
    identifier = card["story_id"]
    print(f"Close interrupted story {identifier} only after saving or stopping in the native console.")
    reason = input("Enter the observed interruption reason to save this card as incomplete, or press Enter to keep it open: ").strip()
    if not reason:
        return False
    result = close_interrupted_story(config, identifier, reason)
    print(f"Saved {identifier} as incomplete. {result['reason']}")
    close_block(config, card)
    return True


def show_prediction(decision):
    learned = decision.get("learned_evidence", {"status": "historical_rule_only"})
    reference = decision.get("slot_reference")
    if reference and reference["reason"]:
        print("Physical container sides are unresolved. " + reference["reason"])
    if learned["status"] == "scored":
        heads = learned["receipt"]["heads"]
        print("ROSMASTER pre-inspection choice", heads["events_only"]["decision"], "(frozen events-only policy)")
        print("Reveal-supervised comparison", heads["events_plus_reveal"]["decision"])
    elif learned["status"] == DEFERRED:
        failure = learned.get("online_attempt")
        if failure is not None:
            print(f"Online learned forecasting was unavailable. {failure['reason']}")
        print("Private inputs are sealed. The frozen learned forecast will run as blind offline replay after reveal.")
    else:
        print("Frozen learned prediction", learned.get("reason", learned["status"]))
    print("RGB rule comparison", decision["query_all"]["decision"], flush=True)


def show_selected_card(card, state):
    print(f"Block {card['block']} · {card['family']} · {card['task']}")
    if state is None:
        print(card["reset"])
        for region in card["placement_regions"]:
            print(region)
        print(card["withheld"])
    elif state["status"] == "pending_reveal":
        print("Resume the saved forecast with the recorded terminal relation unchanged.")
        return
    markers = state["markers"] if state else {}
    if "placement_closed" not in markers:
        print(card["placement"])
    interrupted_prompt = state and ("relocation_prompt_opened" in state or "relocation_prompt_withheld" in state)
    if "relocation_closed" not in markers and not interrupted_prompt:
        print(carrier_instruction(card))
        print(card["relocation"])


def run(config, *, evaluation_mode="online"):
    card, state = next_story(config)
    identifier = card["story_id"]
    show_selected_card(card, state)
    mode = state.get("evaluation_mode", "online") if state else evaluation_mode
    print(f"Learned forecast mode: {mode.replace('_', ' ')}.")
    if state is None:
        print("Place a green station card on ROSMASTER's left and a blue one on its right. Both cameras should see both cards.")
        story_preflight(config)
        if input("Press Enter to start when ready, or type q to return to the menu: ").strip():
            return
        state = begin(config, identifier, staged=True, evaluation_mode=mode)
    decision = record_or_resume(config, identifier, state)
    if decision is None:
        return
    if decision.get("status") == "incomplete":
        print(f"Saved {identifier} as incomplete. {decision['reason']}")
    else:
        request_reveal(config, identifier, decision)
    close_block(config, card)


def record_or_resume(config, identifier, state):
    if state["status"] == "placement_recording" and same_recording_boot(state):
        if "staged_studio_recording" in state:
            print(f"Studio episode {state['staged_studio_episode_id']} is already saved.")
            if state.get("reason"):
                print(state["reason"])
            print("Use x from the menu to close the card as incomplete if the scene changed.")
            prompt = "Press Enter to retry Explorer if the scene stayed fixed, or q to return: "
        else:
            state = record_placement(config, identifier, state)
            print("Save the Studio episode and stop native control. Keep the container still for the next phase.")
            prompt = "Press Enter when Studio has saved and stopped, or q to resume later: "
        if input(prompt).strip():
            return None
        state = start_relocation(config, identifier)
    if state["status"] == "preparing":
        print("Waiting for the native Explorer start receipt. Reopen this card after the capture acknowledges.")
        return None
    if state["status"] == "recording" and same_recording_boot(state):
        state = physical_steps(config, identifier, state)
        print("Keep the terminal relation unchanged. Waiting for the native recording readback.")
        wait_for_capture(config, state["fleet_capture_id"])
    return predict_selected(config, identifier)


def request_reveal(config, identifier, decision):
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
