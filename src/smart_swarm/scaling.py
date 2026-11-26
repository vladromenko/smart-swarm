"""Capture matched camera opportunities from actual simulated team members."""

from collections import defaultdict
from contextlib import closing
from pathlib import Path
import time

from .episode import LABELS, TRUTH, Replay, atomic, load, rows, sha
from .evidence_inputs import runtime_signature
from .evidence_portfolio import labels_for
from .evidence_rollouts import EvidenceStory, prepare_world, reveal
from .physics import STEP
from .story import AGENTS
from .teacher import move_box
from .vision import relocation

WINDOW_SLOTS = 24
SLOT_STEPS = 60
SLOT_PERIOD_NS = round(SLOT_STEPS * STEP * 1_000_000_000)


class ScalingStory(EvidenceStory):
    def witnesses(self):
        return (*AGENTS[:2], *self.world.parked_observers)

    def transport_delay(self, agent):
        return super().transport_delay(agent if agent == AGENTS[0] else AGENTS[1])

    def capture_window(self, observers):
        endpoints, opportunities = defaultdict(list), []
        for index in range(WINDOW_SLOTS):
            self.world.step(SLOT_STEPS)
            agent = observers[index % len(observers)]
            packet, pictures = self.capture(agent)
            image = pictures[agent + "/front"]
            endpoints[agent].append((packet, image))
            opportunities.append({"source_time_ns": self.world.ns, "agent_id": agent})
        return endpoints, opportunities

    def relocate(self):
        observers = self.witnesses()[1:]
        first, before = self.capture_window(observers)
        move_box(self.world, lambda: None)
        self.world.step(200)
        last, after = self.capture_window(observers)
        for agent in observers:
            initial, initial_view = first[agent][0]
            final, final_view = last[agent][-1]
            self.add_evidence(relocation([initial, final], initial_view, final_view, sensor=agent + "/front"))
        self.recorder.meta["fixed_observation_budget"] = {"mode": "fixed_total", "period_ns": SLOT_PERIOD_NS,
            "allocated_camera_samples": len(before) + len(after), "before_relocation": before,
            "after_relocation": after, "allocated_camera_time_s": (len(before) + len(after)) * SLOT_PERIOD_NS / 1e9}


class PassiveReceiverStory(ScalingStory):
    """Keep fixed receiver-front exposure times across team sizes."""

    def approach(self):
        super().approach()
        if self.family != "F6":
            self.capture(AGENTS[2])

    def relocate(self):
        if self.family == "F6":
            self.capture(AGENTS[2])
            self.world.step(SLOT_STEPS)
        self.capture(AGENTS[2])
        super().relocate()


def describe_capture(story, variation, variant, team_size, receiver_history, lineage):
    family = story.family
    meta = story.recorder.meta
    role = f"primary_scaling_{lineage['split']}" if lineage else "development_scaling_rehearsal"
    meta.update(evidence_variant=variant, scaling_team_size=team_size, physical_variation=variation, experiment_role=role)
    if receiver_history:
        profile = ("front-after-placement-before-and-after-relocation" if family == "F6"
                   else "front-before-placement-before-and-after-relocation")
        meta["scaling_receiver_profile"] = profile
    if lineage:
        meta["scene_group_id"] = lineage["source_group"]
        meta["lineage"]["split"] = lineage["split"]
    meta["source_file_sha256"]["scaling.py"] = sha(Path(__file__))
    atomic(story.path / "recording.json", {**meta, "recording_integrity": "recording"})


def record(assets, path, seed, family, variant, team_size, timeout, *, receiver_history=False, lineage=None):
    world, variation, pickup = prepare_world(assets, seed, family, variant=variant, seconds=timeout, team_size=team_size)
    story_type = PassiveReceiverStory if receiver_history else ScalingStory
    with closing(world):
        story = story_type(path, family, world, seed, variant == "hidden-surface" or family == "F2")
        try:
            describe_capture(story, variation, variant, team_size, receiver_history, lineage)
            wall_started = time.monotonic()
            story.scene(pickup)
            query, decision, report = story.ask()
            finish, terminal, slot = reveal(story, decision)
            story.recorder.append(LABELS, "settled-event", labels_for(story, terminal, slot))
            story.recorder.meta.update(status="valid", measured_capture_wall_s=time.monotonic() - wall_started)
            story.recorder.close(query, decision, finish)
            return {"path": str(path), "team_size": team_size, "family": family, "variant": variant,
                    "contact_gate": story.recorder.meta["physical_story_gate"], "engineering_rule_decision": report["query_all"]["decision"],
                    "terminal_outcome": terminal["ground_truth"], "serialized_communication": report["cost"]}
        except (Exception, KeyboardInterrupt) as error:
            story.recorder.quarantine(str(error) or type(error).__name__)
            raise


def confirm_scope(client, ungranted, sender_frame):
    for frame, cutoff in ungranted:
        try:
            client.image(frame, until_ns=cutoff)
        except KeyError:
            pass
        else:
            raise ValueError(f"Replay leaked an out-of-scope frame from {sender_frame}")


def audit_camera(client, query, evaluator):
    packets = client.packets()
    first = packets[0]["cameras"][0]
    frame_id = first["frame_id"]
    stamps = [int(packet["source_time"]["monotonic_ns"]) for packet in packets]
    client.image(frame_id)
    confirm_scope(client, [(frame_id, stamps[0] - 1), (evaluator, None)], client.agent)
    confirm_scope(query, [(frame_id, None), (evaluator, None)], client.agent)
    return stamps, first["sha256"], frame_id


def audit_record(path):
    manifest = load(path / "manifest.json")
    grants = manifest["policy_cameras"]
    clients = {agent: Replay(path, agent) for agent in grants}
    query = clients[AGENTS[2]]
    evaluator = rows(path / TRUTH / "reveal.jsonl")[0]["camera"]["frame_id"]
    budget = manifest["fixed_observation_budget"]
    if budget["allocated_camera_samples"] != WINDOW_SLOTS * 2:
        raise ValueError("Fixed-total schedule has the wrong number of camera samples")
    observers = [agent for agent in grants if agent not in (AGENTS[0], AGENTS[2])]
    audited = {agent: audit_camera(clients[agent], query, evaluator) for agent in observers}
    if len(observers) > 1:
        peers = observers[1:] + observers[:1]
        for agent, peer in zip(observers, peers, strict=True):
            confirm_scope(clients[agent], [(audited[peer][2], None)], peer)
    counts = {agent: len(trace[0]) for agent, trace in audited.items()}
    hashes = {agent: trace[1] for agent, trace in audited.items()}
    planned = sorted((row["source_time_ns"], row["agent_id"])
                     for phase in ("before_relocation", "after_relocation") for row in budget[phase])
    actual = sorted((stamp, agent) for agent, trace in audited.items() for stamp in trace[0])
    if (planned != actual or len(actual) != budget["allocated_camera_samples"]
            or len(grants) != manifest["scaling_team_size"] or len(set(hashes.values())) != len(hashes)):
        raise ValueError("Separate camera samples or instantiated observers differ from the study design")
    receiver_stamps = verify_receiver_history(query, manifest, evaluator)
    return {"camera_samples_by_relocation_witness": counts, "total_granted_camera_samples": sum(
                len(packet["cameras"]) for client in clients.values() for packet in client.packets()),
            "first_scoped_rgb_sha256_by_relocation_witness": hashes,
            "observation_budget": budget, "live_observer_roles": list(grants), "native_time_and_cross_agent_scope_verified": True,
            "receiver_camera_times_ns": receiver_stamps,
            "measured_capture_wall_s": manifest["measured_capture_wall_s"]}


def verify_receiver_history(client, manifest, evaluator):
    packets = client.packets()
    stamps = [int(packet["source_time"]["monotonic_ns"]) for packet in packets]
    has_history = "scaling_receiver_profile" in manifest
    expected = 3 if has_history else 1
    if len(stamps) != expected or sorted(set(stamps)) != stamps or stamps[-1] != client.query:
        raise ValueError("Receiver's private history differs from the declared exposure schedule")
    if has_history and stamps[1] >= manifest["fixed_observation_budget"]["before_relocation"][0]["source_time_ns"]:
        raise ValueError("The receiver lost its separate pre-relocation exposure")
    for packet, stamp in zip(packets, stamps, strict=True):
        image = packet["cameras"][0]["frame_id"]
        client.image(image, until_ns=stamp)
        confirm_scope(client, [(image, stamp - 1), (evaluator, None)], client.agent)
    return stamps


def matching_budgets(records):
    schedule = []
    for result in records:
        budget = result["observation_budget"]
        times = tuple(tuple(row["source_time_ns"] for row in budget[phase])
                      for phase in ("before_relocation", "after_relocation"))
        schedule.append((times, tuple(result.get("receiver_camera_times_ns", ())), result["total_granted_camera_samples"]))
    if len(set(schedule)) != 1:
        raise ValueError("Teams have different total camera opportunities or simulation-time exposure windows")


def verify_hidden_pair(first, second):
    paths = [Path(row["path"]) for row in (first, second)]
    equal = runtime_signature(paths[0]) == runtime_signature(paths[1])
    predictions = [Replay(path, AGENTS[2]).predict() for path in paths]
    match = all(predictions[0][field] == predictions[1][field] for field in ("belief", "posterior", "decision"))
    observed = [row["engineering_rule_decision"] for row in (first, second)]
    if not equal or not match or first["terminal_outcome"] == second["terminal_outcome"]:
        raise ValueError("Held-back evaluator truth changed the permitted inputs or equivalent predictions, or did not vary")
    if observed != ["request_inspection"] * 2 or predictions[0]["decision"] != "request_inspection":
        raise ValueError("The F6 rehearsal should request inspection before an evaluator reveal")
    return True
