"""Exercise installed Studio and LeRobot timing paths with memory-only test devices."""

import argparse
from importlib.metadata import version
import json
from math import ceil
from pathlib import Path
from signal import SIGINT, raise_signal
import subprocess
from threading import Event
from time import monotonic, monotonic_ns, sleep
from types import SimpleNamespace
from unittest.mock import patch

from smart_swarm.cli import ROOT, settings
from smart_swarm.episode import atomic, load, sha
from smart_swarm.research_jobs import admitted
from smart_swarm.studio import native_status, studio_env

FPS = 30
CHUNK = 50
PILOT = ROOT / "evidence/learning-pilot-v3"
TIMING_SHA = "e5031a2d411c9ee1ec40c68688ca793fe0e6b0e6d8eca7dba038d0e653a1336e"


class StubPolicy:
    def __init__(self, delay, after_prediction=None):
        self.delay = delay
        self.after_prediction = after_prediction
        self.resets = 0
        self.config = SimpleNamespace(use_amp=False)

    def reset(self):
        self.resets += 1

    def select_action(self, observation):
        import torch

        sleep(self.delay)
        if self.after_prediction is not None:
            self.after_prediction()
        return torch.zeros((1, 6), dtype=torch.float32)

    def predict_action_chunk(self, observation, **kwargs):
        import torch

        sleep(self.delay)
        return torch.arange(CHUNK, dtype=torch.float32).reshape(1, CHUNK, 1).expand(1, CHUNK, 6)


def passthrough(value):
    return value


def stub_processors(postprocess=passthrough):
    for processor in (passthrough, postprocess):
        processor.steps = ()
        processor.reset = lambda: None
    return passthrough, postprocess


def official_engine(config, policy, shutdown, postprocess=passthrough):
    from lerobot.rollout.inference import create_inference_engine
    from lerobot.utils.feature_utils import hw_to_dataset_features
    from so101_lab.hardware import JOINTS

    keys = [f"{name}.pos" for name in JOINTS]
    local = dict.fromkeys(keys, float)
    observations = hw_to_dataset_features(local, "observation")
    features = {**observations, **hw_to_dataset_features(local, "action")}
    pre, post = stub_processors(postprocess)
    engine = create_inference_engine(config, policy=policy, preprocessor=pre, postprocessor=post,
                                     robot_wrapper=SimpleNamespace(robot_type="so101_follower"),
                                     hw_features=observations, dataset_features=features, ordered_action_keys=keys,
                                     task="Technical memory fixture", fps=FPS, device="cpu", shutdown_event=shutdown)
    return engine, keys, features


def synchronous_case(delay, stop_during_inference):
    from lerobot.rollout.inference import SyncInferenceConfig
    from lerobot.rollout.strategies.base import BaseStrategy
    from lerobot.utils.process import ProcessSignalHandler

    handler = ProcessSignalHandler(use_threads=True, display_pid=False) if stop_during_inference else None
    shutdown = handler.shutdown_event if handler else Event()
    policy = StubPolicy(delay, (lambda: raise_signal(SIGINT)) if handler else None)
    engine, keys, features = official_engine(SyncInferenceConfig(), policy, shutdown)
    calls, observations = [], []

    def observe():
        observations.append(monotonic_ns())
        return dict.fromkeys(keys, 0.)

    def receive(action):
        calls.append({"age_ms": (monotonic_ns() - observations[-1]) / 1e6, "shutdown_event_set": shutdown.is_set(),
                      "action_keys": list(action)})
        shutdown.set()

    robot = SimpleNamespace(inner=SimpleNamespace(is_connected=False), send_action=receive, get_observation=observe)
    config = SimpleNamespace(interpolation_multiplier=1, fps=FPS, duration=.5, use_torch_compile=False,
                             return_to_initial_position=False, display_data=False)
    ctx = SimpleNamespace(runtime=SimpleNamespace(cfg=config, shutdown_event=shutdown),
                          policy=SimpleNamespace(inference=engine), hardware=SimpleNamespace(robot_wrapper=robot, teleop=None),
                          processors=SimpleNamespace(robot_observation_processor=passthrough, robot_action_processor=lambda p: p[0]),
                          data=SimpleNamespace(dataset_features=features, ordered_action_keys=keys))
    strategy = BaseStrategy(config)
    strategy.setup(ctx)
    try:
        strategy.run(ctx)
    finally:
        strategy.teardown(ctx)
    if len(calls) > 1 or any(row["action_keys"] != keys for row in calls) or policy.resets != 1:
        raise ValueError("The official synchronous fixture observed an unexpected command or reset")
    return {"injected_delay_s": delay, "cooperative_stop_during_inference": stop_during_inference,
            "owned_process_sigint_count": handler.counter if handler else 0,
            "memory_sink_calls": calls, "policy_reset_count": policy.resets, "teardown_motion_tested": False}


def rtc_case(delay, execution_horizon, guidance):
    from lerobot.policies.rtc.configuration_rtc import RTCConfig
    from lerobot.rollout.inference import RTCInferenceConfig

    def pause_after_prediction(actions):
        engine.pause()
        return actions

    cfg = RTCConfig(execution_horizon=execution_horizon, max_guidance_weight=guidance)
    engine, keys, _ = official_engine(RTCInferenceConfig(rtc=cfg), StubPolicy(delay), Event(), pause_after_prediction)
    engine.start()
    try:
        engine.notify_observation(dict.fromkeys(keys, 0.))
        engine.resume()
        queue = engine.action_queue
        deadline = monotonic() + 6
        while queue.queue is None and not engine.failed and monotonic() < deadline:
            sleep(.002)
        if engine.failed or queue.queue is None:
            raise RuntimeError("The native RTC fixture did not publish its action queue")
        remaining = queue.qsize()
        first = engine.get_action(None)
        engine.reset()
        cleared = engine.get_action(None) is None
        if not cleared or (first is not None and int(first[0]) != CHUNK - remaining):
            raise ValueError("The native RTC queue did not preserve the retained action index")
    finally:
        engine.stop()
    return {"injected_delay_s": delay, "predicted_actions": CHUNK, "retained_actions": remaining,
            "discarded_prefix": CHUNK - remaining, "first_retained_index": int(first[0]) if first is not None else None,
            "reset_emptied_queue": cleared}


def studio_age_case():
    import numpy as np
    from so101_lab import hardware, inference

    lifecycle = []

    class MemorySource:
        def __init__(self, settings, teleoperate):
            self.calibration = {"values": {"follower": {name: {"range_min": 0, "range_max": 4095} for name in hardware.JOINTS}}}

        def start(self, **kwargs):
            lifecycle.append("start")

        def observe(self):
            stamp = monotonic_ns() - 3_000_000_000
            sample = {"image": np.zeros((4, 4, 3), dtype=np.uint8), "timestamp_ns": stamp}
            return {"timestamp_ns": stamp, "observation.state": [0.] * 6, "images": {"wrist": sample, "top": sample}}

        def send(self, action, observation):
            lifecycle.append("memory_send")
            return {"action": action}

        def close(self):
            lifecycle.append("close")
            return []

    loaded = SimpleNamespace(contract=SimpleNamespace(mode="rgb_state", cameras=("wrist", "top")),
                             policy=SimpleNamespace(name="act"), checkpoint=Path("memory"), predict=lambda *a: [0.] * 6)
    with patch.object(hardware, "HardwareSource", MemorySource):
        try:
            inference._run_policy_loaded(loaded, {"explicit_user_start": True, "max_steps": 1}, None, lambda: False)
        except RuntimeError as error:
            if lifecycle != ["start", "close"]:
                raise
            return {"default_observation_age_limit_ms": inference.MAX_OBSERVATION_AGE_MS,
                    "test_observation_age_ms": 3000, "memory_device_lifecycle": lifecycle, "refusal": str(error)}
    raise ValueError("The direct Studio fixture accepted the old observation")


def read_native_profiles(checkpoints):
    from so101_lab.inference import load_checkpoint, official_lerobot_command

    report = {}
    for kind, path in checkpoints.items():
        seal = path.parent / "checkpoint.json"
        if seal.exists():
            raise ValueError("Recheck this test against the newly installed checkpoint registration")
        try:
            load_checkpoint(path, device="cpu")
        except FileNotFoundError as error:
            if Path(error.filename) != seal:
                raise
        else:
            raise ValueError("The native loader accepted an unregistered Smart Swarm checkpoint")
        command = official_lerobot_command(path, "Technical memory fixture", FPS, .5, "cpu", policy_type=kind)
        options = dict(argument[2:].split("=", 1) for argument in command if argument.startswith("--") and "=" in argument)
        report[kind] = {"checkpoint": str(path), "native_registration_missing": str(seal),
                        "official_engine": options["inference.type"],
                        "official_cameras": list(json.loads(options["robot.cameras"])),
                        "return_to_initial_position": options["return_to_initial_position"],
                        "rtc_execution_horizon": options.get("inference.rtc.execution_horizon"),
                        "rtc_max_guidance_weight": options.get("inference.rtc.max_guidance_weight")}
    return report


def measured_latency_queue(timing_path, horizon, guidance):
    import torch
    from lerobot.policies.rtc.action_queue import ActionQueue
    from lerobot.policies.rtc.configuration_rtc import RTCConfig

    if sha(timing_path) != TIMING_SHA:
        raise ValueError("The original SmolVLA production timing receipt changed")
    data = load(timing_path)
    queue = ActionQueue(RTCConfig(execution_horizon=horizon, max_guidance_weight=guidance))
    commands = torch.zeros((CHUNK, 6))
    results = []
    for row in data["requests"]:
        measured = row["complete_worker_ns"]
        skip = ceil(measured * FPS / 1e9)
        queue.clear()
        queue.merge(commands, commands, skip)
        results.append({"issue": row["issue"], "saved_cpu_latency_s": measured / 1e9,
                        "expired_indices_at_30hz": skip, "actions_remaining": queue.qsize()})
    return results


def native_probe():
    from lerobot.rollout.inference import rtc, sync
    from lerobot.rollout.strategies import base, core
    from lerobot.utils import process
    from so101_lab import inference

    handoff = load(ROOT / "handover/act-clocked-execution-handoff.json")
    inventory = load(PILOT / "training-inventory-final.json")
    smol, = [job for job in inventory["jobs"] if job.get("optimizer_updates") == 85 and job.get("model") == "smolvla"]
    paths = {"act": Path(handoff["manifest"]["checkpoint"]), "smolvla": Path(smol["checkpoint"])}
    profiles = read_native_profiles(paths)
    timing = profiles["smolvla"]
    horizon, guidance = int(timing["rtc_execution_horizon"]), float(timing["rtc_max_guidance_weight"])
    if profiles["act"]["official_engine"] != "sync" or timing["official_engine"] != "rtc":
        raise ValueError("The installed Studio routes changed")
    static = measured_latency_queue(PILOT / "smolvla-cpu-chunk-stage-timing.json", horizon, guidance)
    return {"scope": "Installed software paths with memory-only devices and synthetic action tensors",
            "lerobot_version": version("lerobot"), "implementation_sha256": {unit.__name__: sha(Path(unit.__file__))
             for unit in (rtc, sync, base, core, process, inference)}, "candidates": profiles,
            "studio_direct_observation_age": studio_age_case(),
            "official_sync": {"immediate_result": synchronous_case(0., False), "delayed_result": synchronous_case(.09, False),
                              "cooperative_stop": synchronous_case(.09, True)},
            "official_rtc": {"short_delay": rtc_case(.06, horizon, guidance), "over_full_chunk": rtc_case(2., horizon, guidance),
                             "archived_cpu_latency_applied_to_queue": static, "smolvla_timing_sha256": TIMING_SHA},
            "physical_actuator_requests": 0, "model_optimizer_updates": 0, "hardware_execution_qualified": False}


def run(config):
    before = native_status(config)
    with admitted(config, "smart-swarm-native-policy-timing-fixture", archive_only=True) as (grant, heartbeat):
        env = {**studio_env(config), "PYTHONPATH": str(config["studio_project"] / "src") + ":" + str(ROOT / "src")}
        child = subprocess.run([config["studio_python"], __file__, "--native-probe"], env=env,
                               capture_output=True, text=True, timeout=60)
        if child.returncode:
            raise RuntimeError(child.stderr)
        heartbeat()
    after = native_status(config)
    if before["native_stop_generation"] != after["native_stop_generation"]:
        raise ValueError("Native STOP generation changed during the fixture")
    return {**json.loads(child.stdout), "reservation": grant, "native_stop_generation": after["native_stop_generation"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--native-probe", action="store_true")
    mode.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.native_probe:
        print(json.dumps(native_probe()))
    else:
        if args.output.exists():
            parser.error("Choose a new timing receipt path")
        atomic(args.output, run(settings()))
        print(args.output)


if __name__ == "__main__":
    main()
