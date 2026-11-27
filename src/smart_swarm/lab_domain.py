"""Presentation-free Smart Swarm phases for Homebrain Lab and the terminal client."""
import time
from .episode import atomic, load, sha
from .physical_story_source import STATIONARY_LEAD_IN_S, SETTLED_TANK_S, FLEET_WINDOW_S

MIN_RELOCATION_S = 15


def relocation_readiness(state, now_ns, rgb_started):
    acknowledgement = state.get('fleet_acknowledgement')
    if acknowledgement is None:
        return {'ready': False, 'reason': 'Waiting for native recording acknowledgement', 'seconds_remaining': None}
    age = (now_ns - acknowledgement['monotonic_ns']) / 1e9
    remaining = state['fleet_window_s'] - age
    ready = age >= STATIONARY_LEAD_IN_S and rgb_started and remaining >= MIN_RELOCATION_S
    return {'ready': ready, 'seconds_remaining': max(0, round(remaining, 1)),
            'hold_seconds': max(0, round(STATIONARY_LEAD_IN_S - age, 1)),
            'reason': None if ready else 'Waiting for stationary lead-in and archived RGB with time for relocation',
            'settled_hold_seconds': SETTLED_TANK_S}


def session_state(config, task):
    if task.startswith('component-pilot-'):
        from .lab_pilot import pilot_path
        path = pilot_path(config, task)
        return load(path) if path.exists() else None
    path = config['data_root'] / 'physical-stories' / task / 'trial.json'
    return load(path) if path.exists() else None


def phase(config, task):
    state = session_state(config, task)
    if state is None:
        return 'setup', ['start'], 'Prepare the scene and confirm readiness before recording.'
    status, markers = state['status'], state['markers']
    if status in ('complete', 'incomplete', 'rejected'):
        return status, ['review'], 'Recording is saved. Read admission and outcome receipts before reset.'
    if status == 'pending_reveal':
        return status, ['reveal', 'close_incomplete'], 'Keep the relation fixed. Reveal the outcome only in the evaluator view.'
    if status == 'placement_recording':
        if 'placement_closed' in markers:
            return 'save_placement', ['finish_recording', 'start_mobile', 'close_incomplete'], 'Save the native episode and stop controls. Keep the carrier fixed.'
        return status, ['start_recording', 'mark', 'finish_recording', 'close_incomplete'], 'Use the native leader arm. Confirm the observed placement after withdrawal.'
    if status == 'preparing':
        return status, ['refresh'], 'Waiting for the native Explorer acknowledgement.'
    if 'relocation_closed' in markers:
        return 'settled_capture', ['predict', 'close_incomplete'], 'Keep the carrier still for Tank and then ROSMASTER until native readback ends.'
    if 'relocation_prompt_opened' in state:
        return 'relocation', ['mark', 'close_incomplete'], 'Move the carrier once. Confirm immediately when it settles, then keep it still.'
    return 'mobile_lead_in', ['grant_relocation', 'close_incomplete'], 'Keep the scene still while recorded RGB and stationary lead-in are checked.'


def attended_begin(config):
    from .attended_so101 import idle_hardware_snapshot
    from .physical_story_source import homebrain_boot
    from .studio import native_status
    root = config['data_root'] / 'lab'
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'attended-window.json'
    value = {'before': idle_hardware_snapshot(native_status(config)), 'started_monotonic_ns': time.monotonic_ns(),
             'homebrain_boot_id': homebrain_boot()}
    atomic(path, value)
    return {'receipt': str(path), 'sha256': sha(path)}


def attended_finish(config, observations):
    from .attended_so101 import idle_hardware_snapshot, verify_window, receipt_path, SCHEMA
    from .physical_story_source import homebrain_boot
    from .studio import native_status
    if observations != {'framing_tracking_stop_confirmed': True}:
        raise ValueError('Explicit attended framing, tracking and STOP observations are required')
    before = load(config['data_root'] / 'lab/attended-window.json')
    if before['homebrain_boot_id'] != homebrain_boot():
        raise ValueError('The attended check crossed a Homebrain reboot')
    after, finished = idle_hardware_snapshot(native_status(config)), time.monotonic_ns()
    verify_window(before['before'], after, before['started_monotonic_ns'], finished)
    path = receipt_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {**before, 'schema': SCHEMA, 'after': after, 'finished_monotonic_ns': finished,
              'operator_confirmed_framing_tracking_and_stop': True, 'recorded_unix_s': time.time(),
              'motion_requested_by_smart_swarm': False}
    atomic(path, record)
    return {'receipt': str(path), 'sha256': sha(path)}
