"""Typed task descriptions, source allocations and eligibility without a project UI."""
from hashlib import sha256
from .contract import CONTRACT_SHA, VERSION, json_bytes
from .episode import load
from .physical_story_plan import prepare
from .study_collection import prepare_study
from .lab_domain import phase

API = 'homebrain.lab.v1'
PROJECT = 'smart-swarm'


def descriptor():
    return {'api_version': API, 'project_id': PROJECT, 'version': '0.2.0',
            'protocol_revisions': [VERSION, 'physical-stories-v1', 'smart-swarm-physical-study-v1'],
            'contract_sha256': CONTRACT_SHA, 'modes': ['hardware', 'mujoco', 'replay', 'fixture'],
            'task_groups': ['station_setup', 'component_pilot', 'development', 'study', 'paired_control'],
            'required_capabilities': ['studio.record', 'studio.control', 'studio.stop', 'explorer.capture', 'compute.reserve'],
            'plugin_entrypoint': 'smart_swarm.lab_api:HomebrainProject',
            'transport': 'stdio-jsonl', 'schema': 'schemas/homebrain.lab.v1'}


def all_cards(config):
    original = prepare(config)['cards']
    path = config['data_root'] / 'collection-plans/smart-swarm-physical-study-v1.json'
    return original + (load(path)['cards'] if path.exists() else [])


def task_card(config, raw):
    step, actions, instruction = phase(config, raw['story_id'])
    return {'api_version': API, 'project_id': PROJECT, 'task_id': raw['story_id'],
            'plan_revision': raw.get('plan_version', 'physical-stories-v1'), 'protocol_revision': VERSION,
            'dataset_intent': raw.get('collection_kind', 'development'), 'source_group_id': raw['source_group_id'],
            'split': raw['split'], 'family': raw['family'], 'required_robots': ['SO101', 'Tank', 'ROSMASTER'],
            'ordered_cameras': ['so101-1/wrist', 'so101-1/top', 'tank-1/front', 'rosmaster-1/front'],
            'title': raw['task'], 'setup': raw['placement_regions'], 'placement': raw['placement'],
            'relocation': raw['relocation'], 'withheld': raw['withheld'], 'reset': raw['reset'],
            'setup_media': ['docs/setup/table-floor-station.svg'], 'native_control_profile': 'existing leader-arm and native STOP',
            'step': step, 'instruction': instruction, 'allowed_actions': actions,
            'outcomes': ['placement_succeeded', 'placement_failed', 'unknown'],
            'validation': 'Native finalized RGB/state/actions and causal clocks, then source-group scientific admission',
            'assistance': 'Record human placement or relocation explicitly. Complete failed placement is story data.'}


def station_card():
    return {'api_version': API, 'project_id': PROJECT, 'task_id': 'station-setup', 'title': 'Verify the SO101 station',
            'dataset_intent': 'technical_setup', 'source_group_id': 'station-qualification', 'split': 'development',
            'plan_revision': 'station-v1', 'protocol_revision': VERSION,
            'required_robots': ['SO101'], 'ordered_cameras': ['so101-1/wrist', 'so101-1/top'],
            'setup': ['Place one cube and an open carrier on the table within the already calibrated SO101 reach.',
                      'Check pickup, release and final containment in wrist and top views.',
                      'For mobile stories verify that Tank and ROSMASTER can see the chosen destination and evaluator scene.'],
            'setup_media': ['docs/setup/table-floor-station.svg'], 'step': 'attended_setup',
            'instruction': 'Check camera framing, follow the native leader and use STOP hands. Confirm only after readback.',
            'allowed_actions': ['capture_station', 'begin_attended_check', 'confirm_attended_check', 'readiness', 'submit_job', 'confirm_station', 'freeze_study'],
            'native_control_profile': 'existing Studio controls', 'outcomes': ['verified', 'unverified'],
            'validation': 'New native following and calibration readback followed by a new confirmed STOP'}


def revision(config, task):
    if task.startswith('component-pilot-'):
        from .lab_pilot import pilot_path
        path = pilot_path(config, task)
    else:
        path = config['data_root'] / 'physical-stories' / task / 'trial.json'
    value = path.read_bytes() if path.exists() else json_bytes({'task': task, 'state': 'unstarted'})
    return sha256(value).hexdigest()


def task_list(config, group='development', offset=0, limit=24):
    if not 0 <= offset or not 1 <= limit <= 50:
        raise ValueError('Task pagination requires offset >= 0 and limit 1..50')
    if group == 'station_setup':
        return {'tasks': [station_card()], 'total': 1}
    if group == 'component_pilot':
        from .lab_pilot import pilot_cards
        return {'tasks': pilot_cards(config)[offset:offset+limit], 'total': 4}
    if group in ('study', 'paired_control'):
        prepare_study(config)
    cards = all_cards(config)
    selected = [row for row in cards if (row.get('collection_kind', 'development') == group
                 or (group == 'study' and row.get('collection_kind') == 'event_memory'))]
    return {'tasks': [task_card(config, row) for row in selected[offset:offset+limit]], 'total': len(selected)}
