"""Expose the four existing SO101 component attempts without mobile dependencies."""
from .episode import atomic, load
from .studio import native_binding, native_request, native_status, pending_outcome, label_attempt, sync_pilot
from .attended_so101 import pilot_readiness


def pilot_cards(config=None):
    from .so101_pilot import prepare
    group = prepare(config)['scene_group_id'] if config else 'component-attended-session'
    return [{'api_version': 'homebrain.lab.v1', 'project_id': 'smart-swarm', 'task_id': f'component-pilot-{i}',
             'plan_revision': 'so101-component-v1', 'protocol_revision': 'robot-platform/1.0.0',
             'title': f'SO101 component placement {i} of 4', 'dataset_intent': 'technical_operator_pilot',
             'source_group_id': group, 'split': 'development',
             'required_robots': ['SO101'], 'ordered_cameras': ['so101-1/wrist','so101-1/top'],
             'setup': ['One cube and one open container within the verified table reach.',
                       'Restore the taped start and keep the final result visible until outcome confirmation.'],
             'setup_media': ['docs/setup/table-floor-station.svg'], 'step': 'component_pilot',
             'instruction': 'Start native recording, demonstrate one placement, save, stop and confirm the actual outcome.',
             'native_control_profile': 'existing Studio leader and STOP', 'outcomes': ['placement_succeeded','placement_failed','unknown'],
             'validation': 'Exact native saved episode, wrist/top order, both calibrations and observed assistance',
             'allowed_actions': ['start','start_recording','finish_recording','stop_recording','observe_pilot','review_pilot']}
            for i in range(1,5)]


def pilot_path(config, task):
    if task not in {row['task_id'] for row in pilot_cards()}:
        raise ValueError('Unknown SO101 component attempt')
    return config['data_root'] / 'lab' / f'{task}.json'


def start(config, task):
    path = pilot_path(config, task)
    if path.exists():
        return load(path)
    ready = pilot_readiness(config)
    if ready['status'] != 'READY_FOR_SO101_PILOT':
        raise ValueError('Complete the current attended SO101 check before recording')
    binding = native_binding(config)
    records = native_request(config, f"/api/episodes/{binding['dataset_id']}?include_frames=false")
    if len(records) != int(task.rsplit('-',1)[1])-1:
        raise ValueError('Select the next SO101 attempt in the four-attempt pilot')
    state = {'status': 'placement_recording', 'studio_dataset_id': binding['dataset_id'],
             'preexisting_native_episode_ids': [row['id'] for row in records], 'markers': {}, 'task_id': task}
    atomic(path, state)
    return state


def observe(config, task, args):
    path = pilot_path(config, task)
    state = load(path)
    binding = native_binding(config)
    candidate = pending_outcome(config, binding)
    if candidate is None or candidate['recording']['episode_id'] in state['preexisting_native_episode_ids']:
        raise ValueError('Wait for this native episode to save and controls to stop before outcome review')
    result = label_attempt(config, binding, candidate, args['outcome'], assistance=args['assistance'])
    state.update(status='complete', native_episode_id=candidate['recording']['episode_id'], observed_outcome=args['outcome'])
    atomic(path, state)
    return result
