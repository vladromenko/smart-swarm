"""Allowlisted project actions using the existing native and scientific operations."""
import time
from .episode import atomic, load, sha
from .lab_domain import attended_begin, attended_finish, relocation_readiness, session_state
from .physical_story_source import begin, mark, start_relocation, claim_relocation_prompt
from .physical_story_replay import recover_story, predict_selected, terminal_reveal, close_interrupted_story


class StillPending(ValueError):
    """A native operation is still running; reconcile its receipts before retrying."""


def require_study_release(config, task):
    from .lab_cards import all_cards
    card, = [row for row in all_cards(config) if row['story_id'] == task]
    if card.get('plan_version'):
        from .lab_release import verify
        verify(config)


def native_record(config, task, action, args):
    from .studio import native_request, native_status
    state = session_state(config, task)
    raw = native_request(config, '/api/status')['recording']
    dataset = state['studio_dataset_id']
    if action == 'start_recording':
        opened = config['data_root'] / 'lab' / f'{task}-native-start.json'
        if opened.exists() or raw.get('active') or args != {'explicit_user_start': True}:
            raise ValueError('Recording needs an explicit operator start and idle native writer')
        status = native_status(config)
        result = native_request(config, '/api/record/start', {'dataset_id': dataset, 'episode_count':1, 'auto_advance':False,
                                'expected_stop_generation': status['native_stop_generation']})
        opened.parent.mkdir(parents=True,exist_ok=True)
        atomic(opened, {'acknowledgement':result,'dataset_id':dataset})
        return result
    if not raw.get('active') or raw.get('dataset_id') != dataset:
        raise ValueError('The native writer does not belong to this project attempt')
    if raw.get('episode_id') in state['preexisting_native_episode_ids']:
        raise ValueError('The native episode predates this attempt')
    if action == 'stop_recording':
        if args != {'explicit_user_stop': True}:
            raise ValueError('Explicit native STOP is required')
        return native_request(config, '/api/record/stop', {})
    acknowledgement = native_request(config, '/api/record/end', {})
    path = config['data_root'] / 'lab' / f'{task}-native-finish.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic(path, {'acknowledgement': acknowledgement, 'native_episode_id': raw.get('episode_id'),
                  'meaning': 'Native save requested. Admission awaits completed writer and source readback.'})
    return acknowledgement


def relocation(config, task):
    from .physical_story_operator import archived_rgb_started
    from .physical_story_source import fleet_manifest
    state = session_state(config, task)
    if 'relocation_prompt_opened' in state:
        return {'prompt_already_granted': True}
    ready = relocation_readiness(state, time.monotonic_ns(), archived_rgb_started(config, state))
    if not ready['ready'] or fleet_manifest(config, state['fleet_capture_id'])['phase'] != 'recording':
        raise StillPending(ready['reason'] or 'Native capture ended')
    saved, granted = claim_relocation_prompt(config, task, state['fleet_capture_id'])
    return {'prompt_granted': granted, 'receipt': str(config['data_root'] / 'physical-stories' / task / 'trial.json')}


def review(config):
    from .physical_story_motor import review_pending_motor
    from .physical_evidence_dataset import publish
    from .physical_motor_dataset import publish as motors
    result = review_pending_motor(config)
    histories, prefixes = publish(config), motors(config)
    record = {'motor_review': result, 'admitted_histories': histories['admitted_histories'],
            'accepted_motor_prefixes': prefixes['prefix_count'], 'exclusions': histories['exclusions'],
            'evidence_manifest': histories['manifest'], 'motor_manifest': prefixes['manifest']}
    atomic(config['data_root'] / 'lab/corpus-progress.json', record)
    return record


def execute(config, request):
    task, name, args = request['target_id'], request['action'], request['arguments']
    if task.startswith('component-pilot-') and name not in ('start_recording','finish_recording','stop_recording'):
        from .lab_pilot import start, observe
        from .studio import sync_pilot, native_binding
        if name == 'start':
            return start(config, task)
        if name == 'observe_pilot':
            return observe(config, task, args)
        if name == 'review_pilot':
            return sync_pilot(config, native_binding(config))
    if name == 'start':
        require_study_release(config, task)
        return begin(config, task, staged=True, evaluation_mode=args.get('evaluation_mode', 'blind_offline'))
    if name in ('start_recording', 'finish_recording','stop_recording'):
        return native_record(config, task, name, args)
    if name == 'mark':
        return mark(config, task, **args)
    if name == 'start_mobile':
        return start_relocation(config, task)
    if name == 'grant_relocation':
        return relocation(config, task)
    if name == 'predict':
        state = session_state(config, task)
        from .physical_story_source import fleet_manifest, FLEET_TERMINAL
        if fleet_manifest(config, state['fleet_capture_id'])['phase'] not in FLEET_TERMINAL:
            raise StillPending('Waiting for native archive finalization')
        return predict_selected(config, task)
    if name == 'reveal':
        return terminal_reveal(config, task, args['outcome'], args['relations_preserved'])
    if name == 'close_incomplete':
        return close_interrupted_story(config, task, args['reason'])
    return auxiliary(config, request)


def auxiliary(config, request):
    action, args = request['action'], request['arguments']
    if action == 'submit_job':
        from .lab_jobs import submit_motor
        return submit_motor(config, args)
    if action == 'freeze_study':
        from .lab_release import freeze
        return freeze(config)
    if action == 'readiness':
        from .attended_so101 import pilot_readiness
        from .physical_story_source import story_readiness
        return {'component': pilot_readiness(config), 'distributed': story_readiness(config)}
    if action == 'begin_attended_check':
        return attended_begin(config)
    if action == 'confirm_attended_check':
        return attended_finish(config, args)
    if action == 'capture_station':
        from .lab_station import capture
        return capture(config)
    if action == 'confirm_station':
        from .lab_station import confirm
        return confirm(config, args)
    if action == 'review':
        return review(config)
    if action == 'refresh':
        trial = config['data_root'] / 'physical-stories' / request['target_id']
        return recover_story(config, trial)
    raise ValueError('Unsupported project action')
