"""Noninteractive JSON facade for homebrain.lab.v1, with durable action correlation."""
from concurrent.futures import ThreadPoolExecutor
from fcntl import flock, LOCK_EX, LOCK_NB, LOCK_UN
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import subprocess
import re
from .episode import atomic, load
from .lab_cards import API, PROJECT, all_cards, descriptor, revision, station_card, task_card, task_list
from .lab_domain import session_state
from .lab_store import ActionStore
from . import lab_actions

FIELDS = {'project_id','session_id','target_id','action','expected_revision','idempotency_key','arguments'}


class ProjectFacade:
    def __init__(self, config):
        self.config = config
        self.root = config['data_root'] / 'lab'
        self.store = ActionStore(self.root / 'actions.sqlite')
        self.owner = (self.root / 'facade.lock').open('a')
        flock(self.owner, LOCK_EX | LOCK_NB)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='smart-swarm-domain')
        self.reconcile()

    def reconcile(self):
        for identifier, request in self.store.interrupted():
            result = {'status': 'unknown', 'operation_id': identifier,
                      'resulting_revision': revision(self.config, request['target_id']),
                      'reason': {'code': 'PROCESS_INTERRUPTED', 'message': 'Inspect project/native receipts before another side effect'},
                      'native_acknowledgement': None, 'receipts': []}
            self.store.complete(identifier, result)

    def snapshot(self, target='station-setup'):
        state = session_state(self.config, target)
        card = self.present_card(target)
        receipts = []
        trial = self.config['data_root'] / 'physical-stories' / target
        for name in ('trial.json','motor-intake.json','motor-segment.json','blind-offline-forecast.json'):
            path = trial / name
            if path.is_file():
                receipts.append({'path': str(path), 'sha256': sha256(path.read_bytes()).hexdigest()})
        return {'api_version': API, 'project_id': PROJECT, 'task_id': target, 'attempt_id': target if state else None,
                'session_id': target, 'revision': revision(self.config, target), 'step': card['step'],
                'instruction': card['instruction'], 'allowed_actions': card['allowed_actions'], 'receipts': receipts,
                'native_ids': {key: state.get(key) for key in ('studio_dataset_id','native_episode_id','fleet_capture_id')} if state else {},
                'pending_work': self.store.interrupted(), 'readiness': self.readiness(), 'blockers': self.blockers(card)}

    def present_card(self, target):
        if target == 'station-setup':
            return station_card()
        if target.startswith('component-pilot-'):
            from .lab_pilot import pilot_cards
            card, = [row for row in pilot_cards(self.config) if row['task_id'] == target]
            return card
        return task_card(self.config, self.card(target))

    def card(self, identifier):
        matches = [card for card in all_cards(self.config) if card['story_id'] == identifier]
        if len(matches) != 1:
            raise ValueError('Unknown task ID')
        return matches[0]

    def readiness(self):
        path = self.root / 'last-readiness.json'
        return load(path) if path.exists() else {'status': 'not_checked', 'action': 'readiness'}

    def blockers(self, card):
        from .physical_story_source import homebrain_boot
        path = self.root / 'station-profile.json'
        if card.get('dataset_intent') in ('technical_setup','technical_operator_pilot'):
            return []
        station = load(path) if path.exists() else {}
        current = station.get('distributed_verified') and station.get('homebrain_boot_id') == homebrain_boot()
        return [] if current else ['DISTRIBUTED_SIGHTLINES_UNVERIFIED']

    def action(self, request):
        validate_request(request)
        existing = self.store.existing(request)
        if existing is not None:
            return existing
        if self.store.interrupted():
            raise ValueError('A project operation is pending. Read its progress before another action')
        if self.store.interrupted(('unknown',)) and request['action'] not in ('readiness','refresh'):
            raise ValueError('An uncertain operation requires receipt reconciliation before another side effect')
        snapshot = self.snapshot(request['target_id'])
        if request['action'] not in snapshot['allowed_actions']:
            raise ValueError('Action is not allowed by the current project journal')
        if request['action'] == 'start' and snapshot['blockers']:
            raise ValueError('Verify table/floor witness and evaluator sightlines before distributed capture')
        result, fresh = self.store.reserve(request, snapshot['revision'])
        if fresh:
            self.executor.submit(self.perform, result['operation_id'], request)
        return result

    def perform(self, identifier, request):
        try:
            config = {key: str(value) if isinstance(value, Path) else value for key, value in self.config.items()}
            path = self.root / f'{identifier}-request.json'
            atomic(path, {'request': request, 'config': config})
            destination = self.root / f'{identifier}-worker.json'
            with (self.root / f'{identifier}.log').open('a') as log:
                done = subprocess.run([sys.executable, '-m', 'smart_swarm.lab_worker', str(path), str(destination)],
                                      stdout=log, stderr=subprocess.STDOUT, check=True)
            response = load(destination)
            if response['status'] != 'accepted':
                result = {'status': response['status'], 'reason': {'code': 'DOMAIN_FAILURE', 'message': response['reason']},
                          'native_acknowledgement': None, 'receipts': [str(destination)]}
                result.update(operation_id=identifier, resulting_revision=revision(self.config, request['target_id']))
                return self.store.complete(identifier, result)
            value = response['value']
            if request['action'] == 'readiness':
                atomic(self.root / 'last-readiness.json', value)
            result = {'status': 'accepted', 'reason': None, 'native_acknowledgement': value,
                      'receipts': [str(self.root / f'{identifier}.json')]}
        except Exception as error:
            state = 'unknown' if isinstance(error, (TimeoutError, OSError, lab_actions.StillPending)) else 'rejected'
            result = {'status': state, 'reason': {'code': type(error).__name__, 'message': str(error)},
                      'native_acknowledgement': None, 'receipts': []}
        result.update(operation_id=identifier, resulting_revision=revision(self.config, request['target_id']))
        atomic(self.root / f'{identifier}.json', result)
        self.store.complete(identifier, result)

    def call(self, message):
        method, args = message['method'], message.get('params', {})
        from .lab_jobs import recipes
        handlers = {'descriptor': lambda: descriptor(), 'tasks': lambda: task_list(self.config, **args), 'jobs': recipes,
                    'snapshot': lambda: self.snapshot(**args), 'action': lambda: self.action(args),
                    'events': lambda: self.store.events(**args), 'operation': lambda: self.store.operation(**args),
                    'reconcile': lambda: self.reconcile_operation(**args)}
        if method not in handlers:
            raise ValueError('Unknown facade method')
        return handlers[method]()

    def reconcile_operation(self, identifier):
        from .lab_reconcile import evidence
        saved = self.store.operation(identifier)
        if saved['status'] != 'unknown':
            return saved
        request, = [row for key, row in self.store.interrupted(('unknown',)) if key==identifier]
        proof = evidence(self.config, request)
        if proof is None:
            return saved
        result = {**saved, 'status':'accepted', 'reason':None, 'native_acknowledgement':proof,
                  'resulting_revision':revision(self.config, request['target_id'])}
        return self.store.complete(identifier, result)

    def close(self):
        self.executor.shutdown(wait=True)
        flock(self.owner, LOCK_UN)
        self.owner.close()


def validate_request(request):
    if set(request) != FIELDS or request['project_id'] != PROJECT or not isinstance(request['arguments'], dict):
        raise ValueError('Invalid homebrain.lab.v1 action fields or project identity')
    for field in FIELDS - {'arguments'}:
        if not isinstance(request[field], str) or not 1 <= len(request[field]) <= 256:
            raise ValueError(f'Invalid action field {field}')
    if request['session_id'] != request['target_id']:
        raise ValueError('Session must reference its allocated project task')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,150}', request['target_id']):
        raise ValueError('Invalid project task identifier')


def main():
    from .cli import settings
    from .lab_api import HomebrainProject
    facade = HomebrainProject(settings())
    try:
        for line in sys.stdin:
            try:
                message = json.loads(line)
                reply = {'id': message.get('id'), 'result': facade.call(message)}
            except (ValueError, KeyError, TypeError, OSError) as error:
                reply = {'error': {'code': type(error).__name__, 'message': str(error)}}
            sys.stdout.write(json.dumps(reply, allow_nan=False) + '\n')
            sys.stdout.flush()
    finally:
        facade.close()


if __name__ == '__main__':
    main()
