"""Persist explicit study allocation separately from the development deck."""
from collections import Counter
from hashlib import sha256
from .contract import json_bytes
from .episode import atomic, load
from .physical_story_plan import ROOT, story_card

REVISION = 'smart-swarm-physical-study-v1'
QUOTAS = {'train': (27, 27, 27, 27, 26, 26), 'validation': (7, 7, 7, 7, 6, 6),
          'test': (20, 20, 20, 20, 20, 20)}


def allocation():
    profile = load(ROOT / 'configs/physical-stories.json')
    cards = []
    for split, quotas in QUOTAS.items():
        for family, count in zip(profile['families'], quotas, strict=True):
            for number in range(count):
                group = f'{REVISION}-{split}-layout-{number + 1:03}'
                card = story_card(profile, number + 1, family, f'{REVISION}-{split}')
                cards.append({**card, 'story_id': f'{group}-{family.lower()}', 'source_group_id': group,
                              'split': split, 'collection_kind': 'event_memory', 'plan_version': REVISION})
    for number in range(24):
        group = f'{REVISION}-paired-layout-{number + 1:03}'
        for family in ('F1', 'F6'):
            card = story_card(profile, number + 1, family, group)
            cards.append({**card, 'story_id': f'{group}-{family.lower()}', 'source_group_id': group,
                          'split': 'test', 'collection_kind': 'paired_control', 'plan_version': REVISION})
    return cards


def prepare_study(config):
    directory = config['data_root'] / 'collection-plans'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{REVISION}.json'
    cards = allocation()
    digest = sha256(json_bytes(cards)).hexdigest()
    record = {'schema': REVISION, 'cards': cards, 'allocation_sha256': digest,
              'history_targets': {split: sum(counts) for split, counts in QUOTAS.items()},
              'family_quotas': {split: dict(zip(('F1','F2','F3','F4','F5','F6'), counts)) for split, counts in QUOTAS.items()},
              'paired_control_groups': 24, 'paired_control_histories': 48,
              'source_groups': dict(Counter(card['split'] for card in {c['source_group_id']: c for c in cards}.values())),
              'eligibility': 'Requires reviewed development deck and frozen verified station/protocol',
              'stopping': 'Stop each split at its accepted quota. Preserve incomplete attempts. No automatic expansion.',
              'rounding': 'Train 160 and validation 40 distribute remainders to F1-F4. Test has 20 per family.',
              'operator_time': {'status': 'provisional', 'basis': 'No attended pilot duration measured'}}
    if not path.exists():
        atomic(path, record)
    saved = load(path)
    if saved['allocation_sha256'] != digest or saved['cards'] != cards:
        raise ValueError('The frozen physical study allocation changed')
    return {**saved, 'path': str(path)}


def collection_progress(config):
    plan = prepare_study(config)
    statuses = Counter()
    for card in plan['cards']:
        path = config['data_root'] / 'physical-stories' / card['story_id'] / 'trial.json'
        statuses[load(path)['status'] if path.exists() else 'unstarted'] += 1
    return {key: value for key, value in plan.items() if key != 'cards'} | {'attempt_statuses': dict(statuses)}
