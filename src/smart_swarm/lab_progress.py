"""Read project counters from journals and admission receipts, never button clicks."""
from collections import Counter
from .episode import load


def progress(config):
    root = config['data_root']
    states = [load(path) for path in (root/'physical-stories').glob('*/trial.json')]
    history = root/'lab/corpus-progress.json'
    reviewed = load(history) if history.exists() else {}
    statuses = Counter(row['status'] for row in states)
    native = {row['native_episode_id'] for row in states if row.get('native_episode_id') and row.get('native_story_validation')}
    return {'contract_version':'homebrain.lab.v1','scope':'Smart Swarm physical stories', 'attempts':len(states),
            'saved_raw':len(native),'technically_valid':reviewed.get('admitted_histories',0),
            'admitted_story':reviewed.get('admitted_histories',0),
            'successful_unassisted_motor':reviewed.get('accepted_motor_prefixes',0),
            'assisted':sum(any('human' in (mark.get('assistance') or '') for mark in row.get('markers',{}).values()) for row in states),
            'valid_failures':reviewed.get('valid_failures',0),'unknown':sum(row.get('observed_terminal_outcome')=='unknown' for row in states),
            'pending':sum(statuses[name] for name in ('preparing','placement_recording','recording','pending_reveal')),
            'incomplete':statuses['incomplete'], 'denominators':{'development_attempts':24,'study_histories':320,'paired_controls':48},
            'source_groups':{'attempted':len({row['card']['source_group_id'] for row in states})}}
