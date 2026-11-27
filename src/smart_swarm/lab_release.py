"""Release immutable physical study assignments only after observed development qualification."""
from .episode import atomic, load, sha
from .study_collection import prepare_study
from .physical_story_source import homebrain_boot


def freeze(config):
    from .physical_evidence_dataset import publish
    from .physical_learned import PROFILE
    station = config['data_root'] / 'lab/station-profile.json'
    if not station.exists() or load(station).get('homebrain_boot_id') != homebrain_boot():
        raise ValueError('A current attended distributed station qualification is required')
    corpus = publish(config)
    development = [row for row in corpus['records'] if row['split']=='development'
                   and row.get('condition_audit',{}).get('status') in ('observed','withholding_only')]
    counts = {family:sum(row['family']==family for row in development) for family in ('F1','F2','F3','F4','F5','F6')}
    if any(count<4 for count in counts.values()):
        raise ValueError('Four verified development histories per family are required before study release')
    plan = prepare_study(config)
    release = {'schema':'smart-swarm-study-release-v1','allocation_sha256':plan['allocation_sha256'],
               'station_sha256':sha(station),'evidence_profiles_sha256':sha(PROFILE),
               'development_corpus':corpus['manifest'],'family_coverage':counts}
    path = config['data_root'] / 'lab/study-release.json'
    if path.exists() and load(path)!=release:
        raise ValueError('An existing frozen study release cannot be changed')
    atomic(path,release)
    return {'receipt':str(path),'sha256':sha(path)}


def verify(config):
    from .physical_learned import PROFILE
    path = config['data_root'] / 'lab/study-release.json'
    if not path.exists():
        raise ValueError('STUDY_NOT_RELEASED. Review development coverage and freeze station/model/protocol before study capture')
    record = load(path)
    plan = prepare_study(config)
    if record['allocation_sha256'] != plan['allocation_sha256'] or record['evidence_profiles_sha256'] != sha(PROFILE):
        raise ValueError('The study allocation or frozen visual evidence model profile changed')
