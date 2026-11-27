"""Project facade integrity, duplicate requests, allocation and native save authority."""
import json
from pathlib import Path
import pytest
from smart_swarm import lab_actions
from smart_swarm.episode import atomic, load
from smart_swarm.lab_cards import descriptor, task_list
from smart_swarm.lab_facade import ProjectFacade
from smart_swarm.study_collection import prepare_study


def request(facade, action='readiness', key='one', target='station-setup', arguments=None):
    return {'project_id': 'smart-swarm', 'session_id': target, 'target_id': target,
            'action': action, 'arguments': arguments or {}, 'expected_revision': facade.snapshot(target)['revision'],
            'idempotency_key': key}


def test_study_allocation_persists_and_keeps_group_descendants_together(tmp_path):
    config = {'data_root': tmp_path}
    first = prepare_study(config)
    assert len(first['cards']) == 368 and first['history_targets'] == {'train':160,'validation':40,'test':120}
    assert first['source_groups'] == {'train':27,'validation':7,'test':44}
    assert prepare_study(config) == first
    groups = {}
    for card in first['cards']:
        assert groups.setdefault(card['source_group_id'], card['split']) == card['split']
    atomic(Path(first['path']), {**first,'cards':first['cards'][1:]})
    with pytest.raises(ValueError, match='allocation changed'):
        prepare_study(config)


def test_duplicate_request_retains_result_and_rejects_key_reuse(tmp_path, monkeypatch):
    facade = ProjectFacade({'data_root':tmp_path})
    calls = []
    def finish(identifier, req):
        calls.append(req)
        facade.store.complete(identifier, {'status':'accepted','operation_id':identifier,'resulting_revision':'new',
                                          'reason':None,'native_acknowledgement':{},'receipts':[]})
    monkeypatch.setattr(facade,'perform',finish)
    req = request(facade)
    result = facade.action(req)
    facade.executor.shutdown(wait=True)
    assert facade.action(req)['operation_id'] == result['operation_id'] and len(calls)==1
    with pytest.raises(ValueError, match='already used'):
        facade.action({**req,'arguments':{'forged':True}})
    facade.close()


def test_restart_marks_unknown_and_never_reissues_native_action(tmp_path):
    facade = ProjectFacade({'data_root':tmp_path})
    req = request(facade)
    pending, _ = facade.store.reserve(req, req['expected_revision'])
    with pytest.raises(BlockingIOError):
        ProjectFacade({'data_root':tmp_path})
    facade.close()
    resumed = ProjectFacade({'data_root':tmp_path})
    assert resumed.action(req)['status'] == 'unknown'
    assert resumed.store.operation(pending['operation_id'])['reason']['code'] == 'PROCESS_INTERRUPTED'
    assert [row['sequence'] for row in resumed.store.events()] == [1,2]
    resumed.close()


def test_facade_uses_staged_blind_route_and_rejects_generic_next(tmp_path, monkeypatch):
    config = {'data_root':tmp_path}
    card = task_list(config)['tasks'][0]
    calls = []
    monkeypatch.setattr(lab_actions,'require_study_release',lambda *_:None)
    monkeypatch.setattr(lab_actions,'begin',lambda *args,**kwargs:calls.append(kwargs))
    req = {'target_id':card['task_id'],'action':'start','arguments':{'evaluation_mode':'blind_offline'}}
    lab_actions.execute(config,req)
    assert calls == [{'staged':True,'evaluation_mode':'blind_offline'}]
    facade = ProjectFacade(config)
    with pytest.raises(ValueError, match='not allowed'):
        facade.action(request(facade,action='next'))
    facade.close()


def test_finish_only_requests_native_save_and_does_not_admit(tmp_path, monkeypatch):
    config = {'data_root':tmp_path}
    task = task_list(config)['tasks'][0]['task_id']
    trial = tmp_path/'physical-stories'/task
    trial.mkdir(parents=True)
    atomic(trial/'trial.json',{'studio_dataset_id':'ds_1234567890123456','preexisting_native_episode_ids':[]})
    calls = []
    def native(_, endpoint, payload=None):
        calls.append(endpoint)
        if endpoint == '/api/status':
            return {'recording':{'active':True,'dataset_id':'ds_1234567890123456','episode_id':'ep_current'}}
        return {'command_queued':'end'}
    monkeypatch.setattr('smart_swarm.studio.native_request',native)
    result = lab_actions.native_record(config,task,'finish_recording',{})
    assert result == {'command_queued':'end'} and calls == ['/api/status','/api/record/end']
    assert not (trial/'motor-intake.json').exists()
    assert load(tmp_path/'lab'/f'{task}-native-finish.json')['native_episode_id']=='ep_current'


def test_component_pilot_is_available_without_mobile_station(tmp_path):
    facade = ProjectFacade({'data_root':tmp_path})
    assert facade.snapshot('component-pilot-1')['blockers'] == []
    assert len(task_list(facade.config,group='component_pilot')['tasks'])==4
    facade.close()


def test_declared_schema_fixtures_match_public_contract():
    import jsonschema
    names = {'descriptor':'ProjectDescriptor','task-card':'TaskCard','snapshot':'SessionSnapshot','action-request':'ActionRequest'}
    for name, definition in names.items():
        scoped = load(Path(f'schemas/homebrain.lab.v1/{definition}.schema.json'))
        jsonschema.validate(load(Path(f'fixtures/lab/{name}.json')), scoped)


def test_canonical_revisions_and_result_types_follow_the_native_contract(tmp_path, monkeypatch):
    import jsonschema
    from smart_swarm.lab_api import HomebrainProject
    facade = HomebrainProject({'data_root':tmp_path})
    def finish(identifier, req):
        facade.domain.store.complete(identifier, {'status':'accepted','operation_id':identifier,
          'resulting_revision':req['expected_revision'],'reason':None,'native_acknowledgement':{},'receipts':[]})
    monkeypatch.setattr(facade.domain,'perform',finish)
    snapshot = facade.snapshot()
    req = {'contract_version':'homebrain.lab.v1','project_id':'smart-swarm','session_id':'station-setup',
           'target_id':'station-setup','action':'readiness','arguments':{},'expected_revision':snapshot['revision'],
           'idempotency_key':'canonical-key-0001'}
    pending = facade.action(req)
    facade.domain.executor.shutdown(wait=True)
    assert facade.action(req)['operation_id']==pending['operation_id']
    result = facade.operation(pending['operation_id'])
    assert result['status']=='accepted' and result['revision']==snapshot['revision']
    jsonschema.validate(result,load(Path('schemas/homebrain.lab.v1/ActionResult.schema.json')))
    for event in facade.events():
        jsonschema.validate(event,load(Path('schemas/homebrain.lab.v1/LabEvent.schema.json')))
    facade.close()


def test_source_group_motor_gate_excludes_the_development_deck():
    from smart_swarm.lab_jobs import motor_eligibility
    assert motor_eligibility({'prefixes':[]})['eligible'] is False
