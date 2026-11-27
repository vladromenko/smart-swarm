"""Canonical Homebrain Lab wire types mapped onto Smart Swarm's domain journals."""
from .lab_facade import ProjectFacade
from .lab_cards import descriptor, revision
from .lab_jobs import recipes
import re

VERSION = 'homebrain.lab.v1'


class HomebrainProject:
    def __init__(self, config):
        self.domain = ProjectFacade(config)

    def descriptor(self):
        record = descriptor()
        return {'contract_version':VERSION, 'project_id':record['project_id'], 'title':'Smart Swarm',
                'version':record['version'], 'protocol_revision':record['protocol_revisions'][-1],
                'modes':record['modes'], 'robot_ids':['so101','tank','rosmaster'],
                'task_groups':[{'id':name,'title':name.replace('_',' ')} for name in record['task_groups']],
                'plugin_location':'smart_swarm.lab_api:HomebrainProject', 'capabilities':record['required_capabilities']}

    def tasks(self, **options):
        page = self.domain.call({'method':'tasks','params':options})
        return {'tasks':[wire_card(card) for card in page['tasks']], 'total':page['total']}

    def snapshot(self, target='station-setup'):
        raw = self.domain.snapshot(target)
        number = self.domain.store.revision(target, raw['revision'])
        return {'contract_version':VERSION, 'session_id':raw['session_id'], 'project_id':raw['project_id'],
                'task_id':target, 'robot_id':'so101', 'domain':'physical', 'revision':number, 'state':raw['step'],
                'instruction':raw['instruction'], 'permitted_actions':raw['allowed_actions'],
                'native_ids':{key:value for key,value in raw['native_ids'].items() if value is not None},
                'readiness':raw['readiness'], 'recording':{}, 'receipts':raw['receipts'],
                'pending_work':[{'operation_id':identifier,'action':req['action']} for identifier,req in raw['pending_work']],
                'blockers':raw['blockers'], 'recovery':None, 'project_state':{'task_revision_sha256':raw['revision']}}

    def action(self, request):
        allowed = {'contract_version','project_id','session_id','target_id','action','expected_revision','idempotency_key','arguments'}
        if set(request)-allowed or not re.fullmatch(r'[a-z][a-z0-9_]{1,47}', request.get('action','')):
            raise ValueError('Invalid canonical action fields')
        key = request.get('idempotency_key')
        if not isinstance(key,str) or not 8 <= len(key) <= 128:
            raise ValueError('Canonical idempotency keys must contain 8..128 characters')
        if request.get('contract_version',VERSION) != VERSION:
            raise ValueError('Unsupported Homebrain Lab contract version')
        task = request.get('target_id') or request.get('session_id') or 'station-setup'
        if request.get('session_id') not in (None,task):
            raise ValueError('Homebrain session envelope must bind this project task')
        current = self.snapshot(task)
        digest = self.domain.store.revision_digest(task, request['expected_revision'])
        raw = {key:request[key] for key in ('project_id','action','idempotency_key')}
        raw.update(session_id=task,target_id=task,expected_revision=digest,arguments=request.get('arguments',{}))
        return self.result(self.domain.action(raw), task)

    def result(self, raw, task='station-setup'):
        reason = raw.get('reason') or {}
        return {'contract_version':VERSION, 'status':raw['status'], 'operation_id':raw['operation_id'],
                'revision':self.domain.store.number_for_digest(task,raw['resulting_revision']),
                'reason_code':reason.get('code',raw['status'].upper()), 'message':reason.get('message','Project receipts govern phase progress'),
                'native_acknowledgement':raw.get('native_acknowledgement') or {},
                'receipts':[item if isinstance(item,dict) else {'path':item} for item in raw.get('receipts',[])]}

    def call(self, message):
        name, params = message['method'], message.get('params',{})
        from .lab_progress import progress
        handlers = {'descriptor':self.descriptor,'tasks':lambda:self.tasks(**params),'snapshot':lambda:self.snapshot(**params),
                    'progress':lambda:progress(self.domain.config),
                    'action':lambda:self.action(params), 'jobs':recipes,
                    'operation':lambda:self.operation(params['identifier']),
                    'events':lambda:self.events(**params),
                    'reconcile':lambda:self.reconcile_operation(params['identifier'])}
        if name not in handlers:
            raise ValueError('Unknown project facade method')
        return handlers[name]()

    def operation(self, identifier):
        request = self.domain.store.request(identifier)
        return self.result(self.domain.store.operation(identifier), request['target_id'])

    def reconcile_operation(self, identifier):
        request = self.domain.store.request(identifier)
        return self.result(self.domain.reconcile_operation(identifier), request['target_id'])

    def events(self, after=0, limit=50):
        return [self.event(raw) for raw in self.domain.store.events(after,limit)]

    def event(self, raw):
        request = self.domain.store.request(raw['operation_id'])
        value = self.result(raw,request['target_id'])
        return {'contract_version':VERSION,'sequence':raw['sequence'],'session_id':request['session_id'],
                'kind':'project_operation','revision':value['revision'],
                'created_at':raw['created_at'],'payload':value}

    def close(self):
        self.domain.close()


def wire_card(raw):
    return {'contract_version':VERSION,'project_id':raw['project_id'],'task_id':raw['task_id'],'title':raw['title'],
            'protocol_revision':raw['protocol_revision'],'plan_revision':raw['plan_revision'],
            'robot_id':'so101','domain':'physical','dataset_intent':raw['dataset_intent'],
            'camera_roles':raw['ordered_cameras'],'setup':raw['setup'],'setup_media':raw['setup_media'],
            'control_profile':raw['native_control_profile'],'source_group':raw['source_group_id'],'split':raw['split'],
            'phases':[{'id':raw['step'],'instruction':raw['instruction'],
                       **{key:raw[key] for key in ('placement','relocation','withheld','reset','assistance') if key in raw}}],
            'allowed_actions':raw['allowed_actions'],'outcome_requirements':raw['outcomes'],
            'eligibility':'requires_project_and_native_receipts','blockers':[], 'native':{'validation':raw['validation']}}
