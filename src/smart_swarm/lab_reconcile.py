"""Read-only reconciliation of uncertain operations from authoritative project receipts."""
from .lab_domain import session_state
from .episode import load


def evidence(config, request):
    action, args = request['action'], request['arguments']
    if action in ('start_recording','finish_recording'):
        suffix = 'native-start' if action=='start_recording' else 'native-finish'
        path = config['data_root']/'lab'/f"{request['target_id']}-{suffix}.json"
        if path.exists():
            return {'native_request_receipt':load(path)}
    state = session_state(config, request['target_id'])
    if state is None:
        return None
    if action == 'start' and state['status'] in ('placement_recording','preparing','recording'):
        return {'project_state':state['status'],'original_task':request['target_id']}
    if action == 'mark':
        marker = state.get('markers',{}).get(args['phase'])
        mapping = {'assistance':'assistance','placement_outcome':'observed_placement_outcome','observed_change':'observed_change'}
        if marker and all(marker.get(mapping[key]) == value for key,value in args.items() if key!='phase'):
            return {'saved_marker':marker}
    if action == 'grant_relocation' and state.get('relocation_prompt_opened'):
        return {'prompt_opened':state['relocation_prompt_opened']}
    if action == 'start_mobile' and state.get('fleet_acknowledgement') and state.get('fleet_capture_id'):
        return {'native_capture_id':state['fleet_capture_id'],'acknowledgement':state['fleet_acknowledgement']}
    if action == 'predict' and state.get('decision'):
        return {'decision':state['decision'],'state':state['status']}
    if action == 'reveal' and state['status']=='complete' and state.get('observed_terminal_outcome')==args['outcome']:
        return {'saved_outcome':state['observed_terminal_outcome']}
    if action == 'close_incomplete' and state['status']=='incomplete':
        return {'reason':state['reason']}
    return None
