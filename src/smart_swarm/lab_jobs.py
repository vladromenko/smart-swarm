"""Versioned scientific job recipes admitted through the installed compute interface."""
from pathlib import Path
from .episode import load


def recipes():
    return {'version':'smart-swarm-jobs-v1', 'compute_owner':'Homebrain integration owner', 'recipes':[
        {'id':'native-act-fit-v1','purpose':'native motor learning','model':'ACT','data':'verified physical prefix train/validation groups',
         'source':'hardware','camera_order':['wrist','top'],'updates_cap':5000,'wall_seconds_cap':7200,
         'minimum_source_groups':10,'compute_profile':'existing native CPU-only reservation',
         'entrypoint':['scripts/train_physical_motor.py','--model','act'],'status':'requires_admitted_data'},
        {'id':'native-smolvla-fit-v1','purpose':'native motor learning','model':'SmolVLA','data':'same admitted native motor cohort',
         'source':'hardware','camera_order':['wrist','top'],'updates_cap':2000,'wall_seconds_cap':7200,
         'minimum_source_groups':10,'compute_profile':'existing native CPU-only reservation',
         'entrypoint':['scripts/train_physical_motor.py','--model','smolvla'],'status':'requires_verified_initialization_and_data'},
        {'id':'visual-evidence-adaptation-v1','purpose':'native visual evidence adaptation','model':'frozen event-only and event-plus-reveal references',
         'data':'reviewed role-scoped native histories, train/validation groups and masked unknown labels',
         'compute_profile':'separate existing native CPU reservation','updates_cap':1500,'wall_seconds_cap':7200,
         'status':'requires_first_native_evaluation_before_adaptation'},
        {'id':'simulation-motor-evaluation-v1','purpose':'simulation closed-loop evaluation','model':'ACT or SmolVLA',
         'data':'matching simulator inputs and execution profile','compute_profile':'existing native compute/render admission',
         'entrypoint':['smart-swarm','evaluate','--protocol','motor-pilot'],'status':'requires_matching_source_and_checkpoint_profile'},
        {'id':'physical-evaluation-v1','purpose':'attended physical motor and memory evaluation','data':'fresh frozen source allocations',
         'compute_profile':'native policy ownership and STOP plus allocated inference',
         'status':'requires_attended_station_and_native_checkpoint_qualification'}]}


def motor_eligibility(record):
    from .physical_motor_dataset import check_segments
    check_segments(record['prefixes'], technical=False)
    groups = {row['scene_group_id'] for row in record['prefixes'] if row['split']=='train'}
    validation = {row['scene_group_id'] for row in record['prefixes'] if row['split']=='validation'}
    ready = len(groups) >= 10 and bool(validation)
    return {'eligible':ready,'train_source_groups':len(groups),'validation_source_groups':len(validation),
            'reason':None if ready else 'INSUFFICIENT_DATA. Need 10 independent training groups and separate validation.'}


def submit_motor(config, args):
    import subprocess
    import sys
    from uuid import uuid4
    from .episode import sha
    from .physical_motor_dataset import publish
    from .training import ROOT
    record = publish(config)
    eligibility = motor_eligibility(record)
    if not eligibility['eligible']:
        raise ValueError(eligibility['reason'])
    chosen, = [recipe for recipe in recipes()['recipes'] if recipe['id'] == args['recipe_id']]
    if chosen['id'] not in ('native-act-fit-v1','native-smolvla-fit-v1'):
        raise ValueError('This job recipe has no admitted execution yet')
    corpus = Path(record['manifest'])
    if args['corpus_sha256'] != sha(corpus):
        raise ValueError('The native corpus changed after job review')
    output = config['data_root'] / 'training-runs' / ('native-motor-' + uuid4().hex[:12])
    model = 'act' if chosen['model']=='ACT' else 'smolvla'
    runtime = load(ROOT/'configs/training.json')['runtime']
    command = [runtime, str(ROOT/'scripts/train_physical_motor.py'), '--model', model, '--corpus', str(corpus),
               '--output', str(output), '--updates', str(chosen['updates_cap']), '--wall-seconds', str(chosen['wall_seconds_cap'])]
    if model == 'smolvla':
        weights = load(ROOT/'configs/training.json')['weights']['smolvla']
        backbone = load(ROOT/'configs/training.json')['weights']['vision_language_backbone']
        command += ['--weights', weights, '--backbone', backbone]
    subprocess.run(command, check=True, timeout=chosen['wall_seconds_cap']+30)
    return {'run_receipt':str(output/'run.json'), 'status':load(output/'run.json')['status']}
