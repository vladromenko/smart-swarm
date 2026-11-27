"""Remember an attended table sightline qualification against native observations."""
import time
from .episode import atomic, load, sha
from .physical_framing import capture_framing
from .physical_story_source import homebrain_boot
from .fleet_adapter import native_team_status

FLAGS = ('tank_sees_initial_and_final', 'rosmaster_sees_destinations', 'top_sees_final_relation', 'scene_remains_on_table')


def confirm(config, arguments):
    if set(arguments) != {*FLAGS, 'framing_receipt_sha256'} or not all(arguments[k] is True for k in FLAGS):
        raise ValueError('Confirm every actual table, witness and evaluator sightline at the station')
    path = config['data_root'] / 'lab/station-views.json'
    views = load(path)
    if sha(path) != arguments['framing_receipt_sha256'] or views['homebrain_boot_id'] != homebrain_boot():
        raise ValueError('The station view receipt changed or belongs to an earlier Homebrain boot')
    team = native_team_status(config)
    roles = team['story_observation_readiness']['roles']
    if not all(row['rgb_fresh'] and row['clock_within_limit'] for row in roles.values()):
        raise ValueError('Live native witness RGB and clocks must be available for station qualification')
    record = {'schema':'smart-swarm-table-visible-station-v1', 'distributed_verified':True,
              'homebrain_boot_id':homebrain_boot(), 'observations':arguments, 'views':views,
              'verified_unix_s':time.time(), 'source':'explicit_attended_operator_sightline_observation',
              'transfer_protocol':'No table-to-floor transfer. Keep all object relations in the fixed top camera view.'}
    destination = config['data_root'] / 'lab/station-profile.json'
    atomic(destination,record)
    return {'receipt':str(destination),'sha256':sha(destination)}


def capture(config):
    framing = capture_framing(config)
    path = config['data_root'] / 'lab/station-views.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {'homebrain_boot_id':homebrain_boot(),'studio':framing,'mobile':native_team_status(config),
              'meaning':'Operator qualification views only. Excluded from learned runtime and research admission.'}
    atomic(path, record)
    return {'receipt':str(path),'sha256':sha(path),'views':record}
