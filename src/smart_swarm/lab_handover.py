"""Publish the project plugin registration into the private cross-chat handover."""
import os
from pathlib import Path
import subprocess
import sys
from .episode import atomic, sha
from .lab_cards import descriptor


def publish(config, validation):
    root = Path(__file__).resolve().parents[2]
    directory = Path(os.environ.get('HOMEBRAIN_LAB_HANDOVER_DIR', Path.home() / '.local/share/homebrain-lab/handover'))
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    commit = subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    owner = directory / 'integration-owner.json'
    record = {**descriptor(), 'repository_path': str(root), 'commit': commit,
              'project_facade_entrypoint':'smart_swarm.lab_api:HomebrainProject',
              'contract_version':'homebrain.lab.v1', 'source_commit':commit,
              'configuration_path':os.environ.get('SMART_SWARM_CONFIG'),
              'entrypoint_argv': [sys.executable, '-m', 'smart_swarm.lab_facade'],
              'working_directory': str(root), 'environment': {'PYTHONPATH': str(root / 'src'),
                 **({'SMART_SWARM_CONFIG':os.environ['SMART_SWARM_CONFIG']} if 'SMART_SWARM_CONFIG' in os.environ else {})},
              'config_environment': 'SMART_SWARM_CONFIG', 'data_root': str(config['data_root']),
              'schema_path': str(root / 'schemas/homebrain.lab.v1'),
              'schema_sha256': sha(root / 'schemas/homebrain.lab.v1.json'),
              'canonical_schema_hashes':{path.name:sha(path) for path in (root/'schemas/homebrain.lab.v1').glob('*.json')},
              'fixtures': str(root / 'fixtures/lab'), 'validation': validation,
              'integration_owner_receipt': {'path': str(owner), 'sha256': sha(owner)} if owner.exists() else None,
              'common_ui_journey': 'pending_integration_owner', 'physical_verification': 'pending_attended_setup'}
    atomic(directory / 'smart-swarm.json', record)
    return record
