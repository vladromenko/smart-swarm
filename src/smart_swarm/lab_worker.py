"""Run one explicit project operation with stdout isolated from JSON transport."""
from pathlib import Path
import sys
from .episode import atomic, load
from .lab_actions import execute, StillPending
from subprocess import SubprocessError


def main():
    payload = load(Path(sys.argv[1]))
    config = payload['config']
    for key in ('data_root', 'asset_root', 'platform_handover', 'platform_project', 'studio_project'):
        if key in config:
            config[key] = Path(config[key])
    try:
        response = {'status': 'accepted', 'value': execute(config, payload['request'])}
    except Exception as error:
        response = {'status': 'unknown' if isinstance(error, (OSError, TimeoutError, SubprocessError)) else 'rejected',
                    'reason': f'{type(error).__name__}. {error}'}
    atomic(Path(sys.argv[2]), response)


if __name__ == '__main__':
    main()
