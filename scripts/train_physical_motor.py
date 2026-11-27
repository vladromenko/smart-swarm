"""Bounded physical prefix training worker; requires the existing native reservation."""
import argparse
from pathlib import Path
import os
import sys
import time
from smart_swarm.cli import settings
from smart_swarm.physical_training import native_train
from smart_swarm.research_jobs import admitted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', choices=('act','smolvla'), default='act')
    parser.add_argument('--technical', action='store_true')
    parser.add_argument('--updates', type=int, default=1)
    parser.add_argument('--wall-seconds', type=int, default=90)
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--backbone', type=Path)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        native_train(args.corpus, args.output, model=args.model, technical=args.technical,
                     updates=args.updates, wall_seconds=args.wall_seconds, weights=args.weights, backbone=args.backbone)
    else:
        from smart_swarm.research_jobs import execute
        from smart_swarm.training import offline
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with admitted(settings(), 'smart-swarm-native-motor-training', archive_only=True) as (_, heartbeat):
            command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--worker']
            with args.output.with_suffix('.log').open('x') as stream:
                code = execute(command, offline(), stream, time.monotonic()+args.wall_seconds, heartbeat)
            if code:
                raise RuntimeError(f'Native training worker failed with exit code {code}')


if __name__ == '__main__':
    main()
