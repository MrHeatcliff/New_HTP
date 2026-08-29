#!/usr/bin/env python3
"""Validate the repository's Weights & Biases setup without exposing secrets."""

import argparse
import os
import sys

import wandb

from corewm_eval.config import WANDB_ENTITY, wandb_project_for_game


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--entity', default=os.environ.get('WANDB_ENTITY', WANDB_ENTITY))
  parser.add_argument(
      '--project', default=os.environ.get('WANDB_PROJECT', 'dreamv3-alien'))
  parser.add_argument('--game', help='Use the canonical dreamv3 project for this game.')
  parser.add_argument('--smoke', action='store_true',
                      help='Create and finish one online setup-smoke run.')
  return parser.parse_args()


def main():
  args = parse_args()
  if args.game:
    args.project = wandb_project_for_game(args.game)
  api = wandb.Api(timeout=30)
  viewer_entity = getattr(api.viewer, 'entity', None)
  if not viewer_entity:
    raise RuntimeError('W&B authentication succeeded but viewer entity is empty.')
  if args.entity != viewer_entity:
    print(
        f'WARNING: configured entity {args.entity!r} differs from the '
        f'authenticated default {viewer_entity!r}.', file=sys.stderr)

  print(f'wandb_version={wandb.__version__}')
  print(f'authenticated_entity={viewer_entity}')
  print(f'configured_entity={args.entity}')
  print(f'configured_project={args.project}')
  print('credentials_source=wandb_api (secret not displayed)')

  if args.smoke:
    run = wandb.init(
        entity=args.entity,
        project=args.project,
        name='setup-smoke',
        job_type='setup',
        tags=['setup', 'smoke', 'htp'],
        config={
            'purpose': 'Verify HTP repository W&B logging.',
            'smoke_only': True,
        },
    )
    if run is None:
      raise RuntimeError('wandb.init() did not return a run.')
    run.log({'setup/ok': 1, 'setup/wandb_version': wandb.__version__}, step=0)
    run_url = run.url
    run.finish()
    print(f'smoke_run_url={run_url}')
    print('online_logging_ok=true')
  else:
    print('auth_ok=true')


if __name__ == '__main__':
  main()
