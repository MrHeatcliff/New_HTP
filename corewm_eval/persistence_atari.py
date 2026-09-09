"""Matched short continuation and real policy evaluation, separate from paper runs."""
import json
import argparse
import os
from pathlib import Path
import subprocess
import sys


def run(only=None):
  root = Path('paper_artifacts/persistence_research/atari_matched')
  root.mkdir(parents=True, exist_ok=True)
  source = Path('production_runs/corewm_atari100k_v1/training/wave1/full/alien/seed_0/attempt_002/ckpt/env_action_steps_000100000')
  common = [sys.executable, '-u', '-m', 'dreamerv3.main_htp', '--configs',
            'atari100k', 'size12m', 'corewm_full', '--task', 'atari100k_alien',
            '--seed', '0', '--env.atari100k.use_seed', 'True',
            '--logger.outputs', 'jsonl', '--jax.prealloc', 'False',
            '--run.log_policy_video', 'False']
  for name, scale, isotropy in [('baseline', 0.,0.), ('all_lags', .1,0.), ('isotropic', .1,.1), ('isotropic_001', .1,.01), ('persistent_1', 1.,.01)]:
    if only and name != only:
      continue
    train = root / name
    if train.exists() or (root / (name + '_eval')).exists():
      raise FileExistsError(f'Refusing to mix a new experiment with existing results: {train}')
    command = common + ['--logdir', str(train), '--run.steps', '3000',
        '--run.exact_env_action_budget', 'True', '--run.action_milestones', '3000',
        '--run.from_checkpoint', str(source), '--agent.htp.persistence_scale', str(scale),
        '--agent.htp.persistence_all_lags', 'True',
        '--agent.htp.persistence_isotropy', str(isotropy)]
    env = {**os.environ, 'PYTHONHASHSEED': '0', 'PAPER_DETERMINISTIC_UUID': '1'}
    for stage, cmd in [('train', command), ('eval', common + [
        '--logdir', str(root / (name + '_eval')), '--script', 'eval_only',
        '--run.eval_eps', '20', '--run.steps', '1000000',
        '--run.from_checkpoint', str(train / 'ckpt/env_action_steps_000003000')]),
        ('representation', [sys.executable, '-u', '-m', 'corewm_eval.persistence_representation', name])]:
      (root / f'{name}_{stage}_command.json').write_text(json.dumps(cmd, indent=2))
      print(f'Start {name} {stage}', flush=True)
      with (root / f'{name}_{stage}.log').open('w') as log:
        subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
      print(f'Finished {name} {stage}', flush=True)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--only', choices=('baseline','all_lags','isotropic','isotropic_001','persistent_1'))
  run(parser.parse_args().only)
