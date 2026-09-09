"""Evaluate completed long runs without changing their checkpoints."""
import json
import argparse
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path('paper_artifacts/persistence_research/long_audit')
RUNS = {
    'breakout_constraint': 'constraint_long_breakout_seed0',
    'alien_constraint': 'constraint_long_alien_seed0',
    'breakout_baseline': 'run_007',
}


def main(output_root=None, runs=None):
  root = Path(output_root) if output_root else ROOT
  root.mkdir(parents=True, exist_ok=True)
  for name, slug in (runs or RUNS).items():
    source = Path('production_runs/experiment_tracker') / slug
    latest = (source / 'ckpt/latest').read_text().strip()
    checkpoint = source / 'ckpt' / latest
    scores = [json.loads(line)['episode/score'] for line in (source / 'scores.jsonl').read_text().splitlines()]
    summary = {'source': str(source), 'checkpoint': str(checkpoint),
               'training_episodes': len(scores), 'last20_training_return': float(np.mean(scores[-20:]))}
    (root / f'{name}_training.json').write_text(json.dumps(summary,indent=2))
    out = root / name
    if out.exists():
      raise FileExistsError(out)
    cmd = [sys.executable, '-u', '-m', 'dreamerv3.main_htp', '--configs',
           'htp_atari100k', 'size25m', '--task', f'atari100k_{name.split("_")[0]}',
           '--seed', '0', '--env.atari100k.use_seed', 'True', '--jax.prealloc', 'False',
           '--logger.outputs', 'jsonl', '--logdir', str(out), '--script', 'eval_only',
           '--run.eval_eps', '50', '--run.steps', '1000000', '--run.from_checkpoint', str(checkpoint)]
    (root / f'{name}_command.json').write_text(json.dumps(cmd,indent=2))
    print(f'Evaluating {name}',flush=True)
    with (root / f'{name}.log').open('w') as log:
      subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                     env={**os.environ, 'PYTHONHASHSEED':'0'}, check=True)
    result = json.loads((out / 'paper_artifacts/final_eval.json').read_text())
    print(json.dumps({key: result[key] for key in ('eval_episodes','eval_score_mean','eval_score_std')}),flush=True)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(); parser.add_argument('--output-root')
  parser.add_argument('--run', nargs=2, action='append', metavar=('LABEL','RUN_DIRECTORY'))
  args = parser.parse_args()
  main(args.output_root, dict(args.run) if args.run else None)
