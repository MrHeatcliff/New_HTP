"""One-factor paired continuation from the completed Alien constraint model."""
import json
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main(output_root=None):
  root = Path(output_root or 'paper_artifacts/persistence_research/whitening_trial')
  root.mkdir(parents=True, exist_ok=True)
  source = Path('production_runs/experiment_tracker/constraint_long_alien_seed0')
  checkpoint = source / 'ckpt' / (source / 'ckpt/latest').read_text().strip()
  common = [sys.executable, '-u', '-m', 'dreamerv3.main_htp', '--configs',
            'htp_atari100k', 'size25m', '--task', 'atari100k_alien', '--seed', '0',
            '--env.atari100k.use_seed', 'True', '--jax.prealloc', 'False',
            '--logger.outputs', 'jsonl', '--run.log_policy_video', 'False']
  for metric in ('pooled','whitened'):
    train = root / metric
    if train.exists(): raise FileExistsError(train)
    traincmd = common + ['--logdir',str(train),'--run.steps','3000',
        '--run.exact_env_action_budget','True','--run.action_milestones','3000',
        '--run.from_checkpoint',str(checkpoint),'--agent.htp.persistence_scale','1.0',
        '--agent.htp.persistence_all_lags','True','--agent.htp.persistence_isotropy','0.01',
        '--agent.htp.persistence_metric',metric]
    evalcmd = common + ['--logdir',str(root / f'{metric}_eval'),'--script','eval_only',
        '--run.eval_eps','50','--run.steps','1000000','--run.from_checkpoint',
        str(train / 'ckpt/env_action_steps_000003000')]
    repcmd = [sys.executable,'-u','-m','corewm_eval.persistence_representation',metric,
        '--source-run',str(train),'--output-root',str(root)]
    for stage, cmd in [('train',traincmd),('eval',evalcmd),('representation',repcmd)]:
      (root / f'{metric}_{stage}_command.json').write_text(json.dumps(cmd,indent=2))
      print(f'Start {metric} {stage}',flush=True)
      with (root / f'{metric}_{stage}.log').open('w') as log:
        subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,
                       env={**os.environ,'PYTHONHASHSEED':'0','PAPER_DETERMINISTIC_UUID':'1'})
      print(f'Finished {metric} {stage}',flush=True)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(); parser.add_argument('--output-root')
  main(parser.parse_args().output_root)
