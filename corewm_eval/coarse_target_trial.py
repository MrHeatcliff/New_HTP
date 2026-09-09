"""One-factor coarse reconstruction target experiment; execute only in Slurm."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from threadpoolctl import threadpool_limits
from .persistence_research_report import (
    load, returns, checkpoint_updates, control_comparison, representation_comparison,
    scale_resistant_diagnostics, block_future_probes)


def main(root, factor='coarse_target'):
  if not os.environ.get('SLURM_JOB_ID'):
    raise RuntimeError('Run numerical research through Slurm')
  root = Path(root); root.mkdir(parents=True,exist_ok=True)
  source = Path('production_runs/experiment_tracker/constraint_whitened_p01_alien_seed0')
  checkpoint = source/'ckpt'/(source/'ckpt/latest').read_text().strip()
  arms = [('raw',1,False),('coarse',16,False)] if factor == 'coarse_target' else [
      ('unmasked',1,False),('masked',1,True)]
  names = [arm[0] for arm in arms]
  manifest = {'source':str(checkpoint),'source_sha256':hashlib.sha256((checkpoint/'agent.pkl').read_bytes()).hexdigest(),
      'changed_factor':('first reconstruction target causal window: 1 versus 16'
                        if factor == 'coarse_target' else 'prediction episode-boundary mask: off versus on'),
      'additional_actions':5000,'seed':0,'episodes':50,
      'persistence_metric':'whitened','persistence_scale':.1,'isotropy':.01,
      'arms':arms,'note':'Fresh replay in each arm; exact action budget, update counts audited.'}
  (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
  common = [sys.executable,'-u','-m','dreamerv3.main_htp','--configs','htp_atari100k','size25m',
      '--task','atari100k_alien','--seed','0','--env.atari100k.use_seed','True',
      '--jax.prealloc','False','--logger.outputs','jsonl','--run.log_policy_video','False',
      '--agent.htp.persistence_metric','whitened','--agent.htp.persistence_scale','0.1',
      '--agent.htp.persistence_all_lags','True','--agent.htp.persistence_isotropy','0.01']
  for name,window,masked in arms:
    train=root/name
    if train.exists(): raise FileExistsError(train)
    config = common + ['--agent.htp.recon.first_target_window',str(window),
                       '--agent.htp.pdyn.mask_episode_boundaries',str(masked)]
    stages = [
      ('train',config+['--logdir',str(train),'--run.steps','5000','--run.exact_env_action_budget','True',
                      '--run.action_milestones','5000','--run.from_checkpoint',str(checkpoint)]),
      ('eval',config+['--logdir',str(root/f'{name}_eval'),'--script','eval_only','--run.eval_eps','50',
                     '--run.steps','1000000','--run.from_checkpoint',str(train/'ckpt/env_action_steps_000005000')]),
      ('representation',[sys.executable,'-u','-m','corewm_eval.persistence_representation',name,
                          '--source-run',str(train),'--output-root',str(root)])]
    for stage,cmd in stages:
      (root/f'{name}_{stage}_command.json').write_text(json.dumps(cmd,indent=2))
      print(f'Start {name} {stage}',flush=True)
      with (root/f'{name}_{stage}.log').open('w') as log:
        subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,
            env={**os.environ,'PYTHONHASHSEED':'0','PAPER_DETERMINISTIC_UUID':'1'})
      print(f'Finished {name} {stage}',flush=True)
  rng=np.random.default_rng(90916)
  a,b=[load(root/f'{name}_representation.json') for name in names]
  split=load('production_runs/corewm_atari100k_v1/shared_diagnostic_trajectories/alien/seed_0/split_manifest.json')
  initial=checkpoint_updates(checkpoint)
  report={'manifest':manifest,'limitations':['One continuation seed; not training from scratch.',
      'Episode bootstrap is conditional on trained checkpoints.','Fixed RSSM probes are not semantic labels.',
      'Exact actions do not guarantee equal updates; consult update counts.',
      'Episode masking also reduces effective loss weight in proportion to invalid pairs.'],
      'updates':{name:checkpoint_updates(root/name/'ckpt/env_action_steps_000005000')-initial for name in names},
      'control':control_comparison(returns(root/f'{names[0]}_eval'),returns(root/f'{names[1]}_eval'),rng),
      'prediction':representation_comparison(a,b,rng),'rank':[a['effective_rank'],b['effective_rank']],
      'cka':[a['block12_cka'],b['block12_cka']],
      'blocks':{name:scale_resistant_diagnostics(root/f'{name}_representation.npz',split) for name in names},
      'block_prediction':{name:block_future_probes(root/f'{name}_representation.npz',split) for name in names}}
  (root/'comparison.json').write_text(json.dumps(report,indent=2))
  print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
  parser=argparse.ArgumentParser(); parser.add_argument('root')
  parser.add_argument('--factor',choices=['coarse_target','episode_mask'],default='coarse_target')
  args=parser.parse_args()
  with threadpool_limits(limits=1): main(args.root,args.factor)
