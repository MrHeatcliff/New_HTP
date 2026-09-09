"""Audit the completed long whitened-0.1 run against the earlier long run."""
import argparse
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from .persistence_research_report import (
    load, returns, control_comparison, representation_comparison,
    scale_resistant_diagnostics, block_future_probes)


def run(root):
  root = Path(root)
  old_rep = Path('paper_artifacts/persistence_research/long_audit/alien_constraint_representation')
  old_eval = Path('paper_artifacts/persistence_research/slurm_3328/long_audit/alien_constraint')
  new_rep = root/'current_representation'
  split = load('production_runs/corewm_atari100k_v1/shared_diagnostic_trajectories/alien/seed_0/split_manifest.json')
  rng = np.random.default_rng(240909)
  report = {'comparison':'long pooled weight 1.0 versus long whitened weight 0.1',
            'limitations':['Both normalization and weight differ in this long-run comparison.',
                           'One training seed per condition; no matched unconstrained Alien size25m baseline.',
                           'Intervals condition on trained checkpoints; semantic roles are not directly labeled.']}
  report['control'] = control_comparison(returns(old_eval),returns(root/'eval/alien_whitened_p01'),rng)
  a,b = load(old_rep.with_suffix('.json')),load(new_rep.with_suffix('.json'))
  report['prediction'] = representation_comparison(a,b,rng)
  report['rank'] = [a['effective_rank'],b['effective_rank']]
  report['cka'] = [a['block12_cka'],b['block12_cka']]
  report['blocks'] = {name:scale_resistant_diagnostics(path.with_suffix('.npz'),split)
                      for name,path in [('previous',old_rep),('current',new_rep)]}
  report['block_prediction'] = {name:block_future_probes(path.with_suffix('.npz'),split)
                                for name,path in [('previous',old_rep),('current',new_rep)]}
  # Measure reset-crossing opportunity in stored replay chunks without training.
  replay = Path('production_runs/experiment_tracker/constraint_whitened_p01_alien_seed0/replay')
  bad = {lag:0 for lag in (1,2,4,8,16)}; total = dict(bad)
  files = sorted(replay.glob('*.npz'))
  for file in files:
    with np.load(file) as chunk:
      if 'is_first' not in chunk: continue
      episode = np.cumsum(chunk['is_first'].astype(np.int32))
      for lag in bad:
        if len(episode)>lag:
          bad[lag] += int((episode[lag:]!=episode[:-lag]).sum())
          total[lag] += len(episode)-lag
  report['replay_cross_episode'] = {'files':len(files),'scope':'within stored chunks; not the replay sampling distribution',
      'by_stride':{lag:{'invalid':bad[lag],'pairs':total[lag],
                        'fraction':bad[lag]/total[lag] if total[lag] else None} for lag in bad}}
  (root/'comparison.json').write_text(json.dumps(report,indent=2))
  print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
  parser=argparse.ArgumentParser(); parser.add_argument('root'); args=parser.parse_args()
  with threadpool_limits(limits=1): run(args.root)
