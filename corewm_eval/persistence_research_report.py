"""Episode-level uncertainty and scale-resistant diagnostics for research runs.

Run on a Slurm allocation. Intervals condition on trained checkpoints; they do
not estimate training-seed uncertainty or establish semantic identification.
"""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits


def load(path):
  return json.loads(Path(path).read_text())


def interval(values):
  return np.quantile(values, [.025, .975]).tolist()


def returns(path):
  summary = load(path / 'paper_artifacts/final_eval.json')
  scores = np.array([json.loads(line)['episode/score'] for line in
                     (path / 'scores.jsonl').read_text().splitlines()])
  assert len(scores) == summary['eval_episodes'] >= 50
  assert np.isfinite(scores).all()
  return scores


def checkpoint_updates(path):
  with (Path(path)/'agent.pkl').open('rb') as stream:
    return int(pickle.load(stream)['counters']['updates'])


def control_comparison(a, b, rng):
  diff = b[rng.integers(len(b), size=(20000,len(b)))].mean(1)
  diff -= a[rng.integers(len(a), size=(20000,len(a)))].mean(1)
  return {'reference_mean': float(a.mean()), 'candidate_mean': float(b.mean()),
          'mean_difference': float(b.mean()-a.mean()), 'difference_ci95': interval(diff),
          'episodes': [len(a),len(b)]}


def representation_comparison(a, b, rng):
  rows = {}
  for lag in ('16','32','64'):
    left, right = a['horizons'][lag], b['horizons'][lag]
    ids = sorted(left['episode_statistics'])
    assert ids == sorted(right['episode_statistics'])
    sample = rng.integers(len(ids), size=(20000,len(ids)))
    values = []
    for item in (left,right):
      stats = item['episode_statistics']
      sse = np.array([stats[key]['sse'] for key in ids])
      sst = np.array([stats[key]['sst'] for key in ids])
      values.append(1-sse[sample].sum(1)/sst[sample].sum(1))
    rows[lag] = {'reference_r2':left['r2'], 'candidate_r2':right['r2'],
                 'difference_ci95':interval(values[1]-values[0]),
                 'raw_normalized_change':[left['normalized_change'],right['normalized_change']]}
  return rows


def scale_resistant_diagnostics(path, split):
  with np.load(path) as cache:
    z, ids, ts = cache['z'].astype(np.float64), cache['episode_id'], cache['timestep']
  test = np.isin(ids, split['test'])
  train = np.isin(ids, split['train'])
  rows = []
  for lo,hi in ((0,128),(128,256)):
    x = z[:,lo:hi]
    centered = x[train]-x[train].mean(0)
    covariance = centered.T @ centered / len(centered)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    ridge = max(eigenvalues.mean()*1e-4,1e-8)
    whitener = (eigenvectors/np.sqrt(np.maximum(eigenvalues,0)+ridge)) @ eigenvectors.T
    transformed = (x-x[train].mean(0)) @ whitener
    test_cov = np.cov(x[test].T)
    eig = np.maximum(np.linalg.eigvalsh(test_cov),0)
    changes = {}
    for lag in (1,4,8,16,32,64):
      valid = test[:-lag] & (ids[:-lag]==ids[lag:]) & (ts[lag:]==ts[:-lag]+lag)
      a,b = transformed[:-lag][valid], transformed[lag:][valid]
      changes[str(lag)] = float(np.square(a-b).mean()/2)
    rows.append({'block':[lo,hi], 'effective_rank':float(eig.sum()**2/np.square(eig).sum()),
                 'whitened_change':changes, 'whitener_fit':'train episodes only'})
  return rows


def block_future_probes(path, split):
  """Dimension-matched blocks and marginal prefix utility on fixed targets."""
  with np.load(path) as cache:
    z, ids, ts = cache['z'].astype(np.float64), cache['episode_id'], cache['timestep']
  with np.load('production_runs/corewm_atari100k_v1/offline_cache/alien/seed_0/full/representations.npz') as source:
    lookup = {(int(e),int(t)): i for i,(e,t) in enumerate(zip(source['episode_id'],source['timestep']))}
    h = source['h'][[lookup[(int(e),int(t))] for e,t in zip(ids,ts)]].astype(np.float64)
  h = h @ np.random.default_rng(123).normal(size=(h.shape[1],64))/np.sqrt(h.shape[1])
  masks = {key:np.isin(ids,split[key]) for key in ('train','validation','test')}
  h = (h-h[masks['train']].mean(0))/np.maximum(h[masks['train']].std(0),1e-6)
  rows = {}
  for lag in (1,16,32,64):
    valid = (ids[:-lag]==ids[lag:]) & (ts[lag:]==ts[:-lag]+lag)
    m = {key:value[:-lag]&valid for key,value in masks.items()}
    y = h[lag:]; scores = {}
    for name,lo,hi in (('block1',0,128),('block2',128,256),('prefix2',0,256)):
      x = z[:-lag,lo:hi]
      x = (x-x[m['train']].mean(0))/np.maximum(x[m['train']].std(0),1e-6)
      x = np.c_[x,np.ones(len(x))]
      gram = x[m['train']].T @ x[m['train']]
      rhs = x[m['train']].T @ y[m['train']]
      best = None
      for alpha in (.01,1.,100.,10000.):
        weights = np.linalg.solve(gram+alpha*np.eye(x.shape[1]),rhs)
        error = np.square(x[m['validation']] @ weights-y[m['validation']]).mean()
        if best is None or error<best[0]: best=(error,weights)
      target = y[m['test']]; pred = x[m['test']] @ best[1]
      scores[name] = float(1-np.square(pred-target).sum()/np.square(target-target.mean(0)).sum())
    scores['prefix2_marginal_gain'] = scores['prefix2']-scores['block1']
    rows[str(lag)] = scores
  return rows


def run(root):
  root = Path(root); rng = np.random.default_rng(2409)
  report = {'limitations':[
      'Exploratory comparisons; coefficients were selected during prior research.',
      'One training seed per real condition; bootstrap only covers sampled episodes.',
      'Breakout historical baseline used two training GPUs; constraint run resumed from two onto one.',
      'No matched unconstrained size25m Alien baseline is available.',
      'Fixed RSSM target probes are predictive proxies, not semantic ground truth.',
      'Raw Euclidean change is not invariant to axis rescaling; compare whitened diagnostics too.']}
  audit = root/'long_audit'
  report['breakout_policy'] = control_comparison(returns(audit/'breakout_baseline'),
                                                returns(audit/'breakout_constraint'),rng)
  alien = returns(audit/'alien_constraint')
  report['alien_policy'] = {'mean':float(alien.mean()),'episodes':len(alien),'matched_baseline':None}
  synth = load(root/'synthetic/synthetic_whitening_5000.json')
  report['synthetic'] = synth
  trial = root/'whitening_trial'
  source = Path('production_runs/experiment_tracker/constraint_long_alien_seed0')
  original = source/'ckpt'/(source/'ckpt/latest').read_text().strip()
  initial_updates = checkpoint_updates(original)
  report['continuation_updates'] = {name:checkpoint_updates(
      trial/name/'ckpt/env_action_steps_000003000')-initial_updates for name in ('pooled','whitened')}
  report['same_number_of_updates'] = len(set(report['continuation_updates'].values())) == 1
  report['whitening_policy'] = control_comparison(returns(trial/'pooled_eval'),returns(trial/'whitened_eval'),rng)
  a,b = [load(trial/f'{name}_representation.json') for name in ('pooled','whitened')]
  report['whitening_prediction'] = representation_comparison(a,b,rng)
  report['whitening_rank'] = [a['effective_rank'],b['effective_rank']]
  report['whitening_cka'] = [a['block12_cka'],b['block12_cka']]
  split = load('production_runs/corewm_atari100k_v1/shared_diagnostic_trajectories/alien/seed_0/split_manifest.json')
  report['scale_resistant_blocks'] = {name:scale_resistant_diagnostics(trial/f'{name}_representation.npz',split)
                                      for name in ('pooled','whitened')}
  report['block_future_probes'] = {name:block_future_probes(trial/f'{name}_representation.npz',split)
                                  for name in ('pooled','whitened')}
  (root/'research_comparison.json').write_text(json.dumps(report,indent=2))
  print(json.dumps({key:value for key,value in report.items() if key not in ('synthetic','scale_resistant_blocks')},indent=2))


if __name__ == '__main__':
  parser=argparse.ArgumentParser(); parser.add_argument('root'); args=parser.parse_args()
  with threadpool_limits(limits=1): run(args.root)
