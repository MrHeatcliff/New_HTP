"""Fixed-episode representation diagnostics for the short Alien continuation."""
import argparse
import json
from pathlib import Path
import elements
import jax
import numpy as np
import ruamel.yaml as yaml
from threadpoolctl import threadpool_limits
from dreamerv3 import main_htp
from corewm_eval.extraction import ExtractionSpec, evaluation_seed, extract_readonly
from corewm_eval.paper_pipeline import _padded_batch
from corewm_eval.shared_dataset import load_shared_diagnostic_trajectories


def run(name, analyze_only=False, source_run=None, output_root=None):
  root = Path(output_root or 'paper_artifacts/persistence_research/atari_matched')
  root.mkdir(parents=True, exist_ok=True)
  dataset = Path('production_runs/corewm_atari100k_v1/shared_diagnostic_trajectories/alien/seed_0')
  split = json.loads((dataset / 'split_manifest.json').read_text())
  selected = {key: set(split[key][:10]) for key in ('train', 'validation', 'test')}
  if analyze_only:
    with np.load(root / f'{name}_representation.npz') as cache:
      z, ids, ts = cache['z'], cache['episode_id'], cache['timestep']
  else:
    _, episodes = load_shared_diagnostic_trajectories(dataset)
    episodes = [ep for ep in episodes if int(ep['episode_id'][0]) in set.union(*selected.values())]
    source = Path(source_run) if source_run else root / name
    raw = yaml.YAML(typ='safe').load((source / 'config.yaml').read_text())
    if raw['task'] != 'atari100k_alien':
      raise ValueError('This fixed diagnostic dataset is Alien-only')
    raw['jax'].update(precompile=False, prealloc=False, transfer_guard=False)
    agent = main_htp.make_agent(elements.Config(raw))
    checkpoint = (source / 'ckpt' / (source / 'ckpt/latest').read_text().strip()
                  if source_run else source / 'ckpt/env_action_steps_000003000')
    cp = elements.Checkpoint(); cp.agent = agent; cp.load(checkpoint, keys=['agent'])
    dyn = raw['agent']['dyn'][raw['agent']['dyn']['typ']]
    spec = ExtractionSpec(int(dyn['deter']), (int(dyn['stoch']),int(dyn['classes'])),
                          tuple(raw['agent']['htp']['proj']['dims']))
    zs, ids, ts = [], [], []
    for ep in episodes:
      length = min(256, len(ep['action']))
      ep = {key: value[:length] for key,value in ep.items()}
      obs, actions, _ = _padded_batch([ep], 256)
      result = extract_readonly(agent, obs, actions, spec, evaluation_seed(0,int(ep['episode_id'][0])), decode=False)
      zs.append(np.asarray(result['z'][0,:length], np.float32))
      ids.append(ep['episode_id']); ts.append(ep['timestep'])
    z, ids, ts = np.concatenate(zs), np.concatenate(ids), np.concatenate(ts)
    np.savez(root / f'{name}_representation.npz', z=z, episode_id=ids, timestep=ts)
  with np.load('production_runs/corewm_atari100k_v1/offline_cache/alien/seed_0/full/representations.npz') as ref:
    lookup = {(int(e),int(t)): i for i,(e,t) in enumerate(zip(ref['episode_id'], ref['timestep']))}
    h = ref['h'][[lookup[(int(e),int(t))] for e,t in zip(ids,ts)]].astype(np.float32)
  # Fixed targets: same source-checkpoint features for both arms, not their own moving z.
  h = h @ np.random.default_rng(123).normal(size=(h.shape[1],64)).astype(np.float32) / np.sqrt(h.shape[1])
  masks = {key: np.isin(ids,list(value)) for key,value in selected.items()}
  mean, std = h[masks['train']].mean(0), h[masks['train']].std(0)
  h = (h - mean) / np.maximum(std,1e-6)
  first = z[masks['test'],:128]; second = z[masks['test'],128:256]
  first = first - first.mean(0); second = second - second.mean(0)
  eig = np.maximum(np.linalg.eigvalsh(first.T @ first / len(first)),0)
  metrics = {'effective_rank': float(eig.sum()**2 / np.square(eig).sum()),
      'mean_variance': float(eig.mean()), 'dead_coordinates': int((first.var(0)<1e-6).sum()),
      'block12_cka': float(np.square(first.T @ second).sum() / np.sqrt(np.square(first.T @ first).sum()*np.square(second.T @ second).sum())),
      'horizons': {}, 'episodes_per_split': 10, 'max_steps_per_episode': 256,
      'target': 'fixed source RSSM h, random projection 64, train-standardized; no actions in probe'}
  for lag in (1,4,8,16,32,64):
    valid = (ids[:-lag] == ids[lag:]) & (ts[lag:] == ts[:-lag] + lag)
    x = z[:-lag,:128]; y = h[lag:]
    xm, xs = x[masks['train'][:-lag]&valid].mean(0), x[masks['train'][:-lag]&valid].std(0)
    x = np.c_[(x-xm)/np.maximum(xs,1e-6), np.ones(len(x))]
    m = {key: value[:-lag]&valid for key,value in masks.items()}
    best = None
    for alpha in (.01,1.,100.,10000.):
      w = np.linalg.solve(x[m['train']].T @ x[m['train']] + alpha*np.eye(x.shape[1]), x[m['train']].T @ y[m['train']])
      error = np.square(x[m['validation']]@w-y[m['validation']]).mean()
      if best is None or error<best[0]: best = (error,w,alpha)
    target = y[m['test']]; pred = x[m['test']] @ best[1]
    delta = z[lag:,:128][m['test']] - z[:-lag,:128][m['test']]
    test_ids = ids[:-lag][m['test']]
    sse = np.square(pred-target).sum(-1)
    sst = np.square(target-target.mean(0)).sum(-1)
    metrics['horizons'][str(lag)] = {'r2': float(1-sse.sum()/sst.sum()),
        'episode_statistics': {str(int(e)): {'sse': float(sse[test_ids==e].sum()),
            'sst': float(sst[test_ids==e].sum())} for e in np.unique(test_ids)},
        'normalized_change': float(np.square(delta).mean()/np.maximum(first.var(),1e-8)), 'ridge':best[2]}
  (root / f'{name}_representation.json').write_text(json.dumps(metrics,indent=2))
  print(json.dumps(metrics),flush=True)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(); parser.add_argument('name')
  parser.add_argument('--analyze-only', action='store_true')
  parser.add_argument('--source-run')
  parser.add_argument('--output-root')
  args = parser.parse_args()
  with threadpool_limits(limits=1):
    run(args.name,args.analyze_only,args.source_run,args.output_root)
