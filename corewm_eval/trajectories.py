import json
from pathlib import Path

import numpy as np

from .config import SPLIT_COUNTS


def make_split_manifest(episode_ids, seed=0, counts=SPLIT_COUNTS):
  ids = np.asarray(sorted(set(int(x) for x in episode_ids)), dtype=np.int64)
  expected = sum(counts.values())
  if len(ids) != expected:
    raise ValueError(f'Expected {expected} unique episodes, got {len(ids)}')
  shuffled = np.random.default_rng(seed).permutation(ids)
  a, b = counts['train'], counts['train'] + counts['validation']
  manifest = {
      'split_rng_seed': int(seed),
      'train': shuffled[:a].tolist(),
      'validation': shuffled[a:b].tolist(),
      'test': shuffled[b:].tolist(),
  }
  validate_split_manifest(manifest)
  return manifest


def validate_split_manifest(manifest):
  sets = {key: set(map(int, manifest[key])) for key in ('train', 'validation', 'test')}
  if sets['train'] & sets['validation'] or sets['train'] & sets['test'] or sets['validation'] & sets['test']:
    raise ValueError('Episode leakage across splits')
  return True


def save_split_manifest(path, manifest, game, training_seed):
  validate_split_manifest(manifest)
  payload = {**manifest, 'game': game, 'training_seed': int(training_seed)}
  Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')


def valid_horizon_indices(episode_ids, timesteps, horizon):
  episode_ids = np.asarray(episode_ids)
  timesteps = np.asarray(timesteps)
  if episode_ids.shape != timesteps.shape:
    raise ValueError((episode_ids.shape, timesteps.shape))
  indices = []
  for t in range(len(episode_ids) - horizon):
    target = t + horizon
    if episode_ids[t] == episode_ids[target] and timesteps[target] == timesteps[t] + horizon:
      indices.append(t)
  return np.asarray(indices, dtype=np.int64)


def one_hot_action_windows(actions, starts, horizon, cardinality):
  actions = np.asarray(actions, dtype=np.int64)
  starts = np.asarray(starts, dtype=np.int64)
  if (actions < 0).any() or (actions >= cardinality).any():
    raise ValueError('Action outside configured cardinality')
  eye = np.eye(cardinality, dtype=np.float32)
  return np.stack([eye[actions[t:t + horizon]].reshape(-1) for t in starts])
