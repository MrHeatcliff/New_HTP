"""Lossless shared diagnostic trajectory persistence."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .extraction import evaluation_seed
from .trajectories import make_split_manifest, save_split_manifest

REQUIRED_FIELDS = (
    'observation', 'action', 'reward', 'continuation', 'is_terminal',
    'episode_id', 'timestep')


def persistent_episode_id(game, training_seed, checkpoint_hash, collection_index):
  payload = f'{game}\0{int(training_seed)}\0{checkpoint_hash}\0{int(collection_index)}'
  # 63-bit IDs remain portable through NumPy, JSON, JAX, and parquet.
  return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], 'big') & ((1 << 63) - 1)


def validate_episode(episode):
  missing = set(REQUIRED_FIELDS) - set(episode)
  if missing:
    raise ValueError(f'Missing trajectory fields: {sorted(missing)}')
  lengths = {key: len(np.asarray(episode[key])) for key in REQUIRED_FIELDS}
  if len(set(lengths.values())) != 1:
    raise ValueError(f'Inconsistent trajectory lengths: {lengths}')
  ids = np.asarray(episode['episode_id'], np.int64)
  if len(set(ids.tolist())) != 1:
    raise ValueError('episode_id must be constant within an episode')
  np.testing.assert_array_equal(
      np.asarray(episode['timestep']), np.arange(len(ids), dtype=np.int64))


def save_shared_diagnostic_trajectories(path, episodes, metadata):
  path = Path(path)
  path.mkdir(parents=True, exist_ok=False)
  if len(episodes) != 100:
    raise ValueError(f'Expected exactly 100 episodes, got {len(episodes)}')
  rows = []
  for episode in episodes:
    validate_episode(episode)
    episode_id = int(np.asarray(episode['episode_id'])[0])
    filename = f'episode_{episode_id}.npz'
    np.savez_compressed(path / filename, **{
        key: np.asarray(value) for key, value in episode.items()})
    rows.append({'episode_id': episode_id, 'file': filename,
                 'length': len(episode['action'])})
  manifest = {'dataset_kind': 'shared_diagnostic_trajectories',
              'episodes': rows, **metadata}
  (path / 'dataset_manifest.json').write_text(
      json.dumps(manifest, indent=2, sort_keys=True) + '\n')
  split = make_split_manifest([row['episode_id'] for row in rows], seed=0)
  save_split_manifest(
      path / 'split_manifest.json', split, metadata['game'],
      metadata['source_training_seed'])
  return manifest


def load_shared_diagnostic_trajectories(path):
  path = Path(path)
  manifest = json.loads((path / 'dataset_manifest.json').read_text())
  episodes = []
  for row in manifest['episodes']:
    with np.load(path / row['file']) as data:
      episode = {key: np.asarray(data[key]) for key in data.files}
    validate_episode(episode)
    if int(episode['episode_id'][0]) != int(row['episode_id']):
      raise ValueError('Persistent episode ID differs from manifest')
    episodes.append(episode)
  return manifest, episodes


def episode_rng_key(episode_id, start_timestep=0, stream=0):
  return evaluation_seed(0, int(episode_id), start_timestep, stream)


def assert_bit_identical_inputs(episodes_a, episodes_b):
  if len(episodes_a) != len(episodes_b):
    raise AssertionError('Different episode counts')
  for left, right in zip(episodes_a, episodes_b):
    for key in ('observation', 'action', 'episode_id', 'timestep'):
      np.testing.assert_array_equal(left[key], right[key])
  return True
