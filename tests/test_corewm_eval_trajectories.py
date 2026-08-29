import numpy as np
import pytest
import json

from corewm_eval.trajectories import (
    make_split_manifest, one_hot_action_windows, valid_horizon_indices,
    validate_split_manifest)
from corewm_eval.shared_dataset import (
    assert_bit_identical_inputs, episode_rng_key,
    load_shared_diagnostic_trajectories, persistent_episode_id,
    save_shared_diagnostic_trajectories)


def test_episode_disjoint_60_20_20_split():
  manifest = make_split_manifest(range(100), seed=0)
  assert tuple(map(len, (manifest['train'], manifest['validation'], manifest['test']))) == (60, 20, 20)
  assert validate_split_manifest(manifest)


def test_split_leakage_fails_loudly():
  with pytest.raises(ValueError):
    validate_split_manifest({'train': [1], 'validation': [1], 'test': [2]})


def test_horizon_extraction_never_crosses_episode():
  episode = np.repeat([0, 1], [66, 66])
  timestep = np.r_[np.arange(66), np.arange(66)]
  starts = valid_horizon_indices(episode, timestep, 64)
  np.testing.assert_array_equal(starts, [0, 1, 66, 67])
  assert all(episode[t] == episode[t + 64] for t in starts)
  assert all(timestep[t + 64] == timestep[t] + 64 for t in starts)


def test_action_window_uses_t_through_t_plus_k_minus_one():
  actions = np.array([0, 1, 2, 1])
  result = one_hot_action_windows(actions, [0], horizon=3, cardinality=3)
  expected = np.eye(3, dtype=np.float32)[[0, 1, 2]].reshape(1, -1)
  np.testing.assert_array_equal(result, expected)


def test_persistent_episode_ids_survive_serialization_and_order(tmp_path):
  episodes = []
  for index in range(100):
    eid = persistent_episode_id('alien', 0, 'abc123', index)
    length = 2
    episodes.append({
        'observation': np.full((length, 2, 2, 3), index, np.uint8),
        'action': np.arange(length, dtype=np.int32),
        'reward': np.zeros(length, np.float32),
        'continuation': np.ones(length, bool),
        'is_terminal': np.array([False, True]),
        'episode_id': np.full(length, eid, np.int64),
        'timestep': np.arange(length, dtype=np.int64),
    })
  before_keys = [episode_rng_key(ep['episode_id'][0]) for ep in episodes]
  save_shared_diagnostic_trajectories(tmp_path / 'dataset', episodes, {
      'game': 'alien', 'source_training_seed': 0, 'checkpoint_hash': 'abc123'})
  manifest, loaded = load_shared_diagnostic_trajectories(tmp_path / 'dataset')
  after_keys = [episode_rng_key(ep['episode_id'][0]) for ep in loaded]
  assert_bit_identical_inputs(episodes, loaded)
  for left, right in zip(before_keys, after_keys):
    np.testing.assert_array_equal(left, right)
  assert manifest['dataset_kind'] == 'shared_diagnostic_trajectories'
