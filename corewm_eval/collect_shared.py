"""Future collector for frozen Backbone shared diagnostic trajectories."""

import hashlib
from pathlib import Path

import elements
import embodied
import numpy as np

from dreamerv3 import main

from .shared_dataset import persistent_episode_id, save_shared_diagnostic_trajectories
from .metadata import sha256_file


def _parameter_hash(agent):
  digest = hashlib.sha256()
  for key, value in sorted(agent.save()['params'].items()):
    digest.update(key.encode()); digest.update(np.asarray(value).tobytes())
  return digest.hexdigest()


def collect(config, checkpoint, output, game, training_seed,
            checkpoint_hash=None, episodes=100, evaluation_seed=0):
  if episodes != 100:
    raise ValueError('Final shared diagnostic dataset requires exactly 100 episodes')
  agent = main.make_agent(config)
  checkpoint = Path(checkpoint)
  actual_checkpoint_hash = sha256_file(checkpoint / 'agent.pkl')
  if checkpoint_hash is not None and checkpoint_hash != actual_checkpoint_hash:
    raise ValueError('Provided checkpoint hash does not match agent.pkl')
  checkpoint_hash = actual_checkpoint_hash
  cp = elements.Checkpoint(); cp.agent = agent
  cp.load(checkpoint, keys=['agent'])
  before = _parameter_hash(agent)
  # Explicit env seed makes episode/reset RNG reproducible; no learning/replay exists.
  env = main.make_env(config, 0, seed=int(evaluation_seed))
  driver = embodied.Driver([lambda: env], parallel=False)
  collected, current = [], None

  def record(tran, worker):
    nonlocal current
    if tran['is_first']:
      index = len(collected)
      eid = persistent_episode_id(game, training_seed, checkpoint_hash, index)
      current = {key: [] for key in (
          'observation', 'action', 'reward', 'continuation', 'is_terminal',
          'episode_id', 'timestep')}
    current['observation'].append(np.asarray(tran['image']).copy())
    current['action'].append(np.asarray(tran['action']).copy())
    current['reward'].append(np.asarray(tran['reward']).copy())
    current['continuation'].append(np.asarray(~tran['is_terminal']).copy())
    current['is_terminal'].append(np.asarray(tran['is_terminal']).copy())
    current['episode_id'].append(eid)
    current['timestep'].append(len(current['timestep']))
    if tran['is_last']:
      collected.append({key: np.asarray(value) for key, value in current.items()})

  driver.on_step(record)
  policy = lambda *args: agent.policy(*args, mode='eval')
  driver.reset(agent.init_policy)
  driver(policy, episodes=100)
  driver.close()
  after = _parameter_hash(agent)
  if before != after:
    raise AssertionError('Frozen Backbone parameters changed during collection')
  return save_shared_diagnostic_trajectories(output, collected, {
      'game': game, 'source_training_seed': int(training_seed),
      'checkpoint': str(checkpoint), 'checkpoint_hash': checkpoint_hash,
      'evaluation_seed': int(evaluation_seed), 'model_frozen_verified': True})
