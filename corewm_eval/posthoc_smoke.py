"""Alien-only infrastructure smoke for matched post-hoc reconstruction."""

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .posthoc_recon import (
    PosthocArchitecture, PosthocOptimizer, architecture_metadata, initialize,
    make_optimizer, parameter_counts, parameter_hash, reconstruction_metrics,
    train_step, write_json)


METHODS = ('flat', 'pdyn_only', 'full')


def _sha256(path):
  digest = hashlib.sha256()
  with Path(path).open('rb') as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
      digest.update(chunk)
  return digest.hexdigest()


def _identity_hash(ids, timesteps):
  digest = hashlib.sha256()
  digest.update(np.asarray(ids, np.int64).tobytes())
  digest.update(np.asarray(timesteps, np.int64).tobytes())
  return digest.hexdigest()


def _json_metrics(metrics):
  return {key: [None if np.isnan(x) else float(x) for x in value]
          for key, value in metrics.items()}


def run(cache_root, dataset, output, decoder_seed=0, updates=3, batch_size=8):
  """Run a deliberately tiny smoke, never a paper post-hoc fit."""
  cache_root, dataset, output = map(Path, (cache_root, dataset, output))
  output.mkdir(parents=True, exist_ok=False)
  split = json.loads((dataset / 'split_manifest.json').read_text())
  split_hash = _sha256(dataset / 'split_manifest.json')
  architecture, opt_config = PosthocArchitecture(), PosthocOptimizer()
  optimizer = make_optimizer(opt_config)
  reports, common_identity, common_init_hash, common_counts = {}, None, None, None

  for method in METHODS:
    cache = cache_root / method
    metadata = json.loads((cache / 'metadata.json').read_text())
    if metadata['split_manifest_hash'] != split_hash:
      raise AssertionError(f'{method}: split manifest mismatch')
    checkpoint_file = Path(metadata['checkpoint']) / 'agent.pkl'
    checkpoint_before = _sha256(checkpoint_file)
    with np.load(cache / 'representations.npz') as stored:
      ids = np.asarray(stored['episode_id'], np.int64)
      timesteps = np.asarray(stored['timestep'], np.int64)
      identity = _identity_hash(ids, timesteps)
      if common_identity is None:
        common_identity = identity
      elif identity != common_identity:
        raise AssertionError(f'{method}: trajectories differ across methods')
      masks = {name: np.isin(ids, np.asarray(split[name], np.int64))
               for name in ('train', 'validation', 'test')}
      indices = {name: np.flatnonzero(mask) for name, mask in masks.items()}
      if min(map(len, indices.values())) < batch_size:
        raise AssertionError('Insufficient samples for smoke')
      # Read only the fixed smoke subset from each method's own frozen h/z.
      chosen = {name: value[:batch_size] for name, value in indices.items()}
      z_all, h_all = np.asarray(stored['z']), np.asarray(stored['h'])
      batches = {name: (
          jnp.asarray(z_all[value], jnp.bfloat16),
          jnp.asarray(h_all[value], jnp.bfloat16))
          for name, value in chosen.items()}
      del z_all, h_all

    params = initialize(decoder_seed, architecture)
    init_hash = parameter_hash(params)
    counts = parameter_counts(params)
    if common_init_hash is None:
      common_init_hash, common_counts = init_hash, counts
    elif init_hash != common_init_hash or counts != common_counts:
      raise AssertionError(f'{method}: unmatched initialization or parameter counts')
    opt_state = optimizer.init(params)
    train_trace = []
    for update in range(int(updates)):
      params, opt_state, loss, levels, grads = train_step(
          params, opt_state, *batches['train'], optimizer,
          architecture=architecture)
      grad_norm = float(jax.device_get(
          jnp.sqrt(sum(jnp.square(x).sum() for x in grads.values()))))
      row = {'update': update + 1, 'loss': float(loss),
             'per_level_loss': [float(x) for x in levels],
             'gradient_norm': grad_norm}
      if not np.isfinite([row['loss'], row['gradient_norm'],
                          *row['per_level_loss']]).all():
        raise AssertionError(row)
      train_trace.append(row)

    # These metrics validate wiring only. They are not final paper estimates.
    validation = reconstruction_metrics(
        params, *batches['validation'], architecture)
    test = reconstruction_metrics(params, *batches['test'], architecture)
    checkpoint_after = _sha256(checkpoint_file)
    if checkpoint_before != checkpoint_after:
      raise AssertionError(f'{method}: production checkpoint changed')
    report = {
        'method': metadata['method'], 'SMOKE_ONLY': True,
        'not_for_paper_metrics': True, 'diagnostic_decoder_seed': decoder_seed,
        'smoke_updates': int(updates), 'smoke_batch_size': int(batch_size),
        'episode_split_counts': {
            key: len(split[key]) for key in ('train', 'validation', 'test')},
        'smoke_sample_counts': {key: len(value) for key, value in chosen.items()},
        'trajectory_identity_hash': identity, 'split_manifest_hash': split_hash,
        'initial_parameter_hash': init_hash,
        'final_diagnostic_parameter_hash': parameter_hash(params),
        'parameter_counts_by_level': counts,
        'checkpoint_agent_hash_before': checkpoint_before,
        'checkpoint_agent_hash_after': checkpoint_after,
        'production_checkpoint_unchanged': True,
        'production_representation_parameters_loaded_by_optimizer': False,
        'representation_gradient_blocked': True,
        'train_trace': train_trace,
        'validation_wiring_metrics': _json_metrics(validation),
        'test_wiring_metrics': _json_metrics(test),
    }
    write_json(output / method / 'smoke_report.json', report)
    reports[method] = report
    del batches, params, opt_state

  summary = {
      'status': 'PASS', 'SMOKE_ONLY': True, 'methods': list(METHODS),
      'same_shared_trajectories': True, 'same_split_manifest': True,
      'matched_initialization': True, 'matched_parameter_counts': True,
      'trajectory_identity_hash': common_identity,
      'initial_parameter_hash': common_init_hash,
      'parameter_counts_by_level': common_counts,
      **architecture_metadata(architecture, opt_config),
      'final_training_budget_status': 'UNSPECIFIED_REQUIRES_USER_DECISION',
      'missing_final_protocol_fields': [
          'optimizer_updates_or_epoch_budget', 'minibatch_construction',
          'validation_frequency', 'early_stopping_or_checkpoint_selection_rule'],
  }
  write_json(output / 'metadata.json', summary)
  print(json.dumps(summary, indent=2, sort_keys=True))
  return summary


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--cache-root', required=True)
  parser.add_argument('--dataset', required=True)
  parser.add_argument('--output', required=True)
  parser.add_argument('--decoder-seed', type=int, default=0)
  parser.add_argument('--updates', type=int, default=3)
  parser.add_argument('--batch-size', type=int, default=8)
  args = parser.parse_args()
  run(args.cache_root, args.dataset, args.output, args.decoder_seed,
      args.updates, args.batch_size)
