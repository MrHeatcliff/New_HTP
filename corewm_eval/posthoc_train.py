"""Frozen paper protocol for matched post-hoc reconstruction probes."""

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from .posthoc_recon import (
    PosthocArchitecture, PosthocOptimizer, architecture_metadata, initialize,
    make_optimizer, metrics_from_sufficient_statistics, parameter_counts,
    parameter_hash, probe_not_converged, qualifying_improvement, reconstruct,
    train_step, write_json)


def _sha256(path):
  digest = hashlib.sha256()
  with Path(path).open('rb') as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
      digest.update(chunk)
  return digest.hexdigest()


def _production_parameter_hash(path):
  with Path(path).open('rb') as stream:
    payload = pickle.load(stream)['params']
  return parameter_hash(payload)


def _save_params(path, params):
  arrays = {key: np.asarray(value) for key, value in params.items()}
  np.savez(path, **arrays)


def _load_params(path):
  with np.load(path) as source:
    return {key: jnp.asarray(source[key]) for key in source.files}


def _evaluate(params, z, h, indices, architecture, chunk_size=1024):
  predict = jax.jit(lambda p, x, y: reconstruct(p, x, y, architecture))
  sse = np.zeros(len(architecture.prefix_dims), np.float64)
  target_sum = np.zeros(architecture.feat_dim, np.float64)
  target_square_sum, count = 0.0, 0
  for offset in range(0, len(indices), int(chunk_size)):
    chosen = indices[offset:offset + int(chunk_size)]
    target = np.asarray(h[chosen], np.float32)
    predictions = predict(
        params, jnp.asarray(z[chosen], jnp.bfloat16),
        jnp.asarray(target, jnp.bfloat16))
    target64 = target.astype(np.float64)
    target_sum += target64.sum(0)
    target_square_sum += float(np.square(target64).sum())
    count += len(chosen)
    for level, prediction in enumerate(predictions):
      residual = target64 - np.asarray(prediction, np.float64)
      sse[level] += np.square(residual).sum()
  result = metrics_from_sufficient_statistics(
      sse, target_sum, target_square_sum, count, architecture.feat_dim)
  return result, int(count)


def _write_curve(path, rows):
  fields = ['update', 'validation_score', 'qualifying_improvement',
            'consecutive_without_improvement', 'best_update',
            *[f'R2_rec_l{x}' for x in range(1, 6)]]
  with Path(path).open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader(); writer.writerows(rows)


def run(method, cache, dataset, output, decoder_seed=0, batch_size=1024,
        max_updates=10000, validation_interval=250,
        minimum_updates_before_stop=2000, patience=8,
        min_improvement=1e-4, eval_chunk_size=1024):
  if (decoder_seed, batch_size, max_updates, validation_interval,
      minimum_updates_before_stop, patience, min_improvement) != (
          0, 1024, 10000, 250, 2000, 8, 1e-4):
    raise ValueError('Refusing to deviate from frozen post-hoc paper protocol')
  cache, dataset, output = map(Path, (cache, dataset, output))
  output.mkdir(parents=True, exist_ok=False)
  metadata = json.loads((cache / 'metadata.json').read_text())
  split = json.loads((dataset / 'split_manifest.json').read_text())
  split_hash = _sha256(dataset / 'split_manifest.json')
  if metadata['split_manifest_hash'] != split_hash:
    raise AssertionError('Split manifest mismatch')
  checkpoint_file = Path(metadata['checkpoint']) / 'agent.pkl'
  checkpoint_hash_before = _sha256(checkpoint_file)
  production_params_before = _production_parameter_hash(checkpoint_file)

  with np.load(cache / 'representations.npz') as stored:
    ids = np.asarray(stored['episode_id'], np.int64)
    timesteps = np.asarray(stored['timestep'], np.int64)
    z, h = np.asarray(stored['z']), np.asarray(stored['h'])
  identity = hashlib.sha256(ids.tobytes() + timesteps.tobytes()).hexdigest()
  indices = {name: np.flatnonzero(np.isin(ids, np.asarray(split[name], np.int64)))
             for name in ('train', 'validation', 'test')}
  if any(not len(value) for value in indices.values()):
    raise AssertionError({key: len(value) for key, value in indices.items()})

  # Generate the full method-independent stream once. It is indexed relative
  # to the canonically ordered eligible TRAIN states and sampled uniformly.
  rng = np.random.default_rng(decoder_seed)
  stream = rng.integers(
      0, len(indices['train']), size=(max_updates, batch_size),
      dtype=np.int32)
  stream_hash = hashlib.sha256(stream.tobytes()).hexdigest()

  architecture, opt_config = PosthocArchitecture(), PosthocOptimizer()
  params = initialize(decoder_seed, architecture)
  init_hash = parameter_hash(params)
  counts = parameter_counts(params)
  optimizer = make_optimizer(opt_config)
  opt_state = optimizer.init(params)
  step_fn = jax.jit(lambda p, state, x, y: train_step(
      p, state, x, y, optimizer, architecture))

  validation_curve, training_curve = [], []
  best_score, best_update, no_improvement = -np.inf, 0, 0
  best_path = output / 'best_diagnostic_params.npz'
  start = time.time()

  def validate(update):
    nonlocal best_score, best_update, no_improvement
    values, count = _evaluate(
        params, z, h, indices['validation'], architecture, eval_chunk_size)
    score = float(np.mean(values['R2_rec']))
    improved = (update == 0 or
                qualifying_improvement(score, best_score, min_improvement))
    if improved:
      best_score, best_update, no_improvement = score, int(update), 0
      _save_params(best_path, params)
    else:
      no_improvement += 1
    row = {
        'update': int(update), 'validation_score': score,
        'qualifying_improvement': bool(improved),
        'consecutive_without_improvement': int(no_improvement),
        'best_update': int(best_update),
        **{f'R2_rec_l{i + 1}': float(value)
           for i, value in enumerate(values['R2_rec'])}}
    validation_curve.append(row)
    print(json.dumps({'method': method, 'validation_states': count, **row}),
          flush=True)

  validate(0)
  stopping_reason, executed = 'MAX_UPDATES', 0
  for update in range(1, max_updates + 1):
    chosen = indices['train'][stream[update - 1]]
    params, opt_state, loss, levels, grads = step_fn(
        params, opt_state, jnp.asarray(z[chosen], jnp.bfloat16),
        jnp.asarray(h[chosen], jnp.bfloat16))
    loss_value = float(loss)
    if not np.isfinite(loss_value):
      raise FloatingPointError((update, loss_value))
    executed = update
    training_curve.append({'update': update, 'loss': loss_value})
    if update % validation_interval == 0:
      validate(update)
      if update >= minimum_updates_before_stop and no_improvement >= patience:
        stopping_reason = 'EARLY_STOP'
        break

  _write_curve(output / 'validation_curve.csv', validation_curve)
  with (output / 'training_curve.csv').open('w', newline='') as stream_file:
    writer = csv.DictWriter(stream_file, fieldnames=('update', 'loss'))
    writer.writeheader(); writer.writerows(training_curve)

  params = _load_params(best_path)
  test_metrics, test_count = _evaluate(
      params, z, h, indices['test'], architecture, eval_chunk_size)
  for key, value in test_metrics.items():
    if key.startswith('Delta_'):
      if not np.isnan(value[0]) or not np.isfinite(value[1:]).all():
        raise AssertionError((key, value))
    elif not np.isfinite(value).all():
      raise AssertionError((key, value))
  nonconverged = bool(stopping_reason == 'MAX_UPDATES' and
                      probe_not_converged(validation_curve, best_update))
  checkpoint_hash_after = _sha256(checkpoint_file)
  production_params_after = _production_parameter_hash(checkpoint_file)
  if (checkpoint_hash_before != checkpoint_hash_after or
      production_params_before != production_params_after):
    raise AssertionError('Production checkpoint/parameters changed')

  rows = []
  for level in range(5):
    rows.append({
        'method': metadata['method'], 'game': metadata['game'],
        'training_seed': metadata['training_seed'], 'level': level + 1,
        'prefix_dim': architecture.prefix_dims[level],
        'block_dim': architecture.prefix_dims[level] - (
            architecture.prefix_dims[level - 1] if level else 0),
        **{key: float(value[level]) if np.isfinite(value[level]) else None
           for key, value in test_metrics.items()}})
  with (output / 'test_metrics.csv').open('w', newline='') as stream_file:
    writer = csv.DictWriter(stream_file, fieldnames=rows[0])
    writer.writeheader(); writer.writerows(rows)
  report = {
      'status': 'PASS', 'method': metadata['method'], 'game': metadata['game'],
      'evaluation_code_hashes': {
          'posthoc_train.py': _sha256(__file__),
          'posthoc_recon.py': _sha256(Path(__file__).with_name('posthoc_recon.py'))},
      'training_seed': metadata['training_seed'], 'diagnostic_decoder_seed': 0,
      'state_counts': {key: int(len(value)) for key, value in indices.items()},
      'initialization_hash': init_hash, 'parameter_counts_by_level': counts,
      'minibatch_index_stream_hash': stream_hash,
      'minibatch_rng': 'numpy.default_rng(0)/PCG64; relative canonical TRAIN indices; replacement=True',
      'trajectory_identity_hash': identity, 'split_manifest_hash': split_hash,
      'optimizer_updates': int(executed), 'selected_best_update': int(best_update),
      'stopping_reason': stopping_reason, 'best_validation_score': best_score,
      'PROBE_NOT_CONVERGED': nonconverged,
      'test_metrics': {key: [None if np.isnan(x) else float(x) for x in value]
                       for key, value in test_metrics.items()},
      'test_evaluations': 1, 'validation_checks': len(validation_curve),
      'checkpoint_agent_hash_before': checkpoint_hash_before,
      'checkpoint_agent_hash_after': checkpoint_hash_after,
      'production_parameter_hash_before': production_params_before,
      'production_parameter_hash_after': production_params_after,
      'production_frozen_verified': True,
      'elapsed_seconds': time.time() - start,
      **architecture_metadata(architecture, opt_config),
  }
  write_json(output / 'metadata.json', report)
  print(json.dumps(report, indent=2, sort_keys=True), flush=True)
  return report


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--method', required=True)
  parser.add_argument('--cache', required=True)
  parser.add_argument('--dataset', required=True)
  parser.add_argument('--output', required=True)
  args = parser.parse_args()
  run(args.method, args.cache, args.dataset, args.output)
