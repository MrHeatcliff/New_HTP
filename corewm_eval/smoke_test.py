"""Phase-2 unit and real-checkpoint integration smoke entrypoint."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import elements
import numpy as np
import ruamel.yaml as yaml

from dreamerv3 import main_htp

from .decoder import assert_image_domain, normalize_ground_truth
from .extraction import (
    ExtractionSpec, evaluation_seed, extract_readonly, open_loop_readonly,
    previous_actions)
from .metadata import sha256_file, write_metadata
from .metrics import assert_nonnegative, cka_matrix, reconstruction_metrics
from .open_loop import rollout_error_families, validate_shared_backbone_rollout
from .slicing import block_dims


def _tree_hash(tree):
  digest = hashlib.sha256()
  for key in sorted(tree):
    value = np.asarray(tree[key])
    digest.update(key.encode())
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
  return digest.hexdigest()


def _load_config(path):
  data = yaml.YAML(typ='safe').load(Path(path).read_text())
  data['jax']['precompile'] = False
  data['jax']['prealloc'] = False
  # The read-only extractor intentionally returns large host-side diagnostics.
  # Disable only JAX's implicit-transfer guard; model numerics are unchanged.
  data['jax']['transfer_guard'] = False
  data['logger']['outputs'] = ['jsonl']
  return elements.Config(data)


def _load_smoke_trajectory(replay_dir, length=65):
  for path in sorted(Path(replay_dir).glob('*.npz')):
    with np.load(path) as data:
      if len(data['image']) >= length:
        keys = ('image', 'reward', 'is_first', 'is_last', 'is_terminal', 'action')
        return path, {key: np.asarray(data[key][:length]) for key in keys}
  raise RuntimeError(f'No contiguous smoke trajectory of length {length} in {replay_dir}')


def run_integration(checkpoint, config_path, replay_dir, output):
  checkpoint, config_path = Path(checkpoint), Path(config_path)
  agent_file = checkpoint / 'agent.pkl'
  before_checkpoint = sha256_file(agent_file)
  config = _load_config(config_path)
  agent = main_htp.make_agent(config)
  cp = elements.Checkpoint()
  cp.agent = agent
  cp.load(checkpoint, keys=['agent'])
  host_params = agent.save()['params']
  if 'dyn/dynin0/kernel' not in host_params:
    raise AssertionError(f'Loaded checkpoint lacks dyn parameters: {list(host_params)[:20]}')
  params_before = _tree_hash(host_params)
  updates_before = int(agent.n_updates)

  replay_path, trajectory = _load_smoke_trajectory(replay_dir)
  obs = {key: value[None] for key, value in trajectory.items() if key != 'action'}
  prevact = {'action': previous_actions(trajectory['action'])[None]}
  dyn = config.agent.dyn[config.agent.dyn.typ]
  dims = tuple(map(int, config.agent.htp.proj.dims))
  spec = ExtractionSpec(int(dyn.deter), (int(dyn.stoch), int(dyn.classes)), dims)
  seed = evaluation_seed(0, episode_id=0, start_timestep=0, stream=0)
  first = extract_readonly(agent, obs, prevact, spec, seed, host_params)
  second = extract_readonly(agent, obs, prevact, spec, seed, host_params)

  h, z = first['h'], first['z']
  assert h.shape[-1] == spec.deter_dim + int(np.prod(spec.stoch_shape)) == 2560
  assert z.shape[-1] == dims[-1] == 2048
  assert tuple(x.shape[-1] for x in first['prefixes']) == dims
  expected_blocks = block_dims(dims)
  assert tuple(x.shape[-1] for x in first['blocks']) == expected_blocks
  np.testing.assert_array_equal(np.concatenate(first['blocks'], -1), z)
  np.testing.assert_array_equal(first['prefixes'][1][..., :dims[0]], first['prefixes'][0])
  np.testing.assert_allclose(
      np.asarray(first['h'], np.float32), np.asarray(second['h'], np.float32),
      rtol=0, atol=0)
  np.testing.assert_allclose(
      np.asarray(first['z'], np.float32), np.asarray(second['z'], np.float32),
      rtol=0, atol=0)
  for key in first['decoded_original']:
    np.testing.assert_allclose(
        first['decoded_original'][key], first['decoded_recovered'][key],
        rtol=1e-6, atol=1e-6)
    assert_image_domain(first['decoded_original'][key], f'decoded_original/{key}')
    for decoded in first['decoded_prefixes']:
      assert_image_domain(decoded[key], f'decoded_prefix/{key}')
  assert_image_domain(normalize_ground_truth(trajectory['image']), 'ground_truth')

  errors, gains, gains_dim = reconstruction_metrics(
      h, first['cumulative_reconstructions'], expected_blocks)
  assert_nonnegative('E_rec', errors)
  flat_blocks = [x.reshape((-1, x.shape[-1])) for x in first['blocks']]
  cka = cka_matrix(flat_blocks)
  assert cka.shape == (5, 5) and np.isfinite(cka).all()

  start_state = {
      'deter': first['posterior']['deter'][:, 0],
      'stoch': first['posterior']['stoch'][:, 0],
  }
  rollout_actions = {'action': trajectory['action'][:64][None]}
  rollout = open_loop_readonly(
      agent, start_state, rollout_actions, spec,
      evaluation_seed(0, 0, 0, stream=1), host_params)
  assert rollout['h_tilde'].shape == (1, 64, 2560)
  assert rollout['z_tilde'].shape == (1, 64, 2048)
  validate_shared_backbone_rollout(
      rollout['h_tilde'], rollout['cumulative_reconstructions'])
  np.testing.assert_array_equal(
      np.asarray(rollout['actions_used']['action']), rollout_actions['action'])
  gt_future = normalize_ground_truth(trajectory['image'][1:65])
  full_future = np.asarray(rollout['decoded_full']['image'][0])
  prefix_future = [np.asarray(x['image'][0]) for x in rollout['decoded_prefixes']]
  rollout_metrics = rollout_error_families(
      gt_future, full_future, prefix_future,
      np.asarray(rollout['h_tilde'][0]),
      [np.asarray(x[0]) for x in rollout['cumulative_reconstructions']])
  assert all(np.isfinite(np.asarray(value)).all()
             for row in rollout_metrics for value in row.values())
  horizon_indices = [k - 1 for k in (1, 2, 4, 8, 16, 32, 64)]
  rollout_summary = {
      name: [[float(row[name][index]) for index in horizon_indices]
             for row in rollout_metrics]
      for name in ('E_h', 'E_prefix', 'E_backbone', 'E_total')}

  params_after = _tree_hash(agent.save()['params'])
  after_checkpoint = sha256_file(agent_file)
  assert params_before == params_after
  assert updates_before == int(agent.n_updates)
  assert before_checkpoint == after_checkpoint

  output = Path(output)
  output.mkdir(parents=True, exist_ok=True)
  report = {
      'SMOKE_ONLY': True,
      'checkpoint': str(checkpoint),
      'checkpoint_sha256_before': before_checkpoint,
      'checkpoint_sha256_after': after_checkpoint,
      'checkpoint_unchanged': before_checkpoint == after_checkpoint,
      'parameter_hash_before': params_before,
      'parameter_hash_after': params_after,
      'parameters_unchanged': params_before == params_after,
      'optimizer_updates_before': updates_before,
      'optimizer_updates_after': int(agent.n_updates),
      'ema_update_called': False,
      'replay_modified': False,
      'replay_source': str(replay_path),
      'evaluation_seed': 0,
      'rng_derivation': 'numpy.SeedSequence([global_seed, episode_id, start_timestep, stream]) -> uint32[2]',
      'h_shape': list(h.shape),
      'z_shape': list(z.shape),
      'prefix_dims': list(dims),
      'block_dims': list(expected_blocks),
      'decoder_roundtrip_max_abs': float(max(
          np.max(np.abs(first['decoded_original'][k] - first['decoded_recovered'][k]))
          for k in first['decoded_original'])),
      'continuous_reconstructed_stoch_decoder_finite': True,
      'E_rec_smoke_only': errors.tolist(),
      'Delta_rec_smoke_only': [
          None if not np.isfinite(value) else float(value) for value in gains],
      'Delta_rec_dim_smoke_only': [
          None if not np.isfinite(value) else float(value) for value in gains_dim],
      'cka_shape': list(cka.shape),
      'cka_finite': bool(np.isfinite(cka).all()),
      'deterministic_repeat_equal': True,
      'open_loop_horizons': [1, 2, 4, 8, 16, 32, 64],
      'open_loop_h_shape': list(rollout['h_tilde'].shape),
      'open_loop_z_shape': list(rollout['z_tilde'].shape),
      'open_loop_shared_backbone_verified': True,
      'open_loop_recorded_actions_exact': True,
      'open_loop_all_error_families_finite': True,
      'open_loop_metrics_smoke_only': rollout_summary,
  }
  (output / 'integration_report.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
  with (output / 'open_loop_metrics_smoke_only.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=(
        'level', 'prefix_dim', 'horizon', 'E_h', 'E_prefix',
        'E_backbone', 'E_total'))
    writer.writeheader()
    for level, (prefix_dim, row) in enumerate(zip(dims, rollout_metrics), 1):
      for horizon in (1, 2, 4, 8, 16, 32, 64):
        index = horizon - 1
        writer.writerow({
            'level': level, 'prefix_dim': prefix_dim, 'horizon': horizon,
            **{name: float(row[name][index]) for name in (
                'E_h', 'E_prefix', 'E_backbone', 'E_total')}})
  write_metadata(
      output / 'metadata.json', SMOKE_ONLY=True, checkpoint=str(checkpoint),
      checkpoint_hash=before_checkpoint, config=str(config_path), game='alien',
      seed=0, env_steps=1320, prefix_dims=list(dims),
      block_dims=list(expected_blocks), split_definition='not_applicable_phase2_integration',
      probe_config='not_run_phase2')
  print(json.dumps(report, indent=2, sort_keys=True))
  return report


def main(argv=None):
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint')
  parser.add_argument('--config')
  parser.add_argument('--replay-dir')
  parser.add_argument('--output', default='/tmp/corewm-eval-phase2-alien-smoke')
  args = parser.parse_args(argv)
  if not all((args.checkpoint, args.config, args.replay_dir)):
    parser.error('--checkpoint, --config, and --replay-dir are required')
  run_integration(args.checkpoint, args.config, args.replay_dir, args.output)


if __name__ == '__main__':
  main()
