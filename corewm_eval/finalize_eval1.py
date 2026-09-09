"""Validate and combine native and matched post-hoc Evaluation 1 outputs."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METHODS = ('flat', 'pdyn_only', 'full')


def run(posthoc_root, native_root, output):
  posthoc_root, native_root, output = map(
      Path, (posthoc_root, native_root, output))
  output.mkdir(parents=True, exist_ok=False)
  posthoc = {name: json.loads(
      (posthoc_root / name / 'metadata.json').read_text()) for name in METHODS}
  geometry = {name: json.loads(
      (native_root / name / 'metadata.json').read_text()) for name in METHODS}
  full_native = geometry['full']

  equality_fields = (
      'state_counts', 'initialization_hash', 'split_manifest_hash',
      'trajectory_identity_hash', 'minibatch_index_stream_hash')
  for field in equality_fields:
    values = [json.dumps(posthoc[name][field], sort_keys=True) for name in METHODS]
    if len(set(values)) != 1:
      raise AssertionError((field, values))
  for name in METHODS:
    result = posthoc[name]
    if not result['production_frozen_verified']:
      raise AssertionError((name, 'production not frozen'))
    if result['checkpoint_agent_hash_before'] != result['checkpoint_agent_hash_after']:
      raise AssertionError((name, 'checkpoint changed'))
    if result['production_parameter_hash_before'] != result['production_parameter_hash_after']:
      raise AssertionError((name, 'production parameters changed'))
    for key, values in result['test_metrics'].items():
      array = np.asarray([np.nan if x is None else x for x in values], float)
      valid = array[1:] if key.startswith('Delta_') else array
      if not np.isfinite(valid).all():
        raise AssertionError((name, key, values))
    if geometry[name]['split_manifest_hash'] != result['split_manifest_hash']:
      raise AssertionError((name, 'geometry split differs'))

  dims = (128, 256, 512, 1024, 2048)
  rows = []
  for name in METHODS:
    result, geom = posthoc[name], geometry[name]
    metrics = result['test_metrics']
    row = {
        'method': result['method'],
        **{f'posthoc_R2_rec_prefix_{dim}': metrics['R2_rec'][index]
           for index, dim in enumerate(dims)},
        **{f'posthoc_E_norm_prefix_{dim}': metrics['E_norm'][index]
           for index, dim in enumerate(dims)},
        'mean_conditional_block_R2': float(np.mean(geom['conditional_R2'])),
        'mean_offdiag_CKA': geom['mean_offdiag_CKA'],
        'best_validation_score': result['best_validation_score'],
        'selected_best_update': result['selected_best_update'],
        'PROBE_NOT_CONVERGED': result['PROBE_NOT_CONVERGED'],
    }
    rows.append(row)
  with (output / 'matched_posthoc_table.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=rows[0])
    writer.writeheader(); writer.writerows(rows)

  native_rows = []
  for index, dim in enumerate(dims):
    native_rows.append({
        'method': 'Full', 'level': index + 1, 'prefix_dim': dim,
        'native_E_rec': full_native['E_rec'][index],
        'native_Delta_rec': None if index == 0 else full_native['Delta_rec'][index - 1],
        'native_Delta_rec_dim': (
            None if index == 0 else full_native['Delta_rec_dim'][index - 1]),
    })
  with (output / 'full_native_reconstruction.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=native_rows[0])
    writer.writeheader(); writer.writerows(native_rows)

  report = {
      'status': ('BLOCKED_POSTHOC_PROBE_BUDGET_INSUFFICIENT'
                 if any(posthoc[name]['PROBE_NOT_CONVERGED'] for name in METHODS)
                 else 'PASS'),
      'game': 'alien', 'training_seed': 0,
      'matched_protocol_verified': True,
      'same_fields_verified': list(equality_fields),
      'state_counts': posthoc['flat']['state_counts'],
      'initialization_hash': posthoc['flat']['initialization_hash'],
      'minibatch_index_stream_hash': posthoc['flat']['minibatch_index_stream_hash'],
      'split_manifest_hash': posthoc['flat']['split_manifest_hash'],
      'posthoc': {name: {
          key: posthoc[name][key] for key in (
              'method', 'optimizer_updates', 'selected_best_update',
              'stopping_reason', 'best_validation_score',
              'PROBE_NOT_CONVERGED', 'test_metrics')}
          for name in METHODS},
      'native_full': {key: full_native[key] for key in (
          'E_rec', 'Delta_rec', 'Delta_rec_dim')},
      'conditional_redundancy': {
          name: geometry[name]['conditional_R2'] for name in METHODS},
      'block_cka': {name: {
          'shape': geometry[name]['cka_shape'],
          'mean_offdiag_CKA': geometry[name]['mean_offdiag_CKA']}
          for name in METHODS},
      'production_hashes_unchanged': True,
      'metrics_finite_except_defined_level1_delta_na': True,
      'remaining_wave1_jobs_launched': 0,
  }
  (output / 'eval1_alien_seed0.json').write_text(
      json.dumps(report, indent=2, sort_keys=True) + '\n')
  print(json.dumps(report, indent=2, sort_keys=True))
  return report


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--posthoc-root', required=True)
  parser.add_argument('--native-root', required=True)
  parser.add_argument('--output', required=True)
  args = parser.parse_args()
  run(args.posthoc_root, args.native_root, args.output)
