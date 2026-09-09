"""Read-only validation of completed production training attempts."""

import hashlib
import json
import math
from pathlib import Path

from .config import FINAL_CHECKPOINTS


FROZEN_COMMIT = '5ee4f27a0ba7fd7bdbdcbd4823fb2d539f8b4b47'
FULL_PREFIX_DIMS = [128, 256, 512, 1024, 2048]
FULL_STRIDES = [16, 8, 4, 2, 1]


def checkpoint_tree_hash(path):
  """Hash a checkpoint directory using the frozen production convention."""
  path = Path(path)
  digest = hashlib.sha256()
  for item in sorted(x for x in path.rglob('*') if x.is_file()):
    digest.update(str(item.relative_to(path)).encode())
    with item.open('rb') as stream:
      for chunk in iter(lambda: stream.read(1024 * 1024), b''):
        digest.update(chunk)
  return digest.hexdigest()


def _numbers(value):
  if isinstance(value, dict):
    for child in value.values():
      yield from _numbers(child)
  elif isinstance(value, list):
    for child in value:
      yield from _numbers(child)
  elif isinstance(value, (int, float)) and not isinstance(value, bool):
    yield float(value)


def _jsonl(path):
  path = Path(path)
  if not path.exists():
    return []
  rows = []
  for lineno, line in enumerate(path.read_text().splitlines(), 1):
    if not line.strip():
      continue
    try:
      rows.append(json.loads(line))
    except json.JSONDecodeError as exc:
      raise AssertionError(f'{path}:{lineno}: {exc}') from exc
  return rows


def validate_full_training_attempt(run_dir, *, verify_hashes=True):
  """Validate one exited Full attempt without inspecting its reward."""
  run_dir = Path(run_dir).resolve()
  launch_path = run_dir / 'launch.json'
  launch = json.loads(launch_path.read_text())
  if launch.get('exit_code') != 0 or launch.get('status') != 'TRAINING_EXITED':
    raise AssertionError(('training process did not exit cleanly', launch.get('status')))
  expected = {
      'protocol_id': 'corewm_atari100k_v1', 'method': 'Full',
      'variant': 'Full', 'git_commit': FROZEN_COMMIT,
      'use_recon': True, 'use_pdyn': True, 'htp_enabled': True,
      'projection_enabled': True, 'prefix_dims': FULL_PREFIX_DIMS,
      'strides': FULL_STRIDES,
  }
  for key, value in expected.items():
    if launch.get(key) != value:
      raise AssertionError((key, launch.get(key), value))

  manifest_path = run_dir / 'paper_artifacts/action_checkpoints_manifest.json'
  rows = json.loads(manifest_path.read_text())
  if [row['milestone'] for row in rows] != list(FINAL_CHECKPOINTS):
    raise AssertionError('Missing or incorrectly ordered exact action milestones')
  for row in rows:
    milestone = int(row['milestone'])
    checks = {
        'checkpoint_action_milestone': milestone,
        'env_action_steps': milestone,
        'git_commit': FROZEN_COMMIT,
        'method': 'Full', 'htp_enabled': True,
        'projection_enabled': True, 'use_recon': True, 'use_pdyn': True,
        'prefix_dims': FULL_PREFIX_DIMS, 'strides': FULL_STRIDES,
    }
    for key, value in checks.items():
      if row.get(key) != value:
        raise AssertionError((milestone, key, row.get(key), value))
    checkpoint = Path(row['checkpoint'])
    if not checkpoint.is_dir():
      raise AssertionError(f'Missing checkpoint: {checkpoint}')
    if verify_hashes and checkpoint_tree_hash(checkpoint) != row['checkpoint_hash']:
      raise AssertionError(f'Checkpoint hash mismatch: {checkpoint}')

  metric_rows = []
  for relative in ('metrics.jsonl', 'paper_artifacts/train_metrics.jsonl'):
    metric_rows.extend(_jsonl(run_dir / relative))
  if not metric_rows:
    raise AssertionError('No local machine-readable training metrics')
  nonfinite = [value for row in metric_rows for value in _numbers(row)
               if not math.isfinite(value)]
  if nonfinite:
    raise AssertionError(f'Non-finite training values: {len(nonfinite)}')
  htp = [row for row in metric_rows if 'train/htp/rec_branch_executed' in row]
  if not htp:
    raise AssertionError('No runtime HTP branch evidence')
  if not all(bool(row['train/htp/rec_branch_executed']) for row in htp):
    raise AssertionError('Full reconstruction branch not always active')
  if not all(bool(row['train/htp/pdyn_branch_executed']) for row in htp):
    raise AssertionError('Full predictive-dynamics branch not always active')

  final = rows[-1]
  result = {
      'status': 'VALID_COMPLETE', 'checkpoint_count': len(rows),
      'nonfinite_training_values': 0, 'exact_action_budget': True,
      'checkpoint_hashes_verified': bool(verify_hashes),
      'runtime_objectives_verified': True,
      'runtime_prefix_strides_verified': True,
      'env_action_steps': final['env_action_steps'],
      'driver_callbacks': final['driver_callbacks'],
      'reset_callbacks': final['reset_callbacks'],
      'replay_insertions': final['replay_insertions'],
      'optimizer_updates': final['optimizer_updates'],
      'reset_fraction': final['reset_callbacks_per_env_action'],
      'updates_per_action': final['optimizer_updates_per_env_action'],
      'final_checkpoint': final['checkpoint'],
      'final_checkpoint_hash': final['checkpoint_hash'],
  }
  return result


def mark_full_training_valid(run_dir, *, verify_hashes=True):
  """Validate then atomically mark an attempt; policy eval remains separate."""
  run_dir = Path(run_dir).resolve()
  result = validate_full_training_attempt(run_dir, verify_hashes=verify_hashes)
  launch_path = run_dir / 'launch.json'
  launch = json.loads(launch_path.read_text())
  launch['status'] = 'VALID_COMPLETE'
  launch['training_status'] = 'VALID_COMPLETE'
  launch['policy_eval_status'] = launch.get('policy_eval_status', 'PENDING')
  launch['representation_eval_status'] = launch.get(
      'representation_eval_status', 'PENDING')
  launch['validation'] = result
  temporary = launch_path.with_suffix('.tmp')
  temporary.write_text(json.dumps(launch, indent=2) + '\n')
  temporary.replace(launch_path)
  return result
