"""Accounting and validation helpers for the Wave-1 W&B media failure."""

import argparse
import hashlib
import json
from pathlib import Path

from .phase3_prepare import PROTOCOL_ID


ROOT = Path(__file__).resolve().parents[1]
FAILURE_REASON = 'wandb_video_logger_format'


def _write(path, payload):
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')


def _sha256_tree(path):
  digest = hashlib.sha256()
  for child in sorted(Path(path).rglob('*')):
    if child.is_file():
      digest.update(str(child.relative_to(path)).encode())
      digest.update(child.read_bytes())
  return digest.hexdigest()


def mark_failed_attempts(protocol_root):
  """Persistently classify exactly the 36 video-failed scientific attempts."""
  protocol_root = Path(protocol_root)
  attempts = []
  for launch_path in sorted(
      (protocol_root / 'training' / 'wave1').glob(
          '*/*/seed_*/attempt_*/launch.json')):
    launch = json.loads(launch_path.read_text())
    # The two initial online-W&B/DNS failures have no W&B run ID and retain
    # their distinct infrastructure classification.
    if launch.get('status') != 'FAILED' or not launch.get('wandb_run_id'):
      continue
    previous = launch['status']
    launch.update({
        'original_status': launch.get('original_status', previous),
        'status': 'TECHNICAL_FAILED',
        'technical_failure_reason': FAILURE_REASON,
        'invalid_for_paper': True,
        'scientific_seed_counted': False,
        'logger_fix_revision': 0,
    })
    _write(launch_path, launch)
    marker = {
        'protocol_id': PROTOCOL_ID,
        'method': launch['variant'],
        'game': launch['game'],
        'scientific_seed': launch['seed'],
        'attempt_id': launch['attempt_id'],
        'wandb_run_id': launch['wandb_run_id'],
        'technical_status': 'TECHNICAL_FAILED',
        'reason': FAILURE_REASON,
        'invalid_for_paper': True,
        'scientific_seed_counted': False,
    }
    _write(launch_path.parent / 'technical_status.json', marker)
    offline = launch.get('wandb_offline_directory')
    if offline:
      _write(Path(offline) / 'technical_status.json', marker)
    attempts.append({
        **marker, 'attempt_dir': str(launch_path.parent),
        'wandb_offline_directory': offline})
  if len(attempts) != 36:
    raise AssertionError(
        f'Expected exactly 36 video-failed attempts, found {len(attempts)}')

  invalid_checkpoints = []
  for attempt in attempts:
    if attempt['method'] != 'Backbone':
      continue
    checkpoint = Path(attempt['attempt_dir']) / 'ckpt/env_action_steps_000010000'
    if not checkpoint.exists():
      continue
    record = {
        'protocol_id': PROTOCOL_ID,
        'method': attempt['method'], 'game': attempt['game'],
        'scientific_seed': attempt['scientific_seed'],
        'attempt_id': attempt['attempt_id'],
        'checkpoint_action_milestone': 10_000,
        'checkpoint_path': str(checkpoint),
        'checkpoint_tree_sha256': _sha256_tree(checkpoint),
        'status': 'INVALID_FOR_PAPER',
        'reason': FAILURE_REASON,
        'prohibited_uses': [
            'AUC', 'shared_diagnostic_trajectories', 'paper_metrics',
            'restart_or_resume'],
    }
    _write(checkpoint / 'INVALID_FOR_PAPER.json', record)
    invalid_checkpoints.append(record)
  if len(invalid_checkpoints) != 5:
    raise AssertionError(
        f'Expected five invalid Backbone 10k checkpoints, found '
        f'{len(invalid_checkpoints)}')

  accounting = {
      'protocol_id': PROTOCOL_ID,
      'classification': 'TECHNICAL LOGGING FAILURE',
      'reason': FAILURE_REASON,
      'failed_scientific_attempt_count': len(attempts),
      'invalid_backbone_10k_checkpoint_count': len(invalid_checkpoints),
      'attempts': attempts,
      'invalid_checkpoints': invalid_checkpoints,
  }
  _write(protocol_root / 'failed_attempt_accounting.json', accounting)

  ledger_path = protocol_root / 'wave1_submission_ledger.json'
  ledger = json.loads(ledger_path.read_text())
  failed_ids = {row['wandb_run_id'] for row in attempts}
  changed = 0
  for row in ledger['submitted']:
    run_id = f'cw1j{int(row["job_index"]):03d}a{int(row["attempt"]):03d}'
    if run_id in failed_ids:
      row['active_claim'] = False
      row['technical_status'] = 'TECHNICAL_FAILED'
      row['reason'] = FAILURE_REASON
      changed += 1
  if changed != 36:
    raise AssertionError(f'Expected 36 ledger rows, updated {changed}')
  _write(ledger_path, ledger)
  return accounting


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--protocol-root', type=Path,
      default=ROOT / 'production_runs' / PROTOCOL_ID)
  args = parser.parse_args()
  result = mark_failed_attempts(args.protocol_root)
  print(json.dumps({
      'failed_attempts': result['failed_scientific_attempt_count'],
      'invalid_checkpoints': result['invalid_backbone_10k_checkpoint_count'],
  }, indent=2))


if __name__ == '__main__':
  main()
