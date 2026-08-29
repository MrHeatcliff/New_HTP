"""Validate and write the final pre-training audit decision."""

import json
from pathlib import Path

from .config import FINAL_CHECKPOINTS


REQUIRED_MANIFEST_FIELDS = {
    'method', 'game', 'seed', 'git_commit', 'resolved_config_hash',
    'env_action_steps', 'driver_callbacks', 'reset_callbacks',
    'replay_insertions', 'optimizer_updates', 'checkpoint_action_milestone',
    'checkpoint_hash', 'htp_enabled', 'projection_enabled', 'use_recon',
    'use_pdyn', 'prefix_dims', 'strides',
}


def generate(root='paper_artifacts/pretraining_audit'):
  root = Path(root)
  terminal = json.loads((root / 'terminal_audit_summary.json').read_text())
  differential = json.loads((root / 'legacy_test_diff.json').read_text())
  jobs = json.loads(Path(
      'paper_artifacts/corewm_phase3/training_manifest.json').read_text())
  manifest_valid = (
      len(jobs) == 780 and
      all(REQUIRED_MANIFEST_FIELDS <= set(job) for job in jobs) and
      all(job['checkpoint_action_milestone'] == list(FINAL_CHECKPOINTS)
          for job in jobs) and
      all(job['prefix_dims'] == 'N/A' and job['strides'] == 'N/A'
          for job in jobs if job['method'] == 'Backbone'))
  inert = terminal['mismatch_count'] == 0
  no_new_failures = differential['acceptance_new_failures_empty']
  status = 'PASS' if inert and no_new_failures and manifest_valid else 'BLOCKED'
  report = {
      'status': status,
      'paper_scale_training_launched': False,
      'train_ratio_counter': 'Driver callbacks/replay insertions (unchanged)',
      'canonical_budget_counter': 'env_action_steps',
      'paper_axis': 'env_action_steps',
      'auc_action_milestones': list(FINAL_CHECKPOINTS),
      'reset_observations_remain_in_replay': True,
      'terminal_audit': terminal,
      'legacy_differential': {
          key: differential[key] for key in (
              'base_commit', 'baseline_total', 'baseline_passed_count',
              'baseline_failed_count', 'current_total',
              'current_passed_count', 'current_failed_count',
              'newly_failing_node_ids',
              'previously_failing_now_passing_node_ids',
              'acceptance_new_failures_empty')},
      'training_manifest': {
          'jobs': len(jobs), 'required_fields_valid': manifest_valid,
          'backbone_projection_fields_explicit_na': True},
      'deterministic_resume_required': False,
      'uninterrupted_training_required': True,
      'failed_job_policy': 'restart seed from initialization',
      'remaining_blockers': [] if status == 'PASS' else [
          name for name, okay in (
              ('is_terminal mismatch', inert),
              ('new legacy test failure', no_new_failures),
              ('invalid training manifest', manifest_valid)) if not okay],
  }
  (root / 'pretraining_report.json').write_text(
      json.dumps(report, indent=2, sort_keys=True) + '\n')
  return report


if __name__ == '__main__':
  print(json.dumps(generate(), indent=2, sort_keys=True))
