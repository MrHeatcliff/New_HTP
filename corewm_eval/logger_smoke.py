"""Run and compare isolated logger-remediation training smokes."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from .phase3_prepare import PROTOCOL_ID, VARIANTS


ROOT = Path(__file__).resolve().parents[1]
REMEDIATION_ROOT = (
    ROOT / 'production_runs' / PROTOCOL_ID / 'logger_remediation')


def _variant_slug(variant):
  return variant.lower().replace('-', '_')


def _configs(variant):
  return [
      'atari100k', 'size12m', 'corewm_paper_protocol', 'wandb',
      *VARIANTS[variant]]


def _sha256(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(kind, variant='Full'):
  if variant not in VARIANTS:
    raise ValueError(variant)
  definitions = {
      'equivalence_local': (1320, (660, 1320), True, False),
      'equivalence_scalar': (1320, (660, 1320), True, True),
      'equivalence_local_isolated': (1320, (660, 1320), True, False),
      'equivalence_scalar_isolated': (1320, (660, 1320), True, True),
      'first_flush_12k': (12000, (10000, 12000), False, True),
      'config_flush': (1320, (660, 1320), True, True),
  }
  steps, milestones, fast_flush, wandb_enabled = definitions[kind]
  suffix = _variant_slug(variant) if kind == 'config_flush' else 'full_alien'
  logdir = REMEDIATION_ROOT / kind / suffix
  if logdir.exists():
    raise FileExistsError(f'Never overwrite a remediation run: {logdir}')
  logdir.parent.mkdir(parents=True, exist_ok=True)
  module = 'dreamerv3.main' if variant == 'Backbone' else 'dreamerv3.main_htp'
  command = [
      sys.executable, '-m', module, '--configs', *_configs(variant),
      '--task', 'atari100k_alien', '--seed', '0', '--logdir', str(logdir),
      '--run.steps', str(steps), '--run.action_milestones',
      *(str(x) for x in milestones)]
  if fast_flush:
    command += ['--run.log_every', '1', '--run.report_every', '1']
  if not wandb_enabled:
    command += ['--logger.outputs', 'jsonl', 'scope']
  env = os.environ.copy()
  env.update({
      'PAPER_PROTOCOL_ID': PROTOCOL_ID,
      'PAPER_EXPERIMENT_ID': f'{PROTOCOL_ID}_logger_remediation',
      'PAPER_ATTEMPT_ID': kind,
      'PAPER_LOGGER_FIX_REVISION': '1',
      'PAPER_METHOD': variant,
      'PAPER_CONDITION': _variant_slug(variant),
      'PAPER_DETERMINISTIC_UUID': '1',
      'PAPER_DETERMINISM_TRACE': '1',
      'PAPER_ACTION_SEMANTICS_TRACE': '1',
      'WANDB_MODE': 'offline',
      'WANDB_ENTITY': 'ttdat170703-ho-chi-minh-city-university-of-technology',
      'WANDB_PROJECT': 'dreamv3-up_n_down-alien',
      'WANDB_RUN_ID': f'logger-{kind}-{_variant_slug(variant)}',
      'WANDB_RUN_NAME': f'LOGGER_TEST__{kind}__{_variant_slug(variant)}',
      'WANDB_TAGS': ','.join((
          PROTOCOL_ID, 'logger_remediation', kind, variant,
          'SMOKE_ONLY', 'logger_fix_revision_1')),
      'WANDB_DIR': str(REMEDIATION_ROOT / 'wandb'),
  })
  (logdir.parent / f'{suffix}_command.json').write_text(
      json.dumps(command, indent=2) + '\n')
  result = subprocess.run(command, cwd=ROOT, env=env, check=False)
  status = {
      'kind': kind, 'variant': variant, 'steps': steps,
      'milestones': list(milestones), 'wandb_enabled': wandb_enabled,
      'media_enabled': False, 'exit_code': result.returncode,
      'logdir': str(logdir),
  }
  if result.returncode == 0:
    manifest = json.loads((
        logdir / 'paper_artifacts/action_checkpoints_manifest.json').read_text())
    status['checkpoint_steps'] = [x['env_action_steps'] for x in manifest]
    status['final_checkpoint_hash'] = manifest[-1]['checkpoint_hash']
    final = manifest[-1]
    status['env_action_steps'] = final['env_action_steps']
    status['driver_callbacks'] = final['driver_callbacks']
    status['optimizer_updates'] = final['optimizer_updates']
    status['reset_callbacks'] = final['reset_callbacks']
  status_path = logdir.parent / f'{suffix}_status.json'
  status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + '\n')
  if result.returncode:
    raise SystemExit(result.returncode)
  if status['checkpoint_steps'] != list(milestones):
    raise AssertionError(status)
  if status['env_action_steps'] != steps:
    raise AssertionError(status)
  return status


def compare_equivalence():
  roots = {
      name: REMEDIATION_ROOT / name / 'full_alien'
      for name in (
          'equivalence_local_isolated', 'equivalence_scalar_isolated')}
  for path in roots.values():
    if not path.exists():
      raise FileNotFoundError(path)
  checkpoint_rel = Path('ckpt/env_action_steps_000001320')
  files = [
      'agent.pkl', 'env_action_steps.pkl', 'driver_callbacks.pkl',
      'reset_callbacks.pkl', 'replay_insertions.pkl', 'raw_ale_frames.pkl',
      'action_metadata.pkl', 'replay.pkl', 'step.pkl']
  hashes = {
      mode: {name: _sha256(path / checkpoint_rel / name) for name in files}
      for mode, path in roots.items()}
  mismatches = {
      name: {mode: values[name] for mode, values in hashes.items()}
      for name in files
      if len({values[name] for values in hashes.values()}) != 1}
  def semantic_rows(path, keys):
    return [
        {key: row.get(key) for key in keys}
        for row in (json.loads(line) for line in path.read_text().splitlines())]

  traces = {
      'action_trace': (
          Path('paper_artifacts/determinism/action_trace.jsonl'),
          ('event_index', 'worker', 'transition_hash', 'action_hash', 'reward',
           'is_first', 'is_last', 'is_terminal', 'actions',
           'env_action_steps', 'driver_callbacks', 'optimizer_updates')),
      'batch_trace': (
          Path('paper_artifacts/determinism/batch_trace.jsonl'),
          ('step', 'update_index_in_step', 'optimizer_updates_before',
           'optimizer_updates_after', 'batch_hash', 'stepid_hash',
           'reward_hash', 'action_hash', 'is_first_hash', 'is_last_hash',
           'is_terminal_hash')),
      'event_trace': (
          Path('paper_artifacts/env_action_semantics/event_trace.jsonl'),
          ('event_index', 'is_first', 'is_last', 'action_executed',
           'env_action_steps', 'driver_callbacks', 'reset_callbacks',
           'replay_insertions', 'model_updates', 'actor_updates',
           'critic_updates', 'updates_this_event')),
  }
  trace_hashes = {}
  for name, (relative, keys) in traces.items():
    values = {
        mode: semantic_rows(path / relative, keys)
        for mode, path in roots.items()}
    encoded = {
        mode: hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
        for mode, rows in values.items()}
    trace_hashes[name] = encoded
    if len(set(encoded.values())) != 1:
      mismatches[name] = {
          mode: {'sha256': encoded[mode], 'rows': len(values[mode])}
          for mode in values}
  result = {
      'checkpoint_action_step': 1320,
      'checkpoint_file_hashes': hashes,
      'trace_hashes': trace_hashes,
      'mismatches': mismatches,
      'pass': not mismatches,
      'comparison_scope': (
          'model/optimizer/RNG checkpoint, counters, action trace, update trace'),
  }
  output = REMEDIATION_ROOT / 'training_equivalence.json'
  output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
  if mismatches:
    raise AssertionError(json.dumps(mismatches, indent=2))
  return result


def main():
  parser = argparse.ArgumentParser()
  sub = parser.add_subparsers(dest='command', required=True)
  execute = sub.add_parser('run')
  execute.add_argument('--kind', required=True, choices=(
      'equivalence_local', 'equivalence_scalar', 'first_flush_12k',
      'equivalence_local_isolated', 'equivalence_scalar_isolated',
      'config_flush'))
  execute.add_argument('--variant', default='Full', choices=tuple(VARIANTS))
  sub.add_parser('compare-equivalence')
  args = parser.parse_args()
  result = (
      run(args.kind, args.variant) if args.command == 'run'
      else compare_equivalence())
  print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
  main()
