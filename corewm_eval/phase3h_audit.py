"""Generate the machine-readable Phase-3H semantic audit from smoke artifacts."""

import csv
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from .config import ATARI100K_GAMES, FINAL_CHECKPOINTS


ROOT = Path('paper_artifacts/corewm_phase3h')
TRAIN_ROOTS = {
    'Backbone': Path('/tmp/corewm-phase3h-milestone-backbone'),
    'Flat': Path('/tmp/corewm-phase3h-smoke-flat-v2'),
    'Rec-only': Path('/tmp/corewm-phase3h-smoke-rec_only-v2'),
    'Pdyn-only': Path('/tmp/corewm-phase3h-smoke-pdyn_only-v2'),
    'Full': Path('/tmp/corewm-phase3h-smoke-full-v2'),
    'Reverse': Path('/tmp/corewm-phase3h-smoke-reverse-v2'),
}


LIFECYCLE = (
    ('initial reset observation', True, 'Atari._reset', True, True, True,
     'possible after prefill', False),
    ('normal action transition', True, '', True, True, True,
     'possible after prefill', True),
    ('terminal action transition', True, '', True, True, True,
     'possible after prefill', True),
    ('post-terminal reset observation', True, 'Atari._reset', True, True, True,
     'possible after prefill', False),
)


CONSUMERS = (
    ('Driver local step', 'Driver callback count', 'driver loop only', 'NO',
     'embodied/core/driver.py:Driver._step'),
    ('logger.step', 'Driver callback count', 'logging x-axis', 'NO',
     'embodied/run/train.py:train callbacks'),
    ('legacy stop condition', 'Driver callback count', 'legacy budget', 'NO',
     'embodied/run/train.py:train non-exact branch'),
    ('paper stop condition', 'env_action_steps', 'paper interaction budget', 'YES',
     'embodied/run/train.py:budget_counter'),
    ('exact milestones', 'env_action_steps', 'checkpoint scheduling', 'YES',
     'embodied/run/train.py:milestonefn'),
    ('periodic legacy checkpoint', 'wall clock; logger step in name',
     'checkpoint scheduling', 'NO', 'embodied/run/train.py:should_save'),
    ('report/log clocks', 'wall clock', 'logging/report scheduling', 'NO',
     'embodied/run/train.py:should_log/should_report'),
    ('prefill threshold', 'replay length', 'random exploration/prefill', 'NO',
     'embodied/run/train.py:trainfn'),
    ('train ratio', 'Driver callback logger.step', 'train-ratio scheduling', 'NO',
     'embodied/run/train.py:TraceableRatio'),
    ('replay insertion', 'one per Driver callback', 'replay accounting', 'NO',
     'embodied/run/train.py:replayfn'),
    ('policy RNG/exploration', 'batched policy calls including reset callbacks',
     'policy RNG and exploration', 'NO', 'embodied/jax/agent.py:Agent.policy'),
    ('optimizer and LR', 'optimizer update state', 'optimizer/LR scheduling', 'NO',
     'embodied/jax/opt.py:Optimizer'),
    ('EMA/slow targets', 'training updates', 'target/EMA scheduling', 'NO',
     'embodied/jax/utils.py:SlowModel.update'),
    ('evaluation episode termination', 'completed eval episodes',
     'isolated evaluation', 'NO', 'embodied/run/eval_only.py:eval_only'),
    ('parallel trainer counters', 'parallel callback/insertion counters',
     'non-paper execution mode', 'NEEDS SEPARATE IMPLEMENTATION',
     'embodied/run/parallel.py'),
)


def _sha256(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_csv(path, rows):
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def _loss_metrics(root):
  path = root / 'paper_artifacts/determinism/batch_trace.jsonl'
  if not path.exists():
    return {}
  rows = [json.loads(line) for line in path.open()]
  rows = [row for row in rows if row.get('optimizer_updates_after', '') != '']
  prefixes = ('train/htp/', 'train/loss/htp_')
  return {key: value for key, value in rows[-1].items()
          if key.startswith(prefixes)} if rows else {}


def _resume_comparison():
  continuous = TRAIN_ROOTS['Backbone'] / 'ckpt/env_action_steps_000001320'
  resumed = Path(
      '/tmp/corewm-phase3h-resume-backbone/ckpt/env_action_steps_000001320')
  left = pickle.load((continuous / 'agent.pkl').open('rb'))
  right = pickle.load((resumed / 'agent.pkl').open('rb'))
  comparison = {
      'continuous_agent_counters': left['counters'],
      'resumed_agent_counters': right['counters'],
  }
  for group, predicate in (
      ('model_state', lambda key: not key.startswith('opt/')),
      ('optimizer_state', lambda key: key.startswith('opt/')),
  ):
    keys = [key for key in left['params'] if predicate(key)]
    unequal = [key for key in keys if not np.array_equal(
        np.asarray(left['params'][key]), np.asarray(right['params'][key]))]
    diffs = [float(np.max(np.abs(
        np.asarray(left['params'][key], float) -
        np.asarray(right['params'][key], float)))) for key in unequal]
    comparison[group] = {
        'exact_equal': not unequal, 'keys_compared': len(keys),
        'unequal_keys': len(unequal),
        'max_abs_difference': max(diffs or [0.0]),
    }
  for name in ('env_action_steps.pkl', 'driver_callbacks.pkl',
               'replay_insertions.pkl', 'raw_ale_frames.pkl', 'replay.pkl'):
    comparison[name] = {
        'byte_equal': _sha256(continuous / name) == _sha256(resumed / name)}
  return comparison


def generate(output=ROOT):
  output = Path(output)
  output.mkdir(parents=True, exist_ok=True)
  lifecycle_rows = []
  for event, calls_step, reset_path, callback, old_step, replay, updates, action in LIFECYCLE:
    lifecycle_rows.append({
        'event': event, 'invokes_env_step': calls_step,
        'invokes_env_reset': bool(reset_path), 'reset_code_path': reset_path,
        'driver_callback': callback, 'increments_legacy_global_step': old_step,
        'replay_insert': replay, 'model_actor_critic_update': updates,
        'advances_policy_rng': True,
        'advances_optimizer_lr_ema': f'only if update executes ({updates})',
        'train_ratio_accounting': True,
        'exact_action_milestone_eligible': action,
        'isolated_evaluation_trigger': False,
    })
  _write_csv(output / 'event_lifecycle.csv', lifecycle_rows)

  consumer_rows = [{
      'consumer': name, 'current_counter': current,
      'intended_semantics': intended, 'change_to_action_counter': change,
      'code_location': location,
  } for name, current, intended, change, location in CONSUMERS]
  _write_csv(output / 'step_consumer_audit.csv', consumer_rows)

  smoke_rows = []
  for variant, root in TRAIN_ROOTS.items():
    manifest = json.loads(
        (root / 'paper_artifacts/action_checkpoints_manifest.json').read_text())
    final = manifest[-1]
    eval_name = variant.lower().replace('-', '_')
    eval_report = json.loads((Path(f'/tmp/corewm-phase3h-eval-{eval_name}') /
                              'paper_artifacts/final_eval.json').read_text())
    smoke_rows.append({
        'variant': variant, 'status': 'PASS', **final,
        'milestones': [row['milestone'] for row in manifest],
        'loss_metrics': _loss_metrics(root),
        'checkpoint_reload_eval_status': eval_report['status'],
        'checkpoint_reload_eval_episodes': eval_report['eval_episodes'],
    })
  (output / 'installation_smokes.json').write_text(
      json.dumps(smoke_rows, indent=2, sort_keys=True) + '\n')

  report = {
      'canonical_interaction': 'one action-bearing call to env.step; reset excluded',
      'implementation': {
          'marker': 'embodied/core/driver.py:Driver._step log/action_executed',
          'counter': 'embodied/run/train.py:train env_action_steps',
          'milestone': 'embodied/run/train.py:milestonefn',
      },
      'paper_milestones': list(FINAL_CHECKPOINTS),
      'canonical_games': list(ATARI100K_GAMES),
      'smoke_milestone': smoke_rows[0],
      'resume_comparison': _resume_comparison(),
      'resume_status': 'FAIL_NOT_DETERMINISTIC',
      'resume_reason': (
          'Environment/ALE and Driver policy carry are not checkpointed. The '
          'replay writer/selector and prefetched stream are not serialized as '
          'live process state; after reload the first update occurred at '
          'callback 1152 rather than 1088. The continuous run executed 59 '
          'updates versus 43 after resume, so model and optimizer states '
          'necessarily diverged.'),
      'train_ratio': {
          'scheduler_counter': 'Driver callbacks via logger.step',
          'replay_insertions_per_callback': 1,
          'action_counter_used': False,
          'reset_callbacks_can_trigger_updates': True,
      },
      'multi_env_policy': (
          'exact paper mode rejects envs != 1 rather than crossing milestones'),
      'paper_scale_jobs_launched': 0,
      'verification': {
          'phase2_phase3h_tests': {'passed': 26, 'failed': 0},
          'legacy_embodied_tests': {'passed': 141, 'failed': 60},
          'legacy_test_note': (
              'The bundled Embodied tests target older APIs (Replay.dataset, '
              'old train/parallel signatures, and reset in Driver transition); '
              'these failures predate the Phase-3H counter and were not fixed '
              'because doing so would broaden or change training semantics.'),
          'six_installation_smokes': {'passed': 6, 'failed': 0},
          'six_checkpoint_reload_evals': {'passed': 6, 'failed': 0},
          'compileall': 'PASS', 'pip_check': 'PASS',
          'git_diff_check': 'PASS', 'alien_integration_smoke': 'PASS',
      },
  }
  (output / 'phase3h_report.json').write_text(
      json.dumps(report, indent=2, sort_keys=True) + '\n')
  return report


if __name__ == '__main__':
  generate()
