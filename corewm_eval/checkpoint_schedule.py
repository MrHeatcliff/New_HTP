"""Audit interaction/frame/logger/checkpoint counter semantics."""

import json
from pathlib import Path

import numpy as np


def audit_smoke(replay_dir, reported_step, optimizer_updates, action_repeat=4):
  replay_dir = Path(replay_dir)
  driver_rows = reset_rows = terminal_rows = 0
  for path in replay_dir.glob('*.npz'):
    with np.load(path) as data:
      driver_rows += len(data['is_first'])
      reset_rows += int(data['is_first'].sum())
      terminal_rows += int(data['is_terminal'].sum())
  action_bearing = driver_rows - reset_rows
  return {
      'driver_callback_rows': driver_rows,
      'reset_observation_rows': reset_rows,
      'agent_actions_actual': action_bearing,
      'raw_frames_nominal_upper_bound': action_bearing * int(action_repeat),
      'logger_counter_reported_as_agent_actions': int(reported_step),
      'optimizer_updates': int(optimizer_updates),
      'checkpoint_keyed_to_driver_counter': True,
      'checkpoint_keyed_to_exact_agent_actions': False,
      'status': 'FAIL_FOR_PAPER_SCHEDULE',
      'reason': 'Driver callback counter includes reset observations that execute no ALE action.',
      'minimal_change': (
          'Maintain a separate action counter incremented only when is_first is false; '
          'drive termination and exact 10k checkpoint triggers from that counter, and '
          'run evaluations as isolated checkpoint-based jobs.'),
  }


def write_audit(path, **kwargs):
  result = audit_smoke(**kwargs)
  path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
  return result
