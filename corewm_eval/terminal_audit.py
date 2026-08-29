"""Audit incoming Atari terminal flags before the legacy _obs overwrite."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from embodied.envs.atari import Atari

from .config import QUALITATIVE_GAMES
from .phase3_prepare import resolve


def _category(info):
  causes = [name for name in (
      'game_over', 'time_limit', 'life_loss_observed',
      'life_loss_terminal', 'life_loss_last') if info[name]]
  return '+'.join(causes) if causes else 'no_recorded_cause'


def audit_game(game, seed=0, endings=3, max_action_steps=30_000):
  config = resolve('Backbone')
  kwargs = dict(config.env.atari100k)
  configured_use_seed = bool(kwargs.pop('use_seed'))
  kwargs['seed'] = int(seed)
  env = Atari(game, **kwargs)
  rng = np.random.default_rng(np.random.SeedSequence([seed, QUALITATIVE_GAMES.index(game)]))
  action_steps = 0
  episodes = 0
  counts = dict(
      num_is_last=0, num_true_terminal=0,
      num_is_last_not_terminal=0, num_terminal_not_last=0,
      num_game_over=0, num_time_limit=0, num_life_loss_observed=0)
  mismatches = []
  try:
    env.step({'action': np.int32(0), 'reset': True})
    while action_steps < max_action_steps and episodes < endings:
      action = np.int32(rng.integers(len(env.actionset)))
      obs = env.step({'action': action, 'reset': False})
      action_steps += 1
      info = dict(env.last_transition_audit)
      last = bool(info['incoming_is_last'])
      terminal = bool(info['incoming_is_terminal'])
      counts['num_is_last'] += int(last)
      counts['num_true_terminal'] += int(terminal)
      counts['num_is_last_not_terminal'] += int(last and not terminal)
      counts['num_terminal_not_last'] += int(terminal and not last)
      counts['num_game_over'] += int(info['game_over'])
      counts['num_time_limit'] += int(info['time_limit'])
      counts['num_life_loss_observed'] += int(info['life_loss_observed'])
      if last != terminal:
        mismatches.append({
            'game': game, 'seed': int(seed), 'action_step': action_steps,
            'episode_index': episodes, 'episode_ale_frames': int(env.duration),
            'incoming_is_last': last, 'incoming_is_terminal': terminal,
            'returned_is_terminal_after_legacy_overwrite': bool(obs['is_terminal']),
            'reason_category': _category(info), **info,
        })
      if last:
        episodes += 1
        if episodes < endings and action_steps < max_action_steps:
          env.step({'action': np.int32(0), 'reset': True})
  finally:
    env.close()
  row = {
      'game': game, 'seed': int(seed), 'action_steps': action_steps,
      'episodes_ended': episodes, **counts,
      'mismatch_categories': sorted({x['reason_category'] for x in mismatches}),
      'paper_lives_setting': kwargs['lives'],
      'configured_episode_ale_frame_limit': int(kwargs.get('length', 108_000)),
      'action_repeat': int(kwargs['repeat']),
      'paper_config_use_seed': configured_use_seed,
      'audit_seed_override': int(seed),
      'sufficient_endings_reached': episodes >= endings,
  }
  return row, mismatches


def run(output, seed=0, endings=3, max_action_steps=30_000):
  output = Path(output)
  output.mkdir(parents=True, exist_ok=True)
  rows, mismatches = [], []
  for game in QUALITATIVE_GAMES:
    row, events = audit_game(game, seed, endings, max_action_steps)
    rows.append(row)
    mismatches.extend(events)
    print(json.dumps(row, sort_keys=True))
  with (output / 'terminal_audit.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
  with (output / 'terminal_mismatches.jsonl').open('w') as stream:
    for event in mismatches:
      stream.write(json.dumps(event, sort_keys=True) + '\n')
  summary = {
      'games': list(QUALITATIVE_GAMES), 'seed': int(seed),
      'requested_endings_per_game': int(endings),
      'max_action_steps_per_game': int(max_action_steps),
      'total_action_steps': sum(x['action_steps'] for x in rows),
      'total_is_last': sum(x['num_is_last'] for x in rows),
      'total_true_terminal': sum(x['num_true_terminal'] for x in rows),
      'total_is_last_not_terminal': sum(
          x['num_is_last_not_terminal'] for x in rows),
      'total_terminal_not_last': sum(
          x['num_terminal_not_last'] for x in rows),
      'all_requested_endings_reached': all(
          x['sufficient_endings_reached'] for x in rows),
      'mismatch_count': len(mismatches),
      'decision': 'INERT' if not mismatches else 'STOP_FOR_DECISION',
  }
  (output / 'terminal_audit_summary.json').write_text(
      json.dumps(summary, indent=2, sort_keys=True) + '\n')
  return rows, mismatches, summary


def main(argv=None):
  parser = argparse.ArgumentParser()
  parser.add_argument('--output', default='paper_artifacts/pretraining_audit')
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--endings', type=int, default=3)
  parser.add_argument('--max-action-steps', type=int, default=30_000)
  args = parser.parse_args(argv)
  run(args.output, args.seed, args.endings, args.max_action_steps)


if __name__ == '__main__':
  main()
