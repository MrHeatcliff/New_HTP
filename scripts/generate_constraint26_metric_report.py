#!/usr/bin/env python3
"""Render the completed 26-game constraint run into an auditable Markdown report."""
import json
import math
import statistics
import sys
from pathlib import Path


ROOT = Path('/home/vn-user0101/Dat/HTS-Dreamer/production_runs/constraint_full26_seed0_OzirkNB8')
OUTPUT = Path('paper_artifacts/constraint26_metrics.md')


def rows(path):
  return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def mean_or_none(values):
  values = [float(x) for x in values if x not in (None, '') and math.isfinite(float(x))]
  return statistics.fmean(values) if values else None


def fmt(value, digits=4):
  return '—' if value is None else f'{value:.{digits}f}'


def cell(value, digits=2):
  return '—' if value is None else f'{value:.{digits}f}'


def table(headers, values):
  lines = ['| ' + ' | '.join(headers) + ' |',
           '| ' + ' | '.join(['---'] * len(headers)) + ' |']
  lines += ['| ' + ' | '.join(map(str, row)) + ' |' for row in values]
  return '\n'.join(lines)


def main():
  status = json.loads((ROOT / 'status.json').read_text())
  if status.get('status') != 'COMPLETE' or len(status.get('results', {})) != 26:
    raise RuntimeError('Expected a completed 26-game source run')
  records = []
  for game in sorted(status['results']):
    run = ROOT / 'runs' / game
    episodes = rows(run / 'paper_artifacts/episode_scores.jsonl')
    train = rows(run / 'paper_artifacts/train_metrics.jsonl')
    final = max((x for x in train if int(x.get('agent_actions', -1)) <= 110000),
                key=lambda x: int(x['agent_actions']))
    late = [x['episode_score'] for x in episodes if 90000 <= int(x['agent_actions']) <= 100000]
    terminal = [x['episode_score'] for x in episodes if 100000 < int(x['agent_actions']) <= 110000]
    result = status['results'][game]
    records.append({'game': game, 'late_score': mean_or_none(late), 'late_n': len(late),
        'final_score': mean_or_none(terminal), 'final_n': len(terminal),
        'updates': int(final['optimizer_updates']), 'minutes': float(result['seconds']) / 60,
        'pdyn': final.get('train/htp/pdyn_prediction_only'),
        'persistence': final.get('train/htp/prefix_persistence'),
        'isotropy': final.get('train/htp/prefix_isotropy'),
        'rec': final.get('train/htp/raw_rec_loss'),
        'd1': final.get('train/htp_pdyn_delta1'), 'd2': final.get('train/htp_pdyn_delta2'),
        'd4': final.get('train/htp_pdyn_delta4'), 'd8': final.get('train/htp_pdyn_delta8'),
        'd16': final.get('train/htp_pdyn_delta16'),
        'cross16': final.get('train/htp/cross_episode_delta16'),
        'r0': final.get('train/htp_rec_l0'), 'r4': final.get('train/htp_rec_l4')})
  macro = {key: mean_or_none([row[key] for row in records]) for key in records[0] if key not in ('game', 'late_n', 'final_n')}
  control = [[r['game'], cell(r['late_score']), r['late_n'], cell(r['final_score']), r['final_n'],
              f"{r['updates']:,}", cell(r['minutes'], 1)] for r in records]
  dynamics = [[r['game'], fmt(r['pdyn']), fmt(r['persistence']), fmt(r['isotropy']), fmt(r['rec']),
               fmt(r['d1']), fmt(r['d2']), fmt(r['d4']), fmt(r['d8']), fmt(r['d16']),
               fmt(r['cross16']), fmt(r['r0']), fmt(r['r4'])] for r in records]
  control.append(['Macro mean', cell(macro['late_score']), '—', cell(macro['final_score']), '—',
                  f"{round(macro['updates']):,}", cell(macro['minutes'], 1)])
  dynamics.append(['Macro mean', *[fmt(macro[key]) for key in ('pdyn', 'persistence', 'isotropy', 'rec', 'd1', 'd2', 'd4', 'd8', 'd16', 'cross16', 'r0', 'r4')]])
  text = f'''# Constraint Suite: 26-Game Metric Report

## Scope and provenance

This report is rendered directly from the completed run at `{ROOT}`. The run contains 26 Atari-100k games, seed 0, 110,000 exact environment actions per game, with progressive reconstruction and multi-stride prefix dynamics enabled. The first-prefix constraint package is whitened persistence (weight 0.1, all lags) plus isotropy (weight 0.01). It does **not** establish a comparison against an unconstrained control: results here describe this one condition only.

Every game finished successfully. The macro rows are unweighted means over games; they are not normalized Atari scores and should not be used to rank algorithms. `final_eval.json` records `status: not_run`, so the control values below are training episode returns, not isolated evaluation returns.

## What each metric means

| Family | Logged metric | Interpretation | Caveat |
| --- | --- | --- | --- |
| Control | `episode_score` | Environment return from a completed training episode. | Policy is still training; episodes within one seed are not independent training replicates. |
| Multi-stride prediction | `train/htp/pdyn_prediction_only`, `train/htp_pdyn_delta{{1,2,4,8,16}}` | Total prediction-only loss and its per-stride components. Lower is only an optimization diagnostic, not evidence of semantic stability. | Targets are learned slow features. |
| Temporal constraint | `train/htp/prefix_persistence` | Whitened first-prefix temporal displacement, averaged across lags 1--16, before its 0.1 loss weight. | Lower can also discard dynamic information. |
| Anti-collapse | `train/htp/prefix_isotropy` | Deviation of first-prefix covariance from isotropy. | It is not effective rank or a semantic diversity measurement. |
| Reconstruction | `train/htp/raw_rec_loss`, `train/htp_rec_l0`, `train/htp_rec_l4` | Progressive reconstruction loss and the first/final prefix-level errors. | Reconstruction fidelity does not prove long-horizon predictiveness. |
| Boundary audit | `train/htp/cross_episode_delta16` | Fraction of candidate stride-16 pairs crossing episode resets. | In this run boundary masking was disabled; this reports exposure, not a correction. |

The following diagnostics were requested but were **not collected for all 26 checkpoints**: fixed-trajectory future-target ridge $R^2$, per-block effective rank, dead-coordinate count, whitened change per block, block-1/block-2 CKA, and isolated evaluation return with confidence intervals. They exist only for later four-game diagnostic jobs, and are deliberately not imputed here.

## Control and training completion

`90--100k` and `100--110k` are means over complete training episodes whose exact `agent_actions` fall in each interval. `n` is the number of episodes, so a dash means no episode completed in that window.

{table(['Game', '90--100k score', 'n', '100--110k score', 'n', 'updates', 'wall min'], control)}

## Final logged HTP metrics by game

Each value is the final logged aggregate at or before 110,000 environment actions. `cross16` is a fraction; all remaining columns are losses or diagnostics on their native scale.

{table(['Game', 'pdyn', 'persist', 'isotropy', 'raw rec', 'd1', 'd2', 'd4', 'd8', 'd16', 'cross16', 'rec l0', 'rec l4'], dynamics)}

## Interpretation boundaries

- The suite confirms that both reconstruction and multi-stride branches executed and logged throughout all 26 games, but it does not identify which component caused a control change.
- A lower prediction loss or persistence score alone cannot show that a compact prefix carries invariant semantic content; it may reflect scale, smoothing, or information removal.
- The appropriate evidence for that causal claim remains the paired ablations and fixed-trajectory representation diagnostics described above, ideally repeated across independent training seeds.
'''
  OUTPUT.parent.mkdir(parents=True, exist_ok=True)
  OUTPUT.write_text(text)
  print(f'Wrote {OUTPUT} from {len(records)} completed games')


if __name__ == '__main__':
  main()
