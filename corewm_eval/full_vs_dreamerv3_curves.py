"""Plot completed Full CoRe-WM policy evals against W&B DreamerV3 curves."""

import argparse
import json
from pathlib import Path

from .config import ATARI100K_GAMES, TRAINING_SEEDS


PROTOCOL_ID = 'corewm_atari100k_v1'
PRODUCTION_COMMIT = '5ee4f27a0ba7fd7bdbdcbd4823fb2d539f8b4b47'
CHECKPOINTS = tuple(range(10_000, 100_001, 10_000))


def _binned(frame, bin_width=5_000):
  """Match the original DreamerV3/Harmony per-seed 5k binning."""
  import numpy as np

  work = frame.copy()
  work['bin'] = np.minimum((work.agent_steps // bin_width).astype(int),
                           100_000 // bin_width - 1)
  per_seed = work.groupby(
      ['method', 'game', 'seed', 'bin'], as_index=False)['return'].mean()
  aggregate = per_seed.groupby(
      ['method', 'game', 'bin'])['return'].agg(
          mean='mean', std='std', seeds='count').reset_index()
  aggregate['sem'] = aggregate['std'].fillna(0) / np.sqrt(aggregate['seeds'])
  aggregate['agent_steps'] = (aggregate['bin'] + 0.5) * bin_width
  return per_seed, aggregate


def _full_summary_paths(evaluations_root):
  root = Path(evaluations_root)
  paths = list((root / 'stage1_full' / 'full').glob(
      '*/seed_*/attempt_*/policy_summary.json'))
  alien_seed0 = root / 'wave1' / 'full' / 'alien' / 'seed_0' / 'policy_summary.json'
  if alien_seed0.exists():
    paths.append(alien_seed0)
  return sorted(paths)


def _validated_commit(payload, summary_path, rows):
  """Resolve commit without mutating legacy summaries that omitted the field."""
  commit = payload.get('git_commit')
  if commit is not None:
    return commit
  if payload.get('logger_fix_revision') != 1:
    raise AssertionError(f'Missing frozen provenance in {summary_path}')
  commits = set()
  for row in rows:
    run_meta = (summary_path.parent / f'{int(row["checkpoint"]):06d}' /
                'paper_artifacts' / 'run_meta.json')
    if not run_meta.exists():
      raise AssertionError(f'Missing checkpoint run metadata: {run_meta}')
    commits.add(json.loads(run_meta.read_text()).get('code_commit'))
  if len(commits) != 1:
    raise AssertionError(f'Inconsistent checkpoint commits in {summary_path}: {commits}')
  return commits.pop()


def load_full_policy_curves(evaluations_root):
  """Load and strictly validate the frozen 26-game x 5-seed Full matrix."""
  import pandas as pd

  expected = {(game, int(seed)) for game in ATARI100K_GAMES
              for seed in TRAINING_SEEDS}
  found = {}
  records = []
  sources = []
  for path in _full_summary_paths(evaluations_root):
    payload = json.loads(path.read_text())
    if payload.get('method') != 'Full':
      continue
    key = (payload['game'], int(payload.get('training_seed', payload.get('seed', -1))))
    # Current summaries store the scientific seed in each row.
    rows = payload.get('rows', [])
    if rows:
      row_seeds = {int(row['seed']) for row in rows}
      if len(row_seeds) != 1:
        raise AssertionError(f'Ambiguous training seed in {path}: {row_seeds}')
      key = (payload['game'], row_seeds.pop())
    if key not in expected:
      raise AssertionError(f'Unexpected Full evaluation summary {key}: {path}')
    if key in found:
      raise AssertionError(f'Duplicate Full evaluation summary {key}: {found[key]} and {path}')
    if payload.get('protocol_id') != PROTOCOL_ID:
      raise AssertionError(f'Protocol mismatch in {path}')
    if _validated_commit(payload, path, rows) != PRODUCTION_COMMIT:
      raise AssertionError(f'Commit mismatch in {path}')
    for flag in ('all_checkpoint_hashes_unchanged', 'all_episode_counts_exact',
                 'all_returns_finite'):
      if payload.get(flag) is not True:
        raise AssertionError(f'{flag} is not true in {path}')
    checkpoints = tuple(int(row['checkpoint']) for row in rows)
    if checkpoints != CHECKPOINTS:
      raise AssertionError(f'Checkpoint schedule mismatch in {path}: {checkpoints}')
    episodes = tuple(int(row['episodes']) for row in rows)
    if episodes != (10,) * 9 + (100,):
      raise AssertionError(f'Episode schedule mismatch in {path}: {episodes}')
    found[key] = str(path.resolve())
    sources.append(str(path.resolve()))
    for row in rows:
      records.append({
          'method': 'Full CoRe-WM',
          'game': key[0],
          'seed': key[1],
          'agent_steps': int(row['checkpoint']),
          'return': float(row['raw_return']),
          'episodes': int(row['episodes']),
          'checkpoint_hash': row['checkpoint_hash'],
      })
  missing = sorted(expected - set(found))
  if missing:
    raise AssertionError(f'Missing Full policy curves ({len(missing)}): {missing}')
  if len(found) != 130 or len(records) != 1300:
    raise AssertionError((len(found), len(records)))
  return pd.DataFrame(records), sources


def load_dreamerv3_wandb_curves(data_path):
  """Load the already-downloaded W&B DreamerV3 episode-return histories."""
  import pandas as pd

  path = Path(data_path)
  frame = pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path)
  frame = frame[frame.method == 'DreamerV3'].copy()
  expected = {(game, int(seed)) for game in ATARI100K_GAMES
              for seed in TRAINING_SEEDS}
  found = set(zip(frame.game, frame.seed.astype(int)))
  if found != expected:
    raise AssertionError(
        f'DreamerV3 W&B coverage mismatch; missing={sorted(expected-found)}, '
        f'unexpected={sorted(found-expected)}')
  return frame


def aggregate_curves(full, dreamer):
  """Aggregate per-seed means before cross-seed mean and SEM."""
  import numpy as np
  import pandas as pd

  dreamer_per_seed, dreamer_aggregate = _binned(dreamer, bin_width=5_000)
  dreamer_aggregate = dreamer_aggregate.copy()
  dreamer_per_seed = dreamer_per_seed.copy()
  full_per_seed = full[['method', 'game', 'seed', 'agent_steps', 'return']].copy()
  full_aggregate = full_per_seed.groupby(
      ['method', 'game', 'agent_steps'])['return'].agg(
          mean='mean', std='std', seeds='count').reset_index()
  full_aggregate['sem'] = full_aggregate['std'].fillna(0) / np.sqrt(
      full_aggregate['seeds'])
  aggregate = pd.concat(
      [dreamer_aggregate, full_aggregate], ignore_index=True, sort=False)
  per_seed = pd.concat(
      [dreamer_per_seed, full_per_seed], ignore_index=True, sort=False)
  return per_seed, aggregate


def plot_curves(full, dreamer, output_dir, sources, dreamer_source):
  import os
  os.environ.setdefault('MPLCONFIGDIR', '/tmp/corewm-matplotlib')
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  from matplotlib.ticker import MultipleLocator
  import pandas as pd

  methods = ('DreamerV3', 'Full CoRe-WM')
  colors = {'DreamerV3': '#3B82F6', 'Full CoRe-WM': '#F59E0B'}
  per_seed, aggregate = aggregate_curves(full, dreamer)
  output_dir = Path(output_dir)
  png_dir, pdf_dir = output_dir / 'png', output_dir / 'pdf'
  png_dir.mkdir(parents=True, exist_ok=True)
  pdf_dir.mkdir(parents=True, exist_ok=True)

  def draw(ax, game, overview=False):
    for method in methods:
      values = aggregate[(aggregate.game == game) & (aggregate.method == method)]
      if values.empty:
        raise AssertionError(f'No aggregate values for {(method, game)}')
      values = values.sort_values('agent_steps')
      x = values.agent_steps.to_numpy()
      mean = values['mean'].to_numpy()
      sem = values['sem'].to_numpy()
      label = method if overview else f'{method} (n=5 seeds)'
      ax.plot(x, mean, color=colors[method], lw=1.5 if overview else 2.2,
              label=label)
      ax.fill_between(x, mean - sem, mean + sem, color=colors[method],
                      alpha=.16 if overview else .18, linewidth=0)
    ax.set_xlim(0, 100_000)
    ax.set_xticks([0, 25_000, 50_000, 75_000, 100_000])
    ax.set_xticklabels(['0', '25k', '50k', '75k', '100k'],
                       fontsize=7 if overview else None)
    ax.xaxis.set_minor_locator(MultipleLocator(5_000))
    ax.grid(which='major', alpha=.22 if overview else .28)
    ax.grid(which='minor', axis='x', alpha=.06 if overview else .09)
    ax.set_title(game.replace('_', ' ').title(), fontsize=10 if overview else None)

  for game in ATARI100K_GAMES:
    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    draw(ax, game)
    ax.set_xlabel('Environment interactions (agent actions)')
    ax.set_ylabel('Episode return')
    ax.legend(frameon=False, loc='best')
    fig.savefig(png_dir / f'{game}.png', dpi=180)
    fig.savefig(pdf_dir / f'{game}.pdf')
    plt.close(fig)

  fig, axes = plt.subplots(7, 4, figsize=(16, 22), constrained_layout=True)
  axes = axes.ravel()
  for ax, game in zip(axes, ATARI100K_GAMES):
    draw(ax, game, overview=True)
  handles = [plt.Line2D([0], [0], color=colors[m], lw=2, label=m)
             for m in methods]
  legend_ax = axes[len(ATARI100K_GAMES)]
  legend_ax.axis('off')
  legend_ax.legend(handles=handles, loc='center', frameon=False, fontsize=11)
  note_ax = axes[len(ATARI100K_GAMES) + 1]
  note_ax.axis('off')
  note_ax.text(
      .5, .5,
      'Shading: mean ± 1 SEM across 5 training seeds.\n'
      'Full: isolated checkpoint eval; DreamerV3: W&B training episodes.',
      ha='center', va='center', fontsize=9)
  for ax in axes[len(ATARI100K_GAMES) + 2:]:
    ax.axis('off')
  fig.supxlabel('Environment interactions (agent actions)')
  fig.supylabel('Episode return')
  fig.savefig(output_dir / 'all_games_overview.png', dpi=180)
  fig.savefig(output_dir / 'all_games_overview.pdf')
  plt.close(fig)

  per_seed.to_csv(output_dir / 'per_seed.csv', index=False)
  aggregate.to_csv(output_dir / 'aggregate.csv', index=False)
  full.to_csv(output_dir / 'full_policy_eval_rows.csv', index=False)
  metadata = {
      'protocol': PROTOCOL_ID,
      'full_git_commit': PRODUCTION_COMMIT,
      'games': list(ATARI100K_GAMES),
      'training_seeds': list(map(int, TRAINING_SEEDS)),
      'methods': list(methods),
      'full_source': {
          'kind': 'isolated policy evaluation',
          'checkpoints': list(CHECKPOINTS),
          'episodes_per_checkpoint': [10] * 9 + [100],
          'summary_count': len(sources),
          'summaries': sources,
      },
      'dreamerv3_source': {
          'kind': 'W&B training episode returns',
          'path': str(Path(dreamer_source).resolve()),
          'x_axis': 'agent_actions_est',
          'bin_width_actions': 5_000,
      },
      'aggregation': {
          'Full CoRe-WM': 'checkpoint return mean per seed; mean +/- 1 SEM across seeds',
          'DreamerV3': 'episode returns averaged within 5k bins per seed; mean +/- 1 SEM across seeds',
      },
      'comparability_note': (
          'The two curves use different evaluation sources: isolated checkpoint policy '
          'evaluation for Full CoRe-WM versus W&B training episode returns for DreamerV3.'),
      'axes': {'x_min': 0, 'x_max': 100_000, 'major_tick_actions': 25_000,
               'minor_tick_actions': 5_000, 'y': 'raw episode return'},
      'files': {'individual_png': 26, 'individual_pdf': 26,
                'overview_png': 'all_games_overview.png',
                'overview_pdf': 'all_games_overview.pdf'},
  }
  (output_dir / 'plot_metadata.json').write_text(
      json.dumps(metadata, indent=2) + '\n')
  return metadata


def main(argv=None):
  parser = argparse.ArgumentParser()
  parser.add_argument('--evaluations-root', required=True)
  parser.add_argument('--dreamerv3-data', required=True)
  parser.add_argument('--output-dir', required=True)
  args = parser.parse_args(argv)
  full, sources = load_full_policy_curves(args.evaluations_root)
  dreamer = load_dreamerv3_wandb_curves(args.dreamerv3_data)
  result = plot_curves(full, dreamer, args.output_dir, sources,
                       args.dreamerv3_data)
  print(json.dumps(result, indent=2))


if __name__ == '__main__':
  main()
