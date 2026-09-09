"""Add the completed seed-0 constraint suite to the published curve layout."""
import argparse
import hashlib
import json
from pathlib import Path

from .config import ATARI100K_GAMES
from .full_vs_dreamerv3_curves import _binned


SUITE_LABEL = 'Constraint suite (seed 0; training episodes)'


def load_suite(root):
  import pandas as pd
  root = Path(root)
  manifest = json.loads((root / 'manifest.json').read_text())
  status = json.loads((root / 'status.json').read_text())
  # The manifest is immutable submission provenance and deliberately retains
  # its initial RUNNING state; status.json is the terminal state record.
  if status['status'] != 'COMPLETE':
    raise AssertionError('Suite did not complete')
  if tuple(manifest['games']) != ATARI100K_GAMES or len(status['results']) != 26:
    raise AssertionError('Game coverage mismatch')
  rows = []
  for game in ATARI100K_GAMES:
    result = status['results'][game]
    if not result['complete'] or result['exit_code'] != 0:
      raise AssertionError((game, result))
    # `scores.jsonl` indexes driver callbacks and includes reset callbacks.
    # The paper-artifact stream records `agent_actions`, the requested x-axis.
    path = root / 'runs' / game / 'paper_artifacts' / 'episode_scores.jsonl'
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if not records or any('agent_actions' not in row or 'episode_score' not in row for row in records):
      raise AssertionError(f'Malformed scores: {path}')
    for row in records:
      step, score = int(row['agent_actions']), float(row['episode_score'])
      if not (0 < step <= 110_000) or not __import__('math').isfinite(score):
        raise AssertionError((game, step, score))
      rows.append({'method': SUITE_LABEL, 'game': game, 'seed': 0,
                   'agent_steps': step, 'return': score})
  return pd.DataFrame(rows), manifest, status


def load_existing(path):
  import pandas as pd
  root = Path(path)
  aggregate = pd.read_csv(root / 'aggregate.csv')
  full = aggregate[aggregate.method == 'Full CoRe-WM'].copy()
  dreamer = aggregate[aggregate.method == 'DreamerV3'].copy()
  expected = set(ATARI100K_GAMES)
  if set(full.game) != expected or set(dreamer.game) != expected:
    raise AssertionError('Existing figure game coverage mismatch')
  return full, dreamer


def aggregate(frame, is_training):
  import numpy as np
  if is_training:
    per_seed, output = _binned(frame, bin_width=5_000)
  else:
    per_seed = frame.copy()
    output = per_seed.groupby(['method','game','agent_steps'])['return'].agg(
        mean='mean',std='std',seeds='count').reset_index()
    output['sem'] = output['std'].fillna(0) / np.sqrt(output['seeds'])
  return per_seed, output


def render(full, dreamer, suite, output):
  import os
  os.environ.setdefault('MPLCONFIGDIR', '/tmp/corewm-matplotlib')
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  from matplotlib.ticker import MultipleLocator
  import pandas as pd
  _, suite_aggregate = aggregate(suite, True)
  curves = [
      ('DreamerV3', dreamer, '#3B82F6', 'n=5 seeds; training episodes'),
      ('Full CoRe-WM', full, '#F59E0B', 'n=5 seeds; isolated eval'),
      (SUITE_LABEL, suite_aggregate, '#7C3AED', 'n=1; training episodes'),
  ]
  aggregate_csv = pd.concat([frame.assign(source_note=note) for _,frame,_,note in curves], ignore_index=True, sort=False)
  output = Path(output); png=output/'png'; pdf=output/'pdf'
  png.mkdir(parents=True,exist_ok=False); pdf.mkdir()
  def draw(ax, game, overview=False):
    for label,_,color,_ in curves:
      values=aggregate_csv[(aggregate_csv.game == game)&(aggregate_csv.method == label)].sort_values('agent_steps')
      ax.plot(values.agent_steps,values['mean'],color=color,lw=1.5 if overview else 2.2,label=label)
      # n=1 has zero SEM; plot no pseudo-uncertainty band.
      if values.seeds.iloc[0] > 1:
        ax.fill_between(values.agent_steps,values['mean']-values['sem'],values['mean']+values['sem'],color=color,alpha=.16,linewidth=0)
    ax.set_xlim(0,100_000); ax.set_xticks([0,25000,50000,75000,100000])
    ax.set_xticklabels(['0','25k','50k','75k','100k'],fontsize=7 if overview else None)
    ax.xaxis.set_minor_locator(MultipleLocator(5_000)); ax.grid(which='major',alpha=.25); ax.grid(which='minor',axis='x',alpha=.08)
    ax.set_title(game.replace('_',' ').title(),fontsize=10 if overview else None)
  for game in ATARI100K_GAMES:
    fig,ax=plt.subplots(figsize=(7.2,4.5),constrained_layout=True); draw(ax,game)
    ax.set_xlabel('Environment interactions (agent actions)'); ax.set_ylabel('Episode return'); ax.legend(frameon=False,loc='best',fontsize=8)
    fig.savefig(png/f'{game}.png',dpi=180); fig.savefig(pdf/f'{game}.pdf'); plt.close(fig)
  fig,axes=plt.subplots(7,4,figsize=(16,22),constrained_layout=True); axes=axes.ravel()
  for ax,game in zip(axes,ATARI100K_GAMES): draw(ax,game,overview=True)
  handles=[plt.Line2D([0],[0],color=color,lw=2,label=label) for label,_,color,_ in curves]
  axes[26].axis('off'); axes[26].legend(handles=handles,loc='center',frameon=False,fontsize=10)
  axes[27].axis('off'); axes[27].text(.5,.5,'Shading: mean ± 1 SEM where n=5.\nPurple: new seed-0 training episodes (no shading).',ha='center',va='center',fontsize=9)
  for ax in axes[28:]: ax.axis('off')
  fig.supxlabel('Environment interactions (agent actions)'); fig.supylabel('Episode return')
  fig.savefig(output/'all_games_overview.png',dpi=180); fig.savefig(output/'all_games_overview.pdf'); plt.close(fig)
  aggregate_csv.to_csv(output/'aggregate.csv',index=False)
  suite.to_csv(output/'constraint_suite_episode_rows.csv',index=False)


def main(argv=None):
  parser=argparse.ArgumentParser(); parser.add_argument('--suite-root',required=True); parser.add_argument('--existing-figures',required=True); parser.add_argument('--output-dir',required=True)
  args=parser.parse_args(argv); output=Path(args.output_dir)
  if output.exists(): raise FileExistsError(output)
  full,dreamer=load_existing(args.existing_figures); suite,manifest,status=load_suite(args.suite_root)
  render(full,dreamer,suite,output)
  metadata={'methods':['DreamerV3','Full CoRe-WM',SUITE_LABEL],'suite_root':str(Path(args.suite_root).resolve()),
      'suite_manifest_sha256':hashlib.sha256((Path(args.suite_root)/'manifest.json').read_bytes()).hexdigest(),
      'suite_status_sha256':hashlib.sha256((Path(args.suite_root)/'status.json').read_bytes()).hexdigest(),
      'suite_actions':manifest['actions_per_game'],'suite_seed':manifest['seed'],
      'suite_method_config':'size25m; whitened persistence 0.1; isotropy 0.01; target window 1; reset mask off',
      'comparability_note':'Purple is a single seed and training-episode returns. It is not a 5-seed estimate or isolated checkpoint evaluation; do not compare uncertainty bands or claim significance from this plot.',
      'source_existing_figures':str(Path(args.existing_figures).resolve()),'games':list(ATARI100K_GAMES)}
  (output/'plot_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
  print(json.dumps(metadata,indent=2))


if __name__=='__main__': main()
