"""Download and plot external DreamerV3/HarmonyDream Atari learning curves."""

import argparse
import json
import re
from pathlib import Path

from .config import ATARI100K_GAMES, TRAINING_SEEDS, wandb_project_for_game, WANDB_ENTITY


METHOD_TAGS = {
    'DreamerV3': 'official_dreamerv3',
    'Harmony Dreamer': 'harmonydream',
}
GAME_TAG_ALIASES = {'jamesbond': 'james_bond'}


def _api():
  import wandb
  return wandb.Api(timeout=90)


def audit_runs():
  api = _api(); selected = {}; inventory = {}
  for game in ATARI100K_GAMES:
    project = wandb_project_for_game(game)
    runs = list(api.runs(f'{WANDB_ENTITY}/{project}', per_page=200))
    inventory[game] = []
    for run in runs:
      tags = set(run.tags)
      method = next((name for name, tag in METHOD_TAGS.items() if tag in tags), None)
      if method is None:
        continue
      seed_tags = [tag for tag in tags if tag.startswith('seed') and tag[4:].isdigit()]
      configured_seed = dict(run.config).get('seed')
      named_seed = re.search(r'(?:^|[-_])seed[_-]?(\d+)(?:$|[-_])', run.name.lower())
      if len(seed_tags) == 1:
        seed = int(seed_tags[0][4:])
      elif configured_seed is not None and int(configured_seed) in TRAINING_SEEDS:
        seed = int(configured_seed)
      elif named_seed and int(named_seed.group(1)) in TRAINING_SEEDS:
        seed = int(named_seed.group(1))
      else:
        inventory[game].append({'id': run.id, 'name': run.name,
                                'problem': 'missing_or_ambiguous_seed_tag'})
        continue
      row = {'method': method, 'game': game, 'seed': seed, 'project': project,
             'id': run.id, 'name': run.name, 'state': run.state,
             'tags': sorted(tags), 'url': run.url}
      inventory[game].append(row)
      key = (method, game, seed)
      if key in selected:
        raise RuntimeError(f'Duplicate candidate runs for {key}: {selected[key]} and {row}')
      selected[key] = row
  expected = {(method, game, seed) for method in METHOD_TAGS
              for game in ATARI100K_GAMES for seed in TRAINING_SEEDS}
  missing = sorted(expected - set(selected))
  nonfinished = [row for row in selected.values() if row['state'] != 'finished']
  return {'entity': WANDB_ENTITY, 'methods': list(METHOD_TAGS),
          'selected_count': len(selected), 'expected_count': len(expected),
          'missing': [{'method': x[0], 'game': x[1], 'seed': x[2]} for x in missing],
          'nonfinished': nonfinished, 'selected': list(selected.values()),
          'inventory': inventory}


def inspect_run(project, run_id):
  run = _api().run(f'{WANDB_ENTITY}/{project}/{run_id}')
  keys = (
      'episode/score', 'train/agent_step', 'train/env_frame',
      'train/episode_return', 'Episode-Rewards/rank_0/env-0',
      'Episode-Steps/rank_0/env-0')
  result = {'project': project, 'id': run.id, 'name': run.name,
            'state': run.state, 'tags': list(run.tags),
            'config': dict(run.config), 'summary_keys': sorted(dict(run.summary))}
  histories = {}
  for key in keys:
    rows = list(run.scan_history(keys=[key], page_size=10_000))
    histories[key] = {'count': len(rows), 'first': rows[:3], 'last': rows[-3:]}
  result['histories'] = histories
  return result


def download_one(audit_path, output_dir, index, refresh=False):
  audit=json.loads(Path(audit_path).read_text());meta=audit['selected'][int(index)]
  if meta['state']!='finished':
    return {'index':int(index),'status':'SKIPPED_NONFINISHED','run':meta}
  slug='dreamerv3' if meta['method']=='DreamerV3' else 'harmony_dreamer'
  target=Path(output_dir)/'runs'/slug/meta['game']/f'seed_{meta["seed"]}.json'
  if target.exists() and not refresh:
    payload=json.loads(target.read_text())
    return {'index':int(index),'status':'CACHED','target':str(target),'points':len(payload['points'])}
  run=_api().run(f'{WANDB_ENTITY}/{meta["project"]}/{meta["id"]}')
  if meta['method']=='DreamerV3':
    history=list(run.scan_history(keys=['episode/score','agent_actions_est'],page_size=10_000))
    points=[{'agent_steps':float(row['agent_actions_est']),'return':float(row['episode/score'])}
            for row in history if row.get('episode/score') is not None and row.get('agent_actions_est') is not None]
    x_source='agent_actions_est'
  else:
    repeat=int(dict(run.config).get('env',{}).get('atari',{}).get('repeat',0))
    if repeat!=4:raise AssertionError((meta['id'],'Harmony action repeat',repeat))
    history=list(run.scan_history(keys=['episode/score'],page_size=10_000))
    points=[{'agent_steps':float(row['_step'])/repeat,'return':float(row['episode/score'])}
            for row in history if row.get('episode/score') is not None]
    x_source='_step / configured action_repeat=4'
  points=sorted((row for row in points if 0<=row['agent_steps']<=100_000),key=lambda row:row['agent_steps'])
  if not points:raise AssertionError(f'No valid curve points: {meta}')
  payload={'source':meta,'x_source':x_source,'x_cap_agent_steps':100_000,'points':points}
  target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(payload,indent=2)+'\n')
  return {'index':int(index),'status':'COMPLETE','target':str(target),'points':len(points)}


def download_histories(audit_path, output_dir, refresh=False, workers=8):
  from concurrent.futures import ThreadPoolExecutor, as_completed
  audit = json.loads(Path(audit_path).read_text())
  output_dir = Path(output_dir); records = []; run_rows = []

  def fetch(meta):
    slug = 'dreamerv3' if meta['method'] == 'DreamerV3' else 'harmony_dreamer'
    target = output_dir / 'runs' / slug / meta['game'] / f'seed_{meta["seed"]}.json'
    if target.exists() and not refresh:
      return meta, json.loads(target.read_text())
    run = _api().run(f'{WANDB_ENTITY}/{meta["project"]}/{meta["id"]}')
    if meta['method'] == 'DreamerV3':
      history = list(run.scan_history(
          keys=['episode/score', 'agent_actions_est'], page_size=10_000))
      points = [{'agent_steps': float(row['agent_actions_est']),
                 'return': float(row['episode/score'])}
                for row in history if row.get('episode/score') is not None and
                row.get('agent_actions_est') is not None]
      x_source = 'agent_actions_est'
    else:
      config = dict(run.config)
      repeat = int(config.get('env', {}).get('atari', {}).get('repeat', 0))
      if repeat != 4:
        raise AssertionError((meta['id'], 'Harmony action repeat', repeat))
      history = list(run.scan_history(keys=['episode/score'], page_size=10_000))
      points = [{'agent_steps': float(row['_step']) / repeat,
                 'return': float(row['episode/score'])}
                for row in history if row.get('episode/score') is not None]
      x_source = '_step / configured action_repeat=4'
    points = [row for row in points if 0 <= row['agent_steps'] <= 100_000]
    points.sort(key=lambda row: row['agent_steps'])
    if not points:
      raise AssertionError(f'No valid curve points: {meta}')
    payload = {'source': meta, 'x_source': x_source,
               'x_cap_agent_steps': 100_000, 'points': points}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + '\n')
    return meta, payload

  finished=[]
  for meta in audit['selected']:
    if meta['state'] != 'finished':
      run_rows.append({**meta, 'download_status': 'SKIPPED_NONFINISHED'})
      continue
    finished.append(meta)
  results=[]
  with ThreadPoolExecutor(max_workers=int(workers)) as pool:
    futures={pool.submit(fetch,meta):meta for meta in finished}
    for future in as_completed(futures):
      results.append(future.result())
  results.sort(key=lambda item:(item[0]['game'],item[0]['method'],item[0]['seed']))
  for meta,payload in results:
    run_rows.append({**meta, 'download_status': 'COMPLETE',
                     'x_source': payload['x_source'],
                     'point_count': len(payload['points']),
                     'min_agent_steps': min(row['agent_steps'] for row in payload['points']),
                     'max_agent_steps': max(row['agent_steps'] for row in payload['points'])})
    for point in payload['points']:
      records.append({'method': meta['method'], 'game': meta['game'],
                      'seed': meta['seed'], 'wandb_project': meta['project'],
                      'wandb_run_id': meta['id'], **point})
  import pandas as pd
  output_dir.mkdir(parents=True, exist_ok=True)
  frame = pd.DataFrame(records).sort_values(['game','method','seed','agent_steps'])
  frame.to_parquet(output_dir / 'raw_episode_returns.parquet', index=False)
  frame.to_csv(output_dir / 'raw_episode_returns.csv', index=False)
  manifest = {'audit': str(Path(audit_path).resolve()), 'records': len(frame),
              'completed_runs': sum(row['download_status']=='COMPLETE' for row in run_rows),
              'skipped_nonfinished_runs': sum(row['download_status']!='COMPLETE' for row in run_rows),
              'runs': run_rows}
  (output_dir / 'download_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
  return manifest


def _binned(frame, bin_width=5_000):
  import numpy as np
  import pandas as pd
  work = frame.copy()
  work['bin'] = np.minimum((work.agent_steps // bin_width).astype(int),
                           100_000 // bin_width - 1)
  per_seed = work.groupby(['method','game','seed','bin'],as_index=False)['return'].mean()
  aggregate = per_seed.groupby(['method','game','bin'])['return'].agg(
      mean='mean', std='std', seeds='count').reset_index()
  aggregate['sem'] = aggregate['std'].fillna(0) / np.sqrt(aggregate['seeds'])
  aggregate['agent_steps'] = (aggregate['bin'] + 0.5) * bin_width
  return per_seed, aggregate


def plot_curves(data_path, audit_path, output_dir):
  import os
  os.environ.setdefault('MPLCONFIGDIR', '/tmp/corewm-matplotlib')
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  from matplotlib.ticker import MultipleLocator
  import numpy as np
  import pandas as pd
  frame = pd.read_parquet(data_path); audit = json.loads(Path(audit_path).read_text())
  per_seed, aggregate = _binned(frame)
  output_dir = Path(output_dir); png_dir=output_dir/'png';pdf_dir=output_dir/'pdf'
  png_dir.mkdir(parents=True,exist_ok=True);pdf_dir.mkdir(parents=True,exist_ok=True)
  colors={'DreamerV3':'#3B82F6','Harmony Dreamer':'#F59E0B'}
  coverage=[]
  for game in ATARI100K_GAMES:
    fig,ax=plt.subplots(figsize=(7.2,4.5),constrained_layout=True)
    present=[]
    for method in METHOD_TAGS:
      values=aggregate[(aggregate.game==game)&(aggregate.method==method)]
      seeds=sorted(frame[(frame.game==game)&(frame.method==method)].seed.unique().tolist())
      complete = len(seeds) == 5
      coverage.append({'game':game,'method':method,'seeds':seeds,
                       'seed_count':len(seeds),'complete_five_seed_set':complete})
      if values.empty or not complete:continue
      present.append(method);x=values.agent_steps.to_numpy();mean=values['mean'].to_numpy();sem=values['sem'].to_numpy()
      ax.plot(x,mean,color=colors[method],lw=2.2,label=f'{method} (n={len(seeds)} seeds)')
      ax.fill_between(x,mean-sem,mean+sem,color=colors[method],alpha=.18,linewidth=0)
    ax.set_xlim(0,100_000);ax.set_xticks([0,25_000,50_000,75_000,100_000]);ax.set_xticklabels(['0','25k','50k','75k','100k'])
    ax.xaxis.set_minor_locator(MultipleLocator(5_000));ax.grid(which='major',alpha=.28);ax.grid(which='minor',axis='x',alpha=.09)
    ax.set_xlabel('Environment interactions (agent actions)');ax.set_ylabel('Episode return')
    ax.set_title(game.replace('_',' ').title());ax.legend(frameon=False,loc='best')
    if 'Harmony Dreamer' not in present:
      harmony_n = len(frame[(frame.game==game)&
                            (frame.method=='Harmony Dreamer')].seed.unique())
      note = ('Harmony data unavailable in current W&B audit' if harmony_n == 0
              else f'Harmony incomplete ({harmony_n}/5 finished seeds); curve withheld')
      ax.text(.99,.02,note,transform=ax.transAxes,ha='right',va='bottom',fontsize=8,color='#8A3B12')
    fig.savefig(png_dir/f'{game}.png',dpi=180);fig.savefig(pdf_dir/f'{game}.pdf');plt.close(fig)
  fig,axes=plt.subplots(7,4,figsize=(16,22),constrained_layout=True);axes=axes.ravel()
  for ax,game in zip(axes,ATARI100K_GAMES):
    for method in METHOD_TAGS:
      values=aggregate[(aggregate.game==game)&(aggregate.method==method)]
      seed_count = frame[(frame.game==game)&(frame.method==method)].seed.nunique()
      if values.empty or seed_count != 5:continue
      x=values.agent_steps.to_numpy();mean=values['mean'].to_numpy();sem=values['sem'].to_numpy()
      ax.plot(x,mean,color=colors[method],lw=1.5,label=method);ax.fill_between(x,mean-sem,mean+sem,color=colors[method],alpha=.16,linewidth=0)
    ax.set_xlim(0,100_000);ax.set_xticks([0,25_000,50_000,75_000,100_000]);ax.set_xticklabels(['0','25k','50k','75k','100k'],fontsize=7)
    ax.xaxis.set_minor_locator(MultipleLocator(5_000));ax.grid(which='major',alpha=.22);ax.grid(which='minor',axis='x',alpha=.06);ax.set_title(game.replace('_',' ').title(),fontsize=10)
  handles=[plt.Line2D([0],[0],color=colors[m],lw=2,label=m) for m in METHOD_TAGS]
  legend_ax = axes[len(ATARI100K_GAMES)]; legend_ax.axis('off')
  legend_ax.legend(handles=handles,loc='center',ncol=1,frameon=False,fontsize=11)
  note_ax = axes[len(ATARI100K_GAMES)+1]; note_ax.axis('off')
  note_ax.text(.5,.5,'Curves require all 5 finished seeds.\nShading: mean ± 1 SEM.',
               ha='center',va='center',fontsize=10)
  for ax in axes[len(ATARI100K_GAMES)+2:]:ax.axis('off')
  fig.supxlabel('Environment interactions (agent actions)');fig.supylabel('Episode return')
  fig.savefig(output_dir/'all_games_overview.png',dpi=180);fig.savefig(output_dir/'all_games_overview.pdf');plt.close(fig)
  pd.DataFrame(coverage).to_csv(output_dir/'coverage.csv',index=False)
  per_seed.to_csv(output_dir/'binned_per_seed.csv',index=False);aggregate.to_csv(output_dir/'binned_aggregate.csv',index=False)
  meta={'bin_width_actions':5_000,'major_tick_actions':25_000,'minor_tick_actions':5_000,
        'aggregation':'episode returns averaged within 5k bins per seed; mean +/- 1 SEM across seeds',
        'curve_inclusion':'a method/game curve is plotted only when all 5 seeds are finished',
        'x_axis':{'DreamerV3':'agent_actions_est','Harmony Dreamer':'W&B _step / action_repeat(4)'},
        'files':{'individual_png':26,'individual_pdf':26,'overview_png':'all_games_overview.png','overview_pdf':'all_games_overview.pdf'}}
  (output_dir/'plot_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
  return meta


def main(argv=None):
  parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest='command', required=True)
  audit = sub.add_parser('audit'); audit.add_argument('--output'); audit.add_argument('--quiet', action='store_true')
  inspect = sub.add_parser('inspect'); inspect.add_argument('--project', required=True); inspect.add_argument('--run-id', required=True)
  download = sub.add_parser('download'); download.add_argument('--audit',required=True); download.add_argument('--output-dir',required=True); download.add_argument('--refresh',action='store_true'); download.add_argument('--workers',type=int,default=8)
  plot = sub.add_parser('plot'); plot.add_argument('--data',required=True); plot.add_argument('--audit',required=True); plot.add_argument('--output-dir',required=True)
  one = sub.add_parser('download-one'); one.add_argument('--audit',required=True); one.add_argument('--output-dir',required=True); one.add_argument('--index',type=int,required=True); one.add_argument('--refresh',action='store_true')
  args = parser.parse_args(argv)
  if args.command=='audit':result=audit_runs()
  elif args.command=='inspect':result=inspect_run(args.project,args.run_id)
  elif args.command=='download':result=download_histories(args.audit,args.output_dir,args.refresh,args.workers)
  elif args.command=='download-one':result=download_one(args.audit,args.output_dir,args.index,args.refresh)
  else:result=plot_curves(args.data,args.audit,args.output_dir)
  text = json.dumps(result, indent=2, default=str) + '\n'
  if getattr(args, 'output', None):
    from pathlib import Path
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(text)
  if not getattr(args, 'quiet', False):
    print(text)


if __name__ == '__main__': main()
