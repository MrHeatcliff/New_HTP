"""One-factor rank-floor research, four fixed games on two shared H200s."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
import numpy as np
from threadpoolctl import threadpool_limits
from .h200_suite import slots, atomic
from .persistence_research_report import checkpoint_updates, control_comparison

GAMES=('boxing','up_n_down','frostbite','road_runner')
REPO=Path(os.environ.get('HTP_RESEARCH_REPO','/home/vn-user0101/Dat/HTS-Dreamer'))
SOURCE=REPO/'production_runs/constraint_full26_seed0_OzirkNB8'
FACTOR=os.environ.get('HTP_FOUR_GAME_FACTOR','rank_floor')
TRIAL_ACTIONS=int(os.environ.get('HTP_TRIAL_ACTIONS','20000'))
CONTINUATION_SEEDS=tuple(map(int,os.environ.get('HTP_CONTINUATION_SEEDS','0').split(',')))


def arms_for(factor):
  if factor=='rank_floor': return [('isotropy',0,0.0),('rank32',32,0.0)]
  if factor=='decorrelation': return [('isotropy',0,0.0),('decorrelated',0,0.01)]
  if factor=='decorrelation_guard':
    return [('isotropy',0,0.0),('decorrelated',0,0.01),('guarded',0,0.01)]
  raise ValueError(f'Unknown factor: {factor}')


def sha(path):
  result=hashlib.sha256()
  with Path(path).open('rb') as stream:
    for part in iter(lambda:stream.read(1024*1024),b''): result.update(part)
  return result.hexdigest()


def audit(root):
  status=json.loads((SOURCE/'status.json').read_text())
  assert status['status']=='COMPLETE' and len(status['results'])==26
  rows=[]
  for game,result in status['results'].items():
    assert result['complete'] and result['exit_code']==0
    episodes=[json.loads(line) for line in (SOURCE/'runs'/game/'paper_artifacts/episode_scores.jsonl').read_text().splitlines()]
    late=[x['episode_score'] for x in episodes if 90000<=x['agent_actions']<=100000]
    final=[x['episode_score'] for x in episodes if 100000<x['agent_actions']<=110000]
    rows.append({'game':game,'training_90k_100k_mean':float(np.mean(late)) if late else None,
        'episodes_90k_100k':len(late),'training_100k_110k_mean':float(np.mean(final)) if final else None,
        'seconds':result['seconds']})
  # Retain the historical baseline's source and seed distinction explicitly.
  with (REPO/'paper_artifacts/full_vs_dreamerv3_learning_curves/aggregate.csv').open() as stream:
    historical=list(csv.DictReader(stream))
  for row in rows:
    row['historical_last_points']={}
    for method in ('Full CoRe-WM','DreamerV3'):
      matching=[x for x in historical if x['game']==row['game'] and x['method']==method]
      point=max(matching,key=lambda x:float(x['agent_steps']))
      row['historical_last_points'][method]={'mean':float(point['mean']),'actions':float(point['agent_steps']),'seeds':int(point['seeds'])}
  atomic(root/'suite_observations.json',{'games':rows,'selected':GAMES,
      'note':'Exploratory selection from training curves; historical Full is isolated eval over 5 seeds. No causal win/loss claim.',
      'plot_audit':'Previously published purple final bin folded >100k into 95–100k. This report keeps those windows separate.'})


def common(game, floor, decorrelation=0.0, guard=0.0, seed=0):
  return [sys.executable,'-u','-m','dreamerv3.main_htp','--configs','htp_atari100k','size25m',
      '--task',f'atari100k_{game}','--seed',str(seed),'--env.atari100k.use_seed','True',
      '--jax.prealloc','False','--logger.outputs','jsonl','--run.log_policy_video','False',
      '--agent.htp.persistence_metric','whitened','--agent.htp.persistence_scale','0.1',
      '--agent.htp.persistence_all_lags','True','--agent.htp.persistence_isotropy','0.01',
      '--agent.htp.persistence_min_rank',str(floor),'--agent.htp.recon.first_target_window','1',
      '--agent.htp.persistence_decorrelation',str(decorrelation),
      '--agent.htp.persistence_block2_rank_scale',str(guard),'--agent.htp.persistence_block2_min_rank','8.0',
      '--agent.htp.pdyn.mask_episode_boundaries','False']


def worker_one(root,game,seed=0):
  out=root/game; out.mkdir(exist_ok=False)
  source=SOURCE/'runs'/game; checkpoint=source/'ckpt/env_action_steps_000100000'
  digest=sha(checkpoint/'agent.pkl'); initial=checkpoint_updates(checkpoint)
  arms=arms_for(FACTOR)
  names=[name for name,_,_ in arms]
  primary=names[-2:] if FACTOR=='decorrelation_guard' else names
  manifest={'game':game,'source':str(checkpoint),'source_sha256':digest,
      'changed_factor':{'rank_floor':'isotropy versus participation-rank floor 32',
          'decorrelation':'block-2 CKA regularizer weight: 0 versus 0.01; isotropy unchanged',
          'decorrelation_guard':'block-2 rank-floor-8 weight: 0 versus 0.001; CKA stays 0.01; extra isotropy anchor'}[FACTOR],
      'primary_comparison':primary,
      'new_actions_per_arm':TRIAL_ACTIONS,'evaluation_episodes':20,'source_training_seed':0,
      'continuation_seed':seed,'evaluation_seed':0,
      'replay':'Fresh replay for both continuation arms; actual update counts audited.'}
  atomic(out/'manifest.json',manifest)
  def execute(label,cmd):
    atomic(out/f'{label}_command.json',cmd)
    atomic(out/'status.json',{'stage':label,'status':'RUNNING'})
    started=time.monotonic()
    print(f'{time.strftime("%Y-%m-%d %H:%M:%S")} START {game} seed={seed} {label}',flush=True)
    with (out/f'{label}.log').open('w') as stream:
      subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,check=True)
    print(f'{time.strftime("%Y-%m-%d %H:%M:%S")} END {game} seed={seed} {label} seconds={time.monotonic()-started:.1f}',flush=True)
  def evaluate(label,config,cp):
    config=list(config); config[config.index('--seed')+1]='0'
    execute(label,config+['--script','eval_only','--logdir',str(out/label),
        '--run.eval_eps','20','--run.steps','1000000','--run.from_checkpoint',str(cp)])
  representation=[sys.executable,'-u','-m','corewm_eval.four_game_representation']
  dataset=out/'fixed_clips'
  cache_root=os.environ.get('HTP_FIXED_CLIPS_ROOT')
  if cache_root:
    previous=Path(cache_root)/game
    prior=json.loads((previous/'manifest.json').read_text())
    if prior['source_sha256']!=digest: raise RuntimeError('Diagnostic source checkpoint mismatch')
    shutil.copytree(previous/'fixed_clips',dataset)
    manifest['fixed_clips_reused_from']=str(previous/'fixed_clips')
    manifest['fixed_clips_hashes']={p.name:sha(p) for p in sorted(dataset.glob('*.npz'))}
    if manifest['fixed_clips_hashes']!={p.name:sha(p) for p in sorted((previous/'fixed_clips').glob('*.npz'))}:
      raise RuntimeError('Copied diagnostics changed')
  else:
    execute('prepare',representation+['prepare','--source',str(source),'--checkpoint',str(checkpoint),'--dataset',str(dataset)])
  execute('source_representation',representation+['analyze','--dataset',str(dataset),'--output',str(dataset/'source.json')])
  evaluate('source_eval',common(game,0),checkpoint)
  # Counterbalance stage order across games; each slot owns one game throughout.
  if len(arms)==3:
    offset=(GAMES.index(game)+seed)%len(arms)
    arms=arms[offset:]+arms[:offset]
  elif (GAMES.index(game)+seed)%2: arms.reverse()
  manifest['execution_order']=[name for name,_,_ in arms]; atomic(out/'manifest.json',manifest)
  for name,floor,decorrelation in arms:
    guard=0.001 if name=='guarded' else 0.0
    train=out/name; config=common(game,floor,decorrelation,guard,seed)
    execute(f'{name}_train',config+['--logdir',str(train),'--run.steps',str(TRIAL_ACTIONS),
        '--run.exact_env_action_budget','True','--run.action_milestones',
        *[str(x) for x in range(10000,TRIAL_ACTIONS+1,10000)],
        '--run.from_checkpoint',str(checkpoint)])
    final=train/f'ckpt/env_action_steps_{TRIAL_ACTIONS:09d}'
    if not (final/'agent.pkl').is_file(): raise RuntimeError('Missing completed action checkpoint')
    evaluate(f'{name}_eval',config,final)
    execute(f'{name}_extract',representation+['extract','--source',str(train),'--checkpoint',str(final),
        '--dataset',str(dataset),'--output',str(out/f'{name}_representation.npz')])
    execute(f'{name}_analyze',representation+['analyze','--dataset',str(dataset),
        '--output',str(out/f'{name}_representation.json')])
  if sha(checkpoint/'agent.pkl')!=digest: raise RuntimeError('Source checkpoint changed')
  def scores(label):
    values=np.array([json.loads(line)['episode/score'] for line in (out/label/'scores.jsonl').read_text().splitlines()])
    assert len(values)==20 and np.isfinite(values).all()
    return values
  rng=np.random.default_rng(20260910)
  reports={name:json.loads((out/f'{name}_representation.json').read_text()) for name in names}
  predictions={}
  for lag in ('16','32','64'):
    predictions[lag]={}
    for block in ('block1','block2','prefix2'):
      stats=[reports[name]['horizons'][lag][block] for name in primary]
      ids=sorted(stats[0]['episodes']); assert ids==sorted(stats[1]['episodes'])
      draw=rng.integers(len(ids),size=(20000,len(ids)))
      values=[]
      for stat in stats:
        sse=np.array([stat['episodes'][key]['sse'] for key in ids]); sst=np.array([stat['episodes'][key]['sst'] for key in ids])
        values.append(1-sse[draw].sum(1)/np.maximum(sst[draw].sum(1),1e-12))
      predictions[lag][block]={'reference_r2':stats[0]['r2'],'candidate_r2':stats[1]['r2'],
          'difference_ci95':np.quantile(values[1]-values[0],[.025,.975]).tolist()}
  comparison={'manifest':manifest,'control':control_comparison(scores(f'{primary[0]}_eval'),scores(f'{primary[1]}_eval'),rng),
      'control_vs_isotropy':{name:control_comparison(scores('isotropy_eval'),scores(f'{name}_eval'),rng) for name in names if name!='isotropy'},
      'source_eval_mean':float(scores('source_eval').mean()),'representation':reports,'prediction_intervals':predictions,
      'control_vs_source':{name:control_comparison(scores('source_eval'),scores(f'{name}_eval'),rng) for name in names},
      'updates':{name:checkpoint_updates(out/name/f'ckpt/env_action_steps_{TRIAL_ACTIONS:09d}')-initial for name in reports},
      'limitations':['This record has one continuation seed and 20 eval episodes; consult other replicate records.',
          'Continuation seeds share the same pretrained seed-0 checkpoint; not independent training seeds.',
          'Intervals condition on trained checkpoints; games selected after inspecting prior curves.',
          'Diagnostics cover episode starts, not complete gameplay; fixed h is not semantic ground truth.',
          'Require predictive gains plus retained control and no severe redundancy or collapse.',
          'Fresh-replay continuation can degrade both arms independently of the changed regularizer.',
          'CKA penalization can reduce block-2 variance; CKA alone cannot establish success.']}
  comparison['training_guard_diagnostics']={}
  for name in names:
    path=out/name/'paper_artifacts/train_metrics.jsonl'
    logged=[json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    diagnostics={}
    for metric in ('train/htp/block2_effective_rank','train/htp/block2_rank_guard',
                   'train/htp/block2_rank_guard_active','train/htp/block12_redundancy'):
      values=[float(row[metric]) for row in logged if metric in row]
      if values:
        diagnostics[metric]={'logged_samples':len(values),'min':min(values),'max':max(values),
            'mean_of_logged_aggregates':float(np.mean(values))}
    comparison['training_guard_diagnostics'][name]=diagnostics
  atomic(out/'comparison.json',comparison)
  atomic(out/'status.json',{'status':'COMPLETE'})


def worker(root,game):
  if len(CONTINUATION_SEEDS)==1:
    return worker_one(root,game,CONTINUATION_SEEDS[0])
  out=root/game; out.mkdir(exist_ok=False)
  results={}
  for seed in CONTINUATION_SEEDS:
    repeat=root/'replicates'/f'seed_{seed}'; repeat.mkdir(parents=True,exist_ok=True)
    atomic(out/'status.json',{'status':'RUNNING','continuation_seed':seed})
    worker_one(repeat,game,seed)
    results[str(seed)]=json.loads((repeat/game/'comparison.json').read_text())
  atomic(out/'comparison.json',{'game':game,'factor':FACTOR,'replicates':results,
      'note':'Replicates are continuation seeds sharing a pretrained checkpoint. Do not pool episodes as independent trained policies.'})
  atomic(out/'status.json',{'status':'COMPLETE'})


def supervise(root):
  if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('Slurm only')
  devices=[]
  for device in os.environ['CUDA_VISIBLE_DEVICES'].split(','):
    row=subprocess.check_output(['nvidia-smi','-i',device,'--query-gpu=uuid,name','--format=csv,noheader'],text=True)
    uuid,name=[x.strip() for x in row.split(',')]; assert 'H200' in name; devices.append(uuid)
  assigned=slots(sorted(os.sched_getaffinity(0)),devices)
  audit(root)
  atomic(root/'manifest.json',{'job':os.environ['SLURM_JOB_ID'],'games':GAMES,'slots':assigned,
      'factor':FACTOR,'arms':[name for name,_,_ in arms_for(FACTOR)],'new_actions_per_arm':TRIAL_ACTIONS,'source_actions':100000,
      'continuation_seeds':CONTINUATION_SEEDS,
      'estimated_hours':[1,1.5],'hard_time_limit_hours':4,'frozen_source':str(Path.cwd())})
  active={}; state={'status':'RUNNING','games':{}}
  def stop(signum,frame): raise RuntimeError(f'Signal {signum}')
  signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
  try:
    for game,slot in zip(GAMES,assigned):
      env=dict(os.environ,CUDA_VISIBLE_DEVICES=slot['gpu'],OMP_NUM_THREADS='8',MKL_NUM_THREADS='1',
          OPENBLAS_NUM_THREADS='1',XLA_PYTHON_CLIENT_PREALLOCATE='false',XLA_PYTHON_CLIENT_MEM_FRACTION='0.4',
          PYTHONHASHSEED='0',PAPER_DETERMINISTIC_UUID='1',PYTHONUNBUFFERED='1')
      cmd=['taskset','-c',','.join(map(str,slot['cpus'])),sys.executable,'-u','-m',
           'corewm_eval.four_game_rank_trial',str(root),'--worker',game]
      log=(root/f'{game}_worker.log').open('w')
      proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
      active[game]=(proc,log); state['games'][game]={'pid':proc.pid,'slot':slot}
      print(f'START {game}: GPU {slot["gpu"]}, CPUs {slot["cpus"]}',flush=True)
    atomic(root/'status.json',state)
    with (root/'telemetry.jsonl').open('w') as telemetry:
      while any(p.poll() is None for p,_ in active.values()):
        gpu_rows=[]
        for device in devices:
          row=subprocess.check_output(['nvidia-smi','-i',device,
              '--query-gpu=memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True)
          used,total,util=map(float,row.strip().split(',')); gpu_rows.append([device,used,total,util])
          if used>.9*total: raise RuntimeError('GPU memory safety threshold exceeded')
        rows=[r.split() for r in subprocess.check_output(['ps','-eo','pid=,ppid=,rss='],text=True).splitlines()]
        owned={p.pid for p,_ in active.values()}
        for _ in range(8): owned.update(int(r[0]) for r in rows if int(r[1]) in owned)
        rss=sum(int(r[2]) for r in rows if int(r[0]) in owned)/1024
        if rss>210*1024: raise RuntimeError('Host RSS safety threshold exceeded')
        for game,(p,_) in active.items(): state['games'][game]['exit_code']=p.poll()
        atomic(root/'status.json',state)
        telemetry.write(json.dumps({'time':time.time(),'gpus':gpu_rows,'rss_mib':rss})+'\n'); telemetry.flush()
        time.sleep(5)
    completed={game:json.loads((root/game/'comparison.json').read_text()) for game,(p,_) in active.items()
               if p.returncode==0 and (root/game/'comparison.json').exists()}
    atomic(root/'comparison.json',completed)
    state['status']='COMPLETE' if len(completed)==4 else 'COMPLETE_WITH_FAILURES'
  except BaseException as exc:
    state.update(status='INTERRUPTED',error=str(exc)); raise
  finally:
    for game,(p,log) in active.items():
      if p.poll() is None:
        try: os.killpg(p.pid,signal.SIGTERM)
        except ProcessLookupError: pass
      try: p.wait(timeout=15)
      except subprocess.TimeoutExpired:
        os.killpg(p.pid,signal.SIGKILL); p.wait()
      state['games'][game]['exit_code']=p.returncode; log.close()
    atomic(root/'status.json',state)
  if state['status']!='COMPLETE': raise RuntimeError(state['status'])


if __name__=='__main__':
  parser=argparse.ArgumentParser(); parser.add_argument('root'); parser.add_argument('--worker',choices=GAMES)
  args=parser.parse_args(); root=Path(args.root).resolve()
  if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('Slurm only')
  if TRIAL_ACTIONS<10000 or TRIAL_ACTIONS%10000 or not CONTINUATION_SEEDS or len(set(CONTINUATION_SEEDS))!=len(CONTINUATION_SEEDS):
    raise ValueError('Use distinct seeds and a positive multiple of 10000 actions')
  with threadpool_limits(limits=1):
    if args.worker: worker(root,args.worker)
    else: supervise(root)
