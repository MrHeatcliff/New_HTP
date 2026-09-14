"""Read-only two-training-seed EDA; no training/evaluation/model changes."""
import hashlib
import json
import os
from pathlib import Path


def main():
  assert os.environ.get('SLURM_JOB_ID')
  import numpy as np
  import pandas as pd
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  from ruamel.yaml import YAML
  out=Path('paper_artifacts/upndown_harmony_seed0_vs_seed1');out.mkdir(parents=True,exist_ok=True)
  roots=[Path('production_runs/reborn_harmony_2ew6wt0w'),Path('production_runs/harmony_seeds1to4_mi8h6pay')]
  trains=[roots[0]/'harmony/up_n_down/full',roots[1]/'seed_1/up_n_down/full']
  evals=[roots[0]/'harmony/evaluation/up_n_down',roots[1]/'seed_1/up_n_down/evaluation']
  def read(p): return [json.loads(x) for x in p.read_text().splitlines()]
  def flat(d,p=''):
    result={}
    for k,v in d.items():
      key=p+'.'+k if p else k
      if isinstance(v,dict):result.update(flat(v,key))
      else:result[key]=v
    return result
  configs=[flat(YAML(typ='safe').load((p/'config.yaml').read_text())) for p in trains]
  diff={k:[configs[0].get(k),configs[1].get(k)] for k in set(configs[0])|set(configs[1]) if configs[0].get(k)!=configs[1].get(k)}
  files=['dreamerv3/reborn.py','dreamerv3/agent_reborn.py','dreamerv3/agent_htp.py','dreamerv3/configs.yaml','embodied/envs/atari.py']
  hashes={name:[hashlib.sha256((r/'source'/name).read_bytes()).hexdigest() for r in roots] for name in files}
  assert all(a==b for a,b in hashes.values())
  stats=[]; ep_records=[]; train_records=[]; metric_frames=[]
  keys=['train/ent/action','train/rand/action','train/loss/value','train/loss/rew','train/loss/image',
        'train/opt/grad_norm','train/harmony/observation/weight','train/harmony/task/weight',
        'train/harmony/effective_beta_observation','train/reborn/imag_return_raw_mean',
        'train/reborn/l1/prefix_rate_nats','train/reborn/l5/prefix_rate_nats',
        'train/reborn/l1/mu_variance','train/reborn/l5/mu_variance',
        'train/reborn/l1/mu_dead_fraction','train/reborn/l5/mu_dead_fraction',
        'train/reborn/l1/q_td_mae','train/reborn/l1/q_td_mae_nonzero_reward',
        'train/reborn/l1/q_target_abs','train/reborn/l1/sf_td_mse','train/reborn/l1/sf_zero_mse',
        'train/reborn/l5/prefix_image_mse','train/reborn/valid_transition_count',
        'train/reborn/nonzero_reward_transition_count','train/reborn/terminal_transition_count',
        'train/rew','train/ret','train/val','train/tar','train/adv_std']
  for seed,(train,ev) in enumerate(zip(trains,evals)):
    episodes=read(train/'paper_artifacts/episode_scores.jsonl')
    for r in episodes:
      train_records.append(dict(seed=seed,actions=r['agent_actions'],score=r['episode_score'],
          length_actions=r['episode_length']-1,optimizer_updates=r['optimizer_updates']))
    for step in range(10000,100001,10000):
      rows=read(ev/f'{step:06d}/paper_artifacts/eval_metrics.jsonl')
      s=np.array([r['episode_score'] for r in rows]);a=np.array([r['episode_length']-1 for r in rows])
      assert len(rows)==(100 if step==100000 else 10) and (a>0).all()
      stats.append(dict(seed=seed,checkpoint=step,n=len(rows),mean=s.mean(),median=np.median(s),
          p10=np.quantile(s,.1),p90=np.quantile(s,.9),mean_actions=a.mean(),density=s.sum()/a.sum(),cap=int((a==27000).sum())))
      for score,length in zip(s,a):ep_records.append(dict(seed=seed,checkpoint=step,score=score,actions=length))
    m=pd.DataFrame(read(train/'metrics.jsonl'))
    m=m[m['train/ent/action'].notna()].copy()
    # Logger step = driver callbacks * repeat; episode logs give exact action anchors.
    # Interpolation only aligns metric timestamps, never fills returns or missing episodes.
    x=np.array([r['driver_callbacks']*4 for r in episodes]);y=np.array([r['agent_actions'] for r in episodes])
    m=m[(m.step>=x.min())&(m.step<=x.max())]
    m['actions_est']=np.interp(m.step,x,y);m['seed']=seed
    m['sf_relative']=m['train/reborn/l1/sf_td_mse']/m['train/reborn/l1/sf_zero_mse']
    m['nonzero_fraction']=m['train/reborn/nonzero_reward_transition_count']/m['train/reborn/valid_transition_count']
    m['terminal_fraction']=m['train/reborn/terminal_transition_count']/m['train/reborn/valid_transition_count']
    metric_frames.append(m[['seed','step','actions_est',*keys,'sf_relative','nonzero_fraction','terminal_fraction']])
  metrics=pd.concat(metric_frames,ignore_index=True)
  metrics.to_csv(out/'training_metrics.csv',index=False)
  pd.DataFrame(train_records).to_csv(out/'training_episodes.csv',index=False)
  pd.DataFrame(ep_records).to_csv(out/'evaluation_episodes.csv',index=False)
  df=pd.DataFrame(stats);df.to_csv(out/'evaluation_summary.csv',index=False)
  metrics['window']=pd.cut(metrics.actions_est,[0,20000,40000,60000,80000,100000],labels=['0-20K','20-40K','40-60K','60-80K','80-100K'])
  windows=metrics.groupby(['seed','window'],observed=True).mean(numeric_only=True).reset_index()
  windows.to_csv(out/'metric_windows.csv',index=False)
  final=[df[(df.seed==s)&(df.checkpoint==100000)].iloc[0] for s in (0,1)]
  ratios={k:float(final[0][k]/final[1][k]) for k in ['mean','mean_actions','density']}
  payload=dict(config_differences=diff,source_hashes=hashes,final_ratios_seed0_over_seed1=ratios,
      metric_timestamp='interpolated exact-action anchors from driver_callbacks*4; only within anchor range',
      training_episode_counts=[sum(r['seed']==s for r in train_records) for s in (0,1)])
  (out/'audit.json').write_text(json.dumps(payload,indent=2,default=str)+'\n')
  colors=['#7c3aed','#e09b24']
  fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
  for seed in (0,1):
    d=df[df.seed==seed];e=pd.DataFrame(ep_records);e=e[(e.seed==seed)&(e.checkpoint==100000)]
    for ax,key in zip(axes.flat[:4],['mean','mean_actions','density','cap']):
      ax.plot(d.checkpoint/1000,d[key]/d.n if key=='cap' else d[key],'.-',color=colors[seed],label=f'seed {seed}')
      ax.set(xlabel='Checkpoint (K actions)',title=key)
    s=np.sort(e.score);axes[1,1].step(s,np.arange(1,len(s)+1)/len(s),where='post',color=colors[seed])
    axes[1,2].scatter(e.actions,e.score,s=14,alpha=.6,color=colors[seed])
  axes[1,1].set(title='Final return ECDF',xlabel='Return',ylabel='Cumulative fraction')
  axes[1,2].set(title='Final episodes',xlabel='Actions',ylabel='Return')
  axes[0,0].legend();fig.suptitle('Up N Down | Harmony seed 0 vs 1 | isolated evaluation')
  for ax in axes.flat:ax.grid(alpha=.2)
  for ext in ('png','pdf'):fig.savefig(out/f'evaluation_decomposition.{ext}',dpi=170)
  plt.close(fig)
  selected=['train/ent/action','train/reborn/imag_return_raw_mean','train/loss/value',
    'train/harmony/observation/weight','train/harmony/effective_beta_observation','train/harmony/task/weight',
    'train/reborn/l1/prefix_rate_nats','train/reborn/l1/mu_variance','train/reborn/l5/prefix_image_mse',
    'train/reborn/l1/q_td_mae','sf_relative','terminal_fraction']
  fig,axes=plt.subplots(4,3,figsize=(15,14),layout='constrained')
  for ax,key in zip(axes.flat,selected):
    for seed in (0,1):
      d=metrics[metrics.seed==seed];ax.plot(d.actions_est/1000,d[key],color=colors[seed],label=f'seed {seed}',lw=1)
    ax.set(title=key.replace('train/',''),xlabel='Agent actions (K; aligned)');ax.grid(alpha=.2)
  axes[0,0].legend();fig.suptitle('Logged training statistics | no smoothing | observational, not causal')
  for ext in ('png','pdf'):fig.savefig(out/f'training_signals.{ext}',dpi=170)
  plt.close(fig)
  print(json.dumps(payload,indent=2,default=str))
  print(df.to_string(index=False))
  print(windows[['seed','window',*selected]].to_string(index=False))


if __name__=='__main__':main()
