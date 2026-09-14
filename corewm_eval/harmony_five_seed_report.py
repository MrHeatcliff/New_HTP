"""Final five-seed report, stratified bootstrap, and matched-measurement curves."""
import hashlib
import json
import os
from pathlib import Path


def main():
  assert os.environ.get('SLURM_JOB_ID'), 'CPU Slurm required'
  import numpy as np
  import pandas as pd
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  from scipy.stats import trim_mean
  from .config import ATARI100K_GAMES as games
  from .hns_reference import load_reference
  from .full_vs_dreamerv3_curves import _binned
  out=Path('paper_artifacts/harmony_five_seed_results');out.mkdir(parents=True,exist_ok=True)
  root=Path('production_runs/harmony_seeds1to4_mi8h6pay')
  assert json.loads((root/'status.json').read_text())['completed']==104
  sources={}
  def readjson(p):
    sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text())
  rows=readjson(Path('paper_artifacts/harmony_vs_dreamerv3_26_learning_curves/evaluation_results.json'))
  for seed in range(1,5):
    for game in games:
      r=readjson(root/f'seed_{seed}/{game}/evaluation/summary.json')
      assert len(r)==10 and all(x['training_seed']==seed and x['game']==game for x in r)
      rows.extend(r)
  assert len(rows)==1300
  seen=set()
  for r in rows:
    key=(r['game'],r['training_seed'],r['checkpoint'])
    assert key not in seen;seen.add(key)
    assert r['evaluation_seed']==0 and r['episodes']==len(r['returns'])==(100 if r['checkpoint']==100000 else 10)
    assert np.isfinite(r['returns']).all() and np.isclose(np.mean(r['returns']),r['mean'])
  assert seen=={(g,s,k) for g in games for s in range(5) for k in range(10000,100001,10000)}
  (out/'evaluation_results.json').write_text(json.dumps(rows,indent=2)+'\n')
  refs=load_reference('baselines.yaml','atari57_gamer')
  sources['baselines.yaml']=hashlib.sha256(Path('baselines.yaml').read_bytes()).hexdigest()
  data=pd.DataFrame([{k:v for k,v in r.items() if k!='returns'} for r in rows])
  data.to_csv(out/'evaluation_checkpoints.csv',index=False)
  final=data[data.checkpoint==100000].copy()
  final['random']=final.game.map(lambda g:refs[g][0]);final['human']=final.game.map(lambda g:refs[g][1])
  final['hns']=(final['mean']-final.random)/(final.human-final.random)
  final.to_csv(out/'final_per_seed.csv',index=False)
  matrix=final.pivot(index='training_seed',columns='game',values='hns').reindex(index=range(5),columns=games).to_numpy()
  assert matrix.shape==(5,26) and np.isfinite(matrix).all()
  def aggregates(x):
    return np.array([x.mean(),np.median(x.mean(0)),trim_mean(x,.25,axis=None),np.maximum(1-x,0).mean()])
  point=aggregates(matrix)
  # Independently resample training runs within each fixed game; episodes are not runs.
  rng=np.random.default_rng(20260914);reps=20000
  boot=np.empty((reps,4)); thresholds=np.unique(np.r_[np.linspace(0,4,81),matrix.ravel(),20])
  profiles=np.empty((reps,len(thresholds)))
  for b in range(reps):
    sample=matrix[rng.integers(0,5,size=(5,26)),np.arange(26)[None,:]]
    boot[b]=aggregates(sample)
    profiles[b]=(sample.ravel()[:,None]>thresholds).mean(0)
  ci=np.quantile(boot,[.025,.975],axis=0)
  metrics={k:dict(estimate=float(v),ci95_low=float(ci[0,i]),ci95_high=float(ci[1,i]))
           for i,(k,v) in enumerate(zip(['mean_hns','median_hns','iqm_hns','optimality_gap'],point))}
  metrics['superhuman_games']=int((matrix.mean(0)>1).sum())
  (out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
  assert np.isclose(point[2],np.sort(matrix.ravel())[32:-32].mean())
  assert np.isclose(point[3],1-np.minimum(matrix,1).mean())
  pergame=final.groupby('game')['mean'].agg(['mean','std']).reindex(games)
  pergame['hns']=matrix.mean(0)
  pergame.to_csv(out/'per_game.csv')
  curve=(matrix.ravel()[:,None]>thresholds).mean(0);pci=np.quantile(profiles,[.025,.975],axis=0)
  pd.DataFrame(dict(threshold=thresholds,fraction_above=curve,ci_low=pci[0],ci_high=pci[1])).to_csv(out/'performance_profile.csv',index=False)
  plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
  fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
  for ax,limit in zip(axes,[2,20]):
    ax.step(thresholds,curve,where='post',color='#7c3aed')
    ax.fill_between(thresholds,pci[0],pci[1],step='post',alpha=.2,color='#7c3aed')
    ax.axvline(1,color='gray',ls='--');ax.set(xlim=(0,limit),ylim=(0,1),xlabel='HNS threshold',ylabel='Fraction of runs × games above threshold');ax.grid(alpha=.2)
  fig.suptitle('Harmony | final 100K | 5 seeds × 26 games | pointwise 95% bootstrap CI')
  for ext in ('png','pdf'):fig.savefig(out/f'performance_profile.{ext}',dpi=180)
  plt.close(fig)
  train=[]
  for seed in range(5):
    for game in games:
      if seed:
        p=root/f'seed_{seed}/{game}/full/paper_artifacts/episode_scores.jsonl'
      else:
        base=Path('production_runs/reborn_harmony_2ew6wt0w/harmony') if game in ('boxing','up_n_down','frostbite','road_runner') else Path('production_runs/harmony_remaining22_zvn1urx5')
        p=base/game/'full/paper_artifacts/episode_scores.jsonl'
      sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
      for line in p.read_text().splitlines():
        r=json.loads(line);assert r['seed']==seed
        if 0<r['agent_actions']<=100000:
          train.append(dict(method='Harmony',game=game,seed=seed,agent_steps=r['agent_actions'],**{'return':r['episode_score']}))
  train=pd.DataFrame(train);assert train.groupby(['game','seed']).ngroups==130
  train.to_csv(out/'training_episodes.csv',index=False)
  ps,agg=_binned(train);ps.to_csv(out/'training_per_seed.csv',index=False)
  baseline=Path('paper_artifacts/harmony_corewm_dreamerv3_26_training_curves/aggregate.csv')
  sources[str(baseline)]=hashlib.sha256(baseline.read_bytes()).hexdigest()
  old=pd.read_csv(baseline);old=old[old.method.isin(['DreamerV3','CoRe-WM old'])]
  combined=pd.concat([old,agg],ignore_index=True);combined.to_csv(out/'training_aggregate.csv',index=False)
  evalagg=data.groupby(['game','checkpoint'])['mean'].agg(mean='mean',std='std',seeds='count').reset_index()
  evalagg['sem']=evalagg['std']/np.sqrt(evalagg.seeds);evalagg['agent_steps']=evalagg.checkpoint;evalagg['method']='Harmony'
  p=Path('paper_artifacts/full_vs_dreamerv3_learning_curves/aggregate.csv');sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
  oldeval=pd.read_csv(p);oldeval=oldeval[oldeval.method=='Full CoRe-WM'].copy();oldeval['method']='CoRe-WM old'
  evalcombined=pd.concat([oldeval,evalagg],ignore_index=True);evalcombined.to_csv(out/'evaluation_aggregate.csv',index=False)
  styles=[('DreamerV3','#3b82f6'),('CoRe-WM old','#e09b24'),('Harmony','#7c3aed')]
  def draw(ax,g,frame,kind):
    for method,color in styles:
      d=frame[(frame.game==g)&(frame.method==method)].sort_values('agent_steps')
      if d.empty:continue
      if kind=='training':
        d=d.set_index('bin').reindex(range(20));x=(np.arange(20)+.5)*5000
      else:x=d.agent_steps
      ax.plot(x,d['mean'],'.-',color=color,lw=1.6,markersize=3,label=method+' (5 seeds)')
      sem=d['sem'].where(d.seeds>=2)
      ax.fill_between(x,d['mean']-sem,d['mean']+sem,color=color,alpha=.15,lw=0)
    ax.set(title=g.replace('_',' ').title(),xlim=(0,100000),ylabel=kind.title()+' episode return',xlabel='Agent actions')
    ax.set_xticks([0,50000,100000],['0','50K','100K']);ax.grid(alpha=.2);ax.tick_params(labelsize=8)
  for kind,frame in [('training',combined),('evaluation',evalcombined)]:
    for ext in ('png','pdf'):(out/kind/ext).mkdir(parents=True,exist_ok=True)
    for g in games:
      fig,ax=plt.subplots(figsize=(8,5));draw(ax,g,frame,kind);ax.legend(frameon=False,fontsize=9)
      fig.text(.5,.01,'Mean ±1 SEM | No smoothing | '+('5K bins; missing bins remain gaps' if kind=='training' else '10 episodes at 10K–90K; 100 at 100K'),ha='center',fontsize=8)
      fig.tight_layout(rect=(0,.04,1,1))
      for ext in ('png','pdf'):fig.savefig(out/kind/ext/f'{g}.{ext}',dpi=170)
      plt.close(fig)
    fig,axes=plt.subplots(7,4,figsize=(19,23))
    for ax,g in zip(axes.flat,games):draw(ax,g,frame,kind)
    for ax in list(axes.flat)[26:]:ax.set_visible(False)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.suptitle(f'Atari 100K | {kind.title()} returns | 26 games × 5 seeds',fontsize=20,y=.995)
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.98),ncol=3,frameon=False)
    fig.text(.5,.95,'Bands: ±1 SEM | No smoothing | Model sizes differ; not a controlled architecture ablation',ha='center')
    fig.tight_layout(rect=(0,0,1,.935),h_pad=2)
    for ext in ('png','pdf'):fig.savefig(out/kind/f'all_games_overview.{ext}',dpi=170)
    plt.close(fig)
  meta=dict(job=os.environ['SLURM_JOB_ID'],sources=sources,seeds=list(range(5)),games=list(games),
      checkpoint_points=1300,evaluation_episodes=sum(r['episodes'] for r in rows),final_episodes=13000,
      bootstrap=dict(replicates=reps,rng_seed=20260914,unit='training run within each fixed game',
          method='percentile 2.5/97.5; independent within-game resampling; no episode resampling',profile_ci='pointwise, not simultaneous'),
      model_size_caveat='Harmony size25m versus historical CoRe-WM size12m',
      missing_training_bins='available-seed means, no imputation; no SEM with <2 seeds')
  (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
  text='# Harmony — kết quả đầy đủ Atari 100K, 26 game × 5 seeds\n\n'
  text+='Đã hoàn tất seeds 0–4: **130 runs, 1.300 checkpoint evaluations, 24.700 eval episodes**, trong đó 13.000 final episodes. Seeds 1–4 từ job 4080, hoàn tất 14/09/2026 21:20 UTC+7 sau khoảng 32h57m; seed 0 tái sử dụng.\n\n## Headline: checkpoint cuối 100K\n\n| Metric | Estimate | 95% bootstrap CI |\n|---|---:|---:|\n'
  for k in ['mean_hns','median_hns','iqm_hns','optimality_gap']:
    m=metrics[k];scale=1 if k=='optimality_gap' else 100;label=k+(' (%)' if scale==100 else '')
    text+=f"| {label} | {scale*m['estimate']:.3f} | [{scale*m['ci95_low']:.3f}, {scale*m['ci95_high']:.3f}] |\n"
  text+=f"\n**Games above human: {metrics['superhuman_games']}/26**, xét mean HNS qua 5 seeds > 1. Không chọn best checkpoint.\n\n"
  text+='## Final score từng game\n\nMỗi seed: mean 100 eval episodes. SD là sample SD giữa 5 training seeds, không phải CI.\n\n| Game | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Mean ± SD | HNS (%) |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
  for g in games:
    values=final[final.game==g].sort_values('training_seed')['mean'];r=pergame.loc[g]
    text+=f"| {g} | "+' | '.join(f'{v:.2f}' for v in values)+f" | {r['mean']:.2f} ± {r['std']:.2f} | {100*r.hns:.2f} |\n"
  text+='''
## Cách tính và giới hạn

HNS(g,s) = (mean_final_return(g,s) − random(g)) / (human(g) − random(g)).
Dùng reference đầy đủ precision từ `baselines.yaml:atari57_gamer`; không clip HNS.
Mean/Median HNS: mean/median của 26 game means qua seeds. IQM: `scipy.stats.trim_mean`
trên 130 run×game HNS, bỏ floor(130×0.25)=32 điểm mỗi đuôi, trung bình 66 điểm còn lại.
Optimality gap: mean(max(1−HNS,0)) trên 130 điểm; ngưỡng human-level, thấp hơn tốt hơn.

95% CI: 20.000 stratified bootstrap replicates; mỗi game cố định resample 5 training
runs có hoàn lại, độc lập giữa games, rồi tính lại estimator. RNG seed 20260914;
percentile 2.5/97.5. Không bootstrap episodes làm training seeds, không resample games.
CI phản ánh uncertainty từ 5 runs quan sát trên suite cố định, không bảo đảm bao phủ
mọi failure mode. Performance profile dùng toàn bộ run×game; bands là pointwise CI,
không phải simultaneous confidence band. Công thức estimator tương ứng
[rliable](https://github.com/google-research/rliable/blob/master/rliable/metrics.py);
bootstrap ở đây được triển khai bằng NumPy, không gọi trực tiếp rliable.

Script kiểm tra đủ matrix 26×5×10, không duplicate, episode counts, finite returns,
mean khớp raw episodes; đối chiếu IQM/gap bằng công thức tương đương.

## Performance profile

![Profile](performance_profile.png)

## Training curves: cả ba bản đủ 5 seeds

![Training overview](training/all_games_overview.png)

Harmony, DreamerV3 và CoRe-WM cũ đều dùng **training return**: bin 5K actions mỗi
seed, rồi mean ±1 SEM giữa seeds có dữ liệu. Số seeds/bin nằm trong CSV; bin trống
giữ gap, không nội suy, không smoothing. SEM không phải 95% CI. Harmony/CoRe-WM dùng
agent_actions gốc; DreamerV3 dùng reference training đã cache. Điểm cuối bin không
thay thế final evaluation. Không pooling episodes qua seeds để bias seed có episode ngắn.

## Isolated evaluation curves

![Evaluation overview](evaluation/all_games_overview.png)

Chỉ Harmony và CoRe-WM cũ: cùng lịch 10K–100K và số eval episodes/checkpoint.
Không đưa DreamerV3 training return vào hình evaluation. Cùng measurement/budget
không đảm bảo matched architecture: Harmony size25m, CoRe-WM cũ size12m. Không quy
mọi khác biệt cho method hoặc tuyên bố thắng paper DreamerV3 từ các hình này.
Không báo probability of improvement với DreamerV3 do thiếu matched final eval data.

## Observation chính

Up N Down: seeds 0/2 đạt 196432.5/97739.8 nhưng seeds 1/3/4 chỉ 5279.3/12935.3/11140.1.
Thành công cao được tái lập ở hơn một seed, nhưng không ổn định; Mean HNS vẫn nhạy
đuôi lớn nên phải đọc cùng Median/IQM và CI. Frostbite có nhóm seeds 0/2/4 khoảng
2800–2900, seeds 1/3 khoảng 240–280; không thể kết luận nguyên nhân chỉ từ score.
Đây là cấu hình được chọn sau screening, không phải đánh giá xác nhận hoàn toàn
độc lập khỏi quá trình chọn method. Evaluation có giới hạn episode; không ngoại suy
return không giới hạn của Up N Down. Chi tiết seed-0 long episodes:
[báo cáo trước](../upndown_long_episode_report/README.md).

[EDA chi tiết Up N Down seed 0 vs seed 1](../upndown_harmony_seed0_vs_seed1/README.md)
được giữ như phân tích hai seeds, không thay thế kết quả đủ năm seeds ở trên.

## Hình riêng từng game

| Game | Training PNG | Training PDF | Eval PNG | Eval PDF |
|---|---|---|---|---|
'''
  for g in games:text+=f'| {g} | [PNG](training/png/{g}.png) | [PDF](training/pdf/{g}.pdf) | [PNG](evaluation/png/{g}.png) | [PDF](evaluation/pdf/{g}.pdf) |\n'
  text+='\n## Dữ liệu và tái tạo\n\n[Raw evaluation returns](evaluation_results.json) · [Final per seed](final_per_seed.csv) · [Metrics + CI](metrics.json) · [Provenance](metadata.json)\n\nTraining episodes, per-seed bins, aggregates, profile CSV được lưu cùng thư mục. Không commit checkpoints/replay/ROM.\n\nSubmit CPU: `sbatch scripts/slurm_harmony_five_seed_report.sh`. Script: [harmony_five_seed_report.py](../../corewm_eval/harmony_five_seed_report.py). Tái tạo cần training logs gốc; toàn bộ raw eval returns và processed curve data đã được export để người đọc phân tích mà không cần checkpoint. [Cách train lại](../../reproduction/harmony/README.md).\n'
  (out/'README.md').write_text(text)
  print(json.dumps(metrics,indent=2));print('PASS: 1300 evaluations, 24700 episodes; reports and all figures generated.')


if __name__=='__main__':main()
