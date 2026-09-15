"""Explicitly selection-biased best-five-of-six visualization, not headline results."""
import hashlib
import json
import os
from pathlib import Path


def main():
  assert os.environ.get('SLURM_JOB_ID')
  import numpy as np
  import pandas as pd
  from scipy.stats import trim_mean
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  from .config import ATARI100K_GAMES as games
  from .hns_reference import load_reference
  from .full_vs_dreamerv3_curves import _binned
  out=Path('paper_artifacts/harmony_best5_of6');out.mkdir(parents=True,exist_ok=True)
  prior=Path('paper_artifacts/harmony_five_seed_results')
  root=Path('production_runs/harmony_seeds5to14_p5_tr_l0/seed_5')
  sources={}
  def record(p):sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return p
  rows=json.loads(record(prior/'evaluation_results.json').read_text())
  train=pd.read_csv(record(prior/'training_episodes.csv'))
  more=[]
  for g in games:
    assert json.loads((root/g/'stage.json').read_text())['status']=='COMPLETE'
    rs=json.loads(record(root/g/'evaluation/summary.json').read_text())
    assert all(r['training_seed']==5 and r['game']==g for r in rs)
    rows.extend(rs)
    for line in record(root/g/'full/paper_artifacts/episode_scores.jsonl').read_text().splitlines():
      r=json.loads(line);assert r['seed']==5
      if 0<r['agent_actions']<=100000:more.append(dict(method='Harmony',game=g,seed=5,agent_steps=r['agent_actions'],**{'return':r['episode_score']}))
  assert len(rows)==1560
  seen=set()
  for r in rows:
    key=(r['game'],r['training_seed'],r['checkpoint']);assert key not in seen;seen.add(key)
    assert len(r['returns'])==r['episodes']==(100 if r['checkpoint']==100000 else 10)
    assert np.isfinite(r['returns']).all() and np.isclose(np.mean(r['returns']),r['mean'])
  assert seen=={(g,s,k) for g in games for s in range(6) for k in range(10000,100001,10000)}
  refs=load_reference(record(Path('baselines.yaml')),'atari57_gamer')
  df=pd.DataFrame([{k:v for k,v in r.items() if k!='returns'} for r in rows])
  df['hns']=[(r['mean']-refs[r['game']][0])/(refs[r['game']][1]-refs[r['game']][0]) for r in rows]
  final=df[df.checkpoint==100000]
  ranking=final.groupby('training_seed').hns.mean().reset_index(name='mean_hns').sort_values(['mean_hns','training_seed'],ascending=[False,True])
  chosen=ranking.training_seed.iloc[:5].astype(int).tolist();excluded=ranking.training_seed.iloc[5:].astype(int).tolist()
  ranking['selected']=ranking.training_seed.isin(chosen);ranking.to_csv(out/'seed_ranking.csv',index=False)
  final.to_csv(out/'all_six_final_scores.csv',index=False)
  (out/'all_six_evaluation_results.json').write_text(json.dumps(rows,indent=2)+'\n')
  def metrics(seeds):
    x=final[final.training_seed.isin(seeds)].pivot(index='training_seed',columns='game',values='hns').reindex(columns=games).to_numpy()
    return dict(mean_hns=float(x.mean()),median_hns=float(np.median(x.mean(0))),iqm_hns=float(trim_mean(x,.25,axis=None)),
        gap=float(np.maximum(1-x,0).mean()),superhuman=int((x.mean(0)>1).sum()))
  scores={'original_0_to_4':metrics(range(5)),'all_six':metrics(range(6)),'selected_five':metrics(chosen)}
  (out/'metrics.json').write_text(json.dumps(scores,indent=2)+'\n')
  train=pd.concat([train,pd.DataFrame(more)],ignore_index=True)
  train.to_csv(out/'all_six_training_episodes.csv',index=False)
  ps,agg=_binned(train[train.seed.isin(chosen)])
  ps.to_csv(out/'selected_training_per_seed.csv',index=False)
  baseline=pd.read_csv(record(prior/'training_aggregate.csv'));baseline=baseline[baseline.method!='Harmony']
  training=pd.concat([baseline,agg],ignore_index=True);training.to_csv(out/'training_aggregate.csv',index=False)
  ev=df[df.training_seed.isin(chosen)].groupby(['game','checkpoint'])['mean'].agg(mean='mean',std='std',seeds='count').reset_index()
  ev['sem']=ev['std']/np.sqrt(ev.seeds);ev['agent_steps']=ev.checkpoint;ev['method']='Harmony'
  old=pd.read_csv(record(prior/'evaluation_aggregate.csv'));old=old[old.method!='Harmony']
  evaluation=pd.concat([old,ev],ignore_index=True);evaluation.to_csv(out/'evaluation_aggregate.csv',index=False)
  per=final[final.training_seed.isin(chosen)].groupby('game').agg(mean_return=('mean','mean'),sd=('mean','std'),hns=('hns','mean')).reindex(games)
  per.to_csv(out/'selected_per_game.csv')
  plt.rcParams.update({'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
  styles=[('DreamerV3','#3b82f6'),('CoRe-WM old','#e09b24'),('Harmony','#7c3aed')]
  def draw(ax,g,frame,kind):
    for name,color in styles:
      d=frame[(frame.game==g)&(frame.method==name)].sort_values('agent_steps')
      if d.empty:continue
      if kind=='training':d=d.set_index('bin').reindex(range(20));x=(np.arange(20)+.5)*5000
      else:x=d.agent_steps
      label='Harmony (selected best 5/6)' if name=='Harmony' else name+' (original 5 seeds)'
      ax.plot(x,d['mean'],'.-',color=color,lw=1.6,markersize=3,label=label)
      sem=d['sem'].where(d.seeds>=2);ax.fill_between(x,d['mean']-sem,d['mean']+sem,color=color,alpha=.15,lw=0)
    ax.set(title=g.replace('_',' ').title(),xlim=(0,100000),xlabel='Agent actions',ylabel=kind.title()+' return')
    ax.set_xticks([0,50000,100000],['0','50K','100K']);ax.grid(alpha=.2)
  for kind,frame in [('training',training),('evaluation',evaluation)]:
    for ext in ('png','pdf'):(out/kind/ext).mkdir(parents=True,exist_ok=True)
    for g in games:
      fig,ax=plt.subplots(figsize=(8,5));draw(ax,g,frame,kind);ax.legend(fontsize=8,frameon=False)
      fig.suptitle('POST-HOC SEED SELECTION — exploratory, not headline results',fontsize=10,color='#a12222')
      fig.text(.5,.01,f'Harmony seeds {sorted(chosen)} | ±1 SEM (selected runs, not selection-adjusted CI) | No smoothing',ha='center',fontsize=8)
      fig.tight_layout(rect=(0,.04,1,.96))
      for ext in ('png','pdf'):fig.savefig(out/kind/ext/f'{g}.{ext}',dpi=170)
      plt.close(fig)
    fig,axes=plt.subplots(7,4,figsize=(19,23))
    for ax,g in zip(axes.flat,games):draw(ax,g,frame,kind)
    for ax in list(axes.flat)[26:]:ax.set_visible(False)
    handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.98),ncol=3,frameon=False)
    fig.suptitle(f'{kind.title()} curves | POST-HOC BEST 5 OF 6 SEEDS | Exploratory only',y=.996,fontsize=18,color='#a12222')
    fig.text(.5,.951,f'Harmony seeds {sorted(chosen)} selected by final 26-game Mean HNS; excluded {excluded}\nOriginal baselines are not seed-selected | Bands ±1 SEM | Not an unbiased method comparison',ha='center')
    fig.tight_layout(rect=(0,0,1,.93),h_pad=2)
    for ext in ('png','pdf'):fig.savefig(out/kind/f'all_games_overview.{ext}',dpi=170)
    plt.close(fig)
  text=f'''# Harmony: chọn 5 seeds tốt nhất trong 6 — phân tích hậu nghiệm

**Đây là selection-biased exploratory report, không phải kết quả headline hay bằng
chứng cải thiện không chệch.** Giữ nguyên [báo cáo chính seeds 0–4](../harmony_five_seed_results/README.md).

Quy tắc: tính Mean HNS final 100K trên toàn bộ 26 game cho từng seed trong 0–5,
lấy 5 seeds cao nhất; tie-break seed ID tăng dần. **Dùng cùng một tập seed cho tất cả
game/checkpoint**, không chọn 5 seeds khác nhau cho mỗi game.

**Chọn {sorted(chosen)}; loại {excluded}.** Tiêu chí Mean HNS dễ bị chi phối bởi các
game có HNS cực lớn như Up N Down. Không lựa chọn lại tiêu chí sau khi nhìn kết quả.

## Xếp hạng

| Seed | Mean HNS (%) | Giữ |
|---|---:|---|
'''
  for r in ranking.itertuples():text+=f'| {r.training_seed} | {100*r.mean_hns:.2f} | {r.selected} |\n'
  text+='\n## Aggregate đối chiếu\n\n| Bộ seeds | Mean HNS (%) | Median (%) | IQM (%) | Gap | Superhuman |\n|---|---:|---:|---:|---:|---:|\n'
  for name,m in scores.items():text+=f"| {name} | {100*m['mean_hns']:.2f} | {100*m['median_hns']:.2f} | {100*m['iqm_hns']:.2f} | {m['gap']:.4f} | {m['superhuman']}/26 |\n"
  text+='''
Mỗi run lấy mean 100 final eval episodes; HNS dùng baselines.yaml:atari57_gamer,
không clip. Mean/Median qua game means; IQM trim 25% mỗi đuôi trên run×game scores
theo SciPy floor convention; gap = mean(max(1-HNS,0)). Không báo bootstrap CI trên
bộ đã chọn như thể bộ seed được định trước. Dải curves là ±1 SEM của các seeds
được chọn, không bao gồm selection uncertainty; không dùng nó để kiểm định thắng baseline.

## Training curves

![Training](training/all_games_overview.png)

Cả ba dùng training return, bin 5K, trung bình mỗi seed rồi giữa seeds có dữ liệu.
Bin trống giữ gap, không smoothing. Baselines dùng nguyên 5 seeds cũ, không tuyển
chọn; Harmony size25m và CoRe-WM cũ size12m. Vì vậy đây không phải đối chứng công bằng
để khẳng định method vượt baseline.

## Evaluation curves

![Evaluation](evaluation/all_games_overview.png)

Harmony selected và CoRe-WM cũ: isolated eval 10K–100K, 10 episodes ở 10K–90K,
100 ở 100K. Không đưa DreamerV3 training data vào evaluation chart.

## Final scores của bộ được chọn

| Game | Mean ± SD | HNS (%) |
|---|---:|---:|
'''
  for g,r in per.iterrows():text+=f'| {g} | {r.mean_return:.2f} ± {r.sd:.2f} | {100*r.hns:.2f} |\n'
  text+='\n## Hình riêng\n\n| Game | Training PNG/PDF | Eval PNG/PDF |\n|---|---|---|\n'
  for g in games:text+=f'| {g} | [PNG](training/png/{g}.png) / [PDF](training/pdf/{g}.pdf) | [PNG](evaluation/png/{g}.png) / [PDF](evaluation/pdf/{g}.pdf) |\n'
  text+='\n## Nguồn và tái tạo\n\nLưu đầy đủ raw eval returns của cả 6 seeds, final scores, ranking, training episodes và aggregates trong thư mục này, không chỉ lưu seeds được chọn. [Metadata](metadata.json) ghi SHA256 nguồn. Chạy `sbatch scripts/slurm_harmony_selected_seeds.sh`; [script](../../corewm_eval/harmony_selected_seeds.py). Chỉ CPU EDA, không sửa job training đang chạy. Không gộp các seeds >5 đang chạy vào phép chọn này.\n'
  (out/'README.md').write_text(text)
  (out/'metadata.json').write_text(json.dumps(dict(sources=sources,selected=sorted(chosen),excluded=excluded,
      selection='post-hoc top five of seeds 0–5 by final 26-game mean HNS; same seeds for all games',
      biased=True,job=os.environ['SLURM_JOB_ID']),indent=2)+'\n')
  print(ranking.to_string(index=False));print(json.dumps(scores,indent=2));print('PASS: all six full matrices validated; 52 per-game plots plus overviews.')


if __name__=='__main__':main()
