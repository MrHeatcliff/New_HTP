"""Two-curve view of existing best-five-of-six data; no re-selection."""
import hashlib
import json
import os
from pathlib import Path


def main():
  assert os.environ.get('SLURM_JOB_ID'), 'Use CPU Slurm'
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  import numpy as np
  import pandas as pd
  from .config import ATARI100K_GAMES
  source=Path('paper_artifacts/harmony_best5_of6/training_aggregate.csv')
  data=pd.read_csv(source)
  data=data[data.method.isin(['DreamerV3','Harmony'])]
  assert set(data.method)=={'DreamerV3','Harmony'}
  out=Path('paper_artifacts/harmony_best5_vs_dreamerv3')
  for ext in ('png','pdf'):(out/ext).mkdir(parents=True,exist_ok=True)
  data.to_csv(out/'aggregate.csv',index=False)
  plt.rcParams.update({'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
  def draw(ax,game):
    for method,color,label in [('DreamerV3','#3b82f6','DreamerV3 (original 5 seeds)'),
        ('Harmony','#7c3aed','Harmony (selected best 5/6)')]:
      d=data[(data.game==game)&(data.method==method)].set_index('bin').reindex(range(20))
      x=(np.arange(20)+.5)*5000
      ax.plot(x,d['mean'],'.-',color=color,label=label,lw=1.8,markersize=3)
      sem=d['sem'].where(d.seeds>=2)
      ax.fill_between(x,d['mean']-sem,d['mean']+sem,color=color,alpha=.16,lw=0)
    ax.set(title=game.replace('_',' ').title(),xlabel='Agent actions',ylabel='Training episode return',xlim=(0,100000))
    ax.set_xticks([0,25000,50000,75000,100000],['0','25K','50K','75K','100K'])
    ax.grid(alpha=.2);ax.tick_params(labelsize=8)
  for game in ATARI100K_GAMES:
    fig,ax=plt.subplots(figsize=(8,5));draw(ax,game);ax.legend(fontsize=9,frameon=False)
    fig.suptitle('POST-HOC BEST 5 OF 6 — exploratory, not headline results',fontsize=10,color='#a12222')
    fig.text(.5,.01,'Harmony seeds 0,2,3,4,5 | 5K bins | ±1 SEM, not selection-adjusted | No smoothing',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,.96))
    for ext in ('png','pdf'):fig.savefig(out/ext/f'{game}.{ext}',dpi=170)
    plt.close(fig)
  fig,axes=plt.subplots(7,4,figsize=(19,23))
  for ax,g in zip(axes.flat,ATARI100K_GAMES):draw(ax,g)
  for ax in list(axes.flat)[26:]:ax.set_visible(False)
  handles,labels=axes.flat[0].get_legend_handles_labels()
  fig.suptitle('Harmony best 5/6 vs DreamerV3 | Training returns | 26 games',fontsize=19,y=.995)
  fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.98),ncol=2,frameon=False)
  fig.text(.5,.95,'POST-HOC SELECTION: Harmony seeds 0,2,3,4,5 selected using final Mean HNS\nExploratory only; baseline not seed-selected | ±1 SEM | No smoothing',ha='center',color='#a12222')
  fig.tight_layout(rect=(0,0,1,.93),h_pad=2)
  for ext in ('png','pdf'):fig.savefig(out/f'all_games_overview.{ext}',dpi=170)
  plt.close(fig)
  text='''# Harmony best 5/6 vs DreamerV3 — hai đường training return

![Tổng quan](all_games_overview.png)

[PDF tổng quan](all_games_overview.pdf)

Tím: Harmony seeds **0,2,3,4,5**, chọn hậu nghiệm theo final Mean HNS trên 26 game
trong seeds 0–5. Xanh: DreamerV3 nguyên 5 seeds. Đây là **phân tích exploratory có
selection bias**, không thay báo cáo chính hoặc chứng minh thắng baseline.

Cả hai dùng training return: bin 5K actions, mean từng seed rồi giữa seeds có dữ
liệu, dải ±1 SEM (không selection-adjusted CI). Không smoothing, bin trống giữ gap.
Chỉ bỏ đường CoRe-WM khỏi hình, không tính lại/chọn lại seeds hoặc sửa dữ liệu.
[Quy tắc chọn seed và kết quả đủ 6 seeds](../harmony_best5_of6/README.md).

| Game | PNG | PDF |
|---|---|---|
'''
  for g in ATARI100K_GAMES:text+=f'| {g} | [PNG](png/{g}.png) | [PDF](pdf/{g}.pdf) |\n'
  text+='\nTái tạo: `sbatch scripts/slurm_harmony_best_vs_dreamer.sh`. CPU plotting only.\n'
  (out/'README.md').write_text(text)
  (out/'metadata.json').write_text(json.dumps(dict(source=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
      selected_seeds=[0,2,3,4,5],job=os.environ['SLURM_JOB_ID'],post_hoc_selection=True),indent=2)+'\n')
  print('COMPLETE: 26 two-curve PNG/PDF figures and overview.')


if __name__=='__main__':main()
