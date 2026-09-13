"""Three training-return curves, resolving old runs from evaluated checkpoints."""
import hashlib
import json
import os
from pathlib import Path
import shlex


def main():
  assert os.environ.get('SLURM_JOB_ID'), 'Slurm required'
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  import numpy as np
  import pandas as pd
  from .config import ATARI100K_GAMES
  from .full_vs_dreamerv3_curves import _full_summary_paths, _binned, PRODUCTION_COMMIT
  root = Path('production_runs/corewm_atari100k_v1/evaluations')
  records, sources, seen = [], [], set()
  for summary in _full_summary_paths(root):
    payload = json.loads(summary.read_text())
    game = payload['game']; seed = int(payload['rows'][0]['seed'])
    assert (game,seed) not in seen
    seen.add((game,seed))
    meta = json.loads((summary.parent/'100000/paper_artifacts/run_meta.json').read_text())
    cmd = shlex.split(meta['command'])
    checkpoint = Path(cmd[cmd.index('--run.from_checkpoint')+1])
    path = checkpoint.parent.parent/'paper_artifacts/episode_scores.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    steps = []
    for r in rows:
      assert r['seed'] == seed and r['condition'] == 'full'
      assert r['code_commit'] == PRODUCTION_COMMIT and r['action_repeat'] == 4
      step, score = r['agent_actions'], r['episode_score']
      assert np.isfinite(score)
      steps.append(step)
      if 0 < step <= 100000:
        records.append(dict(method='CoRe-WM old',game=game,seed=seed,agent_steps=step,**{'return':score}))
    assert all(a < b for a,b in zip(steps,steps[1:])), path
    sources.append(dict(game=game,seed=seed,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        evaluated_checkpoint=str(checkpoint)))
  assert seen == {(g,s) for g in ATARI100K_GAMES for s in range(5)}
  episodes = pd.DataFrame(records)
  per_seed, core = _binned(episodes)
  ref = Path('paper_artifacts/harmony_vs_dreamerv3_26_training_curves/aggregate.csv')
  cached = pd.read_csv(ref)
  combined = pd.concat([cached,core],ignore_index=True)
  out = Path('paper_artifacts/harmony_corewm_dreamerv3_26_training_curves')
  for ext in ('png','pdf'): (out/ext).mkdir(parents=True,exist_ok=True)
  combined.to_csv(out/'aggregate.csv',index=False)
  episodes.to_csv(out/'corewm_training_episodes.csv',index=False)
  per_seed.to_csv(out/'corewm_per_seed.csv',index=False)
  plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
  def title(g): return {'up_n_down':'Up N Down','ms_pacman':'Ms. Pac-Man','jamesbond':'James Bond'}.get(g,g.replace('_',' ').title())
  styles=[('DreamerV3','#3B82F6','DreamerV3 (5 seeds)'),('CoRe-WM old','#E09B24','CoRe-WM old (5 seeds)'),
          ('Reborn + Harmony','#7C3AED','Harmony new (seed 0)')]
  def draw(ax,game):
    for method,color,label in styles:
      d=combined[(combined.game==game)&(combined.method==method)].set_index('bin').reindex(range(20))
      x=(np.arange(20)+.5)*5000
      ax.plot(x,d['mean'],color=color,label=label,lw=1.8,marker='.',markersize=3)
      if method!='Reborn + Harmony':
        sem=d['sem'].where(d.seeds>=2)
        ax.fill_between(x,d['mean']-sem,d['mean']+sem,color=color,alpha=.15,lw=0)
    ax.set(title=title(game),xlim=(0,100000),xlabel='Agent actions',ylabel='Training episode return')
    ax.set_xticks([0,25000,50000,75000,100000],['0','25K','50K','75K','100K'])
    ax.grid(alpha=.2);ax.tick_params(labelsize=8)
  for game in ATARI100K_GAMES:
    fig,ax=plt.subplots(figsize=(8,5));draw(ax,game);ax.legend(frameon=False,fontsize=9)
    fig.text(.5,.015,'All methods: training returns | 5K bins | Bands: ±1 SEM | Empty bins remain gaps | No smoothing',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,1))
    for ext in ('png','pdf'):fig.savefig(out/ext/f'{game}.{ext}',dpi=180)
    plt.close(fig)
  fig,axes=plt.subplots(7,4,figsize=(19,23))
  for ax,game in zip(axes.flat,ATARI100K_GAMES):draw(ax,game)
  for ax in list(axes.flat)[26:]:ax.set_visible(False)
  handles,labels=axes.flat[0].get_legend_handles_labels()
  fig.suptitle('Atari 100K · Training returns · Harmony / CoRe-WM / DreamerV3',fontsize=19,y=.995)
  fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.98),ncol=3,frameon=False)
  fig.text(.5,.955,'Same measurement: training episode return · 5K-action bins · Bands: ±1 SEM across available seeds\nHarmony: seed 0 only · Empty bins remain gaps · No evaluation substitution · No smoothing',ha='center',fontsize=11)
  fig.tight_layout(rect=(0,0,1,.935),h_pad=2,w_pad=1.6)
  for ext in ('png','pdf'):fig.savefig(out/f'all_games_overview.{ext}',dpi=180)
  plt.close(fig)
  (out/'metadata.json').write_text(json.dumps(dict(job=os.environ['SLURM_JOB_ID'],corewm_sources=sources,
      cached_harmony_dreamer=str(ref),cached_sha256=hashlib.sha256(ref.read_bytes()).hexdigest(),
      measurement='training episode returns only',corewm_runs=130,corewm_episodes=len(records),
      bin_width=5000,band='±SEM across available seeds; omitted with fewer than two seeds',
      missing_bins='NaN, no interpolation',new_harmony_seeds_excluded=[1,2,3,4]),indent=2)+'\n')
  text='''# Cả ba phương pháp theo training return — 26 game

![Tổng quan](all_games_overview.png)

[PDF tổng quan](all_games_overview.pdf)

**Cả ba đường đều dùng training episode return, không dùng evaluation.**
Xanh: DreamerV3 5 seeds; cam: CoRe-WM cũ 5 seeds; tím: Reborn + Harmony seed 0.

Trung bình episode trong bin 5K actions của từng seed, sau đó trung bình giữa seeds
có dữ liệu. Dải ±1 SEM; bỏ dải khi bin có ít hơn 2 seeds. Cột `seeds` trong CSV ghi
số seed đóng góp thực tế. Bin không có episode kết thúc để trống, không nội suy;
marker giúp nhìn thấy các điểm cô lập. Không smoothing. Bin cuối ở 97.5K gồm
episode kết thúc trong [95K,100K]; không phải final evaluation tại 100K.

CoRe-WM cũ được lấy từ đúng training attempt mà final evaluation trước đây nạp
checkpoint, không chọn attempt có score tốt nhất. Đã kiểm tra đủ 26 × 5 runs,
seed/condition/production commit/action repeat và thời điểm episode tăng nghiêm ngặt.
Alien seed 0 dùng attempt_002, đúng nguồn checkpoint đã evaluation.
CoRe-WM và Harmony dùng agent_actions trong log gốc, không chia logger step ước lượng.
DreamerV3 dùng dữ liệu training đã lưu của hình trước.

Cùng loại metric và binning không đồng nghĩa mọi cấu hình đều matched:
CoRe-WM cũ dùng size12m; Harmony mới dùng size25m, số seeds cũng khác nhau.
Không dùng hình để quy cải thiện duy nhất cho objective. Các seed Harmony 1–4 đang
chạy chưa được ghép vào. Episode dài của Up N Down có thể tạo khoảng trống hợp lệ.

| Game | PNG | PDF |
|---|---|---|
'''
  for game in ATARI100K_GAMES:text+=f'| {title(game)} | [PNG](png/{game}.png) | [PDF](pdf/{game}.pdf) |\n'
  text+='\n[Aggregate](aggregate.csv) · [CoRe-WM episodes](corewm_training_episodes.csv) · [Per-seed bins](corewm_per_seed.csv) · [Provenance](metadata.json)\n\nTái tạo: `sbatch scripts/slurm_three_method_training_curves.sh`. CPU Slurm, không chạm các job training.\n'
  (out/'README.md').write_text(text)
  print(f'PASS: 130 CoRe-WM training runs, {len(records)} episodes; 26 three-method figures generated.')


if __name__=='__main__':main()
