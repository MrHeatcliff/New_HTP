"""Export isolated evaluation curves; retain baseline measurement provenance."""
import json
from pathlib import Path


def render(root, output):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  from matplotlib.ticker import MultipleLocator
  import pandas as pd
  from .reborn_report import GAMES, LABELS
  output.mkdir(parents=True,exist_ok=True)
  rows=[]
  for game in GAMES:
    data=json.loads((root/game/'summary.json').read_text())
    assert [r['checkpoint'] for r in data] == list(range(10000,100001,10000))
    assert [r['episodes'] for r in data] == [10]*9+[100]
    rows.extend(data)
  (output/'evaluation_results.json').write_text(json.dumps(rows,indent=2)+'\n')
  frame=pd.DataFrame([{k:v for k,v in row.items() if k != 'returns'} for row in rows])
  frame.to_csv(output/'evaluation_summary.csv',index=False)
  reference=Path('paper_artifacts/full_vs_dreamerv3_constraint_suite_learning_curves/aggregate.csv')
  baseline=pd.read_csv(reference)
  baseline=baseline[(baseline.method == 'DreamerV3')&baseline.game.isin(GAMES)]
  baseline.to_csv(output/'dreamerv3_training_reference.csv',index=False)
  def draw(ax,game):
    b=baseline[baseline.game == game].sort_values('agent_steps')
    r=frame[frame.game == game].sort_values('checkpoint')
    ax.plot(b.agent_steps,b['mean'],'--',color='#3B82F6',lw=2,label='DreamerV3: training (5 seeds)')
    ax.fill_between(b.agent_steps,b['mean']-b['sem'],b['mean']+b['sem'],color='#3B82F6',alpha=.16,linewidth=0)
    ax.plot(r.checkpoint,r['mean'],'o-',color='#7C3AED',lw=2.2,markersize=4,label='Reborn: isolated eval (seed 0)')
    ax.set_title(LABELS[game]); ax.set_xlim(0,100000)
    ax.set_xticks([0,25000,50000,75000,100000],['0','25k','50k','75k','100k'])
    ax.xaxis.set_minor_locator(MultipleLocator(5000))
    ax.grid(alpha=.25); ax.grid(which='minor',axis='x',alpha=.08)
    ax.set_xlabel('Environment interactions (agent actions)'); ax.set_ylabel('Episode return')
  for game in GAMES:
    fig,ax=plt.subplots(figsize=(7.2,4.5),constrained_layout=True)
    draw(ax,game); ax.legend(frameon=False,fontsize=8)
    for ext in ('png','pdf'): fig.savefig(output/f'{game}.{ext}',dpi=180)
    plt.close(fig)
  fig,axes=plt.subplots(2,2,figsize=(11,7.5),constrained_layout=True)
  for ax,game in zip(axes.flat,GAMES): draw(ax,game)
  axes[0,0].legend(frameon=False,fontsize=8)
  fig.suptitle('Reborn checkpoint evaluation: 10 episodes at 10K–90K; 100 at 100K\nDashed DreamerV3 reference: training returns ±1 SEM (not matched evaluation)',fontsize=11)
  for ext in ('png','pdf'): fig.savefig(output/f'learning_curves.{ext}',dpi=200)
  plt.close(fig)
  text='''# Reborn: evaluation riêng theo checkpoint

Protocol lấy từ `corewm_eval/full_policy_stage.py`: eval seed 0, một environment,
10 episode tại mỗi checkpoint 10K–90K và 100 episode tại 100K. Bốn game dùng
training seed 0. Load checkpoint bằng source đóng băng của run; không train lại.
SHA256 checkpoint được kiểm tra trước và sau evaluation. Noise Reborn vẫn bật
đúng cấu hình đã train, policy dùng mode eval.

![Checkpoint evaluation](learning_curves.png)

Đường tím là evaluation riêng tại từng checkpoint. Đường xanh đứt là training
returns DreamerV3 5 seed từ dữ liệu W&B đã có trong repo, dải ±1 SEM giữa seed.
Đây chưa phải so sánh hai phương pháp bằng cùng protocol evaluation; không có
checkpoint DreamerV3 tương ứng được sử dụng trong báo cáo này. Lịch evaluation
trên là protocol CoRe-WM đã áp dụng trong repo, không phải tuyên bố mọi thiết
lập đều trùng evaluation chính thức của DreamerV3. Reborn một training seed
không có dải bất định giữa training seed. Không dùng training score để lấp
điểm evaluation.

| Game | Mean final evaluation (100 episodes) |
|---|---:|
'''
  for row in rows:
    if row['checkpoint']==100000: text+=f"| {LABELS[row['game']]} | {row['mean']:,.2f} |\n"
  text+='''
Điểm final này thay thế ước lượng 20 episode trước cho báo cáo mới; dữ liệu cũ
được giữ trong `../reborn_four_game_results/` để truy xuất. JSON chứa từng
episode return và checkpoint hash; CSV chứa đủ 40 điểm evaluation.

Implementation: `corewm_eval/reborn_checkpoint_eval.py` chạy và kiểm tra số
episode/hash; `corewm_eval/reborn_checkpoint_report.py` tổng hợp và vẽ.
'''
  (output/'RESULTS_VI.md').write_text(text)


if __name__ == '__main__':
  import sys
  render(Path(sys.argv[1]), Path(sys.argv[2]))
