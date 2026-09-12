"""Validated four-arm results, learning curves, and shared-state actor audits."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ARMS=('full','low_rate','no_q','deterministic')
GAMES=('boxing','up_n_down','frostbite','road_runner')
LABELS=('Full','Lower rate','No auxiliary Q','Deterministic')
COLORS=('#64748b','#7c3aed','#ea580c','#059669')


def main(root,out):
  root,out=Path(root),Path(out); out.mkdir(parents=True,exist_ok=True)
  assert json.loads((root/'status.json').read_text())['status']=='COMPLETE'
  rows=[]; records=[]; audits={}; diagnostics={}
  rng=np.random.default_rng(3854)
  for arm in ARMS:
    audits[arm]={}; diagnostics[arm]={}
    for game in GAMES:
      assert json.loads((root/arm/game/'stage.json').read_text())['status']=='COMPLETE'
      data=json.loads((root/arm/'evaluation'/game/'summary.json').read_text())
      assert [r['checkpoint'] for r in data]==list(range(10000,100001,10000))
      assert [r['episodes'] for r in data]==[10]*9+[100]
      for row in data:
        v=np.asarray(row['returns']); assert len(v)==row['episodes'] and np.isfinite(v).all()
        assert np.isclose(v.mean(),row['mean']) and len(row['checkpoint_hash'])==64
        records.append(dict(arm=arm,**row))
      v=np.asarray(data[-1]['returns'])
      ci=np.quantile(v[rng.integers(0,len(v),(10000,len(v)))].mean(1),[.025,.975])
      auc=sum((a['mean']+b['mean'])*.5*(b['checkpoint']-a['checkpoint']) for a,b in zip(data,data[1:]))/90000
      rows.append(dict(arm=arm,game=game,final_mean=float(v.mean()),median=float(np.median(v)),
          ci_low=float(ci[0]),ci_high=float(ci[1]),min=float(v.min()),max=float(v.max()),auc_mean=auc))
      folder='actor_audit_own' if arm=='full' else 'actor_audit_shared'
      audits[arm][game]=json.loads((root/arm/game/folder/'audit.json').read_text())
      assert audits[arm][game]['status']=='COMPLETE'
      diagnostics[arm][game]=json.loads((root/arm/game/'diagnostic/diagnostics.json').read_text())
      assert diagnostics[arm][game]['status']=='COMPLETE'
  frame=pd.DataFrame(rows); frame.to_csv(out/'summary.csv',index=False)
  (out/'evaluation_results.json').write_text(json.dumps(records,indent=2)+'\n')
  (out/'shared_state_audits.json').write_text(json.dumps(audits,indent=2)+'\n')
  (out/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
  (out/'provenance.json').write_text(json.dumps(dict(run_root=str(root.resolve()),job=3854,
      manifest=json.loads((root/'manifest.json').read_text()),
      source_sha256=json.loads((root/'source_sha256.json').read_text()),
      tests=(root/'tests.log').read_text(),
      uncertainty='Final-score bootstrap CI over evaluation episodes, not training seeds.'),indent=2)+'\n')
  def save(fig,name):
    for ext in ('png','pdf'):fig.savefig(out/f'{name}.{ext}',dpi=200,bbox_inches='tight')
    plt.close(fig)
  def curve(ax,g):
    for arm,label,color in zip(ARMS,LABELS,COLORS):
      v=[r for r in records if r['arm']==arm and r['game']==g]
      ax.plot([r['checkpoint']/1000 for r in v],[r['mean'] for r in v],'o-',color=color,label=label,ms=3,lw=2)
    ax.set_title(g.replace('_',' ').title());ax.set_xlabel('Environment actions (K)')
    ax.set_ylabel('Evaluation return');ax.set_xlim(0,100);ax.grid(alpha=.22)
  fig,axs=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
  for ax,g in zip(axs.flat,GAMES):curve(ax,g)
  axs[0,0].legend(frameon=False,fontsize=8)
  fig.suptitle('Isolated checkpoint evaluation — one training seed\n10 episodes at 10K–90K; 100 at 100K',fontsize=12)
  save(fig,'learning_curves')
  for g in GAMES:
    fig,ax=plt.subplots(figsize=(7.2,4.5),constrained_layout=True);curve(ax,g);ax.legend(frameon=False);save(fig,g)
  fig,axs=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
  for ax,g in zip(axs.flat,GAMES):
    v=frame[frame.game==g].set_index('arm').loc[list(ARMS)]
    ax.bar(range(4),v.final_mean,color=COLORS)
    ax.errorbar(range(4),v.final_mean,yerr=[v.final_mean-v.ci_low,v.ci_high-v.final_mean],fmt='none',color='black',capsize=3)
    ax.set_xticks(range(4),LABELS,rotation=15,fontsize=8);ax.set_title(g.replace('_',' ').title());ax.grid(axis='y',alpha=.2)
  fig.suptitle('Final evaluation: 100 episodes; CI reflects episode variation only')
  save(fig,'final_evaluation')
  fig,axs=plt.subplots(2,4,figsize=(15,6.5),constrained_layout=True)
  for j,g in enumerate(GAMES):
    for i,metric in enumerate(('actor_tv','value_abs_change')):
      v=[np.mean([r['interventions']['resampled_noise'][metric] for r in audits[a][g]['rows']]) for a in ARMS]
      axs[i,j].bar(range(4),v,color=COLORS);axs[i,j].set_xticks(range(4),LABELS,rotation=30,fontsize=7)
      axs[i,j].set_ylabel(metric);axs[i,j].grid(axis='y',alpha=.2)
    axs[0,j].set_title(g.replace('_',' ').title())
  fig.suptitle('Noise sensitivity on shared Full trajectories\nDeterministic zero is structural; these are local interventions, not control scores',fontsize=11)
  save(fig,'noise_sensitivity')
  fig,axs=plt.subplots(2,4,figsize=(16,7),constrained_layout=True)
  branches=audits['full']['boxing']['branches']
  for j,g in enumerate(GAMES):
    norms=np.array([[np.mean([r['gradients']['reborn_code']['norm'][k] for r in audits[a][g]['rows']]) for k in range(9)] for a in ARMS])
    cos=np.array([[np.mean([r['gradients']['reborn_code']['cosine_with_actor_critic'][k] for r in audits[a][g]['rows'] if r['gradients']['reborn_code']['cosine_with_actor_critic'][k] is not None])
          if any(r['gradients']['reborn_code']['cosine_with_actor_critic'][k] is not None for r in audits[a][g]['rows']) else np.nan for k in range(4)] for a in ARMS])
    im=axs[0,j].imshow(np.ma.masked_where(norms<=0,np.log10(np.maximum(norms,1e-12))),aspect='auto',vmin=-5,vmax=1,cmap='viridis')
    im2=axs[1,j].imshow(np.ma.masked_invalid(cos),aspect='auto',vmin=-.25,vmax=.25,cmap='coolwarm')
    axs[0,j].set_xticks(range(9),branches,rotation=70,fontsize=7);axs[1,j].set_xticks(range(4),branches[:4])
    for i in range(2):axs[i,j].set_yticks(range(4),LABELS,fontsize=8)
    axs[0,j].set_title(g.replace('_',' ').title())
  fig.colorbar(im,ax=list(axs[0]),label='log10 weighted gradient norm (zero masked)',shrink=.7)
  fig.colorbar(im2,ax=list(axs[1]),label='Cosine with actor + critics (undefined masked)',shrink=.7)
  fig.suptitle('Projection gradients on shared states, pre-AGC / pre-Adam; local diagnostic')
  save(fig,'gradient_audit')
  text='''# Reborn: kết quả hoàn tất bốn ablation

Job 3854 hoàn tất: 16 run × 100K actions, 160 điểm checkpoint evaluation,
3.040 episode evaluation. Mỗi arm chỉ có training seed 0. Final score dùng
100 episode; 10K–90K dùng 10 episode/checkpoint. CI bootstrap episode không
phải bất định giữa training seed. Tất cả stage diagnostic/audit đã hoàn tất.

| Game | Full | Giảm beta 10× | Bỏ Q | Bỏ noise |
|---|---:|---:|---:|---:|
'''
  for g in GAMES:text+='| '+g+' | '+' | '.join(f'{frame[(frame.arm==a)&(frame.game==g)].iloc[0].final_mean:,.2f}' for a in ARMS)+' |\n'
  text+='\n![Learning curves](learning_curves.png)\n\n![Final evaluation](final_evaluation.png)\n\n'
  text+='''## Up N Down: kết quả cuối và cập nhật giả thuyết

Deterministic đạt **38.945,7**, median **37.805**, range **7.240–82.270** trên
100 episode: tốt hơn Full **5.424,5** khoảng **7,18×**, nhưng thấp hơn low_rate
**175.267,8** khoảng **4,50×**. Low_rate median **186.550**, range
**55.040–251.020**: lợi ích không đến từ một outlier duy nhất.

Deterministic đã lên **103.929 ở 30K**, **142.337 ở 60K**, rồi tụt **10.722 ở
80K**. Vì vậy bỏ noise giúp policy học sớm nhưng chưa xử lý ổn định theo thời
gian; các điểm trung gian chỉ có 10 episode nên vẫn chứa evaluation variance.
Low_rate thắng final rõ trong run này; cần so AUC và nhiều training seed trước
khi chọn cấu hình. AUC trung bình 10K–100K lưu trong summary.csv, không ngoại
suy về 0 actions và không gọi đây là HNS AUC.

AUC trung bình Up N Down: Full **3.157,7**, low_rate **50.817,4**, no_Q
**3.492,5**, deterministic **59.306,5**. Bỏ noise tốt hơn low_rate về diện tích
đường học trong lần chạy này, dù thấp hơn rõ ở final: xếp hạng phụ thuộc vào
sample efficiency hay endpoint. Đây không phải bằng chứng low_rate thống trị
toàn bộ quá trình học.

Auxiliary Q rollout MAE/zero-MAE ở deterministic là khoảng
**[0,554; 0,713; 0,953; 1,211; 1,363]**. Ba horizon dài tốt hơn zero, hai horizon
ngắn vẫn kém zero theo MAE. Bỏ noise chưa tự sửa Q calibration. Không suy
từ zero noise sensitivity ở deterministic thành bằng chứng control tốt: giá
trị đó bằng zero theo kiến trúc.

## Kết luận theo giả thuyết

- **Rate quá mạnh:** được ủng hộ ở Up N Down và một phần Road Runner. Boxing
  giảm final dù học tốt hơn ở một số checkpoint; Frostbite học chậm hơn. Giảm
  rate không phải cải thiện phổ quát. Mu/probe trên trajectory riêng không
  cùng phân phối nên không dùng để chứng minh SNR tăng trên mọi game.
- **Q gây hại chung:** không được ủng hộ. No-Q thua Full ở Boxing, Up N Down,
  Frostbite, chỉ thắng Road Runner. Gradient Q không có cosine âm nhất quán
  với actor–critic. Nên giữ grounding reward, kiểm tra trọng số thay vì bỏ Q
  toàn bộ. Q-head không được supervise ở no-Q nên không chấm nó như head học.
- **Noise gây khó control:** được ủng hộ mạnh nhất ở Boxing (42,98 → 70,75)
  và được hỗ trợ bởi Up N Down/Road Runner. Frostbite gần như không thay đổi
  final. Deterministic bỏ diễn giải stochastic information bound; đây là
  ablation chẩn đoán, không tự là phương pháp cuối.

![Noise sensitivity](noise_sensitivity.png)

Giảm beta hạ actor TV khi resample noise trên cùng Full trajectories:
Boxing 0,249 → 0,232; Up N Down 0,170 → 0,107; Frostbite 0,129 → 0,050;
Road Runner 0,135 → 0,070. Actor ít nhạy noise hơn chưa đủ đảm bảo score tăng.

![Gradient audit](gradient_audit.png)

Audit dùng cùng ba episode prefix (tối đa 64 state), hai PRNG replicate;
norm/cosine trước AGC/Adam, không phải optimizer update thực tế. Gradient
reward/critic trên projection có thể lớn hơn Q. Reconstruction gradient nhỏ
ở checkpoint cuối không chứng minh nó vô ích lúc đầu. Coarse-to-fine là tính
chất output Haar; chưa cô lập nhân quả đóng góp reconstruction trong vòng này.

## Giới hạn và hướng nghiên cứu tiếp

Full chạy lại khác lần Reborn đầu ở ba game dù cùng seed; Boxing tái hiện
đúng. So sánh source model chỉ thấy bổ sung metric trong hai file Reborn,
nhưng nguyên nhân phân kỳ chưa xác định. Không kết luận rằng instrumentation
hoàn toàn vô ảnh hưởng tới số học/scheduling. Cần audit reproducibility và
thêm training seed cho hướng low_rate/deterministic có lợi.

Critic complete-return calibration chỉ hợp lệ trên clip terminal thật, không
dùng return cắt ngắn như target chính xác. Coverage khác nhau giữa policy;
không so raw critic MAE giữa game hoặc phân phối khác như một causal metric.
Shared-state audit chỉ đo gradient và reliance, không on-policy calibration.

Ưu tiên tiếp: xác nhận rate–noise tradeoff và policy degradation cuối training,
giữ Q làm mặc định; không thêm regularizer mới chỉ vì CKA hoặc probe chưa đẹp.

## Dữ liệu và implementation

`evaluation_results.json`: từng episode return, checkpoint hash, seed và arm.
`summary.csv`: final mean/median, episode CI, range và normalized-by-duration AUC.
`shared_state_audits.json`: gradient/intervention đầy đủ trên shared states.
`diagnostics.json`: probes, CKA, calibration, episode coverage cho từng arm/game.
`provenance.json`: source SHA256, manifest job, test output (9 passed).

Protocol: [REBORN_HYPOTHESIS_SUITE_VI.md](../persistence_research/REBORN_HYPOTHESIS_SUITE_VI.md).
Runner: `corewm_eval/reborn_hypothesis_suite.py`; audit:
`corewm_eval/reborn_actor_audit.py`; report: `corewm_eval/reborn_hypothesis_report.py`.
'''
  (out/'RESULTS_VI.md').write_text(text)
  print(frame.to_string(index=False))


if __name__=='__main__':main(*sys.argv[1:])
