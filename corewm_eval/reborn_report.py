"""Aggregate the completed Reborn screening into reviewable tables and plots."""
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

GAMES = ('boxing', 'up_n_down', 'frostbite', 'road_runner')
LABELS = {'boxing': 'Boxing', 'up_n_down': 'Up N Down',
          'frostbite': 'Frostbite', 'road_runner': 'Road Runner'}


def bootstrap_mean(values, seed):
  values = np.asarray(values, np.float64)
  rng = np.random.default_rng(seed)
  samples = values[rng.integers(0, len(values), (20000, len(values)))].mean(1)
  return [float(x) for x in np.quantile(samples, (.025, .975))]


def episode_curve(path, width=10000):
  rows = [json.loads(x) for x in Path(path).read_text().splitlines()]
  points = [(float(x['step']) / 4, float(x['episode/score']))
            for x in rows if 'episode/score' in x]
  bins = []
  for end in range(width, 100001, width):
    vals = [score for step, score in points if end-width < step <= end]
    bins.append((end, float(np.mean(vals)) if vals else None, len(vals)))
  return bins


def ratio(a, b):
  return float(a / b) if b else None


def main(run_root, prior_summary, output):
  root, output = Path(run_root), Path(output)
  output.mkdir(parents=True, exist_ok=True)
  prior = json.loads(Path(prior_summary).read_text())
  result = {'job': 3846, 'run_root': str(root.resolve()), 'games': {},
      'protocol': {'training_seed': 0, 'evaluation_seed': 0,
                   'actions': 100000, 'evaluation_episodes': 20,
                   'diagnostic_policy_episodes': 30}}
  for index, game in enumerate(GAMES):
    control = json.loads((root/game/'control.json').read_text())
    diag = json.loads((root/game/'diagnostic/diagnostics.json').read_text())
    calibration = {key: np.mean([x[key] for x in diag['calibration']], 0).tolist()
                   for key in ('sf_mse','sf_zero_mse','q_mae','q_zero_mae')}
    mu = np.asarray([x['r2_vs_train_mean'] for x in diag['probes_mu']])
    sampled = np.asarray([x['r2_vs_train_mean'] for x in diag['probes_sampled_z']])
    probe = {}
    for name, values in [('mu', mu), ('sampled_z', sampled)]:
      probe[name] = {
          'detail_mean_r2': values[:, :160].reshape(6,5,32).mean(2).tolist(),
          'sf_mean_r2': values[:, 160:320].reshape(6,5,32).mean(2).tolist(),
          'q_mean_r2': values[:, 320:325].tolist()}
    old = float(prior[game]['means']['full'])
    result['games'][game] = {'evaluation_mean': control['mean'],
        'evaluation_ci95_episode_bootstrap': bootstrap_mean(control['scores'], 3846+index),
        'evaluation_scores': control['scores'], 'prior_constrained_full_mean': old,
        'reborn_minus_prior': control['mean']-old,
        'reborn_over_prior': ratio(control['mean'], old),
        'anchors': sum(x['anchors'] for x in diag['episodes']),
        'mu_mean_offdiagonal_cka': float((np.asarray(diag['mu_cka']).sum()-5)/20),
        'mu_block_variance': diag['mu_block_variance'],
        'mu_participation_rank': diag['mu_participation_rank'],
        'band_mse': np.mean(diag['band_mse'],0).tolist(),
        'band_removed_mse': np.mean(diag['band_removed_mse'],0).tolist(),
        'calibration': calibration, 'probes': probe,
        'learning_curve': episode_curve(root/game/'full/metrics.jsonl'),
        'prior_learning_curve': episode_curve(
            Path(prior_summary).parent/game/'full/metrics.jsonl')}
  (output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
  write_csv(result, output/'summary.csv')
  plot_control(result, output)
  plot_learning(result, output)
  plot_representation(result, output)
  plot_calibration(result, output)
  plot_cka(result, root, output)
  write_markdown(result, output/'RESULTS_VI.md')


def save(fig, output, stem):
  fig.savefig(output/f'{stem}.png', dpi=220, bbox_inches='tight')
  fig.savefig(output/f'{stem}.pdf', bbox_inches='tight')
  plt.close(fig)


def write_csv(result, path):
  with path.open('w', newline='') as stream:
    writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(['game','reborn_mean','ci95_low','ci95_high',
                     'prior_constrained_full','difference','ratio','anchors','mu_offdiag_cka'])
    for game,row in result['games'].items():
      writer.writerow([game,row['evaluation_mean'],*row['evaluation_ci95_episode_bootstrap'],
          row['prior_constrained_full_mean'],row['reborn_minus_prior'],row['reborn_over_prior'],
          row['anchors'],row['mu_mean_offdiagonal_cka']])


def plot_control(result, output):
  fig, axes = plt.subplots(1,4,figsize=(13,3.2),constrained_layout=True)
  for ax,(game,row) in zip(axes,result['games'].items()):
    vals = [row['prior_constrained_full_mean'],row['evaluation_mean']]
    lo,hi = row['evaluation_ci95_episode_bootstrap']
    ax.bar([0,1],vals,color=['#9ca3af','#2563eb'],width=.66)
    ax.errorbar(1,vals[1],yerr=[[vals[1]-lo],[hi-vals[1]]],fmt='none',color='#111827',capsize=4)
    ax.set_xticks([0,1],['Prior\nconstraint','Reborn'])
    ax.set_title(LABELS[game]); ax.grid(axis='y',alpha=.22)
    ax.text(1,vals[1],f' {vals[1]:,.1f}',ha='center',va='bottom',fontsize=8)
  fig.suptitle('Final evaluation at 100K actions (one training seed)')
  save(fig,output,'control_comparison')


def plot_learning(result, output):
  fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
  for ax,(game,row) in zip(axes.flat,result['games'].items()):
    for key,label,color in [('prior_learning_curve','Prior constraint','#9ca3af'),
                            ('learning_curve','Reborn','#2563eb')]:
      curve=np.asarray(row[key],float)
      ax.plot(curve[:,0]/1000,curve[:,1],marker='o',label=label,color=color,lw=2)
    ax.set_title(LABELS[game]); ax.set_xlabel('Environment actions (K)')
    ax.set_ylabel('Mean training episode return / 10K bin'); ax.grid(alpha=.22)
  axes[0,0].legend(frameon=False)
  fig.suptitle('Learning behavior (seed 0; episode means are not fixed-policy evaluation)')
  save(fig,output,'learning_curves')


def plot_representation(result, output):
  fig,axes=plt.subplots(2,4,figsize=(14,6.5),constrained_layout=True)
  prefixes=np.arange(6)
  for col,(game,row) in enumerate(result['games'].items()):
    for name,color,style in [('mu','#2563eb','-'),('sampled_z','#dc2626','--')]:
      probe=row['probes'][name]
      axes[0,col].plot(prefixes,np.mean(probe['sf_mean_r2'],1),style,color=color,
          marker='o',label=name)
      axes[1,col].plot(prefixes,np.mean(probe['q_mean_r2'],1),style,color=color,
          marker='o',label=name)
    axes[0,col].set_title(LABELS[game]); axes[0,col].set_ylabel('Same-target SF probe R²')
    axes[1,col].set_ylabel('Same-target Q probe R²'); axes[1,col].set_xlabel('Available prefix')
    for ax in axes[:,col]:
      ax.axhline(0,color='#111827',lw=.7); ax.grid(alpha=.2)
      ax.set_xticks(prefixes,['Action','P1','P2','P3','P4','P5'])
  axes[0,0].legend(frameon=False)
  fig.suptitle('Incremental utility: each prefix is evaluated on identical targets')
  save(fig,output,'incremental_probe_utility')


def plot_calibration(result, output):
  fig,axes=plt.subplots(2,4,figsize=(14,6),constrained_layout=True)
  for col,(game,row) in enumerate(result['games'].items()):
    cal=row['calibration']; levels=np.arange(1,6)
    sf=np.asarray(cal['sf_mse'])/np.maximum(cal['sf_zero_mse'],1e-12)
    q=np.asarray(cal['q_mae'])/np.maximum(cal['q_zero_mae'],1e-12)
    axes[0,col].plot(levels,sf,marker='o',color='#059669')
    axes[1,col].plot(levels,q,marker='o',color='#7c3aed')
    axes[0,col].set_title(LABELS[game]); axes[0,col].set_ylabel('SF MSE / zero MSE')
    axes[1,col].set_ylabel('Q MAE / zero MAE'); axes[1,col].set_xlabel('Level')
    for ax in axes[:,col]:
      ax.axhline(1,color='#dc2626',ls='--',lw=1); ax.grid(alpha=.2); ax.set_xticks(levels)
  fig.suptitle('Frozen-policy rollout calibration (below 1 beats zero predictor)')
  save(fig,output,'rollout_calibration')


def plot_cka(result, root, output):
  fig,axes=plt.subplots(1,4,figsize=(13,3),constrained_layout=True)
  for ax,game in zip(axes,GAMES):
    diag=json.loads((root/game/'diagnostic/diagnostics.json').read_text())
    im=ax.imshow(diag['mu_cka'],vmin=0,vmax=1,cmap='viridis')
    ax.set_title(LABELS[game]); ax.set_xticks(range(5),range(1,6)); ax.set_yticks(range(5),range(1,6))
  fig.colorbar(im,ax=axes,label='Linear CKA on μ')
  fig.suptitle('Block similarity without injected noise')
  save(fig,output,'mu_block_cka')


def write_markdown(result, path):
  rows=[]
  for game,x in result['games'].items():
    lo,hi=x['evaluation_ci95_episode_bootstrap']
    rows.append(f"| {LABELS[game]} | {x['evaluation_mean']:,.1f} | [{lo:,.1f}, {hi:,.1f}] | "
                f"{x['prior_constrained_full_mean']:,.1f} | {x['reborn_minus_prior']:+,.1f} |")
  text='''# Kết quả screening Reborn trên bốn game Atari 100K

Job 3846 hoàn thành thành công trên hai H200 trong 1 giờ 09 phút. Mỗi game được
train từ đầu với seed 0 và 100.000 environment actions, sau đó chạy 20 episode
evaluation seed 0. Khoảng tin cậy dưới đây bootstrap **episode evaluation**, nên
không đại diện cho bất định giữa các training seed.

| Game | Reborn | 95% CI episode | constrained-full cũ | Chênh lệch |
|---|---:|---:|---:|---:|
'''+"\n".join(rows)+'''

![Control comparison](control_comparison.png)

Kết quả control không cho thấy cải thiện đồng đều. Up N Down tăng 6.804 điểm
so với constrained-full screening cùng seed/budget. Boxing giảm 32,15 điểm,
Road Runner giảm 4.030 điểm, và Frostbite giảm 1.778,5 điểm. Hai run dùng cùng
100K action và seed 0, nhưng kiến trúc, số parameter và objective khác nhau;
đây là comparison định hướng, chưa phải paired causal estimate.

![Learning curves](learning_curves.png)

Reborn có 49,36M trainable parameters so với 40,60M của constrained CoRe-WM.
Vì vậy kết quả không parameter-matched. Training episode curve mô tả hành vi
trong lúc policy thay đổi và chỉ dùng để tìm thời điểm học; final evaluation
mới là control endpoint chính.

## Representation và predictive outcomes

![Incremental probes](incremental_probe_utility.png)

Probe dùng split theo episode và chấm mọi prefix trên cùng target. Frostbite và
Road Runner có incremental utility rõ trên cả mean code và sampled code.
Up N Down có tín hiệu tốt trên mean code nhưng noise làm giảm mạnh lợi ích của
prefix lớn. Boxing yếu nhất: sampled-code probe nhìn chung không hơn action-only.
Điều này cho thấy `mu` không collapse nhưng fixed unit noise có thể làm giảm
usable information khi signal-to-noise ratio thấp.

![Rollout calibration](rollout_calibration.png)

SF head thắng zero predictor rất rõ trên cả bốn game. Q head thắng zero rõ ở
Frostbite và Road Runner; ở Boxing và Up N Down, các horizon ngắn/prefix lớn có
tỉ lệ lỗi lớn hơn 1. TD objective vì vậy chưa tạo Q calibration đồng đều. Đây
là failure cụ thể cần tách bằng ablation `no_q`, scale Q và stochastic noise.

![CKA](mu_block_cka.png)

Mean off-diagonal CKA vẫn cao: Boxing 0,881; Up N Down 0,828; Frostbite 0,784;
Road Runner 0,760. Fixed Haar target và cumulative rate chưa đủ tạo block độc
lập. Tuy vậy CKA cao không phủ định incremental utility: Frostbite/Road Runner
vẫn cải thiện probe khi thêm prefix. Tiêu chí chính là same-target utility,
không phải ép CKA về zero.

## Chẩn đoán giả thuyết

- H1 được ủng hộ một phần: detail reconstruction và same-target utility rõ ở
  Frostbite/Road Runner, yếu hoặc bị noise phá ở Boxing/Up N Down.
- H2 được ủng hộ cho SF; Q chỉ được ủng hộ ở hai game. Bootstrap TD loss nhỏ
  không đủ dự đoán rollout calibration.
- H3: không thấy collapse theo dead-coordinate/variance của `mu`, nhưng effective
  rank chỉ khoảng 4–10 và sampled probes cho thấy bottleneck có thể quá mạnh.
- H4 bị bác bỏ ở screening hiện tại: control giảm ở 3/4 game, dù representation
  metrics ở Frostbite rất tốt. Representation target fit chưa đủ bảo đảm control.

Failure ưu tiên tiếp theo là đường stochastic code và Q grounding. Vòng ablation
hợp lý nhất giữ mọi thứ cố định và thay một yếu tố: `deterministic` so với full
trên Boxing/Frostbite, vì hai game phân biệt được trường hợp sampled utility yếu
và trường hợp representation tốt nhưng control xấu. Sau đó mới xét `no_q` hoặc
giảm kích thước head để tách gradient conflict và parameter-count confound.

## Artifact và giới hạn

`summary.json` chứa toàn bộ score, CI, probe, calibration và learning-curve bins;
`summary.csv` là bảng compact. Diagnostic dùng 30 frozen-policy clips, horizon
128 và episode-held-out ridge probes. Clip bắt đầu từ reset và episode variation
không thay thế nhiều training seeds. Một outlier thấp xuất hiện trong Road Runner
evaluation, nên cần thêm seed trước khi kết luận ranking.
'''
  path.write_text(text)


if __name__ == '__main__':
  import sys
  main(*sys.argv[1:])
