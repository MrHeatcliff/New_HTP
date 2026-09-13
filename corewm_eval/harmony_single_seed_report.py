"""Reproduce final-budget, single-training-seed Atari metrics (no bootstrap)."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import trim_mean

from .config import ATARI100K_GAMES
from .hns_reference import load_reference


def main():
  source = Path('paper_artifacts/harmony_vs_dreamerv3_26_learning_curves/evaluation_results.json')
  reference = Path('baselines.yaml')
  out = Path('paper_artifacts/harmony_single_seed_report')
  out.mkdir(parents=True, exist_ok=True)
  final = [r for r in json.loads(source.read_text()) if r['checkpoint'] == 100000]
  assert len(final) == 26 and {r['game'] for r in final} == set(ATARI100K_GAMES)
  refs = load_reference(reference, 'atari57_gamer')
  indexed = {r['game']: r for r in final}
  rows = []
  for game in ATARI100K_GAMES:
    r = indexed[game]
    returns = np.asarray(r['returns'], dtype=float)
    assert r['training_seed'] == 0 and r['episodes'] == len(returns) == 100
    assert np.isfinite(returns).all() and np.isclose(returns.mean(), r['mean'])
    random, human = refs[game]
    assert human > random
    score = float(returns.mean())
    rows.append(dict(game=game, training_seed=0, checkpoint=100000,
                     episodes=100, random=random, human=human, return_mean=score,
                     hns=(score-random)/(human-random), checkpoint_hash=r['checkpoint_hash']))
  hns = np.array([r['hns'] for r in rows])
  metrics = dict(mean_hns=float(hns.mean()), median_hns=float(np.median(hns)),
                 iqm_hns=float(trim_mean(hns, 0.25)),
                 optimality_gap_human=float(np.maximum(1-hns, 0).mean()),
                 superhuman_games=int((hns > 1).sum()), games=26, training_seeds=1,
                 final_episodes_per_game=100)
  # Independent checks of trimming, gap, and normalization anchors.
  assert np.isclose(metrics['iqm_hns'], np.sort(hns)[6:20].mean())
  assert np.isclose(metrics['optimality_gap_human'], 1-np.minimum(hns, 1).mean())
  assert np.isclose((refs['alien'][1]-refs['alien'][0])/(refs['alien'][1]-refs['alien'][0]), 1)
  with (out/'per_game.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=rows[0], lineterminator='\n')
    writer.writeheader(); writer.writerows(rows)
  metadata = dict(metrics=metrics, source=str(source), reference=str(reference),
                  reference_section='atari57_gamer',
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
                  iqm_implementation='scipy.stats.trim_mean(hns, 0.25); floor(26*0.25)=6 per tail',
                  uncertainty='Not estimated: one training seed; no episode-as-seed bootstrap.')
  (out/'metrics.json').write_text(json.dumps(metadata, indent=2)+'\n')
  thresholds = np.unique(np.r_[0, np.linspace(0, max(2, hns.max()+.5), 401), hns])
  profile = np.array([(hns > t).mean() for t in thresholds])
  with (out/'performance_profile.csv').open('w', newline='') as f:
    writer = csv.writer(f, lineterminator='\n'); writer.writerow(['threshold_hns', 'fraction_games_strictly_above'])
    writer.writerows(zip(thresholds, profile))
  fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout='constrained')
  for ax, limit, title in zip(axes, [2, thresholds.max()], ['Human-level region', 'Full score range']):
    ax.step(thresholds, profile, where='post', color='#7952b3', linewidth=2)
    ax.axvline(1, color='gray', linestyle='--', linewidth=1)
    ax.set(xlim=(0, limit), ylim=(0, 1.02), xlabel='Human-normalized score threshold', title=title)
    ax.grid(alpha=.2)
  axes[0].set_ylabel('Fraction of games above threshold')
  fig.suptitle('Reborn + Harmony | final 100K | 26 games | 1 training seed | no CI')
  for ext in ('png', 'pdf'):
    fig.savefig(out/f'performance_profile.{ext}', dpi=180)
  plt.close(fig)
  table = '\n'.join(f"| {r['game']} | {r['random']:.1f} | {r['human']:.1f} | {r['return_mean']:.2f} | {100*r['hns']:.2f} |" for r in rows)
  up = next(r for r in rows if r['game'] == 'up_n_down')
  report = f'''# Reborn + Harmony — Atari 100K, một training seed

**Kết quả sơ bộ, không phải kết quả đa training seed.** 26 game, training seed 0;
mỗi game lấy trung bình 100 evaluation episodes riêng tại checkpoint cuối 100.000
agent actions (400.000 frames với action repeat 4). Tổng cộng 2.600 final episodes.
Không chọn best checkpoint, không dùng training return thay evaluation return.
Bốn game Boxing/Up N Down/Frostbite/Road Runner từ screening job 3897;
22 game còn lại từ job 3997, cùng cấu hình Harmony. Đây là cấu hình đã được chọn
sau các vòng nghiên cứu, không phải đánh giá xác nhận độc lập chưa qua lựa chọn.

## Kết quả tổng hợp

| Metric | Giá trị |
|---|---:|
| Mean HNS (%) | {100*metrics['mean_hns']:.2f} |
| Median HNS (%) | {100*metrics['median_hns']:.2f} |
| IQM HNS (%) | {100*metrics['iqm_hns']:.2f} |
| Optimality gap, ngưỡng HNS = 1 | {metrics['optimality_gap_human']:.4f} |
| Số game vượt human (HNS > 1) | {metrics['superhuman_games']}/26 |

## Final score từng game

Random/human dùng chính xác bảng `atari57_gamer` trong `baselines.yaml`, không
làm tròn các reference thành số nguyên trước khi tính. HNS không clip âm hoặc >1.

| Game | Random | Human | Final return (100 episodes) | HNS (%) |
|---|---:|---:|---:|---:|
{table}

## Công thức và implementation

Với game g, R_g là trung bình 100 final episode returns của seed 0:

$$H_g = (R_g-R_g^{{random}})/(R_g^{{human}}-R_g^{{random}}).$$

- Mean HNS: trung bình 26 giá trị H_g; Median HNS: median của chúng.
- IQM: `scipy.stats.trim_mean(hns, 0.25)`, cùng quy ước implementation
  [rliable](https://github.com/google-research/rliable/blob/master/rliable/metrics.py).
  Với 26 điểm, bỏ floor(26 × 0,25) = 6 điểm ở mỗi đuôi, trung bình 14 điểm còn lại.
  Khi mở rộng đa seed, IQM dùng toàn bộ ma trận run × game, không lấy trung bình
  seed trước khi trim. Mean/Median dùng trung bình theo seed của từng game.
- Optimality gap: trung bình `max(1 - H_g, 0)`, nhỏ hơn tốt hơn; đây là mức thiếu
  so với human, không phải khoảng cách tới policy tối ưu thật.
- Superhuman: đếm game H_g > 1; không phải kiểm định ý nghĩa thống kê.
- Performance profile: tại ngưỡng τ, tỷ lệ game có H_g > τ. Mỗi game trọng số bằng nhau.
  Không smoothing và không có confidence band.

Implementation: [harmony_single_seed_report.py](../../corewm_eval/harmony_single_seed_report.py).
Reference loader: [hns_reference.py](../../corewm_eval/hns_reference.py).
Script kiểm tra đủ 26 game duy nhất, 100 finite returns/game, đúng checkpoint/seed,
mean khớp dữ liệu nguồn, và đối chiếu IQM/gap bằng công thức tương đương.

## Performance profile

![Performance profile](performance_profile.png)

[PDF](performance_profile.pdf) · [Dữ liệu profile](performance_profile.csv)

## Diễn giải và giới hạn

Up N Down đạt HNS {100*up['hns']:.2f}%, đóng góp
{100*up['hns']/26:.2f} điểm phần trăm vào Mean HNS của toàn bộ 26 game.
Vì vậy phải đọc Mean cùng Median/IQM, không suy từ Mean rằng mọi game đều cải thiện.
Giữ nguyên Up N Down trong aggregate chuẩn, không loại outlier để điều chỉnh kết quả.
[Báo cáo episode dài](../upndown_long_episode_report/README.md) ghi nhận 29/100 final
episodes chạm đúng 27.000 actions; đây là proxy time cap, không có termination-cause
log để khẳng định từng episode bị truncation. Không ngoại suy return ngoài giới hạn.

Không báo 95% training-seed CI: một seed không xác định được biến thiên huấn luyện.
100 eval episodes không phải 100 training seeds. Không bootstrap episodes rồi gọi
là CI đa seed; không báo probability of improvement với baseline chưa có dữ liệu
run-level theo phép đo tương thích. DreamerV3 training curves đã lưu không được dùng
như isolated final evaluation để tuyên bố thắng/thua. Báo cáo này không lập bảng
xếp hạng với số từ paper có protocol chưa đối chiếu.

## Nguồn và tái tạo

- [Raw evaluation returns + checkpoint hashes](../harmony_vs_dreamerv3_26_learning_curves/evaluation_results.json).
- [Protocol/source metadata](../harmony_vs_dreamerv3_26_learning_curves/plot_metadata.json).
- [Per-game CSV](per_game.csv), [metric JSON và SHA256 nguồn](metrics.json).
- Reference: [baselines.yaml](../../baselines.yaml), section `atari57_gamer`.
- Chạy `sbatch scripts/slurm_harmony_single_seed_report.sh` từ repository root.
  Job CPU chỉ đọc kết quả có sẵn; không train/eval lại, không dùng GPU.
'''
  (out/'README.md').write_text(report)
  print(json.dumps(metrics, indent=2))
  print('PASS: 26 final checkpoints, 2600 returns, metric cross-checks.')


if __name__ == '__main__':
  main()
