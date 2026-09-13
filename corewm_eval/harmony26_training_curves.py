"""Like-for-like 5K-binned training returns; never substitute evaluation scores."""
import json
import os
import hashlib
from pathlib import Path


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Use Slurm for plotting'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from .config import ATARI100K_GAMES
    from .full_vs_dreamerv3_curves import _binned

    out = Path('paper_artifacts/harmony_vs_dreamerv3_26_training_curves')
    for ext in ('png', 'pdf'):
        (out/ext).mkdir(parents=True, exist_ok=True)
    ref = Path('paper_artifacts/full_vs_dreamerv3_constraint_suite_learning_curves/aggregate.csv')
    base = pd.read_csv(ref)
    base = base[base.method == 'DreamerV3'].copy()
    assert set(base.game) == set(ATARI100K_GAMES)
    records, sources = [], []
    for game in ATARI100K_GAMES:
        root = Path('production_runs/reborn_harmony_2ew6wt0w/harmony') if game in (
            'boxing', 'up_n_down', 'frostbite', 'road_runner') else Path('production_runs/harmony_remaining22_zvn1urx5')
        path = root/game/'full/paper_artifacts/episode_scores.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            step, score = row['agent_actions'], row['episode_score']
            if 0 < step <= 100000:
                assert np.isfinite(score)
                records.append(dict(method='Reborn + Harmony', game=game, seed=0,
                                    agent_steps=step, **{'return': score}))
        sources.append(dict(game=game, path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    episodes = pd.DataFrame(records)
    assert set(episodes.game) == set(ATARI100K_GAMES)
    per_seed, harmony = _binned(episodes, bin_width=5000)
    combined = pd.concat([base, harmony], ignore_index=True)
    combined.to_csv(out/'aggregate.csv', index=False)
    episodes.to_csv(out/'harmony_training_episodes.csv', index=False)
    per_seed.to_csv(out/'harmony_per_seed.csv', index=False)
    base.to_csv(out/'dreamerv3_training_reference.csv', index=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    def label(g):
        return {'up_n_down':'Up N Down', 'ms_pacman':'Ms. Pac-Man', 'jamesbond':'James Bond'}.get(g, g.replace('_',' ').title())
    def draw(ax, game, small=False):
        for data, color, name in ((base, '#3B82F6', 'DreamerV3 (5 seeds)'),
                                  (harmony, '#7C3AED', 'Reborn + Harmony (seed 0)')):
            d = data[data.game == game].set_index('bin').reindex(range(20))
            x = (np.arange(20)+.5)*5000
            ax.plot(x, d['mean'], color=color, lw=1.8, label=name,
                    marker='.' if name.startswith('Reborn') else None, markersize=4)
            if name.startswith('Dreamer'):
                ax.fill_between(x, d['mean']-d['sem'], d['mean']+d['sem'], color=color, alpha=.16, linewidth=0)
        ax.set_title(label(game), fontsize=11 if small else 14)
        ax.set_xlim(0,100000)
        ax.set_xticks([0,25000,50000,75000,100000], ['0','25K','50K','75K','100K'])
        ax.set_xlabel('Agent actions', fontsize=9 if small else 11)
        ax.set_ylabel('Training episode return', fontsize=9 if small else 11)
        ax.tick_params(labelsize=8 if small else 10)
        ax.grid(alpha=.2)
    for game in ATARI100K_GAMES:
        fig, ax = plt.subplots(figsize=(7.4,4.8))
        draw(ax,game)
        ax.legend(frameon=False, fontsize=9)
        fig.text(.5,.015,'Training vs training · 5K bins · Blue band: ±1 SEM · No smoothing', ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.04,1,1))
        for ext in ('png','pdf'):fig.savefig(out/ext/f'{game}.{ext}',dpi=180)
        plt.close(fig)
    fig, axes = plt.subplots(7,4,figsize=(18,22))
    for ax, game in zip(axes.flat,ATARI100K_GAMES):draw(ax,game,True)
    for ax in list(axes.flat)[26:]:ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle('Atari 100K · Training returns · Harmony vs DreamerV3',fontsize=20,y=.996)
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.979),ncol=2,frameon=False)
    fig.text(.5,.955,'Both methods: per-seed episode averages in 5K-action bins\nDreamerV3: 5 seeds ±1 SEM; Harmony: seed 0 · Empty bins remain gaps · No smoothing',ha='center',va='top',fontsize=10)
    fig.tight_layout(rect=(0,0,1,.925),h_pad=2,w_pad=1.6)
    for ext in ('png','pdf'):fig.savefig(out/f'all_games_overview.{ext}',dpi=180)
    plt.close(fig)
    missing={g:sorted(set(range(20))-set(harmony[harmony.game==g]['bin'].astype(int))) for g in ATARI100K_GAMES}
    (out/'metadata.json').write_text(json.dumps(dict(job=os.environ['SLURM_JOB_ID'],sources=sources,
        reference=str(ref),reference_sha256=hashlib.sha256(ref.read_bytes()).hexdigest(),
        measurement='training returns for both methods',bin_width=5000,missing_harmony_bins=missing,
        harmony_training_seeds=[0],dreamerv3_training_seeds=[0,1,2,3,4],band='DreamerV3 ±SEM',
        x_source='Harmony episode_scores.jsonl agent_actions; baseline cached agent_steps',
        evaluation_used=False,smoothing=None),indent=2)+'\n')
    text = '# Harmony vs DreamerV3 — training curves, 26 game\n\n![Overview](all_games_overview.png)\n\n[PDF tổng quan](all_games_overview.pdf)\n\n'
    text += ('**Cả hai đường đều là training episode return**, gom bin 5K actions theo đúng hàm binning của hình DreamerV3 trước đó. '
             'Lấy trung bình episode trong từng bin/từng seed, rồi trung bình giữa seed. DreamerV3: 5 seed, dải ±1 SEM; '
             'Harmony: seed 0, không vẽ dải bất định giữa seed. Không EMA smoothing, không dùng evaluation để điền dữ liệu.\n\n'
             'Trục x của Harmony dùng agent_actions đã ghi trong episode_scores.jsonl, không ước lượng bằng logger step/4. '
             'Bin [0,5K) đặt ở 2.5K; bin cuối [95K,100K] ở97.5K. Episode được gán theo thời điểm kết thúc. '
             'Bin không có episode được để trống, không nội suy. Policy sống lâu có thể gây gap; đó không tự động là lỗi logging.\n\n'
             '**Đây là so sánh cùng loại metric, không bảo đảm đã khớp mọi cấu hình/model size và số seed.** '
             'Điểm cuối là trung bình training trong bin cuối, không phải final evaluation100episode. '
             '[Bản evaluation riêng vẫn được giữ](../harmony_vs_dreamerv3_26_learning_curves/README.md).\n\n'
             '| Game | PNG | PDF | Bin thiếu (index0–19) |\n|---|---|---|---|\n')
    for g in ATARI100K_GAMES:text+=f'| {label(g)} | [PNG](png/{g}.png) | [PDF](pdf/{g}.pdf) | {missing[g]} |\n'
    text+='\n[Aggregate CSV](aggregate.csv) · [Harmony episode rows](harmony_training_episodes.csv) · [Provenance](metadata.json)\n\nTái tạo bằng scripts/slurm_harmony26_training_curves.sh. Chỉ dựng hình bằng CPU Slurm, không train/eval lại.\n'
    (out/'README.md').write_text(text)
    print(json.dumps(dict(status='COMPLETE',games=26,episodes=len(episodes),missing_bins=missing)))


if __name__ == '__main__':main()
