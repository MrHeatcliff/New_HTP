"""Render completed Harmony evaluation against the cached Dreamer training reference."""
import hashlib
import json
import os
from pathlib import Path


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Run plotting through Slurm'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from .config import ATARI100K_GAMES

    output = Path('paper_artifacts/harmony_vs_dreamerv3_26_learning_curves')
    for folder in ('png', 'pdf'):
        (output / folder).mkdir(parents=True, exist_ok=True)
    old = Path('production_runs/reborn_harmony_2ew6wt0w/harmony/evaluation')
    new = Path('production_runs/harmony_remaining22_zvn1urx5/evaluation')
    reference = Path('paper_artifacts/full_vs_dreamerv3_constraint_suite_learning_curves/aggregate.csv')
    baseline = pd.read_csv(reference)
    baseline = baseline[baseline.method == 'DreamerV3'].copy()
    assert set(baseline.game) == set(ATARI100K_GAMES)
    assert np.isfinite(baseline[['mean', 'sem', 'agent_steps']]).all().all()
    rows, sources = [], []
    for game in ATARI100K_GAMES:
        root = old if game in ('boxing', 'up_n_down', 'frostbite', 'road_runner') else new
        path = root / game / 'summary.json'
        data = json.loads(path.read_text())
        assert [r['checkpoint'] for r in data] == list(range(10000, 100001, 10000))
        assert [r['episodes'] for r in data] == [10] * 9 + [100]
        for row in data:
            assert row['game'] == game and row['training_seed'] == 0 and row['evaluation_seed'] == 0
            assert len(row['returns']) == row['episodes'] and np.isfinite(row['returns']).all()
            assert np.isclose(np.mean(row['returns']), row['mean'])
            assert len(row['checkpoint_hash']) == 64
        assert not baseline[baseline.game == game].agent_steps.duplicated().any()
        rows.extend(data)
        sources.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    assert len(rows) == 260
    (output / 'evaluation_results.json').write_text(json.dumps(rows, indent=2) + '\n')
    frame = pd.DataFrame([{k: v for k, v in r.items() if k != 'returns'} for r in rows])
    frame.to_csv(output / 'evaluation_summary.csv', index=False)
    baseline.to_csv(output / 'dreamerv3_training_reference.csv', index=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})

    def title(game):
        return {'up_n_down': 'Up N Down', 'ms_pacman': 'Ms. Pac-Man',
                'jamesbond': 'James Bond', 'qbert': 'Qbert'}.get(game, game.replace('_', ' ').title())

    def draw(ax, game, small=False):
        b = baseline[baseline.game == game].sort_values('agent_steps')
        h = frame[frame.game == game].sort_values('checkpoint')
        ax.plot(b.agent_steps, b['mean'], '--', color='#3B82F6', lw=1.8,
                label='DreamerV3: training (5 seeds)')
        ax.fill_between(b.agent_steps, b['mean'] - b['sem'], b['mean'] + b['sem'],
                        color='#3B82F6', alpha=.16, linewidth=0)
        ax.plot(h.checkpoint, h['mean'], 'o-', color='#7C3AED', lw=1.9, markersize=3,
                label='Reborn + Harmony: isolated eval (seed 0)')
        ax.set_title(title(game), fontsize=11 if small else 14)
        ax.set_xlim(0, 100000)
        ax.set_xticks([0, 25000, 50000, 75000, 100000], ['0', '25K', '50K', '75K', '100K'])
        ax.grid(alpha=.2)
        ax.tick_params(labelsize=8 if small else 10)
        ax.set_xlabel('Agent actions', fontsize=9 if small else 11)
        ax.set_ylabel('Episode return', fontsize=9 if small else 11)

    for game in ATARI100K_GAMES:
        fig, ax = plt.subplots(figsize=(7.4, 4.8))
        draw(ax, game)
        ax.legend(frameon=False, fontsize=8, loc='best')
        fig.text(.5, .015, 'Different measurement protocols; blue band = ±1 SEM across training seeds.',
                 ha='center', fontsize=8, color='#475569')
        fig.tight_layout(rect=(0, .04, 1, 1))
        for ext in ('png', 'pdf'):
            fig.savefig(output / ext / f'{game}.{ext}', dpi=180)
        plt.close(fig)
    fig, axes = plt.subplots(7, 4, figsize=(18, 22))
    for ax, game in zip(axes.flat, ATARI100K_GAMES):
        draw(ax, game, True)
    for ax in list(axes.flat)[26:]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .979), ncol=2, frameon=False, fontsize=10)
    fig.suptitle('Atari 100K · Reborn + Harmony vs DreamerV3', fontsize=20, y=.996)
    fig.text(.5, .955, 'Harmony: 10 eval episodes at 10K–90K; 100 at 100K. DreamerV3: training returns ±1 SEM.\n'
             'Different measurement protocols · One Harmony training seed · No smoothing or synthetic points',
             ha='center', va='top', fontsize=10, color='#475569')
    fig.tight_layout(rect=(0, 0, 1, .925), h_pad=2.0, w_pad=1.6)
    for ext in ('png', 'pdf'):
        fig.savefig(output / f'all_games_overview.{ext}', dpi=180)
    plt.close(fig)
    metadata = {'job': os.environ['SLURM_JOB_ID'], 'games': 26, 'evaluation_points': len(rows),
                'evaluation_episodes': sum(r['episodes'] for r in rows), 'training_seed': 0,
                'sources': sources, 'baseline_source': str(reference),
                'baseline_sha256': hashlib.sha256(reference.read_bytes()).hexdigest(),
                'caveat': 'Harmony isolated evaluation vs DreamerV3 training reference; not matched evaluation.',
                'smoothing': None, 'band': 'DreamerV3 ±1 SEM; no Harmony seed-uncertainty band'}
    (output / 'plot_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    readme = '# Harmony vs DreamerV3 — toàn bộ 26 game Atari\n\n'
    readme += '![26-game overview](all_games_overview.png)\n\n[PDF tổng quan](all_games_overview.pdf)\n\n'
    readme += ('Đường **tím**: Reborn + Harmony, training seed 0, evaluation riêng mỗi checkpoint; '
               '10 episode tại 10K–90K và 100 episode tại 100K. Đường **xanh đứt**: DreamerV3 '
               'training returns từ W&B đã lưu, aggregate 5 seed theo bin 5K; dải ±1 SEM giữa seed.\n\n'
               '**Hai đường khác loại phép đo; đây là reference comparison, không phải matched evaluation.** '
               'Không dùng bảng này để tuyên bố thắng/thua baseline theo cùng protocol. Harmony một seed '
               'không có dải bất định giữa training seed. Không smoothing, không thêm điểm 0, không lấy training '
               'score lấp evaluation. Nối thẳng các checkpoint chỉ để hiển thị.\n\n'
               'Bốn game Boxing/Up N Down/Frostbite/Road Runner lấy từ screening job 3897; '
               '22 game còn lại từ job 3997, cùng cấu hình Harmony. Đủ 260 điểm và 4.940 episode. '
               'Nguồn/hash và quy ước vẽ nằm trong [metadata](plot_metadata.json).\n\n'
               '| Game | Final Harmony (100 episodes) | PNG | PDF |\n|---|---:|---|---|\n')
    for game in ATARI100K_GAMES:
        final = frame[(frame.game == game) & (frame.checkpoint == 100000)]['mean'].iloc[0]
        readme += f'| {title(game)} | {final:,.2f} | [PNG](png/{game}.png) | [PDF](pdf/{game}.pdf) |\n'
    readme += ('\n[Evaluation CSV](evaluation_summary.csv) · [Episode returns + checkpoint hashes](evaluation_results.json)'
               ' · [DreamerV3 reference CSV](dreamerv3_training_reference.csv)\n\n'
               'Tái tạo: submit `scripts/slurm_harmony26_curves.sh`; script `corewm_eval/harmony26_curves.py`. '
               'Chỉ đọc kết quả đã có; không train hoặc evaluation lại, không dùng GPU.\n')
    (output / 'README.md').write_text(readme)
    print(json.dumps({'status': 'COMPLETE', 'output': str(output), 'games': 26, 'points': 260}))


if __name__ == '__main__':
    main()
