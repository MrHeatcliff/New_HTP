"""Plot only the two CSV files beside this script; no training repo required."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).resolve().parent
GAMES = ('boxing', 'up_n_down', 'frostbite', 'road_runner')

def read(name):
    with (HERE / name).open(newline='') as file:
        return list(csv.DictReader(file))

def main():
    curves, finals = read('learning_curves.csv'), read('results.csv')
    methods = sorted({(r['role'], r['arm']) for r in finals},
                     key=lambda x: (x[0] != 'ablation', x[1]))
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.6), constrained_layout=True)
    for ax, game in zip(axes.flat, GAMES):
        for i, (role, arm) in enumerate(methods):
            points = sorted((r for r in curves if r['game'] == game and r['arm'] == arm),
                            key=lambda r: int(r['agent_steps']))
            assert [int(r['agent_steps']) for r in points] == list(range(10000, 100001, 10000))
            final = next(r for r in finals if r['game'] == game and r['arm'] == arm)
            assert abs(float(points[-1]['mean_return']) - float(final['mean_return'])) < 1e-6
            color = ('#2463A6', '#D47829')[i]
            ax.plot([int(r['agent_steps']) / 1000 for r in points],
                    [float(r['mean_return']) for r in points],
                    color=color, linestyle='-' if i == 0 else '--',
                    linewidth=1.8, marker='o', markersize=3, label=arm)
            ax.scatter([100], [float(final['mean_return'])], color=color,
                       marker='D', s=38, zorder=5)
        ax.set(title=game.replace('_', ' ').title(), xlabel='Agent actions (×1,000)',
               ylabel='Mean evaluation return', xlim=(0, 104))
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:,.0f}'))
        ax.grid(alpha=0.18)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside upper center', ncol=2, frameon=False)
    fig.supxlabel('Seed 0 · 10 episodes/checkpoint; 100 at 100K (◆) · No smoothing or seed confidence bands', fontsize=9)
    for suffix in ('png', 'pdf'):
        fig.savefig(HERE / f'learning_curves.{suffix}', dpi=180,
                    metadata={'Creator': 'plot.py'} if suffix == 'pdf' else None)
    plt.close(fig)

if __name__ == '__main__':
    main()
