"""Evaluate the existing 130 Full checkpoints; all computation belongs in Slurm."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path('/home/vn-user0101/Dat/HTS-Dreamer/production_runs/corewm_atari100k_v1')
HORIZONS = (1, 2, 4, 8, 16, 32, 64)


def dump(path, data):
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(data, indent=2, allow_nan=False))


def digest(path):
  h = hashlib.sha256()
  with path.open('rb') as f:
    for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
  return h.hexdigest()


def source(game, seed):
  if game == 'alien' and seed == 0:
    candidates = list((ROOT / 'training/wave1/full/alien/seed_0').glob('**/ckpt/env_action_steps_000100000'))
  else:
    candidates = list((ROOT / f'training/stage1_full/{game}/seed_{seed}').glob('*/ckpt/env_action_steps_000100000'))
  if len(candidates) != 1:
    raise RuntimeError(f'Ambiguous checkpoint {game}/{seed}: {candidates}')
  return candidates[0]


def grid(path, images, labels, columns):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  fig, axes = plt.subplots(len(images), len(columns), figsize=(2 * len(columns), 1.7 * len(images)), squeeze=False)
  for i, row in enumerate(images):
    for j, im in enumerate(row):
      axes[i, j].imshow(np.clip(im, 0, 1)); axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
      if i == 0: axes[i, j].set_title(str(columns[j]))
      if j == 0: axes[i, j].set_ylabel(labels[i])
  fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def worker(out, game, seed):
  import elements
  import embodied
  from dreamerv3 import main_htp
  from .smoke_test import _load_config, _tree_hash
  from .extraction import ExtractionSpec, evaluation_seed, previous_actions
  from .manuscript_eval import evaluate_readonly
  from .metrics import cka_matrix
  cp = source(game, seed)
  config = _load_config(cp.parent.parent / 'config.yaml')
  config = config.update(logdir=str(out / game / f'seed_{seed}/runtime'))
  agent = main_htp.make_agent(config)
  checkpoint = elements.Checkpoint(); checkpoint.agent = agent; checkpoint.load(cp, keys=['agent'])
  before = _tree_hash(agent.save()['params']); updates = int(agent.n_updates)
  sha = digest(cp / 'agent.pkl')
  dyn = config.agent.dyn[config.agent.dyn.typ]
  dims = tuple(map(int, config.agent.htp.proj.dims))
  spec = ExtractionSpec(int(dyn.deter), (int(dyn.stoch), int(dyn.classes)), dims)
  destination = out / game / f'seed_{seed}'; destination.mkdir(parents=True, exist_ok=True)
  data = out / game / 'clips'; data.mkdir(parents=True, exist_ok=True)
  # Seed-0 source policy creates four predeclared early-episode clips shared by
  # all five checkpoints. Each begins at a genuine environment reset.
  if seed == 0:
    env = main_htp.make_env(config, 0, use_seed=False, seed=260909)
    driver = embodied.Driver([lambda: env], parallel=False)
    collected = []
    def record(tran, worker):
      collected.append({k: np.asarray(tran[k]).copy() for k in
          ('image', 'reward', 'is_first', 'is_last', 'is_terminal', 'action')})
    driver.on_step(record)
    try:
      for episode in range(4):
        collected.clear(); driver.reset(agent.init_policy)
        driver(lambda *args: agent.policy(*args, mode='eval'), steps=96)
        values = {k: np.stack([x[k] for x in collected]) for k in collected[0]}
        # Never silently join two episodes; retain the first terminal observation.
        terminals = np.flatnonzero(values['is_last'])
        if len(terminals): values = {k: v[:terminals[0] + 1] for k, v in values.items()}
        if len(values['image']) < 81:
          raise RuntimeError(f'Clip too short for fixed start=16, horizon=64: {game}/{episode}')
        np.savez_compressed(data / f'episode_{episode}.npz', **values)
    finally: driver.close()
  all_z, errors, full_errors = [], [], []
  for episode in range(4):
    with np.load(data / f'episode_{episode}.npz') as f: ep = dict(f)
    obs = {k: v[None] for k, v in ep.items() if k != 'action'}
    actions = {'action': ep['action'][None]}
    result = evaluate_readonly(agent, obs, {'action': previous_actions(ep['action'])[None]},
        actions, spec, evaluation_seed(0, episode, 16), 16)
    np.testing.assert_array_equal(result['used_actions']['action'], actions['action'][:, 16:80])
    gt = ep['image'][17:81].astype(np.float32) / 255
    errors.append(np.stack([np.abs(p[0] - gt).mean((1, 2, 3)) for p in result['prefixes']]))
    full_errors.append(np.abs(result['full'][0] - gt).mean((1, 2, 3)))
    all_z.append(result['one_z'])
    if episode == 0:
      idx = np.array(HORIZONS) - 1
      grid(destination / 'prefix_imagination.png', [gt[idx], result['full'][0, idx],
          *[p[0, idx] for p in result['prefixes']]],
          ['GT', 'Base dynamics', *[f'Prefix {i+1}' for i in range(len(dims))]], HORIZONS)
      positions = [0, 16, 32, 64]
      grid(destination / 'one_step_blocks.png',
          [ep['image'][np.array(positions)+1] / 255, result['one_full'][positions],
           *[p[positions] for p in result['one_blocks']]],
          ['GT', 'One step', *[f'Block {i+1}' for i in range(len(dims))]], positions)
  z = np.concatenate(all_z)
  cka = cka_matrix([z[:, lo:hi] for lo, hi in zip((0, *dims[:-1]), dims)])
  mae = np.mean(errors, axis=0); base = np.mean(full_errors, axis=0)
  import matplotlib.pyplot as plt
  fig, ax = plt.subplots(); im = ax.imshow(cka, vmin=0, vmax=1); fig.colorbar(im, ax=ax)
  ax.set(xlabel='Block', ylabel='Block', xticks=range(len(dims)), yticks=range(len(dims)),
         xticklabels=range(1, len(dims)+1), yticklabels=range(1, len(dims)+1))
  fig.tight_layout(); fig.savefig(destination / 'cka.png', dpi=140); plt.close(fig)
  fig, ax = plt.subplots()
  for i, row in enumerate(mae): ax.plot(range(1, 65), row, label=f'Prefix {i+1}')
  ax.plot(range(1, 65), base, '--', label='Base dynamics'); ax.legend()
  ax.set(xlabel='Horizon (actions)', ylabel='Pixel MAE [0,1]')
  fig.tight_layout(); fig.savefig(destination / 'mae.png', dpi=140); plt.close(fig)
  assert before == _tree_hash(agent.save()['params']) and updates == int(agent.n_updates)
  assert sha == digest(cp / 'agent.pkl')
  dump(destination / 'result.json', {'game': game, 'seed': seed, 'checkpoint': str(cp),
      'checkpoint_sha256': sha, 'prefix_dims': dims, 'cka': cka.tolist(),
      'prefix_mae': mae.tolist(), 'backbone_mae': base.tolist(),
      'clip_sha256': {p.name: digest(p) for p in sorted(data.glob('*.npz'))},
      'one_step_samples': len(z), 'rollout_starts': 4, 'parameters_unchanged': True})


def report(out):
  from .full_vs_dreamerv3_curves import load_full_policy_curves
  from .hns_reference import load_reference
  frame, sources = load_full_policy_curves(ROOT / 'evaluations')
  refs = load_reference('baselines.yaml', 'atari57_gamer')
  lines = ['# CoRe-WM: đánh giá 26 game × 5 seed', '',
      'Policy: checkpoint 100k actions, 100 evaluation episodes/seed; AUC hình thang trên 10k–100k, chia 90k. HNS không clip; mean/median tính sau khi lấy trung bình 5 seed mỗi game.', '',
      'Representation: 4 clip từ policy seed 0 mỗi game dùng chung cho 5 checkpoint; rollout bắt đầu ở posterior t=16, chạy 64 actions thật. Đây là diagnostic đầu episode. CKA dùng các dự đoán một bước độc lập từ từng posterior. Mọi prefix dùng chung một rollout; reconstruction không đưa lại vào dynamics. Block-only được tính bằng hiệu hai cumulative reconstruction, tương đương đúng D_l(z_l), tránh bias từ head bị mask.', '',
      'Checkpoint có 5 block: dùng tất cả 5 block. Các mốc 1,2,4,8,16,32,64 là horizon. Base dynamics trong hình là RSSM của cùng checkpoint; chưa phải mô hình baseline huấn luyện độc lập.', '',
      '| Game | Return mean ± SD (5 seed) | HNS % | AUC HNS % | CKA off-diagonal | MAE prefix 1 @64 | MAE full prefix @64 | MAE base @64 |',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
  hnss = []
  for game in sorted(refs):
    random, human = refs[game]; sub = frame[frame.game == game]
    final = sub[sub.agent_steps == 100000]['return'].to_numpy()
    hns = (final.mean() - random) / (human - random); hnss.append(hns)
    auc = []
    for seed in range(5):
      curve = sub[sub.seed == seed].sort_values('agent_steps')
      auc.append(np.trapz((curve['return'].to_numpy()-random)/(human-random), curve.agent_steps.to_numpy()) / 90000)
    paths = [out / game / f'seed_{seed}/result.json' for seed in range(5)]
    if all(p.exists() for p in paths):
      results = [json.loads(p.read_text()) for p in paths]
      c = np.mean([np.array(r['cka'])[np.triu_indices(len(r['cka']), 1)].mean() for r in results])
      vals = [c, np.mean([r['prefix_mae'][0][-1] for r in results]),
              np.mean([r['prefix_mae'][-1][-1] for r in results]), np.mean([r['backbone_mae'][-1] for r in results])]
      extra = ' | '.join(f'{v:.4f}' for v in vals)
    else: extra = 'pending | pending | pending | pending'
    lines.append(f'| {game} | {final.mean():.2f} ± {final.std(ddof=1):.2f} | {100*hns:.2f} | {100*np.mean(auc):.2f} | {extra} |')
  lines += ['', f'Mean HNS: {100*np.mean(hnss):.2f}%; median HNS: {100*np.median(hnss):.2f}%; games above human: {sum(x>1 for x in hnss)}/26.', '',
            '## Figures', '']
  for game in sorted(refs):
    lines += [f'### {game}', '', *[f'![{game} {name}]({game}/seed_0/{name}.png)' for name in ('one_step_blocks', 'cka', 'prefix_imagination', 'mae')], '']
  (out / 'README.md').write_text('\n'.join(lines) + '\n')
  dump(out / 'policy_sources.json', sources)


def main():
  if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('Slurm required')
  p = argparse.ArgumentParser(); p.add_argument('out', type=Path); p.add_argument('--game'); p.add_argument('--seed', type=int); p.add_argument('--report', action='store_true')
  args = p.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
  if args.report: report(args.out)
  elif args.game: worker(args.out, args.game, args.seed)
  else:
    from .config import ATARI100K_GAMES
    from concurrent.futures import ThreadPoolExecutor
    report(args.out)
    cpus = sorted(os.sched_getaffinity(0))
    devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    if len(cpus) < 32 or len(devices) != 2:
      raise RuntimeError('Expected two GPUs and 32 CPUs')
    def slot(index):
      env = dict(os.environ, CUDA_VISIBLE_DEVICES=devices[index//2])
      for game in ATARI100K_GAMES[index::4]:
        for seed in range(5):
          target = args.out / game / f'seed_{seed}/result.json'
          if target.exists(): continue
          print(f'START {game} seed={seed} slot={index}', flush=True)
          with (args.out / f'{game}_{seed}.log').open('w') as log:
            subprocess.run(['taskset', '-c', ','.join(map(str, cpus[8*index:8*index+8])),
                sys.executable, '-u', '-m', 'corewm_eval.manuscript_suite', str(args.out),
                '--game', game, '--seed', str(seed)], env=env,
                stdout=log, stderr=subprocess.STDOUT, check=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
      list(pool.map(slot, range(4)))
    report(args.out)
    dump(args.out / 'status.json', {'status': 'COMPLETE', 'checkpoints': 130})


if __name__ == '__main__': main()
