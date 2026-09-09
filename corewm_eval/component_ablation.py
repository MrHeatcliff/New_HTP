"""From-scratch 2x2 loss ablation; one Slurm allocation, four pinned games."""
import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .h200_suite import atomic, slots

GAMES = ('boxing', 'up_n_down', 'frostbite', 'road_runner')
ARMS = {'flat': (False, False), 'rec_only': (True, False),
        'pdyn_only': (False, True), 'full': (True, True)}
MODE = os.environ.get('HTP_COMPONENT_MODE', 'factorial')
if MODE == 'direct_h':
  ARMS = {'h_control': (False, False), 'h_pdyn': (False, True)}
if MODE == 'direct_h_constraint':
  ARMS = {'h_pdyn': (False, True), 'h_pdyn_constraint': (False, True)}


def command(game, arm, output):
  rec, pdyn = ARMS[arm]
  cmd = [sys.executable, '-u', '-m', 'dreamerv3.main_htp', '--configs',
      'htp_atari100k', 'size25m', '--task', f'atari100k_{game}', '--seed', '0',
      '--env.atari100k.use_seed', 'True', '--logdir', str(output),
      '--jax.prealloc', 'False', '--logger.outputs', 'jsonl',
      '--run.log_policy_video', 'False', '--agent.htp.enabled', 'True',
      '--agent.htp.use_proj', 'True', '--agent.htp.grad_to_backbone', 'False',
      '--agent.htp.use_recon', str(rec), '--agent.htp.use_pdyn', str(pdyn),
      '--agent.htp.use_vicreg', 'False', '--agent.htp.persistence_scale', '0.0',
      '--agent.htp.persistence_isotropy', '0.0', '--agent.htp.persistence_min_rank', '0.0',
      '--agent.htp.persistence_decorrelation', '0.0',
      '--agent.htp.persistence_block2_rank_scale', '0.0',
      '--agent.htp.recon.first_target_window', '1',
      '--agent.htp.pdyn.mask_episode_boundaries', 'False']
  if MODE in ('direct_h', 'direct_h_constraint'):
    cmd[cmd.index('--agent.htp.use_proj') + 1] = 'False'
    cmd[cmd.index('--agent.htp.grad_to_backbone') + 1] = 'True'
    cmd += ['--agent.htp.pdyn.dims', '128', '256', '512', '1024', '3840']
  if arm == 'h_pdyn_constraint':
    cmd[cmd.index('--agent.htp.persistence_scale') + 1] = '0.1'
    cmd[cmd.index('--agent.htp.persistence_isotropy') + 1] = '0.01'
    cmd += ['--agent.htp.persistence_metric', 'whitened',
            '--agent.htp.persistence_all_lags', 'True']
  return cmd


def worker(root, game, slot):
  out = root / game
  out.mkdir()
  env = dict(os.environ, CUDA_VISIBLE_DEVICES=slot['gpu'], OMP_NUM_THREADS='8',
      OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONHASHSEED='0',
      PAPER_DETERMINISTIC_UUID='1', XLA_PYTHON_CLIENT_PREALLOCATE='false',
      XLA_PYTHON_CLIENT_MEM_FRACTION='0.4')
  def execute(label, cmd):
    atomic(out / f'{label}_command.json', cmd)
    atomic(out / 'status.json', {'status': 'RUNNING', 'stage': label})
    print(f'START {game} {label} gpu={slot["gpu"]} cpus={slot["cpus"]}', flush=True)
    with (out / f'{label}.log').open('w') as stream:
      subprocess.run(['taskset', '-c', ','.join(map(str, slot['cpus'])), *cmd],
          env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    print(f'END {game} {label}', flush=True)
  repo = Path(os.environ['HTP_RESEARCH_REPO'])
  dataset = out / 'fixed_clips'
  shutil.copytree(repo / 'production_runs/rank32_four_game_TN09vuQY' / game / 'fixed_clips', dataset)
  names = list(ARMS)
  offset = GAMES.index(game) % len(names)
  results = {}
  for arm in names[offset:] + names[:offset]:
    train = out / arm
    execute(f'{arm}_train', command(game, arm, train) + ['--run.steps', '100000',
        '--run.exact_env_action_budget', 'True', '--run.action_milestones',
        *map(str, range(10000, 100001, 10000))])
    cp = train / 'ckpt/env_action_steps_000100000'
    if not (cp / 'agent.pkl').is_file():
      raise RuntimeError(f'Missing checkpoint: {cp}')
    evaluation = out / f'{arm}_eval'
    execute(f'{arm}_eval', command(game, arm, evaluation) + ['--script', 'eval_only',
        '--run.eval_eps', '20', '--run.steps', '1000000', '--run.from_checkpoint', str(cp)])
    rep = [sys.executable, '-u', '-m', 'corewm_eval.four_game_representation']
    execute(f'{arm}_extract', rep + ['extract', '--source', str(train), '--checkpoint',
        str(cp), '--dataset', str(dataset), '--output', str(out / f'{arm}_representation.npz')])
    execute(f'{arm}_analyze', rep + ['analyze', '--dataset', str(dataset),
        '--output', str(out / f'{arm}_representation.json')])
    scores = [json.loads(line)['episode/score'] for line in (evaluation / 'scores.jsonl').read_text().splitlines()]
    if len(scores) != 20:
      raise RuntimeError('Expected 20 complete evaluation episodes')
    results[arm] = {'scores': scores, 'mean': sum(scores) / len(scores)}
    atomic(out / 'results.json', results)
  m = {k: v['mean'] for k, v in results.items()}
  if MODE == 'direct_h_constraint':
    atomic(out / 'contrasts.json', {
        'constraint_on_h_pdyn': m['h_pdyn_constraint'] - m['h_pdyn'],
        'limitation': 'One training seed; tests the persistence+isotropy package on h, not projection removal.'})
    atomic(out / 'status.json', {'status': 'COMPLETE'})
    return
  if MODE == 'direct_h':
    atomic(out / 'contrasts.json', {
        'pdyn_on_h': m['h_pdyn'] - m['h_control'],
        'limitation': 'One training seed; not a comparison isolating projection removal from full CoRe-WM.'})
    atomic(out / 'status.json', {'status': 'COMPLETE'})
    return
  atomic(out / 'contrasts.json', {
      'rec_without_pdyn': m['rec_only'] - m['flat'],
      'rec_with_pdyn': m['full'] - m['pdyn_only'],
      'pdyn_without_rec': m['pdyn_only'] - m['flat'],
      'pdyn_with_rec': m['full'] - m['rec_only'],
      'interaction': m['full'] - m['rec_only'] - m['pdyn_only'] + m['flat'],
      'limitation': 'One training seed; episode variation is not training-seed uncertainty.'})
  atomic(out / 'status.json', {'status': 'COMPLETE'})


def main(root):
  if not os.environ.get('SLURM_JOB_ID'):
    raise RuntimeError('Slurm only')
  root = Path(root).resolve()
  devices = []
  for device in os.environ['CUDA_VISIBLE_DEVICES'].split(','):
    row = subprocess.check_output(['nvidia-smi', '-i', device,
        '--query-gpu=uuid,name', '--format=csv,noheader'], text=True)
    uuid, name = map(str.strip, row.split(','))
    if 'H200' not in name:
      raise RuntimeError(row)
    devices.append(uuid)
  assigned = slots(sorted(os.sched_getaffinity(0)), devices)
  atomic(root / 'manifest.json', {'job': os.environ['SLURM_JOB_ID'], 'arms': ARMS,
      'games': GAMES, 'slots': assigned, 'seed': 0, 'actions': 100000,
      'initialization': 'from scratch, no checkpoint continuation',
      'source': str(Path.cwd()), 'start': time.time()})
  atomic(root / 'status.json', {'status': 'RUNNING'})
  try:
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
      futures = [pool.submit(worker, root, game, slot) for game, slot in zip(GAMES, assigned)]
      for future in concurrent.futures.as_completed(futures):
        future.result()
  except BaseException as exc:
    atomic(root / 'status.json', {'status': 'FAILED', 'error': str(exc)})
    raise
  atomic(root / 'status.json', {'status': 'COMPLETE'})


if __name__ == '__main__':
  main(sys.argv[1])
