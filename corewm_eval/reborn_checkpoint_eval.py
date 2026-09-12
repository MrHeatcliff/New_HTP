"""Isolated Reborn checkpoint evaluation using the repository policy protocol."""
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from .h200_suite import atomic, slots
from .reborn_experiment import GAMES, command


def digest(path):
  sha = hashlib.sha256()
  for file in sorted(path.rglob('*')):
    if file.is_file():
      sha.update(str(file.relative_to(path)).encode())
      with file.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
          sha.update(block)
  return sha.hexdigest()


def worker(root, output, game, slot):
  (output/game).mkdir(parents=True,exist_ok=True)
  source = root/'source'
  env = dict(os.environ, CUDA_VISIBLE_DEVICES=slot['gpu'], OMP_NUM_THREADS='8',
      OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONPATH=str(source),
      XLA_PYTHON_CLIENT_PREALLOCATE='false', PAPER_TRAINING_SEED='0',
      PAPER_EVALUATION_SEED='0', PAPER_METHOD='Reborn', WANDB_MODE='disabled')
  rows = []
  for step in range(10000,100001,10000):
    checkpoint = root/game/'full/ckpt'/f'env_action_steps_{step:09d}'
    assert (checkpoint/'agent.pkl').is_file(), checkpoint
    before = digest(checkpoint)
    destination = output/game/f'{step:06d}'
    episodes = 100 if step == 100000 else 10
    cmd = command(game,destination)+['--script','eval_only','--run.from_checkpoint',
        str(checkpoint),'--run.envs','1','--run.eval_eps',str(episodes),
        '--run.steps','1000000000']
    atomic(output/game/'progress.json',dict(status='RUNNING',checkpoint=step,completed=rows))
    atomic(output/game/f'request_{step}.json',dict(command=cmd,hash=before,episodes=episodes))
    print(f'START {game} {step} episodes={episodes}',flush=True)
    with (output/game/f'eval_{step}.log').open('w') as stream:
      subprocess.run(['taskset','-c',','.join(map(str,slot['cpus'])),*cmd],
          cwd=source,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    scores = [json.loads(line)['episode/score'] for line in
              (destination/'scores.jsonl').read_text().splitlines()]
    assert len(scores) == episodes and all(map(math.isfinite,scores)), (game,step,len(scores))
    after = digest(checkpoint)
    assert before == after, 'Checkpoint changed during evaluation'
    rows.append(dict(game=game,checkpoint=step,episodes=episodes,returns=scores,
        mean=sum(scores)/len(scores),checkpoint_hash=before,training_seed=0,evaluation_seed=0))
    atomic(output/game/'summary.json',rows)
    print(f'END {game} {step} mean={rows[-1]["mean"]}',flush=True)
  atomic(output/game/'progress.json',dict(status='COMPLETE',completed=rows))


def main(root, output):
  assert os.environ.get('SLURM_JOB_ID'), 'Submit through Slurm'
  root,output=Path(root).resolve(),Path(output).resolve()
  output.mkdir(parents=True,exist_ok=True)
  devices=[]
  for device in os.environ['CUDA_VISIBLE_DEVICES'].split(','):
    devices.append(subprocess.check_output(['nvidia-smi','-i',device,
        '--query-gpu=uuid','--format=csv,noheader'],text=True).strip())
  assigned=slots(sorted(os.sched_getaffinity(0)),devices)
  atomic(output/'manifest.json',dict(job=os.environ['SLURM_JOB_ID'],source=str(root/'source'),
      games=GAMES,slots=assigned,start=time.time(),episode_schedule=[10]*9+[100],
      training_seed=0,evaluation_seed=0))
  try:
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
      futures=[pool.submit(worker,root,output,g,s) for g,s in zip(GAMES,assigned)]
      for future in concurrent.futures.as_completed(futures):
        future.result()
    atomic(output/'status.json',dict(status='COMPLETE',stage='evaluation'))
  except BaseException as exc:
    atomic(output/'status.json',dict(status='FAILED',error=str(exc)))
    raise


if __name__ == '__main__':
  main(*sys.argv[1:])
