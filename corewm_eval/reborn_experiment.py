"""Four games, one allocation; training, evaluation, then frozen diagnostics."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .h200_suite import atomic, slots

GAMES = tuple(x for x in os.environ.get(
    'REBORN_GAMES', 'boxing,up_n_down,frostbite,road_runner').split(',') if x)


def command(game, output, arm='full'):
  # One-factor ablations are explicit and reusable by later experiment rounds.
  arms = {
      'full': {}, 'no_rec': {'use_rec':False}, 'no_outcome': {'use_outcome':False},
      'no_q': {'q_weight':0.0}, 'no_rate': {'beta':0.0},
      'flat_rate': {'cumulative_rate':False}, 'no_context': {'context':False},
      'deterministic': {'stochastic':False},
      'affine_control': {'use_rec':False,'use_outcome':False,'beta':0.0},
  }
  cmd = [sys.executable,'-u','-m','dreamerv3.main_htp','--configs','atari100k','size25m',
      '--task',f'atari100k_{game}','--seed','0','--env.atari100k.use_seed','True',
      '--logdir',str(output),'--jax.prealloc','False','--logger.outputs','jsonl',
      '--run.log_policy_video','False','--agent.htp.enabled','False',
      '--agent.reborn.enabled','True']
  for k,v in arms[arm].items():
    cmd += [f'--agent.reborn.{k}', str(v)]
  return cmd


def worker(root, game, slot):
  out = root/game; out.mkdir()
  env = dict(os.environ,CUDA_VISIBLE_DEVICES=slot['gpu'],OMP_NUM_THREADS='8',
      OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONHASHSEED='0',
      PAPER_DETERMINISTIC_UUID='1',XLA_PYTHON_CLIENT_PREALLOCATE='false',
      XLA_PYTHON_CLIENT_MEM_FRACTION='0.40',PYTHONUNBUFFERED='1')
  def execute(stage, cmd):
    atomic(out/f'{stage}_command.json',cmd)
    atomic(out/'status.json',dict(status='RUNNING',stage=stage,start=time.time()))
    print(f'START {game} {stage} gpu={slot["gpu"]} cpus={slot["cpus"]}',flush=True)
    started = time.time()
    with (out/f'{stage}.log').open('w') as stream:
      subprocess.run(['taskset','-c',','.join(map(str,slot['cpus'])),*cmd],
          env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    atomic(out/f'{stage}_timing.json',dict(seconds=time.time()-started))
    print(f'END {game} {stage}',flush=True)
  try:
    train = out/'full'
    execute('train',command(game,train)+['--run.steps','100000',
        '--run.exact_env_action_budget','True','--run.action_milestones',
        *map(str,range(10000,100001,10000))])
    cp = train/'ckpt/env_action_steps_000100000'
    if not (cp/'agent.pkl').is_file():
      raise RuntimeError(f'Missing final checkpoint {cp}')
    execute('eval',command(game,out/'eval')+['--script','eval_only','--run.eval_eps','20',
        '--run.steps','1000000','--run.from_checkpoint',str(cp)])
    scores = [json.loads(x)['episode/score'] for x in (out/'eval/scores.jsonl').read_text().splitlines()]
    if len(scores) != 20:
      raise RuntimeError(f'Expected 20 evaluation episodes, got {len(scores)}')
    atomic(out/'control.json',dict(scores=scores,mean=sum(scores)/len(scores),training_seed=0))
    execute('diagnostic',[sys.executable,'-u','-m','corewm_eval.reborn_diagnostic',str(train),str(out/'diagnostic')])
    atomic(out/'status.json',dict(status='COMPLETE'))
  except BaseException as exc:
    atomic(out/'status.json',dict(status='FAILED',error=str(exc)))
    raise


def main(root):
  if not os.environ.get('SLURM_JOB_ID'):
    raise RuntimeError('Slurm required')
  root = Path(root)
  devices = []
  for device in os.environ['CUDA_VISIBLE_DEVICES'].split(','):
    uuid,name = subprocess.check_output(['nvidia-smi','-i',device,
        '--query-gpu=uuid,name','--format=csv,noheader'],text=True).strip().split(', ')
    if 'H200' not in name:
      raise RuntimeError(f'Expected H200, got {name}')
    devices.append(uuid)
  cpus = sorted(os.sched_getaffinity(0))
  if len(devices) == 1 and len(GAMES) == 2:
    if len(cpus) < 16:
      raise ValueError('Two-game mode requires 16 allocated CPUs')
    assigned = [dict(slot=i,gpu=devices[0],cpus=cpus[i*8:i*8+8]) for i in range(2)]
  else:
    assigned = slots(cpus,devices)
  if len(assigned) != len(GAMES):
    raise ValueError(f'Require one worker slot per game: {len(assigned)} != {len(GAMES)}')
  atomic(root/'manifest.json',dict(job=os.environ['SLURM_JOB_ID'],slots=assigned,games=GAMES,
      actions=100000,training_seed=0,arm='full',start=time.time(),source=str(Path.cwd())))
  atomic(root/'status.json',dict(status='RUNNING'))
  try:
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
      futures = [pool.submit(worker,root,g,s) for g,s in zip(GAMES,assigned)]
      for future in concurrent.futures.as_completed(futures):
        future.result()
    atomic(root/'status.json',dict(status='COMPLETE'))
  except BaseException as exc:
    atomic(root/'status.json',dict(status='FAILED',error=str(exc)))
    raise


if __name__ == '__main__':
  main(sys.argv[1])
