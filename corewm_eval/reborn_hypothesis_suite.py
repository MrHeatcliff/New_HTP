"""Four matched arms, four games per wave, one allocation, eight CPUs per game."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .h200_suite import atomic, slots
from .reborn_experiment import command, GAMES
from .reborn_checkpoint_eval import worker as evaluate

ARMS=('full','low_rate','no_q','deterministic')


def run_stage(output, name, cmd, slot, source):
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=slot['gpu'],OMP_NUM_THREADS='8',
      OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',XLA_PYTHON_CLIENT_PREALLOCATE='false',
      PYTHONUNBUFFERED='1',PYTHONHASHSEED='0',PAPER_DETERMINISTIC_UUID='1')
  atomic(output/f'{name}_request.json',dict(command=cmd,slot=slot,source=str(source)))
  atomic(output/'stage.json',dict(stage=name,status='RUNNING',started=time.time()))
  print(f'START {output.name} {name}',flush=True)
  start=time.time()
  with (output/f'{name}.log').open('w') as stream:
    subprocess.run(['taskset','-c',','.join(map(str,slot['cpus'])),*cmd],cwd=source,
        env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
  atomic(output/f'{name}_timing.json',dict(seconds=time.time()-start))
  print(f'END {output.name} {name}',flush=True)


def worker(root, arm, game, slot):
  source=root/'source'; wave=root/arm; out=wave/game; out.mkdir(parents=True)
  train=out/'full'
  run_stage(out,'train',command(game,train,arm)+['--run.steps','100000',
      '--run.exact_env_action_budget','True','--run.action_milestones',
      *map(str,range(10000,100001,10000))],slot,source)
  atomic(out/'stage.json',dict(stage='checkpoint_eval',status='RUNNING'))
  evaluate(wave,wave/'evaluation',game,slot,arm)
  run_stage(out,'diagnostic',[sys.executable,'-u','-m','corewm_eval.reborn_diagnostic',
      str(train),str(out/'diagnostic')],slot,source)
  run_stage(out,'actor_audit_own',[sys.executable,'-u','-m','corewm_eval.reborn_actor_audit',
      str(train),str(out/'diagnostic'),str(out/'actor_audit_own')],slot,source)
  if arm != 'full':
    run_stage(out,'actor_audit_shared',[sys.executable,'-u','-m','corewm_eval.reborn_actor_audit',
        str(train),str(root/'full'/game/'diagnostic'),str(out/'actor_audit_shared')],slot,source)
  atomic(out/'stage.json',dict(status='COMPLETE'))


def main(root):
  assert os.environ.get('SLURM_JOB_ID'), 'Slurm required'
  root=Path(root).resolve(); source=root/'source'
  devices=[]
  for d in os.environ['CUDA_VISIBLE_DEVICES'].split(','):
    uuid,name=subprocess.check_output(['nvidia-smi','-i',d,'--query-gpu=uuid,name',
        '--format=csv,noheader'],text=True).strip().split(', ')
    assert 'H200' in name
    devices.append(uuid)
  assigned=slots(sorted(os.sched_getaffinity(0)),devices)
  atomic(root/'manifest.json',dict(job=os.environ['SLURM_JOB_ID'],arms=ARMS,games=GAMES,
      training_seed=0,actions=100000,slots=assigned,source=str(source),start=time.time()))
  try:
    run_stage(root,'tests',[sys.executable,'-m','pytest','tests/test_reborn.py',
        'tests/test_h200_suite.py','-q'],assigned[0],source)
    for arm in ARMS:
      wave=root/arm; wave.mkdir()
      (wave/'source').symlink_to(source,target_is_directory=True)
      atomic(root/'status.json',dict(status='RUNNING',arm=arm))
      with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(worker,root,arm,g,s) for g,s in zip(GAMES,assigned)]
        for f in concurrent.futures.as_completed(futures): f.result()
    atomic(root/'status.json',dict(status='COMPLETE'))
  except BaseException as exc:
    atomic(root/'status.json',dict(status='FAILED',error=str(exc)))
    raise


if __name__ == '__main__': main(sys.argv[1])
