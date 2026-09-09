"""One allocation, two GPUs, four CPU-pinned training slots, canonical suite."""
import collections
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .config import ATARI100K_GAMES


def slots(cpus, devices):
  if len(cpus) < 32 or len(devices) != 2 or len(set(devices)) != 2:
    raise ValueError('Require 32 distinct allocated CPUs and two distinct GPUs')
  if len(set(cpus)) != len(cpus):
    raise ValueError('Duplicate CPU IDs')
  return [{'slot':i,'gpu':devices[i//2],'cpus':list(cpus[i*8:i*8+8])}
          for i in range(4)]


def command(python, game, out, cpu_ids):
  task = {'jamesbond':'james_bond'}.get(game,game)
  return ['taskset','-c',','.join(map(str,cpu_ids)),python,'-u','-m','dreamerv3.main_htp',
      '--configs','htp_atari100k','size25m','--task',f'atari100k_{task}',
      '--seed','0','--env.atari100k.use_seed','True','--logdir',str(out),
      '--jax.prealloc','False','--logger.outputs','jsonl','--run.log_policy_video','False',
      '--run.steps','110000','--run.exact_env_action_budget','True',
      '--run.action_milestones',*[str(x) for x in range(10000,110001,10000)],
      '--agent.htp.persistence_metric','whitened','--agent.htp.persistence_scale','0.1',
      '--agent.htp.persistence_all_lags','True','--agent.htp.persistence_isotropy','0.01',
      '--agent.htp.recon.first_target_window','1',
      '--agent.htp.pdyn.mask_episode_boundaries','False']


def atomic(path, value):
  temp=path.with_suffix('.tmp'); temp.write_text(json.dumps(value,indent=2)+'\n')
  temp.replace(path)


def main(root):
  if not os.environ.get('SLURM_JOB_ID'):
    raise RuntimeError('Slurm only')
  root=Path(root).resolve(); output=root/'runs'; output.mkdir(exist_ok=False)
  visible=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
  devices=[]; hardware=[]
  for device in visible:
    row=subprocess.check_output(['nvidia-smi','-i',device,
        '--query-gpu=uuid,name,memory.total','--format=csv,noheader,nounits'],text=True).strip()
    uuid,name,total=[x.strip() for x in row.split(',')]
    if 'H200' not in name: raise RuntimeError(row)
    devices.append(uuid); hardware.append({'uuid':uuid,'name':name,'memory_mib':float(total)})
  workers=slots(sorted(os.sched_getaffinity(0)),devices)
  state={'job_id':os.environ['SLURM_JOB_ID'],'status':'RUNNING','hardware':hardware,
      'slots':workers,'games':list(ATARI100K_GAMES),'seed':0,'actions_per_game':110000,
      'source':str(Path.cwd()),'results':{},'pending':list(ATARI100K_GAMES)}
  atomic(root/'manifest.json',state)
  queue=collections.deque(ATARI100K_GAMES); active={}
  def persist():
    state['pending']=list(queue)
    state['active']={str(k):{'game':v['game'],'pid':v['process'].pid} for k,v in active.items()}
    atomic(root/'status.json',state)
  def stop(signum,frame):
    raise RuntimeError(f'Received signal {signum}')
  signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
  try:
    with (root/'telemetry.jsonl').open('w') as telemetry:
      while queue or active:
        for worker in workers:
          key=worker['slot']
          if key in active or not queue: continue
          game=queue.popleft(); out=output/game
          cmd=command(sys.executable,game,out,worker['cpus'])
          atomic(root/f'{game}_command.json',cmd)
          env=dict(os.environ,CUDA_VISIBLE_DEVICES=worker['gpu'],
              OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='8',MKL_NUM_THREADS='1',
              XLA_PYTHON_CLIENT_PREALLOCATE='false',XLA_PYTHON_CLIENT_MEM_FRACTION='0.40',
              PYTHONHASHSEED='0',PAPER_DETERMINISTIC_UUID='1',PYTHONUNBUFFERED='1')
          log=(root/f'{game}.log').open('w')
          process=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
          active[key]={'game':game,'process':process,'log':log,'start':time.time()}
          print(f'START slot={key} game={game} gpu={worker["gpu"]} cpus={worker["cpus"]}',flush=True)
          persist()
        for key,item in list(active.items()):
          code=item['process'].poll()
          if code is None: continue
          item['log'].close()
          ckpt=output/item['game']/'ckpt/env_action_steps_000110000'
          complete=code==0 and (ckpt/'agent.pkl').is_file()
          state['results'][item['game']]={'exit_code':code,'complete':complete,
              'seconds':time.time()-item['start'],'checkpoint':str(ckpt)}
          del active[key]; persist()
          print(f'END game={item["game"]} exit={code} complete={complete}',flush=True)
        readings=[]
        for gpu in devices:
          row=subprocess.check_output(['nvidia-smi','-i',gpu,
              '--query-gpu=memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True)
          used,total,util=map(float,row.strip().split(','))
          readings.append({'gpu':gpu,'used_mib':used,'total_mib':total,'util':util})
          if used > .90*total: raise RuntimeError(f'GPU memory safety threshold: {gpu}')
        rows=[x.split() for x in subprocess.check_output(['ps','-eo','pid=,ppid=,rss='],text=True).splitlines()]
        owned={v['process'].pid for v in active.values()}
        for _ in range(8): owned.update(int(r[0]) for r in rows if int(r[1]) in owned)
        rss=sum(int(r[2]) for r in rows if int(r[0]) in owned)/1024
        telemetry.write(json.dumps({'time':time.time(),'gpus':readings,'rss_mib':rss,'active':len(active)})+'\n')
        telemetry.flush()
        if rss > 210*1024: raise RuntimeError('Host RSS safety threshold exceeded')
        if queue or active: time.sleep(5)
    state['status']='COMPLETE' if all(v['complete'] for v in state['results'].values()) else 'COMPLETE_WITH_FAILURES'
  except BaseException as exc:
    state['status']='INTERRUPTED'; state['error']=str(exc)
    raise
  finally:
    persist()
    for item in active.values():
      p=item['process']
      if p.poll() is None:
        try: os.killpg(p.pid,signal.SIGTERM)
        except ProcessLookupError: pass
    for item in active.values():
      try: item['process'].wait(timeout=15)
      except subprocess.TimeoutExpired:
        os.killpg(item['process'].pid,signal.SIGKILL); item['process'].wait()
      item['log'].close()
  if state['status']!='COMPLETE': raise RuntimeError(state['status'])


if __name__=='__main__': main(sys.argv[1])
