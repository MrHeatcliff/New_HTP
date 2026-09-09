"""Slurm-only sequential versus concurrent two-game capacity benchmark."""
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time


def main():
  if not os.environ.get('SLURM_JOB_ID'):
    raise RuntimeError('Submit through Slurm')
  actions = int(os.environ.get('HTP_CAPACITY_ACTIONS', '5000'))
  if actions < 2048:
    raise ValueError('Action budget must exceed replay prefill')
  root = Path(sys.argv[1]); root.mkdir(parents=True, exist_ok=False)
  cpus = sorted(os.sched_getaffinity(0))
  if len(cpus) < 16:
    raise RuntimeError(f'Need 16 allocated CPUs, got {cpus}')
  device = os.environ.get('CUDA_VISIBLE_DEVICES', '')
  if not device or ',' in device:
    raise RuntimeError(f'Expected one allocated GPU, got {device!r}')
  inventory = subprocess.check_output(['nvidia-smi', '-i', device,
      '--query-gpu=uuid,name,memory.total', '--format=csv,noheader,nounits'], text=True)
  (root/'hardware.txt').write_text(inventory + '\nCPUs: ' + str(cpus))
  if 'H200' not in inventory:
    raise RuntimeError(inventory)
  env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='8',
      MKL_NUM_THREADS='1', XLA_PYTHON_CLIENT_PREALLOCATE='false',
      XLA_PYTHON_CLIENT_MEM_FRACTION='0.40', PYTHONUNBUFFERED='1',
      PYTHONHASHSEED='0', PAPER_DETERMINISTIC_UUID='1')
  # MEM_FRACTION is not treated as a hard cap when preallocation is disabled.
  results = {}

  def run_stage(stage, games):
    workers = []; samples = []; start = time.monotonic()
    failure = None
    try:
      for index, game in enumerate(games):
        out = root/f'{stage}_{game}'
        command = ['taskset','-c',','.join(map(str,cpus[index*8:index*8+8])),
            sys.executable,'-u','-m','dreamerv3.main_htp',
            '--configs','htp_atari100k','size25m', '--task',f'atari100k_{game}',
            '--seed','0','--env.atari100k.use_seed','True','--logdir',str(out),
            '--jax.prealloc','False','--logger.outputs','jsonl',
            '--run.log_policy_video','False','--run.steps',str(actions),
            '--run.exact_env_action_budget','True','--run.action_milestones',str(actions),
            '--agent.htp.persistence_metric','whitened',
            '--agent.htp.persistence_scale','0.1','--agent.htp.persistence_all_lags','True',
            '--agent.htp.persistence_isotropy','0.01',
            '--agent.htp.recon.first_target_window','1',
            '--agent.htp.pdyn.mask_episode_boundaries','False']
        (root/f'{stage}_{game}_command.json').write_text(json.dumps(command,indent=2))
        log = (root/f'{stage}_{game}.log').open('w')
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                env=env, start_new_session=True)
        workers.append((game,proc,log))
      with (root/f'{stage}_telemetry.jsonl').open('w') as telemetry:
        while any(p.poll() is None for _,p,_ in workers):
          gpu = subprocess.check_output(['nvidia-smi','-i',device,
              '--query-gpu=memory.used,memory.total,utilization.gpu,utilization.memory,power.draw',
              '--format=csv,noheader,nounits'],text=True).strip()
          used,total,util,memutil,power = map(float,gpu.split(','))
          processes = subprocess.check_output(['ps','-eo','pid=,ppid=,rss=,pcpu=,nlwp='],text=True)
          rows = [line.split() for line in processes.splitlines()]
          owned = {p.pid for _,p,_ in workers}
          for _ in range(8):
            owned.update(int(r[0]) for r in rows if int(r[1]) in owned)
          mine = [r for r in rows if int(r[0]) in owned]
          sample = dict(seconds=time.monotonic()-start,gpu_used_mib=used,
              gpu_total_mib=total,gpu_util=util,gpu_memory_util=memutil,power_w=power,
              process_rss_mib=sum(int(r[2]) for r in mine)/1024,
              process_cpu_percent=sum(float(r[3]) for r in mine),
              threads=sum(int(r[4]) for r in mine),
              active=sum(p.poll() is None for _,p,_ in workers))
          samples.append(sample); telemetry.write(json.dumps(sample)+'\n'); telemetry.flush()
          if used > .90*total or sample['process_rss_mib'] > 100*1024:
            raise RuntimeError('Safety threshold exceeded (90% GPU or 100 GiB summed RSS)')
          if any(p.poll() not in (None,0) for _,p,_ in workers):
            raise RuntimeError('Training child failed; inspect per-game logs')
          time.sleep(2)
      if any(p.returncode != 0 for _,p,_ in workers):
        raise RuntimeError('Training child failed')
    except Exception as exc:
      failure = str(exc)
    finally:
      for _,p,_ in workers:
        if p.poll() is None:
          os.killpg(p.pid,signal.SIGTERM)
      for _,p,log in workers:
        try: p.wait(timeout=15)
        except subprocess.TimeoutExpired:
          os.killpg(p.pid,signal.SIGKILL); p.wait()
        log.close()
    elapsed = time.monotonic()-start
    result = dict(games=games,elapsed_seconds=elapsed,error=failure,
        exit_codes={game:p.returncode for game,p,_ in workers},
        gpu_peak_mib=max((s['gpu_used_mib'] for s in samples),default=0),
        gpu_total_mib=samples[0]['gpu_total_mib'] if samples else 0,
        gpu_util_mean=statistics.mean(s['gpu_util'] for s in samples) if samples else 0,
        rss_peak_mib=max((s['process_rss_mib'] for s in samples),default=0),
        cpu_percent_mean=statistics.mean(s['process_cpu_percent'] for s in samples) if samples else 0)
    results[stage]=result
    (root/'results.json').write_text(json.dumps(results,indent=2))
    print(json.dumps({stage:result}),flush=True)
    if failure: raise RuntimeError(failure)
    return result

  if os.environ.get('HTP_CAPACITY_PAIRED_ONLY') == '1':
    gate = json.loads(Path(os.environ['HTP_CAPACITY_GATE']).read_text())
    if gate['decision']['concurrency'] != 2 or gate['pair']['error'] is not None:
      raise RuntimeError('Two-game mode requires a successful capacity benchmark')
    run_stage('pair',['alien','breakout'])
    return
  alien=run_stage('solo_alien',['alien'])
  breakout=run_stage('solo_breakout',['breakout'])
  if alien['gpu_peak_mib'] + breakout['gpu_peak_mib'] > .80*alien['gpu_total_mib']:
    results['decision']={'concurrency':1,'reason':'Projected pair exceeds 80% VRAM'}
  elif alien['rss_peak_mib'] + breakout['rss_peak_mib'] > 80*1024:
    results['decision']={'concurrency':1,'reason':'Projected pair exceeds 80 GiB summed RSS'}
  else:
    pair=run_stage('pair',['alien','breakout'])
    speedup=(alien['elapsed_seconds']+breakout['elapsed_seconds'])/pair['elapsed_seconds']
    results['decision']={'concurrency':2 if speedup > 1.05 else 1,
        'aggregate_wall_time_speedup':speedup,
        'reason':'Choose two only if total completion time improves by >5%',
        'limitations':'Short fresh-training benchmark including compilation; not a long-run capacity guarantee. No test above concurrency two.'}
  (root/'results.json').write_text(json.dumps(results,indent=2))
  print(json.dumps(results['decision']),flush=True)


if __name__ == '__main__':
  main()
