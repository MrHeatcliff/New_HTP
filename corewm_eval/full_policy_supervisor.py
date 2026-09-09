"""Rolling two-slot supervisor for Full isolated policy evaluations."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from .full_policy_stage import CONTROL_ROOT, EVAL_ROOT, _atomic, _load, submit_once


def _queue():
  rows=subprocess.check_output(['squeue','-h','-u',os.environ['USER'],'-o','%A|%j|%T'],text=True).splitlines()
  return {x[0]:{'name':x[1],'state':x[2]} for x in (row.split('|',2) for row in rows)}


def _attempt(row):
  root=EVAL_ROOT/row['game']/f'seed_{row["training_seed"]}'
  candidates=sorted(root.glob('attempt_*')) if root.exists() else []
  for path in reversed(candidates):
    launch=path/'launch.json'
    if launch.exists(): return path
  return None


def reconcile():
  ledger_path=CONTROL_ROOT/'submission_ledger.json';lock_path=CONTROL_ROOT/'submission.lock'
  queue=_queue()
  with lock_path.open('r+') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX);ledger=_load(ledger_path);changed=False;blocked=[]
    for row in ledger['submitted']:
      if not row.get('active_claim',True):continue
      if str(row['slurm_job_id']) in queue:continue
      attempt=_attempt(row)
      if attempt is None:
        row['settle_polls']=int(row.get('settle_polls',0))+1;changed=True
        if row['settle_polls']<4:continue
        row.update(active_claim=False,status='TECHNICAL_FAILED',failure_reason='missing evaluation attempt artifacts');blocked.append(row);continue
      launch=_load(attempt/'launch.json')
      if launch.get('status')=='VALID_COMPLETE':
        row.update(active_claim=False,status='VALID_COMPLETE',attempt=str(attempt),validated_unix_time=time.time());changed=True
      else:
        row['settle_polls']=int(row.get('settle_polls',0))+1;changed=True
        if row['settle_polls']<4:continue
        row.update(active_claim=False,status='TECHNICAL_FAILED',attempt=str(attempt),failure_reason=f'evaluator exited with status {launch.get("status")}');blocked.append(row)
    if changed:_atomic(ledger_path,ledger)
    return {'blocked':blocked,
      'valid':sum(row.get('status')=='VALID_COMPLETE' for row in ledger['submitted']),
      'active_claims':sum(row.get('active_claim',True) for row in ledger['submitted']),
      'queue_own':sum(item['name'].startswith('cs1-eval-full-') for item in queue.values()),
      'queue_unrelated_ignored':sum(not item['name'].startswith('cs1-eval-full-') for item in queue.values())}


def supervise(manifest,worktree,interval,max_jobs):
  CONTROL_ROOT.mkdir(parents=True,exist_ok=True)
  process_lock=CONTROL_ROOT/'supervisor.lock';process_lock.touch(exist_ok=True)
  heartbeat=CONTROL_ROOT/'supervisor_status.json'
  with process_lock.open('r+') as lock:
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError as exc:raise RuntimeError('Full policy supervisor already running') from exc
    while True:
      state=reconcile();state.update(pid=os.getpid(),status='RUNNING',updated_unix_time=time.time(),max_concurrent_own_jobs=max_jobs)
      if state['blocked']:
        state['status']='BLOCKED_TECHNICAL_FAILURE';_atomic(heartbeat,state);raise RuntimeError('Policy evaluator technical failure')
      if state['valid']>=129:
        state['status']='FULL_POLICY_EVALUATION_COMPLETE';_atomic(heartbeat,state);return
      state['last_submission']=submit_once(manifest,worktree,max_jobs);_atomic(heartbeat,state);time.sleep(interval)


def main(argv=None):
  p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--worktree',required=True);p.add_argument('--interval',type=int,default=30);p.add_argument('--max-jobs',type=int,default=2);a=p.parse_args(argv)
  if a.max_jobs!=2:raise ValueError('Production authorization is frozen at max_jobs=2')
  supervise(a.manifest,a.worktree,a.interval,a.max_jobs)


if __name__=='__main__':main()
