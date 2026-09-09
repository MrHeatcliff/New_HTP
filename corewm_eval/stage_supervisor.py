"""Conservative rolling supervisor for the Full production stage.

This process is orchestration-only: it never imports or mutates the training
model. It validates an exited attempt before it makes the next slot available,
and stops on any technical/validation failure.
"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from .method_stage import PRODUCTION, STAGE_ROOT, submit_once
from .production_validate import mark_full_training_valid


def _load(path):
  return json.loads(Path(path).read_text())


def _atomic(path, value):
  path = Path(path)
  temporary = path.with_suffix(path.suffix + '.tmp')
  temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
  temporary.replace(path)


def _queue_ids():
  output = subprocess.check_output(
      ['squeue', '-h', '-u', os.environ['USER'], '-o', '%A|%j|%T'],
      text=True)
  rows = [line.split('|', 2) for line in output.splitlines()]
  # Deliberately return every job for observability, but only cs1-full jobs are
  # acted on by matching their ledger IDs below.
  return {row[0]: {'name': row[1], 'state': row[2]} for row in rows}


def _attempt_dir(row):
  root = (PRODUCTION / 'training' / 'stage1_full' / row['game'] /
          f'seed_{row["seed"]}')
  candidates = sorted(root.glob('attempt_*')) if root.exists() else []
  matches = []
  for path in candidates:
    launch_path = path / 'launch.json'
    if not launch_path.exists():
      continue
    launch = _load(launch_path)
    if (int(launch.get('job_index', -1)) == int(row['job_index']) and
        int(launch.get('seed', -1)) == int(row['seed'])):
      matches.append(path)
  return matches[-1] if matches else None


def reconcile(ledger_path, lock_path):
  """Validate exited active claims and return a concise supervisor state."""
  queue = _queue_ids()
  with Path(lock_path).open('r+') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    ledger = _load(ledger_path)
    changed = False
    blocked = []
    for row in ledger['submitted']:
      if not row.get('active_claim', True):
        continue
      job_id = str(row['slurm_job_id'])
      if job_id in queue:
        continue
      attempt = _attempt_dir(row)
      if attempt is None:
        # Slurm may briefly remove a job before its final filesystem writes are
        # visible. Give that state several polling intervals before blocking.
        row['missing_attempt_polls'] = int(row.get('missing_attempt_polls', 0)) + 1
        changed = True
        if row['missing_attempt_polls'] >= 4:
          row.update(active_claim=False, status='TECHNICAL_FAILED',
                     failure_reason='no_attempt_artifacts_after_slurm_exit')
          blocked.append(row)
        continue
      launch = _load(attempt / 'launch.json')
      try:
        if launch.get('status') == 'VALID_COMPLETE':
          validation = launch['validation']
        elif launch.get('status') == 'TRAINING_EXITED' and launch.get('exit_code') == 0:
          validation = mark_full_training_valid(attempt)
        elif launch.get('status') in ('LAUNCHING', None):
          row['unfinished_launch_polls'] = int(row.get('unfinished_launch_polls', 0)) + 1
          changed = True
          if row['unfinished_launch_polls'] < 4:
            continue
          raise RuntimeError('Slurm exited but launch metadata remained unfinished')
        else:
          raise RuntimeError(
              f'training status={launch.get("status")} exit={launch.get("exit_code")}')
      except Exception as exc:
        failure = {
            'status': 'TECHNICAL_FAILED', 'reason': str(exc),
            'traceback': traceback.format_exc(), 'attempt': str(attempt),
            'slurm_job_id': job_id, 'time': time.time()}
        _atomic(attempt / 'supervisor_failure.json', failure)
        row.update(active_claim=False, status='TECHNICAL_FAILED',
                   failure_reason=str(exc), attempt=str(attempt))
        blocked.append(row); changed = True
        continue
      row.update(active_claim=False, status='VALID_COMPLETE',
                 attempt=str(attempt), validation=validation,
                 validated_unix_time=time.time())
      changed = True
    if changed:
      _atomic(ledger_path, ledger)
    valid_ordinals = {int(row['ordinal']) for row in ledger['submitted']
                      if row.get('status') == 'VALID_COMPLETE'}
    active = [row for row in ledger['submitted'] if row.get('active_claim', True)]
    return {
        'blocked': blocked, 'valid_remaining_jobs': len(valid_ordinals),
        'active_claims': len(active), 'queue_own_full': sum(
            item['name'].startswith('cs1-full-') for item in queue.values()),
        'queue_unrelated_ignored': sum(
            not item['name'].startswith('cs1-full-') for item in queue.values())}


def supervise(manifest, worktree, interval, max_jobs):
  STAGE_ROOT.mkdir(parents=True, exist_ok=True)
  ledger = STAGE_ROOT / 'submission_ledger.json'
  ledger_lock = STAGE_ROOT / 'submission.lock'
  process_lock = STAGE_ROOT / 'supervisor.lock'
  heartbeat = STAGE_ROOT / 'supervisor_status.json'
  process_lock.touch(exist_ok=True)
  with process_lock.open('r+') as lock:
    try:
      fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
      raise RuntimeError('A Full-stage supervisor is already running') from exc
    while True:
      state = reconcile(ledger, ledger_lock)
      state.update(pid=os.getpid(), status='RUNNING', updated_unix_time=time.time(),
                   max_concurrent_own_jobs=max_jobs)
      if state['blocked']:
        state['status'] = 'BLOCKED_TECHNICAL_FAILURE'
        _atomic(heartbeat, state)
        raise RuntimeError(
            f'Supervisor stopped on {len(state["blocked"])} technical failure(s)')
      if state['valid_remaining_jobs'] >= 129:
        state['status'] = 'TRAINING_MATRIX_COMPLETE_PENDING_STAGE_EVALUATION'
        _atomic(heartbeat, state)
        return
      submission = submit_once(manifest, worktree, max_jobs=max_jobs)
      state['last_submission'] = submission
      _atomic(heartbeat, state)
      time.sleep(interval)


def main(argv=None):
  parser = argparse.ArgumentParser()
  parser.add_argument('--manifest', required=True)
  parser.add_argument('--worktree', required=True)
  parser.add_argument('--interval', type=int, default=30)
  parser.add_argument('--max-jobs', type=int, default=2)
  args = parser.parse_args(argv)
  if args.max_jobs != 2:
    raise ValueError('Production authorization is frozen at exactly max_jobs=2')
  supervise(args.manifest, args.worktree, args.interval, args.max_jobs)


if __name__ == '__main__':
  main()
