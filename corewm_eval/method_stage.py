"""Method-by-method production execution without changing training semantics."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .config import ATARI100K_GAMES, TRAINING_SEEDS
from .phase3_prepare import PROTOCOL_ID
from .production_validate import mark_full_training_valid


ROOT = Path(__file__).resolve().parents[1]
MASTER_MANIFEST = ROOT / 'paper_artifacts/corewm_phase3/training_manifest.json'
PRODUCTION = ROOT / 'production_runs' / PROTOCOL_ID
STAGE_ROOT = PRODUCTION / 'stages' / 'full'
FROZEN_COMMIT = '5ee4f27a0ba7fd7bdbdcbd4823fb2d539f8b4b47'


def _sha256(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(worktree, *args):
  return subprocess.check_output(
      ['git', '-C', str(worktree), *args], text=True).strip()


def _load_json(path):
  return json.loads(Path(path).read_text())


def _task_game(game):
  return {'jamesbond': 'james_bond'}.get(game, game)


def _valid_full_runs():
  valid = {}
  roots = [
      PRODUCTION / 'training' / 'wave1' / 'full',
      PRODUCTION / 'training' / 'stage1_full']
  for root in roots:
    if not root.exists():
      continue
    for launch_path in root.glob('**/launch.json'):
      launch = _load_json(launch_path)
      if launch.get('variant') != 'Full' or launch.get('status') != 'VALID_COMPLETE':
        continue
      key = (launch['game'], int(launch['seed']))
      if key in valid:
        raise RuntimeError(f'Duplicate valid Full runs: {valid[key]} and {launch_path}')
      valid[key] = str(launch_path)
  return valid


def select_remaining_full(master, valid):
  full = [row for row in master if row['variant'] == 'Full']
  if len(full) != 130:
    raise AssertionError(len(full))
  expected = {(game, seed) for game in ATARI100K_GAMES for seed in TRAINING_SEEDS}
  observed = {(row['game'], int(row['seed'])) for row in full}
  if observed != expected:
    raise AssertionError((expected - observed, observed - expected))
  if set(valid) - expected:
    raise AssertionError(set(valid) - expected)
  order = {game: index for index, game in enumerate(ATARI100K_GAMES)}
  remaining = [row for row in full if (row['game'], int(row['seed'])) not in valid]
  remaining.sort(key=lambda row: (order[row['game']], int(row['seed'])))
  return full, remaining


def generate_manifest(output):
  protocol = _load_json(PRODUCTION / 'protocol.json')
  if protocol['git_commit'] != FROZEN_COMMIT:
    raise AssertionError(protocol['git_commit'])
  if _sha256(MASTER_MANIFEST) != protocol['manifest_sha256']:
    raise AssertionError('Frozen master manifest hash mismatch')
  master = _load_json(MASTER_MANIFEST)
  valid = _valid_full_runs()
  full, remaining = select_remaining_full(master, valid)
  if (len(full), len(valid), len(remaining)) != (130, 1, 129):
    raise AssertionError((len(full), len(valid), len(remaining)))
  if ('alien', 0) not in valid:
    raise AssertionError('Full/alien/seed0 is not registered VALID_COMPLETE')
  payload = {
      'protocol_id': PROTOCOL_ID,
      'stage': 1,
      'method': 'Full',
      'execution_order': ['Full', 'Pdyn-only', 'Rec-only', 'Flat', 'Backbone'],
      'required_full_runs': 130,
      'existing_valid_full_runs': 1,
      'remaining_full_runs': 129,
      'existing_valid': [
          {'game': game, 'seed': seed, 'launch': path}
          for (game, seed), path in sorted(valid.items())],
      'jobs': [{**row, 'training_status': 'PENDING',
                'policy_eval_status': 'PENDING',
                'representation_eval_status': 'PENDING'} for row in remaining],
  }
  output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
  return payload


def _verify_clean_worktree(worktree):
  worktree = Path(worktree).resolve()
  if _git(worktree, 'rev-parse', 'HEAD') != FROZEN_COMMIT:
    raise RuntimeError('Production worktree commit mismatch')
  status = _git(worktree, 'status', '--porcelain')
  if status:
    raise RuntimeError(f'Production worktree is dirty:\n{status}')
  protocol = _load_json(PRODUCTION / 'protocol.json')
  manifest = worktree / 'paper_artifacts/corewm_phase3/training_manifest.json'
  if _sha256(manifest) != protocol['manifest_sha256']:
    raise RuntimeError('Clean-worktree manifest differs from frozen protocol')
  return worktree


def _next_attempt(game, seed):
  candidates = [
      PRODUCTION / 'training' / 'wave1' / 'full' / game / f'seed_{seed}',
      PRODUCTION / 'training' / 'stage1_full' / game / f'seed_{seed}']
  attempts = []
  for root in candidates:
    for path in root.glob('attempt_*') if root.exists() else ():
      try:
        attempts.append(int(path.name.split('_')[-1]))
      except ValueError:
        pass
  return max(attempts, default=0) + 1


def run_job(stage_manifest, ordinal, worktree):
  worktree = _verify_clean_worktree(worktree)
  stage = _load_json(stage_manifest)
  if (stage['protocol_id'], stage['method'], len(stage['jobs'])) != (
      PROTOCOL_ID, 'Full', 129):
    raise RuntimeError('Invalid Stage-1 manifest')
  job = stage['jobs'][int(ordinal)]
  if job['variant'] != 'Full':
    raise RuntimeError(job)
  attempt = _next_attempt(job['game'], int(job['seed']))
  attempt_id = f'attempt_{attempt:03d}'
  run_dir = (PRODUCTION / 'training' / 'stage1_full' / job['game'] /
             f'seed_{job["seed"]}' / attempt_id)
  if run_dir.exists():
    raise FileExistsError(run_dir)
  run_dir.mkdir(parents=True)
  launch = {
      **job, 'stage': 1, 'attempt_id': attempt_id,
      'git_commit': FROZEN_COMMIT, 'training_worktree': str(worktree),
      'training_worktree_clean': True,
      'protocol_path': str(PRODUCTION / 'protocol.json'),
      'stage_manifest': str(Path(stage_manifest).resolve()),
      'status': 'LAUNCHING'}
  launch_path = run_dir / 'launch.json'
  launch_path.write_text(json.dumps(launch, indent=2) + '\n')

  env = os.environ.copy()
  condition = 'full'
  run_id = f'cs1f{int(job["job_index"]):03d}a{attempt:03d}'
  env.update({
      'PAPER_PROTOCOL_ID': PROTOCOL_ID,
      'PAPER_EXPERIMENT_ID': PROTOCOL_ID,
      'PAPER_ATTEMPT_ID': attempt_id,
      'PAPER_LOGGER_FIX_REVISION': '1',
      'PAPER_METHOD': 'Full', 'PAPER_CONDITION': condition,
      'PAPER_METHOD_STAGE': '1',
      'WANDB_ENTITY': job['wandb_entity'],
      'WANDB_PROJECT': job['wandb_project'], 'WANDB_MODE': 'offline',
      'WANDB_RUN_ID': run_id,
      'WANDB_GROUP': f'{PROTOCOL_ID}-{job["game"]}',
      'WANDB_JOB_TYPE': 'full',
      'WANDB_RUN_NAME': (
          f'{PROTOCOL_ID}__full__{job["game"]}__seed{job["seed"]}__{attempt_id}'),
      'WANDB_TAGS': ','.join((
          PROTOCOL_ID, 'production', 'stage1_full', 'Full', job['game'],
          f'seed{job["seed"]}', attempt_id)),
      'WANDB_DIR': str(PRODUCTION / 'wandb'),
      'PYTHONPATH': str(worktree),
  })
  python = Path(sys.executable)
  command = [
      str(python), '-m', 'dreamerv3.main_htp', '--configs', *job['configs'],
      '--task', f'atari100k_{_task_game(job["game"])}',
      '--seed', str(job['seed']), '--logdir', str(run_dir)]
  (run_dir / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
  try:
    result = subprocess.run(command, cwd=worktree, env=env, check=False)
  except BaseException:
    launch['status'] = 'FAILED_TO_EXECUTE'
    launch_path.write_text(json.dumps(launch, indent=2) + '\n')
    raise
  launch['exit_code'] = result.returncode
  launch['status'] = 'TRAINING_EXITED' if result.returncode == 0 else 'FAILED'
  launch['wandb_run_id'] = run_id
  matches = sorted((PRODUCTION / 'wandb' / 'wandb').glob(
      f'offline-run-*-{run_id}'))
  launch['wandb_offline_directory'] = str(matches[-1]) if matches else None
  launch_path.write_text(json.dumps(launch, indent=2) + '\n')
  if result.returncode:
    raise SystemExit(result.returncode)
  # Validation is deliberately outside the training process. It only reads the
  # completed artifacts and updates orchestration metadata after training exits.
  validation = mark_full_training_valid(run_dir)
  print(json.dumps({'run_dir': str(run_dir), 'validation': validation}, indent=2))


def validate_job(run_dir, skip_checkpoint_hashes=False):
  return mark_full_training_valid(
      run_dir, verify_hashes=not skip_checkpoint_hashes)


def submit_once(stage_manifest, worktree, max_jobs=2):
  _verify_clean_worktree(worktree)
  stage = _load_json(stage_manifest)
  if len(stage['jobs']) != 129:
    raise AssertionError(len(stage['jobs']))
  STAGE_ROOT.mkdir(parents=True, exist_ok=True)
  ledger_path, lock_path = STAGE_ROOT / 'submission_ledger.json', STAGE_ROOT / 'submission.lock'
  lock_path.touch(exist_ok=True)
  with lock_path.open('r+') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    ledger = {'submitted': []} if not ledger_path.exists() else _load_json(ledger_path)
    queue = subprocess.check_output(
        ['squeue', '-h', '-u', os.environ['USER'], '-o', '%A|%j'],
        text=True).splitlines()
    active = [line for line in queue if line.partition('|')[2].startswith('cs1-full-')]
    slots = max(0, int(max_jobs) - len(active))
    claimed = {int(row['ordinal']) for row in ledger['submitted']
               if (row.get('active_claim', True) or
                   row.get('status') == 'VALID_COMPLETE')}
    new = []
    (ROOT / 'slurm_logs').mkdir(exist_ok=True)
    for ordinal, job in enumerate(stage['jobs']):
      if ordinal in claimed or len(new) >= slots:
        continue
      name = f'cs1-full-{job["game"]}-s{job["seed"]}'
      command = [
          'sbatch', '--parsable', '--job-name', name,
          '--partition', 'gpu_junior',
          '--export', (
              f'ALL,COREWM_STAGE_ORDINAL={ordinal},'
              f'COREWM_STAGE_MANIFEST={Path(stage_manifest).resolve()},'
              f'COREWM_CLEAN_WORKTREE={Path(worktree).resolve()}'),
          str(ROOT / 'scripts/slurm_corewm_full_stage.sh')]
      result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
      if result.returncode:
        if ('QOSMaxSubmitJobPerUserLimit' in result.stderr or
            'QOSMaxGRESPerUser' in result.stderr):
          break
        raise RuntimeError(result.stderr.strip())
      row = {
          'ordinal': ordinal, 'job_index': job['job_index'],
          'slurm_job_id': result.stdout.strip(), 'game': job['game'],
          'seed': job['seed'], 'variant': 'Full',
          'active_claim': True,
          'submitted_unix_time': time.time()}
      ledger['submitted'].append(row); new.append(row); claimed.add(ordinal)
      tmp = ledger_path.with_suffix('.tmp')
      tmp.write_text(json.dumps(ledger, indent=2, sort_keys=True) + '\n')
      tmp.replace(ledger_path)
    return {
        'active_full_jobs_before': len(active),
        'unrelated_jobs_ignored': len(queue) - len(active),
        'available_slots': slots, 'new_submissions': new,
        'claimed_stage_jobs': len(claimed),
        'unclaimed_stage_jobs': 129 - len(claimed)}


def main(argv=None):
  parser = argparse.ArgumentParser()
  sub = parser.add_subparsers(dest='command', required=True)
  generate = sub.add_parser('generate')
  generate.add_argument('--output', required=True)
  run = sub.add_parser('run-job')
  run.add_argument('--manifest', required=True)
  run.add_argument('--ordinal', required=True, type=int)
  run.add_argument('--worktree', required=True)
  submit = sub.add_parser('submit-once')
  submit.add_argument('--manifest', required=True)
  submit.add_argument('--worktree', required=True)
  submit.add_argument('--max-jobs', type=int, default=2)
  validate = sub.add_parser('validate-job')
  validate.add_argument('--run-dir', required=True)
  validate.add_argument('--skip-checkpoint-hashes', action='store_true')
  args = parser.parse_args(argv)
  if args.command == 'generate':
    print(json.dumps(generate_manifest(args.output), indent=2))
  elif args.command == 'run-job':
    run_job(args.manifest, args.ordinal, args.worktree)
  elif args.command == 'validate-job':
    print(json.dumps(validate_job(
        args.run_dir, args.skip_checkpoint_hashes), indent=2))
  else:
    print(json.dumps(submit_once(
        args.manifest, args.worktree, args.max_jobs), indent=2))


if __name__ == '__main__':
  main()
