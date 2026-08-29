"""Production protocol freezing and immutable Wave-1 job execution."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from .config import ATARI100K_GAMES, FINAL_CHECKPOINTS, TRAINING_SEEDS
from .phase3_prepare import PROTOCOL_ID


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'paper_artifacts/corewm_phase3/training_manifest.json'


def _sha256(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(*args):
  return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def _tree_hash(folder):
  digest = hashlib.sha256()
  for path in sorted(Path(folder).rglob('*.py')):
    digest.update(str(path.relative_to(folder)).encode())
    digest.update(path.read_bytes())
  return digest.hexdigest()


def require_clean_tree():
  status = _git('status', '--porcelain')
  if status:
    raise RuntimeError(f'Production requires a clean working tree:\n{status}')


def freeze_protocol(output):
  require_clean_tree()
  output = Path(output)
  output.mkdir(parents=True, exist_ok=False)
  manifest = json.loads(MANIFEST.read_text())
  if len(manifest) != 680:
    raise AssertionError(len(manifest))
  main = sum(row['variant'] != 'Reverse' for row in manifest)
  reverse = sum(row['variant'] == 'Reverse' for row in manifest)
  wave1 = sum(row['wave'] == 1 for row in manifest)
  assert (main, reverse, wave1) == (650, 30, 36)
  resolved = {}
  for variant in ('backbone', 'flat', 'rec_only', 'pdyn_only', 'full', 'reverse'):
    path = ROOT / f'paper_artifacts/corewm_phase3/resolved_{variant}.yaml'
    resolved[variant] = {'path': str(path.relative_to(ROOT)), 'sha256': _sha256(path)}
  conda = json.loads(subprocess.check_output(
      ['conda', 'list', '-n', 'htp', '--json'], text=True))
  packages = {row['name']: row['version'] for row in conda}
  record = {
      'protocol_id': PROTOCOL_ID,
      'git_commit': _git('rev-parse', 'HEAD'),
      'dirty_worktree': False,
      'python': platform.python_version(),
      'conda_environment': 'htp',
      'packages': packages,
      'resolved_configs': resolved,
      'manifest': str(MANIFEST.relative_to(ROOT)),
      'manifest_sha256': _sha256(MANIFEST),
      'canonical_games': list(ATARI100K_GAMES),
      'training_seeds': list(TRAINING_SEEDS),
      'interaction_budget': {
          'counter': 'env_action_steps',
          'definition': 'one action actually passed to env.step; reset does not count',
          'final': 100_000,
      },
      'checkpoint_milestones': list(FINAL_CHECKPOINTS),
      'evaluation_episodes': {'10000-90000': 10, '100000': 100},
      'hns_reference': {
          'path': 'baselines.yaml', 'sha256': _sha256(ROOT / 'baselines.yaml')},
      'evaluation_code_version': _tree_hash(ROOT / 'corewm_eval'),
      'training_matrix': {'main': main, 'reverse': reverse, 'total': len(manifest)},
      'wave1_jobs': wave1,
  }
  (output / 'protocol.json').write_text(
      json.dumps(record, indent=2, sort_keys=True) + '\n')
  return record


def _task_game(game):
  return {'jamesbond': 'james_bond'}.get(game, game)


def run_job(protocol_path, index, attempt):
  require_clean_tree()
  protocol_path = Path(protocol_path).resolve()
  protocol = json.loads(protocol_path.read_text())
  if protocol['protocol_id'] != PROTOCOL_ID:
    raise RuntimeError((protocol['protocol_id'], PROTOCOL_ID))
  commit = _git('rev-parse', 'HEAD')
  if commit != protocol['git_commit']:
    raise RuntimeError(f'Commit mismatch: {commit} != {protocol["git_commit"]}')
  if _sha256(MANIFEST) != protocol['manifest_sha256']:
    raise RuntimeError('Training manifest changed after protocol freeze.')
  jobs = json.loads(MANIFEST.read_text())
  job = jobs[int(index)]
  if job['job_index'] != int(index) or job['wave'] != 1:
    raise RuntimeError(f'Index {index} is not an approved Wave-1 job: {job}')
  attempt_id = f'attempt_{int(attempt):03d}'
  run_dir = (
      ROOT / 'production_runs' / PROTOCOL_ID / 'training' / 'wave1' /
      job['variant'].lower().replace('-', '_') / job['game'] /
      f'seed_{job["seed"]}' / attempt_id)
  if run_dir.exists():
    raise FileExistsError(
        f'{run_dir} already exists; never resume/overwrite a production attempt.')
  run_dir.mkdir(parents=True)
  launch = {
      **job, 'attempt_id': attempt_id, 'git_commit': commit,
      'protocol_path': str(protocol_path), 'status': 'LAUNCHING'}
  (run_dir / 'launch.json').write_text(json.dumps(launch, indent=2) + '\n')
  env = os.environ.copy()
  env.update({
      'PAPER_PROTOCOL_ID': PROTOCOL_ID,
      'PAPER_EXPERIMENT_ID': PROTOCOL_ID,
      'PAPER_ATTEMPT_ID': attempt_id,
      'PAPER_METHOD': job['variant'],
      'PAPER_CONDITION': job['variant'].lower().replace('-', '_'),
      'WANDB_ENTITY': job['wandb_entity'],
      'WANDB_PROJECT': job['wandb_project'],
      'WANDB_MODE': 'online',
      'WANDB_GROUP': f'{PROTOCOL_ID}-{job["game"]}',
      'WANDB_JOB_TYPE': job['variant'].lower().replace('-', '_'),
      'WANDB_RUN_NAME': (
          f'{PROTOCOL_ID}__{job["variant"].lower().replace("-", "_")}__'
          f'{job["game"]}__seed{job["seed"]}__{attempt_id}'),
      'WANDB_TAGS': ','.join((
          PROTOCOL_ID, 'production', 'wave1', job['variant'], job['game'],
          f'seed{job["seed"]}', attempt_id)),
  })
  module = 'dreamerv3.main' if job['variant'] == 'Backbone' else 'dreamerv3.main_htp'
  command = [
      sys.executable, '-m', module, '--configs', *job['configs'],
      '--task', f'atari100k_{_task_game(job["game"])}',
      '--seed', str(job['seed']), '--logdir', str(run_dir)]
  (run_dir / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
  try:
    result = subprocess.run(command, cwd=ROOT, env=env, check=False)
  except BaseException:
    launch['status'] = 'FAILED_TO_EXECUTE'
    (run_dir / 'launch.json').write_text(json.dumps(launch, indent=2) + '\n')
    raise
  launch['exit_code'] = result.returncode
  launch['status'] = 'TRAINING_EXITED' if result.returncode == 0 else 'FAILED'
  (run_dir / 'launch.json').write_text(json.dumps(launch, indent=2) + '\n')
  if result.returncode:
    raise SystemExit(result.returncode)


def _verify_protocol(protocol_path):
  require_clean_tree()
  protocol_path = Path(protocol_path).resolve()
  protocol = json.loads(protocol_path.read_text())
  if protocol['protocol_id'] != PROTOCOL_ID:
    raise RuntimeError((protocol['protocol_id'], PROTOCOL_ID))
  if _git('rev-parse', 'HEAD') != protocol['git_commit']:
    raise RuntimeError('Current commit differs from the frozen protocol.')
  if _sha256(MANIFEST) != protocol['manifest_sha256']:
    raise RuntimeError('Training manifest changed after protocol freeze.')
  return protocol_path, protocol


def submit_wave1_once(protocol_path, max_user_jobs=2):
  """Fill free Slurm slots without ever exceeding the user-wide job limit."""
  protocol_path, _ = _verify_protocol(protocol_path)
  root = protocol_path.parent
  ledger_path = root / 'wave1_submission_ledger.json'
  lock_path = root / 'wave1_submission.lock'
  lock_path.touch(exist_ok=True)
  with lock_path.open('r+') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    ledger = ({'submitted': []} if not ledger_path.exists()
              else json.loads(ledger_path.read_text()))
    submitted = {int(row['job_index']) for row in ledger['submitted']}
    jobs = json.loads(MANIFEST.read_text())
    wave1 = [row for row in jobs if row['wave'] == 1]
    assert len(wave1) == 36
    user = os.environ['USER']
    active = subprocess.check_output(
        ['squeue', '-h', '-u', user, '-o', '%A'], text=True).splitlines()
    slots = max(0, int(max_user_jobs) - len(active))
    new_rows = []
    (ROOT / 'slurm_logs').mkdir(exist_ok=True)
    for job in (row for row in wave1 if row['job_index'] not in submitted):
      if len(new_rows) >= slots:
        break
      name = f'cw1-{job["job_index"]:02d}-{job["variant"].lower().replace("-", "")}-{job["game"]}'
      command = [
          'sbatch', '--parsable', '--job-name', name,
          '--export', (
              f'ALL,COREWM_JOB_INDEX={job["job_index"]},COREWM_ATTEMPT=1'),
          str(ROOT / 'scripts/slurm_corewm_wave1.sh')]
      result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
      if result.returncode:
        # A concurrent user submission may consume the slot between squeue and
        # sbatch. Leave the manifest row unclaimed so the next poll retries it.
        if 'QOSMaxSubmitJobPerUserLimit' in result.stderr:
          break
        raise RuntimeError(result.stderr.strip())
      row = {
          'job_index': job['job_index'], 'slurm_job_id': result.stdout.strip(),
          'variant': job['variant'], 'game': job['game'], 'seed': job['seed'],
          'attempt': 1, 'submitted_unix_time': time.time()}
      ledger['submitted'].append(row)
      new_rows.append(row)
      submitted.add(job['job_index'])
      tmp = ledger_path.with_suffix('.tmp')
      tmp.write_text(json.dumps(ledger, indent=2, sort_keys=True) + '\n')
      tmp.replace(ledger_path)
    return {
        'active_user_jobs_before': len(active), 'available_slots': slots,
        'new_submissions': new_rows, 'submitted_wave1': len(submitted),
        'remaining_wave1': len(wave1) - len(submitted)}


def supervise_wave1(protocol_path, max_user_jobs=2, poll_seconds=60):
  """Submit the approved Wave 1 gradually, then stop before Wave 2."""
  while True:
    status = submit_wave1_once(protocol_path, max_user_jobs)
    print(json.dumps(status, sort_keys=True), flush=True)
    if status['remaining_wave1'] == 0:
      return
    time.sleep(poll_seconds)


def main(argv=None):
  parser = argparse.ArgumentParser()
  sub = parser.add_subparsers(dest='command', required=True)
  freeze = sub.add_parser('freeze')
  freeze.add_argument('--output', required=True)
  run = sub.add_parser('run-job')
  run.add_argument('--protocol', required=True)
  run.add_argument('--index', required=True, type=int)
  run.add_argument('--attempt', default=1, type=int)
  submit = sub.add_parser('submit-wave1-once')
  submit.add_argument('--protocol', required=True)
  submit.add_argument('--max-user-jobs', default=2, type=int)
  supervise = sub.add_parser('supervise-wave1')
  supervise.add_argument('--protocol', required=True)
  supervise.add_argument('--max-user-jobs', default=2, type=int)
  supervise.add_argument('--poll-seconds', default=60, type=int)
  args = parser.parse_args(argv)
  if args.command == 'freeze':
    print(json.dumps(freeze_protocol(args.output), indent=2))
  elif args.command == 'run-job':
    run_job(args.protocol, args.index, args.attempt)
  elif args.command == 'submit-wave1-once':
    print(json.dumps(submit_wave1_once(
        args.protocol, args.max_user_jobs), indent=2))
  else:
    supervise_wave1(args.protocol, args.max_user_jobs, args.poll_seconds)


if __name__ == '__main__':
  main()
