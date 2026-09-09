"""Isolated Full-stage policy evaluation and rolling Slurm orchestration."""

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from .config import ATARI100K_GAMES, FINAL_CHECKPOINTS, TRAINING_SEEDS
from .hns_reference import load_reference
from .method_stage import FROZEN_COMMIT, PRODUCTION, ROOT, _task_game, _verify_clean_worktree
from .policy import human_normalized_score, normalized_auc
from .production_validate import checkpoint_tree_hash


EVAL_ROOT = PRODUCTION / 'evaluations' / 'stage1_full' / 'full'
CONTROL_ROOT = PRODUCTION / 'stages' / 'full_policy_eval'
EVALUATION_SEED = 0


def _load(path):
  return json.loads(Path(path).read_text())


def _atomic(path, value):
  path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + '.tmp')
  temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
  temporary.replace(path)


def _sha256(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _full_training_runs():
  found = {}
  roots = [PRODUCTION / 'training/wave1/full', PRODUCTION / 'training/stage1_full']
  for root in roots:
    if not root.exists(): continue
    for launch_path in root.glob('*/*/attempt_*/launch.json'):
      launch = _load(launch_path)
      if launch.get('variant') != 'Full' or launch.get('status') != 'VALID_COMPLETE':
        continue
      key = (launch['game'], int(launch['seed']))
      if key in found: raise RuntimeError(f'Duplicate valid Full training run {key}')
      found[key] = launch_path.parent
  expected = {(game, seed) for game in ATARI100K_GAMES for seed in TRAINING_SEEDS}
  if set(found) != expected: raise AssertionError((expected - set(found), set(found) - expected))
  return found


def _validate_existing_alien_seed0(training):
  summary_path = PRODUCTION / 'evaluations/wave1/full/alien/seed_0/policy_summary.json'
  summary = _load(summary_path)
  rows = summary['rows']
  if [row['checkpoint'] for row in rows] != list(FINAL_CHECKPOINTS):
    raise AssertionError('Existing Alien seed0 policy milestones differ')
  if [row['episodes'] for row in rows] != [10] * 9 + [100]:
    raise AssertionError('Existing Alien seed0 episode counts differ')
  manifest = _load(training / 'paper_artifacts/action_checkpoints_manifest.json')
  expected_hash = {row['milestone']: row['checkpoint_hash'] for row in manifest}
  for row in rows:
    if row['checkpoint_hash'] != expected_hash[row['checkpoint']]:
      raise AssertionError('Existing Alien seed0 checkpoint hash differs')
    if len(row['returns']) != row['episodes'] or not all(
        math.isfinite(float(value)) for value in row['returns']):
      raise AssertionError('Existing Alien seed0 return validation failed')
  return summary_path


def generate_manifest(output):
  training = _full_training_runs()
  existing = _validate_existing_alien_seed0(training[('alien', 0)])
  jobs = []
  for game in ATARI100K_GAMES:
    for seed in TRAINING_SEEDS:
      if (game, seed) == ('alien', 0): continue
      run = training[(game, seed)]
      checkpoints = _load(run / 'paper_artifacts/action_checkpoints_manifest.json')
      if [row['milestone'] for row in checkpoints] != list(FINAL_CHECKPOINTS):
        raise AssertionError((game, seed, 'checkpoint schedule'))
      jobs.append({
          'ordinal': len(jobs), 'method': 'Full', 'game': game, 'training_seed': seed,
          'evaluation_seed': EVALUATION_SEED, 'training_run': str(run),
          'training_launch_hash': _sha256(run / 'launch.json'),
          'checkpoints': [{
              'milestone': row['milestone'], 'checkpoint': row['checkpoint'],
              'checkpoint_hash': row['checkpoint_hash'],
              'episodes': 100 if row['milestone'] == 100_000 else 10}
              for row in checkpoints]})
  if len(jobs) != 129: raise AssertionError(len(jobs))
  payload = {
      'protocol_id': 'corewm_atari100k_v1', 'git_commit': FROZEN_COMMIT,
      'method': 'Full', 'required_policy_curves': 130,
      'existing_valid_policy_curves': 1, 'remaining_policy_curves': 129,
      'evaluation_seed': EVALUATION_SEED, 'existing': str(existing), 'jobs': jobs}
  _atomic(output, payload)
  return payload


def _next_attempt(game, seed):
  root = EVAL_ROOT / game / f'seed_{seed}'
  values = []
  for path in root.glob('attempt_*') if root.exists() else ():
    try: values.append(int(path.name.rsplit('_', 1)[1]))
    except ValueError: pass
  return max(values, default=0) + 1


def _episode_returns(outdir):
  score_file = outdir / 'scores.jsonl'
  returns = [float(json.loads(line)['episode/score'])
             for line in score_file.read_text().splitlines() if line.strip()]
  if not all(math.isfinite(value) for value in returns):
    raise AssertionError('Non-finite evaluation return')
  return returns


def run_job(manifest_path, ordinal, worktree):
  worktree = _verify_clean_worktree(worktree)
  manifest = _load(manifest_path)
  if manifest['remaining_policy_curves'] != 129: raise AssertionError('Bad eval manifest')
  job = manifest['jobs'][int(ordinal)]
  attempt_number = _next_attempt(job['game'], job['training_seed'])
  attempt_id = f'attempt_{attempt_number:03d}'
  run_dir = EVAL_ROOT / job['game'] / f'seed_{job["training_seed"]}' / attempt_id
  run_dir.mkdir(parents=True)
  launch_path = run_dir / 'launch.json'
  launch = {**job, 'attempt_id': attempt_id, 'status': 'EVALUATING',
            'protocol_id': manifest['protocol_id'], 'git_commit': FROZEN_COMMIT,
            'training_worktree': str(worktree), 'started_unix_time': time.time()}
  _atomic(launch_path, launch)
  rows = []
  reference = load_reference(ROOT / 'baselines.yaml', 'atari57_gamer')[job['game']]
  random_score, human_score = reference
  for checkpoint_row in job['checkpoints']:
    milestone = checkpoint_row['milestone']; checkpoint = Path(checkpoint_row['checkpoint'])
    before = checkpoint_tree_hash(checkpoint)
    if before != checkpoint_row['checkpoint_hash']:
      raise AssertionError(f'Pre-evaluation checkpoint hash mismatch: {checkpoint}')
    outdir = run_dir / f'{milestone:06d}'
    outdir.mkdir()
    env = os.environ.copy()
    env.update({
        'PAPER_PROTOCOL_ID': manifest['protocol_id'],
        'PAPER_EXPERIMENT_ID': 'corewm_atari100k_v1_policy_eval',
        'PAPER_ATTEMPT_ID': f'{attempt_id}_eval_{milestone:06d}',
        'PAPER_LOGGER_FIX_REVISION': '1', 'PAPER_METHOD': 'Full',
        'PAPER_CONDITION': 'full', 'PAPER_TRAINING_SEED': str(job['training_seed']),
        'PAPER_EVALUATION_SEED': str(EVALUATION_SEED), 'WANDB_MODE': 'disabled',
        'PYTHONPATH': str(worktree)})
    command = [str(Path(sys.executable)), '-m', 'dreamerv3.main_htp', '--configs',
        'atari100k', 'size12m', 'corewm_paper_protocol', 'wandb', 'corewm_full',
        '--script', 'eval_only', '--task', f'atari100k_{_task_game(job["game"])}',
        '--seed', str(EVALUATION_SEED), '--logdir', str(outdir),
        '--run.from_checkpoint', str(checkpoint), '--run.steps', '1000000000',
        '--run.envs', '1', '--run.eval_eps', str(checkpoint_row['episodes']),
        '--run.log_every', '1000', '--logger.outputs', 'jsonl',
        '--jax.prealloc', 'False', '--env.atari100k.use_seed', 'True']
    _atomic(outdir / 'evaluation_request.json', {
        'command': command, 'game': job['game'], 'training_seed': job['training_seed'],
        'evaluation_seed': EVALUATION_SEED, **checkpoint_row})
    result = subprocess.run(command, cwd=worktree, env=env, check=False)
    if result.returncode: raise RuntimeError(f'Evaluator exited {result.returncode} at {milestone}')
    returns = _episode_returns(outdir)
    if len(returns) != checkpoint_row['episodes']:
      raise AssertionError((milestone, len(returns), checkpoint_row['episodes']))
    final = _load(outdir / 'paper_artifacts/final_eval.json')
    if final.get('status') != 'complete' or Path(final['checkpoint_path']) != checkpoint:
      raise AssertionError(f'Wrong evaluator completion/checkpoint at {milestone}')
    after = checkpoint_tree_hash(checkpoint)
    if after != before: raise AssertionError(f'Evaluation mutated checkpoint: {checkpoint}')
    mean = sum(returns) / len(returns)
    rows.append({
        'method': 'Full', 'game': job['game'], 'seed': job['training_seed'],
        'evaluation_seed': EVALUATION_SEED, 'checkpoint': milestone,
        'checkpoint_hash': before, 'episodes': len(returns), 'returns': returns,
        'raw_return': mean,
        'HNS': float(human_normalized_score(mean, random_score, human_score))})
    _atomic(outdir / 'validation.json', {
        'status': 'VALID_COMPLETE', 'checkpoint_hash_before': before,
        'checkpoint_hash_after': after, 'episode_count': len(returns),
        'all_returns_finite': True, 'training_writes': False})
  auc = float(normalized_auc(FINAL_CHECKPOINTS, [row['HNS'] for row in rows]))
  summary = {
      'protocol_id': manifest['protocol_id'], 'git_commit': FROZEN_COMMIT,
      'method': 'Full', 'game': job['game'], 'training_seed': job['training_seed'],
      'evaluation_seed': EVALUATION_SEED, 'attempt_id': attempt_id,
      'random_score': random_score, 'human_score': human_score, 'rows': rows,
      'normalized_auc': auc, 'all_episode_counts_exact': True,
      'all_returns_finite': True, 'all_checkpoint_hashes_unchanged': True,
      'evaluation_writes_to_training_state': False}
  _atomic(run_dir / 'policy_summary.json', summary)
  with (run_dir / 'policy_curve.csv').open('w', newline='') as stream:
    fields = ('method','game','seed','evaluation_seed','checkpoint','checkpoint_hash',
              'episodes','raw_return','HNS')
    writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
    writer.writerows({key: row[key] for key in fields} for row in rows)
  launch.update(status='VALID_COMPLETE', completed_unix_time=time.time(),
                normalized_auc=auc, final_return=rows[-1]['raw_return'],
                final_checkpoint_hash=rows[-1]['checkpoint_hash'])
  _atomic(launch_path, launch)
  training_launch = Path(job['training_run']) / 'launch.json'
  training_meta = _load(training_launch)
  training_meta['policy_eval_status'] = 'VALID_COMPLETE'
  training_meta['policy_eval_summary'] = str(run_dir / 'policy_summary.json')
  _atomic(training_launch, training_meta)
  print(json.dumps({'status':'VALID_COMPLETE','run_dir':str(run_dir),'auc':auc}, indent=2))


def submit_once(manifest_path, worktree, max_jobs=2):
  _verify_clean_worktree(worktree)
  manifest = _load(manifest_path)
  CONTROL_ROOT.mkdir(parents=True, exist_ok=True)
  ledger_path = CONTROL_ROOT / 'submission_ledger.json'; lock_path = CONTROL_ROOT/'submission.lock'
  lock_path.touch(exist_ok=True)
  with lock_path.open('r+') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    ledger = {'submitted': []} if not ledger_path.exists() else _load(ledger_path)
    queue = subprocess.check_output(['squeue','-h','-u',os.environ['USER'],'-o','%A|%j'],text=True).splitlines()
    active = [line for line in queue if line.partition('|')[2].startswith('cs1-eval-full-')]
    slots = max(0, int(max_jobs)-len(active))
    claimed = {int(row['ordinal']) for row in ledger['submitted']
               if row.get('active_claim',True) or row.get('status')=='VALID_COMPLETE'}
    new=[]
    for job in manifest['jobs']:
      ordinal=job['ordinal']
      if ordinal in claimed or len(new)>=slots: continue
      name=f'cs1-eval-full-{job["game"]}-s{job["training_seed"]}'
      command=['sbatch','--parsable','--job-name',name,'--partition','gpu_junior',
          '--export',f'ALL,COREWM_EVAL_ORDINAL={ordinal},COREWM_EVAL_MANIFEST={Path(manifest_path).resolve()},COREWM_CLEAN_WORKTREE={Path(worktree).resolve()}',
          str(ROOT/'scripts/slurm_corewm_full_policy_eval.sh')]
      result=subprocess.run(command,cwd=ROOT,text=True,capture_output=True)
      if result.returncode: raise RuntimeError(result.stderr.strip())
      row={'ordinal':ordinal,'game':job['game'],'training_seed':job['training_seed'],
           'slurm_job_id':result.stdout.strip(),'active_claim':True,
           'submitted_unix_time':time.time()}
      ledger['submitted'].append(row);new.append(row);claimed.add(ordinal);_atomic(ledger_path,ledger)
    return {'active_before':len(active),'new_submissions':new,
            'valid_or_active':len(claimed),'remaining':129-len(claimed)}


def main(argv=None):
  parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
  p=sub.add_parser('generate');p.add_argument('--output',required=True)
  p=sub.add_parser('run-job');p.add_argument('--manifest',required=True);p.add_argument('--ordinal',type=int,required=True);p.add_argument('--worktree',required=True)
  p=sub.add_parser('submit-once');p.add_argument('--manifest',required=True);p.add_argument('--worktree',required=True);p.add_argument('--max-jobs',type=int,default=2)
  args=parser.parse_args(argv)
  if args.command=='generate': print(json.dumps(generate_manifest(args.output),indent=2))
  elif args.command=='run-job': run_job(args.manifest,args.ordinal,args.worktree)
  else: print(json.dumps(submit_once(args.manifest,args.worktree,args.max_jobs),indent=2))


if __name__=='__main__': main()
