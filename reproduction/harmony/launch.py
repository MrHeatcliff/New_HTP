"""Portable, opt-in submission of the exact job-4080 source snapshot."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile

ARCHIVE_SHA256 = '86f6584eaec0889f7abf16bdbea7ba39e81aef62192ba80f75db7f47675e5f5c'


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--run-root', required=True, type=Path, help='New output directory on shared storage')
  parser.add_argument('--python', default=sys.executable, help='Absolute Python path visible on compute nodes')
  parser.add_argument('--partition', default='gpu_general')
  parser.add_argument('--qos', default='gpu_general_qos')
  parser.add_argument('--gres', default='gpu:nvidia_h200:2')
  parser.add_argument('--account')
  parser.add_argument('--dependency', help='Optional Slurm dependency, e.g. afterok:1234')
  parser.add_argument('--submit', action='store_true', help='Actually submit ONE job; otherwise only prepare')
  args = parser.parse_args()
  archive = Path(__file__).with_name('frozen_source.tar.gz')
  assert hashlib.sha256(archive.read_bytes()).hexdigest() == ARCHIVE_SHA256, 'Archive checksum mismatch'
  python = Path(args.python).expanduser().resolve()
  if not python.is_file(): parser.error('Python executable not found')
  if args.submit and not shutil.which('sbatch'): parser.error('sbatch not available')
  root = args.run_root.expanduser().resolve()
  root.mkdir(parents=True, exist_ok=False)  # Never overwrite/resume a previous experiment.
  with tarfile.open(archive, 'r:gz') as tar:
    members = tar.getmembers()
    for member in members:
      parts = Path(member.name).parts
      if not parts or parts[0] != 'source' or '..' in parts or Path(member.name).is_absolute():
        raise ValueError(f'Unsafe archive path: {member.name}')
      if not (member.isfile() or member.isdir()): raise ValueError('Links/devices forbidden')
    tar.extractall(root, members=members, filter='data')
  source = root/'source'
  hashes = {str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source.rglob('*') if p.is_file()}
  (root/'source_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
  # Do not use the historical shell script: its paths belong to the original server.
  invocation = [str(python), '-u', '-m', 'corewm_eval.harmony_multiseed', str(root)]
  wrap = ('export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8 '
          'XLA_PYTHON_CLIENT_PREALLOCATE=false; exec '+shlex.join(invocation))
  cmd = ['sbatch','--parsable','--job-name=harmony-seeds1to4',
         '--partition='+args.partition,'--nodes=1','--ntasks=1','--cpus-per-task=32',
         '--gres='+args.gres,'--mem=192G','--time=2-00:00:00',
         '--chdir='+str(source),'--output='+str(root/'slurm-%j.out'),
         '--error='+str(root/'slurm-%j.err')]
  if args.qos: cmd += ['--qos='+args.qos]
  if args.account: cmd += ['--account='+args.account]
  if args.dependency: cmd += ['--dependency='+args.dependency]
  cmd += ['--wrap',wrap]
  record = dict(status='PREPARED',root=str(root),command=cmd,archive_sha256=ARCHIVE_SHA256,
                seeds=[1,2,3,4],games=26,runs=104,evaluation_seed=0,
                gpus=2,concurrent_games=4,cpus_per_game=8)
  if args.submit:
    record.update(status='SUBMITTED',job_id=subprocess.check_output(cmd,text=True).strip())
  (root/'submission.json').write_text(json.dumps(record,indent=2)+'\n')
  print(json.dumps(record,indent=2))


if __name__ == '__main__': main()
