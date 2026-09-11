"""Freeze the working source and submit exactly one Slurm allocation."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

repo = Path(__file__).resolve().parents[1]
refresh = len(sys.argv) == 3 and sys.argv[1] == '--refresh-held'
if refresh:
  root = Path(sys.argv[2]).resolve()
  if root.parent != repo/'production_runs' or not root.name.startswith('reborn_four_game_'):
    raise ValueError('Expected an existing Reborn run root')
  submission = json.loads((root/'submission.json').read_text())
  state = subprocess.check_output(['squeue','-h','-j',submission['job_id'],'-o','%T|%r'],text=True).strip()
  if state != 'PENDING|JobHeldUser':
    raise RuntimeError(f'Refresh allowed only before allocation, with job held: {state}')
  if (root/'tests.log').exists() or (root/'manifest.json').exists():
    raise RuntimeError('Cannot change source after execution started')
  previous = root/'source_before_review'
  (root/'source').rename(previous)
  (root/'source_sha256.json').rename(root/'source_sha256_before_review.json')
else:
  if len(sys.argv) != 1:
    raise ValueError('Usage: submit_reborn.py [--refresh-held RUNROOT]')
  root = Path(tempfile.mkdtemp(prefix='reborn_four_game_',dir=repo/'production_runs'))
source = root/'source'; source.mkdir()
files = subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard',
    'dreamerv3','embodied','corewm_eval','tests','scripts','requirements.txt','baselines.yaml',
    'paper_artifacts/persistence_research/REBORN_PROTOCOL_VI.md'],cwd=repo,text=True).splitlines()
manifest = {}
for name in sorted(set(files)):
  original = repo/name
  if not original.is_file():
    continue
  target = source/name
  target.parent.mkdir(parents=True,exist_ok=True)
  shutil.copy2(original,target)
  manifest[name] = hashlib.sha256(target.read_bytes()).hexdigest()
(root/'source_sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
if refresh:
  print(f'Refreshed held job {submission["job_id"]}; preserved prior source in {previous}')
  sys.exit(0)
job = subprocess.check_output(['sbatch','--parsable',str(source/'scripts/slurm_reborn.sh'),str(root)],cwd=repo,text=True).strip()
(root/'submission.json').write_text(json.dumps({'job_id':job,'root':str(root),'branch':
    subprocess.check_output(['git','branch','--show-current'],cwd=repo,text=True).strip()},indent=2)+'\n')
print(json.dumps({'job_id':job,'root':str(root)},indent=2))
