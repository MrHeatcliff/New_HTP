"""Freeze source and submit exactly one hypothesis-suite allocation."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

repo=Path(__file__).resolve().parents[1]
root=Path(tempfile.mkdtemp(prefix='reborn_hypotheses_',dir=repo/'production_runs'))
source=root/'source'; source.mkdir()
files=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard',
    'dreamerv3','embodied','corewm_eval','tests','scripts','requirements.txt','baselines.yaml',
    'paper_artifacts/persistence_research/REBORN_HYPOTHESIS_SUITE_VI.md'],cwd=repo,text=True).splitlines()
manifest={}
for name in sorted(set(files)):
  original=repo/name
  if not original.is_file(): continue
  target=source/name; target.parent.mkdir(parents=True,exist_ok=True)
  shutil.copy2(original,target)
  manifest[name]=hashlib.sha256(target.read_bytes()).hexdigest()
(root/'source_sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
job=subprocess.check_output(['sbatch','--parsable',str(source/'scripts/slurm_reborn_hypotheses.sh'),
    str(root)],cwd=repo,text=True).strip()
record=dict(job_id=job,root=str(root),arms=['full','low_rate','no_q','deterministic'],
    games=['boxing','up_n_down','frostbite','road_runner'],gpus=2,cpus_per_game=8)
(root/'submission.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
