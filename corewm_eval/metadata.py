import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path):
  digest = hashlib.sha256()
  with Path(path).open('rb') as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b''):
      digest.update(chunk)
  return digest.hexdigest()


def git_commit():
  return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()


def write_metadata(path, **fields):
  payload = {
      'git_commit': git_commit(),
      'evaluation_timestamp': datetime.now(timezone.utc).isoformat(),
      'evaluation_code_version': 'phase2-v1',
      **fields,
  }
  Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
  return payload
