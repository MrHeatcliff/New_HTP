"""CPU-only preparation tests; never call sbatch or request a GPU."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class PreparationTest(unittest.TestCase):
  def test_prepare_and_frozen_seed_commands(self):
    launcher = Path(__file__).with_name('launch.py').resolve()
    with tempfile.TemporaryDirectory(prefix='harmony-package-test-') as tmp:
      root = Path(tmp)/'run'
      cmd = [sys.executable,str(launcher),'--run-root',str(root)]
      result = subprocess.run(cmd,check=True,capture_output=True,text=True)
      record = json.loads(result.stdout)
      self.assertEqual(record['status'],'PREPARED')
      self.assertNotIn('job_id',record)
      self.assertEqual(record['runs'],104)
      for name,expected in json.loads((root/'source_sha256.json').read_text()).items():
        self.assertEqual(hashlib.sha256((root/'source'/name).read_bytes()).hexdigest(),expected)
      check = '''from pathlib import Path
from corewm_eval.harmony_multiseed import train_command
from corewm_eval.h200_suite import slots
for seed in range(1,5):
    cmd = train_command('alien',Path('/unused'),seed)
    assert cmd[cmd.index('--seed')+1] == str(seed)
    assert cmd[cmd.index('--agent.reborn.harmony')+1] == 'True'
    assert cmd[cmd.index('--run.steps')+1] == '100000'
assert len(slots(list(range(32)),['gpu-a','gpu-b'])) == 4
print('PASS frozen scheduler seed/config/slot checks')
'''
      subprocess.run([sys.executable,'-c',check],cwd=root/'source',check=True)
      again = subprocess.run(cmd,capture_output=True,text=True)
      self.assertNotEqual(again.returncode,0)  # Existing runs must never be overwritten.


if __name__ == '__main__': unittest.main()
