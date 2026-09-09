import pytest
from corewm_eval.h200_suite import slots, command
from corewm_eval.config import ATARI100K_GAMES


def test_four_disjoint_eight_cpu_slots_two_per_gpu():
  assigned=slots(list(range(32,64)),['GPU-A','GPU-B'])
  assert [s['gpu'] for s in assigned]==['GPU-A','GPU-A','GPU-B','GPU-B']
  assert all(len(s['cpus'])==8 for s in assigned)
  assert len({c for s in assigned for c in s['cpus']})==32
  with pytest.raises(ValueError): slots(list(range(16)),['GPU-A','GPU-B'])
  with pytest.raises(ValueError): slots(list(range(32)),['GPU-A','GPU-A'])


def test_full_suite_command_budget_alias_and_memory():
  assert len(ATARI100K_GAMES)==len(set(ATARI100K_GAMES))==26
  cmd=command('python','jamesbond','/example',list(range(8)))
  assert cmd[2]=='0,1,2,3,4,5,6,7'
  assert cmd[cmd.index('--task')+1]=='atari100k_james_bond'
  assert cmd[cmd.index('--run.steps')+1]=='110000'
  start=cmd.index('--run.action_milestones')+1
  assert cmd[start:start+11]==[str(x) for x in range(10000,110001,10000)]
  assert cmd[cmd.index('--jax.prealloc')+1]=='False'
