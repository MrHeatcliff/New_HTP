from corewm_eval.component_ablation import ARMS, command
from corewm_eval import component_ablation


def test_factorial_commands_change_only_two_toggles():
  configs = {}
  for arm in ARMS:
    cmd = command('boxing', arm, '/tmp/same-output')
    assert '--run.from_checkpoint' not in cmd
    for flag in ('--agent.htp.use_recon', '--agent.htp.use_pdyn'):
      index = cmd.index(flag)
      cmd[index + 1] = 'MASKED'
    configs[arm] = cmd
  assert all(cmd == configs['flat'] for cmd in configs.values())
  assert set(ARMS.values()) == {(False, False), (True, False), (False, True), (True, True)}


def test_direct_h_only_prediction_changes(monkeypatch):
  monkeypatch.setattr(component_ablation, 'MODE', 'direct_h')
  monkeypatch.setattr(component_ablation, 'ARMS', {'h_control': (False, False), 'h_pdyn': (False, True)})
  a = command('boxing', 'h_control', '/tmp/same')
  b = command('boxing', 'h_pdyn', '/tmp/same')
  assert a[a.index('--agent.htp.use_proj') + 1] == 'False'
  assert a[a.index('--agent.htp.use_recon') + 1] == 'False'
  assert a[a.index('--agent.htp.grad_to_backbone') + 1] == 'True'
  b[b.index('--agent.htp.use_pdyn') + 1] = 'False'
  assert a == b


def test_direct_h_constraint_package(monkeypatch):
  monkeypatch.setattr(component_ablation, 'MODE', 'direct_h_constraint')
  monkeypatch.setattr(component_ablation, 'ARMS', {'h_pdyn': (False, True), 'h_pdyn_constraint': (False, True)})
  a = command('boxing', 'h_pdyn', '/tmp/same')
  b = command('boxing', 'h_pdyn_constraint', '/tmp/same')
  assert b[b.index('--agent.htp.use_proj') + 1] == 'False'
  assert b[b.index('--agent.htp.use_recon') + 1] == 'False'
  assert b[b.index('--agent.htp.use_pdyn') + 1] == 'True'
  assert b[b.index('--agent.htp.grad_to_backbone') + 1] == 'True'
  assert b[b.index('--agent.htp.persistence_metric') + 1] == 'whitened'
  assert b[b.index('--agent.htp.persistence_all_lags') + 1] == 'True'
  for flag, value in (('--agent.htp.persistence_scale', '0.1'),
                      ('--agent.htp.persistence_isotropy', '0.01')):
    assert b[b.index(flag) + 1] == value
    b[b.index(flag) + 1] = '0.0'
  assert b[:-4] == a
