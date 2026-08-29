"""Read-only evaluation of actual HTP loss code paths on stored smoke replay."""

import argparse
import json
from pathlib import Path

import elements
import jax
import ninjax as nj
import numpy as np
import ruamel.yaml as yaml

from dreamerv3 import main_htp

from .extraction import previous_actions


def run(root, output):
  root = Path(root)
  raw = yaml.YAML(typ='safe').load((root / 'config.yaml').read_text())
  raw['jax']['precompile'] = False
  raw['jax']['prealloc'] = False
  raw['jax']['transfer_guard'] = False
  config = elements.Config(raw)
  agent = main_htp.make_agent(config)
  latest = (root / 'ckpt/latest').read_text().strip()
  cp = elements.Checkpoint(); cp.agent = agent
  cp.load(root / 'ckpt' / latest, keys=['agent'])
  replay_file = next((root / 'replay').glob('*.npz'))
  with np.load(replay_file) as data:
    obs = {key: np.asarray(data[key][:64])[None] for key in (
        'image', 'reward', 'is_first', 'is_last', 'is_terminal')}
    prevact = {'action': previous_actions(data['action'][:64])[None]}
  carry = agent.model.init_train(1)[:3]
  def lossfn(carry, obs, prevact):
    return agent.model.loss(
        carry, obs, prevact, training=False)[1][2]['losses']
  _, losses = nj.pure(lossfn)(
      agent.save()['params'], carry, obs, prevact,
      seed=np.asarray([0, 0], np.uint32), create=False, modify=True, ignore=True)
  values = {key: float(np.asarray(value).mean()) for key, value in
            jax.tree.map(np.asarray, losses).items() if key.startswith('htp_')}
  if not all(np.isfinite(value) and value > 0 for value in values.values()):
    raise AssertionError(values)
  payload = {'root': str(root), 'replay': str(replay_file),
             'losses': values, 'read_only': True}
  output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
  print(json.dumps(payload, sort_keys=True))
  return payload


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--root', required=True)
  parser.add_argument('--output', required=True)
  args = parser.parse_args()
  run(args.root, args.output)
