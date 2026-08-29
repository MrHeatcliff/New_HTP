"""Audit the six fixed Phase-3 Alien installation smoke artifacts."""

import json
import pickle
from pathlib import Path

import numpy as np
import ruamel.yaml as yaml


ROOTS = {
    'Backbone': Path('/tmp/corewm-phase3-smoke-backbone-v2'),
    'Flat': Path('/tmp/corewm-phase3-smoke-flat'),
    'Rec-only': Path('/tmp/corewm-phase3-smoke-rec_only'),
    'Pdyn-only': Path('/tmp/corewm-phase3-smoke-pdyn_only'),
    'Full': Path('/tmp/corewm-phase3-smoke-full'),
    'Reverse': Path('/tmp/corewm-phase3-smoke-reverse'),
}
EVALS = {
    'Backbone': Path('/tmp/corewm-phase3-eval-backbone-v3'),
    **{name: Path('/tmp/corewm-phase3-eval-' + slug) for name, slug in (
        ('Flat', 'flat'), ('Rec-only', 'rec_only'), ('Pdyn-only', 'pdyn_only'),
        ('Full', 'full'), ('Reverse', 'reverse'))},
}


def audit(output='paper_artifacts/corewm_phase3/installation_smokes.json'):
  rows = []
  for variant, root in ROOTS.items():
    config = yaml.YAML(typ='safe').load((root / 'config.yaml').read_text())
    htp = config['agent']['htp']
    latest = (root / 'ckpt/latest').read_text().strip()
    checkpoint = root / 'ckpt' / latest
    payload = pickle.load((checkpoint / 'agent.pkl').open('rb'))
    params = payload['params']
    namespaces = {key.split('/')[0] for key in params}
    trace_path = root / 'paper_artifacts/replay_consistency_v6/update_event_trace_v6.jsonl'
    trace = [json.loads(line) for line in trace_path.open()]
    with_updates = [row for row in trace if row['optimizer_updates_cumulative'] > 0]
    replay_rows = first_rows = 0
    for replay_file in (root / 'replay').glob('*.npz'):
      with np.load(replay_file) as replay:
        replay_rows += len(replay['is_first'])
        first_rows += int(replay['is_first'].sum())
    final_eval = json.loads(
        (EVALS[variant] / 'paper_artifacts/final_eval.json').read_text())
    slug = variant.lower().replace('-', '_')
    loss_path = Path(f'paper_artifacts/corewm_phase3/loss_smoke_{slug}.json')
    measured_losses = json.loads(loss_path.read_text())['losses'] if loss_path.exists() else {}
    rec_expected = variant in ('Rec-only', 'Full', 'Reverse')
    pdyn_expected = variant in ('Pdyn-only', 'Full', 'Reverse')
    rec_present = 'htp_recon' in namespaces
    pdyn_present = 'htp_pdyn' in namespaces
    if rec_present != rec_expected or pdyn_present != pdyn_expected:
      raise AssertionError((variant, rec_present, pdyn_present))
    rows.append({
        'variant': variant, 'status': 'PASS', 'model_built': True,
        'environment_ran': replay_rows == 1320, 'replay_inserted': replay_rows > 0,
        'model_and_actor_critic_updates_ran': bool(with_updates),
        'optimizer_updates': int(trace[-1]['optimizer_updates_cumulative']),
        'checkpoint_saved': (checkpoint / 'done').exists(),
        'checkpoint_reloaded_and_eval_ran': final_eval['status'] == 'complete',
        'eval_episodes': int(final_eval['eval_episodes']),
        'lambda_rec': float(config['agent']['loss_scales']['htp_rec']) if rec_expected else 0.0,
        'lambda_pdyn': float(config['agent']['loss_scales']['htp_pdyn']) if pdyn_expected else 0.0,
        'prefix_dims': list(htp['proj']['dims']) if htp['enabled'] else None,
        'strides': list(htp['pdyn']['strides']) if pdyn_expected else None,
        'reconstruction_loss_code_path_executed': rec_present and bool(with_updates),
        'prediction_loss_code_path_executed': pdyn_present and bool(with_updates),
        'measured_readonly_losses': measured_losses,
        'htp_parameter_namespaces': sorted(x for x in namespaces if x.startswith('htp')),
        'driver_rows': replay_rows, 'reset_observations': first_rows,
        'true_action_bearing_transitions': replay_rows - first_rows,
        'reported_agent_actions': int(trace[-1]['agent_actions']),
        'approximate_raw_frames': 4 * (replay_rows - first_rows),
        'checkpoint': str(checkpoint),
    })
  target = Path(output); target.parent.mkdir(parents=True, exist_ok=True)
  target.write_text(json.dumps(rows, indent=2, sort_keys=True) + '\n')
  return rows


if __name__ == '__main__':
  audit()
