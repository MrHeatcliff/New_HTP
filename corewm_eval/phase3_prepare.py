"""Resolve matched configs and emit the non-executing paper run manifest."""

import csv
import hashlib
import json
from pathlib import Path

import elements
import ruamel.yaml as yaml

from .config import (
    ATARI100K_GAMES, FINAL_CHECKPOINTS, QUALITATIVE_GAMES, TRAINING_SEEDS,
    WANDB_ENTITY, wandb_project_for_game)

VARIANTS = {
    'Backbone': (),
    'Flat': ('corewm_flat',),
    'Rec-only': ('corewm_rec_only',),
    'Pdyn-only': ('corewm_pdyn_only',),
    'Full': ('corewm_full',),
    'Reverse': ('corewm_reverse',),
}
MAIN_POLICY_VARIANTS = ('Backbone', 'Flat', 'Rec-only', 'Pdyn-only', 'Full')
PROTOCOL_ID = 'corewm_atari100k_v1'
BASE_CONFIGS = ('atari100k', 'size12m', 'corewm_paper_protocol', 'wandb')


def resolve(variant, configs_path='dreamerv3/configs.yaml'):
  configs = yaml.YAML(typ='safe').load(Path(configs_path).read_text())
  config = elements.Config(configs['defaults'])
  for name in (*BASE_CONFIGS, *VARIANTS[variant]):
    config = config.update(configs[name])
  return config


def audit_row(variant, config):
  htp = config.agent.htp
  projected = bool(htp.enabled and htp.use_proj)
  rec = bool(htp.enabled and htp.use_recon)
  pdyn = bool(htp.enabled and htp.use_pdyn)
  return {
      'variant': variant,
      'base_config': ' '.join(BASE_CONFIGS),
      'HTP_enabled': bool(htp.enabled), 'projection_enabled': projected,
      'D': int(htp.proj.dims[-1]) if projected else None,
      'prefix_dims': list(htp.proj.dims) if projected else None,
      'lambda_rec': float(config.agent.loss_scales.htp_rec) if rec else 0.0,
      'lambda_pdyn': float(config.agent.loss_scales.htp_pdyn) if pdyn else 0.0,
      'stride_assignment': list(htp.pdyn.strides) if pdyn else None,
      'actor_input': 'full_z_2048' if projected else 'backbone_h_2560',
      'critic_input': 'full_z_2048' if projected else 'backbone_h_2560',
      'world_model_backbone': 'DreamerV3 size12m RSSM deter=2048 stoch=32x16',
      'optimizer_config': dict(config.agent.opt),
      'replay_config': dict(config.replay),
      'environment_config': dict(config.env.atari100k),
      'training_budget': int(config.run.steps),
      'checkpoint_schedule': list(FINAL_CHECKPOINTS),
      'evaluation_schedule': {
          '10k_to_90k_episodes': 10, '100k_episodes': 100},
  }


def _flat(value, prefix=''):
  if isinstance(value, elements.Config):
    return dict(value.flat)
  if isinstance(value, dict):
    result = {}
    for key, child in value.items():
      result.update(_flat(child, f'{prefix}.{key}' if prefix else str(key)))
    return result
  return {prefix: value}


def _resolved_hash(config):
  payload = json.dumps(
      dict(sorted(_flat(config).items())), default=str,
      separators=(',', ':'), sort_keys=True).encode()
  return hashlib.sha256(payload).hexdigest()


def prepare(output='paper_artifacts/corewm_phase3'):
  output = Path(output)
  output.mkdir(parents=True, exist_ok=True)
  configs, audits = {}, []
  for variant in VARIANTS:
    config = resolve(variant)
    configs[variant] = config
    config.save(output / f'resolved_{variant.lower().replace("-", "_")}.yaml')
    audits.append(audit_row(variant, config))
  (output / 'config_audit.json').write_text(json.dumps(audits, indent=2) + '\n')
  full = _flat(configs['Full'])
  intentional = {
      'Backbone': ('agent.htp.enabled',),
      'Flat': ('agent.htp.use_recon', 'agent.htp.use_pdyn'),
      'Rec-only': ('agent.htp.use_pdyn',),
      'Pdyn-only': ('agent.htp.use_recon',),
      'Full': (),
      'Reverse': ('agent.htp.pdyn.strides', 'agent.htp.pdyn.enforce_coarse_to_fine'),
  }
  differences = []
  for variant, config in configs.items():
    current = _flat(config)
    for key in sorted(set(full) | set(current)):
      if current.get(key) != full.get(key):
        differences.append({
            'variant': variant, 'field': key, 'value': current.get(key),
            'full_value': full.get(key),
            'intentional': key in intentional[variant]})
  (output / 'config_differences.json').write_text(
      json.dumps(differences, indent=2, default=str) + '\n')
  unexpected = [row for row in differences if not row['intentional']]
  if unexpected:
    raise AssertionError(f'Unexpected matched config differences: {unexpected}')
  jobs = []
  approved = []
  for variant in MAIN_POLICY_VARIANTS:
    approved.extend(
        (variant, game, seed)
        for game in ATARI100K_GAMES for seed in TRAINING_SEEDS)
  approved.extend(
      ('Reverse', game, seed)
      for game in QUALITATIVE_GAMES for seed in TRAINING_SEEDS)
  approved.sort(key=lambda item: (
      1 if item[2] == 0 and item[1] in QUALITATIVE_GAMES else 2,
      QUALITATIVE_GAMES.index(item[1]) if item[1] in QUALITATIVE_GAMES else 99,
      tuple(VARIANTS).index(item[0]), ATARI100K_GAMES.index(item[1]), item[2]))
  for job_index, (variant, game, seed) in enumerate(approved):
    config = configs[variant]
    htp = config.agent.htp
    enabled = bool(htp.enabled)
    projected = bool(enabled and htp.use_proj)
    wave = 1 if seed == 0 and game in QUALITATIVE_GAMES else 2
    jobs.append({
            'job_index': job_index,
            'protocol_id': PROTOCOL_ID,
            'wave': wave,
            'method': variant, 'variant': variant, 'game': game, 'seed': seed,
            'wandb_entity': WANDB_ENTITY,
            'wandb_project': wandb_project_for_game(game),
            'configs': list((*BASE_CONFIGS, *VARIANTS[variant])),
            'status': 'PENDING_APPROVAL',
            'git_commit': 'CAPTURE_AT_RUNTIME',
            'resolved_config_hash': _resolved_hash(config),
            'env_action_steps': 'CAPTURE_AT_RUNTIME',
            'driver_callbacks': 'CAPTURE_AT_RUNTIME',
            'reset_callbacks': 'CAPTURE_AT_RUNTIME',
            'replay_insertions': 'CAPTURE_AT_RUNTIME',
            'optimizer_updates': 'CAPTURE_AT_RUNTIME',
            'checkpoint_action_milestone': list(FINAL_CHECKPOINTS),
            'checkpoint_hash': 'CAPTURE_PER_MILESTONE',
            'htp_enabled': enabled,
            'projection_enabled': projected,
            'use_recon': bool(htp.use_recon) if enabled else 'N/A',
            'use_pdyn': bool(htp.use_pdyn) if enabled else 'N/A',
            'prefix_dims': list(htp.proj.dims) if projected else 'N/A',
            'strides': list(htp.pdyn.strides) if enabled else 'N/A',
            'checkpoints': (
                list(FINAL_CHECKPOINTS) if variant in MAIN_POLICY_VARIANTS
                else [100_000])})
  main_jobs = [row for row in jobs if row['variant'] in MAIN_POLICY_VARIANTS]
  reverse_jobs = [row for row in jobs if row['variant'] == 'Reverse']
  wave1_jobs = [row for row in jobs if row['wave'] == 1]
  wave2_jobs = [row for row in jobs if row['wave'] == 2]
  assert len(main_jobs) == 5 * 26 * 5 == 650
  assert len(reverse_jobs) == 1 * 6 * 5 == 30
  assert len(jobs) == 680
  assert len(wave1_jobs) == 36
  assert len(wave2_jobs) == 644
  (output / 'training_manifest.json').write_text(json.dumps(jobs, indent=2) + '\n')
  eval_jobs = []
  for job in jobs:
    checkpoints = FINAL_CHECKPOINTS if job['variant'] in MAIN_POLICY_VARIANTS else (100_000,)
    for checkpoint in checkpoints:
      eval_jobs.append({
          'protocol_id': PROTOCOL_ID, 'wave': job['wave'],
          'variant': job['variant'], 'game': job['game'], 'seed': job['seed'],
          'checkpoint': checkpoint,
          'episodes': 100 if checkpoint == 100_000 else 10,
          'execution': 'isolated_checkpoint_job', 'status': 'PENDING_APPROVAL'})
  assert len(eval_jobs) == 650 * 10 + 30 == 6530
  assert sum(row['episodes'] for row in eval_jobs if row['checkpoint'] < 100_000) == 58_500
  assert sum(row['episodes'] for row in eval_jobs if row['checkpoint'] == 100_000) == 68_000
  (output / 'evaluation_manifest.json').write_text(
      json.dumps(eval_jobs, indent=2) + '\n')
  return audits, differences, jobs


if __name__ == '__main__':
  prepare()
