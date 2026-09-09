"""Real-checkpoint offline paper analyses on shared diagnostic trajectories."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import elements
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from dreamerv3 import main_htp

from .decoder import (
    assert_image_domain, normalize_ground_truth, unflatten_rssm_state)
from .extraction import (
    ExtractionSpec, evaluation_seed, extract_readonly, open_loop_readonly)
from .metrics import assert_nonnegative
from .open_loop import rollout_error_families, validate_shared_backbone_rollout
from .phase3_prepare import resolve
from .probes import fit_ridge_protocol
from .shared_dataset import load_shared_diagnostic_trajectories
from .slicing import block_dims
from .temporal import (
    mean_long_horizon_r2, predictability_half_life, spearman_all_finite)
from .trajectories import one_hot_action_windows, valid_horizon_indices


VARIANTS = {
    'flat': 'Flat', 'rec_only': 'Rec-only', 'pdyn_only': 'Pdyn-only',
    'full': 'Full', 'reverse': 'Reverse'}
HORIZONS = (1, 2, 4, 8, 16, 32, 64)


def _sha256(path):
  digest = hashlib.sha256()
  with Path(path).open('rb') as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
      digest.update(chunk)
  return digest.hexdigest()


def _parameter_hash(agent):
  digest = hashlib.sha256()
  for key, value in sorted(agent.save()['params'].items()):
    digest.update(key.encode()); digest.update(np.asarray(value).tobytes())
  return digest.hexdigest()


def _make_agent(slug, checkpoint, runtime):
  config = resolve(VARIANTS[slug]).update(
      task='atari100k_alien', seed=0, logdir=str(runtime),
      **{'jax.prealloc': False, 'env.atari100k.use_seed': True})
  agent = main_htp.make_agent(config)
  cp = elements.Checkpoint(); cp.agent = agent
  cp.load(checkpoint, keys=['agent'])
  dyn = config.agent.dyn[config.agent.dyn.typ]
  dims = tuple(map(int, config.agent.htp.proj.dims))
  spec = ExtractionSpec(
      int(dyn.deter), (int(dyn.stoch), int(dyn.classes)), dims)
  cardinality = int(agent.act_space['action'].high)
  return config, agent, spec, cardinality


def _padded_batch(episodes, max_length):
  batch, time = len(episodes), int(max_length)
  obs = {
      'image': np.zeros((batch, time, 64, 64, 3), np.uint8),
      'reward': np.zeros((batch, time), np.float32),
      'is_first': np.ones((batch, time), bool),
      'is_last': np.zeros((batch, time), bool),
      'is_terminal': np.zeros((batch, time), bool),
  }
  prevact = np.zeros((batch, time), np.int32)
  lengths = []
  for index, episode in enumerate(episodes):
    length = len(episode['action']); lengths.append(length)
    obs['image'][index, :length] = episode['observation']
    obs['reward'][index, :length] = episode['reward']
    obs['is_first'][index, :length] = False
    obs['is_first'][index, 0] = True
    obs['is_last'][index, :length] = ~episode['continuation'].astype(bool)
    obs['is_terminal'][index, :length] = episode['is_terminal'].astype(bool)
    prevact[index, 1:length] = np.asarray(episode['action'][:-1], np.int32)
  return obs, {'action': prevact}, lengths


def extract_dataset(slug, checkpoint, dataset, output, batch_size=1):
  output = Path(output); output.mkdir(parents=True, exist_ok=False)
  if int(batch_size) != 1:
    raise ValueError(
        'Production extraction uses batch_size=1 so every episode receives '
        'its own order-independent RNG key')
  manifest, episodes = load_shared_diagnostic_trajectories(dataset)
  split = json.loads((Path(dataset) / 'split_manifest.json').read_text())
  config, agent, spec, cardinality = _make_agent(
      slug, checkpoint, output / 'agent_runtime')
  before = _parameter_hash(agent); updates_before = int(agent.n_updates)
  checkpoint_before = _sha256(Path(checkpoint) / 'agent.pkl')
  maximum = max(len(ep['action']) for ep in episodes)
  all_z, all_h, all_errors, all_ids, all_t, all_actions = [], [], [], [], [], []
  for offset in range(0, len(episodes), int(batch_size)):
    group = episodes[offset:offset + int(batch_size)]
    obs, prevact, lengths = _padded_batch(group, maximum)
    eid = int(group[0]['episode_id'][0])
    result = extract_readonly(
        agent, obs, prevact, spec,
        evaluation_seed(0, eid, 0, stream=0), decode=False)
    for index, (episode, length) in enumerate(zip(group, lengths)):
      h = np.asarray(result['h'][index, :length], np.float32)
      z = np.asarray(result['z'][index, :length], np.float32)
      recons = [np.asarray(x[index, :length], np.float32)
                for x in result['cumulative_reconstructions']]
      errors = (np.stack([np.square(h - x).mean(-1) for x in recons], -1)
                if recons else np.empty((length, 0), np.float32))
      if recons:
        assert_nonnegative('E_rec', errors)
      all_h.append(h); all_z.append(z); all_errors.append(errors)
      all_ids.append(np.asarray(episode['episode_id'], np.int64))
      all_t.append(np.asarray(episode['timestep'], np.int64))
      all_actions.append(np.asarray(episode['action'], np.int32))
    print(f'EXTRACTED {min(offset + len(group), len(episodes))}/100', flush=True)
  after = _parameter_hash(agent); checkpoint_after = _sha256(Path(checkpoint) / 'agent.pkl')
  if before != after or updates_before != int(agent.n_updates):
    raise AssertionError('Model or optimizer changed during extraction')
  if checkpoint_before != checkpoint_after:
    raise AssertionError('Checkpoint changed during extraction')
  arrays = {
      'z': np.concatenate(all_z), 'h': np.concatenate(all_h),
      'E_rec_per_sample': np.concatenate(all_errors),
      'episode_id': np.concatenate(all_ids), 'timestep': np.concatenate(all_t),
      'action': np.concatenate(all_actions),
  }
  np.savez(output / 'representations.npz', **arrays)
  metadata = {
      'protocol': 'corewm_atari100k_v1', 'evaluation_code_version': _sha256(__file__),
      'method': VARIANTS[slug], 'game': manifest['game'], 'training_seed': 0,
      'checkpoint': str(checkpoint), 'checkpoint_agent_hash': checkpoint_before,
      'source_dataset': str(dataset),
      'source_dataset_manifest_hash': _sha256(Path(dataset) / 'dataset_manifest.json'),
      'split_manifest_hash': _sha256(Path(dataset) / 'split_manifest.json'),
      'samples': int(len(arrays['z'])), 'episodes': len(episodes),
      'h_dim': int(arrays['h'].shape[-1]), 'z_dim': int(arrays['z'].shape[-1]),
      'prefix_dims': list(spec.prefix_dims), 'block_dims': list(block_dims(spec.prefix_dims)),
      'action_cardinality': cardinality, 'model_frozen_verified': True,
      'reconstruction_heads_available': bool(all_errors[0].shape[-1]),
      'optimizer_updates_unchanged': True, 'checkpoint_unchanged': True,
      'split_counts': {key: len(split[key]) for key in ('train','validation','test')},
      'rng_derivation': 'SeedSequence([0, persistent_episode_id, 0, 0])',
      'batch_size': int(batch_size), 'padding_masked_from_all_outputs': True,
  }
  if (metadata['h_dim'], metadata['z_dim'], metadata['prefix_dims'], metadata['block_dims']) != (
      2560, 2048, [128,256,512,1024,2048], [128,128,256,512,1024]):
    raise AssertionError(metadata)
  (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
  print(json.dumps(metadata, indent=2))


def _load_cache(cache):
  cache = Path(cache)
  with np.load(cache / 'representations.npz') as data:
    arrays = {key: np.asarray(data[key]) for key in data.files}
  metadata = json.loads((cache / 'metadata.json').read_text())
  return metadata, arrays


def _split_masks(ids, split):
  return {key: np.isin(ids, np.asarray(split[key], np.int64))
          for key in ('train','validation','test')}


def _cka_gpu(blocks):
  centered = []
  for block in blocks:
    value = jnp.asarray(block)
    centered.append(value - value.mean(0, keepdims=True))
  matrix = np.empty((len(blocks), len(blocks)), np.float64)
  norms = [jnp.sqrt(jnp.square(x.T @ x).sum()) for x in centered]
  for i, left in enumerate(centered):
    for j in range(i, len(centered)):
      right = centered[j]
      value = float(jnp.square(left.T @ right).sum() / (norms[i] * norms[j]))
      matrix[i, j] = matrix[j, i] = value
  if not np.isfinite(matrix).all() or not np.array_equal(matrix, matrix.T):
    raise AssertionError(matrix)
  return matrix


def eval1(cache, dataset, output):
  output = Path(output); output.mkdir(parents=True, exist_ok=False)
  metadata, data = _load_cache(cache)
  split = json.loads((Path(dataset) / 'split_manifest.json').read_text())
  if metadata['split_manifest_hash'] != _sha256(Path(dataset) / 'split_manifest.json'):
    raise AssertionError('Split manifest mismatch')
  dims = tuple(metadata['prefix_dims']); bdims = block_dims(dims)
  z, errors = data['z'], data['E_rec_per_sample']
  blocks = [z[:, lo:hi] for lo, hi in zip((0,*dims[:-1]), dims)]
  cka = _cka_gpu(blocks)
  if not np.allclose(np.diag(cka), 1, atol=2e-4):
    raise AssertionError(np.diag(cka))
  np.save(output / 'block_cka.npy', cka)
  np.savetxt(output / 'block_cka.csv', cka, delimiter=',')
  np.savez(output / 'per_sample_reconstruction.npz',
           episode_id=data['episode_id'], timestep=data['timestep'], E_rec=errors)
  has_recon = bool(metadata['reconstruction_heads_available'])
  if has_recon:
    means = errors.mean(0); assert_nonnegative('E_rec', means)
    gains = np.r_[np.nan, means[:-1] - means[1:]]
    gains_dim = gains / np.asarray(bdims)
  else:
    means = gains = gains_dim = np.full(5, np.nan)
  masks = _split_masks(data['episode_id'], split)
  probes = []
  for level in range(1, 5):
    result, _ = fit_ridge_protocol(
        z[masks['train'], :dims[level-1]], blocks[level][masks['train']],
        z[masks['validation'], :dims[level-1]], blocks[level][masks['validation']],
        z[masks['test'], :dims[level-1]], blocks[level][masks['test']])
    probes.append({'level':level+1, **result.to_dict()})
  with (output / 'conditional_r2.csv').open('w', newline='') as stream:
    writer=csv.DictWriter(stream, fieldnames=('level','selected_alpha','test_r2','train_samples','validation_samples','test_samples','validation_r2_grid','scaler_metadata'))
    writer.writeheader(); writer.writerows({**row,
      'validation_r2_grid':json.dumps(row['validation_r2_grid']),
      'scaler_metadata':json.dumps(row['scaler_metadata'])} for row in probes)
  rows=[]
  for level,(dim,bdim) in enumerate(zip(dims,bdims),1):
    rows.append({'method':metadata['method'],'game':'alien','seed':0,'level':level,
      'prefix_dim':dim,'block_dim':bdim,
      'E_rec':None if not has_recon else float(means[level-1]),
      'Delta_rec':None if level==1 or not has_recon else float(gains[level-1]),
      'Delta_rec_dim':None if level==1 or not has_recon else float(gains_dim[level-1]),
      'conditional_R2':None if level==1 else float(probes[level-2]['test_r2'])})
  with (output/'incremental_metrics.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
  if has_recon:
    fig,ax=plt.subplots(figsize=(5,3));ax.plot(dims,means,marker='o');ax.set(xlabel='Prefix dimension',ylabel='E_rec');fig.tight_layout();fig.savefig(output/'reconstruction_curve.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(5,3));ax.plot(dims[1:],gains[1:],marker='o',label='Delta');ax.plot(dims[1:],gains_dim[1:],marker='x',label='Delta/dim');ax.legend();fig.tight_layout();fig.savefig(output/'marginal_gain.pdf');plt.close(fig)
  fig,ax=plt.subplots(figsize=(4,4));image=ax.imshow(cka,vmin=0,vmax=1);fig.colorbar(image,ax=ax);fig.tight_layout();fig.savefig(output/'cka_heatmap.pdf');plt.close(fig)
  report={'method':metadata['method'],
    'E_rec':means.tolist() if has_recon else None,
    'Delta_rec':gains[1:].tolist() if has_recon else None,
    'Delta_rec_dim':gains_dim[1:].tolist() if has_recon else None,
    'reconstruction_status':('AVAILABLE' if has_recon else
      'BLOCKED_NO_TRAINED_RECONSTRUCTION_HEADS'),
    'conditional_R2':[x['test_r2'] for x in probes],
    'cka_shape':list(cka.shape),'cka_diagonal':np.diag(cka).tolist(),
    'mean_offdiag_CKA':float(cka[np.triu_indices(5,1)].mean()),'finite':True,
    'split_manifest_hash':metadata['split_manifest_hash']}
  (output/'metadata.json').write_text(json.dumps({**metadata,**report},indent=2)+'\n')
  print(json.dumps(report,indent=2))


def eval2b(cache, dataset, output):
  output=Path(output);output.mkdir(parents=True,exist_ok=False)
  metadata,data=_load_cache(cache); split=json.loads((Path(dataset)/'split_manifest.json').read_text())
  if metadata['split_manifest_hash'] != _sha256(Path(dataset)/'split_manifest.json'): raise AssertionError('Split mismatch')
  dims=tuple(metadata['prefix_dims']); bdims=block_dims(dims); z=data['z']; ids=data['episode_id']; times=data['timestep']; actions=data['action']; cardinality=int(metadata['action_cardinality'])
  matrix=np.empty((5,7),np.float64); rows=[]
  for li,(lo,hi) in enumerate(zip((0,*dims[:-1]),dims)):
    for ki,k in enumerate(HORIZONS):
      starts=valid_horizon_indices(ids,times,k); source_ids=ids[starts]
      windows=one_hot_action_windows(actions,starts,k,cardinality)
      x=np.concatenate([z[starts,:hi],windows],-1); y=z[starts+k,lo:hi]
      masks=_split_masks(source_ids,split)
      result,_=fit_ridge_protocol(x[masks['train']],y[masks['train']],x[masks['validation']],y[masks['validation']],x[masks['test']],y[masks['test']])
      matrix[li,ki]=result.test_r2
      rows.append({'method':metadata['method'],'game':'alien','seed':0,'block':li+1,'block_dim':bdims[li],'k':k,'R2':result.test_r2,'selected_alpha':result.selected_alpha,'train_samples':result.train_samples,'validation_samples':result.validation_samples,'test_samples':result.test_samples,'validation_r2_grid':json.dumps(result.validation_r2_grid)})
      print(f'PROBE level={li+1} k={k} R2={result.test_r2:.6g}',flush=True)
  np.save(output/'r2_matrix.npy',matrix)
  with (output/'r2_matrix.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
  taus=np.asarray([predictability_half_life(HORIZONS,row) for row in matrix]); meanlong,perblock=mean_long_horizon_r2(matrix,HORIZONS); rho=spearman_all_finite(taus)
  half=[{'method':metadata['method'],'game':'alien','seed':0,'block':i+1,'tau':None if not np.isfinite(taus[i]) else float(taus[i]),'R2_k1':float(matrix[i,0]),'MeanLongR2_l':float(perblock[i])} for i in range(5)]
  with (output/'half_life.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=half[0]);writer.writeheader();writer.writerows(half)
  summary={'method':metadata['method'],'shape':list(matrix.shape),'MeanLongR2':meanlong,'MeanLongR2_l':perblock.tolist(),'tau':[x['tau'] for x in half],'Spearman':None if not np.isfinite(rho) else float(rho),'all_finite_R2':bool(np.isfinite(matrix).all()),'split_manifest_hash':metadata['split_manifest_hash']}
  (output/'metadata.json').write_text(json.dumps({**metadata,**summary},indent=2)+'\n')
  fig,ax=plt.subplots(figsize=(6,4));im=ax.imshow(matrix,aspect='auto');ax.set_xticks(range(7),HORIZONS);ax.set_yticks(range(5),range(1,6));fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(output/'level_horizon_heatmap.pdf');plt.close(fig)
  fig,ax=plt.subplots(figsize=(6,4));
  for i,row in enumerate(matrix):ax.plot(HORIZONS,row,marker='o',label=f'Block {i+1}')
  ax.set_xscale('log',base=2);ax.legend();fig.tight_layout();fig.savefig(output/'predictability_decay_curves.pdf');plt.close(fig)
  print(json.dumps(summary,indent=2))


def _save_image_grid(path, rows, column_titles, row_titles=None):
  nrows, ncols = len(rows), len(column_titles)
  fig, axes = plt.subplots(nrows, ncols, figsize=(2 * ncols, 2 * nrows),
                           squeeze=False)
  for i, row in enumerate(rows):
    for j, value in enumerate(row):
      axes[i, j].imshow(np.clip(value, 0, 1))
      axes[i, j].axis('off')
      if i == 0:
        axes[i, j].set_title(column_titles[j], fontsize=8)
    if row_titles:
      axes[i, 0].set_ylabel(row_titles[i])
  fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def _posterior_qualitative(agent, params, spec, episodes, split, output):
  test_ids = set(map(int, split['test']))
  episode = next(ep for ep in episodes if int(ep['episode_id'][0]) in test_ids)
  obs, prevact, lengths = _padded_batch([episode], len(episode['action']))
  eid = int(episode['episode_id'][0])
  result = extract_readonly(
      agent, obs, prevact, spec, evaluation_seed(0, eid, 0, 0),
      params=params, decode=True)
  for key in result['decoded_original']:
    np.testing.assert_allclose(
        result['decoded_original'][key], result['decoded_recovered'][key],
        rtol=1e-6, atol=1e-6)
  index = lengths[0] // 2
  gt = normalize_ground_truth(episode['observation'][index])
  original = np.asarray(result['decoded_original']['image'][0, index])
  prefixes = [np.asarray(x['image'][0, index]) for x in result['decoded_prefixes']]
  assert_image_domain(gt, 'qualitative_ground_truth')
  assert_image_domain(original, 'qualitative_backbone')
  for level, image in enumerate(prefixes, 1):
    assert_image_domain(image, f'qualitative_prefix_{level}')
  path = output / 'progressive_reconstruction_alien_0.png'
  _save_image_grid(path, [[gt, original, *prefixes]], [
      'Ground truth', 'Backbone h', 'Prefix 128', 'Prefix 256',
      'Prefix 512', 'Prefix 1024', 'Prefix 2048'])
  return {
      'episode_id': eid, 'timestep': index,
      'selection_rule': 'midpoint of first manifest-ordered TEST episode',
      'decoder_roundtrip_max_abs': float(max(np.max(np.abs(
          result['decoded_original'][key] - result['decoded_recovered'][key]))
          for key in result['decoded_original'])),
      'artifact': str(path),
  }


def eval2a(cache, checkpoint, dataset, output, eval1_output=None, max_starts=None):
  output = Path(output); output.mkdir(parents=True, exist_ok=False)
  metadata, data = _load_cache(cache)
  manifest, episodes = load_shared_diagnostic_trajectories(dataset)
  split = json.loads((Path(dataset) / 'split_manifest.json').read_text())
  if metadata['method'] != 'Full':
    raise ValueError('Open-loop prefix retention is primary Full CoRe-WM only')
  if metadata['split_manifest_hash'] != _sha256(Path(dataset) / 'split_manifest.json'):
    raise AssertionError('Split manifest mismatch')
  config, agent, spec, cardinality = _make_agent(
      'full', checkpoint, output / 'agent_runtime')
  before = _parameter_hash(agent); updates_before = int(agent.n_updates)
  checkpoint_before = _sha256(Path(checkpoint) / 'agent.pkl')
  with jax._src.config.explicit_device_put_scope():
    params = jax.tree.map(jax.device_put, agent.save()['params'])
  ids, times, actions = data['episode_id'], data['timestep'], data['action']
  observations = np.concatenate([ep['observation'] for ep in episodes])
  starts = valid_horizon_indices(ids, times, 64)
  starts = starts[np.isin(ids[starts], np.asarray(split['test'], np.int64))]
  if max_starts is not None:
    starts = starts[:int(max_starts)]
  if not len(starts):
    raise ValueError('No valid 64-step TEST rollout starts')
  count = len(starts) * 5 * 64
  sample = {
      'start_episode': np.empty(count, np.int64),
      'start_t': np.empty(count, np.int32),
      'horizon': np.empty(count, np.int16),
      'level': np.empty(count, np.int8),
      'prefix_dim': np.empty(count, np.int16),
      'E_h': np.empty(count, np.float32),
      'E_prefix': np.empty(count, np.float32),
      'E_backbone': np.empty(count, np.float32),
      'E_total': np.empty(count, np.float32),
  }
  sums = {name: np.zeros((5, 64), np.float64) for name in (
      'E_h', 'E_prefix', 'E_backbone', 'E_total')}
  cursor = 0; selected_images = None; selected_info = None
  for number, start in enumerate(starts):
    eid, timestep = int(ids[start]), int(times[start])
    state = unflatten_rssm_state(
        data['h'][start:start + 1], spec.deter_dim, spec.stoch_shape)
    action_seq = {'action': actions[start:start + 64][None]}
    if (action_seq['action'] < 0).any() or (action_seq['action'] >= cardinality).any():
      raise ValueError('Action outside actual Atari cardinality')
    rollout = open_loop_readonly(
        agent, state, action_seq, spec,
        evaluation_seed(0, eid, timestep, stream=1), params=params)
    validate_shared_backbone_rollout(
        rollout['h_tilde'], rollout['cumulative_reconstructions'])
    np.testing.assert_array_equal(rollout['actions_used']['action'], action_seq['action'])
    if not np.all(ids[start + 1:start + 65] == eid):
      raise AssertionError('Rollout crossed an episode boundary')
    gt = normalize_ground_truth(observations[start + 1:start + 65])
    full = np.asarray(rollout['decoded_full']['image'][0])
    prefix_images = [np.asarray(x['image'][0]) for x in rollout['decoded_prefixes']]
    prefix_h = [np.asarray(x[0]) for x in rollout['cumulative_reconstructions']]
    assert_image_domain(gt, 'ground_truth'); assert_image_domain(full, 'full_backbone')
    for level, image in enumerate(prefix_images, 1):
      assert_image_domain(image, f'prefix_{level}')
    rows = rollout_error_families(
        gt, full, prefix_images, np.asarray(rollout['h_tilde'][0]), prefix_h)
    for level, row in enumerate(rows):
      sl = slice(cursor, cursor + 64); cursor += 64
      sample['start_episode'][sl] = eid; sample['start_t'][sl] = timestep
      sample['horizon'][sl] = np.arange(1, 65); sample['level'][sl] = level + 1
      sample['prefix_dim'][sl] = spec.prefix_dims[level]
      for name in sums:
        assert_nonnegative(name, row[name]); sums[name][level] += row[name]
        sample[name][sl] = row[name]
    if selected_images is None:
      selected_images = (gt, full, prefix_images)
      selected_info = (eid, timestep)
    if (number + 1) % 100 == 0 or number + 1 == len(starts):
      print(f'ROLLOUTS {number + 1}/{len(starts)}', flush=True)
  if cursor != count:
    raise AssertionError((cursor, count))
  for name in sums:
    sums[name] /= len(starts); assert_nonnegative(name, sums[name])
  np.savez(output / 'retention_metrics_per_sample.npz', **sample)
  with (output / 'retention_metrics.csv').open('w', newline='') as stream:
    fields = ('method','game','seed','horizon','level','prefix_dim',
              'E_h','E_prefix','E_backbone','E_total')
    writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
    for level, dim in enumerate(spec.prefix_dims):
      for horizon in range(1, 65):
        writer.writerow({'method':'Full','game':'alien','seed':0,
          'horizon':horizon,'level':level+1,'prefix_dim':dim,
          **{name:float(sums[name][level,horizon-1]) for name in sums}})
  gt, full, prefix_images = selected_images
  indices = [x - 1 for x in HORIZONS]
  grid = [[gt[i], full[i], *[images[i] for images in prefix_images]] for i in indices]
  grid_path = output / 'prefix_rollout_grid_alien_0.png'
  _save_image_grid(grid_path, grid, [
      'Ground Truth','Full Backbone','Prefix 128','Prefix 256','Prefix 512',
      'Prefix 1024','Prefix 2048'], [f'k={k}' for k in HORIZONS])
  for metric, filename in (
      ('E_h','latent_retention_curves.pdf'),
      ('E_prefix','image_prefix_distortion_curves.pdf')):
    fig,ax=plt.subplots(figsize=(6,4))
    for level,dim in enumerate(spec.prefix_dims):
      ax.plot(range(1,65),sums[metric][level],label=f'{dim}')
    ax.legend();ax.set(xlabel='Horizon',ylabel=metric);fig.tight_layout();fig.savefig(output/filename);plt.close(fig)
  fig,ax=plt.subplots(figsize=(6,4));ax.plot(range(1,65),sums['E_backbone'][0],label='Backbone')
  for level,dim in enumerate(spec.prefix_dims):ax.plot(range(1,65),sums['E_total'][level],label=f'Total {dim}')
  ax.legend();fig.tight_layout();fig.savefig(output/'total_vs_backbone_error.pdf');plt.close(fig)
  qualitative = None
  if eval1_output:
    eval1_output = Path(eval1_output); eval1_output.mkdir(parents=True, exist_ok=True)
    qualitative = _posterior_qualitative(agent, params, spec, episodes, split, eval1_output)
  after = _parameter_hash(agent); checkpoint_after = _sha256(Path(checkpoint)/'agent.pkl')
  if before != after or updates_before != int(agent.n_updates) or checkpoint_before != checkpoint_after:
    raise AssertionError('Evaluation mutated model, optimizer, or checkpoint')
  report = {
      'method':'Full','game':'alien','seed':0,'valid_rollout_starts':len(starts),
      'horizons':list(HORIZONS),'prefix_dims':list(spec.prefix_dims),
      'metric_shapes':{name:list(value.shape) for name,value in sums.items()},
      'all_metrics_finite_nonnegative':True,'same_h_tilde_for_all_prefixes':True,
      'prefix_never_transitioned':True,'recorded_actions_exact':True,
      'episode_boundaries_verified':True,'image_domain':'[0,1]',
      'rng_derivation':'SeedSequence([0, episode_id, start_timestep, 1])',
      'selected_rollout':{'episode_id':selected_info[0],'start_t':selected_info[1]},
      'qualitative_eval1':qualitative,'checkpoint_unchanged':True,
      'model_frozen_verified':True,'split_manifest_hash':metadata['split_manifest_hash'],
      'debug_max_starts':max_starts,
  }
  (output/'metadata.json').write_text(json.dumps({**metadata,**report},indent=2)+'\n')
  print(json.dumps(report,indent=2))


def main(argv=None):
  parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
  p=sub.add_parser('extract');p.add_argument('--method',choices=VARIANTS,required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--dataset',required=True);p.add_argument('--output',required=True);p.add_argument('--batch-size',type=int,default=1)
  p=sub.add_parser('eval1');p.add_argument('--cache',required=True);p.add_argument('--dataset',required=True);p.add_argument('--output',required=True)
  p=sub.add_parser('eval2b');p.add_argument('--cache',required=True);p.add_argument('--dataset',required=True);p.add_argument('--output',required=True)
  p=sub.add_parser('eval2a');p.add_argument('--cache',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--dataset',required=True);p.add_argument('--output',required=True);p.add_argument('--eval1-output');p.add_argument('--max-starts',type=int)
  args=parser.parse_args(argv)
  if args.command=='extract':extract_dataset(args.method,args.checkpoint,args.dataset,args.output,args.batch_size)
  elif args.command=='eval1':eval1(args.cache,args.dataset,args.output)
  elif args.command=='eval2b':eval2b(args.cache,args.dataset,args.output)
  else:eval2a(args.cache,args.checkpoint,args.dataset,args.output,args.eval1_output,args.max_starts)


if __name__=='__main__':main()
