"""Matched post-hoc progressive reconstruction probes.

The probe reuses DreamerV3's actual ``ProgressiveRecon`` implementation but
lives outside the production agent/checkpoint. Inputs and targets are detached,
so only the freshly initialized diagnostic heads can be updated.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import optax

from dreamerv3.htp import ProgressiveRecon
from embodied.jax import opt as embodied_opt


@dataclass(frozen=True)
class PosthocArchitecture:
  prefix_dims: tuple = (128, 256, 512, 1024, 2048)
  feat_dim: int = 2560
  hidden: int = 512
  layers: int = 1
  norm: str = 'rms'
  act: str = 'silu'
  outscale: float = 1.0


@dataclass(frozen=True)
class PosthocOptimizer:
  lr: float = 4e-5
  agc: float = 0.3
  eps: float = 1e-20
  beta1: float = 0.9
  beta2: float = 0.999
  warmup: int = 1000


def make_module(architecture=PosthocArchitecture(), name='posthoc_recon'):
  """Instantiate the exact reconstruction class used by native CoRe-WM."""
  return ProgressiveRecon(
      feat_dim=architecture.feat_dim, dims=architecture.prefix_dims,
      hidden=architecture.hidden, layers=architecture.layers,
      norm=architecture.norm, act=architecture.act,
      outscale=architecture.outscale, name=name)


def make_pure(architecture=PosthocArchitecture()):
  module = make_module(architecture)
  return module, nj.pure(lambda z, h: module(
      jax.lax.stop_gradient(z), jax.lax.stop_gradient(h)))


def initialize(seed, architecture=PosthocArchitecture()):
  """Initialize matched parameters without using any production parameters."""
  _, pure = make_pure(architecture)
  z = jnp.zeros((1, 1, architecture.prefix_dims[-1]), jnp.bfloat16)
  h = jnp.zeros((1, 1, architecture.feat_dim), jnp.bfloat16)
  params, _ = pure({}, z, h, seed=int(seed), create=True)
  return params


def parameter_counts(params, levels=5):
  result = {}
  for level in range(levels):
    marker = f'/head{level}/'
    result[level + 1] = int(sum(
        np.prod(value.shape) for key, value in params.items()
        if marker in key))
  if not all(result.values()):
    raise AssertionError(f'Missing head parameters: {result}')
  return result


def parameter_hash(params):
  digest = hashlib.sha256()
  for key, value in sorted(params.items()):
    array = np.asarray(value)
    digest.update(key.encode())
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, np.int64).tobytes())
    digest.update(array.tobytes())
  return digest.hexdigest()


def make_optimizer(config=PosthocOptimizer()):
  """Exact native optimizer chain for the active production configuration."""
  schedule = optax.constant_schedule(config.lr)
  if config.warmup:
    ramp = optax.linear_schedule(0.0, config.lr, config.warmup)
    schedule = optax.join_schedules([ramp, schedule], [config.warmup])
  return optax.chain(
      embodied_opt.clip_by_agc(config.agc),
      embodied_opt.scale_by_rms(config.beta2, config.eps),
      embodied_opt.scale_by_momentum(config.beta1, False),
      optax.scale_by_learning_rate(schedule))


def loss_and_levels(params, z, h, architecture=PosthocArchitecture()):
  _, pure = make_pure(architecture)
  _, (loss, levels) = pure(
      params, z, h, seed=0, create=False, modify=False)
  return loss.mean().astype(jnp.float32), tuple(
      value.mean().astype(jnp.float32) for value in levels)


def reconstruct(params, z, h_template,
                architecture=PosthocArchitecture()):
  module = make_module(architecture)
  pure = nj.pure(lambda value, template: module.reconstruct(
      jax.lax.stop_gradient(value), jax.lax.stop_gradient(template)))
  _, cumulative = pure(
      params, z, h_template, seed=0, create=False, modify=False)
  return cumulative


def train_step(params, opt_state, z, h, optimizer,
               architecture=PosthocArchitecture()):
  def objective(current):
    return loss_and_levels(current, z, h, architecture)
  (loss, levels), grads = jax.value_and_grad(objective, has_aux=True)(params)
  updates, opt_state = optimizer.update(grads, opt_state, params)
  params = optax.apply_updates(params, updates)
  return params, opt_state, loss, levels, grads


def reconstruction_metrics(params, z, h,
                           architecture=PosthocArchitecture()):
  cumulative = reconstruct(params, z, h, architecture)
  target = np.asarray(h, np.float64)
  mean = target.mean(axis=0, keepdims=True)
  denominator = float(np.square(target - mean).sum())
  if not np.isfinite(denominator) or denominator <= 0:
    raise ValueError(f'Degenerate TEST target denominator: {denominator}')
  mse, r2 = [], []
  for prediction in cumulative:
    residual = target - np.asarray(prediction, np.float64)
    sse = float(np.square(residual).sum())
    mse.append(sse / target.size)
    r2.append(1.0 - sse / denominator)
  mse, r2 = np.asarray(mse), np.asarray(r2)
  if not np.isfinite(mse).all() or not np.isfinite(r2).all():
    raise AssertionError((mse, r2))
  if np.any(mse < -1e-8):
    raise AssertionError(mse)
  return {
      'E_post': mse,
      'R2_rec': r2,
      'E_norm': 1.0 - r2,
      'Delta_E_post': np.r_[np.nan, mse[:-1] - mse[1:]],
      'Delta_R2_rec': np.r_[np.nan, r2[1:] - r2[:-1]],
  }


def metrics_from_sufficient_statistics(sse, target_sum, target_square_sum,
                                       sample_count, feat_dim):
  sse = np.asarray(sse, np.float64)
  target_sum = np.asarray(target_sum, np.float64)
  sst = float(target_square_sum - np.square(target_sum).sum() / sample_count)
  if sample_count <= 0 or not np.isfinite(sst) or sst <= 0:
    raise ValueError((sample_count, sst))
  mse = sse / (sample_count * feat_dim)
  r2 = 1.0 - sse / sst
  if not np.isfinite(mse).all() or not np.isfinite(r2).all():
    raise AssertionError((mse, r2))
  if np.any(mse < -1e-8):
    raise AssertionError(mse)
  return {
      'E_post': mse, 'R2_rec': r2, 'E_norm': 1.0 - r2,
      'Delta_E_post': np.r_[np.nan, mse[:-1] - mse[1:]],
      'Delta_R2_rec': np.r_[np.nan, r2[1:] - r2[:-1]],
  }


def qualifying_improvement(score, best, min_improvement=1e-4):
  return bool(score > best + min_improvement)


def probe_not_converged(validation_curve, best_update):
  """Frozen technical non-convergence rule for a max-budget probe."""
  by_update = {int(row['update']): float(row['validation_score'])
               for row in validation_curve}
  if 9000 not in by_update or 10000 not in by_update:
    return False
  improvement = by_update[10000] - by_update[9000]
  exceeds = improvement > 0.002 and not np.isclose(
      improvement, 0.002, rtol=0, atol=1e-12)
  return bool(best_update >= 9500 and exceeds)


def architecture_metadata(architecture=PosthocArchitecture(),
                          optimizer=PosthocOptimizer()):
  return {
      'implementation_class': 'dreamerv3.htp.ProgressiveRecon',
      'head_class': 'dreamerv3.htp.ResidualReconHead',
      'architecture': asdict(architecture),
      'optimizer': {
          **asdict(optimizer),
          'chain': ['adaptive_gradient_clipping', 'scale_by_rms',
                    'scale_by_momentum', 'scale_by_learning_rate'],
          'weight_decay': 0.0,
          'schedule': 'constant_after_linear_warmup',
      },
      'progressive_recurrence': 'hhat_l = stop_gradient(hhat_l-1) + Gpost_l(block_l)',
      'representation_inputs_detached': True,
      'production_checkpoint_parameters_reused': False,
  }


def write_json(path, payload):
  path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
