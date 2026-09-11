"""Fixed orthonormal refinement and rate-priced predictive outcomes.

Pure target/loss functions are deliberately separate from neural modules so
ablation changes do not silently alter normalization or episode semantics.
"""
import elements
import embodied.jax
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

SIZES = (4, 8, 16, 32, 64)
BAND_DIMS = (48, 144, 576, 2304, 9216)
PIXELS = 64 * 64 * 3
sg = jax.lax.stop_gradient


def haar_bands(image):
  """Orthonormal RGB Haar coefficients, coarse 4x4 followed by details."""
  if image.shape[-3:] != (64, 64, 3):
    raise ValueError('Reborn requires 64x64 RGB images')
  x = image.astype(jnp.float32)
  details = []
  while x.shape[-3] > 4:
    a, b = x[..., ::2, ::2, :], x[..., ::2, 1::2, :]
    c, d = x[..., 1::2, ::2, :], x[..., 1::2, 1::2, :]
    detail = jnp.stack(((a-b+c-d)/2, (a+b-c-d)/2, (a-b-c+d)/2), -1)
    details.append(detail.reshape((*x.shape[:-3], -1)))
    x = (a+b+c+d)/2
  return (x.reshape((*x.shape[:-3], -1)), *reversed(details))


def haar_image(bands):
  """Synthesize a prefix; absent details are zero, preserving coarse output."""
  x = bands[0].reshape((*bands[0].shape[:-1], 4, 4, 3))
  for level in range(1, 5):
    h = x.shape[-3]
    if level < len(bands):
      detail = bands[level].reshape((*x.shape, 3))
      u, v, w = (detail[..., i] for i in range(3))
    else:
      u = v = w = jnp.zeros_like(x)
    out = jnp.zeros((*x.shape[:-3], h*2, h*2, 3), x.dtype)
    out = out.at[..., ::2, ::2, :].set((x+u+v+w)/2)
    out = out.at[..., ::2, 1::2, :].set((x-u+v-w)/2)
    out = out.at[..., 1::2, ::2, :].set((x+u-v-w)/2)
    x = out.at[..., 1::2, 1::2, :].set((x-u-v+w)/2)
  return x


def sample_code(mu, seed, stochastic=True):
  noise = jax.random.normal(seed, mu.shape, dtype=jnp.float32) if stochastic else jnp.zeros_like(mu)
  return (mu.astype(jnp.float32) + noise).astype(nn.COMPUTE_DTYPE)


def rate_terms(mu, dims, cumulative=True):
  block = jnp.stack([0.5*jnp.square(mu[..., lo:hi].astype(jnp.float32)).sum(-1)
      for lo, hi in zip((0, *dims[:-1]), dims)], -1)
  prefix = jnp.cumsum(block, -1)
  return (prefix.mean(-1) if cumulative else block.sum(-1)), block, prefix


def transition_mask(first, last):
  return (~first[:, 1:] & ~last[:, :-1]).astype(jnp.float32)


def masked_mean(value, valid):
  return (jnp.where(valid > 0, value, 0) * valid).sum() / jnp.maximum(valid.sum(), 1)


def bellman_targets(features, reward, terminal, next_psi, next_q, gamma):
  cont = 1-terminal.astype(jnp.float32)
  return (sg((1-gamma)*features + gamma*cont[..., None]*next_psi),
          sg(reward + gamma*cont*next_q))


class AffineProjection(nj.Module):
  dim: int = 2048

  def __call__(self, h):
    if self.dim > h.shape[-1]:
      raise ValueError('Output dimension must not exceed backbone dimension')
    return self.sub('affine', nn.Linear, self.dim)(nn.cast(h)).astype(jnp.float32)


class Refinement(nj.Module):
  dims: tuple = (128, 256, 512, 1024, 2048)
  hidden: int = 512
  context: bool = True

  def __call__(self, z):
    bands = []
    for i, (lo, hi, outdim) in enumerate(zip((0, *self.dims[:-1]), self.dims, BAND_DIMS)):
      x = z[..., :hi] if self.context else z[..., lo:hi]
      x = self.sub(f'head{i}', nn.MLP, 1, self.hidden, act='silu', norm='rms')(nn.cast(x))
      bands.append(self.sub(f'band{i}', nn.Linear, outdim)(x).astype(jnp.float32))
    return tuple(bands)


class Outcomes(nj.Module):
  dims: tuple = (128, 256, 512, 1024, 2048)
  hidden: int = 512

  def __call__(self, z, action):
    outputs = []
    for i, (dim, size) in enumerate(zip(self.dims, SIZES)):
      x = nn.cast(jnp.concatenate((z[..., :dim], action), -1))
      x = self.sub(f'trunk{i}', nn.MLP, 2, self.hidden, act='silu', norm='rms')(x)
      psi = self.sub(f'psi{i}', nn.Linear, size*size*3)(x).astype(jnp.float32)
      q = self.sub(f'q{i}', embodied.jax.MLPHead, elements.Space(np.float32, ()),
          'symexp_twohot', layers=0, units=self.hidden, bins=255, outscale=0.0)(x, z.ndim-1)
      outputs.append((psi, q))
    return outputs


def refinement_loss(predicted, target):
  errors = tuple(jnp.square(p-t).sum(-1)/PIXELS for p, t in zip(predicted, target))
  return sum(errors), errors


def outcome_loss(online, target, features, reward, terminal, valid, gammas, q_weight):
  total = jnp.zeros_like(reward)
  metrics = {}
  for i, ((psi, q), (next_psi, next_q), gamma) in enumerate(zip(online, target, gammas)):
    fixed = jnp.concatenate(features[:i+1], -1)
    yp, yq = bellman_targets(fixed, reward, terminal, next_psi, next_q.pred(), gamma)
    pe = jnp.square(psi-yp).sum(-1)/PIXELS
    qe = q.loss(yq)
    total += (pe + q_weight*qe)/len(gammas)
    for key, value in dict(sf_td_mse=pe, q_twohot=qe, q_td_mae=jnp.abs(q.pred()-yq),
        q_target_abs=jnp.abs(yq), sf_target_energy=jnp.square(yp).sum(-1)/PIXELS,
        sf_zero_mse=jnp.square(yp).sum(-1)/PIXELS).items():
      metrics[f'reborn/l{i+1}/{key}'] = masked_mean(value, valid)
    nonzero = valid*(reward != 0)
    metrics[f'reborn/l{i+1}/q_td_mae_nonzero_reward'] = masked_mean(jnp.abs(q.pred()-yq), nonzero)
  # The agent averages B*T: normalize by actual valid pairs, then pad to B*T.
  scaled = jnp.where(valid > 0, total, 0)*valid * (valid.shape[0]*(valid.shape[1]+1))/jnp.maximum(valid.sum(), 1)
  return jnp.pad(scaled, ((0, 0), (0, 1))), metrics
