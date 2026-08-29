import numpy as np


def assert_nonnegative(name, values, tolerance=1e-8):
  values = np.asarray(values, np.float64)
  if not np.isfinite(values).all():
    raise AssertionError(f'{name} contains NaN/Inf')
  minimum = float(values.min()) if values.size else 0.0
  if minimum < -float(tolerance):
    raise AssertionError(f'{name} must be non-negative; minimum={minimum}')
  return values


def _finite(name, value):
  value = np.asarray(value)
  if not np.isfinite(value).all():
    raise ValueError(f'{name} contains NaN or Inf')
  return value


def multivariate_r2(target, prediction):
  target = _finite('target', target).astype(np.float64)
  prediction = _finite('prediction', prediction).astype(np.float64)
  if target.shape != prediction.shape:
    raise ValueError((target.shape, prediction.shape))
  residual = np.square(target - prediction).sum()
  centered = target - target.mean(axis=0, keepdims=True)
  denominator = np.square(centered).sum()
  if denominator <= 0:
    raise ValueError('R2 target is numerically degenerate')
  return float(1.0 - residual / denominator)


def progressive_sum(residuals):
  if not residuals:
    raise ValueError('At least one residual is required')
  out, running = [], np.zeros_like(np.asarray(residuals[0]))
  for residual in residuals:
    running = running + np.asarray(residual)
    out.append(running.copy())
  return tuple(out)


def reconstruction_errors(h, cumulative_recons):
  h = _finite('h', h).astype(np.float64)
  errors = []
  for recon in cumulative_recons:
    recon = _finite('reconstruction', recon).astype(np.float64)
    if recon.shape != h.shape:
      raise ValueError((h.shape, recon.shape))
    errors.append(float(np.square(h - recon).mean(axis=-1).mean()))
  return np.asarray(errors)


def reconstruction_metrics(h, cumulative_recons, block_dimensions):
  errors = reconstruction_errors(h, cumulative_recons)
  block_dimensions = np.asarray(block_dimensions, dtype=np.int64)
  if len(errors) != len(block_dimensions):
    raise ValueError((len(errors), len(block_dimensions)))
  gains = np.full(len(errors), np.nan)
  gains[1:] = errors[:-1] - errors[1:]
  gains_per_dim = gains / block_dimensions
  return errors, gains, gains_per_dim


def linear_cka(x, y):
  x = _finite('x', x).astype(np.float64)
  y = _finite('y', y).astype(np.float64)
  if x.ndim != 2 or y.ndim != 2 or len(x) != len(y):
    raise ValueError((x.shape, y.shape))
  x = x - x.mean(axis=0, keepdims=True)
  y = y - y.mean(axis=0, keepdims=True)
  cross = x.T @ y
  numerator = np.square(cross).sum()
  norm_x = np.sqrt(np.square(x.T @ x).sum())
  norm_y = np.sqrt(np.square(y.T @ y).sum())
  denominator = norm_x * norm_y
  if denominator <= np.finfo(np.float64).tiny:
    raise ValueError('CKA denominator is numerically degenerate')
  return float(numerator / denominator)


def cka_matrix(blocks):
  matrix = np.empty((len(blocks), len(blocks)), np.float64)
  for i, x in enumerate(blocks):
    for j, y in enumerate(blocks):
      matrix[i, j] = linear_cka(x, y)
  if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12):
    raise AssertionError('CKA matrix is not symmetric')
  return matrix
