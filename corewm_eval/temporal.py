import numpy as np
from scipy.stats import spearmanr


def predictability_half_life(horizons, r2):
  horizons, r2 = np.asarray(horizons), np.asarray(r2, np.float64)
  if r2[0] <= 0:
    return np.nan
  qualified = horizons[r2 >= 0.5 * r2[0]]
  return float(qualified.max()) if len(qualified) else np.nan


def spearman_all_finite(half_lives):
  half_lives = np.asarray(half_lives, np.float64)
  if len(half_lives) != 5 or not np.isfinite(half_lives).all():
    return np.nan
  return float(spearmanr(np.arange(1, 6), half_lives).statistic)


def mean_long_horizon_r2(matrix, horizons, long_horizons=(16, 32, 64)):
  matrix, horizons = np.asarray(matrix, np.float64), np.asarray(horizons)
  indices = [int(np.where(horizons == k)[0][0]) for k in long_horizons]
  per_block = matrix[:, indices].mean(axis=1)
  return float(per_block.mean()), per_block
