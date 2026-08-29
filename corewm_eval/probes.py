from dataclasses import asdict, dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .config import ALPHA_GRID
from .metrics import multivariate_r2


@dataclass
class RidgeResult:
  selected_alpha: float
  validation_r2_grid: dict
  test_r2: float
  train_samples: int
  validation_samples: int
  test_samples: int
  scaler_metadata: dict

  def to_dict(self):
    return asdict(self)


def fit_ridge_protocol(x_train, y_train, x_val, y_val, x_test, y_test, alphas=ALPHA_GRID):
  arrays = [np.asarray(x, np.float64) for x in (x_train, y_train, x_val, y_val, x_test, y_test)]
  x_train, y_train, x_val, y_val, x_test, y_test = arrays
  x_scaler, y_scaler = StandardScaler(), StandardScaler()
  xs, ys = x_scaler.fit_transform(x_train), y_scaler.fit_transform(y_train)
  xv = x_scaler.transform(x_val)
  grid = {}
  best_alpha, best_score = None, -np.inf
  for alpha in sorted(float(x) for x in alphas):
    model = Ridge(alpha=alpha, fit_intercept=True).fit(xs, ys)
    pred = y_scaler.inverse_transform(model.predict(xv))
    score = multivariate_r2(y_val, pred)
    grid[f'{alpha:.17g}'] = score
    if score > best_score:
      best_alpha, best_score = alpha, score
  x_tv, y_tv = np.concatenate([x_train, x_val]), np.concatenate([y_train, y_val])
  x_final, y_final = StandardScaler(), StandardScaler()
  model = Ridge(alpha=best_alpha, fit_intercept=True).fit(
      x_final.fit_transform(x_tv), y_final.fit_transform(y_tv))
  prediction = y_final.inverse_transform(model.predict(x_final.transform(x_test)))
  result = RidgeResult(
      best_alpha, grid, multivariate_r2(y_test, prediction), len(x_train), len(x_val), len(x_test),
      {'x_mean': x_final.mean_.tolist(), 'x_scale': x_final.scale_.tolist(),
       'y_mean': y_final.mean_.tolist(), 'y_scale': y_final.scale_.tolist()})
  return result, prediction
