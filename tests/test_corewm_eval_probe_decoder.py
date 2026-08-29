import numpy as np

from corewm_eval.decoder import flatten_rssm_state, unflatten_rssm_state
from corewm_eval.probes import fit_ridge_protocol


def test_rssm_flat_structured_roundtrip():
  rng = np.random.default_rng(0)
  state = {'deter': rng.normal(size=(3, 8)), 'stoch': rng.normal(size=(3, 2, 4))}
  recovered = unflatten_rssm_state(flatten_rssm_state(state), 8, (2, 4))
  np.testing.assert_array_equal(recovered['deter'], state['deter'])
  np.testing.assert_array_equal(recovered['stoch'], state['stoch'])


def test_ridge_protocol_selects_without_test_leakage_and_predicts():
  rng = np.random.default_rng(0)
  x = rng.normal(size=(180, 4))
  y = x @ rng.normal(size=(4, 3)) + 0.001 * rng.normal(size=(180, 3))
  result, prediction = fit_ridge_protocol(
      x[:100], y[:100], x[100:140], y[100:140], x[140:], y[140:])
  assert result.test_r2 > .99
  assert prediction.shape == y[140:].shape
  assert result.train_samples == 100 and result.validation_samples == 40 and result.test_samples == 40
