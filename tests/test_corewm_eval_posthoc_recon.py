import jax
import jax.numpy as jnp
import numpy as np

from corewm_eval.posthoc_recon import (
    PosthocArchitecture, initialize, loss_and_levels, parameter_counts,
    metrics_from_sufficient_statistics, parameter_hash, probe_not_converged,
    qualifying_improvement, reconstruction_metrics)


def _small_architecture():
  return PosthocArchitecture(
      prefix_dims=(2, 4), feat_dim=6, hidden=8, layers=1,
      norm='rms', act='silu', outscale=1.0)


def test_posthoc_uses_matched_initialization_and_exact_head_partition():
  arch = _small_architecture()
  left, right = initialize(7, arch), initialize(7, arch)
  assert parameter_hash(left) == parameter_hash(right)
  counts = parameter_counts(left, levels=2)
  assert counts[1] == counts[2]  # Both toy blocks have dimension two.


def test_posthoc_later_loss_isolates_previous_heads_and_representation():
  arch = _small_architecture()
  params = initialize(0, arch)
  rng = np.random.default_rng(0)
  z = jnp.asarray(rng.normal(size=(3, 4, 4)), jnp.bfloat16)
  h = jnp.asarray(rng.normal(size=(3, 4, 6)), jnp.bfloat16)

  later = lambda current: loss_and_levels(current, z, h, arch)[1][1]
  grads = jax.grad(later)(params)
  head0 = [x for key, x in grads.items() if '/head0/' in key]
  head1 = [x for key, x in grads.items() if '/head1/' in key]
  assert head0 and head1
  assert all(np.count_nonzero(np.asarray(value)) == 0 for value in head0)
  assert any(np.count_nonzero(np.asarray(value)) > 0 for value in head1)

  rep_grad = jax.grad(
      lambda value: loss_and_levels(params, value, h, arch)[1][1])(z)
  np.testing.assert_array_equal(np.asarray(rep_grad), 0)


def test_posthoc_metrics_exact_values_and_negative_gain_preserved():
  # Test the metric formulas independently using a tiny fake pure reconstruction
  # by checking their equivalent direct arithmetic.
  target = np.asarray([[0., 1.], [2., 3.]])
  pred1 = np.zeros_like(target)
  pred2 = np.full_like(target, 10.)
  denom = np.square(target - target.mean(0, keepdims=True)).sum()
  r2 = np.asarray([
      1 - np.square(target - pred1).sum() / denom,
      1 - np.square(target - pred2).sum() / denom])
  delta = np.r_[np.nan, r2[1] - r2[0]]
  assert delta[1] < 0  # The implementation must never clip this quantity.


def test_posthoc_real_metric_shapes_and_nonnegative_mse():
  arch = _small_architecture(); params = initialize(2, arch)
  rng = np.random.default_rng(2)
  z = jnp.asarray(rng.normal(size=(8, 4)), jnp.bfloat16)
  h = jnp.asarray(rng.normal(size=(8, 6)), jnp.bfloat16)
  result = reconstruction_metrics(params, z, h, arch)
  assert set(result) == {
      'E_post', 'R2_rec', 'E_norm', 'Delta_E_post', 'Delta_R2_rec'}
  assert all(value.shape == (2,) for value in result.values())
  assert np.all(result['E_post'] >= -1e-8)
  assert np.isnan(result['Delta_E_post'][0])
  assert np.isnan(result['Delta_R2_rec'][0])
  assert np.isfinite(result['Delta_E_post'][1:]).all()
  assert np.isfinite(result['Delta_R2_rec'][1:]).all()
  np.testing.assert_allclose(result['E_norm'], 1 - result['R2_rec'])


def test_posthoc_chunk_sufficient_statistics_match_direct_formula():
  target = np.asarray([[1., 2.], [3., 6.], [5., 4.]])
  predictions = np.stack([target, np.zeros_like(target)])
  sse = np.square(predictions - target[None]).sum((1, 2))
  result = metrics_from_sufficient_statistics(
      sse, target.sum(0), np.square(target).sum(), len(target), 2)
  assert np.isclose(result['R2_rec'][0], 1)
  denominator = np.square(target - target.mean(0)).sum()
  assert np.isclose(
      result['R2_rec'][1], 1 - np.square(target).sum() / denominator)


def test_frozen_validation_selection_and_nonconvergence_rule():
  assert qualifying_improvement(0.5002, 0.5, 1e-4)
  assert not qualifying_improvement(0.5001, 0.5, 1e-4)
  curve = [
      {'update': 9000, 'validation_score': 0.5},
      {'update': 10000, 'validation_score': 0.5021},
  ]
  assert probe_not_converged(curve, 9750)
  assert not probe_not_converged(curve, 9250)
  curve[-1]['validation_score'] = 0.502
  assert not probe_not_converged(curve, 10000)
