import numpy as np
import jax
import jax.numpy as jnp

from corewm_eval import metrics
from corewm_eval.config import ATARI100K_GAMES, wandb_project_for_game
from corewm_eval.policy import human_normalized_score, normalized_auc, ordered_game_values
from corewm_eval.policy import score_matrix_from_seed_dicts
from corewm_eval.policy import rliable_aggregate_intervals
from corewm_eval.slicing import split_prefixes_blocks
from corewm_eval.temporal import predictability_half_life, spearman_all_finite


def test_prefix_block_exact_slicing():
  z = np.arange(2 * 2048).reshape(2, 2048)
  result = split_prefixes_blocks(z, [128, 256, 512, 1024, 2048])
  assert result.block_dims == (128, 128, 256, 512, 1024)
  np.testing.assert_array_equal(result.prefixes[1][..., :128], result.prefixes[0])
  np.testing.assert_array_equal(np.concatenate(result.blocks[:3], -1), result.prefixes[2])
  np.testing.assert_array_equal(np.concatenate(result.blocks, -1), z)


def test_progressive_reconstruction_arithmetic_and_gains():
  residuals = [np.array([1., 2.]), np.array([3., 4.]), np.array([-1., 1.])]
  cumulative = metrics.progressive_sum(residuals)
  np.testing.assert_allclose(cumulative, [[1, 2], [4, 6], [3, 7]])
  errors = np.array([4.0, 3.0, 1.5])
  gains = np.r_[np.nan, errors[:-1] - errors[1:]]
  np.testing.assert_allclose(gains[1:], [1.0, 1.5])


def test_progressive_reconstruction_refactor_values_loss_and_gradients():
  from dreamerv3.htp import cumulative_residual_reconstruction
  residuals = tuple(jnp.asarray(x) for x in np.random.default_rng(0).normal(size=(3, 2, 5)))
  target = jnp.asarray(np.random.default_rng(1).normal(size=(2, 5)))
  def old_formula(xs):
    current, outputs = jnp.zeros_like(target), []
    for residual in xs:
      current = jax.lax.stop_gradient(current) + residual
      outputs.append(current)
    return tuple(outputs)
  old, new = old_formula(residuals), cumulative_residual_reconstruction(residuals, target)
  for left, right in zip(old, new):
    np.testing.assert_allclose(left, right, rtol=0, atol=0)
  loss = lambda fn, xs: sum(jnp.square(target - value).mean() for value in fn(xs)) / len(xs)
  np.testing.assert_allclose(loss(old_formula, residuals), loss(
      lambda xs: cumulative_residual_reconstruction(xs, target), residuals))
  old_grad = jax.grad(lambda *xs: loss(old_formula, xs), argnums=(0, 1, 2))(*residuals)
  new_grad = jax.grad(lambda *xs: loss(
      lambda ys: cumulative_residual_reconstruction(ys, target), xs),
      argnums=(0, 1, 2))(*residuals)
  for left, right in zip(old_grad, new_grad):
    np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-6)


def test_canonical_game_order_and_exact_set():
  values = {game: index for index, game in enumerate(reversed(ATARI100K_GAMES))}
  expected = [values[game] for game in ATARI100K_GAMES]
  np.testing.assert_array_equal(ordered_game_values(values), expected)
  import pytest
  with pytest.raises(ValueError):
    ordered_game_values({game: 0 for game in ATARI100K_GAMES[:-1]})


def test_wandb_project_routing_covers_canonical_games():
  projects = [wandb_project_for_game(game) for game in ATARI100K_GAMES]
  assert len(projects) == 26
  assert len(set(projects)) == 26
  assert wandb_project_for_game('alien') == 'dreamv3-alien'
  assert wandb_project_for_game('jamesbond') == 'dreamv3-james_bond'
  assert wandb_project_for_game('up_n_down') == 'dreamv3-up_n_down'


def test_score_matrix_canonical_order_ignores_mapping_insertion_order():
  rng = np.random.default_rng(7)
  result = {}
  for seed in rng.permutation(5):
    order = rng.permutation(len(ATARI100K_GAMES))
    result[int(seed)] = {
        ATARI100K_GAMES[index]: seed * 100 + index for index in order}
  matrix = score_matrix_from_seed_dicts(result)
  expected = np.asarray([
      [seed * 100 + index for index in range(26)] for seed in range(5)])
  np.testing.assert_array_equal(matrix, expected)


def test_nonnegative_metrics_fail_without_clamping():
  import pytest
  metrics.assert_nonnegative('ok', [0.0, 1.0, -1e-9])
  with pytest.raises(AssertionError):
    metrics.assert_nonnegative('bad', [0.0, -1e-7])


def test_repository_manuscript_hns_table_matches_baselines(tmp_path):
  from corewm_eval.hns_reference import audit_repository_manuscript_table
  rows = audit_repository_manuscript_table(tmp_path / 'diff.csv')
  assert len(rows) == 26 and all(row['match'] for row in rows)


def test_multivariate_r2_perfect_mean_and_negative():
  y = np.arange(12, dtype=float).reshape(6, 2)
  assert np.isclose(metrics.multivariate_r2(y, y), 1)
  mean = np.broadcast_to(y.mean(0), y.shape)
  assert np.isclose(metrics.multivariate_r2(y, mean), 0)
  assert metrics.multivariate_r2(y, -y) < 0


def test_linear_cka_properties():
  rng = np.random.default_rng(0)
  x, y = rng.normal(size=(100, 7)), rng.normal(size=(100, 11))
  assert np.isclose(metrics.linear_cka(x, x), 1)
  assert np.isclose(metrics.linear_cka(x, y), metrics.linear_cka(y, x))
  matrix = metrics.cka_matrix([x, y])
  assert matrix.shape == (2, 2) and np.isfinite(matrix).all()


def test_hns_unclipped():
  assert human_normalized_score(2, 2, 12) == 0
  assert human_normalized_score(12, 2, 12) == 1
  assert human_normalized_score(22, 2, 12) > 1


def test_auc_constant_and_two_point():
  checkpoints = np.array([10_000, 50_000, 100_000])
  assert np.isclose(normalized_auc(checkpoints, np.full(3, 2.5)), 2.5)
  expected = (0.5 * (0 + 1) * 40_000 + 0.5 * (1 + 3) * 50_000) / 90_000
  assert np.isclose(normalized_auc(checkpoints, [0, 1, 3]), expected)


def test_half_life_literal_max_and_nan_spearman():
  assert predictability_half_life([1, 2, 4, 8, 16], [.8, .7, .5, .39, .2]) == 4
  assert predictability_half_life([1, 2, 4, 8], [.8, .2, .5, .1]) == 4
  assert np.isnan(predictability_half_life([1, 2], [-.1, .2]))
  assert np.isnan(spearman_all_finite([1, 2, np.nan, 8, 16]))


def test_rliable_wrapper_point_estimates_match_library_metrics():
  from rliable import metrics as rliable_metrics
  scores = np.arange(130, dtype=float).reshape(5, 26) / 10
  points, intervals = rliable_aggregate_intervals({'toy': scores}, reps=100, seed=0)
  expected = np.array([
      rliable_metrics.aggregate_mean(scores),
      rliable_metrics.aggregate_median(scores),
      rliable_metrics.aggregate_iqm(scores)])
  np.testing.assert_allclose(points['toy'], expected)
  assert intervals['toy'].shape == (2, 3)
