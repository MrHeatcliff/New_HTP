import numpy as np
from corewm_eval.manuscript_eval import residual_blocks
from corewm_eval.metrics import cka_matrix


def test_block_decoding_keeps_only_its_residual_including_bias():
  residuals = [np.array([3., 7.]), np.array([-2., 4.]), np.array([8., -9.])]
  cumulative = tuple(np.cumsum(residuals, axis=0))
  for actual, expected in zip(residual_blocks(cumulative), residuals):
    np.testing.assert_array_equal(actual, expected)


def test_complete_cka_with_unequal_block_widths():
  rng = np.random.default_rng(0)
  x = rng.normal(size=(40, 3))
  matrix = cka_matrix([x, 7 * x + 12, rng.normal(size=(40, 9))])
  np.testing.assert_allclose(np.diag(matrix), 1)
  np.testing.assert_allclose(matrix, matrix.T)
  assert np.isclose(matrix[0, 1], 1)
  assert matrix[0, 2] < matrix[0, 1]
