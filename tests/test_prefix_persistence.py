import jax
import jax.numpy as jnp
import numpy as np
from dreamerv3.htp import prefix_persistence, prefix_isotropy


def test_episode_constant_codes_and_boundaries():
  z = jnp.array([[[1.], [1.], [9.], [9.]]])
  reset = jnp.array([[True, False, True, False]])
  assert float(prefix_persistence(z, reset, 1, 1)) == 0
  assert float(prefix_persistence(z, reset, 1, 2)) == 0
  assert float(prefix_persistence(z, jnp.zeros_like(reset), 1, 1)) > 0


def test_scale_invariance_and_only_compact_prefix_gradient():
  z = jnp.arange(24, dtype=jnp.float32).reshape(2, 4, 3)
  reset = jnp.zeros((2, 4), bool)
  value = prefix_persistence(z, reset, 1, 1)
  np.testing.assert_allclose(value, prefix_persistence(z * .01, reset, 1, 1), rtol=1e-5)
  grad = jax.grad(lambda x: prefix_persistence(x, reset, 1, 1))(z)
  assert np.isfinite(grad).all()
  np.testing.assert_array_equal(grad[..., 1:], 0)


def test_no_pairs_and_collapsed_code_are_finite():
  z = jnp.ones((2, 4, 3)); reset = jnp.ones((2, 4), bool)
  for stride in (1, 4, 8):
    assert float(prefix_persistence(z, reset, 2, stride)) == 0
    grad = jax.grad(lambda x: prefix_persistence(x, reset, 2, stride))(z)
    assert np.isfinite(grad).all()


def test_all_lags_detects_periodic_endpoint_alias():
  z = jnp.tile(jnp.array([1., -1.]), 8).reshape(1, 16, 1)
  reset = jnp.zeros((1,16), bool)
  assert float(prefix_persistence(z, reset, 1, 4)) == 0
  assert float(prefix_persistence(z, reset, 1, 4, all_lags=True)) > .5


def test_isotropy_detects_duplicate_coordinates():
  balanced = jnp.array([[1.,1.], [1.,-1.], [-1.,1.], [-1.,-1.]])
  duplicate = jnp.stack([balanced[:,0], balanced[:,0]],-1)
  assert float(prefix_isotropy(balanced,2)) == 0
  assert float(prefix_isotropy(duplicate,2)) > .9
  np.testing.assert_allclose(prefix_isotropy(duplicate*.01,2), prefix_isotropy(duplicate,2))


def test_whitening_rejects_axis_rescaling_shortcut():
  slow = jnp.broadcast_to(jnp.arange(4)[:,None], (4,16))
  fast = jnp.broadcast_to(jnp.tile(jnp.array([1.,-1.]),8), (4,16))
  z = jnp.stack([slow,fast],-1)
  reset = jnp.zeros((4,16),bool)
  scaled = z * jnp.array([10.,1.])
  assert prefix_persistence(scaled,reset,2,1) < .03 * prefix_persistence(z,reset,2,1)
  np.testing.assert_allclose(
      prefix_persistence(z,reset,2,1,metric='whitened'),
      prefix_persistence(scaled,reset,2,1,metric='whitened'),rtol=.01)


def test_whitening_masks_boundaries_and_keeps_degenerate_gradients_finite():
  z = jnp.ones((2,4,3))
  reset = jnp.ones((2,4),bool)
  fn = lambda value: prefix_persistence(value,reset,2,2,True,'whitened')
  assert float(fn(z)) == 0
  assert np.isfinite(jax.grad(fn)(z)).all()


def test_whitening_preserves_one_dimensional_objective():
  z = jnp.arange(24,dtype=jnp.float32).reshape(2,12,1)
  reset = jnp.zeros((2,12),bool)
  np.testing.assert_allclose(prefix_persistence(z,reset,1,3),
      prefix_persistence(z,reset,1,3,metric='whitened'),rtol=.001)
