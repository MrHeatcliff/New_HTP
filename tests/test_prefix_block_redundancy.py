import jax
import jax.numpy as jnp
import numpy as np
import pytest
from dreamerv3.htp import prefix_block_redundancy


def test_copy_is_detected_but_independent_blocks_are_not():
  a=jnp.array([1.,1.,-1.,-1.])
  b=jnp.array([1.,-1.,1.,-1.])
  assert float(prefix_block_redundancy(jnp.stack([a,a],-1),1,2)) > .999
  assert float(prefix_block_redundancy(jnp.stack([a,b],-1),1,2)) == 0


def test_partial_redundancy_has_only_second_block_gradient():
  a=jnp.array([1.,1.,-1.,-1.]); b=jnp.array([1.,-1.,1.,-1.])
  z=jnp.stack([a,a+b,b],-1)
  value=lambda x: prefix_block_redundancy(x,1,2)
  grad=jax.grad(value)(z)
  np.testing.assert_array_equal(grad[:,[0,2]],0)
  assert float(jnp.linalg.norm(grad[:,1])) > 0
  np.testing.assert_allclose(value(z*jnp.array([.01,10.,1.])),value(z),rtol=1e-5)


def test_degenerate_blocks_finite_and_invalid_boundaries_rejected():
  z=jnp.ones((2,8,4))
  fn=lambda x:prefix_block_redundancy(x,1,3)
  assert float(fn(z))==0
  assert np.isfinite(jax.grad(fn)(z)).all()
  with pytest.raises(ValueError): prefix_block_redundancy(z,2,2)
