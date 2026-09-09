import jax
import jax.numpy as jnp
import numpy as np
from dreamerv3.htp import prefix_isotropy, prefix_rank_floor, prefix_participation_rank


def test_sufficient_diversity_stops_pressure_to_fill_unused_dimensions():
  x=jnp.array([[1.,1.,0.,0.],[1.,-1.,0.,0.],[-1.,1.,0.,0.],[-1.,-1.,0.,0.]])
  assert float(prefix_isotropy(x,4)) > 0
  assert float(prefix_rank_floor(x,4,1.5)) == 0
  np.testing.assert_array_equal(jax.grad(lambda y: prefix_rank_floor(y,4,1.5))(x),0)


def test_below_floor_keeps_isotropy_gradient_and_prefix_isolation():
  x=jnp.array([[1.,.1,0.,7.],[1.,-.1,0.,8.],[-1.,.1,0.,9.],[-1.,-.1,0.,10.]])
  a=jax.grad(lambda y: prefix_rank_floor(y,3,2.5))(x)
  b=jax.grad(lambda y: prefix_isotropy(y,3))(x)
  np.testing.assert_allclose(a,b,atol=1e-5,rtol=1e-4)
  np.testing.assert_array_equal(a[:,-1],0)
  np.testing.assert_allclose(prefix_rank_floor(x*.01,3,2.5),prefix_rank_floor(x,3,2.5),rtol=1e-4)


def test_constant_representation_is_penalized_and_gradients_finite():
  x=jnp.ones((2,8,4))
  assert float(prefix_rank_floor(x,4,2)) > 0
  assert np.isfinite(jax.grad(lambda y: prefix_rank_floor(y,4,2))(x)).all()


def test_second_block_guard_isolates_direct_gradients_and_reports_collapse():
  x=jnp.array([[1.,1.,.1,9.],[1.,1.,-.1,8.],[-1.,-1.,.1,7.],[-1.,-1.,-.1,6.]])
  penalty=lambda z:prefix_rank_floor(z[...,1:3],2,1.8)
  grad=jax.grad(penalty)(x)
  np.testing.assert_array_equal(grad[:,[0,3]],0)
  assert float(jnp.linalg.norm(grad[:,1:3])) > 0
  assert float(prefix_participation_rank(jnp.ones((8,2)),2)) == 0
