import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import pytest
from dreamerv3.htp import MultiStridePDyn


def fixture(masked=True):
  model = MultiStridePDyn(dims=(2,),strides=(2,),layers=0,outscale=0.,
                         mask_episode_boundaries=masked,name='pdyn')
  z = jnp.ones((1,6,2),jnp.bfloat16)
  actions = jnp.ones((1,6,1),jnp.bfloat16)
  reset = jnp.array([[True,False,False,True,False,False]])
  pure = nj.pure(lambda target, flags: model(z,target,actions,reset=flags))
  params,_ = pure({},z,reset,seed=0,create=True)
  return model,pure,params,z,actions,reset


def test_mask_removes_only_pairs_crossing_reset_in_real_loss():
  _,pure,params,z,_,reset = fixture()
  _,(loss,levels) = pure(params,z,reset,seed=0,create=False,modify=False)
  np.testing.assert_array_equal(loss,[[1,0,0,1,0,0]])
  assert float(levels[0]['cross_episode_fraction']) == .5
  assert float(levels[0]['err_mean']) == 1.
  _,(legacy,_) = fixture(False)[1]({},z,reset,seed=0,create=True)
  np.testing.assert_array_equal(legacy,[[1,1,1,1,0,0]])


def test_all_reset_pairs_zero_and_target_gradient_remains_stopped():
  _,pure,params,z,_,reset = fixture()
  flags = jnp.ones_like(reset)
  loss = lambda target: pure(params,target,flags,seed=0,create=False,modify=False)[1][0].astype(jnp.float32).sum()
  assert float(loss(z)) == 0
  np.testing.assert_array_equal(jax.grad(loss)(z),0)


def test_mask_requires_correct_reset_input():
  model,_,_,z,actions,_ = fixture()
  with pytest.raises(ValueError,match='requires reset'):
    nj.pure(lambda: model(z,z,actions))({},seed=0,create=True)
