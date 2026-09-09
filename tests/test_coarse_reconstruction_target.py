import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
from dreamerv3.htp import causal_episode_average, ProgressiveRecon


def test_causal_mean_never_crosses_reset_or_uses_future():
  x = jnp.array([[[1.],[3.],[5.],[100.],[200.]]])
  reset = jnp.array([[True,False,False,True,False]])
  result = causal_episode_average(x,reset,3)
  np.testing.assert_allclose(result[...,0],[[1,2,3,100,150]])
  changed = x.at[:,4].set(999.)
  np.testing.assert_array_equal(causal_episode_average(changed,reset,3)[:,:4],result[:,:4])
  np.testing.assert_array_equal(causal_episode_average(x,None,1),x)


def test_window_rolls_and_each_batch_row_has_its_own_resets():
  x = jnp.arange(1,13,dtype=jnp.float32).reshape(2,6,1)
  reset = jnp.array([[True,False,False,False,False,False],[True,False,True,False,False,False]])
  np.testing.assert_allclose(causal_episode_average(x,reset,2)[...,0],
                             [[1,1.5,2.5,3.5,4.5,5.5],[7,7.5,9,9.5,10.5,11.5]])


def test_only_first_progressive_loss_changes_and_no_parameters_are_added():
  z = jnp.ones((1,5,4),jnp.bfloat16)
  h = jnp.arange(10,dtype=jnp.bfloat16).reshape(1,5,2)
  reset = jnp.array([[True,False,False,False,False]])
  raw = ProgressiveRecon(feat_dim=2,dims=(2,4),hidden=8,layers=0,outscale=0.,name='rec')
  coarse = ProgressiveRecon(feat_dim=2,dims=(2,4),hidden=8,layers=0,outscale=0.,first_target_window=3,name='rec')
  base = nj.pure(lambda: raw(z,h,reset=reset))
  other = nj.pure(lambda: coarse(z,h,reset=reset))
  params,(_,before) = base({},seed=0,create=True)
  after_params,(_,after) = other(params,seed=0,create=False,modify=False)
  assert set(params) == set(after_params)
  np.testing.assert_array_equal(before[1],after[1])
  assert not np.array_equal(before[0],after[0])
