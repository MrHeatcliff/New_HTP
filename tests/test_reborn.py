import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import pytest
from dreamerv3 import reborn as rb


def test_haar_roundtrip_parseval_and_no_coarse_overwrite():
  image = jnp.asarray(np.random.default_rng(5).normal(size=(2, 64, 64, 3)), jnp.float32)
  bands = rb.haar_bands(image)
  np.testing.assert_allclose(rb.haar_image(bands), image, atol=1e-6)
  np.testing.assert_allclose(sum((x*x).sum() for x in bands), (image*image).sum(), rtol=2e-6)
  for level in range(1, 5):
    earlier = rb.haar_bands(rb.haar_image(bands[:level]))
    later = rb.haar_bands(rb.haar_image(bands[:level+1]))
    for a, b in zip(earlier[:level], later[:level]):
      np.testing.assert_allclose(a, b, atol=1e-6)
  zeros = tuple(jnp.zeros_like(x) for x in bands)
  loss, _ = rb.refinement_loss(zeros, bands)
  np.testing.assert_allclose(loss, (image*image).mean((-3,-2,-1)), rtol=2e-6)


def test_cumulative_rate_weights_and_noise_does_not_prove_signal():
  mu = jnp.ones((2, 5))
  cost, block, prefix = rb.rate_terms(mu, (1, 2, 3, 4, 5))
  np.testing.assert_allclose(cost, 1.5)
  grad = jax.grad(lambda x: rb.rate_terms(x, (1,2,3,4,5))[0].sum())(mu)
  np.testing.assert_allclose(grad[0], [1,.8,.6,.4,.2])
  noise = rb.sample_code(jnp.zeros((2048,)), jax.random.PRNGKey(0))
  assert .8 < np.asarray(noise, float).var() < 1.2
  assert float(rb.rate_terms(jnp.zeros((1,5)), (1,2,3,4,5))[0][0]) == 0


def test_episode_safe_terminal_and_timeout_targets_and_detach():
  first = jnp.array([[True,False,False,True,False]])
  last = jnp.array([[False,False,True,False,False]])
  np.testing.assert_array_equal(rb.transition_mask(first,last), [[1,1,0,1]])
  features = jnp.ones((1,2,3))*4
  args = (features,jnp.array([[2.,3.]]),jnp.array([[True,False]]),
      jnp.ones_like(features)*10,jnp.ones((1,2))*20)
  psi,q = rb.bellman_targets(*args, .5)
  np.testing.assert_allclose(psi, [[[2,2,2],[7,7,7]]])
  np.testing.assert_allclose(q, [[2,13]])
  psi,q = rb.bellman_targets(*args, 0.)
  np.testing.assert_allclose(psi, features)
  np.testing.assert_allclose(q, args[1])
  np.testing.assert_array_equal(jax.grad(lambda x: rb.bellman_targets(x,*args[1:],.5)[0].sum())(features), jnp.zeros_like(features))
  assert float(rb.masked_mean(jnp.ones((1,2)),jnp.zeros((1,2)))) == 0


def test_module_context_and_affine_capacity():
  projection = rb.AffineProjection(dim=10, name='code')
  rec = rb.Refinement(dims=(2,4,6,8,10), hidden=8, name='rec')
  def forward():
    mu = projection(jnp.ones((1,2,12)))
    return mu, rec(mu)
  params, (mu, bands) = nj.pure(forward)({}, seed=0, create=True)
  assert mu.shape == (1,2,10)
  assert [b.shape[-1] for b in bands] == list(rb.BAND_DIMS)
  assert len([key for key in params if key.startswith('code/') and key.endswith('/kernel')]) <= 1
  bad = rb.AffineProjection(dim=13, name='bad')
  with pytest.raises(ValueError, match='dimension'):
    nj.pure(lambda: bad(jnp.ones((1,12))))({},seed=0,create=True)


def test_agent_end_to_end_and_ema():
  """Compile real model update, noisy imagination and all auxiliary heads."""
  import elements
  import ruamel.yaml as yaml
  from pathlib import Path
  from dreamerv3.agent_reborn import Agent_Reborn
  raw = yaml.YAML(typ='safe').load((Path(__file__).parents[1]/'dreamerv3/configs.yaml').read_text())
  config = elements.Config(raw['defaults']).update(raw['size1m'])
  cfg = elements.Config(**config.agent, replay_context=0).update({
      'reborn.enabled':True, 'reborn.dims':[2,4,8,16,32], 'reborn.hidden':16,
      'imag_length':2, 'imag_last':2})
  spaces = dict(image=elements.Space(np.uint8,(64,64,3)), reward=elements.Space(np.float32),
      **{k:elements.Space(bool) for k in ('is_first','is_last','is_terminal')})
  model = object.__new__(Agent_Reborn)
  model.__init__(spaces, {'action':elements.Space(np.int32,(),0,4)}, cfg)
  obs = {k:jnp.zeros((1,3,*s.shape),s.dtype) for k,s in spaces.items()}
  obs['is_first'] = obs['is_first'].at[:,0].set(True)
  act = {'action':jnp.zeros((1,3),jnp.int32)}
  def forward():
    carry = model.init_train(1)[:3]
    return model.loss(carry,obs,act,True)
  state, (loss, aux) = jax.jit(nj.pure(forward), static_argnames=('create',))({},seed=0,create=True)
  assert np.isfinite(float(loss))
  assert float(aux[-1]['reborn/valid_transition_count']) == 2
  assert any('slow_reborn_outcomes' in k for k in state)
  # A genuine optimization step checks gradient paths through scan and EMA.
  def update():
    return model.opt(model.loss, model.init_train(1)[:3], obs, act, True, has_aux=True)
  state, result = jax.jit(nj.pure(update), static_argnames=('create',))(state,seed=1,create=True)
  assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(result))
  def ema():
    model.slow_code.update()
    model.slow_outcomes.update()
  state, _ = nj.pure(ema)(state,seed=2,create=True)
  # Changing a later detail head may not remove earlier context gradients.
  assert any('reborn_refinement/' in k for k in state)


def test_real_rollout_targets_stop_after_terminal():
  from corewm_eval.reborn_diagnostic import rollout_targets
  bands = [np.ones((5,1),np.float32)*2]*5
  sf,q = rollout_targets(bands,np.array([0,3,7,100,100]),
      np.array([False,False,True,False,False]),np.array([0]),(.5,.5,.5,.5,0),horizon=4)
  np.testing.assert_allclose(q, [[6.5,6.5,6.5,6.5,3]])
  np.testing.assert_allclose(sf[0], [[1.5]])


def test_ablation_switches_are_explicit():
  from corewm_eval.reborn_experiment import command
  base = command('boxing','/tmp/reborn-test')
  assert base[base.index('--agent.htp.enabled')+1] == 'False'
  assert '--agent.reborn.enabled' in base
  for arm,key,val in [('no_rec','use_rec','False'),('no_outcome','use_outcome','False'),
      ('no_q','q_weight','0.0'),('no_rate','beta','0.0'),('no_context','context','False')]:
    assert command('boxing','/tmp/reborn-test',arm) == base+[f'--agent.reborn.{key}',val]
