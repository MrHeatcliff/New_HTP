import numpy as np
import jax
import jax.numpy as jnp
import ninjax as nj


def test_driver_marks_reset_callbacks_but_counts_only_action_transitions():
  import embodied
  from embodied.envs import dummy

  env = dummy.Dummy('disc', length=3)
  agent = embodied.RandomAgent(env.obs_space, env.act_space)
  env.close()
  driver = embodied.Driver([lambda: dummy.Dummy('disc', length=3)])
  driver.reset(agent.init_policy)
  transitions = []
  driver.on_step(lambda tran, _: transitions.append(tran))
  driver(agent.policy, episodes=2)
  flags = [bool(x['log/action_executed']) for x in transitions]
  first = [bool(x['is_first']) for x in transitions]
  last = [bool(x['is_last']) for x in transitions]
  assert first == [True, False, False, False, True, False, False, False]
  assert last == [False, False, False, True, False, False, False, True]
  assert flags == [False, True, True, True, False, True, True, True]
  assert sum(flags) == 6


def test_paper_artifact_base_uses_bound_action_and_raw_frame_counters(tmp_path):
  import elements
  from embodied.run.paper_artifacts import PaperArtifactWriter

  args = elements.Config(
      task='atari100k_alien', seed=0,
      env={'atari100k': {'repeat': 4}}, agent={},
      batch_size=16, batch_length=64, train_ratio=256,
      exact_env_action_budget=True)
  writer = PaperArtifactWriter(tmp_path, args)
  actions, resets = elements.Counter(10), elements.Counter(2)
  replay, frames = elements.Counter(12), elements.Counter(39)
  writer.bind_runtime_counters(
      actions, resets, replay, frames, optimizer_updates=lambda: 3)
  row = writer._base(elements.Counter(12))
  assert row['step'] == 10
  assert row['legacy_logger_step'] == 12
  assert row['driver_callbacks'] == 12
  assert row['env_steps'] == row['agent_actions'] == 10
  assert row['frames'] == row['realized_frames'] == 39
  assert row['reset_callbacks'] == 2
  assert row['replay_insertions'] == 12
  assert row['optimizer_updates'] == 3
  assert np.isclose(row['reset_callbacks_per_env_action'], 0.2)
  assert np.isclose(row['optimizer_updates_per_env_action'], 0.3)
  assert np.isclose(row['expected_updates_per_driver_callback'], 0.25)
  assert np.isclose(row['expected_updates_per_agent_action'], 0.30)


def test_full_and_reverse_temporal_targets_and_action_windows():
  from dreamerv3.htp import MultiStridePDyn

  expected = {
      'full': (16, 8, 4, 2, 1),
      'reverse': (1, 2, 4, 8, 16),
  }
  for strides in expected.values():
    for stride in strides:
      source, actions, target = MultiStridePDyn.temporal_indices(20, stride)
      source, actions, target = map(np.asarray, (source, actions, target))
      np.testing.assert_array_equal(target, source + stride)
      for row, t in zip(actions, source):
        np.testing.assert_array_equal(row, np.arange(t, t + stride))


def test_enforce_coarse_to_fine_only_validates_stride_order():
  from dreamerv3.htp import MultiStridePDyn
  import pytest

  with pytest.raises(AssertionError):
    MultiStridePDyn(
        dims=(2, 4), strides=(1, 2), enforce_coarse_to_fine=True,
        name='invalid')
  reverse = MultiStridePDyn(
      dims=(2, 4), strides=(1, 2), enforce_coarse_to_fine=False,
      name='reverse')
  assert tuple(reverse.dims) == (2, 4)
  assert tuple(reverse.strides) == (1, 2)


def test_multistride_gradient_isolation_and_ema_target_stop_gradient():
  from dreamerv3.htp import MultiStridePDyn

  module = MultiStridePDyn(
      dims=(2, 4), strides=(2, 1), hidden=8, layers=1, norm='rms',
      act='gelu', outscale=1.0, name='pdyn')
  z = jnp.ones((1, 5, 4), jnp.bfloat16)
  grad = jax.grad(lambda value: module._gi_prefix(value, 1).sum())(z)
  np.testing.assert_array_equal(np.asarray(grad[..., :2]), 0)
  np.testing.assert_array_equal(np.asarray(grad[..., 2:4]), 1)

  one = MultiStridePDyn(
      dims=(2,), strides=(1,), hidden=8, layers=1, norm='rms',
      act='gelu', outscale=1.0, name='one')
  pure = nj.pure(lambda online, target, actions: one(online, target, actions)[0])
  online = jnp.ones((1, 4, 2), jnp.bfloat16)
  target = jnp.ones((1, 4, 2), jnp.bfloat16)
  actions = jnp.ones((1, 4, 3), jnp.bfloat16)
  params, _ = pure({}, online, target, actions, seed=0, create=True)

  def loss(slow_target):
    _, value = pure(
        params, online, slow_target, actions, seed=0,
        create=False, modify=False)
    return value.mean().astype(jnp.float32)

  target_grad = jax.grad(loss)(target)
  np.testing.assert_array_equal(np.asarray(target_grad), 0)


def test_actual_progressive_heads_optimize_and_stop_gradient_isolates_levels():
  from dreamerv3.htp import ProgressiveRecon

  module = ProgressiveRecon(
      feat_dim=6, dims=(2, 4), hidden=16, layers=1, norm='rms',
      act='gelu', outscale=1.0, name='recon')
  pure = nj.pure(lambda z, h: module(z, h))
  rng = np.random.default_rng(0)
  z = jnp.asarray(rng.normal(size=(8, 4, 4)), jnp.bfloat16)
  h = jnp.asarray(rng.normal(size=(8, 4, 6)), jnp.bfloat16)
  params, _ = pure({}, z, h, seed=0, create=True)

  def level_losses(current):
    _, (_, levels) = pure(
        current, z, h, seed=0, create=False, modify=False)
    return jnp.stack([x.mean().astype(jnp.float32) for x in levels])

  before = np.asarray(level_losses(params))
  objective = lambda current: level_losses(current).mean()
  for _ in range(100):
    grads = jax.grad(objective)(params)
    params = jax.tree.map(lambda value, grad: value - 0.02 * grad, params, grads)
  after = np.asarray(level_losses(params))
  assert np.isfinite(after).all()
  assert np.all(after < before), (before, after)

  later_loss = lambda current: level_losses(current)[1]
  later_grads = jax.grad(later_loss)(params)
  head0 = [value for key, value in later_grads.items() if '/head0/' in key]
  head1 = [value for key, value in later_grads.items() if '/head1/' in key]
  assert head0 and head1
  assert all(np.count_nonzero(np.asarray(value)) == 0 for value in head0)
  assert any(np.count_nonzero(np.asarray(value)) > 0 for value in head1)
