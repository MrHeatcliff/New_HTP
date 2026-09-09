"""Controlled synthetic mechanism test; not an Atari control-performance claim."""
import json
import argparse
from pathlib import Path
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import optax
from dreamerv3.htp import MultiStridePDyn, prefix_persistence, prefix_isotropy


def dataset(seed, episodes=128, length=48):
  rng = np.random.default_rng(seed)
  goal = rng.uniform(-1, 1, (episodes, 1)).repeat(length, 1)
  action = rng.uniform(-1, 1, (episodes, length, 1))
  position = np.zeros((episodes, length))
  position[:, 0] = rng.uniform(-1, 1, episodes)
  for t in range(1, length):
    position[:, t] = .95 * position[:, t-1] + .2 * action[:, t-1, 0]
  phase = rng.uniform(-np.pi, np.pi, (episodes, 1)) + .7 * np.arange(length)
  state = np.stack([goal, position, np.cos(phase), np.sin(phase),
                    rng.normal(size=goal.shape), rng.normal(size=goal.shape)], -1)
  return state.astype('float32'), action.astype('float32')


def ridge(x, y, test):
  x = np.c_[x, np.ones(len(x))]; test = np.c_[test, np.ones(len(test))]
  return test @ np.linalg.solve(x.T @ x + .01 * np.eye(x.shape[1]), x.T @ y)


def r2(y, p):
  return float(1 - np.square(y-p).sum() / np.maximum(np.square(y-y.mean(0)).sum(), 1e-12))


def run(seed, scale, steps=1000, all_lags=False, isotropy=0., metric='pooled'):
  train, actions = dataset(seed)
  test, _ = dataset(seed + 10000)
  reset = np.zeros(train.shape[:2], bool); reset[:, 0] = True
  module = MultiStridePDyn(dims=(2,4,6), strides=(16,4,1),
                          hidden=32, layers=1, act='gelu', name='pdyn')
  pure = nj.pure(lambda z, a: module(z.astype(jnp.bfloat16),
      jax.lax.stop_gradient(z).astype(jnp.bfloat16), a.astype(jnp.bfloat16))[0].mean().astype(jnp.float32))
  initial = jnp.asarray(np.random.default_rng(seed).normal(size=(6,6)) * .2)
  params_dyn, _ = pure({}, jnp.asarray(train) @ initial, jnp.asarray(actions), seed=seed, create=True)
  params = {'encoder': initial, 'decoder': jnp.eye(6), 'dyn': params_dyn}
  optimizer = optax.adam(1e-3); state = optimizer.init(params)
  def objective(p, x, a):
    z = x @ p['encoder']
    _, prediction = pure(p['dyn'], z, a, seed=0, create=False, modify=False)
    reconstruction = jnp.square(z @ p['decoder'] - x).mean()
    return (reconstruction + prediction
            + scale * prefix_persistence(z, jnp.asarray(reset), 2, 16, all_lags, metric)
            + isotropy * prefix_isotropy(z, 2))
  @jax.jit
  def step(p, s):
    loss, grad = jax.value_and_grad(objective)(p, jnp.asarray(train), jnp.asarray(actions))
    updates, s = optimizer.update(grad, s, p)
    return optax.apply_updates(p, updates), s, loss
  for _ in range(steps):
    params, state, loss = step(params, state)
  encoder = np.asarray(params['encoder']); z = train @ encoder; v = test @ encoder
  slow = ridge(z[..., :2].reshape(-1,2), train[..., :1].reshape(-1,1), v[..., :2].reshape(-1,2))
  future = ridge(z[:, :-16, :2].reshape(-1,2), train[:, 16:, :4].reshape(-1,4), v[:, :-16, :2].reshape(-1,2))
  # Frozen representation -> fitted controller -> new closed-loop trajectories.
  # Full code matches the production actor input; compact-only is diagnostic.
  target_action = np.clip((train[..., 0] - .95 * train[..., 1]) / .2, -1, 1)
  returns = {}
  for dim in (2,6):
    rng = np.random.default_rng(seed + 20000)
    goal = rng.uniform(-1,1,128); position = rng.uniform(-1,1,128)
    phase = rng.uniform(-np.pi,np.pi,128); total = np.zeros(128)
    for t in range(100):
      obs = np.stack([goal, position, np.cos(phase+.7*t), np.sin(phase+.7*t),
                      rng.normal(size=128), rng.normal(size=128)], -1)
      action = ridge(z[..., :dim].reshape(-1,dim), target_action.ravel(), (obs @ encoder)[:, :dim])
      position = .95 * position + .2 * np.clip(action, -1, 1)
      total -= np.square(position-goal)
    returns[str(dim)] = float(total.mean())
  covariance = np.cov(v[..., :2].reshape(-1,2).T)
  eig = np.linalg.eigvalsh(covariance)
  redundancy = ridge(z[..., 2:].reshape(-1,4), z[..., :2].reshape(-1,2), v[..., 2:].reshape(-1,4))
  horizons = {}
  for lag in (1,4,8,16):
    predicted = ridge(z[:, :-lag, :2].reshape(-1,2), train[:, lag:, :2].reshape(-1,2), v[:, :-lag, :2].reshape(-1,2))
    horizons[str(lag)] = r2(test[:, lag:, :2].reshape(-1,2), predicted)
  return dict(seed=seed, scale=scale, metric=metric, all_lags=all_lags, isotropy=isotropy, steps=steps, loss=float(loss), slow_r2=r2(test[..., :1].reshape(-1,1), slow),
              control_state_future_r2=horizons,
              future_r2=r2(test[:, 16:, :4].reshape(-1,4), future),
              persistence=float(prefix_persistence(jnp.asarray(v), jnp.asarray(reset), 2,16)),
              min_variance=float(eig.min()), effective_rank=float(eig.sum()**2/(eig**2).sum()),
              redundancy_r2=r2(v[..., :2].reshape(-1,2), redundancy), control_return=returns)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--steps', type=int, default=1000)
  parser.add_argument('--refinement', action='store_true')
  parser.add_argument('--whitening', action='store_true')
  parser.add_argument('--output-root', default='paper_artifacts/persistence_research')
  args = parser.parse_args()
  output = Path(args.output_root); output.mkdir(parents=True, exist_ok=True)
  rows = []
  if args.whitening:
    for seed in range(3):
      for metric in ('pooled', 'whitened'):
        row = run(seed, 1., steps=args.steps, all_lags=True, isotropy=.01, metric=metric)
        rows.append(row); print(json.dumps(row), flush=True)
        (output / f'synthetic_whitening_{args.steps}.json').write_text(json.dumps(rows,indent=2))
    raise SystemExit(0)
  for seed in range(3):
    variants = ((0.,False,0.),(.1,True,0.),(.1,True,.1)) if args.refinement else (
        (0.,False,0.),(.01,False,0.),(.1,False,0.),(.1,True,0.))
    for scale, all_lags, isotropy in variants:
      row = run(seed, scale, steps=args.steps, all_lags=all_lags, isotropy=isotropy)
      rows.append(row); print(json.dumps(row), flush=True)
      filename = f'synthetic_refinement_{args.steps}.json' if args.refinement else 'synthetic_results.json'
      (output / filename).write_text(json.dumps(rows, indent=2))
