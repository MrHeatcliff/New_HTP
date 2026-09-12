"""Dreamer backbone with conditional Haar refinement and successor outcomes."""
import embodied.jax
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

from .agent_htp import Agent_HTP, sg, sample, prefix_fn, imag_loss, repl_loss, isimage
from .htp import flatten_action_dict
from . import reborn as rb


class Agent_Reborn(Agent_HTP):

  def __init__(self, obs_space, act_space, config):
    if config.htp.enabled:
      raise ValueError('Reborn replaces HTP objectives: set htp.enabled=false')
    super().__init__(obs_space, act_space, config)
    c = config.reborn
    self.dims = tuple(c.dims)
    if len(self.dims) != 5 or not all(a < b for a, b in zip((0, *self.dims[:-1]), self.dims)):
      raise ValueError('Require five increasing positive prefix sizes')
    if self.dims[-1] > self.feat_dim:
      raise ValueError('Affine projection D must be <= backbone feature dimension')
    if len(c.strides) != 5 or any(x < 1 for x in c.strides):
      raise ValueError('Require five positive effective horizons')
    for name in ('beta', 'lambda_rec', 'lambda_outcome', 'q_weight'):
      if not np.isfinite(c[name]) or c[name] < 0:
        raise ValueError(f'Invalid reborn.{name}')
    self.gammas = tuple(1-1/x for x in c.strides)
    self.code = rb.AffineProjection(dim=self.dims[-1], name='reborn_code')
    self.slow_code = embodied.jax.SlowModel(
        rb.AffineProjection(dim=self.dims[-1], name='slow_reborn_code'),
        source=self.code, rate=c.ema_rate)
    self.refinement = rb.Refinement(dims=self.dims, hidden=c.hidden, context=c.context, name='reborn_refinement') if c.use_rec else None
    self.outcomes = rb.Outcomes(dims=self.dims, hidden=c.hidden, name='reborn_outcomes') if c.use_outcome else None
    self.slow_outcomes = embodied.jax.SlowModel(
        rb.Outcomes(dims=self.dims, hidden=c.hidden, name='slow_reborn_outcomes'),
        source=self.outcomes, rate=c.ema_rate) if c.use_outcome else None
    self.modules += [self.code] + ([self.refinement] if c.use_rec else []) + ([self.outcomes] if c.use_outcome else [])
    self.harmony = bool(c.get('harmony', False))
    if self.harmony:
      if not (c.use_rec and c.use_outcome):
        raise ValueError('Harmony requires reconstruction and outcome tasks')
      self.harmony_obs = rb.TaskHarmonizer(name='harmony_obs')
      self.harmony_task = rb.TaskHarmonizer(name='harmony_task')
      self.modules += [self.harmony_obs,self.harmony_task]
    # Parent construction has no parameters yet; reconfigure its optimizer.
    self.opt = embodied.jax.Optimizer(self.modules, self._make_opt(**config.opt), summary_depth=1, name='opt')
    self.scales.update(reborn_rate=c.lambda_rec*c.beta)
    if c.use_rec:
      self.scales['reborn_rec'] = c.lambda_rec
    if c.use_outcome:
      self.scales['reborn_outcome'] = c.lambda_outcome

  @property
  def policy_keys(self):
    return '^(enc|dyn|dec|pol|reborn_code)/'

  def feat2tensor(self, feat):
    return rb.sample_code(self.code(self.feat2h(feat)), nj.seed(), self.config.reborn.stochastic)

  def train(self, carry, data):
    result = super().train(carry, data)
    if self.outcomes is not None:
      self.slow_code.update()
      self.slow_outcomes.update()
    return result

  def auxiliary(self, h, z, mu, obs, prevact):
    c = self.config.reborn
    losses, metrics = {}, {}
    bands = rb.haar_bands(obs['image'].astype(jnp.float32)/255)
    rate, block, prefix = rb.rate_terms(mu, self.dims, c.cumulative_rate)
    losses['reborn_rate'] = rate
    for i, (lo, hi) in enumerate(zip((0, *self.dims[:-1]), self.dims)):
      signal = mu[..., lo:hi].astype(jnp.float32)
      variance = signal.var((0, 1))
      metrics.update({f'reborn/l{i+1}/rate_nats': block[..., i].mean(),
          f'reborn/l{i+1}/prefix_rate_nats': prefix[..., i].mean(),
          f'reborn/l{i+1}/mu_variance': variance.mean(),
          f'reborn/l{i+1}/mu_dead_fraction': (variance < 1e-6).mean(),
          f'reborn/l{i+1}/noise_mse': jnp.square(z[..., lo:hi].astype(jnp.float32)-signal).mean()})
    if self.refinement is not None:
      predicted = self.refinement(z)
      losses['reborn_rec'], errors = rb.refinement_loss(predicted, bands)
      for i, err in enumerate(errors):
        metrics[f'reborn/l{i+1}/band_mse_per_pixel'] = err.mean()
        metrics[f'reborn/l{i+1}/band_zero_mse'] = (jnp.square(bands[i]).sum(-1)/rb.PIXELS).mean()
        # Orthogonality: omitted bands contribute exactly their true energy.
        metrics[f'reborn/l{i+1}/prefix_image_mse'] = (sum(errors[:i+1]) +
            sum(jnp.square(b).sum(-1)/rb.PIXELS for b in bands[i+1:])).mean()
    if self.outcomes is not None:
      action = flatten_action_dict(prevact, self.act_space)[:, 1:]
      online = self.outcomes(z[:, :-1], action)
      # Current actor chooses a' using online code; target predictor uses EMA code.
      next_action = flatten_action_dict(sample(self.pol(sg(z[:, 1:]), 2)), self.act_space)
      target_mu = self.slow_code(sg(h[:, 1:]))
      target_z = rb.sample_code(target_mu, nj.seed(), c.stochastic)
      target = self.slow_outcomes(sg(target_z), sg(next_action))
      valid = rb.transition_mask(obs['is_first'], obs['is_last'])
      losses['reborn_outcome'], more = rb.outcome_loss(online, target,
          [b[:, 1:] for b in bands], obs['reward'][:, 1:], obs['is_terminal'][:, 1:],
          valid, self.gammas, c.q_weight)
      metrics.update(more)
      metrics['reborn/valid_transition_count'] = valid.sum()
      metrics['reborn/valid_transition_fraction'] = valid.mean()
      metrics['reborn/nonzero_reward_transition_count'] = (valid*(obs['reward'][:, 1:] != 0)).sum()
      metrics['reborn/terminal_transition_count'] = (valid*obs['is_terminal'][:, 1:]).sum()
      metrics['reborn/ema_mu_mse'] = jnp.square(target_mu-sg(mu[:, 1:])).mean()
    return losses, metrics

  def imagine_codes(self, starts, length, training):
    """Sample z once per state; retain it for action logp, value and rewards."""
    def step(state, _):
      z = self.feat2tensor(sg(state))
      action = sample(self.pol(z, 1))
      next_state, _ = self.dyn.imagine(state, action, 1, training, single=True)
      return sg(next_state), (z, action)
    state, (codes, acts) = nj.scan(step, nn.cast(starts), (), length, axis=1)
    last = self.feat2tensor(sg(state))
    lastact = sample(self.pol(last, 1))
    codes = jnp.concatenate((codes, last[:, None]), 1)
    acts = jax.tree.map(lambda a, b: jnp.concatenate((a, b[:, None]), 1), acts, lastact)
    return codes, acts

  def loss(self, carry, obs, prevact, training):
    enc_carry, dyn_carry, dec_carry = carry
    reset = obs['is_first']
    B, T = reset.shape
    enc_carry, enc_entries, tokens = self.enc(enc_carry, obs, reset, training)
    dyn_carry, dyn_entries, losses, repfeat, metrics = self.dyn.loss(dyn_carry, tokens, prevact, reset, training)
    dec_carry, dec_entries, recons = self.dec(dec_carry, repfeat, reset, training)
    h = self.feat2h(repfeat)
    # One posterior noise draw, reused by all projected heads. Detach only the
    # auxiliary backbone path; values remain identical on both paths.
    mu = self.code(h)
    z = rb.sample_code(mu, nj.seed(), self.config.reborn.stochastic)
    aux_mu = self.code(h if self.config.reborn.grad_to_backbone else sg(h))
    aux_z = (aux_mu + sg(z.astype(jnp.float32)-mu)).astype(nn.COMPUTE_DTYPE)
    auxloss, auxmetrics = self.auxiliary(h, aux_z, aux_mu, obs, prevact)
    losses.update(auxloss)
    metrics.update(auxmetrics)
    losses['rew'] = self.rew(sg(z, skip=self.config.reward_grad), 2).loss(obs['reward'])
    con = (~obs['is_terminal']).astype(jnp.float32)
    if self.config.contdisc:
      con *= 1-1/self.config.horizon
    losses['con'] = self.con(z, 2).loss(con)
    for key, recon in recons.items():
      value = obs[key]
      target = value.astype(jnp.float32)/255 if isimage(self.obs_space[key]) else value
      losses[key] = recon.loss(sg(target))
    assert all(x.shape == (B, T) for x in losses.values())

    K = min(self.config.imag_last or T, T)
    starts = self.dyn.starts(dyn_entries, dyn_carry, K)
    inp, imgact = self.imagine_codes(starts, self.config.imag_length, training)
    los, imgloss_out, mets = imag_loss(imgact, self.rew(inp, 2).pred(),
        self.con(inp, 2).prob(1), self.pol(inp, 2), self.val(inp, 2), self.slowval(inp, 2),
        self.retnorm, self.valnorm, self.advnorm, update=training,
        contdisc=self.config.contdisc, horizon=self.config.horizon, **self.config.imag_loss)
    losses.update({k: v.mean(1).reshape((B, K)) for k, v in los.items()})
    metrics.update(mets)
    if self.config.repval_loss:
      # Reuse posterior noise even if the backbone gradient route is detached.
      replay_mu = self.code(h if self.config.repval_grad else sg(h))
      replay_z = (replay_mu + sg(z.astype(jnp.float32)-mu)).astype(nn.COMPUTE_DTYPE)
      last, term, rew = [obs[k][:, -K:] for k in ('is_last', 'is_terminal', 'reward')]
      boot = imgloss_out['ret'][:, 0].reshape(B, K)
      inp = replay_z[:, -K:]
      los, _, mets = repl_loss(last, term, rew, boot, self.val(inp, 2),
          self.slowval(inp, 2), self.valnorm, update=training,
          horizon=self.config.horizon, **self.config.repl_loss)
      losses.update(los)
      metrics.update(prefix_fn(mets, 'reploss'))
    assert set(losses) == set(self.scales), (set(losses), set(self.scales))
    metrics.update({f'loss/{k}': v.mean() for k, v in losses.items()})
    metrics.update({f'weighted_loss/{k}': v.mean()*self.scales[k] for k, v in losses.items()})
    metrics['reborn/weighted_sf'] = self.config.reborn.lambda_outcome*metrics.get('reborn/sf_loss_component',0.)
    metrics['reborn/weighted_q'] = self.config.reborn.lambda_outcome*metrics.get('reborn/q_loss_component',0.)
    metrics['reborn/imag_return_raw_mean'] = imgloss_out['ret'].mean()
    metrics['reborn/imag_return_raw_std'] = imgloss_out['ret'].std()
    loss = sum(v.mean()*self.scales[k] for k, v in losses.items())
    if self.harmony:
      observation = self.scales['reborn_rec']*losses['reborn_rec'].mean()+metrics['reborn/weighted_sf']
      task = self.scales['rew']*losses['rew'].mean()+metrics['reborn/weighted_q']
      obs_loss,wo,so = self.harmony_obs(observation)
      task_loss,wt,st = self.harmony_task(task)
      fixed = sum(v.mean()*self.scales[k] for k,v in losses.items()
                  if k not in ('reborn_rec','reborn_outcome','rew'))
      loss = fixed+obs_loss+task_loss
      for name,raw,weight,s in [('observation',observation,wo,so),('task',task,wt,st)]:
        metrics.update({f'harmony/{name}/raw':raw,f'harmony/{name}/weight':weight,
            f'harmony/{name}/log_scale':s,f'harmony/{name}/weighted':weight*raw,
            f'harmony/{name}/scalar_gradient':-weight*raw+jax.nn.sigmoid(s),
            f'harmony/{name}/stationary_weight_batch':(jnp.sqrt(1+4/jnp.maximum(raw,1e-12))-1)/2})
      metrics['harmony/effective_beta_observation'] = self.config.reborn.beta/wo
      metrics['harmony/actual_rec'] = wo*self.scales['reborn_rec']*losses['reborn_rec'].mean()
      metrics['harmony/actual_sf'] = wo*metrics['reborn/weighted_sf']
      metrics['harmony/actual_q'] = wt*metrics['reborn/weighted_q']
      metrics['harmony/actual_reward'] = wt*self.scales['rew']*losses['rew'].mean()
    metrics['loss/total_actual'] = loss
    outs = {'tokens': tokens, 'repfeat': repfeat, 'losses': losses}
    return loss, ((enc_carry, dyn_carry, dec_carry),
        (enc_entries, dyn_entries, dec_entries), outs, metrics)
