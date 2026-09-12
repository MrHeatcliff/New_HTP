"""Frozen-policy rollout calibration and episode-held-out incremental probes.

Runs inside the same allocation after training. Uses real rewards/images, never
TD bootstrap targets for calibration; state coverage and truncation are reported.
"""
import json
import os
from pathlib import Path
import sys
import elements
import embodied
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import ruamel.yaml as yaml
from dreamerv3 import main_htp, reborn as rb
from dreamerv3.htp import flatten_action_dict
from .h200_suite import atomic


def rollout_targets(bands, reward, terminal, starts, gammas, horizon=128):
  """Finite real-trajectory returns; terminal image/reward included exactly once."""
  sf, q = [], []
  for i, gamma in enumerate(gammas):
    fixed = np.concatenate(bands[:i+1], -1)
    features = np.zeros((len(starts), fixed.shape[-1]), np.float32)
    returns = np.zeros(len(starts), np.float32)
    alive = np.ones(len(starts), np.float32)
    for k in range(1, horizon+1):
      index = starts+k
      features += ((1-gamma)*gamma**(k-1)*alive)[:,None]*fixed[index]
      returns += gamma**(k-1)*alive*reward[index]
      alive *= ~terminal[index]
    sf.append(features)
    q.append(returns)
  return sf, np.stack(q,-1)


def ridge_matrix(x, y, ids, dims):
  """Same target columns for every prefix; lambda selected on held-out episodes."""
  from scipy.linalg import solve
  splits = (ids % 3 == 0, ids % 3 == 1, ids % 3 == 2)
  train, validation, test = splits
  if min(m.sum() for m in splits) < 2:
    raise ValueError('Insufficient independent episodes for held-out probes')
  mean = y[train].mean(0)
  scale = np.maximum(y[train].std(0), 1e-6)
  y = (y-mean)/scale
  scores = []
  for dim in (0, *dims):
    if dim == 0:
      # x suffix contains action; action-only is the zero-prefix baseline.
      a = x[:, dims[-1]:]
    else:
      a = np.concatenate((x[:,:dim], x[:,dims[-1]:]),-1)
    a = (a-a[train].mean(0))/np.maximum(a[train].std(0),1e-5)
    a = np.column_stack((a,np.ones(len(a))))
    at, yt = a[train], y[train]
    dual = at.shape[1] > len(at)
    gram = at@at.T if dual else at.T@at
    rhs = yt if dual else at.T@yt
    best = None
    for ridge in (0.01, 1., 100.):
      coef = solve(gram+ridge*np.eye(len(gram)),rhs,assume_a='pos')
      coef = at.T@coef if dual else coef
      validation_error = np.square(a[validation]@coef-y[validation]).mean()
      if best is None or validation_error < best[0]:
        best = validation_error,coef,ridge
    error = np.square(a[test]@best[1]-y[test]).mean(0)
    baseline = np.square(y[test]).mean(0)
    scores.append({'prefix_dim':dim,'ridge':best[2],
        'normalized_mse':error.tolist(), 'r2_vs_train_mean':(1-error/np.maximum(baseline,1e-8)).tolist()})
  return scores


def extract_chunk(model, carry, obs, prevact, actions):
  enc, dyn = carry
  enc, _, tokens = model.enc(enc, obs, obs['is_first'], False)
  dyn, _, feat = model.dyn.observe(dyn,tokens,prevact,obs['is_first'],False)
  mu = model.code(model.feat2h(feat))[:,::16]
  z = rb.sample_code(mu, nj.seed(), model.config.reborn.stochastic)
  flat = flatten_action_dict({'action':actions[:,::16]},model.act_space)
  bands = model.refinement(z) if model.refinement is not None else ()
  pred = model.outcomes(z,flat) if model.outcomes is not None else ()
  # Paired interventions reuse noise/context. These measure reliance, not the
  # benefit of a retrained probe (which is measured separately below).
  removed = []
  if model.refinement is not None:
    for i,(lo,hi) in enumerate(zip((0,*model.dims[:-1]),model.dims)):
      ablated = z.at[...,lo:hi].set(z[...,lo:hi]-mu[...,lo:hi])
      removed.append(model.refinement(ablated)[i])
  policy = model.pol(z,2)['action']
  n = int(model.act_space['action'].high)
  prob = jnp.stack([policy.prob(jnp.full(z.shape[:2],a,jnp.int32)) for a in range(n)],-1)
  voffset,vscale=model.valnorm.stats()
  return (enc,dyn), dict(mu=mu[0], z=z[0], bands=tuple(b[0] for b in bands),
      removed=tuple(b[0] for b in removed), psi=tuple(p[0] for p,q in pred),
      q=jnp.stack([q.pred()[0] for p,q in pred],-1) if pred else jnp.zeros((mu.shape[1],5)),
      reward=model.rew(z,2).pred()[0], continuation=model.con(z,2).prob(1)[0],
      actor_entropy=policy.entropy()[0], actor_prob=prob[0],
      value=(model.val(z,2).pred()*vscale+voffset)[0])


def main(source, output):
  if not os.environ.get('SLURM_JOB_ID'):
    raise RuntimeError('Slurm required')
  source, output = Path(source), Path(output)
  output.mkdir(parents=True,exist_ok=True)
  raw = yaml.YAML(typ='safe').load((source/'config.yaml').read_text())
  raw['jax'].update(precompile=False,prealloc=False,transfer_guard=False)
  config = elements.Config(raw)
  agent = main_htp.make_agent(config)
  cp = elements.Checkpoint(); cp.agent = agent
  cp.load(source/'ckpt/env_action_steps_000100000',keys=['agent'])
  params = agent.save()['params']
  atomic(output/'parameter_counts.json', {'by_module': {
      name:sum(int(np.prod(v.shape)) for k,v in params.items() if k.startswith(name+'/'))
      for name in sorted({k.split('/')[0] for k in params})}})
  env = main_htp.make_env(config,0,use_seed=False,seed=260912)
  driver = embodied.Driver([lambda:env],parallel=False)
  current = []
  def record(tran, worker):
    if current and current[-1]['is_last']:
      return
    current.append({k:np.asarray(tran[k]).copy() for k in
        ('image','action','reward','is_first','is_last','is_terminal')})
  driver.on_step(record)
  fn = jax.jit(nj.pure(lambda carry,obs,prevact,actions:
      extract_chunk(agent.model,carry,obs,prevact,actions)),
      static_argnames=('create','modify','ignore'))
  rng = np.random.default_rng(901)
  sketches = [rng.normal(size=(d,32)).astype(np.float32)/np.sqrt(32) for d in rb.BAND_DIMS]
  all_mu, all_z, all_y, all_ids, all_actions = [],[],[],[],[]
  calibration, band_errors, removed_errors, rows = [],[],[],[]
  try:
    for eid in range(30):
      current.clear()
      driver.reset(agent.init_policy)
      driver(lambda *args:agent.policy(*args,mode='eval'),steps=1024)
      ep = {k:np.stack([r[k] for r in current]) for k in current[0]}
      np.savez_compressed(output/f'episode_{eid:02d}.npz',**ep)
      n = len(current)
      starts = np.arange(0,n-128,16)
      if not len(starts):
        rows.append({'episode':eid,'length':n,'anchors':0})
        continue
      bands = [np.asarray(x) for x in rb.haar_bands(jnp.asarray(ep['image'],jnp.float32)/255)]
      sf, returns = rollout_targets(bands,ep['reward'],ep['is_terminal'],starts,agent.model.gammas)
      enc = agent.model.enc.initial(1); dyn = agent.model.dyn.initial(1)
      carry = (enc,dyn)
      pieces = []
      for lo in range(0, int(starts[-1])+1, 128):
        hi = min(lo+128,n)
        obs = {k:v[None,lo:hi] for k,v in ep.items() if k != 'action'}
        prev = np.zeros((1,hi-lo),np.int32)
        if lo:
          prev[0] = ep['action'][lo-1:hi-1]
        else:
          prev[0,1:] = ep['action'][:hi-1]
        _, (carry, data) = fn(params,carry,obs,{'action':prev},ep['action'][None,lo:hi],
            seed=np.array([eid,lo],np.uint32),create=False,modify=True,ignore=True)
        pieces.append(jax.tree.map(np.asarray,data))
      data = jax.tree.map(lambda *xs:np.concatenate(xs,0)[:len(starts)],*pieces)
      y_detail = np.concatenate([b[starts]@s for b,s in zip(bands,sketches)],-1)
      # Each timescale uses the SAME full-image Haar sketch for every prefix.
      # Fixed sketch is a diagnostic target reduction, never a training target.
      temporal = []
      for gamma in agent.model.gammas:
        target = np.zeros((len(starts),32),np.float32)
        alive = np.ones(len(starts),np.float32)
        sketch = sum(b@s for b,s in zip(bands,sketches))
        for k in range(1,129):
          target += ((1-gamma)*gamma**(k-1)*alive)[:,None]*sketch[starts+k]
          alive *= ~ep['is_terminal'][starts+k]
        temporal.append(target)
      all_y.append(np.concatenate((y_detail,*temporal,returns),-1))
      all_mu.append(data['mu']); all_z.append(data['z']); all_ids.append(np.full(len(starts),eid))
      all_actions.append(np.eye(int(agent.act_space['action'].high))[ep['action'][starts]])
      if data['psi']:
        calibration.append(dict(sf_mse=[float(np.square(p-t).sum(-1).mean()/rb.PIXELS) for p,t in zip(data['psi'],sf)],
            sf_zero_mse=[float(np.square(t).sum(-1).mean()/rb.PIXELS) for t in sf],
            q_mae=np.abs(data['q']-returns).mean(0).tolist(),q_zero_mae=np.abs(returns).mean(0).tolist()))
      if data['bands']:
        band_errors.append([float(np.square(p-b[starts]).sum(-1).mean()/rb.PIXELS) for p,b in zip(data['bands'],bands)])
        removed_errors.append([float(np.square(p-b[starts]).sum(-1).mean()/rb.PIXELS) for p,b in zip(data['removed'],bands)])
      nz = ep['reward'][starts] != 0
      # Only truly terminated clips provide an unbootstrapped complete return.
      # Censored/time-limit clips must not be presented as exact value targets.
      critic = dict(critic_complete_episode=bool(ep['is_terminal'][-1]),
                    critic_complete_return_mae=None,critic_complete_return_zero_mae=None)
      if ep['is_terminal'][-1]:
        discount=1-1/config.agent.horizon
        returns_complete=np.zeros(n,np.float64)
        for t in range(n-2,-1,-1):
          returns_complete[t]=ep['reward'][t+1]+discount*(not ep['is_terminal'][t+1])*returns_complete[t+1]
        critic.update(critic_complete_return_mae=float(np.abs(data['value']-returns_complete[starts]).mean()),
            critic_complete_return_zero_mae=float(np.abs(returns_complete[starts]).mean()))
      rows.append(dict(episode=eid,length=n,anchors=len(starts),
          **critic,
          reward_mae=float(np.abs(data['reward']-ep['reward'][starts]).mean()),
          reward_zero_mae=float(np.abs(ep['reward'][starts]).mean()),
          reward_nonzero_count=int(nz.sum()),
          reward_nonzero_mae=float(np.abs(data['reward'][nz]-ep['reward'][starts][nz]).mean()) if nz.any() else None,
          terminal_count=int(ep['is_terminal'][starts].sum()),
          continuation_brier=float(np.square(data['continuation']-(~ep['is_terminal'][starts])*
              (1-1/config.agent.horizon if config.agent.contdisc else 1.)).mean()),
          actor_entropy=float(data['actor_entropy'].mean())))
      atomic(output/'progress.json',{'episodes':rows,'status':'COLLECTING'})
  finally:
    driver.close()
  mu,z,y,ids,actions = map(np.concatenate,(all_mu,all_z,all_y,all_ids,all_actions))
  np.savez_compressed(output/'probe_data.npz',mu=mu,z=z,target=y,episode_id=ids,action=actions)
  blocks = [mu[:,lo:hi].astype(np.float64) for lo,hi in zip((0,*agent.model.dims[:-1]),agent.model.dims)]
  grams = [(b-b.mean(0))@(b-b.mean(0)).T for b in blocks]
  cka = [[float((a*b).sum()/max(np.sqrt((a*a).sum()*(b*b).sum()),1e-12)) for b in grams] for a in grams]
  result = dict(status='COMPLETE',episodes=rows,calibration=calibration,
      band_mse=band_errors, band_removed_mse=removed_errors,mu_cka=cka,
      mu_participation_rank=[float(np.trace(g)**2/max((g*g).sum(),1e-12)) for g in grams],
      mu_block_variance=[float(b.var(0).mean()) for b in blocks],
      probes_mu=ridge_matrix(np.concatenate((mu,actions),-1),y,ids,agent.model.dims),
      probes_sampled_z=ridge_matrix(np.concatenate((z,actions),-1),y,ids,agent.model.dims),
      target_columns={'detail_bands':'0:160, five 32-D fixed sketches',
          'same_spatial_target_by_timescale':'160:320, five 32-D sketches', 'Q_by_timescale':'320:325'},
      limitations=['30 frozen-policy clips of up to 1024 actions; restricted state coverage.',
          '128-step finite returns, longest-discount tail gamma^128; not exact infinite returns.',
          'One stochastic prediction and one real rollout per anchor; Monte Carlo MSE includes outcome variance.',
          'Ridge uses episode splits eid modulo 3; shared lambda selected over all standardized targets.',
          'Incremental probes use fixed sketches, full coefficient error uses the trained heads.',
          'Episode variation is not training-seed uncertainty.'])
  atomic(output/'diagnostics.json',result)
  plot_diagnostics(result,output)


def plot_diagnostics(result, output):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  fig, axes = plt.subplots(2,3,figsize=(14,7),constrained_layout=True)
  for row,key in enumerate(('probes_mu','probes_sampled_z')):
    values = np.array([x['r2_vs_train_mean'] for x in result[key]])
    for col,(start,end,label) in enumerate(((0,160,'Detail band'),(160,320,'Timescale SF'),(320,325,'Timescale Q'))):
      matrix = values[:,start:end]
      if col < 2:
        matrix = matrix.reshape(6,5,32).mean(-1)
      ax = axes[row,col]
      im = ax.imshow(matrix,aspect='auto',cmap='coolwarm',vmin=-1,vmax=1)
      ax.set_title(f'{key}: {label} R²')
      ax.set_xticks(range(5),['1','2','3','4','5'])
      ax.set_yticks(range(6),['Action only','P1','P2','P3','P4','P5'])
      fig.colorbar(im,ax=ax)
  fig.savefig(output/'prefix_target_probe_matrix.png',dpi=180)
  plt.close(fig)
  fig,ax = plt.subplots(figsize=(5,4),constrained_layout=True)
  im = ax.imshow(result['mu_cka'],vmin=0,vmax=1,cmap='viridis')
  ax.set_title('Block CKA on mean code μ (no sampled noise)')
  ax.set_xticks(range(5),range(1,6)); ax.set_yticks(range(5),range(1,6))
  fig.colorbar(im,ax=ax)
  fig.savefig(output/'mu_block_cka.png',dpi=180)
  plt.close(fig)


if __name__ == '__main__':
  main(*sys.argv[1:])
