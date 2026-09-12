"""Frozen-state paired interventions and actual loss gradients; no optimizer step.

Gradients are pre-AGC/pre-Adam diagnostic gradients, not realized updates.
All loss branches reuse identical input and PRNG seed within each comparison.
"""
import json
import os
from pathlib import Path
import sys
import elements
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import ruamel.yaml as yaml
from dreamerv3 import main_htp
from .h200_suite import atomic

BRANCHES=('rec','rate','sf','q','reward','continuation','actor','critic','replay_critic')


def gradient_objective(model, obs, prev, weights):
  _, (_, _, outs, metrics) = model.loss(model.init_train(1)[:3],obs,prev,False)
  losses=outs['losses']
  names=('reborn_rec','reborn_rate',None,None,'rew','con','policy','value','repval')
  vals=[]
  for i,name in enumerate(names):
    if name is None:
      val=metrics['reborn/'+('sf_loss_component' if i==2 else 'q_loss_component')]*model.config.reborn.lambda_outcome
    else:
      val=losses[name].mean()*model.scales[name]
    vals.append(val)
  return jnp.dot(jnp.stack(vals),weights)


def interventions(model,obs,prev):
  reset=obs['is_first']
  _,_,tokens=model.enc(model.enc.initial(1),obs,reset,False)
  _,_,feat=model.dyn.observe(model.dyn.initial(1),tokens,prev,reset,False)
  mu=model.code(model.feat2h(feat))
  eps=jax.random.normal(nj.seed(),mu.shape,dtype=jnp.float32)
  second=jax.random.normal(nj.seed(),mu.shape,dtype=jnp.float32)
  active=float(model.config.reborn.stochastic)
  z=mu+active*eps
  def heads(code):
    pol=model.pol(code,2)['action']
    n=int(model.act_space['action'].high)
    probs=jnp.stack([pol.prob(jnp.full(code.shape[:2],a,jnp.int32)) for a in range(n)],-1)
    off,scale=model.valnorm.stats()
    return dict(prob=probs,value=model.val(code,2).pred()*scale+off,
        reward=model.rew(code,2).pred(),continuation=model.con(code,2).prob(1))
  base=heads(z)
  variants={'resampled_noise':heads(mu+active*second),'mean_code':heads(mu)}
  for i,(lo,hi) in enumerate(zip((0,*model.dims[:-1]),model.dims)):
    variants[f'remove_mu_block_{i+1}']=heads(z.at[...,lo:hi].add(-mu[...,lo:hi]))
  def compare(other):
    p,q=base['prob'],other['prob']
    return dict(actor_tv=(.5*jnp.abs(p-q).sum(-1)).mean(),
        actor_kl=(p*(jnp.log(jnp.maximum(p,1e-8))-jnp.log(jnp.maximum(q,1e-8)))).sum(-1).mean(),
        actor_argmax_flip=(p.argmax(-1)!=q.argmax(-1)).mean(),
        value_abs_change=jnp.abs(base['value']-other['value']).mean(),
        reward_abs_change=jnp.abs(base['reward']-other['reward']).mean(),
        continuation_abs_change=jnp.abs(base['continuation']-other['continuation']).mean())
  metrics={name:compare(value) for name,value in variants.items()}
  metrics['base']=dict(actor_entropy=-(base['prob']*jnp.log(jnp.maximum(base['prob'],1e-8))).sum(-1).mean(),
      value_abs_mean=jnp.abs(base['value']).mean(),reward_mae=jnp.abs(base['reward']-obs['reward']).mean(),
      reward_zero_mae=jnp.abs(obs['reward']).mean(),
      continuation_brier=jnp.square(base['continuation']-(1-obs['is_terminal'].astype(jnp.float32))*
          (1-1/model.config.horizon if model.config.contdisc else 1.)).mean())
  return metrics


def summarize_gradients(grads):
  modules=('reborn_code','pol','val','rew','con')
  result={}
  for module in modules:
    vectors=[]
    for grad in grads:
      chunks=[np.asarray(v,np.float32).ravel() for k,v in sorted(grad.items()) if k.startswith(module+'/')]
      vectors.append(np.concatenate(chunks) if chunks else np.zeros(1,np.float32))
    norms=np.array([np.linalg.norm(v) for v in vectors])
    dots=np.array([[float(np.dot(a,b)) for b in vectors] for a in vectors])
    denom=norms[:,None]*norms[None,:]
    # Undefined cosine for zero gradients stays null, never interpreted as agreement.
    cosine=[[float(dots[i,j]/denom[i,j]) if denom[i,j]>1e-16 else None
             for j in range(len(grads))] for i in range(len(grads))]
    ac=sum(vectors[6:9])
    acnorm=float(np.linalg.norm(ac))
    result[module]=dict(norm=norms.tolist(),cosine=cosine,
        actor_critic_norm=acnorm,
        cosine_with_actor_critic=[float(np.dot(v,ac)/(n*acnorm)) if n*acnorm>1e-16 else None
                                for v,n in zip(vectors,norms)])
  return result


def main(source, clips, output):
  assert os.environ.get('SLURM_JOB_ID'), 'Slurm required'
  source,clips,output=map(Path,(source,clips,output)); output.mkdir(parents=True,exist_ok=True)
  raw=yaml.YAML(typ='safe').load((source/'config.yaml').read_text())
  raw['jax'].update(precompile=False,prealloc=False,transfer_guard=False)
  agent=main_htp.make_agent(elements.Config(raw))
  cp=elements.Checkpoint(); cp.agent=agent
  cp.load(source/'ckpt/env_action_steps_000100000',keys=['agent'])
  params=agent.save()['params']; model=agent.model
  def grad_call(obs,prev,weights):
    return nj.grad(lambda:gradient_objective(model,obs,prev,weights),
        [model.code,model.pol,model.val,model.rew,model.con])()[2]
  gradfn=jax.jit(nj.pure(grad_call),static_argnames=('create','modify','ignore'))
  paired=jax.jit(nj.pure(lambda obs,prev:interventions(model,obs,prev)),
      static_argnames=('create','modify','ignore'))
  rows=[]
  # Fixed episode selection, two independent noise/imagined action replicates.
  for eid in (0,10,20):
    with np.load(clips/f'episode_{eid:02d}.npz') as ep:
      length=min(64,len(ep['reward']))
      obs={k:ep[k][None,:length] for k in ('image','reward','is_first','is_last','is_terminal')}
      prev={'action':np.concatenate((np.zeros((1,1),np.int32),ep['action'][None,:length-1]),1)}
    for replicate in range(2):
      seed=np.array([eid+991,replicate],np.uint32)
      _,pair=paired(params,obs,prev,seed=seed,create=False,modify=True,ignore=True)
      grads=[]
      for i in range(len(BRANCHES)):
        _,grad=gradfn(params,obs,prev,jnp.eye(len(BRANCHES))[i],seed=seed,
                     create=False,modify=True,ignore=True)
        grads.append(jax.tree.map(np.asarray,grad))
      rows.append(dict(episode=eid,replicate=replicate,states=length,
          interventions=jax.tree.map(lambda x:float(np.asarray(x)),pair),
          gradients=summarize_gradients(grads)))
      atomic(output/'audit.json',dict(status='RUNNING',branches=BRANCHES,rows=rows))
      print(f'AUDIT episode={eid} replicate={replicate} complete',flush=True)
  atomic(output/'audit.json',dict(status='COMPLETE',branches=BRANCHES,rows=rows,
      clips=str(clips.resolve()),checkpoint=str(source.resolve()),
      limitations=['Three episode prefixes (up to 64 states), restricted coverage; two PRNG replicates.',
          'Pre-AGC/pre-Adam gradients of actual weighted losses; not realized optimizer updates.',
          'Cosine describes local gradient conflict, not causal proof of long-term control effects.',
          'Block removal and mean-code intervention are out of distribution; measure reliance only.',
          'Paired interventions reuse backbone state and noise; environment is not rolled out.',
          'Value sensitivity is not value calibration; no truncated return treated as exact critic target.']))


if __name__ == '__main__':
  main(*sys.argv[1:])
