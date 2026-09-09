"""Fixed source-policy clips and held-out probes for four-game research."""
import argparse
import json
import os
from pathlib import Path
import elements
import embodied
import numpy as np
import ruamel.yaml as yaml
from threadpoolctl import threadpool_limits
from dreamerv3 import main_htp
from .extraction import ExtractionSpec, evaluation_seed, extract_readonly
from .paper_pipeline import _padded_batch
from .persistence_research_report import scale_resistant_diagnostics


def agent_for(source, checkpoint):
  raw=yaml.YAML(typ='safe').load((Path(source)/'config.yaml').read_text())
  raw['jax'].update(precompile=False,prealloc=False,transfer_guard=False)
  config=elements.Config(raw)
  agent=main_htp.make_agent(config)
  cp=elements.Checkpoint(); cp.agent=agent; cp.load(checkpoint,keys=['agent'])
  dyn=raw['agent']['dyn'][raw['agent']['dyn']['typ']]
  dims = (raw['agent']['htp']['proj']['dims'] if raw['agent']['htp'].get('use_proj', True)
          else raw['agent']['htp']['pdyn']['dims'])
  spec=ExtractionSpec(int(dyn['deter']),(int(dyn['stoch']),int(dyn['classes'])),
                      tuple(dims))
  return config,agent,spec


def prepare(source, checkpoint, root):
  root=Path(root); root.mkdir(exist_ok=False)
  config,agent,spec=agent_for(source,checkpoint)
  # Override use_seed explicitly so make_env honors this diagnostic seed.
  env=main_htp.make_env(config,0,use_seed=False,seed=260909)
  driver=embodied.Driver([lambda:env],parallel=False)
  current=[]; ended=False
  def record(tran,worker):
    nonlocal ended
    if ended: return
    current.append({k:np.asarray(tran[k]).copy() for k in ('image','action','reward','is_terminal','is_last')})
    ended=bool(tran['is_last'])
  driver.on_step(record)
  try:
    for eid in range(30):
      current=[]; ended=False
      driver.reset(agent.init_policy)
      driver(lambda *args:agent.policy(*args,mode='eval'),steps=256)
      if len(current)<66: raise RuntimeError('Diagnostic episode too short for horizon 64')
      ep={'observation':np.stack([t['image'] for t in current]),
          'action':np.array([t['action'] for t in current]),
          'reward':np.array([t['reward'] for t in current]),
          'is_terminal':np.array([t['is_terminal'] for t in current]),
          'continuation':~np.array([t['is_terminal'] for t in current]),
          'episode_id':np.full(len(current),eid,np.int64),'timestep':np.arange(len(current))}
      np.savez_compressed(root/f'episode_{eid:02d}.npz',**ep)
  finally: driver.close()
  order=np.random.default_rng(202609).permutation(30)
  split={key:order[i*10:(i+1)*10].tolist() for i,key in enumerate(('train','validation','test'))}
  (root/'split.json').write_text(json.dumps(split,indent=2))
  (root/'metadata.json').write_text(json.dumps({'source_checkpoint':str(checkpoint),
      'task':config.task,'episodes':30,'max_observations':256,
      'limitation':'Independent source-policy episode starts; early-state diagnostic, not full state coverage.'},indent=2))
  extract(agent,spec,root,root/'source.npz',True)


def extract(agent,spec,dataset,output,save_target=False):
  zs=[]; hs=[]; ids=[]; ts=[]
  for path in sorted(Path(dataset).glob('episode_*.npz')):
    with np.load(path) as stored: ep={k:stored[k] for k in stored.files}
    obs,act,_=_padded_batch([ep],256); length=len(ep['action'])
    result=extract_readonly(agent,obs,act,spec,evaluation_seed(0,int(ep['episode_id'][0])),decode=False)
    zs.append(result['z'][0,:length].astype(np.float32))
    if save_target: hs.append(result['h'][0,:length].astype(np.float32))
    ids.append(ep['episode_id']); ts.append(ep['timestep'])
  values={'z':np.concatenate(zs),'episode_id':np.concatenate(ids),'timestep':np.concatenate(ts)}
  if save_target:
    h=np.concatenate(hs)
    values['target']=h @ np.random.default_rng(123).normal(size=(h.shape[1],64)).astype(np.float32)/np.sqrt(h.shape[1])
  np.savez(output,**values)


def analyze(dataset,representation,output):
  dataset=Path(dataset)
  with np.load(representation) as cache: z,ids,ts=[cache[k].astype(np.float64) for k in ('z','episode_id','timestep')]
  with np.load(dataset/'source.npz') as source:
    np.testing.assert_array_equal(ids,source['episode_id']); np.testing.assert_array_equal(ts,source['timestep'])
    h=source['target'].astype(np.float64)
  split=json.loads((dataset/'split.json').read_text())
  masks={k:np.isin(ids,v) for k,v in split.items()}
  h=(h-h[masks['train']].mean(0))/np.maximum(h[masks['train']].std(0),1e-6)
  a=z[masks['test'],:128]; b=z[masks['test'],128:256]
  a=a-a.mean(0); b=b-b.mean(0)
  report={'blocks':scale_resistant_diagnostics(representation,split),
      'cka':float(np.square(a.T@b).sum()/max(np.sqrt(np.square(a.T@a).sum()*np.square(b.T@b).sum()),1e-12)),
      'mean_variance':float(a.var(0).mean()),'dead_coordinates':int((a.var(0)<1e-6).sum()),
      'target':'Fixed source-checkpoint h projected to 64 dimensions; no action inputs; not semantic labels.',
      'horizons':{}}
  for lag in (1,4,8,16,32,64):
    valid=(ids[:-lag]==ids[lag:])&(ts[lag:]==ts[:-lag]+lag)
    m={k:v[:-lag]&valid for k,v in masks.items()}; y=h[lag:]; probes={}
    for name,lo,hi in (('block1',0,128),('block2',128,256),('prefix2',0,256)):
      x=z[:-lag,lo:hi]
      x=(x-x[m['train']].mean(0))/np.maximum(x[m['train']].std(0),1e-6)
      x=np.c_[x,np.ones(len(x))]; gram=x[m['train']].T@x[m['train']]; rhs=x[m['train']].T@y[m['train']]
      best=None
      for alpha in (.01,1.,100.,10000.):
        w=np.linalg.solve(gram+alpha*np.eye(x.shape[1]),rhs)
        error=np.square(x[m['validation']]@w-y[m['validation']]).mean()
        if best is None or error<best[0]: best=(error,w,alpha)
      target=y[m['test']]; pred=x[m['test']]@best[1]; test_ids=ids[:-lag][m['test']]
      sse=np.square(pred-target).sum(-1); sst=np.square(target-target.mean(0)).sum(-1)
      if sst.sum()<=1e-12: raise RuntimeError('Degenerate probe target')
      probes[name]={'r2':float(1-sse.sum()/sst.sum()),'ridge':best[2],
          'episodes':{str(int(e)):{'sse':float(sse[test_ids==e].sum()),'sst':float(sst[test_ids==e].sum())} for e in np.unique(test_ids)}}
    report['horizons'][str(lag)]=probes
  Path(output).write_text(json.dumps(report,indent=2,allow_nan=False))


def main():
  if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('Slurm only')
  parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=['prepare','extract','analyze'])
  parser.add_argument('--source'); parser.add_argument('--checkpoint'); parser.add_argument('--dataset',required=True)
  parser.add_argument('--output'); args=parser.parse_args()
  if args.mode=='prepare': prepare(args.source,args.checkpoint,args.dataset)
  elif args.mode=='extract':
    _,agent,spec=agent_for(args.source,args.checkpoint)
    extract(agent,spec,args.dataset,args.output)
  else: analyze(args.dataset,Path(args.output).with_suffix('.npz'),args.output)


if __name__=='__main__':
  with threadpool_limits(limits=1): main()
