import pandas as pd

from corewm_eval.wandb_baseline_curves import _binned


def test_binning_is_per_seed_before_cross_seed_average():
  frame=pd.DataFrame([
      {'method':'M','game':'g','seed':0,'agent_steps':1000,'return':1.0},
      {'method':'M','game':'g','seed':0,'agent_steps':2000,'return':3.0},
      {'method':'M','game':'g','seed':1,'agent_steps':3000,'return':6.0},
  ])
  per_seed,aggregate=_binned(frame,bin_width=5000)
  assert sorted(per_seed['return'])==[2.0,6.0]
  row=aggregate.iloc[0]
  assert row['mean']==4.0 and row['seeds']==2 and row['agent_steps']==2500
