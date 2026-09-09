from corewm_eval.constraint_suite_curves import SUITE_LABEL, aggregate
import pandas as pd


def test_seed_one_training_curve_has_binned_mean_and_zero_sem():
  frame=pd.DataFrame({'method':[SUITE_LABEL]*2,'game':['alien']*2,'seed':[0,0],
      'agent_steps':[1000,2000],'return':[3.,5.]})
  _,result=aggregate(frame,True)
  assert result['seeds'].tolist()==[1]
  assert result['mean'].tolist()==[4.]
  assert result['sem'].tolist()==[0.]
