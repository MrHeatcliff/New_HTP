from corewm_eval.four_game_rank_trial import arms_for, common


def test_decorrelation_ablation_changes_only_one_config_option():
  arms=arms_for('decorrelation')
  assert arms[0][1]==arms[1][1]==0
  baseline=common('boxing',arms[0][1],arms[0][2])
  candidate=common('boxing',arms[1][1],arms[1][2])
  changed=[i for i,(a,b) in enumerate(zip(baseline,candidate)) if a!=b]
  assert len(baseline)==len(candidate) and len(changed)==1
  assert candidate[changed[0]-1]=='--agent.htp.persistence_decorrelation'


def test_rank_guard_ablation_preserves_cka_and_first_prefix_isotropy():
  arms=arms_for('decorrelation_guard')
  assert len(arms)==3 and arms[1][1:]==arms[2][1:]==(0,0.01)
  baseline=common('boxing',0,0.01,guard=0,seed=1)
  candidate=common('boxing',0,0.01,guard=.001,seed=1)
  changed=[i for i,(a,b) in enumerate(zip(baseline,candidate)) if a!=b]
  assert len(changed)==1
  assert candidate[changed[0]-1]=='--agent.htp.persistence_block2_rank_scale'
  assert candidate[candidate.index('--seed')+1]=='1'
