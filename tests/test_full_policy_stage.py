from corewm_eval.config import ATARI100K_GAMES, FINAL_CHECKPOINTS, TRAINING_SEEDS


def test_full_policy_stage_cardinality_and_schedule():
  all_jobs={(game,seed) for game in ATARI100K_GAMES for seed in TRAINING_SEEDS}
  assert len(all_jobs)==130
  assert len(all_jobs-{('alien',0)})==129
  assert FINAL_CHECKPOINTS==tuple(range(10_000,100_001,10_000))
  assert [10]*9+[100]==[100 if step==100_000 else 10 for step in FINAL_CHECKPOINTS]
