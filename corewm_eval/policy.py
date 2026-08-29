import numpy as np
from rliable import library as rly
from rliable import metrics

from .config import ATARI100K_GAMES, PERFORMANCE_THRESHOLDS


def ordered_game_values(result_by_game, games=ATARI100K_GAMES):
  expected, observed = set(games), set(result_by_game)
  if observed != expected:
    raise ValueError(
        f'Game set mismatch: missing={sorted(expected - observed)}, '
        f'extra={sorted(observed - expected)}')
  return np.asarray([result_by_game[game] for game in games])


def score_matrix_from_seed_dicts(result_by_seed, seeds=range(5), games=ATARI100K_GAMES):
  """Convert mappings to [seed, game] without relying on dictionary order."""
  expected, observed = set(seeds), set(result_by_seed)
  if observed != expected:
    raise ValueError(
        f'Seed set mismatch: missing={sorted(expected - observed)}, '
        f'extra={sorted(observed - expected)}')
  return validate_score_matrix(np.stack([
      ordered_game_values(result_by_seed[seed], games) for seed in seeds]))


def human_normalized_score(score, random_score, human_score):
  if human_score == random_score:
    raise ValueError('Human and random scores are equal')
  return (np.asarray(score, np.float64) - random_score) / (human_score - random_score)


def normalized_auc(checkpoints, hns, final_budget=100_000):
  checkpoints, hns = np.asarray(checkpoints, np.float64), np.asarray(hns, np.float64)
  if checkpoints.ndim != 1 or hns.shape[-1] != len(checkpoints):
    raise ValueError((checkpoints.shape, hns.shape))
  if checkpoints[-1] != final_budget or not np.all(np.diff(checkpoints) > 0):
    raise ValueError('Incomplete or unordered checkpoint schedule')
  return np.trapz(hns, checkpoints, axis=-1) / (final_budget - checkpoints[0])


def validate_score_matrix(scores):
  scores = np.asarray(scores, np.float64)
  if scores.shape != (5, 26):
    raise ValueError(f'Expected [5,26], got {scores.shape}')
  if not np.isfinite(scores).all():
    raise ValueError('Scores contain NaN/Inf')
  return scores


def performance_profile(scores, thresholds=PERFORMANCE_THRESHOLDS):
  scores = validate_score_matrix(scores)
  thresholds = np.asarray(thresholds, np.float64)
  return np.asarray([(scores > tau).mean() for tau in thresholds])


def rliable_aggregate_intervals(score_dict, reps=50_000, seed=0):
  score_dict = {key: validate_score_matrix(value) for key, value in score_dict.items()}
  aggregate = lambda x: np.array([
      metrics.aggregate_mean(x), metrics.aggregate_median(x),
      metrics.aggregate_iqm(x)])
  return rly.get_interval_estimates(
      score_dict, aggregate, reps=int(reps), confidence_interval_size=0.95,
      random_state=np.random.RandomState(seed))


def rliable_probability_of_improvement(scores_a, scores_b, reps=50_000, seed=0):
  pair = {'comparison': (validate_score_matrix(scores_a), validate_score_matrix(scores_b))}
  fn = lambda x, y: metrics.probability_of_improvement(x, y)
  return rly.get_interval_estimates(
      pair, fn, reps=int(reps), confidence_interval_size=0.95,
      random_state=np.random.RandomState(seed))
