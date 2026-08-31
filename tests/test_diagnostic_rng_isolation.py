import numpy as np

from embodied.core.replay import Replay


def _populate(replay, count=16):
  stepids = np.zeros((1, 20), np.uint8)
  for key in range(count):
    replay.sampler[key] = stepids
    for sampler in replay.report_samplers.values():
      sampler[key] = stepids


def test_report_sampling_does_not_advance_training_selector_rng():
  baseline = Replay(length=1, seed=7)
  with_report = Replay(length=1, seed=7)
  _populate(baseline)
  _populate(with_report)
  for _ in range(100):
    with_report.report_samplers['report']()
  assert [baseline.sampler() for _ in range(100)] == [
      with_report.sampler() for _ in range(100)]


def test_report_and_eval_selectors_are_distinct_objects():
  replay = Replay(length=1, seed=7)
  assert replay.report_samplers['report'] is not replay.sampler
  assert replay.report_samplers['eval'] is not replay.sampler
  assert replay.report_samplers['eval'] is not replay.report_samplers['report']
