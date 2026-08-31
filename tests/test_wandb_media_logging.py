import numpy as np
import pytest
import wandb

from dreamerv3.wandb_output import WandBOutput


class Recorder:

  def __init__(self, fail_media=False):
    self.calls = []
    self.fail_media = fail_media

  def __call__(self, summaries):
    self.calls.append(tuple(summaries))
    if self.fail_media and any(np.asarray(value).ndim >= 2 for _, _, value in summaries):
      raise ValueError(
          'wandb.Video requires a `format` argument when initializing with a numpy array.')


def representative_payload():
  # report_length=32, 10 padded time steps, H=3*64+4, W=6*(64+4).
  return np.zeros((42, 196, 408, 3), np.uint8)


def test_old_wandb_video_call_reproduces_original_exception():
  value = representative_payload().transpose(0, 3, 1, 2)
  with pytest.raises(ValueError, match='requires a `format` argument'):
    wandb.Video(value)


def test_production_media_disabled_retains_scalars_without_serialization():
  recorder = Recorder(fail_media=True)
  output = WandBOutput('unused', media_enabled=False, output=recorder)
  output(((10, 'train/loss', np.asarray(1.25)),
          (10, 'report/openloop/image', representative_payload())))
  assert len(recorder.calls) == 1
  names = [name for _, name, _ in recorder.calls[0]]
  assert 'train/loss' in names
  assert 'report/openloop/image' not in names
  assert 'logger/wandb_media_items_skipped' in names


def test_optional_media_failure_is_narrowly_fail_open():
  recorder = Recorder(fail_media=True)
  output = WandBOutput('unused', media_enabled=True, output=recorder)
  output(((10, 'train/loss', np.asarray(1.25)),
          (10, 'report/openloop/image', representative_payload())))
  assert len(recorder.calls) == 2
  names = [name for _, name, _ in recorder.calls[-1]]
  assert names == ['train/loss', 'logger/wandb_media_log_failures']


def test_non_media_wandb_failure_remains_fatal():
  class Broken:
    def __call__(self, summaries):
      raise RuntimeError('scalar backend failed')
  output = WandBOutput('unused', media_enabled=False, output=Broken())
  with pytest.raises(RuntimeError, match='scalar backend failed'):
    output(((10, 'train/loss', np.asarray(1.25)),))
