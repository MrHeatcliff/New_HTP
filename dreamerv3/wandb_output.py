"""W&B output policy for production training.

Local JSONL/Scope outputs remain authoritative. Production W&B receives scalar
and histogram summaries only, so optional media encoding cannot stop training.
"""

import collections

import numpy as np

import elements


class WandBOutput:

  def __init__(self, name, media_enabled=True, output=None, **kwargs):
    self.media_enabled = bool(media_enabled)
    self._output = output or elements.logger.WandBOutput(name, **kwargs)

  @staticmethod
  def _partition(summaries):
    scalars, media = [], []
    for item in summaries:
      value = item[2]
      ndim = 0 if isinstance(value, str) else np.asarray(value).ndim
      (media if ndim >= 2 else scalars).append(item)
    return scalars, media

  @staticmethod
  def _counter(media, name):
    counts = collections.Counter(step for step, _, _ in media)
    return [(step, name, np.asarray(count)) for step, count in counts.items()]

  def __call__(self, summaries):
    scalars, media = self._partition(summaries)
    if not self.media_enabled:
      payload = scalars + self._counter(
          media, 'logger/wandb_media_items_skipped')
      if payload:
        self._output(tuple(payload))
      return
    try:
      self._output(summaries)
    except ValueError as exc:
      # Narrow compatibility guard for optional W&B video serialization only.
      # Scalar/network/local-logger failures remain fatal and visible.
      if not media or 'wandb.Video requires a `format` argument' not in str(exc):
        raise
      print(
          'WARNING: optional W&B media serialization failed; retaining scalar '
          f'logging and continuing: {exc}')
      payload = scalars + self._counter(
          media, 'logger/wandb_media_log_failures')
      if payload:
        self._output(tuple(payload))
