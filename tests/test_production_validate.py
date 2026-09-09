import hashlib
import json

import pytest

from corewm_eval.production_validate import (
    FULL_PREFIX_DIMS, FULL_STRIDES, checkpoint_tree_hash,
    validate_full_training_attempt)


def test_checkpoint_tree_hash_is_content_and_path_stable(tmp_path):
  checkpoint = tmp_path / 'checkpoint'
  checkpoint.mkdir()
  (checkpoint / 'a').write_bytes(b'one')
  (checkpoint / 'b').write_bytes(b'two')
  expected = hashlib.sha256(b'aonebtwo').hexdigest()
  assert checkpoint_tree_hash(checkpoint) == expected


def test_validator_rejects_nonfinite_training_metric(tmp_path):
  run = tmp_path / 'attempt_001'
  (run / 'paper_artifacts').mkdir(parents=True)
  launch = {
      'exit_code': 0, 'status': 'TRAINING_EXITED',
      'protocol_id': 'corewm_atari100k_v1', 'method': 'Full',
      'variant': 'Full',
      'git_commit': '5ee4f27a0ba7fd7bdbdcbd4823fb2d539f8b4b47',
      'use_recon': True, 'use_pdyn': True, 'htp_enabled': True,
      'projection_enabled': True, 'prefix_dims': FULL_PREFIX_DIMS,
      'strides': FULL_STRIDES}
  (run / 'launch.json').write_text(json.dumps(launch))
  (run / 'paper_artifacts/train_metrics.jsonl').write_text(
      json.dumps({'loss': float('nan')}) + '\n')
  (run / 'paper_artifacts/action_checkpoints_manifest.json').write_text('[]')
  with pytest.raises(AssertionError):
    validate_full_training_attempt(run, verify_hashes=False)
