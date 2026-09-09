import json

from corewm_eval.stage_supervisor import _attempt_dir


def test_attempt_lookup_is_stable_by_job_metadata(tmp_path, monkeypatch):
  from corewm_eval import stage_supervisor
  production = tmp_path / 'prod'
  monkeypatch.setattr(stage_supervisor, 'PRODUCTION', production)
  root = production / 'training/stage1_full/alien/seed_3'
  for number, index in ((1, 54), (2, 99)):
    attempt = root / f'attempt_{number:03d}'
    attempt.mkdir(parents=True)
    (attempt / 'launch.json').write_text(json.dumps({
        'job_index': index, 'seed': 3}))
  assert _attempt_dir({'game': 'alien', 'seed': 3, 'job_index': 54}).name == 'attempt_001'
