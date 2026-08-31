import json

from corewm_eval.config import QUALITATIVE_GAMES
from corewm_eval.phase3_prepare import MAIN_POLICY_VARIANTS, prepare, resolve
from corewm_eval.production import _task_game


def test_production_matrix_is_exactly_680(tmp_path):
  _, _, jobs = prepare(tmp_path)
  main = [row for row in jobs if row['variant'] in MAIN_POLICY_VARIANTS]
  reverse = [row for row in jobs if row['variant'] == 'Reverse']
  wave1 = [row for row in jobs if row['wave'] == 1]
  wave2 = [row for row in jobs if row['wave'] == 2]
  assert (len(main), len(reverse), len(jobs)) == (650, 30, 680)
  assert (len(wave1), len(wave2)) == (36, 644)
  assert {row['game'] for row in reverse} == set(QUALITATIVE_GAMES)
  assert {row['seed'] for row in reverse} == set(range(5))
  assert [row['job_index'] for row in jobs] == list(range(680))
  persisted = json.loads((tmp_path / 'training_manifest.json').read_text())
  assert persisted == jobs


def test_production_task_alias_is_explicit():
  assert _task_game('alien') == 'alien'
  assert _task_game('jamesbond') == 'james_bond'


def test_production_disables_training_media_only():
  config = resolve('Full')
  assert tuple(config.logger.outputs) == ('jsonl', 'scope', 'wandb')
  assert config.logger.wandb_media is False
  assert config.run.log_policy_video is False
