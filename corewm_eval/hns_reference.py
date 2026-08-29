"""Canonical-order HNS reference table auditing."""

import csv
from pathlib import Path

import ruamel.yaml as yaml

from .config import ATARI100K_GAMES

TITLE_KEYS = {
    'alien': 'Alien', 'amidar': 'Amidar', 'assault': 'Assault',
    'asterix': 'Asterix', 'bank_heist': 'BankHeist',
    'battle_zone': 'BattleZone', 'boxing': 'Boxing', 'breakout': 'Breakout',
    'chopper_command': 'ChopperCommand', 'crazy_climber': 'CrazyClimber',
    'demon_attack': 'DemonAttack', 'freeway': 'Freeway',
    'frostbite': 'Frostbite', 'gopher': 'Gopher', 'hero': 'Hero',
    'jamesbond': 'Jamesbond', 'kangaroo': 'Kangaroo', 'krull': 'Krull',
    'kung_fu_master': 'KungFuMaster', 'ms_pacman': 'MsPacman', 'pong': 'Pong',
    'private_eye': 'PrivateEye', 'qbert': 'Qbert', 'road_runner': 'RoadRunner',
    'seaquest': 'Seaquest', 'up_n_down': 'UpNDown'}


def load_reference(path, section=None):
  data = yaml.YAML(typ='safe').load(Path(path).read_text())
  if section:
    data = data[section]
  result = {}
  for game in ATARI100K_GAMES:
    key = game if game in data else f'atari_{game}'
    if key not in data:
      raise ValueError(f'Missing HNS reference game: {game}')
    random_score, human_score = data[key]
    result[game] = (float(random_score), float(human_score))
  extras = set(data) - set(ATARI100K_GAMES) - {
      f'atari_{game}' for game in ATARI100K_GAMES}
  # Extra Atari57 entries are allowed for baselines.yaml; canonical extraction
  # still follows the explicit 26-game order above.
  return result


def audit_references(manuscript_path, output,
                     yaml_path='baselines.yaml', yaml_section='atari57_gamer'):
  manuscript = load_reference(manuscript_path)
  baseline = load_reference(yaml_path, yaml_section)
  rows = []
  for game in ATARI100K_GAMES:
    mr, mh = manuscript[game]
    yr, yh = baseline[game]
    rows.append({
        'game': game, 'manuscript_random': mr, 'yaml_random': yr,
        'manuscript_human': mh, 'yaml_human': yh,
        'match': mr == yr and mh == yh})
  output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
  with output.open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=rows[0])
    writer.writeheader(); writer.writerows(rows)
  if not all(row['match'] for row in rows):
    raise AssertionError([row for row in rows if not row['match']])
  return rows


def audit_repository_manuscript_table(
    output, yaml_path='baselines.yaml', yaml_section='atari57_gamer'):
  from embodied.run.paper_artifacts import ATARI_HNS_REFERENCES
  if set(TITLE_KEYS.values()) != set(ATARI_HNS_REFERENCES):
    raise AssertionError('Repository manuscript HNS table is not exactly 26 games')
  baseline = load_reference(yaml_path, yaml_section)
  rows = []
  for game in ATARI100K_GAMES:
    mr, mh = map(float, ATARI_HNS_REFERENCES[TITLE_KEYS[game]])
    yr, yh = baseline[game]
    rows.append({
        'game': game, 'manuscript_random': mr, 'yaml_random': yr,
        'manuscript_human': mh, 'yaml_human': yh,
        'match': mr == yr and mh == yh})
  output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
  with output.open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=rows[0])
    writer.writeheader(); writer.writerows(rows)
  if not all(row['match'] for row in rows):
    raise AssertionError([row for row in rows if not row['match']])
  return rows
