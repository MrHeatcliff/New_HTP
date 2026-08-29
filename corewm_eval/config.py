import numpy as np

EVALUATION_SEED = 0
TRAINING_SEEDS = (0, 1, 2, 3, 4)
ATARI100K_GAMES = (
    'alien', 'amidar', 'assault', 'asterix', 'bank_heist', 'battle_zone',
    'boxing', 'breakout', 'chopper_command', 'crazy_climber', 'demon_attack',
    'freeway', 'frostbite', 'gopher', 'hero', 'jamesbond', 'kangaroo',
    'krull', 'kung_fu_master', 'ms_pacman', 'pong', 'private_eye', 'qbert',
    'road_runner', 'seaquest', 'up_n_down')
HORIZONS = (1, 2, 4, 8, 16, 32, 64)
LONG_HORIZONS = (16, 32, 64)
ALPHA_GRID = np.logspace(-6, 6, 13)
PERFORMANCE_THRESHOLDS = np.linspace(0.0, 8.0, 161)
FINAL_CHECKPOINTS = tuple(range(10_000, 100_001, 10_000))
QUALITATIVE_GAMES = (
    'alien', 'bank_heist', 'frostbite', 'kangaroo', 'ms_pacman', 'seaquest')
SPLIT_COUNTS = {'train': 60, 'validation': 20, 'test': 20}

WANDB_ENTITY = 'ttdat170703-ho-chi-minh-city-university-of-technology'
WANDB_GAME_ALIASES = {'jamesbond': 'james_bond'}


def wandb_project_for_game(game):
  """Return the existing team project assigned to one canonical Atari game."""
  if game not in ATARI100K_GAMES:
    raise ValueError(f'Not a canonical Atari100K game: {game!r}')
  return f'dreamv3-{WANDB_GAME_ALIASES.get(game, game)}'
