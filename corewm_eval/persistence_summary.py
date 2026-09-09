"""Exploratory episode bootstrap; does not estimate training-seed uncertainty."""
import json
from pathlib import Path
import numpy as np


def run():
  root = Path('paper_artifacts/persistence_research/atari_matched')
  rng = np.random.default_rng(137)
  names = [name for name in ('baseline','all_lags','isotropic','isotropic_001','persistent_1')
           if (root / f'{name}_representation.json').exists()
           and (root / f'{name}_eval/paper_artifacts/final_eval.json').exists()]
  scores = {name: np.array([json.loads(line)['episode/score'] for line in
      (root / f'{name}_eval/scores.jsonl').read_text().splitlines()]) for name in names}
  reps = {name: json.loads((root / f'{name}_representation.json').read_text()) for name in names}
  rows = []
  for name in names:
    row = dict(name=name, score=float(scores[name].mean()), episodes=len(scores[name]),
        rank=reps[name]['effective_rank'], cka=reps[name]['block12_cka'],
        change16=reps[name]['horizons']['16']['normalized_change'],
        future_r2={lag: reps[name]['horizons'][lag]['r2'] for lag in ('16','32','64')})
    if name != 'baseline':
      bs = scores['baseline']; cs = scores[name]
      difference = cs[rng.integers(len(cs),size=(20000,len(cs)))].mean(1) - bs[rng.integers(len(bs),size=(20000,len(bs)))].mean(1)
      row['score_difference_ci95'] = np.quantile(difference,[.025,.975]).tolist()
      row['r2_difference_ci95'] = {}
      for lag in ('16','32','64'):
        a = reps['baseline']['horizons'][lag].get('episode_statistics')
        b = reps[name]['horizons'][lag].get('episode_statistics')
        if a is None or b is None: continue
        ids = sorted(a); assert ids == sorted(b)
        sample = rng.integers(len(ids),size=(20000,len(ids)))
        ar = 1-np.array([a[e]['sse'] for e in ids])[sample].sum(1)/np.array([a[e]['sst'] for e in ids])[sample].sum(1)
        br = 1-np.array([b[e]['sse'] for e in ids])[sample].sum(1)/np.array([b[e]['sst'] for e in ids])[sample].sum(1)
        row['r2_difference_ci95'][lag] = np.quantile(br-ar,[.025,.975]).tolist()
    rows.append(row)
  payload = {'scope':'one source checkpoint, one continuation seed, exploratory intervals conditional on trained checkpoints', 'rows':rows}
  (root / 'comparison.json').write_text(json.dumps(payload,indent=2))
  print(json.dumps(payload,indent=2))


if __name__ == '__main__':
  run()
