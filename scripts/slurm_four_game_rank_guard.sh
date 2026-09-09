#!/usr/bin/env bash
#SBATCH --job-name=rankguard-four-games
#SBATCH --partition=gpu_junior
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_h200:2
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=/home/vn-user0101/Dat/HTS-Dreamer/slurm_logs/rankguard-four-%j.out
#SBATCH --error=/home/vn-user0101/Dat/HTS-Dreamer/slurm_logs/rankguard-four-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit using sbatch}"
readonly ROOT="${1:?Supply frozen experiment directory}"
readonly PYTHON=/home/vn-user0101/Dat/HTS-Dreamer/.venv/bin/python
export HTP_RESEARCH_REPO=/home/vn-user0101/Dat/HTS-Dreamer
export HTP_FOUR_GAME_FACTOR=decorrelation_guard HTP_TRIAL_ACTIONS=10000 HTP_CONTINUATION_SEEDS=0,1
export HTP_FIXED_CLIPS_ROOT="$HTP_RESEARCH_REPO/production_runs/rank32_four_game_TN09vuQY"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1
cd "$ROOT/source"
"$PYTHON" -m py_compile corewm_eval/four_game_rank_trial.py corewm_eval/four_game_representation.py
"$PYTHON" -m pytest tests/test_prefix_block_redundancy.py tests/test_prefix_rank_floor.py \
  tests/test_prefix_persistence.py tests/test_phase3h_semantics.py tests/test_pdyn_episode_mask.py \
  tests/test_coarse_reconstruction_target.py tests/test_h200_suite.py tests/test_four_game_factors.py -q
exec "$PYTHON" -u -m corewm_eval.four_game_rank_trial "$ROOT"
