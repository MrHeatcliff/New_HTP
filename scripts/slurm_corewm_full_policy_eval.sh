#!/usr/bin/env bash
#SBATCH --partition=gpu_junior
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/vn-user0101/Dat/HTS-Dreamer/slurm_logs/corewm-v1-full-eval-%j.out
#SBATCH --error=/home/vn-user0101/Dat/HTS-Dreamer/slurm_logs/corewm-v1-full-eval-%j.err
set -euo pipefail
cd /home/vn-user0101/Dat/HTS-Dreamer
exec /home/vn-user0101/.conda/envs/htp/bin/python -m corewm_eval.full_policy_stage run-job \
  --manifest "${COREWM_EVAL_MANIFEST}" \
  --ordinal "${COREWM_EVAL_ORDINAL}" \
  --worktree "${COREWM_CLEAN_WORKTREE}"
