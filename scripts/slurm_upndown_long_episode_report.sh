#!/usr/bin/env bash
#SBATCH --job-name=upndown-long-episodes
#SBATCH --partition=cpu_general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/home/vn-user0101/Dat/HTS-Dreamer/slurm_logs/upndown-long-episodes-%j.out
#SBATCH --error=/home/vn-user0101/Dat/HTS-Dreamer/slurm_logs/upndown-long-episodes-%j.err
set -euo pipefail
cd /home/vn-user0101/Dat/HTS-Dreamer
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec /home/vn-user0101/.conda/envs/htp/bin/python -m corewm_eval.upndown_long_episode_report
