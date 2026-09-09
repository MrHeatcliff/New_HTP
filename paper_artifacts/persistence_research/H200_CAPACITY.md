# One H200: measured two-game sharing

Job 3334, H200 with 143771 MiB VRAM, 16 allocated CPU cores, 128 GiB host
memory. Same size25m HTP settings in every run: whitened persistence 0.1,
isotropy 0.01, causal target window 1, episode mask off, seed 0. Each fresh
training run executed 5000 actions. Eight pinned CPU cores per process.
Preallocation disabled; no MPS or global GPU settings changed.

| Execution | Wall time (s) | Peak GPU memory (MiB) | Peak summed process RSS (MiB) |
| --- | ---: | ---: | ---: |
| Alien alone | 138.11 | 7103 | 3269.57 |
| Breakout alone | 138.12 | 7103 | 3112.75 |
| Both simultaneously | 209.36 | 14189 | 6354.52 |

All four training children exited zero. Aggregate wall-time speedup 1.3194x,
including initialization, compilation, replay prefill, and checkpoint writes.
Steady training samples showed 85–88% GPU utilization solo and 99–100% paired;
whole-run mean utilization was approximately 49% solo and 72% paired.
CPU telemetry from `ps %cpu` is lifetime-average, not instantaneous CPU load.
RSS sums can double-count shared pages. GPU statistics are sampled every 2 s,
so reported peaks are sampled peaks, not guaranteed transient maxima.

Recommendation: two concurrent games on one H200 for this measured workload.
No evidence supports extrapolating to three or more games merely from free
VRAM. Long replay growth, evaluation, other model sizes, and different games
need separate verification. This is a capacity test, not a control-quality
comparison; both sequential and simultaneous training metrics are exploratory.

Reusable launcher: `scripts/slurm_h200_two_games.sh [actions-per-game]`.
Default 5000; long budget must be explicitly supplied. It requires this successful
benchmark report, isolates output per Slurm ID, and terminates only its own child
process groups if GPU memory exceeds 90% or summed RSS exceeds 100 GiB. The
XLA memory fraction is NOT assumed to enforce a hard cap with preallocation off.
The original benchmark execution path was verified in job 3334; the convenience
paired-only launcher was subsequently added and shell syntax-checked, but has
not yet been separately submitted. No extra long training job was launched.

Raw evidence: `h200_capacity_3334/results.json` and per-stage telemetry JSONL.
