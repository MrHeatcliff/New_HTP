# Decorrelation outcome and second-block diversity guard

Verified Slurm job 3368: COMPLETED, ExitCode=0:0, runtime 00:32:45,
2026-09-09 14:42:10–15:14:55 UTC+7. All four workers exited zero;
29 tests passed. Scheduler log: `slurm_logs/decorrel-four-3368.out`.
Detailed stage logs and comparison.json are under
`production_runs/decorrelation_four_game_ro2vQg01/`.

| Game | Isotropy return | Decorrelation return | Conditional difference 95% interval | Block-2 rank, old → new | CKA, old → new |
| --- | ---: | ---: | --- | --- | --- |
| Boxing | 63.55 | 30.35 | [-43.65, -22.5] | 7.96 → 5.56 | 0.429 → 0.363 |
| Up N Down | 6716.5 | 4947 | [-2706.51, -830] | 9.79 → 4.02 | 0.458 → 0.341 |
| Frostbite | 2526 | 2540 | [-249, 276] | 4.54 → 4.42 | 0.407 → 0.383 |
| Road Runner | 12215 | 15540 | [2285, 4365] | 15.21 → 4.88 | 0.508 → 0.308 |

CKA decreased in all games, but second-block rank decreased too. This supports
the suspected degeneracy of suppressing shared signals instead of retaining
useful complementary information. It does not establish that rank loss caused
the control changes. First-prefix long-horizon improvements were limited/mixed,
and whitened displacement often increased. Do not promote decorrelation alone.
No all-coordinate constant collapse was observed in the first prefix; lost
effective rank in block 2 is the relevant failure signal.

The Road Runner isotropy control differs across repeats: 15545 in job 3353
versus 12215 in job 3368, with 4734 versus 4735 updates. Thus a conditional
episode-bootstrap interval does not settle training/continuation variability.
The source Road Runner policy returned 16485 and source Up N Down 13565;
both methods can underperform their source. Fresh-replay continuation remains
a confound for source-relative changes. All evidence is exploratory, with
20 policy episodes per checkpoint and no multiplicity correction.

Next single-factor comparison: CKA weight 0.01 versus the same CKA weight plus
a second-block participation-rank floor of 8, weight 0.001. The additional
penalty is max(0, D2/max(r2,1) - D2/8). First-prefix isotropy stays 0.01,
whitened persistence stays 0.1, rank32 remains disabled. The isotropy-only arm
is included as a contemporaneous anchor. The candidate adds no parameters.
Direct guard gradients affect block 2 only; shared-trunk effects still exist.
Exact constants have zero covariance escape gradient, and an in-batch floor
does not guarantee held-out rank. The chosen floor is a mechanism probe, not
a universal intrinsic dimensionality claim (Frostbite's anchor is below 8).

Use four game slots, two per H200, eight disjoint CPU IDs each. Each slot runs
three conditions for continuation seeds 0 and 1, sequentially: 24 total short
continuations, at most four running concurrently. Each starts from that game's
same seed-0 100k checkpoint with fresh replay and 10k new actions. This is
shorter than the prior 20k follow-up; conclusions are restricted to this budget.
Continuation seed varies environment/replay initialization, not pretrained
training seed. Evaluate every final policy with evaluation seed 0 and 20
complete episodes. Keep results by continuation seed; do not pool the episodes
as independent trained policies. Reuse the exact hash-verified diagnostic clips
from job 3353 and report validation-selected future probes, CKA, both block
ranks, whitened displacement, and logged guard activation.

Primary causal contrast: guarded versus decorrelated, one changed loss scale.
Also report both against isotropy and source. Require control retention and
useful future information beyond a rank/CKA metric change across both seeds.
If held-out rank improves but control stays worse, reject the notion that rank
loss alone explains the failure; reconsider forcing block independence. If the
guard never activates, the test is inconclusive about protecting diversity.
If outcomes disagree across seeds, prioritize continuation/replay variability
before adding more regularizers. The original stable-semantic-prefix goal
remains open.

Expected runtime: 60–90 minutes after allocation; time limit four hours. All
tests, metrics, and experiments run in Slurm. Frozen source and per-stage
timestamps make the run auditable. Only startup is checked interactively.
