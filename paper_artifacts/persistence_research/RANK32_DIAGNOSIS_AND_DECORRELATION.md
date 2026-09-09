# Rank-floor outcome and block-decorrelation experiment

Job 3353 completed all four game workers with exit code zero. The status file
records COMPLETE; the scheduler has since evicted this job from live records.
The run took approximately 33 minutes from manifest creation to completion.

| Game | Isotropy control | Rank-32 control | Episode-bootstrap difference 95% interval | First-prefix rank, old → new | CKA, old → new |
| --- | ---: | ---: | --- | --- | --- |
| Boxing | 63.55 | 49.6 | [-27.0, -0.4] | 65.31 → 34.94 | 0.429 → 0.623 |
| Up N Down | 6716.5 | 5419.0 | [-2268.03, -346.99] | 55.13 → 33.09 | 0.458 → 0.602 |
| Frostbite | 2336.5 | 2622.0 | [-22.51, 597.0] | 46.68 → 25.21 | 0.449 → 0.701 |
| Road Runner | 15545 | 15255 | [-1340, 810] | 64.27 → 29.03 | 0.482 → 0.677 |

Twenty evaluation episodes and one continuation seed per condition. Intervals
condition on checkpoints; exploratory comparisons are not multiplicity-adjusted.
Boxing's first-prefix future R2 improved at horizons 16 and 32, but policy
performance decreased and whitened displacement increased. Other games did not
show convincing first-prefix predictive gains. Both rank and redundancy moved
in the wrong direction for the intended goal. Reject rank floor 32 as a promoted
method; the hypothesis that relaxing isotropy would give broadly better stable,
useful prefixes was not supported. A training-batch rank floor is not a held-out
rank guarantee: Frostbite/Road Runner fell below 32 on the fixed test clips.

Source Up N Down evaluation was 13565, versus 6716.5 even in the isotropy
continuation. Fresh replay, continuation, and policy sample variability remain
important alternative explanations for source-to-continuation changes. Update
counts differed by 0–5 across paired arms. Do not attribute all source-relative
degradation to the experimental penalty.

Next hypothesis: explicitly discouraging linearly shared block content while
retaining first-prefix isotropy will promote complementary later-block features
without the diversity loss observed under rank relaxation. Compare isotropy
alone with isotropy plus 0.01 linear CKA between centered block 1 and block 2.
Block 1 is detached in this extra loss; only block 2 gets direct activation
gradients, although shared projection parameters can indirectly affect block 1.
No parameters are added. Both arms have rank floor disabled. This isolates one
factor within the new pair; it is not a causal comparison against the previous
rank32 arm, which differs in two settings.

Risks: CKA only tests linear redundancy; second-block collapse is a degenerate
minimum; over-separating shared control state may harm policy or prefix dynamics.
Thus reduced CKA alone is insufficient. Require retained block-2 rank/variance,
long-horizon first-prefix and cumulative-prefix prediction, and control. A
positive outcome still requires additional seeds and complete-episode probes.

Use the same four games, 100k source checkpoints, and 20k new actions per arm.
Copy the existing fixed diagnostic clips with verified hashes to keep the
representation measurements identical. Source evaluation is repeated; the
report now includes policy comparisons against the source as well as the paired
control. Two sequential conditions per game, four game workers simultaneously,
two per H200 and eight disjoint CPU IDs per worker. All computation stays in
one Slurm job. Frozen source copies isolate queued subprocesses from edits.

Expected duration after allocation: 30–50 minutes, based on job 3353, with a
four-hour scheduler limit. Only startup is checked; the user will request the
next diagnosis after completion. Raw prior evidence is
`production_runs/rank32_four_game_TN09vuQY/comparison.json`.
