# Four-game rank-floor refinement

The 26-game run completed, but comparisons with historical curves remain
exploratory: one constrained training seed versus five baseline seeds, and
training episodes versus isolated evaluation for the historical Full method.
Select Boxing and Up N Down (apparent weaknesses), Frostbite and Road Runner
(apparent strengths) as development cases. They are no longer held-out games.

The existing purple curves have a plotting defect: `_binned` folds episodes
above 100k into the final 95–100k bin. This experiment's initial audit reads
the exact `agent_actions` fields and keeps 90–100k and 100–110k separate.
Published figures have not been regenerated in this turn. Raw logger `step`
also includes action-repeat scaling as well as reset callbacks; do not use it
as executed actions without consulting the artifact counters.

Hypothesis: covariance isotropy keeps rewarding additional varying directions
even after the prefix has enough diversity. This can favor fast nuisance
information and compete with temporal persistence. For covariance C,
participation rank r = tr(C)^2 / tr(C^2). For nondegenerate C the old penalty
is D/r - 1. The new penalty is max(0, D/r - D/32), with the same weight 0.01.
Below rank 32 it retains the old local gradient; above rank 32 it stops pushing
toward full isotropy. Numerical floors keep degenerate cases finite. Exact
constants retain zero covariance-gradient escape, so this is not a collapse
guarantee. Dimensional concentration can also artificially reduce the whitened
persistence objective averaged over D; predictive probes must accompany rank.

Four game workers execute simultaneously: two workers per H200, eight disjoint
CPU IDs per worker. Each game uses its own frozen suite checkpoint at exactly
100k actions. Both arms load that same checkpoint, use fresh replay, seed 0,
and 20k new actions. The sole changed method option is
`agent.htp.persistence_min_rank`: 0 (historical isotropy) versus 32. All other
weights, projection dimensions, predictive horizons, target window 1, and reset
mask off are identical. Arm order is counterbalanced across the four games.
This is eight continuations in two sequential stages per game, not eight
simultaneous training processes. Exact optimizer update deltas are reported;
equal action budgets do not ensure equal updates when episode lengths differ.

Each game collects 30 independently reset source-policy clips, at most 256
observations each, truncated at natural termination. This covers early states
and is not a full-game diagnostic distribution. A fixed random split uses 10
clips each for train/validation/test. The source model's h projected to 64
dimensions supplies a frozen probe target for both arms. Probe regularization
is chosen on validation only. Report matched-width block-1/block-2 and combined
prefix R2 at 1/4/8/16/32/64, train-fitted whitening displacement, effective rank,
mean variance, dead coordinates and CKA. These are predictive proxies, not
semantic ground truth. Source and both final policies get 20 isolated complete
evaluation episodes each. Bootstrap intervals only quantify conditional
episode uncertainty, not training-seed uncertainty or noninferiority.

Accept the hypothesis provisionally only if long-horizon prediction improves
without severe diversity loss/redundancy and control is retained across both
weak and strong development cases. Better slowness with worse R2 is a failure;
mixed/noisy results require refinement or additional seeds. No outcome is
assumed before `comparison.json` is produced.

Expected duration: 45–90 minutes after allocation, based on the 26-game run's
per-game training times. Evaluation episode lengths and extraction compilation
add uncertainty. Scheduler time limit is four hours. The user requested no
long-term monitoring; verify startup and leave the job to produce its reports.
