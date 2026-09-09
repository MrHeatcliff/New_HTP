# Persistence research: current protocol

The target is an empirical improvement in compact-prefix stability and useful
long-horizon prediction without loss of information or control performance.
None of the implemented losses guarantees identification of semantic factors.

## Execution

All numerical work, including tests and statistical analysis, must execute in
Slurm. Keep at most one research allocation active/submitted at a time. Do not
cancel, modify, or use other users' jobs. Login-node work is limited to reading
status/files and editing source. Job 3328 runs its stages sequentially.

Do not modify the training or evaluation source used by an active experiment.
New analysis is implemented in separate files and scheduled after it finishes.
Keep interrupted artifacts as interrupted artifacts; use fresh output roots.

## Current hypothesis

The pooled temporal penalty is a ratio of total squared displacement to total
variance. It is invariant to uniform scale but not to separate rescaling of
coordinates. Increasing the amplitude of a slow coordinate can lower the loss
without removing fast information from the prefix. A covariance-normalized
displacement should reduce this shortcut. A relative ridge makes the solve
finite, but exact collapse and extreme anisotropy remain possible failure modes.

The current paired experiment changes only `persistence_metric`, from `pooled`
to `whitened`. Both arms use persistence weight 1.0, all lags 1..16, isotropy
weight 0.01, the same completed Alien size25m source checkpoint, continuation
seed 0, and 3,000 additional environment actions. This is short continuation,
not independent training from scratch. Replay is newly populated in both arms.

## Measurements

- Synthetic: 3 paired seeds, 5,000 updates; held-out slow-factor and control-state
  probes, eigenvalue participation ratio, minimum variance, cross-block probe
  redundancy, and frozen-representation closed-loop controller returns.
- Real control: 50 evaluation episodes per checkpoint. Compare the two Alien
  continuation arms directly. Breakout's historical baseline uses the same size
  and nominal budget but differs in training-device history; report that caveat.
  There is no matched unconstrained size25m Alien baseline in this experiment.
- Real representation: fixed shared Alien episodes; 10 train, 10 validation,
  10 test episodes, up to 256 observations from each episode. Ridge regularization
  is selected on validation only. Future targets are a fixed random projection
  of the original cached RSSM features, not each arm's moving latent target.
- Report prediction at 16, 32, and 64 steps, participation ratio, coordinate
  variance, and block-1/block-2 CKA. Also evaluate temporal displacement after
  fitting a whitener on TRAIN episodes, so improvements in raw Euclidean
  displacement cannot by themselves count as successful stability learning.
- Prediction intervals resample matched test episodes. Policy intervals
  independently resample the episode returns of each checkpoint. These intervals
  are exploratory and conditional on trained models; they do not measure
  variation across training seeds. No noninferiority claim follows merely from
  an insignificant difference or an improved point estimate.

## Diagnosis and subsequent decisions

If only raw displacement improves, reject a claim of semantic stability. If
rank improves but predictive information or control worsens, do not promote the
constraint. Synthetic success alone does not justify an Atari claim. An
encouraging real result requires independent training seeds and held-out
confirmation before changing defaults. All constraint weights remain zero by
default; `pooled` remains the compatibility default metric.

The auxiliary synthetic harness uses a linear encoder/decoder and actual
MultiStridePDyn heads, with a detached online target rather than the production
EMA projection and progressive reconstruction. Its full-code controller is
especially permissive because an invertible linear encoder preserves all input
information. Treat these results as mechanism tests, not Dreamer policy results.

## September 9 continuation

Long run 3330 completed successfully: Alien size25m, seed 0, nominal 110k
training steps, whitened persistence at weight 0.1, all lags, isotropy 0.01.
Its evaluation must not be compared with training episode scores. The comparison
against the earlier long pooled-weight-1 run changes both metric and weight;
it is a candidate comparison, not an isolated attribution to either factor.

Job 3331 evaluates that checkpoint and measures representation properties on
the same fixed diagnostic episodes. It also runs tests for a separate correction:
`agent.htp.pdyn.mask_episode_boundaries`. When enabled, source/target pairs that
cross an episode reset have zero prediction loss. The option defaults to false
to preserve historical controls. The agent always supplies reset flags, and
per-stride cross-episode fractions are now available as diagnostics. The helper
rejects missing reset flags when masking is enabled.

Reset masking is not a semantic-invariance constraint and is not presented as
one. It removes invalid temporal supervision. Its effect on policy and learned
representations requires a separate paired experiment; do not mix it into the
normalization/weight comparison while claiming a single changed factor.

### Completed audit (job 3331)

The new long Alien checkpoint returned 812.8 versus 434.0 for the earlier
pooled-weight-1 checkpoint over 50 evaluation episodes each. The independent
episode-bootstrap difference interval was approximately [270.6, 502.6]. This
does not measure training-seed uncertainty or isolate the two changed factors.
First-prefix effective rank increased from 7.87 to 40.49 and block CKA fell
from 0.845 to 0.356. Train-fitted whitened displacement improved at long lags.
However, first-prefix future-probe improvements were inconclusive, and block 2
still outperformed block 1 at horizons 16 and 32. Therefore the desired temporal
hierarchy is NOT established, despite improved control and diversity.
Full numbers: `latest_3331/comparison.json`.

### One-factor refinement (job 3332)

Hypothesis: forcing every progressive decoder, including the smallest prefix,
to reconstruct the instantaneous RSSM feature rewards fast details and conflicts
with long-horizon stability. Replace only the first decoder's target with a
causal 16-observation rolling average. Other decoder targets, predictive losses,
weights, and episode-mask setting are unchanged. The average never crosses
episode boundaries and uses available history at replay-chunk starts. This is
low-pass supervision, not a guarantee of semantic identity; risks include lagged
state, removal of control-relevant signals, and redundancy with later prefixes.

The implementation defaults to window 1 (historical behavior) and adds no model
parameters. Job 3332 runs unit tests followed by paired window-1/window-16
continuations from exactly the same checkpoint, 5,000 new environment actions
each with fresh replay, then 50 policy episodes and fixed representation probes
per arm. Both arms use whitened persistence 0.1 and isotropy 0.01; reset masking
stays off. Arms run sequentially inside a single one-GPU Slurm allocation.
Outputs: `coarse_target_3332/`. No effect of this refinement is established
until its comparison report exists. This short one-seed test cannot establish
noninferiority or replace matched from-scratch, multi-seed controls.

### Coarse-target diagnosis and next experiment

Job 3332 produced the complete report and all six train/eval/extraction stages
finished. Slurm accounting was unavailable at the subsequent check, so scheduler
exit status could not be independently retrieved. Twenty unit tests passed.
Control: raw 499.4, coarse 742.8 (50 episodes each); conditional bootstrap
difference interval [132.2, 363.4]. However, first-prefix future R2 decreased
at all reported long horizons: 16, 0.15956 to 0.15075; 32, 0.09391 to 0.08588;
64, 0.03748 to 0.03300 (difference interval [-0.00756, -0.00170]). Rank improved
37.73 to 41.98 and CKA fell 0.399 to 0.336. Whitened displacement improved
slightly, but block 2 remained more predictive at horizons 16/32. Do not promote
causal averaging as a solution to temporal hierarchy. Both continuations also
had lower point-estimate policy scores than their source checkpoint (812.8);
fresh-replay continuation is not a clean test of from-scratch training quality.
Updates were 980 versus 981, an additional causal attribution limitation.

Next hypothesis: invalid prediction pairs crossing episode resets undermine
long-stride learning more than short-stride learning. The prior replay audit
found 2.37% invalid stride-16 pairs. Job 3333 compares unmasked/masked prediction
from the SAME original whitened-0.1 long checkpoint, with window 1 in BOTH arms.
Only the already-tested episode-mask option changes. Each arm receives 5,000
new actions, 50 evaluation episodes, and fixed representation diagnostics;
update counts are audited. Both arms run sequentially inside one GPU allocation.
Masking is a supervision-correctness hypothesis, not a semantic-invariance
guarantee. Zeroing invalid pairs also reduces average dynamics loss weight;
a positive result would need a weight-matched follow-up to separate those effects.
Output: `episode_mask_3333/`. No result is assumed in advance.
