# CoRe-WM constraint research: hypotheses, interventions, evidence, and current diagnosis

Last updated: 12 September 2026. Branch: `constraint`.

This report reconstructs the research sequence from the first request to make compact prefixes temporally stable through the completed constrained factorial job 3669. It distinguishes a measured result from an interpretation. A hypothesis is marked **supported locally**, **rejected**, **mixed**, or **open**. “Supported locally” never means a theorem or broad Atari claim: most development experiments use one training seed, four selected games, short continuations, or limited early-episode probes.

## 1. Original weakness and research target

The original Multi-Stride Prefix Dynamics objective required

$$
z_t^{(1:\ell)} \longrightarrow z_{t+\Delta_\ell}^{(1:\ell)}
$$

to be predictable from the current prefix and intervening actions. Predictability alone does not guarantee that the same prefix keeps the same stable factors or semantic role over time. A rapidly changing but deterministic signal can be easy to predict. A constant representation can be temporally stable but useless.

The empirical target was therefore conjunctive:

1. compact prefix changes slowly at longer horizons;
2. it retains predictive and task-relevant information;
3. it does not collapse or become redundant;
4. it does not reduce policy performance.

No implemented loss identifies semantic factors by itself. Every conclusion below is bounded by the targets and trajectories actually measured.

## 2. Reference method and constraint package

The reference representation is

$$
h_t=[h_t^{\mathrm{deter}},\mathrm{vec}(h_t^{\mathrm{stoch}})],
\qquad z_t=S_\psi(h_t),
$$

with cumulative dimensions `[128,256,512,1024,2048]` and assigned strides `[16,8,4,2,1]`. Actor, critic, reward and continuation heads consume the full $z$, while policy imagination still uses the base RSSM.

The selected constraint package adds to multi-stride prediction:

$$
\mathcal L_C=0.1\mathcal L_{\mathrm{persist}}+0.01\mathcal L_{\mathrm{iso}}.
$$

Persistence measures covariance-normalized displacement of the first 128-dimensional prefix over every lag 1 through 16, excluding pairs crossing episode resets:

$$
\mathcal L_{\mathrm{persist}}=
\frac1K\sum_{k=1}^{K}\frac1{2d}
\mathrm{tr}\left[(C_k+\epsilon_kI)^{-1}Q_k\right],
\qquad d=128.
$$

Isotropy penalizes concentration of covariance in a few directions:

$$
\mathcal L_{\mathrm{iso}}=
\frac1d\left\|\frac{C}{\max(\mathrm{tr}(C)/d,10^{-8})}-I\right\|_F^2.
$$

The complete implementation-level derivation is in [CONSTRAINT_MATH_AND_INTUITION.md](CONSTRAINT_MATH_AND_INTUITION.md). The 26-game constraint suite used exactly whitened persistence 0.1, all lags, isotropy 0.01, latent $h$ reconstruction, first target window 1, and prediction reset mask off. All 26 saved configs match.

## 3. Evidence ladder and common limitations

Experiments in this history are not equally strong:

* Synthetic optimization tests mechanism in a simplified linear setting.
* Short continuation tests sensitivity after loading one checkpoint, with fresh replay. It is not independent training.
* Four-game from-scratch runs provide causal within-run arm comparisons for one training seed.
* Evaluation-episode bootstrap intervals condition on already trained policies. They do not measure training-seed uncertainty.
* Fixed representation probes use source-policy early episodes, typically 10 train, 10 validation and 10 test episodes. Their target is often a random projection of source RSSM features, not semantic labels.
* Horizon 64 is extrapolative: the first prefix is trained at stride 16 and actor imagination length is 15 in inspected configs.
* The four development games are no longer held out after repeated use.

## 4. Chronological research record

### 4.1. Pooled temporal penalty and covariance whitening

**Hypothesis.** The original pooled ratio can be reduced by increasing the amplitude of a slow coordinate, hiding fast coordinates inside total variance. Covariance-normalized displacement should remove this directional scale shortcut.

**Intervention.** Replace pooled displacement/total-variance normalization with a full covariance solve plus relative ridge. Add first-prefix isotropy because whitening alone does not prevent concentration or exact collapse. Mechanism tests used paired synthetic seeds; Atari used Alien continuations and a later long checkpoint.

**Evidence.** The long whitened-0.1 Alien checkpoint scored 812.8 versus 434.0 for an earlier pooled-weight-1 checkpoint over 50 episodes; conditional difference interval approximately `[270.6,502.6]`. First-prefix effective rank increased `7.87→40.49`, block CKA fell `.845→.356`, and held-out whitened displacement improved. First-prefix future-probe gains were inconclusive, and block 2 remained more predictive at horizons 16 and 32.

**Diagnosis: supported as a better constraint package, temporal hierarchy still open.** The control comparison changed both metric and weight and did not use independent training seeds. Diversity and control improved, but the desired semantic/predictive hierarchy was not established. This package became the research reference, not a guarantee.

### 4.2. Causal coarse reconstruction target

**Hypothesis.** Forcing the smallest prefix to reconstruct instantaneous $h_t$ rewards fast details and conflicts with its stride-16 role. A causal 16-observation average for only the first reconstruction target should improve stable long-horizon content.

**Intervention.** Compare first-target windows 1 and 16 while keeping prediction and constraints fixed. Job 3332 continued the same Alien checkpoint for 5k actions per arm.

**Evidence.** Control increased `499.4→742.8`, conditional interval `[132.2,363.4]`; rank increased `37.73→41.98`; CKA fell `.399→.336`; displacement improved slightly. However B1 future R2 fell at every reported long horizon: `.15956→.15075` at 16, `.09391→.08588` at 32, `.03748→.03300` at 64. Both arms also remained below the source policy score 812.8.

**Diagnosis: rejected as a temporal-hierarchy solution.** It improved some proxies and short-continuation control but worsened the core predictive measurement. Fresh replay and one continuation seed further limit attribution. Causal averaging was not promoted.

### 4.3. Remove prediction pairs crossing episode resets

**Hypothesis.** Invalid source-target pairs crossing an episode boundary damage long-stride learning. The audit found 2.37% invalid stride-16 pairs.

**Intervention.** Job 3333 compared prediction reset mask off/on from the same Alien checkpoint, 5k additional actions, window 1, 50 evaluation episodes.

**Evidence.** Masking improved B1 R2 at horizon 32 from `.08847→.10837`, interval `[.00562,.03407]`, and horizon 64 from `.03444→.03887`, interval `[.00096,.00767]`. Rank fell `37.51→36.43`; control fell `575.6→475.4`, difference `-100.2`, interval `[-298,49]`. Prediction loss weight also decreased when invalid pairs were removed.

**Diagnosis: supervision bug confirmed; control benefit unproven.** Episode-safe targets are conceptually correct, but this continuation did not establish improved control and was not weight-matched. Historical reference runs keep mask off for comparability. A future clean correction should normalize by valid-pair count and rerun from scratch.

### 4.4. Full 26-game constraint suite

**Hypothesis.** The whitened persistence 0.1 + isotropy 0.01 package is viable across Atari rather than only Alien.

**Intervention.** Job 3335 trained all 26 Atari 100K games, seed 0, size25m, 110k actions per game, two H200 with two simultaneous games/GPU and eight CPUs/game.

**Evidence.** All 26 games completed with matching saved configs and nonzero logged prediction, persistence and isotropy metrics. The run supplies the purple line in `full_vs_dreamerv3_constraint_suite_learning_curves`. It is one-seed training-episode data, while historical curves include five seeds and, for Full CoRe-WM, isolated evaluations.

**Diagnosis: feasibility supported, comparative performance not settled.** The suite shows the package trains across all games without universal numerical collapse. It is not a five-seed controlled comparison against the unconstrained Full method. The constraint remained a development reference because earlier paired evidence was encouraging, not because this suite proved semantic hierarchy.

### 4.5. Relax isotropy with a rank floor

**Hypothesis.** Full isotropy may keep rewarding fast nuisance dimensions after enough diversity exists. Stop the penalty once participation rank reaches 32.

**Intervention.** Job 3353 continued four 100k source checkpoints for 20k actions, comparing isotropy against rank-floor 32.

| Game | Isotropy | Rank floor 32 | B1 rank old→new | CKA old→new |
|---|---:|---:|---:|---:|
| Boxing | 63.55 | 49.6 | 65.31→34.94 | .429→.623 |
| Up N Down | 6716.5 | 5419 | 55.13→33.09 | .458→.602 |
| Frostbite | 2336.5 | 2622 | 46.68→25.21 | .449→.701 |
| Road Runner | 15545 | 15255 | 64.27→29.03 | .482→.677 |

**Diagnosis: rejected.** Control decreased clearly in Boxing and Up N Down; rank and redundancy moved opposite the intended direction. A training-batch floor did not guarantee held-out rank. Better B1 R2 in isolated cases did not compensate for control/diversity failure.

### 4.6. Block-1/block-2 decorrelation

**Hypothesis.** Explicitly reducing linear CKA should make block 2 complementary rather than redundant with block 1.

**Intervention.** Job 3368 compared isotropy alone against isotropy plus CKA weight 0.01, with block 1 detached in the CKA loss.

| Game | Isotropy | + decorrelation | B2 rank old→new | CKA old→new |
|---|---:|---:|---:|---:|
| Boxing | 63.55 | 30.35 | 7.96→5.56 | .429→.363 |
| Up N Down | 6716.5 | 4947 | 9.79→4.02 | .458→.341 |
| Frostbite | 2526 | 2540 | 4.54→4.42 | .407→.383 |
| Road Runner | 12215 | 15540 | 15.21→4.88 | .508→.308 |

**Diagnosis: rejected.** The optimized metric improved in all four games, but B2 rank collapsed further and control decreased in two games. This is direct evidence that reducing CKA alone can suppress shared useful signal rather than create complementary content. A B2 rank guard was proposed; no completed comparison artifact for that follow-up is present, so it is not counted as evidence.

### 4.7. Remove projection/reconstruction and predict directly from h

**Hypothesis.** Multi-stride prediction directly on $h_t$ may make the projection and progressive reconstruction unnecessary. Constraint should remain because earlier research selected it.

**Intervention.** Direct-$h$ ablations removed projection and reconstruction, enabled gradient to backbone, then compared prediction without and with the persistence/isotropy package on four games.

| Game | h prediction | h prediction + constraint |
|---|---:|---:|
| Boxing | 65.35 | 68.1 |
| Up N Down | 7241 | 6508 |
| Frostbite | 248.5 | 307.5 |
| Road Runner | 10445 | 8755 |

**Diagnosis: mixed and not promoted.** Constraint helped Boxing/Frostbite but hurt Up N Down/Road Runner. More fundamentally, fixed coordinate slices of $h$ have no learned ordered partition, so assigning horizons to the first 128/256/... coordinates does not provide a strong mechanism for semantic hierarchy. This supports retaining a learned projection/reconstruction structure as the reference, but does not prove every reconstruction objective is useful.

### 4.8. Spatial image reconstruction: latent, uniform, and pyramid targets

**Hypothesis.** Reconstructing spatial images by resolution may make prefix roles explicit: compact prefixes retain coarse persistent structure and later blocks add detail. Competing concern: spatially coarse is not temporally slow, and 4x4 can erase small control-relevant objects.

**Intervention.** Job 3405 trained latent-$h$, full-resolution image, and `[4,8,16,32,64]` pyramid targets from scratch, seed 0, four games, 100k actions. All retained prediction and constraint.

| Game | Latent h | Uniform image | Pyramid 4x4→64 |
|---|---:|---:|---:|
| Boxing | 80.05 | 81.85 | 91.25 |
| Up N Down | 27616.5 | 9286 | 35569 |
| Frostbite | 2083 | 2753.5 | 3238 |
| Road Runner | 9035 | 19000 | 9775 |

Pyramid beat uniform in three games, but B1 R2@64 changed `-.0561→-.0561`, `.2119→.1936`, `.3729→.3730`, `.1966→.1682` respectively. Up N Down control improved while B1/B2 rank decreased and CKA increased. No first-prefix dead coordinates were observed.

**Diagnosis: control signal at one seed; hierarchy unsupported.** Pixel target difficulty and background dominance make reconstruction loss insufficient evidence. Control and predictive probes did not move together.

### 4.9. Soften the first spatial target from 4x4 to 8x8

**Hypothesis.** The first 4x4 target is too coarse and removes small useful objects; `[8,16,32,64,64]` should recover control while preserving long-horizon image information.

**Seed-1 screening, job 3502:** soft beat 4x4 pyramid in Boxing `72.55 vs 64.2`, Frostbite `2489.5 vs 371.5`, Road Runner `13525 vs 8450`, but lost Up N Down `6417 vs 7164.5`. It beat uniform only in Road Runner.

**Seed-2 preregistered replication, job 3525:** soft lost to 4x4 pyramid in all four games:

| Game | Uniform | Pyramid 4x4 | Soft 8x8 |
|---|---:|---:|---:|
| Boxing | 72.05 | 74.15 | 63.55 |
| Up N Down | 176639.5 | 70242 | 66576.5 |
| Frostbite | 2312.5 | 3625.5 | 391 |
| Road Runner | 1855 | 13820 | 8420 |

Additional absolute-future-image B1 R2@64 on seed 1 was almost unchanged or moved opposite control: Frostbite `.5101→.4907`; Road Runner `.3564→.3564`. Delta-image probes had a current-image shortcut and were not accepted as motion semantics.

**Diagnosis: rejected.** The stated expectation—soft beats pyramid in at least three games on seed 2—failed. The 8x8 schedule is not a reliable improvement. Large run variation, such as Road Runner uniform `19000→1945→1855`, confirms that evaluation-episode intervals cannot substitute for training-seed replication.

### 4.10. Audit what the learned components actually do

**Hypotheses.** Predictor may copy persistent targets or ignore actions; aggregate reward loss may hide rare-event failure; actor may not use compact prefixes; posterior state may be adequate while imagination loses task signal.

**Intervention.** Jobs 3526 and 3668 evaluated 12 frozen seed-1 image checkpoints on ten common held-out episodes/game. Tests included actual predictor versus copy, rolled/zero actions, reward stratification, same-RNG RSSM rollout, same-window posterior/imagined heads, and equal-width 128-coordinate actor interventions.

**Evidence.** B1 predictor/copy MSE ratios ranged `.431–.690`; rolling actions increased B1 MSE by about `2.9–15.8%`. Episode-bootstrap intervals for copy-minus-predictor and rolled-minus-true were positive in all 12 checkpoints. This contradicts the strong “only copies/ignores action” hypothesis.

Reward failure differed by game. On identical frames:

* Frostbite uniform reward MAE `.401→1.125` from posterior to imagination, paired difference `.723 [.552,.879]`; pyramid `.232→.373`; soft `.156→.461`.
* Up N Down pyramid degraded `6.471→8.538`, difference `2.067 [1.120,3.050]`.
* Road Runner posterior was already near the zero-reward baseline: pyramid `10.416`, soft `10.485`, baseline `10.46875`; rollout added little error.
* All clips contained zero terminal events, so continuation calibration remained unmeasured.

Equal-width block interventions showed lower actor KL for B1 than the mean of B5 chunks in all 12 configurations; Road Runner uniform was `.0036 vs .0409`. Perturbation covariance still differs, and reliance is not usefulness.

**Diagnosis: several local failures identified, no single global cause.** Actual PDyn learns beyond copying and uses actions on these clips. Frostbite/Up N Down expose imagination-path reward degradation; Road Runner exposes reward prediction/information or sparse-coverage failure already on posterior states. Actor relies less on B1 under intervention, consistent with full-$z$ policy heads not being forced to exploit the compact prefix. No terminal evidence exists.

### 4.11. Constrained 2x2 factorial on the original latent-h formulation

**Hypothesis H1.** Prediction contributes to policy learning primarily when reconstruction preserves useful information; expect a positive interaction.

**Hypothesis H2.** Reconstruction supplies most benefit; prediction may optimize its target without helping control.

**Intervention.** Job 3669 added `prediction_loss_scale`, allowing prediction MSE to be zeroed while keeping persistence 0.1 and isotropy 0.01 active. Four from-scratch seed-0 arms trained for 100k actions:

| Report name | h reconstruction R | prediction P | constraints C |
|---|---:|---:|---:|
| flat | 0 | 0 | 1 |
| rec_only | 1 | 0 | 1 |
| pdyn_only | 0 | 1 | 1 |
| full | 1 | 1 | 1 |

`flat` means constraints-only auxiliary training, not DreamerV3. Full objective is

$$
\mathcal L=\mathcal L_{\mathrm{other}}+R\mathcal L_{\mathrm{rec}}+
\lambda_{\mathrm{pdyn}}(P\mathcal L_{\mathrm{prediction}}+0.1\mathcal L_{\mathrm{persist}}+0.01\mathcal L_{\mathrm{iso}}).
$$

All 16 conditions completed. Means over 20 final evaluation episodes:

| Game | Constraint only | + reconstruction | + prediction | Full |
|---|---:|---:|---:|---:|
| Boxing | 43.2 | 56.9 | 70.1 | **80.05** |
| Up N Down | **81872** | 6652 | 31592 | 13565 |
| Frostbite | 2387 | **3157** | 429.5 | 2038 |
| Road Runner | 14950 | 13025 | 6960 | **15790** |

Factorial effects with conditional episode-bootstrap 95% intervals:

| Game | Prediction effect without/with reconstruction | Reconstruction effect without/with prediction | Interaction |
|---|---|---|---:|
| Boxing | `+26.9 [18.0,35.65]` / `+23.15 [16.05,30.30]` | `+13.7 [3.75,23.70]` / `+9.95 [4.60,15.45]` | `-3.75 [-14.95,7.60]` |
| Up N Down | `-50280 [-68799.5,-32042.8]` / `+6913 [648.0,14622.1]` | `-75220 [-90548.7,-60821.9]` / `-18027 [-31066.5,-5606.0]` | `+57193 [37755.5,77034.7]` |
| Frostbite | `-1957.5 [-2142,-1767]` / `-1119 [-1578.5,-700]` | `+770 [473.5,1131]` / `+1608.5 [1260,1948.5]` | `+838.5 [343,1296]` |
| Road Runner | `-7990 [-8795,-7125]` / `+2765 [1870,3690]` | `-1925 [-2820,-1065]` / `+8830 [7945,9655]` | `+10755 [9510,11985]` |

The intervals only quantify evaluation-episode variation conditional on four trained seed-0 policies/game. Arms were independently trained; they do not provide training-seed confidence intervals.

Representation behavior also lacks one universal explanation:

* Boxing Full is best in control, but B1 R2@64 is negative in all arms and differences are tiny (`-.0542` to `-.0583`). Prediction raises B2 rank relative to constraints-only, consistent with preserved diversity, but not proof of useful semantics.
* Up N Down Full has the best B1 R2@16 `.4972` and R2@64 `.2233`, yet constraint-only has six times the Full return. Better source-feature prediction did not imply better control.
* Frostbite prediction-only has the best B1 R2@16 `.5707` and R2@64 `.3835`, but the worst return 429.5 and highest B1 displacement 1.332. Reconstruction-only gives the best return and best R2@64 `.3886`.
* Road Runner Full gives the best return and R2@16 `.3035`; prediction-only gives the best R2@64 `.1914` but low return. Full B2 rank is only 5.88 and CKA is highest `.685`, so its control gain does not establish block distinctness.
* Every arm has zero dead B1 coordinates, excluding exact coordinate collapse under that threshold. Effective-rank concentration and redundancy remain.

**Diagnosis: H1 is mixed, H2 is rejected as a universal account.** Interaction is positive in Up N Down, Frostbite and Road Runner, but absent/negative in Boxing. Prediction helps strongly in Boxing, hurts strongly in Frostbite, and changes sign with reconstruction in Up N Down/Road Runner. Reconstruction likewise helps Boxing/Frostbite but can hurt Up N Down/Road Runner. Full is best only in Boxing and narrowly in Road Runner; it is not uniformly optimal.

This is the strongest component-attribution result so far. It shows both auxiliaries can cause negative transfer and their effects are highly context-dependent. It does not identify whether the context is game dynamics, reward sparsity, representation gradient conflict, exploration, or one-seed training variation.

## 5. Hypothesis ledger

| ID | Hypothesis | Evidence status | Decision |
|---|---|---|---|
| H0 | Predictability alone guarantees stable semantic role | Rejected conceptually and empirically | Add direct diagnostics/constraints; never make this guarantee |
| H1 | Covariance whitening reduces pooled scale shortcut | Supported as mechanism/reference | Keep whitened persistence package for controlled research |
| H2 | Whitened persistence + isotropy establishes temporal hierarchy | Open | Control/diversity improved in Alien, hierarchy probes inconclusive |
| H3 | Causal averaging of first reconstruction target solves conflict | Rejected | Do not promote window 16 |
| H4 | Reset-crossing targets harm long-horizon prediction | Supported locally | Mask improves some R2; control/weight-matched effect unresolved |
| H5 | Rank-floor 32 avoids nuisance isotropy pressure | Rejected | Rank, CKA and control often worsen |
| H6 | CKA decorrelation creates complementary block 2 | Rejected | CKA falls by suppressing B2 rank/useful shared signal |
| H7 | Direct prediction from h makes projection/reconstruction unnecessary | Not supported | Mixed control; h coordinate prefixes lack learned ordering |
| H8 | Spatial pyramid directly grounds useful temporal hierarchy | Not supported | One-seed control gains do not match long-horizon probes |
| H9 | 4x4 is too coarse; 8x8 is reliable improvement | Rejected by seed 2 | Soft loses pyramid on all four replication games |
| H10 | PDyn merely copies or ignores action | Rejected on tested clips | Predictor beats copy and is action-sensitive |
| H11 | Reward failure mainly originates in imagination | Game-dependent | Supported in Frostbite/some Up N Down; Road Runner fails earlier |
| H12 | Actor underuses compact prefix | Supported as local reliance pattern | Equal-width B1 perturbations affect policy less; usefulness still open |
| H13 | Prediction helps mainly with reconstruction | Mixed | Positive interaction in 3/4, but sign/main effects vary strongly |
| H14 | Reconstruction is always necessary | Rejected | Constraints-only dominates Up N Down and is competitive elsewhere |
| H15 | Multi-stride prediction always improves control | Rejected | Helps Boxing, hurts Frostbite, context-dependent elsewhere |

## 6. What has actually improved the method?

The only retained core improvement is the **whitened persistence 0.1 + all-lag + isotropy 0.01 package**, because it addresses a real mathematical shortcut and showed encouraging Alien control/diversity evidence. Its semantic claim remains limited. None of the subsequent rank, CKA, temporal averaging, direct-h or spatial-target modifications has earned promotion.

The research infrastructure improved materially:

* reset-crossing prediction pairs are observable and maskable;
* rank, CKA, dead coordinates, displacement and long-horizon probes are measured on fixed splits;
* actual PDyn heads are tested against copy/action controls;
* posterior and same-RNG imagined reward errors are compared on matched frames;
* actor reliance is tested with equal-width interventions;
* prediction can now be disabled independently while constraints remain active;
* manifests preserve seed, action budget, source checkpoint and failure limitations.

These changes improve attribution, not the final method score by themselves.

## 7. Current failure model

The evidence no longer supports one global failure. The present model is:

1. **Auxiliary conflict is conditional.** Reconstruction and prediction can each help or hurt. Their interaction can rescue one another in some games, especially Road Runner, but does not do so universally.
2. **Prediction learns its own supervised signal, but that signal can be misaligned with control.** Frostbite prediction-only is the clearest counterexample: strong future probes coexist with poor return.
3. **The policy has no structural reason to use compact prefixes.** All control heads consume full $z$; later chunks often influence actor output more. A good B1 may remain incidental to control.
4. **World-model failures occur at different locations.** Frostbite exposes degradation through imagination; Road Runner reward error is already high on posterior data. One representation regularizer cannot be assumed to fix both.
5. **Current probes measure accessibility, not semantic identity.** Random projected $h$, image averages, pixel MAE and CKA can all improve while control worsens.
6. **Training variance is large.** Single-seed rankings reversed repeatedly. Future promotion requires independent from-scratch seeds.

## 8. Next hypotheses and improvement plan

The next cycle should locate negative transfer before changing representation losses.

### N1. Gradient conflict between auxiliary and task objectives

**Hypothesis.** Negative-transfer games exhibit larger opposing gradients from reconstruction/prediction versus reward/value/policy objectives in the shared projection $S_\psi$.

**Measurement before intervention.** On identical replay batches from completed latent factorial checkpoints, compute per-loss gradient norm and cosine for projection trunk and each output block. Separate reconstruction, prediction, persistence, isotropy, reward, continuation, value, replay-value and policy. Normalize neither by raw loss nor parameter count without reporting both.

**Expected evidence.** Frostbite prediction arms and Up N Down reconstruction arms show persistent negative cosine or excessive auxiliary/task norm ratios compared with helpful Boxing effects. If no relationship appears, reject gradient conflict as the primary explanation.

**Possible fix only after support.** Stop-gradient routing, adaptive gradient projection, or loss weighting for the identified conflicting component. Change one route/weight at a time and retain a fixed no-fix arm.

### N2. Task information lost in h→z versus present but unused

**Hypothesis.** Some failures arise because task information exists in $h$ but is lost in $z$; others arise because information exists in $z$ but actor/reward heads fail to exploit it.

**Measurement.** Fit equal-protocol held-out probes from $h$, B1, cumulative prefixes and full $z$ to immediate reward, nonzero-reward class, discounted return and terminal events. Use on-policy/coverage-aware datasets and matched train/validation/test episodes. Terminal trajectories must be collected because current clips have none. Then freeze representation and retrain matched lightweight heads.

**Expected evidence.** If h probe succeeds and z probe fails, change projection/reconstruction. If z probe succeeds but existing head/retrained head differ, address head optimization or policy use. If h already fails, focus on encoder/RSSM or data coverage.

### N3. Task-selective temporal prediction

**Hypothesis.** Uniform latent MSE rewards predictable nuisance dimensions. Prediction should prioritize future features carrying reward/value/continuation information rather than every coordinate equally.

**Prerequisite.** N2 must identify retained task information and N1 must rule in/out optimization conflict. Otherwise adding reward weighting risks amplifying sparse/noisy heads.

**Candidate fix.** Weight prediction residual directions by held-out task relevance or predict compact task sufficient statistics alongside latent targets. Keep action-shuffle/copy controls and test whether B1 gains task probes, control and stable rank together.

### N4. Policy access to hierarchy

**Hypothesis.** Full-z heads bypass compact-prefix specialization. Prefix dropout or horizon-conditioned access can force robust use of B1 while preserving full capacity when needed.

**Prerequisite and risk.** Equal-width reliance evidence supports investigating this, but forced prefix use can remove fast control signals. First evaluate frozen head retraining on prefixes. A training experiment should compare one access change, monitor entropy/reward coverage, and reject it if control declines despite higher B1 reliance.

### N5. Correct reset masking with matched optimization weight

**Hypothesis.** Episode-safe prediction improves representation when valid-pair normalization keeps effective level weight constant.

**Fix.** Normalize each level by valid count and broadcast a weight-matched loss rather than zeroing invalid pairs under a fixed outer mean. Then compare from scratch, one factor only. This is a correctness refinement, not a semantic constraint.

## 9. Immediate research decision

Do not promote spatial 8x8, rank floor, CKA decorrelation, causal averaging, or direct-h prediction. Do not remove reconstruction or prediction globally based on four games and one seed. Keep the 26-game constraint formulation as the reference checkpoint family while diagnosing conditional negative transfer.

The next computation should be a **read-only gradient/task-information audit on the completed latent factorial checkpoints**, followed by one targeted intervention selected from its failure pattern. This has higher diagnostic value than another broad loss sweep and respects the requirement to change one principal factor per causal experiment.

## 10. Artifact index

| Stage | Main artifact |
|---|---|
| Constraint mathematics | `CONSTRAINT_MATH_AND_INTUITION.md` |
| Whitening/Alien/coarse target/reset mask | `RESEARCH_PROTOCOL.md`, `latest_3331/`, `coarse_target_3332/`, `episode_mask_3333/` |
| 26-game suite | `production_runs/constraint_full26_seed0_OzirkNB8`, `full_vs_dreamerv3_constraint_suite_learning_curves/` |
| Rank floor | `FOUR_GAME_RANK_FLOOR.md`, `RANK32_DIAGNOSIS_AND_DECORRELATION.md` |
| Decorrelation | `DECORRELATION_DIAGNOSIS_AND_RANK_GUARD.md`, `production_runs/decorrelation_four_game_ro2vQg01/` |
| Direct h | `production_runs/direct_h_constraint_NpMOUZXh/` |
| Spatial seed 0/1/2 | `SPATIAL_RECONSTRUCTION.md`, `SPATIAL_SEED1_DIAGNOSIS.md`, `production_runs/spatial_seed2_UOW5Hr6o/` |
| Component audit | `COMPONENT_FAILURE_AUDIT.md`, `COMPONENT_AUDIT_RESULTS_3526.md`, `MATCHED_AUDIT_RESULTS_3668.md` |
| Latent constrained factorial | `CONSTRAINED_FACTORIAL_PROTOCOL.md`, `production_runs/constrained_factorial_NOBwTnNw/factorial_summary.json` |

The local `production_runs` paths contain raw checkpoints/logs and are not assumed to be present on GitHub. This Markdown contains the decision-relevant numbers; the machine-readable factorial summary remains at `production_runs/constrained_factorial_NOBwTnNw/factorial_summary.json` on the research server.
