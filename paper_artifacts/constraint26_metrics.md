# Constraint Suite: 26-Game Metric Report

## Scope and provenance

This report is rendered directly from the completed run at `/home/vn-user0101/Dat/HTS-Dreamer/production_runs/constraint_full26_seed0_OzirkNB8`. The run contains 26 Atari-100k games, seed 0, 110,000 exact environment actions per game, with progressive reconstruction and multi-stride prefix dynamics enabled. The first-prefix constraint package is whitened persistence (weight 0.1, all lags) plus isotropy (weight 0.01). It does **not** establish a comparison against an unconstrained control: results here describe this one condition only.

Every game finished successfully. The macro rows are unweighted means over games; they are not normalized Atari scores and should not be used to rank algorithms. `final_eval.json` records `status: not_run`, so the control values below are training episode returns, not isolated evaluation returns.

## What each metric means

| Family | Logged metric | Interpretation | Caveat |
| --- | --- | --- | --- |
| Control | `episode_score` | Environment return from a completed training episode. | Policy is still training; episodes within one seed are not independent training replicates. |
| Multi-stride prediction | `train/htp/pdyn_prediction_only`, `train/htp_pdyn_delta{1,2,4,8,16}` | Total prediction-only loss and its per-stride components. Lower is only an optimization diagnostic, not evidence of semantic stability. | Targets are learned slow features. |
| Temporal constraint | `train/htp/prefix_persistence` | Whitened first-prefix temporal displacement, averaged across lags 1--16, before its 0.1 loss weight. | Lower can also discard dynamic information. |
| Anti-collapse | `train/htp/prefix_isotropy` | Deviation of first-prefix covariance from isotropy. | It is not effective rank or a semantic diversity measurement. |
| Reconstruction | `train/htp/raw_rec_loss`, `train/htp_rec_l0`, `train/htp_rec_l4` | Progressive reconstruction loss and the first/final prefix-level errors. | Reconstruction fidelity does not prove long-horizon predictiveness. |
| Boundary audit | `train/htp/cross_episode_delta16` | Fraction of candidate stride-16 pairs crossing episode resets. | In this run boundary masking was disabled; this reports exposure, not a correction. |

The following diagnostics were requested but were **not collected for all 26 checkpoints**: fixed-trajectory future-target ridge $R^2$, per-block effective rank, dead-coordinate count, whitened change per block, block-1/block-2 CKA, and isolated evaluation return with confidence intervals. They exist only for later four-game diagnostic jobs, and are deliberately not imputed here.

## Control and training completion

`90--100k` and `100--110k` are means over complete training episodes whose exact `agent_actions` fall in each interval. `n` is the number of episodes, so a dash means no episode completed in that window.

| Game | 90--100k score | n | 100--110k score | n | updates | wall min |
| --- | --- | --- | --- | --- | --- | --- |
| alien | 748.46 | 13 | 590.00 | 13 | 27,261 | 73.0 |
| amidar | 115.15 | 13 | 143.92 | 12 | 26,954 | 71.7 |
| assault | 605.77 | 13 | 635.73 | 11 | 26,449 | 72.6 |
| asterix | 556.82 | 22 | 694.12 | 17 | 26,794 | 72.2 |
| bank_heist | 36.25 | 16 | 42.94 | 17 | 26,524 | 70.9 |
| battle_zone | 5833.33 | 6 | 9500.00 | 6 | 27,126 | 71.2 |
| boxing | 50.00 | 11 | 55.33 | 12 | 26,357 | 72.5 |
| breakout | 5.88 | 25 | 6.31 | 26 | 26,813 | 72.2 |
| chopper_command | 865.22 | 23 | 860.87 | 23 | 27,282 | 70.9 |
| crazy_climber | 53100.00 | 4 | 82800.00 | 2 | 27,166 | 71.0 |
| demon_attack | 209.58 | 12 | 145.00 | 15 | 26,802 | 72.1 |
| freeway | 0.00 | 4 | 0.00 | 5 | 26,628 | 72.6 |
| frostbite | 2484.00 | 10 | 2951.11 | 9 | 27,180 | 71.1 |
| gopher | 1860.00 | 7 | 2672.00 | 5 | 26,698 | 70.3 |
| hero | 7509.67 | 15 | 7518.50 | 10 | 26,557 | 72.8 |
| jamesbond | 176.67 | 15 | 276.92 | 13 | 26,769 | 72.3 |
| kangaroo | 1141.18 | 17 | 1187.50 | 16 | 27,167 | 71.2 |
| krull | 9015.00 | 6 | 8571.25 | 8 | 27,137 | 71.2 |
| kung_fu_master | 24066.67 | 3 | 19675.00 | 4 | 26,706 | 72.3 |
| ms_pacman | 1382.14 | 14 | 1293.85 | 13 | 26,729 | 72.4 |
| pong | -9.33 | 3 | -4.67 | 3 | 27,155 | 71.1 |
| private_eye | -263.25 | 4 | -276.67 | 3 | 27,153 | 71.1 |
| qbert | 839.06 | 16 | 908.33 | 15 | 26,810 | 72.1 |
| road_runner | 16930.00 | 10 | 16160.00 | 10 | 26,681 | 72.5 |
| seaquest | 420.00 | 9 | 395.56 | 9 | 26,870 | 41.4 |
| up_n_down | 3879.00 | 10 | 12613.33 | 3 | 26,671 | 41.6 |
| Macro mean | 5059.89 | — | 6516.01 | — | 26,863 | 69.5 |

## Final logged HTP metrics by game

Each value is the final logged aggregate at or before 110,000 environment actions. `cross16` is a fraction; all remaining columns are losses or diagnostics on their native scale.

| Game | pdyn | persist | isotropy | raw rec | d1 | d2 | d4 | d8 | d16 | cross16 | rec l0 | rec l4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| alien | 0.0638 | 0.8320 | 1.2378 | 0.0158 | 0.0754 | 0.0779 | 0.0779 | 0.0764 | 0.0385 | 0.0263 | 0.0169 | 0.0150 |
| amidar | 0.0472 | 0.8370 | 1.3000 | 0.0151 | 0.0413 | 0.0467 | 0.0564 | 0.0624 | 0.0564 | 0.0236 | 0.0171 | 0.0136 |
| assault | 0.0677 | 0.8372 | 0.8047 | 0.0120 | 0.0426 | 0.0605 | 0.0838 | 0.0997 | 0.0956 | 0.0197 | 0.0126 | 0.0113 |
| asterix | 0.0574 | 0.8421 | 1.0174 | 0.0176 | 0.0490 | 0.0561 | 0.0688 | 0.0791 | 0.0668 | 0.0364 | 0.0194 | 0.0159 |
| bank_heist | 0.0658 | 0.8607 | 1.9203 | 0.0128 | 0.0728 | 0.0805 | 0.0858 | 0.0791 | 0.0391 | 0.0260 | 0.0135 | 0.0129 |
| battle_zone | 0.0813 | 0.8755 | 0.7656 | 0.0109 | 0.0822 | 0.0985 | 0.1117 | 0.1104 | 0.0375 | 0.0093 | 0.0111 | 0.0109 |
| boxing | 0.0882 | 0.8768 | 0.5906 | 0.0110 | 0.0720 | 0.0913 | 0.1128 | 0.1230 | 0.0899 | 0.0104 | 0.0113 | 0.0108 |
| breakout | 0.0693 | 0.8522 | 1.1866 | 0.0191 | 0.0665 | 0.0731 | 0.0830 | 0.0912 | 0.0697 | 0.0484 | 0.0212 | 0.0179 |
| chopper_command | 0.0828 | 0.8767 | 1.5511 | 0.0201 | 0.1164 | 0.1095 | 0.0986 | 0.0848 | 0.0349 | 0.0313 | 0.0204 | 0.0201 |
| crazy_climber | 0.0431 | 0.8224 | 0.9010 | 0.0123 | 0.0354 | 0.0411 | 0.0483 | 0.0571 | 0.0603 | 0.0064 | 0.0135 | 0.0114 |
| demon_attack | 0.0818 | 0.8397 | 0.8158 | 0.0136 | 0.0579 | 0.0728 | 0.0988 | 0.1193 | 0.1116 | 0.0187 | 0.0147 | 0.0126 |
| freeway | 0.0061 | 0.7585 | 0.4403 | 0.0030 | 0.0043 | 0.0049 | 0.0058 | 0.0079 | 0.0124 | 0.0073 | 0.0040 | 0.0026 |
| frostbite | 0.0397 | 0.8165 | 1.0398 | 0.0141 | 0.0339 | 0.0379 | 0.0451 | 0.0528 | 0.0530 | 0.0263 | 0.0154 | 0.0132 |
| gopher | 0.0571 | 0.8470 | 0.7026 | 0.0133 | 0.0482 | 0.0571 | 0.0688 | 0.0777 | 0.0667 | 0.0130 | 0.0140 | 0.0127 |
| hero | 0.0362 | 0.8508 | 1.6543 | 0.0143 | 0.0332 | 0.0374 | 0.0425 | 0.0470 | 0.0411 | 0.0240 | 0.0154 | 0.0137 |
| jamesbond | 0.0455 | 0.8266 | 0.9571 | 0.0140 | 0.0449 | 0.0469 | 0.0531 | 0.0584 | 0.0488 | 0.0299 | 0.0149 | 0.0133 |
| kangaroo | 0.0514 | 0.8465 | 1.1255 | 0.0170 | 0.0551 | 0.0574 | 0.0612 | 0.0635 | 0.0454 | 0.0321 | 0.0176 | 0.0165 |
| krull | 0.0499 | 0.8441 | 0.6892 | 0.0130 | 0.0419 | 0.0493 | 0.0582 | 0.0656 | 0.0644 | 0.0121 | 0.0140 | 0.0121 |
| kung_fu_master | 0.0677 | 0.8446 | 0.6715 | 0.0109 | 0.0526 | 0.0654 | 0.0804 | 0.0972 | 0.0834 | 0.0073 | 0.0114 | 0.0104 |
| ms_pacman | 0.0461 | 0.8339 | 1.3950 | 0.0154 | 0.0552 | 0.0541 | 0.0515 | 0.0517 | 0.0397 | 0.0243 | 0.0170 | 0.0143 |
| pong | 0.0611 | 0.8493 | 1.1336 | 0.0202 | 0.0495 | 0.0582 | 0.0741 | 0.0881 | 0.0712 | 0.0116 | 0.0225 | 0.0188 |
| private_eye | 0.0454 | 0.8969 | 1.0225 | 0.0107 | 0.0427 | 0.0517 | 0.0600 | 0.0640 | 0.0298 | 0.0062 | 0.0108 | 0.0109 |
| qbert | 0.0604 | 0.8283 | 1.2657 | 0.0166 | 0.0573 | 0.0634 | 0.0703 | 0.0767 | 0.0675 | 0.0367 | 0.0180 | 0.0155 |
| road_runner | 0.0592 | 0.8598 | 0.6581 | 0.0115 | 0.0452 | 0.0573 | 0.0688 | 0.0819 | 0.0792 | 0.0208 | 0.0121 | 0.0110 |
| seaquest | 0.0566 | 0.8175 | 0.9226 | 0.0137 | 0.0494 | 0.0556 | 0.0668 | 0.0772 | 0.0665 | 0.0147 | 0.0146 | 0.0128 |
| up_n_down | 0.0684 | 0.8368 | 0.9236 | 0.0136 | 0.0487 | 0.0623 | 0.0785 | 0.0970 | 0.0997 | 0.0141 | 0.0149 | 0.0125 |
| Macro mean | 0.0577 | 0.8427 | 1.0266 | 0.0139 | 0.0528 | 0.0603 | 0.0697 | 0.0765 | 0.0604 | 0.0206 | 0.0149 | 0.0132 |

## Interpretation boundaries

- The suite confirms that both reconstruction and multi-stride branches executed and logged throughout all 26 games, but it does not identify which component caused a control change.
- A lower prediction loss or persistence score alone cannot show that a compact prefix carries invariant semantic content; it may reflect scale, smoothing, or information removal.
- The appropriate evidence for that causal claim remains the paired ablations and fixed-trajectory representation diagnostics described above, ideally repeated across independent training seeds.
