# Chord-change classifier: architecture comparison (2026-07-04)

Leave-one-song-out cross-validation (5 folds, one song held out each time) on `features.csv`. Features: beat phase (one-hot), beats since last change, bass pitch-class change/onset, onset density, chroma novelty, distance to segment boundary, loop-position fraction. Excludes anything that leaks the ground-truth chord identity (`gt_root`, `B_bass_is_root_or_fifth`, `C_bigram_*` -- see script docstring for why).

| Model | Accuracy | F1 | AUC | Brier | LogLoss |
|---|---|---|---|---|---|
| Majority class | 0.570 | 0.000 | 0.407 | 0.250 | 0.693 |
| Heuristic (phase==0 or bass_changed) | 0.623 | 0.632 | 0.640 | 0.377 | 2.602 |
| Logistic (main effects) | 0.675 | 0.614 | 0.722 | 0.224 | 0.682 |
| Logistic (+ phase x bass interactions) | 0.683 | 0.613 | 0.726 | 0.223 | 0.681 |
| Decision Tree (depth=4) | 0.724 | 0.709 | 0.789 | 0.210 | 0.723 |
| Random Forest (200 trees) | 0.754 | 0.727 | 0.805 | 0.175 | 0.538 |
| Gradient Boosting | 0.773 | 0.746 | 0.831 | 0.170 | 0.567 |
| MLP (16, 8) | 0.710 | 0.672 | 0.769 | 0.238 | 1.059 |

## Per-song breakdown (best model)

Best model by pooled AUC: **Gradient Boosting**

| song | n | accuracy | f1 | auc | brier |
|---|---|---|---|---|---|
| 001 | 292 | 0.777 | 0.747 | 0.832 | 0.193 |
| 002 | 242 | 0.876 | 0.868 | 0.946 | 0.088 |
| 003 | 313 | 0.585 | 0.532 | 0.717 | 0.298 |
| 004 | 238 | 0.794 | 0.754 | 0.882 | 0.142 |
| 005 | 304 | 0.865 | 0.852 | 0.928 | 0.105 |

## Feature importances (RF / GB, fit on all 5 songs, descriptive)

| feature | rf_importance | gb_importance |
|---|---|---|
| beats_since_change | 0.411 | 0.518 |
| phase_3 | 0.125 | 0.119 |
| chroma_dist | 0.121 | 0.119 |
| onset_density | 0.076 | 0.089 |
| loop_position_frac | 0.061 | 0.059 |
| phase_0 | 0.058 | 0.027 |
| dist_to_boundary | 0.041 | 0.023 |
| bass_changed | 0.040 | 0.017 |
| phase_2 | 0.033 | 0.014 |
| phase_1 | 0.024 | 0.015 |
| bass_onset | 0.009 | 0.000 |
| phase_4 | 0.001 | 0.000 |
