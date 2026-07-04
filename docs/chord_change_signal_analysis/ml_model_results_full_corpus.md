# Chord-change classifier: full 909-song corpus (2026-07-04)

10-fold GroupKFold cross-validation (grouped by song, so no song's beats leak across train/test) on `features_symbolic.csv` -- 303894 rows, 909 songs, fully symbolic (no audio, no Basic Pitch; bass from POP909's own PIANO-track ground truth, chroma/onset from a MIDI note-onset piano-roll). Same feature set and leakage exclusions as the 5-song audio run (`ml_model_results.md`) -- see that file's header for what's excluded and why.

| Model | Accuracy | F1 | AUC | Brier | LogLoss |
|---|---|---|---|---|---|
| Majority class | 0.646 | 0.000 | 0.490 | 0.229 | 0.650 |
| Heuristic (phase==0 or bass_changed) | 0.646 | 0.549 | 0.637 | 0.354 | 2.447 |
| Logistic (main effects) | 0.693 | 0.491 | 0.727 | 0.196 | 0.576 |
| Logistic (+ phase x bass interactions) | 0.700 | 0.525 | 0.732 | 0.195 | 0.573 |
| Decision Tree (depth=6) | 0.819 | 0.704 | 0.870 | 0.130 | 0.413 |
| Random Forest (200 trees) | 0.825 | 0.717 | 0.875 | 0.130 | 0.415 |
| Gradient Boosting | 0.826 | 0.723 | 0.873 | 0.127 | 0.410 |
| PyTorch MLP (32,16) on M4 GPU | 0.824 | 0.722 | 0.877 | 0.127 | 0.405 |

## Feature importances (RF / GB, fit on all songs, descriptive)

| feature | rf_importance | gb_importance |
|---|---|---|
| beats_since_change | 0.501 | 0.571 |
| bass_changed | 0.166 | 0.194 |
| phase_1 | 0.078 | 0.141 |
| chroma_dist | 0.073 | 0.029 |
| phase_3 | 0.049 | 0.018 |
| onset_density | 0.043 | 0.014 |
| phase_0 | 0.027 | 0.005 |
| phase_2 | 0.024 | 0.010 |
| dist_to_boundary | 0.014 | 0.008 |
| bass_onset | 0.013 | 0.004 |
| loop_position_frac | 0.007 | 0.000 |
| phase_4 | 0.003 | 0.005 |
