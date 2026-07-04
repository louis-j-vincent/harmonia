# Chord-change signal analysis — master summary (2026-07-04, updated after periodicity-bug fix)

**Question:** what should a real chord-change detector learn from and use —
harmonic-rhythm timing (A), bass patterns (B), key/bigram patterns (C), raw
note/chroma patterns (D), or song structure (E) — and where do pairs of
these carry genuinely complementary information vs redundant/overlapping
information?

**Method:** built a shared, validated per-beat feature table
(`features.csv`, 1389 rows, 5 rendered POP909 songs, 10 metrics across the
5 categories — see `README.md`), passed 22/22 garde-fou sanity checks
(`scripts/validate_chord_change_features.py`, which caught 2 real bugs
along the way — see "Process notes" below), then delegated 6 of the
highest-priority cross-category pairs (out of 10 possible) to 3 parallel
subagents for joint-distribution analysis. Full detail in
`findings_A_vs_BD.md`, `findings_B_vs_DC.md`, `findings_AE_DE.md`.
`PRIOR_FINDINGS.md` has the individual (non-joint) correlations from
earlier sessions that this builds on. One of the joint analyses (A×E)
surfaced a real bug in production code (`harmonia/models/periodicity.py`),
which was fixed and the affected analysis rerun — see "structural
discovery" section below.

**Caveat that applies to everything below:** n=5 songs. Every pair analysis
found real heterogeneity across the 5 songs, sometimes reversing the pooled
pattern. Read all verdicts as "true of this small sample, worth checking on
more data," not as settled corpus-wide facts.

## Results at a glance

| pair | redundant? | complementary (adds joint predictive power)? | confidence |
|---|---|---|---|
| **A (phase) × B (bass changed)** | No (Cramér's V=0.13, weak) | **Yes — the clearest positive result** | Moderate |
| A (phase) × D (onset density, chroma novelty) | Partial overlap (η²=0.014, 0.038 — small) | Some, but modest | Moderate |
| **B (bass changed) × D (chroma novelty)** | No (r=0.43, real but not a subset) | **Yes — combining lifts P(change) from 0.43 to 0.58** | Moderate |
| B (bass changed) × D (onset density) | No — but also unrelated (d=0.09, ~independent) | No | Moderate |
| B (bass root/fifth) × C (bigram log-prob) | — | Inconclusive (sign flips across songs, small n) | Low |
| A (phase) × E (loop position) — **corrected** | No, but now (mostly) nested (4/5 songs a clean subset, was 1/5) after fixing the periodicity phase bug | **No — settled negative**, not just inconclusive: 4/5 songs show equal-or-lower P(change) at loop-start | Moderate (up from Low — this is now a fair comparison, see below) |
| D (chroma/onset) × E (segment boundary) | No (correlation ≈ 0) | Not demonstrated | Low-moderate |

**Update, same day:** the A×E pair below originally surfaced a real bug in
`harmonia/models/periodicity.py` (see "structural discovery" section) —
fixing it and rerunning changed the verdict from "inconclusive" to a
settled "no." Every other row in this table is unaffected: regenerating
`features.csv` with the fix changed only the `E_position_in_loop` column,
verified directly against the pre-fix file.

## The single most useful finding

**Beat phase and bass-pitch-class-change combine to sharpen the estimate
well beyond either alone.** The two signals are only weakly coupled to each
other (bass changes 60% of the time on a downbeat vs 44-49% elsewhere — a
real but modest dependence), which is exactly the profile you want for two
features worth combining rather than picking one:

- Best single signal: `phase==0` alone → P(chord changed) = 0.672.
- Best combination: `phase==2 AND bass_changed` → P(chord changed) =
  **0.775** — higher than either signal reaches alone, anywhere.
- Worst combination: `phase==3 AND bass unchanged` → P(chord changed) =
  **0.098** — lower than either signal's floor alone.
- Within every phase bucket, `bass_changed=True` raises P(chord changed)
  relative to `bass_changed=False` — it's adding information at every
  phase, not just echoing the phase signal.

See `plots/pair_A_phase_vs_B_bass_chordchange_grid.png`. Caveat: song 003's
phase-2 beats are 0% chord-change regardless of bass state, so the pooled
0.775 figure blends songs with different metrical behavior — read the
*direction* of the effect as solid, the exact magnitude as this-sample-only.

**Bass change and chroma novelty are correlated (r=0.43) but not
redundant** — 57% distributional overlap remains, and combining
`bass_changed AND chroma_high` lifts P(chord changed) from 0.43 to 0.58.
**Bass change and onset density are essentially unrelated** (d=0.09) — a
genuinely separate, independent signal, for better or worse.

## An unexpected, more structural discovery — found AND fixed

Testing `A_beat_phase × E_position_in_loop` jointly (rather than trusting
each in isolation) surfaced a real bug in `harmonia/models/periodicity.py`,
not just a data finding: **`score_periods()` only ever detects the period
*length*, never the phase offset** — verified directly, its scoring is
`np.diagonal(ssm, offset=L).mean()`, which averages over every possible
starting position simultaneously, by construction. There was no code
anywhere that solved for "which beat is the true start of a repeat."
Consequence: `E_position_in_loop = beat_idx % period` used beat 0 of the
song as an arbitrary phase reference, with no guarantee it coincided with
a real repeat boundary — explaining why loop-start and downbeat were
**completely non-overlapping sets in 2 of 5 songs** (003, 004).

**Fixed the same day.** Added `find_loop_phase(period, is_downbeat)`,
which anchors phase 0 to the first annotated downbeat instead of beat 0 of
the song (`harmonia/models/periodicity.py`, tested in
`tests/test_periodicity.py`, wired into
`scripts/build_chord_change_features.py`). An SSM-self-similarity-based
approach was considered first and rejected on inspection: a cleanly
repeating signal is equally self-similar under *any* phase choice by
construction, so the SSM alone can never break that symmetry — only
external timing information (the downbeat grid) can. Verified the fix
directly changes only what it should: `features.csv` was regenerated and
every column except `E_position_in_loop` (and the new `E_loop_phase`) is
byte-identical to the pre-fix file.

**Measured effect:** pooled loop-start/downbeat overlap went from a
39%-ish pooled rate (1/5 songs a clean subset, 2/5 fully disjoint) to
91.5% (4/5 clean subsets, the fifth at 93.8%). Songs 003 and 004 — the
two that were fully disjoint — picked up non-zero phase corrections
(phase=2) exactly as predicted. Re-running the joint analysis this bug
surfaced changed the *conclusion*: the original "positive lift"
(63.8%→89.4%) was a Simpson's-paradox artifact of comparing sets that
didn't actually overlap; with the sets properly nested, 4 of 5 songs now
show equal-or-lower P(chord changed) at loop-start vs. other downbeats —
a settled "no combination benefit" rather than "inconclusive, needs more
data." Full corrected write-up: `findings_AE_DE.md`; bug + fix recorded
in `docs/known_issues.md`.

## What still needs a bigger sample

- `B_bass_is_root_or_fifth × C_bigram_logprob_atomic`: direction of the
  (small) effect flips sign across songs; n≈600 split into groups of
  200-390 isn't enough to resolve. Would need the full 909-song symbolic
  corpus's worth of real audio (not available) or many more rendered songs.
- `D × E` (chroma/onset vs segment-boundary distance): correlation is
  essentially zero (ρ≈0.06-0.08) and the direction flips between songs
  (003 spikes at boundaries, 005 troughs there) — genuinely inconclusive,
  not just underpowered.
- The 4 lower-priority pairs not analyzed this round: A×C, C×D, C×E, B×E.

## Follow-up work (2026-07-04, same day)

- **A first supervised model was built and compared across architectures**
  (logistic regression, decision tree, random forest, gradient boosting,
  small MLP) predicting `chord_changed` from the non-leaking features in
  this table, evaluated with leave-one-song-out CV (the right protocol
  given 5 heterogeneous songs). Gradient boosting won (AUC 0.831, pooled),
  clearly ahead of logistic regression (0.722-0.726) — the phase×bass
  interaction this investigation found is real, but a linear model with
  that interaction manually added still can't match what tree ensembles
  find on their own. **The single most important feature by a wide margin
  was `A_beats_since_change`** (41-52% of RF/GB importance) — a duration/
  run-length signal that, notably, was never actually tested in any of the
  6 pairwise analyses above. Full results: `ml_model_results.md`, plot:
  `plots/ml_model_comparison.png`. Song 003 remains the hardest
  out-of-sample case for every architecture (AUC as low as 0.36 for
  logistic regression) — consistent with its consistently anomalous
  behavior throughout this whole investigation.
- **A separate design pass** produced a concrete proposal for
  multi-hypothesis structure detection (validated via cross-repeat
  harmonic agreement against the full 909-song symbolic corpus, not just
  these 5 audio songs) plus trigram/timing priors, directly responding to
  why category E looked unhelpful here and how category C could be
  revisited properly. See `docs/structure_trigram_design_2026-07-04.md`.
- **The design's proposed cheapest first experiment was run**: Cross-Repeat
  Harmonic Agreement (CRHA) across all 909 songs, fully symbolic, no audio
  (`scripts/run_structure_validation.py`). Result: **23.4% of songs (213/909)**
  have a structure hypothesis whose cross-repeat chord agreement clears a
  shuffled-label null by a real margin — a genuine but minority phenomenon,
  not a corpus-wide property, and short periods (1-2 bars) are
  disproportionately false positives (accompaniment repeating, not
  harmony) vs. long periods (4-8 bar phrases) among the songs that do pass.
  Checking where songs 001-005 land produced a sharper, more important
  finding than expected: **song 001 (Candidate C's worst regression) has
  the highest CRHA margin of all 5 songs** — its harmonic repetition is
  real and correctly validated, which means Candidate C's failure there
  was about audio-evidence surface variation, not a bad structure
  hypothesis. Full results: `structure_validation_results.csv`, plot:
  `plots/crha_structure_validation.png`, write-up:
  `docs/structure_trigram_design_2026-07-04.md`'s "Results" section.
- **The ML classifier comparison was rerun on the full 909-song corpus**
  (fully symbolic, `features_symbolic.csv`, `scripts/build_symbolic_features.py`
  / `scripts/train_chord_change_classifier_full.py`, 10-fold GroupKFold by
  song instead of LOSO). All tree/neural architectures converged tightly
  around AUC 0.87-0.88 (up from 0.79-0.83 at n=5), and a PyTorch MLP
  trained on the M4 GPU edged out gradient boosting for the first time
  (AUC 0.877 vs 0.873) — the extra data let the neural net's smoother
  interactions catch up to and pass tree ensembles. The bigger correction:
  **`bass_changed`'s true feature importance was hidden at n=5** (4%
  there vs 17-19% at full scale) — confirming the earlier pairwise finding
  was real and that the 5-song multivariate importance ranking shouldn't
  be trusted on its own. `loop_position_frac`/`dist_to_boundary` stayed
  negligible at both scales, reinforcing the structure-doesn't-help
  finding with far more statistical power. Full results:
  `ml_model_results_full_corpus.md`, comparison plot:
  `plots/ml_5song_vs_909song_comparison.png`.

## Process notes (for anyone re-running this)

- `scripts/build_chord_change_features.py` builds the table;
  `scripts/validate_chord_change_features.py` must pass all checks (22/22
  as of this update) before trusting any analysis on top of it. It caught
  two real bugs on first run: pandas silently coercing all-digit `song_id`
  strings to int64, and a buggy `groupby().apply().shift(-1)` check in the
  validator itself (the underlying feature was already correct — the check
  was wrong, a useful reminder that a garde-fou needs its own
  sanity-checking too). Two more checks were added after the periodicity
  phase-offset fix specifically to guard against that bug regressing.
- One subagent (`findings_B_vs_DC.md`'s analysis) hit a tool restriction
  preventing it from writing its own `.md` report file; its numeric
  results were relayed in its final response and the file was written
  manually from that content. No data or analysis was lost, just the
  file-creation step.
- Nothing in this investigation is wired into `harmonia/models/chord_hmm.py`
  or any pipeline code — purely exploratory. No git commits were made.
