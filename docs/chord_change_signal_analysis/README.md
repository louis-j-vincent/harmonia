# Chord-change signal analysis — feature table reference

**Goal:** decide what a real chord-change *detector* should learn from and
use, by finding where there's real (non-redundant) signal across 5
candidate categories: (A) harmonic-rhythm timing, (B) bass patterns, (C)
key/bigram patterns, (D) raw note/chroma patterns, (E) song structure.
This is exploratory/hypothesis-generating — nothing here is wired into
`harmonia/models/chord_hmm.py`.

## Data

`features.csv`, built by `scripts/build_chord_change_features.py`, one row
per beat across all 5 rendered POP909 songs (001-005), **1389 rows total**.
Validated by `scripts/validate_chord_change_features.py` (20/20 checks
passing as of 2026-07-04) — re-run that script if you regenerate the CSV.

**Load it with `pd.read_csv(..., dtype={"song_id": str})`** — song IDs are
all-digit strings ("001", "002", ...) and pandas silently coerces them to
int64 (stripping leading zeros) without the explicit dtype. This was caught
by the validation script; don't reintroduce it.

Ground truth (`chord_changed`, `gt_root`, `gt_label`) and the beat/downbeat
grid both come from **POP909's own annotation files** (`beat_midi.txt`,
`chord_midi.txt`), not our audio beat tracker — deliberate, since this
analysis is about timing and we want the least noisy available reference.
**Important consequence, verified across all 5 songs:** beat *counts* here
can differ from other scripts in this repo that use the audio-derived
(librosa) beat grid instead. Checked directly (POP909 tempo vs our
librosa-derived tempo): 001 90.0 vs 89.1 BPM, 003 82.0 vs 80.7, 004 71.5 vs
71.8, 005 64.9 vs 64.6 — all close. **Song 002 is the one exception**: 63.0
vs 129.2 BPM, a genuine tempo-*octave* error in our librosa beat tracker
(129.2/63.0 = 2.05x), not a general property of this table or a
unit-conversion bug. This is the same kind of tempo-octave instability
already flagged in `docs/known_issues.md`'s soundfont section (a different
song's beat count shifted ~2x between two renders there) — now confirmed
directly against POP909's own ground truth and isolated to song 002
specifically. Anything measured "in beats" for song 002 is only comparable
within this table (which uses POP909's grid throughout), not directly
against `plot_structure_proposal_illustrations.py` or
`plot_chord_change_correlates.py`'s own numbers, unless rescaled by ~2x.

## Columns

| column | type | meaning |
|---|---|---|
| `song_id`, `beat_idx`, `time_s` | id | which song/beat |
| `chord_changed` | bool | **the label** — did the GT chord change at this beat vs the previous one |
| `gt_root`, `gt_label` | int/str | GT chord root pitch class (-2 = no GT coverage, -1 = N/no-chord) and full label, for reference |
| `A_beat_phase` | int | beats since the most recent downbeat (0 = downbeat itself); -1 before the first downbeat; rarely >3 (0.29% of rows) where POP909's own annotation has a longer-than-4/4 gap between downbeats — real data, not a bug |
| `A_beats_since_change` | int | how many consecutive prior beats had the same GT chord, as of just before this beat (0 on the beat right after a change) |
| `B_bass_changed` | bool | did the inferred bass note's pitch class change vs the previous beat (`bass_track.py::infer_bass_track_learned`, forward-filled) |
| `B_bass_onset` | bool | was there a fresh onset-detected bass note this beat (vs a carried-forward/held one) |
| `B_bass_is_root_or_fifth` | bool | does the inferred bass note's pitch class equal the GT chord's root or fifth |
| `C_bigram_logprob_atomic` | float, NaN unless `chord_changed` | log P(this chord's canonical degree+quality \| previous chord's), from the **mode-agnostic pooled** bigram table fit across all 909 POP909 songs using each song's own best-fit parent scale (see `scripts/scale_taxonomy.py`, `docs/scale_taxonomy_2026-07-03.md`) |
| `C_bigram_logprob_mode` | float, NaN unless `chord_changed` | same, but from the bigram table specific to this song's own annotated major/minor mode (canonicalised to the relative-major tonic) |
| `C_bigram_mode_delta` | float, NaN unless `chord_changed` | `C_bigram_logprob_mode - C_bigram_logprob_atomic` — positive means the mode-specific table thought this transition was more likely than the mode-agnostic pooled one; near zero means mode-awareness doesn't matter for this transition |
| `D_onset_density` | float >= 0 | total onset activation across all 88 piano keys this beat (raw attack energy) |
| `D_chroma_cosine_dist` | float in [0,2] | 1 - cosine similarity between this beat's and the previous beat's raw (onset-based) chroma vector — a local, unsmoothed "how different does this beat sound" signal |
| `E_dist_to_segment_boundary` | int >= 0 | beats to the nearest structural-segment boundary (`harmonia.models.structure.Segmenter`, SSM+checkerboard-novelty based) |
| `E_position_in_loop` | int | `(beat_idx - E_loop_phase) mod E_detected_period` — position within the detected repeating structural loop, phase-anchored to the first annotated downbeat (`harmonia.models.periodicity.find_loop_phase`) so position 0 lines up with a real downbeat rather than beat 0 of the song. **Was a real bug before 2026-07-04**: it used `beat_idx mod period` directly, which left loop-start and downbeat as completely disjoint sets in 2/5 songs — see `docs/known_issues.md` #1 and `findings_AE_DE.md`. |
| `E_detected_period` | int | the song's detected loop length in beats (`harmonia.models.periodicity.score_periods`, top-1 candidate); 0 if none detected |
| `E_loop_phase` | int | the phase offset used to compute `E_position_in_loop` (see above); 0 if no downbeat was found to anchor to |

## What "redundant" vs "complementary" means here

For a pair of metrics from different categories, we care about the JOINT
relationship, not just each one's individual correlation with
`chord_changed` (already partially known from earlier sessions — see
`docs/chord_change_signal_analysis/PRIOR_FINDINGS.md`):

- **Redundant**: metric Y is mostly just a proxy for metric X (e.g. if
  `B_bass_changed` is true almost only when `A_beat_phase==0`, then knowing
  the bass changed doesn't add information beyond knowing it's a downbeat).
  Evidence: strong correlation/dependence between X and Y themselves,
  regardless of the chord-change label.
- **Complementary**: X and Y are roughly independent of each other, but
  each still correlates with `chord_changed` on its own (or, better, their
  COMBINATION predicts `chord_changed` better than either alone — e.g. "bass
  changed AND it's a downbeat" is a much stronger signal than either fact
  alone). This is what you want to find for a joint/Bayesian model — signals
  worth combining, not signals that duplicate each other.

For a continuous-vs-binary pair, use point-biserial correlation (=Pearson
correlation directly). For continuous-vs-continuous, Pearson/Spearman. For
categorical-vs-categorical (e.g. `A_beat_phase` vs a binned other variable),
a contingency table + Cramér's V, or just grouped-mean plots. Always report
both the raw joint-distribution PLOT and a summary statistic — the plot is
usually more informative than the single number for catching non-linear
relationships.

## Prior findings this connects to

See `docs/chord_change_signal_analysis/PRIOR_FINDINGS.md` for the individual
(non-joint) correlations already established in earlier sessions, so new
analysis builds on top of them rather than re-deriving from scratch.
