# Success metric — LOCKED BEFORE any variant was run (2026-07-30)

Corpus: the 50 **canonical** baked charts under `docs/plots/inferred_*.html`.
(60 files exist; 10 are experiment re-bakes of a song that already has a base
chart — `_npattern`, `_readable`, `_loopdemo`, `_missedchords`, `_nfix`,
`_bestfit`, `_phone`, `_barlocked*`, and one rename. Some are STALE: chiquitita's
`_npattern` still carries a 225-bar pre-octave-fix grid where the base has 58.
Keeping them would double-weight those songs and pollute the mean.)

Input is the `const P = {...}` payload baked into each chart — already
post-regrid, so no audio and no server is needed. Verified: it reproduces the
committed Don't Know Why baseline `A×3 B A×2 C×2 A×2 C×2 A×4 B×2 D` exactly.

## The five numbers, per chart

| # | name | definition | direction |
|---|---|---|---|
| M1 | `n_written` | distinct sections the chart WRITES (after `_group_to_min_bars` + `_fold_units` + `_collapse_endings`) | lower, floor ≈ 2 |
| M2 | `len_honest` | share of vocabulary sections whose bar span is an exact whole number of its own loop (`(bar1-bar0) % d_bars == 0`) | 1.0 |
| M3 | `sane` | ≤5 distinct letters AND the dominant letter covers ≥30% of bars AND recurs ≥2× | True |
| M4 | `rec_cov` | share of song bars covered by a vocabulary item that occurs ≥2 times | higher |
| M5 | `frag` | number of play-order sections shorter than 4 bars ("one-off fragments") | lower |

M3 is the headline: it is the closest thing to "a musician would call this form
sane". M1 alone can be gamed by merging everything; M4 alone by minting one
giant pattern. They are reported together and a variant only wins if it moves
M3 up without moving M4 down.

## The blur prior (b) is scored separately

Boundary set from Foote checkerboard novelty on the Gaussian-blurred chord-tone
SSM vs the refined detector's letter-change bars, at ±2 bars tolerance:
precision, recall, F1, swept over σ ∈ {2,3,4,6,8} bars over all 50 charts.
This measures AGREEMENT with the sharp detector, not truth — stated up front
because the blur's whole job is to be a *prior* on the sharp detector.

## What this metric does NOT measure

* **Truth.** Only the ~25 `irealb_*` twins carry real forms, and at least one is
  mis-titled (contains a different tune). Everything above is self-consistency.
* **Chord correctness.** All variants read the same decoded chords; a wrong chord
  is wrong in every arm.
* **Playback.** M2 is a proxy for the playhead drift the under-fold rule fixed;
  it does not re-measure drift.
