# Joint analysis: A×E and D×E pairs

Analysis of `features.csv` (1389 rows, 5 songs: 001-005). Loaded with
`pd.read_csv(path, dtype={"song_id": str})` per the README gotcha.
Script used for this analysis is not checked in (exploratory, run ad hoc);
plots are saved under `plots/`.

## Pair 1: `A_beat_phase` × `E_position_in_loop`

**UPDATED 2026-07-04 after fixing the root cause this pair surfaced.** The
original version of this analysis (below the line) found loop-start and
downbeat as disjoint sets in 2 of 5 songs and used that as evidence the two
signals were "differently anchored." Investigating *why* led to a real bug,
not just a data quirk: `harmonia/models/periodicity.py::score_periods()`
only ever detects a loop's *length*, never its *phase* — `E_position_in_loop
= beat_idx % period` was silently assuming beat 0 of the song is also beat
0 of the loop. Fixed by adding `find_loop_phase()`, which anchors position 0
to the first annotated downbeat instead
(`harmonia/models/periodicity.py`, `tests/test_periodicity.py`, wired into
`scripts/build_chord_change_features.py`; see `docs/known_issues.md` #1).
`features.csv` was regenerated with the fix — **only `E_position_in_loop`
and the new `E_loop_phase` column changed**, every other column is
byte-identical to before (verified directly). The results below are the
corrected numbers; the original (pre-fix) numbers are kept underneath for
the record, since the *methodological lesson* (a Simpson's-paradox artifact
from misaligned sets) is itself worth keeping.

**Setup.** All 5 songs have `E_detected_period > 0` (four songs at period
16, song 004 at period 32) — none excluded. Defined `loop_start =
(E_position_in_loop == 0)` and `downbeat = (A_beat_phase == 0)`.

### Check: is loop-start the same SET of beats as downbeat, per song? (corrected)

Now, mostly yes — loop-start is a clean subset of downbeats in 4 of 5
songs, and close in the fifth:

| song | period | phase anchor | loop-start beats | downbeats | intersection | loop-start subset of downbeats? |
|---|---|---|---|---|---|---|
| 001 | 16 | 0 | 19 | 73 | 19 | yes (100%) |
| 002 | 16 | 0 | 16 | 60 | 15 | close (93.8%) |
| 003 | 16 | 2 | 20 | 79 | 20 | yes (100%) |
| 004 | 32 | 2 | 8 | 59 | 8 | yes (100%) |
| 005 | 16 | 0 | 19 | 77 | 13 | no (68.4%) |

Songs 003 and 004 — the two that had **zero** overlap before the fix — now
anchor cleanly at phase 2 instead of phase 0, exactly the correction the
bug predicted. Pooled overlap rate: 91.5% (up from a pooled number that
was dominated by two fully-disjoint songs before). The residual misses in
002 and 005 are not a sign the fix is incomplete: both songs are
independently already flagged (see the `A_beat_phase` validation note,
`scripts/validate_chord_change_features.py`) as having rare irregular
inter-downbeat gaps (>4 beats) elsewhere in the song — a single irregular
bar shifts every loop-start beat after it by that same drift, since this
fix assumes one fixed period+phase for the whole song. That's a distinct,
smaller residual limitation (irregular meter drift), not the phase-anchor
bug itself.

Pooled contingency (`A_beat_phase` in 0-3 × `loop_start`): Cramer's V =
0.389 (chi2=208.6, p=6e-45, N=1289) — noticeably stronger than the
pre-fix 0.267, which makes sense: `loop_start` is now genuinely
(near-)nested inside `downbeat`, so of course the two are more associated
than when a third of the loop-start beats were scattered onto the wrong
phase.

### Does loop-start add predictive power beyond downbeat status? (corrected)

Pooled, restricting to **downbeats only** (`A_beat_phase==0`, n=348):
P(chord_changed | loop_start=False) = 68.1% (n=273) vs P(chord_changed |
loop_start=True) = 64.0% (n=75) — pooled now shows a small **negative**
difference, not the misleading +26-point "lift" the pre-fix data showed.
Per song, now that every song contributes real data to both cells:

| song | loop_start=False | loop_start=True |
|---|---|---|
| 001 | 98.1% (n=54) | 94.7% (n=19) |
| 002 | 84.4% (n=45) | 80.0% (n=15) |
| 003 | 0.0% (n=59) | 0.0% (n=20) |
| 004 | 64.7% (n=51) | 75.0% (n=8) |
| 005 | 96.9% (n=64) | 92.3% (n=13) |

Four of five songs (001, 002, 003, 005) show loop-start beats at
**equal-or-lower** P(chord_changed) than other downbeats; only song 004
shows a positive difference, and its n=8 loop-start cell is too small to
read much into. Song 003 is flat 0% either way (it just doesn't change
chords on any downbeat, loop-start or not).

**Verdict: NOT complementary — this settles what the pre-fix data left
ambiguous.** With the phase bug fixed, the loop-start/downbeat sets are
(mostly) genuinely nested rather than accidentally disjoint, which makes
this a fair comparison for the first time. The fair comparison shows no
positive lift: loop-start status does not sharpen the downbeat-based
estimate, and if anything points slightly negative. This actually
*resolves* the original "inconclusive" verdict rather than just refining
it — the earlier ambiguity was a data-quality artifact (mismatched sets
from the phase bug), not real statistical uncertainty. The two features
remain "not redundant" in the sense that they're computed differently and
don't pick out identical beat sets, but combining them is not worth the
complexity for chord-change prediction.

Plot: `plots/pair_A_phase_vs_E_loop.png` is STALE (built from the pre-fix
data) — regenerate before reusing it; the numbers above are from a rerun
plotting script, not yet re-saved as a PNG.

---

### Original pre-fix analysis (2026-07-03), kept for the methodological record

The write-up below is what this section said before the phase-anchor bug
was found and fixed. It's preserved because the mistake it made — reading
a Simpson's-paradox artifact as a real "combination effect" — is a useful
cautionary example, not because its numbers are still trusted.

No, not in general — the two sets are neither equal nor even nested the
same way across songs:

| song | period | loop-start beats | downbeats | intersection | loop-start subset of downbeats? |
|---|---|---|---|---|---|
| 001 | 16 | 19 | 73 | 19 | yes |
| 002 | 16 | 16 | 60 | 15 | no (1 loop-start beat is NOT a downbeat) |
| 003 | 16 | 20 | 79 | 0 | no (zero overlap) |
| 004 | 32 | 8 | 59 | 0 | no (zero overlap) |
| 005 | 16 | 19 | 77 | 13 | no |

Song 003 and song 004 have **zero overlap** between loop-start beats and
downbeats — the detected periodicity phase is offset from the annotated
downbeat grid entirely. Song 001 is the only song where loop-start is a
clean subset of downbeats (consistent with "period is a multiple of 4 and
phase-aligned"), and even there it's a subset, not an equal set (only 19 of
73 downbeats are loop-starts, since 001 is much longer than one loop). So
the "loop start is always a downbeat by construction" assumption in the
prompt does **not** hold empirically for 3 of 5 songs — the periodicity
detector (`harmonia.models.periodicity.score_periods`) evidently doesn't
anchor its phase to the downbeat annotation, and/or `beat_idx` numbering
doesn't start at a downbeat for every song. This is itself a useful
finding: `E_position_in_loop==0` is not simply redundant with
`A_beat_phase==0` — it's a differently-anchored (and per song 003/004,
essentially uncorrelated) periodic signal.

Pooled contingency (`A_beat_phase` binned 0/1/2/3+/neg x `loop_start`):
Cramer's V = 0.267 (chi2=98.9, p=1.7e-20, N=1389) — a real but moderate
association, driven mostly by the fact that `loop_start` beats never fall
on phase 1 (0 of 348) and rarely on phase 3+ (1 of 346), i.e. loop starts
are concentrated on phases 0 and 2. That's consistent with a period that's
a multiple of 4 relative to *some* internal 4-beat cycle, but that cycle
isn't the downbeat grid in 3/5 songs (see above).

Pooled, restricting to **downbeats only** (`A_beat_phase==0`, n=348):
P(chord_changed | loop_start=False) = 63.8% (n=301) vs P(chord_changed |
loop_start=True) = 89.4% (n=47). That looks like a large lift, but it's not
robust per-song:

| song | loop_start=False | loop_start=True |
|---|---|---|
| 001 | 98.1% (n=54) | 94.7% (n=19) |
| 002 | 84.4% (n=45) | 80.0% (n=15) |
| 003 | 0.0% (n=79) | (no loop-start downbeats) |
| 004 | 66.1% (n=59) | (no loop-start downbeats) |
| 005 | 96.9% (n=64) | 92.3% (n=13) |

In every song that has both groups (001, 002, 005), loop-start downbeats
have **equal or slightly lower** P(chord_changed) than non-loop-start
downbeats — the opposite direction of the pooled 63.8% -> 89.4% "lift."
That pooled number is a Simpson's-paradox artifact: it's dominated by
song 003, which has 0% chord-change on all 79 downbeats (song 003's
downbeats and loop-starts never coincide, so it contributes 79 zero-value
points only to the `loop_start=False` bucket) and song 004, which has no
loop-start downbeats at all, pulling the `False` pooled average down
further.

**Original verdict (superseded above): inconclusive / not what it first
looks like.**

## Pair 2: `D_chroma_cosine_dist` / `D_onset_density` × `E_dist_to_segment_boundary`

**Setup.** Binned `E_dist_to_segment_boundary` into {0, 1, 2, 3+} beats
from the nearest structural boundary. Bin sizes (pooled, all 5 songs,
N=1389): 0->59, 1->118, 2->118, 3+->1094. The "3+" bin dominates by
construction (`E_dist_to_segment_boundary` mean is 7.5 beats, i.e. most
beats are far from any boundary — segments are typically many beats long).

### Does chroma novelty / onset density spike at boundaries?

Kruskal-Wallis across the 4 bins: chroma novelty H=10.48, p=0.015; onset
density H=11.24, p=0.011 — both nominally significant, but Spearman
correlation between the raw (unbinned) distance and each metric is tiny:
rho=0.075 (p=0.005) for chroma, rho=0.057 (p=0.032) for onset density.
These are real but very weak monotonic associations — not a "chroma
novelty decays smoothly with distance" pattern.

Per-bin summary (pooled):

| dist bin | n | chroma median | chroma mean | onset median | onset mean |
|---|---|---|---|---|---|
| 0 | 59 | 0.133 | 0.283 | 27.28 | 25.81 |
| 1 | 118 | 0.258 | 0.311 | 24.66 | 24.26 |
| 2 | 118 | 0.126 | 0.223 | 23.24 | 23.01 |
| 3+ | 1094 | 0.200 | 0.265 | 26.09 | 26.46 |

Notably, the median chroma novelty **at the boundary itself (bin 0, median
0.133) is not the highest** — bin 1 (one beat away) has a higher median
(0.258) and the "3+" bin (far from any boundary) sits in between. There is
no monotonic decay pattern at all; the group means are all within a fairly
narrow band (0.22-0.31) with heavily overlapping distributions (see
boxplot — IQRs overlap substantially across all 4 bins, and outliers reach
1.0 in every bin).

Per-song medians (chroma) show why the pooled picture is so muddled: song
003's bin-0 median is 0.58 (a real spike at boundaries) while song 005's
bin-0 median is 0.07 (a trough) — the direction of the boundary effect is
not consistent across songs, so pooling washes it into a near-null pooled
signal that's only "significant" by Kruskal-Wallis because of the large N
in the 3+ bin, not because of a strong or consistent effect size.

Onset density shows the same story: no consistent monotonic pattern, and
per-song medians range narrowly (18-34) with no systematic peak at
distance 0.

I also checked chroma novelty at boundary vs away, split by
`chord_changed` (median, at-boundary vs away): for `chord_changed=False`,
0.130 (n=31) vs 0.134 (n=761) — essentially identical; for
`chord_changed=True`, 0.158 (n=28) vs 0.300 (n=569) — chroma novelty is
actually **higher away from boundaries** among true chord-change beats,
the opposite of "novelty concentrates at boundaries." This directly
supports the hypothesis in the prompt that structural boundaries and
beat-to-beat chord novelty are largely different phenomena.

**Verdict: NOT redundant, but "complementary" in the sense of adding joint
predictive power is not demonstrated either — the boundary-distance signal
shows essentially no relationship to local chroma/onset novelty in this
sample.** The Kruskal-Wallis p-values are nominally significant (large N
in the 3+ bin does that even for a rho of about 0.06-0.08), but the effect
is not visually or practically meaningful (heavily overlapping boxes,
non-monotonic per-song medians, inconsistent sign across songs). The
honest read: `E_dist_to_segment_boundary` (built from its own
SSM/checkerboard-novelty detector) and beat-level chroma novelty are two
different signals that don't track each other — good news for a joint
model (they're not wasted redundant features) but this dataset does not
show a clean "boundary implies chroma spike" relationship to exploit, and
the tiny bin-0 sample (n=59, spread over 5 songs, so about 12 boundary
beats/song) is not enough to resolve song-level heterogeneity.

Plot: `plots/pair_D_chroma_vs_E_boundary.png` — box plots of both metrics
across the 4 distance bins, with Kruskal-Wallis p and Spearman rho in the
subplot titles.

## Summary

| Pair | Redundant? | Complementary? | Confidence |
|---|---|---|---|
| A_beat_phase x E_position_in_loop | No, but now (mostly) nested — loop-start is a subset of downbeats in 4/5 songs after the phase-anchor fix (was 1/5, with 2/5 fully disjoint, before it) | No — the corrected, fair per-song comparison shows equal-or-negative lift in 4/5 songs; the earlier apparent "positive lift" was a Simpson's-paradox artifact of the phase bug | Moderate (the fix makes this a fair comparison for the first time; still only 5 songs) |
| D_chroma/onset x E_dist_to_segment_boundary | No — correlation with boundary distance is near-zero (rho 0.06-0.08) and non-monotonic per song | Not demonstrated — boundary distance doesn't predict local novelty well enough to combine usefully as tested here | Low-moderate (bin-0 n=59 across 5 songs is thin; direction of effect flips between songs 003 and 005; unaffected by the periodicity fix) |

Both pairs turned out to be **not redundant** (each metric is not simply a
proxy for the other), which is a useful negative result for feature
selection — neither pair can be safely collapsed to one signal. Neither
pair shows the stronger "combination beats either alone" pattern the A×B
and B×D pairs did (see `findings_A_vs_BD.md`, `findings_B_vs_DC.md`) — for
A×E this is now a settled negative result (not just an underpowered one),
since the phase-anchor fix removed the specific artifact that made the
pre-fix data ambiguous; for D×E it remains genuinely underpowered at 5
songs and worth re-checking on a larger sample before ruling out entirely.
