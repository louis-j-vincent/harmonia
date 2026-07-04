# Joint analysis: A_beat_phase x B_bass_changed, A_beat_phase x D_* (onset/chroma)

Data: `features.csv`, 1389 rows, 5 songs (001-005), loaded with
`dtype={"song_id": str}`. All analyses below restrict to `A_beat_phase` in
{0,1,2,3} (1381/1389 rows) - the rare phase in {-1,4,5} rows (8 total) are
dropped, too few to say anything about.

All numbers are pooled across the 5 songs unless stated otherwise. Caveat
that applies to everything in this file: n=5 songs is a very small,
non-random sample (song 001 is already flagged in PRIOR_FINDINGS.md as
metrically atypical - 100th percentile regularity), and per-song breakdowns
below show real heterogeneity, sometimes reversing the pooled pattern. Treat
pooled numbers as descriptive of this 5-song sample, not as corpus-wide
claims.

## Pair 1: A_beat_phase x B_bass_changed

Plots: `plots/pair_A_phase_vs_B_bass.png` (P(bass_changed | phase) bar
chart), `plots/pair_A_phase_vs_B_bass_chordchange_grid.png` (2x4 heatmap of
P(chord_changed | bass_changed, phase)).

### Does phase predict bass_changed? (redundancy check)

P(B_bass_changed | A_beat_phase), pooled:

| phase | P(bass_changed) | n |
|---|---|---|
| 0 | 0.603 | 348 |
| 1 | 0.448 | 348 |
| 2 | 0.493 | 343 |
| 3 | 0.436 | 342 |

Overall rate 0.495 (n=1381). Chi-square test of independence (phase x
bass_changed): chi2=24.23, dof=3, p=2.2e-05 - statistically significant
association, but Cramer's V = 0.13, a small effect. Bass is somewhat more
likely to change on the downbeat (60.3%) than on beat 3 (43.6%), a ~17-point
spread, but far from deterministic in either direction - bass changes
plenty off the downbeat (>43% of every phase bucket) and stays put on the
downbeat almost 40% of the time. Verdict: weak dependence, not redundancy.
Knowing beat_phase shifts your bass-change estimate by ~17 points at most,
nowhere near collapsing it to a near-0/near-1 indicator the way true
redundancy would.

### Does the combination beat either signal alone for predicting chord_changed?

P(chord_changed) single-signal baselines:

| signal | value | P(chord_changed) | n |
|---|---|---|---|
| bass_changed | False | 0.331 | 697 |
| bass_changed | True | 0.535 | 684 |
| phase | 0 | 0.672 | 348 |
| phase | 1 | 0.359 | 348 |
| phase | 2 | 0.574 | 343 |
| phase | 3 | 0.120 | 342 |

Joint P(chord_changed | bass_changed, phase):

| bass_changed | phase 0 | phase 1 | phase 2 | phase 3 |
|---|---|---|---|---|
| False | 0.609 (n=138) | 0.323 (n=192) | 0.379 (n=174) | 0.098 (n=193) |
| True | 0.714 (n=210) | 0.404 (n=156) | 0.775 (n=169) | 0.148 (n=149) |

All 8 cells have n>=138, comfortably above the n<20 caution threshold.

Findings:
- The best single-signal cell is phase=0 alone at 0.672. The best joint cell
  is (bass_changed=True, phase=2) at 0.775 - clearly higher than any
  single-signal number, and higher than (bass_changed=True, phase=0)=0.714
  too. This is a genuinely interesting combination effect: phase=2 alone is
  a middling signal (0.574) but combined with bass_changed=True it jumps to
  0.775, higher than what either the downbeat or the bass signal delivers by
  itself anywhere in the table.
- At the low end, (bass_changed=False, phase=3) = 0.098 is lower than either
  single-signal floor (phase=3 alone: 0.120; bass_changed=False alone:
  0.331) - again the combination sharpens the estimate beyond either
  marginal.
- Within every phase, bass_changed=True raises P(chord_changed) relative to
  bass_changed=False (largest gap at phase=2: 0.775 vs 0.379, +40 points;
  smallest at phase=0: 0.714 vs 0.609, +11 points) - i.e. bass_changed keeps
  adding information at every phase, it doesn't just echo phase.

Verdict: complementary. Phase and bass_changed are only weakly coupled to
each other (Cramer's V=0.13), and their combination produces both a
stronger positive cell (0.775) and a stronger negative cell (0.098) than
either marginal signal reaches alone. This is exactly the "signals worth
combining" pattern described in the README, not redundancy.

Caution - per-song heterogeneity is large. Breaking the (bass_changed,
phase) cells down by song shows the pooled phase=2 result is not uniform:
song 003 has P(chord_changed)=0.0 at phase=2 for both bass_changed values
(n=16 and n=61), while songs 001/002/005 show phase=2 rates of 0.86-1.0 when
bass_changed=True. The pooled 0.775 figure is a blend of songs with quite
different metrical behavior at that phase, not a uniform effect across the
5-song sample. The direction (bass_changed=True > False within a phase)
does hold in most per-song slices, but magnitudes vary a lot and the
smallest per-song cells (e.g. n=16, n=17, n=18) are individually
unreliable. Read the pooled combination effect as suggestive, not as a
settled per-song fact.

## Pair 2: A_beat_phase x D_onset_density and x D_chroma_cosine_dist

Plot: `plots/pair_A_phase_vs_D_onset_and_chroma.png` (side-by-side box
plots, one per metric, grouped by phase 0-3; n printed on each box).

### D_onset_density by phase

| phase | mean | median | std | n |
|---|---|---|---|---|
| 0 | 28.08 | 27.23 | 10.55 | 348 |
| 1 | 25.51 | 24.90 | 9.81 | 348 |
| 2 | 25.96 | 25.76 | 10.44 | 343 |
| 3 | 24.78 | 24.48 | 11.04 | 342 |

One-way ANOVA: F=6.33, p=0.0003 (statistically significant). Effect size
eta-squared = 0.014 - phase explains only ~1.4% of the variance in onset
density. The means differ by about 3.3 units (28.08 vs 24.78, downbeat vs
weakest beat) against a within-group std of ~10-11, i.e. the difference
between phases is small relative to the spread within any single phase -
the boxplots show heavily overlapping distributions.

### D_chroma_cosine_dist by phase

| phase | mean | median | std | n |
|---|---|---|---|---|
| 0 | 0.339 | 0.286 | 0.252 | 348 |
| 1 | 0.261 | 0.197 | 0.227 | 348 |
| 2 | 0.251 | 0.167 | 0.224 | 343 |
| 3 | 0.215 | 0.133 | 0.216 | 342 |

One-way ANOVA: F=17.87, p<0.0001. Effect size eta-squared = 0.038 - larger
than onset density's but still small in absolute terms (~3.8% of variance
explained). The downbeat (phase=0) mean (0.339) is about 1.6x phase=3's
mean (0.215), a real and monotonic-looking decline from phase 0 to phase 3,
but again with substantial within-group spread (std ~0.22-0.25, comparable
to the between-group mean differences).

### Verdict

Both D_onset_density and D_chroma_cosine_dist are statistically
significantly elevated on the downbeat and decline monotonically through
the beat cycle (phase 0 > 1 ~ 2 > 3 for chroma distance; roughly 0 > 2 ~ 1 >
3 for onset density, with 1 and 2 close). This is consistent with the
hypothesis that "something happens" more at strong metrical positions,
matching what phase alone already tells you.

However, the effect sizes are small (eta-squared 0.014 and 0.038) - phase
explains well under 5% of the variance in either D metric. This means:
partial overlap, not redundancy. There is a real, non-trivial shared
component (both D metrics correlate with phase, and PRIOR_FINDINGS.md
already shows both correlate with chord_changed on their own, r=0.24 and
r=0.22 for song 001), but the vast majority of variance in onset density
and chroma novelty is not explained by beat phase - these metrics are
mostly carrying information beyond "is this a downbeat." A model combining
phase with either D metric should still gain from both, though the
marginal value of D on top of phase is probably smaller than the marginal
value the B_bass_changed combination showed in Pair 1, precisely because
Pair 1's phase/bass association was even weaker (Cramer's V=0.13) while
still producing a strong joint effect - here the shared component, while
modest, is picked up by a formal significance test, so this pair sits
closer to "complementary with some overlap" than Pair 1's cleaner
"complementary."

Same per-song caution applies: these ANOVAs pool 5 songs with different
tempos, one of which (002) has a known tempo-octave beat-tracking
discrepancy noted in the README (irrelevant here since this table uses
POP909's own annotated grid, but still: 5 songs is a small, non-random
sample and song-level medians/means for these D metrics were not
separately re-verified beyond the pooled ANOVA above).
