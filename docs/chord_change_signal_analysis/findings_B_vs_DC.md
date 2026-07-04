# Findings: B (bass) × D (chroma/onset) and B (bass) × C (bigram)

Analysis by a delegated subagent, 2026-07-03. Data: `features.csv`
(1389 rows, 5 songs). See `README.md` for column definitions and
`PRIOR_FINDINGS.md` for the individual (non-joint) correlations this
builds on.

## Pair 1: `B_bass_changed` × `D_chroma_cosine_dist` / `D_onset_density`

Full 1389 rows, split by `B_bass_changed`: n=684 (bass changed), n=705
(bass unchanged).

**`D_chroma_cosine_dist`:** mean 0.370 when bass changed vs 0.166 when it
didn't. Cohen's d = 0.96 (large effect), Welch t=17.8, p=4e-63 — a large,
highly significant difference. But the KDE overlap coefficient between the
two distributions is 0.57 — substantial overlap remains. Point-biserial
r(bass_changed, chroma_cosine_dist) = 0.43.

**Verdict: complementary, with a real shared component.** Bass change and
chroma-shift are correlated (r=0.43) but bass change is not simply a subset
of "the chroma changed a lot" — there's a lot of overlap in the
distributions, meaning bass can change without a big chroma shift and vice
versa often enough to matter. Combining the two signals
(`bass_changed=True AND chroma_high`, chroma_high = above the median) moves
P(chord_changed) from 0.429 (bass_changed alone) up to 0.580 — a real,
meaningful lift from combining rather than using either alone.

**`D_onset_density`:** Cohen's d = 0.09 (negligible), p=0.08 (not
significant at conventional thresholds), KDE overlap = 0.90 (nearly
identical distributions).

**Verdict: not redundant, but also not related.** Onset density looks
essentially independent of whether the bass pitch class changed — these two
signals don't overlap at all, in either the "redundant" or "combines well"
sense. Onset density's own individual correlation with chord_changed (r=
+0.24, per PRIOR_FINDINGS.md) doesn't appear to be mediated through bass
motion at all.

Plot: `plots/pair_B_bass_vs_D_chroma_onset.png` (box + violin plots for
both metrics, split by `B_bass_changed`) — visually confirms the described
overlap patterns (chroma: visibly separated distributions with overlap;
onset density: near-identical distributions).

## Pair 2: `B_bass_is_root_or_fifth` × `C_bigram_logprob_atomic`

Restricted to `chord_changed=True` rows only (`C_bigram_logprob_atomic` is
only defined there) — n=597 total: 390 with bass on root/fifth, 207 not.

Mean bigram log-probability: -1.96 (bass on root/fifth) vs -2.09 (bass not
on root/fifth). Cohen's d = 0.10 (negligible), Welch t=1.15, p=0.25 — not
statistically significant.

Per-song breakdown shows the direction of the (small, non-significant)
effect **flips sign** across songs (001/002 go one way, 003/004/005 the
other), and one song's "not root/fifth" group has only n=11 — far too small
to draw a per-song conclusion from.

**Verdict: inconclusive.** The pooled 5-song sample (n=597, further split
into groups of ~200-390) doesn't have the statistical power to distinguish
this from independence, and the sign-flipping across songs is a warning
sign against over-interpreting the pooled mean difference as real. This
would need the full 909-song symbolic corpus (which has real chord
sequences but no audio-derived bass track) or many more rendered-audio
songs to settle properly — flagged as a genuine open question, not a
negative result.

Plot: `plots/pair_B_rootfifth_vs_C_bigram.png` (box + violin plots for
`C_bigram_logprob_atomic`, split by `B_bass_is_root_or_fifth`).
