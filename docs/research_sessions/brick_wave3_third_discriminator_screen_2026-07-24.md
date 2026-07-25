# Wave-3: 3rd-degree (maj↔min) discriminator — premise screen, 2026-07-24

Continuation of `brick_wave2_bricks_2026-07-24.md` ("next hypothesis: close_to_you's
strict .235 is driven by musx FAMILY errors (dom→maj, min→maj) that need the 3rd, not
the 7th — a 3rd-degree discriminator screen is the untried quality lever"). Final,
cheap, decisive premise-screen — **no brick built**. Baseline unchanged: pooled root
0.7367 / majmin 0.7102 / 7ths 0.5173 / partial 0.6419 / strict 0.4774 / bass 0.7440,
7 frozen songs, 1654s (`harmonia.eval.accuracy_score`, read-only).

## Scoping (music theory, per brief)
Family errors split by which note actually discriminates them:
- **maj↔min**: 3rd differs (root+3 vs root+4) → in scope for a 3rd-degree discriminator.
- **dom↔maj**: both have a major 3rd, differ only in the b7 (root+10) → this is the SAME
  lever Wave-2 Brick B already tried and killed (cross-segment b7 leak). Out of scope here.

## Method
Reused the Wave-2 `_label_segments` monkeypatch pattern (`scratchpad/screen_third.py`,
template `scratchpad/screen_B.py`): captured per-FINAL-pass-segment (root, raw treble
chroma = `seg_feat[:,12:]`, i.e. `feat[s:e].mean(0)[12:]` where `feat = nf.pool_beats(...)`
— confirmed L2-per-half-normalized, C-frame, absolute pitch-class, 24d `[bass(12), treble(12)]`
via `harmonia/models/nnls_features.py:pool_beats` — the same convention `fifth_discriminator.py`
and `_fifth_corrected_quality` both use), joined to frozen GT by segment-midpoint lookup.
Restricted to **ROOT-correct** segments where `chord_family(gt) != chord_family(pred)`
(the `harmonia.eval.accuracy_score.chord_family` 7-way bucket), on `close_to_you` +
`blue_bossa_backing` (the two songs the brief names as losing to family errors on
already-correct roots). Disk-safe: both WAVs were already cached from Wave-1/2
(`scratchpad/wavtmp/`), so this used zero fresh decodes (df stayed at 5 GiB, never
approached the 1.5 GiB floor).

## Step 2 — mass quantification: STOP condition triggers

| song | maj↔min mass (s) | other mass (s) [mostly dom↔maj] |
|---|---|---|
| close_to_you | 3.38 | 29.11 |
| blue_bossa_backing | 5.99 | 39.56 |
| **POOLED** | **9.38** | **68.67** |

maj↔min pairs, both songs: **100% `(gt=min, pred=maj)`** — zero instances of the reverse
direction (gt=maj mispredicted as min) anywhere in either song's root-correct/family-wrong set.
"other" mass is dominated by `dom→maj` (26.4s close_to_you, 12.4s blue_bossa_backing) and
`maj→dom` (22.8s blue_bossa_backing) — i.e. the dead 7th-degree lever, confirming it's still
the larger share (68.67s vs 9.38s, ~88% of the family-wrong-but-root-correct mass).

**9.38s pooled < the 15s STOP floor**, and 9.38s / 1654s (full-7 pooled duration) = **0.57pp**,
well under the 2pp bar. Per the brief's own criterion this is decisive: **STOP — the maj↔min
mass is too small to be a lever**, before even running the separation check. Ran it anyway
(cheap, data already captured) for completeness and to close the axis honestly.

## Step 3 — 3rd-degree separation check: confounded, and inverted

16 segments in the maj↔min bucket (all `gt=min7`, `pred=maj` — see above). `disc = c[root+3] −
c[root+4]` on the L1-renormalized raw treble chroma (>0 should favor min, <0 should favor maj):

| | n | median disc |
|---|---|---|
| true-min (all 16) | 16 | **−0.021** |
| true-maj | 0 | n/a (no reverse-direction instances in this sample) |

Duration-weighted **% correctly classified by sign: 25.6% (2.40s / 9.38s)** — worse than a
coin flip, and the median disc for true-min segments is *negative* (the chroma leans major-3rd
even though GT says minor7). A rule that ignored the chroma entirely and always guessed "min"
whenever family disagrees would score 100% on this sample (trivially, since it's 16/16 one-
directional) — the one-note evidence is actively pointing the wrong way, not just weak.

**Same cross-boundary leakage confound as Wave-2 Brick B, confirmed from GT context:**
- `blue_bossa_backing` 52.77–54.37s: GT is one `C:min7` chord (52.725–54.325s), preceded by
  `G:7` (dominant, major 3rd = B) and followed by `D:hdim7`. Music-x-lab's fine segmentation
  split this single GT chord into four ~0.4s micro-segments. The first two (52.77–53.57,
  right after the G7→Cm7 boundary) are WRONG (disc negative, maj-leaning); the back two
  (53.57–54.37, farther from the boundary) flip to OK (disc positive). Textbook onset-adjacent
  leakage from the preceding dominant's major 3rd.
- `close_to_you` 63.67–65.70s: GT `E:min7` spans 63.032–65.747s; the captured window covers
  nearly the whole chord (2.03s of 2.715s) yet is still maj-leaning — not a micro-segment this
  time, so leakage alone doesn't fully explain it; still consistent with genuine per-song
  quality-classifier bias rather than a recoverable acoustic signal.

Short-segment mass (<0.6s, the fast-harmonic-rhythm micro-segments) = 5.19s of the 9.38s bucket
— over half the already-tiny target mass sits exactly in the boundary-bleed-prone regime Brick B
diagnosed. Verdict: **confounded**, and additionally too small to matter even in principle.

## Decision: DROP

Both stop conditions from the brief fire independently:
1. Mass: 9.38s pooled ≈ 0.57pp of pooled duration, under the 15s / 2pp floor.
2. Separation: 25.6% correct (worse than chance), sign inverted on the median, and the same
   cross-segment leakage confound that killed Brick B is visibly present (leading micro-segments
   of a single GT chord flip sign near a dominant-chord boundary).

No `harmonia/models/third_discriminator.py` or `tests/test_third_discriminator.py` was built —
the brief's build gate ("clean separation ≥~75% correct + clear sign split AND meaningful mass")
fails on both counts, decisively enough that building-then-measuring would just reproduce Brick
B's mirage pattern at a smaller scale. No pipeline files touched (read-only); no wiring; no git.

## Bottom line: the quality axis is now CLOSED for post-hoc chroma discrimination

Across Wave-2 and Wave-3, every note-level chroma discriminator aimed at *quality* refinement
on the current 7-song benchmark has failed for the same underlying reason — **music-x-lab's
segment-level label is the stronger source, and any one-note acoustic tie-break computed on a
mean-pooled segment window inherits cross-boundary leakage from the neighboring chord** (Brick
B: b7 leak on min→min7; Wave-3: 3rd-degree leak on min→maj, same mechanism, opposite degree).
The one quality-axis brick that DID hold (Wave-1 `fifth_discriminator.py`, root-level V-vs-I,
not quality) worked specifically because it fires on genuinely *sustained* tonic-vs-dominant
ambiguity in major keys, not on transient family disagreement at chord boundaries — a different
regime. The close_to_you/blue_bossa_backing family-error mass is real (98.05s pooled, root-
correct-but-family-wrong) but 88% of it is the dom↔maj/7th-degree kind, already proven
unfixable by Brick B; only 9.38s (9.6%) is the 3rd-degree kind this screen targeted, and that
slice is itself confounded. **Closing this axis**: further gains need either (a) a genuinely
boundary-bleed-free feature (e.g. `_clip_pool` in `yt_chord_corpus.py`, frame-clipped to
[t0,t1) with zero adjacent-beat bleed, already exists for a *different* purpose — worth trying
as the feature source before declaring quality unfixable outright) or (b) improving the upstream
root/quality source itself (music-x-lab retraining / ensemble reweighting), not further post-hoc
adjudication between existing sources.

## Files
- New (this screen): `docs/research_sessions/brick_wave3_third_discriminator_screen_2026-07-24.md`
  (this file) only. No module, no test — DROP verdict, nothing to stage besides this doc.
- Scratchpad (throwaway, not staged): `scratchpad/screen_third.py`, `scratchpad/analyze_third.py`,
  `scratchpad/third_records.json`, `scratchpad/third_mass.json`.
- Read-only reuse: `harmonia.eval.accuracy_score` (`load_frozen_gt`, `run_prediction`,
  `chord_family`), `harmonia.models.chord_pipeline_v1._label_segments` (monkeypatched, not
  edited), `harmonia.models.nnls_features.pool_beats` (read for the treble-slice convention).
