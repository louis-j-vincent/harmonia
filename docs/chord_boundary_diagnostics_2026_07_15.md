# Chord-boundary (segmentation) diagnostics — deployed pipeline, real inference (2026-07-15)

Resolves the "labeling vs boundaries" disagreement directly: runs the **actual
deployed inference function**, `infer_chords_billboard_v1` in
`harmonia/models/chord_pipeline_v1.py` (the same call `scripts/harmonia_server.py
._run_analysis` makes — read-only, not modified, imported and called from a
standalone script), on real YouTube audio for 4 Billboard training-corpus songs,
and compares its own emitted chord CHANGE POINTS to Billboard `chords_full` GT
change points. This had never been measured: the Billboard training corpus was
built by sampling BP48 features directly at Billboard's GT timestamps (see
`docs/root_inference_diagnostics_2026_07_15.md` and the "Root-accuracy campaign"
entries in `docs/known_issues.md`), so segmentation quality at real inference
time — with no GT available — was never in the loop until this session.

Repro: `scratchpad/boundary_diag.py` (also copied to session scratchpad),
results `scratchpad/boundary_diag_results.json`. Plots:
`docs/plots/chord_boundary_diag_bb_{1111,887,1027,362}.png`. Same 4 songs as
`docs/root_inference_diagnostics_2026_07_15.md` for direct continuity (spans
root-acc 0.99 → 0.10). WAVs downloaded one at a time and deleted immediately
(disk was at 99%/2.4 GB free the whole session; verified stable throughout).

## Method

- GT change points: `mirdata.initialize("billboard").track(tid).chords_full`
  interval starts (excluding t=0), same source as the alignment-check plots.
- Inferred change points: `start_s` of every chord span `infer_chords_billboard_v1`
  emits, excluding the first. This is a direct, board-level property of the
  deployed decoding: per-beat root+quality argmax, **coalesced by merging
  consecutive beats with identical (root, quality)** — no duration prior, no
  smoothing (see the function's module docstring).
- Matching: greedy one-to-one, tolerance = 0.5 s and = 1 beat-duration
  (song-specific, from `chart.tempo_bpm`). Precision/recall/F1 over boundary
  sets. For every missed GT boundary (FN), checked whether an inferred segment
  spans continuously across it (= a **merge** — the disruptive-for-correction
  failure mode) vs. a **shift** (inferred boundary exists nearby but outside
  tolerance).

## Results

| song | root acc (labeling) | n GT changes | n inferred changes | P (0.5s) | R (0.5s) | F1 (0.5s) | F1 (1beat) | FN that are merges |
|------|---------------------:|-------------:|--------------------:|---------:|---------:|----------:|-----------:|--------------------:|
| bb_1111 (clean)   | 0.99 | 3   | 144 | 0.007 | 0.33 | **0.01** | 0.01 | 2/2 (100%) |
| bb_887 (De La Soul)| 0.70 | 193 | 355 | 0.518 | 0.95 | **0.67** | 0.67 | 9/9 (100%) |
| bb_1027 (Greg Kihn)| 0.32 | 142 | 337 | 0.374 | 0.89 | **0.53** | 0.53 | 14/16 (88%) |
| bb_362 (hard)      | 0.10 | 83  | 163 | 0.362 | 0.71 | **0.48** | 0.57 | 23/24 (96%) |

## What the plots show

All 4 plots (waveform / GT chord grid / inferred chord grid, shared time axis,
matched boundaries green, missed-GT red-dashed, spurious-inferred magenta-dotted)
show the **same pattern**: recall is consistently high (71–95%) — nearly every
real chord change DOES have an inferred boundary near it — but **precision
collapses** (0.4–52%) because the deployed segmentation fires 1.8–2.6× (and on
the near-static bb_1111, **48×**) more boundaries than actually exist. The
`inferred (deployed pipeline)` row on every plot is visually a dense picket
fence of magenta ticks, densest exactly where the labeling problem is worst
(bb_1111's harmony barely moves — 2 real chords over 145s — yet the model
chatters between adjacent root/quality guesses almost every beat).

bb_1111 is the sharpest illustration: GT has essentially one chord change (Bb:maj
→ Eb:7, plus a trailing N). The deployed pipeline emits **145 chord spans**. Root
accuracy sampled at GT intervals is 99% (correct on average within the true span)
but the pipeline's *own* segmentation is unusable — it never actually outputs a
clean 2-chord chart; it outputs 145 chattering fragments that happen to average
out to the right label per GT window.

## Direct answer to the "couldn't correct" report

The user reported opening the Wheel/Compass editor on songs from this corpus and
being unable to correct chords because "the boundaries... seemed kind of wrong."
This diagnostic reproduces that mechanically: the **dominant failure mode is
over-segmentation / spurious splitting**, not missing or badly-shifted
boundaries. Missed GT boundaries are rare (5–29% of GT boundaries depending on
song) and when they do happen they are overwhelmingly true **merges** (88–100%
of FNs) — one inferred span silently covering two different GT chords, which
*would* also block correction (no sub-span to select). But the bigger, more
pervasive problem visible in every plot is the opposite: a real chord is
typically split into 2–4+ tiny inferred fragments (median inferred segment is a
handful of beats or less; bb_1111's 145 fragments over 3 GT chords is the
extreme case). A user trying to fix "this one wrong chord" in the Wheel editor
is confronted with a cluster of adjacent micro-segments, several with different
(and sometimes flip-flopping) labels, none of which cleanly corresponds to the
GT chord they're trying to fix — consistent with "the boundaries seemed kind of
wrong" as an honest description of what's on screen. **This is a genuine,
previously-unmeasured segmentation-quality bug in the deployed
`infer_chords_billboard_v1` path**, specifically its "no duration prior, no
smoothing, merge-identical-only" coalescing scheme.

## Verdict on the labeling-vs-boundaries disagreement

**The user is right, and the earlier "labeling is the bottleneck" conclusion
does not transfer to this deployed path.** That conclusion came from a
different corpus (jazz1460) and a different decoder (`infer_chords_v1`'s
semi-Markov/CRF-ish stack with duration priors) being fed oracle GT boundaries
— a test of "does the *classifier* need better boundaries," which found ~0
gain because that decoder already places boundaries reasonably (known_issues
#1's benchmark: exact-beat F1 0.78, ±1-beat 0.86) and its ceiling is elsewhere.
`infer_chords_billboard_v1` is architecturally different and much weaker on
this specific axis: it has **no boundary/duration model at all** — segmentation
is a side effect of independent per-beat argmax noise, coalesced only when two
*consecutive* beats happen to agree exactly. That is why its boundary F1
(0.01–0.67 here) is far below the jazz1460 learned detector's 0.78–0.86: this
isn't a harder segmentation problem, it's a segmentation step that was never
actually built for this backend. So: labeling accuracy (root/quality, #31) is
real and unsolved, but for the **currently deployed real-audio path**, boundary
placement — specifically chattering over-segmentation — is *also* a severe,
independent, user-visible bottleneck that blocks the correction workflow
outright, regardless of what the root/quality numbers say. Fix belongs on
`infer_chords_billboard_v1`'s coalescing step (duration prior / semi-Markov
merge / minimum-segment-length smoothing over the per-beat argmax stream),
not on retraining root/quality heads.
