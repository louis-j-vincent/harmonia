# Quality-model selection: cascade vs. flat classifier — a product decision

**Status:** decided (default), not yet wired into the production pipeline.
**Source data:** `docs/known_issues.md` 2026-07-17, "Addendum 2 — the cascade
BUILT + evaluated end-to-end"; numbers from `scratchpad/nnls_cascade_pipeline.py`
(log `scratchpad/nnls_cascade_pipeline.log`, result
`scratchpad/nnls_cascade_pipeline.json`), 5-seed RWC-Popular CV, NNLS-24 front-end.

## The two options

Both consume the same NNLS-24 front-end; they differ only in how the 7-way
chord-quality decision is made.

1. **Flat NNLS 7-way classifier.** One softmax over
   {maj, min, dom, hdim, dim, aug, sus}, class-weighted training.
2. **Cascade (soft-hierarchical).** Stage 1 is a 3-way router
   {maj, min, residual}; chords routed to "residual" go to a Stage-2 5-way
   specialist {dom, hdim, dim, aug, sus} trained only on residual chords.
   Final probabilities are the soft product of the two stages.

## The numbers (pooled 5-seed, 7-way, RWC-Popular)

| System | raw acc | bal acc | Δraw vs flat | Δbal vs flat |
|---|---|---|---|---|
| **Flat NNLS 7-way** | 0.749 | **0.657** | — | — |
| Cascade HARD routing | 0.804 | 0.615 | +0.054 | −0.042 |
| **Cascade SOFT hierarchical** | **0.830** | 0.587 | **+0.081** | **−0.070** |
| Cascade CONF routing (τ=0.7) | 0.727 | 0.634 | −0.022 | −0.023 |

Mechanism (from known_issues.md): the cascade's Stage 1 nails the 87.2%
maj/min majority at 0.953 binary accuracy, and the soft product multiplies
residual-class probability by P(residual)<1 — fewer majority→rare mistakes,
higher raw accuracy. But the flat classifier's class-weighted softmax
*already* implicitly protects rare-class recall; the cascade's explicit split
erodes that protection rather than adding to it. So the tradeoff is not "the
cascade is a strict improvement, minus a rounding error" — it is a real
raw↔balanced reallocation, confirmed on a held-out 5-seed CV, not a single
lucky split.

## Recommended default: **flat NNLS classifier**

**Reasoning.** CLAUDE.md frames this project's audience as "the human is an
ML PhD and jazz musician" — i.e. someone doing serious harmonic analysis, not
a casual listener who only cares about getting the I–IV–V chords right. Rare
qualities (dim, aug, sus, half-diminished) are exactly the vocabulary that
matters most in jazz analysis and least in casual pop use — they're also the
classes the balanced-accuracy metric protects and the ones a play-along-chart
user would tolerate being smoothed to the nearest maj/min. Given that
framing, the −7pp balanced-accuracy cost of the cascade is a worse trade than
the +8.1pp raw-accuracy gain it buys, because the raw-accuracy gain is
concentrated on chords (common maj/min) that were already easy, while the
balanced-accuracy loss lands on chords (dim/aug/sus/hdim) this project's
stated user most needs correctly identified.

This is a recommendation, not a silent default: if a future deployment target
shifts toward a casual play-along-chart use case (raw-accuracy-dominated,
rare qualities rounded away without much user cost), the cascade
(soft-hierarchical variant, +8.1pp raw) is the better choice and this doc's
default should flip for that build, not be silently overridden.

## Non-default use case: cascade

Use the **soft-hierarchical cascade** when the deployment target is
common-chord-dominated (e.g. a casual play-along chart for a pop tune where
users mainly care about getting the big maj/min/dom shapes right and rarely
encounter or care about dim/aug/sus/hdim precision). Avoid CONF routing
(τ=0.7) — it is dominated by flat NNLS on both metrics (0.727/0.634 vs
0.749/0.657) — and avoid HARD routing, which is dominated by SOFT (0.804/0.615
vs 0.830/0.587): if a cascade is used at all, use the soft-hierarchical
variant.

## How to wire this in (not yet done)

`harmonia/models/chord_pipeline_v1.py` currently resolves the quality/family
decision through `_get_family_clf()` / `_get_ctx_clf()` (module-level lazy
singletons around line 975–1014), backed by `_FamilyClassifier` (defined at
line 616) — this is the BP48-era flat classifier, not the NNLS-24 cascade or
flat-NNLS classifier described above. As of this writing there is **no NNLS-24
wiring in `chord_pipeline_v1.py` yet** (confirmed by grep — no `nnls`/`NNLS`
references in the file), so this doc's recommendation is not in tension with
any in-flight edit there.

Suggested integration shape for whoever does the NNLS-24 production migration
(check `docs/known_issues.md` for that agent's current progress before
editing this file, per CLAUDE.md's cross-session-conflict guidance):

- Add a `quality_mode: Literal["flat", "cascade"] = "flat"` parameter
  (default per this doc's recommendation) on whatever new NNLS-24 quality
  classifier class/loader replaces or sits alongside `_FamilyClassifier`.
- Load either the flat 7-way head or the {router + residual-specialist} pair
  behind that flag, mirroring the existing `_get_family_clf()` /
  `_get_ctx_clf()` lazy-singleton pattern (one cached instance per mode, not
  reloaded per call).
- Surface the flag up through whatever CLI/config entry point selects models
  today (e.g. `scripts/harmonia_server.py`, `scripts/render_youtube_chart.py`)
  so a "play-along chart" build can opt into `cascade` without a code change.
- Keep both code paths behind the same evaluation harness
  (`scripts/rwc_nnls_multihead_cv.py` / `scratchpad/nnls_cascade_pipeline.py`)
  so a future re-sweep can re-confirm the tradeoff after any retraining.
