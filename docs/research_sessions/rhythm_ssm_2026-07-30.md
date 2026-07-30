# Rhythm SSM from a separated drum stem — 2026-07-30

Budget: ~2h. Elapsed at write-up: ~45 min (env `date` reported unusually fast
wall-clock progress through this session; treat the phase order, not the
literal minute count, as the record).

## Brief

Build a rhythmic-pattern SSM from a separated DRUMS stem, on the SAME
half-bar slot grid as `scratchpad/pattern_slide.py`'s chord SSM, and check
whether it carries independent section (verse/chorus/bridge) information —
motivated by: harmony is ambiguous between sections (they can share chords),
drums are not.

## What was built

`scratchpad/rhythm_ssm.py`:
- `separate_drums()` — demucs (htdemucs, `--two-stems=drums`, MPS device).
  Already installed in `.venv` (4.1.0), model weights already cached
  (`~/.cache/huggingface/hub/models--adefossez--HTDemucs`). ~45-50s per
  3.5min song on Apple Silicon MPS (measured: This Love 205s audio → 46.6s
  wall time). Disk: 29 GiB free before, stems (~36 MB/song) written to the
  **session tmp scratchpad**, not the repo, to avoid bloating `scratchpad/`
  with binaries — `HARMONIA_STEM_CACHE` env var overrides the path.
- `_drum_onset_bands()` — 3-band onset-strength envelope (kick/snare/hats-ish
  mel-channel split) via `librosa.onset.onset_strength_multi`.
- `build_slot_patches()` — per-slot patch: (3 bands × 8 sub-steps),
  interpolated from the envelope, L2-normalised.
- 5 comparison variants, all in `[0,1]`:
  - `baseline` — plain cosine.
  - `tolerance` — max cosine over a ±(7.5%,15%) time-shift grid (drum jitter).
  - `smooth` — Gaussian-smoothed envelope (σ=50ms) before plain cosine.
  - `pearson` — mean-centred (subtract the song-average patch) then cosine,
    rescaled `(r+1)/2`. Hypothesis: a steady 4/4 groove has a shared
    always-on backbone (kick on 1&3, hats every 8th) that inflates raw
    cosine everywhere; this was explicitly the mechanism already documented
    for chroma-DTW in `docs/known_issues.md` ("raw cosine sits at a ~0.5 DC
    floor... mean-centring is the biggest single win") — the closest prior
    art in this repo, so tested directly rather than re-derived from scratch.
  - `pearson_tolerance` — both combined.
- `scratchpad/rhythm_calibrate.py` — reproduces Louis's own calibration
  protocol for the chord SSM (same-section vs different-section slide-
  diagonal score ranges) but driven generically off `pattern_slide.
  slide_across_x`, so it runs unmodified on any S matrix. Reference labelling
  = `pattern_slide.segment()`'s chord-SSM-derived blocks (the only
  "known-good" segmentation available without hand GT beyond This Love).
- `scratchpad/rhythm_plot.py` — side-by-side PNG: chord SSM / best rhythm
  variant / baseline rhythm variant, boundaries overlaid.

## Sanity check (rule #1: verify the harness before trusting it)

Ran the calibration protocol on the **chord SSM itself** first. This Love:
same-section 0.477–1.000 (median 0.99), different-section 0.168–0.697 (median
0.40) — median gap +0.59, reproduces the range documented in
`pattern_slide.py`'s own comment block (0.76–1.00 / 0.17–0.70) closely enough
(the small discrepancy is 1st/2nd-ending pairs pulling the same-section floor
down, which that comment block itself flags as expected). Harness trusted.

## Results — same-section vs different-section, median gap (same − diff)

| song | chord SSM (reference) | rhythm: baseline | tolerance | smooth | pearson | pearson+tol |
|---|---|---|---|---|---|---|
| This Love (n_bars=80, programmed pop) | +0.590 | +0.008 | −0.010 | **+0.040** | +0.055 | −0.002 |
| Every Breath You Take (n_bars=56, live drums) | +0.055 | +0.066 | +0.128 | **+0.227** | +0.071 | +0.063 |
| Billie Jean (n_bars=71) | **degenerate — see below** | — | — | — | — | — |

Full range/IQR numbers are in the script output (`rhythm_calibrate.py`); the
table above is the single comparable number per the brief's ask.

**This Love: no variant works.** All 5 median gaps are ≈0 (−0.01 to +0.06);
raw ranges overlap completely (e.g. `tolerance` diff range 0.665–0.932 fully
swallows the same-section range 0.706–1.000). Confirmed visually: side-by-side
PNG (`docs/research_sessions/rhythm_ssm_this_love_2026-07-30.png`) shows the
chord SSM's checkerboard block structure with NO corresponding structure in
either rhythm variant — the rhythm SSM is a near-uniform bright field.
Root-caused with a raw-envelope timeline
(`/private/tmp/.../scratchpad/env_timeline.png`, not committed — see below):
all 3 bands fire at every 8th/16th-note position almost continuously across
the WHOLE song, verse and chorus alike — the song's actual production (dense,
compressed, four-on-the-floor pop) does not vary its *drum pattern* between
sections; the section cue is in the bass/synth/vocal layering, which this
feature never sees. **This is not obviously a feature-engineering failure —
it may be a real property of this song's production**, though I could not
fully rule out demucs bleed (mel-band separation on a compressed programmed
mix is a plausible confound I didn't have budget to isolate further; the
`no_drums.wav` companion stem was not inspected for leakage).

**Every Breath You Take: a real, if modest, signal, on a live-drums song.**
`smooth` gives same-section IQR [0.870,0.962] vs different-section IQR
[0.536,0.837] — visibly separated, and the side-by-side PNG
(`docs/research_sessions/rhythm_ssm_the_police_every_breath_2026-07-30.png`)
shows genuine block structure in the rhythm SSM (a distinct low-similarity
zone spanning roughly the B/C sections) that loosely tracks the chord
boundaries. Notably, rhythm's median gap (+0.227) *beats* this song's own
chord-SSM median gap (+0.055) — on this song, drums are the stronger signal.

**Billie Jean: the calibration protocol itself breaks, which is the
motivating scenario made literal.** The song is harmonically near-static
(mostly F#m throughout, famously); `pattern_slide.segment()` labels the
*entire song* as one section "A" (0 different-section pairs found) — there is
nothing to calibrate against. This is exactly Louis's premise: harmony has no
signal here. Independent check (no chord reference, just inspecting the
rhythm SSM directly): the first ~20 slots (~bars 0–10, the sparse intro before
the full groove enters — visually a distinct bright block in
`docs/research_sessions/rhythm_ssm_michael_jackson_billie_jean_2026-07-30.png`)
are more self-similar to each other than to the rest of the song, in every
variant tested — gap ranges from +0.037 (`smooth`) to +0.263 (`pearson`).
Modest, but it is a real boundary the harmonic pipeline cannot find in
principle (n_bars=71/bar_sec=4.111s is also very likely a 2-bar-unit octave
error in `rigid_grid_for` given this song's slow harmonic rhythm — flagged as
a known limitation of that function already, not a rhythm-SSM issue, but it
means the exact bar numbers above are approximate, not precise).

## What does NOT work, stated explicitly (rule #4)

- Plain cosine on raw onset-envelope patches (`baseline`): median gap ≈0 or
  even slightly negative on This Love. Confirmed root cause: NOT simply "no
  discriminative information" — the visual block structure IS present for
  Police under `baseline` too (arguably the cleanest-looking blocks of the 3
  panels) but is masked by a handful of extreme outlier pairs (same-section
  min literally 0.000) that dominate the min/max range metric — the summary
  statistic, not just the feature, needs to be outlier-robust (median/IQR,
  not min/max) before judging a variant.
- No single tolerance/smoothing/mean-centring fix is a consistent win across
  songs with only 3 data points: `smooth` wins clearly on Police, `pearson`
  wins clearest on the Billie Jean intro-block test, `pearson_tolerance`
  under-performs both on every test. **Do not read the current `variant=
  "smooth"` default as validated — it's the best of a small, inconsistent
  field.**
- Jitter-tolerance (max over time-shifts) mechanically inflates ALL pairwise
  scores (it's a max over more candidates), which is *why* it tends to shrink
  rather than grow the same/different gap — a real, generalisable mechanism,
  not specific to this feature.

## Recommendation

**Not ready to fuse into the segmenter as built.** The signal is real on one
of three songs (Police, live drums) and plausibly real but unquantified on a
second (Billie Jean, where it matters most — harmony gives zero signal there)
but is null on the primary target song (This Love). n=3 is a hypothesis-
generation sample, not a validated feature (rule #5) — before any fusion
work: (1) get bar-exact hand GT for ≥2 more songs the way This Love has it,
so Billie Jean and any dense-programmed-pop song can be calibrated properly
instead of falling back to a degenerate or unquantified reference; (2) test
whether the This Love null result is demucs bleed vs a real production
property by inspecting `no_drums.wav` and/or trying the 4-stem model instead
of 2-stem; (3) if the null holds, the fusion prior should be **confidence-
weighted by production style** (sparse/live drums → trust rhythm; dense
compressed programmed pop → down-weight it), not a fixed-weight fusion.

## Files

- `scratchpad/rhythm_ssm.py` — the module (target deliverable).
- `scratchpad/rhythm_calibrate.py` — calibration harness (reused for all 3 songs).
- `scratchpad/rhythm_plot.py` — diagnostic PNG generator.
- `docs/research_sessions/rhythm_ssm_this_love_2026-07-30.png`
- `docs/research_sessions/rhythm_ssm_the_police_every_breath_2026-07-30.png`
- `docs/research_sessions/rhythm_ssm_michael_jackson_billie_jean_2026-07-30.png`

---

## Fill detection — the reframe (2026-07-30, same session, ~1.5h follow-on)

Louis's reframe after reading the null above: "the rythmic SSM isn't enough
of itself, what we need more than pattern differences, is just to detect
fill ins, and small drum changes right before rythm changes." Stop asking
"is bar X's groove like bar Y's" (identity, washed out on This Love). Ask "is
bar X unlike its own local neighbourhood" (novelty/outlier) — a drummer fills
before a section change and often crashes on the new downbeat. Prediction to
test: This Love's near-uniform 8th/16th activity, which is *why* the identity
SSM was null, should make a fill a cleaner LOCAL outlier against a flat
baseline. **This prediction held** — the null did not reproduce here.

Reused the cached demucs stems (no re-separation). Built
`scratchpad/drum_fills.py`: `fill_scores(audio_path, bar_bounds_sec) ->
(n_bars,)`, `component_scores()` exposing 4 raw signals per bar plus 2
blends, `radius=4` local neighbourhood by default:

1. **density** — total onset energy this bar vs local median (robust z, MAD-normalised).
2. **groove_dev** — 1 − cosine(this bar's onset patch, LOCAL median patch of the neighbourhood).
3. **crash** — high-band transient at a bar's OWN downbeat, re-indexed back one bar (crash at b+1's downbeat = fill evidence for bar b).
4. **flurry** — mid-band onset PEAK COUNT excess (a fill is several close hits, not just more energy).

### Root-caused bug: a symmetric local window straddles the boundary it's trying to detect

First pass used a SYMMETRIC `[b-4, b+4]` reference window. On Every Breath
You Take this produced a clearly wrong top-4 for `groove_dev`
(`[7, 9, 10, 12]` — bar 12 is the boundary's own first bar, not the fill
target bar 11) and a strongly NEGATIVE median gap (fill-bar median 0.470 vs
other-bars median 0.905, gap **−0.435**). Root cause: a window straddling a
real boundary blends two different patterns into one "local normal", so
EVERY bar within `radius` of a boundary — on both sides — reads as deviant,
not just the true fill bar. Confirmed by inspecting raw values bar-by-bar
(alternating 0/1 across bars 8–16 with no relation to the true boundary at
12). **Fix**: `causal=True` — the reference window for bar `b` is
`[b-radius, b)` only, never bars after `b`. A causal window cannot straddle
a boundary it hasn't reached yet. This is now the default.

### Result: `groove_dev` (causal) is the one component with a reproducible signal

Same/different median-gap and AP-vs-chance, `radius=4`, causal, 3 songs
(This Love = hand GT; Every Breath You Take / Billie Jean = model-derived
boundaries from `pattern_slide.segment()`, NOT hand GT):

| component | This Love gap / AP (chance 0.081) | Every Breath gap / AP (chance 0.073) | Billie Jean gap / AP (chance 0.086) |
|---|---|---|---|
| density | +0.000 / 0.080 (1.0x) | +0.163 / 0.157 (2.1x) | +0.096 / 0.142 (1.7x) |
| **groove_dev** | **+0.069 / 0.137 (1.7x)** | **+0.089 / 0.315 (4.3x)** | **+0.080 / 0.195 (2.3x)** |
| crash | −0.057 / 0.060 (0.7x) | +0.000 / 0.162 (2.2x) | +0.139 / 0.104 (1.2x) |
| flurry | **+0.185 / 0.214 (2.6x, best on this song)** | +0.000 / 0.131 (1.8x) | +0.042 / 0.102 (1.2x) |
| combined_mean | −0.020 / 0.092 | +0.043 / 0.118 | +0.081 / 0.132 |
| combined_max | +0.040 / 0.099 | +0.033 / 0.152 | +0.107 / 0.113 |

**`groove_dev` is the only component with a positive gap on all 3 songs**,
and beats or ties every other single component's AP on 2 of 3 (loses only to
`flurry` on This Love). Neither blend (`combined_mean`, `combined_max`) beats
the best single component on ANY song — "do not just ship a blend" was the
right instruction; `fill_scores()` now defaults to `variant="groove_dev"`,
not a blend.

Louis's prediction was directionally right: This Love, the song where the
identity SSM was completely null (median gap ≈0 for every variant), is
**not** null for fill detection — `flurry` reaches 2.6x chance AP there,
`groove_dev` 1.7x. The reframe recovers signal specifically where the
original approach failed, which is the strongest evidence in this session
that novelty (not identity) is the right lens for drums.

### What does NOT work / caveats (rule #4)

- **Absolute performance is still weak.** Best precision@k across all 9
  (song × component) cells that clear chance is 0.33 (2/6, This Love
  `flurry`) — most land at 0.17–0.25 (1 hit). This is a WEAK prior (1.7–4.3x
  better than a random bar), not a reliable standalone boundary detector.
- **`groove_dev` saturates near 1.0 on Every Breath You Take specifically**
  (46–60% of post-warmup bars score >0.9 regardless of `radius` tested,
  3–10) — a live drummer's genuine bar-to-bar pattern variability compresses
  the score's dynamic range even outside real fills, which is *why* its AP
  there, while the best of the three songs, is riding on a mostly-tied
  top of the ranking rather than a clean separation (visible in the PNG:
  the green `groove_dev` curve sits at 0.8–1.0 almost everywhere after bar
  4). This did NOT happen on This Love or Billie Jean (0% saturation at any
  radius tested) — genre/performance-style dependent, same axis as the
  identity-SSM's This Love-vs-Police split above.
- **`crash` is the weakest and noisiest component** on 2/3 songs (near or
  below chance, visibly spiky with no boundary relation in the PNGs) —
  the "crash on the new downbeat" cue, plausible in music theory, is not
  reliably recovered by a raw high-band transient-at-downbeat measure
  through a demucs drum stem at this budget.
- `radius` swept 3/4/6/8/10 on all 3 songs (`groove_dev` only, causal): no
  value is a clean, universal winner (This Love flat ~0.11-0.12 AP across
  all radii; Police peaks at radius=4 (0.315); Billie Jean peaks at
  radius=3 (0.213)). `radius=4` was kept as a reasonable, not tuned, default.
- Every Breath You Take / Billie Jean boundaries are model-derived
  (`pattern_slide.segment()`), not hand GT — for Billie Jean in particular
  these are loop-length changes within a single harmonically-static "A"
  label, not true section changes, so treat that column as the weakest
  evidence of the three.

### Recommendation

`groove_dev` (causal local-neighbourhood cosine deviation) is a real,
reproducible, but WEAK boundary prior — worth wiring as a small additive
boost to boundary probability at bar `b+1` when `fill_scores()[b]` is high,
fused with the existing chord-SSM boundary signal, not as a standalone
detector or a hard gate. Before fusing: get hand GT boundaries for Every
Breath You Take and Billie Jean (their current model-derived references are
the weakest link in every number above) so the AP/gap numbers can be
trusted rather than treated as directional.

### Files (fill detection)

- `scratchpad/drum_fills.py` — `fill_scores()` / `component_scores()` (target deliverable).
- `scratchpad/drum_fills_eval.py` — precision@k / AP / PR-curve / same-different-gap harness + PNG generator.
- `docs/research_sessions/drum_fills_this_love_2026-07-30.png`
- `docs/research_sessions/drum_fills_the_police_every_breath_2026-07-30.png`
- `docs/research_sessions/drum_fills_michael_jackson_billie_jean_2026-07-30.png`
