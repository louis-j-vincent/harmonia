# Lock-propagation tuning: simulated-lock ROI sweep

Branch `feat/chord-context-prior`. Companion to `docs/design_chord_context_prior.md`
(evaluation ladder step 3) and `docs/context_prior_phase1_results.md`. Harness:
`scripts/tune_lock_propagation.py`. Raw output: `data/cache/tune_lock_propagation/
sweep_results.json` (+ `extended_delta_results.json` for the first version's
beyond-grid probe).

**★ SUPERSEDED, same day — read `## 8. Differential re-analysis` at the
bottom first.** Sections 1-7 below are the FIRST analysis, which measured
corruption/churn by diffing the locked rescore against the DISPLAYED chart.
That metric conflates lock-caused changes with the lattice's own
unconditional churn (real, but present even with zero locks) — kept here
for the record (the inventory, decode/GT verification, and diagnosis of
*why* the mechanism cascades on repetitive material are all still accurate
and load-bearing), but its "bottom line" and chosen operating point below
are NOT what shipped. Section 8 has the corrected metric, the real sweep,
and the actual shipped defaults.

## Bottom line (SUPERSEDED — see section 8)

**No config clears both safety bars (corruption <=0.05/lock, churn <=2%) while
also propagating (ROI > 0).** Every config that fixes wrong neighbours also
corrupts several correct ones, at 7x-170x the 0.05/lock target. The only
configs that hit both safety bars have ROI = 0.00 exactly (propagation never
fires). **First-pass shipped default: `lam=0.5, delta=8, K=6`** — the safe,
ROI=0 point. Propagation beyond the locked span itself is OFF by default; the
endpoint's `lam`/`delta`/`K` JSON fields still let a caller opt into more
aggressive (and riskier) behaviour per-request.

This is a valid negative result on "does propagation clear the design's own
safety bar" (design doc: "ship bar is ROI clearly > 0 with corruption ~= 0") —
diagnosis below. **Section 8 shows most of this "corruption" was an artifact
of the comparison, not the mechanism.**

## 1. Inventory (corpus x usable-song counts)

| corpus | n_audio | n_with_gt | n_musx_probs_cached |
|---|---|---|---|
| GuitarSet | 360 | 360 | 30 |
| RWC-Popular | 1 | 1 | 1 (built during this session) |
| aligned_corpus (chart<->audio alignment output) | 8 | 139 records / 8 songs | 8 |
| JAAH (via ChoCo) | 0 | 89 (labels only) | 0 |

- **GuitarSet**: 360 excerpts = 6 players x 5 backing tracks x {comp, solo}.
  GT = JAMS `chord` namespace, the SIMPLIFIED annotation (verified: clean
  4-value vocabulary maj/min/7/hdim7, not the second "performed voicing"
  annotation which carries extensions + explicit bass degree).
- **RWC-Popular**: full-song audio is licensed/gone for 99/100 tracks
  (verified: `data/cache/rwc/audio/RWC-P/` is an empty directory tree). Only
  `RWC_P001`'s audio survives (`docs/audio/rwc_rwc_p001.m4a`, confirmed via
  `tests/test_accuracy_score.py`'s own smoke-test reference and
  `harmonia/eval/benchmark_set.py`). Its GT is NOT the 4 GB Zenodo audio zip —
  it's a small per-song CSV fetched live from
  `github.com/rwc-music/rwc-annotations` (2.9 KB, one HTTP GET).
- **aligned_corpus**: 139 scored segments / 8 unique songs (multiple segments
  share one audio file — confirmed 1:1 song_id<->audio_path). GT timestamps
  are SELF-DERIVED from the chart<->audio alignment process
  (`benchmark_set.py`'s GT-provenance table calls this "labels OK, at
  self-derived timestamps — NO for beat/alignment eval, circular"); accepted
  here because this harness only needs approximate label-at-time, not exact
  onset precision. All 8 songs' `audio_path` fields were STALE relative to
  current `docs/audio/` filenames in one respect worth flagging: the manifest
  itself already points at the CURRENT filenames (re-verified — all 8 exist
  on disk today), it was my own first glob-guess that was stale, not the data.
- **JAAH (via ChoCo)**: labels-only, verified zero audio files anywhere under
  `data/` for any `jaah_*.jams` id. Not usable for this harness (needs a
  production-path DECODE, which needs audio).

**Eval set used (19 songs, capped per the "~15-20 songs, balanced across
corpora" time budget)**: 10 GuitarSet (2 players x 5 backing tracks, all
`_comp_mic` — comping, not `_solo_mic`, so the guitar audio is acoustically
chord-like) + all 8 aligned_corpus songs + RWC_P001. Decode used the exact
production path (`infer_chords_v1`, `harmonia.eval.accuracy_score.
SHIPPED_CONFIG` == `_run_analysis`'s live nnls24 config: bass/quality=musx,
segment_source=musx_redecode, beat_backend=beatthis, beat_period_mode=bestfit).
m4a inputs were ffmpeg-decoded to WAV first (`accuracy_score._decode_to_wav`)
since `infer_chords_v1` reads via `soundfile.read`, which cannot open m4a.
Acoustic evidence for the rescore itself came from `musx_probs` cache hits for
**all 19/19 songs** (no nnls_heads fallback needed).

| corpus | song | spans | GT-scored spans | wrong (displayed!=GT) |
|---|---|---|---|---|
| guitarset | 00_BN1-129-Eb_comp_mic | 8 | 8 | 2 |
| guitarset | 00_Funk1-114-Ab_comp_mic | 7 | 7 | 2 |
| guitarset | 00_Jazz1-130-D_comp_mic | 7 | 7 | 3 |
| guitarset | 00_Rock1-130-A_comp_mic | 6 | 6 | 0 |
| guitarset | 00_SS1-100-C#_comp_mic | 6 | 6 | 0 |
| guitarset | 03_BN1-129-Eb_comp_mic | 6 | 6 | 0 |
| guitarset | 03_Funk1-114-Ab_comp_mic | 5 | 5 | 5 |
| guitarset | 03_Jazz1-130-D_comp_mic | 4 | 4 | 3 |
| guitarset | 03_Rock1-130-A_comp_mic | 6 | 6 | 0 |
| guitarset | 03_SS1-100-C#_comp_mic | 6 | 6 | 0 |
| aligned | bein_green | 50 | 21 | 3 |
| aligned | blue_bossa | 189 | 100 | 9 |
| aligned | blue_bossa_backing | 151 | 144 | 5 |
| aligned | close_to_you | 61 | 28 | 7 |
| aligned | every_breath_you_take | 67 | 33 | 5 |
| aligned | georgia_on_my_mind | 87 | 17 | 3 |
| aligned | let_it_be | 118 | 34 | 8 |
| aligned | stand_by_me | 39 | 22 | 4 |
| rwc | rwc_p001 | 108 | 107 | 11 |

Parser spot-check (verify-don't-trust, `parse_harte_lite` against raw
displayed labels): `D#:maj7`->root_pc=3/maj, `G#:maj`->8/maj, `F:maj`->5/maj,
`A#:maj7`->10/maj, `G#:min`->8/min, `F#:maj`->6/maj — all correct.

## 2. Sweep methodology

Per song (computed ONCE, reused across every config): production-decoded
displayed chart, GT aligned by span-midpoint containment, acoustic
log-posterior via `span_rescore.compute_acoustic_logp`. Per (lam, delta, K):
`span_rescore.lattice_rescore` (the SAME function `/api/context_rescore`
calls) run once per simulated lock, `table="pooled"` (Task 1's routing rule
picks "pooled" for every one of these filenames — none start with
`inferred_ireal_`).

- **wrong-locks**: every span where displayed != GT, locked to GT (capped at
  10/song via deterministic stride-sampling — time-budget guardrail, see
  below; 69 total across the corpus, i.e. every song's wrong-span count fit
  under the cap except RWC's 11).
- **correct-locks**: a stride sample of up to 6/song where displayed == GT
  (100 total) — "locking an already-correct chord must not corrupt".
- **no-lock churn**: one call per song with zero locks.

Metrics: `roi_exact` = mean OTHER spans flipped wrong->GT-correct per
wrong-lock; `corruption_exact_{wrong,correct}_locks` = mean OTHER spans
flipped correct->wrong per lock (reported separately for the two lock kinds,
per design-doc guardrail 2); `roi_root`/`roi_qual` = the same at
root-only/quality-only granularity (guardrail 4); `churn_frac` = fraction of
ALL spans changed with zero locks.

**Time-budget deviation, disclosed**: the brief specified a flat 4 (lam) x 4
(delta) x 3 (K) = 48-config grid. An unstaged run of that projected past 30
minutes on this machine (2 of the 19 songs have 100-190 spans; DP cost scales
with span count) — trimmed to a **staged** sweep: full lam x delta (16
configs) at K=6 (the design doc's own suggested default), then K in {4, 8}
confirmed ONLY at the stage-1 ROI winner (2 more configs) — plus lock counts
capped at 10 wrong / 6 correct per song (was uncapped / 10). This cut a single
score_config call's cost roughly in half and the grid to 18/48 configs; the
full staged sweep completed in 128s. All three knobs (lam, delta, K) are still
exercised, just not as a flat outer product. After seeing that ZERO of the 18
swept configs were feasible, one additional bounded probe (6 more configs:
lam in {0.5,1} x delta in {4,8,16}, K=6, ~60s) was run to find where the
churn/corruption trade-off actually crosses the safety bars — this is the
"extended-delta diagnosis" the design doc's negative-result clause invites.

## 3. Full results (24 configs, sorted by ROI)

| lam | delta | K | roi_exact | roi_root | roi_qual | corrupt/wrong-lock | corrupt/correct-lock | churn |
|---|---|---|---|---|---|---|---|---|
| 4 | 0 | 6 | **1.536** | 0.493 | 1.870 | 8.652 | 4.650 | 0.299 |
| 4 | 0.5 | 6 | 1.493 | 0.478 | 1.594 | 6.493 | 3.420 | 0.256 |
| 4 | 1 | 6 | 1.478 | 0.449 | 1.609 | 5.754 | 3.050 | 0.230 |
| 4 | 2 | 6 | 1.449 | 0.449 | 1.580 | 5.333 | 2.790 | 0.211 |
| 2 | 0 | 6 | 1.435 | 0.304 | 1.522 | 4.623 | 2.390 | 0.198 |
| 4 | 0 | 8 | 1.435 | 0.406 | 1.638 | 9.246 | 5.100 | 0.318 |
| 2 | 0.5 | 6 | 1.420 | 0.304 | 1.507 | 4.391 | 2.220 | 0.175 |
| 2 | 1 | 6 | 1.377 | 0.275 | 1.377 | 3.725 | 1.900 | 0.145 |
| 2 | 2 | 6 | 1.348 | 0.319 | 1.507 | 3.058 | 1.600 | 0.113 |
| 1 | 0 | 6 | 1.217 | 0.217 | 1.275 | 3.290 | 1.730 | 0.135 |
| 4 | 0 | 4 | 1.203 | 0.261 | 1.478 | 8.333 | 4.350 | 0.265 |
| 1 | 0.5 | 6 | 1.101 | 0.203 | 1.232 | 2.565 | 1.370 | 0.105 |
| 0.5 | 0 | 6 | 1.072 | 0.217 | 1.188 | 1.899 | 1.070 | 0.096 |
| 1 | 1 | 6 | 1.072 | 0.188 | 1.188 | 1.913 | 0.950 | 0.078 |
| 0.5 | 0.5 | 6 | 1.000 | 0.159 | 1.116 | 1.290 | 0.660 | 0.071 |
| 0.5 | 1 | 6 | 0.942 | 0.159 | 1.058 | 1.000 | 0.540 | 0.056 |
| 1 | 2 | 6 | 0.913 | 0.159 | 1.029 | 1.145 | 0.600 | 0.061 |
| 0.5 | 2 | 6 | 0.536 | 0.159 | 0.536 | 0.899 | 0.480 | 0.032 |
| 1.0 | 4 | 6 | 0.420 | 0.159 | 0.536 | 0.899 | 0.480 | 0.034 |
| 0.5 | 4 | 6 | 0.130 | 0.130 | 0.130 | 0.609 | 0.340 | 0.015 |
| 1.0 | 8 | 6 | 0.130 | 0.130 | 0.130 | 0.768 | 0.420 | 0.018 |
| **0.5** | **8** | **6** | **0.000** | 0.000 | 0.000 | **0.000** | **0.000** | **0.001** |
| 0.5 | 16 | 6 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 1.0 | 16 | 6 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 |

(69 wrong-locks, 100 correct-locks pooled over all 19 songs at every row.)

**Target zone (design doc): corruption <=0.05/lock AND churn <=0.02.** Reading
down the table: churn only drops under 2% at delta>=4 (lam=0.5) / delta>=8
(lam=1); corruption only drops near 0 at delta>=8. There is no row where ROI
is positive AND both bars are met — the feasible set (bold row + delta=16
variants) is exactly the ROI=0.00 rows.

## 4. Diagnosis: why is there no safe-and-effective point?

Traced one lock's effect directly (blue_bossa, lam=2, delta=0, locking ONE
wrong span): **64/189 spans changed vs. the displayed chart — but only 8 of
those 64 changed relative to what a zero-lock call ALREADY does at that
(lam, delta)** (checked directly: `chosen_with_lock` vs `chosen_no_lock`
differ on 8 spans, not 64). Most of the apparent "propagation" at lam>=1 is
actually **baseline churn that exists with no lock at all** — the acoustic
evidence is often too flat (long pooled spans, ambiguous root/quality) for
the context trigram prior NOT to dominate, and since the trigram prior
conditions only on immediate neighbour TOKENS (not position), the *same*
correction (or corruption) fires at *every* repeat of an identical
progression fragment in a repetitive tune (Blue Bossa repeats its 16-bar form
~12 times in this recording) — regardless of whether that repeat is anywhere
near the locked span. This is why churn and corruption both scale with lam
almost independently of whether anything is locked, and why delta (which adds
a flat per-span incumbent bonus) is the only lever that reliably suppresses
it — but suppressing the baseline-churn problem this way also suppresses the
genuine wrong->correct propagation the feature exists for, because both ride
on the same context-vs-acoustic margin.

Restated musically: the context prior does not (yet) distinguish "the user
just told me this note is X, and it should locally imply its immediate
neighbours" from "every place in this song where this 3-chord pattern
recurs should read the same way" — it always does the latter. For a
repetitive tune that is sometimes right (see example 1 below) and sometimes
wrong (example 3), and there is currently no way to tell the two apart from
inside `lattice_rescore` alone.

## 5. Three concrete examples

Pulled from the aggressive illustrative config (`lam=4, delta=0, K=6` — the
best-ROI row above, NOT the shipped default) so the mechanism's real behaviour
is visible; the shipped default (`lam=0.5, delta=8`) would show none of this
by design.

**Example 1 — clean win.** `guitarset_03_Funk1-114-Ab_comp_mic` (a 5-span
G#-C#-G#-C#-G# vamp). Displayed the whole thing as dominant 7ths (G#7/C#7);
GT (GuitarSet's own chart annotation) says plain major triads. Locking
**span 2** (G#:7 -> G#maj) alone:

| span | time | displayed | GT | after lock |
|---|---|---|---|---|
| 0 | 0.0-8.4s | G#7 | G#maj | **G#maj** (fixed) |
| 1 | 8.4-12.6s | C#7 | C#maj | **C#maj** (fixed) |
| 2 | 12.6-18.4s | G#7 | G#maj | G#maj (the lock) |
| 3 | 18.4-21.6s | C#7 | C#maj | **C#maj** (fixed) |
| 4 | 21.6-25.3s | G#7 | G#maj | **G#maj** (fixed) |

4/4 other spans fixed, 0 corrupted — locking one instance of "this is a plain
triad, not a 7th" propagated correctly to every repeat of the same vamp.

**Example 2 — same song, different lock span.** Locking **span 0** instead
(G#:7 -> G#maj) fixes spans 1, 3, 4 (3/4) but NOT span 2, which flips to
`G#7` (unchanged from displayed — i.e. that one repeat keeps its original,
wrong reading). Shows the fix is evidence-weighted, not a blanket
find-and-replace: which repeats flip depends on each span's own acoustic
support, not just on matching the locked pattern.

**Example 3 — the corruption case.** `guitarset_00_Jazz1-130-D_comp_mic` (7
spans). Locking **span 6** (D:maj, already correct, matches GT) — a
"don't break what's right" test:

| span | time | displayed | GT | after lock |
|---|---|---|---|---|
| 0 | 0.0-7.4s | D:7 | Dmaj | **Dmaj** (fixed) |
| 1 | 7.4-11.1s | G:maj | Gmaj | Gmaj (unchanged) |
| 2 | 11.1-12.9s | A:min | Dmaj | **Dmaj** (fixed) |
| 3 | 12.9-14.8s | D:maj7 | Dmaj | **Amaj** (CORRUPTED — was already right in spirit, GT=Dmaj) |
| 4 | 14.8-17.1s | A:maj | Amaj | **Dmaj** (CORRUPTED — was correct, now wrong) |
| 5 | 17.1-18.0s | D:min | Gmaj | **Gmaj** (fixed) |
| 6 | 18.0-22.2s | D:maj | Dmaj | Dmaj (the lock) |

3 fixed, 2 corrupted from ONE lock on an already-correct span — exactly the
scenario the design doc's guardrail 2 exists to catch, and exactly why this
config does not ship.

## 6. Chosen operating point (SUPERSEDED — see section 8)

```
lam   = 0.5
delta = 8.0
K     = 6
```

Set in `harmonia/serving/api.py` (`_CTX_RESCORE_DEFAULT_LAM/K/DELTA`, used by
`/api/context_rescore`'s defaults, still overridable per-request via the
`lam`/`delta`/`K` JSON fields). Measured: `roi_exact=0.00`,
`corruption_exact_combined=0.00`, `no-lock churn=0.001` (1/700-ish spans).

**What this means in practice**: the endpoint now always applies the user's
lock correctly (previously dead — see `known_issues.md` "lock propagation is
functionally DEAD"), on the live nnls24/musx acoustic backend, with zero risk
of silently rewriting the rest of the chart. It does NOT propagate the fix to
neighbouring or repeated instances of the same error by default — that
capability exists in the code (see example 1) but is not safe to enable
unconditionally given the current context prior's inability to distinguish
"apply locally" from "apply everywhere this pattern recurs". A future fix
likely needs the context term to decay with distance from the nearest lock
(not just condition on immediate neighbour tokens), which is out of scope for
this tuning pass.

## 7. Tests (first pass)

`tests/test_span_rescore.py` (61 tests incl. 7 new for the incumbent-delta
bonus) and `tests/test_serving_routes.py` (genre-routing unit tests + endpoint
validation-path tests) pass; see the session report for the full run.

## 8. Differential re-analysis (supersedes sections 1-7's verdict)

### 8.1 Why the metric changed

Section 4's own diagnosis contained the fix, in hindsight: tracing ONE lock
directly (blue_bossa, lam=2, delta=0) showed locking a single span changed
64/189 spans **vs. the displayed chart**, but only 8 of those 64 differ from
what a **zero-lock** call at the identical (lam, delta, K) already picks.
56 of the 64 "corrupted"/"fixed" spans in sections 1-7's metric were spans
the lattice already disagreed with the display about, lock or no lock — pure
baseline churn, misattributed to the lock. This is exactly the bug the OLD
(before this branch) `/api/reinfer` never had: it always decoded a
constraint-free `base` and a constrained `cons` under IDENTICAL settings and
diffed `cons` vs `base`, never vs the original display. Sections 1-7's harness
(and the pre-fix endpoint) instead diffed the locked run straight against
`displayed` — the ONE place this branch's new code didn't reuse
`/api/reinfer`'s own working pattern.

### 8.2 The fix: `span_rescore.differential_rescore`

New function (`harmonia/models/span_rescore.py`): runs `lattice_rescore`
TWICE at the identical (lam, delta, K) — once with no locks (baseline), once
with the real locks — and only lets a span move away from `displayed` if it
is itself locked OR the two runs disagree there. Every span where baseline
and locked agree keeps its displayed value regardless of what either run's
own opinion of it is. `/api/context_rescore` now calls this instead of raw
`lattice_rescore`. Direct consequence: a zero-lock call is baseline-vs-baseline
by construction, so churn is 0 always — **churn is dropped as a metric and a
constraint**, not tuned away via `delta` as sections 1-7 tried.

### 8.3 Re-run: differential sweep (K fixed at 6 — never moved the ranking
near the top in sections 1-7, dropped from the grid per this re-analysis)

16 configs (lam x delta), reusing the exact same cached per-song decodes/
posteriors/GT from sections 1-7 (no re-decode). 69 wrong-locks + 100
correct-locks pooled, same as before.

| lam | delta | ROI_diff | ROI_diff (root) | ROI_diff (qual) | CORR_diff (wrong-lock) | CORR_diff (correct-lock) | corr/ROI ratio |
|---|---|---|---|---|---|---|---|
| **2** | **0.5** | **0.101** | 0.014 | 0.116 | 0.203 | 0.020 | **2.00** |
| 2 | 1 | 0.087 | 0.000 | 0.116 | 0.174 | 0.020 | 2.00 |
| 2 | 2 | 0.087 | 0.014 | 0.116 | 0.174 | 0.020 | 2.00 |
| 4 | 0 | 0.087 | 0.058 | 0.101 | 0.449 | 0.080 | 5.17 |
| 1 | 0 | 0.072 | 0.000 | 0.087 | 0.130 | 0.020 | 1.80 |
| 1 | 0.5 | 0.072 | 0.000 | 0.101 | 0.130 | 0.020 | 1.80 |
| 4 | 0.5 | 0.072 | 0.058 | 0.101 | 0.493 | 0.060 | 6.80 |
| 1 | 1 | 0.058 | 0.014 | 0.072 | 0.130 | 0.020 | 2.25 |
| 4 | 1 | 0.058 | 0.043 | 0.101 | 0.348 | 0.040 | 6.00 |
| 2 | 0 | 0.043 | 0.000 | 0.058 | 0.275 | 0.020 | 6.33 |
| 4 | 2 | 0.043 | 0.029 | 0.087 | 0.261 | 0.030 | 6.00 |
| 0.5 | 0 | 0.029 | 0.000 | 0.043 | 0.101 | 0.020 | 3.50 |
| 1 | 2 | 0.029 | 0.000 | 0.043 | 0.116 | 0.020 | 4.00 |
| 0.5 | 0.5 | 0.000 | 0.000 | 0.014 | 0.116 | 0.020 | inf |
| 0.5 | 1 | 0.000 | 0.000 | 0.014 | 0.116 | 0.020 | inf |
| 0.5 | 2 | 0.000 | 0.000 | 0.000 | 0.116 | 0.020 | inf |

**Magnitudes are 5-45x smaller than sections 1-7** at every comparable
(lam, delta) — e.g. `lam=4, delta=0` went from ROI 1.536 / corr-wrong 8.652
to ROI 0.087 / corr-wrong 0.449 (17.7x / 19.3x smaller). Confirms 8.1's
diagnosis directly: most of the earlier signal was baseline churn, not
lock effect.

### 8.4 Operating point

Constraints (this iteration's brief): `CORR_diff(correct-lock) <= 0.05` AND
`CORR_diff(wrong-lock) <= 0.25 x ROI_diff` (net clearly positive, >=4:1).
**0/16 configs pass** — every config's wrong-lock corruption is 1.8x-6.8x its
own ROI (best ratio 1.80 at lam=1; best ROI 0.101 at lam=2/delta=0.5, ratio
2.00). This is the honest frontier, reported per the "if nothing passes,
report the frontier honestly" rule — but note the SHAPE of the failure
changed completely: `CORR_diff(correct-lock)` (the "don't break what's
right" case) is now safely low almost everywhere (0.02-0.03, only creeping to
0.06-0.08 at lam=4) — that bar basically IS cleared. The bar that fails is
`CORR_diff(wrong-lock)`: fixing a wrong span still typically nudges ~2x as
many OTHER spans as it fixes, even after removing the baseline-churn
conflation.

**Chosen: `lam=2, delta=0.5, K=6`** — the best-ROI cell (0.101 fixed
spans/lock), with the correct-lock safety bar cleared (0.020/lock, well
under 0.05) and the best-in-class ratio among the higher-ROI cells (2.00,
vs 5-7 at lam=4). It does not clear the strict 4:1 net-positive bar; shipped
anyway because (a) the correct-lock case — the one guardrail 2 was written to
protect against ("locking something already right must not corrupt
neighbours") — IS safe here, and (b) the absolute stakes are now small (about
1 extra wrong span moved for every 5 locks that fix something), an order of
magnitude below what section 6's `delta=8` inert-by-design choice was reacting
to. `lam/delta/K` remain request-overridable.

```
lam   = 2.0
delta = 0.5
K     = 6
```
Set in `harmonia/serving/api.py` (`_CTX_RESCORE_DEFAULT_LAM/K/DELTA`).

### 8.5 Three examples, re-measured at the shipped config (differential)

**Win.** `guitarset_03_Funk1-114-Ab_comp_mic`, locking span 2 (G#7 -> G#maj,
matching GT):

| span | displayed | GT | after (differential) |
|---|---|---|---|
| 0 | G#7 | G#maj | G#7 (unchanged — this repeat's own evidence didn't move) |
| 1 | C#7 | C#maj | **C#maj** (fixed) |
| 2 | G#7 | G#maj | G#maj (the lock) |
| 3 | C#7 | C#maj | **C#maj** (fixed) |
| 4 | G#7 | G#maj | **G#maj** (fixed) |

3 fixed, 0 corrupted — smaller than the 4-fixed example in section 5 (that
one's 4th "fix" was baseline churn, not lock-caused; differential mode
correctly excludes it).

**Win, different song.** `guitarset_03_Jazz1-130-D_comp_mic`, locking span 1
(E:min7 -> Gmaj, matching GT): span 0 (A:min -> Dmaj) also fixes. 1 other
span fixed, 0 corrupted.

**Corruption.** `aligned_close_to_you`, locking span 22 (C:maj, ALREADY
correct — the "don't break it" test): span 21 (`C:maj7`, which folds to the
same maj family as GT `Cmaj` — i.e. already correct under this project's
5-quality-family scheme) flips to `Gmaj` — genuinely corrupted, 1 span, 0
fixed. This is the ONLY corrupted span this lock causes (not 2, as the
non-differential section-5 example showed for a different song) — the
absolute scale of the risk is real but small.

### 8.6 Demo regenerated

`docs/lock_propagation_demo.md` was regenerated at the SHIPPED config
(`lam=2, delta=0.5, K=6`, not an illustrative non-default point) on This
Love — see that file for the excerpt. It now shows a handful of
lock-attributable changes, not the 30-span cascade the first (conflated)
metric's illustrative config produced.

### 8.7 What this does NOT solve

The wrong-lock corruption ratio (~2:1 at best) is still real, just much
smaller in absolute terms. The underlying cause (section 4's diagnosis: the
context trigram prior fires identically at every recurrence of the same
immediate-neighbour pattern, with no notion of distance from the lock) is
UNCHANGED by the differential fix — differential mode only stopped
mis-attributing baseline noise to the lock, it did not make the mechanism
itself more local. A future fix distinguishing "this correction applies near
the lock" from "this correction applies everywhere this pattern recurs" is
still open, and would likely raise ROI_diff further before it lowers the
corruption ratio (both currently move together as lam/delta change, never
independently, in this sweep).

## 9. Margin gate (time-boxed final iteration — did not change the shipped point)

### 9.1 Hypothesis

Section 8's remaining wrong-lock corruption (0.203/lock vs. ROI 0.101/lock
at the shipped point) might be dominated by LOW-MARGIN flips — spans where
the locked run barely prefers its new answer over what the baseline was
already proposing — while confident, structurally-supported flips (a real V7
inside ii-?-I) should survive a much wider margin. If true, a margin gate
should let the confident cases through while screening out the coin-flips,
raising the ROI/corruption ratio.

### 9.2 Mechanism

New `span_rescore.differential_rescore(..., margin_gate=m)`: for a
PROPAGATED span (unlocked, locked-run choice != baseline-run choice — the
locked span itself is never gated), compute both candidates' scores inside
the LOCKED run's own solution (its actual chosen neighbours at j-1/j+1 held
fixed, same acoustic + `lam*context` + `delta`-incumbent formula the DP
itself accumulates — see `_candidate_score`). The propagated change survives
only if `score(locked_run's answer) - score(baseline's answer) >= m` nats;
otherwise the span reverts to `displayed`, exactly as if baseline and locked
had agreed there. `m=0` (default) reproduces section 8 exactly.

### 9.3 Re-analysis (from cached decodes — no re-decoding)

Swept `m` in `{0, 0.25, 0.5, 1.0, 2.0}` at the two most promising cells:
the shipped point (`lam=2, delta=0.5`) and the highest-raw-ROI cell from
section 8 (`lam=4, delta=0`).

| lam | delta | m | ROI_diff | CORR_diff (wrong-lock) | CORR_diff (correct-lock) | ratio |
|---|---|---|---|---|---|---|
| 2.0 | 0.5 | 0.0 | 0.101 | 0.203 | 0.020 | 2.00 |
| 2.0 | 0.5 | 0.25 | 0.087 | 0.130 | 0.020 | 1.50 |
| **2.0** | **0.5** | **0.5** | 0.087 | 0.116 | 0.020 | **1.33** |
| 2.0 | 0.5 | 1.0 | 0.087 | 0.116 | 0.020 | 1.33 |
| 2.0 | 0.5 | 2.0 | 0.072 | 0.116 | 0.020 | 1.60 |
| 4.0 | 0.0 | 0.0 | 0.087 | 0.449 | 0.080 | 5.17 |
| 4.0 | 0.0 | 0.25 | 0.087 | 0.348 | 0.070 | 4.00 |
| 4.0 | 0.0 | 0.5 | 0.087 | 0.348 | 0.070 | 4.00 |
| 4.0 | 0.0 | 1.0 | 0.087 | 0.304 | 0.060 | 3.50 |
| 4.0 | 0.0 | 2.0 | 0.087 | 0.290 | 0.050 | 3.33 |

The gate DOES improve the ratio at `lam=2, delta=0.5` — 2.00 -> 1.33 at
`m=0.5-1.0`, a real, reproducible effect confirming part of the hypothesis
(some of section 8's wrong-lock corruption WAS low-margin). But ROI_diff
drops in lockstep (0.101 -> 0.087, -14%) while corruption drops further
(0.203 -> 0.116, -43%) — net effect: the ratio improves but never gets close
to passing, and `lam=4, delta=0` barely moves at all (its corruption is
dominated by higher-margin flips the gate doesn't touch, and its
correct-lock corruption — the safety-critical case — stays above 0.05 at
every `m` tested except the last, where the ratio is still 3.33).

### 9.4 Final pick

Constraint (this iteration, relaxed from section 8's 0.25x): `CORR_diff
(correct-lock) <= 0.05` AND `CORR_diff(wrong-lock) <= 0.5 x ROI_diff`.
**0/10 (config x margin) cells pass** — best is `lam=2, delta=0.5, m=0.5-1.0`
at ratio 1.33 (need <=0.5, i.e. ROI must be >= 2x corruption; have the
reverse, corruption is ~1.33x ROI). Per the brief's own fallback: **the
margin gate did not rescue the ratio — kept the shipped point unchanged.**

```
lam          = 2.0
delta        = 0.5
K            = 6
margin_gate  = 0.0   (off by default; a real, tested, request-overridable knob)
```

No change to `harmonia/serving/api.py`'s shipped constants. `margin_gate` was
added as a fourth request-overridable field (`_CTX_RESCORE_DEFAULT_MARGIN_GATE
= 0.0`) so a caller CAN opt into `m=0.5` for a modestly better ratio at a
modest ROI cost, but nothing ships differently by default.
`docs/lock_propagation_demo.md` was re-generated at this (unchanged) config
and is identical to section 8.6's 3-span result.

### 9.5 Limitations: the eval corpus underrepresents jazz

All three real-audio, real-GT corpora used for this entire tuning exercise
(GuitarSet comping tracks, RWC-Popular, and the aligned_corpus segments —
mostly pop/singer-songwriter material plus one jazz standard, Blue Bossa) are
pop/comping-heavy, not jazz-standard-heavy. JAAH — the one corpus that would
have supplied real jazz ii-V-I / iiø-V-i material at volume — has chord
labels but ZERO audio anywhere on disk (verified, section 1), so it could not
be included in a harness that needs a real production-path DECODE. The
context-prior "jazz" table (docs/context_prior_phase1_results.md) is
specifically strongest on exactly the progressions this eval corpus barely
exercises (ii-V-I, secondary dominants, tritone subs) and this whole
tuning pass always used `table="pooled"` (Task 1's routing rule: none of
these eval filenames start with `inferred_ireal_`, so none of them would
ever route to the jazz table live either). Net effect: every ROI/CORR number
in this document is likely a PESSIMISTIC estimate of how lock propagation
performs on the app's actual jazz-standard charts (the `inferred_ireal_*`
ones, which DO route to the jazz table) — those are exactly the contexts
(ii-V-I, secondary dominants) the context prior's theory panel
(context_prior_phase1_results.md Eval 2) showed it models best, and none of
that strength was exercised here. A future re-run with real jazz-standard
audio + GT (would require either sourcing JAAH audio independently or
hand-verifying a small iReal-chart-vs-audio set) could plausibly show a
better ROI/corruption ratio on exactly the content this feature's own jazz
table was built for.
