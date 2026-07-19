# Henny & Gingerale — Occam / N-chord / loop-pooling session — 2026-07-19

Slug `mayer_hawthorne_henny_gingerale`, video `gmfcYli6vV4`, key A major.
Budget 4h. Three user-reported problems; make the chart the simplest pattern
consistent with the evidence.

## Brief (restated)
1. **N no-chord propagation** (root-caused, top priority): musx `.lab` says
   `0.0→18.3s N`; `_infer_nnls24` treats `mx_root<0` as "musx unavailable" and
   falls to the NNLS heads which always invent a chord. Fix: musx N → first-class
   no-chord (empty/N.C. cell, conf 0, excluded from section content); NNLS-only
   path gets a chroma energy/flatness N gate; calibrator clamps conf 0 on N;
   app_shell renders N empty.
2. **Loop-phase evidence pooling**: A section is A|Bm7 2-bar loop; chart
   interleaves spurious E7s. Pool same-loop-phase bars within the same family;
   a deviation survives only if its own evidence beats the pooled consensus by a
   clear margin.
3. **Occam post-pass**: after decode, find the minimal repeating pattern
   (loop period + families from barlocked) explaining the sequence; re-emit as
   that pattern, keeping deviations only with a decisive log-likelihood margin.
   Uses ONLY the song's own structure (corpus LM/grammar priors are dead here).

## Baseline (in-process production decode, stem-keyed gmfcYli6vV4.wav)
Params: feature_frontend=nnls24, bass=musx, quality=musx, seg=nnls (server prod).
- key A major, tempo 123.4, **117 chords**, sections `IBABAB...` (barlocked found
  Intro + A/B loop but A/B letters swapped — known limitation).
- **Intro 0-18.3s (= musx N span) = dense junk**: B:hdim7/D#, G#:hdim7/C, D:7,
  E:min, A#:dim/F, C#:aug/B, F#:7/D#, F:7/A#... 22-69% displayed conf. Confirms
  Part 1 root cause exactly.
- Post-intro: A|Bm7 loop present but polluted with spurious 1-beat chords
  (Part 2/3).

Artifacts: `scratchpad/baseline_henny.py`, stem-keyed audio in scratchpad.

## Part 1 — N no-chord propagation — DONE + TESTED (in-process chain)
Files:
- `harmonia/models/musx_bass.py::no_chord_per_segment` — explicit N/X midpoint mask
  (distinguishes true N from "no overlap").
- `harmonia/models/chord_pipeline_v1.py::_infer_nnls24` — per-segment `seg_no_chord`
  (musx primary; `_nnls_no_chord_segs` raw-energy gate as musx-absent fallback);
  labeling loop emits `NO_CHORD_LABEL="N"` at conf 0; final loop clamps conf 0 and
  skips the isotonic calibrator on N.
- `harmonia/output/chart_interactive.py` — bake marks `entry["nc"]=True` when the
  exact ireal token is `N`/`N.C.`/`X` (parse_token would read "N" as C = pc 0).
- `harmonia/output/chart_model.py::to_chart_model` — carries `nc`, sets sentinel
  `q="N"`, conf 0; a sidecar correction clears nc.
- `harmonia/output/app_shell.html` — nc cell renders faint "N.C.", still clickable
  in annotate mode; carried into S.chords.

Result (in-process infer→bake→chart_model): Henny intro 0-18.75s = single N.C. cell
+ empty bars, **0 invented chords**; total chords 117→85. NNLS-only energy gate
calibrated on real audio: intro treble-energy 0.52× median vs body 0.97× → gate at
0.35× (energy only; flatness did NOT separate: intro 0.49 < body 0.54).
Tests: `tests/test_chart_model.py` +3 (17 passed); musx N mask unit test.
Free positive: aretha (5C4FnlftQt4) musx lab has N 82.6-99.7s = the documented
a-cappella bridge → musx-N path finds it automatically.

## Part 3 — Occam post-pass (opt-in HARMONIA_OCCAM_POSTPASS) — DONE
Subsumes Part 2 (loop-phase pooling). `occam_compress_bars` +
`_apply_occam_to_coalesced` in chord_pipeline_v1. Key design iterations (logged
because each failure informed the next):
1. Rigid modulo-P phase pooling → REJECTED: real decodes insert/delete chords so
   the loop phase drifts; coverage stuck at 0.39 (= single-chord rate) for all P.
2. Raw-frequency vocabulary → REJECTED: admitted the spurious E into vocab {A,B,E}
   because the decode over-hallucinates E (class-weighting bias).
3. Dominant reciprocal-bigram vocabulary → WORKS: vamp = the unordered root pair
   maximizing c[x→y]+c[y→x]. henny → {A,B}, just-aint → {E,F#}. Robust to a
   frequently-invented third chord.
4. Added "too many exceptions → abstain" (dev-frac > 0.35) as a pure-Occam gate.
   Separates clean vamps but NOT abba (0.335 ≈ henny/just-aint 0.32-0.34) → opt-in.

Gate: family = maximal non-N run (barlocked A/B are loop PHASES, not families).
Snap off-vocab bars to best in-bar vocab member; keep deviation iff own posterior
beats snap target by log(4) AND ≥0.55 (DL-vs-evidence, explicit). Uses ONLY the
song's structure — no corpus grammar prior.

## Gate results
- Henny intro: renders EMPTY N.C. (screenshot henny_nfix_intro_NC), was a big C
  (the standalone renderGrid dropped nc — fixed). PASS.
- Henny A section: Occam → clean A|Bm7 early/mid, spurious E7 + maj/maj7 wobble
  snapped; high-margin tail deviations kept per margin rule (logged). PASS (partial
  on the degraded tail — honest miss, see below).
- Two fresh bakes byte-identical (base + occam). Stability PASS.
- No-regression (in-process real infer_chords_v1, Occam OFF = default):
  just_aint Intro/A/B split UNCHANGED (IABABA); autumn/let_it_be/commodores/aretha/
  abba N spans all intros/outros/bridges, no mid-song spurious N, sections intact.
- aretha a-cappella bridge (82.6-99.7s) auto-detected as N.C. — free positive PASS.
- HTTP-server gate NOT run via live yt-dlp (network); validated the exact server
  code path in-process (real infer_chords_v1 + real chart_to_interactive_inputs +
  render_interactive + payload_from_chart_html + to_chart_model). Residual untested:
  live download + Flask routing only.

## Shipped
- Part 1 (N propagation): DEFAULT ON. Clean win, no regressions.
- Part 3 (Occam): OPT-IN (HARMONIA_OCCAM_POSTPASS=1). abba false-positive means
  don't default without user A/B.
Published: docs/plots/inferred_mayer_hawthorne_henny_gingerale_{nfix,npattern}.html
(→ gmfcYli6vV4). Original junk-intro chart left as the "before".

## Honest misses / next steps
- Occam tail: progressive decode degradation in the 2nd half keeps high-confidence
  misreads that survive the margin. Not fixable by post-hoc compression without
  overriding genuine evidence. Would need better 2nd-half emission (upstream).
- abba false-positive: dev-frac can't separate an A-E-backbone through-composed
  song from a true 2-chord vamp. A diatonic-function or section-repeat consistency
  signal might, but that's future work.
- Occam is 2-chord-vamp only; 3+-chord loops abstain (safe, but uncompressed).

## MISSION 2 (Bayesian Occam prior) — Part A premise-check: PASS, informative
pop400 (n=345): dev-frac vs dominant 2-chord vamp median 0.443 (p10 0.166, p90 0.667),
45% have dominant alternation, min-vocab median 5 (2-chord 4%, 3-4chord 43%, 5+ 53%).
jazz1460 (n=1460): median 0.574, 21% alternation, 87% 5+ chords. pop != jazz clearly.
abba's razor 0.335 = 33rd pct of pop (simpler side). KEY: pop is mostly 3-5-chord
loops, NOT 2-chord — my Occam's 2-chord restriction misses most pop (motivates
vocab extension). Plot docs/plots/occam_simplicity_prior_2026_07_19.png.
Implication for anti-crush: GT "deviations" from a 2-chord vamp are mostly REAL
3rd/4th chords, so the arbitration must be permissive when evidence is confident.

## MISSION 2 Parts B+C — DONE
Part B (positional prior): near-flat (0.36-0.41), pop deviations are recurring loop
chords not cadences — honest negative, positional term kept minimal.
Part C (Bayes arbitration): log-odds(individual) = conf_weight(1.5)*calibrated_conf*
[log post(own)-log post(pattern)] + base(0.38 corpus) + coverage + positional.
- ANTI-CRUSH: 100.0% of 25,120 pop400 GT bars unchanged (razor acted on 139 tunes).
- Real audio: henny A|Bm7 / just-aint E|F#m / abba A/E/C#m preserved; 2-run stable.
- Flips vs old hard margin: henny 16->22 kept (borderline log_odds~0, real-audio
  calibration compressed #26), just-aint 0, abba +1. Logged occam_flip_compare.py.
- conf_weight tuned 3.0->1.5: separates abba's real D/C#m (conf 0.56) from henny
  noise (conf 0.34); the calibrated conf IS the discriminator.
Net: principled, passes every gate; marginally more permissive than the hand
threshold on henny (a wash on real audio, clear win in principle + anti-crush).
Recommendation: keep opt-in with the razor; sharpens as #26/#29 calibration improves.
