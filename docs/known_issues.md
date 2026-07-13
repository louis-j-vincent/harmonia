# Harmonia — Known Issues

Living tracker of known limitations in the current pipeline, ordered by how much
each is currently limiting end-to-end accuracy. Distinct from `architecture_extensions.md`
(forward-looking design ideas) and `suggestions.md` (specific stage-1/stage-5
improvement proposals) — this file is "what's actually wrong right now."

**Status note (2026-07-06): `harmonia/pipeline.py` (`HarmoniaPipeline`) and
`harmonia/models/chord_hmm.py` (`ChordInferrer`/`build_emission_matrix`) are
FROZEN, not deleted.** Last touched at commit `cb8fcf8` ("verify: production
pipeline reproduces logged key-prior perfs"). Twelve minutes later, `ddf9679`
("feat: end-to-end v0 pipeline") introduced a separate, script-based system
(`scripts/pipeline_v0.py`, `scripts/chord_change_engine.py`) built on a new
synthetic jazz-standards corpus (iReal/MMA-rendered, see
`docs/blog/05-turning-the-pipeline-around.md`) and trained classifiers (root/
family/seventh models) instead of template-dot-product HMM emission scoring.
Every commit since builds on that new system; none touch `chord_hmm.py`
again. **Issues #0–#8 below (the POP909/HMM-era investigation, including #5's
emission-template-geometry finding) are about the frozen module** — real,
still correctly fixed and tested (193/193 passing), but no longer on the
critical path for current accuracy numbers. The new system's own issues are
#9 onward. Don't spend a fresh session's budget on #0–#8 assuming they still
gate production accuracy; check whether `chord_hmm.py` is even imported by
the current entry point first.

---

## ACTIVE ISSUES — QUICK REFERENCE

One line per issue. Read **only this section** in pre-flight; read a specific §N only when actively working on that issue.

| # | Title | Status | Next action |
|---|---|---|---|
| 1 | Chord-change temporal resolution | OPEN — root cause: emission discriminability; 3 fixes (A/B/C) rejected. **madmom tested 2026-07-14: does NOT fix the tempo octave-lock (0/10 corpus, worse on anchored songs) — see §9 addendum + docs/madmom_reinference_results.md**. **Octave-lock sub-problem UNSOLVABLE blind (2026-07-14): a blind audio-only disambiguator caps at 3/8 (38%) vs oracle 8/8 — audio-internal signals (onset-ACF, harmonic-rhythm, metrical-alternation) are octave-symmetric or prefer the WRONG 2× octave; only an EXTERNAL tempo prior helps, and a single-centre prior can't cover the 65–225 BPM span. `scripts/disambiguate_octave.py`, `docs/octave_disambiguator_results.md`, plot `docs/plots/octave_accuracy_per_song.png`** | Wire bass-change-signal detector; improved emission model. Octave-lock: NOT a blind-signal problem — use a **style-conditioned tempo prior** (ballad 50–90 / bossa 120–160 / bebop 180–260 via `infer_style_posteriors`), or human tap, or lead-sheet tempo metadata. Tracker choice is irrelevant (both land in [55,215], pick wrong multiple) |
| 2 | Soundfont quality | DONE — MuseScore General sf2 adopted, +12% boundary-F | — |
| 3 | Zero test coverage on pipeline.py / mirex_eval.py | OPEN — process risk, no audio fixtures | Add audio fixtures or mocked activations for pytest |
| 4 | Lossy quality mapping in `_label_to_mireval` | OPEN, low priority — phase-2+ approximation only | Revisit when phase-2+ vocabulary enabled |
| 5 | Emission-template geometry (cosine fix net-negative) | SUPERSEDED — Gen-2 no longer uses template scoring | — (chord_hmm.py frozen) |
| 6 | `build_emission_matrix` drops intervals > 11 | FIXED 2026-07-03 — chord_hmm.py frozen anyway | — |
| 7 | `POP909Parser` discarded GT downbeat column | FIXED 2026-07-03 | — |
| 8 | Dead code crashing if called | FIXED 2026-07-03 | — |
| 9 | Beat tracking not the bottleneck; stem isolation ineffective | MEASURED 2026-07-06 — negative finding | — |
| 10 | Family-emission features unnormalized (duration-dependent) | FIXED 2026-07-06 — L2-norm in chord_change_engine | — (DB cache still raw-sum; consumers must normalize) |
| 11 | MIREX numbers undersold by GT-source mismatch | FIXED 2026-07-06 — harness aligned to song_chord_spans | — |
| 12 | Motif stacking: decision-level voting null on clean audio | MEASURED 2026-07-07 — hurts on hard audio | Feature-level BP-activation pooling across motif instances |
| 13 | `vary_voicings` omitted pitch classes | FIXED 2026-07-07 — DB regen blocked, see #15 | Unblocked by #15 |
| 14 | Production upgraded to Gen-2 (chord_pipeline_v1) | DONE 2026-07-08 — root 78.6% / majmin 73.6% POP909 | — |
| 15 | accomp_db regen blocked by full disk | OPEN — data/cache/accomp_varied/ stale | Free ≥500 MB; regen accomp_varied/ |
| 16 | ctx_v2 model trained | DONE 2026-07-09 — 87.7% fam, 87.6% root (oracle MIDI) | — |
| 17 | beat_seq_model_v3 integration | SUPERSEDED — v4 (93.3% per-beat) was shipped; v3 not wired | — |
| 18 | v3 design-brief baselines misattributed | RECORDED — provenance documented; real baselines in §18 | — |
| 19 | Domain gap: MMA synth → real YouTube recordings | MEASURED 2026-07-13 (Mission 4) — prod pipeline on 7195 real chords (iReal GT): root 59% / exact(root+q5) 32%; **dom7 21% exact / 55% fam-or-better — the old "dom7 0%" is REFUTED**, failure is dom→maj quality confusion not collapse. Quality head q5 acc 44%. Chroma diag: b7 present (dom b7=0.49) but low-contrast (maj b7=0.35), covariate shift not missing info. **RETRAIN DONE 2026-07-14 (Mission 2)** — 5-way q5 head retrained on real corpus_50 feat48 (`data/models/quality_head_v1.pt`; trainer `scripts/train_quality_head.py`, report `scripts/mission2_quality_report.py`). Song-grouped 5-fold CV (root=iReal oracle, n=4143): strict q5 acc **34.7%→57.5% (+22.9pp)**, majmin (third) **61.6%→73.3% (+11.7pp)** vs a same-arch synth-q5 baseline (audio_chord_features base7→q5). Caveat: the synth baseline is degenerate on real audio (over-predicts modal `dom`, recall 0.98, ≈ the 0.344 majority floor), so the mechanism-robust number is majmin +11.7pp. Isotonic calibration on the head's softmax max-prob (song-held-out): ECE **0.122→0.019 (<0.05 PASS)**, saved `data/models/quality_head_v1_calibrator.npz`. hdim/dim still weak (rare). Full writeup `docs/mission_2_real_audio_quality_head_results.md`. NOT YET wired into the pipeline (`_FamilyClassifier` still live) — integration is next | Contrast features (HPSS/whitening) secondary. Do NOT chase "recover b7". Next: wire quality_head_v1.pt into chord_pipeline_v1 as the q5 emission + re-run Mission 1 end-to-end |
| 28 | Merge / evidence-pooling √N denoising validated on REAL audio | MEASURED 2026-07-13 (Mission 4) — pooling feat48 across repeats of the same chord within a song: q5 acc 43.8→53.8% (+10.0pp), grows with reps (≥5: +9.8pp). First real-audio confirmation of Mission 3's pooled-emission claim. **Mission-4 prep (2026-07-13): AUTO-merge detection+eval built** (`scripts/detect_auto_merges.py` + `scripts/eval_auto_merge.py`, brief in `docs/mission_4_auto_merge_brief.md`) — fires a merge only when same-label + equal-bars + structural-conf AND acoustic-conf both >0.75 (never blind; yields to user assertions). Self-test passes; synth smoke run wires detect→pool→score cleanly (1/3 songs fired, +4pp 7ths on it, 0 regressions). **GATED on Mission 1 benchmark** (not yet built) — eval exits(2) until then | Run `eval_auto_merge.py` once Mission 1 lands; ship in-pipeline auto-fire iff Δ7ths ≥ +5pp with 0 regressions on the 20 songs |
| 29 | Real-audio calibration map is ROOT-BLIND (regression vs the fused design) | OPEN 2026-07-13 — the `real` map is fitted on `confidence_raw` (quality head only), so `root_conf` is NOT folded into the displayed confidence on the default (real) path; the `synth` map still uses the fused score. Verified: Autumn Leaves real-map conf mean 0.154 / max 0.46 vs synth-map 0.604 / 1.00. Labels identical (467/467) — display-only, no decode impact | Re-fit the real map on the fused score (`confidence_raw × root_conf`) so #26's root-blindness fix also holds on real audio; until then treat real-audio confidence as quality-only. **Mission-3 prep DONE 2026-07-13 (pending Mission 1+2 to run):** pipeline now honors a `score_kind` field on the saved map — `_get_conf_calibrator` reads it, the ChordChart build feeds `fused` vs `conf` accordingly (legacy real map still `conf`, so current behavior unchanged until refit). `scripts/calibrate_quality.py` refits the real map on the fused score + saves `score_kind="fused"`; `scripts/eval_calibration.py` reports ECE real-vs-synth with the <0.05 gate; shared harvest in `scripts/_calib_common.py`. Validated end-to-end on a 6-song synthetic proxy: fused CV ECE 0.085 < conf-only 0.127 (root-folding already helps). `scripts/exp_trigram_gated.py --fused-gate` re-tests the entropy gate on honest (fused) uncertainty. Blocked on Mission 1 benchmark (`data/real_audio_benchmark/aligned_chords_per_song.json`) + Mission 2 head |
| 20 | Diatonic quality prior | PASS on POP909 (93.3% > 60%); FAIL on jazz1460 (49.4%) | Implement prior for POP909 decoding; keep disabled for jazz |
| 30 | Alignment has no structural QA gate (iReal↔inferred slips are silent) | DESIGN 2026-07-13 (Mission 6) — misaligned alignments silently corrupt training (`yt_chord_corpus` writes wrong labels), eval (mislabels hits as misses, cf. #20), and the displayed chart. 6 failure shapes characterized (chorus slip, phase offset, slipped repeat, warp hole, 2× tempo, wrong transpose). Design: a **structural validator** (NOT a new aligner) using 3 relative signals — repeat-consistency (same-label sections' inferred-content agreement, reuses #22's bridge-contrast), elastic boundary-IoU, per-section family-fraction — combined into `align_score`∈[0,1] + verdict {OK/SUSPECT/MISALIGNED/UNVERIFIABLE}, localizing the slipped section. Sidesteps #20's SNR wall (all signals compare audio to itself / chart to itself, never to absolute time). Docs: `mission_6_alignment_problem.md`, `mission_6_elastic_matching_design.md`, `mission_6_implementation.md` | Build `harmonia/models/alignment_validator.py` + `scripts/validate_chart_alignment.py`; run 3-pilot premise check (does repeat_consistency separate on real audio?) BEFORE the 20-song injected-slip gate; ship display-only banner first, training-filter only after ≥80% slip-recall @ ≤10% FP. **PREMISE CHECK DONE 2026-07-13** (`scripts/test_mission6_repeat_consistency.py`, `docs/mission_6_premise_check_results.md`, plot `docs/plots/mission6_premise_check.png`): CONDITIONAL PASS on 3 real-audio pilots (Autumn Leaves, Ghost-of-a-Chance = #20 pilot fresh inference, Let It Be = documented #22 natural slip). Global Δ=within−cross separates natural cases 3/3 (aligned +0.052/+0.061 vs slipped −0.009, margin ~0.06, matches design estimate). BUT two build-changing findings: (1) global Δ is √N-diluted → nearly blind to a *single* localized slip (Autumn Leaves 1-of-8 A's corrupted moved Δ only +0.003); localization must use the **per-instance sibling-mean z-score** (victim z=−3.47 on distinct-section Autumn Leaves), not global Δ. (2) The z-outlier is contrast-limited: fires only when swapped sections are harmonically distinct — on low-bridge-contrast Ghost (within 0.707 vs cross 0.645) a slip is undetectable (z=+0.41) → must return UNVERIFIABLE, add within/cross spread as abstain trigger. Aligned floor (+0.05) sits on the proposed 0.05 threshold w/ n=1 natural slip → threshold not yet trustworthy | Scale to 20-song harness with the two fixes (per-instance z localizer + bridge-contrast abstain); inject **localized single-instance** slips w/ verified donor≠victim; ship display-only banner on global Δ first (trustworthy as aggregate), training-filter only after per-instance z calibrated to ≥80% recall @ ≤10% FP. **PHASE 2 DONE 2026-07-13 — GATE PASSES** (`harmonia/models/alignment_validator.py`, `tests/test_alignment_validator.py` 5/5 green, `scripts/validate_chart_alignment.py`, `docs/mission_6_phase2_results.md`): 20-song injected-slip harness (jazz1460+pop400, looped 3-chorus + 10% noise; Type A rotate 25% / B swap 50% / C phase 25%) clears all 3 criteria on **10/10 seeds** — mean recall 91%, FP 4%, localisation 98%, ROC-AUC 0.98. Localiser = per-instance cross-match slip-score `xmatch−own_sim` (>0.15) + strong-z (<−3), robust to inference noise where raw z<−2 was not. Two calibration fixes vs the skeleton (drove FP 30%→4%): (a) aggregate score uses **mean** family not min + dropped `sig(bridge)` from the score (bridge is a tune property, not alignment quality — it stays as the abstain gate & localiser basis only); (b) added a **uniform-median-family-floor→MISALIGNED** branch for global slips that keep repeat_consistency intact. **Display banner SHIPPED** in `api_irealb_align` (green/yellow/red/gray + suspect sections + `validation` JSON block). NOT solved: adjacent same-label sections merge (relies on multi-chorus separation); boundary_f1 is phase-blind (Type C caught via family, not Signal 2); synthetic FP ≠ real FP | **Next: training-filter** in `yt_chord_corpus._build_records` behind `--require-alignment-ok`, but ONLY after re-running the harness on **real** YouTube alignments to confirm the ≤10% FP holds on real audio (synthetic FP is not real FP). Then eval-filter in `eval_yt_model` (exclude MISALIGNED, log drops) |
| 21 | Chord progression encoder | REVERSED by §25 (2026-07-13) — reranker default now OFF; the bypass harness's +0.7-1.0pp was a proxy artifact, real path shows −3.6pp | Re-enter encoder as a transition factor in the joint decode (audit step 2), not a greedy rerank |
| 22 | Section structure (AABA / form boundaries) | RESOLVED (2026-07-12) — labels A/B/C + chart chips wired | Eval labelling accuracy on iRealb/POP909; tune sim_threshold; centroid-rep option |
| 25 | `eval_irealb_e2e.py` bypasses ctx model — reranker default-ON reversed on real path | FOUND 2026-07-13 — rerankers OFF = majmin 84.0%/7ths 59.2% (best); 801d byte-identical to 684d with reranker off | Use real-path evals for decisions; wire encoder into joint decode |
| 26 | Displayed confidence was uncalibrated, root-blind, stale after rerank | RESOLVED 2026-07-13 — fused root×quality conf + isotonic map; test ECE 0.233→0.037. **Mission 4 (2026-07-13): two-domain calibration shipped** — synth map ECE 0.465 on REAL audio (amplifies overconfidence to 0.533!); real-audio reliability is near-FLAT (conf 0.98→48% correct), so k-selection was near-random on real audio. New `confidence_calibration_real.npz` (fit on 7002 real segments) + `infer_chords_v1(audio_domain=...)`, default "real" for server: real ECE 0.465→0.007 (5-fold song-held-out CV), collapses displayed conf to base rate ~0.44. Synth path untouched (ECE 0.037) | Refit real map on PRODUCTION confidence_raw (current fit uses baseline-LR proxy) once real audio+GT is re-obtainable; nightly reliability check |
| 27 | Joint root×quality segment Viterbi (audit step 2) | GATE PASSED 2026-07-13 — jazz majmin 84.0→86.2, 7ths 59.2→60.5, POP909 neutral; default ON, calibration refit on joint path. **Mission 1 (2026-07-13): transition slot stays EMPTY** — key-local bigram (H1), encoder shallow fusion (H2), density-ratio fusion (H3) ALL net-negative on jazz majmin (optimum λ→0); diagnosed dead ends, all wired default-OFF. **Mission 2 (2026-07-13): per-beat semi-Markov GATE PASSED, default ON** — explicit-duration Viterbi (jazz1460 dur prior) as the segmenter feeding the joint labeler: jazz held-out root 88.7→89.4 / majmin 86.2→86.6; POP909 root 76.9→78.6 / majmin 50.1→51.1 / 7ths 45.9→47.0 (all up) | **Duration/boundary evidence IS the live lever** (unlike the grammar slot) — semi-Markov shipped; Mission 3 = user-input factors on its pooled-emission interface |

---

Baseline referenced throughout: MIREX weighted-overlap accuracy on the 5 rendered
POP909 songs (piano patch, `prog0`), after the session-4 bugfixes (N-collapse,
zero-duration events, confidence underflow, `_label_to_mireval` crash):

| song | n_events | root | majmin | 7ths | tetrads |
|---|---|---|---|---|---|
| 001 | 14 | 31.1% | 32.1% | 2.8% | 2.7% |
| 002 | 61 | 21.0% | 11.5% | 10.4% | 10.0% |
| 003 | 20 | 22.8% | 19.1% | 0.7% | 0.6% |
| 004 | 28 | 18.0% | 15.1% | 10.5% | 9.7% |
| 005 | 68 | 20.5% | 7.3% | 4.4% | 4.8% |

Two foundational bugs were found and fixed after that baseline (both real,
both independent of the issue #1 investigation below): `BASIC_PITCH_FRAME_RATE`
was off by exactly 2x (see "Resolved" section), and `key_prior_per_beat` (see
issue #1's follow-up). Combined, they moved the 5-song mean from
root=21.5%/majmin=15.4% to root=33.0%-35.5%/majmin=27.1%-29.6% — bigger than
any single fix in the issue #1 A/B/C investigation. **Issue #0 below (found
and fixed 2026-07-02) turned out to be a real, independent calibration bug —
validated across all 5 songs against real ground truth — but re-checking
issue #1's song-001 regression afterward showed it wasn't the explanation
for that regression; that mystery is still open.**

---

## 0. Key inference posterior was near-uniform / uncalibrated — RESOLVED 2026-07-02

**Found while investigating why `key_prior_per_beat` (issue #1) helped 4/5
songs but hurt song 001.** Checked every structural segment's inferred key
for song 001 — all 16 segments resolved to "F# major", each with **bit-for-bit
identical confidence, 0.043** (`1/24 = 0.0417`, i.e. essentially uniform over
all 24 candidate keys).

**Root cause, `harmonia/theory/key_profiles.py::infer_key()`:**
`log_likelihood = KEY_PROFILES @ chroma_norm` computed a correlation between
two L1-normalized distributions and treated the result directly as a
log-likelihood — mathematically bounded to ~10% relative posterior
concentration regardless of input. A secondary bug neutralized the
Dirichlet-style confidence-scaling term for the same reason (the caller
pre-normalized `chroma` before `infer_key` ever saw its magnitude).

**Fix:** proper multinomial log-likelihood over raw (unnormalized) chroma
counts (`sum_i chroma_raw[i] * log(profile_k[i])`), naturally scaling with
the amount of evidence. Both call sites (`structure.py::_make_segment`,
`key_profiles.py::activations_to_chroma`) now pass raw chroma through
instead of normalizing it away.

**A second, related bug surfaced during validation:** once raw magnitude
reached `infer_key()`, confidence saturated to bit-exact `1.0` for almost
every real segment, even the shortest ones — chroma sums land in the
hundreds because raw per-frame/per-beat activation-probability magnitude
isn't a genuine independent-trial count (many pitch classes co-sound within
one beat). Fixed by L1-normalizing each beat/frame individually before
summing into the aggregate chroma (`_beat_chroma(..., norm="l1")` in
structure.py; the same treatment in `activations_to_chroma`), so raw
magnitude reflects a real evidence count (~n_beats or ~n_frames with
signal) instead of an inflated one.

**Validation** (`scripts/validate_key_inference.py`, all 5 POP909 songs,
against `key_audio.txt` ground truth — unused anywhere in this project
before this session):

| song | GT key | global inferred | global conf | duration-weighted seg. acc. | confidence range |
|---|---|---|---|---|---|
| 001 | Gb:maj | F# major | 1.000 | 100.0% | 0.30–0.92 (16 distinct values) |
| 002 | B:maj | B major | 1.000 | 41.2% | 0.19–0.89 |
| 003 | Bb:maj | A# major | 1.000 | 78.9% | 0.25–0.93 |
| 004 | Eb:min | F# major | 1.000 | 42.1% | 0.23–0.60 |
| 005 | G:maj | G major | 1.000 | 92.2% | 0.36–0.88 |

Global-key accuracy: **4/5**. The one miss (song 004) is a textbook
relative-major/minor confusion — Eb minor and Gb/F# major share the same 7
diatonic pitch classes, a known limitation of pure Krumhansl-Schmuckler
profile matching, not a bug from this session. Confidence is now genuinely
informative: song 001 (unambiguous, all segments correct) has mean
confidence 0.541; song 004 (the one that's actually wrong) has mean
confidence 0.348 — lower exactly where the model is in fact less reliable.
Per-segment consistency (41–100%) is noisier than global-key accuracy,
largely explained by real tonicization of the dominant/subdominant within
sections (song 002's "misses" are consistently E major/F# major — the IV
and V of B major) — a per-segment KS-profile estimate correctly picks this
up as local emphasis, not calibration noise.

**Re-checked issue #1's `key_prior_per_beat` song-001 regression (see issue
#1 below) now that key inference is calibrated — it did not resolve.**
Same magnitude as before (root 33.3%→22.6%, majmin 34.0%→21.9%). This
makes sense in hindsight: song 001's MAP key (tonic/mode) was already
correct even under the old broken calibration — only *confidence* was
wrong, and `build_key_prior()` only ever consumes `.tonic`/`.mode`, never
`.confidence`. Fixing calibration without changing which key wins the
argmax can't move a downstream consumer that never looked at the
confidence value in the first place. The song-001 regression's real cause
remains open and unexplained — see issue #1.

Tests: `tests/test_theory.py::TestKeyInferenceCalibration`,
`tests/test_structure.py` (written red-first against the old bug, then
used to confirm the fix). `harmonia/data/pop909_parser.py` gained
`KeyEvent`/`POP909Song.key_events`/`key_at_time()` to load
`key_audio.txt`, reusable for future work.

---

## 20. Beat-grid iReal→audio alignment (Mission 1 Phase 1) FAILS the ±150ms gate — OPEN, 2026-07-13

Validated `scripts/mission_1_build_benchmark.py` (`extract_beat_grid` +
`align_ireal_to_beat_grid`) on 3 pilot songs before scaling to 20.
Objective check: chroma-template correlation at reference chords, best-offset
search = alignment error (no by-ear playback available in-session; the offset
proxy is reproducible — see scratchpad `verify_align.py` / `diagnose.py`).

Audio: `docs/audio/{ghost_of_a_chance,a_foggy_day,airegin}.m4a`.

| song | style | chart/librosa/true BPM | raw mean/max off | oracle-fit mean/max off |
|---|---|---|---|---|
| A Ghost Of A Chance | Ballad | 70 / **117.5** / ~58 | 749 / ≥1000ms | 555 / 800ms |
| A Foggy Day | Med Swing | 140 / 129 / ~120 | 600 / 840ms | 363 / 800ms |
| Airegin | Up Swing | 220 / 287 / ~290 | 376 / 720ms | 413 / 800ms |

**Gate: FAIL** (all songs ≫ ±200ms). Two independent root causes:
1. **Tempo-octave error** (issue #1 pattern): ballad librosa BPM 117.5 = 2.03×
   true ~58. Off-by-2x beat grid ⇒ chart-beat N lands at ½ the true time.
2. **Structural, not just a tuning bug**: even an *oracle* global linear fit
   (best constant tempo + fitted intro offset, tuned to maximise chroma match)
   still leaves 363–555ms mean / 800ms residual. A single global beat-index→time
   map can't track human rubato/swing, intros/pickups, or the fact that the
   1-chorus iReal chart ≠ the multi-chorus recording (Foggy Day audio = 1013s
   vs ~84s head). `extract_beat_grid`'s downbeats are a fake every-4th-beat
   heuristic, so there is no re-anchoring.

**Fixed en route:** `beat_track(onset_env=…)` → `onset_envelope=…` (librosa 0.11
kwarg rename; the skeleton would have crashed on first real call).

**Proposed alternative (non-circular, handles rubato):** chord-template-chromagram
DTW — synthesize a chroma sequence from the *iReal GT chords themselves* (not
model output, so NOT circular) and DTW-align it to the audio CQT chroma,
restricted to the first head (subsequence DTW to skip intro/solos). Local warping
absorbs rubato; expect ≤±150ms. Fallback: manual downbeat anchors at a few
section boundaries + piecewise-linear interpolation. Est. 1 build-day for DTW
path + re-run this 3-song gate before scaling.

---

## 1. Chord-change temporal resolution is far coarser than reality — OPEN, root cause characterized, 3 fixes tried and rejected

**Symptom:** GT chords change roughly every 2 beats (~1.3s at 89 BPM). Predicted
chords last 15–35 beats on average. This is why `root`/`majmin` are moderate but
`7ths`/`tetrads` are near-zero — the model gets the coarse harmonic family right
sometimes but doesn't track actual chord-to-chord motion.

**Ruled out:** the HMM transition matrix's `self_transition_boost`. Sweeping it
2.0 → 0.05 (40x range) changed event count only 14 → 17 on song 001. Not a
transition-tuning problem.

**Working hypotheses (being tested, see below):**
1. Emission observations (`beat_probs`) are never loudness-normalized before the
   `beat_probs @ E.T` dot product against chord templates (rows of `E` are
   L1-normalized, `beat_probs` isn't). A dominant, possibly-recurring bass note
   can dominate the emission vector's magnitude/shape and steer the argmax
   toward the same "chord family" regardless of the (weaker, sparser) inner
   voices that actually distinguish e.g. B:maj from C#:maj from Bb:min from
   Eb:min. Evidence: MIDI 49 (C#3) and MIDI 47 (B2) are each the loudest key in
   ~15% of all beats in song 001 — a strong, narrow concentration.
2. The self-transition-boost duration model is **memoryless (geometric)**: the
   probability of staying on the same chord for one more beat is constant
   regardless of how long you've already been on it. Real chord durations
   cluster around a typical value (~2 beats here) rather than following a
   memoryless process — a semi-Markov / explicit-duration model would let
   "probability of staying" decay as time-on-chord grows past the typical
   duration, rather than staying flat forever.
3. No use is made of song structure (repeated verse/chorus vamps). If a 4- or
   8-beat harmonic loop repeats several times, overlaying repeats gives more
   observations per "slot" in the loop, and cross-repeat consistency is itself
   evidence about where real chord boundaries are.

**Plan:** see chat log 2026-07-02 for the full design discussion. Three fix
candidates, to be A/B tested independently before being combined:
- **A — emission signal quality** (L1-normalize `beat_probs`; adaptive per-song
  onset threshold). Cheap, low risk.
- **B — explicit-duration decoding** (semi-Markov-style duration prior fit to
  the empirical POP909 chord-duration histogram, replacing the flat geometric
  self-transition boost). Moderate effort, well-established technique.
- **C — periodicity/structure folding** (rank candidate loop lengths via the
  self-similarity matrix's diagonal-averaged autocorrelation, constrained to
  musically plausible multiples of the detected bar length; use top candidates
  as an ensembled/voted prior rather than committing to one). Novel, higher
  risk/effort — sequenced after A and B are validated.

Isolated metrics for A/B testing (chosen so each hypothesis is tested directly,
not just via the confounded end-to-end weighted accuracy):
- **Per-beat emission argmax root-accuracy vs GT** (bypasses the HMM entirely)
  — isolates whether raw evidence discriminates chords better. Used for A.
- **Chord-boundary F-score** (predicted vs GT change-points, with a small beat
  tolerance) — isolates whether the *rate* of chord changes matches reality,
  independent of whether the root/quality is exactly correct. Used for B.
- **Cross-repeat label consistency** (agreement between predicted chords at
  matching slots across detected loop repeats) — used for C.
- MIREX weighted accuracy (root/majmin/7ths/tetrads) as the final downstream
  check once a fix looks good in isolation.

### Candidate A results (2026-07-02) — tested, not adopted

Harness: `scripts/experiment_issue1.py --sweep` (metric 1) and `--sweep-full`
(metrics 2+3), all 5 songs, `prog0`.

| Variant | metric 1 (5-song mean) | boundary F | root | majmin |
|---|---|---|---|---|
| baseline | 16.8% | 0.215 | 22.7% | 17.0% |
| A1: L1-normalize `beat_probs` | 16.8% (byte-identical) | 0.215 (byte-identical) | 22.7% | 17.0% |
| A2: adaptive percentile threshold (best: p97) | 17.1% | 0.214 | 23.0% | 17.0% |
| A3: `sqrt` compression | 17.8% (best on metric 1) | **0.167** | **21.4%** | **16.3%** |
| A3: `log1p` compression | 16.8% | 0.179 | 22.3% | 16.8% |

**A1 (L1-normalize) is not just weak, it's provably a no-op.** Viterbi's
recursion is `viterbi[t,c] = max_i(viterbi[t-1,i] + log_A[i,c]) + log_emission[t,c]`.
L1-normalizing beat `t`'s observation subtracts the same constant
`log(row_sum[t])` from `log_emission[t,c]` for every chord `c` — a uniform
per-timestep shift can never change which path Viterbi prefers, at any step,
for any input. Confirmed empirically (byte-identical output on all 5 songs,
both metrics) and formally (`tests/test_chord_hmm.py::TestEmissionPreprocessing::
test_normalize_emission_does_not_change_decoded_path`). Implemented as
`ChordInferrer(normalize_emission=...)`, kept only for that regression test.

**A2 (adaptive threshold)** moves metric 1 a little (this song's fixed 0.3
threshold already sits near the 95th percentile, so percentile thresholds only
diverge from baseline at the tails, e.g. p97) but the full-pipeline effect is
within noise. Implemented as `PitchExtractor(onset_percentile=...)` /
`HarmoniaPipeline(onset_percentile=...)` — real code path, not adopted as the
new default given no measurable downstream win.

**A3 (nonlinear compression, sqrt/log1p)** is the most interesting result:
it *is* a real, non-inert transform (per-element, not per-timestep-uniform —
changes relative weight of loud vs. soft notes within a beat) and it clearly
improved the isolated per-beat metric (16.8%→17.8% with sqrt). But it made the
**full pipeline worse** (boundary F 0.215→0.167, root 22.7%→21.4%). Working
explanation: compression shrinks `log_emission`'s dynamic range across
candidate chords at a beat; Viterbi sums `log_emission + log_transition +
log_init`, so shrinking the emission term's spread makes the (already too
sticky, see B below) transition prior relatively *more* influential, not less
— sharper per-beat evidence gets more thoroughly overridden by the prior.
Implemented as `ChordInferrer(compress_emission="sqrt"|"log1p")` — real code
path, not adopted as the default.

**Conclusion:** no Candidate A variant is adopted as the new default. The A3
finding is independent evidence (beyond the original plan's reasoning) that
**Candidate B needs to land before emission-quality work can show results** —
the duration prior is currently strong enough to absorb improvements in
per-beat evidence quality. Proceeding to B next.

### Candidate B results (2026-07-02) — tested, not adopted

Fit `harmonia/theory/duration_prior.py::fit_duration_prior()` from all 909
POP909 songs' text annotations (119,901 chord events, no audio needed).
Confirmed the geometric-model critique directly: `P(duration=2 beats)=49.2%`
is *higher* than `P(duration=1)=15.0%` — a geometric distribution can never
have an interior peak like that (it's always maximised at its minimum), so
this is empirical proof the true duration shape can't be represented by
`self_transition_boost` at any value. Mean duration 2.49 beats, secondary
mode at 4 beats. N (no-chord) durations are 98.5% a single beat, confirming
it needed its own separate duration model.

Implemented `viterbi_duration_aware()` in `harmonia/models/chord_hmm.py` — a
segmental/explicit-duration Viterbi (`O(T x D x C^2)`, `delta[t,j] = max`
over duration `d` and predecessor `i` of `delta[t-d,i] + transition +
duration(j,d) + segment_emission(j)`), wired in via
`ChordInferrer(duration_prior=...)`.

First attempt forbade same-state transitions between segments (textbook
HSMM: persistence should come only from `log_duration`, not the transition
matrix). This backfired badly: whenever a stretch was genuinely longer than
the duration model's cap (`D=32` beats), the decoder was forced to "fake" a
change into some other state just to keep going — in practice usually a
near-duplicate quality of the same root (observed directly: `C#sus4(31) ->
C#7sus4(32) -> C#7sus4(31)` back to back on song 001). Fixed by allowing
same-state segment chaining (an "escape valve" for long stable regions) —
mechanically correct (see `tests/test_chord_hmm.py::TestViterbiDurationAware::
test_escape_valve_for_long_stable_regions`) but the pathological pattern
persisted anyway on real audio, because the decoder was choosing to
alternate between near-duplicate qualities *even when allowed not to* — the
emission evidence itself was (marginally) rewarding the alternation.

Swept `self_transition_boost` blended with the duration prior (0.0, 0.5,
1.0, 2.0 — i.e. from "pure" HSMM to increasingly hybrid) on the full
5-song pipeline; none matched the plain-geometric baseline:

| Config | boundary F | root | majmin |
|---|---|---|---|
| baseline (no duration prior) | 0.215 | 22.7% | 17.0% |
| duration-aware, boost=0.0 | 0.175 | 21.7% | 9.7% |
| duration-aware, boost=0.5 | 0.157 | 22.3% | 10.3% |
| duration-aware, boost=1.0 | 0.151 | 22.0% | 10.7% |
| duration-aware, boost=2.0 | 0.146 | 21.7% | 10.6% |

Root accuracy is roughly flat across all configs (the decoder gets *when*
to place boundaries closer to right); `majmin` collapses from 17.0% to
~10% in every configuration (it gets *what quality* wrong far more often).

**Conclusion: not adopted.** This converges with Candidate A on the same
diagnosis: forcing more frequent segment boundaries (matching the true ~2
beat harmonic rhythm) doesn't help once you get there, because the
per-segment emission evidence isn't reliable enough to discriminate between
similar chord qualities (e.g. sus4 vs 7sus4 vs dom7, which share most of
their template) — it just exposes that weakness more often than the
original sticky-but-rarely-wrong-when-it-commits geometric model did. Both
A and B independently point at the same root cause: **emission
discriminability, not decoder structure, is the binding constraint.**
Proceeding to Candidate C, which is the one candidate that targets
improving the emission evidence itself (via cross-repeat averaging) rather
than reshaping how existing evidence is used.

### Candidate C periodicity premise check (2026-07-02, song 001 only)

Before implementing, checked the premise directly: `scripts/plot_periodicity_diagnostic.py
--song 001` computes `score(L) = mean_i SSM[i, i+L]` (autocorrelation of the
self-similarity matrix already built for segmentation) and plots it alongside
the SSM itself → `docs/plots/inference/pop909_001/ssm_periodicity.png`.

Result: a sharp, clear peak at **L=32 beats (8 bars)**, score 0.82 — the
single highest of any lag tested, well above its immediate neighbors.
Harmonics confirm it's real structure, not noise: L=64 (`2×32`) also peaks
(0.80), and L=16 (`32/2`) shows a smaller secondary peak (0.72) — consistent
with an 8-bar section built from two similar 4-bar halves. L=4 and L=8
(bar/2-bar) show no distinct peak — no evidence of fine-grained accompaniment-
pattern repetition at that resolution via plain chroma similarity. L=1 also
scores high (0.78) but is a known false-signal (adjacent beats are usually
still the same chord — the issue #1 over-smoothing problem itself, not
structure), which is exactly why the candidate search is constrained to bar
multiples rather than an unconstrained lag sweep.

Premise validated for song 001; not yet checked on 002-005 (deferred until
Candidate C implementation, per plan).

### Candidate C results (2026-07-02) — tested, not adopted

Implemented `harmonia/models/periodicity.py`: `score_periods()` (reuses
`build_ssm()` from segmentation, candidate periods constrained to
`beats_per_bar x {1,2,4,8}`, drops harmonics of an already-kept period) and
`fold_beat_probs()` (circular-average every beat with all beats an exact
multiple of the period away). Wired into `ChordInferrer`/`HarmoniaPipeline`
as additional weighted emission terms alongside the raw per-beat one — an
ensemble, not a replacement, consistent with every other prior in this
codebase.

Tested on the 5-song set (new soundfont as the base, see issue #2 below),
sweeping `periodicity_weight` from 0.1 to 1.0:

| periodicity_weight | boundary F | root | majmin |
|---|---|---|---|
| 0.0 (baseline, no folding) | 0.241 | 21.5% | 15.4% |
| 0.1 | 0.235 | 21.8% | 15.1% |
| 0.25 | 0.235 | 22.6% | 15.2% |
| 0.5 | 0.237 | 22.6% | 14.7% |
| 1.0 | 0.240 | 22.3% | **11.7%** |

At light weights it's essentially a wash (all three metrics within noise of
baseline); at full weight `majmin` drops noticeably, same pattern as B.
Song 001 — the one with by far the strongest, cleanest periodicity signal
(L=32, score 0.82, see premise check above) — regressed the *most* at full
weight (`majmin` 32.7% -> 15.3%), which is the opposite of what the
hypothesis predicted and is the most informative single data point here.

**Working explanation:** the SSM similarity peak at L=32 most likely
reflects repetition of the *accompaniment pattern and rhythmic texture*
(instrumentation, energy, overall pitch-class distribution), not
necessarily identical chord-for-chord harmony at every slot. Real songs
vary their harmony between repeats of a section — a second verse often
reharmonizes a beat or two even when the underlying groove is identical.
Averaging genuinely different chords together at those slots produces a
blurred composite that isn't a better version of either — it's evidence
for neither, which is exactly the kind of thing that damages quality
discrimination specifically (`majmin`/`tetrads`) while leaving the coarser
`root` signal comparatively unharmed. High self-similarity in a chroma-only
SSM is necessary but not sufficient evidence that harmony repeats
identically — it's also satisfied by "the rhythm and instrumentation
repeat, harmony merely correlates."

**Conclusion: not adopted** (`use_periodicity=False` remains the default).
This is the third candidate in a row to converge on the same result: no
amount of reshaping *how* existing per-beat evidence is used — via
emission preprocessing (A), duration modeling (B), or cross-repeat
averaging (C) — recovers accuracy that isn't already latent in the
per-beat evidence itself. The mechanical implementation (period detection,
folding, weighted ensembling) is correct and tested regardless
(`tests/test_periodicity.py`, `tests/test_chord_hmm.py::TestFoldedViews`)
in case a future direction wants to reuse it — e.g. restricted to only the
segments/sections where cross-repeat agreement is independently confirmed
to be high, rather than applied uniformly.

### Issue #1 status: not resolved, but well-characterized

Three structurally different fixes (A: emission preprocessing, B: explicit
duration modeling, C: structural cross-repeat averaging) were each
implemented properly, tested in isolation with a metric chosen to match
their specific hypothesis, and validated end-to-end across all 5 songs.
None improved the full pipeline. All three converge on the same diagnosis:
**per-beat/per-segment emission evidence cannot reliably discriminate
between chords that share most of their template** (sus4 vs 7sus4 vs
dom7, maj vs maj7, etc.) — every fix that made the decoder more
responsive to that evidence (more frequent, more accurately-timed
switching) just exposed this weakness more often, which specifically
tanks `majmin`/`tetrads` (quality-sensitive) while leaving `root`
(quality-blind) comparatively stable across every experiment run in this
investigation.

Issue #2 (soundfont) was a real, if modest, net win — better transcription
quality demonstrably helps timing (`boundary F` +12% relative, the best
result of anything tried) without helping quality discrimination. That's
consistent with the diagnosis, not a counterexample: better audio fidelity
improves *how much signal exists*, not *how well the emission model
separates similar-quality chord templates given that signal* — those are
different bottlenecks, and only the second one is what's currently
blocking `majmin`/`tetrads`.

**What this points to next**, in rough priority order: (1) the emission
*model* itself — `build_emission_matrix`'s chord templates may not be
sharp enough to separate closely-related qualities even with perfect
audio (worth checking directly: does the emission matrix's own row-to-row
cosine similarity show sus4/7sus4/dom7 as nearly indistinguishable
templates, independent of any real audio?); (2) `docs/suggestions.md`'s
still-untried Stage 1 ideas (hybrid onset+note observation, MAX vs SUM
pooling) which target Basic Pitch's raw output rather than anything
downstream of it; (3) revisiting whether phase-1's 15-quality vocabulary
is more granular than the acoustic evidence can actually support, i.e.
whether some of these quality distinctions should be merged rather than
forced.

### `key_prior_per_beat`'s song-001 regression, re-checked post-calibration-fix (2026-07-02) — still open

Issue #0's fix was suspected to possibly explain why `key_prior_per_beat`
helps songs 002-005 but hurts song 001 (root 33.3%→22.6%, majmin
34.0%→21.9%). Re-ran the same A/B comparison
(`scripts/experiment_issue1.py --sweep-key-prior`, `v005_musescoregeneral`
renders) now that `infer_key()` is properly calibrated — **the regression
is unchanged, to within noise**:

| song | root (off→on) | majmin (off→on) |
|---|---|---|
| 001 | 33.3%→22.6% (-10.7pp) | 34.0%→21.9% (-12.1pp) |
| 002 | 37.4%→37.8% (+0.4pp) | 28.2%→38.5% (+10.3pp) |
| 003 | 32.0%→27.9% (-4.1pp) | 22.4%→27.9% (+5.5pp) |
| 004 | 43.3%→37.6% (-5.6pp) | 29.2%→30.0% (+0.8pp) |
| 005 | 32.5%→40.5% (+8.0pp) | 19.0%→31.4% (+12.4pp) |

Makes sense on reflection: song 001's MAP key (F# major) was already
correct even under the old, badly-miscalibrated confidence — `infer_key()`
picked the right `tonic`/`mode` in every segment before this session's fix
too (see issue #0's original write-up). `build_key_prior()` only ever
consumes `.tonic`/`.mode`, never `.confidence`. Fixing calibration without
changing which key wins the argmax cannot move a downstream consumer that
never looked at confidence in the first place — issue #0 and this
regression are independent problems that happened to surface in the same
investigation. **Root cause of the song-001 regression is still
unexplained** — worth a fresh, narrowly-scoped look (e.g. per-beat
diatonic-boost interaction with song 001's specific chord vocabulary or
voicings) rather than folding it into the still-on-hold issue #1 A/B/C
investigation.

### Bass-note motion as a chord-change signal — exploratory, 2026-07-02, groundwork laid

New hypothesis, not yet part of the A/B/C investigation above: bass motion
might carry useful signal about *when* a chord actually changes (issue #1's
open problem), distinct from *what* it changes to. A walking bass moves
every beat without necessarily implying a new chord; a bass pitch-class
change that coincides with other evidence changing is more likely a real
chord change. New reusable tooling, all exploratory (nothing wired into
`harmonia/models/chord_hmm.py` yet):

- `scripts/bass_track.py` — `infer_bass_track_learned()` (audio-only bass
  detector), `rolling_key_track()` (dense per-beat key estimate, diagnostic
  only), `true_bass_track()` (ground-truth bass from POP909's symbolic
  `PIANO` MIDI track).
- `scripts/plot_bass_and_key_tracks.py` — per-song 4-panel visual (note
  probs, chroma, bass or rolling-key track, GT chords), same layout as
  `plot_note_probs_vs_gt.py`.
- `scripts/analyze_bass_patterns.py` — cross-song empirical distributions
  (all 5 songs, pooled, against GT chord annotations):
  - Bass scale-degree relative to concurrent GT chord root: **63.6% root,
    11.7% fifth** (75% combined), third only 1.7% — strongly confirms the
    "bass favours root/fifth" intuition, and echoes issue #1's earlier
    finding that the third is the acoustically weakest chord tone.
  - Bass-change is real but soft evidence for chord-change: P(chord
    changed | bass changed) = 49.7% vs P(chord changed | bass same) =
    26.9% — roughly doubles the odds, nowhere near deterministic.
  - Bass pitch-class runs are shorter than GT chord runs (mean 2.05 vs
    2.59 beats) — bass does subdivide harmony somewhat, but the gap is
    modest, not dramatic.
- `scripts/learn_bass_distribution.py` — used POP909's `PIANO` track as
  ground truth (not just audio) to learn, rather than guess, bass-detector
  thresholds:
  - True "no bass at all" is rare in this corpus: **0.4%** (7/1584 beats)
    — POP909 piano arrangements have near-continuous LH accompaniment.
  - True bass register is narrow: MIDI 37-61 (C#2-C#4), 99th percentile
    F#3 — a register ceiling is well-justified and free (no measured
    downside).
  - Tested whether an "isolation gap" (semitone distance from the lowest
    active note to the next one up) can distinguish a real bass note from
    the bottom of a closely-voiced chord: ground truth shows a real
    difference (median gap 5 semitones when bass is truly present vs 2
    when truly absent), **but a full grid search against ground truth
    found no gap threshold that improves detection** — see
    `docs/plots/inference/bass_patterns/bass_detector_v1_vs_v2.png|.` Two
    reasons: only 7 true no-bass beats total (too few to learn a reliable
    per-beat threshold from), and the dominant error mode is different
    from what was hypothesized — **even when a real bass note is present
    and freshly struck, the audio's lowest active key names the wrong
    pitch class ~48% of the time.** That's raw pitch-detection noise in
    the bass register itself, not a "confused with a nearby chord tone"
    problem a smarter post-hoc filter can fix.

**Where this leaves things:** the register ceiling is adopted (free win,
`infer_bass_track_learned`'s default). The isolation-gap idea is
documented but disabled (evidence was inconclusive, not negative — worth
revisiting with more ground-truth data). The real open question this
surfaced is *why* the raw audio-derived bass pitch-class is wrong ~48% of
the time even under ideal conditions — likely the next thing worth
understanding if bass-based chord-change detection is pursued further.

### Oracle-segment chord reconstruction — 1-hour sprint, 2026-07-02, strong result

**The question:** given *correct* chord-change timing (GT segment
boundaries from `chord_midi.txt`), can bass evidence + full chroma + the
real key/scale reconstruct the right chord label? This decouples "what
chord is it" from "when does it change" — the two problems issue #1 has
been conflating throughout the whole A/B/C investigation above. If
labelling-given-correct-timing works well, the remaining problem is purely
about *when* to change, which can reuse the bass-change correlation prior
already measured above.

**Method** (`scripts/experiment_bass_chord_inference.py`): per oracle GT
segment, score every (root, quality) pair as

```
score(root, quality) = w_bass · log(bass_pc[root] + fifth_weight · bass_pc[root+7] + eps)
                      + w_key  · log(diatonic_boost if root in GT-key scale else 1.0)
                      + w_chroma · log(cosine(chroma_seg, template(root, quality)) + eps)
```

`bass_pc` is the segment's chroma, but folded with an octave weight
*centred on the learned true-bass register* (Gaussian, center=46,
sigma=9 — from this session's earlier `learn_bass_distribution.py`
finding: true bass lives in MIDI 37-61) instead of the mid-register
weighting the existing pipeline's emission matrix uses for quality
matching. `chroma_seg` uses the existing mid-register-weighted chroma. The
`fifth_weight` term implements the requested heuristic directly: **seeing
the chord's fifth in the bass is itself real (if weaker) evidence for the
root**, not just the root's own presence — captures "1 to 5" walking
bass without needing to hard-detect a single "the bass note."

**Iteration** (all 5 songs pooled, root accuracy at oracle boundaries):

| step | root acc |
|---|---|
| chroma + key only (no bass) | 53.3% |
| + bass (root only, no fifth heuristic) | 73.4% |
| + root/fifth heuristic (fifth_weight=0.4) | 80.3% |
| + tuned weights (fifth_weight=0.8, w_bass=1.5) | **83.2%** |

Ablations at the tuned point: removing the diatonic key-prior term
(`w_key=0`) cost essentially nothing (80.1% vs 80.3%) — bass+chroma
evidence is specific enough that the soft diatonic prior rarely needs to
break a tie. Removing chroma (`w_chroma=0`, bass+key only) left root
accuracy unchanged (80.5%) but collapsed majmin (quality) accuracy from
64.9% to 47.5% — **root is almost entirely a bass question; quality is
almost entirely a chroma-template question.** That factorization is the
main structural finding here, more than any single number.

**The dominant remaining error, found by inspecting song 001's
"surprisingly bad" per-event misses:** POP909 chord labels sometimes
encode a bass-note inversion (e.g. `F#:maj7/5` = F# major 7 with C#, the
5th, in the bass) — `POP909Parser` silently discards this
("`Bass inversions (/bass_note) are ignored — we model root position
chords`"), so the stored GT `root` is the *functional* root, not
necessarily the note actually sounding in the bass. A model that
deliberately trusts the sounding bass note will disagree with that label
by construction. Checked directly: **10-18% of chord_midi.txt lines per
song carry a slash marker**; pooled across all 5 songs, root accuracy on
non-inversion labels is **86.8% (n=545)** vs **38.1% on inversion labels
(n=63)**. This is exactly the "ground truth might not be completely true"
case — on inversions, the model and the label are answering different
questions (sounding bass vs. functional root), not one being simply wrong.
`--exclude-slash` reproduces this split directly.
See `docs/plots/inference/bass_patterns/bass_chord_inference_summary.png`.

**Final numbers, oracle boundaries, non-inversion labels, pooled:
root 86.8%, root+majmin-bucket 69.7%** — dramatically higher than the full
pipeline's real numbers (root ~33-35%, majmin ~27-30%), strong evidence
that **timing, not labelling, is the dominant error source once bass
evidence is used properly** — consistent with, and sharpening, issue #1's
long-standing diagnosis.

**Not yet done (ran out of the 1-hour budget):** wiring this into an
actual chord-*change* detector (the suggested next step: blend the
bass-change-correlation prior measured earlier — P(chord changed | bass
changed)=49.7% vs 26.9% — with chroma novelty to place boundaries, then
apply this scoring formula per detected segment) and a full end-to-end
re-evaluation against the real MIREX pipeline. The scoring formula itself
is validated and ready to reuse for that; `harmonia/models/chord_hmm.py`
is untouched — this is still exploratory, no pipeline integration yet.

### `harmonia/models/periodicity.py::score_periods()` detects period length only, never phase offset — found 2026-07-04, FIXED 2026-07-04

Surfaced while jointly analyzing `A_beat_phase` × `E_position_in_loop` for
`docs/chord_change_signal_analysis/` (see that folder's `findings_AE_DE.md`
and `SUMMARY.md`): loop-start beats (`beat_idx % detected_period == 0`) and
annotated downbeats turned out to be **completely disjoint sets** in 2 of 5
songs (003, 004) and only a partial subset in the other 3 — i.e. "position 0
within the detected loop" frequently does not land on a real downbeat at
all.

**Root cause, confirmed by reading the function directly:**
`score_periods()`'s only scoring line is
`scores = {L: float(np.diagonal(ssm, offset=L).mean()) for L in candidates}`
— for each candidate period length `L`, this averages the self-similarity
matrix's `L`-diagonal across *every* starting position `i` simultaneously.
That's correct for answering "does a repeat of length `L` exist somewhere,"
but by construction it never identifies *which* beat index is the true
start of a repeat — there is no companion computation anywhere in the file
that solves for a phase offset. Every downstream consumer that needs
"position within the loop" (e.g. `E_position_in_loop =
beat_idx % period` in `scripts/build_chord_change_features.py`) is
therefore forced to use beat 0 of the song as an arbitrary phase reference,
with no guarantee it coincides with any real repeated-section boundary.

**Consequence:** anything built on "distance into the loop" or "loop-start"
today is silently mis-phased for a substantial fraction of songs. This is
distinct from, and additional to, the already-documented Candidate C
finding above (that cross-repeat chord *averaging* hurts `majmin`/`tetrads`)
— that result was about content once phase is known; this is about not
even knowing the correct phase in the first place.

**Fixed.** Added `find_loop_phase(period, is_downbeat)` to
`harmonia/models/periodicity.py`: anchors phase 0 to the first annotated
downbeat (`downbeat_idxs[0] % period`), rather than beat 0 of the song.
Considered, and rejected, a chroma-self-similarity-based approach first
(score each candidate phase by mutual similarity of its members) — proved
unsound on inspection: a cleanly repeating signal is, by construction,
equally self-similar under *any* phase choice, so the SSM alone can never
break that symmetry. Only external information (the downbeat annotation)
can. `tests/test_periodicity.py::TestFindLoopPhase` (5 tests, including a
song-with-pickup-beats case) covers this. Wired into
`scripts/build_chord_change_features.py` (`E_position_in_loop` now uses
`(beat_idx - loop_phase) % period`; new `E_loop_phase` column added for
transparency), and a new garde-fou pair in
`scripts/validate_chord_change_features.py` guards against regressing to
the old zero-overlap signature.

**Effect of the fix, measured directly** (`features.csv` regenerated;
verified only `E_position_in_loop`/`E_loop_phase` changed, every other
column byte-identical): loop-start/downbeat overlap went from 1/5 songs a
clean subset (with 2/5 *fully disjoint*) to 4/5 a clean subset and the
fifth at 93.8% — pooled overlap 39%→91.5%. Songs 003 and 004, the two that
were fully disjoint before, picked up non-zero phase corrections (phase=2)
exactly as the bug predicted. The residual gap in songs 002/005 (also the
two songs independently flagged as having rare irregular >4-beat
inter-downbeat gaps) is a separate, smaller limitation — this fix assumes
one fixed period+phase for the whole song, which drifts if a bar
elsewhere is irregular — not a sign the anchoring itself is wrong.

Re-running the `A_beat_phase × E_position_in_loop` joint analysis this bug
had originally surfaced (`docs/chord_change_signal_analysis/findings_AE_DE.md`)
with corrected data changed the conclusion, not just the numbers: the
pre-fix "positive lift" (63.8%→89.4%) was a Simpson's-paradox artifact of
comparing sets that didn't actually overlap; with the sets properly
nested, the fair comparison shows **no lift** (4 of 5 songs show
equal-or-lower P(chord_changed) for loop-start vs. other downbeats). This
turns what was an "inconclusive, needs more data" verdict into a settled
negative result.

---

## 2. Soundfont quality — TESTED 2026-07-02, modest win, worth keeping

**Found the actual bug first:** `data/soundfonts/GeneralUser.sf2` is
mislabeled — `strings` on the file shows its real internal name is
`"Vintage Dreams Waves v 2.0"` (Ian Wilson, 1996), a small 307KB soundfont,
not the real GeneralUser GS (~30MB) the filename claims. All 5 songs'
`prog0` renders (the ones used throughout this investigation) were
synthesized with this low-fidelity file the whole time.

Downloaded a real high-quality replacement:
[MuseScore_General.sf2](https://ftp.osuosl.org/pub/musescore/soundfont/MuseScore_General/MuseScore_General.sf2)
(215MB, MuseScore's own well-regarded GM soundfont — a solid, readily
available substitute for GeneralUser GS). Re-rendered all 5 songs'
`001.mid`-`005.mid` through it with identical `RenderConfig` (only the
soundfont differs) → `*_v005_musescoregeneral.wav`.

| Metric | old (Vintage Dreams) | new (MuseScore General) |
|---|---|---|
| per-beat argmax root-accuracy (metric 1) | 16.8% | 17.3% |
| boundary F (metric 2) | 0.215 | **0.241** |
| root (metric 3) | 22.7% | 21.5% |
| majmin (metric 3) | 17.0% | 15.4% |

**A different pattern than candidates A or B:** boundary F improved
meaningfully (the best improvement of any experiment run so far), and the
raw per-beat metric improved slightly too — real evidence the better
soundfont genuinely improves Basic Pitch's transcription, giving more
accurately-timed chord changes (`n_events` roughly doubled on most songs,
moving closer to GT's true rate). But root/majmin didn't follow — they're
flat to slightly worse. Same underlying story as A and B: getting *when*
right doesn't automatically get *what* right when quality discrimination
is the binding constraint.

**Side finding, not yet investigated:** song 005's beat count changed from
595 to 298 beats (almost exactly 2x) between the two renders of the *same*
MIDI file — the new soundfont's different acoustic characteristics (attack/
reverb) evidently threw off librosa's beat tracker into a different tempo
octave for that one song. A real confound worth being aware of when
comparing soundfonts: differences aren't purely about transcription
quality, they can also silently change the beat grid itself.

**Confirmed as a real, recurring failure mode (2026-07-04), isolated to a
specific song:** comparing our librosa-derived tempo against POP909's own
annotated tempo directly (not soundfont-vs-soundfont this time) across all
5 songs: 001 90.0 vs 89.1 BPM, 002 **63.0 vs 129.2 BPM**, 003 82.0 vs 80.7,
004 71.5 vs 71.8, 005 64.9 vs 64.6. Four of five songs agree with ground
truth to within ~1 BPM; **song 002 alone** shows our beat tracker locking
onto almost exactly double the true tempo (ratio 2.05x). Not yet
investigated why song 002 specifically is vulnerable (vs. 001/003/004/005
being fine) — worth checking if `docs/chord_change_signal_analysis/` work
continues, since anything measuring "beats" for song 002 via our audio beat
tracker (not POP909's own grid) inherits this error silently.

**Conclusion: adopt as the new default going forward** (real, if modest,
net-positive result — nothing else tested so far has improved boundary F
this much) but it does not resolve issue #1 on its own; proceeding to
Candidate C with this as the new baseline soundfont.

---

## 3. Zero test coverage on `pipeline.py` and `eval/mirex_eval.py` — OPEN

Both are only exercised by manually running `scripts/evaluate.py` against real
audio — not part of the fast `pytest` suite. This is a process risk, not an
accuracy bug: both the N-collapse bug and the `_label_to_mireval` crash existed
undetected for an unknown time precisely because nothing exercised those code
paths automatically. Needs audio fixtures or mocked activations to test
properly; deferred due to effort/cost tradeoff so far.

---

## 4. Lossy quality mapping in `_label_to_mireval` for phase-2+ chords — OPEN, low priority

mir_eval has no native shorthand for some altered/suspended-7th qualities
(`7b9`, `7#9`, `7sus4`, augmented-7th, etc.) — `_QUALITY_TO_MIREVAL` in
`harmonia/eval/mirex_eval.py` approximates these down to the nearest supported
quality for *scoring* purposes only (doesn't affect model predictions). Low
priority because the default chord vocabulary (phase 1) barely uses these
qualities; revisit if/when extending to phase 2+ (9ths, altered dominants).

---

## 5. Emission-template geometry cannot separate the qualities it's blamed for missing — CONFIRMED 2026-07-03, no audio involved

**Superseded 2026-07-06: the production path no longer scores chords via
template dot-product at all** — see the status note at the top of this file.
The new trained root/family models don't have this failure mode by
construction (a learned classifier boundary, not a fixed-template argmax), so
this finding no longer describes a live bottleneck. Left below as a real,
correctly-diagnosed result about `chord_hmm.py`, in case that module is ever
revived.

This is issue #1's "what this points to next (1)", now actually run
(`scripts/check_emission_separability.py`, pure template geometry, no
audio anywhere). Two results:

**(a) Under a PERFECT observation — the chord's own emission row fed back as
the observation — the emission model misidentifies 9 of 180 chords.** Every
`X:7sus4` (9 of 12 outright, the other 3 by margins of +0.2%) scores higher
against the sus2 template rooted a whole tone below (`D:7sus4` → `C:sus2`,
etc.). Root is wrong, not just quality. Mechanism: 7sus4 {0,5,7,10} and the
bVII's sus2 {10,0,5 rel.} share their three strongest pitch classes, and
**row-L1-normalisation makes templates with fewer notes systematically
"sharper"** — a triad's row concentrates its mass on 3 pitch classes where a
tetrad spreads it over 4, so a tetrad's own ideal observation often dot-products
higher against a subset-triad's row than against its own. This is a structural
bias of `beat_probs @ E.T` with row-normalised E, and it can never be fixed by
better audio — it's a ceiling. Directly explains the A/B/C convergence
("more responsive decoding exposes quality confusion"): the confusion is
partly built into the template geometry itself.

**(b) Same-root cosine similarities quantify the rest:** dim/ø7 and dim/°7
0.90, min7/ø7 and maj7/augMaj7 and 7/aug7 0.89, maj/maj7, maj/7, min/min7,
sus4/7sus4 all 0.87. °7's three enharmonic transpositions are 0.993 apart
cross-root (expected — genuinely the same pitch-class set; arguably should be
merged in the vocabulary rather than scored as errors).

**Implications, in order of likely value:** (1) reconsider row-normalisation
(e.g. normalise by L2, or score with cosine instead of dot product, so note
*count* stops being a bias); (2) merge or de-duplicate near-degenerate
vocabulary entries (°7 transpositions; possibly 7sus4-vs-sus2 needs an
explicit disambiguating weight on the 7th); (3) this strengthens the case
that phase-1's 15 qualities exceed what the template scoring can support.

**(1) implemented and A/B tested 2026-07-03 — real fix at the layer it
targets, net negative on the full pipeline, not adopted.**
`ChordInferrer(emission_scoring="cosine"|"dot")` (default `"dot"`, unchanged
behaviour) L2-normalizes both the observation and each template row before
scoring instead of the raw `beat_probs @ E.T` dot product.
`tests/test_chord_hmm.py::TestEmissionScoring` confirms cosine scoring
resolves **all 9/9** ideal-observation misidentifications from the finding
above (dot scoring still reproduces all 9, unchanged — the defect is real
and this is a real fix for it in isolation).

Full 5-song pipeline (`scripts/experiment_issue1.py --sweep-emission-scoring`,
`v005_musescoregeneral` renders):

| variant | boundary F | root | majmin |
|---|---|---|---|
| dot (baseline) | 0.276 | 33.3% | 29.9% |
| cosine | 0.263 | 29.1% | 26.4% |

4 of 5 songs regress on both root and majmin (song 003 alone improves,
marginally). **Same convergence as issue #1's candidates A/B/C**: a fix
that is provably correct against its own targeted diagnostic still nets
negative end-to-end. Working explanation, by analogy with Candidate A3's
`sqrt`/`log1p` finding: cosine similarity is bounded to `[0, 1]` regardless
of how much evidence a beat carries (a beat with 6 clearly-struck notes and
a beat with 1 faint one can produce similar-magnitude scores if their
*shapes* are similarly close to a template), which compresses
`log_emission`'s dynamic range across beats — Viterbi's per-beat evidence
term becomes weaker relative to the (unboosted-in-this-experiment but still
present) transition/duration prior, letting the prior override real
per-beat signal more often than the dot product's magnitude-sensitive
scores did. Consistent with the standing diagnosis: **emission
discriminability improvements keep getting absorbed or outweighed by the
decoder's prior structure**, not just failing to help.

**Conclusion: not adopted** (`emission_scoring="dot"` remains the default).
Real, tested code path (`tests/test_chord_hmm.py::TestEmissionScoring`,
`scripts/experiment_issue1.py --sweep-emission-scoring`) kept for reuse —
e.g. worth revisiting in combination with a duration-aware or
periodicity-folded decoder where the emission term's relative weight is
controlled by a separate tunable, rather than only by its own dynamic
range.

---

## 6. `build_emission_matrix` silently drops template intervals > 11 — FIXED 2026-07-03 (was phase ≥ 2 only, latent under the phase-1 default)

`interval = (pc - root) % 12` is always 0–11, but phase-2+ templates keep
extension intervals as 13/14/15/17/18/21 in `weights` — `if interval in
template.weights` never matched them. Confirmed directly: with
`max_phase=2`, `E[C:9]` and `E[C:7]` differed by at most 0.0025 (a
renormalisation echo of the 5th's weight, 0.25 vs 0.3), and C:9's mass on
pitch-class D (its defining 9th) equalled the noise floor. **Every
9th/11th/13th chord's emission row was silently just its underlying 7th
chord.** Harmless at the phase-1 default, would have been guaranteed
confusion the day phase 2 was enabled. `ChordTemplate.to_weight_vector()`
already folded with `% 12` correctly — the two code paths disagreed.

**Fix:** `build_emission_matrix` now folds template interval keys mod 12
before the lookup (max, not sum, on collision). `tests/test_chord_hmm.py::
TestEmissionExtensionIntervals` (3 tests, red-first: confirmed failing
against the old code, e.g. C:9's ninth-pitch-class mass equalled C:7's
noise floor) — checks the 9th now carries real mass, that no two
same-root qualities at `max_phase=2` share an emission row, and that every
row's above-floor pitch-class set matches its template's own
`to_weight_vector()` support. Phase-1 rows are byte-identical (all phase-1
intervals are already ≤ 11) — this only changes behaviour once phase 2+ is
enabled.

---

## 7. `POP909Parser` discarded beat_midi.txt's ground-truth downbeat column — FIXED 2026-07-03

`beat_midi.txt` is 3 columns: time, half-bar flag (spacing 2), **downbeat
flag (spacing exactly 4)** — verified on song 001. `_parse_beat_file()` kept
only column 0, so `POP909Song` had no downbeat data, while
`docs/architecture_extensions.md` item #9 (beat-phase-aware harmonic-rhythm
prior) was blocked on "needs real downbeat detection (madmom or
equivalent)" — **the ground-truth downbeats were already in the annotation
file being parsed.** Meanwhile `scripts/build_chord_change_features.py` and
`scripts/build_symbolic_features.py` each carried a private
`_load_pop909_beat_grid()` that read column 2 correctly — duplicated
parsing logic that could drift from the canonical parser.

**Fix:** `POP909Song` gains `is_downbeat: np.ndarray` (parallel to
`beat_times`) and a `downbeat_times` property; `_parse_beat_file()` returns
both columns. The two scripts' `_load_pop909_beat_grid()` is now a thin
wrapper delegating to `POP909Parser` (kept only because
`build_symbolic_features.py` imports it by name) instead of re-reading the
file. `tests/test_pop909_parser.py::TestDownbeatGroundTruth` (3 tests)
confirms 73 downbeats on song 001 at exact 4-beat spacing and that
`downbeat_times` is a true subset of `beat_times`. (Real audio-only
downbeat detection is still needed for non-POP909 input eventually, but
every POP909 experiment can use GT downbeats today.)

---

## 8. Dead code that crashed if ever called — FIXED 2026-07-03, low priority

- `RhythmAnalyser.analyse_from_midi()` crashed immediately:
  `pm.get_tempo_change_times()` is not a pretty_midi API
  (`get_tempo_changes()` returns the `(times, tempi)` tuple directly).
  Confirmed by running it on 001.mid before the fix. No callers anywhere.
  **Fixed** (one-line call-site correction);
  `tests/test_rhythm.py::TestAnalyseFromMidi` (new file) covers both real
  POP909 MIDI and a synthetic no-tempo-event case.
- `KeyEvent.is_no_chord` / `KeyEvent.duration_beats()` referenced
  `self.quality` / `self.end_beat`, which don't exist on `KeyEvent`
  (copy-paste from `ChordEvent`). No callers — **removed**.
- `POP909Song.chord_at_beat(beat)` compared its argument against
  `start_beat`/`end_beat`, which hold **seconds** — any future caller
  passing a beat index would have gotten silently wrong results. No
  callers today. **Renamed to `chord_at_time(t)`** with a docstring
  explaining the seconds gotcha, rather than left as a trap.

Also quantified while scanning, not acted on: `maj6`/`min6` labels (723 +
58 = 781 events corpus-wide, ~0.65%) are mapped to plain maj/min triads by
`_QUALITY_MAP` — same "GT-mapping artifact" family as the inversion finding
in issue #1, but an order of magnitude rarer; and `evaluate_song()` returns
an all-zeros score on any internal exception (logged as a warning only) — a
crashed song silently drags dataset averages down instead of being
excluded (related to issue #3's zero-coverage risk). Neither is fixed yet.

---

## 9. Beat tracking is NOT the end-to-end bottleneck; drum/bass stem isolation does not help on synthetic data — MEASURED 2026-07-06

Context: end-to-end v0 (`scripts/pipeline_v0.py`) hits ~67% root with detected
beats+boundaries but 86.8% with oracle boundaries, so "where does the gap go?"
Three probes, all against the **known MMA tempo grid** (exact GT beats =
`k·60/tempo`, no count-in — verified) and the GT `section_per_bar`:

- **Beat tracking (`scripts/beat_tracking_experiment.py`).** librosa
  `beat_track` on the full mix scores F=0.879 clean, **0.872 degraded** — noise
  barely dents it. An isolated drum stem does NOT help (0.876 clean, 0.827
  degraded — *worse* degraded); stripping drums hurts (0.72). So beat detection
  is not the bottleneck, and drum-stem-based beat tracking is a dead end. The
  67→86.8% gap is segmentation + emission on real evidence, not beat placement.

- **Drum-self-similarity structure prior (`scripts/stem_benefit.py`).** Premise
  falsified before building: **MMA renders ONE groove for the whole tune** — the
  drum voice set is identical across sections (e.g. A vs B both {35,37,44,46,63,
  64}) and per-bar hit density is flat (14–28) with no jump at section
  boundaries. Real drummers mark A→B→bridge with fills/pattern swaps; our
  synthetic data does not reproduce this, so the drum-structure prior is
  **untestable on the current DB** (rule #3/#5: the phenomenon is absent from GT).
  To pursue it we'd need real audio or per-section MMA grooves.

- **Bass-stem isolation for root.** Isolated bass stem → per-beat root = 0.20;
  low register of the full mix (what pipeline_v0 uses) = 0.24. Isolation did NOT
  help — the full-mix low register already separates bass well enough, and the
  MuseScore bass stem's muddy low end reads no better through Basic Pitch.
  (Per-beat exact-match is a harsh metric vs MIREX weighted overlap; the
  *relative* comparison shares the grid so it holds regardless.)

Method note also logged: Foote **checkerboard novelty finds local contrast, but
AABA structure lives in repetition** (bar i ≈ bar i+16 when A returns). On
symbolic chords the novelty detector scores only F=0.25 for section boundaries
because jazz harmonic rhythm (ii-V-I every 2 bars) produces stronger *local*
novelty than the sections do. A repetition/time-lag SSM is the right tool for
sectional structure, not novelty — but see the constant-groove blocker above.

Licensing (asked re: a paid product): full runtime stack is commercial-safe —
MIT/BSD/Apache/ISC/PSF (numpy, scipy, librosa, basic-pitch, music21, torch,
torchaudio, onnxruntime, scikit-learn, …). Two watch-items: `soxr` is
LGPL-2.1 (fine as a shared lib, or avoid via librosa `res_type`), `tqdm` is
MPL-2.0 (file-level, fine). One landmine: **`madmom`** (listed as an optional
dep in pyproject) has a non-commercial research-license clause — must NOT ship
in a commercial build. Source separation for the real-audio path: **HDemucs is
already bundled in torchaudio** (MIT weights from Meta), so no new dependency —
though its MUSDB18-HQ training provenance is worth a lawyer's glance for a paid
product; Spleeter (Deezer, MIT code+models) is the belt-and-suspenders fallback.

**Addendum 2026-07-14 — madmom installed + wired, MEASURED: does NOT fix the
octave-lock (issue #1 premise), reinforcing this section.** Full writeup:
`docs/madmom_reinference_results.md`; data `docs/tempo_comparison_madmom.json`;
plot `docs/plots/tempo_comparison_madmom.png`.
- `madmom 0.16.1` now builds+imports on py3.12/numpy-2.x via a compat shim
  (`rhythm.py::_ensure_madmom_compat`, restores `collections`/`np.float` names).
  `infer_chords_v1(beat_backend="madmom")` is opt-in; **default stays librosa**.
- On the 10-song `docs/audio` corpus (faithful 44.1k production load path):
  madmom fixes the octave on **0/10**. Where it diverges from librosa it prefers
  the *half*-note pulse; on the only exact reference (blue_bossa 150-BPM backing
  track) madmom = **75.0 = exactly ½×** vs librosa 99.4 — madmom is *worse* on
  every reference-anchored song. Where the two agree (ghost, autumn, airegin,
  adele) they agree at the same *wrong* octave. Both trackers sit in [55,215] and
  merely pick the wrong multiple → the real lever is an octave *disambiguator*
  (match beat count to harmonic-rhythm / chord-change rate), not the tracker.
- madmom **downbeat** detection crashes under numpy 2.x (`inhomogeneous shape`
  in `DBNDownBeatTrackingProcessor`) on every song → no metre benefit here.
- **Licence reminder still stands** (above): madmom is non-commercial research
  licence — must NOT ship in a paid build. It's a dev/eval convenience only.
- **Env landmine found:** the editable install maps `harmonia` → the stale
  `~/harmonia` clone; scripts run as files (incl. the server) import the STALE
  copy unless the canonical repo is forced onto `sys.path`/`PYTHONPATH`. So the
  madmom wiring will not reach the server without repointing the editable install
  or launching with this repo on the path.

---

## 10. Family-emission features are unnormalized summed chroma (duration-dependent) — FIXED 2026-07-06 (in chord_change_engine)

`build_audio_chord_features.reg_chroma`/`full_chroma` return raw *summed* chroma,
whose magnitude scales with the number of beats in the segment. A family model
trained on oracle-segment scales and applied to segments of a different typical
duration (e.g. the coarse chord-change engine's merged segments) receives inputs
off the training distribution even after StandardScaler → degraded quality. This
silently capped end-to-end majmin at 58.8% despite family-given-root being 94.4%.
Fix in `chord_change_engine.py` (`norm_blocks`): L2-normalize each 12-chroma block
so features are duration-invariant → majmin 58.8% → 82.8% (clean), 78.6% (degraded).
The extracted table `data/cache/audio_chord_features.npz` itself is still raw-sum;
any consumer must normalize per-block. (Same silent-calibration family as issue #0
and the frame-rate/tempo bugs in CLAUDE.md rule #1.)

---

## 11. Chord-change engine's MIREX root/majmin are UNDERSOLD by a GT-source mismatch — found 2026-07-06

The engine reports MIREX root ~75% on oracle boundaries, but the root MODEL is fine:
applied to freshly-extracted oracle segments and scored against its own training
target (`song_chord_spans` root), it gets **91.6%** on held-out-ish eval songs
(matching its 93% CV). The gap is in the eval harness, not the model:

- estimated segmentation comes from `gt_chord_per_beat` (a per-beat re-parse of
  `ireal`/`mma`), while the MIREX reference is built from `song_chord_spans`.
- these two GT chord representations disagree (raw span-vs-per-beat agreement ~62%,
  much of it boundary-timing), so est segments straddle ref-chord boundaries and
  MIREX penalizes correct roots that land on a misaligned segment.

Consequence #1: the reported ORACLE-bounds numbers (root ~75%) were artifactual —
the true labeling ceiling is **~91% root**. Fix: build the oracle segmentation from
`song_chord_spans` (same source as the reference), not `gt_chord_per_beat`.

Consequence #2 (REVISES an earlier conclusion): with the corrected ceiling, standalone
root ~74% vs labeling ceiling ~91% means **segmentation costs ~17 points after all** —
so the session-earlier claim that "segmentation is at its useful ceiling / oracle
boundaries don't help" (built on the artifactual 75% oracle number) is WRONG. Boundary
quality does matter; the zoom / better-segmentation work is worth revisiting. The
standalone eval itself uses a consistent `song_chord_spans` reference so its ~74/70%
figures are honest; only the oracle-bounds *comparison* was corrupted.

Fix: align the harness to a single GT chord source before trusting oracle-vs-coarse
comparisons. Related: rule #3 (ground truth is a measurement) and issue #1.

**FIXED 2026-07-06 (commit 72af4ec):** segmentation, per-beat GT, change-times and the
reference now all derive from `song_chord_spans`. Revealed the true ceiling (oracle
root 89% / majmin 94%) and that boundaries cost ~9-10 pts. Follow-on boundary fix
(commit f9ae502): lower θ (0.15→0.08, favour recall) + coalesce adjacent same-chord
segments → GT-grid majmin 83.7→89.4%; disjoint standalone majmin 70.5→74.9%. The
earlier "exact placement 0.50 / zoom headroom" and "segmentation at its ceiling" were
both artifacts of this bug.

Separately (root model tuning, `scripts/root_improve.py`): templates-as-features +
MLP lifts per-segment root CV 93.4% → 95.0% (bass sub-bands don't help); a modest,
real gain available once the harness is fixed and worth wiring.

---

## 12. Motif stacking: 51% of chord-slots are redundant repeated patterns — MEASURED 2026-07-07

**The compression opportunity.** Across the 150-song corpus, half the chord stream
is repetition of a small set of bar-aligned motifs (ii-V, turnarounds, dominant
cycles). A greedy motif detector (transpose-invariant "shape" view) compresses
the average song from 47 chords to 22 meaningful units (mean 51%, median 51%),
using only 4.2 unique motifs per song. Best cases hit 84%.

**What it enables (future):**
- **Folded lead-sheet rendering** — repeat brackets, motif labels, compressed charts.
- **Voting across motif copies** — if the same ii-V appears 16 times, the model's
  strongest decode can correct the weakest.
- **Structural prior for the HMM** — knowing "this bar is probably another ii-V"
  is free evidence that the current HMM doesn't use.

**Measured accuracy gain from motif voting (2026-07-07, N=150 songs):**

| Condition | Family | Seventh | Exact |
|-----------|--------|---------|-------|
| Audio only | 96.6% | 93.4% | 91.3% |
| Motif fold | 96.8% | 93.0% | 90.6% |
| GT fold (oracle) | 96.7% | 93.7% | 91.6% |

Deltas: motif +0.2% / −0.4% / −0.7%; GT oracle ceiling +0.1% / +0.3% / +0.3%.
**Conclusion (clean audio):** Near-zero gain — no headroom to close.

**Hard-audio blind experiment (N=150, full multi-stem + SNR 3–20 dB, no GT helpers):**

| Condition | Family | Seventh | Exact |
|-----------|--------|---------|-------|
| Blind audio | 54.3% | 25.2% | 22.4% |
| Blind + motif fold | 54.3% | 24.4% | 21.1% |

**Conclusion (hard audio):** Decision-level motif voting *hurts* (−0.8% / −1.3%).
With 54% family accuracy the inferred chords are too noisy — grouping averages
errors together. The correct approach is **feature-level averaging**: pool raw BP
activations across motif instances *before* classifying. Not yet implemented.

**What it doesn't solve:** section-level folding (recovering AABA labels from chords
alone) tops out at meanARI ~0.49 on GT chords. The human's section labels don't
correspond to clean chord-repeat units — a 16-bar "A" section typically has two
different 8-bar halves. Melody/timbre boundaries needed for that.

**Files:** `harmonia/models/motif.py`, `harmonia/models/block_fold.py`,
`scripts/demo_motif.py`, `scripts/render_motif_chart.py`.

---

## 13. `vary_voicings` created independence by omitting pitch classes (wrong knob) — FIXED 2026-07-07

**Symptom:** `--fold --vary` in `chord_change_engine.py` netted −13 majmin vs no-fold, even
though structure folding (+pooling) should help. `structure_fold_experiment.py` showed
independent repeats adding +8.3 root pts vs +6.1 correlated — so independence helps. The
combined flag hurt because `vary_voicings` was creating "independence" by omitting 1–2 upper
pitch classes per occurrence, which thinned the harmony and confused the chord classifier.

**Root cause (`scripts/build_accomp_audio_hard.py`, `vary_voicings`):** the function dropped
notes with `if n.pitch % 12 in omit: continue` (PC-level omission) and also randomly dropped
individual notes with 12% probability (`if r.random() < 0.12: continue`). Both changed the
chroma vector, defeating the purpose (same chord, different surface).

**Fix:** remove all note-dropping. Keep all pitch classes intact. Vary only:
- octave shifts (30% per non-bass note, ±1 octave)
- velocity (±25% swing)
- micro-timing jitter (±15ms)

Independence now comes from the audio surface (different BP onset errors per repeat) not from
changing the harmony. Verified: note count unchanged, PC set identical, zero PC diff on one
song. The `--fold --vary` combination should now be re-evaluated end-to-end.

**What this does NOT fix:** the DB has not been regenerated yet. Existing `data/accomp_db/`
renders used the old omit-voices vary_voicings. Re-running `--vary` in the engine uses the
fixed function, but a full DB regen (for training fold-robust models) is a separate step.

---

## 14. Production entry point upgraded to Gen-2 (chord_pipeline_v1) — DONE 2026-07-08

The production pipeline was upgraded from the frozen Gen-1 `HarmoniaPipeline` / `ChordInferrer`
to the new Gen-2 `chord_pipeline_v1.infer_chords_v1`. Both `scripts/harmonia_server.py` (web
server) and `scripts/render_youtube_chart.py` (CLI) now call `infer_chords_v1`.

**Gen-2 improvements wired in (all calibration bugs fixed by design):**
- Beat-sequence model (`beat_seq_model.npz`, 88.3% per-beat root CV) — window=2, SUM pooling
- Trained family + seventh classifiers from `audio_chord_features.npz` (norm_blocks, root-shift)
- Tempo-grid de-jitter (uniform grid at librosa tempo + circular-mean phase)
- Coarse segmentation at θ=0.08 cosine novelty with same-label coalescence
- Optional ctx family model (`ctx_family_model.npz`, 86.9% CV) — auto-loaded when present

**Measured on POP909 (5 songs, musescoregeneral renders):**
| pipeline | root | majmin |
|---|---|---|
| Gen-1 (frozen) | ~33% | ~29% |
| Gen-2 v1 | **60.5%** | **39.1%** |

**Ctx model training completed 2026-07-08:** `harmonia/models/ctx_family_model.npz` saved.
Gate accuracy 86.9% vs 85.1% LR baseline on MMA jazz (trained on 60 songs). Neutral on POP909
(different domain — expected; should help on jazz audio).

---

## 17. beat_seq_model_v3: architecturally key-invariant root+quality heads — TRAINED 2026-07-08, integration OPEN

New root/quality model that achieves key-invariance *by construction* instead of
by rotation augmentation (v2's mechanism). `scripts/train_beat_seq_model_v3.py`,
saved to `harmonia/models/beat_seq_model_v3.npz` (local-only, `*.npz` gitignored).

- **ROOT head — canonical-form scorer (equivariant by weight-tying).** For each of
  12 candidate roots r, roll all chroma blocks in the 240d windowed feature by -r
  (candidate → pc 0), score with a *shared* 96-unit ReLU MLP → scalar; argmax over
  candidates = root. Same function scores every candidate, so rolling input by s and
  label by s only cyclically permutes the 12 scores → identical loss.
- **QUALITY head — DFT magnitudes (invariant by construction).** Each 12d chroma
  block → `|rfft|[:7]`; |DFT| is exactly shift-invariant, so the head cannot encode a
  key-biased quality prior. 4 blocks × 7 × 5 beats = 140d → LR over 5 classes
  {major, minor, dom7, maj7, dim}.

**Eval (per-beat, POP909 001-005, pure-numpy `V3Model.predict_proba`):**
- v3 root **78.2%** (77.6% excl. song 002's 2× tempo), majmin **62.2%**.
- v2 root via the *same* harness: **79.4%**. So v3 ~ties v2 on root while dropping
  the augmentation dependency, and adds a majmin capability v2 lacked.

**⚠ Calibration discrepancy — the stated v2 baseline of "root 70.4% / majmin 45.0%"
could NOT be reproduced.** Through `train_beat_seq_model_v3.py --eval` (renders in
`data/renders/pop909/*/`, tempo-grid beats, GT via `POP909Parser.chord_at_time`), v2
scores **79.4% root**, not 70.4%. Either the 70.4 figure is stale or it used a
different beat grid / valid-beat counting convention. Both v2 and v3 are scored by
byte-identical code, so the fair comparison is 78.2 vs 79.4 — but do NOT re-quote
70.4 without finding its provenance. (Surfaced by the built-in `[calibration]` line;
process-rule #1.)

**Ablations (reuse cached trainset, no re-render):**
- Chroma-template prior (`chroma_root_template.npz` log-lik as a per-candidate
  feature): **no help** — 78.2% off vs 78.1% on. The MLP already extracts root
  evidence from the rolled chroma. Final model ships with it OFF.
- Rotation augmentation: **a no-op for v3** (78.2 no-aug vs 76.3 with-aug, within
  init/SGD noise), exactly as the equivariance argument predicts — vs v2, which
  *needed* it (60.5→70.4). Goal met: augmentation is now optional insurance.
- 60 epochs overfit (95.5% train → 76.4% eval); 40 epochs generalizes better (78.2%).

**Next:** v3 is NOT a drop-in for `_BeatSeqModel` (canonical scorer ≠ flat-LR npz
format). Wiring it into `chord_pipeline_v1` needs a small loader class there
(`_get_beat_seq` + a `V3Model`-style predictor) — left untouched for now. Then run
end-to-end POP909 MIREX (not per-beat) to compare against the ctx-v2 root head (#16).

---

## 18. beat_seq_model_v3 "design-brief" baselines were misattributed numbers — RECORDED 2026-07-09

A v3 design brief circulated with a "diagnostic block" instructing a rebuild of
beat_seq_model_v3 to beat **"POP909 001-005: overall 77.6% root, interior 93.6%,
boundary 64.1%, beat-1 62.2%, 70.4% pipeline."** A provenance hunt (process rule
#1/#2) found **none of these are a real POP909 per-beat breakdown** — they are
unrelated real numbers relabeled, plus two with no provenance at all:

| brief claim | actual source | what it really is |
|---|---|---|
| 77.6% overall per-beat root | `blog/14` L164 | jazz1460 standalone-disjoint root (30 *odd* songs, tempo grid) — **not POP909**. The one legit number, wrong corpus. |
| 93.6% "interior" root | `chord_change_engine_2026-07-06.md` L276 | jazz1460 **majmin oracle-bounds ceiling** — not interior, not root, not POP909 |
| 62.2% "beat-1 downbeat" root | commit `08db7b2` message | v3's overall **majmin CV** ("78.2% root, 62.2% majmin") — relabeled |
| 70.4% pipeline / 64.1% boundary | — | no accuracy provenance found (appear only as timestamp/row values in `chord_change_signal_analysis/features.csv`) |

**Same family as issue #17's calibration warning** ("v2 baseline of 70.4% could
NOT be reproduced"). The real, single-harness POP909 001-005 per-beat numbers
(v2, renders + BP-cache, both beat grids) are:

| | librosa grid | GT beat grid |
|---|---|---|
| overall root | 79.4% | 81.9% |
| interior | 83.9% | 85.9% |
| boundary | 72.8% | 76.0% |
| downbeat (pos-0) | 84.9% (**best**, librosa `b%4`) | 79.4% (~avg, GT phase) |
| P4/P5 (+5/+7) error share | 45.4% | 46.6% |

So: (1) "interior is saturated at 93.6%" is false — real headroom (~84-86%) is
spread across interior *and* boundary beats. (2) "beat-1 is worst (62.2%)" is
reversed on the librosa grid the pipeline actually uses, and only mildly true
(76.7%) under GT-downbeat phase the pipeline doesn't have. (3) **Architecture B
of the brief (canonical-form root scorer) is already built** (`beat_seq_model_v3.npz`,
issue #17) and scores 78.2% vs v2's 79.4% — the "biggest gain" exists and is
net-neutral, because P4/P5 confusion is acoustic (5th-apart, bass) not a learnable
key-bias. **The one solidly-real diagnostic is the P4/P5 confusion (46% of errors),
independently corroborated the same day by the bass-tracking session ("root
classifier's F#/C# confusion, not segmentation").**

**Canonical GT going forward: irealb/jazz1460** (metronomic, exact beat grid,
GT = the chart via `song_chord_spans`), *not* POP909 001-005 — which is why the
77.6% jazz-standalone figure looked unreproducible when re-measured on POP909.
Reusable real-breakdown harness left in `scripts/diag_boundary_interior.py`,
`diag_downbeat_position.py`, `diag_grid_and_errors.py`.

### Reframe — the real lever is a root PROGRESSION/context model, not more single-beat arch (2026-07-09)

Premise check on the canonical corpus (`scripts/diag_fifth_confusion_jazz.py`, 25 odd
jazz songs, exact tempo grid, v2 per-beat): **+5/+7 (root↔4th/5th) = 51.0%** of root
errors (confirms the "52%" claim *on jazz*, where walking bass makes it worse than
POP909's 46%). Crucially those errors are **rescuable, not an acoustic wall**: of the
5th-apart errors, the true root is in v2's **top-2 for 85.9%**, top-3 for 95.2%, and
is an adjacent beat's argmax for 82.4% → **92.5% rescuable** by a context/progression
prior. So the true root is present in the neighbourhood; it just loses the per-beat
argmax to the 5th. The wall is NOT Basic Pitch bass transcription.

### Vacuum bake-off of root models (`scripts/bakeoff_root_models.py`, 2026-07-09)

Leak-free ranking: irealb/jazz1460, ORACLE chord segments (isolates root-ID from
segmentation), mean chroma per chord ±4 chords context, disjoint 35-train/35-eval
(1501 eval chords), every learnable model trained on train split only. Metric =
per-segment root accuracy (unweighted / dur-weighted) + +5/7 share of remaining errors.

| model | root | wtd | +5/7 |
|---|---|---|---|
| **canon ⊕ bass-anchored (avg) — super-model** | **96.1%** | **96.1%** | 43.1% |
| canon (full-chroma, key-agnostic, ±4 ctx) | 95.8% | 95.9% | 23.8% |
| bass-anchored (full chroma) | 94.7% | 94.9% | 43.8% |
| twopath (canon+abs merged) | 94.1% | 94.0% | 30.7% |
| abs_query (chord's own mean chroma) | 92.7% | 93.8% | 54.5% |
| ltas_canon (canonical over 5-family log-lik) | 92.3% | 91.8% | 24.1% |
| abs_ctx (plain LR, key-biased) | 91.4% | 91.7% | 38.8% |
| viterbi (abs_ctx + relative root-bigram) | 90.5% | 90.6% | 29.6% |
| bass=root (naive dominant-bass anchor) | 77.7% | 81.0% | 57.2% |

Findings: (1) **within-chord mean chroma alone → 92.7%** (averaging over the chord's
beats makes the root dominate the 1-then-5 bass, as predicted). (2) **key-agnostic
canonical + ±4 context is the single best design (95.8%)** and halves the 5th-error
share (54.5→23.8) — the rescuability thesis realized. (3) the "two-path" (adding a
separate ABSOLUTE root path) *underperforms* canon-alone: canon already sees the query
chroma, and the absolute path relearns non-transferable key-specific mappings on a
disjoint-key eval. (4) **bass-anchoring** (roll rotation by the OBSERVED bass PC, not
the oracle root — removes the "we already assumed root" caveat of the family-LL model;
predict root as offset-from-bass) is a strong standalone model (94.7%) and its ensemble
with canon is the **new best (96.1%)** — the two rotation-anchoring strategies
(search-all-12 vs bass-observed) are complementary. (5) family-LL infers root well
(92.3%/86.1%) but always loses ~3-4pp to raw chroma. (6) the relative root-bigram
progression prior is **net-negative on oracle segments** (emissions already near
ceiling; over-smooths) — its real test is the per-beat regime, deferred to the per-beat
bake-off. Corollary: root-ID is ~96% GIVEN correct boundaries, so most real-pipeline
root error is now a SEGMENTATION problem (rule #3 / issue #11), not a root-model one.

### Per-beat bake-off + PROMOTED to production as beat_seq_model_v4 (2026-07-09)

`scripts/bakeoff_root_perbeat.py` (same clean disjoint jazz split, per-beat grid,
4900 eval beats, 28% boundary):

| model | overall | interior | boundary |
|---|---|---|---|
| canon ⊕ bass-anchored + viterbi | 93.5% | 94.3% | 91.6% |
| **canon ⊕ bass-anchored (shipped)** | **93.3%** | 93.3% | 93.2% |
| canon (key-agnostic) | 92.9% | 92.7% | 93.4% |
| bass-anchored (full chroma) | 90.5% | 90.3% | 91.0% |
| abs_ctx (LR ±4, key-biased) ≈ v2 | 86.7% | 86.9% | 86.2% |

The key-agnostic canonical scorer beats the v2-style LR by **+6.2pp per-beat** on a
leak-free split (86.7→92.9), +6.6 with bass-anchoring; the Viterbi progression prior
adds only +0.2 (it finally helps on noisy per-beat emissions, unlike oracle segments,
but not worth the decoding complexity — downstream segmentation already smooths).

**Promoted (`scripts/train_beat_seq_model_v4.py` → `harmonia/models/beat_seq_model_v4.npz`).**
canon MLP (±4, 96-hidden) ⊕ bass-anchored LR, trained on jazz1460 (70) + POP909 (60,
001-005 held out), **20 epochs** (40 overfit — 98.4% train, regressed POP909; process
rule #1). Held-out POP909 001-005 per-beat: **v2 79.4% → v4 80.4%** (interior tie,
**boundary 72.8→75.1**) — a clean gain on BOTH the canonical jazz corpus and pop, no
regression. Wired: `_BeatSeqModelV4` (self-contained numpy loader) in
`chord_pipeline_v1.py`; `_get_beat_seq()` now prefers v4 → v2 → v1. Full suite 223/223
green.

### End-to-end MIREX on held-out irealb + gmerge segmentation shipped (2026-07-09)

`scripts/eval_irealb_e2e.py`, 25 held-out jazz songs (index ≥70, unseen by v4),
real components (v4 root + baseline family classifier), mir_eval root/majmin vs chart:

| grid | segmentation | root | majmin | seg/GT |
|---|---|---|---|---|
| exact | oracle | 95.5% | 92.3% | 0.98 |
| exact | gmerge | 91.8% | 87.5% | 1.00 |
| exact | gridmerge (old production) | 77.1% | 72.9% | 0.67 |
| tempo (detected beats) | oracle | 91.7% | 88.3% | 0.98 |
| **tempo | gmerge (shipped)** | **88.7%** | **84.0%** | 1.12 |
| tempo | gridmerge (old production) | 70.8% | 66.1% | 0.73 |

The old production segmentation (`_merge_grid_by_root`: fixed 2/4-beat cells merged
by root) **under-segmented** (seg/GT 0.73 — merged across mid-cell ii-V changes):
end-to-end root only 70.8%. Replacing it with **gmerge** (`_root_change_segs`: cut
wherever the per-beat root argmax changes) lifts end-to-end root **+17.9pp → 88.7%**
(majmin +17.9 → 84.0), within ~3pp of the oracle-segmentation ceiling (91.7%). Beat
tracking itself costs only ~3pp (exact-grid oracle 95.5 → tempo 91.7). HCDF (frame-level
tonnetz novelty, `scripts/hcdf_boundary_probe.py`) was tried first and is DOMINATED by
grid methods on these metronomic corpora (pop 73.5 / jazz 87.5 vs gmerge 80.2 / 98.0) —
a live-audio tool, shelved for synthetic corpora. Wired: `infer_chords_v1` default
(non-bass) path now uses `_root_change_segs`; majmin here uses the baseline family clf
(ctx model would lift it). 223/223 tests green. Open: end-to-end on real YouTube audio
(where HCDF should finally pay off) and ctx-model majmin.

### Joint vs split chord prediction — split wins, DFT quality inferior (2026-07-09)

`scripts/bakeoff_chord_joint.py`, same oracle-segment vacuum (disjoint 35/35, 1495
eval chords), root/majmin/7ths:

| model | root | majmin | 7ths |
|---|---|---|---|
| **split** (canon root + root-conditioned quality LR) | **95.7%** | **92.3%** | **83.7%** |
| dft (canon root + \|DFT\| quality — the v3 head) | 95.7% | 80.2% | 65.8% |
| joint (one weight-tied head, root × quality) | 93.3% | 90.1% | 79.7% |

(1) **Joint HURTS** (−2.4pp root, −2.2 majmin, −4 7ths): a single 12×6 softmax lets
quality ambiguity leak into the root argmax. Keep root and quality as SEPARATE
canonical-frame classifiers (= what the pipeline already does). (2) **\|DFT\| quality
(v3 head) is materially worse** (−12 majmin, −18 7ths): magnitude discards the phase
that separates {0,4,7} from {0,3,7}; the production root-conditioned family clf is the
right choice. (3) **7ths headroom is SEGMENTATION, not the quality model**: 83.7% given
oracle segments vs 58.6% end-to-end (scorecard) — a ~25pp boundary/beat gap. Confirms
segmentation as the binding constraint for quality too.

### Pushed on segmentation — gmerge is near-optimal; the 7ths lever is the LABELER (2026-07-09)

`scripts/eval_seg_variants.py`, end-to-end tempo grid, v4 root + baseline family clf.
Premise: gmerge cuts only on root change, so it misses quality-only boundaries — but
those are only **6.0% (jazz) / 7.5% (pop)** of GT boundaries. Tested two fixes:

| variant | irealb root/mm/7th | POP909 root/mm/7th |
|---|---|---|
| oracle | 91.7 / 88.3 / 62.1 | 84.2 / 78.4 / 43.3 |
| **gmerge (current)** | **88.7 / 84.0 / 58.6** | **78.6 / 73.6 / 41.8** |
| gmerge_vit (Viterbi self-transition smooth) | 86.8 / 81.8 / 56.6 | 76.2 / 71.2 / 42.6 |
| gmerge_qual (root OR v3-quality change) | 88.7 / 84.0 / 58.5 | 78.6 / 72.0 / 42.0 |

**No variant beats plain gmerge** on either corpus. Viterbi-smoothing over-merges
(seg/GT 0.79-0.90 — deletes real 2-beat ii-V chords; the "1-beat flips" are often real);
quality-boundaries over-segment (seg/GT 1.11-1.34 — the noisy v3 quality signal adds more
false splits than the 6% real ones, cf. the bass-tracking lesson). **Segmentation is
near-solved by gmerge**: within 3pp (irealb) / 5.6pp (pop) of oracle, and the residual is
per-beat-root noise + beat tracking, not addressable by boundary heuristics.

**Redirect — the 7ths bottleneck is the LABELER, not segmentation.** Even with ORACLE
boundaries, 7ths is only 62.1% (irealb) / 43.3% (pop) using the baseline family clf's b7
model — vs **83.7%** for the canonical 6-class quality LR in the oracle-segment vacuum
(bakeoff_chord_joint). So the ~20pp of 7ths headroom is the seventh-labeler
(fam_clf.b7 → a root-conditioned canonical quality head), not boundaries. That is the
next lever, ahead of any further segmentation work.

### Full-pipeline scorecard (all improvements, end-to-end from audio, 2026-07-09)

| corpus | root | majmin | 7ths |
|---|---|---|---|
| irealb (held-out 25) | 88.7% | 84.0% | 58.6% |
| POP909 (001-005) | 78.6% | 73.6% | 41.8% |

vs Gen-2 v1 (2026-07-08) POP909 60.5/39.1 and Gen-1 ~33/29 → this session +18pp root /
+34pp majmin on POP909. ctx = baseline majmin here (quality-given-root already ~95%; root
is binding). Both corpora are synthetic; real live/YouTube audio still unmeasured.

---

## 16. ctx v2 model trained — integrate into chord_pipeline_v1 — DONE 2026-07-09

`scripts/train_ctx_model_v2.py` completed 3000 steps. Final best checkpoint (step ~2860):
- **fam_acc 87.7%, root_acc 87.6%, mirex_mm proxy 79.8%** (vs v1 family-only: ~39% majmin proxy)
- maj_acc 92.3%, min_acc 81.7% — no class bias
- Features: 684d = chroma(12) + ctx_ll(540) + root_intervals(108) + bsm_rel(12) + bsm_abs(12)
- Architecture: dual-head MLP (shared 256→128 trunk → family_head(5) + root_head(12))
- Loss: 0.6·CE_family + 0.4·CE_root; checkpoint saved to `harmonia/models/ctx_v2.npz` (838KB)

**Note:** root_acc 87.6% is on the iReal val set (oracle MIDI beat rolls). On real audio
(Basic Pitch → beat quantisation), root will degrade — the bsm_abs features are the load-bearing
addition over v1.

**Next:** update `chord_pipeline_v1._get_ctx_clf()` to load `ctx_v2.npz` and adapt
`_CtxFamilyClassifier` to handle the new 684d feature vector + dual-head weights. Then run
POP909 MIREX eval to get real (not proxy) end-to-end numbers.

---

## 19. Domain gap: models trained on MMA synth audio fail on real YouTube recordings — UNDER INVESTIGATION 2026-07-09

All current classifiers (`_FamilyClassifier`, `_BeatSeqModel`, `_CtxFamilyClassifierV2`) are trained on
synthetic MMA-rendered audio (accomp_db). Real YouTube recordings differ in:
- Timbre (live piano/ensemble vs. General MIDI synth)
- Recording quality, reverb, microphone characteristics
- Tempi and rubato (DTW alignment handles some of this)

**Investigation approach:** build a (YouTube audio, iReal Pro GT) paired corpus via DTW alignment
(`irealb_aligner.align_irealb_to_inferred`), then train a separate quality+root MLP on it.

**Corpus:** `harmonia/data/yt_chord_corpus.py` builds the corpus. Features:
- 48-dim root-shifted BP chroma (same pathway as `_FamilyClassifier.predict`)
- 12-dim root-shifted CQT chroma (librosa, bins_per_octave=36)
- Labels: 7-class quality (maj/min/dom/hdim/dim/aug/sus), 12-class root

**Results (2026-07-09):**

| Model | Quality val | Root val | Notes |
|-------|-------------|----------|-------|
| 10-song pilot, 7-class MLP, 1 val song | 26.5% | 63.3% | too noisy |
| 10-song LOSO, LR 60-dim | 49.4% | — | ceiling estimate |
| 10-song LOSO, RF 100 trees | 56.3% | — | non-linear helps |
| 50-song, 7-class MLP, 7 val songs | **53.0%** | 65.9% | meaningful |
| 50-song, 7-class MLP, context=1 | 53.6% | 63.7% | root overfit (train 97%) |
| 50-song, **3-class MLP** (maj/min/dom) | **62.0%** | — | +9pp from class simplification |

**Existing _FamilyClassifier (synth-trained) on same 7 val songs:**
- Strict: 41.0% (can't predict dom — merges to maj)
- Lenient (dom→maj credited): 60.3%
- maj=87%, min=46%, dom=0%, hdim=0%

**3-class yt model vs existing:** maj=63% vs 87%, min=63% vs 46%, dom=58% vs 0%.
Trade-off: better min/dom at the cost of maj. In jazz, dom/min distinction is critical → **3-class yt model preferred** for real audio.

**Feature analysis:** BP chroma m3/M3 ratio: 2.89 (min) vs 0.73 (dom) — mean separation is real.
Min/dom confusion is not a feature problem; it's a data-diversity problem.
Context windowing (±1 segment) hurts with 50 songs (LOSO 49.4%→43.3%) but may help at 200+ songs.

**Next:** 200-song corpus build in progress; 3-class model training on 50-song corpus.
After 200 songs: retrain 3-class, compare, then integrate best model into chord_pipeline_v1 as real-audio quality head.

---

**OPEN sub-question:** at what scale does context windowing start helping? Hypothesis: ≥200 songs.
The beat_seq_model_v4 uses ±4 beat window (88.3% root) — its success is context, not just scale.

---

### Mission 1 · Phase 1B (2026-07-13): chord-template↔chromagram DTW alignment — FAILED (full writeup: docs/mission_1_phase1b_results.md)

Phase 1's beat-grid alignment failed (600–1000 ms; tempo octaves + rubato). Proposed fix
was subsequence DTW of an iReal-GT chord template (non-circular: no model predictions) against
audio CQT chroma, absorbing rubato via local warping. **Built it fully; re-validated on the 3
pilots (Ghost/Foggy/Airegin) via chord-*change-point* correlation. FAIL: mean 1169/1504/1478 ms
vs a ±150 ms gate.**

Root cause is the **representation, not the algorithm** (cheap rigid-correlation premise checks,
independent of the DTW code): a synthetic chord template has almost no harmonic SNR against
**full-mix CQT chroma** of real jazz recordings. Raw cosine sits at a ~0.5 DC floor for *any*
alignment (percussion/reverb/melody/bass fill every bin); mean-centring (Pearson) is the biggest
single win but leaves a key-discrimination gap of only **≈0.02–0.04 cosine** — near noise. Tuning
is fine (~0 semitone). Ghost/Foggy are in the chart key (chroma-hist corr 0.85/0.74) and Ghost —
best key separation — aligns best (median 432 ms), confirming the SNR↔alignment link, still 3× over
gate. **`airegin.m4a` is transposed +2 semitones vs the F-minor chart** (corr 0.57 at +2 vs ~0.00 at
0) — a different-key recording, impossible to align in principle. Aggravator: the pilot files are
5–17 min full tracks (Foggy = 17 min) with verse intros + solo choruses, so subsequence-DTW's free
audio-skip mis-locks to late regions.

**Go/No-Go: No-Go** on frame-level template↔full-mix-chroma DTW for building the benchmark.
Recommended fallback: **manual downbeat anchors + piecewise-linear** (~15 min/song, ±100–200 ms,
no SNR dependence) to unblock the 20-song benchmark; keep chord-recognition-**posterior**-DTW or
beat-synchronous-chroma DTW as later automation. And curate inputs first — verify each audio's key
matches its chart (the Airegin +2 check) and trim to the head.

### Mission 4 (2026-07-13): real-audio calibration + domain-gap re-measurement on the NEW pipeline

**Inventory of real-audio GT actually on disk (checked before building anything):**
- `docs/audio/*.m4a` = 3 raw recordings (autumn_leaves, nina_simone_feeling_good, beatles_let_it_be). soundfile can't read m4a → convert via ffmpeg.
- `data/cache/yt_corpus/audio/` = 1 wav (Cm0O4IhLcPY, 882s) — NOT a corpus song, no GT (a full mix).
- `data/cache/yt_corpus/corpus_50.npz` = **7195 real-audio chord segments, 50 songs, iReal GT** (root+quality), with cached 48-dim features + t0/t1 + a `match` field. The `match` field = the production pipeline's per-chord agreement with iReal GT at build time (exact=root+family right, family=root right family wrong, mismatch=root wrong).
- **Trap found (rule #1):** the one time-aligned chart, `docs/plots/irealb_autumn_leaves.html` (`window.P.chords`), spans only 0–160s but the current autumn_leaves.m4a is **422s** — the GT is **orphaned** from a since-deleted shorter download. Matching the new pipeline to it gives 7.8% root acc (≈ chance, flat offset dist) and a sliding-window premise-check bottoms at DTW cost 0.60. **Do NOT report that as the domain gap** — it is a stale-GT artifact. Network fetch of fresh charts is 403-blocked, so no new time-aligned pairs could be built. Conclusion: there is **no usable (audio, time-aligned GT) pair on disk**; all real-audio measurement below routes through the corpus_50 segments (raw BP activations aren't cached — `bp_cache/` empty — so the full segmenter can't be re-run; only the labeling+confidence layer is reproducible from cached feat48, which is fine because that is exactly where the calibrator lives).

**A. Domain-gap accuracy (production pipeline, n=7195, iReal GT, `match` field).** Lower bounds — DTW misalignments inflate "mismatch".

| GT family | n | exact | family | mismatch | root acc (ex+fam) |
|-----------|----|-------|--------|----------|-------------------|
| maj | 1746 | 0.53 | 0.11 | 0.35 | 0.65 |
| min | 2324 | 0.31 | 0.30 | 0.39 | 0.61 |
| dom | 2566 | 0.21 | 0.34 | 0.44 | 0.55 |
| hdim | 203 | 0.16 | 0.39 | 0.45 | 0.55 |
| dim | 163 | 0.11 | 0.30 | 0.59 | 0.41 |
| **all** | 7195 | **0.32** | 0.27 | 0.41 | **0.59** |

**dom7 is NOT 0% (the #19 headline is stale)** — it's 21% exact / 55% fam-or-better. The residual is dom→maj family confusion, i.e. the b7. Chroma diagnostic (root-aligned peak-normed CQT): dom b7=0.49 vs M7=0.41; maj b7=0.35 — the b7 IS present but the dom-vs-maj contrast is only ~0.14 and every degree floors at ~0.35–0.65 (reverb/overtone/vocal smear). **Covariate shift, not masked information.**

**B. Calibration on real audio (quality head, n=7002, 5-way q5).** q5 acc = 43.9% but displayed `confidence_raw` mean = 0.90 → **overconfidence +0.47, ECE 0.465**. Reliability is near-**flat** (conf 0.98 → 48% correct), so k-selection of the lowest-confidence chords is near-random on real audio. Passing raw through the **synth** isotonic map makes it worse (ECE 0.533). **Fix shipped:** `scripts/fit_confidence_calibration_real.py` → `data/cache/confidence_calibration_real.npz` (isotonic on the 7002 real segments); `infer_chords_v1(audio_domain="synth"|"real")`, default **"real"** for the server path, selects the map (`_get_conf_calibrator`). Real ECE **0.465 → 0.007** (5-fold song-held-out CV); collapses displayed conf toward base rate ~0.44 and caps at the reliability ceiling ~0.45. Synth path untouched (ECE 0.037). **Caveat:** the real map is fit on a PROXY score (baseline-LR `_FamilyClassifier` on cached feat48), not production ctx/joint `confidence_raw`, and root_conf isn't folded in; it transfers safely only because it is nearly flat. Refit on production confidence_raw once real audio+GT is re-obtainable.

**C. Merge / evidence-pooling √N test on real audio (issue #28).** Pooling feat48 across repeats of the same chord within a song then classifying once: q5 acc **43.8 → 53.8% (+10.0pp)**, growing with repetition count (reps≥5: +9.8pp). First real-audio confirmation of Mission 3's pooled-emission denoising — pool BP evidence, don't vote on labels.

**D. Highest-leverage training intervention (ranked).** (1) **Retrain the q5 quality head on real-audio chroma** — the corpus already has 7002 labeled real segments; the gap is emission covariate shift (flat/smeared chroma), and a synth-trained head is the wrong prior. Expected biggest single win on dom/min. (2) Contrast-enhancing features (HPSS before chroma, per-chord chroma whitening) to widen the b7/M7 and m3/M3 gaps — feature-side, complementary. (3) Section-merge pooling in the labeler (from C) — cheap, +10pp where repeats exist. Do **not** invest in "recovering the b7" — it is present.

**Literature (10-min scan):** Calibration under distribution shift (Ovadia et al., NeurIPS 2019) — temperature/isotonic maps fit on in-domain data do **not** transfer to shifted domains and can worsen ECE; our synth→real 0.037→0.465→(0.533 after synth map) is a textbook instance, motivating a domain-selected map. crema (McFee) / madmom chord models are trained on real annotated audio (Isophonics/Billboard/RWC) with CQT front-ends — the recipe worth copying for the future 200-song corpus is real-annotated-audio + CQT, which is the direction of the yt_corpus build.

---

## 20. Chord quality inference ignores diatonic scale prior per section — key-relative ctx FEATURE + two-pass wiring: REAL prod win (jazz1460 majmin +2.7pp, minor-family +9pp), opt-in 2026-07-13

> **Update (2026-07-13) — TWO deliverables (see docs/nightly_runs.md 2026-07-13):**
>
> **(a) `LocalKeySeqGRU` wired as a diatonic-prior reranker** (`use_local_key_prior`,
> default OFF; commit 80c17fc). This finally replaces the noisy single-window `infer_key`
> the 2026-07-12 note called out as "the next lever." It is the more-reliable key source,
> but on **held-out jazz1460 it is still net-negative** (majmin 84.0→83.0% at boost 4,
> monotone with boost; root unchanged): jazz is only ~49% diatonic, so snapping genuine
> secondary dominants to diatonic is *wrong*. Default stays OFF — the payoff corpus is
> diatonic pop/standards, not jazz (POP909 clip verified to make clean corrections, e.g.
> `D#:7→D#:min7`). **A section-local *snapping* prior is the wrong shape for jazz, full stop.**
>
> **(b) The better use of the same information is a key-relative INPUT FEATURE on the
> acoustic classifier, not a post-hoc snap** (volet 2, commit 736b57c). Adding a 117-d
> "scale-degree-vs-local-key" block (key-agnostic, transpose-invariant) to `_CtxFamilyClassifierV2`
> lifts held-out VAL **family accuracy +4.3pp (0.845→0.888), minor-family +7.6pp** at equal
> training budget — the minor-family gain being exactly the Georgia/"Let It Be" `A major`
> vs `Am` error. Raw heuristic (v2) beats consolidated (v3) here (v3 erases the local
> functional signal). This is a **bootstrap upper bound** (GT context quality at train
> time); the realizable prod gain needs a two-pass retrain+wiring, blocked by disk <10 GB
> this session. This is now the primary #20 lever.

> **Resolution — TWO-PASS PROD, REAL GAIN (2026-07-13, later session):** the two-pass
> wiring + full-budget retrain are done, and the bootstrap's minor-family win **survives
> the passage to noisy predicted context**. `ctx_v3.npz` = full-budget retrain (3000
> steps, `--local-key v2`, 801d). `infer_chords_v1(ctx_classifier_variant="801d_two_pass")`
> runs the non-circular scheme: pass-1 684d ctx_v2 over the whole song → the raw-v2
> continuity teacher reads a local key per chord off that *predicted* sequence → pass-2
> 801d model re-scores each segment with the 117d key-relative block, its refined q5
> log-probs feeding the shared #21 progression reranker (the ctx family's realization
> path). End-to-end on **held-out jazz1460 (25 songs, idx 70–95, MuseScore render, prod
> defaults, `scripts/eval_two_pass_801d.py`):**
>
> | variant | root | majmin | 7ths | maj | min | dom | hdim | dim |
> |---|---|---|---|---|---|---|---|---|
> | 684d (current prod) | 88.7% | 80.4% | 56.7% | 90% | 73% | 78% | 47% | 50% |
> | **801d two-pass** | 88.7% | **83.1%** | **57.7%** | 88% | **82%** | 79% | **68%** | 56% |
> | Δ | 0 | **+2.7pp** | +1.0pp | −2pp | **+9pp** | +1pp | **+21pp** | +6pp |
>
> (GT-chord support/family: maj 219, min 341, dom 534, hdim 38, dim 16.) The
> **minor-family +9pp** (73→82%) is the Georgia/"Let It Be" `A major`-where-`Am`
> confusion — and it is *larger* than the bootstrap's GT-context +7.6pp, not smaller, so
> the two-pass noise did NOT eat the gain (the teacher reads the local key robustly even
> off imperfect pass-1 quality; on jazz's ~49% diatonicism the vi/ii-of-minor signal is
> especially informative). hdim +21pp is a bonus (the m7b5 ii-of-minor is now placed by
> key). The −2pp maj is the expected small trade (a few maj→min flips); net majmin is
> clearly positive. Root is unchanged by construction (root comes from beat_seq_v4, not
> the ctx head). **Default stays `684d`** (opt-in `ctx_classifier_variant="801d_two_pass"`)
> pending a POP909 confirm + a UX check, but this is the first *realizable* #20 win.
> Ref case `C-G-Am-F` (`scripts/ref_test_let_it_be.py`): on a deliberately sparse 2-voice
> synthetic render (degraded root detection) the 801d still flips the first `A:7→A:min`
> vs 684d — directional, the aggregate jazz1460 number is the real evidence.
> Volet A wiring: commit two-pass (this session); tests `tests/test_two_pass_ctx_inference.py`.

> **Resolution (2026-07-12, `apply_diatonic_prior` in `chord_pipeline_v1.py`):** the
> section-local, confidence-gated diatonic prior is implemented and unit-tested
> (`tests/test_diatonic_prior.py`, 9 tests) but ships **default OFF**
> (`use_diatonic_prior=False`).  Rationale: the GT premise is real (POP909 = 93.3%
> diatonic in local key) but the *inferred* local key (`infer_key` over a ±4-bar
> window) is not accurate enough to exploit it.  End-to-end MIREX
> (`scripts/eval_diatonic_prior.py`, tempo grid + gmerge, 25 held-out jazz1460 +
> 5 POP909):
>
> | corpus | variant | root | majmin | 7ths |
> |---|---|---|---|---|
> | jazz1460 | baseline | 88.7% | 84.0% | 58.6% |
> | jazz1460 | +prior (thr 0.65) | 88.7% | 83.2% | 58.1% |
> | POP909 | baseline | 78.6% | 73.6% | 41.8% |
> | POP909 | +prior (thr 0.65) | 78.6% | 73.0% | 41.6% |
>
> Fire-outcome tally on POP909 is a coin-flip (default thr 0.65: 3 wrong→correct
> vs 3 correct→wrong; thr 0.90: 13 vs 8).  A (boost, thr) sweep peaks at
> boost 4.0 / thr 0.80 → POP909 majmin **+0.1pp** (within noise, n=5) — never a
> credible gain.  The mechanism is correct (on a real F-major clip it flips
> I:min→maj, IV:7→maj7, vi:7→min7 as expected); the bottleneck is local-key
> inference, not the prior.  **Next lever:** wire the unused
> `harmonia/theory/local_key.py` (HMM local key) in place of the single-window
> `infer_key`, re-validate local-key accuracy, then re-sweep.

> **Premise check (2026-07-12, `scripts/check_diatonic_premise.py`):** on held-out jazz1460
> (idx 70–95, 1128 GT chords) only **49.4%** of chords are diatonic in the song key (52.4% even
> using the *trusted* iReal key annotation) — well below the 60% gate. Cause is genuine jazz
> chromaticism/modality (secondary dominants, tritone subs, dom7/blues tonics), not a bug. A
> **strict, global-key** diatonic prior is therefore the wrong tool here and was **not implemented**.
> If revisited: needs a *soft, section-local, confidence-gated* weight with Mixolydian/blues-tonic
> tolerance, and the local-key premise must be re-validated first. May still pay off on POP909
> (higher diatonicism) — untested. See `docs/nightly_runs.md` 2026-07-12 entry.

**Observed on:** "Georgia On My Mind" (live iPhone test, `chord_pipeline_v1`). Root detection is often correct;
chord *family* (maj/min/dom/…) is frequently wrong despite the harmonic context making it nearly unambiguous.

**Root cause hypothesis:** The current family classifier (`_FamilyClassifier` / `_CtxFamilyClassifierV2`) is a
pure acoustic model — it does not condition on the *key* of the current section. In a diatonic context almost
every chord family is determined by the root's scale degree:

| scale degree (major key) | diatonic quality |
|---|---|
| I | maj (or maj7) |
| II | min7 |
| III | min7 |
| IV | maj (or maj7) |
| V | dom7 |
| VI | min7 |
| VII | hdim7 (m7b5) |

The model should use this as a strong prior and **only override it toward a non-diatonic quality when the
acoustic evidence is confident and coherent across the section** (e.g. a #IV°7 substitution, a secondary
dominant). Currently the family head treats every beat independently with no diatonic bias at all —
so acoustic noise (e.g. a passing tone, reverb) routinely flips a min7 chord to maj or dom.

**Proposed fix (nuclear subtask):**

1. Within each coarse segment, compute the section key via `infer_key()` on the segment's chroma (already
   available from `_BeatSeqModelV4`'s canonical rotation).
2. Derive the **diatonic quality prior** for each candidate root: `prior[quality] = high` if `(root, quality)`
   is diatonic in the section key, `low` otherwise. A simple log-weight (e.g. `+log(5)` for diatonic,
   `0` for chromatic) is enough to start.
3. Combine: `log_posterior = log_likelihood_acoustic + diatonic_log_prior`.
4. Override only when `max acoustic confidence > threshold_chromatic` (to be tuned) — this prevents the prior
   from suppressing a real secondary dominant (V/V etc.) the acoustic model sees clearly.

**Key diagnostic to run first (cheapest premise check):**

```python
# On Georgia / 5 POP909 songs: for each GT chord, what fraction is diatonic in GT key?
# → sets the ceiling gain from a perfect diatonic prior
```

If ≥80% of GT chords are diatonic in their section key, the prior can correct a large fraction of
the observed family errors at near-zero cost. This is a cheap O(1) check before any implementation.

**What this does NOT solve:** chromatic/borrowed chords (secondary dominants, modal interchange,
tritone subs) will require higher acoustic confidence to escape the prior — may under-predict these
in jazz contexts where chromatic movement is common. Tune the threshold on jazz1460 to avoid
over-suppressing secondary dominants.

**Owning agent:** dedicated Opus agent — see nightly_agent_runbook.md §Multi-agent strategy.

---

## 21. Structural chord progression priors not exploited — bigram/trigram coherence model — OPEN (bigram premise MARGINAL) 2026-07-12

**Update 2026-07-12 (real per-q5 acoustic prior for the encoder reranker):** The
`ProgressionEncoder` reranker (commit `f7ecd3c`) combined `log_acoustic + w·log_encoder`
where `log_acoustic` was a **confidence-gated one-hot** on the greedy q5 class — a
diagnostic run showed this pinned the acoustic prior near-degenerate whenever
`conf > ~0.65`, leaving the encoder no real evidence to argue against (root cause of
the standalone-cloze (83.9%) vs end-to-end (+0.7pp majmin) gap). Fix: the acoustic
classifier already computes two heads with real posteriors — the 5-class family
distribution (`p_fam`, major/minor/dim/aug/sus — from `_FamilyClassifier`/the
entropy-gated ctx blend `p_mix`) and the base7 seventh distribution (`p7`, from
`b7_clf`) — that were previously collapsed to a single scalar `conf`. New
`_family_q5_logprobs()` (`harmonia/models/chord_pipeline_v1.py`) combines both into a
real 5-class q5 (maj/min/dom/hdim/dim) log-probability vector: minor/augmented/
suspended map 1:1 onto q5 (min/maj/maj respectively, no split needed); major splits
into maj-vs-dom and diminished splits into dim-vs-hdim using each branch's b7
posterior mass, renormalized within that branch. `predict(..., return_q5proba=True)`
on `_FamilyClassifier`/`_CtxFamilyClassifier`/`_CtxFamilyClassifierV2` exposes this;
`rerank_progression_qualities(..., aco_logprobs=...)` uses it directly when given,
falling back to the old one-hot when `None` (back-compat preserved — existing
external call sites in `scripts/eval_diatonic_prior.py`, `eval_seg_variants.py`,
`diag_diatonic_prior_pop.py` are unaffected).

irealb held-out e2e sweep (`scripts/eval_irealb_e2e.py`, tempo/gmerge config, n=25):

  | variant | root | majmin | 7ths |
  |---|---|---|---|
  | baseline (no encoder) | 88.7% | 84.0% | 58.6% |
  | encoder + one-hot (old, w=0.5) | 88.7% | 84.7% | 58.9% |
  | encoder + real q5 logprobs (w=0.2) | 88.7% | 84.8% | 59.0% |
  | encoder + real q5 logprobs (w=0.5) | 88.7% | 84.8% | 59.0% |
  | encoder + real q5 logprobs (w=1.0) | 88.7% | 84.7% | 58.9% |
  | encoder + real q5 logprobs (w=2.0) | 88.7% | **85.0%** | **59.0%** |

  **+1.0pp majmin over baseline, +0.3pp over the old one-hot prior at its best
  weight — real but modest** (did not clear the ≥85.5% "true improvement" bar set
  going in). `progression_weight` default bumped 0.5→2.0 to match the new prior's
  calibration. Root is unaffected (the encoder only reranks quality, never root).
  Not yet re-run on POP909; the irealb/jazz1460 numbers are the trusted eval per
  the canonical-GT provenance note. See `docs/nightly_runs.md` for the run log.

---

**Update 2026-07-12 (premise check, nightly agent):** Subtask #1 (the pre-registered
premise gate) was run via `scripts/check_bigram_premise.py` on the iReal corpus
(`data/accomp_db/db.jsonl`, 1458 jazz / 1856 total songs). Transpose-invariant bigram
`(interval=(root_j−root_i) mod 12, quality_i, quality_j)`:

  | corpus | top-50 coverage | top-20 | info gain H(next)−H(next\|q_prev) |
  |---|---|---|---|
  | jazz1460 (target) | **63.8%** | 50.3% | 0.90 bits (17% of 5.25) |
  | pop400 | 62.9% | 43.3% | — |
  | blues50 | 98.3% | 86.2% | — |
  | gate to implement | **≥ 70%** | — | — |

**Verdict: MARGINAL — 63.8% < 70% gate → a plain global bigram prior was NOT built**
(CLAUDE.md rule #2 — don't move the goalpost after seeing data). The signal is real but
weak: the previous chord removes only 17% of next-chord uncertainty. Top bigrams are
textbook (`min7 →+5 dom7 →+5 maj7` = ii-V-I; `m7b5 →+5 dom7alt` = minor ii-V), which is
exactly the point — **the dominant pattern is a *trigram* (ii-V-I); a bigram splits it into
`ii→V` and `V→I` and cannot enforce the full cadence.** blues50 (formulaic 12-bar) scores
98.3%, confirming the metric behaves and jazz genuinely has a long tail (1314 unique
transpose-invariant bigrams). **Recommended next (ranked): (1) re-run the premise on
transpose-invariant *trigrams*; (2) condition the prior on the detected section / local
key (issue #22's `ChordChart.sections`) instead of a global matrix; (3) the reranking
transformer in the spec below.** See `docs/nightly_runs.md` 2026-07-12 entry.

---


**Observed on:** "Georgia On My Mind" (live iPhone test). Even when individual chords are plausible, the
sequence of chords per sub-section (A or B phrase) is incoherent — the model jumps to rare or unexpected
chords that no musician would play in that position, when a much more likely progression would explain the
same acoustic evidence.

**Root cause:** Current pipeline is a per-beat greedy argmax (v4 root model) followed by per-segment family
classification. There is **no progression model** — no learned sense that "ii-V-I in Bb" is 100× more
likely than a random sequence of chords at the same positions. The chord stream is essentially IID given
the audio, with no structural self-consistency constraint.

**What the literature says (to be verified by the agent):**

- **Markov/HMM chord transition models** (Raphael & Stoddard 2004, Papadopoulos & Peeters 2007, Klapuri et
  al.) — standard first approach; transitions learned from corpora. Already partially implemented in this
  project via `viterbi_duration_aware` (issue #1) but with a bad emission model. The real question is
  whether a *data-driven* bigram matrix from iReal Pro (2229 songs) is strong enough to act as a useful
  prior on top of the v4 per-beat output.
- **Neural chord sequence models** (McLeod & Steedman 2021, Chen & Su 2019, Kosta et al. 2022) — treat
  chord prediction as a seq2seq task; transformers / LSTMs over chord labels or multi-hot representations.
- **Functional harmony bigrams** (jazz-specific): ii-V, V-I, I-IV, I-VI-ii-V turnarounds, tritone subs.
  A vocabulary of ~20 functional bigrams covers the majority of jazz progressions.
- **ATIS / grammar-based approaches** (Rohrmeier 2011, Martin 2018) — formal harmonic grammar generating
  chord sequences; more expensive to train but explicit and interpretable.

**Proposed model architecture (user spec):**

Input to a coherence/validation model:
- Last 4–8 "nuclear bigrams" (the most likely pair at each step from the progression decoder) + their
  confidence scores
- Next 4–8 predicted bigrams + confidences (look-ahead from the per-beat model)
- Current per-chord acoustic confidence

Output: corrected chord for the current position, or a probability distribution over corrections.

This is a **sequence-to-sequence reranking** model — it sees the local context window and adjusts the
greedy per-beat prediction to be consistent with the typical chord grammar of the key/section. Attention
(a small Transformer encoder over the bigram context) is well-suited here: each bigram position can attend
to all others in the window, so the model can detect "this ii-V is structurally consistent with what came
before/after" without a fixed-order Markov assumption.

**Nuclear subtask for the agent:**

1. **Premise check first (CLAUDE.md rule #2):** compute corpus-level bigram coverage on jazz1460 —
   what fraction of consecutive chord pairs are in the top-50 most common bigrams (transpose-invariant)?
   If ≥70%, a bigram prior is worth implementing; if <50%, skip to trigrams or a full sequence model.
2. Build a **transpose-invariant bigram matrix** from the iReal Pro 2229-song corpus (the same corpus
   used for `train_online.py`) using root-relative intervals. Store as `data/cache/chord_bigrams.npz`.
3. Wire as a Viterbi log-prior on top of `_BeatSeqModelV4`'s per-beat root distribution (don't replace,
   *combine* — cf. canon⊕bass-anchored ensemble result in issue #18 bake-off).
4. Evaluate on jazz1460 held-out 25 songs + Georgia on my mind (manual listen check).

**Can attention help here?** Yes — specifically for the *look-ahead* part of the user spec. A causal
Transformer sees past bigrams cleanly; looking ahead at "what the per-beat model predicts for the next
8 bars" and using that to refine the current chord requires bidirectional or encoder-style attention
(non-causal). Small model (≤4 heads, ≤2 layers, sequence length ≤16 bigrams) is likely sufficient and
fast to train on the iReal corpus.

**What this does NOT solve:** issues #20 (diatonic prior) and #22 (section structure) are prerequisites
or complements — a bigram prior on a poorly-segmented score is noise. Attack #20 and #22 first.

**Owning agent:** dedicated Opus agent — see nightly_agent_runbook.md §Multi-agent strategy.

---

## 22. Global section structure (AABA/AABA') inference is poor — PARTIALLY RESOLVED 2026-07-12

**Update 2026-07-12 (chord-SSM section detector, nightly agent):** Boundary detection now
implemented in `harmonia/models/section_structure.py` and wired into `chord_pipeline_v1`
(new `ChordChart.sections` field). Symbolic chord-SSM + jazz form-length prior recovers
8/16-bar section boundaries. Held-out jazz1460 boundary-F (GT section markers, ±1 bar):

  | variant | boundary-F | prec | rec | eval set |
  |---|---|---|---|---|
  | gmerge baseline (chord cuts) | 0.097 | 0.055 | 0.992 | 301 AABA tunes, GT chords |
  | chord-SSM sections (ceiling) | **0.986** | 0.987 | 0.987 | 301 AABA tunes, GT chords |
  | chord-SSM sections (end-to-end) | **0.844** | 0.889 | 0.833 | 12 AABA tunes, inferred chords |

  Premise check (`scripts/premise_check_chord_ssm.py`) confirmed the mechanism before
  implementing (CLAUDE.md rule #2): on the symbolic chord-SSM the bridge B is correctly less
  similar to A than the two A's are to each other (chord-SSM beats acoustic-SSM 7/8 tunes);
  the *acoustic* SSM carries ~0 section signal (bridge-contrast ±0.003) — the crux of this
  issue. Checkerboard novelty is a poor detector for both (3/8); the working detector is
  repetition + form-length prior, NOT novelty. Diagnostic: `docs/plots/section_ssm_aaba.png`.

  **Still open:** (a) section *labelling* (which span is A vs B) is not done — only lengths /
  boundaries; (b) through-composed / all-sections-similar tunes (e.g. "Dat Dere") and tunes
  where the two A8 phrases differ enough to over-split still miss (the 15% wrong-signed tail
  from the 371-tune bridge-contrast survey); (c) ~~section *phase* (pickup/intro offset) is
  assumed 0~~ **FIXED 2026-07-12, see below**; (d) not yet wired into the interactive chart
  renderer / not evaluated on POP909 or YouTube audio (the "Georgia On My Mind" origin case).

  **Update 2026-07-12 (section-phase correction — cycle-shift bug, e.g. Let It Be):**
  Concrete instance found on real audio (Let It Be, live iPhone test): the C-G-Am-F loop
  (I-V-vi-IV in C) was phase-shifted so the tonic C, which opens each 4-bar cycle, landed
  *last*. The period length (4 bars) was detected correctly; only the phase was wrong. The
  earlier phase fix — `periodicity.find_loop_phase()` (`## score_periods()` entry, "FIXED
  2026-07-04") — anchors phase on `is_downbeat`, which is POP909 **ground truth**
  (`beat_midi.txt` col 3) and does not exist on YouTube/real audio, so the bug was intact for
  the production task. New fix recovers phase from **harmonic-progression likelihood** instead:
  `correct_section_phase()` in `section_structure.py` scores each of the P candidate phases of
  a P-bar loop by the summed per-period log-likelihood under a key-relative bigram LM
  (`build_progression_model()` / cached `data/cache/chord_bigrams.npz`) and picks the max. The
  LM's BOS (start) distribution — built from corpus song-openers, peaked on the tonic I
  (`(root_rel 0, maj)` logp −0.72, next `(0, min)` −1.89) — is what breaks the cyclic symmetry
  a bare transpose-invariant bigram table cannot (rotating a loop preserves every consecutive
  pair; transitions alone are phase-blind). Wired into `infer_chords_v1` as
  `use_phase_correction=True` (opt-in, ON by default); it shifts the **section grid only**, so
  the chord labels/times are byte-identical with it on vs off (verified end-to-end) → zero
  regression risk to root/majmin/7ths metrics. `scripts/eval_irealb_e2e.py` reconstructs the
  pipeline stage-by-stage and never computes sections, so it does not exercise (or regress on)
  this path. **NOT solved here:** Option A (the issue-#21 `ProgressionEncoder`) is unusable for
  phase — it is a *quality* cloze model (root-relative, transpose-invariant) whose summed
  log-probs are invariant to a whole-loop rotation, so it carries no phase signal; a bigram LM
  with a key-relative BOS distribution is the smallest model that does. Tests:
  `tests/test_phase_correction.py` (synthetic Let It Be loop, all 3 wrong-start phases
  recovered).

  **Update 2026-07-12 (bigram → trigram phase-scoring LM):** the bigram above contradicted
  #21's own conclusion — the #21 premise-check found bigrams MARGINAL on jazz (63.8% cloze
  < 70% gate) *precisely because* the resolving ii–V–I is a **trigram** pattern, which is what
  motivated the ±6-context `ProgressionEncoder`. `build_progression_model()` now also builds a
  key-relative **trigram** table `tri` (5,12,5,12,5) of `(q_prev2, Δ21, q_prev1) → (Δn, q_next)`
  (all root-deltas relative to the middle chord → transpose-invariant like the bigram), plus
  context counts `ctx`. `_next_logprob()` interpolates trigram-MLE with the bigram via a
  Witten-Bell-style weight `λ = c/(c+5)` (`c` = trigram context count) → unseen contexts back
  off cleanly to the (smoothed) bigram. The tonic-peaked BOS prior is **unchanged** — it, not
  the model order, breaks the rotation symmetry. Cache moved to
  `data/cache/chord_progression_model.npz` (old bigram-only `.npz` is auto-rebuilt); public
  API (`load_progression_model`, `correct_section_phase`) unchanged, so `chord_pipeline_v1`
  needs no edit. **Demonstrated gain (not a null result):** two synthetic jazz loops with an
  embedded ii–V–I —
  `Dm7 G7 | Cmaj7 Em7 | Am7 D7 | Dm7 G7` (C major) and
  `Am F | Bm7b5 E7 | Am Dm | Bm7b5 E7` (A minor) — where the **bigram picks the wrong phase in
  a near-tie** (C-major: correct phase 0.03 nats behind the bigram argmax; A-minor: an exact
  tie) because its pair statistics are nearly rotation-symmetric, while the **trigram recovers
  the tonic-opening phase decisively** (+2.8 / +0.96 nats) by scoring the intact ii–V–I triple.
  New tests in `tests/test_phase_correction.py` pin this *delta* (trigram corrects a phase the
  bigram-only model gets wrong). Non-regression: `shift` feeds only `apply_phase_shift` →
  section grid, never chord labels/times (chord_pipeline_v1.py:1769-1771), so chords stay
  byte-identical; full suite 271 passed. Cost: 18k-cell dense trigram tensor (75 KB npz),
  negligible.

**Observed on:** "Georgia On My Mind" (live iPhone test). The A and B sections (and their
sub-phrases) are not correctly identified; the chord chart does not reflect the AABA form and
its repeats.

**Root cause:** Section segmentation in `chord_pipeline_v1` uses a cosine-novelty boundary detector
(gmerge) calibrated at the chord level. This is good for detecting *chord changes* (≤2-beat boundaries)
but poor at detecting *section boundaries* (8–16-bar boundaries) because:
- Cosine novelty detects local contrast, not long-range repetition (Foote checkerboard — see issue #9).
- AABA structure lives in *repetition* (bar i ≈ bar i+16/32), not local novelty.
- Jazz harmonic rhythm (ii-V-I every 2 bars) produces stronger local novelty than section boundaries.

**What the literature says (to be verified by the agent):**

- **MSAF (Music Structure Analysis Framework)** (Nieto & Bello 2014/2016) — standard benchmark suite;
  multiple structure analysis algorithms compared on RWC/SALAMI datasets. SSM-based methods dominate.
- **Repetition-based methods:** SF (Structure Features, Serra et al. 2014), Foote (2000), NMF-based
  decomposition (Nieto & Jehan 2013), C-NMF (Nieto & Bello 2015). The SSM diagonal approach
  (`score_periods()` in `harmonia/models/periodicity.py`) is an instance of this family, but was
  found to detect *accompaniment* repetition not *harmonic* repetition (issue #1/C).
- **Symbolic methods** (for our synthetic/quantized domain): chord-level SSM (treat the chord sequence
  as a "symbolic audio" and compute self-similarity directly on chord labels or chroma) — likely
  better than audio SSM for our use case.
- **Learning-based:** SALAMI/RWC-trained boundary detectors (Ullrich et al. 2014 CNN; McFee & Ellis
  2014 spectral clustering). These need real-audio or large-scale real training data, may not
  transfer from MMA synth.
- **Key insight for jazz standards:** AABA structure is *highly regular* — 32-bar AABA where A=8
  bars, B=8 bars, with occasional 16-bar or 64-bar variants. A prior over standard jazz form lengths
  (8, 16, 32, 64 bars) is a strong structural constraint the pipeline currently does not exploit.

**Nuclear subtask for the agent:**

1. **Literature survey** (3–5 most relevant papers) + assess which approach is feasible given our
   corpus (synthetic metronomic MMA, no real-audio structure labels).
2. **Chord-level SSM** — build the SSM on the inferred chord sequence (root×quality, transposition-
   invariant representation) rather than audio chroma. Score the diagonal as in `score_periods()`.
   On a metronomic corpus, the chord SSM should be much cleaner than the audio SSM for detecting
   section repetition.
3. **Form-length prior** — given the dominant-period detection, snap to the nearest standard jazz
   form (8/16/32/64 bars) and use that as a prior for how many sections to expect and where
   section boundaries land.
4. **Evaluate:** on jazz1460 (where the iReal form is the GT), compute ARI / boundary F between
   detected sections and iReal section markers. Compare chord-SSM vs audio-SSM vs current gmerge.

**What this does NOT solve:** within-section chord inference (#20, #21) — fix section boundaries
first, then the bigram/diatonic priors operate on cleaner sub-sequences. Recommended attack order:
#20 (diatonic prior, cheapest) → #22 (section structure) → #21 (bigram/attention model).

**Owning agent:** dedicated Opus agent — see nightly_agent_runbook.md §Multi-agent strategy.

---

## 23. Learned section-local key model — wired to prod (reranker + ctx feature) 2026-07-13

> **Update (2026-07-13):** the per-chord `LocalKeySeqGRU` (distilled from the rule-based
> heuristic, v3 dominant-chain consolidation, `data/cache/local_key_seq_gru.pt`) reached
> production in two forms, both transpose-equivariant end-to-end:
> - **as a reranker** in `infer_chords_v1` (`use_local_key_prior`, commit 80c17fc) — works
>   mechanically but net-negative on jazz (see #20a); default OFF.
> - **as the teacher for a key-relative ctx-classifier feature** (commit 736b57c) — the
>   strong result: family +4.3pp / minor +7.6pp on the bootstrap (#20b). Notably, the
>   **raw per-chord heuristic labels beat the v3-consolidated ones for this feature** — the
>   consolidation that de-zigzags the *key track* erases the local functional cue the
>   *family* classifier needs (a secondary dominant read as chromatic vs "acting as a V").
>   So the #23 consolidation is right for key-labeling, wrong for the family feature.
> Remaining: two-pass prod inference + full-budget retrain (blocked by disk <10 GB). See
> docs/nightly_runs.md 2026-07-13.

Follow-up to #20's "next lever" (wire a real local-key model in place of the noisy
single-window `infer_key`). Instead of patching the existing inference, we built a
**learned** section-key predictor with a symbolic dataset from iRealb. Phase-1
(symbolic-only, no audio) is complete; this entry logs the premise-check + first model.

**New code:**
- `harmonia/models/local_key_data.py` — rules-based oracle section-key labeler +
  dataset builder. One example per *section instance* (contiguous same-label bar run):
  `(chord_seq: list[(root_pc, q5)], oracle_key_idx ∈ 0..23)`. Oracle = duration-weighted
  chord-tone chroma matched to the 24 Krumhansl profiles (`key_profiles.infer_key`) with
  a **margin gate against the song global key** — a section is only marked *modulated*
  when its chords beat the global key by ≥ `margin` nats ("hold home key until forced out").
- `harmonia/models/local_key_model.py` — `LocalKeyGRU`, a bi-GRU over (root,quality)
  chord embeddings → 24-key logits, with **transpose-equivariant** training (random
  ±k-semitone augmentation shifts chords and label together).
- `scripts/check_local_key_premise.py`, `scripts/train_local_key_model.py`,
  `tests/test_local_key_data.py` (7 tests).

**Premise-check #1 — is section-local key a real problem on iRealb, or does it collapse
to a global-key problem?** Ran the oracle over the whole corpus (1856 songs, 6418 section
instances). Reported both the raw 24-class modulation rate AND the **collection-level**
rate (relative maj/min flips, e.g. C-major↔A-minor, share the same 7 diatonic pcs → give
the *identical* diatonic quality prior, so they are NOT real modulations for the downstream
prior and are filtered out).

  | corpus | sections | collection-modulation rate (margin 6) |
  |---|---|---|
  | jazz1460 | 4173 | **24.2%** |
  | pop400 | 2142 | 23.8% |
  | blues50 | 103 | 4.9% |

  Robust to the margin gate (jazz 29.8% → 24.2% → 18.4% across margins 3/6/10). 947/1856
  songs have ≥1 modulated section. **Oracle validated** (CLAUDE rule #1) two ways: (a) an
  independent membership oracle (best diatonic collection by chord-root coverage, gain ≥0.15)
  gives 17.1% jazz / 15.2% pop — same ballpark; (b) in oracle-modulated sections the local
  collection genuinely covers far more chord roots than the global one (jazz 0.72→0.87,
  pop 0.81→0.93), so the calls are real signal, not noise.

  **Verdict: section-local key is a real, non-trivial phenomenon (~15–25% of sections),
  NOT reducible to a pure global-key problem.** Note `pop400` (iReal-pop standards/bossa)
  is NOT POP909 — it modulates far more than POP909's 93.3%-diatonic pop, so the two
  should not be conflated.

**Model (Step 3):** LocalKeyGRU trained to imitate the oracle from chords alone, 40 epochs.

  | | val acc | modulated subset | non-modulated |
  |---|---|---|---|
  | baseline (always global key) | 74.4% | 0.0% (by construction) | 100% |
  | LocalKeyGRU | **83.7%** | **69.8%** (n=325) | 88.5% (n=944) |

  **+9.3pp over the global baseline — clears the ≥5pp "worth a sequential model" gate.**
  The model recovers ~70% of true modulations from chord symbols alone (the transferable
  capability for phase-2 audio). It does NOT trivially reproduce the oracle (transpose
  augmentation forces a genuine equivariant function, not memorization).

**What this does NOT solve / caveats:**
- The 11.5% *false-modulation* rate on non-modulated sections is the deployment risk: for a
  never-modulating diatonic pop song (e.g. **"Let It Be"** — the bug that motivated #20), a
  model that hallucinates a modulation would *hurt* the diatonic prior. Any prod wiring must
  keep the margin/confidence gate at inference (only override the global key when confident).
- The Let-It-Be `A major` vs `Am` error is itself a **within-global-key family** error, not a
  modulation — for that specific song a *robust global key* + diatonic prior already fixes it.
  The section-local model's payoff is the **jazz long tail**, not diatonic pop.
- Symbolic only. The oracle is a deterministic function of the (clean) chords, so on symbolic
  input the model can at best imitate it; the real value is transferring the learned mapping
  to noisy audio where the clean rule is unavailable.

**Phase-2 plan (audio, NOT started — next session):** MMA renders already exist at
`data/accomp_db/audio/` + `audio_hard/` (WAVs) with MIDI at `midi_path` per db record;
chroma is extracted via `PitchExtractor`/`analyze_accomp_emission.pc_vector` and folded with
`key_profiles.activations_to_chroma`. Plan: (1) extract per-section chroma from the MMA
renders, keeping the *same* oracle labels (chart is ground truth); (2) train an audio-input
version of LocalKeyGRU (chroma/quality features per beat) with the identical section split;
(3) target: match the symbolic model's 69.8% modulated-recall from audio; (4) then wire as
the local-key source for `chord_pipeline_v1`'s `apply_diatonic_prior`, replacing the
single-window `infer_key`, and re-run `scripts/eval_diatonic_prior.py`. **Disk gate: keep
≥10 GB free before any audio render/extraction (currently ~1.7 GB — must free space first).**

**Update 2026-07-12 — ported the client-side continuity heuristic as a zero-parameter
baseline (was never compared against the oracle):** the app already ships a rules-based
local-key tracker in `chart_interactive.py` (`continuity()`/`coreTones()`) that had been
mirrored server-side as `theory.local_key.continuity_scale_track` but never scored against
the section-key oracle. New `harmonia/models/local_key_heuristic.py` runs that per-chord
tracker over each song (seeded on the global key), reduces each section to one key by a
duration-weighted vote, and scores it on the **identical** GRU val split
(`scripts/eval_local_key_baselines.py`, `tests/test_local_key_heuristic.py`, 5 tests).

  | baseline | val acc | modulated recall |
  |---|---|---|
  | always-global-key | 74.4% | 0.0% (by construction) |
  | **continuity heuristic (ported, 0 params)** | **54.1%** | **23.7%** (n=325) |
  | LocalKeyGRU | 83.7% | 69.8% |

  **The heuristic is a NEGATIVE baseline** — it loses ~20pp to the trivial always-global
  baseline and ~30pp to the GRU. Root cause is **over-modulation**: it flags 47.6% of
  sections as modulated vs the oracle's 25.6%, because it jumps to a neighbouring collection
  on *any* out-of-scale chord (secondary dominant, tritone sub, borrowed iv…) with no
  hysteresis/margin gate. On non-modulated sections it stays on the global collection only
  69.3% of the time. Even ignoring relative maj/min flips (collection-level accuracy) it is
  only 60.3%, still below baseline. **Verdict: the GRU's +9.3pp is NOT reproducible by the
  zero-parameter tracker — this strengthens the case for the learned model.** Caveat: the
  heuristic optimises voice-leading continuity for *colouring*, not oracle imitation, so the
  comparison is somewhat unfair to it; but against the very quantity the GRU is judged on it
  is decisively worse, which is the question that was asked.

**Update 2026-07-12 (evening) — harmonic-minor-aware rewrite of the tracker
(`continuity_scale_track_v2`): the heuristic is no longer a negative baseline vs
v1, and one motivating musical bug is fixed.** Root-caused the over-modulation on
"Autumn Leaves" (Cm7 F7 Bb^7 Eb^7 Am7b5 D7b13 Gm6, static G minor, no real
modulation): v1 oscillated `Bb→…→G→F→Bb…` because it tested conformity against
the 12 **major** collections only (= natural minor). A minor key's own V7
(D7b13's raised leading tone F#) and i6 (Gm6's major 6th E) are *out* of the
natural-minor collection, so v1 read every functional-minor cadence as a key
change. Fix: a collection now also admits the **harmonic-minor** colour of its
relative minor (raised 7th) always, and the **melodic-minor** colour (raised
6th+7th) *surgically* — only for a chord rooted on the relative-minor tonic (the
i6 case). Full-scale melodic acceptance was tried and rejected (it pulls in
sharp-side harmony and dropped accuracy 55→44%). Jump tie-breaks now use a
2-chord lookahead (was 1). A minor-key home seed labels an all-diatonic run as
its minor tonic (not the relative major).

  | baseline | val acc | modulated recall |
  |---|---|---|
  | always-global-key | 74.4% | 0.0% |
  | continuity heuristic **v1** (natural-only) | 54.1% | 23.7% |
  | **continuity heuristic v2 (harmonic-minor-aware)** | **55.3%** | **27.7%** |
  | LocalKeyGRU | 83.7% | 69.8% |

  Strict improvement on both metrics (`continuity_scale_track_v2` in
  `theory/local_key.py`, now used by `local_key_heuristic.py`;
  `tests/test_local_key.py` +5 tests). "Autumn Leaves" now holds one stable
  "G minor" (0 collection changes) instead of oscillating. **Honest remainder:**
  v2 is still ~19pp under the trivial always-global baseline and ~28pp under the
  GRU — a purely *local* rule cannot resolve genuine progressive modulation
  (e.g. All The Things You Are's bridge, which v2 walks Bb→Eb→…→G but only
  approximately) nor the oracle's global-key hysteresis. The GRU's advantage
  stands; v2's value is interpretability + the fixed musical artifact for the
  app's "show keys" colouring. **Not yet done: port the fix to the prod JS
  (`chart_interactive.py::continuity()`)** — deferred as a UI deployment
  decision for the user, since it changes what the iPhone app displays.

**Update 2026-07-12 (late) — new model distilled from the HEURISTIC, not the
oracle; per-chord (not per-section) sequence labeling.** User decision (expert
musician): on the disagreement cases examined together (Criss Cross, Dear Old
Stockholm, A Beautiful Friendship) the **rule-based heuristic**
`continuity_scale_track_v2` tracks *what an improviser would actually play on
this exact chord* better than the section oracle, even though it is noisier
locally. So the teacher is now the heuristic, and the granularity is **per chord
across the whole song** (section boundaries no longer the unit of prediction).
This model complements — does not delete — the oracle-trained `LocalKeyGRU`.

- **New code:** `harmonia/models/local_key_seq_data.py` (per-chord distillation
  dataset), `harmonia/models/local_key_seq_model.py` (`LocalKeySeqGRU`, a
  **many-to-many** bi-GRU tagger emitting a key at every position),
  `scripts/train_local_key_seq_model.py`, `tests/test_local_key_seq.py` (11 tests).
- **Transpose-equivariant BY CONSTRUCTION** (not by augmentation, unlike the
  oracle `LocalKeyGRU`): both input roots and target keys are encoded *relative
  to the song's global tonic* (`tokens_to_rel_example`), so a song and all 12 of
  its transpositions produce a bit-identical (input, target) pair — the model
  learns each harmonic motif once. Verified: the ABF motif in E major is exactly
  the C-major prediction shifted +4 (`relative preds identical: True`; test
  `test_relative_encoding_is_transpose_invariant_by_construction`). This lifted
  val accuracy vs an absolute-encoding + augmentation variant by +3.6pp pop /
  +1.3pp jazz.
- **Per-position key accuracy vs the heuristic teacher (val, whole-song bi-GRU,
  pure distillation):** pop-like (pop400+blues50) **82.9%** (n=7659 chords),
  jazz1460 **85.4%** (n=14708). As the user predicted, pop is the calmer split.
- **The honest negative result — smoothing the secondary-dominant chain did NOT
  work.** Goal was: read `Em7 A7 D7 G7#5` (ABF section B, home C) as one gesture
  resolving to a single key, not 3-4 keys flickering past. The pure-distillation
  model faithfully reproduces the teacher's flicker (F→…→C→F→Bb→Eb, **5
  collection changes, identical to the heuristic**). Adding a tunable **soft
  collection-churn penalty** (`--churn-weight`, penalises `1-<q_{t-1},q_t>` on
  the softmax folded to 12 collections) trims *ambiguous* churn corpus-wide
  (jazz 44.0→42.8 changes/100 chords at w=1.0) but leaves the ABF chain at 5
  changes and costs fidelity + the clean equivariance demo. **Root cause:** each
  secondary dominant's own chord tones (A7's C#, D7's F#, …) strongly contradict
  staying in the prior collection, so *any* collection-continuity model — the
  heuristic, its distilled student, or a churn-regularised student — is pushed to
  flip there. Reading the chain as one descending-fifths gesture toward the final
  resolution needs a **functional/relational feature** (detect the fifths cycle
  and bind to its target), NOT more context over collection-membership. That is
  the clear next lever, deferred.
- **Genuine collection changes are preserved** (the capability that must NOT be
  lost to smoothing): the model fires both jumps in the user's spec case
  Gm7→F / Eb→Bb, in C major *and* transposed to A major (A→D→G) — tests
  `test_model_resolves_genuine_collection_change_gm7_eb`,
  `..._in_another_key`.
- **Shipped checkpoint:** `data/cache/local_key_seq_gru.pt` = pure distillation
  (w=0): best fidelity, provable equivariance. Churn penalty kept as a
  documented, off-by-default lever. **NOT wired into `chord_pipeline_v1`** —
  scoped to training + eval, per the session brief; prod integration is a
  separate user-validated step.

**Update 2026-07-12 (night) — the deferred functional feature + consolidation,
built. The ABF zigzag is FIXED (not just characterised).** The prior update's
clear next lever ("a functional/relational feature that detects the fifths cycle
and binds to its target, NOT more collection context") is now implemented on two
levers at once — a better *input representation* and a better *distillation
target* — exactly because either alone is insufficient (a relational input
feature is useless if the target still teaches the zigzag).

- **Lever 1 — relational input feature (`local_key_seq_data.rel_features`,
  transpose-invariant by construction).** Two per-chord features feed the tagger
  alongside `(root_rel, q5)`: `interval_to_next = (root[i+1]-root[i]) % 12`
  (13-way one-hot: 0..11 + a "no next" slot) and `is_dominant_prep`
  (`q5==dom AND interval_to_next==5` — "I am functioning as the V7 of the next
  chord"). Both are root *differences* ⇒ unchanged under transposition. Wired as
  two extra additive embeddings in `LocalKeySeqGRU`
  (`intv_emb`+`domprep_emb`; `use_rel_feats=True`, ablatable).
- **Lever 2 — deterministic consolidation of the distillation TARGET
  (`theory.local_key.consolidate_dominant_chains`).** Post-processes the raw v2
  track before distillation: finds maximal runs of ≥2 consecutive *dominants*
  each descending a perfect fifth to the next (a lone secondary dominant is left
  alone — v2 already labels it with its target collection), absorbs a leading
  **ii** (m7/m7b5 a fifth above the first V), and relabels the whole run with the
  key the chain **resolves to** (the following chord's key, or the implied
  `(root+5)` at a section end — home mode when that root is the home tonic, so
  `A7 D7 G7#5` in C → C **major**, the home it is the V of). This is the "v3"
  target; `build_seq_examples` now emits both `y` (v3) and `y_v2` (raw, for
  reference eval).
- **The ABF case is now solved end to end.** `G-7 C7 F^7 Bb7 E-7 A7 D7 G7#5`
  (home C): the tail `E-7 A7 D7 G7#5`, which v2 read as **C, F, Bb, Eb** (5
  collection changes across the section), the trained MODEL now reads as a single
  **C major** (2 changes) — matching the v3 target exactly, and the
  transpose-equivariance demo still holds bit-for-bit (E-major preds = C-major
  preds +4). Genuine borrowing is preserved: in a song-length context the model
  still fires C→F (Gm7) and F→Bb (Eb) (`_BORROW_CTX` test); Autumn Leaves stays a
  static G minor (0 changes).
- **Per-position accuracy (val, whole-song bi-GRU, no churn penalty):** pop-like
  **acc(v3) 82.7%**, jazz1460 **acc(v3) 84.0%** — on par with the pre-fix pure
  distillation (82.9 / 85.4 vs v2). Against the *old* v2 target the same model
  scores acc(v2) 81.4 pop / **78.6 jazz** — the ~7pp jazz drop vs v2 is the point:
  the model deliberately no longer imitates the v2 dominant-chain zigzag.
- **Churn (collection changes / 100 chords).** Corpus-wide the consolidation
  alone cuts the target churn **jazz 44.04 → 40.02 (−4.02)**, pop 23.41 → 22.43;
  it relabels 6.5% of jazz chords across **49.5% of jazz songs** (dominant chains
  are common) vs 2.3% / 28.4% for pop. On val the trained model tracks the v3
  target: jazz raw-v2 44.04 → v3-target 39.96 → **model 40.86**; pop 21.96 →
  21.58 → 23.11. This beats today's blunt anti-churn penalty (42.8 jazz) via the
  principled route and *without* costing fidelity or the equivariance demo.
- **Generalisation (spot-checked beyond ABF):** rhythm-changes bridge
  `D7 G7 C7 F7` (Bb) → one Bb (churn 3→0); Sweet Georgia Brown `D7 G7 C7 F6`
  (F) → one collection (3→0). Conservative where it should be: two *separate*
  ii-Vs (`A-7 D7 G^7 | D-7 G7 C^7`) are NOT merged, and the ATTYA bridge (a
  genuine mixed-quality progressive modulation) is left exactly as v2 labelled
  it — the rule requires ≥2 chained *dominants*, so it cannot flatten a real
  modulation.
- **Honest remainders:** (a) the arrival *inherits the resolution chord's raw v2
  label*, so when that chord is itself collection-ambiguous the tonic name can be
  off (Sweet Georgia Brown's chain lands on the "Bb major" collection F6 sits in,
  not "F major") — the *collection* is right, only the maj/rel-minor tonic label
  is debatable; (b) a tritone-sub chain (roots move by semitone, not a fifth) is
  out of scope; (c) a chain that dangles into a *minor* resolution at a section
  end defaults to major unless it coincides with the home minor tonic. (d) Still
  **NOT wired into prod** — training + eval only, per brief.
- **Code:** `theory/local_key.py` (`consolidate_dominant_chains`,
  `is_dominant_quality`), `models/local_key_seq_data.py` (`rel_features`,
  `build_rel_example`, dual targets), `models/local_key_seq_model.py` (feature
  embeddings, dict-based `collate`), `scripts/train_local_key_seq_model.py`
  (dual-target eval), tests: `test_local_key.py` +6, `test_local_key_seq.py` +5
  (52 local-key tests green).

---

## 33. Section-merge was silently rejected and reported as success — FIXED 2026-07-13

`pool_beat_evidence` requires the merged spans to have **equal beat count**
("equal musical length" is a documented v1 precondition — it lines the spans up
beat-by-beat). When they don't, it raises `ValueError`, and
`chord_pipeline_v1` catches it, logs

```
WARNING  chord_pipeline_v1: section-merge rejected (spans cover [143, 174] beats …)
```

and **decodes unconstrained anyway**. Graceful degradation is the right call
inside the pipeline — but `/api/reinfer` returned `200 / n_changed: 0` with no
indication anything was refused, so the app showed its "Merged — one shared
reading" banner for a merge that never happened. A UI cannot be honest about a
model it cannot hear.

**Fixed on both ends:**
- `/api/reinfer` attaches a `rejected: [...]` list (captured from the pipeline
  logger) and the app refuses to claim success when it is non-empty.
- The form ribbon now greys out sections that *cannot* be pooled with the one
  you picked (unequal bar count), prints each section's length on its chip, and
  says why on tap. The dead end is no longer reachable.

**Availability of the feature, measured over the library** (equal-length section
pairs / all pairs): 130/370, and all 12 multi-section charts have ≥1 mergeable
pair — but it is very uneven. Feeling Good has 6 same-letter equal-length pairs
(A¹–A⁴ are all 16 bars); Autumn Leaves has effectively none, because its A
sections are 72/88/128-bar solo choruses that the segmenter never cuts to equal
length. **The merge feature's premise — that repeats segment to equal musical
length — holds for tight pop forms and fails for jazz solo choruses**, which is
where the pooling would have been most valuable. Lifting the equal-length
precondition (e.g. DTW-align the two spans before pooling) is the obvious next
step and is not done.

---

## 30. Every real-audio chart in `major` was baked as `minor` — FIXED 2026-07-13

**Silent calibration bug (pattern #1), found by unit-testing the most basic
load-bearing assumption of a stage I was about to build on** (the ChartModel
adapter reads `P.home`), exactly as the counter-rule says to.

`chart_interactive._parse_home_key` served two key-string dialects: the iReal
DB format (`"Ab"`, `"G-"`) and `chord_pipeline_v1.global_key` (`"G# major"`).
It decided the mode with

```python
mode = "minor" if "-" in key[i:] or "m" in key[i:] else "major"
```

The word **"major" contains an "m"**. So every key string in the pipeline's
own format parsed as minor: `_parse_home_key("C major") == (0, "minor")`. The
true-minor charts came out right by luck ("minor" also contains an "m"), which
is why this never looked obviously broken.

**Impact — not display-only.** The chart's client JS derives its relative-major
reference from that field (`chart_interactive.py:1049`):

```js
const maj = h.mode === "major" ? h.tonic : mod(h.tonic + 3, 12);
```

so on every affected chart the scale/function analysis and key colouring were
keyed to a tonic **three semitones off**. 9 of 17 charts in `docs/plots` were
wrong (all the major ones — Autumn Leaves, Let It Be, Georgia On My Mind, …).

**Fixed:** "maj" is now tested before the bare "m" (`tests/test_chart_model.py::
TestParseHomeKey` is red against the old behaviour). `render_interactive` now
also bakes the raw `keyName` into the payload, so the string is recoverable
without re-parsing. Already-baked charts were repaired in place by
`scripts/fix_chart_home_mode.py` (idempotent; recovers the key from the
chart's own subhead — no re-inference needed).

---

## 31. `P.sections` holds the KEY NAME on real-audio charts, not a form letter — WORKED AROUND 2026-07-13

Per-bar section labels are supposed to be form letters (A/B/C). On symbolic
(iReal) charts they are. On real-audio charts, `Chart.section_per_bar` is
filled with the local key, so `P.sections` reads
`["G# major", "G# major", …]` for all 330 bars.

Anything that reconstructs the form by grouping runs of equal per-bar labels
therefore sees **one section spanning the whole tune, named "G# major"** — and
it looks plausible enough to ship. The real form is in `P.sectionChips`
(`[{label, start_s}, …]`, one entry per segment of the changepoint
segmentation).

**Worked around, not fixed:** `chart_model._section_runs` prefers
`sectionChips`, falls back to per-bar letters only when they actually look like
form letters (≤2 chars, no space), and falls back again to a single section.
The underlying naming collision in `section_per_bar` is still there — it is a
field doing two jobs depending on which pipeline filled it.

---

## 32. The 390×844 headless-screenshot recipe in our own docs renders at 500px — FIXED 2026-07-13

Every doc in this repo that tells you to verify phone layout with

```
google-chrome --headless --screenshot=x.png --window-size=390,844 URL
```

is wrong on macOS: **Chrome clamps its window to a 500px minimum width**,
renders the page at 500 CSS px, and scales the image down to 390. It looks like
a phone screenshot. It is not one. `window.innerWidth` inside that page reads
**500**, and a layout that overflows at 390 looks perfectly fine in the image.

This is pattern #6 (a component swap — here, the verification instrument —
changing more than the target metric) applied to our own tooling: the app grid
was clipping bars 3 and 4 of every row at 390px, and the "verification"
screenshot showed a clean 2-column chart.

**Fixed:** `scripts/phone_screenshot.py` drives Chrome's DevTools Protocol and
sets `Emulation.setDeviceMetricsOverride` (a real 390-CSS-px mobile viewport,
@2x, touch on) — what DevTools mobile emulation actually does. It can also run
JS in the page (`--eval`) and tap by label (`--click "text=Annotate"`), so a
layout claim can be checked by measurement (`grid.scrollWidth` vs
`clientWidth`) and not by eyeballing a rescaled picture.

---

## 24. Chord quality display corrupted for every YouTube analysis since section-chips wiring — FIXED 2026-07-12

**Two silent calibration bugs (pattern #1) in the label → iReal token converter,
`scripts/render_youtube_chart.py::_split_label`/`_QUALITY_TO_IREAL`, found while
adding chord-suggestion data (which flows through the same function).**

1. **Colon separator never handled.** `chord_pipeline_v1.infer_chords_v1`
   always emits labels as `f"{NOTE[root]}:{sev_h}"` (e.g. `"D:maj7"`).
   `_split_label` was written for a concatenated format (`"F#min7"`, docstring
   example, no colon) — a leftover from before `chord_pipeline_v1` became the
   production pipeline. `_split_label("D:maj7")` correctly found the root
   (`"D"`) but left the quality tail as `":maj7"`, which doesn't match any key
   in `_QUALITY_TO_IREAL` and fell through unchanged. Every chord's `exact`
   *and* `seventh` *and* `family` level ended up showing the identical raw
   `":maj7"` string — the family/seventh collapse never ran either, since that
   lookup fails the same way.
2. **`hdim7`/`dim7`/`minmaj7` vocabulary mismatch**, independent of the colon
   bug. `_QUALITY_TO_IREAL`/`_QUALITY_TO_FAMILY` were written against
   `chord_vocabulary.ChordQuality.value` strings (an older pipeline's
   vocabulary: `"ø7"`, `"°7"`, `"mMaj7"`), not `chord_pipeline_v1`'s actual
   `sev_h` names (`_SEV_TO_Q5` in that file: `"hdim7"`, `"dim7"`,
   `"minmaj7"`). These three qualities fell through to the raw name even after
   fixing the colon issue.

**Impact — real, not hypothetical:** every song analyzed through the live
mobile app (YouTube search → analyze) since `render_youtube_chart.py`'s
`chart_to_interactive_inputs` became the wiring for the production endpoint
showed garbled chord quality symbols (literally `":maj7"` etc. instead of
`"^7"`) in the chart UI. Root, bass, section boundaries, and confidence
values were **not** affected (`parse_token`'s root/bass extraction succeeds
independent of the quality tail; confidence is a separate float). Audited
every `docs/plots/inferred_*.html` by grepping for `'"q": ":'`: **8 of 15
charts were corrupted** — `autumn_leaves` (1944 corrupted quality fields),
`autumn_leaves_remastered` (444), `love_is` (744), `my_baby_just_cares_for_me`
(696), `nina_simone_feeling_good` (501), `ray_charles_georgia_on_my_mind`
(534), `the_beatles_let_it_be` (504), `yam_b_cane` (516). The other 7 predate
this pipeline wiring and were unaffected.

**Fixed:** both tables corrected in `scripts/render_youtube_chart.py` (colon
split handled first; `hdim7`/`dim7`/`minmaj7` aliases added to both dicts).
**Repaired retroactively** without re-running inference — the corrupted
`"q"` string *is* the recoverable raw `sev_h` (with a leading `:` if that was
the failure mode), so `scripts/fix_colon_quality_labels.py` re-derives the
correct family/seventh/exact tokens from it and patches the 8 baked chart
JSONs in place. Verified 0 remaining `'"q": ":'` matches across all 15 charts
after the fix, and spot-checked `inferred_autumn_leaves.html` visually
(headless-Chrome screenshot) before/after.

**What this fix does NOT solve:** it's a display-string bug only — the
underlying root/quality *classification* (the actual `sev_h` the model
predicted) was always correct in the data model, this bug only broke how it
got printed. No re-evaluation of pipeline accuracy is needed. Also does not
touch `chord_hmm.py`'s separate (frozen, unused) label path.

---

## 25. `eval_irealb_e2e.py` bypass harness mismeasured the progression reranker — default-ON REVERSES on the real path — FOUND 2026-07-13

**Pattern #1/#6 again, this time in the eval harness, and it flips a shipped
default.** `eval_irealb_e2e.py` — the harness used to justify issue #21's
"reranker default ON, +1.0pp majmin" decision — bypasses the ctx family
classifier entirely (uses the raw family LR + the confidence-gated one-hot
acoustic prior). The production path (`infer_chords_v1`) runs the ctx model
and feeds real q5 log-probs to the reranker — a different acoustic-prior
geometry than the one the reranker weight was tuned against.

**Re-measured on the real path** (`scripts/eval_two_pass_801d.py`, held-out
jazz1460 idx 70–95, n=25, MuseScore render, both arms bit-for-bit reproduced):

| ctx variant | reranker | root | majmin | 7ths | min | hdim |
|---|---|---|---|---|---|---|
| 684d | OFF | 88.7% | **84.0%** | **59.2%** | 83% | 68% |
| 684d | ON (prod default, w=2.0) | 88.7% | 80.4% | 56.7% | 73% | 47% |
| 801d_two_pass | OFF | 88.7% | **84.0%** | **59.2%** | 83% | 68% |
| 801d_two_pass | ON | 88.7% | 83.1% | 57.7% | 82% | 68% |

Reranker marginal on the real path: **−3.6pp majmin / −2.5pp 7ths (684d)**,
**−0.9pp / −1.5pp (801d)** — negative in every cell, larger in magnitude than
the +1.0pp the bypass harness claimed. **The default-ON decision reverses.**

**Two corollaries.** (a) The 801d two-pass "+2.7pp majmin gain" (issue #20's
2026-07-13 entry) is real only *relative to the damage the reranker does*:
with the reranker OFF, 801d's hard argmax is **byte-identical to 684d** (its
refined q5 distribution only ever flows into the reranker). 801d is a no-op
until some consumer of its distribution exists — don't flip its default.
(b) Every prior conclusion measured with `eval_irealb_e2e.py` (ProgressionEncoder
gain, phase-fix effect on e2e numbers) carries the same proxy-path caveat and
should be re-read as "on the bypass path", not "in prod".

**Action taken 2026-07-13:** POP909 cross-check first
(`scripts/eval_pop909_reranker_ab.py`, 5 songs, v005 renders, real path):
reranker ON vs OFF is **byte-identical on every song** — the reranker never
fires on POP909 at all (no rerank-failure warnings; the toggle demonstrably
works, the jazz 2x2 shows ON≠OFF through the same parameter). Both corpora
therefore agree OFF ≥ ON, and `use_progression_prior` default flipped to
False in `infer_chords_v1`; the reranker stays available opt-in. The encoder
itself is NOT dead — its information belongs in a joint decode as a
transition factor (audit build-order step 2), not as a greedy post-hoc
override tuned against a proxy. Why it never fires on POP909 is unexplained
(plausibly the jazz-trained ctx q5 log-probs are peaked/overconfident on pop
input — pattern worth checking when calibrating), noted, not investigated.

**What this does NOT solve:** the reranker-OFF config still leaves the
progression/local-key evidence unused at inference; hdim/dim remain weak
(68%/56%); and the harness family (`eval_irealb_e2e.py`) still exists and can
mislead again — prefer `eval_two_pass_801d.py`-style real-path evals for any
future decision.

---

## 26. Displayed chord confidence was uncalibrated, root-blind, and stale after rerank — RESOLVED 2026-07-13 (audit step 1)

The app's core premise (show where the model is unsure) was unbacked in three
stacked ways, all fixed this session:

1. **Stale after rerank** (`3e9f0f4`): the 8a/8b rerankers flipped a quality
   but carried the pre-rerank conf. Both rerankers now return the posterior of
   the decision they actually made (`return_post=True`); write-back consumes
   it at flipped positions. Red-first tests in `tests/test_rerank_confidence.py`.
2. **Root-blind** (`fa088d4`): quality heads never see the root. Output chords
   now carry `root_conf` (span-mean beat_seq posterior at the label's root)
   and `confidence_raw`; the displayed `confidence` fuses conf × root_conf.
3. **Uncalibrated**: isotonic map fitted by
   `scripts/fit_confidence_calibration.py` on jazz1460 songs (interleaved
   song-disjoint splits, eval set 70..95 excluded), saved to
   `data/cache/confidence_calibration.npz`, auto-loaded like every other
   artifact. Display-layer only — applied at output assembly after every
   label/gate, cannot change a decision by construction.

**Numbers (disjoint test split, 38 songs / ~1300 chords; target = root pc AND
q5 family correct at chord midpoint):** raw fused ECE 0.2325 → calibrated
**0.0366** (gate < 0.05 PASS). First attempt with block splits failed the gate
(0.0561) because the blocks had a real difficulty shift (fit 82.4% vs test
74.4% base accuracy) — the interleaved song-disjoint split removed the
confound. Autumn Leaves e2e: pinned-at-1.0 chords 14 → 2; a raw-1.00 quality
call over a weak root (root_conf 0.35) now correctly displays 0.65.

**What this does NOT solve:** (a) calibration is fitted on MMA-synth audio —
it will be overconfident on real recordings until re-fitted there (issue #19);
(b) the raw fused score has a non-monotone dip at ~[0.6,0.7) (two populations
mixed — isotonic maps it conservatively but a 2-feature calibration could do
better); (c) the fit script doesn't yet save per-chord preds, so the
reliability *plot* needs a re-run (add --save-preds); (d) suggestions'
probabilities are still the raw pre-rerank posteriors.

---

## 27. Joint (root × quality) segment Viterbi — GATE PASSED 2026-07-13 (audit step 2, default ON)

`harmonia/models/joint_decode.py` + `use_joint_decode` in `infer_chords_v1`
(default ON since the same day, after refitting the confidence calibration on
the joint path — the joint `conf` is a forward–backward max-marginal, different
semantics from the greedy family max-prob the first map was fitted on):
per segment, top-K=3 candidate roots (GT-root top-3 coverage on
real fit-split segments: **99.3%**, premise script
`scripts/premise_joint_root_coverage.py`) × 5 qualities; exact Viterbi +
forward–backward over the segment chain; `conf` = the state's max-marginal.
Subsumes/disables the two-pass, local-key and progression rerankers.
Segmentation unchanged — the decode only relabels.

**Gate (real path, `scripts/eval_joint_decode.py` / `eval_pop909_joint.py`):**

| corpus | arm | root | majmin | 7ths |
|---|---|---|---|---|
| jazz1460 idx 70–95 n=25 | greedy (defaults) | 88.7% | 84.0% | 59.2% |
| | **joint w=0** | 88.7% | **86.2%** | **60.5%** |
| POP909 5-song v005 | greedy | 77.1% | 50.0% | 45.8% |
| | joint w=0 | 76.9% | 50.1% | 45.9% (3/5 songs byte-identical) |

Two findings the gate forced out:

1. **`_family_q5_logprobs` is contaminated as an emission** (first gate run
   FAILED −5.3pp majmin): it folds aug+sus family mass onto `maj`, so a chord
   the family head calls minor can have `maj` as its q5 argmax. Fixed locally
   in `joint_decode` (greedy anchor: the classifier's own call is raised to
   emission argmax; `_family_q5_logprobs` itself untouched — the suggestions
   display still consumes its raw form, which therefore still carries this
   bias, see #26d).
2. **The corpus progression bigram is net-negative as a transition factor on
   jazz** (fit-split sweep idx 20–30, w ∈ {0, .1, .25, .5, 1, 2}: every
   positive weight snaps min/hdim/dim toward the majority-major prior; even
   w=0.1 costs min 80→66). `joint_transition_weight` defaults to 0.0 — the
   realized gain is entirely the root×quality *emission coupling* (the decode
   may pick a top-2/3 root whose quality evidence is stronger). Consistent
   with the "global bigram progression prior premise MARGINAL" dead-end.

**What this does NOT solve:** (a) default stays OFF — flipping it is an
orchestrator/user call; (b) the transition slot is wired but empty (w=0) — the
encoder/grammar information still needs a key-*local*, calibrated factor to be
net-positive on jazz; (c) non-argmax candidate roots reuse the greedy top-1
neighbour context in the ctx classifier (v1 approximation); (d) per-beat
semi-Markov (durations, `viterbi_duration_aware`) is the next step and needs
per-beat emissions plus the same anchor fix.

### Mission 1 (2026-07-13): three grammar factors for the empty transition slot — ALL DEAD ENDS

Goal: fill the `joint_transition_weight=0` slot with a grammar factor that lifts
jazz majmin past 86.2 (held-out gate). Three hypotheses tested on the FIT split
(jazz1460 idx 20–30, n=10; w=0 baseline root 92.4 / majmin **88.4** / 7ths 62.0).
**Every factor's optimum is λ→0** (monotone decreasing in weight) — not a tuning
problem, the factor points the wrong way on net. Numbers (majmin, best λ):

| factor | mechanism | best majmin | family damage |
|---|---|---|---|
| **H1 key-local bigram** | transition re-referenced to per-chord local key (`joint_local_key_transition`) | 78.0 @0.25 | min 80→56, dim 44→6 — *worse* than global |
| global bigram (prior) | transition, global tonic | 85.1 @0.25 | min 80→65, dim 44→28 |
| **H2 encoder shallow fusion** | ProgressionEncoder `log P(q\|ctx)` as per-cand-root EMISSION factor, centre masked (`joint_progression_fusion`) | 87.5 @0.5 | min 80→71, dim 44→28 |
| **H3 density-ratio fusion** | H2 minus encoder marginal `log[P(q\|ctx)/P(q)]` (`joint_fusion_subtract_prior`) | 86.0 @0.25 | maj 93→90, min 80→69, dim 44→50 (bias *flips* to rare classes) |

Diagnoses (each falsifies its hypothesis, not just "didn't help"):
- **H1**: the continuity-teacher local tonic changes on **46% of adjacent chord
  pairs even on CLEAN GT tokens** (`_localkey_track_from_qualities_v2`). A bigram
  FIT under one global reference, applied under a per-chord-shifting reference,
  makes ~half of all transition lookups correspond to no real root motion — mass
  concentrates onto the major-family cells *harder* than the global reference, so
  H1 is strictly worse. The reference-frame idea would need the bigram *re-fit*
  under the local reference; Korzeniowski's "bigram gains are marginal" + global
  already net-negative make that unpromising.
- **H2/H3**: raw shallow fusion carries the corpus label prior (majority-major) →
  the classic ACR label-bias snap (Korzeniowski & Widmer keep P(y) uniform).
  Density-ratio subtraction (H3) removes it but then *over*-rewards rare classes
  (dom/dim up, maj down) — still net-negative because at w=0 majmin is already
  ~88 and its residual errors are ACOUSTIC (maj↔dom, 5th-apart root), which a
  quality grammar cannot fix. Root is flat (92.4→92.5) across every arm: the
  emission coupling already resolves roots the grammar might have.
- The **unmasked-centre** fusion variant is exactly the reversed #21/#25 rerank
  (already net-negative), so it is pre-falsified — the masked/density-ratio forms
  here are the new, more principled attempts.

**Outcome:** no default changed. The three factors are wired default-OFF behind
`joint_local_key_transition` / `joint_progression_fusion` /
`joint_fusion_subtract_prior` in `infer_chords_v1` (+ `local_tonic` / `q5_bonus`
hooks in `joint_decode`, `_progression_fusion_bonus_fn` helper), with unit tests
(transposition invariance + w/λ=0 reproduction). `joint_transition_weight`
stays 0.0. **The grammar slot on the SEGMENT decode is a dead end for jazz
majmin**; the lever the numbers point to is Mission 2 (per-beat semi-Markov with
explicit durations) — Korzeniowski's own result is that ACR gains come from
segment/duration models, not frame/label LMs, and here the residual errors are
acoustic, so duration-aware emission (not a quality prior) is the next move.

**Addendum (2026-07-13, trigram falsification check, `scripts/exp_trigram_fusion.py`):**
a scale-relative TRIGRAM over (deg, q5) states (add-k + bigram/unigram backoff, fit
on jazz1460 excl. fit+gate) fused via the iterated `q5_bonus` path — semi-Markov ON,
fit baseline root 93.5 / majmin **89.7** / 7ths 62.4 — is ALSO net-negative at every
λ ∈ {0.1, 0.25, 0.5} × {left, full-context} × {raw, density-ratio} (best arm: full
λ=0.1, majmin 87.6, −2.1pp; same min 80→70 / hdim 89→67 major-snap signature; ratio
variants worse, −5.5 to −11.9pp). Sparsity is NOT the failure mode (100% trigram-
context / 82.9% exact-trigram coverage of fit-split GT) — the trigram confirms,
not escapes, the bigram/encoder diagnosis: residual errors are acoustic.

**Addendum 2 (2026-07-13, ENTROPY-GATED trigram on the ii-V-I slice,
`scripts/exp_trigram_gated.py`) — the user's counter-example, properly tested.**
The user objected: "when there is a 2-5-1, we know the 5 has a dominant 7" — and
the always-on sweeps above could not see it, because (a) a global λ lets
corpus-marginal losses swamp cadence wins, and (b) **MIREX majmin maps dom7→maj,
so a V:maj→dom7 fix is INVISIBLE in majmin** (it lives in 7ths + dom-recall). So:
a slice metric over GT ii-V instances (key-agnostic, catching tonicized ii-Vs) +
a trigram that fires ONLY where it is sharp (predictive entropy < thr), the
acoustics are uncertain (p_max < τ), and both context chords are confidently
decoded.

**Result: the premise is already satisfied by the acoustic model.** On 68 GT ii-V
instances (fit split, semi-Markov ON), the production decoder already gets the V
chord right **65/68 (95.6%) at q5 AND root+q5** — there are only 3 errors to win.
Every gated arm (λ ∈ {0.5, 1.0} × H_thr ∈ {1.0, 1.75, 2.5} × τ ∈ {0.65, 0.8})
is numerically IDENTICAL to baseline (7ths 62.4, majmin 89.7, root 93.5, min 80%,
hdim 89%); gate firing rate 0–5%.

**The per-error diagnostic is the real finding — it explains why grammar cannot
help, and it is not "grammar is worthless":**

| on the 3 V-chord errors | value |
|---|---|
| context chords confidently decoded (gate iii) | **0%** (never) |
| acoustic p_max, median (gate ii wants LOW) | **0.69** — one error at **0.93** |
| trigram entropy H, median (gate i wants LOW) | 2.05 (vs 1.68 on correct) |

Two independent killers: (1) **the context is broken exactly where it is needed** —
a trigram can only recognize a ii-V-I if it first sees a correct `ii`, and in 0/3
failures were the neighbours confidently decoded (when the model is lost, it is
lost about the context too — grammar can only exploit a cadence the model has
already half-recognized); (2) **the acoustics are confidently WRONG, not
uncertain** (`F:maj` at p_max 0.93), so an "intervene where unsure" gate never
opens. That is a **q5-level calibration failure, not a missing-grammar failure**.

**Reframe:** the grammar slot is not dead because chord statistics are worthless —
it is dead because grammar needs *trustworthy neighbours* and *honest uncertainty*,
and precisely where we fail we have neither. This retro-explains why Mission 3's
constrained decode DOES benefit from the transition factor (tw=2.0): **a human
confirm supplies exactly the trustworthy anchor the trigram was missing.** Route
to revive grammar: fix per-quality calibration first (see #26/#29), then re-test
the gate — do NOT re-test grammar before that.

### Mission 2 (2026-07-13): per-beat semi-Markov (explicit duration) — GATE PASSED, default ON

`harmonia/models/semi_markov_decode.py` + `use_semi_markov` (default **ON**,
`semi_markov_dur_weight=0.25`) in `infer_chords_v1`. An explicit-duration Viterbi
over (root×q5) with a jazz1460-fit duration prior decides the SEGMENT BOUNDARIES;
the existing joint decode then labels root×quality on those segments. Reuses the
frozen `chord_hmm.viterbi_duration_aware` (O(T·D·C²), MAP/Viterbi semiring — no
sum-product forward-backward, sidestepping the HSMM log-space-scaling caveat).

**Why Gen-2 succeeds where Gen-1 Candidate B failed (#1):** Candidate B forced
the true ~2-beat rhythm onto a weak per-SEGMENT emission and just exposed the
weakness (majmin 17→10). Gen-2 has a strong per-beat root posterior (beat_seq_v4,
96% per-beat on clean renders) and — critically — does NOT trust the decode for
quality: quality is re-labeled by the joint decode's top-K root×quality coupling
(the v3 per-beat quality head is only 51.7% q5-exact, premise check a — using it
for the label craters majmin ~8pp). So the decode owns ROOTS + BOUNDARIES only.

**Premise checks (rule #2, both passed):**
- (a) v3 quality head 51.7% per-beat q5-exact vs v4 root 96.3% → quality NOT
  trusted to the decode (drove the architecture above).
- (b) jazz1460 GT chord durations are sharply NON-geometric: d=2 57%, d=4 30%,
  odd durations ~0%, d=1 4% — a much sharper boundary signal than POP909's.
  `scripts/build_duration_prior_jazz.py` fits {pooled,(5,D) per-q5} PMFs
  (excludes gate idx 70–95), cached `data/cache/duration_prior_jazz1460.npz`.

**Label-bias discipline (Korzeniowski & Widmer, uniform-prior):** the duration
prior carries a "long⇒major" bias (maj/min have more 4/8-beat mass). Default uses
a QUALITY-INDEPENDENT pooled prior (zero quality label-bias); a per-q5 variant
enters as a density ratio log[P(d|q)/P(d)] (`semi_markov_per_quality_dur`, OFF).

**Sweep (fit jazz idx 20–30, n=10):** `w=0` is BIT-IDENTICAL to production joint
(segmentation reduces exactly to root-change argmax) — the clean degenerate
check. `w=0.25` best: root 92.4→93.5, majmin 88.4→89.7, 7ths 62.0→62.4. Higher w
over-merges and eats short dim/hdim chords (dur=1.0: hdim 89→67).

**Gate (held-out):**

| corpus | arm | root | majmin | 7ths |
|---|---|---|---|---|
| jazz1460 idx 70–95 n=25 | prod (joint) | 88.7 | 86.2 | 60.5 |
| | **sm dur=0.25** | **89.4** | **86.6** | 60.4 |
| POP909 5-song v005 | prod (joint) | 76.9 | 50.1 | 45.9 |
| | **sm dur=0.25** | **78.6** | **51.1** | **47.0** |

Root gate (>89.0) AND majmin gate (>86.5) both pass; no metric regresses >0.5
(jazz 7ths −0.1). Total jazz root errors 110→97. The jazz1460-fit prior
generalizes to POP909 (both 2/4-beat harmonic rhythm). **Headline: the ROOT lever
Mission 1 pointed to is real** — duration/boundary evidence, not grammar.

**What this does NOT solve / caveats (rule #4):** (a) the 5th-apart *share* of
root errors is unchanged (~35–40%) — merging fixes ISOLATED single-beat root
errors, not systematic span-level 5th-of-bass confusions (those stay acoustic, as
Mission 1 predicted); the +13 fewer errors are mostly non-5th. (b) **Production
default changed** — other eval harnesses' unqualified `infer_chords_v1(...)` calls
now get semi-Markov ON; pass `use_semi_markov=False` for a true pre-M2 baseline.
(c) `data/cache/duration_prior_jazz1460.npz` is gitignored; a fresh checkout
without it falls back to root-change segmentation (warned, not crashed) — rebuild
with `scripts/build_duration_prior_jazz.py`. Tests: `tests/test_semi_markov_decode.py`
(degenerate=per-beat argmax, duration-override, transposition invariance,
pooled-prior quality-independence).

---

## 34. Mission 5 LLM priors — glue WIRED + verified, but non-circular symbolic eval INFEASIBLE on this corpus — 2026-07-13

Full writeup: `docs/mission_5_part_ab_results.md`. Prior context:
`docs/mission_5_audit.md` (V1 was unmeasured: not an LLM, never wired, circular
sim).

**Part A (DONE, verified).** The analyst priors now enter the production joint
decode through three of the four seams, behind `use_llm_priors=False` (default
OFF ⇒ bit-identical to production):
`harmonia/models/chord_pipeline_v1.py::apply_llm_priors` + helper
`bars_to_segment_groups` (slot-wise repeat→segment tie), wired into
`infer_chords_v1(use_llm_priors=, llm_analysis=, llm_song=, llm_playlist=,
llm_max_nats=)`. Settings: `LLM_KEY_TRUST=0.60`, `max_nats=8.0`, transition bias
OFF (#27 saturated). End-to-end proof (not just a unit test): an extreme
dominant-everywhere analysis on POP909 render 001 moves 116/131 decoded labels
toward dom — the q5_bonus provably reaches the real emission. Tests:
`tests/test_llm_priors_glue.py`. Unsolved remainder (CLAUDE.md #4):
`bars_to_segment_groups` assumes bar 1 = beat 0 and fixed `beats_per_bar` (4/4);
does not recover bar↔beat phase (pickup bars misalign pooled slots).

**Part B1 (INCONCLUSIVE — premise falsified).**
`scripts/eval_llm_priors.py --cross-source` derives priors from iReal source A,
scores against a different lead-sheet B of the same tune. Cheap premise check
(CLAUDE.md #2) first: of 40 titles present in ≥2 of 7 playlists, 30 are
byte-identical transcriptions (0% disagreement → trivially circular), 7 are
homonyms/transpositions/length-mismatches (different song), leaving **2 valid
pairs** (Blue Room, C'est Si Bon) at only 3–5% disagreement. With A≈B, Δcross
(+4.2pp) ≈ Δcirc (+4.0pp) **by construction** — the numeric +2pp gate passes but
the test has no power to measure analyst transfer, and the +4.2 is carried by a
single tune. The script prints `Test power: ... INADEQUATE` and VERDICT
INCONCLUSIVE. **Do NOT proceed to Part C on this basis.** The real gate is Part
B2: prior from chart, GT from the audio's annotation on the Mission-1 real-audio
benchmark (`data/real_audio_benchmark/`, #20/#28), the only setup that is both
non-circular and has genuine chord-level disagreement. Gated on Mission 1.

---

## 36. GT-eval UNBLOCKED — first honest iReal-GT accuracy on real audio — 2026-07-14

Resolves the #35 blocker. #35's own prescription ("export inference + iReal GT on
a shared audio clock, same DTW pass") is already implemented as
`harmonia.irealb_aligner.align_irealb_to_inferred`. The stale `irealb_<slug>.html`
artifacts were from an OLD aligner that under-detected repeats (autumn_leaves GT
160 s vs inferred 422 s); the CURRENT aligner tiles correctly (fresh run: GT span
422 = 422 s, 8 choruses). So re-run the alignment fresh — do not trust the cached
HTML match fields.

Deliverables: `scripts/validate_against_ireal.py`,
`data/ireal_gt_validation_set.json`, `docs/ireal_validation_results.md`,
`docs/plots/ireal_accuracy_comparison.png`. Inferred chords are read from the
embedded `const P` in `inferred_<slug>.html` (no pipeline re-run); iReal GT from
the corpus tune → `tune_to_mma`. 9 of 14 mapped songs pass the coverage gate
(2 680 chords).

**Headline: pooled root 0.47, family 0.40, joint 0.27 — BUT the honest number is
the lift over a spurious-alignment floor.** The aligner picks the best of 12
transpositions + time-warps, so it aligns a WRONG chart to ~0.34 root agreement
(measured: each chart aligned to 2–3 deliberately-wrong tunes). Mean per-song
lift over floor is only **+0.13**, and it is **carried by two clean-audio songs**
(let_it_be +0.38, blue_bossa_150bpm backing-track +0.27). Real full-mix jazz
recordings sit at +0.04…+0.10 — barely above chance under this alignment.
Quoting "47 % root accuracy" bare would be the fabricated-number trap (CLAUDE.md
#2/#3); always report against the floor.

Family breakdown (root+family, pooled): maj 0.46 / min 0.22 / dom 0.21 / hdim
0.14 — corroborates #35 finding #3 (quality collapses toward maj/dom; ø/dim
nearly lost). This is the highest-leverage fix (M2 quality head).

Not done (harness ready, scoped out): baseline-vs-retrained-vs-LLM A/B needs the
audio pipeline re-run per config; `validate_against_ireal.py` scores any set of
`inferred_<slug>.html` so the A/B is just re-render + re-point. Conf-vs-accuracy
is confounded by the two-domain split (#26) — read within domain, not pooled.

## 35. Failure-mode dashboard v2 + GT-eval is BLOCKED by a chart timeline mismatch — 2026-07-14

Deliverables: `docs/error_analysis_dashboard_v2.html`,
`docs/failure_mode_analysis.md`, `scripts/analyze_failure_modes.py`,
`scripts/build_failure_dashboard.py` (19 inferred charts, 3,384 chords).

**Data-integrity finding (CLAUDE.md #1/#2 premise-screen paid off).** The
`irealb_<slug>.html` GT charts and the `inferred_<slug>.html` charts are on
**different timelines**, so no valid GT-anchored root/quality accuracy can be
computed from these artifacts. Evidence: autumn_leaves GT span 160s vs inferred
422s (2.64×); Let It Be 10.3s vs 243s (23.6×). Two alignment strategies
contradict each other — absolute-time overlap → ~0.11 root-acc (chance, from the
mismatch); free NW sequence alignment → 1.0 (cherry-picks 1 of ~10× more
inferred segments). A naive time-overlap comparison would have reported "11% root
accuracy, model is broken" — a fabricated number. **Fix before any accuracy
panel: export inference + iReal GT on a shared audio clock (same DTW pass).**

**What IS measurable (model self-signal, no GT):** each inferred chord's `sug`
(ranked root/quality rivals) + `lv.exact.c`. Findings:
1. **Perfect-4th/5th dominates root ambiguity — 33.8% of all root competitor
   mass** (+5 17.3%, +7 16.4%; uniform baseline ~17% for 2 of 12 intervals).
   Direction balanced (118 up-a-fifth, 109 up-a-fourth). This quantifies the
   long-suspected 5th-apart acoustic confusion (cf. #5 template geometry) from
   the model's own posterior. Suggested lever: bass/transition tiebreak applied
   only when top-2 roots are a fifth apart within a small margin.
2. **Two-domain confidence split, not gradual degradation** — 4 songs
   (blue_bossa, blue_bossa_150bpm, adele_hello, muppets_kermit) are ~globally
   uncertain (mean conf ≈ 0.17–0.22, ≥99% chords <0.4) while the other 15 sit at
   mean conf ≈ 0.75–0.9. Consistent with the real-audio calibration work (#26,
   Mission 4). Inspect which domain/calibrator each low-conf song hits.
3. **Quality collapses to maj/dom** (41%/28%/22% maj/dom/min; ø+dim ~6% combined)
   — a ø/dim reading as its relative maj/dom is a systematic quality error GT
   eval would catch.

Secondary: in some chords the decoded `root` diverges from its own `sug` argmax
by a fifth (chosen-root posterior ≈0 while a fifth rival dominates) — HMM
prior/transition overriding weak acoustics; worth a separate look.

## 15. accomp_db regen (fixed vary_voicings) blocked by full disk — OPEN 2026-07-08

The `vary_voicings` fix (issue #13, committed 2026-07-07) corrects the function in code, but
the existing `data/cache/accomp_varied/` (347MB) was built with the old bug (pitch-class
omission). The `--fold --vary` re-evaluation and fold-robust model retraining require a fresh
regen.

**Blocked:** disk is at 100% (`df -h` shows 259Mi free on 228Gi total). The regen (`build_accomp_audio_hard.py --vary-voicings`) would fail mid-run.

**To unblock:** free ≥500MB on the data volume. The safest cleanup targets:
- `data/cache/accomp_varied/*.npz` (347MB, stale — safe to delete, will be rebuilt)
- `data/cache/accomp_blind/` (391MB, used for blind-eval scripts — check if still needed)
- Any WAV files under `data/accomp_db/` (should have been cleaned up after BP extraction)

After cleanup, regenerate: `.venv/bin/python scripts/build_accomp_audio_hard.py --n-songs 60 --vary-voicings`

---

## Resolved (session 4, 2026-07-01)

- NO_CHORD absorbing-state collapse in `build_transition_matrix` (Viterbi
  predicted "N" for 100% of a song).
- Zero-duration / cross-segment-overlapping chord events in `_compress_path`.
- Confidence always exactly 0.0 (cumulative-log-prob underflow).
- `_label_to_mireval` suffix-ordering bug crashing evaluation on min-maj7/
  dim7/half-dim7 chords (3 of 5 test songs).
- `evaluate_pop909` beat-index-vs-seconds bug.
- `PitchActivations.chroma()` used `note_probs` (near-constant sustain signal)
  instead of `onset_probs` — only affected the cosmetic `global_key` field.
