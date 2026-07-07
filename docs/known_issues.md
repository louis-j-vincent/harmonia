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
  strongest decode can correct the weakest. Zero headroom on clean MMA (family
  already ~100%), but should pay on real-audio corpora where family drops.
- **Structural prior for the HMM** — knowing "this bar is probably another ii-V"
  is free evidence that the current HMM doesn't use.

**What it doesn't solve:** section-level folding (recovering AABA labels from chords
alone) tops out at meanARI ~0.49 on GT chords. The human's section labels don't
correspond to clean chord-repeat units — a 16-bar "A" section typically has two
different 8-bar halves. Melody/timbre boundaries needed for that.

**Files:** `harmonia/models/motif.py`, `harmonia/models/block_fold.py`,
`scripts/demo_motif.py`, `scripts/render_motif_chart.py`.

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
