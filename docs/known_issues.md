# Harmonia — Known Issues

Living tracker of known limitations in the current pipeline, ordered by how much
each is currently limiting end-to-end accuracy. Distinct from `architecture_extensions.md`
(forward-looking design ideas) and `suggestions.md` (specific stage-1/stage-5
improvement proposals) — this file is "what's actually wrong right now."

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
