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
2026-07-02, not yet fixed) is likely bigger still — treat it as the current
top priority, ahead of resuming the issue #1 investigation.**

---

## 0. Key inference posterior is near-uniform / uncalibrated — OPEN, top priority

**Found while investigating why `key_prior_per_beat` (issue #1) helped 4/5
songs but hurt song 001.** Checked every structural segment's inferred key
for song 001 — all 16 segments resolved to "F# major", each with **bit-for-bit
identical confidence, 0.043** (`1/24 = 0.0417`, i.e. essentially uniform over
all 24 candidate keys). Printing the full posterior for one segment:

```
F# major     0.04296   <- picked (correctly, for this song)
C# major     0.04249
B major      0.04239
D# minor     0.04236
...
C major      0.04077   <- "worst" of all 24 keys
```

Spread between best and worst key: `0.04296` vs `0.04077` — a ~5% relative
difference, even though the underlying chroma is genuinely informative (C#
at 21% and F# at 26% of total chroma energy — the correct dominant pitch
classes, directly visible in
`docs/plots/inference/pop909_001/note_probs_vs_gt_C2_C5.png`'s chroma panel).
F# major wins by a margin of 0.00047 — a coin flip that happened to land
right, not a confident inference.

**Root cause, `harmonia/theory/key_profiles.py::infer_key()`:**
`log_likelihood = KEY_PROFILES @ chroma_norm` computes a correlation between
the (L1-normalized) chroma and the (L1-normalized) Krumhansl-Schmuckler
profile, and treats the result directly as a log-likelihood. Since both
inputs are probability distributions over 12 pitch classes, this dot product
is mathematically bounded to a narrow range (~0.06–0.16) *regardless of
input* — `exp(0.16)/exp(0.06) ≈ 1.1`, meaning the resulting posterior can
never be more than ~10% concentrated on any single key no matter how clean
or ambiguous the actual harmonic content is. There's a secondary,
compounding bug: the code has a Dirichlet-style term
(`1 + alpha * total / 12`) apparently meant to sharpen the posterior when
there's more acoustic evidence in a segment, but `total = chroma.sum()`
arrives pre-normalized to ~1.0 by the caller (`structure.py::_make_segment`
already L1-normalizes `chroma` before `infer_key` ever sees it), so that
term is neutralized to a near-constant ~1.08 regardless of how much real
evidence a given segment actually has.

**Why this matters more than issue #1:** every downstream use of key
(diatonic-quality prior for chord decoding, modulation detection via
`detect_modulations`'s `confidence >= 0.6` threshold — which can now
*never* fire, since confidence tops out around 0.05) is built on a key
estimate that's essentially arbitrary. It happened to land correctly for
song 001, by a hair, but there's no principled reason to expect that holds
for other songs, and the reported `confidence` field is meaningless
everywhere it's used.

**Fix in progress (next session, delegated — see handoff prompt in
project memory / chat log 2026-07-02):** replace the bounded-correlation
pseudo-log-likelihood with a proper multinomial (or Dirichlet-multinomial)
log-likelihood — treat each KS profile as a probability distribution over
pitch classes for that key, and compute the log-likelihood of the
*unnormalized* chroma (pseudo-)counts under it
(`sum_i chroma_raw[i] * log(profile[i])`), which naturally scales with the
amount of evidence rather than needing an ad hoc temperature constant, and
fixes the `total`-neutralization bug for free since it operates on raw
counts throughout.

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
