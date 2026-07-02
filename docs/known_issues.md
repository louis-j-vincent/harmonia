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

---

## 1. Chord-change temporal resolution is far coarser than reality — OPEN, actively being worked

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

---

## 2. Soundfont quality (VintageDreams, ~307KB) — OPEN, untested

All POP909 renders are synthesized with a small, low-fidelity soundfont.
GeneralUser GS (~30MB) likely has more realistic attack/decay characteristics,
which Basic Pitch (trained on real acoustic recordings) may transcribe more
reliably — particularly onset salience of inner voices, which is directly
relevant to issue #1. Not yet A/B tested.

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
