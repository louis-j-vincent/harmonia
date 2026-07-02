# Building Harmonia: teaching a machine to hear jazz changes (Part 1)

*First in a series on building a Bayesian chord-recognition system for solo jazz
piano — from raw audio to a Viterbi decoder that (mostly) knows a ii–V–I when it
hears one.*

## Why

I'm a jazz pianist with an ML PhD, which is a dangerous combination, because at
some point you inevitably ask: *can I build something that transcribes my own
playing into chord symbols?* Not "detect C major vs A minor" — real jazz voicings,
rootless left-hand shells, chromatic passing chords, the whole vocabulary.

There's no shortage of chord-recognition papers and tools. Almost none of them are
built for solo piano, and almost none of them treat the problem the way a musician
actually thinks about it: as a *hierarchy* of decisions — what key are we in, what's
the harmonic rhythm, what's the bass doing, what's the actual voicing above it —
where each level constrains but doesn't dictate the ones below it. That soft
hierarchy is the design principle I keep coming back to across this whole project:
**priors should regularize, never override.** If the acoustic evidence strongly
supports an out-of-key chord, the data should win. Priors resolve ambiguity; they
don't censor it.

So: Harmonia. Named for the thing it's trying to recognize.

## The pipeline, in one breath

Audio goes in, a stream of chord symbols with timestamps comes out, via five stages:

1. **Pitch extraction** — Spotify's [Basic Pitch](https://github.com/spotify/basic-pitch)
   (ONNX backend) turns raw audio into an `(frames × 88 keys)` tensor of onset and
   note-sustain probabilities. This is the only deep-learned component; everything
   downstream is explicit probabilistic modeling.
2. **Rhythm** — a beat tracker (librosa) builds a beat grid; frame-level pitch
   activations get pooled into beat-level observation vectors.
3. **Structure** — a self-similarity matrix with a checkerboard kernel segments the
   piece into structurally coherent sections (verse/chorus-scale, not chord-scale).
4. **Key inference** — Krumhansl-Schmuckler profile correlation, run per segment, so
   key can shift across the piece instead of being fixed globally.
5. **Chord decoding** — a Bayesian HMM: emission probabilities from the beat-level
   pitch vectors, a key-conditioned prior favoring diatonic chords, and a
   jazz-progression-aware transition matrix (ii–V–I and friends get a bump). Viterbi
   decodes the most likely chord sequence.

Every stage after Basic Pitch is a probability model I designed by hand, which is
the whole point — I want to *understand* why the system says what it says, and be
able to argue with it in the same language I'd argue with a music theory student.

## Where it actually stood: evaluating against ground truth

Design intuitions are worth nothing until you measure against real chord labels. I
used [POP909](https://github.com/music-x-lab/POP909-Dataset) — 909 pop songs with
hand-annotated chord charts — rendered to audio via FluidSynth, and scored
predictions against ground truth with `mir_eval`'s MIREX weighted-overlap metrics
(root, major/minor, sevenths, full tetrad accuracy).

The first time I ran the full pipeline end-to-end, root accuracy was hovering around
**1.4%**. Not "needs tuning" bad — *structurally broken* bad. Something was
swallowing the whole prediction.

This is where the project stopped being architecture and became debugging, and
honestly, debugging a probabilistic model is its own particular flavor of fun: the
bug isn't a stack trace, it's a *wrong belief*, hiding inside a matrix of numbers
that all look individually plausible.

## Bug #1: the absorbing state

Turning on some diagnostics showed the Viterbi path was predicting `NO_CHORD` for
essentially 100% of every song, regardless of what was actually being played. That
smelled like a transition-matrix problem, not an emission problem — something was
making "stay on no-chord" pathologically attractive.

It was. `build_transition_matrix` builds each chord's row by layering a jazz
progression prior on top of a flat baseline, then boosts every diagonal
(self-transition) unconditionally to encourage harmonic stability — chords tend to
last more than one beat. But the loop that applies the *progression* weighting
skipped the `NO_CHORD` row entirely (there's no meaningful "progression" going out
of silence). So real chords' rows got inflated by the progression prior on top of
the self-transition boost, while NO_CHORD kept its bare baseline — except NO_CHORD's
diagonal still got the unconditional self-transition boost.

Net effect: `P(N → N) ≈ 53%` while `P(chord → chord) ≈ 4%`. NO_CHORD became a black
hole. Once the decoder touched it — and with those odds, it touched it almost
immediately — it could never leave. I confirmed the emission evidence alone (no
transition matrix at all) ranked the true chord far above N at essentially every
beat; the acoustic evidence was fine. The prior was just steamrolling it, which is
exactly the failure mode the "priors regularize, never override" principle exists to
prevent — and exactly what happens when you implement that principle inconsistently
across one row of a matrix.

**Fix:** give NO_CHORD the same flat baseline treatment as real chords for their
N-transitions, and its own separately tunable self-transition boost (default 0.5,
versus 2.0 for real chords) — both now exposed as pipeline parameters rather than
buried constants.

## Bug #2: zero-duration chords

With the absorbing state fixed, some songs still errored out downstream: chord
events with `start_time == end_time`. A Viterbi run that happened to end exactly on
a structural segment's last beat was clamping its end time to that last valid beat
instead of extending to the true segment boundary — and my first attempted fix
(just extrapolate forward) overshot and started *overlapping the next segment's
first event* instead. The actual fix needed the *true* next-beat time from the
full, un-sliced beat grid, and only extrapolation for the genuinely final segment of
the whole track, where no such boundary exists. Two off-by-one-feeling bugs wearing
one trenchcoat.

## Bug #3: a confidence score that was always exactly zero

Every predicted chord reported `confidence = 0.0`. Every single one. That's a
suspiciously round number for something computed from real audio.

The original computation was `exp(mean of cumulative Viterbi log-probabilities))`.
Cumulative log-probabilities are a running *sum* over the whole decoded run — they
grow more negative without bound as the run gets longer. Averaging them and
exponentiating still leaves you exponentiating something that underflows to zero for
any run longer than about five beats. The bug wasn't in the math, exactly — it was
in reusing a global path-probability quantity (useful for comparing whole
*sequences*) as if it were a local, per-decision confidence (useful for comparing
individual *chords*). Different question, same-looking formula.

**Fix:** confidence is now the mean per-beat softmax-normalized emission posterior
at the decoded state — bounded in `(0, 1]`, and it actually varies.

## Bug #4: `EmMaj7` breaks everything

Once predictions looked sane, evaluation itself started crashing — silently, on 3 of
the first 5 test songs, zeroing their score via a caught exception with no useful
message. The culprit: `_label_to_mireval`'s suffix-replacement logic checked for a
generic `"7"` substring *before* checking more specific suffixes like `"mMaj7"`,
`"°7"`, `"ø7"`. So `"EmMaj7"` got the generic rule applied first and came out the
other side as the syntactically invalid Harte chord string `"EmMaj:7"` — invalid
enough that `mir_eval` just threw.

This is the most boring bug of the four and also the most illustrative one: it's a
classic ordering bug in what amounts to a priority list of string patterns, and it
hid for however long because nothing exercised the min-maj7/dim7/half-dim7 code path
until real chord data did. **Fix:** parse root and quality explicitly against
Harmonia's own known vocabulary instead of pattern-matching suffixes, then map
through an explicit quality table.

## Where that left things

| song | events | root | maj/min | 7ths | tetrads |
|---|---|---|---|---|---|
| 001 | 14 | 31.1% | 32.1% | 2.8% | 2.7% |
| 002 | 61 | 21.0% | 11.5% | 10.4% | 10.0% |
| 003 | 20 | 22.8% | 19.1% | 0.7% | 0.6% |
| 004 | 28 | 18.0% | 15.1% | 10.5% | 9.7% |
| 005 | 68 | 20.5% | 7.3% | 4.4% | 4.8% |

Up from 1.4% (complete N-collapse) to root accuracy in the 18–31% range, with no
more crashes across any of the five songs. That is — let's be honest — still a
*long* way from good. A system that gets the root right one time in five isn't
transcribing anything usable yet. But going from "broken in a way that hides all
signal" to "working badly, in a way I can now actually measure and improve" is the
real milestone. You cannot optimize what you cannot see.

And the *shape* of the remaining error is itself informative, which is where the
next post picks up: root and major/minor accuracy are moderate, but seventh-chord
and full-tetrad accuracy are still near zero. The model gets the coarse harmonic
family right some of the time but isn't tracking real chord-to-chord motion — ground
truth chords change roughly every two beats, predicted chords last fifteen to
thirty-five beats on average. That's not a transition-matrix tuning problem either
(I checked — sweeping the self-transition boost across a 40x range barely moves the
event count). It smells like an upstream signal-quality problem in how beat-level
pitch evidence gets pooled, which means the next investigation starts back at Basic
Pitch's raw onset activations, not the HMM at all.

*Next: chasing chord-change resolution — why the model hears the right harmonic
neighborhood but can't track a ii–V that goes by in two beats.*
