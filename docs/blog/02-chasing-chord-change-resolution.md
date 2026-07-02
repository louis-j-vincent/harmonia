# Building Harmonia: chasing chord-change resolution (Part 2)

*Second in a series on building a Bayesian chord-recognition system for solo jazz
piano. [Part 1](01-origin-story.md) covered four bugs that took root accuracy from
1.4% to the 18–31% range. This one is about what's left, and why two well-motivated
fixes both failed in the same instructive way.*

## The shape of the remaining error

Part 1 ended on a specific, measurable symptom: ground-truth chords in POP909 change
roughly every two beats. My predicted chords were lasting fifteen to thirty-five
beats on average. Root and major/minor accuracy were moderate; seventh-chord and
full-tetrad accuracy were near zero. The model was getting the coarse harmonic
neighborhood right sometimes, but it wasn't tracking real chord-to-chord motion at
all.

I had three hypotheses, and I wanted each one tested in isolation before combining
anything — end-to-end weighted accuracy conflates too many stages to tell you *why*
a change helped or hurt.

- **A — emission signal quality.** The beat-level observation vector is never
  loudness-normalized before being compared against chord templates. A dominant,
  recurring bass note could plausibly swamp the softer inner voices that actually
  distinguish one chord from another.
- **B — the duration model is memoryless.** The only thing encoding "how long
  should a chord last" is a constant added to the transition matrix's diagonal,
  which implies a *geometric* distribution — constant hazard of switching, no
  matter how long you've already stayed on a chord. Real harmonic rhythm clusters
  around a typical duration; it isn't memoryless.
- **C — song structure goes unused.** Repeated verse/chorus vamps aren't exploited
  at all. If an eight-bar loop repeats, overlaying the repeats should improve the
  signal-to-noise ratio of "what's actually happening at this position in the loop."

I built a small harness with metrics chosen to isolate each hypothesis rather than
just averaging everything together: per-beat emission argmax accuracy (bypasses the
decoder entirely — pure evidence-quality check) for A, chord-boundary F-score
(are the changes happening at the right *times*, independent of whether the label is
right) for B, and MIREX weighted accuracy as the umbrella sanity check for all three.

## A: a fix that is mathematically incapable of working

I tried three variants under candidate A: L1-normalizing the beat vector, an
adaptive per-song onset threshold, and nonlinear compression (`sqrt`, `log1p`) of
the observation before the dot product.

The normalization result was flat — not "small effect," *exactly* identical output,
down to the byte, on every song. That's not noise. It's provable: the Viterbi
recursion is `viterbi[t,c] = max_i(viterbi[t-1,i] + transition[i,c]) + emission[t,c]`.
L1-normalizing a beat's observation vector subtracts the *same constant* from
`emission[t,c]` for every candidate chord `c` at that beat. A uniform shift across
every option at a given timestep can never change which option wins — not at that
step, and not for the rest of the decoded path, since the shift just propagates
forward as a constant offset. I'd built a fix for the right *intuition* (loud bass
notes dominating) using the wrong *operationalization* — a transform that was
mathematically guaranteed to be inert before I ever ran it. I wrote a test that
proves this rather than just asserting it, which felt like the right way to close
the question permanently instead of half-remembering it later.

Compression was the interesting one: it's a genuinely non-inert, per-element
transform, and it *did* improve the isolated per-beat metric — 16.8% up to 17.8%
with `sqrt`. Then it made the full pipeline *worse* (boundary F dropped from 0.215
to 0.167). My best explanation: compression shrinks the emission term's dynamic
range across candidate chords, and since Viterbi sums `emission + transition +
init`, shrinking one term just hands relatively more influence to the others —
including the still-oversized self-transition boost from bug #1's era. Sharpening
the evidence backfired because something else was still drowning it out.

None of candidate A's variants got adopted. But the compression result was a real
clue: **fixing the emission signal in isolation can't show up downstream while the
duration prior is still overriding it.** Time for B.

## B: proving the duration model wrong, then still losing

POP909's chord annotations, at beat-level resolution, are ground truth for exactly
the question I needed answered: how long does a chord actually last? I fit an
empirical distribution from all 909 songs' annotations — no audio needed, just the
text files, so this cost about a second of parsing for ~120,000 labeled chord
events.

The result was the clean kind of confirmation you don't often get in this line of
work: `P(duration = 2 beats) = 49.2%`, higher than `P(duration = 1 beat) = 15.0%`.
A geometric distribution — which is what a constant self-transition probability
implies, no matter what value you tune it to — is *always* maximized at its minimum
value. It cannot have an interior peak. So this wasn't "the geometric model is
probably a bad fit," it was direct proof the true distribution has a shape the
current architecture cannot express at any setting.

That justified building a proper explicit-duration decoder — a segmental Viterbi
that jointly picks segment boundaries and chord labels using the fitted duration
distribution instead of inferring persistence from a self-transition weight. The
textbook version of this (a Hidden Semi-Markov Model) forbids the same chord from
appearing in two consecutive segments, since persistence is supposed to be entirely
the duration model's job now. I implemented it that way first, and it broke in a
way I found genuinely interesting: whenever a stretch of audio was genuinely longer
than the duration model's cap, the decoder was forced to fake a change into *some*
other chord just to keep decoding — and it usually picked a near-duplicate quality
of the same chord (`C#sus4` → `C#7sus4` → `C#7sus4`, three "different" labels in a
row that are really the same harmony wearing slightly different clothes). Forbidding
self-transitions had turned "there's nothing more to say here" into "say something
anyway."

I fixed that — allowing the same chord to legitimately span multiple chained
segments as an escape valve for long stable regions — and it made zero difference on
real audio. The decoder kept alternating between near-duplicate qualities even when
explicitly allowed not to, because the *emission evidence itself* was marginally
rewarding the alternation. I swept the blend between the old boost and the new
duration prior from pure-duration to heavily-boosted, and every single configuration
made `majmin` worse than doing nothing — from 17.0% down to roughly 10%, consistently,
regardless of blend. Root accuracy stayed roughly flat throughout.

That's the tell. Forcing chord boundaries to land at the statistically correct
*times* (which the duration model genuinely does better) doesn't help if the
decoder can't reliably tell `sus4` from `7sus4` from `dom7` once it gets there —
templates for those qualities share most of their notes. More frequent, more
accurately-timed decisions just create more opportunities to get the *quality*
wrong, and quality is exactly what `majmin` and `tetrads` measure. **A and B failed
for the same underlying reason, from two completely different directions**: the
bottleneck isn't decoder structure, it's how discriminable the acoustic evidence is
in the first place.

## An accidental find: the soundfont was lying about its own name

Before starting on candidate C — which is the one hypothesis that actually targets
evidence *quality* rather than decoder logic — I went back to a much cruder
suspicion I'd been sitting on: the soundfont used to render POP909's MIDI to audio
was small, 307KB, plausibly low-fidelity. I ran `strings` on the file to check its
embedded metadata before bothering to source a replacement, mostly out of habit.

It came back `"Vintage Dreams Waves v 2.0"` — a completely different, much smaller
1996-era soundfont than the file's own name (`GeneralUser.sf2`) claimed to be. Every
render used throughout this whole investigation had quietly been synthesized with
the wrong instrument the entire time.

I downloaded a real high-quality GM soundfont (MuseScore's own, 215MB, a solid
substitute for the intended GeneralUser GS) and re-rendered all five songs with
identical settings, changing only the soundfont. The result was a genuinely
different pattern from A and B: boundary F-score improved from 0.215 to 0.241 — the
best improvement any experiment in this whole investigation has produced — and the
raw per-beat evidence-quality metric improved slightly too. Real signal that a
better instrument genuinely helps Basic Pitch transcribe more accurately. But root
and major/minor accuracy stayed flat to slightly worse, the same story as
everywhere else: getting *when* right doesn't automatically buy you *what*.

One loose thread worth naming honestly rather than quietly ignoring: one song's
detected beat count changed by almost exactly 2x between the two soundfont renders
of the *identical* MIDI file. Different attack and reverb characteristics apparently
pushed the beat tracker into a different tempo octave for that one song. Soundfont
choice doesn't just affect transcription quality — it can silently change the beat
grid you're measuring everything else against, which is a confound worth
remembering the next time a comparison looks cleaner than it should.

## Where that leaves things

Two candidates tested, two rejected, converging on the same diagnosis from
different angles, plus a genuinely embarrassing but easy-to-fix data bug found along
the way. Not the outcome I was hoping to report, but a coherent one — and coherent
negative results are worth exactly as much as positive ones if you're trying to
actually understand a system instead of just poking it until the number goes up.

*Next: candidate C, periodicity folding — the one idea in this batch that tries to
improve the evidence itself instead of reshaping how existing evidence gets used,
using the self-similarity matrix the segmentation stage already builds to find and
exploit repeated song structure.*
