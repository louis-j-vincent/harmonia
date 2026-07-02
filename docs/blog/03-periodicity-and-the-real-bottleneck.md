# Building Harmonia: periodicity, a mislabeled soundfont, and the real bottleneck (Part 3)

*Third in a series on building a Bayesian chord-recognition system for solo jazz
piano. [Part 2](02-chasing-chord-change-resolution.md) covered why two well-motivated
fixes — emission preprocessing and explicit-duration decoding — both failed in the
same instructive way. This one closes out the investigation: a third fix, an
embarrassing data bug, and what all of it adds up to.*

## An unglamorous detour first

Before touching candidate C, I went back to a suspicion I'd been carrying since Part
1: the soundfont used to render POP909's MIDI files to audio was small — 307KB,
plausibly a toy. I ran `strings` on the file before bothering to source a
replacement, mostly out of habit rather than expectation.

It came back `"Vintage Dreams Waves v 2.0"`, a completely different 1996-era
soundfont than what the filename (`GeneralUser.sf2`) claimed to contain. Every
render used throughout this entire investigation had quietly been synthesized with
the wrong instrument. Not a subtle bug — the file was just mislabeled, sitting there
the whole time.

I downloaded a real high-quality GM soundfont and re-rendered all five songs with
identical settings, changing only the instrument. The result was informative in a
way that fit the emerging pattern rather than breaking it: boundary F-score improved
from 0.215 to 0.241 — the best single improvement of anything I'd tried — and the
raw per-beat evidence-quality metric improved slightly too. Real signal that better
audio genuinely helps transcription. But root and major/minor accuracy stayed flat
to slightly worse, same story as everywhere else: getting *when* a chord changes
right doesn't automatically buy you *what* it changed to. One more thing worth
naming honestly: one song's detected beat count changed by almost exactly 2x between
the two soundfont renders of the *identical* MIDI file — different attack and reverb
characteristics apparently pushed the beat tracker into a different tempo octave.
Soundfont choice doesn't just affect transcription, it can silently move the ruler
you're measuring everything else against.

## Candidate C: does the song repeat itself?

The third hypothesis was different in kind from the first two. A and B both tried to
extract more out of the *existing* per-beat evidence — better preprocessing, better
decoding logic. Candidate C tried to get *more* evidence: if an eight-bar
accompaniment loop repeats four times over a verse, averaging beat `t` with beat
`t + 32`, `t + 64`, `t + 96` should improve the signal-to-noise ratio of "what's
actually happening at this position in the loop," since whatever's just noise or a
passing tone should differ between repeats while the real harmony shouldn't.

First I wanted to know if the premise was even true, before writing any decoding
logic. The self-similarity matrix the segmentation stage already builds for its own
purposes turns out to answer this almost for free: average the matrix along its
L-th off-diagonal — every pair of beats exactly `L` apart — for a handful of
musically sensible candidate lags (bar multiples: 4, 8, 16, 32 beats), and you get a
clean autocorrelation-style profile. Song 001 produced a genuinely striking result: a
sharp peak at L=32 beats, score 0.82, with harmonics at 16 and 64 beats confirming it
was real structure and not noise. Real repeated structure, unambiguously present,
easy to find.

So I built it: period detection constrained to musically plausible lags, a folding
function that circular-averages each beat with its same-slot repeats, and wired the
folded views into the decoder as additional weighted evidence — an ensemble on top
of the raw per-beat signal, not a replacement, matching the "priors regularize,
never override" principle I keep coming back to in this project.

Song 001 — the one with by far the cleanest periodicity signal — regressed the
*most*. Major/minor accuracy dropped from 32.7% to 15.3% at full blend weight. That's
backwards from what the hypothesis predicted, and it's the most useful single data
point to come out of candidate C. My best explanation: high self-similarity in a
chroma-only comparison is necessary but not sufficient evidence that the *harmony*
repeats identically — it's equally well satisfied by "the rhythm and instrumentation
repeat, and the harmony is merely correlated with them." Real songs reharmonize
between repeats of a section constantly — a second verse changes a chord or two even
when the groove underneath is untouched. Averaging two genuinely different chords
together at the same slot doesn't produce a cleaner version of either one; it
produces a blur that's evidence for neither, and that blur damages exactly the
quality-sensitive metrics, for exactly the same reason candidate B did.

I swept the blend weight from barely-there to full strength, hoping a lighter touch
would capture some of the benefit without the damage. At low weights it was a wash —
every metric within noise of doing nothing. At full weight, the damage showed up.
Never better than baseline, at any setting.

## The whole picture

Three structurally different fixes, each properly implemented, each tested in
isolation with a metric chosen to match its specific hypothesis, each validated
across all five songs. None of them moved the needle. And they all failed for
recognizably the same reason, from three different directions:

- **A** made the emission signal sharper, and that sharpness got drowned out by
  everything downstream still expecting the old, blunter signal.
- **B** made the decoder place chord boundaries at statistically correct times, and
  every new boundary was one more opportunity to guess the wrong quality.
- **C** gave the decoder more evidence per beat, and averaging that evidence across
  repeats blurred exactly the distinctions that mattered.

None of these are failures of effort or of technique — the duration prior really is
provably the wrong shape (a geometric distribution can't have an interior peak, and
the real data does), the periodicity really is there in the self-similarity matrix,
the soundfont really was mislabeled and really was hurting things. Every diagnosis
was correct. None of the fixes helped, because none of them touch the actual
constraint: **the per-beat emission model cannot reliably tell apart chord qualities
that share most of their template** — a sus4 triad and a dominant-7-sus4 differ by
one note. No amount of deciding *when* to trust that signal, or *how long* to trust
it, or *how many repeats* to average it over, changes how discriminating the signal
is in the first place.

That's not where I expected three weeks of work to land, but it's a more useful
place to be than "root accuracy is 22%, not sure why." I know what's not the
bottleneck now, with reasonably strong evidence for each exclusion. What's left
points upstream: at the emission model's own chord templates (are sus4 and 7sus4
built too similarly to ever be separable, independent of audio quality at all?), and
further upstream still, at Basic Pitch's raw output itself, which I haven't touched
in this entire investigation.

*Next: either the emission templates directly, or back to Basic Pitch's raw onset
activations — whichever turns out to be the more tractable place to actually move
this number.*
