# Building Harmonia: when the bottleneck moves (Part 9)

*Ninth in a series. [Part 8](08-hierarchies-all-the-way-down.md) left the system with
a calibrated chord tree but a lingering discomfort: it had never been tested honestly
end-to-end, from raw audio with no chart to help. This part is the honest test — and
the story of how the bottleneck I was sure I understood turned out to be somewhere
else entirely.*

## The plan, and the premise-check that broke it

The idea was clean and I believed it. A chord changes on a metrical grid — every 1,
2, or 4 beats — and that grid can differ per section (the A section might move every
two beats, the bridge every four). So: anchor on beat and structure, estimate the
harmonic-rhythm period per section, merge beats into blocks of that size, then ask at
each block boundary "same chord as before, or new?" It's how I'd do it by ear.

The one load-bearing assumption was that merging into blocks makes a real chord change
cleanly separable from a held chord. I checked it first, and it held beautifully:
the change-vs-hold signal goes from a coin-flip 0.64 at the single-beat level to a
near-perfect **0.96 at two-beat blocks**. Merging is the whole trick — it averages
out the noise that makes per-beat evidence useless.

Then I checked the *other* premise — the per-section period — across all 1136 songs,
and it evaporated. Chord changes land on *every* beat of the bar (39/29/20/13% on
beats 1–4), and within-section spacings of 1, 2, 3, and 4 beats are almost equally
common. There is no clean period to estimate; 92% of sections don't have one. The
elegant part of the plan was fitting noise. So I kept the merge (fixed at two beats,
the best single grid) and dropped the estimator. The coarse engine landed at a
change-detection F of 0.89.

## Four ways to zoom, all wrong

The coarse grid misses the ~40% of changes that fall between its blocks, so the plan
called for a "zoom" — go back into each segment at beat resolution and recover them.
I tried four ways. Naive beat-level novelty: worse. Per-track self-similarity (the
clever idea — split out bass and piano and look at each): defeated by *walking bass*,
which changes the bass note on 58% of beats regardless of the chord, so no per-track
cue beats the mixed signal. Top-down divisive splitting: the first cut of a
many-chord section has two muddy halves and under-segments. Pooled-halves boundary
snapping: moves boundaries *around* the true change, not onto it.

All four failed for one reason, and it's worth stating plainly: **you cannot have
beat resolution and a clean signal at the same time.** The evidence is ~0.65 at the
beat, and the only thing that cleans it up — pooling — is exactly what destroys the
resolution. That's not a bug to fix; it's the shape of the problem.

So I did the experiment that decides whether zooming is even worth it: I fed the
model *perfect* boundaries. Accuracy didn't move. Perfect segmentation gave the same
numbers as the coarse engine. The thing I'd spent days on — where the chords *change*
— was not the bottleneck at all.

## The bottleneck was the note, not the moment

With good boundaries and only ~40% majmin, the arithmetic was loud: the *labels* were
wrong. And they decomposed cleanly. Given the correct root, the family model is 94%.
So the root was the problem — and the root was being read off the loudest bass note,
which walking bass makes a liar 32% of the time.

A trained twelve-way root classifier, reading "root-ness" from the whole chroma
instead of the loudest low note, went to **93%** held-out. Wired in, majmin jumped
from 39.5% to 58.8%. Then a second, quieter culprit surfaced — the family features
were unnormalized sums whose magnitude scaled with segment length, so the model met
inputs it was never trained on. One line of normalization: **58.8% to 82.8%.** A
silent scale bug had been hiding twenty points the whole time.

That is the fourth time in this project a low-level calibration error masqueraded as a
hard modeling problem. I have a rule for it now. It still got me.

## The honest number

The last crutch was the beat grid — I'd been handing the model perfect beat times
from the score. Swapping in beats detected from the audio cost twenty majmin points,
which stopped me cold, because the beat tracker's F-measure was a healthy 0.87. The
tempo was right to within 2%; it was the *phase* that jittered, and pooling is
merciless about phase. Since this accompaniment audio is metronomic, imposing a
uniform grid at the detected tempo recovered almost all of it.

Then the number that actually counts. Root and family models trained only on
even-numbered songs, evaluated fully standalone — raw audio in, no chart, no beats,
no structure — on thirty odd-numbered songs: **root 74%, majmin 70%.** Up from 39.5%
where this session's labeling work began.

It is a smaller number than the 82% I got before I started removing the crutches one
by one, and that gap *is* the point. Every crutch I pulled — the oracle beats, the
oracle structure, the overlapping train set — took a few points with it, and what's
left is the part that's real.

## What I stopped believing

Structure detection, which I'd queued as the next big task, turned out to be both
unnecessary (removing it costs half a point) and impossible from harmony alone (every
method scores ~0.25 — jazz A and B sections share a key and a vocabulary; they differ
in *melody*, which I'm not looking at). Two findings, one decision: closed.

The tree from [Part 8](08-hierarchies-all-the-way-down.md) held up on real audio. Given
the root, the seventh is 88% and the exact chord 84%, and — the part I care about most
— the confidence gate is honest at every level. Ask for the seventh only when the model
is sure, and it's right 96% of the time on the 62% it's willing to answer. Confident
"Cmaj7," unsure "C major," never a confident lie.

The map I started with was wrong about where the hard part was. The premise-checks kept
saying so, one at a time, and the only thing to do was believe them.
