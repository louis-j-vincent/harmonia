# Building Harmonia: a coin flip wearing a lab coat (Part 4)

*Fourth in a series on building a Bayesian chord-recognition system for solo jazz
piano. [Part 3](03-periodicity-and-the-real-bottleneck.md) closed out the issue #1
investigation — three fixes, three well-characterized failures, all converging on
"the emission model can't discriminate similar chord qualities." While chasing why
one of those fixes helped four songs and hurt a fifth, I found something more
foundational underneath all of it.*

## The detour that mattered more than the destination

I was trying to understand why `key_prior_per_beat` — a fix that applies the
key-conditioned diatonic-quality prior at every beat instead of just the first beat
of a segment — helped songs 002 through 005 and quietly wrecked song 001. My first
guess was that song 001's ground-truth harmony must include something non-diatonic
to its key, F# major, that the prior was fighting. I checked. It didn't. Every chord
in song 001's repeating progression — I, V, iii, vi — is exactly diatonic. Wrong
hypothesis, cleanly ruled out, which meant the real explanation was somewhere else.

So I went one level down: was the *detected* key for song 001 actually right, every
time, for every segment? I printed it out. All sixteen structural segments resolved
to "F# major" — with **bit-for-bit identical confidence, 0.043**, regardless of
whether the segment was 11 beats or 35. `0.043 ≈ 1/24`, which is what you'd get from
a perfectly uniform distribution over all 24 candidate keys. The chroma was picking
the right key, every time. It just had no idea it was right.

## A correlation score wearing a log-likelihood's coat

The bug, once I found it, was almost embarrassingly clean:

```python
log_likelihood = KEY_PROFILES @ chroma_norm
```

Both `KEY_PROFILES` (the Krumhansl-Schmuckler key-profile rows) and `chroma_norm`
(the input chroma) are L1-normalized — each sums to 1, so each is a genuine
probability distribution over 12 pitch classes. Their dot product is a *convex
combination*, mathematically bounded to whatever range the profile itself spans —
roughly 0.06 to 0.16 for the major-key profile. That bounded number was then handed
straight to `exp()` and treated as if it were a log-likelihood. `exp(0.16) /
exp(0.06) ≈ 1.10`. A hard ceiling of about 10% relative concentration between the
best- and worst-fitting key, baked into the arithmetic, regardless of how clean or
ambiguous the actual input was. There was a second bug riding along with it — a
Dirichlet-style "more evidence should mean more confidence" term that had been
silently neutralized, because the chroma it was measuring had already been
normalized to sum to 1 by the caller before it ever arrived. The exact information
the term needed had been thrown away one function call earlier.

This is the kind of bug that's dangerous precisely because it isn't loud. The model
was still picking the right key. It just couldn't tell you that, or ever be
*confident* about it, which meant every downstream consumer of that confidence value
— modulation detection, in this codebase — was built on a number that looked like
probability but functioned like noise.

The fix matches what a Bayesian treatment of this problem should have been from the
start: score the *raw, unnormalized* chroma counts against each key profile as a
proper multinomial log-likelihood, `sum_i chroma_raw[i] * log(profile_k[i])`. This is
additive in the evidence by construction — more observed signal naturally produces a
more separated posterior, no ad hoc temperature constant required, and it fixes the
neutralized-Dirichlet-term bug for free, since it never throws away magnitude in the
first place.

I wrote the tests before touching the implementation, on purpose — a set of
calibration tests that should obviously hold if the fix were correct (confidence
concentrates given strong evidence, confidence increases with more evidence at fixed
shape, confidence for the exact song-001 case exceeds 0.3 instead of sitting at
0.043). Run against the old code, three of them failed immediately — that failure
*was* the bug, made concrete instead of just described. Then I made them pass.

## Fixing the ceiling exposed a floor

Here's the part I didn't expect. The moment raw chroma magnitude started flowing
through, confidence didn't land somewhere reasonable — it slammed to bit-exact `1.0`
for almost every real segment, including the shortest one in the whole song.
Printing the actual numbers made it obvious why: a typical segment's raw chroma sums
to somewhere between 180 and 1000. Feed that much "evidence" into a multinomial
log-likelihood against two competing profiles that differ at all, and the log-odds
gap between best and second-best key blows out to dozens or hundreds of nats. `exp()`
of a negative number that large underflows to zero before it ever reaches a
denominator. The posterior wasn't confident. It was numerically saturated — the same
failure as before, just at the opposite end of the number line. A confidence field
that's always exactly 0.043 and a confidence field that's always exactly 1.0 are
equally useless; neither one is telling you anything you didn't already know from
the argmax.

The actual problem was a modeling one, not an arithmetic one this time: summed raw
activation-probability magnitude across a beat isn't a genuine independent-trial
count. Multiple pitch classes co-sound within a single beat — that's just what music
is — so treating "how loud were all 88 keys, summed, across 35 beats" as if it were
"how many independent categorical draws did I observe" inflates the effective sample
size by whatever factor of polyphony happens to be present, which has nothing to do
with how much *real* evidence about the key actually exists.

The fix follows directly from stating the problem that way: treat each beat as
exactly one unit of evidence. L1-normalize each beat's chroma internally — so its
own polyphony doesn't inflate its total weight — then sum across beats. Now a
segment's raw magnitude is proportional to genuinely interpretable evidence:
roughly the number of beats with real signal in them, not however many notes
happened to be sounding. Re-running the same song-001 segments after this fix,
confidence spread from 0.30 (an 11-beat segment) to 0.92 (the 35-beat one) — sixteen
distinct values, correctly ordered by how much evidence each segment actually had,
still picking F# major every single time.

## Does it generalize, or did I just get lucky twice on the same song?

This was the part of the session I was most insistent on not skipping. Song 001 had
already burned me once — a hypothesis that looked right ("non-diatonic harmony") and
turned out to be completely wrong once checked directly. I wasn't going to trust
"this fixes song 001's numbers" as evidence the fix was actually *correct*.

POP909 ships a `key_audio.txt` ground-truth file per song — real annotated key
labels, sitting unused in the dataset the entire time I'd been working with it. I
wired up a small parser for it and validated the fix in the order the methodology
demanded: synthetic unambiguous cases first (a pure C-major triad held for 32 beats
should infer C major confidently — it does), then one real segment, then whole-song
consistency, then generalization across all five available songs against real
ground truth.

Global key came back correct for 4 of 5 songs. The miss — song 004, ground truth Eb
minor, inferred F# major — isn't a new bug. It's a textbook confusion: Eb minor and
Gb/F# major are relative keys, sharing the exact same seven diatonic pitch classes,
differing only in which one you'd call "home." Pure Krumhansl-Schmuckler profile
matching is well known to struggle exactly here, and tellingly, the model's own
confidence on song 004 was honest about it — mean confidence 0.348, versus 0.541 on
song 001, where every segment was unambiguously right. That's what calibration is
*for*: not being right 100% of the time, but knowing, in a way you can actually read
off the number, when you're on shakier ground.

## The thing I was most curious about: did any of this explain the original mystery?

I still owed an answer to the question that started all of this — does
`key_prior_per_beat`'s song-001 regression go away once key inference is properly
calibrated? I re-ran the exact same A/B comparison from the earlier session.

It didn't move. Root accuracy still drops from 33.3% to 22.6% on song 001 when the
per-beat key prior is switched on; majmin still drops from 34.0% to 21.9%. Almost
identical to the original numbers.

In hindsight this makes complete sense, and I probably should have seen it coming:
song 001's key was already being identified correctly — tonic and mode both right —
even under the badly miscalibrated confidence. The only thing that changed this
session was *how confident the model was allowed to say it was*, and the code that
consumes key inference downstream (`build_key_prior`, which builds the
diatonic-quality boost fed into the chord decoder) never once looks at the
confidence field. It only ever asks for the MAP tonic and mode. Fixing a number that
nothing downstream reads can't move a metric that depends on a different number
entirely. The two bugs shared an investigation but never shared a mechanism — issue
#0 was real, worth fixing, and independent of issue #1's still-unexplained
regression.

## Where this leaves things

Three real, load-bearing findings from one detour: a genuine calibration bug fixed
and validated against ground truth for the first time in this project, a subtler
second bug (evidence inflation) that only became visible once the first one was
gone, and a negative result — the song-001 mystery is still open, but it's now
definitively *not* explained by key-inference calibration, which narrows the search
meaningfully instead of leaving it as a loose end wrapped around the wrong culprit.

None of this touches issue #1's actual diagnosis — emission evidence still can't
reliably discriminate chord qualities that share most of their template — but that
was never really the point of this detour. The point was: don't build the next
layer on a foundation you haven't actually checked. `key_prior_per_beat` is the
second fix in this project's history to have looked directly downstream of a bug
this fundamental. I'd rather find the third one this way — by validating the
foundation first — than by chasing another A/B/C sweep that quietly inherits it.

*Next: either a narrowly-scoped look at what's actually driving song 001's
`key_prior_per_beat` regression now that key inference is ruled out as the cause, or
back to issue #1's emission-template-geometry question — whichever turns out to be
more tractable.*
