# 18 — Duration was the lever

Mission 1 spent three careful attempts trying to put a *grammar* factor into the
joint chord decoder — a key-local bigram, an encoder shallow-fusion, a
density-ratio version of the same — and every one of them had its optimum at
"weight zero." The diagnosis that fell out was uncomfortable but sharp: on the
segment decoder the grammar slot is saturated, and the residual root errors are
*acoustic* (a bass playing the 5th, so a whole span reads a fifth off), not
grammatical. A chord-quality language model can't fix a root that the ear itself
hears wrong.

Mission 1 pointed at the one lever it hadn't pulled: **duration**. Not "what
chord comes next," but "how long chords last." Korzeniowski & Widmer's 2018
result is exactly this — in automatic chord recognition, the gains live in the
*duration* model, not the frame- or label-level language model.

## The trap we'd already stepped in once

We had tried explicit-duration decoding before (Gen-1, "Candidate B"). It failed
badly — majmin collapsed from 17% to 10%. But the post-mortem was precise about
*why*: forcing the true ~2-beat harmonic rhythm onto a weak per-*segment*
emission just gave the weak emission more chances to be wrong. The binding
constraint back then was emission discriminability, not decoder structure.

That constraint is gone. Gen-2 has a per-beat root model that's 96% accurate on
clean renders. So this time the premise-check was: is there duration *structure*
worth modeling, and can we trust a per-beat *quality* signal too?

- Jazz chord durations are almost comically non-geometric: 57% of chords last
  exactly 2 beats, 30% last 4, and *essentially none* last 1 or 3. That's a very
  loud boundary prior — "don't you dare carve a 1-beat segment here."
- The per-beat quality head, though, is only 52% accurate. So we do **not** let
  the duration decoder pick quality. It decides roots and boundaries; the strong
  segment classifier still gets the final say on maj-vs-min.

## The one architecture bug worth naming

The first wiring forced the decoder's own root and classified quality at it. Root
accuracy held — but majmin dropped 8 points. The joint decoder's whole trick is
choosing root *and* quality together (it'll take a slightly-less-likely root if
its quality evidence is much stronger). Forcing the root throws that coupling
away. Fix: let the semi-Markov decide only *boundaries*, then hand those segments
to the unchanged joint labeler. With the duration weight at zero this is now
bit-for-bit the old pipeline — the cleanest possible sanity check.

## Numbers

Held-out jazz: root 88.7 → **89.4**, majmin 86.2 → **86.6**. POP909, which the
prior was never fit on: root 76.9 → **78.6**, majmin 50.1 → **51.1**, sevenths
45.9 → **47.0** — every metric up. The jazz "chords come in 2s and 4s" prior
transfers to pop because pop harmonic rhythm lives on the same grid.

The honest caveat: the *fraction* of root errors that are fifth-apart didn't
budge. Duration merging fixes the isolated single-beat slip, not the systematic
"the bass is playing the fifth for two whole bars" confusion. Those stay
acoustic — exactly what Mission 1 said. But we still deleted 13 of 110 root
errors on the held-out set, and it shipped.
