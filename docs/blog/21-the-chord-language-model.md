# 21 — A language model whose words are half-bars

*2026-08-01, branch `feat/chord-lm`, commits 5dbd132→…*

Louis's brief: build a language model over chords, on a metrical grid — half-bar
tokens, chord families rather than full qualities, a token for *no chord* and a
token for *same as before*. Check whether it already exists; if not, build it.
And: "pars du principe que ça devrait marcher".

It works. 79.5% of half-bars named correctly on held-out songs, against a 45.2%
floor and a 57.2% n-gram ceiling. But the interesting part is that it took three
rounds to get there, and all three blockers were **measurement** problems wearing
modelling clothes.

## It did not already exist

Every chord LM in the field — Chordonomicon's GPT-2 over 666k songs, our own
`train_progression_lm.py`, our own `ProgressionEncoder` — models the *sequence of
chord changes*. It knows `Dm → G7 → C` and has no way to say the Dm lasted two
bars.

I ran the two published models rather than reading their cards. MusicLang's
5.6M-parameter GPT-2 loads and generates fine, and its vocabulary of 50 tokens
contains no unit of time at all. Asked for a progression it emits
`CHORD_DEGREE__1` three times in a row — an event model's only way to express a
hold, and exactly the pathology a `%` token exists to prevent. The one HuggingFace
repo whose name promised per-bar tokens ships a tokenizer and no weights.

Our pipeline's hardest question is *when* the harmony changes. That question is
not expressible in any of them.

## Three bugs, none of them in the model

**The masked model looked worse than the causal one.** It has strictly more
information — both sides of the slot instead of one — so scoring below the causal
model was impossible if both had converged. They hadn't: a masked model gets a
gradient on the ~16% of slots it masks, a causal one on all of them. Same epochs
= six times less signal. At matched signal the ordering flipped to the one
information theory demands.

**The context-length ablation was measuring the wrong thing.** Re-scoring each
slot with only the last *k* visible showed short context doing suspiciously well.
Truncating the input had left-shifted the positional embeddings, so a window from
bar 30 was being told it was at bar 0. The ablation was measuring a distribution
shift. The bug flattered short context by ~10 points.

**The repetition cache looked worthless.** Its top-1 was 94% correct. Its
contribution was +0.6pp. Both were true: a smoothing constant of 0.05, spread
across 90 tokens, gave a chord seen once before a probability of 0.19. Right
answer, no confidence, nothing to transfer. Every argmax metric said it was fine.
Printing mean *p(true)* next to top-1 found it in one line, and the contribution
went to +8.8pp.

That third one is the reusable lesson: **accuracy cannot see a calibration bug.**
The regression test asserts the probability, not the argmax; the argmax version
passes either way.

## The thing Louis predicted

Mid-session he said he expected a model with whole-song context to *easily redo
the patterns* — charts are built out of literal repeats, so bar 33 of an AABA
tune is bar 1 again.

So I split the test slots by whether their previous four bars had already
occurred verbatim earlier in the same song (35.5% had), and compared the model
against a rule that just copies whatever followed last time.

| | model | copy rule |
|---|---|---|
| absolute positions | 72.1% | 91.3% |

Nineteen points, sitting there. The diagnosis: with a learned absolute position
table, "find this pattern somewhere earlier" is a different attention pattern for
every offset the pattern might sit at. Relative positions make it one pattern.
Swapping in rotary embeddings:

| | model | copy rule |
|---|---|---|
| rotary positions | **88.0%** | 91.3% |

Gap closed to 3.3 points — and novel-context slots improved too (65.7% → 70.5%),
so it wasn't only bookkeeping. The explicit cache, worth +8.8pp on the absolute
model, drops to +1.3pp on the rotary one, and its fitted trust at an 8-bar match
falls from 0.90 to 0.50. Two independent measurements agreeing that the
architecture absorbed the signal.

## What it writes

From a two-bar minor ii-V, at temperature 0.8:

```
| D:hdim  %       | G:dom   %       | C:min   %       | F:dom   %       |
| F:min   %       | Bb:dom  %       | Eb:maj  %       | %       %       |
    ... then this eight-bar phrase, verbatim, three more times
```

A descending-fifths chain that then repeats its own phrase. From a Bb7 prompt it
writes a blues head and modulates through Bbm–Eb7–Ab.

## What it found in our own charts

Pointed at `harmonia_min/state/charts/`, it immediately caught an inconsistency
rather than a disagreement of taste. In **She Will Be Loved** our pipeline writes
the same chord both ways in the same song: `Bb` as a major triad 28 times, `Bb7`
14 times. The LM proposes the plain triad at all 14 dominant spellings and never
the reverse. Which is right needs the ear — but the pipeline cannot be right both
ways, and 28-vs-14 is instability, not variation.

## Where this does not go yet

Nowhere. It is not wired into `infer_chords_v1`, and the number that matters for
wiring it — 73.3% chord identity where the harmony moves — says it is a prior,
not an oracle. That is precisely the shape that failed in issue #21, when the
ProgressionEncoder was bolted on as a post-hoc reranker and cost 3.6pp on the
real path (post 17). It belongs as a factor in a joint decode, measured on
`infer_chords_v1` and not on a proxy harness.

Full writeup and every table: `docs/chord_lm_2026-08-01.md`.
