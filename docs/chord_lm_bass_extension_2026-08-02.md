# Adding the bass to the chord LM — design

*2026-08-02, from Louis's ear verdicts on three contested chords*

## What the verdicts showed

Louis judged three of the LM's contested half-bars by ear:

| case | chart | LM | ear | who was right |
|---|---|---|---|---|
| Stand By Me 121.8s | E:maj | **D:maj** | D:maj | the LM |
| Close to You 14.0s | B:maj | C:maj | **B:sus** | neither — chart had the root, LM lost it |
| She Will Be Loved 49.6s | **C:min** | Eb:maj | C:min | the chart |

All three disagreements are about the **root**. And the musx bass plane settles
all three, confidently:

| case | ear | bass posterior (top 2) |
|---|---|---|
| Stand By Me 121.8s | D | **D 0.82** / E 0.10 |
| Close to You 14.0s | B | **B 0.96** / N 0.01 |
| She Will Be Loved 49.6s | C | **C 0.90** / Bb 0.08 |

Across all 19 contested half-bars, the bass backs the **chart's** root in 16 of
19, and contradicts the LM's proposed root almost everywhere — Gb 0.93 against
the LM's A, B 0.99 against its C, C 0.98 against its Db, D 0.87 against its Bb.
The one time the bass sides with the LM is 121.8s, which is the one case
Louis's ear also gave the LM.

Then the corpus-level number, over all 120 LM disagreements on the five charts:

| what the LM changes | share |
|---|---|
| the **root** | **85.0%** |
| the quality only (root kept) | 15.0% |

**So the LM is spending 85% of its disagreements on the one axis where good
acoustic evidence already exists, and it loses.** Its own lane — quality, and
whether the harmony changes at all — is the 15%. Louis's "c'est un Bsus" is
exactly that lane: the root was right, the quality was wrong, and no bass model
can tell you a sus from a triad.

## The division of labour this implies

    root / bass          <- acoustic. The bass plane already knows.
    quality + family     <- the LM. Grammar, and nothing else has it.
    harmonic rhythm      <- the LM (change vs hold), measured 91.1% accurate.

The LM should be **conditioned on** the bass, not competing with it.

## Level 1 — fuse at decode time, no retraining

The cheapest thing that could work, and therefore the thing to measure first
(CLAUDE.md rule #2: screen the premise before implementing):

    score(chord) = log P_LM(chord | context) + w · log P_bass(bass_of(chord))

`P_bass` is `musx.frame_posteriors(...)[1]`, already computed for every song and
cached — 13 columns, **column 0 is N and column i≥1 is pitch class i−1**
(determined empirically, 93.8% agreement with the decoded triad root on Let It
Be; it is not documented anywhere).

`bass_of(chord)` is not simply the root: a C/E sounds an E. So the term should be
a small mixture over the chord's own tones, root-weighted —
`P_bass(root)·0.8 + P_bass(third)·0.1 + P_bass(fifth)·0.1` — rather than a hard
root lookup, or every first-inversion chord gets punished for being played the
way pop plays it.

`w` fitted on GuitarSet (verified GT, audio shipped with it). Expected effect,
from the 19 cases above: it removes ~15 bad proposals and keeps the good one.

**This is where to start**, and it is a day of work, not a week.

## Level 2 — condition the LM on the bass (the real extension)

Add a second input stream so the model *sees* the bass at every slot, including
the masked one. That is what turns "grammar guessing a root" into "grammar
choosing a quality for a root it can see".

    input(slot) = chord-token embedding
                + slot-in-bar embedding
                + bass embedding          <- new

Three decisions, with the reasoning:

1. **Feed the posterior, not the argmax.** A projected 13-vector carries the
   model's uncertainty; an argmax throws away exactly the cases that matter
   (the ambiguous ones we want the LM to arbitrate). `Linear(13 -> d_model)`.

2. **Drop the bass channel at random during training** (say 20% of slots). Two
   reasons: the bass model is sometimes wrong or silent, and a model that has
   become bass-dependent collapses there; and it keeps the pure-grammar
   behaviour available, which is what the cloze evaluation measures today.

3. **Training data — restore the slash basses I threw away.** `vocab.py` drops
   `/A` from `D-7/A` on purpose (a grammar model needs ii-V-I to look like
   ii-V-I). But the iReal corpus *has* those inversions, and the project already
   knows they are 10–18% of lines. They should be kept as a SEPARATE bass label
   rather than discarded, so the bass channel trains on real inversions instead
   of assuming root position everywhere.

   For the rest, the training-time bass has to be simulated — and its noise
   profile must be calibrated on the real model's error profile (measurable on
   GuitarSet), or the LM will learn to trust a bass that is cleaner than the one
   it will meet.

## Level 3 — predict the bass too

A second output head over 13 bass classes, so the LM proposes `C/E`, not just
`C`. This is the one that lines up with the project's actual target (sounding
bass, `corpus_schema.sounding_bass_pc`) and with what the chart format already
stores (`bass` field, `-1` for root position).

## The trap to avoid

`vocab.py` roots are **functional**; the pipeline's are the **sounding bass**.
Fusing them without stating the mapping is exactly CLAUDE.md error-pattern #3 —
attributing a disagreement to the model when the two sides encode different
things. Every fusion term above has to be explicit about which one it is
speaking, and the tests have to pin it.

## Order

1. Level 1 fusion, measured on GuitarSet. If it does not move, levels 2–3 rest
   on a false premise and we find out in a day.
2. Restore slash basses in the iReal loader (data work, useful regardless).
3. Level 2, with the bass-noise profile calibrated on GuitarSet.
4. Level 3 only once 1–2 have paid.
