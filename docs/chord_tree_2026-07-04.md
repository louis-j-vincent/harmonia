# Hierarchical chord detection — how deep can we reliably name a chord?

Exploratory experiment (`scripts/experiment_chord_tree.py`), 2026-07-04, in
response to the idea: organize chords as a tree by how much they specify, and
detect at each level, only going deeper when the evidence supports it.

## The tree

- **Level 1 — family** (decided by the third + fifth): major / minor /
  diminished / augmented / suspended.
- **Level 2 — the seventh** stacked on top: none(triad/6) / dominant-7th /
  major-7th / diminished-7th.
- **Level 3 — the exact chord** including color notes (9ths, alterations, 6).

A C-major triad {C,E,G} is the parent of Cmaj7, C7, C6 (all contain it); C7
{C,E,G,Bb} is the parent of C9, C7b9 (all contain it). Deeper = more notes
specified = quieter, harder-to-hear distinctions.

## Result (2,540 test chords, fake songs, root given, train/test split by song)

| level | perfect notes (ceiling) | real audio (Basic Pitch) |
|---|---|---|
| L1 — family (maj/min/dim/aug/sus) | 94.6% | **80.6%** |
| L2 — + which seventh | 95.4% | 60.8% |
| L3 — exact chord (~15 types) | 90.8% | 53.2% |

**The staircase is the whole point.** With perfect notes every level is
~90%+ — the tree structure isn't needed, you can just name the exact chord.
With real audio, each step down the tree costs ~20 points: the family is
reliable (81%), the seventh is a near-coin-flip on top of it, and the exact
chord is a coin flip. This is the acoustic reality: the note that decides the
family (the *third*) is loud-ish; the notes that decide finer types are quiet
and often not even played.

## Confidence-gated tree walk (the actual proposed procedure)

Pick a family; only choose the seventh among that family's children; only pick
the exact chord among that node's children; descend while the winner clearly
beats the runner-up. "Never-wrong" = we never output a label that contradicts
the truth (stopping shallow and just naming the family is allowed and counts as
fine).

| confidence demanded | real-audio never-wrong | avg depth reached |
|---|---|---|
| low (margin≥0.02) | 57.2% | 2.13 / 3 |
| medium (margin≥0.05) | 63.7% | 1.72 / 3 |
| high (margin≥0.10) | **74.9%** | 1.24 / 3 |

Being cautious genuinely buys correctness: demanding more confidence lifts
never-wrong from 57% to 75%, at the cost of answering shallower (usually just
"it's a major-family chord" plus sometimes the seventh). This is strictly
better than the current pipeline's behaviour of always committing to one exact
chord and being wrong ~half the time.

## The most useful diagnostic: why level 1 fails

Level-1 confusions on real audio: **major→suspended (152)**, major→minor (132),
minor→major (47), major→augmented (43), major→diminished (37).

Every top confusion is a *third* problem. When Basic Pitch under-detects the
third (the quiet note), a major chord's leftover evidence is just root + fifth
— which looks exactly like a suspended chord (no third) or gets flipped to
minor. **The entire chord-quality bottleneck reduces to one thing: hearing the
third.** This is consistent with the earlier finding (session 6) that the third
is the weakest chord tone acoustically (24–42% of the root's salience).

## Recommendation

1. **Default output granularity = Level 1 (family).** From audio it's ~81%
   reliable vs ~53% for exact chords — report "C major" confidently instead of
   gambling on "Cmaj7". This is a presentation/decoding choice, not a retrain.
2. **Offer Level 2 (the seventh) only when confident** (the gated walk). Users
   who want "Cmaj7 vs C7" get it when the evidence is there and a plain "C
   major" when it isn't — honest instead of a coin flip.
3. **Consider merging `suspended` into an "unclear third" bucket at Level 1**,
   since most sus predictions on real audio are actually thirds that went
   missing, not genuine suspensions. Worth checking against how often true sus
   chords get correctly caught before doing this.
4. **The one lever that improves every level at once is third detection** — the
   Stage-1 audio→notes work (learned emission templates, per-key third
   sensitivity) all cash out here. Nothing below Level 1 improves until the
   third is heard more reliably.

## Caveats

- Synthetic audio (MMA renders): real recordings will score somewhat lower at
  every level; the *ordering* and the third-detection diagnosis transfer.
- "Root given" — this measures quality only. Root/family detection from the
  bass (the two-stage design) is a separate, already-strong signal.
- Exploratory only; nothing wired into the pipeline. The tree in
  `experiment_chord_tree.py::TREE` is the concrete structure to promote into
  `harmonia/theory/` if adopted.
