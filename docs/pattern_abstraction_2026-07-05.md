# A hierarchy of progression patterns — finding the best representation (2026-07-05)

User's idea: just as chords form a tree (family → seventh → exact), *progressions*
form one too. Tritone subs are functionally equivalent (V7→I ≡ bII7→I), so ii-V-I
and ii-bII-I are the same pattern; encoding chords by function should collapse the
pattern vocabulary and let a model learn faster from less data.

Tested (`scripts/experiment_pattern_abstraction.py`) on the full jazz corpus
(1,454 songs, 64,834 chords) at four representation levels.

## Pattern concentration by representation level

| encoding | chord types | distinct trigrams | top-50 trigrams cover | next-chord acc |
|---|---|---|---|---|
| literal (degree, 7th) | 157 | 11,077 | 27% | 35.3% |
| tritone-folded dominants | 139 | 9,289 | 30% | 36.2% |
| **degree + family (drop the 7th)** | **60** | **6,879** | **34%** | **41.2%** |
| tritone-fold + family | 66 | 7,164 | 35% | 40.0% |

## Three findings

1. **The idea is right — abstraction concentrates patterns and improves
   prediction.** Going from the literal encoding to the family level cuts distinct
   3-chord patterns by 38% (11,077 → 6,879) and lifts next-chord accuracy by ~6
   points. And it degrades far less with little data: with only 58 training songs
   the family encoding already scores 39.1% vs the literal's 32.9%. That
   data-efficiency is exactly the payoff you predicted.

2. **The biggest lever is dropping the 7th (the family level), not tritone subs.**
   The tritone-fold alone barely moves anything (157→139 types) because — the
   honest reason — **notated tritone subs are rare in lead-sheet jazz**: ii-bII-I
   occurs 44 times vs ii-V-I's 1,577 (the collapse *does* correctly merge them,
   it's just that there's little to merge). Tritone subs are mostly a
   *performance/reharmonization* choice, absent from the written charts. On real
   recordings they'd be common and this brick would matter much more — a corpus
   limitation, not a flaw in the idea.

3. **Abstractions must be coherent or they backfire.** Tritone-fold *on top of*
   family is slightly worse than family alone (66 vs 60 types): my fold routes
   dominants into a separate "DOM" class that no longer merges with the major
   family, fragmenting the space. Lesson: the equivalence classes have to be
   designed as one consistent quotient, not stacked ad hoc.

## The unifying result

The representation that best concentrates *progression* patterns is the **same
family level** that the chord tree (`harmonia/theory/chord_tree.py`) already uses
for individual chords. So the chord tree's family layer is not just a good chord
abstraction — it is also the natural alphabet for learning progressions. That's a
clean unification: one hierarchy serves both the chord-quality decision and the
pattern-learning problem.

## Recommendation

- Learn progression priors at the **family level** (degree + maj/min/dim/aug/sus)
  — it's where patterns are dense and predictable, and it stays robust with little
  data. Use the finer seventh/exact levels only where the evidence (audio) already
  pins them, exactly as the confidence-gated tree does.
- Keep **tritone-sub equivalence as a defined class in the pattern quotient**, but
  don't expect much from it on notated lead sheets; revisit it on real-recording
  or bebop corpora where reharmonization is frequent.
- Your "next 4 chords reveal a section change" point (a bridge entry shifts the
  priors) is the natural next study: condition the progression model on
  proximity to a section boundary — the accompaniment DB has ground-truth
  section labels to test it directly.
