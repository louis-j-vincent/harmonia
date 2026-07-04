# Prior individual (non-joint) findings

Established in earlier sessions, before this joint/pairwise investigation.
New analysis should build on these, not re-derive them.

- **Beat phase (category A):** point-biserial correlation with chord-change,
  song 001 alone: r=+0.53 (strongest single signal found so far).
  `P(chord change | phase)` for song 001: 97.3%/37.0%/69.9%/1.4% for beats
  0-3. Corpus-wide (909 songs, pooled): much softer — 50.8%/42.0%/29.3%/
  22.8%. Song 001 is the most metrically regular song in the whole corpus
  for this statistic (100th percentile) — not representative on its own.
- **Bass pitch-class change (category B):** r=+0.40 (song 001). Corpus-wide
  contingency: P(chord changed | bass changed)=49.7% vs P(chord changed |
  bass same)=26.9%. Bass sits on the chord's root or fifth 75.3% of the
  time when a real chord is present (63.6% root + 11.7% fifth).
- **Onset density (category D):** r=+0.24 (song 001).
- **Chroma novelty, SSM-checkerboard, small kernel (category D):** r=+0.22
  (song 001).
- **Bass onset presence (category B):** r=+0.16 (song 001) — weaker than
  bass pitch-class change itself.
- **7th quality (category C-adjacent):** chromatic dom7 chords resolve down
  a fifth (functional-dominant behaviour) at 53.4%, matching the primary
  diatonic V's own 52.4%, regardless of scale position. Chromatic maj7
  chords only 12.1%. The specific 7th predicts function better than scale
  position alone.
- **Chord duration (category A):** full corpus PMF peaks at 2 beats
  (49.2%), not 1 — proof the true shape isn't geometric/memoryless.
- **Mode-agnostic parent-scale identification:** 95.3% agreement with the
  GT-implied 7-note collection, using chord content alone (no key lookup) —
  confirms "which 7 notes" and "which one is home" are separable problems.

Full detail: `docs/scale_taxonomy_2026-07-03.md`,
`docs/known_issues.md` issue #1's subsections,
`docs/plots/inference/bass_patterns/`, `docs/plots/structure_proposal/`.
