# Handoff: symbolic song-structure detection (learned, adaptive hierarchy)

_Written 2026-07-18, end of an overnight session. This carries the full reasoning
trail, not just a task list — read it before touching any of the referenced code.
The user (ML PhD, jazz musician) cares about WHY each approach failed or worked,
not just the numbers._

## The goal

Detect song structure (section boundaries + which sections repeat — "this is the
same as bars 9-16") from the CHORD SEQUENCE alone, no audio needed. This sits on
top of an already-solved chord-recognition pipeline (NNLS-24 + music-x-lab, see
`docs/session_2026_07_17_bass_root_capstone.md` — root/quality/bass are strong;
structure is the open problem).

## The corpus and evaluation (don't re-derive, reuse)

- **iReal Pro corpus**: `harmonia/data/ireal_corpus.py` parses `data/ireal/*.txt`
  (~2399 tunes total, **1992 tunes have real multi-section ground truth** —
  `*A/*B/*C` markers = `section_per_bar`). This solved an earlier "only 3 songs
  have audio+structure GT" problem — no audio needed, this is pure symbolic data.
- **Metric**: V-measure (`mir_eval.segment.nce`), per-bar, reported as `V_F`.
- **Splits**: always song-level (never chord/bar-level) to avoid leakage.
- **Known gotcha**: title collisions are real — e.g. "Yesterday" in this corpus
  is Kern's "Yesterdays" or "Yesterday's Gardenias," NOT the Beatles song. Don't
  assume a title match is the right song without checking.

## What was tried, in order, and why each was rejected (or worked)

All artifacts are in `scratchpad/symstruct*.py` — read the most recent ones
directly, this summary is not a substitute for the code.

1. **Flat block8** (`symstruct.py`): fixed 8-bar blocks, exact transposition-
   invariant chord-sequence match, greedy A/B/C labeling. **V_F = 0.68-0.70**
   (varies by split). This has been the benchmark to beat all night.

2. **Fixed-scale hierarchy** (`symstruct_hier.py`): independently run flat
   matching at global scales {4,8,16,32}, try to pick the best scale per song.
   **Failed** — no unsupervised selector could beat flat block8, because ~56% of
   songs are genuinely 8-bar and picking wrong on the majority costs more than
   the gain on the minority that want 16/32. Oracle ceiling (perfect hindsight
   scale-picking) = **0.732** — this number gets cited a lot, it is NOT
   deployable, it's a benchmark showing even perfect global-scale selection has
   limited headroom.

3. **Grammar induction / RePair-Sequitur** (`symstruct_grammar.py`): bottom-up
   pattern compression from 4-bar nuclear blocks, merging repeated adjacent
   bigrams into composite symbols recursively. Two hypotheses were raised and
   **both falsified with real numbers** (worth knowing so they aren't re-tried):
   - *"Adjacency-only merging can't find AABA forms"* — false. Flat block8
     already compares every block against ALL earlier ones, not just
     neighbors, so adjacency was never the bottleneck. All-pairs matching
     tested explicitly on AABA-shaped tunes: 69% recovery vs greedy's 68% —
     no real difference.
   - **The actual bottleneck, found via a clean-GT sanity check**: even on
     PERFECT iReal chord data with perfect bars, transposition-invariant exact
     matching only recovers **33% of true same-section repeats at 59%
     precision**. Same sections often have different chords (fills,
     turnarounds, altered endings); different sections can collide under
     transposition. **Every hard-matching method is capped by this ceiling,
     regardless of hierarchy depth or merge strategy.** This is the key
     finding of the whole night — it reframed the problem from "smarter
     matching logic" to "need a fundamentally more flexible similarity."

4. **Learned similarity (CURRENT WINNING APPROACH)** (`symstruct_learned.py`):
   train a small BiLSTM encoder with a metric-learning/contrastive loss
   (InfoNCE) on 4-bar chord blocks, so same-GT-section blocks embed close
   together even when their literal chords differ.
   - **Critical finding on transposition**: forcing transposition-invariance
     via random-key augmentation ACTIVELY HURTS. Within a real song, repeated
     sections are almost always in the SAME key — training the model to treat
     all 12 keys as equivalent throws away that signal and creates false
     matches between different sections that happen to share a shape in
     different keys.
   - **The winning variant**: normalize each WHOLE SONG to a canonical tonic
     (transpose the entire song rigidly by one fixed amount, e.g. tonic→C)
     as a **fixed input representation**, not a training augmentation. This
     preserves 100% of within-song relative structure (a whole-song rigid
     shift can never change which blocks match which) while removing
     cross-song key variation, making patterns easier to learn.
   - **Validated results, 5 seeds, paired same-split comparison**:
     seed0 +0.005, seed1 +0.003, seed2 +0.008, seed3 +0.012, seed4 +0.022 vs
     flat block8. 5/5 positive, sign test p≈0.03, mean margin **+0.010**.
     Confirmed not an overfitting artifact (2x larger encoder = 0.702, no
     meaningful gain — the small model already suffices).
   - **Beats the hard-matching P/R ceiling directly**: at equal precision,
     learned recall 0.364 vs hard-matching's ~0.32; at equal recall, learned
     precision 0.675 vs hard's ~0.59.
   - **Status**: validated research result, real but MODEST margin (+0.01).
     NOT recommended for deployment yet — still below the 0.732 oracle
     (though that's an unrealistic hindsight number), and inherits the same
     real-audio transfer risk as everything else tonight (this was all
     trained/evaluated on clean iReal data; noisy predicted chords + bar-grid
     drift on real recordings is untested).

## Correction from the user (2026-07-18, after this handoff was drafted)

**Use 2-bar nuclear blocks as the default going forward, not 4-bar.** The
earlier finding "4-bar clearly beats 2-bar" (in the "What was tried" section
above, item 3/grammar induction) was measured under HARD/EXACT chord-sequence
matching, where finer 2-bar blocks are more fragile to partial mismatches.
That finding does NOT necessarily hold for the learned-similarity approach
(item 4) — the whole point of a learned, tolerant similarity function is that
it should handle the finer granularity's fragility better than exact matching
did. The user's original hierarchical intuition explicitly started from "a
nuclear 2 or 4 bar level," and their explicit instruction now is to make
2-bar the default nuclear unit for the learned/adaptive work. Re-validate the
learned encoder and the adaptive hierarchy phase with 2-bar nuclear blocks as
the primary configuration (4-bar can stay as a comparison point, but 2-bar is
the one to optimize).

## What's IN PROGRESS right now (pick this up first)

An agent (was running as `a51d5fec3a80dc91d` in the prior session) was mid-way
through the natural next phase when this handoff was written:

**Adaptive agglomerative hierarchy using the learned key-normalized similarity
as the merge criterion**, replacing the earlier fixed 4→8→16→32 doubling
scaffold. The point: real song structure doesn't come in power-of-2 blocks — a
section should grow only as far as the learned similarity actually supports it,
bottom-up, with variable-length results (an 11-bar section shouldn't be forced
into the nearest power of 2). Use the winning encoder's embeddings directly as
the agglomerative merge score at each step, on top of the `symstruct_hier.py`/
`symstruct_grammar.py` scaffolding for the tree structure itself. Evaluate the
same way (V-measure, 1992-tune corpus) against flat block8 (0.68-0.70) and the
oracle (0.732). **Check if this run finished and what it found before
re-starting it** — check `docs/known_issues.md`'s most recent STRUCTURE entries
first, and look for a result file under `scratchpad/`.

## Explicitly deferred (user's words: "for further investigations later")

**Learned harmonic equivalence** (e.g. tritone substitutions treated as "the
same" for structure-matching purposes, beyond simple transposition). This is a
promising extension of the same learned-similarity idea — a ii-V-I and its
tritone-sub variant are functionally the same progression but look different
under exact/transposition-only matching. Don't start this until the adaptive
hierarchy phase above is resolved; the user asked for it to be explicitly
queued, not built yet.

## Known unresolved gap (don't let this get lost)

**Everything above was validated on clean, hand-entered iReal chord data.**
Nothing has been tested against our OWN pipeline's real predicted chords on
real audio (noisy chord labels + the known bar-grid/downbeat drift issue,
CLAUDE.md error pattern #1). This is the actual deployment blocker, not the
algorithm itself — before shipping ANY of this to the live app, it needs
validation on real predicted-chord output, ideally with a synthetic noise
stress-test first (reuse the noise-injection approach from
`symstruct_robust.py`) before spending time on real audio.

## Process notes for whoever picks this up

- **Run everything synchronously/foreground.** Do not launch a background
  training/eval run and end your turn "waiting for a monitor" — this was a
  repeated, costly failure mode across many agents tonight (the harness does
  not resume idle background waits productively; always block until you have
  a real result before stopping).
- **No commits** — a separate commit-coordinator process handles git; just
  write files and log findings to `docs/known_issues.md` incrementally, not
  only at the end.
- **Production server**: `scripts/harmonia_server.py` / `harmonia/output/
  app_shell.html` may be live and in active use — check `lsof -i :7771`
  before assuming anything about server state, and don't touch UI/server
  files for this structure-detection work unless/until it's genuinely ready
  to deploy (it is not, as of this handoff).
- **Honesty bar**: this project had a real integrity incident earlier
  tonight (an agent fabricated/conflated results, caught by audit — see
  `docs/known_issues.md`'s "PHASE-0 AUDIT" entry). Every number in this
  handoff traces to a real file; keep that standard. Multi-seed validation
  is mandatory for any headline claim, per the pattern that caught the
  earlier "does the margin hold up" question above.

## Key files reference

| File | What it is |
|---|---|
| `harmonia/data/ireal_corpus.py` | iReal Pro corpus parser, GT section labels |
| `scratchpad/symstruct.py` | Flat block8, chord-signature core, corpus loader |
| `scratchpad/symstruct_hier.py` | Fixed-scale hierarchy attempt (rejected) |
| `scratchpad/symstruct_grammar.py` | Grammar induction + clean-GT sanity check |
| `scratchpad/symstruct_robust.py` | Synthetic bar-drift noise injection for stress-testing |
| `scratchpad/symstruct_learned.py` | The winning learned key-norm encoder |
| `docs/known_issues.md` | Search "STRUCTURE" for the full dated entry trail |
