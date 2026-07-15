# Chord hierarchy audit: maj vs dom (2026-07-15, Agent 2 Part 1)

**Task premise:** "Our hierarchy has maj→dom (dom is a child of maj), which makes
the model classify dom as maj *by construction*." Fix so maj and dom are siblings.

**Finding: the premise is already satisfied where it matters. No label-space
change is needed, and the one place a maj⊇dom relation exists is a musically
correct display layer that should NOT be flattened.** (CLAUDE.md rule 2 — screen
the premise cheaply before implementing.)

## Where "hierarchy" lives, and what each layer actually does

1. **Flat 5-class quality disambiguator** (the model that decides third/seventh):
   label space is `['maj','min','dom','hdim','dim']`, stored as distinct
   `quality_idx` 0..4 in `billboard_training_corpus_full.npz`. maj (0) and dom (2)
   are **already independent sibling classes**. The softmax head has one logit per
   class; nothing forces dom through a maj node. There is no maj→dom parent-child
   link here to break.

2. **`harmonia/theory/chord_tree.py` `_FAMILY`** (reporting/display only): maps
   `DOM7 → Family.MAJOR`, alongside `MAJ7 → Family.MAJOR`. This is the *level-1
   family* used by `HierarchicalReporter` to back off to a coarse label when the
   evidence is weak. It is **music-theoretically correct**: a dominant 7th is a
   major triad (major 3rd + perfect 5th) plus a ♭7 — its family *is* major. maj7
   and dom7 are siblings one level down (`_BASE_SEVENTH`: `MAJ7→MAJ7`,
   `DOM7→DOM7`), which is exactly the sibling relationship the task wants, at the
   correct level. Flattening dom into a top-level family peer of "major" would be
   musically wrong and would only change human-facing `reported_label`s, not the
   disambiguator's decisions.

3. Neither `build_key_prior` nor `build_index` collapses dom into maj; the HMM
   vocabulary carries `DOM7` and `MAJ7` as separate states.

## Cheap premise check (the falsification)

Trained the flat 12→64→32→5 MLP on the GT-root-relative frame (root fixed at
chroma index 0 via `np.roll(chroma,-root)`), inverse-freq class weights,
song-stratified 80/10/10. On the 11,384-chord held-out test set:

- **23.4%** of test chords are predicted **dom**, and **dom recall = 0.665**.

If dom were "classified as maj by construction," dom-predicted fraction would be
~0 and dom recall ~0. It is not. **Premise falsified** — consistent with the
repo's own prior note (commit a73b065: "premise already satisfied").

## Conclusion / action

- No change to the flat quality label space (already siblings).
- No change to `chord_tree.py` `_FAMILY` (dom-under-major is correct theory and
  is display-only; flattening it is a musically-wrong edit to a surface shared
  across the pipeline, and risky with concurrent sessions editing that layer).
- The real dom bottleneck is **acoustic + root-frame**, not hierarchical — dom
  differs from maj only by the ♭7, and from a *wrong* root frame that ♭7 lands
  in the wrong bin (see `chord_quality_report.md` and known_issues Phase 2B).
