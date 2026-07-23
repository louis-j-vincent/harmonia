# Chord-detector improvement backlog (2026-07-23)

**Distinction:** the *refactoring* makes the code clean at IDENTICAL behavior (proven byte-for-byte,
Phase 3). This backlog is the OTHER axis Louis asked for — make the detector *better*. Each item is
built as a **default-OFF brick**, measured on the 7 frozen Brick-0 songs (`accuracy_score.py`), kept
only if a real gain + **no regression** on the parity net.

**Real-target baseline (7 frozen songs):** root **0.737** · maj/min 0.710 · **7ths 0.517** (weakest)
· partial 0.642 · strict 0.477 · **bass 0.744**. Proven bottleneck (3 research runs): the **upstream
root SOURCE**, not segmentation / timing / post-hoc editing.

---

## P1 — ROOT SOURCE (the ~26pp lever; the proven bottleneck)
The single biggest prize. ~42% of root errors are fifth-related (P5 23% + P4 19%), the rest m3/TT/m7.

- **P1a — Bass-informed root prior (highest EV).** Bass (0.744) already beats root (0.737) AND root
  gates bass (bass-miss = 308s root-wrong vs 8s root-ok). Feed the actually-played bass note back
  into the root decision (bass→root feedback), instead of root→bass one-way. Well-motivated, low-disk.
- **P1b — musx vs NNLS-head reweighting.** The shipped root = music-x-lab labels (musx dominates; the
  NNLS head is discarded on the nnls24 path). Screen an ensemble weighting / confidence-gated blend.
- **P1c — Better note-detection front-end for root.** The NNLS/Basic-Pitch chroma is where fifth
  energy leaks in. Longer-horizon; needs front-end work.

## P2 — QUALITY / 7ths (weakest metric, 0.517; strict quality 0.477)
- **P2a — 7th disambiguation via the "one note" (Louis's method, transposed).** Just as the 4th
  degree separates a key from its fifth, the **7th degree** separates maj7 / dom7 / m7 / 6. Screen
  whether chroma energy on that one note cleanly separates the qualities on the confusion spans —
  the analog of the validated `fifth_discriminator`, applied to quality.
- **P2b — quality from musx vs head blend** (mirror of P1b for quality).

## P3 — NO-CHORD, productionize (validated +2.1pp, currently dormant)
`no_chord_policy` gave **+2.10pp root** on the 7, zero regressions, but suppress-all is unsafe where
real silence exists (pop intros). **Build the silence-guard** = `musx-N ∧ energy-N` intersect +
duration guard, then wire ON. Near-term, concrete, mostly-built. **This is the readiest real gain.**

## P4 — FIFTH-DISCRIMINATOR, extend (validated theory, benchmark-limited)
`fifth_discriminator` is CORRECT (georgia 88% vs 38% baselines) but only +0.05pp here (1 major-key
tune). Re-evaluate on a **major-key-heavy jazz corpus** where it pays; investigate a minor/modal
analog (does a one-note discriminator exist for the minor-key confusions on blue_bossa?).

## P5 — ALIGNMENT (owned by the concurrent fusion lane — coordinate, don't duplicate)
Beat/downbeat: jazz first-beat failures + octave-locks. The fusion aligner (other session) is already
on this. Chord accuracy inherits it: bad bar-1 phase → wrong bar chords. Track their result.

---

## Ruled OUT this session (do NOT chase)
- **Segmentation / grid unification** — NEGATIVE (per-bar −20.6pp; grid-snap hurts). STEP 8b/9.
- **Flip-margin gate** — +0.17pp, musx already coalesces. Dead on shipped path.
- **Post-hoc root editing without acoustic evidence** — a flat key prior can't beat fifth-confusion
  (root_resolve −4.3pp). Only the *right acoustic note* works (P2a/P1a), not re-adjudication.

## Method reminder (every item)
Default-OFF brick → premise-screen cheaply first (CLAUDE.md #2) → measure on/off delta on the 7
frozen songs → keep only if real gain AND parity net stays green → inspectable artifact → then wire
(coordinated, since wiring touches the contended `chord_pipeline_v1.py`).
