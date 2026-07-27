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

---

## 2026-07-27 — HEADLINE METRIC = family-level partial credit (Louis's call)

Louis: report the metric that is right when the FAMILY is right, regardless of whether the
7th was caught. **That metric already exists and is already computed: `partial_credit`
(0.6419) vs `mirex_sevenths` (0.5173)** — the gap between them IS the 7th-omission forgiveness.
Use `partial_credit` as the headline; keep `sevenths`/`strict` as secondary diagnostics.

**Asymmetry, deliberate (kept):** `_FAMILY` forgives `min7→min` (129.2s) and `maj7→maj`
(42.7s) — same family — but NOT `7→maj` (49.3s), because `dom` is its own family. Musically
right: a dominant without its 7th loses the tritone and the function changes, whereas
Cmaj7→C / Dm7→Dm keep theirs. So ~172s of the 221s omission loss is already forgiven.

**Consequence (honest):** the `seventh_upgrade` brick (+2.59pp sevenths, +0.75 strict) moves
`partial_credit` by **exactly 0.0000** — it only performs min→min7, a case the family metric
already credits. **Under the headline metric that brick is worth nothing.** It stays dormant;
revisit only if `sevenths` becomes a target in its own right.

## 2026-07-27 — ALL EFFORT → BOUNDARY BLEED (Louis's directive)

The diagnosis (`dc6926a`) reframed the problem: **70.5% of wrong-root duration is an adjacent
GT chord's root** (lift 1.75 vs chance; 94.3% of "fifth-related" error duration is the
neighbour), 2:1 toward holding the PREVIOUS chord, 48-72% localised within 0.5s of the shared
edge. Oracle: re-cutting our own labels on GT boundaries = **root +6.56pp**, of which a
jitter-only snap recovers **+5.45** → **83% of the temporal prize is MISPLACED boundaries**
(unbiased median +0.01s, ±1-beat quantised), not missing ones. This is now the single focus.

**Already measured and REFUTED as boundary signals — do not re-run:** chroma novelty (0.547
vs shipped 0.554), onset strength (0.367), HPSS (0.370), onset+chroma (0.482) — percussive
energy is blind to harmonic change; musx segmentation (−1.97), union (−1.97, `_coalesce_labeled`
erases added cuts), gated under-seg repair (−0.34), onset-hint retiming (−0.44), within-beat
trim/shift/weight (≈0), metrical snap (−1.49 raw / +0.27 gated), majority vote (dead on premise).

**Untapped:** music-x-lab emits FRAME-LEVEL chord posteriors that we discard (we read only the
argmax `.lab` at segment midpoints). That is the natural evidence for ±1-beat boundary placement
and the natural input to a supervised beat-level boundary classifier.
