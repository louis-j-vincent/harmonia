# This Love — mistakes, fixes, and the roadmap to the target spec (2026-07-29)

Retrospective on the grid+sections work, so the errors don't recur. Target =
`docs/this_love_target_spec.md` (Louis's lead sheet).

## My (orchestration) mistakes — and the counter-rules

1. **Tested through the wrong path.** I reconstructed `bars` from the raw payload
   (`bars[c["bar"]]`), but `to_chart_model` builds DIFFERENT display bars — so my
   checks said "Cm" while the app (and Louis) saw "G". Cost several turns.
   → **Always verify through `harmonia.serving.render._chart_model_for`**, never a
   hand reconstruction. (Matches CLAUDE.md "verify what a thing DOES".)

2. **Over-engineered per-section bar LENGTH.** I chased variable bar lengths per
   section. The real convention is **one uniform bar; a faster section = more
   chords per bar (≤2)**, not a shorter bar.
   → Uniform grid; harmonic-rhythm difference = chords/bar.

3. **Forced one-chord-per-bar in my own render** → dropped B's Fm/Eb, showing
   `Cm Bb` instead of `Cm Fm | Bb Eb`.
   → Keep ≤2 chords/bar (split bars); the app already does this.

4. **Section-level phase band-aids for a grid problem.** `_align_repeats` /
   pickup shifts fought a symptom; the cause was the bad bar grid.
   → Fix the grid first (rigid_grid), then the section content falls out.

5. **Showed only the first 3 sections** → cut the C bridge off, then wondered why
   C was "missing".
   → Render/inspect the WHOLE form.

## The algorithm's (pipeline) mistakes — and the fixes

1. **Bad beat/bar grid** — glued G7+Cm into one ~5 s bar (root cause of the whole
   thread). → **FIXED**: `rigid_grid.rigid_grid_for` recovers period+phase from
   the chord onsets; `apply_rigid_grid` re-quantises. G7 and Cm now separate.

2. **Intro pickup mis-binned.** The pre-grid `C` at t=0.44 (before the first bar
   edge) is clamped into bar 0 and steals the downbeat, pushing G7 to bar 4.
   → **FIX (this round)**: `apply_rigid_grid(drop_before_grid=True)` drops chords
   whose onset precedes the first bar edge; the hook passes it. A opens on G.

3. **No repeat folding into ×N.** The form prints in play order (ABABACBB), not
   `A×3 … B×3`. The machinery EXISTS (`_fold_section_loops`,
   `_fold_repeating_section_groups`). → Next: run it on the regridded bars.

4. **1st/2nd endings not surfaced.** B and C diverge only in their last two bars.
   `_detect_endings` EXISTS (This Love was its motivating case). → Next: wire it
   on the regridded bars (its earlier input was the bad grid).

5. **1-vs-2 chords per bar not chosen per section.** A is 1 chord/bar (drop the
   passing `Fdim`); B is 2 chords/bar. → Next: per-section chord DENSITY (chords
   per bar) picks the max-per-bar for that section, instead of a global cap of 2.

6. **Grid octave ambiguity corpus-wide** (held chords double the bar, 2 chords/bar
   halve it) — 11/18 recover the octave. This Love is fine. → Future: an
   accent/tempo cue (drums, beatthis tempo octave) to break the octave.

## Roadmap (ordered)

1. ✅ Regrid from onsets (rigid_grid). 2. ✅ Drop the intro pickup. 3. Per-section
chords-per-bar (A→1, B→2, drop passing). 4. Fold internal loop repeats → `×N`
badges. 5. Detect 1st/2nd endings on B and C. 6. Emit the compact form string.
7. (corpus) break the grid octave with a tempo cue.

Acceptance = the rendered chart equals `docs/this_love_target_spec.md`.
