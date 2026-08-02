# Design handoffs — analysis + implementation plan (2026-08-02, rev. 2)

**Rev. 2 correction:** the first version of this doc analysed `handoff 3/` and found it
already shipped. Louis then pointed out missing features (piano chord animation…) —
right call: `handoff 3/` was **not the newest drop**. Full picture below.

## 1. The three drops, dated

| Drop | Date | Status |
|---|---|---|
| `handoff 2/` | 2026-07-13 | **Byte-identical duplicate of `handoff 3/`** (`diff -rq` clean) |
| `handoff 3/` | 2026-07-13 | Already shipped in the old app; server-side gaps in harmonia_min (see §4) |
| **`Harmonia.zip` → `design_handoff_new_chord_ux/`** | **2026-07-21 22:07** | **Newest. Zero traces in either app shell** (`reduceQ`, `glyphCompact`, `analysisLens`, `keyHighlightStyle`, `immersive`, Learn: 0 hits in both `app_shell.html`) |

The zip contains: `README.md` (the contract), `concept-engine.js` (all algorithms,
dependency-free vanilla JS, "copy verbatim"), the concept board
(`Harmonia Concept — New Chord UX.dc.html`, panels 1–7), and an updated
`Harmonia App.dc.html` integration target. Extracted to scratchpad
`design_handoff_new_chord_ux/` for this session; re-extract from the zip anytime.

**Sibling document:** the same day (21-07, 15:09, commit `de4e7ad`) Louis committed
`docs/pedagogical_mode_design_2026_07_21.md` + `scratchpad/pedagogical_mode_prototype.html`
— his own research/design on the same territory, but deeper on the algorithms:
real jazz voicing conventions (§8), voice-leading suggestions (§9), diatonic-vs-altered
extensions (§10.3), bass-invariance constraint (§11). **Authority split for
implementation: pedagogical spec = the algorithms; new-chord-ux handoff = the visuals
and interactions.** Where they conflict on voicings, the spec (v3, later that day) wins.

## 2. What the new handoff adds (6 features + notation)

| # | Feature | Backend needed? |
|---|---|---|
| 1 | **Learn mode + L1/L2/L3 ladder** — relabel every chord to a simpler vocabulary via pure lookup `reduceQ` (L1 triads, L2 +3 sevenths, L3 full). Root & timing never change | No |
| 2 | **Chord sheet with piano illustration** — tap a chord → bottom sheet: piano drawing, bass note accented, 3 voicing cards (close / shell / rootless) via `voicing()` + `renderVoicing()` | No |
| 3 | **Immersive reading** — chrome slides away, floating mini-transport, pull-to-reveal | No |
| 4 | **Voltas (1./2. endings)** — alternate final bar on its own line with bracket | **Yes** — endings structure in ChartModel; `irealb_aligner.py` already parses `[n1…][n2…]` |
| 5 | **Key-analysis lenses** — Global key vs Tonicization; full-cell colour bands joining across same-key bars | Global: no. **Tonicization: yes** — per-chord `localKey` pc. Note: harmonia_min already emits `keySegments` (`pipeline.py:343`, `harmonic_key.py`) — likely feedable with a small adapter |
| 6 | **Play-along voicing bar** — pinned live keyboard at the bottom showing the current chord's voicing on each playhead change | No |
| — | **Compact notation** — accidental+quality stacked in a narrow column beside a full-size root (`△ − ø °` glyphs), plus dark theme + chart-colour prefs | No |

## 3. Implementation plan — new-chord-ux track

**Target surface: `harmonia_min/app_shell.html`** (the active app; a plain served
file, no migrate-script machinery). Never touch :7771 / `chart_interactive.py` —
owned by other sessions; the zip README's "integrate into chart_interactive.py"
predates harmonia_min. Extend the existing store (`S.mode`, `S.colorMode`,
`S.chords[]`, playhead) — no parallel state.

Ranked — Louis flagged #2 first, and A/B/C share one engine:

- **A. Voicing engine + chord sheet (#2)** (~½ day). Port `IV[q]`, `voicing()`,
  `renderVoicing()` from `concept-engine.js`, reconciled with pedagogical spec §8
  (CLOSE root position at L1/L2, real jazz voicings at L3) and §11 (bass-invariance).
  Bottom sheet on chord tap in **Read** mode (Annotate keeps the editor).
  **Done when:** tap any chord on iPhone → sheet with piano, bass accented, 3 voicings.
- **B. Play-along voicing bar (#6)** (~2–3 h after A). Drive from the existing
  playhead index, update on chord change only.
- **C. Learn ladder (#1)** (~2–3 h). `reduceQ` verbatim + `Advanced ⇄ Learn` pill +
  "simpler" tag on changed cells. Purely a render-time relabel.
- **D. Compact notation + themes** (~½ day). `glyphCompact` with the reserved
  accidental slot; visualisation preference, not a mode. Dark palette from the README.
  ⚠ must compose with the 2026-08-02 density work (`SZ1/SZ2/BARH`, quarter-position
  grid) — check two-chord bars at 390 px, rendered pixels not JS widths.
- **E. Immersive mode (#3)** (~2–3 h).
- **F. Global-key lens (#5a)** (~2 h) — full-cell bands, `khue(pc)=(pc*7)%12/12*360`,
  neutral fallback when circle-of-fifths colours are off.
- **G. Tonicization lens (#5b)** — adapter from `keySegments` → per-chord `localKey`,
  then same band renderer. Gate behind confidence; fall back to global.
- **H. Voltas (#4)** — ChartModel schema (`shared` + `endings[]`), coordinate with
  `folding.py` (an ending is a divergent tail — today's fold handles tails by
  covering the whole pass; voltas are the *right* representation of that case).
  Do last; schema touches the fold contract.

A–C are one coherent sprint (~1.5 days) delivering everything Louis noticed missing.

## 4. Server-gap track (from rev. 1 — still valid, unchanged)

`harmonia_min/app_shell.html` is a copy of the old shell; all handoff-3 surfaces
exist as client JS but several endpoints 404:

- **P1 — annotation persistence** (~1–2 h): `POST /api/annotations/<file>` 404s
  *silently* (`app_shell.html:3321`) — every confirmed chord is lost on reload.
  Sidecar in `state/annotations/`, rehydrate on chart-model load, toast on failure.
- **P2 — minimal re-infer** (~½ day): route `/api/context_rescore/<file>` onto the
  already-copied but unrouted `chord_context_prior.py` / `span_rescore.py`.
  Old `/api/reinfer` machinery is on the DO-NOT-IMPORT list — rebuild, don't port.
- **P3 — graceful 404 degradation** (~1 h): hide/disable Record, Jam, Training,
  iReal toggle, section-merge game tile (that one is a hard Flask 404 page).
- **P4 — section merge** (after P2, validated by ear).

Suggested interleave: **P1 first** (data loss), then the A–C sprint, then P2.

## 5. Do-not-do / cleanup

- No port of `handoff 2|3/js/*` engines (mock-driven, older than the live shell).
- No un-folding of Louis's 2026-08-02 chart decisions (×N strip-only, badge above
  row, règle d'or lattice) to match July demos.
- Cleanup once the plan is accepted: delete `handoff 2/` (verified byte-identical
  duplicate), archive `handoff 3/` + `Harmonia.zip` under `docs/archive/`,
  commit the untracked `docs/handoff3_integration_diff.md`.
