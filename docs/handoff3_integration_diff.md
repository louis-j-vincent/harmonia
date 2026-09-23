# handoff 3 → shipped app: integration diff & report

**Branch:** `feat/ui-design-integration` · **Date:** 2026-07-29 · **No code changed** (diff-only, as requested).

## One-line verdict

**The handoff design is already integrated and live.** The shipped app (`app_shell.html`,
labelled *"design handoff 2"* in the server) implements every surface and nearly every
feature in `handoff 3/HANDOFF.md`, and it does so wired to **real** endpoints where
handoff 3's reference engines still use **mocks**. There is little-to-nothing *net-new*
to port; the only honest open question is **cosmetic/UX drift**, which needs a visual pass.

## Why the earlier "MISSING" scare was wrong

A name-level grep said `buildIReal`, `confirmChord`, `buildRequest`, etc. were "missing".
They're not missing features — the shipped template is a **separate implementation** of the
same design under its own function names (`renderGrid`, `colOf`/`keyHueCss`, the merge/
re-infer flow), not a copy of handoff 3's engines. Feature-level parity is what matters,
and it's high.

## Surface-by-surface

| Surface | handoff 3 feature | Shipped status | Delta / risk |
|---|---|---|---|
| **Chart viewer** | iReal layout, 3 modes (Read/Analyse/Annotate), Analyse Function\|Key sub-colour, repeat-fold `×N`, playhead, circle-of-fifths **rotor** transpose | ✅ all present (`ireal` layout, `MODES` w/ annotate, `colOf`+`keyHueCss`, `×${span}` fold, rotor) | Cosmetic only — needs pixel check |
| **Chord editor** | Compass + Guide, plus Cylinder + Chip-grid edit-by-hand | ✅ all four tabs live (`Wheel / Suggestions / Compass / Guide`, `data-mode`), cylinder + chips present | handoff 3 has a standalone `buildCylinder`; shipped folds it in differently — no UX gap |
| **Re-infer loop** | confidence %, family fallback <0.42, confirm hard-clamp, re-infer POST, propagation banner, section merge | ✅ shipped: `↻ Re-infer with my fixes`, `🔗 Merge sections`, confirmed-badge, real `POST /api/reinfer` | **⚠ regression trap:** handoff 3's `harmonia_reinfer.js` uses `mockReinfer`. Shipped is already real-endpoint. **Do not "port" this — it would go backwards.** |
| **Import / search** | search-first entry, 6-stage "analysing" narration | ✅ shipped as the one-page app (`/` → `app_shell.html`: search → analyse → chart → annotate) | Cosmetic only |
| **Launcher** | redesign, check()/poll, states, copy-command | ✅ `harmonia.html` already = the redesign (`HLAUNCH`, ring, copy) **and keeps the real `check()` + redirect** the handoff asked to preserve | None |

## What handoff 3 actually is

`handoff 3/` is untracked and never committed. Its `.dc.html` files are dated 2026-07-13 —
the **same design era** as the shipped app, which the server literally names *"design
handoff 2"*. The `js/` engines are **mock-driven reference implementations** of the shipped
design, not a newer spec. The integration surface (`chart_model.py` adapter, `app_shell.html`)
is also **currently modified in the working tree** — likely a concurrent session — which is
another reason not to bulk-overwrite.

## Recommendation

1. **Do not do a "full re-integration" from handoff 3.** It would regress real endpoints to
   mocks and risks clobbering committed + in-progress UI work.
2. The remaining value is **catching cosmetic/UX drift** between the handoff 3 `*_demo.html`
   targets and the live app. That is a **visual pass**, not a code merge: render each live
   surface at 390×844, put it beside the matching `handoff 3/js/*_demo.html`, list only the
   pixel/behaviour differences, and port those individually.
3. If nothing meaningful differs visually, the correct outcome is: **close this out, the
   design is shipped** — and delete the stale `handoff 3/` drop + `MISSION_COMPLETION_GUIDE.md`
   blueprint (its "SPLICES markers" instructions no longer match the current migrate script).

## Visual pass results (2026-07-29, 390×844 headless Chrome)

Screenshots in scratchpad `vdiff/`. Verdict: **no regressions; a few intentional evolutions;
one cosmetic chart delta worth a look.**

| Surface | Static-comparable? | Result |
|---|---|---|
| **Launcher** | yes | ✅ **Pixel-match**, and correctly *drops* the demo's bottom preview state-switcher (as the handoff instructed). Nothing to do. |
| **Chart** | yes | ✅ Very close (3-mode control, G-min pill, iReal barlines, split bars, right-edge playhead, "certainty ≥ 0.60" caption all present). **3 minor deltas** below. |
| **Import / search** | yes | ↗ Live has **evolved past** the demo: dropped the big *"What do you want to play?"* serif headline for a compact wordmark+tagline, and **added** YouTube\|iReal toggle, Record, Jam Mode, Training mode. Not a regression. |
| **Chord editor** | no (tap-to-open modal) | Blank statically. Code parity confirmed: Wheel/Suggestions/Compass/Guide + cylinder + chips all live. Needs an interactive check to compare pixels. |
| **Re-infer loop** | no (Annotate mode + tap) | Demo shows conf-%, family fallback, `?` flag, FORM merge ribbon, Re-infer button — all present live, wired to the **real** endpoint (demo is mock). Needs interactive check. |

### The 3 chart deltas (demo → live)
1. **Section markers.** Demo puts boxed `A/B/C` **inline on the double left-barline** of the
   row each section starts, with a **`×2` repeat-fold badge**. Live puts `A/B/C` as a **chip
   strip above the grid**, no `×N` badge. *(Plausibly an intentional evolution — the shipped
   chip-strip + hand-drawn section labels are a sanctioned feature. Confirm intent before "fixing".)*
2. **Top bar.** Demo = *"change key"* pill + a **Print · gig chart** button. Live = key
   dropdown + song title, **no Print button**.
3. **Intro card.** Demo has a *"READING VIEW — press ▶ to play along"* card; live uses the
   bottom *"Auto: …"* caption instead.

(Chord-identity differences between the two are real-inference vs hand-typed demo data — **not**
a design delta.)

## Final call

Design is shipped and matches the handoff. The only *portable* items are the three chart
cosmetics above — and #1 is likely a deliberate divergence, so **confirm intent, don't
auto-port.** Recommend: decide per-item whether to pull the demo's section-marker/Print/intro
styling back in, then close this out and clean up the stale `handoff 3/` drop + the
`MISSION_COMPLETION_GUIDE.md` blueprint (its SPLICES instructions no longer match the migrate script).
