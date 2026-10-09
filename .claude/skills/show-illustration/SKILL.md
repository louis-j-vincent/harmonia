---
name: show-illustration
description: Use whenever a change, a measurement or a claim has something to LOOK at — any UI/chart/glyph/layout change, any before/after, any "it works now", any diagnostic. Always put the picture in front of Louis (SendUserFile, render) in the same message as the claim; a link or a sentence is not an illustration.
---

# Always show the illustration

Louis, 2026-10-09, after "ça marche" / "c'est fait" messages that carried
links and numbers but no picture, and a bare « show me » when he could not
see what I meant: **skill → always show illustration.**

## The rule

When you report a visual result, the image comes WITH the report — not on
request, not as a link he has to open, not "I checked it myself".

1. **Capture the real thing**, from the real app (Playwright, 390 px wide,
   `?open=<key>` deep link, scale 2) — never a mock-up of what it should
   look like. Crop to the part that matters; one image per idea.
2. **Send it** with `SendUserFile` (`display: "render"`, `status: "normal"`),
   with a one-line caption saying what to look at. He is usually on another
   device: an inline tool-result image is invisible to him, a sent file is not.
3. **Before/after = side by side in ONE image**, labelled "avant" / "après",
   same song, same crop. Two separate files make him compare from memory.
4. **Then** the text: one or two sentences, the live link as a bonus (the
   `100.89.209.63:7772` form), what the picture does NOT show (untested modes,
   devices) in one line.
5. **Look at the image yourself first.** Read the file before sending; if it
   shows a defect (an accidental that slid under the letter, a chord that
   overflows), fix it and re-shoot — don't ship a picture of a bug with a
   caption saying it works.

## What counts as "something to look at"

UI, chart rendering, glyphs, layout, colours, section boundaries, bar grids,
a start-of-song marker, a before/after on any song — and any measurement that
has a shape (a distribution, a curve, a per-bar posterior). Numbers alone are
for the commit message; if he must trust a number, give him a picture of what
the number counts.

## Not an excuse to skip the other rules

Still verify in the rendered app (`verify-rendered-chart`), still ask with
mock-ups when a request is ambiguous (`ask-before-guessing`). This skill is
the delivery step: what was verified is *shown*.

## When no image is possible

Audio-only results: send the clip (`SendUserFile`) the same way. Server-side
only changes with nothing visual (a rebake that changed 0 bars): say "rien à
voir : 0 mesure changée" and give the diff page if one exists. Never silently
fall back to prose.

## Le hook qui l'impose (2026-10-09, « make it a hook »)

`.claude/hooks/show_illustration.py`, branché en `Stop` dans
`.claude/settings.json`. Si, depuis le dernier message de Louis, un fichier
visuel a été modifié (Edit/Write sur `harmonia/static/`, `docs/plots/*.html`,
`tools/*page*.py`, `tools/*demo*.py`) sans `SendUserFile` ensuite, la fin du
tour est REFUSÉE une fois, avec la consigne de capturer, regarder, envoyer.
Garde-fous : `stop_hook_active` (pas de boucle), toute erreur → on laisse finir.
Ne couvre PAS les modifications d'un sous-agent ni un `git commit` seul : là, le
skill reste la seule garantie. Rejoué sur les 79 tours de la session qui l'a vu
naître : 8 blocages, tous sur du visuel livré sans image.
