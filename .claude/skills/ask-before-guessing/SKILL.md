---
name: ask-before-guessing
description: Use before implementing ANY request whose target you had to interpret rather than read off directly — a cosmetic/UI/layout change described in words, a phrase with more than one plausible reading, a "make it look like X" with no picture, or a fix you've already had to re-guess once in this thread. Louis prefers answering a concrete question over discovering a wrong guess two edits later.
---

# Ask with pictures, don't guess with words

Louis, 2026-09-25, after two wrong guesses in a row on the same cosmetic
request (San Francisco Street's folded chart — a "B′" row, then a
"mini-cellule" split, each guessed from a verbal description and each
wrong in a different way):

> « Très bien posez beaucoup + de questions comme ça ça évite d'avoir des
> ambiguïtés ... je préfère largement répondre à tes questions que de te
> laisser deviner. »

## What actually happened

The request ("diviser la cellule de rajout qu'il y a juste avant en deux,
et d'ajouter les deux barres dans cette cellule") had at least three
internally-consistent readings — a display split, a section-label change,
a mini-cell-within-a-cell split — and each got implemented, verified with
Playwright, screenshotted, and pushed to the live app before turning out
to not be the one he meant. Every one of those cycles cost real turns.
`AskUserQuestion` with `preview` (ASCII mockups of the candidate layouts)
resolved it in one round — he picked directly off the pictures.

## The rule

Before implementing a request that involves **any of**:
- a cosmetic, layout, or display change described in prose, not shown
- a phrase that maps to more than one concrete implementation you can
  picture (e.g. "split X in two", "make it look nicer", "like Y but
  smaller") — if you can draw two different honest mockups from the same
  sentence, that's the signal
- a case where you've already implemented your best reading once this
  thread and it was rejected or corrected

... stop and use `AskUserQuestion` with 2–4 concrete options, using
`preview` for anything visual (ASCII box/row mockups of the actual content
— real chord names, real bar numbers, not placeholders). Ground every
option in the REAL data you already have (query the actual chart/state
first), not abstract descriptions — he picks off what's concretely there,
not off adjectives.

This is not a license to ask about everything — an unambiguous instruction
("fix this off-by-one", "commit this") still gets executed directly, no
question. It fires specifically on the guessable middle: prose describing
a shape, a layout, or an intent that could honestly cash out two or more
ways. When in doubt about whether it's ambiguous enough to ask: it
probably is — the cost of one good question is much lower than the cost
of a wrong guess implemented, verified, and shipped.
