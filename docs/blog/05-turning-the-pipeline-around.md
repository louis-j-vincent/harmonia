# Building Harmonia: turning the pipeline around (Part 5)

*Fifth in a series on building a Bayesian chord-recognition system for solo jazz
piano. [Part 4](04-a-coin-flip-wearing-a-lab-coat.md) ended with a corpus-scale
lesson about validating on distributions, not single songs. This part is about a
different kind of move: instead of squeezing more out of the inference direction,
build the generative direction — and get training data with perfect labels for free.*

## The idea

Everything so far has run one way: audio in, chords out, POP909's annotations as
the measuring stick. But POP909 keeps biting in the same three places. It's pop,
not jazz — my hand-written ii-V-I priors are aimed at a vocabulary the corpus
barely uses. It has no section labels — the form-clustering work in the structure
proposal can only be checked by ear. And its piano MIDI has no separate bass stem —
the bass-anchored root inference that scored 86.8% had to *infer* the bass register
instead of just reading it.

So: turn the pipeline around. Take chord charts where the harmony, the form, and
the style are already written down — lead sheets — render a real accompaniment
from them, and keep every intermediate artifact aligned. The chart *is* the ground
truth. Whatever the renderer plays, we know exactly which chord, which section,
which bar it belongs to, because we chose it.

Jazz musicians already have the perfect chart format: iReal Pro. Its community
playlists encode 1,400+ standards as `irealb://` URLs — chords, section markers
(`*A`, `*B`), repeats, codas, style, key, meter. My first thought was to also use
iReal Pro (or GarageBand) to *render* the accompaniment, since the app does exactly
that. That half of the idea died fast: neither has any programmatic interface.
GarageBand doesn't even have an AppleScript dictionary. The format is the treasure;
the app is a dead end.

## Scoping before committing

The rule from earlier sessions applies: minimal test through *all* steps before
believing in any of them. Four candidates, each smoke-tested:

- **pyRealParser** parsed 1,859 charts from three public corpora (1,460 jazz
  standards, 345 pop, 54 blues) on the first try — titles, styles, keys, flattened
  measures. One catch: it throws away the section markers during flattening. A
  sentinel trick (rewrite `*A` to `@A` before flattening; the sentinel survives
  every regex in its repeat-expansion machinery) preserves per-bar section labels.
- **MMA (Musical MIDI Accompaniment)** — a 20-year-old, still-maintained,
  pure-Python accompaniment engine I'd never used. Feed it `Groove Swing` and a
  list of bars, get multi-track MIDI back with tracks *named* `Bass`, `Chord`,
  `Drum`. That naming matters more than it looks: the bass line is extractable by
  construction, no source separation, no register heuristics. 2,022 grooves, a
  163-quality chord table that covers `7alt` and `13sus` natively, deterministic
  under a seed, ~0.2 seconds per song.
- **Hand-rolled pretty_midi comping** — works, 20 lines, and immediately shows why
  it's the fallback rather than the answer: root-fifth bass and block chords is a
  metronome wearing a beret. Kept for augmentation variety later.
- **GarageBand / iReal Pro as renderers** — ruled out, above.

The glue risk was the notation gap: iReal writes `F^7`, `E-7`, `Eh7`, glues chords
together inside a bar (`Eh7A7b9`), and sprinkles pseudo-symbols (`n` for no-chord,
`W` for bass-only slashes, `x` for bar repeats). The end-to-end smoke test on five
standards came back cleaner than expected: zero unmapped tokens, and the forms
read like a fake book index — All The Things You Are came out `A8 B8 C8 D12`,
All Of Me `A8 B8 A8 C8`. The best sanity check was musical: on the bossa grooves,
the rendered bass hit the chart's chord root on 100% of barlines. The chart said
Dm7; the bass played D. The tuple is aligned.

Corpus scale surfaced the real bugs, as it always does. My favorite: pyRealParser
"cleans up" chord strings by deleting single `l` characters (they're rendering
artifacts in the format) — which quietly turns `D7alt` into `D7at`. An altered
dominant, mangled by a regex. Fixed with the world's most specific mapping entry.

## What the database is

One JSON record per song — 1,800 of them: form string and per-bar section labels
(**structure**), a per-beat timeline in both notations (**chords**), the extracted
bass stem as note events (**bass**), and paths to the rendered multi-track file
(**midi**), plus a per-song bass-root agreement score as built-in QA. Jazz corpus
mean: 97%. Blues: 71% — blues bass lines riff on thirds and fifths by design,
which is exactly the kind of thing you want your training data to contain.

This is the dataset POP909 couldn't be: jazz harmony with real ii-V-Is and altered
dominants for the n-gram priors, ground-truth section labels for validating the
form-clustering work, and a clean bass stem for the bass-anchored inference — all
three of the standing complaints, answered by one generative pipeline. Next step
when needed: render the MIDI to audio through the existing FluidSynth renderer and
the full inference pipeline finally has an end-to-end exam it can be graded on
with a perfect answer key.
