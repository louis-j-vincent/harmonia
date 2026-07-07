# Design: the "Jazzify" cursor (conventional → funky reharmonization)

**Goal.** A single slider on the interactive chart that reharmonizes the inferred
changes with increasing boldness — from "add the obvious 7ths" to "really out" —
so a player can dial in how much spice they want. This is a *creative* layer on
top of inference, not a claim about what was heard.

## The core idea

The cursor is an **intensity 0–5**. Each notch is a pure, deterministic
transform `chart → chart`; the slider applies notches `1..k` in order, so 0 is
always the untouched inference and dragging back is a clean undo. Every transform
runs **against the local scale track** (`continuity_scale_track`, already
computed): a substitution or extension is only chosen if it's coherent with the
key the chord is *in*, so the result stays musical instead of random.

Transforms are rhythm-aware. Inserting a ii–V needs beats, so a notch may **split
a bar** (a 4-beat one-chord bar → 2+2) using the existing beat grid; it never
overfills a bar past its meter.

## The ladder (each notch is cumulative)

| notch | name | what it does | needs |
|------:|------|--------------|-------|
| 0 | **As played** | the inferred chart, untouched | — |
| 1 | **Color** | triads → 7ths; add the diatonic 6/7 that's missing; nothing chromatic | scale track |
| 2 | **Extensions** | add 9/11/13 by function — maj7 +9, dom +9 or +13, m7 +11 (only in-scale tensions) | scale track |
| 3 | **Little ii–V's** | before a target reached by a strong root move, drop its ii–V (or just the V) into the bar's back half — "secondary 2-5-1s" | scale track + beat split |
| 4 | **Substitutions** | tritone sub on dominants (G7→D♭7), backdoor (V→♭VII7), diminished passing chords, relative-minor swaps | scale + function |
| 5 | **Altered / funky** | alter the (now secondary/tritone) dominants — 7♭9, 7♯9, 7♯11, 7alt; modal interchange (borrow ♭VI, iv); upper-structure voicings | scale + function |

Notch 1–2 are *safe* (diatonic, reversible, hard to get wrong). 3–5 are the
opinionated ones and where the "funk" lives.

## Why the scale track is the engine

Every decision keys off the per-chord local key already computed for the
highlight:

- **Which tension is in-scale** (notch 2): a ♮13 on a dom is only added if the
  6th degree of the chord's local key is diatonic — otherwise offer ♭13. The
  continuity scale tells us directly.
- **Which ii–V to insert** (notch 3): the target chord's local key gives the ii
  and V roots (ii = key+2 as m7, V = key+7 as 7). Deceptive/minor targets get a
  minor ii–V (ø7–7♭9–m).
- **When a sub is idiomatic** (notch 4): tritone subs apply to *dominant-function*
  chords (identified by function, already classified in `quality_class`), not to
  every major triad.

## Interaction with the rest of the UI

- **Not inference.** When the cursor is > 0, jazzified chords render in a marked
  style (e.g. a small ✦ or a lighter weight) and the certainty colour bar dims —
  these are suggestions, not what the model heard. The original stays one drag
  away.
- **Re-analyze after.** The scale highlight recomputes on the jazzified chart, so
  an inserted secondary ii–V lights up its own tonicization band — the reharm
  *teaches* why it works.
- **Transpose / level compose cleanly** because Jazzify, like transpose, is just
  another client-side transform over the structured chord list.

## Build order (low-risk first)

1. Notches 0–2 (extensions) — pure, local, safe; ship and get feedback.
2. Notch 3 (ii–V insertion + bar splitting) — the first structural change; needs
   the beat-grid handling.
3. Notches 4–5 (subs, alterations) — validate each transform on a handful of
   standards with a jazz ear before enabling.

A "what changed" log (list of applied moves per bar) makes the funk auditable and
doubles as an ear-training explainer.

## Open questions

- **Voice-leading / melody.** We don't track melody, so reharm can clash with a
  tune's top line. Mitigations: keep phrase-final chords stable; prefer subs that
  preserve guide tones (3rds/7ths). A melody-aware version is a later stage.
- **Taste is plural.** "Funky" isn't one axis. A v2 could split the cursor into
  *density* (how many chords) vs *chromaticism* (how out), or offer style presets
  (bebop / modal / Robert Glasper).
