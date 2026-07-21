# Pedagogical Mode — research + design spec (2026-07-21)

Author: design/research agent (standalone, no production files touched).
Prototype: `scratchpad/pedagogical_mode_prototype.html` (open in a browser).

Target user: an adult who has *started* reading chord charts but doesn't yet
grasp inversions / extensions / substitutions ("the user's mother"). This is a
**new, additive mode/toggle** layered on top of the locked "Chord AI for
advanced musicians" product — NOT a change to the jazz-facing chart. It is
philosophically aligned with the locked **simplicity principle (2026-07-19)**:
prefer simple unless evidence decisively supports complexity. Pedagogical mode
is the user-facing surface of that same instinct.

Vocabulary ceiling is the locked **iReal-level contract**:
`{maj7, m7, 7, m7♭5, dim7, sus2/sus4, 6, m6, mMaj7} + /bass` plus the plain
triads `{maj, min, dim, aug}`. Everything below reduces *from* that ceiling; it
never invents vocabulary above it.

---

## 1. Research summary — what the field actually does

### iReal Pro (inspected via Help Center + notation docs)
- **Chord diagrams are an on-demand overlay, not a difficulty system.** You tap
  a "Chord Diagrams" button and toggle Guitar / Piano (two-hands) / Ukulele /
  Chord-Scales; tap-and-hold a measure shows the fingering/voicing, and you can
  cycle alternate voicings. Piano diagrams show a two-hand keyboard.
- **No progressive simplification.** iReal is an *authoring/playback* tool: the
  chart is whatever the user typed. It has rich notation (`-` minor, `^`/△ maj7,
  `o` dim, `ø` half-dim, slash bass) but zero "make this easier" reduction. That
  is exactly the gap this feature fills.
- Takeaway for us: **borrow the tap-a-chord → diagram interaction**; the piano
  two-hand diagram is the closest existing thing to "what a chord looks like."

### Chord AI (App Store / Play listing)
- Real-time detection → shows **finger positions on Guitar / Piano / Ukulele**.
  FREE tier recognizes basic qualities (maj, min, aug, dim, 7, M7, sus); PRO
  adds half-dim, dim7, 9/11/13 etc.
- The free/PRO split is a *recognition-precision* tier, **not** a per-song
  learner-facing "reduce this chart to triads" control. No public evidence of a
  harmonic-simplification hierarchy.
- Takeaway: confirms **piano/guitar diagram on tap** is the expected mental
  model for "show me the chord," and that "basic vs advanced vocabulary" is a
  natural, market-accepted axis — but nobody exposes it as a *pedagogical
  reduction of the same song*.

### Moises — Guitar Chord Finder (closest real precedent)
- Explicitly offers **3 difficulty levels: easy / medium / advanced**, and the
  displayed chords adjust to the selected level.
- BUT this is **voicing/playability** simplification (easier guitar shapes),
  not harmonic reduction. "Easy" = easier-to-fret voicing of the *same* chord,
  not a simpler *chord*. Honest distinction: our Level system reduces the
  **harmony** (drop the 7th, drop the inversion), which is a different and more
  pedagogically ambitious axis than Moises's shape-swapping.

### Swaram — "Beginner Mode"
- A toggle that **auto-simplifies complex chords into easy open chords + capo
  suggestions.** Again playability-oriented (guitar), but it validates the exact
  UX pattern the user asked for: a single **Beginner toggle** that rewrites the
  chart to something a learner can handle.

### Hooktheory / musictheory.net (pedagogy, not detection)
- Hooktheory teaches with a **color-coded, piano-roll-style** interface, chords
  colored by root, and defines an inversion in plain language: *"a chord voicing
  where the lowest note (bass) is not the root; it uses another chord tone —
  the 3rd (1st inversion) or 5th (2nd inversion) — as the bass instead."* Their
  slash-chord study frames inversions as **the same note-set with a different
  bottom note**. This is *exactly* the sentence the user wants taught.
- musictheory.net: staff-based drills for inversions; more formal, less "app."
- Takeaway: **plain-language framing + a visual where the bass note is the
  spatially-lowest highlighted note** is the proven way to teach inversion.

### Honest gaps
- No inspected competitor exposes a **harmonic** (vocabulary-depth) reduction of
  one song into multiple difficulty levels. The playability-based simplifiers
  (Moises, Swaram) are the nearest, and they simplify *fingering*, not *harmony*.
  So piece #1 (below) is genuinely differentiated, not a clone — which also
  means we can't copy a spec; the reduction rules below are ours to justify.

---

## 2. Design — the two orthogonal reduction axes (piece 1)

The core idea: **do NOT conflate two different kinds of simplification.**

- **Axis A — Vocabulary depth (label complexity).** Pure *relabeling* of each
  chord. The number of chords and their timing is UNCHANGED. Always safe; needs
  no key inference. This is the backbone of the difficulty ladder.
- **Axis B — Harmonic density (chord count).** *Merges/removes* passing and
  substitute chords (ii–V→V, secondary dominants, passing diminished). Changes
  the chord COUNT and timing. Requires reliable key + function inference, so it
  is **riskier and OFF by default**. It's the "make the song shorter to read"
  axis, offered as an advanced sub-toggle.

Keeping them separate matters because Axis A can never be "wrong" (Cm7→Cm is
always a legitimate simplification of Cm7), whereas Axis B can restructure the
harmony incorrectly if the key/function guess is off. The known_issues bar-grid
and key-inference caveats mean Axis B must degrade gracefully.

### Axis A — vocabulary-depth levels (the ladder)

All reductions are **monotone supersets**: L1 ⊂ L2 ⊂ L3 in vocabulary, so every
higher level only *adds back* information, never contradicts a lower level.

| Source chord (ceiling, L3) | **L1 — Triads only** | **L2 — Triads + basic 7ths** | **L3 — Full (iReal ceiling)** |
|---|---|---|---|
| maj                | maj | maj | maj |
| maj7               | **maj** | **maj7** | maj7 |
| 6                  | **maj** | **maj** (6 is an extension, not a basic 7th) | 6 |
| 7 (dominant)       | **maj** | **7** | 7 |
| sus4 / sus2        | **maj** (drop the suspension) | **maj** | sus4 / sus2 |
| min                | min | min | min |
| m7                 | **min** | **m7** | m7 |
| m6                 | **min** | **min** | m6 |
| mMaj7              | **min** | **m7** (simplify the rare maj-7-on-minor) | mMaj7 |
| m7♭5 (half-dim)    | **dim** (keep the ♭5 character) | **dim** | m7♭5 |
| dim / dim7         | **dim** | **dim** | dim7 |
| aug / 7♯5          | **aug** | **aug** | aug (7♯5 → 7 at L2) |
| **X / bass (slash)** | **drop bass → root position** | **drop bass** | keep slash |

Optional **Level 0 — "two colours only"** for absolute beginners: collapse to
just **major / minor** (dim → min, aug → maj, sus → maj). Reduces the whole
vocabulary to the two qualities every learner meets first. Off unless requested.

**Music-theory justification (why these specific mappings, not arbitrary):**

1. **Reduce to the underlying triad by stacked-thirds identity.** Every ceiling
   chord is a triad + added thirds (7th/6th/extensions). L1 keeps the triad
   (root + 3rd + 5th) — the note-set that fixes *quality* (major/minor/dim/aug)
   and *root* — and discards the added stack. This is the standard
   "chord = triad with tones piled on top" reduction taught in every method book
   (Complete Jazz Keyboard Method; TalkingBass "simplifying jazz chords":
   *replace 9/11/13 with the 7th, then the 7th with the triad*).
2. **m7♭5 → dim triad, NOT min triad.** The defining feature of a
   half-diminished chord is the **♭5**; dropping to a plain minor triad would
   silently *raise the 5th* and change the sound/function (it would look like a
   ii chord instead of a viiø/ii-of-minor). The diminished **triad**
   (root, ♭3, ♭5) preserves the ♭5 and the unstable character. This is the one
   place where "keep the parent triad" would mislead, so we override it — a
   deliberate, defensible call, consistent with CLAUDE.md's partial-credit
   family logic.
3. **sus → drop the suspension to major.** A sus chord *has no 3rd*; a beginner
   cannot see a quality without a 3rd. The overwhelmingly common resolution of a
   pop/jazz sus is to the **major** triad on the same root (Dsus4→D). We reduce
   to major and (in Learn mode) annotate "the 4th resolves down to the 3rd."
   Stated non-solve: this discards genuine sus2 ambiguity — acceptable at L1/L2.
4. **6 and m6 restore only at L3.** A 6th is an *added tone*, not one of the
   three workhorse 7ths (maj7/m7/dom7) that L2 exists to teach. Restoring it at
   L2 would muddy the "L2 = learn the three 7th-chord families" story. (C6 is
   also enharmonically Am7/C — genuinely a higher-level ambiguity.)
5. **Inversions (slash bass) are the LAST thing restored (L3 only).** The bass ≠
   root distinction is precisely the "subtlety" the target user doesn't yet
   grasp, so L1/L2 present every chord in **root position**. The concept is then
   *taught explicitly* in the illustration (piece 2) before it appears in labels.
6. **L2 = the three diatonic 7th families.** maj7, m7, dominant-7 are the
   backbone of functional harmony; adding exactly these (and nothing altered)
   matches the universal method-book ordering: triads → 7th chords → extensions
   → alterations. Dominant-7 in particular is restored early because it carries
   *function* (the V→I engine) that a triad hides.

### Axis B — harmonic-density reduction (optional, off by default, needs key)

Applied *after* Axis A. Every rule is conservative and reversible, and each
must **degrade to a no-op** when key/function confidence is low (respect the
key-inference and bar-grid caveats in known_issues — never restructure on a
shaky grid).

- **B1 — Secondary-dominant de-tension.** A `V7/x` that resolves to `x` is
  relabeled to the *diatonic* chord on the same root (e.g. in C: `D7 → Dm7`,
  then Axis-A-reduced). Keeps the root motion (which the learner follows), drops
  the borrowed tension. Guard: only when `x` actually follows.
- **B2 — ii–V collapse.** A `ii(m7) → V7` inside one bar that resolves to `I` is
  reduced to just the **V7** (the tension chord that defines the motion), or, if
  configured "target-first," to just the **I**. Halves the chords-per-bar a
  beginner must place. Guard: only for a *complete* ii–V–I; a dangling ii–V is
  left intact.
- **B3 — Passing-chord absorption.** A chord shorter than a threshold
  (≤ ¼ bar, or a chromatic passing diminished like `♯Idim` between `I` and `ii`)
  is absorbed into the neighbour it voice-leads into. Guard: never merge across
  a detected section boundary; never absorb a structurally long chord.

Axis B changes the chord grid, so in the UI it must be visibly labelled
("simplified harmony — 3 chords merged") and be one tap to undo.

### Reduction pipeline (implementable summary)

```
raw_labels (L3 ceiling)
  └─ Axis A: map each label via the level table  → relabeled, same grid  (SAFE)
  └─ Axis B (opt-in, key-gated): merge passing/ii-V/secondary → fewer cells (RISKY)
  └─ render at selected level
```

Axis A is a pure function `simplify(label, level) -> label` (a lookup on the
parsed root+quality+bass — Harmonia already parses these). It should live as a
small standalone module (e.g. `harmonia/pedagogy/reduce.py`) so it can be unit
tested against a table of `(chord, level, expected)` cases, red-first per the
project's test discipline. It does NOT need the audio or the model — it operates
on already-emitted labels, which is why it's safe to build in parallel with the
core-pipeline refactor.

---

## 3. Chord illustration / "what a chord looks like" (piece 2)

**Primary visual: a mini piano keyboard (~2 octaves) with the chord's notes
highlighted and the BASS note visually distinct.** Piano beats guitar here for
one reason: on a keyboard, pitch order is left→right, so **the bottom (bass)
note is literally the leftmost highlighted key** — the entire inversion concept
becomes spatial and obvious. On a guitar fretboard the string ranges overlap and
"which note is on the bottom" is invisible. (We can still offer guitar diagrams
later for players, but the *teaching* visual is piano.)

Three coordinated elements, top to bottom:

1. **Keyboard strip.** Chord tones highlighted; the **bass note** in a stronger
   accent colour with a small "BASS" tag. Note names printed on the highlighted
   keys.
2. **Stacked-thirds "tower."** A vertical stack of dots = the note-set, labelled
   `root / 3rd / 5th / 7th`, teaching the "a chord is a stack of notes" metaphor
   directly. The bass tone is marked in the tower too, so a learner sees "same
   stack, the marked one is on the bottom."
3. **One plain-language caption**, e.g. *"C major 7 = four notes: C E G B.
   Lowest note is C — that's 'root position.'"*

**Teaching the inversion (the specific concept the user named).** Show a
before/after pair with **identical highlighted keys** but the bass marker moved:

- `C` (root position): C–E–G, **C** in bass → caption "C on the bottom."
- `C/E` (1st inversion): same C–E–G, **E** in bass → caption *"Same three notes.
  Only the bottom note changed — now it's E."*

That single side-by-side, with the note-set held constant and only the bass
marker sliding, is the whole lesson. It maps 1:1 to Hooktheory's definition.

---

## 4. Bottom suggestion bar — "like Airbnb" (piece 3)

Airbnb's pattern = a horizontal, thumb-scrollable strip of alternative cards.
In pedagogical mode:

- **Trigger:** tap any chord cell in the chart (Learn mode only).
- **Behaviour:** a bottom sheet slides up with a **horizontal strip of cards**,
  one per available simplification of *that* chord, from simplest → full:
  `Cm (simplest) · Cm7 · Cm7 (full)` — for a half-dim it reads
  `Adim (simplest) · Adim · Am7♭5 (full)`.
- **Each card** shows: the chord label at that level, a **mini piano thumbnail**
  (the piece-2 visual, small), and a one-line "why" (*"7th removed — just the
  basic triad"*).
- **Actions:** tapping a card (a) previews the illustration in the panel above,
  and (b) offers "use this level for the whole song" or "just this chord." So the
  strip is both a per-chord explainer *and* the control for the global level.
- It reuses the **same level ladder** from piece 1 — the strip is literally the
  chord walked up/down Axis A. Optionally it can also surface an *alternative
  reading* card ("model also considered F7sus4") when the model's 2nd-best is
  close, tying into the existing confidence work — but that's a later add.

This keeps the difficulty control discoverable and *local* (you learn what
"simpler" means on the chord in front of you) rather than a hidden global slider.

---

## 5. Pedagogical explanation mode (piece 4)

A **distinct top-level mode**, not tooltips sprinkled into the advanced chart:

- **`Advanced ⇄ Learn` toggle** at the top level. Advanced = the untouched
  jazz-musician chart (do-not-touch). Learn = a separate, softer-skinned view.
- Inside Learn:
  - a **difficulty selector** (L1 / L2 / L3, optionally L0) that drives Axis A;
  - **tap-a-chord** → piece-3 bottom strip + piece-2 illustration + caption;
  - **first-encounter coach-marks**: the first time an inversion (or a 7th, or a
    half-dim) would appear as you raise the level, a one-time explainer bubble
    fires ("This is an inversion — same notes, different bass"). Dismissible,
    shown once.
  - an optional **"Learn" side panel** with the running plain-language
    explanation of the currently-selected chord.
- **Separation guarantee:** none of this renders in Advanced mode; the advanced
  chart's density and notation are unchanged. Learn mode is meant to feel
  *friendly/playful* (consistent with the CLAUDE.md note that the chart surface
  should feel fun), with larger cells, softer colours, and plain language.

Surfacing summary: **one mode toggle** (not mixed in), **one level selector**,
**tap-to-learn** interactions, **once-only coach-marks**. No persistent tooltips
cluttering the advanced view.

---

## 6. Build notes / hand-off for implementation

- **Axis A first, standalone.** `simplify(parsed_chord, level)` is a pure lookup
  on `(root, quality, bass)` — Harmonia already parses these in `parseLabel`.
  Ship it as an isolated module + a red-first table test. Zero pipeline risk;
  can merge independently of the core refactor.
- **Axis B is opt-in and key-gated.** Do not enable until key/function
  confidence and the bestfit beat grid are trusted for the song (both have live
  caveats in known_issues). Degrade to no-op on low confidence.
- **Illustration + strip are pure front-end**, driven by the same
  `chord → note-set` function used in the prototype; no model calls needed.
- **Mode toggle lives above the chart component** so Advanced stays byte-for-byte
  the current experience.

---

## 7. Prototype — what to look at

`scratchpad/pedagogical_mode_prototype.html` (standalone, no dependencies, open
directly in a browser). It demonstrates, on the Autumn Leaves A-section
(`Cm7 | F7 | B♭maj7 | E♭maj7 | Am7♭5 | D7 | Gm6 | Gm`):

1. **Level toggle L1/L2/L3** rewriting the whole chart live via Axis A — watch
   `Cm7`→`Cm`, `Am7♭5`→`A°`(dim)→`Am7♭5`, `Gm6`→`Gm`→`Gm6` as you move levels.
2. **Tap any chord** → Airbnb-style **bottom suggestion strip** with the three
   level-variants of that chord, each with a mini piano thumbnail + a "why" line.
3. **Piano-keyboard illustration** with the bass note accented + a stacked-thirds
   tower + plain-language caption.
4. **A dedicated inversion demo** (`C` vs `C/E`) showing the identical note-set
   with only the bass marker moving — the exact concept the user wants taught.
5. **`Advanced ⇄ Learn` mode toggle** showing Learn as a separate skin.

It uses hardcoded example data (no live inference) — this is a design-validation
prototype, per the brief.

## 8. Voicing conventions — research + implementation (added, prototype v2)

The first prototype lit *pitch classes* (first-occurrence-per-octave), which is
not how a chord is actually voiced — notes appeared doubled/scattered. v2 models
voicings as **sets of absolute pitches** (semitones, C4=60), so real octave
placement, register and inversions are faithful. Each convention below is a
standard, citable jazz-piano voicing, not an invention.

### Beginner illustration (L1 / L2) — CLOSE root position
Root, 3rd, 5th, (7th) stacked within one octave, **no doubling**, root in the
bass. This is the "chord = a stack of thirds" visual every method book opens with
(Complete Jazz Keyboard Method). It is the default and the *only* voicing shown
below L3 — jazz forms are deliberately an L3 concept.

### Advanced illustration (L3) — real jazz-piano voicings
Implemented per chord quality, selectable via a "Voicing" segmented control
(Close / Shell / Rootless / Drop-2). Reference: **Mark Levine, _The Jazz Piano
Book_** (rootless A/B forms, drop-2); shells are the standard Bud-Powell /
comping left-hand voicing.

- **Shell (R-3-7).** Root + 3rd + 7th, 5th omitted. The 3rd and 7th are the
  *guide tones* that define the chord's quality and function; the 5th is
  harmonically inert and dropped. E.g. Cmaj7 → C-E-B. The simplest "real" jazz
  voicing and a good bridge from triads.
- **Rootless A / B forms (Bill Evans / Levine).** Omit the root (the bassist
  plays it), voice 3-5-7-9 (A form) or 7-9-3-5 (B form) in the octave around
  middle C. Verified outputs: **Dm7 A = F-A-C-E**, **G7 A = B-E-F-A** (the 13th
  substitutes for the 5th on dominants, per Levine), **Cmaj7 A = E-G-B-D**. These
  are the canonical bebop comping voicings and voice-lead ii-V-I almost entirely
  by common tone + step.
- **Drop-2.** Take a close-position 4-note chord and drop the **2nd-voice-from-
  the-top down an octave** — the standard technique for opening a close voicing
  into a fuller, playable spread. Verified: Cmaj7 close C-E-G-B → drop the G →
  **G-C-E-B**.

Non-tertian / triad qualities (sus, dim, aug, and plain triads) have no standard
rootless form, so those styles **gracefully fall back to close** rather than
inventing a voicing.

## 9. Voice-leading suggestion mode — algorithm (added, prototype v2)

A "Voicing suggestions" toggle (Learn mode, `#smoothVL`) re-renders the whole
progression so each chord is voiced to **minimise hand movement from the previous
chord's chosen voicing** — the core jazz-piano comping skill ("hold what's
already in place, move only what must move"). Design:

1. **Candidate generation** (`candidates(root, qual, level)`): for each chord,
   enumerate styles × two octave registers (C3, C4) × close-position inversions
   (lift the bottom voice up an octave, repeated). At L3 the style set is
   {close, shell, rootless A, rootless B, drop-2}; below L3 just {close, shell}.
   Candidates are clamped to a C3–B5 keyboard window so the hand can't drift off
   the board.
2. **Cost = optimal assignment, not index-matching** (`assignCost(a, b)`): the
   movement between two voicings is the cost of a **minimum-cost bipartite
   matching** between their note-sets (brute-forced; n ≤ 5). Common tones match
   at distance 0 and are thus "held"; a voice-count change costs a fixed penalty
   (`VOICE_PEN=7`) per unmatched voice. A small register-drift term
   (`0.15·|mean-pitch difference|`) breaks ties toward staying in place.
3. **Cascade** (`voiceLead(song, level)`): the first chord is anchored near
   mid-register (prefers plain close position); every subsequent chord is scored
   against the **actual** voicing chosen for its predecessor, not a default — so
   choices propagate down the whole chart.

**Verified on Autumn Leaves A (L3):** smooth voice-leading total = **28
semitones** vs **166** for the naive independent-root-position rendering — a ~6×
reduction, and the rendered hand visibly stays in one region (e.g. Cm7
C-E♭-G-B♭ → F7 as an inversion C-E♭-F-A, holding C and E♭, moving 2 voices 3
semitones total). The flow panel prints both totals so the effect is inspectable,
not just asserted.

**Stated non-solves (per CLAUDE.md rule 4):** the metric is total L1 semitone
motion with a flat voice-count penalty — it does not model hand *span*/playability
limits, thumb-under fingering, or top-voice melody constraints (a real arranger
also keeps a smooth *soprano* line, not just minimal total motion). Candidate set
is finite (2 registers, close inversions, 5 styles), so it is a strong heuristic,
not a proof-optimal global voice-leading over all enharmonic realisations.

## 10. Round-2 additions — selector-wiring fix, multi-voicing display, diatonic/altered extensions (prototype v3)

### 10.1 Voicing-selector wiring fix (was a real bug, not an algorithm error)
Last round's `voicing()` was correct in isolation but the UI never reached it below
L3: `illoVoicing()` did `const st = (currentLevel()<3) ? "close" : style`, so at the
**default Level 2** the segmented control's choice was silently discarded and every
style rendered close position. Fix: `illoVoicing` honours the selected `style` at
every level; styles that require a 7th (shell/rootless/drop-2) still fall back to
close for reduced triads inside `voicingIntervals()`, so an L1 triad reads as a plain
stack while an L2/L3 four-note chord genuinely re-voices. Re-verified the whole
click→state→render path (setting the global `voiceStyle` as the handler does and
reading the note-set `renderIllo` computes), not just `voicing()` alone: Cmaj7 →
close `C E G B`, shell `C E B`, rootless-A `E G B D`, drop-2 `G C E B` — four distinct
pitch-sets. Process note (CLAUDE.md rule 1/6): "algorithm verified" ≠ "wiring
verified" — a correct pure function can be severed from its UI; test the full path.

### 10.2 Multiple equivalent voicings per level (piece-3 extension)
Each level now surfaces **several** valid voicings of the same chord side by side
(mini-piano thumbnails), reinforcing that voicing is a choice, not a single answer:

| Level | Options shown |
|---|---|
| L1 (easy)   | same triad, two octave placements + a 1st inversion (3rd in bass) |
| L2 (medium) | close (full stack) · shell (R-3-7, drop the inert 5th) · 1st inversion |
| L3 (expert) | close · shell · rootless-A (3-5-7-9) · drop-2 (open spread) |

Cards are de-duplicated by pitch-set, so qualities where a style collapses to close
(e.g. a plain triad has no distinct shell/rootless/drop-2) show only genuinely
different cards — a Gm triad at L3 yields one card, Cmaj7 yields four. High-register
jazz voicings are rendered on a keyboard **sized to fit** (`renderVoicingFit`: nearest
C below the lowest note, enough octaves to cover the top) so nothing clips off the
fixed illustration board.

### 10.3 Diatonic-vs-altered upper extensions at the expert level (piece-4 extension)
**Key assumption (stated explicitly, per the brief).** The prototype's example data
carries no key field, so we assume the song's home key: **B♭ major / relative G
minor** — the whole Autumn Leaves A section (`Cm7 F7 B♭maj7 E♭maj7 · Am7♭5 D7 Gm`) is
diatonic to B♭ major *except* the `D7`, the secondary/functional dominant of G minor.
That single out-of-key dominant is exactly what makes altered tensions teachable here.

**Classifier (`classifyExtensions(ch, KEY)`).** The three upper-extension slots are
the natural (unaltered) intervals above the root: 9th = R+2, 11th = R+5, 13th = R+9.
For each, compare its pitch class to the key scale (B♭ major = {B♭ C D E♭ F G A}):

1. **In the scale → diatonic** (green): a safe added colour (e.g. F7's 9th = G, 13th =
   C are both in B♭ major).
2. **Not in the scale → altered** (amber): snap to the nearest scale tone (±1
   semitone); that scale tone *is* the alteration the key actually wants, named
   ♭9/♯9/♯11/♭13, with a plain-language reason. This is standard secondary-dominant
   practice: the tensions that fit the key are chromatic to the chord.
3. **Natural 11 above a major 3rd → avoid note** (grey): the natural 11 sits a
   semitone above the 3rd (a harsh clash); jazz raises it to ♯11. Flagged separately
   rather than called simply diatonic/altered.

**Worked example (verified).** `D7` in B♭ major/G minor:
- 9th natural = **E** — outside B♭ major → **♭9 (E♭)**, "borrowed for extra tension,
  the classic alteration on a V resolving down a fifth (D7→Gm)."
- 11th natural = **G** — in key, but a semitone above the 3rd (F♯) → **avoid**, use ♯11.
- 13th natural = **B** — outside B♭ major → **♭13 (B♭)**, "the ♭6 of G minor, the dark
  minor-key dominant colour."

Contrast `F7` (diatonic dominant): 9th (G) and 13th (C) both **diatonic** — showing the
learner that the *same* chord quality takes natural or altered tensions depending on
its role in the key. Panel appears only at L3 and only for chords with a 7th.

**Stated non-solves.** Single assumed key for the whole section (no per-chord
tonicization / modal-interchange analysis); the classifier reads tensions off the
prevailing key only, not the chord's own secondary key, so it labels a secondary
dominant's tensions "altered" relative to the home key (which is the intended teaching
frame here, but is a simplification of full functional analysis). 11ths are shown for
all seventh chords including where a ♯11/avoid nuance is genre-dependent.

## 11. Bass-note constraint on voicing candidates (design principle, 2026-07-21)

**User feedback (verbatim reasoning preserved):** "Chords are interchangeable in
terms of which note goes first (upper voices can be freely reordered/reregistered
across voicing candidates), BUT the bass note must stay the same as the intended
chord's bass/inversion — rootless voicings are an explicit, deliberate EXCEPTION to
this rule (their lowest sounding note is often the 3rd or 7th, not the true bass),
and should stay a specifically-documented exception rather than being generalized
into the rule."

**Practical implication:** Any future voicing-candidate generation (e.g. the
voice-leading suggestion algorithm's candidate set) must constrain the bass note to
match the chord's actual/intended bass across all non-rootless voicing styles
(close, shell, drop-2) — only the rootless-voicing candidates are allowed to deviate
from this constraint. Rootless voicings, by definition, omit the root entirely
(bassist plays it), so their lowest sounding note is explicitly the 3rd or 7th, not
the root. That exception must remain explicit in code and documentation, not
generalized into a permissive "any note in the voicing can be the bass" rule.

## Sources
- Mark Levine, _The Jazz Piano Book_ — rootless A/B voicings, drop-2, 13-for-5 on dominants; diatonic vs altered tensions on dominant chords.
- Complete Jazz Keyboard Method — close-position stacked-thirds reduction ordering.
- iReal Pro Help Center — chord diagrams (guitar/piano/uke/scales), chord symbols.
- Chord AI — App Store / Google Play listing (free vs PRO vocabulary tiers).
- Moises Guitar Chord Finder — easy/medium/advanced difficulty levels.
- Swaram — Beginner Mode auto-simplify toggle.
- Hooktheory — chord-inversion guide + slash-chord statistical study (bass≠root framing).
- TalkingBass "Simplifying Jazz Chords"; Complete Jazz Keyboard Method (reduction ordering).
