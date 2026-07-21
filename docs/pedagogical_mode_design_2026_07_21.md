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

## Sources
- iReal Pro Help Center — chord diagrams (guitar/piano/uke/scales), chord symbols.
- Chord AI — App Store / Google Play listing (free vs PRO vocabulary tiers).
- Moises Guitar Chord Finder — easy/medium/advanced difficulty levels.
- Swaram — Beginner Mode auto-simplify toggle.
- Hooktheory — chord-inversion guide + slash-chord statistical study (bass≠root framing).
- TalkingBass "Simplifying Jazz Chords"; Complete Jazz Keyboard Method (reduction ordering).
