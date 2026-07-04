# Scale taxonomy & chord-progression analysis — full reference (2026-07-03)

Detailed reference for a multi-step investigation into how to represent
scales/chords so that chord-progression pattern-mining generalizes properly,
rather than needing a "thousand small rules." Companion to
`docs/known_issues.md` #1 (bass-informed chord reconstruction) and
`docs/architecture_extensions.md` items #9-11 (this doc is the detailed
backing material those items now point to). All code:
`scripts/scale_taxonomy.py` (the taxonomy itself) and
`scripts/plot_structure_proposal_illustrations.py` (all analyses/plots
referenced below). All numbers are from the full 909-song POP909 corpus
(symbolic annotations only, no audio) unless stated otherwise.

---

## 1. What "diatonic" means

A note or chord is **diatonic** to a given 7-note scale if it can be built
entirely from that scale's own notes, with no pitches from outside the
collection. For a **chord** specifically (the sense used everywhere in this
document): stack thirds on a scale degree using *only* notes already in the
scale, and whatever triad quality falls out is "the" diatonic chord for that
degree — e.g. stacking thirds on the 3rd degree of a major scale always
produces a minor triad, not because of an arbitrary rule but because the
scale's own 3rd, 5th, and 7th degrees happen to form a minor-third-plus-
perfect-fifth interval pattern. **Diatonic membership is a joint property of
(scale degree, chord quality) together** — you can't ask "is this chord
diatonic" without asking "diatonic to which scale, and does the quality
match what that scale's own degree would produce." A major triad built on
the 3rd degree of a major scale is *not* diatonic to that scale (the user's
example) even though scale degree 3 itself is a member of the scale — the
quality is wrong for that position. "Chromatic" is the complement: using a
note, or building a chord quality, that scale membership doesn't produce.

---

## 2. The atomic scale taxonomy

**Core idea:** the 7 "church modes" (Ionian, Dorian, Phrygian, Lydian,
Mixolydian, Aeolian, Locrian) are not 7 different scales — they are the same
7-note pitch-class *collection*, with a different member of that collection
picked out as tonic. "D Dorian" and "C major" (C Ionian) are the literal
same 7 notes. So the only genuinely atomic, transposition-covering objects
are:

1. A small number of distinct note **collections** ("families"), each
   defined once by its interval pattern from an arbitrary reference point
   (that family's own "mode 1", by convention — not because mode 1 is
   privileged). Each family has 12 transpositions (fewer for symmetric
   ones, see below).
2. Within a family+transposition, a **modal centre** — which member of the
   collection currently functions as tonic. This is separate, softer, more
   slowly-varying state, *not* part of the collection's identity.

**Consequence used throughout:** whether a chord is diatonic depends only on
(which family, which transposition), never on the modal centre. One
diatonic-triad-by-position table per family, built once, covers every mode
of that family simultaneously.

### Families implemented (with working diatonic-triad tables)

**Major family** — `[0, 2, 4, 5, 7, 9, 11]`. The 7 modes (Ionian through
Locrian) are just this collection read from 7 different starting points.
Diatonic triads by position (derived programmatically, not hand-typed —
verified in `scripts/scale_taxonomy.py`'s self-test):

| interval from mode-1 (Ionian) tonic | 0 | 2 | 4 | 5 | 7 | 9 | 11 |
|---|---|---|---|---|---|---|---|
| triad quality | maj | min | min | maj | maj | min | dim |
| roman numeral | I | ii | iii | IV | V | vi | vii° |

Natural minor (Aeolian) is mode 6 of this same family — tonic sits at
interval 9 above the family's Ionian reference (equivalently, the Ionian/
"relative major" reference sits at interval 3 above a natural-minor tonic).
**Verified directly:** re-indexing this ONE table at +3 reproduces exactly
the natural-minor-by-its-own-tonic table `{0:min, 2:dim, 3:maj, 5:min,
7:min, 8:maj, 10:maj}` that an earlier, less general pass at this analysis
had built by hand as a seemingly-separate table.

**Harmonic minor family** — `[0, 2, 3, 5, 7, 8, 11]` (natural minor with a
raised 7th). Modes: Harmonic minor, Locrian ♮6, Ionian ♯5, Dorian ♯4,
**Phrygian dominant**, Lydian ♯2, Superlocrian ♭♭7. Diatonic triads:

| interval from mode-1 tonic | 0 | 2 | 3 | 5 | 7 | 8 | 11 |
|---|---|---|---|---|---|---|---|
| triad quality | min | dim | aug | min | maj | maj | dim |

Mode 1's own tonic sits at the family's reference point directly (no
transposition needed relative to a minor tonic) — unlike the major family,
where the commonly-used minor mode (Aeolian) is offset from the reference.

### Families documented but NOT yet implemented (no working membership check)

Deferred rather than guessed at, given POP909 is a pop corpus where these
are expected to be rare — flagged here so the taxonomy is complete even
where the code isn't:

- **Melodic minor family** `[0,2,3,5,7,9,11]` — modes include Dorian ♭2,
  Lydian augmented, **Lydian dominant** (mode 4) and the **Altered scale**
  (mode 7, "Superlocrian") — both real, common jazz sounds, low expected
  prevalence in pop.
- **Whole tone** `[0,2,4,6,8,10]` — 6 notes, symmetric under transposition
  by 2 semitones, only 2 distinct transpositions exist. No conventional
  maj/min/dim/aug triads build from it (every stacked-third triad is
  augmented).
- **Octatonic** (whole-half `[0,2,3,5,6,8,9,11]` and half-whole
  `[0,1,3,4,6,7,9,10]`, genuinely different pitch-class sets, not modes of
  each other) — 8 notes, symmetric under transposition by 3 semitones, only
  3 distinct transpositions of each.

---

## 3. Findings, in the order they were established (including a correction)

### 3.1 Diatonic-membership taxonomy, "parallel" framing

`classify_membership(interval, quality, song_mode)` (in `scale_taxonomy.py`)
checks a chord (expressed as interval from the song's OWN annotated tonic)
against: (a) the major family anchored at the song's own tonic if
major-annotated, or tonic+3 if minor-annotated ("diatonic_own"); (b) the
major family anchored at the OTHER position ("parallel_borrow" — e.g. a
borrowed `iv`, `bVI`, `bVII` in a major-key song, or a borrowed major `IV`/
`bIII` in a minor-key song); (c) the harmonic-minor family at the song's own
tonic, no shift ("harmonic_minor_borrow"); (d) "sus" (no third, doesn't
structurally conflict with anything); else "chromatic".

**Corpus-wide breakdown** (`taxonomy_overview.png`, all 909 songs, checked
against each song's own annotated tonic):

| | major-annotated (n=65,947 chords) | minor-annotated (n=54,122 chords) |
|---|---|---|
| diatonic to own mode | 80.8% | 77.9% |
| parallel-mode borrow | 3.5% | 11.3% |
| harmonic-minor-only borrow | 0.0% | 0.0% |
| sus (neutral) | 8.6% | 7.5% |
| chromatic (neither) | 7.1% | 3.2% |

**Asymmetry, unresolved:** minor-annotated songs borrow from the parallel
major over 3x more than major-annotated songs borrow from the parallel
minor (11.3% vs 3.5%). Could be genuine (minor-key pop borrows major color
more than the reverse) or an artifact of POP909's automatic key labeling
being less reliable specifically on minor keys (consistent with the
session-7 finding that relative-major/minor confusion is a known
Krumhansl-Schmuckler failure mode). Not resolved here.

**Correction made mid-investigation:** the harmonic-minor family was
hypothesized to meaningfully re-explain a chunk of the "chromatic" bucket
(specifically the raised-leading-tone V/V7/vii° chords common in minor-key
cadences). Checked directly, position by position, against the already-
correct `parallel_borrow` table: **the V, V7, and vii° chords produced by
harmonic minor's raised 7th are pitch-identical to the parallel major's own
V/V7/vii°** — raising natural minor's 7th degree to create a proper leading
tone is *the same operation*, note for note, as borrowing the parallel
major's leading tone. There is no way to distinguish "this V7 is borrowed
from the parallel major" from "this V7 is harmonic minor's own diatonic V7"
— they're the same chord under two different theoretical framings. Verified
exhaustively: of harmonic minor's 7 diatonic positions, only the augmented
mediant (`III+`, position 3) is NOT already covered by `diatonic_own` or
`parallel_borrow` — confirmed by running `classify_membership` against
every harmonic-minor-family position and checking which category claims it
first (see `scripts/scale_taxonomy.py`'s inline check). `harmonic_minor_
borrow` is consequently ~0.0% of the corpus in practice — real, but a rare
augmented-mediant chord, not the significant reclassification first
hypothesized. This means an earlier description (this session, prior turn)
of the minor-key raised-7th cadence as a distinct "harmonic minor" category
separate from parallel-mode borrowing was imprecise; it should be described
as parallel-mode borrowing throughout.

### 3.2 Canonical (relative-major-referenced) bigram pooling

Computing each chord's scale degree relative to the song's own annotated
tonic (regardless of major/minor) fragments genuinely-identical relative-
pitch events: a minor key's `v→i` cadence (intervals 7→0 from its own
tonic) and a major key's `iii→vi` motion (intervals 4→9) are the *same*
relative-pitch event once you account for the shared 7-note collection —
verified directly (E→A in A minor: intervals 7→0 from A; intervals 4→9 from
C, A minor's relative major). Canonicalizing every song's reference tonic to
its relative-major tonic (tonic+3 for minor-annotated songs, unchanged for
major-annotated) before computing scale degrees, then pooling bigrams
**separately by the song's original annotated mode** (to check whether
canonicalizing actually unifies the language, rather than just asserting
it), gives:

- `III(min)→VI(min)` (the canonicalized natural-minor `v→i`): 3.23% of
  major-annotated songs' transitions, 3.95% of minor-annotated songs' —
  comparable prominence in both, confirming the fragmentation-fix works.
- `V(maj)→I(maj)`: 8.44% major-annotated vs 4.13% minor-annotated — did
  *not* transfer evenly, but for a real reason: in a minor-annotated song's
  canonical frame, degree `I` isn't the song's actual tonal centre (`VI`
  is), so "approach to `I`" is structurally less central there.
- The closest minor-key equivalent of a real authentic cadence,
  `IIImaj→VImin` (harmonic-minor/parallel-major-borrowed `V→i`): 3.01% of
  minor-annotated songs' transitions, rank 7, essentially absent from the
  major-annotated group's top patterns — a real, distinct minor-specific
  idiom, not an artifact.

See `ngram_illustration_canonical_major_vs_minor.png`.

### 3.3 Taxonomy folded into the canonical bigrams + chromatic-only view

`ngram_by_category.png`: the same canonical top-15 bigrams, each bar
coloured by its more-exotic member's category. Only 1 of the top 30 bars
(both groups combined) is non-diatonic (`IIImaj→VImin`, correctly green/
parallel-borrow) — the dominant patterns are overwhelmingly plain diatonic
motion; chromatic/borrowed chords are individually too rare to crack a
frequency-sorted top-15 even though they're collectively ~7-11% of all
chords.

`ngram_chromatic_only.png` fixes that by filtering to non-diatonic bigrams
only: 33.2% of major-annotated transitions and 37.9% of minor-annotated
transitions involve at least one non-diatonic (sus, borrowed, or chromatic)
chord (bigram-level rate is higher than the per-chord-event chromatic rate
above, since a bigram counts as non-diatonic if *either* member is). Major-
annotated songs' non-diatonic content is dominated by sus-chord suspension/
resolution patterns (`Iother→Imaj` etc., mundane) with a smaller genuinely-
chromatic tail (`IIImaj→VImin`, `IImaj→Vmaj` — secondary dominants).
Minor-annotated songs' non-diatonic content is dominated by the parallel-
major-borrow family (`IIImaj→VImin` and several variants involving `IIImaj`
in other contexts).

### 3.4 Are 7ths a real differentiator? (empirical, not assumed)

Tested directly rather than assumed: among chords with "maj"-family triad
quality (which a coarse maj/min bucket would lump together), does the
*specific* raw quality — plain triad vs `maj7` vs `dom7` — predict whether
the chord resolves down a perfect fifth to the next chord (the functional-
dominant signature)? `seventh_differentiation.png`:

| group | n | P(resolves down a 5th) |
|---|---|---|
| chromatic `maj7` | 141 | 12.1% |
| diatonic V `maj7` | 35 | 22.9% |
| ANY chord (baseline) | 106,391 | 27.6% |
| chromatic (plain triad) | 3,395 | 37.3% |
| diatonic V `7` (dom7) | 601 | 49.9% |
| diatonic V (plain triad) | 12,940 | 52.4% |
| chromatic `7` (dom7) | 296 | 53.4% |

**Yes — `dom7` is a real, position-independent differentiator.** A
chromatic `dom7` chord resolves like a dominant (53.4%) at essentially the
same rate as the primary diatonic V itself (52.4%), regardless of scale
position. A chromatic `maj7` chord resolves like a dominant only 12.1% of
the time — barely different in kind from a `maj7` sitting on the actual
diatonic V (22.9%, small n). The specific 7th tells you more about a
chord's function than its scale position does.

### 3.5 Mode-agnostic parent-scale identification (validated)

Decomposes the 24-way (12 keys × major/minor) — or, properly, up-to-84-way
(12 × 7 modes) — key-finding problem into two separably-easier
sub-problems: "which 7-note collection" (12-way) and "which degree is
home" (harder, ≤7-way, not solved here — see §4). `identify_best_parent_
scale()`: for each song, score every major-family transposition T by the
fraction of the song's real chord events whose (root, quality) exactly
matches that T's diatonic-triad table; take the argmax.

**Validated against the GT-implied collection (major tonic, or tonic+3 if
minor) using nothing but chord content — no key_audio.txt lookup at all
during identification:**

| | agreement with GT |
|---|---|
| overall | 866/909 songs (95.3%) |
| major-annotated | 491/502 (97.8%) |
| minor-annotated | 375/407 (92.1%) |

Mean within-song diatonic-triad match rate at the identified T: 81.2%. See
`parent_scale_identification.png` for the full distribution (right-skewed,
most songs 0.7-1.0). The same major/minor asymmetry as §3.1 shows up here
too (97.8% vs 92.1%) — consistent with, not independent evidence for, that
open question.

### 3.6 Atomic (fully mode-agnostic) bigrams + cross-scale tracking

Pools bigrams from ALL 909 songs using each song's own *algorithmically
identified* best-fit T (§3.5), not the GT tonic/mode — genuinely mode-
agnostic pooling, since nothing about the labeling assumes which collection
member is felt as home. Chords whose root isn't even a member of the
identified 7-note collection are tracked separately as "cross-scale" —
candidate real modulations, a strictly stronger criterion than "chromatic"
in §3.1 (a `V/vi` secondary dominant's root is still IN the collection, so
it does not count as cross-scale here; only genuinely foreign notes do,
directly matching the "if a chord that is not in a scale pops out, that
means we are now in a different scale" framing).

**Result:** 6.2% of all 106,391 chord-to-chord transitions are cross-scale
(root not in the identified collection at either end). The remaining 93.8%
pooled bigram table (`atomic_bigrams.png`) is topped by `V→I` (6.98%,
diluted relative to the major-only 8.44% in §3.2, since this pools in
minor-annotated songs' analogous-but-differently-labeled motion too) and
`IV→V` (6.05%).

---

## 4. Open / not yet done

- **Modal-centre inference beyond "trust the annotated major/minor label as
  a two-way prior".** §3.5 solves the collection-identification half
  robustly; *which* of the 7 positions within that collection is actually
  felt as home (Ionian vs Aeolian vs Mixolydian vs Dorian, etc.) still just
  defaults to the GT-annotated major/minor choice, which itself only ever
  distinguishes 2 of the 7 possible modal centres. A real inference step
  (e.g. weighting each candidate centre by total duration/frequency of the
  chord built on it, first/last chord of the piece, etc.) is undesigned —
  flagged, per the original request, as "a problem in itself."
- **Melodic-minor, whole-tone, octatonic membership tables** — documented
  (§2) but not implemented; expected low-prevalence in POP909 specifically,
  unverified.
- **The major/minor parallel-borrow asymmetry** (§3.1, §3.5) — real,
  measured twice independently, not explained.
- **Duration-weighting.** All membership/identification analyses here use
  per-EVENT counts, not duration-weighted ones — a single long chord and a
  single short one count equally. Not checked whether this matters.

Nothing in this document is wired into `harmonia/models/chord_hmm.py` or any
other part of the real pipeline — purely exploratory/analytical, same status
as the rest of `docs/architecture_extensions.md`.
