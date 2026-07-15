# Chord vocabulary alignment — iRealb ↔ Billboard ↔ unified Q5

*2026-07-15. Task 3 of the corrected-GT mission. Companion to
`docs/billboard_retraining_findings.md` and issue #31.*

## Why this exists

The three symbolic sources speak different chord dialects. To train or evaluate one
model across them without a dataset-of-origin bias, every label must collapse to a
single target vocabulary. Harmonia's modelling target is the **5-way quality (Q5)**
`{maj, min, dom, hdim, dim}` plus a 12-way **root** (pitch class, inversions
discarded — CLAUDE.md rule 3: POP909/iReal "root" is functional, not sounding).

## The three dialects

| Source | Notation | Root spelling | Quality tokens | Encodes bass/inv? |
|---|---|---|---|---|
| **Billboard (McGill)** | Harte (`full.lab`) | `C`, `Bb`, `F#` | `maj min 7 min7 maj7 sus4(b7,9) hdim7 dim …` | yes, `X/b7` — dropped |
| **iRealb** | iReal shorthand (`IREAL_TO_MMA` keys, 70 tokens) | `C`, `Bb`, `F#` | `^ - h o 7 -7b5 7alt 69 …` | slash `/G` — dropped |
| **POP909** | root:family (parser) | pc int | maj/min/dom families | no (`/bass` already stripped in GT) |

The **same audible chord** is spelled three ways, e.g. a half-diminished:
Billboard `C:hdim7` · iRealb `Ch7` / `C-7b5` · unified **`hdim`**.
A dominant-7: Billboard `G:7` · iRealb `G7` · unified **`dom`**.

## Unified mapping → Q5

### Billboard Harte → Q5 (`harte_to_q5`, verified on all 114,741 `full.lab` chords)
Rule order (first match wins), after stripping inversion (`/…`) and root:
- contains `hdim` or `min7b5` → **hdim**
- base starts `dim` → **dim**
- base starts `minmaj` → **min**; starts `maj`/`aug`/`+` → **maj**; starts `min`/`-` → **min**
- base ∈ `{7,9,11,13}` → **dom**
- `sus…` → **dom** if `b7` present else **maj**
- base ∈ `{5,1,""}` (power chord / bare root) → **dropped** (no third)
- else → **maj**

Observed Q5 distribution (Billboard, corrected): maj 73,072 · min 26,182 ·
dom 14,975 · hdim 200 · dim 312.

### iRealb shorthand → Q5 (all 70 `IREAL_TO_MMA` tokens)
| Q5 | iReal tokens |
|---|---|
| **maj** | `(maj)` `^` `^7` `^9` `^13` `^7#11` `^7#5` `^9#11` `6` `69` `2` `5` `+` `add9` `sus` `susadd3` |
| **min** | `-` `-6` `-7` `-9` `-11` `-69` `-#5` `-b6` `-7#5` `-^7` `-^9` |
| **dom** | `7` `9` `11` `13` `7b9` `7#9` `7#11` `7b5` `7#5` `7alt` `alt` `13b9` `13#11` `9sus` `7sus` `7susb9` `7b13` … (all root-position 7th-family + altered dominants) |
| **hdim** | `h` `h7` `h9` `-7b5` |
| **dim** | `o` `o7` `o^7` |

## Key differences that would bias a naive merge

1. **`aug` and `sus`.** Billboard folds augmented and plain-sus into **maj** (no
   dominant 7th). iRealb `+`/`sus` likewise → maj, but `7sus`/`9sus` → dom. Both
   dialects now agree under the rules above — *verify this stays true if either
   mapper changes.*
2. **Diminished vs half-diminished.** Billboard writes them out (`dim7`, `hdim7`);
   iReal uses `o`/`h`. Easy to mis-route `h`→dim; the mappers keep them distinct.
3. **Class balance is wildly different, and that is the real bias.** iRealb (jazz)
   is dom/min7/hdim-rich; Billboard (pop) is 64% maj, dom ~13%, hdim/dim <0.5%.
   Merging *feature* corpora would let Billboard's maj mass swamp the jazz 7th
   signal. Any merge must **stratify/re-weight by Q5**, not pool raw.

## Feature-distribution matching & merge — status: BLOCKED (documented, not skipped)

The mission asked to z-normalise NNLS/BP chroma across iRealb + Billboard and save a
merged `combined_training_corpus_normalized.npz`. This is **not sound to ship yet**,
for a load-bearing reason (CLAUDE.md rules 1 & 6):

- **iRealb has no chroma of its own** — it is a symbolic chart corpus. Its only
  audio-feature form is MMA-rendered MIDI (a synthetic timbre) or the real-audio
  YouTube corpus (issue #19), which is extracted in the **48-dim Basic-Pitch**
  domain.
- **Billboard features are McGill NNLS chroma.** NNLS-Chroma and Basic-Pitch chroma
  are different feature spaces (different spectral whitening, octave handling,
  reference pitch). z-normalising each to N(0,1) **hides but does not remove** the
  domain gap — a model trained on the pooled set learns a blend of two sensors, and
  the findings doc already identifies this feature-domain gap as *the* blocker to
  putting any Billboard model into `chord_pipeline_v1`.

**Correct action:** keep vocabulary unified (done, above — this half *is* valid and
load-bearing), but do **not** merge feature distributions across the NNLS↔BP
boundary. Merge only *within* a feature domain: (a) all-NNLS (Billboard only), or
(b) all-BP (corrected-iRealb-render + YouTube), each z-normalised within itself.
Producing a cross-domain `combined_…npz` now would be a silent-calibration trap of
exactly the kind rule 1 warns against — so it is deliberately not written.
