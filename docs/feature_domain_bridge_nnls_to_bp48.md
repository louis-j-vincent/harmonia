# Feature-Domain Bridge: NNLS McGill → BP48 Production

*2026-07-15. Mission: bridge research heads (trained on McGill NNLS chroma) to
the production Basic-Pitch 48-dim feature space. Experiments run **natively in
BP48**, the space every prior head investigation (#31, trigram doc) had NOT
tested — they were all on NNLS/Billboard.*

Repro: `scratchpad/{extract_bp48_absolute,bridge_experiments,plot_bridge}.py`;
data `data/cache/bp48_absolute.npz` (7350 chords / 60 songs, accomp_db synthetic
MMA jazz, absolute-frame BP48). Plot `docs/plots/bridge_nnls_bp48.png`.

## What "BP48" actually is (terminology pinned)

Production `chord_pipeline_v1` feature = **4 root-relative 12-dim chroma blocks**
= `onset ⊕ note ⊕ bass ⊕ treble` (48-dim). It is NOT a 48-bin CQT. Register is
encoded only coarsely: `bass` = onset activity MIDI<52, `treble` = onset MIDI≥60.
NNLS (Billboard, `bothchroma.csv`) = 24-dim `bass ⊕ treble`, A-referenced.

## Q1 — 12 vs 48 dim (q5 quality, root-relative, song-stratified)

| feature | dim | bal acc | dom |
|---|---|---|---|
| onset (single folded chroma) | 12 | 0.908 | 0.90 |
| onset+note (2 channels, no register) | 24 | 0.932 | 0.89 |
| **bass+treble (register split)** | 24 | **0.954** | 0.93 |
| **all 4 blocks** | 48 | **0.971** | 0.95 |

**Going 12→48 buys +6.3pp balanced.** The clean controlled result: at **equal
24 dim**, the register split (bass+treble) beats channel duplication
(onset+note) by **+2.2pp** — so the gain is *register information*, not raw
capacity. This confirms the Chord_AI / #31 "don't fold octaves away" thesis
**natively in BP48**, not just on NNLS. Keep the 48-dim production feature; do
not collapse to 12.

## Q2 — Normalization scheme (48-dim) — the decisive result

| scheme (mission label) | bal acc | dom |
|---|---|---|
| raw / absolute (no norm) | 0.811 | 0.79 |
| **C** — relative to KEY (tonic→C) | 0.819 | 0.83 |
| **A** — relative to ROOT (root→C) | **0.968** | 0.95 |

**Option A (functional root → C) wins by +15pp over Option C (key → C).**
Relative-key barely beats raw. Mechanism confirmed empirically: root-relative
puts the quality-defining intervals (b7 vs maj7, m3 vs M3) at **fixed indices**
the head specializes on; key-relative leaves them scattered across 12 positions
by chord degree — the model would have to learn "12 different rules," exactly
the failure the mission asked about. **This is the single most important bridge
decision and it is unambiguous: normalize by functional root, not by key.**

**Option B** ("bass note of functional root → C") ≡ Option A for non-inversion
chords (bass root = functional root). B only diverges on slash chords/inversions,
where it requires a **bass head** to identify the sounding bass. B is the right
frame *only once a bass/inversion head exists*; until then A and B are identical.
Recommendation: ship A now; B is A + a bass head (see voicing proposal).

## Q3 — The bridge: can NNLS-trained heads transfer to BP48?

| train → test | bal acc | dom |
|---|---|---|
| BP48 → BP48 (native, upper bound) | 0.923 | — |
| NNLS → NNLS (in-domain sanity, matches known ~0.74) | 0.745 | — |
| **NNLS → BP48 (port weights, raw)** | 0.765 | 0.86 |
| NNLS → BP48 + per-dim z-norm "bridge" | 0.673 | 0.71 |

Two findings:
1. **Native BP48 beats a ported NNLS head by ~16pp.** The transferable asset is
   the **recipe** (root-relative + register split + structured heads), NOT the
   weights.
2. **A learned/statistical feature-alignment transform HURTS.** Per-dim
   standardization dropped cross-domain −9pp. Root-relative + per-block L2
   already normalizes the geometry; a linear re-standardization distorts it.
   Do **not** build an NNLS→BP48 feature transformer.

### Why we can't build a "true" paired bridge anyway
McGill Billboard ships no audio we can push through Basic Pitch (known_issues
#31: "BP48 transfer BLOCKED — no Billboard audio"). So NNLS↔BP48 paired data
does not exist. Combined with finding (2), the feature-transform path is both
impossible and undesirable.

## RECOMMENDATION — what bridge to build

**Retrain the recipe natively in BP48; discard NNLS weights.** Concretely:
1. **Feature:** keep production 48-dim (onset⊕note⊕bass⊕treble), per-block L2,
   **root-relative** (Option A). Confirmed best here.
2. **Corpora (BP48-native, already have both audio + labels):** accomp_db
   synthetic MMA (this experiment) + Mission-2 real corpus_50 (`quality_head_v1`)
   + YouTube/iRealb. Billboard/NNLS becomes a **teacher for the architecture and
   the root/majmin priors only**, never a weight source.
3. **Heads:** port the #31/trigram *architecture* (root MLP → root-relative
   quality head with top-k root marginalization + learned trigram context) into
   BP48 by re-extracting features and retraining. No new model design needed —
   just re-fit in this space.
4. **Do NOT** attempt a feature-space transform or z-norm alignment (Q3).

## CRITICAL CAVEAT — synthetic ceiling

All BP48 numbers here are on **clean synthetic MMA renders**; Basic Pitch is far
cleaner on these than on real audio. Absolute values are optimistic (dom 0.95
here vs real-audio dom 0.21 in #19). **The relative comparisons (Q1/Q2/Q3) are
the valid signal; the absolute accuracies are not a real-audio forecast.** Re-run
Q1/Q2 on the Mission-2 real corpus_50 BP48 cache before quoting any absolute
number. The ordering (48>24>12; A≫C≫raw; native≫ported) is expected to hold —
it matches the independently-derived NNLS results — but confirm on real audio.

---

## Q4 — Expert Voicing Model (proposal)

**Goal:** learn complex-chord voicings (7#11, 6/9, add#11, altered dominants) from
the *complete unaltered* chroma — chords the flat 5-way q5 head structurally
cannot express.

### Why the current heads can't do it, and why 48-dim isn't enough for this
The q5 head outputs one of {maj,min,dom,hdim,dim}. 7#11 vs 9 vs 13 differ by
which **upper extensions** sound (#11, 9, 13) — pitch classes a *root-relative
12-dim collapse still contains*, but the 4-block BP48 has only 2 registers
(bass<52 / treble≥60), too coarse to place a #11 in its actual octave. For
voicing you want the **finer register grid the folded blocks threw away**. This
is exactly the Chord_AI/ChordFormer point (`chord_ai_reverse_engineering.md`):
CQT at 24–36 bins/octave, not folded chroma.

### Proposed architecture — a separate, LATE head (not integrated into q5)
Keep it **separate** from root/quality. Rationale: extensions are (a) rare
(long-tail, would destabilize the balanced q5 head), (b) only meaningful once
root is known, (c) a *multi-label* problem (a chord has several extensions at
once), not the single-label q5 softmax.

```
audio → Basic Pitch 88-key roll (do NOT fold)
      → root head (existing, 89% / BP48-retrained)   ── gives root r
      → rotate the 88-roll to root-relative pitch-CLASS×register grid
        (24 or 36 bins/oct CQT is better; 88-roll is the free fallback)
      → VOICING HEAD: MLP/small-transformer, 12–14 independent sigmoids
        = P(scale-degree present) for {b3,3,4,#4/b5,5,#5/b6,6,b7,7,9,#9,b11,13}
        supervised as a root-relative pitch-class BITMAP
      → assemble label: root + triad(from q5) + active-extension sigmoids
```

This is the **McFee/Bello "pitch-class quality bitmap" head** made explicit:
independent sigmoids over root-relative degrees, trained multi-label with
per-degree pos-weights (extensions are sparse). It shares the root frame with
the quality head (Option A), so the b7/maj7/#11 contrasts land at fixed indices.

### Where it plugs in
- **Emission-time, after root, in parallel with q5.** q5 gives the triad+7th
  backbone (high accuracy, common classes); the voicing bitmap *adds* extensions
  on top. Assembly rule: start from q5's triad/7th, then attach any extension
  whose sigmoid > threshold (calibrated per-degree). Conflicts (e.g. bitmap says
  no-b7 but q5 says dom) resolved in q5's favor for the backbone, bitmap only for
  9/11/13 upper structure.
- Keeps q5 stable (don't dilute it with rare classes) while unlocking the Pro-tier
  vocabulary Chord_AI advertises (7#11, 6/9, 9, 13, add#11).

### Data / feasibility
- **Voicing supervision needs inversion/extension-preserving labels.** POP909
  discards them; MMA/iRealb renders **have exact MIDI** → a perfect voicing bitmap
  is free from the rendered corpus (we already read `perfect_chroma` in
  `build_audio_chord_features.py`). Train on rendered voicings first (piano),
  validate on real audio.
- Piano-first (mission ask): render standard jazz-piano voicings (rootless A/B,
  drop-2) via MMA, extract per-degree bitmaps — this directly teaches the
  "standard voicing shapes" the mission wants.
- **Requires octave-preserving features** to reach its ceiling: use the 88-roll
  (already extracted) or add a CQT front-end. The 4-block BP48 is a floor, not
  the target, for this head specifically.

### Recommendation
Build the voicing bitmap head **separate and downstream** of root+q5, on
root-relative octave-preserving features (88-roll now, CQT later), supervised by
rendered-MIDI voicing bitmaps. It is additive: zero risk to the shipped q5/root
heads, and it is the only path to the 7#11/6-9/13 vocabulary. Sequence it
**after** the BP48 retrain of root+q5 (this doc's main recommendation) lands.

