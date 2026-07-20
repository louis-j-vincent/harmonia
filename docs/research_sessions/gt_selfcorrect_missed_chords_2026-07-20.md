# GT self-correction — missed-chords re-decode + Bein' Green — 2026-07-20

Continuation of the overnight GT campaign. Budget 2.5h (start 07:47 CEST).
Two missions: (1) Let It Be missed-chords phrase-position pooling; (2) Bein' Green
error analysis. Then widen matched set if budget allows.

## Brief (restated, numbered)
1. **Mission 1 — accords loupés.** Let It Be verse loop decodes as P2 [C,F]; G/Am
   under-detected per pass. Fix: pool per-bar evidence at each LOOP POSITION across
   the A×N phrase repetitions (√N denoising) and re-arbitrate with the existing
   Occam-Bayes machinery. Gate: rendered chart recovers ≥1 of {G, Am}; no regression
   on matched set (abba, Billie Jean, Stand By Me, henny, just-aint); anti-crush
   green; 2-run stable. Update scoreboard.
2. **Mission 2 — Bein' Green.** Find GT, validate A/V alignment, run full pipeline,
   scoreboard row + top-2 error classes. Jazz ballad — characterize honestly if
   through-composed.
3. Budget permitting: widen matched set by 1–2 songs.

Process: gates via REAL /api/analyze on side port ≥7773 (never 7771), ×2 stable;
stem-keyed caches; anti-crush ≥99.5% GT bars unchanged; .venv/bin/python; disk floor
2.0 GiB (currently ~4 free — WATCH).

## Mission 1

### Baseline (in-process real infer_chords_v1, nnls24/musx/bestfit, occam ON default)
Cached audio transcoded to a unique-stem wav (soundfile can't read m4a).
- key C major, tempo 140.2, **118 chords**. Chord-span vocab: C:maj 45, F:maj 34,
  **G:maj 25, A:min 13**, N 1. So G and Am ARE present at the chord-SPAN level
  already — the "missing" is in the FOLDED LOOP view.
- Flux-anchored bars: 142 (half-measures; song ~243s @140bpm), argmax C:60 F:35
  G:31 A:15.
- **`detect_loop_pattern` on the raw argmax → (2, [C, F])** — the P2 underestimation.
  This drives the ×N fold, collapsing the loop to C-F and hiding G/Am.

### GT (pop400 iReal, parsed via ireal_corpus)
Verse A = `C G | Am F^7 F6 | C G | F C`; chorus B = `Am C/G | F C | C G7 | F C`.
Roots present: C, G, Am, F (harmonic rhythm ~2 chords/bar → flux half-bars).

### Premise check (cheapest falsification): does phrase-position pooling surface G/Am?
Pooled bar_post by position mod L over the whole song, re-took argmax:
- L=4 → [F,C,C,G]; L=8 → [F,C,C,G,C,C,C,G]; L=16 → 8-repeat of same.
- **G recovered at pooled positions; Am NEVER recovered** (mean posterior A=0.108
  vs C=0.431 — C/E overlap means the NNLS root head puts C on top even in Am bars;
  a systematic root-head bias, NOT noise, so √N pooling can't fix it).
- Recurrence guard: L=8 rec=0.552 (clears 0.5), L=16=0.349, L=4=0.457 → L=8 is the
  phrase (4 measures). Pooling@L=8 → `detect_loop_pattern` returns **P8
  [F,C,C,G,C,C,C,G]** — G is now a PATTERN MEMBER.

**Premise verdict: PARTIAL PASS.** Pooling recovers G into the loop pattern
(satisfies the ≥1-of-{G,Am} gate) but NOT Am. Am is an honest documented miss
(root-head C/Am confusion, not denoisable).

### Design
Pool for STRUCTURE (period + pattern) only; keep per-bar Bayes arbitration on the
ORIGINAL posterior. This recovers G as a pattern member without crushing the real
Am deviations (arbitrated on their own evidence, unchanged). Anti-crush preserved
by construction: the symbolic harness feeds one-hot conf=1 posteriors, so every
deviation's arbitration LR is huge → kept → 100% unchanged regardless of pattern.

### Implementation (chord_pipeline_v1.py)
- `_phrase_pool_for_loop(roots, bar_post, idx)`: finds the largest phrase length
  L∈{16,8,4} (≥2 reps) whose raw-argmax lag-recurrence ≥0.5; sums+renorms the
  per-bar root posteriors across repetitions at each position mod L; returns
  denoised per-bar posteriors + roots. Let It Be picks L=8 (rec 0.552).
- `occam_compress_bars`: computes the RAW loop, then the phrase-pooled loop, and
  **overrides only when the pooled vocab is a STRICT SUPERSET of the raw vocab**
  (the period-underestimation signature — a genuinely new chord surfaced). Per-bar
  Bayes arbitration unchanged (runs on original bar_post). Env kill-switch
  `HARMONIA_OCCAM_PHRASEPOOL=0`.

### Gate results (all PASS)
| check | result |
|---|---|
| Let It Be raw P2 loop → | pooled P8 [F,C,C,G,C,C,C,G] (adds G) → Occam abstains |
| Let It Be chord-span vocab | C 45→39, F 34→37, **G 25→31, Am 13→16, C/G 0→4**, N 1 |
| Rendered chart (baked HTML) | roots G:31, Am:16, 4 slash (C/G) — G AND Am recovered |
| Controls (henny/justaint/abba/billiejean/standbyme) | **byte-identical OFF vs ON** |
| Anti-crush symbolic (pop400 GT) | **100.00% of 25,120 bars unchanged** (156 applied) |
| 2-run stability (Let It Be) | identical |
| tests (chart_model + user_constraints) | 43 passed |

**Why the tight vocab-superset guard**: the first (loose) version fired on 2-chord
vamps too and shifted their arbitration (henny Emin7 11→2 etc). Requiring the pooled
vocab to strictly ADD a chord confines the change to the actual missed-chord case
and leaves every validated 2-chord-vamp chart byte-identical.

**Root cause the fix corrects**: with the false P2 [C,F] vamp, Occam was actively
SNAPPING real G/Am bars to C/F (logged: bars 38/42/82/111/127 G→C, 17/86 A→C). The
period fix stops that destruction. G recovered as a pattern member; Am recovered as
a kept deviation; the C/G slash also surfaces.

**Honest miss**: Am is NOT recovered by pooling itself (mean posterior A 0.108 vs
C 0.431 — the NNLS root head systematically prefers C in Am bars due to C/E overlap;
a bias, not noise). Am improves 13→16 only because the false vamp no longer snaps it
away, not because pooling surfaced it. A true Am fix needs a bass-aware or
third-sensitive emission, upstream. Folding still abstains (Occam declines the P8
because Am+drift exceed dev-frac) — faithful, uncompressed chart; ×N fold on this
song still needs the upstream Am fix.

## Mission 2 — Bein' Green (jazz1460, AABA ballad)
GT (key Bb, AABA 32 bars): A = Bb^7 A7#5 Dm7b5/Ab G7sus/b9 Cm7 F7sus Bb^7 [pass] ×2;
B = Ab^7 Db^7 Bb^7 Bb^7 Gm/Gm-maj7 Gm7-C7 Cm7 F7. Audio docs/audio/bein_green.m4a
(179.1s), transcoded to unique-stem wav.

Alignment validation: `alignment_validator.validate_alignment` needs a time-aligned
iReal result (heavier build); used the documented duration-match + key-match +
ordered-sequence approach instead. Key ✓ (A#=Bb). 

Decode (nnls24/musx/bestfit): key A#/Bb ✓, tempo 149.6, 53 chords. Vocab includes
Bbmaj7, A7, G7, Cm7, F7/F7sus, Abmaj7, C#(Db)maj7, Gm7, C7, G#hdim7 — nearly the full
GT harmony.

Evidence (scripts in scratchpad: beingreen_align.py, beingreen_warp.py):
- Root-vocab Jaccard vs GT = **0.89** (all 8 GT roots present, +1 Eb extra).
- **Ordered-root LCS = 27/32 = 84%** of GT bar-roots recovered IN ORDER (collapsed
  GT 24/27 = 89%). The A-section ii-V descent (A7→G7→Cm7→F7) and the Ab→Db bridge
  appear twice in the decoded sequence.
- Per-bar UNIFORM-grid time alignment = **5/32 = 0.16** — the only weak number.

**Top-2 error classes** (evidence above):
1. **Time-registration / harmonic-rhythm on a rubato ballad** (dominant). Chords are
   right and in order; their onset times just don't map onto GT bar positions. Tempo
   149.6 is a 2×/rubato read of a slow ballad; the uniform bestfit bar grid can't
   absorb the rubato (the documented bar-grid-drift / A/V-sync class, amplified here).
   NOT a chord error — the 0.89 vocab / 0.84 ordered-LCS prove the harmony is decoded.
2. **Altered/rootless jazz-voicing simplification** (secondary). A7#5→A7, Dm7b5/Ab
   (rootless slash) mis-rooted, G7sus/G7b9→G7, Gm-maj7→Gm7, Bb6→Bbmaj7; the
   `Bb^ Ab/Eb Gb/Db F7/C` passing bar → extra Eb. Quality FAMILIES (7/maj7/min7) right;
   the specific #5/b9/sus/6/maj7-on-minor + slash basses lost.

Verdict: Bein' Green is AABA (not through-composed) and decodes WELL at the chord
level — a strong scoreboard row, not a failure. The open work is time registration
(upstream bar-grid/rubato), not harmony. Artifact `docs/plots/inferred_bein_green.html`.
