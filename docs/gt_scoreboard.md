# GT self-correction scoreboard — audio→chart vs iReal ground truth

Standing before/after table for the overnight GT campaign (2026-07-20). Goal:
re-infer charts from audio as faithfully as possible, using iReal GT as the
reference (symbolic per-chord similarity + section-structure comparison).

## Matched set (cached audio × iReal GT)
| song | video id | corpus | GT key | GT loop (symbolic) |
|---|---|---|---|---|
| Billie Jean | Zi_XLOBDo_Y | pop400 | F#- | [Bm,F#m,F#m,F#m] P4 |
| Let It Be | QDYfEBY9NM4 | pop400 | C | [C,Am,C,F] P4 |
| Chiquitita (abba) | p9Y3N_2xUsw | pop400 | — | — |
| Autumn Leaves | zTVlrOk9a8M | jazz1460 | — | (through-composed) |
| Easy (Commodores) | saaLW0jiiUE | pop400 | — | [G#,A#] P2 |
| Chain of Fools | 5C4FnlftQt4 | pop400 | C | (1-chord vamp) |
| Bein' Green | (needs analyze) | jazz1460 | — | (AABA head) |
| Beat It | oRdxUFDoQe0 | — (not in GT) | — | tab-verify only |

MJ sourcing: **works with yt-dlp 2026.07.04** (was 403 on the older build). Any
iReal-playlist song whose audio can be fetched can join the set (mind the 2.0 GiB
disk floor; stream-and-delete after feature extraction).

## HELD-OUT validation (songs NOT used to tune the section detector)
| song | key | vocab Jaccard | sections | verdict |
|---|---|---|---|---|
| Stand By Me (Ben E King) | A major ✓ | **1.00** {A,D,E,F#m} | B · A×2 · C | perfect chords + phrase structure — generalizes |
| Every Breath You Take (Police) | G#/Ab major | 0.25 | A | **GT-KEY CAVEAT**: recording is in Ab; iReal notates A → semitone offset, NOT a decode error (rule #3: GT is a measurement too) |
| Georgia On My Mind (Ray Charles) | decoded G / GT-notated F | 0.45 (uninformative) | ABABA | **GT-KEY CAVEAT (2026-07-20)**: Ray Charles' recording is in G; jazz1460 notates F (whole step down). Under the +2 alignment ordered-root LCS = **21/27 = 78%**; vocab Jaccard is uninformative (busy jazz decode spans 11 pcs). Decode key ✓ matches the RECORDING, not the chart |

Held-out takeaway: the largest-unit section detector + chord read generalize
cleanly to unseen songs; the one "failure" is a GT-vs-recording key-notation
mismatch, a scoreboard caveat rather than a model error.

## Chord-vocabulary fidelity (alignment-free, decode vs GT root-set Jaccard)
| song | GT vocab | decoded vocab | Jaccard | key match |
|---|---|---|---|---|
| Billie Jean | F#m,Bm,D,C#7 | F#m,Bm,D,C#7 | **1.00** | ✓ F# minor |
| Let It Be | C,G,Am,F | C,G,Am,F,**C/G**(+A# noise) | **1.00** | ✓ C major |
| Easy | (7 chords) | (10, extra) | 0.70 | — |
| Autumn Leaves | (7) | (8) | 0.67 | — |
| Bein' Green | Bb,A,D,G,C,F,Ab,Db (8) | +Eb (9) | **0.89** | ✓ Bb major |
| Chain of Fools | C | C | 1.00 | ✓ |

### Bein' Green (jazz1460, AABA ballad) — Mission 2 error analysis 2026-07-20
Key ✓ Bb (decoded A#=Bb). **Chords + ORDER recovered well** but TIME-registration
fails: root-vocab Jaccard **0.89** (all 8 GT roots present, +1 Eb), **ordered-root
LCS 27/32 = 84%** of GT bar-roots recovered in sequence (89% on collapsed GT), yet
per-bar uniform-grid alignment only **5/32 = 0.16**. It is AABA (not through-composed):
the decode reproduces the A-section ii-V descent (A7→G7→Cm7→F7) and the Ab→Db bridge,
twice. **Top-2 error classes**: (1) **time-registration / harmonic-rhythm on a rubato
ballad** — chords are right and in order, only their onset times don't map to GT bar
positions (tempo read 149.6 = a 2×/rubato ballad artifact; the uniform bestfit bar
grid can't absorb the rubato — the documented bar-grid-drift / A/V-sync class, here
dominant); (2) **altered/rootless jazz-voicing simplification** — A7#5→A7, Dm7b5/Ab
(rootless slash) mis-rooted, G7sus/G7b9→G7, Gm(maj7)→Gm7, Bb6→Bbmaj7, and the
`Bb^ Ab/Eb Gb/Db F7/C` passing bar read as an extra Eb; partial-credit quality families
(7/maj7/min7) are right, the specific alterations + slash basses are lost.
Artifact `docs/plots/inferred_bein_green.html`. NOT a chord-recognition failure —
the open work is time registration (upstream bar-grid), not the harmony.

Chords are read **well** (user: "il trouve les bons accords, le bon beat"). The
open gaps are STRUCTURE and a few MISSED chords, per the error taxonomy.

## Section drift — PHASE-TOLERANT block matching (2026-07-20)
`_sections_by_largest_unit._sim` was strict position-by-position on the uniform grid →
a 1-bar phrase drift minted a false-B letter (Let It Be identical-but-drifted blocks
scored strict 0.00). Fix: ±1-bar lag trusted only above `_PHASE_STRICT=0.80` (real drift
~1.00, coincidental slide ~0.6 rejected → no over-merge) + trailing-partial-block overlap
match. Matched set: **henny/just-aint/abba/Commodores/aretha/Autumn/Georgia SAME; Billie
Jean bridge PRESERVED (A·B·A×3); Let It Be A×18; Stand By Me / Bein' Green trailing
artifact merged.** Every merge = trailing/drift artifact, no genuine section lost. 2-run
stable, live-serve confirmed, decode/anti-crush untouched. Kill-switch
`HARMONIA_SECTION_PHASE_TOL=0`. **Open**: true verse↔chorus split needs grid-anchored
blocks (deferred).

## Module 1/2 (bass veto / top-2 referee) — premise FALSIFIED
Let It Be's Am is ALREADY correct on the live path (musx root+bass, not the NNLS head);
the C-over-Am bias is NNLS-head-only and never reaches the final decode. Not implemented
(no-op on prod). See ledger 2026-07-20.

## Chord-CHANGE-timing detection F1 (2026-07-20) — vs music-x-lab change times, matched set (9)
Threshold-swept best-F1, same adaptive peak-picker for every novelty curve. Literature
ceiling: Harte & Sandler 2006 HCDF reported change-F ≈ 64.9% (known-hard MIR problem).
| novelty source | mean best-F1@150ms | @250ms | note |
|---|---|---|---|
| raw treble-chroma L2 (current `_chroma_flux`) | 0.122 | 0.275 | baseline |
| HCDF tonal-centroid (Harte-Sandler) on NNLS chroma, σ=4 | **0.186** | 0.42 | +52% rel, wins 7/9 |
| HPSS-harmonic (librosa) → NNLS chroma | mixed | — | helps 2/3, not universal |
| **madmom DeepChroma** (Korzeniowski-Widmer 2016) + HCDF | **0.442** | **0.580** | **3.6×**, wins 9/9 (justaint 0.73, abba 0.57) |

On clean DeepChroma, HCDF-TCS ≈ raw-L2 → the win is chroma QUALITY, not the projection.
Sparse vamps (Billie Jean 6 s gaps 0.17, aretha 1-chord 0.02) stay hard for all, per lit.

**Downstream (honest)**: the pipeline's only flux consumer is the downbeat PHASE φ, which
the raw flux already recovers correctly (DeepChroma agrees 5/6, just a sharper comb) — so
the 3.6× win does NOT change current output. The 173 ms boundary-PLACEMENT noise lives in
the SEGMENTATION stage (`_root_change_segs`/musx times), not the flux. Shipped: kill-
switched flux-novelty selector (`HARMONIA_FLUX_NOVELTY=raw|hcdf|deepchroma`, **default raw
= byte-identical**) + madmom-py312 shim, so a future change-time-consuming boundary
REFINER can build on the validated DeepChroma front-end. Opt-in (madmom broken by default
+ NN cost). Session log `docs/research_sessions/chord_change_timing_2026-07-20.md`.
Refs: Harte & Sandler 2006 (HCDF); Korzeniowski & Widmer 2016 (Deep Chroma, in madmom);
APSIPA 2025 source-sep preproc (DEFERRED, licensing).

## Error taxonomy (ranked by generality × impact)
1. **Section over-collapse / fragmentation** — verse+chorus sharing a chord set
   merged to one letter (Let It Be → one 142-bar A); jazz heads over-split (Autumn
   → 27 sections). **FIXED (over-collapse)** by largest-repeating-unit sections
   (commit below). Autumn over-split (content-relabel path on jazz) OPEN.
2. **Loop-period underestimation on real audio** — GT P4 loops (Billie Jean,
   Let It Be) decode as P2, because one chord dominates (BJ F#m 3/4 positions →
   determinism flat) or the true period's positional coverage is noise-degraded
   below the gate (LIB P4 rec 0.50). Reframed by the user's Flag 2: small loops
   live INSIDE sections, so this matters for the INNER loop display / ×N folding,
   not the section letter. OPEN (needs pooled-evidence re-decode of missed chords).
3. **Missed chords** — Let It Be under-detects G/Am at some positions (the P4→P2
   cause). **PARTLY FIXED 2026-07-20 (phrase-position pooling)**: the P2 [C,F] was a
   FALSE loop that made Occam actively SNAP real G/Am bars to C/F; pooling the per-
   bar posteriors across the 8-bar-phrase repetitions (√N) surfaces G, corrects the
   period to P8 (adds G to the vocab), so Occam no longer destroys them. Let It Be
   rendered chart: G 25→31, Am 13→16, +4 C/G slash; vocab Jaccard 0.80→**1.00**.
   Controls byte-identical, anti-crush 100%, live-server + 2-run confirmed.
   **Am RESIDUAL still upstream**: pooling does NOT surface Am (NNLS root head
   prefers C in Am bars, C/E overlap — a bias not noise); Am gain is only from
   stopping the false-vamp snap. A true Am fix needs bass/third-aware emission.
4. **A/V sync residual** — see below; small (±0.2s), MID marginally >150ms.

## A/V sync — FIXED (real-beat display snapping, 2026-07-20)
Root cause: the uniform bestfit decode grid can't absorb tempo rubato → chord
onsets drift by the grid residual. Fix (display-layer, decode untouched): the bake
snaps each chord's displayed t0/t1 to the nearest DETECTED beat (`ChordChart.
beat_times`); (bar,beat) layout stays uniform so sections/folds are byte-identical.
Serve-path bar1_offset confirmed applied exactly once (saved offsets all 0).
Onset-aligned offsets at start/mid/end (gate ±150ms):
| song | before (uniform) | after (snapped) |
|---|---|---|
| Let It Be | −0.12 / −0.20 / +0.08 | **+0.035 / −0.069 / +0.058** |
| abba | — | **+0.023 / +0.024 / +0.034** |
| Billie Jean | — | **+0.127 / +0.023 / +0.023** |
All within ±150ms, 2-run stable, sections unchanged. Applies to NEW analyses (bake).

## A/V sync measurement (Let It Be) — historical (pre-fix)
- Served m4a 243.03s ≈ analyzed wav 243.05s → **no codec-delay mismatch**.
- Chord-change times vs audio onsets (onset-aligned): START −0.12s, MID −0.20s,
  END +0.08s — within ±0.2s, **NOT monotonically growing**. The chord times track
  the audio; the "1.25s overshoot" was the final held chord extending past the end.
- Uniform bestfit grid residual vs real librosa beats: std 0.72s, up to ±1.5s —
  the drift the uniform grid can't absorb. MID −0.20s (>150ms) comes from this.
- Verdict: **not a gross growing lag**; a display-layer real-beat remap would
  tighten MID under 150ms. Lower priority than structure/missed-chords. OPEN.

## Before / after (shipped this campaign)
| item | before | after | gate |
|---|---|---|---|
| Let It Be sections | one 142-bar "A" | B×2 · **A×15** · C (8-bar phrase folded) | anti-crush 100%, 2-run stable, no-regression on 7 controls |
| abba Chiquitita sections | 15 alternating sections | **A×9 · B×5 · C** (verse/chorus) | same |
| Billie Jean sections | (n/a, new) | **A (verse F#m-Bm) · B (bridge D-F#m-C#7) · A×2 · C** — matches GT form | same |
| aretha Chain of Fools | — | A×2 · B | same |
| A/V sync | display trails audio | playhead lead 0.18s (measured offset ±0.2s, non-growing) | app_shell disk-served |
| MJ sourcing | 403 blocked | Billie Jean (vocab 1.00) + Beat It fetched | — |

`×N folding on real audio is now UNLOCKED` (the repeating phrase is the foldable
unit): Let It Be A×15, abba A×9, Billie Jean A×2 — the coordinator's priority (a).
Screenshots: `scratchpad/screenshots/{letitbe_folded_AxN,billie_jean_sections}_2026_07_20.png`.

**Still falls back (no clean phrase structure → changepoint path, unchanged)**:
Autumn Leaves (jazz head, >4 phrase clusters even at rec_min 0.25 — genuinely
through-composed; its 27-section content-relabel split is the OPEN jazz-structure
item), henny/just-aint/commodores (short 2-chord vamps, no 8/16-bar phrase).

anti-crush symbolic (occam on pop400 GT): **100.00% of 25,120 bars unchanged**,
154/345 tunes read as a loop.

## Directive 2 — multi-factor section-boundary model (learned, song-held-out) — FINDING: keep largest-unit
Candidates = every bar (grid-locked); features per-song z-normalized; labels = GT
section boundaries (iReal *A/*B, snapped by time-fraction) ±1 bar; logistic,
leave-one-song-out on 6 matched songs (Billie Jean, Let It Be, Stand By Me, Easy,
Chain of Fools, Autumn). REAL AUDIO ONLY. `scripts/section_boundary_{features,train}.py`.

**LOSO mean boundary F1: learned model 0.23 vs phrase-position-only 0.34** — the
learned multi-factor model LOSES to the hand baseline on this set (a finding, not a
failure, per process rule). So it is NOT wired; the shipped largest-unit detector
(which already keys off phrase position + chord-block recurrence) stands.

Feature importances (standardized logistic weight | univariate corr with boundary):
| feature | weight | corr | read |
|---|---|---|---|
| phrase_pos (dist to 8-bar mult) | **+0.57** | **+0.16** | DOMINANT — boundaries sit on the 8-bar grid |
| chord_recur (before/after novelty) | −0.77 | −0.14 | boundaries have LOW novelty — the phrase RESTARTS (not a contrast) |
| drum_fill (HPSS perc. energy, prev bar) | −0.29 | **−0.06** | the user's drum-fill idea is NOT supported (as mean perc. energy) |
| timbre_nov (centroid Δ) | +0.20 | +0.03 | weak |
| energy_nov (RMS Δ) | +0.15 | +0.01 | weak |
| harm_rhythm (Δ distinct roots) | +0.14 | −0.03 | weak |
| nc_adj (N.C. within ±1 bar) | +0.11 | +0.04 | weak |
| phrase_restart (repeat of phrase 8 bars ago) | −0.22 | +0.05 | weak |

**Takeaways for the user**: (1) section boundaries are overwhelmingly a PHRASE-GRID
phenomenon — the 8-bar position is the signal, which the largest-unit detector uses;
(2) the acoustic cues (drum fill, energy, timbre) add ~no measurable signal on this
set — the drum-fill-tail hypothesis specifically does not correlate (as mean
percussive energy; a burst/onset-density fill signature is untested and could
differ); (3) chord-recurrence matters as "the phrase repeats here", already captured
by the block-matching. **Caveats**: only 6 songs; GT boundaries snapped by crude
time-fraction (tempo mismatch adds label noise, esp. Autumn 6/318). Revisit with a
larger matched set + proper GT-bar alignment before wiring a learned model.
