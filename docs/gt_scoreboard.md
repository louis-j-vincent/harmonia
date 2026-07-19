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

Held-out takeaway: the largest-unit section detector + chord read generalize
cleanly to unseen songs; the one "failure" is a GT-vs-recording key-notation
mismatch, a scoreboard caveat rather than a model error.

## Chord-vocabulary fidelity (alignment-free, decode vs GT root-set Jaccard)
| song | GT vocab | decoded vocab | Jaccard | key match |
|---|---|---|---|---|
| Billie Jean | F#m,Bm,D,C#7 | F#m,Bm,D,C#7 | **1.00** | ✓ F# minor |
| Let It Be | C,G,Am,F | C,G,Am,F(+G# noise) | 0.80 | ✓ C major |
| Easy | (7 chords) | (10, extra) | 0.70 | — |
| Autumn Leaves | (7) | (8) | 0.67 | — |
| Chain of Fools | C | C | 1.00 | ✓ |

Chords are read **well** (user: "il trouve les bons accords, le bon beat"). The
open gaps are STRUCTURE and a few MISSED chords, per the error taxonomy.

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
   cause). Upstream (emission) — OPEN, priority per user ("les accords loupés").
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
