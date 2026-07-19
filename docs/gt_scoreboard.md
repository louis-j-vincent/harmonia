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

## A/V sync measurement (Let It Be)
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
| Let It Be sections | one 142-bar "A" | C×2 · B×2 · **A×9** · B×4 · D (8-bar phrase) | anti-crush 100%, 2-run stable, no-regression on 7 controls |
| Billie Jean sections | (n/a, new) | A-B-A-C-D phrase blocks | same |
| MJ sourcing | 403 blocked | Billie Jean + Beat It fetched | — |

anti-crush symbolic (occam on pop400 GT): **100.00% of 25,120 bars unchanged**,
154/345 tunes read as a loop.
