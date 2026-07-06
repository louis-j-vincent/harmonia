# Chord-change engine — deep investigation, 2026-07-06

Building the "scaffold → coarse grid → fill → zoom" method for chord-change
detection. Scaffold = GT `section_per_bar` + exact MMA beat grid (structure
detection is a separable, parked problem). Scripts:
`harmonic_rhythm_probe.py`, `period_estimation.py`, `chord_change_engine.py`.

## 1. Foundation validated — merging is the load-bearing lever

`harmonic_rhythm_probe.py`, 12 songs. Merge beats into g-beat blocks, measure
ROC-AUC of adjacent-block chroma+bass distance for "true change" vs "hold":

| grid | AUC | note |
|------|-----|------|
| g=1 (per-beat) | 0.643 | weak — this is the BP noise that tanked per-beat root to 0.20 |
| **g=2** | **0.962** | chroma+bass cleanly separates change from hold |
| g=4 | 0.903 | strong, but starts merging real changes |

→ The same-or-different decision works at the **block** level, not the beat level.

## 2. Per-section "period" premise FALSIFIED (corpus-wide, 1136 songs)

The plan was to estimate a per-section change period ∈ {1,2,4} and merge at it.
Corpus symbolic check (cheap, rule #5) killed it:

- Changes land on **every** beat of the 4/4 bar: beat 1/2/3/4 = 38.6 / 28.7 /
  20.1 / 12.5 %. Not a downbeats-only or half-bar grid.
- A section's changes explained by its best 2-beat grid: only **61%**; best
  4-beat grid: 45%.
- Sections with a *clean* period (≥80% of changes on one phase): period-1 **92%**,
  period-2 5%, period-4 3%.
- Within-section change spacing 1/2/3/4 beats = 19/25/24/18 % — nearly uniform.

→ There is no clean per-section period to estimate; harmonic rhythm is irregular
with a 2-beat mode. Estimating a period would be fitting noise. `period_estimation.py`
confirmed on audio (est accuracy 27%, GT period = 1 for 44/45 sections). **Decision:
drop the period estimator; use a fixed 2-beat coarse grid** (best single grid) and
push the residual to the zoom.

## 3. Coarse engine — fixed 2-beat merge + same-or-different (GT-structure scaffold)

`chord_change_engine.py`, 15 clean songs, θ=0.15 (block cosine-distance cut,
forced boundary at section changes):

| metric | value |
|--------|-------|
| change-detection F (±1 beat) | **0.89** (P 0.91, R 0.88) |
| change-detection F (exact beat) | **0.50** |
| MIREX root | 60.5% |
| MIREX majmin | 39.5% |
| detected/GT segment ratio | 0.91 |

The ±1 vs exact gap (0.89 → 0.50) is the zoom's headroom: half the changes are
placed one beat off because the real change sits on the odd beat *between* two
2-beat blocks. Note root (60.5%) is now the limiter, not segmentation — labeling
on real evidence (parked task #3) is the next bottleneck once boundaries are good.

## 4. Every zoom strategy FAILS; segmentation is not the bottleneck — LABELING is

Four zooms tried to beat the coarse chgF 0.89 / exact 0.50; all failed:

| zoom | chgF | exact | root | why it failed |
|------|------|-------|------|---------------|
| coarse only (baseline) | 0.89 | 0.50 | 60.5% | — |
| naive beat novelty (snap+split) | 0.86 | 0.41 | 50.1% | reintroduces the noisy g=1 signal |
| per-track (bass/chord stems) | — | — | — | premise falsified: walking bass flips 58% of beats; no per-track beat cue beats mixed (all AUC ~0.6–0.67, `pertrack_zoom_probe.py`) |
| divisive top-down pooled split | 0.36 | 0.27 | 39.9% | first split of a many-chord section has two muddy multi-chord halves → under-segments |
| pooled-halves boundary snap ±1 | 0.87 | 0.41 | 56.9% | max-contrast position ≠ true change beat (BP onset smear) |

Exact-beat placement (~0.50) is a **hard ceiling**: the beat-level change signal is
~0.65 AUC on *every* track (mixed or isolated), and the pooling that gives clean
SNR (0.962) destroys the resolution needed to localize a change to one beat.

**Oracle-boundary diagnostic (the decider):** feed GT change beats as boundaries
(chgF=1.00) and label with the same models → root **55.2%**, majmin **36.6%** — no
better than the coarse engine. So perfect segmentation does NOT raise accuracy.
The gap is entirely in LABELING on real evidence: bass-argmax root is wrecked by
walking bass, and the family emission model tops out ~37% majmin. **Conclusion: the
chord-change/segmentation problem is at its useful ceiling (coarse chgF 0.89); the
priority is now labeling — parked task #3.**

### (historical) Naive beat-level zoom FAILS

`--zoom` (snap boundary + split on interior beat-level novelty): chgF 0.89→0.86,
exact 0.50→0.41, root 60.5%→50.1%, ratio 0.91→1.08 (over-segments). Reason: beat-
resolution mixed-chroma novelty IS the noisy g=1 signal (AUC 0.643) that merging
suppressed — splitting on it reintroduces the noise. **The zoom needs a cleaner
cue than mixed beat novelty** → per-track self-similarity (bass-onset motion,
piano SSM), which we get free from the MIDI stems. That is the next step.

## Next

Per-track zoom: within each coarse segment, use the *isolated bass onset* PC-change
and per-instrument SSM to locate interior changes the mixed 2-beat block blurred —
the user's original zoom design, now justified by the naive-zoom failure. Possibly
within-song EM (safe here: within-song, not the correlated-cross-repeat fold that
died before).
