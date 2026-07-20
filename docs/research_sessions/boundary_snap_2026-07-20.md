# Boundary-placement refiner (DeepChroma peak-snap) — 2026-07-20

Budget 3h. Target the 173 ms bar-first chord-onset placement noise (grid-align tool,
std vs raw librosa beats, no systematic bias). Snap each decoded onset to the nearest
DeepChroma novelty peak within ±1 beat when a prominent peak exists; else current
behaviour. Onset TIME only — never labels/roots/quality, never bar/beat indices
(sections read indices, must stay byte-identical). Opt-in kill-switch. Reuses
`_madmom_compat.deepchroma_novelty` (cached).

## Injection point
`scripts/render_youtube_chart.py::chart_to_interactive_inputs._snap` — already the
display-layer time-snap (currently snaps t0/t1 to `pipeline_chart.beat_times`); the
(bar,beat) LAYOUT is computed separately from `start_beat_idx`/`start_s`, so moving
t0/t1 there is byte-identical downstream (same discipline as the flux-φ verification).

## Metric (grid-align tool methodology)
std of (bar-first chord onset − nearest raw librosa beat) across the matched set,
snap-ON vs OFF. Target: materially below 173 ms, mean unchanged (tighten spread, no
new bias). Plus change-F1 vs music-x-lab times.

## PREMISE FINDINGS — DeepChroma peak-snap FALSIFIED; root cause is fold-reconstruction drift

Metric is the offset WRAPPED modulo the beat period (phase within a beat); the tool's
"173ms" is this wrapped std (Let It Be currently 111ms after my A×18 fold change).

1. **Raw (non-folded) onset placement is already perfect**: the current `_snap` snaps
   onsets to `pipeline_chart.beat_times` = the SAME librosa beats the tool references →
   std(onset − nearest raw beat) = **0** for every matched song. Nothing to fix there.
2. **Peak-snap makes RAW onsets WORSE**: snapping to DeepChroma peaks moves onsets off
   the beats to harmonic-change points → std 0→134ms (Let It Be), 0→407 (henny), and
   adds a systematic bias (−57ms Let It Be, +76 Billie Jean). The metric is offset-vs-
   BEATS; peaks aren't on beats.
3. **The 173/111ms is the FOLDED-SECTION reconstruction**: a folded A×N section shows one
   representative phrase replayed by offsetting `rep_bar.t0 + (span_i.start − span_0.start)`
   (app_shell + the diagnostic both do this). The repeats are NOT identically timed (rubato
   drift within each 8-bar phrase), so the offset doesn't tile → ±111ms phase wobble.
4. **The fix is BEAT-snapping the RECONSTRUCTED folded onsets** (Let It Be, wrapped std):
   baseline 111ms → **snap→raw-beat (±1 beat) 24ms** (78% ↓); snap→DeepChroma-peak 104ms
   (no help). So the evidence says beat-align the fold, NOT peak-snap.

**Verdict**: the mission's DeepChroma peak-snap does not fix this metric (it is offset-vs-
beats, and peaks are off-beat by design). The real defect is fold-reconstruction drift,
fixed by snapping the reconstructed folded onsets to real beats. Proceeding with THAT
(evidence-based), reporting the peak-snap falsification honestly.

## Corpus result (7 matched songs, baked charts, fold-reconstruction bar-first wrapped-std)
| song | OFF | beat-snap | peak-snap |
|---|---|---|---|
| Let It Be | 111 | **23** | 104 |
| Stand By Me | 191 | **44** | 233 |
| henny | 126 | **32** | 150 |
| Billie Jean | 79 | **37** | 103 |
| Bein Green | 67 | **42** | 100 |
| abba | 6 | **0** | 55 |
| Commodores | 10 | **9** | 77 |
| **MEAN (ms)** | **84** | **27 (−68%)** | **118 (+40% WORSE)** |

Per-onset beat-snap improves ALL 7; DeepChroma peak-snap degrades ALL 7. Mean unchanged
(no new bias): beat-snap mean −0.1 ms. Definitive.

## SHIPPED (safe, tested, server-side)
- `api_grid_align_data` now returns `downbeat_times_snapped`, `displayed_chords_snapped`,
  and `boundary_offset_stats` {off, beat_snapped} — verified live on a side port:
  Let It Be **off std 110.8ms → beat_snapped 25.2ms**. The grid-align tool can now SHOW
  the before/after directly.
- Artifact `docs/plots/boundary_snap_beforeafter_2026_07_20.png` (OFF vs beat-snap
  histograms, 4 songs).

## NOT shipped this round (deliberate) — production playback fix = precise recommendation
The user-facing fix is a client change: `harmonia/output/app_shell.html` line ~761
reconstructs a folded repeat's onsets as `c.t0 + (sp[0]-base)`; snap each reconstructed
t0/t1 to the nearest baked beat-time within ±1 beat (kill-switch `HARMONIA_BOUNDARY_SNAP`,
default OFF per the mission's opt-in guidance). Plumbing (all additive, low-risk):
1. `chart_interactive.render_interactive` payload += `"beatTimes"` (pass `_rbeats` from
   `render_youtube_chart.chart_to_interactive_inputs`, already computed there for `_snap`).
2. `payload_from_chart_html` already round-trips the whole payload → `to_chart_model`
   exposes `beatTimes`.
3. `app_shell` snaps at the reconstruction line.
I did NOT ship this: it is a cross-surface change to the sensitive `app_shell` UI whose
JS runtime I cannot test headless, and the round's PREMISE (DeepChroma peak-snap) was
corpus-falsified — so shipping the *correct* fix is a scope pivot better made with a
browser-in-the-loop check than by a background agent mid-flight. The science, the tool
before/after, and the exact injection point are all in hand for a fast follow-up.

## Reconciliation with the flagged 173 ms
The tool's 173 ms predates my A×18 phase-tolerant fold (which changed Let It Be's fold
structure); on the current baked chart Let It Be is 111 ms. Same phenomenon, same fix.
