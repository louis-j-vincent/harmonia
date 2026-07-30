# Handoff → the session owning chart_model.py (section chips one bar late)

**From**: the feat/harmonic-key session (2026-07-30). **To**: the concurrent
session with the large uncommitted chart_model.py diff (chordtone
section-repr work). We deliberately did NOT touch chart_model.py to avoid
clobbering you. This documents the bug that belongs to your file, with the
full reasoning trail.

## What already landed (don't redo it)

Commit `9303331` (branch `feat/harmonic-key`), in
`scripts/render_youtube_chart.py`: the bake layer anchored bars **one beat
late** vs the harmony — beat_this's downbeat anchor claims beat k≡2(mod 4)
but harmony changes at k≡1. Baked This Love: 75/120 chords on beat 3, so
every bar-opening chord rendered as the previous bar's tail.
`harmonic_phase_correction()` now rotates the grid phase when non-N chords
have a ≥55% supermajority on one non-zero beat residue AND ≤15% sit on
beat 0 (This Love: 62.5% vs 6.7%). Kill-switch:
`HARMONIA_HARMONIC_REANCHOR=0`. Red-first tests:
`tests/test_render_youtube_chart.py::TestHarmonicBarPhaseReanchor` (7/7).
Corpus scan of 57 baked charts: fires on This Love, Misery, one Let It Be
loop demo only.

## The remaining bug — yours

Section chips are timestamped ~+0.34s AFTER their section's opening chord,
and `chart_model._section_runs` maps chip→bar with a **1e-6 tolerance** on
bar start times. Result: section rows start one bar late in the app, even
after the re-anchor fix above (and the re-anchor shifts bar t0s, so the
1e-6 match can now miss entirely).

**Remedy (agreed with Louis via the harmonic-key session):** map each chip
to the NEAREST bar t0 with a tolerance of about half a bar, instead of the
1e-6 exact match. Please:

1. Red-first test against the current behavior (This Love payload:
   chip at ≈41.8s should land on the bar starting ≈41.5s, currently lands
   one bar late).
2. Check the interaction with `harmonic_phase_correction` — chips were
   computed under the OLD phase; nearest-bar matching should absorb the
   one-beat shift, verify it does.
3. State what the fix does NOT solve (upstream beat_this downbeat phase is
   still wrong inside the pipeline's barlocked sections).

## Ops after both fixes land (for Louis or whoever runs it)

1. **Restart `harmonia_server.py`** — no reloader; the running process
   serves the old module (this has burned us before).
2. Re-analyze the three affected songs (This Love, Misery, the Let It Be
   demo) so their baked charts pick up the corrected bar phase. Re-bake
   only after the chart_model chip fix is in, to avoid a double pass.
