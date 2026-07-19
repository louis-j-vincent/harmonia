# Adversarial review log — 2026-07-17

Grilled tonight's 6 uncommitted files + load-bearing dependencies, looking for
silent bugs/staleness/bias. Durable summary also cross-posted into
`docs/known_issues.md`. Findings below, most important first.

## Fixed

**#1 — Path traversal, `/api/chord-snippet/<filename>`**
(`scripts/harmonia_server.py::_audio_path_for_chart`). Built the audio path
from the URL filename with no traversal guard, unlike the sibling
`api_chart_model` (which has `p.parent != PLOTS_DIR`). Confirmed:
`inferred_../../../../etc/x.html` resolved outside `AUDIO_DIR`. HTTP
exploitability unconfirmed (Flask's default converter blocks raw `/`), fixed
regardless. **Fix**: `.name` on the slug + `p.parent == AUDIO_DIR` check.

**#2 — Broken repeat-detector filter, `librosa_laplacian_sections`**
(`harmonia/models/section_structure.py`). Dropped librosa's
`timelag_filter` wrap around the diagonal-enhancement median filter — ran on
raw time-time recurrence matrix instead of time-lag space, the actual
mechanism that enhances repeat stripes. Confirmed: breaks matrix symmetry
(`max|Rf-Rf.T|≈0.48`), `np.linalg.eigh` then silently reads one triangle,
real-audio cluster labels differ substantially from the correctly-wrapped
version. **Fix**: `librosa.segment.timelag_filter(median_filter)(R, size=(1,7))`.
**Consequence**: invalidated the 3-song V-measure table in `known_issues.md`
(see re-validation below) — corrected in place there.

**#3 — Stale-cache bug, `nnls_features.extract_bothchroma`**
(same root cause `musx_bass.py` already fixed for its own cache, per that
file's own docstring — but never applied here). Keyed on mtime; the server
downloads every YouTube job into a fresh `tempfile.mkdtemp()`, so a
re-analysis of the same video never hits cache. **On the deployed default
path** (`HARMONIA_ANALYZE_FRONTEND=nnls24`); tonight's own new `?seg=` A/B
toggle actively triggers this. Confirmed on disk: 8 distinct cold cache
entries for one video id. **Fix**: key on stem only, matching `musx_bass`.
Deliberately did **not** touch `stage1_pitch.PitchExtractor`'s similar-looking
key (mtime+full-path) — it backs broad local/corpus workflows where that's
correct semantics; "fixing" it the same way would introduce real staleness
there.

All 3 fixes verified: compile clean, 464/464 tests pass, behavior confirmed
empirically (no test coverage existed for any of the three).

## Re-validation: section-fallback V-measure after fixing #2

3-song iReal-GT V-measure, symbolic (unaffected by #2, reused from cache) vs
librosa (re-run fixed):

| song | symbolic | librosa OLD (buggy) | librosa FIXED | gate fires? |
|---|---|---|---|---|
| chain_of_fools | 0.50 | 0.40 | 0.55 | no — symbolic non-degenerate, gate abstains regardless |
| autumn_leaves | 0.00 | 0.18 | 0.25 | yes |
| goodbye_ybr | 0.19 | 0.64 | 0.37 | yes |

Deployed **decisions unchanged** (gate only checks symbolic degeneracy, never
compares scores). autumn's win holds (even slightly better than claimed).
**goodbye's win was overstated ~2x** (0.64 claimed → 0.37 actual) — still a
net win, much smaller margin. Correction posted in `known_issues.md`.

## Bias check — root head trained unweighted, quality head is not

`train_nnls24_heads.py`: quality head uses inverse-frequency class weights;
root head does not. RWC root-class distribution is imbalanced (G 14.4% vs
C#/F#/G# ~4.5%, 3.18x ratio). Confirmed on the **shipped** model
(seed-0 held-out split): 3 worst-recall roots = 3 rarest (G# 80.0%, A# 81.8%,
C# 85.7% vs 91–96% for the rest) — real signal at the tails, not a clean
monotonic frequency relationship overall.

Cheap 1-seed premise-check (retrain root head with/without class weights,
same split): aggregate flat (82.86% vs 82.78%), but per-class deltas
contradictory (A# +11.7pp, G# −5.6pp) — noise-shaped at n≈90/class.

**Multi-seed follow-up (5 seeds, matching the project's own CV convention,
`scratchpad/root_head_cw_multiseed.py`)**: confirms the null result.
Aggregate 79.29% unweighted vs 79.04% weighted (−0.25pp, within the ±2.8pp
seed-to-seed std). Mean recall over the 5 rarest roots (C#/F#/G#/D#/B):
79.9% → 80.4% (**+0.5pp**, negligible). Per-class deltas still don't move
together (A# +4.1pp, G# −0.8pp, C# −1.3pp) — no consistent direction even
with 5x the data. **Verdict: class-weighting the root head is not a fix** —
the bias at the tails is real but this remedy doesn't move it meaningfully.
**Not applied.** Next things worth trying if this is worth more budget: focal
loss (already implemented in `multihead_training.train_clf`, unused here),
more rare-key training data, or accepting the tradeoff since a Bayesian-
optimal classifier legitimately leans toward common keys given a skewed
prior.

## Ruled out (checked, no bug — don't re-litigate)

- `musx_bass._DEFAULT_MUSX` pointing at an ephemeral scratchpad path: resolves
  to the durable git-tracked `harmonia/third_party/` copy first in practice.
- `NNLS24Heads` importing from `scratchpad/`: that dir is git-tracked in this
  repo, not session-ephemeral.
- `musx_bass._parse_root` looked unbounded: does end in `% 12`, correctly
  wrapped (read was just cut off before that line).
- `_NNLS_Q_TO_HARTE`/`_MUSX_Q_TO_SEV` silent `"maj"` fallbacks: both dead code
  — each dict is exhaustive against its model's actual fixed vocabulary
  (verified against `nnls24_heads.npz` and music-x-lab's
  `submission_chord_list.txt`).
- `chord_hmm.py` transition priors: skipped, not investigated — CLAUDE.md
  explicitly flags that module as frozen/non-critical-path.
- Beat-tracker octave-lock in `librosa_laplacian_sections`: already documented
  and bounded (`max_section_bars`) in the code itself; boundary *times* are
  unaffected by a tempo-octave error, only bar *counts* are, and that's
  already clamped.

## Round 4 — segmentation building blocks + new opt-in path: all clean

- `_root_change_segs`, `_musx_boundary_segs`, `_merge_grid_by_root(_and_bass)`:
  read for off-by-one/boundary bugs (empty-input guards, zero-length segment
  risk, duplicate cut dedup). All correct as written.
- `_QUALITY_TO_IREAL` (render_youtube_chart.py): cross-checked against every
  sev_h string producible by `_NNLS_Q_TO_HARTE` and `_MUSX_Q_TO_SEV` —
  exhaustive, no silent-fallthrough gap (the file's own comments show this
  exact bug shape was already caught and fixed once before).
  `_QUALITY_TO_SEVENTH`/`_QUALITY_TO_FAMILY` also exhaustive.
- Tonight's new opt-in `segment_source="musx"` path (never exercised before
  tonight — only reachable via `?seg=musx`): ran it end-to-end on real audio.
  Works, no crash: 36 segs vs 70 for the default NNLS root-change
  segmentation on the same song (over-segmentation reduction as intended).
- Server thread-safety (`_jobs`/`_jobs_lock`, `_yt_audio_meta`,
  `_yt_video_ids`): checked for read-modify-write races across concurrent
  analysis threads. `_jobs` is consistently lock-protected. The other two are
  single-key dict ops on one shared global (not load-mutate-write-back), so
  GIL atomicity covers them — no race.
- Re-confirmed CLAUDE.md's existing `PitchExtractor` cache-key gotcha (misses
  module-level constants like `BASIC_PITCH_FRAME_RATE`) is still accurate —
  not new, just verified the doc hasn't gone stale.

No new bugs this round.
