# Inference pipeline: real stage timing + progressive-analysis animation scope

**Date:** 2026-07-17. **Scope:** profile the LIVE production pipeline stage by
stage with real wall-clock numbers, catalog the information available at each
stage, and propose (not build) a progressive/animated analysis screen. Read-only
profiling — the live server (`http://100.89.209.63:7771`, PID 42364 at time of
writing) was **not** modified or restarted.

Deployed config being profiled (server defaults, `scripts/harmonia_server.py`):
`infer_chords_v1(feature_frontend="nnls24", bass_frontend="musx",
quality_frontend="musx", segment_source="nnls")`. Note `beat_backend` defaults
to **librosa** — the live server does **not** pass `beat_backend`, so madmom is
NOT on the live path (it is opt-in only).

Profilers (scratchpad, mirror `_infer_nnls24` exactly, stem-keyed caches copied
to fresh temp stems so COLD is measured without deleting anyone's cache):
`profile_pipeline.py`, `profile_render_bp.py`.

---

## 1. Stage-by-stage timing (real wall-clock, 3 songs)

Three real songs, different lengths. **Cold** = first analyze of this audio
(caches empty). **Warm** = repeat of the *same video* (stem-keyed cache hit).
"Warmup" = one-time-per-process JIT/model-load cost, paid once on the first
analyze of a server's lifetime, not per song — broken out where it inflated the
first row.

| Stage | 68s (autumn) | 126s (yesterday) | 207s (rwc_p001) | Cached? | Scales with |
|---|--:|--:|--:|---|---|
| 1. Audio load (`sf.read`) | 30 ms | 36 ms | 55 ms | — | duration |
| 2. Beat track (librosa) + grid | 350 ms\* | 660 ms | 950 ms | **NO — recomputed every run** | duration |
| 3. NNLS-24 extract (VAMP `nnls-chroma`) | 1.8 s | 2.8 s | 5.0 s | YES (`nnls_infer`, stem) | duration |
| 4. `pool_beats` (per-beat chroma) | 7 ms | 7 ms | 17 ms | — | duration |
| 5. Root head (NNLS MLP) | ~1 ms\* | 1 ms | 1 ms | model in-proc | — |
| 6. Segmentation (`_root_change_segs`) | <1 ms | <1 ms | <1 ms | — | — |
| 7. **music-x-lab (5-fold subprocess)** | **10.4 s** | **13.1 s** | **17.7 s** | YES (`musx_infer`, stem) | duration |
| 8. Bass route (rule F) + quality paint | 1–5 ms | 1 ms | 1 ms | — | — |
| 9. Label loop + coalesce | 4 ms | 4 ms | 4 ms | — | — |
| 10. Key inference | <1 ms | <1 ms | <1 ms | — | — |
| 11. Section fallback (librosa-Laplacian) | 440 ms\* | 630 ms | 1.05 s | **NO — recomputed every run** | duration |
| 12. Chart render (`render_interactive`) | 8 ms | 8 ms | 8 ms | — | — |
| **infer subtotal (warm process)** | **~14.8 s** | **~17.3 s** | **~24.8 s** | | |

\* First-call warmup (one-time per process, not per song): librosa/numba JIT on
the first `beat_track` ≈ +3.1 s; `get_heads()` model load on the first root head
≈ +1.25 s; first librosa-Laplacian ≈ +1.7 s. Second and later analyses in the
same server process do NOT pay these.

### Post-infer, server-only stages (`_run_analysis`, run BEFORE the job reports "done")

These are **not** part of `infer_chords_v1` — they run in the server job after the
chart is already rendered but before the user is redirected. They were measured
separately (`profile_render_bp.py`, 126 s song):

| Stage | 126s song | Cached? | Notes |
|---|--:|---|---|
| Audio transcode (ffmpeg → AAC/m4a, playback copy) | ~1–3 s | — | duration-scaling |
| **Basic Pitch persist (`PitchExtractor.extract`, server line ~3996)** | **10.9 s COLD** | **temp-path key = ALWAYS cold** | see finding below |
| iReal community fetch (`search_community` + render) | network, ~1–5 s | — | best-effort |

- yt-dlp **download** (stage 0) is network-bound and not measured here; DEPLOY-3
  observed ~4–15 s for a 4-min song.

### End-to-end (excluding download), representative

- **Cold, 126 s song:** ~17 s infer + ~2 s transcode + **~11 s Basic Pitch** +
  ~2 s iReal ≈ **~32 s** wall before "done" (+ download on top).
- **Warm repeat (same video):** NNLS + musx caches hit → infer collapses to
  ~1.5–2 s, BUT beat-track (~0.7 s) + section fallback (~0.6 s) + transcode +
  **Basic Pitch (~11 s, still cold)** are all re-paid. The dominant residual on a
  warm repeat is the redundant Basic Pitch run, not NNLS/musx. This is what
  DEPLOY-3's "warm 17.5 s, remaining = download+NNLS+render" was actually
  measuring — the residual is mostly Basic Pitch, not NNLS.

### The long pole

On the cold path, **music-x-lab (stage 7) is the single dominant cost** at
10–18 s and scaling linearly with duration (~0.085 s per second of audio) — the
"5-fold CQT+XHMM ensemble" noted earlier tonight. **Basic Pitch persist
(~11–18 s cold)** is the *second* dominant cost and is redundant on this path
(finding below). Everything else combined is < 6 s. So the wait the user feels is
essentially `music-x-lab + Basic-Pitch-persist`, ~20–35 s, with the other nine
stages being visual garnish by comparison.

---

## FINDING: Basic Pitch persist (server line ~3996) is redundant + uncached on the nnls24 path — ~11–18 s of pure wait per analyze

`_run_analysis` runs, after rendering the chart:

```python
activations = PitchExtractor(cache_dir=...).extract(audio_path)   # line ~3996
activations.save(PITCH_CACHE_DIR / f"{slug}.npz")
```

The code comment claims this is "a cache hit … `infer_chords_v1()` already
populated PitchExtractor's cache above." **That is false on the live nnls24
path**: `infer_chords_v1` returns early via `_infer_nnls24`, which never calls
`PitchExtractor` — Basic Pitch is a BP48-path artifact. So on every fresh analyze
this is a **full cold Basic Pitch run** (10.9 s for a 126 s song; ~18 s for 4
min), keyed by the ephemeral temp-file path so it never warms across downloads
either. It runs *before* the job is marked "done", so the user waits on it inside
the "Drawing the chart" stage.

The persisted `.npz` is used only later (annotator / re-inference surfaces), never
by the freshly-rendered chart. **Top recommended win:** move this block (and the
best-effort iReal fetch) to a background thread *after* `update("done", …)`, or
gate it behind `feature_frontend`. Removes ~11–18 s of user-visible wait from
every analyze with no downside to chart-open. Touches the server job runner (not
inference logic) — flagged for the user to approve + restart, not applied here.

---

## 2. Information available at the end of each stage

This is the raw material for progressive display — what real data exists to show,
and when.

| After stage | Data now available |
|---|---|
| 1. Audio load | `duration_s`, sample rate, mono waveform. (yt-dlp metadata gives **title** + duration even earlier, during download.) |
| 2. Beat track | **`tempo_bpm`** (single global value), raw jittery beat times, de-jittered uniform beat grid `bt` (`n_beats`), beat `period`. **Time signature is hard-assumed 4/4** (never detected). No downbeats on real audio (GT-only feature). |
| 3. NNLS extract | Per-frame 24-dim NNLS chroma (12 bass-half ⊕ 12 treble-half) + frame times. |
| 4. `pool_beats` | Per-beat 24-dim chroma, `n_beats × 24`. **Key is computable from here** (treble half) — currently deferred to stage 10, could surface at ~4 s. |
| 5. Root head | Per-beat root posterior `n_beats × 12`. |
| 6. Segmentation | Root-change segment boundaries `[(start_beat, end_beat)]`, harmonic-grid size, `n_segs`. |
| 7. music-x-lab | List of `(t0, t1, Harte-label)` from the 5-fold ensemble — **root + quality + bass inversion** per segment. music-x-lab's own timing boundaries are coarse/degenerate (used only for labels; segmentation stays on NNLS unless `seg_source=musx`). |
| 8. Bass route (rule F) | Per-segment **sounding-bass pitch class + inversion flag** (bass ≠ root → slash chord). |
| 8/9. Quality + label | Per-segment `(root_pc, sev_h)` → the actual **chord vocabulary** (maj / min7 / maj7 / dom7 / hdim7 / sus / …). Final chord list: `[{label, start_s, end_s, duration_beats, confidence}]`, `n_chords`. |
| 10. Key inference | **Global key name + confidence** (from aggregate treble chroma). |
| 11. Section fallback | **librosa-Laplacian sections** `[{label: A/B/C…, start_s, end_s}]` — verse/chorus **repeat structure**, `n_sections`. |
| 12. Render | Bar-grid layout, `n_bars`, the interactive HTML chart. |

The genuinely *interesting-to-show-early* data, ranked by (interest ÷ latency):
**tempo** (known at ~1 s), **key** (computable at ~4 s if reordered — currently
hidden until ~20 s), then **chords** (the 10–18 s payoff), then **section map**
(+1 s).

---

## 3. Proposal: progressive analysis screen (design, not built)

### Current state
`harmonia/output/app_shell.html` shows a **3-step static list** driven by the
job record's `stage` (0–3) + `results[]`:
`Fetching the audio` / `Listening to it` (one opaque 15–25 s block) /
`Drawing the chart`. Result chips are filled **only after** each stage returns —
so tempo/key/chords all appear at once, at the very end of the long middle step.
The `HOME_TEMPLATE` variant (`scripts/harmonia_server.py` ~629) doesn't even read
`stage`; it just cycles whimsical jargon and redirects on done.

The middle step is the problem: it owns ~90% of the wall clock and shows nothing
until it's over.

### Proposed steps (weighted by duration AND how interesting the reveal is)

| # | Step label | Real owner | Live chip (real data) | ~Latency |
|---|---|---|---|---|
| 1 | Fetching the audio | yt-dlp download | `Yesterday · 2:05` (title from metadata) | 5–15 s |
| 2 | Finding the beat | librosa beat track | **`97 BPM · 4/4`** | ~1 s |
| 3 | Hearing the harmony | NNLS extract + key | **`Key: F major`** | ~4 s |
| 4 | Naming the chords | **music-x-lab (long pole)** | **`67 chords · Fmaj7 · Dm7 · Gm7 · C7…`** | 10–18 s |
| 5 | Mapping the structure | section fallback | **`A B A C · 4 sections`** | ~1 s |
| 6 | Drawing the chart | render | `50 bars` | instant |

Two quick, satisfying real-data reveals (tempo at ~1 s, key at ~4 s) land
*before* the long chord-recognition wait, so the user sees the model "working" on
real facts about their song within a second, and the unavoidable 10–18 s wait has
an honest, specific owner ("Naming the chords — music-x-lab") instead of a generic
"Listening" spinner. Step 6 becomes instant once the redundant Basic Pitch run is
moved to the background (finding above).

### What data is real vs. what would be faked
All six chips are **real** and already computed — nothing simulated. The only
change needed to surface tempo/key *before* the chord wait is to (a) let the
server update the job record mid-inference, and (b) reorder key inference to run
right after `pool_beats` (it only needs `feat[:,12:]`, currently deferred to after
music-x-lab for no reason). music-x-lab is a blocking subprocess, so step 4 cannot
show *per-chord* streaming without tailing its stdout — the honest treatment is a
spinner + attribution, then reveal the full chord preview when it returns.

### Implementation sketch — polling is sufficient; no SSE/WebSockets needed

The architecture is already a **job-polling model**: `POST /api/analyze` →
`job_id`; client polls `GET /api/job/<id>` every 1.2 s; job record is a dict
updated under `_jobs_lock`. This is *already* the right transport for progressive
updates — the milestones are coarse (6 steps over ~30 s), so 1.2 s polling (or
drop to ~800 ms during analysis) resolves them fine. **SSE/WebSockets would be
smoother but are not warranted** for six second-scale milestones; they'd add infra
for marginal benefit.

The one real blocker: `infer_chords_v1` is a single blocking call, so the server
can't update the job record between stages. Minimal additive fix:

1. **Add an optional `progress_cb: Callable[[str, dict], None] | None = None`** to
   `infer_chords_v1` / `_infer_nnls24`. Default `None` → zero overhead, no
   behavior change (safe silent-swap, CLAUDE.md #6). Call it at stage boundaries:
   `progress_cb("beats", {"tempo_bpm": …})` after the grid;
   `progress_cb("key", {"key": …})` after the reordered key infer;
   `progress_cb("chords", {"n_chords": …, "preview": [labels[:4]]})` after the
   label loop; `progress_cb("sections", {"sections": …})` after the fallback.
2. **In `_run_analysis`**, pass a `progress_cb` that does
   `update("running", stage=…, tempo=…, key=…, chord_preview=…, sections=…)`.
3. **Client** (`app_shell.html::pollJob` + `paintAnalysing`): extend the 6-row
   `STAGES` list and render the new fields (`j.tempo`, `j.key`,
   `j.chord_preview`, `j.sections`) into the per-row chips as they arrive. The
   existing done/error/spinner machinery is unchanged.
4. Move Basic Pitch persist + iReal fetch to a background thread after
   `update("done", …)` so step 6 is instant.

Steps 1–3 are additive and reversible; step 4 changes job ordering. All touch the
server/client, **not** the model math. None of this was built here — this doc is
the scope. Recommended build order if greenlit: (4) first (biggest latency win,
smallest surface), then (1–3) for the progressive chips.
