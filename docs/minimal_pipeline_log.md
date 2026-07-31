# HARMONIA-MIN — minimal pipeline rebuild log

Branch `feat/minimal-pipeline` (from `feat/chord-context-prior`).
Mission: rebuild a working audio → interactive chord chart path from the few
good bricks, ~300 lines of new orchestration, no old-pipeline imports.

## 2026-07-31 — Milestone 1 DONE: chart rendering end-to-end on :7772

`python -m harmonia_min.server` → http://localhost:7772 — the app shell
renders, search finds local audio, Analyse runs the thin pipeline (~4 s on
cached musx posteriors), the chart opens, audio plays, the playhead tracks
the right bar (verified in headless Chrome, screenshots in session scratchpad;
bar-label arithmetic checked against the grid by hand).

### Branch decisions

Skipped ALL ahead branches (none touches a copied brick — verified by diff):
- `feat/harmonic-key` (14 ahead, ACTIVE yesterday): docs + scratchpad UG-scoring
  analysis only. Merging an active session's branch = cross-session conflict risk.
- `feat/downbeat-section-monolith` / `feat/section-cnn*` (11 ahead): section
  detection — out of milestone-1 scope, and sections get scoped with Louis later.

Port note: a leftover `python3 serve.py 7772` (static preview of
`scratchpad/iphone_view`, started 2026-07-30 18:15) was squatting the assigned
port; killed. Restart it on another port if still wanted.

### What was KEPT (copied bricks, each verified against raw data before use)

| brick | copied to | verification |
|---|---|---|
| `musx_redecode.py` | `harmonia_min/musx.py` | posterior layout (T,73)/(T,13)/… rowsums=1.0, T·23.22ms = ffprobe duration ±0.03s |
| beatthis wrapper (re-extracted, ~100 L) | `harmonia_min/beats.py` | This Love 93.75 BPM (true ~95, right octave), 3.94 beats/bar |
| `key_profiles.py` | verbatim | C-triad chroma → "C major" conf 0.91 |
| `nnls_features.py` | copy, scratchpad MLP inlined | checkpoint loads, 7 qualities |
| `span_rescore.py` + `chord_context_prior.py` | copies, QUAL5/SEMITONE_NAMES inlined | import + prior npz loads (NOT yet wired to a route — lock UX is milestone 2) |
| `app_shell.html` | verbatim | renders against the new server |

New code: `labels.py` (musx→iReal-tail mapping, closed 26-label vocab, raises
on unknown — no silent maj default), `pipeline.py` (~230 L orchestration),
`server.py` (~200 L Flask).

### What was CUT

`chord_pipeline_v1` (4,779 L), `chord_head.py` post-passes (Occam, vocab fold,
section arbiter), `chord_hmm`, `section_structure`, repeat folding/endings,
ProgressionEncoder, LLM priors, `/api/reinfer`, annotations, iReal import,
jam/record modes, billboard corpus. Server routes for these return 404; the UI
degrades as designed.

### Deliberate DIVERGENCES from the old pipeline

1. **Bar grid = Beat This!'s real downbeats**, not a synthetic constant-tempo
   lattice, not chord-onset anchors (known_issues "BOTH CLOCKS ARE SYNTHETIC").
   Synthetic bar lines only at the edges (pickup/tail), median-length.
2. **No librosa anywhere, no silent fallback**: beat tracking failure raises
   `BeatTrackingError`. m4a is ffmpeg-transcoded to wav before Beat This!
   (the 2026-07-30 lesson, baked in).
3. **Labels straight from the beat-grid re-decode** — no post-passes at all.
4. **Confidence is acoustic** (mean musx triad posterior over the segment's
   own frames), not the repetition heuristic (whose AUC was 0.480).
5. **Key from decoded chords** (duration-weighted chord tones → KS profiles),
   not a separate chroma pass.
6. **One unfolded section**; `barSpans[r] = [[grid[r], grid[r+1]]]` — the
   playhead map is trivially correct by construction.

### Diffs vs the old pipeline's charts — SIGNAL for Louis, not failures

3 songs analyzed (cached musx posteriors): This Love, Let It Be, Stand By Me.

| song | old chart | new chart | reading |
|---|---|---|---|
| This Love | 81 bars, bar 0 `C G` (roots only at my extraction level), later `D Bb` | 81 bars, `G/B C- | F-7 | % | F-/Ab G/B`, `Dø7 G/B` | new = the actual changes incl. inversions; old/new disagree on bar-0 content and several roots (D/Bb vs Dø7) — worth an ear pass |
| Let It Be | 72 bars, bar 0 crams `C G A` (3 chords/bar) | 77 bars, `C G | A- F | C F | C` | old bar grid glued the intro; new opens on the real changes. 72 vs 77 bars = different grids, not an error flag per se |
| Stand By Me | 89 bars, bar 0 labeled `C` (in A major!) | 93 bars, bar 0 `N.C.` then held | the old "C" was a mislabeled no-chord; new shows the honest N.C. — but 15 intro bars of bass riff stay unlabeled by musx (chord-vs-no-chord discrimination, the known top priority) |

Other observations:
- All 3 songs selected **musx latency 0 ms** (the old 7-song study measured
  60–340 ms optima with per-song selection). Different beat backend feeding the
  grid may explain it; unverified — worth a look before trusting the latency knob.
- Small fixes made during the build: leading-pickup coverage (a chord sounding
  before the first detected downbeat got dropped — Let It Be's opening C was
  20 ms early), leading-N.C. rule (an N segment > half a bar = real played
  intro, shorter = silence).

### How to run

```
.venv/bin/python -m harmonia_min.server        # port 7772, NEVER 7771
# charts persist in harmonia_min/state/charts/*.json
# analyze from the UI (search a local song → Analyse) or pre-seed via pipeline.analyze()
```

### Next milestones (to scope with Louis)

lock/annotation UX (span_rescore + context prior are copied and load — need
`/api/context_rescore` wired), iReal import, sections/folding.

## 2026-07-31 (later) — Grid arrangement REDONE after Louis's review

Louis checked This Love on :7772: musx chords right, **arrangement onto the
grid wrong**. Root cause: my v1 assigned chords to bars by TIME CONTAINMENT
(which bar's [t0,t1) holds the onset). Redecode boundaries sit on beats only
up to the 23.22 ms musx frame grid, so a chord changing ON a bar line lands a
few ms either side → wrong bar → every bar-opening chord rendered as the tail
of the previous bar + a held "%". The live app already solved exactly this;
v2 is a minimal copy of its method (render_youtube_chart.py::
chart_to_interactive_inputs):

- chord onset → nearest detected-beat INDEX; bar/beat = integer arithmetic
  ((idx − off) // bpb, % bpb) — no time containment anywhere;
- bar phase `off` = modal residue of the tracker's downbeat indices, then the
  **harmonic re-anchor** (live thresholds 55%/15%) lets the chords out-vote
  the tracker when they overwhelmingly agree on a non-zero beat-in-bar;
- bar time spans = real beat times at the boundary indices (playhead map).

Result (browser-verified): This Love `G/B | Cm | Fm7 | Dø7 Fm/Ab` one chord
per bar, 4 held bars (was: chords straddling bars + % everywhere). Let It Be
`C G | Am F | C G | F C` (the real two-per-bar form, 0 held). Stand By Me
N.C. intro + E at bar 12. Re-anchor fired on none of the three — the downbeat
vote alone was right; it stays as the safety net it is in live.

## 2026-07-31 — Two one-beat-early chords (Louis, This Love 1:56 / 2:52): diagnosed + fixed

Not a layout bug — the layout put chords exactly where the decode put them
(all onsets within ±11 ms of a beat, all 81 downbeats on one lattice). At the
two phrase turns the raw musx frames go AMBIGUOUS for ~a beat (conf 0.3–0.6
vs 0.9+ elsewhere; band fill), and the flat change penalty made last-beat vs
next-downbeat equal cost — noise picked the early beat. Fix: pass Beat This!
downbeats into the re-decode → the vendored decoder's downbeat-graded
transition costs (15/45/100) break exactly that tie toward the downbeat.
Verified: Fm 116.61→117.21 (own bar), Cm 172.13→172.78 (joins Fm7);
beat-in-bar distribution goes 73/1/38/7 → 76 on beat 0, 41 on beat 2, zero
stragglers. Old study had graded at −0.17 pp label overlap (a wash) — kept ON
here because chart placement is the product. Let It Be / Stand By Me unchanged.

## 2026-07-31 — Dormant-route sweep + brick schema

Audit of everything that could switch chart-grid behavior. Found and removed:
- `redecode(downbeat_times=None)` default — a future caller would silently get
  the flat-penalty tie back. Now a REQUIRED keyword (TypeError without it).
- `redecode_audio()` convenience wrapper (unrouted second call site) — deleted.
- `key_profiles.activations_to_chroma` / `detect_modulations` — tied to the
  cut Basic Pitch front-end — deleted.
- Stale "downbeat grading not recommended" note in musx.py replaced with the
  current placement rationale.
Verified: all 3 charts byte-identical before/after the sweep. Env flags in the
chart path: ZERO (only HARMONIA_MUSX_DIR, clone dir resolution). span_rescore/
chord_context_prior/nnls_features stay as unrouted milestone-2 bricks — no
server route reaches them.

Brick schema: docs/harmonia_min_schema.png.

## 2026-07-31 — Harmonic key analysis added + Louis's two bar rules

`harmonia_min/harmonic_key.py` (~330 L): minimal port of the v7c colour pipe
(scratchpad/colour_hmm_song.py::decode_segments, feat/harmonic-key). Kept: the
five stages (chord chroma → tonic track → mode audit → colours → feedback) and
every v4.1 calibration constant. Cut: structure fold (needs the cut section
machinery; the script treats no-fold as graceful fallback), the global MODE
mutation (now a parameter), the third duplicate quality table (labels.py is
the vocabulary). Wired into pipeline.analyze best-effort: global key = longest
tonic segment; keySegments in the payload; colour/inflect/flag/sug per chord.
Port verified faithful: identical segments/mode/suspects to the reference
script run on the same input.

Bar rules (Louis): (1) a bar lists all chords sounding in it — carried chord
re-written at beat 0 when a new chord arrives mid-bar (carry-marked, excluded
from repetition counts); held bars keep "%". (2) bar cells are a bpb-quarter
grid; each chord sits at its beat quarter. Cap 2→bpb chords/bar; pickups
display at beat 0.

Verified in browser: Let It Be `C·G|Am·F` quarters + carries; This Love
mid-bar G at its real beat 2; Close to You modulation segmented at 98.0s.

FLAGS for the harmonic-key session (not fixed here — their science):
* v7b naming names Close to You seg-2 **Ab**; the v7b commit claims Db and UG
  adjudication (3787cca) says Db. Reproduced on the v7b-era script itself —
  the commit's 3/3 claim does not reproduce. Krumhansl's saturated argmax
  decides between candidates; duration itself favours Db (53s vs 38s).
* colour_chart_song.py::beat_grid reads data/cache/raw_beat_times_v2 — the
  cache serving/audio.py documents as 100% stale librosa beats (v3 superseded).
