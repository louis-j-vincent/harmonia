# `harmonia.dataset` — high-precision chord-training-dataset harvest

Turn (iReal chart, YouTube recording) pairs into **clean (audio-segment → chord)
training rows** via the existing Brick-0 aligner + a strict **3-way confidence
gate**. The dataset's value is *clean* pairs, not coverage:

> **Precision ≫ recall.** Better 40 % of a song at ~96 % label-precision than
> 100 % at 85 %. A false positive is a wrong chord label that poisons training.

Audio is **never copied into the repo** — every row references audio by
`path + [t0, t1]` offsets. Manifests and the download/proposal caches are
gitignored.

## The 3-way gate (the crux)

Each aligned segment (one proposed chord span) is classified from **non-circular**
signals (raw CQT chroma vs chart chord-tones + drum beat-lock — never the shipped
model's own decode):

| bucket | meaning | destination |
|---|---|---|
| **CLEAN** (1) | confident **and** chart matches the recording | `manifest.jsonl` — the high-precision training bulk |
| **REVIEW** (2) | confidently placed in **time** but chart **≠** recording (a substitution / reharmonisation) | `review_manifest.jsonl` — a to-decide queue for ear-labelling |
| **DROP** (3) | low confidence not attributable to a clean substitution | discarded |

The discriminator between REVIEW and DROP is the aligner's **divergence
detectors**: a span that is time-aligned (beat-locked, cross-rep consistent,
covered, transpose-clear) but harmonically off is a *substitution* (valuable, not
noise — e.g. Georgia's charted F#dim played as B7); a span that is merely
poorly/ambiguously aligned is dropped. Bucket 2 feeds a future active-learning
"ear-training" loop, so its rows carry the chart suggestion + why it's uncertain.

## Public API

```python
from harmonia.dataset import add_song, harvest_song, gate_segment, write_manifests

# grow the dataset with one call: chart + YouTube audio -> gated rows
res = add_song({"ireal_file": "pop400", "tune_title": "Stand By Me"},
               "https://youtu.be/hwZNL7QVJjE", song_id="stand_by_me")

# or harvest a song config directly (align -> gate)
res = harvest_song({"song_id": "...", "title": "...", "audio": "docs/audio/x.m4a",
                    "ireal_file": "pop400", "tune_title": "..."})
res.clean_rows      # bucket-1 rows      res.review_rows   # bucket-2 rows
res.dropped         # bucket-3 (t0,t1,label,reason)   res.stats

write_manifests([res])   # -> data/chord_dataset/{manifest,review_manifest}.jsonl
```

`gate_segment(SegmentSignals, GateConfig) -> GateDecision` is **pure and
audio-free** (all per-song normalisation happens in `harvest.py`), so the gate
logic is fully unit-tested with no audio/models — see
`tests/test_dataset_gate.py`.

### Modules

- **`gate.py`** — the pure 3-way gate + `GateConfig` (thresholds) +
  `SegmentSignals` (pre-normalised inputs) + `GateDecision`.
- **`harvest.py`** — `run_aligner` (imports `scripts/brick0_propose.py`
  READ-ONLY, into a private `.dataset_cache/`, never touching `golden/` or
  `docs/brick0_review/`) → `compute_beat_lock` (drum tracker) →
  `build_segment_signals` → `harvest_song` → `write_manifests`.
- **`ingest.py`** — `add_song(chart_ref, youtube_ref)`, a standalone
  `YouTubeFetcher` (yt-dlp; caches to `data/dataset_audio/`, reuses `docs/audio/`),
  and `resolve_chart` (via `harmonia.irealb_fetcher`, read-only).

## Manifest schemas

`data/chord_dataset/manifest.jsonl` (clean GT — one row per contiguous CLEAN run):

```json
{"audio_path": "docs/audio/blue_bossa_150bpm_backing_track.m4a",
 "t0": 4.725, "t1": 54.325,
 "chords": [{"t0": 4.725, "t1": 7.925, "label": "C:min7", "root_pc": 0, "bass_pc": 0}, ...],
 "confidence": 0.80, "song_id": "blue_bossa_backing", "source": "ireal+youtube"}
```

`data/chord_dataset/review_manifest.jsonl` (substitution / to-decide queue):

```json
{"audio_path": "...ray_charles_georgia....m4a", "t0": 26.81, "t1": 30.58,
 "chart_suggestion": "A:maj/C#", "substitution_suggestion": "G:dim",
 "divergence_evidence": {"reasons": ["DIVERGENCE: strong mid-span split ..."],
                         "agreement": 0.31, "beat_lock": 0.7, "xrep_divergence": false},
 "confidence": 0.42, "song_id": "georgia_on_my_mind", "source": "ireal+youtube",
 "status": "unlabeled"}
```

`root_pc`/`bass_pc` are the sounding pitch classes (project target since
2026-07-16; `bass_pc` via `corpus_schema.sounding_bass_pc`). `label` is a
Harte-style token from the shipped vocabulary.

## Gate signals & calibrated operating point

`GateConfig` defaults are the calibrated point. A CLEAN emit requires **all** of:

| signal | default | source |
|---|---|---|
| `agr_keep` — per-chord Pearson agreement | **0.34** | aligner `agreement_detail.per_chord` |
| `agr_norm_keep` — agr / this-song's agreement ceiling | 0.60 | aligner `harmonic_agreement` |
| `beat_lock_min` — per-song-normalised drum reliability over span | 0.45 | drum tracker `reliability` |
| `require_octave_locked` | true | drum tracker `octave_locked` |
| `require_coverage_full` / `drop_near_gap` | true | `section_alignment.gaps_unlabeled` |
| `transpose_margin_min` — best − runner-up transpose score | 0.03 | `agreement_detail.per_transpose` |
| `require_xrep_consistent` | true | `refinement.cross_repetition` |
| `frac_low_veto` — song-level broad-misalignment veto | 0.35 | `harmonic_agreement.frac_low` |

`agr_keep` is the primary precision knob; it was swept during calibration.

### Calibration (non-circular, on the 5 frozen ear-verified songs)

Each song is re-aligned **raw** (label overrides stripped; timing anchors kept
where the frozen GT legitimately used them), so the proposal is independent of
the ear-verified GT. Label precision = duration-weighted fraction of CLEAN spans
whose `(root_pc, quality)` matches the frozen GT.

Pooled precision/recall vs `agr_keep` (5 frozen songs):

| `agr_keep` | P strict | P partial | recall | wrong-chord | outro over-extend |
|---|---|---|---|---|---|
| 0.28 | 0.954 | 0.958 | 0.72 | 0.025 | 0.017 |
| **0.34** | **0.955** | **0.959** | **0.63** | 0.027 | 0.014 |
| 0.40 | 0.958 | 0.962 | 0.54 | 0.027 | 0.011 |
| 0.48 | 0.954 | 0.961 | 0.37 | 0.039 | 0.000 |

Precision is **flat at ~0.955** across thresholds because the residual error is
**not** gate noise — it is two *structural aligner/form* failures the gate cannot
see (the gate faithfully passes confident alignments):

Per-song CLEAN precision @ `agr_keep=0.34`:

| song | P strict | note |
|---|---|---|
| blue_bossa (timing-anchored) | **1.000** | |
| blue_bossa_backing | **1.000** | |
| stand_by_me | **1.000** | |
| every_breath_you_take | 0.88 | ~8 % is outro over-extension past the song's end |
| bein_green | 0.76 | one confidently-**mis-placed section** (high-agr aligner form error) |

**Operating point: `agr_keep = 0.34`** — on the precision plateau (~0.955 strict /
~0.96 partial) at the best recall (~0.63). Three of five frozen songs harvest at
**100 % precision**. The two shortfalls are flagged for the forthcoming
downbeat/form model, not the gate.

### Behaviour on hard / held-out songs (as intended)

- **Autumn Leaves** (unfrozen, hard jazz jam): the gate emits **nothing** — 200
  spans fail the transpose-margin gate (its key is genuinely ambiguous, margin
  0.027) and the rest fail beat-lock. Keeping it would have emitted ~104 s at only
  ~47 % precision. Correctly refused.
- **Let It Be** (held-out): well-aligned overall (`frac_low` 0.015) but only
  **32 %** harvested — the solo-piano intro and dense/ambiguous spans DROP; the
  confident band sections stay. Exactly the "drop the low-confidence regions"
  behaviour.

## Demo harvest (already-local audio)

`harvest_song` over the 9 locally-available BATCH1 songs (recommended mode:
timing anchors kept, label overrides dropped), `agr_keep=0.34`:

- **448** clean (segment → chord) pairs, **18.2 min** of audio; **33** substitution
  candidates in the review queue.
- Frozen-only subtotal: 5 songs, **343** clean pairs, **13.7 min**.
- Per-song kept-fraction ranges from 0.00 (Autumn — refused) to 0.97
  (blue_bossa_backing).
- **Estimated label precision at the gate ≈ 0.955 strict / 0.96 partial** (from the
  frozen calibration).

## Downbeat fold (landed 2026-07-23)

`compute_beat_lock(use_downbeat=True)` (the default) now runs the global-phase
downbeat resolver (`harmonia/align/downbeat.py`, imported READ-ONLY) on the SAME
drum track + raw CQT chroma + bass-salience stream, and folds its per-song
`confidence` / `flagged` into every span's `beat_lock` via `_downbeat_gain`.
`gate.py` is unchanged (it still consumes a single [0, 1] beat-lock).

The fold is **directional and precision-first**:

- **Unflagged, confident downbeat → bounded BOOST** (`1 + 0.30·clip(conf/0.50)`),
  lifting recall on well-placed spans of the confident-downbeat pop songs.
- **Flagged (ambiguous-phase / weak-margin / phase-flip) → ABSTAIN** (gain 1.0):
  no boost, so a flagged song can never be promoted to CLEAN *by the downbeat*
  ("never emit on a shaky downbeat") — but it is **not** capped DOWN. The downbeat
  *phase* is orthogonal to chord-label correctness, so a literal "cap beat_lock
  low" would DELETE verified-correct rows and regress the frozen bar: the
  chroma-flat 9-min Blue Bossa jam is downbeat-FLAGGED (conf ~0.03) yet harvests
  at 100 % label precision (human-anchored timing + high agreement). Capping it
  removes ~235 s of perfect rows and drops duration-weighted pooled frozen
  precision **0.945 → 0.924** — measured, not hypothetical. Precision is paramount,
  so the flag withholds LIFT, it does not tear down independently-justified locks.

Measured demo lift (9 BATCH1 songs, fold OFF → ON): **444 → 449** clean pairs,
**18.02 → 18.32** min, review queue **33 → 33**; frozen-5 strict precision
**0.944 → 0.945** (held, on a consistent scorer; ~0.955 under the calibration
scorer). Recall lifted on the confident-downbeat pop songs — stand_by_me kept
0.575 → 0.650, bein_green 0.526 → 0.568 (its precision *rose* 0.644 → 0.670, the
boosted spans were correct); blue_bossa_backing already saturated. The FLAGGED
jazz/rubato songs (Autumn, Blue Bossa jam, Georgia, Close To You, Let It Be) are
byte-identical (correctly still-conservative — they do NOT lift).

## Integration points (deferred to lane-reconcile, per concurrent-session safety)

- **Fresh-chart persistence** — the aligner consumes charts as `(ireal_file,
  tune_title)` from `data/ireal/`. A chart fetched fresh from `irealb_fetcher`
  (an `irealb://` URL) must be persisted into that on-disk format before it can be
  aligned; that one bridge is left to reconcile (the `irealb_fetcher` lane is
  mid-edit). Songs whose chart is already in `data/ireal/` harvest end-to-end
  today.

## Reproduce

```bash
PYTHONPATH=$PWD .venv/bin/python -m pytest tests/test_dataset_gate.py -q   # audio-free gate tests
# harvest is driven programmatically via harmonia.dataset.harvest_song / add_song.
```
