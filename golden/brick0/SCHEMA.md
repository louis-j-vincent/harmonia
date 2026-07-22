# Brick 0 — frozen real-audio benchmark: GT schema (LOCKED)

This document **locks** the per-song frozen ground-truth format consumed by the
Brick-0 scorer (`harmonia/eval/accuracy_score.py`). One JSON file per song lives
under `golden/brick0/`. A *separate* proposal pipeline (next task) writes these
files; **the scorer only reads them and never re-aligns**.

See `docs/known_issues.md` → "STEP 11" (why the benchmark is needed + the data
inventory) and "BRICK 0 GREENLIT" (guardrails), and
`data/real_audio_benchmark/PROTOCOL.md` (the non-circular alignment spec this
schema serves).

## Why frozen + why these fields (the non-circularity contract)

The old measurement (`scripts/validate_against_ireal.py`) was **circular**: it
DTW-aligned the iReal GT to the model's *own* predictions, so a wrong model bent
the GT to fit itself and scored its error as zero (CLAUDE.md #3). Brick 0 kills
that: the GT timeline is produced + **hand-verified offline** on the real audio
clock, then **frozen**; the scorer overlaps the prediction against the frozen GT
by direct lookup, no re-alignment.

The two independent inputs must come from **different sources**:

- **Chord symbols** ← iReal Pro charts (retain the `/bass` slash → *sounding*
  bass, the project target since 2026-07-16).
- **Beat/downbeat grid** ← an **INDEPENDENT** Beat This! pass (a *producer*,
  never the scorer, and never the model's own decode). Used offline to place
  iReal bars on the audio clock and to sanity-check form; frozen into the GT.

## File: `golden/brick0/<song_id>.gt.json`

```jsonc
{
  "song_id": "blue_bossa",            // stable slug; matches the filename stem
  "title": "Blue Bossa",
  "audio_path": "docs/audio/blue_bossa.m4a",  // repo-relative (or absolute)
  "chart_source": {                   // provenance of the chord symbols
    "ireal_file": "jazz1460",         // data/ireal/<file>.txt
    "index": 47,                      // tune index within that playlist
    "tune_title": "Blue Bossa"
  },
  "transpose_semitones": 0,           // semitones applied to the iReal chart to
                                      //   match the sounding key of THIS audio
  "form": {                           // how the chart was tiled onto the audio
    "section_order": ["A", "A"],      // section labels in play order
    "repeat_counts": {"A": 2},        // repeats per section label
    "intro_bars": 0,
    "outro_bars": 0
  },
  "bar1_anchor_time": 0.83,           // audio time (s) of the first downbeat of
                                      //   bar 1 (the form's t=0 anchor)
  "downbeat_times": [0.83, 2.9, ...], // INDEPENDENT Beat This! downbeats (s).
                                      //   Informational/for form verification;
                                      //   the chord scorer does NOT consume them.
  "gt_chords": [                      // the frozen chord timeline (audio clock)
    {
      "t0": 0.83,                     // onset (s, audio clock)
      "t1": 2.90,                     // offset (s)
      "root_pc": 0,                   // functional root pitch class 0-11 (C=0)
      "quality": "min7",             // Harte/pipeline quality token (see below)
      "bass_pc": 0,                   // SOUNDING bass pc via sounding_bass_pc()
      "label": "C:min7"              // human-readable Harte label (not scored)
    }
    // ... contiguous, non-overlapping, sorted by t0
  ],
  "verified": false                   // HARD GATE — see below
}
```

### Field notes

- **`root_pc`** — functional root (0-11). May be `null` only for a no-chord
  span; then `quality` must be `"N"` and `bass_pc` `null`.
- **`quality`** — a quality token from the shared vocabulary used by the shipped
  pipeline (`harmonia/models/musx_bass._MUSX_Q_TO_SEV` values + Harte
  shorthands): `maj min 7 maj7 min7 dim dim7 hdim7 aug sus2 sus4 6 maj6 min6 9
  maj9 min9 11 13`, and `N` for no-chord. Both GT and prediction go through the
  same `chord_family` / `_sev_class` maps, so partial-credit and MIREX levels
  are consistent across the two.
- **`bass_pc`** — the **SOUNDING** bass pitch class, computed by the producer
  with `harmonia.data.corpus_schema.sounding_bass_pc(label, root_pc)` (root
  itself for a root-position chord; the slash note for `/bass`). This is the
  target of the *sounding-bass root* metric — NOT the functional root.
- **`t0`/`t1`** — audio-clock seconds. The `gt_chords` list should be
  contiguous and non-overlapping; the loader warns on overlaps and raises on a
  non-positive span. Gaps are treated as no-chord.
- **`verified`** — **HARD GATE.** Defaults `false`. The scorer
  (`score_song`) REFUSES to score a song with `verified=false` unless the
  caller passes `allow_unverified=True`, which additionally stamps the result
  `verified=false` and warns loudly. A song flips to `true` only after a human
  confirms transpose + form + bar-1 anchor + the chord-at-time timeline against
  the audio. No fabricated number can ship off an unverified chart.

## What the scorer reports (per song + pooled micro-average + N)

All metrics are duration-weighted over the GT span (frozen-GT lookup):

| metric | definition |
|---|---|
| `mirex_root` | root pc matches (duration-weighted chord-symbol recall) |
| `mirex_majmin` | root + maj/min bucket ({min,hdim,dim}→min, else maj) |
| `mirex_sevenths` | root + seventh-class (maj vs maj7, min vs min7, 7, …) |
| `partial_credit` | root + parent FAMILY (maj7-for-maj → credit) |
| `strict` | root + full (canonicalised) quality token |
| `bass_root` | **sounding** bass pc matches (`sounding_bass_pc` target) |

`mir_eval.chord` is used as an independent cross-check on root/majmin/sevenths
when installed, but the reported numbers come from the module's own
deterministic overlap engine so the headline does not depend on an optional
package being present (reproducibility is a hard requirement).

## Reproducibility contract

Same frozen GT + same audio file ⇒ same numbers. The scoring math is fully
deterministic; the prediction is produced by the shipped pipeline
(`infer_chords_v1` with `accuracy_score.SHIPPED_CONFIG` ==
`PipelineConfig.live_defaults()`), which is deterministic given fixed model
weights (torch eval-mode nets, no sampling). m4a/mp3 inputs are decoded to WAV
with ffmpeg (deterministic) because `soundfile` cannot read them.
