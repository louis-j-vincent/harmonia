# Hands-on guide to Harmonia

_Written 2026-07-17. Purpose: get you back to running the pipeline yourself without
wading through the ~65k LOC of `scripts/` sprawl. Read time ~12 min. Everything with
a ✅ was executed/loaded and verified; ⚠️ was read from code but not run._

**One thing to fix in your head first.** There are TWO parallel systems in this repo
and they do not share weights:

- **The running app / production pipeline** (`infer_chords_v1`, the models in
  `harmonia/models/*.npz`). BP48 features, POP909/YouTube-tuned. This is what
  `render_youtube_chart.py` and `harmonia_server.py` call. This is what produces the
  charts you look at.
- **The research heads** (root 0.89, quality-balanced 0.735, etc.) trained offline on
  RWC/Billboard/JAAH corpora, saved under `data/models/*.pt`. These are **not wired
  into the app** — there is a documented feature-domain gap (McGill-NNLS-clean vs
  production BP48) that has never been bridged. When you see a great SOTA number, it
  is almost always a research head, not the app.

Keep those separate and most of the confusion in `known_issues.md` dissolves.

---

## 1. The pipeline in one page

Six stages, chained in `harmonia/pipeline.py` (`HarmoniaPipeline.run`) and, in the
current production form, in `harmonia/models/chord_pipeline_v1.py`
(`infer_chords_v1`). One canonical file per stage:

1. **Pitch extraction** — `harmonia/models/stage1_pitch.py` (`PitchExtractor`).
   Runs Basic Pitch (ONNX) over the audio → an 88-key piano-roll-ish activation at
   **86.1328125 Hz** (`BASIC_PITCH_FRAME_RATE = 22050/256`; this constant is error
   pattern #1 — don't touch it). Cached to `data/cache/*.npz`; the cache key does NOT
   cover module constants, so clear the cache if you change any.

2. **Beat / rhythm** — `harmonia/models/rhythm.py` (`RhythmAnalyser`). Tempo + beat
   grid (librosa default, madmom optional). Pitch activations get pooled onto beat
   cells here. (Song 002's tempo is a known 2× octave error — inherited by anything
   using the audio beat tracker on it.)

3. **Segmentation** — `harmonia/models/structure.py` (`Segmenter`), with
   section-level structure in `harmonia/models/section_structure.py`. Builds a
   self-similarity matrix over beat-pooled chroma and cuts it into segments; a chord
   is assumed roughly constant within a cell.

4. **Key inference** — `harmonia/theory/key_profiles.py` (`infer_key`,
   `detect_modulations`). Krumhansl-style key-profile correlation per segment,
   yielding a global key + modulation points.

5. **Chord HMM / decoding** — `harmonia/models/chord_hmm.py` (`ChordInferrer`) plus
   the heavier `infer_chords_v1` orchestration (root model → quality/family
   classifier → semi-Markov duration decode → optional joint decode). This is where
   the 12-way root and 7-way quality come together into labels.

6. **Chart output** — `harmonia/pipeline.py::ChordChart` and the interactive
   renderer `harmonia/output/chart_interactive.py` (do not edit — actively iterated).

The 7-way quality vocabulary, fixed order everywhere:
`["maj", "min", "dom", "hdim", "dim", "aug", "sus"]`.

---

## 2. Run a prediction on a new song right now

**The one-liner (✅ verified — ran end-to-end on `demo_audio/example_clean.wav`,
produced 46 chords, key F major, tempo 139.7 BPM):**

```python
from pathlib import Path
from harmonia.models.chord_pipeline_v1 import infer_chords_v1

chart = infer_chords_v1(Path("demo_audio/example_clean.wav"))
chart.print()                 # pretty table
chart.save_json("out.json")   # {label, start_s, end_s, duration_beats, confidence}
```

Any format `soundfile` reads works. Defaults are the production-tuned ones (semi-Markov
on, joint decode on, ctx model on when its cache exists). No checkpoint path to pass —
the models load themselves from `harmonia/models/*.npz`.

**With a browser chart + YouTube download** (⚠️ read, not run this session; needs
`yt-dlp`):

```bash
.venv/bin/python scripts/render_youtube_chart.py https://youtu.be/XYZ --title "My Song"
.venv/bin/python scripts/render_youtube_chart.py --audio song.wav   # skip download
```

Writes an interactive HTML chart (same viewer as `docs/plots/inferred_*.html`) and
opens it. Under the hood it just calls `infer_chords_v1` and adapts the `ChordChart`
to the renderer.

**Heads-up (✅ observed on a clean run):** two optional caches are missing on a fresh
checkout, and the pipeline **silently degrades** rather than failing:
- `data/cache/duration_prior_jazz1460.npz` missing → semi-Markov decode disabled.
- `harmonia/models/ltas_family_dist.npz` (ctx model) missing → context family model
  disabled.
It still runs and produces a chart, but you're not getting the full model. If you
want them, run `scripts/build_duration_prior_jazz.py` (referenced in the warning).

---

## 3. Run the standard evaluation

The established convention this campaign is **multi-seed, song-grouped (song-stratified)
cross-validation** — never a single split (too high-variance on these small corpora),
and always report **balanced** accuracy + per-class recall on quality, never raw
accuracy on the imbalanced set.

Canonical runner (⚠️ read, not run):

```bash
# RWC-Popular, 6 song-stratified seeds, root-roll augmentation on:
.venv/bin/python scripts/train_jaah_cv.py \
    --corpus data/cache/rwc/rwc_bp48_fixed.npz --seeds 6 --roll
# JAAH (default corpus): just drop --corpus
```

`train_jaah_cv.py` is the reference implementation of the methodology: it reuses
`train_real_audio_final.py`'s `_train_head` / `_eval` / `_augment_root_by_roll`
verbatim, does song-grouped splits, and prints root acc / quality-balanced acc / dom
recall as mean ± std. `scripts/chordformer_rwc_cv.py` uses the *identical* CV harness
for the factored-slot experiment — copy either as your template. Both filter records
through `corpus_schema.filter_by_match(match, minimum=MatchQuality.EXACT)`.

Note: this evaluates the **research heads on cached features** (`.npz` corpora),
which is the layer where all the recent SOTA work happens. End-to-end pipeline eval
(audio → labels → MIREX overlap) lives in `harmonia/eval/mirex_eval.py`
(`evaluate_song`, `evaluate_pop909`) — a different, older axis.

---

## 4. The ~10 files that actually matter

Load-bearing modules in `harmonia/` (ranked roughly by how central they are; import
counts are across `scripts/ + harmonia/ + tests/`):

1. `harmonia/models/stage1_pitch.py` — `PitchExtractor`, the ONE feature front-end;
   the frame-rate constant lives here. Every corpus builder routes through it.
2. `harmonia/models/chord_pipeline_v1.py` — `infer_chords_v1`, the current production
   inference orchestration (root → quality → decode). This IS the pipeline.
3. `harmonia/data/corpus_schema.py` — single source of truth for the `match`-quality
   enum, `save_corpus`/`load_corpus`, and **`sounding_bass_pc`** (the new
   sounding-bass-vs-functional-root resolver). Read this before touching any corpus.
4. `harmonia/eval/mirex_eval.py` — `evaluate_song` / `evaluate_pop909`, the MIREX
   weighted-overlap metrics. Pin these; a silently-shifting metric is the worst bug.
5. `harmonia/models/rhythm.py` — `RhythmAnalyser`, tempo + beat grid (18 importers).
6. `harmonia/models/structure.py` — `Segmenter`, SSM segmentation (14 importers).
7. `harmonia/theory/key_profiles.py` — `infer_key`, `detect_modulations` (15
   importers).
8. `harmonia/theory/chord_vocabulary.py` — the chord/quality vocabulary (27
   importers); the shared label alphabet.
9. `harmonia/data/pop909_parser.py` + `harmonia/data/billboard_translator.py` — the
   ONLY two chord-label translators; `/bass`, colon-quality, maj↔maj7 family logic.
10. `harmonia/pipeline.py` — `HarmoniaPipeline` + `ChordChart` (the output
    dataclass). The clean high-level entry / chart schema.

(`harmonia/models/chord_hmm.py` and `harmonia/data/midi_renderer.py` are honorable
mentions — the HMM decoder proper, and the render/synthesis utility with the highest
raw import count because so many scripts sonify their output.)

---

## 5. What NOT to look at

This is normal research sprawl, not a mess — the package (`harmonia/`, ~20k LOC) is
clean; the noise is one-off experiment scaffolding around it. To save yourself time,
skip:

- **The ~30 `scripts/train_*.py` variants.** Most are superseded one-shots
  (`train_beat_seq_model_v2/v3/v4`, `train_billboard_{batched,chord_model,from_features}`,
  `train_yt_{chord_model,exact_matches,real_audio}`, `train_online*`, …). The only
  ones current are `train_real_audio_final.py` (the shared trainer functions) and the
  two CV wrappers `train_jaah_cv.py` / `chordformer_rwc_cv.py`.
- **`scratchpad/`.** Everything here is this session's agent output — `bass_*.py`,
  `joint_*.py`, `oracle_*.py`, `*_result.json`, `*.log`, `*.png`. Verdicts worth
  keeping have already been distilled into `known_issues.md`; the raw files are
  reproduction scripts, not a library. Don't build on them.
- **`scripts/harmonia_server.py`** (~400 KB / 7k LOC). Its own subsystem (the web
  app, GT-playalong routes, annotator tools). Fine to run, but not something to read
  top-to-bottom to understand the model.
- **The many `data/models/*.pt` research heads** (`quality_head_trigram_v1`,
  `root_head_multihead_v1`, `bass_detector_v1`, `billboard_*`, `yt_*`, `cnn_*`,
  `lstm_*`). These are experiment artifacts on the NNLS/RWC/Billboard side, NOT wired
  into the app. The app's weights are the smaller set in `harmonia/models/*.npz`.
- The stale clone at `~/harmonia/` — never work there; this repo is canonical.

If you want the full structural audit, it's already written:
`docs/refactoring_suggestions.md` (findings) + `docs/refactoring_delegation_plan.md`
(the not-yet-executed cleanup plan).

---

## 6. Current state of the art, in one table

**Read the caveats — these numbers do NOT all measure the same thing.** This is the
exact trap that produced a fabricated-number incident on 2026-07-16 (an agent claimed
"NNLS root stable across corpora, Billboard 0.379 → JAAH 0.378" by conflating three
different experiments). The authoritative de-confliction is the **PHASE-0 AUDIT**
entry in `known_issues.md` (2026-07-17); everything below is pulled from there and
from the JAAH NNLS-vs-BP48 entry.

| Metric | Value | Corpus | Audio / feature | Recipe | Protocol | Source |
|---|---|---|---|---|---|---|
| Root acc | **0.890** ⭐ | Billboard | McGill's own audio + McGill `bothchroma` NNLS | nonlinear MLP + root-rel rotation + learned trigram ctx | oracle bnd, single split, 97.7k/884 | #31 Add-4 |
| Quality bal (5-way) | **0.735** | Billboard | same as above (oracle root) | trigram-context quality head | oracle bnd | #31 Add-4 |
| Dom recall | **0.776** | Billboard | same (oracle root, dom-weight×1.8, top-5 marg) | " | oracle bnd | #31 Add-4 |
| Root acc | **0.379** | Billboard | **our re-sourced YouTube** + real VAMP NNLS | **plain MLP**, no rotation/trigram | oracle bnd, GroupKFold-5, 20 songs | "REAL VAMP" 07-15 |
| Root acc | **0.378** | JAAH | our YouTube + within-corpus **NNLS** | original multihead recipe | 8-seed song-grouped | JAAH NNLS-vs-BP48 |
| Quality bal (5-way) | **0.623** | JAAH | NNLS (same as above) | " | 8-seed | JAAH NNLS-vs-BP48 |
| Root acc | ~**0.337** | JAAH | our YouTube + **BP48** (Basic Pitch) | MLP root head | 6 song-strat splits | `train_jaah_cv.py` |

Reading this table:
- The **0.890 headline is real but domain-locked**: it exists only on McGill's own
  clean audio + their canonical NNLS chroma + the full rotation/trigram recipe. It
  has never reproduced on our re-sourced audio or on production BP48 (that gap
  decomposes to ~30pp feature/audio-domain + a recipe downgrade — see the "Why the
  great NNLS numbers don't translate" entry).
- **0.379 (Billboard) and 0.378 (JAAH) are NOT the same measurement as each other or
  as 0.890** — different audio, feature extractor, AND recipe. Do not read them as
  "stable."
- **Genuinely settled recent finding:** on the trusted, chroma-fit-verified **JAAH**
  corpus, a trained head on **NNLS chroma decisively beats BP48** — root +8pp
  (0.378 vs 0.294), quality-balanced +17pp (0.623 vs 0.448), consistent across
  seeds. This is the airtight within-corpus result (identical audio/boundaries/
  pooling, only the chroma extractor differs). Its cross-corpus interpretation is
  confounded by jazz-vs-pop genre — flagged honestly in the source entry.
- **Bass/inversion thread is closed as negative:** windowed context-MLP,
  confidence-gated rescue, and predicted-inversion-degree all failed to beat the
  plain absolute-bass-12 chroma baseline (~0.654 sounding-bass acc). The oracle
  inversion-degree gain (→0.879 on inversions) does NOT survive predicting the degree
  — it's an oracle artifact.

If you cite any single number to someone, cite the whole row (corpus + feature +
recipe + protocol), not the bare value.

---

## 7. Corpus data format

The training corpora are `.npz` files under `data/cache/<corpus>/` (e.g.
`data/cache/rwc/rwc_bp48_fixed.npz`, 13,204 chords / 100 songs;
`data/cache/jaah/jaah_bp48.npz`, 6,677 chords / ~113 songs). Load with
`corpus_schema.load_corpus(path)` (or plain `np.load(path, allow_pickle=True)`).
Verified keys (✅ inspected `rwc_bp48_fixed.npz`):

| key | shape | dtype | meaning |
|---|---|---|---|
| `feat48` | (N, 48) | float32 | BP48 pitch-class features, **root-relative** (rotated so the true root sits at index 0). Use for the **quality** head. |
| `feat48_abs` | (N, 48) | float32 | same features, **absolute** pitch (not rotated). Use for the **root** head — root can't be predicted from the root-relative frame. |
| `root` | (N,) | int32 | functional root pitch class, 0–11 (C=0). |
| `quality_idx` | (N,) | int32 | index into the 7-way vocab `["maj","min","dom","hdim","dim","aug","sus"]`. |
| `quality` | (N,) | str | the quality string (redundant with `quality_idx`). |
| `qualities` | (7,) | str | the vocab itself, for decoding `quality_idx`. |
| `labels` | (N,) | str | original Harte label, e.g. `"Ab:min"`, `"C:maj/D"`. |
| `match` | (N,) | str | trust tier: `"exact"`/`"family"`/`"mismatch"`/`"none"` — filter via `filter_by_match`. |
| `t0`, `t1` | (N,) | float64 | chord span in **seconds** (oracle GT boundaries). |
| `song_id` | (N,) | str | e.g. `"rwc_RWC_P001"` — **group by this for CV splits** to avoid song leakage. |

The **48 dims** are 4 blocks of 12 pitch classes: `[chroma_onset, chroma_note,
bass, treble]` (`feat48_abs = [ch_on, ch_nt, bass, treble]`). The bass block is the
low-register energy that carries inversion information; treble is the upper register.

**The sounding-bass vs functional-root distinction (redefined 2026-07-16).** `root`
stored in the corpus is the *functional* root — for `C:maj/D` that's `C`. The project
target was redefined to the **sounding bass pitch class** — for `C:maj/D` that's `D`.
Use `corpus_schema.sounding_bass_pc(label, root_pc)` to resolve it: no slash → returns
root; `/<degree>` (e.g. `/3`, `/b7`) → `(root + offset) % 12`; `/<note>` (e.g. `/D`) →
absolute note pc. In RWC all 1,633 inverted labels use scale-degree tokens (verified),
87.6% of chords are root-position (so `bass == root` there). POP909 **discards
`/bass`** at parse time, so it can only report `bass == root` and must be excluded
from bass-target work.

Quick poke in a notebook:

```python
from harmonia.data.corpus_schema import load_corpus, sounding_bass_pc
d = load_corpus("data/cache/rwc/rwc_bp48_fixed.npz")
d["feat48_abs"].shape          # (13204, 48)
list(d["qualities"])           # the 7-way vocab
[sounding_bass_pc(l, int(r)) for l, r in zip(d["labels"][:5], d["root"][:5])]
```

---

## Where to look next

- Current "what's true" and the fabricated-number resolution: the **PHASE-0 AUDIT**
  and the tail entries of `docs/known_issues.md` (authoritative, but 8k lines — read
  the tail, not the whole thing).
- Structural cleanup already scoped: `docs/refactoring_suggestions.md` +
  `docs/refactoring_delegation_plan.md`.
- Forward design ideas: `docs/architecture_extensions.md`, `docs/suggestions.md`.
