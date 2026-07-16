# Refactoring suggestions — Harmonia

_Written 2026-07-15 (Opus survey). Findings + proposals only; no code was moved.
The staged execution plan lives in `docs/refactoring_delegation_plan.md`._

This doc exists because of a recurring failure shape: the same data-pipeline step
gets re-implemented in two scratch scripts, they silently diverge on a string
constant, and a downstream trainer drops most of a dataset while still printing a
plausible metric. That is not a one-off — it is the predictable cost of having no
shared, tested, single-source-of-truth modules for core operations. Everything
below is aimed at that root cause, and is anchored to real counts in the repo as of
this date.

---

## 1. Scale of the sprawl (measured, not estimated)

| Area | Count | LOC | Note |
|---|---|---|---|
| `harmonia/` package | 50 `.py` | ~19,800 | Reasonably organized: `data/ eval/ models/ output/ theory/`. This is the good part. |
| `scripts/` | ~200 `.py` (226 dir entries) | **65,081** | 3.3× the package. This is where the mess lives. |
| `scripts/harmonia_server.py` | 1 file | **7,008** | A single monolith — 321 KB. Bigger than most subsystems. |
| `scratchpad/` | 18 items | — | Mixed `.py`/`.json`/`.log` + a committed 4.7 MB `root_posteriors.npz`. |
| `docs/*.md` (top level) | **85** | — | + `blog/` (21), `results/` (10), `plots/`, `logo/`, `pwa/`. |
| `docs/known_issues.md` | 1 file | **3,395 lines / 235 KB** | Append-only. Discussed in §5. |
| `tests/` | 33 files | — | **380 tests collected** — a real suite exists (see §4). |

The package is healthy. The problem is the 65 kLOC of `scripts/` + `scratchpad/`
that has grown as an undifferentiated pile of experiments, one-off migrations,
plotting utilities, and a few pieces of genuinely load-bearing reusable logic —
with no boundary between those categories.

---

## 2. The core bug class: no single source of truth for shared operations

### 2a. The `match` label is a free-string convention (this is the bug that triggered this review)

There is no enum, no constant, no validator for the corpus `match`/quality field.
Grepping the literal values actually written to corpora:

```
114  "exact"
 93  "family"
 45  "none"
 15  "mismatch"
  2  "billboard_gt"   ← scratchpad/build_billboard_pilot.py:129, build_billboard_60.py:190
```

Downstream trainers filter on hardcoded literals, e.g.:

- `scripts/train_real_audio_final.py:179` → `keep = match == "exact"`
- `scripts/train_yt_exact_matches.py:121` → `exact_mask = match == "exact"`
- `scripts/train_yt_real_audio.py:201` → `mask = (match == "exact") | (match == "family")`

So a corpus tagged `"billboard_gt"` (a value only two scratch builders emit, and no
trainer recognizes) is silently filtered to **zero rows** by any `match == "exact"`
gate — the reported bug, exactly. The fix is a single module (e.g.
`harmonia/data/corpus_schema.py`) that defines the allowed match values as an enum
and a `filter_by_match(records, min_quality)` helper both builders and trainers
import. No literal `"exact"` string in a training script survives the refactor.

### 2b. Feature extraction is scattered, not centralized

- **71** scripts import `PitchExtractor`/`stage1_pitch` (good — the shared path exists).
- **23** scripts compute librosa chroma inline (`chroma_cqt`/`cqt(`/…).
- **32** scripts reference `basic_pitch`/`predict(` directly.

This is precisely the surface where CLAUDE.md's error pattern #1 (silent
calibration bugs — frame rate off by 2×, etc.) recurs. Every inline chroma/BP call
is a place a frame-rate or normalization constant can drift out of sync with the
canonical extractor without any test noticing. Target: one feature-extraction entry
point that all corpus builders and all trainers route through; inline `librosa`/`bp`
calls in scripts become a lint smell to hunt down.

### 2c. Corpus packing is implicit and per-script

**34** distinct `np.savez` call sites across `scripts/` + `scratchpad/`. Each
defines its own array names and dtypes ad hoc; the "schema" of a training corpus is
whatever a given builder happened to write. This is the structural reason 2a is even
possible. Target: a single `save_corpus(...)`/`load_corpus(...)` pair with a fixed,
documented key set (features, labels, match, song_id, boundaries…) so a schema
mismatch is a load-time error, not a silent all-rows-dropped.

### 2d. Chord-label translation exists in the package but is re-derived in scripts

`harmonia/data/billboard_translator.py` and `harmonia/data/pop909_parser.py` hold
the canonical translators, yet ~17 translation-flavored `def`s live scattered
(e.g. `scripts/llm_chord_priors.py`, `scripts/analyze_corrections.py`,
`scripts/analyze_accomp_emission.py`). Chord-format translation is exactly the kind
of thing CLAUDE.md rule #3 warns about (POP909 discards `/bass`; "root" is
functional not sounding) — it must have one implementation, tested against known
edge cases, not N.

---

## 3. Dead / archivable / broken

- **Broken entry point:** `pyproject.toml` declares `harmonia = "harmonia.cli:main"`
  but **`harmonia/cli.py` does not exist**. `pip install -e .` gives a `harmonia`
  command that crashes. Either add the CLI or drop the `[project.scripts]` entry.
- **~30 `train_*.py` variants** in `scripts/`, many explicitly versioned/superseded:
  `train_beat_seq_model{,_v2,_v3,_v4}.py`, `train_billboard_{batched,chord_model,from_features}.py`,
  `train_yt_{chord_model,exact_matches,real_audio}.py`, `train_real_audio_final.py`,
  plus `train_ctx_model_v2`, `train_online{,_ctx_model}`, etc. Most are one-shot
  experiments. Only a handful are the current best path; the rest should move to an
  `scripts/archive/` (or be deleted) once the current one is identified in writing.
- **Orphan module:** `harmonia/models/block_fold.py` has **0** importers across
  `scripts/ tests/ harmonia/`. Candidate for removal (verify against git log first).
- **Root-level clutter that should not be at repo root:**
  `Taking over app UIUX.zip` (206 KB), `handoff 2/` (a whole duplicate-looking
  directory with its own `HANDOFF.md` + 6 `*.dc.html`), `alignment_comparison.html`,
  `harmonia.html`, `.coverage`, `.DS_Store` (multiple). None are source; several
  look like editor/download detritus.
- **`scratchpad/root_posteriors.npz` (4.7 MB) appears committed** — large binaries
  in a scratchpad should be gitignored, not versioned.

---

## 4. Test suite: exists, but thin where it matters most

Good news, correcting a session-time assumption: there **is** a live suite —
`tests/` has 33 files, **380 tests collected**, wired via `pyproject.toml`
(`testpaths=["tests"]`, `--cov=harmonia`). CLAUDE.md's "red-first" convention is
backed by real infra.

The gap is **coverage of the load-bearing numeric paths**: total measured coverage
is ~15% of `harmonia/`, and the tests concentrate on models/theory. The functions
whose silent breakage has actually cost this project time are under-covered.

**Unit tests to add first, ranked by "most likely to silently produce wrong
numbers" (matched to CLAUDE.md's six error patterns):**

1. **Feature-extraction calibration** (pattern #1): assert `stage1_pitch`
   frame-rate and any chroma normalization against the upstream library constant
   and a real-file duration. This is the #1 historical foot-gun.
2. **Corpus schema / `match`-value contract** (the trigger bug): a round-trip test
   that a corpus written by the builder is read back with the expected keys, and
   that an unknown `match` value raises instead of silently filtering to empty.
3. **`harmonia/eval/mirex_eval.py`** (pattern #6): metric functions must be pinned
   — a metric that silently shifts is the worst failure mode in a research repo.
   Include a partial-credit-vs-strict pair (CLAUDE.md default).
4. **Chord-label translation** (pattern #3): `/bass` handling, colon-quality labels,
   maj↔maj7 family mapping — the edge cases already documented as having bitten.
5. **Beat/tempo** (pattern #1): a regression pin on song 002's known 2× tempo
   octave so a future beat-tracker swap can't silently reintroduce it.

---

## 5. The "what's true right now?" discoverability problem

`docs/known_issues.md` is 3,395 lines, append-only, and is simultaneously the
project's best asset (a complete reasoning trail) and a liability as a status
source. Concrete contradiction, same file: line 46 treats the accomp-DB regen as
"blocked"; the McGill Billboard audio is asserted "dead end / blocked" in one pass
and "not blocked — 10/10 pilot hit rate" ~20 lines later, with neither original
struck through. A newcomer (or a fresh Claude session) grepping "Billboard" gets a
stale answer with no signal of currency. Billboard alone is spread across **6+**
docs (`MISSION_BILLBOARD_COMPLETION.md`, `billboard_integration_summary.md`,
`billboard_retraining_findings.md`, `billboard_training_plan_SUMMARY.md`,
`billboard_training_results_v2.md` — 4 lines long —, `billboard_translation_validation.md`).

**Proposed fix — split by function, keep the history:**

- **`docs/STATE.md`** (new, short, hand-curated, the single "what is true now"):
  current best model + where its weights are, current best dataset + its path,
  a compact open-issues **table** (id / one-line / status / next action) — and
  nothing else. This is the file a newcomer or new session reads first. Cap it at
  a page or two; if it grows, that's a signal to resolve issues, not to append.
- **`docs/known_issues_log.md`** (rename of today's file): keep it append-only and
  chronological — it is genuinely valuable as the reasoning trail. But it stops
  being cited as "current status"; STATE.md is.
- **`docs/archive/`**: move superseded session/mission dumps here
  (`MISSION_*`, `PHASE_2_*`, per-experiment result files, the 4-line
  `billboard_training_results_v2.md`). They stay readable, out of the top-level
  namespace. Leave forward-looking docs (`architecture_extensions.md`,
  `suggestions.md`, `blog/`) in place.

Do NOT try to auto-merge the 6 Billboard docs into one — that risks losing nuance.
Just point STATE.md at the current one and archive the rest.

---

## 6. Target directory shape (modest, grounded, not a rewrite)

The package (`harmonia/`) is already the right shape — leave its structure alone.
The work is almost entirely about `scripts/` + `scratchpad/` and pulling a few
reusable pieces up into the package. Keep it **minimal and readable** — favor a few
clear files over layered abstraction.

```
harmonia/data/
    corpus_schema.py     NEW — match-value enum + save_corpus/load_corpus + filter_by_match
                         (kills §2a and §2c; every builder/trainer imports this)
    features.py          NEW — the ONE feature-extraction entry point corpora/trainers call
                         (thin wrapper over stage1_pitch; kills the 23+32 inline call sites)
    (billboard_translator.py, pop909_parser.py already here — make them the ONLY translators)

scripts/                 keep as the flat command surface, but with two conventions:
    archive/             NEW — superseded experiments + all *_v2/_v3/_v4 and one-shot migrations
    (the ~15 genuinely-current entry points stay at top level, named clearly)

scratchpad/              stays a true scratchpad; add *.npz/*.log to .gitignore

tests/                   add the 5 §4 tests; the schema + features + mirex tests are the priority
```

`scripts/harmonia_server.py` (7 kLOC) is its own project and should be treated
separately from this modeling refactor — flag it, don't fold it in.

The one-line litmus test for whether the refactor worked: **there is no literal
`"exact"` string, no inline `librosa.feature.chroma`, and no bare `np.savez` of a
training corpus left in any `scripts/train_*.py`** — all three now go through an
imported, tested helper. If that's true, the class of bug that started this can't
recur silently.
