# Architecture proposal — clean rewrite of the inference path (draft for discussion)

_Written 2026-07-21, in response to Louis's request for a ground-up recode
proposal ("tu me recotes tout de zéro... propose-moi une nouvelle
architecture"). Nothing executed — this is the discussion doc, per his own
request to talk it through first._

## 1. The sprawl, measured right now (not the July 15 numbers — it's grown)

| Area | July 15 survey | Now (2026-07-21) |
|---|---|---|
| `harmonia/` package | 50 `.py`, ~19.8k LOC | 130 `.py` (incl. ~80 vendored `third_party/musx`) |
| `scripts/` | ~200 `.py`, 65k LOC | 241 `.py` |
| `scratchpad/` | 18 items | 199 `.py` files |
| `scripts/harmonia_server.py` | 7,008 lines | **10,332 lines** |
| `docs/known_issues.md` | 3,395 lines | **17,824 lines** |
| `harmonia/models/chord_pipeline_v1.py` | — | 4,965 lines, `infer_chords_v1()` takes **~30 boolean/knob kwargs** |

The core inference entry point is a single function with ~30 accumulated
feature flags (`use_beat_seq`, `use_ctx_model`, `use_harmonic_grid`,
`use_bass_tracking`, `use_diatonic_prior`, `use_progression_prior`,
`use_local_key_prior`, `use_phase_correction`, `use_joint_decode`,
`use_semi_markov`, ...) — one per experiment ever run, most now permanently
on or off, none removed. `harmonia_server.py` is a 10k-line monolith mixing
routing, HTML generation, offline analysis, and cache management. This is
the most concrete evidence for "propose-moi une nouvelle architecture."

## 2. There is already a prior proposal — and it argues against a rewrite

`docs/refactoring_suggestions.md` / `docs/refactoring_delegation_plan.md`
(2026-07-15, Opus survey) looked at this exact sprawl and concluded: **the
`harmonia/` package itself is "reasonably organized... the good part."**
The actual mess is `scripts/` + `scratchpad/` (one-off experiments, ~34
different ad-hoc `np.savez` schemas, no shared corpus contract) and
`docs/*.md` (106 files, no single "what's true now" source). Its proposed
fix was deliberately modest: add 2-3 new shared modules
(`corpus_schema.py`, `features.py`), archive dead scripts, split
`known_issues.md` into a curated `STATE.md` + the append-only log — NOT
touch the package's actual structure.

**This directly conflicts with "recode tout de zéro."** I want to flag that
tension rather than silently pick a side.

## 3. Why a full ground-up rewrite is the highest-risk option on the table

This project's own CLAUDE.md rules exist because this exact move has burned
real time before:

- **Rule #1** (silent calibration bugs): four separate times, a low-level
  unit/scale/constant error silently corrupted everything downstream while
  producing plausible numbers. A rewrite is the single easiest way to
  reintroduce every one of these, because the whole premise of "recode from
  zero" is *not* reusing the code that currently encodes the fix.
- **Rule #6** (component swaps change more than the target metric): a
  soundfont swap once silently doubled a song's detected tempo. Rewriting
  the entire inference path is the maximal version of a component swap.
- The current pipeline encodes **~15 non-obvious, hard-won fixes** that
  live only as logic, not as tests that would catch a silent regression:
  sounding-bass target redefinition, A/C-frame chroma indexing (found
  *this week*, cost real corpus rows), fifth-corrected hdim7/dim tie-break,
  bar1-offset-as-bar-drop (not phase-shift), near-boundary onset
  rebalancing, "add9" family fix, unconfirmed-dominant local-key default,
  1st/2nd-ending fold handling, and others in `known_issues.md`. None of
  these are exotic — they're the kind of thing that's invisible until a
  real song hits the exact case, and a rewrite has no automatic way to
  know it needs to preserve them unless something forces the comparison.

**My recommendation: not "don't rewrite" — but "rewrite with a
parity gate," not "rewrite from memory."** Concretely: port the validated
logic stage-by-stage into the new structure (copy the tested function,
then refactor its shape) rather than re-deriving it from the music theory
in your head, and gate every stage's replacement on a **golden-output diff
against the current live pipeline on a frozen benchmark set** before the
old path is deleted. This gets you the clean architecture you're asking
for without the "silently different numbers" risk the project has paid for
four times already. It's slower than a from-scratch rewrite but it's the
only version of "from zero" that doesn't reset progress.

**2026-07-21 addendum, per Louis's explicit correction**: a parity mismatch
is NOT automatically a regression to revert. When old and new disagree,
that disagreement gets investigated against real ground truth (RWC-Popular
+ the aligned real-audio corpus) before deciding which one is right —
Louis's own expectation is that this will *surface real bugs worth fixing*,
not just prove the rewrite faithful. He specifically flagged alignment and
first-beat/downbeat detection as areas he suspects are currently wrong and
wants mismatches there treated as leads, not noise. So each stage's gate
has three outcomes, not two: (a) match → port confirmed correct, (b)
new is right / old was a known bug → keep new, log the fix, (c) old is
right / new introduced a regression → fix the new code before proceeding.
Never silently pick (a) or (c) by default just because old and new differ.

## 4. Proposed target architecture

Config-driven pipeline, one stage = one module with a typed input/output
contract, no boolean-flag god-function:

```
harmonia/
  core/
    audio.py            # load/resample/duration — the one place a sample-rate
                         # or frame-rate constant is allowed to live
    features.py          # ONE feature-extraction entry point (NNLS-24, BP48,
                         # musx-frontend as pluggable FeatureExtractor impls,
                         # not inline calls scattered across 71+ scripts)
  stages/
    beat_grid.py          # beat/downbeat/phase estimation (already extracted
                         # 2026-07-18, reuse as-is)
    segmentation.py       # section boundary detection (SSM / nnls / musx —
                         # pluggable, same interface as today's segment_source)
    chords.py             # root/quality/bass heads — EACH is a swappable
                         # ChordHead implementing predict(features) -> labels,
                         # so nnls24-heads vs musx are two implementations of
                         # one interface, not two branches of one function
    key.py                 # local + global key inference
    structure.py           # A/B/C repeat labeling, folding, 1st/2nd endings
    alignment.py           # bar1-offset / user-correction application
  pipeline.py              # orchestrates stages via one PipelineConfig
                         # dataclass (replaces the 30-kwarg function) —
                         # config is serializable, so a run is reproducible
                         # from a saved config, not from remembering which
                         # 30 flags were on
  theory/                 # unchanged — already the right shape
  data/                   # corpus_schema.py, features.py per the July 15 plan
  eval/                   # mirex_eval.py + a golden-output diff harness
                         # (NEW — the actual rewrite-safety mechanism)
  serving/
    api.py                 # thin Flask routes, delegate to pipeline.py —
                         # NOT 10k lines in one file
    render.py             # chart HTML generation, separate from routing
    cache.py               # one cache-key convention (stem-based, per the
                         # bug this project's own agent just found and fixed)
```

Scripts stay a flat command surface (per the July 15 plan, still correct):
`scripts/train_*.py`, `scripts/eval_*.py`, `scripts/build_*_corpus.py` all
import from `harmonia/core` + `harmonia/stages`, never reimplement feature
extraction or corpus schemas inline. `scripts/archive/` for the ~30
superseded `train_*` variants.

## 5. Decisions (Louis, 2026-07-21)

1. **Scope: everything.** Not just inference + serving — the training/eval
   tooling in `scripts/` (241 files) and the research trail in
   `scratchpad/` (199 files) are in scope too. This changes the July 15
   plan's stance for those directories from "archive, don't rewrite" to
   "rewrite them too, same parity-gate discipline as the core pipeline."
2. **musx (third_party/ISMIR2019...)**: stays vendored as-is — external
   model, not reimplemented.
3. **Parity-gate benchmark: RWC-Popular + the aligned real-audio corpus**
   (148 songs, `data/cache/aligned_corpus/aligned_corpus.npz`) — not the
   still-unbuilt Brick-0 benchmark. RWC-Popular = the RWC Music Database
   popular subset, bundled audio (no YouTube fetch needed), chord labels
   via https://github.com/tmc323/Chord-Annotations, streamed from the
   Zenodo `RWC-P.zip` via `remotezip` — see `docs/known_issues.md`
   "Alternative chord-annotated audio datasets" (2026-07-16) for the full
   provenance.
4. **A parity mismatch is a lead, not a verdict.** When old and new
   disagree, investigate against real GT before deciding which is right —
   see the addendum in §3 above. Louis specifically expects real bugs to
   surface around alignment and first-beat/downbeat detection; treat
   mismatches there as the most likely place to find something real, not
   as noise to suppress.
5. **Pace: wait for the in-progress 12h production audit to land first.**
   Its scope map (which stages are solid vs. weak, right now, on real
   audio) decides which stages get ported carefully vs. redesigned outright
   rather than faithfully preserved.

Next step once the audit lands: turn this into a staged execution plan
(which stage first, what the golden-set looks like per stage, who/what
runs each gate) — not started yet, per point 5.
