# Harmonia — staged rewrite execution plan (2026-07-21)

_Companion to `docs/architecture_proposal_2026_07_21.md` (the "what/why") and
`docs/production_model_audit_2026_07_21.md` (the "where we're weak"). This doc
is the "how/in-what-order". Nothing here is executed yet — this is the plan to
approve before delegating stage 0._

Decisions already locked by Louis (2026-07-21):
- **Scope: everything** — inference pipeline, serving, `scripts/` (241 files),
  `scratchpad/` (199 files), docs. Not just the core pipeline.
- **Parity benchmark: RWC-Popular + `aligned_corpus` (148 songs)** — see §1 for
  the critical caveat on which GT is usable for which gate.
- **A parity mismatch is a lead, not a verdict** — three outcomes (§2). Louis
  explicitly expects real bugs around alignment / first-beat detection.
- **musx (`third_party/ISMIR2019…`) stays vendored as-is.**

---

## 0. The two-net principle (the core methodological idea)

There are two *different* questions, and conflating them is how this project has
burned time before. Every stage is gated on one or both:

- **PARITY net — "did the refactor change behavior?"** New code output vs.
  **the current code's** output, on frozen benchmark audio, label-for-label /
  vector-for-vector. This is independent of whether the output is *correct*. It
  catches port bugs. For a faithfully-ported stage, the target is **exact match
  (diff = 0)**; any diff is either a port bug (fix new) or a discovered latent
  bug (investigate — see §2).
- **ACCURACY net — "is the output actually right?"** New code output vs. **real
  ground truth**. This is where known-weak stages get *improved*, not just
  preserved, and where Louis's suspected alignment bugs get confirmed/fixed.

**Why two nets and not one:** a stage can pass parity (faithful port) while being
wrong (inherited a real bug), or fail parity (intentional redesign) while being
better. You need to know which of the two you're looking at, per stage, in
advance. So every stage below is tagged **PORT** (parity is the goal, exact match
expected) or **REDESIGN** (accuracy is the goal, beating old behavior expected).

---

## 1. ⚠️ Benchmark GT provenance — do NOT evaluate alignment against alignment-derived GT

The single most important measurement subtlety, and the reason the audit's
`root=0.525` number is not a clean chord-recognition figure:

`aligned_corpus`'s per-chord **timestamps are produced by the alignment process
itself** (iReal↔YouTube section alignment). So:

| GT source | Clean for CHORD-LABEL eval? | Clean for BEAT/ALIGNMENT eval? |
|---|---|---|
| **RWC-Popular** (AIST annotations) | ✅ chord labels, studio audio | ✅ **independent beat/downbeat GT** — use this for alignment |
| **POP909** (`beat_midi.txt` col 3) | ✅ chord labels (functional root only — see CLAUDE.md #3) | ✅ **real downbeat GT** — use this for alignment |
| **`aligned_corpus`** (148 songs) | ⚠️ labels OK, but **at self-derived timestamps** | ❌ **circular — its timing IS the alignment output** |

**Rule for the whole rewrite:** alignment / first-beat / beat-grid accuracy is
measured **only** against RWC-AIST beats and POP909 downbeats (independent GT).
`aligned_corpus` is for chord-label accuracy *conditioned on* alignment, and its
chord numbers must always be reported alongside "at aligned timestamps" so nobody
mistakes an alignment error for a chord error again.

**Corollary that reorders the whole plan:** because the audit's chord numbers are
entangled with alignment, we **fix alignment first, then re-measure chords**.
Optimizing chord heads against alignment-contaminated scores would be optimizing
against a ghost. This is why beat/alignment (Phase 2) comes before chords (Phase
3), even though chords are "more central."

**Disk blocker (must clear before Phase 0):** disk is at ~200 MiB / 99% full
(external cause, not this pipeline). The golden-output capture in Phase 0 and any
RWC streaming need headroom. Phase 0 does not start until this is resolved or
quantified as safe. RWC audio streams (not stored); golden outputs are small
(npz/json), but the current floor is too low to trust any write.

---

## 2. The parity-mismatch decision tree (applies at every gate)

When new ≠ old on the parity net:

- **(a) Match** → port confirmed. Proceed.
- **(b) New is right / old was a latent bug** → keep new, **write the bug + fix
  to `known_issues.md`**, update the golden output to the corrected value. This
  is the outcome Louis is hoping for on alignment/first-beat.
- **(c) Old is right / new introduced a regression** → fix new code before
  proceeding. Do **not** advance the phase with a red parity net.

Never silently default to (a) or (c) because outputs "look close." A diff on a
REDESIGN stage is expected and routed to the ACCURACY net (vs GT) to decide (b)
vs (c). A diff on a PORT stage is a bug by definition — investigate which side.

---

## 3. Phases (each phase = one green gate before the next; commit at each gate)

Cross-cutting rules for every phase (inherited from `refactoring_delegation_plan.md`):
research history first; **never move + edit in the same diff** (a move is one
reviewable change, a behavior change is another); small independently-verifiable
diffs; don't touch actively-iterated UI surfaces without an explicit go; delegated
stage-by-stage to subagents with the honesty/multi-seed bar.

### Phase 0 — Foundation & safety net (blocks everything)
**Tag:** neither PORT nor REDESIGN — this builds the nets themselves.
- Resolve/quantify the disk floor (§1).
- **Freeze the parity benchmark**: pick the fixed audio set (RWC-Popular subset +
  a fixed `aligned_corpus` subset + POP909 downbeat-GT subset), write a
  provenance table (which GT is usable for which net, per §1), snapshot the
  current pipeline as the parity oracle (git tag, à la `prod-workable-2026-07-19`).
- **Capture per-stage golden outputs** to disk for every benchmark song:
  features → beats/downbeats → segments → chords → structure → final chart. These
  frozen intermediates let each later stage be ported/diffed independently while
  the old code is progressively deleted.
- **Build the golden-diff harness** (`harmonia/eval/parity.py`): capture, diff,
  and report per-stage divergences with tolerances (exact for labels, epsilon for
  float vectors).
- **Build the calibration unit-test layer** (the thing that would have caught the
  A/C-frame chroma bug and the frame-rate-2× bug): frame rate vs upstream library
  constant, chroma normalization convention, sample rate vs real file duration,
  song-002 2×-tempo octave regression pin. These are red-first pins, not aspirational.
- Lay down the empty new package skeleton (`core/`, `stages/`, `serving/` modules
  with typed input/output contracts) **without moving any logic yet**.
- **Gate:** harness captures current-pipeline output on the frozen set; two
  independent runs diff to zero (self-consistency); calibration pins all green.

### Phase 1 — `core/audio` + `core/features` + `data/corpus_schema` — **PORT**
- One audio loader: sample-rate / resample / duration — the single place those
  constants may live.
- One feature-extraction entry point: NNLS-24, BP48, musx-frontend as pluggable
  `FeatureExtractor` implementations. Kill the 23 inline-librosa-chroma + 32
  direct-basic-pitch call sites (they become a lint smell).
- `corpus_schema.py`: `save_corpus`/`load_corpus` + a `match`-value **enum**
  (kills the free-string `"exact"/"family"/"billboard_gt"` bug that dropped
  corpora to zero rows silently) + `filter_by_match`.
- **Gate (parity + calibration):** feature vectors byte-identical to the current
  extractor on the benchmark; every corpus builder/trainer rerouted through the
  new entry point reproduces its corpus identically; calibration pins still green.

### Phase 2 — `stages/beat_grid` + `stages/alignment` — **REDESIGN** (early, per §1)
The high-value phase. Suspected bug nest; contaminates every downstream metric.
- Port current `bestfit` beat period + Beat This! downbeat anchor as the starting
  point, but treat as a redesign target, not a faithful port.
- **Gate on INDEPENDENT GT only** (RWC-AIST beats, POP909 downbeats — never
  `aligned_corpus`): beat accuracy, downbeat/first-beat accuracy, octave-lock rate.
- **Explicitly investigate** the audit's 6/14 octave-lock disagreements and the
  jazz first-beat failures (rubato intros, AABA with weak downbeat). Each
  confirmed bug → outcome (b): fix + log.
- **Then re-run the chord audit with clean alignment** and report the
  decomposition: how much of `root=0.525` was alignment contamination vs. genuine
  chord error. This number retroactively re-scopes Phase 3.
- **Gate:** beat/downbeat accuracy ≥ current on independent GT; octave-lock cases
  characterized and reduced; the alignment-contamination decomposition written up.

### Phase 3 — `stages/chords` (root/quality/bass heads) — **PORT + isolated improvement**
- Decompose the ~30-flag `infer_chords_v1` into a `ChordHead` interface
  (`nnls24`, `musx` as two implementations) + a small config object. The 30
  booleans become named config fields, not positional god-function kwargs.
- **Gate (parity):** with default config, produces the **exact same labels** as
  today's live call on the benchmark. Port first, prove faithful.
- **Then, separately** (own diff, own accuracy gate vs GT): address the quality
  **class-imbalance** the confusion matrix exposed (rare classes `dim`/`sus`/`hdim`
  essentially never predicted). Do NOT bundle this with the port. Do NOT re-add
  trigram/neighbor context (rejected twice, −7.9pp on real audio). ChordFormer's
  reweighting scheme (lit scan) is a reference to skim, not adopt blind.
- **Gate:** parity green with defaults; any accuracy change measured vs GT, on the
  *clean-alignment* re-measurement from Phase 2, multi-seed.

### Phase 4 — `stages/key` + `stages/structure` — **PORT the working path only**
- Port the working baseline: flat block8 (V_F 0.68-0.70) + the shipped
  `chordtone` section-clustering (53/58 corpus-exact). Port local/global key.
- **Do NOT resurrect** the symbolic learned-similarity thread — its "+0.010 win"
  does not reproduce at fresh seeds (9/9 negative), and the adaptive
  agglomerative extension was decisively re-falsified. It's a dead end for
  beating block8.
- Buisson et al. (audio-native multi-level self-supervised segmentation) is the
  one genuinely-different lead — **flagged as a future dedicated research phase,
  explicitly OUT of scope for this rewrite** (weeks, not part of the port).
- **Gate (parity):** structure/key labels match current on the benchmark.

### Phase 5 — `pipeline.py` orchestration — **PORT** (kills the god-function)
- One `PipelineConfig` dataclass replaces the 30 kwargs. Serializable → a run is
  reproducible from a saved config, not from remembering which flags were on.
- **Gate (full-pipeline parity):** `pipeline.run(config=current_defaults)` ==
  today's `infer_chords_v1(<30 flags>)`, label-for-label, across the whole
  benchmark. This is the capstone parity gate: if green, the new pipeline is a
  behavior-identical replacement of the old core, and the old core can be deleted.

### Phase 6 — `serving/` — **PORT** (break the 10,332-line monolith)
- Split `harmonia_server.py` into `serving/api.py` (thin Flask routes) +
  `serving/render.py` (chart HTML) + `serving/cache.py` (one cache-key
  convention — fixes the stem-collision bug class the audit hit).
- **Gate (route-by-route HTTP parity):** each route returns the same response as
  the monolith on a fixed request set; cache keys unified and collision-tested.
- **Care:** UI surfaces (`app_shell.html`, `chart_interactive.py`) are
  actively-iterated / separately-owned — behavior-preserving decomposition only,
  no redesign, explicit go before touching them.

### Phase 7 — `scripts/` + `scratchpad/` triage — **PORT / archive** (scope=everything)
- Reroute all trainers/builders through `corpus_schema` + `core/features` (no
  inline feature or corpus code survives — enforce with a lint check).
- Archive superseded `train_*_v2/v3/v4` and one-shot migrations to
  `scripts/archive/` — **verify each file individually** against importers + git
  log before moving/deleting (bulk-deletion discipline: never glob-delete off a
  sample).
- `scratchpad/` (199 files) is research history, not code to rewrite line-by-line:
  **triage** — promote the genuinely-reusable pieces into the package, archive the
  rest. Gitignore/remove the committed large binaries (`root_posteriors.npz` 4.7 MB).
- **Gate:** full test suite green; a representative trainer reproduces its
  corpus/model bit-identically through the new shared modules; lint finds zero
  inline feature/corpus sites.

### Phase 8 — docs consolidation (from the July 15 plan)
- `docs/STATE.md` (new, curated, ≤2 pages: current best model + weights path,
  current best dataset + path, open-issues table). The single "what's true now".
- Rename today's `known_issues.md` → `known_issues_log.md` (stays append-only, the
  reasoning trail).
- `docs/archive/` for superseded mission/session dumps.
- **Gate:** STATE.md exists and is accurate; no doc contradicts it.

---

## 4. Dependency order (why this order, one line each)
0 Foundation — nothing can be gated without the nets.
1 Features — every corpus/trainer/stage depends on features being canonical first.
2 Alignment — **before chords**, because it contaminates the chord measurement (§1).
3 Chords — re-measured against clean alignment from 2.
4 Structure/key — consumes chords; port the working path, don't chase dead ends.
5 Pipeline — orchestrates 1-4; the capstone full-parity gate.
6 Serving — wraps the pipeline; behavior-preserving only.
7 Scripts/scratchpad — reroute through 1's canonical modules; archive the rest.
8 Docs — last, reflects the finished state.

## 5. What I need from Louis before delegating Phase 0
1. **Approve the order** — in particular alignment-before-chords (§1). This is the
   one non-obvious reordering; it follows from the contamination argument but it
   means chord improvements wait for Phase 2.
2. **Disk** — the ~200 MiB floor must be cleared (external cause). Do you want me
   to spawn a careful cleanup pass first (individual-file verified, per discipline),
   or is that something you'll handle?
3. **The jazz-vs-pop question** (audit §1) — your ear on whether those low jazz
   scores are mis-alignment vs mis-recognition would pre-load Phase 2's
   investigation, but it's not a blocker; Phase 2 will settle it empirically either
   way.
4. **Pace / autonomy** — same 12h-budget delegated-subagent model as tonight, one
   phase per agent with the parity gate, or do you want to review each phase's gate
   output yourself before the next starts?
