# Harmonia — working conventions for agents

Jazz/pop chord-recognition pipeline (Basic Pitch → beat quantisation → SSM
segmentation → key inference → chord HMM), evaluated on POP909 with MIREX
weighted-overlap metrics. Personal research project; the human is an ML PhD
and jazz musician — be concise, be rigorous, show numbers.

## Where things live

- `docs/known_issues.md` — **authoritative** open-issues tracker ("what's wrong
  right now"). Update it whenever an issue is found, characterized, or resolved.
- `docs/architecture_extensions.md` — forward-looking design ideas.
- `docs/suggestions.md` — specific stage-1/stage-5 improvement proposals.
- `docs/blog/` — narrative devlog; keep it updated after significant sessions.
- `data/` and `.venv` are symlinks into `~/harmonia/`; there is a stale clone
  at `~/harmonia/` — never work there, this repo is the canonical one.
- Renders: use `*_v005_musescoregeneral.wav` (current default soundfont), not
  the old `prog0` Vintage-Dreams renders.

## Hard-won process rules (each one paid for with real wasted effort)

Every major error in this project so far fits one of six patterns. Apply the
counter-rule *before* starting work, not after something looks off.

1. **Silent calibration bugs beat clever experiments.** Four separate times, a
   low-level unit/scale/constant error (frame rate off by 2x, key posterior
   pinned near-uniform, mislabeled soundfont file, beat tracker locking a 2x
   tempo octave) silently corrupted everything downstream while producing
   plausible numbers. → When touching a pipeline stage, first unit-test its
   most basic load-bearing assumption against an external reference (upstream
   library constant, real file duration, GT tempo annotation).
2. **Screen the premise cheaply before implementing.** Issue #1's candidates
   A and B were fully implemented before their premise was checked; both
   failed. Candidate C ran a cheap premise-check script first — that is the
   template. Before any multi-day investigation, write the cheapest check
   that could falsify the idea and run it first.
3. **Ground truth is a measurement too.** POP909 chord labels discard `/bass`
   inversions (10–18% of lines); its "root" is functional, not sounding.
   Before attributing model↔GT disagreement to the model, check what the
   label format actually encodes and discards.
4. **State what a fix does NOT solve.** `score_periods()` detected period
   length but never phase; downstream code silently assumed phase 0. When a
   function solves part of a problem, document the unsolved remainder in its
   docstring and in known_issues.md, even if nothing consumes it yet.
5. **Single-song findings are hypotheses.** Song 001's beat-phase correlation
   collapsed corpus-wide. Corpus-level symbolic checks are cheap
   (`parse_all(require_audio=False)`) — run them before building on a
   one-song result.
6. **Component swaps change more than the target metric.** A soundfont swap
   silently doubled one song's detected tempo. After swapping any component,
   diff all intermediate outputs (beat counts, event counts, shapes), not
   just the metric you meant to move.

## Collaboration conventions

- **Don't commit while the user is still iterating** on interpretation/design
  in chat. Commit-at-checkpoint is fine for well-defined "go implement X"
  tasks; when the exchange is conceptual pushback / "what does X mean?",
  hold commits until the user signals the round is settled. Writing files is
  fine; `git commit` waits for a signal.
- **Time-boxed narrow sprints work well** (best result of the project came
  from a 1-hour oracle-boundary sprint). Prefer a sharp question + small
  script over expanding scope.
- **Handoffs to fresh sessions** should carry the full reasoning trail, not
  just a task list (see `docs/handoff_2026-07-02_key_inference.md` as the
  model).
- Tests are red-first when fixing bugs: write the failing test against the
  old behavior, then fix.

## Environment gotchas

- Python 3.12; numpy `<2.5` (numba); basic-pitch via ONNX (TF backend broken).
- `PitchExtractor` caches to `data/cache/*.npz` — the cache key does NOT
  cover module-level constants; clear the cache after changing any.
- `POP909Song.ChordEvent.start_beat`/`end_beat` are **seconds**, not beat
  indices, despite the names — use `song.chord_at_time(t)`, not a
  now-removed `chord_at_beat`.
- Song 002's librosa-detected tempo is 2x wrong (63 vs 129 BPM GT); anything
  measuring "beats" for that song via our audio beat tracker inherits it.
- `POP909Song.is_downbeat`/`.downbeat_times` are real ground truth (from
  `beat_midi.txt` column 3) — prefer these over audio-only downbeat
  detection for any POP909 experiment.
- `ChordInferrer(emission_scoring=...)`: `"dot"` (default) vs `"cosine"`.
  Cosine is the theoretically correct fix for a confirmed template-geometry
  bug (docs/known_issues.md #5) but is a net negative end-to-end — don't
  flip the default without re-running
  `scripts/experiment_issue1.py --sweep-emission-scoring` first.
