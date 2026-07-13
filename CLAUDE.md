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
   label format actually encodes and discards. When sources disagree, trust
   order is iReal Pro > guitar tabs > model output — but a strong,
   well-supported music-theoretic hypothesis is allowed some leeway against
   a single disputed label, especially from a lower-trust source.
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

## Default working habits (mined from 571 prompts across 17 sessions, 2026-07-11)

These are things the user has had to ask for repeatedly — treat as defaults,
not as requests to wait for.

- **Log findings immediately**, not just at session end — after any
  nontrivial result (bug root-caused, sweep finished, experiment done),
  write it to `docs/known_issues.md` or `docs/blog/` before moving on.
- **Diagnostic plots are the default first move when something looks wrong**,
  not a verbal explanation — generate the plot before or alongside the
  writeup, not only when asked.
- **Don't claim success on a metric alone** — produce something directly
  inspectable (audio render, side-by-side comparison, HTML chart) before
  asserting a fix or improvement worked.
- **Autonomy is granted/revoked per-context, not fixed.** For unattended or
  long runs, state (or ask for) an explicit time budget and a check-in
  cadence up front (~30 min default); don't over-poll. Include a disk-space
  check in that cadence — a real disk-full incident has happened before.
- **Delegated subagents should research existing project history first**
  (known_issues.md, docs/blog/, recent git log) before starting fresh work,
  same as the user does. The user runs multiple concurrent Claude sessions
  on this repo, so cross-session state conflicts (esp. silently clobbering
  another session's UI/design work) are a real, recurring risk — check for
  unfamiliar recent changes before overwriting.
- **Good task handoffs are: ranked priorities, an explicit try-order, and a
  quantitative stopping/continue criterion** — not an open-ended "try to
  improve X."
- **Explain jargon at the ML×music-theory intersection** (e.g. "DFT
  magnitude of a chroma vector," "ARI over segmentation boundaries") even
  though the user is fluent in each field separately — the friction is at
  the combination, not within either domain.
- **UI/aesthetic state on `harmonia/output/chart_interactive.py` needs the
  same log-before-change discipline as modeling decisions** — it has
  regressed silently before with no way to recover intent except guessing.
  That surface is meant to feel fun/playful, not just correct.
- **Prefer partial-credit chord scoring** (predicting maj7 when GT is maj
  should get credit for the parent family) alongside strict exact-match
  when reporting eval numbers.

## Model Tier Delegation Rules (Cost-Conscious Research)

**Always start with Haiku. Let Haiku decide what needs escalation.**

The agent first receives the task at Haiku tier. If it determines the task
needs deeper reasoning, it explicitly proposes escalation and explains why,
following these decision rules:

### Haiku Tier (Always Start Here)
Use for tasks <15 seconds of human thinking:
- Constant/parameter verification: "What is FRAME_RATE in stage1_pitch.py?"
- File existence checks: "List first 3 POP909 audio files in harmonia/data/"
- Simple validation: "Does 44100/512 equal 86.1328125?"
- Basic log scanning: "Find ERROR lines in experiment log"
- Writing straightforward findings to markdown
- Simple syntax/arithmetic checks
- Any routine information retrieval

**Haiku's job**: Provide the answer. If the task is too complex, explicitly
propose escalation with reasoning (see below).

### Escalation to Sonnet (Proposed by Haiku When Needed)
Use for tasks 1–5 minutes of human thinking:
- File/directory exploration: "Find all chord-related files in harmonia/models/"
- Reading/summarizing docs: "Summarize docs/known_issues.md issue #5"
- Designing simple experiments: "Outline a unit test for chord transition validation"
- Generating diagnostic plot code
- Parameter sweep setup: "List 5 beta values between 0.1–0.5"
- Literature-style research: "What does literature say about Dirichlet priors in HMMs?"

**Sonnet's job**: Explore, design, summarize. Answer the question directly. If
deeper reasoning is needed, propose Opus escalation.

### Escalation to Opus (Proposed by Sonnet When Needed)
Use **only** when ALL of these are true:
1. User has explicitly granted time/budget ("I have 20 minutes for this")
2. **AND** the task requires one of:
   - Multi-step music-theoretic reasoning (e.g., analyzing why ii-V-I resolution fails)
   - Bayesian optimization design with expert judgment
   - Adversarial validation isolating competing hypotheses
   - Theoretical synthesis drawing on deep domain knowledge

**Opus's job**: Provide expert reasoning. Answer the question with full
depth.

### The Escalation Protocol
When Haiku/Sonnet encounter a task requiring deeper reasoning:

1. **Explicitly state**: "This task requires escalation to Sonnet/Opus because..."
2. **Give concrete reason** referencing the rules above
3. **Provide a time budget estimate** ("this should take ~10 minutes")
4. **For Opus escalation**: Wait for user confirmation (user has already granted
   budget in the conversation, or agent asks: "Should I use Opus for this?")
5. **Proceed** with the escalated tier only after confirmation

### Examples of Escalation in Action

**Example 1: Haiku receives file validation task**
```
Task: "Check if FRAME_RATE in stage1_pitch.py matches Basic Pitch documentation"
Haiku's response: [Reads file, outputs FRAME_RATE value]
No escalation needed. Task complete.
```

**Example 2: Haiku receives exploration task**
```
Task: "Analyze why chord HMM overpredicts dominant 7ths in minor keys"
Haiku's response: "This task requires Sonnet-tier reasoning because it involves
music-theoretic analysis of prediction patterns across multiple songs. Escalating
to Sonnet to explore chord theory hypotheses and data patterns."
[Escalates to Sonnet]
```

**Example 3: Sonnet receives deep synthesis task**
```
Task: "Design a Bayesian prior update for chord HMM transition matrix considering
rhythm complexity in jazz standards, with exact parameter changes and validation metrics"
Sonnet's response: "This requires Opus-tier expertise because it needs adversarial
validation of competing hypotheses (timbre vs harmony effects) using only MIDI features.
Estimating 12-minute Opus session. Awaiting confirmation before proceeding."
[Waits for user confirmation or explicit budget statement]
```

### Critical Rules
- **Never skip Haiku**: Always start here, even if you think it's obviously
  complex. Haiku is fast enough to validate the assumption.
- **Escalation is the agent's job**: The agent decides when to escalate, based
  on these rules. Don't force a tier—let the agent propose it with reasoning.
- **No Opus without justification**: If escalating to Opus, the agent must cite
  the specific rule (music-theoretic reasoning, Bayesian design, or adversarial
  validation) and the user must have granted time/budget.
- **Log escalation decisions**: When escalating, note why in the response so
  the user can later audit whether escalation was justified.

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
