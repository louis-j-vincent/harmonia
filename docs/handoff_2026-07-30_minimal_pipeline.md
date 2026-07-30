# Handoff: HARMONIA-MIN — minimal rebuild, chart rendering first

2026-07-30. Prompt for a fresh agent. Context: written right after the
chord-context-prior session (branch `feat/chord-context-prior`, see
`docs/blog/2026-07-30-lock-propagation-context-prior.md`). Louis's framing:
the current pipeline is PRESUMED BROKEN — the minimal rebuild must NOT
reproduce it, and must not inherit its weight.

---

You are building HARMONIA-MIN: a minimal, functional harmonia rebuilt from
the few bricks that are actually good, in the repo
/Users/vincente/Documents/Projets Perso/Code/harmonia (the canonical one;
NEVER work in the stale ~/harmonia clone). Python: `.venv/bin/python`.

MISSION. The current pipeline is PRESUMED BROKEN and heavy — do NOT
reproduce it and do NOT import its orchestration. Copy only the small live
leaf bricks below into a fresh `harmonia_min/` package, write a new thin
orchestration (~300 lines), and get ONE thing working first: chart
rendering — audio in, interactive chord chart in the browser. No test
suites (one smoke test max), no eval harnesses, no experiment scripts.

BRANCH. `git fetch --all`; branch `feat/minimal-pipeline` from
`feat/chord-context-prior` (contains feat/harmonic-key + the new lock
propagation). Check `git branch -a --sort=-committerdate` for other
branches ahead with recent commits; merge active ones or list what you
skipped. Never `git add -A`; stage explicit paths; commit at checkpoints.

THE LIVE CORE — copy these files into harmonia_min/ (≈2,500 lines total),
they are self-contained leaves:
- `harmonia/models/nnls_features.py` (221 L) — VAMP NNLS chroma + trained
  root/quality heads; checkpoint `harmonia/models/nnls24_heads.npz`.
- `harmonia/models/musx_redecode.py` (361 L) — the SOTA chord evidence:
  music-x-lab 5-fold frame posteriors (cache `data/cache/musx_probs/`),
  boundary-latency correction, beat-grid redecode. Note its `_InMusxDir`
  chdir trick into the external music-x-lab clone — keep it working.
- `harmonia/theory/key_profiles.py` (242 L) — `infer_key`.
- `harmonia/models/span_rescore.py` (663 L) +
  `harmonia/models/chord_context_prior.py` (718 L) — lock propagation
  (differential re-score + context prior; npz caches under `data/cache/`).
- beatthis: the wrapper `_get_beatthis()` lives INSIDE chord_pipeline_v1
  (4,779-line monolith) — re-extract it (~50 lines) instead of importing;
  read `harmonia/serving/audio.py:126-220` first (m4a→wav lesson: Beat
  This! must get a wav, the m4a path broke silently once). beatthis ONLY,
  librosa is banned (2x tempo octave bug).
- UI: copy `harmonia/output/app_shell.html` and the minimal slice of
  `harmonia/output/chart_model.py` needed to feed it. The UI is the
  product, not the broken part — don't redesign it, but drop server routes
  it calls that you don't serve yet (degrade gracefully).

DO NOT IMPORT (the weight lives here): `chord_pipeline_v1.py` (monolith),
`chord_head.py` (1,212 L of post-passes), `chord_hmm.py` (dead),
`section_structure.py`, `joint_decode`/`semi_markov`/bp48 anything,
`infer_chords_billboard_v1`, ProgressionEncoder, LLM priors, old
`/api/reinfer` machinery, scripts/.

NEW THIN ORCHESTRATION (your ~300 lines, design freedom): audio → beatthis
beats → musx posteriors → segment + label (thin: musx-derived boundaries +
argmax labels from musx/heads evidence, simple coalesce) → infer_key →
chart payload → serve. You are explicitly ALLOWED to diverge from the old
pipeline's choices — e.g. it quantized everything to a synthetic
constant-tempo lattice instead of the real detected beats (known_issues
"BOTH CLOCKS ARE SYNTHETIC"): you may use real beat times. Simplicity
beats faithfulness to the old code everywhere they conflict.

RULES. Don't trust the old code or docs >1 week old — verify any brick you
copy against raw data before building on it (this session alone caught 3
silent bugs that way: a parser defaulting unknown qualities to maj, a
transposed musx array layout, a metric conflating churn with effect).
Serve on port 7772; NEVER touch the server on :7771. Log as you go in
`docs/minimal_pipeline_log.md`.

MILESTONE 1 DONE = qualitative, no equivalence gate (the old pipeline is
not the reference): the app on :7772 analyzes 2-3 songs with cached
musx_probs (e.g. This Love) end-to-end and renders their charts with
working playhead; you report what you kept/cut/diverged-on, and where your
output DIFFERS from the old pipeline's chart, listed as signal for Louis
to inspect — not as failures. Then STOP; next milestones (lock/annotation
UX, iReal import, sections) get scoped with Louis.
