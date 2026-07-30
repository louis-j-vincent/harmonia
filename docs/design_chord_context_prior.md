# Design: chord context prior for lock propagation (branch `feat/chord-context-prior`)

2026-07-30. Owner: autonomous session (Fable). Status: design settled, Phase 1 in
progress.

## Problem

In annotation mode, locking a chord is supposed to re-infer its neighbours.
It does nothing: `/api/reinfer` with confirms-only always takes the
`infer_chords_billboard_v1` branch, which decodes from scratch on a retired
backend and patches ONLY the locked chord's label (see known_issues.md
2026-07-30 "lock propagation is functionally DEAD"). The propagation-capable
code (`user_constraints.py` + joint decode) is unreachable and targets the
retired `bp48` frontend anyway.

## Why this attempt is different from the five failed ones

Full history: see the 2026-07-30 recon (Gen 0–5) — hand priors, bigram
illustration, scale-relative LM (+0.6–0.8pp, never wired), the 2026-07-12/13
joint-decode campaign (every λ→0), the root-motion bigram (+0.8pp of a
+15.7pp ceiling, FAILS). The standing verdict
(`docs/literature_review_chord_lm_attention.md`): a grammar prior over the
model's own noisy emissions is a saturated slot.

**All five attempts conditioned on the model's own (noisy, miscalibrated)
output.** Gen 4's entropy-gated ii-V-I experiment concluded: *"grammar needs
trustworthy neighbours and honest uncertainty calibration, which is exactly
where the model fails."* A user lock supplies the trustworthy neighbour by
construction. This is conditional infilling anchored on ground truth, not an
unconditional decode prior. Published validation of exactly this shape:
SERENADE (arXiv:2310.11165) — user-locked chords injected into a
bidirectional model, ROI 1.18 labels fixed per human correction. Our target
is the cheap/interpretable version of that.

## Representation (Louis's directive, 2026-07-30)

Normalize each trigram **by the candidate ("alleged") middle chord**, and
keep quality as first-class:

- context chord token = (Δ = root interval from candidate root, quality)
- candidate token = quality only (its root is the origin)
- quality classes = existing `QUAL5` (maj / min / dom / hdim / dim) from
  `harmonia/models/progression_encoder.py::FINE_TO_QUAL5`

Example — ii-?-I, candidate G7 in C: prev = (min, −5), next = (maj, +5),
candidate = dom. Fires identically in all 12 keys; no dependency on our key
estimator. Tritone-sub Db7 in the same slot: prev = (min, +1), next =
(maj, −1), candidate = dom — a distinct, learnable pattern.

Scoring candidate c at position t:
`S(c) = P(prev=(Δp,qp), next=(Δn,qn) | q_c)` from smoothed counts,
normalized across the candidate set. Backoff: full triple → prev-only +
next-only target-relative bigrams → candidate-quality unigram
(Witten-Bell; literature says trigram is the sweet spot for harmony and
simple smoothing suffices — Korzeniowski et al. 2018).

## Model + training data

- Trigram/bigram count tables in the representation above. Reuse the
  fitting/backoff machinery style of
  `harmonia/models/section_structure.py::build_progression_model()` (already
  transposition-invariant, already cached) — but the tables are NEW (that
  model keys on neighbour-to-neighbour Δroot, ours keys target-relative).
- Corpora on disk today (no downloads): `data/accomp_db/db.jsonl` (1,856
  songs), POP909 `chord_midi.txt` (909), ChoCo JAMS label-only charts
  (1,623: billboard/isophonics/jaah/rwc-pop/uspop/robbie-williams). Dedupe
  runs of identical chords first (model change-to-change transitions, not
  per-beat frames). Keep a held-out song split.
- Genre note: accomp_db is jazz-heavy, ChoCo+POP909 pop-heavy. v1 pools
  them; a genre-conditioned table is a later knob.

## Integration (pinned after inference-chain recon, 2026-07-30)

Recon facts that force the shape:
- The live analyze path is `NNLS24ChordHead.run_full` (greedy per-segment
  classifier + post-passes). It has **no Viterbi, no constraint seam, no
  prior hook** — `ChordHeadConfig` deliberately excludes `user_constraints`
  (`harmonia/stages/chord_head.py:120-125`).
- The clamp machinery that exists (`joint_decode` `constraints` +
  `CLAMP_NATS=40` soft bonus + `q5_bonus` closure seam) lives on the bp48
  backend only — using it means decoding the correction on a different
  backend than produced the chart (what reinfer wrongly does today).
- Cheap assets on the nnls24 side: NNLS bothchroma is stem-cached
  (`data/cache/nnls_infer/<stem>.npz`); the heads' per-beat root/quality
  posteriors are a sub-second numpy pass over it. The musx .lab cache is
  labels-only (no posteriors) — not usable for re-scoring.

Decision:
- **Post-hoc span-lattice re-score on the displayed chart.** Client sends
  its current chord list + locks; server recomputes per-beat head posteriors
  from the NNLS cache (hit ⇒ sub-second), pools them per displayed span,
  builds a candidate lattice (acoustic top-K + locked chords), and decodes
  with acoustic log-posterior + λ·context-prior, locked spans clamped.
  Boundaries fixed. Stateless; matches whatever the user is looking at,
  including their prior edits.
- Per span, candidate set = acoustic top-K (K≈6; JAAH premise check showed
  top-2 root recall 76.3% vs top-1 60.7%, so the truth is usually in the
  list) + the locked chord where locked.
- Decode = second-order Viterbi / forward-backward over (pair-of-spans)
  states so trigram factors apply exactly; locked spans clamped to a delta.
  Combined score = acoustic log-posterior + λ · context-prior log-prob.
  λ tuned on simulated locks (below), NOT assumed.
- Locality: propagation naturally decays with distance from the lock through
  the lattice; report only spans whose argmax changed.

## Phase 2 wiring contract (so Phase 1 and Phase 2 can build in parallel)

- Model API (Phase 1 provides, Phase 2 consumes):
  `load_context_prior()` → object with
  `score_candidates(prev_token, next_token) -> (60,)` normalized over
  12 roots × QUAL5, tokens = (root_pc, qual5), either side may be None
  (directed-bigram fallback). Until Phase 1 lands, Phase 2 uses a uniform
  stub behind try/except.
- Acoustic side (no beat tracking needed): pool NNLS bothchroma frames
  within each displayed span's [t0,t1) into one 24-d row, run the nnls24
  heads on the pooled rows → per-span root (12,) and quality posteriors
  (folded to QUAL5). Stem-keyed cache hit ⇒ sub-second.
- Lattice: per span, candidates = acoustic top-K (K≈6) ∪ {displayed chord}
  ∪ {locked chord if locked}; locked spans have candidate set = {lock}.
  Exact second-order DP (state = pair of adjacent span choices) maximizing
  Σ log p_acoustic(c_i) + λ · Σ log P_ctx(c_i | c_{i-1}, c_{i+1}).
  Boundaries never move.
- Serving: NEW endpoint `POST /api/context_rescore/<file>` — request =
  client's current chord list (t0, t1, root, quality per span) + locks;
  response mirrors `/api/reinfer`'s `{chords, diff, n_changed}` shape so
  `applyResp` keeps working. `/api/reinfer` stays untouched for merges.
- Client: `app_shell.html` runReinfer posts to the new endpoint when
  merges are empty. `chart_interactive.py` port comes after validation.

## Guardrails

**Staleness rule (Louis, 2026-07-30): do NOT treat pre-deprecation findings
as binding.** The Gen 0–5 verdicts above (λ→0, "root re-ranking is dead",
coverage gates) were all measured on the old bp48 pipeline, deprecated
since. They inform hypotheses; they veto nothing. Anything worth deciding
on gets re-measured on the CURRENT path.

**Don't trust harmonia code (Louis, 2026-07-30).** Any reused utility
(corpus loaders, Harte parsers, QUAL5 maps, nnls heads) must be verified
against raw data before its output feeds a decision: print parsed tokens
next to raw source lines, sanity-check aggregate stats against music-theory
expectations (e.g. dom share high in jazz, V→I motion dominant), and
double-check any number that looks weird before building on it.

Methodological guardrails that stand on their own:
1. **Evaluate on the real production decode path** — never a proxy harness
   (this exact trap reversed a shipped default once, issue #25).
2. Track "corrupted correct neighbours" separately from "fixed wrong
   neighbours" in every eval; net-negative on corruption kills the arm.
3. No λ-injection into the full unconditional decode in this feature. The
   prior only ever acts in a re-score anchored on ≥1 locked chord.
4. Report root-changes and quality-changes as separate metrics — measure,
   don't assume either direction.

## Evaluation ladder (cheap premise first — process rule 2)

1. **Symbolic infilling check (no audio):** mask every middle chord in
   held-out songs; measure top-1/top-3 recall of P(c | prev, next) vs
   unigram + "repeat-a-neighbour" baselines. Minutes to run.
2. **Theory sanity panel (the user's ask):** in a ii-?-I context the ranked
   list should surface V7 / V / tritone-sub Db7-type candidates; also check
   V-of-V, backdoor bVII7, iiø-V-i in minor. Inspectable table, goes in this
   doc.
3. **Simulated-lock ROI on real decodes:** production-path charts for songs
   with GT (JAAH/RWC/aligned-139); lock one wrong-or-right chord to GT,
   re-score, report: neighbours fixed per lock (ROI), neighbours corrupted
   per lock. SERENADE's ROI 1.18 is the reference point; ship bar is
   ROI clearly > 0 with corruption ≈ 0.

## Status log

- 2026-07-30: branch created; recon (4 agents) done except inference-chain
  map; design settled; Phase 1 (corpus + model + checks 1–2) delegated.
