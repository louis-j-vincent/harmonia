# Handoff — extended-vocabulary chord LM (7ths / 6ths / 9ths)

**For:** a fresh session (the "LM agent"), 2026-08-02.
**From:** the harmonia_min server session (P1–P3 landed: commits `4c2152f`,
`e6322c6`, `23bcd1c`, `4d3a721` on `feat/chord-lm`).

## Why this exists

Louis, today: *"on peut augmenter le LM pour qu'il apprenne aussi les 7èmes
6èmes 9èmes (vocabulaire étendu)"*. He is right, and the constraint that
motivates it is already shipped and documented: the lock-propagation endpoint
I just routed (`/api/context_rescore/<file>`) decodes only **QUAL5**
(maj/min/dom/hdim/dim), so **any span the re-score changes loses its seventh**
— a `C-7` comes back `C:min` and the shell renders `Cm`. Unchanged spans keep
their quality, so the damage is confined to exactly the spans the model moved,
but it is real.

## The screen is already done — do NOT redo it, build on it

Cheapest falsifier (project rule #2) was coverage: the trigram key is
`(q_candidate, Δprev, q_prev, Δnext, q_next)`, so a bigger quality alphabet
could in principle explode the key space and leave every cell too sparse to
use. Measured on the full pooled corpus (accomp_db + POP909 + ChoCo), same
accepted label set in every arm, only the class granularity changing
(`scratchpad/lm_vocab_coverage.py`, run 2026-08-02):

| arm | classes | trigrams | distinct keys | ≥5 obs | ≥10 | ≥20 |
|---|---|---|---|---|---|---|
| QUAL5 (ships today) | 5 | 330 985 | 6 659 | 97.9% | 95.7% | 92.9% |
| Q8 (+7ths, sus) | 9 | 341 509 | 12 761 | 95.8% | 91.9% | 86.9% |
| Q16 (+6ths, 9ths) | 18 | 342 929 | 16 438 | 94.4% | 89.6% | 83.7% |

**Verdict: the premise is alive.** Going from 5 to 18 classes costs only
92.9% → 83.7% of occurrences sitting in well-observed keys (≥20), and the key
space grows 2.5×, not |Q|³ — real music uses a small corner of the
combinatorial space.

**Second finding, and the stronger argument for doing this:** the trigram
count *rises* with the finer vocabulary (330 985 → 342 929, +3.6%). Consecutive
identical chords are de-duplicated, so under QUAL5 a real `C → C7` move was
being erased as "same chord". The coarsening was not just blurring labels, it
was **deleting ~12 000 chord changes** from the training signal.

## The job

1. **Pick the vocabulary.** Q16 is affordable; Q8 is the conservative option.
   Decide with the numbers above plus per-class counts (some classes — `aug7`,
   `minmaj7` — may be too rare to earn a slot; fold them). State the choice and
   why.
2. **Extend the corpus parser.** `harmonia_min/chord_context_prior.py`:
   `HARTE_QUAL_TO_QUAL5` + `parse_harte_lite`. Keep the current deliberate
   drops (`"5"` power chords, `"1"`, interval-list chords) — they break the
   sequence rather than being silently folded, on purpose; do not "fix" that.
3. **Backoff is mandatory**, not optional, for the ~16% tail: extended key →
   Q8 key → QUAL5 key → root-evidence only. The existing `score_candidates`
   already has a backoff seam; extend it, and make the level actually used per
   query observable (a counter you can report), so the tail is measurable
   rather than assumed.
4. **The acoustic side must speak the same vocabulary — this is the part that
   is easy to miss.** `harmonia_min/span_rescore.py::acoustic_logp_musx` folds
   music-x-lab's **separate** triad / seventh heads into 60 candidates via
   `_FOLD[t,s,q]` (`_build_fold_tensor`). musx already emits seventh (and
   ninth) information — we are throwing it away in that fold. Extend `_FOLD`
   and `N_CANDIDATES` to the new alphabet, or the LM will propose qualities the
   acoustic model cannot score. Read `acoustic_logp_musx`'s docstring first: it
   documents a real transpose bug (root fast, type slow) that a naive reshape
   reintroduces — verified by `scripts/check_span_rescore_sanity.py`, run it.
5. **The NNLS fallback** (`acoustic_logp_nnls`, 7 quality heads folded to 5)
   needs the same treatment or an explicit "extended vocabulary unavailable on
   this backend" degradation. Do not let it silently emit QUAL5 into an
   extended lattice.

## Acceptance criteria (quantitative, not "it works")

- **Held-out perplexity / infilling recall** of the extended table vs QUAL5,
  on the existing held-out split (`_is_heldout`). Report both; a worse
  perplexity at finer granularity is expected and fine — what matters is
  infilling *accuracy at the family level not regressing* while sevenths
  become predictable at all.
- **Backoff usage histogram**: what fraction of live queries resolve at the
  extended level vs each fallback.
- **End-to-end, through the live route** (`POST /api/context_rescore/<file>`
  on :7772, server restarted first — no reloader): lock a chord on
  `min_maroon_5_this_love`, confirm a changed span now **keeps its seventh**
  (this is the user-visible point of the whole exercise).
- **No regression** on the 6 charts in `harmonia_min/state/charts/`: zero
  locks must still change nothing (structural property of
  `differential_rescore` — if it breaks, you broke the differential design).

## Hard constraints

- **Never edit `harmonia_min/app_shell.html`** — a concurrent session owns it.
  If the client needs a change, write it under "Client asks" in
  `docs/handoff_2026-08-02_min_server_gaps.md` and stop.
- `harmonia_min/server.py`, `annotations.py`, `context_rescore.py` are the
  server session's; coordinate before editing (extending the vocabulary should
  need at most the `_q_tail` / `Q5_HARTE` maps in `context_rescore.py`, which
  are deliberately small and isolated for exactly this).
- Port **7772** only, never 7771. Restart the server after any server-side
  change and say so.
- Branch `feat/chord-lm`, same working tree, stage explicit paths only, never
  `git add -A` / `checkout` / `stash` / `reset` — other sessions' uncommitted
  work lives here.
- Tests red-first; prefer real charts/corpora over synthetic fixtures.
- Log findings in `docs/minimal_pipeline_log.md` as you go, and state what your
  change does **not** solve (project rule #4).

## Known non-solves you inherit

- Boundaries never move in the re-score — a lock can relabel a span, never
  split or merge it.
- Cold `musx_probs` cache silently degrades to the weaker NNLS-24 heads.
- The prior is pooled across genres; jazz/pop arms exist
  (`chord_context_prior_{jazz,pop}.npz`) but nothing selects between them.
- `/api/context_rescore` reports `new_confidence: null` for propagated spans —
  there is no calibrated confidence for a propagated change yet.
