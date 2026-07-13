# Mission 5 V2 — Part A + Part B results

First real wiring of the LLM/offline-analyst priors into the production decoder,
and the first *non-circular* attempt to measure them. Written 2026-07-13.
Companions: `mission_5_audit.md` (why V1 was unmeasured),
`mission_5_v2_deep_analysis.md` (the A/B/C plan),
`mission_5_bayesian_integration.md` (the four seams).

## Part A — glue wired

**Done.** `harmonia/models/chord_pipeline_v1.py` now injects analyst priors into
`joint_decode` through three of the four seams, behind `use_llm_priors=False`
(default OFF ⇒ bit-identical to production).

- `apply_llm_priors(analysis, segs, beat_times, *, inferred_tonic, max_nats, beats_per_bar)`
  → `{tonic, q5_bonus, pool_groups, factors}`.
- Helper `bars_to_segment_groups(pool_group_bars, segs, beat_times, beats_per_bar)`
  maps analyst repeat bar-spans → **slot-wise** tied segment groups (bar 1↔9,
  2↔10, … for a parallel span pair — not the whole strain collapsed to one
  chord).
- Wired into `infer_chords_v1(use_llm_priors=..., llm_analysis=..., llm_song=...,
  llm_playlist=..., llm_max_nats=...)`. The three seams feed the primary decode
  and the (optional) local-key re-decode.

**Settings:** `LLM_KEY_TRUST = 0.60` (below it, keep the audio-inferred tonic);
`max_nats = 8.0` (~5× weaker than a user confirm's ~40 nats, further scaled by
analyst confidence). **Seam 4 (transition bias) is OFF** — the bigram slot is
saturated (#27), as the plan dictates.

**End-to-end verification (not just a unit test).** Driving `infer_chords_v1` on
POP909 render 001 with an extreme "dominant-everywhere" analysis (conf 0.95)
flips the decoded quality distribution from `{maj:91, min:32, dom:8}` to
`{dom:117, maj:3, min:1}` — 116/131 labels move. The q5_bonus provably reaches
the real emission (the audit's "never wired" gap is closed). Test:
`tests/test_llm_priors_glue.py::test_llm_priors_shift_labels_end_to_end`.

Note this does **not** yet solve bar↔beat *phase* (CLAUDE.md #4):
`bars_to_segment_groups` assumes bar 1 starts at beat 0 and a fixed
`beats_per_bar` (4/4). A pickup bar or mis-phased grid misaligns the pooled
slots. Documented in the function docstring and here.

## Part B1 — non-circular eval: **INCONCLUSIVE (corpus has no usable set)**

`scripts/eval_llm_priors.py --cross-source` derives priors from iReal source A
and scores them against a *different* lead-sheet B of the same title. The premise
check (CLAUDE.md #2) — enumerate the multi-source pairs and measure their
disagreement *before* trusting any Δ — falsified the premise: **the symbolic
corpus does not contain a usable non-circular test set.**

Census over 7 playlists (jazz1460, pop400, brazilian220, blues50, latin_salsa50,
country, dixieland1), 40 titles present in ≥2 sources:

| bucket | n | why unusable |
|---|---|---|
| identical transcription (0% disagree) | 30 | same bytes in two playlists → trivially circular |
| key mismatch (homonym / transposed)   | 5  | e.g. "Hello" G vs F#, "Goodnight Irene" G vs Ab — different song or transposed; every root disagrees |
| disagree >50% (different composition)  | 2  | title homonyms ("Blue Eyes Crying in the Rain") |
| length mismatch (>4 bars)              | 1  | form differs → per-bar GT misaligns |
| **VALID** (same key, ≤4-bar diff, 0<dis≤0.5) | **2** | genuine same-tune variants |

The two valid pairs (200 seeds, σ=1.2):

| tune | T | dis% | Δcross | Δcirc |
|---|---|---|---|---|
| Blue Room, The | 32 | 3% | −4.4 | −5.2 |
| C'est Si Bon   | 40 | 5% | +12.9 | +13.3 |
| **mean** | | 4% | **+4.2** | **+4.0** |

**Numeric criterion: PASS (+4.2 ≥ +2pp). Honest verdict: INCONCLUSIVE.**
With only 2 valid pairs at 3–5% disagreement, source-A and source-B priors are
~95% identical, so **Δcross ≈ Δcirc by construction** — the test has no power to
distinguish "the analyst generalizes" from "the two sources are near-copies."
The +4.2pp is the circular signal in disguise, and it is carried by a single
tune (C'est Si Bon +12.9; Blue Room is negative). The script gates this
explicitly: `Test power: 2 valid pair(s), mean disagreement 4% → INADEQUATE`.

For comparison, the original circular sim (Autumn Leaves, prior=GT=same chart)
reports +7 to +50pp depending on σ (audit §1c) — all of which the audit already
showed is a strength-knob artifact.

## Stopping criterion

**Do not proceed to Part C on the strength of Part B1.** Not because the prior
failed (+4.2pp technically clears the gate), but because the measurement is not
trustworthy: the corpus cannot supply a non-circular symbolic test with enough
pairs or enough genuine disagreement. Per CLAUDE.md #5 (few-song findings are
hypotheses) and #1 (a plausible number from a broken measurement is worse than
no number), the honest state is:

- **Part A (wiring): shipped and verified end-to-end.** Ready for a real number.
- **Part B1 (symbolic non-circular eval): infeasible on this corpus.** The real
  gate is **Part B2** — prior from the chart, GT from the audio's own annotation
  on the Mission-1 real-audio benchmark (`data/real_audio_benchmark/`, #20/#28),
  where the audio is a genuinely different source from the chart. That is the
  only setup that both (a) has real chord-level disagreement and (b) is not
  circular. It is gated on Mission 1 landing.

Bottom line: the glue is real and measurable; the honest measurement is not
available symbolically and must come from Part B2. The audit's core claim
stands — Mission 5's value is unknown until real audio, and the +2pp remains a
gate, not a result.

## Repro

```
# Part A end-to-end mechanism + factor unit tests
python -m pytest tests/test_llm_priors_glue.py -q

# Part B1 non-circular eval + corpus census
python scripts/eval_llm_priors.py --cross-source --sigma 1.2 --seeds 200
```
