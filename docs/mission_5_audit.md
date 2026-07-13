# Mission 5 — Deep audit: why only "+2pp", and is the ceiling real?

Audit of the shipped Mission-5 prototype (`scripts/llm_chord_priors.py`,
`scripts/eval_llm_priors.py`, `docs/mission_5_*`). Verified against the code and
by re-running the eval + a factor ablation, 2026-07-13.

## TL;DR (honest)

1. **"+2pp" is not a measured result — it is the *ship gate* threshold.** The
   only place +2pp appears is `mission_5_bayesian_integration.md` §5: "ship
   `use_llm_priors=True` iff Δ(root+q5) ≥ +2pp with 0 regressions." There is **no
   measured real-audio gain at all.** The prototype's own simulation reports
   +6 to +50pp, not +2pp. So the premise "the work delivered +2pp" is itself a
   misread; what was delivered is a design, an offline analyst, and a simulation
   that cannot be trusted (see #3).

2. **The "LLM analyst" never runs as an LLM.** `anthropic` is not installed in
   `.venv` (verified: `import anthropic` → ImportError). Every number in the repo
   comes from the **offline rule-based analyst**, which is not Claude and does no
   music-theoretic reasoning — it counts the chart's own chords. "LLM priors" is
   currently a misnomer for "chart-statistics priors."

3. **The eval is self-referential; its big numbers measure a knob, not
   analysis.** The synthetic ground truth *is* the iReal chart; the priors are
   *derived from that same chart*. So the prior is an aggregate of the answer
   key. The reported +35–50pp measures how hard we tilt (the `max_nats` strength
   knob) toward a lookup of the GT — not the quality of any harmonic analysis. A
   deeper V2 analyst would produce the *same* per-root quality lookup on Autumn
   Leaves and score identically. **The current eval is structurally incapable of
   showing V2 > V1.**

4. **The integration is designed but never wired.** `grep` for
   `to_bayesian_factors` / `use_llm_priors` / `BayesianFactors` across
   `harmonia/` returns only the eval script. The four "seams" into
   `joint_decode` are real (the `tonic`, `q5_bonus`, `pool_groups`,
   `bigram_logp` hooks all exist and were verified), but no glue fills them and
   nothing touches the real pipeline. Mission 5 has produced **zero end-to-end
   audio evidence.**

The +2pp "modest gain" the user is reacting to does not exist as a measurement.
The real state is worse *and* more fixable than that: nothing has been measured
on audio, and the simulation that was measured is circular.

## 1. What V1 actually does (exact, line by line)

### 1a. The analyst (`llm_chord_priors.py`)

Two paths emit an identical JSON schema:

- **LLM path (`llm_analyze`, lines 376–393):** one `messages.create` call to
  `claude-opus-4-8` with a `json_schema` structured output. The prompt
  (`_build_brief`, 343–373) hands Claude the raw per-bar tokens + title/style/key
  and asks for `tonic_pc`, `structure.repeats`, `chord_priors` P(q|root),
  `transition_priors` P(root|prev), `confidence`. **Never executed** (no SDK).

- **Offline path (`offline_analyze`, 222–299) — the one that actually runs:**
  - `key` → tonic/mode by string parsing (`_key_to_tonic`).
  - `structure.repeats` → **exact adjacent token-identical block repeats**
    (`_detect_repeat_spans`, 302–334): greedy largest-L where
    `bars[i:i+L] == bars[i+L:i+2L]` after collapsing `x`/empty bars. Autumn
    Leaves → one group (bars 1–8 ≈ 9–16).
  - `chord_priors` → **per-root observed quality marginal**, Laplace-smoothed
    (254–264): `P(q5 | root) = count(root,q5)/count(root)`. Counts the chart.
  - `transition_priors` → **observed adjacent-root bigram** from the chart
    (266–277): `P(next_root | prev_root)`.
  - `confidence` → `0.55 + 0.30·(diatonic root fraction)` (282–285). Autumn
    Leaves → 0.83.

So the offline analyst is: **key parse + exact-repeat detection + per-root
quality histogram + adjacent-root bigram, all read off the chart.** No functional
labels (I/ii/V), no cadence detection, no secondary dominants, no borrowed
chords, no voice-leading, no position/section conditioning. The `chord_priors`
comment even flags it: *"position-agnostic form v1."*

### 1b. Translation (`to_bayesian_factors`, 428–478)

- `strength = max_nats(8.0) · confidence` → 6.7 nats on Autumn Leaves.
- `quality_bonus[root][q5] = strength·(p − 1/5)` — centred additive per-q5
  log-bonus.
- `root_transition_bias[prev][next] = strength·p`.
- `pool_group_bars` from `structure.repeats`.
- Ceiling 8 nats vs a human confirm's `CLAMP_NATS ≈ 40` (verified in
  `user_constraints.py`). The confidence→strength honesty knob is sound and is
  the best-designed part of V1.

### 1c. The eval (`eval_llm_priors.py`)

- GT = one state per bar = the bar's first chart chord (`ground_truth_states`).
- Emission = GT bump + noise concentrated on the two real confusions (5th-apart
  root, dom↔maj) — this part is well-motivated (#19).
- Coordinate-ascent ICM solver with tied pooled groups.
- Two arms: uninformed (zero trans/q, identity pooling) vs LLM-guided (all
  factors). Reports sweeps, root acc, (root,q5) acc over seeds.

Re-run (offline analyst, 100 seeds):

| σ   | uninf (root,q5) | guided | Δ |
|-----|------|------|------|
| 0.8 | 83.7 | 90.8 | +7.1 |
| 1.2 | 46.2 | 85.4 | +39.2 |
| 1.6 | 26.1 | 76.0 | +50.0 |

## 2. Factor ablation — where the simulated gain comes from

I split the guided arm into its three factors (150 seeds):

| factor (Δ full-label acc)      | σ=1.2 | σ=1.6 |
|--------------------------------|-------|-------|
| quality prior P(q\|root) only  | **+29.1** | **+33.7** |
| transition bias only           | +9.5  | +8.6  |
| pooling only                   | +10.4 | +6.4  |
| all                            | +39.8 | +50.9 |

**The quality prior does ~75% of the work.** And that is exactly the factor most
corrupted by the circularity: on Autumn Leaves each root has one quality in the
chart (G→min, D→dom, A→hdim, C→min, F→dom, Bb→maj, Eb→maj). So `P(q|root)` from
the chart is a near-deterministic **root→quality lookup table**, and the GT
quality is that same table. The simulation is rewarding memorisation of the
answer key, scaled by the strength knob. Push `max_nats` up and Δ climbs toward
100% regardless of any "analysis."

This is the crux: **the number that dominates the eval is the number the eval is
least able to trust.**

## 3. Why "+2pp is the ceiling" is the wrong question

The mission asks whether the design has a +2pp ceiling. That framing accepts a
measurement that was never taken. The correct statements are:

- **On the simulation:** the ceiling is ~+50pp and rising with the strength knob
  — but that number is meaningless (circular), so its size proves nothing.
- **On real audio:** the gain is **unknown**, because nothing is wired. The +2pp
  is a *gate we set*, not a result we got.

So V1 is neither "sound but shallow +2pp" nor "wrong approach capped at +2pp."
It is **an unmeasured design plus a circular sim.** The right next move is not a
fancier analyst — it is to *make the measurement honest*: wire the glue and
score end-to-end on the Mission-1 real-audio benchmark, where the audio is a
different source than the chart (so chart-derived priors are non-circular).

## 4. What V1 is genuinely missing (feeds V2)

Ordered by expected real-audio value, cross-checked against our ablation history:

1. **Position/section-conditional quality** `P(q | root, section)`. V1's marginal
   collapses a root that is a diatonic I in the A-section and a secondary
   dominant in the bridge into one distribution. This is the one lever V1
   explicitly punts and the one the front-end is weakest on (q5 = 44%, #19).
   **Highest-value real lever.**
2. **A non-circular eval.** Score priors from source X against GT from source Y.
   Without this, no analyst change is measurable.
3. **The wiring itself** — the ~30-line glue into `infer_chords_v1`.
4. Everything the mission brief lists as "deep" — cadences, secondary dominants,
   borrowed chords, voice-leading — enters the decoder *as transition bias*, and
   **that slot is saturated** (#27: key-local bigram, encoder fusion,
   density-ratio fusion all net-negative on jazz). Expect ~0 from these. This is
   the single most important constraint the V2 wishlist ignores.

## 5. Verdict

- V1 code quality is good; the confidence→strength honesty knob and the
  reuse-the-user-constraint-interface design are the right instincts.
- V1 evidence is not usable: not an LLM, never wired, circular sim.
- The mission's proposed V2 (cadences/secondary-dominants/voice-leading) mostly
  targets the *saturated* transition slot and would be invisible to the current
  eval. It is the wrong place to spend effort next.

See `mission_5_v2_deep_analysis.md` for the go/no-go and the one V2 change that
is actually worth building.
