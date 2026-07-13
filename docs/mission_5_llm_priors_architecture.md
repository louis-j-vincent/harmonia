# Mission 5 — Architecture: LLM analyst → Bayesian decoder

How the LLM and the Bayesian decoder split the job, what crosses the boundary,
and why the interface is the one we already have. Companion:
`mission_5_llm_priors_research.md` (why), `mission_5_bayesian_integration.md`
(exact code seams), `scripts/llm_chord_priors.py` (prototype).

## 0. The one-line architecture

```
audio ─► [ audio front-end ]───────────────────────────┐  (per-beat root posterior,
   │        stage1_pitch → beat_seq_v4 → ctx q5 head    │   detected key, tempo)
   │                                                    ▼
iReal chart (optional) ─► [ LLM ANALYST ] ─► priors JSON ─► [ to_bayesian_factors ]
   title / style              claude-opus-4-8              │        │
                              (or offline analyst)         │        ▼
                                                           │   factors: tonic, pool
                                                           │   groups, q-bonus,
                                                           │   root-transition bias,
                                                           │   strength(=conf·ceiling)
                                                           ▼        │
                              [ joint_decode / semi_markov_decode ] ◄┘  ← EXACT MAP
                                        │                               (Bayesian keeps
                                        ▼                                control)
                                  ChordChart
```

The LLM never labels chords that ship. It emits *priors*; the exact segment
Viterbi decides the labels, coupling the LLM's priors with the real acoustic
evidence and the transition factor. **Intuition from the LLM, rigour from the
decoder** — and, critically, the decoder can and will overrule a wrong LLM prior
when the acoustic evidence disagrees strongly enough, because the LLM's
contribution is a *finite* additive log-term, not a clamp.

## 1. Division of labour

| Concern | Owner | Why |
|---|---|---|
| Key / mode | **LLM** (fallback: `infer_key`) | LLM recognises the standard; removes #0's maj/min confusion |
| Section structure / which spans repeat | **LLM** (fallback: SSM + `_detect_repeat_spans`) | high-level "these are the same" beats a chroma-SSM peak (#1 Candidate C failure) |
| `P(quality \| root, position)` | **LLM** | targets the 44%-accurate q5 head (#19); LLM knows "V→dom7, vi→min" |
| `P(root \| prev_root)` song-specific bias | **LLM**, softly | the *generic* grammar slot is saturated (#27); only per-song bias is new info |
| Boundary / duration shape | **decoder** (semi-Markov, #27 M2) | already owned; LLM section lengths only sharpen it |
| Acoustic evidence (per-beat root/q5) | **decoder front-end** | the LLM cannot hear |
| Final (root, q5) label + confidence | **decoder** (joint Viterbi + calibrator) | mathematical, interpretable, calibrated (#26) |

## 2. Input to the LLM

Symbolic + numeric only (Claude has no audio here — see research §4):

1. **iReal chart** (raw per-bar tokens with section labels + `x` repeat bars) —
   the primary, highest-trust input when present.
2. **Metadata** — title, style/groove, iReal key signature.
3. **Optional audio summary** — a compact JSON of our own front-end: detected
   key, tempo, and a coarse per-beat root-posterior digest. Lets the LLM
   reconcile chart vs. what was actually played (e.g. a chart in Gm but the
   recording a step down), and is the *only* input in the chart-less case.

The prompt (`_build_brief`) hands the model the chart and asks for the schema in
§3, with an explicit instruction to be honest in `confidence` — high when it
recognises the tune and the harmony is unambiguous, low otherwise.

## 3. Output schema (LLM → priors JSON)

Emitted by **both** the LLM path (via structured output / `json_schema`) and the
offline analyst, so downstream code is path-agnostic:

```jsonc
{
  "key": "G-", "mode": "minor", "tonic_pc": 7,
  "structure": {
    "form": "A16 B8 C8",
    "sections": [{"label":"A","start_bar":1,"end_bar":16}, ...],
    "repeats": [[3,4]]                 // section indices asserted identical → pool
  },
  "chord_priors": {                    // P(quality | root), per pitch-class
    "9": {"maj":0.07,"min":0.07,"dom":0.07,"hdim":0.73,"dim":0.07},  // A → ø (Am7b5)
    "2": {"dom":0.73, ...},            // D → dom7
    "7": {"min":0.81, ...}             // G → min
  },
  "transition_priors": {               // P(next_root | prev_root)
    "9": {"2":1.0}, "2": {"7":1.0}     // A→D→G  (ii-V-i in Gm)
  },
  "confidence": 0.83
}
```

Real output of the offline analyst on Autumn Leaves — the harmony read is
functionally correct (Gm: A→ø, D→dom7, G→min, Bb→maj; ii-V-i motion) and the
8-bar internal repeat (bars 1-8 ≈ 9-16) is detected as a pool group.

## 4. Translation to decoder factors (`to_bayesian_factors`)

The JSON is mapped to the exact objects the decoder's factor interface takes:

- `tonic` / `mode` → `joint_decode(tonic=...)` and the diatonic-prior key.
- `structure.repeats` → **`pool_group_bars`**: bar-spans the decoder ties and
  sums emission over (the `pool_groups` argument to `joint_decode`; P3 √N
  denoising, #28).
- `chord_priors` → **`quality_bonus[root][q5] = strength·(p − 1/5)`**: an
  additive per-q5 log-bonus, centred so the mean quality gets ~0. Feeds the
  `q5_bonus` callback slot in `joint_decode` (the same slot the progression
  encoder's shallow fusion used, #27 H2).
- `transition_priors` → **`root_transition_bias[prev][next] = strength·p`**: a
  log-boost folded into the bigram transition table.
- `confidence` → **`strength = ceiling · confidence`** (nats). Ceiling defaults
  to **8 nats**, vs a human user-confirm's `CLAMP_NATS ≈ 40`
  (`user_constraints.py`) — the LLM is deliberately an order of magnitude weaker
  than a human assertion.

## 5. Why this interface (and not a new one)

`harmonia/models/user_constraints.py` already turns *user* assertions into
decoder factors of exactly these three kinds:

- chord-confirm → additive emission log-bonus on `(root, q5)` cells + a
  boundary hint → **our `quality_bonus`** is the same mechanism, weaker.
- section-merge → tied + pooled segments → **our `pool_group_bars`** is
  literally the same `pool_groups` path.

So the LLM is **"an automated annotator"**: mechanically identical to a user,
at lower authority. Consequences:

1. **Zero new decoder recursion.** Everything routes through `joint_decode`'s
   existing `tonic` / `q5_bonus` / `constraints` / `pool_groups` arguments.
2. **The "never blind" discipline is inherited.** #28's rule — pool only on an
   explicit assertion, never blindly (Candidate C, #1) — holds automatically:
   the LLM *is* the assertion, and a low-confidence LLM produces a weak,
   near-inert prior rather than a blind average.
3. **User always wins.** When both a human and the LLM speak, the human's
   40-nat clamp dominates the LLM's ≤8-nat tilt in the same log-space — the
   correct precedence for free.

## 6. Failure modes and guards

| Risk | Guard |
|---|---|
| LLM hallucinates a wrong key on an unfamiliar tune | confidence-gated strength → weak tilt; acoustic evidence + transition factor overrule |
| LLM asserts a repeat that isn't identical | pooled emission only *sums* evidence — if the spans genuinely differ the summed emission is ambiguous, not confidently wrong; and the assertion is confidence-scaled |
| Circular eval (chart-derived prior scored vs same chart) | eval measures convergence/robustness on held-out synthetic emission, never chart-agreement (research §4) |
| No network / no API key | offline rule-based analyst emits the identical schema — pipeline degrades, never breaks |
| Generic-grammar temptation | transition bias is per-song and soft only; the saturated grammar slot (#27) is left alone |

## 7. Scope cut (per the mission's fallback priority)

Delivered core: research (1) + architecture (2) + working prototype (3) on
Autumn Leaves + integration design (4) + a convergence eval (5, simulation).
The **real-audio** end-to-end number is deliberately *not* claimed — it is gated
on the Mission 1 benchmark (`data/real_audio_benchmark/`, #20/#28), consistent
with how every other real-audio claim in this repo is gated.
