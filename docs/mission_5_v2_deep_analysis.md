# Mission 5 — V2 proposal + go/no-go

Response to the "deeper analysis" brief. Read `mission_5_audit.md` first — the
audit changes the question. Written 2026-07-13.

## The honest reframe

The brief asks for a V2 analyst with cadence detection, secondary dominants,
borrowed-chord ID, and voice-leading, and estimates +5–10pp. Two audit findings
override that plan:

1. **Most of that sophistication lands in the transition slot, which is
   saturated.** Cadential motion (V→I), secondary dominants (V/V→V), tritone subs
   — these are all statements about `P(next_root | prev_root)` or the bigram.
   `known_issues.md` #27 measured three independent transition-fusion attempts on
   jazz (key-local bigram, progression-encoder shallow fusion, density-ratio
   fusion); **all three drove their optimal weight to ~0 or net-negative.** A
   V2 that dresses the same slot in richer theory will, on the weight of that
   evidence, net ~0. Building it is re-litigating a settled ablation.

2. **The current eval cannot see analyst depth.** Because the sim's GT and prior
   both come from the chart (audit §2–3), a "deep" analyst and the trivial
   chart-histogram produce the same `P(q|root)` lookup and score identically.
   You cannot A/B V2>V1 until the eval is non-circular. **Fix the eval before
   touching the analyst.**

So V2 is not "a smarter LLM prompt." V2 is: **(a) make the measurement real,
(b) add the one prior lever that is theoretically unsaturated, (c) then, and
only then, let the LLM fill it.**

## What V2 actually is (3 parts, in order)

### Part A — Wire the glue and get the first real-audio number (must-do)

~30 lines in `infer_chords_v1`, behind `use_llm_priors=False`, filling the four
existing `joint_decode` hooks (all verified present):

```python
def apply_llm_priors(analysis, segs, beat_times):
    f = to_bayesian_factors(analysis)          # existing
    tonic = f.tonic if f.confidence >= KEY_TRUST else inferred_tonic
    def q5_bonus(seg_idx, root):
        row = np.zeros(5)
        for q5, nats in f.quality_bonus.get(root, {}).items():
            row[q5] = nats
        return row
    pool_groups = bars_to_segment_groups(f.pool_group_bars, segs, beat_times)
    # transition bias: default OFF (saturated slot, #27)
    return dict(tonic=tonic, q5_bonus=q5_bonus, pool_groups=pool_groups)
```

Gate (unchanged from integration doc §5): ship iff Δ(root+q5) ≥ +2pp, 0
regressions, on the held-out jazz set **with audio from a different source than
the chart**. This is the number Mission 5 has never had. Effort: ~half a day.

### Part B — A non-circular eval (must-do, precedes any analyst work)

Two options, cheapest first:

- **B1 (cheap, symbolic):** derive priors from **iReal chart A**, score against a
  **different lead-sheet of the same tune** (guitar-tab or a second iReal
  version) as GT. Where the two sources disagree (inversions, tritone subs,
  passing chords) is exactly where a real analyst must earn its keep. `parse_all`
  makes this a <1hr script.
- **B2 (real, gated):** the Mission-1 real-audio benchmark. Prior from chart,
  GT from the audio's annotation. This is the ground truth for the whole
  mission. Gated on Mission 1 landing (#20/#28).

Until B exists, **no analyst change is measurable** — this is the highest-leverage
item in the whole mission and it is not an analyst change at all.

### Part C — The one unsaturated analyst lever: `P(q | root, section)`

This is the *only* piece of the brief's "deep analysis" that targets a live,
unsaturated slot (the q5 head, 44% acc, #19) rather than the dead transition
slot. Concretely, condition the quality prior on section/position so the same
root gets different quality expectations by function:

```jsonc
"chord_priors_by_section": {
  "A": { "2": {"dom":0.8,...} },      // D as V7 in the A-section
  "bridge": { "2": {"min":0.6,...} }  // D as ii of the bridge's local key
}
```

Maps to a **section-indexed `q5_bonus`** — the callback already takes `seg_idx`,
so this needs *no decoder change*, only a per-section lookup. This is where an
LLM genuinely beats the chart histogram: the histogram gives one distribution
per root; Claude can say "this D is a secondary dominant *here* and a ii *there*"
— information the marginal literally cannot represent.

**Note the honest limit:** on a clean fully-diatonic standard like Autumn Leaves,
each root already has one quality everywhere, so section-conditioning adds ~0
there. It pays off on tunes with **functional reuse of a root** (secondary
dominants, modulating bridges, borrowed chords) — Rhythm changes bridge, Giant
Steps, anything with a modulating B-section. So the eval for Part C must be run
on those tunes, not on Autumn Leaves.

## Estimated gain (concrete, and lower than the brief hoped)

| Lever | Slot | Prior evidence | Est. real-audio Δ(root+q5) |
|---|---|---|---|
| Key/mode correct | diatonic prior (#20) | removes #0 confusion | +1 to +3pp, tune-dependent |
| Pooling identical repeats | pool_groups (#28) | **+10pp q5 measured** | +2 to +6pp (already partly banked by #28) |
| `P(q\|root)` marginal (V1) | q5_bonus | — | +1 to +3pp where front-end is weak |
| `P(q\|root,section)` (V2 Part C) | q5_bonus | targets #19, unsaturated | **+2 to +4pp** on functional-reuse tunes, ~0 on plain diatonic |
| Cadence / sec-dom / voice-leading as transition bias | bigram | **#27: net-negative ×3** | ~0 to slightly negative |

**Realistic V2 total: +3 to +8pp on real audio, front-loaded on key + pooling,
with the analyst-depth contribution (Part C) a modest +2–4pp confined to
functionally-complex tunes.** The brief's +5–10pp is reachable only on the harder
tunes and only if the transition-slot items are dropped. On clean standards the
honest number is low single digits — most of which key+pooling already capture
without any LLM at all.

The uncomfortable truth: the biggest wins (key, pooling) need the LLM *least*
(a key-detector and an SSM already do most of it). The LLM's unique contribution
— section-conditional quality on functionally-complex tunes — is real but
narrow.

## Go / No-go

**GO — but not on the brief's V2.** Build in this order, stop at any gate that
fails:

1. **Part A (glue) + Part B1 (symbolic non-circular eval).** ~1 day. This gives
   the first honest Δ. **Stopping criterion:** if V1's chart-histogram priors
   don't clear +2pp on B1's cross-source eval, the whole prior mechanism is too
   weak and you stop — no V2 analyst will rescue a dead injection path.
2. **Part C (section-conditional quality)**, LLM path enabled, evaluated on
   functional-reuse tunes only. **Stopping criterion:** ship iff Part C beats the
   V1 marginal by ≥+2pp on those tunes with 0 regression on plain standards.
3. **NO-GO on cadence / secondary-dominant / voice-leading transition priors**
   until #27's saturation is re-tested for *per-song* bias specifically — and
   even then, default-off, low-strength, measured before enabled. Do not build
   the voice-leading machinery the brief describes; #27 says it won't move the
   metric.

**Why not just declare +2pp the ceiling and stop?** Because the +2pp was never
measured — it's a gate. Parts A+B are cheap (~1 day) and finally produce the
real number. That measurement is worth more than any further design. If A+B come
back under +2pp, *then* the honest conclusion is "LLM priors aren't the lever for
this decoder," and it will be backed by data instead of a circular sim.

## One-paragraph summary for the user

V1 isn't shallow-but-working at +2pp — it's unmeasured. It's the offline
rule-based analyst (Claude never ran; `anthropic` isn't installed), its priors
are chart-count histograms not harmonic analysis, its impressive sim numbers are
circular (prior and ground truth are both the chart), and the injection glue was
never wired into the real pipeline. The fix isn't a fancier analyst — the
brief's cadence/secondary-dominant/voice-leading ideas mostly hit the transition
slot that #27 already proved saturated, and the current eval can't even see
analyst depth. The right V2 is: wire the ~30-line glue, build a non-circular
eval (prior from one source, GT from another), add exactly one new prior
(section-conditional `P(q|root,section)`, the only unsaturated lever), and get
the first real-audio number. Estimated honest gain +3–8pp, front-loaded on
key+pooling (which barely need an LLM), with the LLM's unique value a narrow
+2–4pp on functionally-complex tunes. ~1 day to the first real measurement; stop
if it's under +2pp.
