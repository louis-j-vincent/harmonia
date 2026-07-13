# Mission 6 · Premise check — does `repeat_consistency` separate aligned from slipped on real audio?

**Date:** 2026-07-13
**Verdict:** ⚠️ **CONDITIONAL PASS** — the signal is real on real audio, but the
*global* Δ statistic and the *localization* statistic behave differently and both
have named blind spots. Scale to the 20-song harness **with the two fixes below**,
not as-specified.

Script: `scripts/test_mission6_repeat_consistency.py` (single-use).
Plot: `docs/plots/mission6_premise_check.png`.

## Setup (what was actually run)

Three pilots, all on **real inference from real audio** (`docs/audio/*.m4a`), each
aligned to its iReal chart via the existing `align_irealb_to_inferred`:

| Pilot | Source of inference | iReal chart | Expected | Why |
|---|---|---|---|---|
| Autumn Leaves | saved `inferred_autumn_leaves.html` | jazz1460 (A,B,C) | aligned | mission-named known-good |
| Let It Be | saved `inferred_..._let_it_be...html` | pop400 (A,B,i) | **slipped** | documented #22 cycle-shift slip |
| Ghost of a Chance | **fresh** `HarmoniaPipeline.run()` | jazz1460 (A,B) | aligned | genuine issue-#20 pilot; phase1b's best-aligned |

Note on pilot choice: the three issue-#20 audios are ghost/foggy/airegin, but only
ghost/foggy/airegin *audio* exists — no saved inference. Ran fresh inference on
**Ghost** (the #20 pilot with the best key-separation per phase1b, tempo came out
117.5 = the documented 2× octave). Foggy (17 min) and Airegin (transposed +2, no
honest alignment) were skipped for this cheap gate; Let It Be supplies the needed
*natural slip* instead, and it is a stronger slip reference (a **documented** #22
failure, not an SNR artifact).

**Signal 1 recipe used** (per `mission_6_elastic_matching_design.md`): split the
aligned result into contiguous section instances (A,B,C,A,B,C,…); for each
instance collect the *inferred* chords whose midpoint falls in its time span;
fingerprint = L2-normalized mean `[root one-hot(12) | quality one-hot]`;
`within` = mean cosine over same-label instance pairs, `cross` = over diff-label
pairs, `repeat_consistency = within − cross`.

## Results

| Pilot | Type | within | cross | **Δ (clean)** | verdict | expected | Match? |
|---|---|---|---|---|---|---|---|
| Autumn Leaves | aligned | 0.8273 | 0.7758 | **+0.0515** | OK | aligned | ✓ |
| Let It Be | natural slip | 0.7597 | 0.7691 | **−0.0094** | SLIPPED | slipped | ✓ |
| Ghost of a Chance | aligned | 0.7066 | 0.6452 | **+0.0614** | OK | aligned | ✓ |

**Natural-case separation: 3/3 correct** at the proposed Δ=0.05 threshold. Aligned
tunes land at +0.052 / +0.061; the documented slip lands at −0.009. Separation
≈ 0.06 cosine — right in the design's estimated 0.05–0.10 band.

### Injected localized slip (failure #3) + localization

For each song I also injected a controlled 1-section slip (overwrite one repeated
A-instance's content with a different section's content) and recomputed:

| Pilot | Δ clean | Δ after localized slip | global drop | victim z-score (per-instance) |
|---|---|---|---|---|
| Autumn Leaves | +0.0515 | +0.0482 | +0.0033 | **−3.47** (clean localize) |
| Let It Be | −0.0094 | −0.0100 | +0.0005 | −0.30 (not detected) |
| Ghost of a Chance | +0.0614 | +0.0303 | +0.0311 | +0.41 (not detected) |

## Two findings that change the build plan

**1. The *global* within−cross Δ is nearly blind to a single localized slip.**
Autumn Leaves has 8 A-instances; corrupting one moves the within-mean by only
+0.0033 (7 of 28 A–A pairs touch the victim — √N dilution). The global Δ is a fine
**aggregate coherence gate** (it caught the natural Let It Be slip, which is
*global*), but it is the wrong statistic for the *localized* single-slip that the
design's `suspect_sections` output promises. → **Localization must use a
per-instance outlier score, not the global Δ.**

**2. The per-instance outlier score works — but only when the swapped sections are
harmonically distinct.**
- Autumn Leaves (A/B/C genuinely distinct): the injected victim drops to
  **z = −3.47**, a clean 3.5σ outlier — localization is sharp.
- Ghost (A≈B: within 0.707 vs cross 0.645, only 0.06 apart): swapping A→B content
  barely changes the victim (z = +0.41). Low bridge-contrast tune ⇒ a slip is
  *undetectable in principle*, exactly the design's abstain case.
- Let It Be (already-internally-inconsistent A's from a real slip): z = −0.30; the
  corruption is lost in the tune's own high within-A variance.

## Decision

**CONDITIONAL PASS — proceed to the 20-song harness, with two required changes:**

1. **Score localization with the per-instance sibling-mean z-score**, not the
   global Δ. Keep global Δ only as the coarse OK/SLIPPED aggregate gate. (Directly
   fixes `suspect_sections`, which as-designed would miss single slips in
   many-repeat forms.)
2. **Gate on bridge-contrast:** when `within − cross` is near zero *because the
   sections are genuinely similar* (Ghost), Signal 1 cannot see a slip — return
   `UNVERIFIABLE`, don't force a verdict. Add `cross`/`within` spread as the
   abstain trigger.

Also flag for the harness: the aligned floor (+0.05) sits **right on** the
proposed 0.05 threshold and n=1 natural slip — the threshold is not yet
trustworthy. The 20-song set must inject **localized** slips (single instance,
verified donor≠victim harmonically) and calibrate the threshold on the per-instance
z-score, targeting the Part-2 stopping criterion (≥80% slip-recall @ ≤10% FP).

**Not falsified** (the premise "repeat consistency is a real signal on real audio"
holds), but **not the clean pass the skeleton assumed** either: the load-bearing
statistic for the mission's headline feature (localize the slipped section) is the
per-instance z-score, and it is contrast-limited. Ship the display-only banner on
the *global* Δ first (it is trustworthy as an aggregate), and gate the
training-filter on the per-instance localizer only after the 20-song calibration.
