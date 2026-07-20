# Section discrimination under harmonic ambiguity — research narrative (2026-07-20)

Central remaining problem: verse↔chorus with (near-)identical chords get over-merged.
Corpus study (§H expert_procedure) found 21-25% of pop distinct-section PAIRS share
identical/near-identical chord-root vocab. Full autonomy, iterate (v1, v2, exceptions).

## Literature (checked first, avoid dead ends)
- The harmonically-identical verse/chorus problem is a KNOWN hard case; the standard
  answer (McFee & Ellis 2014, spectral clustering; TISMIR survey) is to combine
  repetition with TIMBRE/instrumentation — i.e. NON-chord evidence. Our constraint is
  chord-only (timbre decode is noisy; the project's multi-factor acoustic classifier
  already LOST to phrase-position). So: squeeze max discrimination from the chord
  SEQUENCE first, accept a ceiling, and only then consider a cheap acoustic confirmer.
- pitchclass2vec (2023) does symbolic structure segmentation with chord embeddings —
  a learned sequence representation; noted, heavier than warranted for v1.

## Checkpoint 1 — H1 premise-check (symbolic, GT-clean pop400+jazz1460)
Question: does order-aware sequence similarity separate the GT-ambiguous same-vocab
section pairs that the CURRENT metric conflates?

Current merge metric = POSITION-WISE agreement (`_sim`, pos-agree>=0.6, phase-tolerant).
Measured similarity distributions (pop400):
| pair type | pos-agree | norm-LCS |
|---|---|---|
| SAME label (want merge) | mean .92 / **median 1.00** | mean .78 / median 1.00 |
| DIFF label, vocab-Jac>=.8 (ambiguous, want split) | mean .78 / **median 1.00** | mean .63 / median .57 |
| DIFF label, all | — | mean .40 |

**Finding: position-agreement is the CULPRIT — ambiguous verse/chorus align at nearly
every position (median 1.00), indistinguishable from same-section.** norm-LCS separates
a little better (.63 vs .78) but no clean global threshold (LCS>=.6 keeps 70% of true
merges but still merges 49% of ambiguous pairs). Chord-SEQUENCE-only has a genuine
ceiling on these pairs — consistent with the literature (needs timbre/melody).

Note: She Will Be Loved (the live case) verse{Bb,C}/chorus{Bb,C,Eb,Ab} is vocab-Jac
**0.5** (superset), NOT in the >=.8 bucket — it over-merged purely via position-agreement
(verse [Bb,C,Bb,C..] vs chorus [Bb,C,Ab,Eb,Bb,C,Bb,Eb] agree 5/8=.63>=.6). The
discriminating signal is CONCENTRATED (the chorus's recurring Eb/Ab), diluted by a
global metric.

## Checkpoint 2 — distinctive-chord veto (v1 candidate)
Idea: block them from merging if one block has a root that RECURS (>=2 bars) but is
ABSENT from the other — catches the concentrated signal a global metric dilutes.
Corpus (pos-agree>=0.6 merge, + veto):
| corpus | SAME merge (want hi) | DIFF merge = over-merge (want lo) | veto effect |
|---|---|---|---|
| pop400 | 92.5% → 83.4% | 30.1% → 21.0% | removes 30% of over-merges, costs 9.1% false-sep |
| jazz1460 | 92.8% → 86.0% | 9.5% → 6.6% | removes 30%, costs 6.8% false-sep |
Net positive but NOT free — 9% of true same-section pairs get falsely split (an
instance with a one-off recurring chord the other lacks). Needs a confidence gate
before it can go on noisy real audio without fragmenting a clean A×15.

## Checkpoint 3 — THE NUMBER: over/under-clustering vs iReal GT (harmony-only baseline)
User asked "à quel point es-tu bon pour séparer A de B sans sur/sous-clusterer" — a real,
defensible number. Measured the SHIPPED clustering-DECISION logic (committed largest-unit
`_sim`: position-wise agreement, phase-tolerant ±1 lag, merge iff ≥0.6) on GT section pairs
(true iReal boundaries — isolates the A-vs-B DECISION from the block-grid/boundary problem).
`scratchpad/decision_eval.py` + `variant_eval.py`.

**Harmony-only baseline (this is the number):**
| | pop400 | jazz1460 |
|---|---|---|
| OVER-merge, EASY (diff harmony, vocab Jac<0.5) | **10.1%** (n=811) | 1.7% (n=1141) |
| OVER-merge, MID (0.5≤Jac<0.8) | 24.6% | 10.5% |
| OVER-merge, HARD (same vocab, Jac≥0.8) | 76.4% | 43.3% |
| UNDER-split (GT same section) | 6.8% (n=864) | 6.6% (n=1009) |

Honest read: on genuinely-different-harmony sections the pipeline is decent on jazz (1.7%
false-merge, jazz aligns to 8/16) but **NOT near-zero on pop (10.1%)** — the top-priority
gap. On same-harmony (HARD) the harmony-only metric fundamentally cannot separate (pop 76%)
— that needs the non-harmonic signals (rhythm/melody/vocal), per the user's own reasoning.

IMPORTANT CAVEAT: this isolates the DECISION metric, not the full audio pipeline. On
synthetic full-pipeline runs `to_chart_model` collapses via its changepoint/Jaccard
FALLBACK (87% of pop → one letter) — but that fallback is NOT what runs on real audio (real
sections come UPSTREAM from `_infer_nnls24` flux-barlocked). So the decision-metric number
is the fair measure of the clustering logic; the real-audio full-pipeline number is separate
(matched set, TODO).

**The two threads feed each other (verified):** the distinctive-chord veto (built for the
HARD case — "don't merge if one block has a recurring root the other lacks", min_recur=2,
min_frac=0.2) also fixes the EASY case:
| pop400 variant | EASY over-merge | under-split |
|---|---|---|
| base (shipped) | 10.1% | 6.8% |
| phase-tol OFF | 8.5% | 7.5% |
| **base + veto** | **3.7%** | 9.8% |
| phaseoff + veto | 3.5% | 10.4% |
The veto cuts EASY over-merge 10.1%→3.7% (−63%) — the current pos-agree metric weights all
positions equally and ignores the DISTINCTIVE chord; the veto targets exactly it. Cost:
+3pp under-split (same-section instances that vary trigger a false veto). Phase-tolerance
contributes ~1.6pp of the easy over-merges (spurious lag matches) — a smaller, separable lever.

**Open trade-off → motivates the non-harmonic signal:** the veto's under-split cost is
harmony-only collateral (can't tell "real section variation" from "different section" by
chords alone). A rhythmic/vocal-activity CONFIRMER (H3/H4) is what resolves BOTH the veto's
under-split cost AND the 76% hard-case — the same signal, per the user's causal point.

## Checkpoint 4 — non-harmonic confirmer feasibility (REAL audio, matched set)
Built cheap acoustic features (librosa, zero new dep): RMS energy, HPSS vocal-band
(250-2500 Hz) harmonic energy + its fraction, percussive onset-density.
`scratchpad/acoustic_confirmer.py` + `energy_ortho.py`. Artifact
`docs/plots/swbl_energy_confirmer_2026_07_20.png`.

**Q(a) hard case — does a non-harmonic signal separate verse/chorus where harmony is
silent?** YES for the clearest live case. She Will Be Loved (verse Cm-Bb vs chorus with
Eb — the M2 case): per-span separation (Cohen's d, chorus vs verse):
| feature | d |
|---|---|
| RMS energy | **+0.88** |
| vocal-band energy | +0.85 |
| onset-density | −0.65 |
| vocal-band FRACTION | −0.58 |
The chorus is LOUDER + has more vocal-band energy — a large effect. **But the useful
signal is LOUDNESS (RMS), NOT vocal-specific**: vocal-band FRACTION (normalised by total
harmonic) goes the WRONG way; vocal-band ENERGY just tracks RMS. So H4 (vocal-activity
proxy) gives NO advantage over H2 (energy/RMS) here — a clean negative on the more novel
idea, a clean positive on the cheaper one. This Love is NULL (all features |d|≤0.22) — not
every song has acoustically-distinct sections.

**Orthogonality across the matched set** (8-bar blocks, energy CoV WITHIN
harmonically-identical block groups — high = energy separates what harmony conflates):
SWBL 0.31, This Love 0.25, Let It Be 0.41, abba 0.44, aretha 0.16 (>0.15 = orthogonal
signal present); Billie Jean 0.14, Commodores 0.11, henny 0.09, Stand By Me 0.07 (low).
**5/9 songs carry section energy-structure orthogonal to harmony** — a real, common signal.

**Double-edged (the honest caveat → confirms H3):** energy also varies WITHIN a true
section (buildups/dynamics — the SWBL artifact shows a broad rise, not a clean verse/chorus
square wave). So energy is a CONFIRMER, arbitrated with harmony+repetition, NEVER a primary
boundary detector — exactly the user's D8 answer. A naive energy threshold would both fix
verse/chorus AND fragment a dynamic single section.

## Proposed v1 arbitration (design; integration DEFERRED — chart_model.py has concurrent WIP)
Calibrated-evidence arbitration (Occam-post-pass philosophy, not a blind override):
1. HARD case (vocab Jaccard≥0.8, harmony silent): if two candidate sections' mean energy
   differs strongly (|d|≥~0.8, calibrated), SPLIT — energy supplies the discrimination.
2. Veto UNDER-split reduction: when the distinctive-chord veto proposes a split, CONFIRM
   only if energy ALSO differs; if energy is similar it's likely a same-section passing
   chord → don't split (recovers the veto's +3pp under-split cost).
Both feed the SAME arbiter, per the user's causal point. **Cannot be GATED yet** — the
over/under-cluster metric needs GT section BOUNDARIES on real audio (the matched set has
GT letters/forms but not per-bar section times; iReal↔audio bar alignment is the known-hard
duration-match problem). Next dependency: a small hand-annotated verse/chorus boundary set
(or SALAMI-style GT) to gate the arbitration honestly. Feasibility is POSITIVE; the gate
infrastructure is the blocker, not the signal.
