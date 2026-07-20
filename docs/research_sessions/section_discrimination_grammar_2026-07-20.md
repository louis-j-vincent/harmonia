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

## Checkpoint 5 — veto + energy arbitration WIRED (importable module) + real-audio gate
Built `harmonia/models/section_arbiter.py` (importable, deterministic, no audio deps —
energy passed in as a per-block scalar): single-linkage on harmony pos-agree≥0.6 +
distinctive-chord veto + energy arbitration (override veto when energy SIMILAR = same
section varied; block a harmony-merge when energy STRONGLY differs = diff section, harmony
silent). NOT wired into chart_model.py (still concurrent WIP) — ready to plug in.

Gate (letter-level GT methodology, per waiver): 6 matched songs in pop400, audio→GT aligned
by chord-DTW, GT letters transferred to audio 8-bar blocks, over/under-clustering measured.
| variant | over-merge | under-split | vs harmony |
|---|---|---|---|
| harmony (shipped) | 71.8% | 16.4% | — |
| +veto | 55.3% | **38.6%** | veto UNUSABLE alone (+22pp under-split from decode noise) |
| +veto+energy (e_diff=1.2) | 59.3% | 23.7% | −12.5pp over / +7.3pp under |
| +veto+energy (e_diff=0.8) | 52.0% | 28.5% | −19.8pp over / +12.1pp under |

**What's validated:** the arbitration DESIGN works mechanically — energy RESCUES the veto's
+22pp under-split blowup (down to +7pp) while keeping most of the over-merge gain. On real
audio the veto alone is unusable (decode noise mints spurious distinctive chords → false
splits); energy is what makes it viable. Energy helped the hard case per-song where it
exists (SWBL 89→60%, Stand By Me 18→9%).

**What's NOT a win (honest):** there is NO configuration that improves BOTH rates — it's a
trade-off FRONTIER (every over-merge reduction costs under-split). Residual over-merge stays
high (52-59%) because the matched set is dominated by harmonically-repetitive HARD cases
(Let It Be, aretha = same chords AND not energy-distinct → stay 100% over-merge, neither
harmony nor energy separates). This is the fundamental chord+energy ceiling — the songs
where verse/chorus share chords AND similar energy genuinely need melody/lyrics/timbre
(which the project's constraints + failed multi-factor experiment rule out).

**Caveats:** absolute numbers inflated by (a) DTW-alignment label noise, (b) small set (6
songs) dominated by repetitive hard cases, (c) pipeline collapsing some songs to 1 cluster.
The RELATIVE comparison (same alignment across variants) is the trustworthy signal:
energy rescues the veto.

**Recommendation (for the user to steer):** the arbiter is ready to wire once chart_model.py
is free. But since it's a trade-off not a strict win, the integration decision needs the
user's error-preference: is verse/chorus OVER-merge (losing the B section, the SWBL/This Love
complaint) worse than occasional UNDER-split (a section shown as two)? The evidence says
over-merge is the user's stated pain → favour the veto+energy@0.8-1.0 operating point, but
gate live on the matched-set FORMS (not just pair-rates) before shipping.

## Checkpoint 6 — arbiter default locked to user's error-preference + robustness fix
User confirmed (2026-07-21): "je préfère l'erreur 2 à l'erreur 1 → privilégie plus de
sections que moins" (UNDER-split preferred over OVER-merge; bias to NOT merge under
ambiguity). Locked `section_arbiter.cluster` defaults: use_veto=True, use_energy=True,
**e_diff=0.8, e_same=0.4** (real-audio gate: 52% over / 28.5% under — trades over→under vs
harmony 71.8%/16.4%, i.e. more sections). e_same=0.4 keeps the veto's split standing when
energy is AMBIGUOUS (0.4≤|dz|≤0.8) = don't-merge-under-ambiguity.
Robustness fix (found via synthetic test): energy uses a per-song Z-SCORE gated by (a)
nb≥4 blocks and (b) CoV(energy)≥0.10 — a flat vamp has no real dynamics, so its z-score
would amplify noise into false splits; there energy is untrusted → harmony(+veto) only.
(A relative-to-median measure was tried and REJECTED: it amplifies normal 10-15% within-
section dynamics → 88-95% under-split. z-score-vs-song-dynamics is the right notion.)

## Checkpoint 7 — DIVISIVE-step premise CONFIRMED (symbolic corpus)
Current `_sections_by_largest_unit` (chart_model.py:443) picks ONE grain L∈{16,8} for the
WHOLE song (first clearing rec_min=0.55, uniform blocks from bar 0 — never tries 4, never
adapts within-song). Confirmed in code. Premise-check `scratchpad/divisive_premise.py`:
| | pop400 (345, n≥16) | jazz1460 (1385) |
|---|---|---|
| single-L picks a grain (16/8) | (most) | 34% (else abstains → changepoint) |
| **songs MIXING phrase scales (GT sections span ≥2 of 4/8/16)** | **92%** | 54% |
| **uniform-L blocks straddling a GT section boundary** | **60.5%** | 48.1% |
| songs with ≥1 straddling block | 94% | 76% |
| straddles at the L/2 midpoint (a clean 16→8/8→4 split fixes) | 46% | 97% |

**STRONGLY CONFIRMED, not rare**: 92% of pop songs mix phrase scales — a single global L
structurally cannot fit them — and 60% of its uniform blocks straddle a real GT boundary.
The divisive step is well-motivated. Design note: jazz straddles sit at the midpoint 97%
(clean AABA subdivision → "split in half" works), but pop only 46% → the pop split point
must be the DETECTED content-change (max internal heterogeneity), not just L/2. This shapes
v1: recursive split at the block's strongest internal boundary when its two parts are
genuinely different material, down to a 4-bar floor, then agglomerate with the veto+energy
arbiter at whatever local scale survives.

## Checkpoint 8 — DIAGNOSTIC: boundary alignment is the dominant lever (8-base validated)
Built 8-base divisive+agglomerative hierarchy (`scratchpad/hierarchical.py`, arbiter reused).
Bar-pair over/under vs GT, pop400:
| variant | over-merge | under-split |
|---|---|---|
| baseline (single-L 16/8, shipped) | 90.5% | 8.2% |
| uniform-8 (no divisive) | 29.4% | 52.0% |
| hier v2 (8-base + divisive, match-others split) | 31.2% | 49.3% |
| **oracle boundaries (GT-perfect segments)** | **25.8%** | **4.9%** |
(jazz1460: baseline 91.9%/4.9%, uniform-8 9.5%/39.3%, oracle 9.0%/3.2%.)

**Three findings:**
1. **"8 as base" (user 2026-07-21) is STRONGLY validated**: switching the block scale 16→8
   drops over-merge 90.5%→29% on pop. The shipped single-L (16-first) is the main cause of
   the collapse-to-one-letter — 16-bar blocks are so long that every block matches every
   other in repetitive pop → all merge. 8-bar blocks are distinguishable.
2. **The divisive step as built adds ~nothing** (hier 31/49 ≈ uniform-8 29/52). Splitting an
   8-block at its midpoint / best-internal-point does NOT recover the real section boundaries.
3. **Under-split (52%) is almost ENTIRELY grid straddling**: oracle-perfect boundaries drop
   it 52%→5% with the SAME clustering. So the dominant missing lever is BOUNDARY DETECTION
   (where do sections actually start), not scale, not the clustering, not the divisive-as-
   built. The residual over-merge floor (26% even with oracle) is the harmony-only ceiling
   (hard same-vocab pairs) — the energy confirmer's target.

**Reframed architecture:** the win is (a) 8-base [SHIP-worthy on its own — big over-merge
drop] + (b) a real BOUNDARY detector so blocks align to section starts [kills the 52% under-
split] + (c) energy arbiter for the hard-case over-merge floor. The "divisive by power-of-2
midpoint" idea is REJECTED by this measurement — replace with content-boundary detection.

## Checkpoint 9 — boundary detection is hard + LLM-global-read premise-check (side mission)
**Content-boundary detection from chords alone is HARD.** Foote-style novelty over the root
sequence: pop400 boundary-F1 **0.38** (jazz 0.32), over/under 32%/53% — does NOT close the
gap to oracle (26/5). Boundary detection is the known-hard part (audio-MIR ceiling ~0.5-0.6);
symbolic-chord-only is weaker.

**LLM-global-read premise-check (coordinator side mission, cheap, no training):** presented 4
pop400 raw chord sequences (bar-indexed, NO GT shown) to the LLM (me), asked for section
boundaries+groupings, scored vs iReal GT. Boundary-F1: Golden Lady 0.62, Blue Tango 0.43,
Lately **0.00**, Save The Last Dance **0.00** (mean ~0.26 — NOT better than the local novelty
0.38). **Three confounds, reported honestly:**
1. All 4 sampled songs are FAMOUS standards → memorization confound (can't rule out recall).
2. **iReal GT is COARSE/IDIOSYNCRATIC**: "Lately" GT boundaries=[4,12] (it labels ~the whole
   61-bar body as ONE section); "Save The Last Dance"=[2,38,46,64]. The LLM's F1=0 there is
   DISAGREEMENT WITH IREAL'S LABELING SCHEME, not a musical error — the LLM proposed a
   reasonable verse/chorus form; iReal just doesn't label that finely. **This caveat applies
   to EVERY over/under-vs-iReal-GT number in this study** — the target itself has label noise.
3. The LLM's read is repetition-based → largely REDUNDANT with the local block model.
**Verdict: NEGATIVE/inconclusive** — no evidence the LLM global read beats or complements the
local hierarchical+arbiter signal on boundary detection, and it is heavily confounded. Not
worth wiring as an arbitration vote on this evidence (consistent with the project's
"priors/LLM dead-to-negative + evaluation-circularity" history, known_issues Mission 5).

## SYNTHESIS (what to build, quantified)
1. **SHIP: 8-bar base scale** (replace cands=(16,8) → base 8, 16 via merge). Over-merge
   90.5%→29% on pop, matching the user's more-sections preference. The single biggest,
   cleanest win; low-risk (scale change). Blocked only by chart_model.py concurrent WIP.
2. **Energy arbiter** for the 26% over-merge floor (hard same-vocab cases) — validated
   (SWBL d=0.88); already an importable module with the user's locked operating point.
3. **Boundary detection** is the dominant remaining lever (oracle under-split 52%→5%) but is
   genuinely HARD (novelty-F1 0.38, LLM inconclusive). This is the honest open frontier —
   NOT solved this session; the divisive-at-power-of-2 idea was measured and REJECTED.
4. **Caveat on the whole metric**: iReal letter-GT is coarse/idiosyncratic → the absolute
   over/under numbers carry label noise; relative comparisons across variants are the signal.

## DESIGN DISTINCTION (preserve explicitly — orchestrator verified 2026-07-21)
The ~5 documented prior failures (progression_prior.py bigram/trigram λ→0, #21/#27; diatonic;
encoder-fusion; density-ratio; key-local) were all DECODE-TIME injected priors — a corpus
statistic used as a CORRECTING FORCE fighting the acoustic emissions during chord decoding.
That pattern is dead. It was NOT a data-quality problem: those priors trained on the SAME
clean `ireal_corpus` used here, and they were FLAT chord-to-chord bigrams that never touched
section labels / % / repeat structure at all.

EVERYTHING in this session is a different, UNTESTED class: **post-hoc STRUCTURAL VOTES over
the finished chord sequence** (segmentation grammar, hierarchical entropy segmentation,
distinctive-chord veto, energy confirmer, LLM-global-read test) — NONE injects a prior into
DECODING. This is the same shape as the veto+energy that DID work. Learning from the FULL rich
notation (sections, %, repeats) as an arbitrated structural vote is the genuinely new,
promising direction — must NOT be conflated with the dead decode-time-override pattern.
(Parser note: D.C./D.S./Coda markers are stripped, not resolved to performance order — fine
for section-CONTENT labeling, would matter only for true linear performance order.)
