# Design: multi-hypothesis structure detection + trigram/timing priors

Produced 2026-07-04 by a dedicated design pass (Plan agent), in response to
the hypothesis: "if bigram/key context is unresolved, it's because we're
not looking at it the right way — trigrams with timing expectations,
combined with a properly validated notion of song structure, should let
repeated passages more robustly identify chords." This is a design
document, not an implementation — nothing here is wired into the pipeline.
See `docs/chord_change_signal_analysis/SUMMARY.md` for the investigation
that prompted it, and `docs/known_issues.md` #1 for the full A/B/C
prior-attempt history this design deliberately builds on rather than
repeats.

## 0. Non-negotiable constraints from prior failures

Three "reshape how evidence is used" attempts (A: emission preprocessing,
B: explicit duration decoding, C: periodicity/structure folding) were each
fully implemented, tested, and rejected (`docs/known_issues.md` #1). Three
lessons must constrain this design:

1. **Candidate C's failure was specific**: high SSM chroma self-similarity
   at a lag is evidence of accompaniment/rhythm repetition, not proof of
   harmonic identity at matching slots. Song 001 — the *cleanest* detected
   periodicity (score 0.82) — regressed the *most* when repeats were
   averaged blindly (`majmin` 32.7%→15.3%). Any new structure mechanism
   must independently verify harmonic identity before pooling evidence
   across repeats, not infer it from the same chroma-SSM signal that
   already failed to guarantee it once.
2. **The periodicity phase-offset bug (found/fixed 2026-07-04, this
   session) proves the SSM cannot resolve phase on its own** — its score
   is invariant to phase by construction; only the downbeat grid broke
   that symmetry. This generalizes: the SSM alone also can't pick between
   several equally self-similar structural hypotheses (AABA vs.
   verse-chorus vs. strophic can all produce similar-looking SSM band
   patterns) — hypothesis *selection* needs an external validation signal,
   the same way phase did.
3. **Simpson's-paradox is a live trap for this exact kind of work** — it
   already happened once in this investigation (pre-fix A×E). Any
   validation plan must report per-song breakdowns, not just pooled
   statistics, and be suspicious of "pooled effect driven by 1-2 songs."

## 1. Multi-hypothesis structure detection

**Mechanism.** Reuse `build_ssm()` (already computed, free). Instead of
`score_periods()`'s current top-1 winner, carry its top-3 non-harmonic
period candidates as parallel hypotheses. For each, anchor phase via the
already-fixed `find_loop_phase()`. Slice into candidate sections, cluster
them (reusing `illustrate_form_clustering()`'s existing nearest-centroid
logic) at 2 threshold settings instead of 1 fixed value (0.85), since that
script already showed real threshold-sensitivity across songs 001/002. A
"structure hypothesis" = `(period, phase, threshold, section_labels)` —
carry the cross product (≤6 hypotheses/song) into validation. Bounded,
cheap, reuses existing code — no new algorithm invented.

**Validation — the part that was previously missing.** Compute
**Cross-Repeat Harmonic Agreement (CRHA)**: for each hypothesis, group
sections by assigned label, and at each beat-position-within-section,
check whether all repeats agree on the GT chord label (from
`chord_midi.txt`). Critically: **run this on the full 909-song symbolic
corpus, not the 5 audio songs** — POP909 ships no section labels, but it
does ship exact chord ground truth for all 909 songs, which is precisely
the "did repeats actually share a chord" signal Candidate C's diagnosis
says was missing, computable with zero audio noise. Compare each
hypothesis's real CRHA against a shuffled-label null (same partition,
random group membership) — a hypothesis only counts as trustworthy if it
clears the null's 95th percentile by a real margin, not just "sounds
high." This directly avoids two things that already bit this project: (a)
validating on 5 non-representative songs (song 001 already collapses to a
single trivial "all-A" section, which would make CRHA meaninglessly
perfect for the wrong reason), and (b) trusting a pooled statistic without
per-song breakdown.

**Decision gate:** measure what fraction of the 909-song corpus has any
hypothesis clearing the trustworthiness bar. Small minority (<15-20%) →
the premise is a niche win, scope down before touching audio. Healthy
majority (>50%) → real license to proceed.

## 2. Trigram priors with timing expectations

**Mechanism.** Direct extension of the bigram tables already in
`build_chord_change_features.py::fit_bigram_tables()` — same
canonicalization (mode-agnostic + relative-major pooling), same
Laplace-smoothing pattern (`_conditional_logprob`), just keyed on
`((chord_a, chord_b), chord_c)` triples instead of pairs.

**Sparsity, quantified honestly.** 36 atomic chord-states (12 roman
degrees × 3 quality buckets) → 1,296 possible bigram contexts, 46,656
possible trigram contexts (36×). The corpus has ~106K transitions, so a
*uniform* spread gives ~82 obs/bigram-context but only ~2.3
obs/trigram-context — a real, quantified sparsity cliff. The real
distribution is Zipfian, though (top-10 bigrams already cover ~46% of all
mass, per `docs/architecture_extensions.md` #10), so specific
high-traffic contexts — critically, **ii→V is already the 6th most common
bigram (3.57%) and V→I is 1st (9.58%)**, so `(ii,V)→I` should land among
the best-populated trigram contexts, not a sparse edge case. This is a
concrete, checkable prediction, not a hope.

**Guarding sparsity:** (1) interpolated backoff to the bigram table when a
trigram context has too few observations (deleted-interpolation style,
matching this codebase's existing "simple, validated" smoothing choices,
not a fancier estimator); (2) reuse the existing canonicalization
machinery rather than inventing a third scheme; (3) empirically test
whether collapsing quality further on the *context* chords (keep full
resolution only on the predicted chord) meaningfully improves coverage,
rather than guessing at the right granularity.

**Timing expectation — test before building.** Before any joint model:
check, symbolically, whether trigram context actually modulates duration.
Extend `duration_prior.py::fit_duration_prior()`'s existing global PMF
(peaks at 2 beats, 49.2%) to condition on "is this chord the resolving
chord of a ii-V-_ context," and run a KS test against the global PMF. If
the conditional and unconditional duration distributions are
indistinguishable, the "trigram implies timing" half of the hypothesis is
falsified cheaply, before any HMM change. Only if that test shows a real
effect does a joint hazard-style model (duration prior modulated by
trigram-context identity) become worth building — and even then, as a
targeted multiplicative modulation of the existing prior, not a redesign.

**The connection the user wants, made precise.** A trigram-timing prior
alone doesn't fix the actual documented bottleneck (`known_issues.md`:
"per-beat emission evidence cannot reliably discriminate between chords
that share most of their template" — sus4 vs 7sus4 vs dom7). This is
exactly where validated structure hypotheses (Section 1) earn their keep:
for a slot that has *passed* CRHA validation, pooling audio observations
across its confirmed repeats (the existing `fold_beat_probs`, but now
gated by CRHA passing instead of applied everywhere — the "restricted to
segments where cross-repeat agreement is independently confirmed" idea
`known_issues.md` already floated as Candidate C's untried follow-up)
gives a cleaner observation to feed the trigram-conditioned duration
model. Structure and trigrams are complementary precisely because one
supplies more/cleaner data and the other supplies a sharper prior over
what that data should look like.

## 3. Smallest first experiment (before touching audio or the HMM)

All on `chord_midi.txt`/`beat_midi.txt`, all 909 songs, **zero audio**:

1. Run CRHA validation on the full corpus (period detection can even run
   directly on the GT chord-label sequence's own self-similarity, sidestepping
   any chroma proxy entirely, since GT chords are the ground truth at
   perfect fidelity). Report the distribution of (CRHA − null) across all
   909 songs and the pass-rate.
2. On the passing subset, fit trigram tables restricted to chords inside
   validated repeat-groups vs. outside, and compare sample counts/entropy
   per context between the two pools — directly tests "does using repeats
   reduce trigram sparsity," in its purest audio-free form.
3. **Decision gate:** if either check comes back weak, the whole direction
   is falsified cheaply — before repeating A/B/C's expensive
   implement-test-reject cycle a fourth time. If both show real signal,
   proceed to a narrow, CRHA-gated `fold_beat_probs` experiment on the
   5-song audio pipeline — not a full rebuild.

## 4. Risks

1. CRHA could replicate Candidate C's failure one level removed — coarse
   label agreement could look fine while the specific sus4/7sus4/dom7
   slots that matter most for perceived quality are exactly the
   disagreeing ones. Mitigation: report worst-slot agreement too, not just
   the mean, and check whether disagreement clusters at cadences/turnarounds.
2. Songs that pass CRHA might be systematically simpler/more repetitive —
   any trigram-sparsity win measured there might not transfer to the
   harder, less repetitive songs where help is actually needed most.
3. The mode-agnostic canonicalization has a known, unresolved asymmetry
   (minor-annotated songs borrow from parallel major ~3x more than the
   reverse, `docs/scale_taxonomy_2026-07-03.md` — possibly a
   Krumhansl-Schmuckler labeling artifact, not a real musical asymmetry).
   Trigram tables inherit this silently.
4. POP909 is pop, not jazz — `ii→V→I` coverage should be good, but
   secondary dominants and tritone subs will likely be rare, same caveat
   already flagged for `bVI→bVII→I`-type chromatic motion. The *shape* of
   the user's jazz intuition is right; the *frequency mix* this table
   learns will be pop-flavored.
5. Even a perfectly validated trigram-timing prior doesn't fix the
   underlying emission-discriminability bottleneck A/B/C all converged on
   — the realistic best case is improved root/timing accuracy, not
   necessarily `majmin`/`tetrads`, unless Risk 1 is specifically checked
   and ruled out.
6. Real (non-POP909) audio has no downbeat ground truth yet —
   `find_loop_phase()` needs a downbeat grid, and
   `docs/architecture_extensions.md` #2 already flags real downbeat
   detection (e.g. madmom) as an unresolved prerequisite. The symbolic-only
   first experiment above sidesteps this, but it must be solved before any
   of this reaches real audio.

## Results: the smallest first experiment, run 2026-07-04

Ran the CRHA validation from Section 3, step 1, exactly as designed:
`scripts/run_structure_validation.py`, all 909 songs, zero audio/MIDI
re-parsing (built entirely from `features_symbolic.csv`'s already-computed
`gt_label`/`A_beat_phase` columns). For each song, tested candidate periods
{4, 8, 16, 32, 64} beats, anchored phase to the first downbeat (same logic
as `find_loop_phase`), grouped beats by position-in-loop, and scored
size-weighted majority-label agreement against a 200-shuffle null. Runtime:
3m19s for the full corpus.

**Result: 213/909 songs (23.4%) have a structure hypothesis whose CRHA
clears the null's 95th percentile by more than 0.15** (`plots/crha_structure_validation.png`,
`structure_validation_results.csv`). Per the decision rule proposed above
(<15-20% → niche win, scope down; >50% → real license to proceed),
**23.4% lands just past the "niche win" boundary — real, but a minority
phenomenon, not a corpus-wide property.** This is a clean, decisive answer,
not an ambiguous one: naive block-repetition structure is a genuine signal
for roughly a quarter of pop songs (the more repetitive ones) and not
worth relying on for the rest.

Two findings sharpen this further:

1. **The margin distribution is heavily right-skewed** (median 0.051, mean
   0.091, max 0.595) — most songs sit close to the null, and the
   "trustworthy" quarter pulls clearly away from the pack rather than the
   whole corpus shifting gradually. This looks like two populations, not
   one continuum — consistent with "some songs are built from cleanly
   repeating blocks, most aren't," rather than "structure helps a little
   everywhere."
2. **Winning period length matters a lot.** Among all songs, short periods
   (4, 8 beats — one or two bars) win almost as often as long ones. But
   among *trustworthy* songs specifically, long periods (16, 32 beats — 4-
   and 8-bar phrases) dominate (84 and 91 songs respectively, vs. only 23
   at period 8 and essentially none at period 4). This is exactly the
   Candidate C failure mode showing up again, quantitatively this time:
   short-period "repetition" is disproportionately accompaniment-pattern
   coincidence (a 1-2 bar bass/comping riff repeating while the harmony
   underneath doesn't), not real chord-level structure — matching the
   original diagnosis that high self-similarity at short lags is weak
   evidence of harmonic identity.

**Decision, per the design's own gate:** this doesn't kill the direction,
but it does scope it down before any audio work. A CRHA-gated
`fold_beat_probs` experiment is worth trying, but only (a) restricted to
period candidates of 16+ beats, given the period-length finding above, and
(b) evaluated with the expectation that it should help roughly a quarter
of songs, not all five of the audio pipeline's existing test songs.

**Checked directly where songs 001-005 land, and it overturns my first
guess in an important way.** I initially expected song 001 — the one with
the strongest detected audio periodicity (0.82) and the one Candidate C
regressed the *most* (`majmin` 32.7%→15.3%) — to be exactly the case CRHA
should flag as untrustworthy. It's the opposite:

| song | best period | CRHA | null-95 | margin | trustworthy? |
|---|---|---|---|---|---|
| 001 | 16 | 0.896 | 0.358 | **0.538** (highest of all 5) | **Yes** |
| 002 | 4 | 0.361 | 0.269 | 0.092 | No |
| 003 | 16 | 0.606 | 0.355 | 0.252 | Yes |
| 004 | 8 | 0.219 | 0.174 | 0.044 | No |
| 005 | 8 | 0.191 | 0.194 | -0.003 | No |

Song 001 has the *strongest* CRHA margin of all 5 songs (0.538, near the
top of the entire 909-song distribution) — its ground-truth chords really
do repeat at ~90% agreement against a ~36% null, and this holds even at
`period=32` specifically (CRHA=0.896, margin=0.490), the exact period
Candidate C actually used for this song. **This means Candidate C's
failure on song 001 was not a false-positive structure hypothesis** — the
harmonic repetition genuinely exists, CRHA correctly says so, and folding
still hurt anyway. That's a more important and more precise finding than
my first hypothesis: it shows CRHA validates harmonic *label* repetition
correctly, but label repetition is not sufficient to guarantee that
*averaging raw audio evidence* across those repeats is safe — a real
repeat can still carry different voicings, passing tones, or ornamentation
each time even when the underlying chord is genuinely the same, and
that's exactly the surface-level variation that fine quality discrimination
(`majmin`/`tetrads`) is sensitive to. CRHA answers "does the structure
exist" correctly; it does not by itself answer "is it safe to fold audio
evidence across it," which is a real, narrower question Risk 1 above
gestured at but this result pins down concretely. Any future CRHA-gated
folding experiment needs a second, audio-level check (e.g. does folding
actually reduce variance in the *emission* evidence at these slots, not
just agreement in the *label*) before trusting a high CRHA score to mean
"safe to average."

## Critical files

- `harmonia/models/periodicity.py` — `score_periods`/`find_loop_phase`/
  `fold_beat_probs`; extend `score_periods` to expose the multi-hypothesis
  grid, add CRHA-gating to `fold_beat_probs` call sites.
- `harmonia/models/structure.py` — `build_ssm`/`_beat_chroma`/`Segmenter`;
  source of the section-chroma aggregation reused for clustering.
- `scripts/build_chord_change_features.py` — `fit_bigram_tables`,
  `_conditional_logprob`, `identify_best_parent_scale`; direct extension
  point for trigram fitting.
- `harmonia/theory/duration_prior.py` — `fit_duration_prior`; extension
  point for trigram-context-conditioned duration PMFs.
- `scripts/plot_structure_proposal_illustrations.py` —
  `illustrate_form_clustering`; existing section-clustering prototype to
  extend, not reimplement.
