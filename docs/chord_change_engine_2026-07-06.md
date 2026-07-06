# Chord-change engine — deep investigation, 2026-07-06

Building the "scaffold → coarse grid → fill → zoom" method for chord-change
detection. Scaffold = GT `section_per_bar` + exact MMA beat grid (structure
detection is a separable, parked problem). Scripts:
`harmonic_rhythm_probe.py`, `period_estimation.py`, `chord_change_engine.py`.

## 1. Foundation validated — merging is the load-bearing lever

`harmonic_rhythm_probe.py`, 12 songs. Merge beats into g-beat blocks, measure
ROC-AUC of adjacent-block chroma+bass distance for "true change" vs "hold":

| grid | AUC | note |
|------|-----|------|
| g=1 (per-beat) | 0.643 | weak — this is the BP noise that tanked per-beat root to 0.20 |
| **g=2** | **0.962** | chroma+bass cleanly separates change from hold |
| g=4 | 0.903 | strong, but starts merging real changes |

→ The same-or-different decision works at the **block** level, not the beat level.

## 2. Per-section "period" premise FALSIFIED (corpus-wide, 1136 songs)

The plan was to estimate a per-section change period ∈ {1,2,4} and merge at it.
Corpus symbolic check (cheap, rule #5) killed it:

- Changes land on **every** beat of the 4/4 bar: beat 1/2/3/4 = 38.6 / 28.7 /
  20.1 / 12.5 %. Not a downbeats-only or half-bar grid.
- A section's changes explained by its best 2-beat grid: only **61%**; best
  4-beat grid: 45%.
- Sections with a *clean* period (≥80% of changes on one phase): period-1 **92%**,
  period-2 5%, period-4 3%.
- Within-section change spacing 1/2/3/4 beats = 19/25/24/18 % — nearly uniform.

→ There is no clean per-section period to estimate; harmonic rhythm is irregular
with a 2-beat mode. Estimating a period would be fitting noise. `period_estimation.py`
confirmed on audio (est accuracy 27%, GT period = 1 for 44/45 sections). **Decision:
drop the period estimator; use a fixed 2-beat coarse grid** (best single grid) and
push the residual to the zoom.

## 3. Coarse engine — fixed 2-beat merge + same-or-different (GT-structure scaffold)

`chord_change_engine.py`, 15 clean songs, θ=0.15 (block cosine-distance cut,
forced boundary at section changes):

| metric | value |
|--------|-------|
| change-detection F (±1 beat) | **0.89** (P 0.91, R 0.88) |
| change-detection F (exact beat) | **0.50** |
| MIREX root | 60.5% |
| MIREX majmin | 39.5% |
| detected/GT segment ratio | 0.91 |

The ±1 vs exact gap (0.89 → 0.50) is the zoom's headroom: half the changes are
placed one beat off because the real change sits on the odd beat *between* two
2-beat blocks. Note root (60.5%) is now the limiter, not segmentation — labeling
on real evidence (parked task #3) is the next bottleneck once boundaries are good.

## 4. Every zoom strategy FAILS; segmentation is not the bottleneck — LABELING is

Four zooms tried to beat the coarse chgF 0.89 / exact 0.50; all failed:

| zoom | chgF | exact | root | why it failed |
|------|------|-------|------|---------------|
| coarse only (baseline) | 0.89 | 0.50 | 60.5% | — |
| naive beat novelty (snap+split) | 0.86 | 0.41 | 50.1% | reintroduces the noisy g=1 signal |
| per-track (bass/chord stems) | — | — | — | premise falsified: walking bass flips 58% of beats; no per-track beat cue beats mixed (all AUC ~0.6–0.67, `pertrack_zoom_probe.py`) |
| divisive top-down pooled split | 0.36 | 0.27 | 39.9% | first split of a many-chord section has two muddy multi-chord halves → under-segments |
| pooled-halves boundary snap ±1 | 0.87 | 0.41 | 56.9% | max-contrast position ≠ true change beat (BP onset smear) |

Exact-beat placement (~0.50) is a **hard ceiling**: the beat-level change signal is
~0.65 AUC on *every* track (mixed or isolated), and the pooling that gives clean
SNR (0.962) destroys the resolution needed to localize a change to one beat.

**Oracle-boundary diagnostic (the decider):** feed GT change beats as boundaries
(chgF=1.00) and label with the same models → root **55.2%**, majmin **36.6%** — no
better than the coarse engine. So perfect segmentation does NOT raise accuracy.
The gap is entirely in LABELING on real evidence: bass-argmax root is wrecked by
walking bass, and the family emission model tops out ~37% majmin. **Conclusion: the
chord-change/segmentation problem is at its useful ceiling (coarse chgF 0.89); the
priority is now labeling — parked task #3.**

## 5. Labeling fix — trained root model (the actual lever)

`root_model_experiment.py`. On ORACLE segments (segmentation removed), root
estimators vs GT root:

| estimator | root acc |
|-----------|----------|
| onset_argmax | 63–68% |
| **bass_argmax (pipeline today)** | **67–68%** |
| template match (root×family) | 66–74% |
| **trained_LR (48d absolute chroma, 5-fold by song)** | **85.9% (n=25) → 93.4% (n=60)** |

Bass-argmax is defeated by walking bass (the bass plays non-root tones most beats);
a trained 12-way classifier on absolute onset+note+bass+treble chroma learns the
"root-ness" pattern and generalizes to held-out songs at ~90%. Wired into the engine
(`chord_change_engine.py --root-model`, model at `harmonia/models/root_model.npz`):

| engine config | root | majmin |
|---------------|------|--------|
| coarse + bass-argmax | 60.5% | 39.5% |
| **coarse + trained root** | **80.9%** | **58.8%** |
| oracle bounds + trained root | 75.2% | 55.9% |

→ Root was the entire labeling gap; the trained root model ~doubles end-to-end
majmin. (Caveat: the wired model was trained on a pool overlapping the 15 eval
songs — the 80.9% MIREX-root is a mild over-estimate; the held-out CV 85.9–93.4%
is the rigorous generalization claim, and the engine number sits below it.)

## 6. Quality/family emission + a silent feature-scale bug (the second lever)

`quality_model_experiment.py` (runs on the extracted oracle-segment table, no
rendering). Family-given-CORRECT-root is already strong; the end-to-end shortfall
was a calibration bug, not a weak quality model:

- family 94.4% / third 95.6% (onset+note+bass+treble, 5-fold by song); `perfect`
  GT-MIDI ceiling 99.4%. **key_prior adds only +0.5%** — the "key picks the third"
  lever is real but redundant when chroma is clean (would matter more when degraded).
  onset chroma carries the third; note/sustain alone is much worse (66%).
- **Silent bug (rule #1):** `reg_chroma`/`full_chroma` produce UNNORMALIZED summed
  chroma whose magnitude scales with segment length. The family model is trained on
  oracle-segment scales but applied to coarse segments of different durations →
  inputs land off-distribution after StandardScaler. Fix: L2-normalize each 12-chroma
  block (`norm_blocks`, train + inference) → duration-invariant.

End-to-end labeling arc (coarse engine, GT-structure scaffold, θ=0.15):

| stage | root | majmin (clean) | majmin (degraded) |
|-------|------|----------------|-------------------|
| bass-argmax + unnormalized family (start) | 60.5% | 39.5% | — |
| + trained root model | 80.9% | 58.8% | — |
| **+ family-feature normalization** | 80.9% | **82.8%** | **78.6%** |

majmin more than doubled (39.5→82.8%). majmin can exceed root because mir_eval
scores majmin only over maj/min reference segments.

**Root model retrained with clean+degraded augmentation** (`--augment`): degraded
root 77.8→**81.5%**, degraded majmin 78.6→**84.1%**, with no clean regression
(clean root 79.8%, majmin 83.7%). The model is now robustly ~80% root / ~84% majmin
on both conditions. Remaining headroom: `perfect` ceiling 99% vs ~95% audio third;
and the GT-structure scaffold still needs replacing with detected structure (parked #1).

## 7. Structure detection (#1) — not needed, and not reliably possible here

Two findings close this task:

- **Not load-bearing.** The engine only uses structure to force a boundary at
  section changes. Dropping it entirely (`--no-structure`, constant section) costs
  **0.5 majmin** (83.7→83.2) — a new section almost always opens with a chord change
  the chroma cut already catches. So the engine is effectively structure-independent
  and works on pure audio (no form info) at majmin ~83%.
- **Not reliably detectable from harmony anyway.** `structure_repetition_ssm.py`, 40
  songs, section-boundary F vs GT: raw-SSM novelty 0.29, diagonally-enhanced novelty
  0.25, spectral clustering with *oracle* cluster-count 0.24 — all weak, barely above
  chance. Jazz AABA sections are defined largely by melody/phrasing; A and B share the
  same key and ii-V vocabulary, so bar-level chord SSMs don't separate them. Matches
  the POP909 CRHA result (only 23% of songs have validatable harmonic repetition).

→ Close #1: structure detection is neither necessary for chord accuracy nor
achievable from harmony on this data. (It could still be pursued for a form-display
UI feature, but would need melodic/phrasing features, not harmony.)

## 12. Segmentation robustness under distortion (2026-07-06, part 4)

The hard degradation is now a full DISTORTION chain (wow/flutter pitch warp, tremolo,
room comb, overdrive, hard-clip, bitcrush + noise/muffle/dropouts), not just noise —
baseline coarse-standalone majmin drops to 64% (was 79% noise-only). Diagnosis of the
segmentation failure: it's a PRECISION collapse (change-P 0.93→0.65, recall holds 0.86,
seg/GT 1.06→1.53) — distortion makes two blocks WITHIN a held chord look different, so
we cut where no change exists.

Fix tried — **adaptive threshold** (`--nov-adapt`: eff_theta = theta + k·median(novelty)):
FAILED. Raising the threshold suppresses spurious boundaries (seg/GT 1.53→1.00 at k=0.5)
but kills real changes just as fast (recall 0.86→0.65), so majmin drops. The premise was
wrong: distortion doesn't shift the novelty distribution up uniformly, it COMPRESSES it —
held chords and real changes both become equally muddy (wow/flutter + muffle corrupt the
chroma), so no global threshold separates them. `--nov-adapt` left off by default.

Deeper implication: under heavy distortion the per-block change signal (chroma+bass
novelty) loses discriminability, and a duration prior can't rescue it either because
real chords ARE ~2 beats (the harmonic-rhythm mode) — indistinguishable from a 2-beat
flicker by duration alone. The only recourse that adds genuinely new information is the
STRUCTURE prior (a repeated chorus gives an independent look at the same chords), which
needs REAL audio (correlated on synthetic). So segmentation robustness under this level
of distortion is evidence-limited, not tuning-limited. Model-based boundaries (segment by
the robust root/family prediction rather than raw chroma novelty) is the remaining
untried lever, but block-level label flicker under distortion limits it too.

## 11. Confidence-gated EM refinement + hard degradation (2026-07-06, part 3)

Vision: keep per-segment root/quality as PROBABILITIES, then leverage key + progression
(+ structure) priors to correct the fuzzy cases. Key design insight (the user's): only
refine LOW-confidence segments — CLAMP the ones the evidence is already sure about, so a
correct-confident chord is never broken. `em_refine_roots` (`--refine`): clamp segments
with root-confidence ≥ `--conf-gate` (0.6), Viterbi-decode the rest with a learned
root-motion prior (`learn_root_transition`, empirical from the DB) + estimated key,
iterate (re-estimate key from the decoded roots).

- Fixed-weight priors (no clamping) HURT everywhere (they tug confident-correct chords);
  jazz non-diatonicism makes the key prior neutral. Clamping fixes this.
- Confidence-gated EM: oracle root 87.8→88.4% / majmin 91.6→92.2% (hard degrade);
  coarse medium-degrade 87.3→87.5% — small but POSITIVE where segmentation is good.
- BUT on coarse HARD-degrade it HURTS (79.3→77.5): there the damage is to SEGMENTATION
  (seg/GT 1.5), so the prior refines labels over a meaningless segment sequence.

Conclusions: (1) the labeling evidence is robust even under strong noise (oracle root
only 89.6→87.8% hard) so the EM's headroom is small on synthetic data; (2) the EM is
the right architecture and is SAFE (won't break confident chords) but should be gated on
segmentation quality; (3) the structure prior — the strongest one conceptually — stays a
dead end on synthetic data (correlated repeats) and needs real audio. `--refine` is
OFF by default.

**Hard degradation** (`build_accomp_audio_hard.strong_nonuniform_degrade`, `--hard-degrade`):
drifting STFT muffle + near-silent dropouts + wide gain/SNR swings + heavy clip. Mostly
breaks segmentation/beat-tracking (chgF 0.89→0.76), not labeling. Playable A/B examples:
`scripts/make_degraded_example.py` → `demo_audio/example_{clean,degraded_medium,degraded_hard}.wav`.

## 10. Harness fix + boundary improvement (2026-07-06, part 2)

The oracle-vs-coarse comparison was corrupted by a GT-source mismatch (known_issues
#11): estimated segmentation came from `gt_chord_per_beat`, the reference from
`song_chord_spans`. Fixed by deriving segmentation, per-beat GT, change-times AND the
reference all from `song_chord_spans`. This revealed the true numbers and reversed the
"segmentation is at its ceiling" conclusion:

| config (15 songs, root model, θ) | root | majmin | chgF | chgF0 | seg/GT |
|----------------------------------|------|--------|------|-------|--------|
| oracle bounds (true ceiling) | 89.1% | 93.6% | 0.99 | 0.99 | 0.97 |
| coarse GT grid, θ=0.15 (old default) | 79.8% | 83.7% | 0.87 | 0.86 | 0.80 |
| **coarse GT grid, θ=0.08 + coalesce** | **85.6%** | **89.4%** | 0.91 | 0.89 | 1.06 |

Two boundary fixes, once the eval was honest:
- **Lower θ (favour recall).** Under-segmenting merges two chords (costly); over-
  segmenting repeats a label (nearly free) — so θ=0.15 was mistuned. θ≈0.08 lifts
  GT-grid majmin 83.7→89.4%, within ~4 of the 93.6% ceiling.
- **Coalesce adjacent same-chord segments** (a repeated chord is one chord): undoes
  the low-θ over-segmentation for free (labels-over-time unchanged → MIREX preserved;
  seg-count back to ~1.0, change-precision recovered).

And the earlier "exact placement 0.50, huge zoom headroom" was itself the harness
artifact: with the fix, coarse chgF0 is 0.84–0.89. Boundaries are placed well; the
residual gap to the ceiling is missed changes merging chords, addressed by the θ/recall
fix above — the four failed "zoom" strategies were solving a mismeasured problem.

**Definitive fully-standalone, fully-disjoint (train even songs, eval 30 odd, tempo-
grid, θ=0.08, all fixes):** root **77.6%** / majmin **74.9%** — up from 74.0% / 70.5%
before these fixes. 30-song overlapping (production model): root ~82% / majmin ~79%.

## 9. Hierarchical quality tree — seventh & exact levels (confidence-gated)

`seventh_model_experiment.py`, oracle-segment table, root-relative, given correct
root, 5-fold by song. Adds the deeper tree levels above family (triad ~94%):

| level | classes | ungated | gate t=0.7 | gate t=0.9 | GT-MIDI ceiling |
|-------|---------|---------|-----------|-----------|-----------------|
| seventh (base7) | 14 | 88.4% | 94.2% @ 82% cov | 96.6% @ 62% cov | 98.6% |
| exact | 18 | 84.4% | 92.7% @ 77% cov | 96.2% @ 56% cov | 98.2% |

- The confidence gate (max softmax prob) is well-calibrated at every level, so the
  project's "report deeper only when confident" rule works: raising the threshold
  trades coverage for accuracy monotonically.
- The 7th is carried by the ONSET attack, not sustain: note-only 49% vs onset 80%
  (surprising — 7ths are held, but the attack captures all tones and the sustain
  channel is near-constant/muddy). key_prior adds nothing (chroma already resolves it).
- These are given CORRECT root; in the standalone pipeline root is ~93%, so end-to-end
  seventh ≈ 0.93 × 0.88 ≈ 0.82 ungated (compounding), higher under the gate.

## 8. Fully standalone (raw audio → chords) — the capstone

Swapped the exact GT beat grid for audio beat tracking (`chord_change_engine.py`,
15 songs, θ=0.15, trained root, no structure):

| beat source | root | majmin (clean) | majmin (degraded) |
|-------------|------|----------------|-------------------|
| GT grid (`spb` from tempo) | 79.8% | 83.7% | 84.1% |
| raw librosa beats | 62.0% | 63.6% | 54.9% |
| **uniform grid at detected tempo (`--tempo-grid`)** | **79.1%** | **81.6%** | **83.7%** |

Raw librosa beats cost ~20 majmin points despite beat-F 0.87 — NOT an octave/count
bug (tempo and beat count are within 2% of GT), pure per-beat PHASE jitter that
smears the pooling. Since MMA renders are metronomic, imposing a uniform grid at the
*detected* tempo (accurate) + circular-mean phase recovers almost all of it. So the
system is fully standalone (no chart, no GT beats, no structure) at **majmin ~82%
clean / ~84% degraded, root ~80%**.

Caveat: the uniform-grid trick assumes a metronomic source (MMA / programmed music);
real human performances with rubato would need genuine per-beat phase tracking, which
librosa does not provide well enough (the ~20-point gap is the cost). For the
accompaniment-DB / programmed-music use case it is a full solution.

**Larger-sample validation (30 songs, rule #5 — the 15-song numbers were mildly
optimistic):** standalone tempo-grid θ=0.15 gives root 78.1% / majmin **75.1%**
clean, root 75.8% / majmin **71.3%** degraded (seg/GT 0.97). θ=0.15 confirmed
optimal (monotonic in the sweep).

**Disjoint held-out (the definitive number, `--parity`):** root AND family models
trained ONLY on even-numbered songs, evaluated standalone on 30 odd songs (no song
overlap): root **74.0%** / majmin **70.5%** (θ=0.15, seg/GT 0.89). ~4 points below
the overlapping run — mild optimism confirmed, but the result holds. So the honest,
fully-standalone, fully-held-out figure is **root ~74% / majmin ~70%** on raw
audio→chords, up from majmin 39.5% at the start of this session's labeling work.

### Reconciliation with the POP909 handoff (2026-07-04)

That investigation found the POP909 production pipeline's bottleneck is **timing,
not labeling** (chords held 15–35 beats vs changing every ~2). This one finds the
opposite on jazz1460 (accompaniment DB): segmentation is at its useful ceiling and
**labeling (root) is the bottleneck**. Both are correct — different data, different
limiter: POP909 is real piano with simple triads (root usually clear, timing hard);
jazz1460 is MMA jazz with walking bass + extended chords (2-beat merge nails timing,
root is hard). The handoff's finding #5 (audio-surface variation between identical-
chord repeats breaks evidence-averaging) matches this project's fold-EM dead end.

### (historical) Naive beat-level zoom FAILS

`--zoom` (snap boundary + split on interior beat-level novelty): chgF 0.89→0.86,
exact 0.50→0.41, root 60.5%→50.1%, ratio 0.91→1.08 (over-segments). Reason: beat-
resolution mixed-chroma novelty IS the noisy g=1 signal (AUC 0.643) that merging
suppressed — splitting on it reintroduces the noise. **The zoom needs a cleaner
cue than mixed beat novelty** → per-track self-similarity (bass-onset motion,
piano SSM), which we get free from the MIDI stems. That is the next step.

## Next

Per-track zoom: within each coarse segment, use the *isolated bass onset* PC-change
and per-instrument SSM to locate interior changes the mixed 2-beat block blurred —
the user's original zoom design, now justified by the naive-zoom failure. Possibly
within-song EM (safe here: within-song, not the correlated-cross-repeat fold that
died before).
