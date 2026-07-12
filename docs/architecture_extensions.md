# Harmonia — Architecture Extensions Spec

Brainstorm output to guide implementation. Existing pipeline (Basic Pitch → madmom →
SSM segmentation → Krumhansl-Schmuckler → Bayesian HMM) stays intact. These are
additive refinements, ordered simplest → most complex.

## 1. Soft hierarchy principle (guiding rule for everything below)

Priors at every level (scale transitions, chord n-grams, timing) should **regularize,
not override**. If the soft note-probability tensor from Basic Pitch strongly supports
a reading that contradicts a prior (e.g. an out-of-scale jazz chord), the data wins.
Priors are there to resolve ambiguity, not to censor valid acoustic evidence. All
priors should have a tunable weight so this balance is adjustable empirically.

## 2. Bass-anchored root inference (near-term, simple) — VALIDATED 2026-07-02/03

Before inferring full chord quality, use the lowest sounding pitch in each beat
window as a strong prior on the chord root. Two-stage inference:
1. Bass note → root candidate (e.g. G in bass → root is G).
2. Given root, infer quality (major/minor/etc.) from the rest of the pitch content.

Slash chords / inversions are a later extension — not needed for Phase 1.

**Update:** exactly this design was prototyped and validated in a 1-hour sprint
(`scripts/experiment_bass_chord_inference.py`) — given oracle (correct) chord-change
timing, bass-register-weighted evidence + a root/fifth-in-the-bass heuristic +
chroma template matching reconstructs chords at 86.8% root accuracy (vs the real
pipeline's ~33-35%). Ablations confirmed the two-stage intuition exactly: root is
almost entirely a bass question, quality almost entirely a chroma question.
Slash-chord/inversion handling is still not done — see docs/known_issues.md #1's
"Oracle-segment chord reconstruction" subsection for the full write-up, including
why inversions look like model errors but are actually a ground-truth labelling
convention mismatch (GT stores the functional root, not the sounding bass note).
Not yet wired into `harmonia/models/chord_hmm.py` — still a standalone scoring
function, no chord-*change* detector built around it yet (see item #9 below).

## 3. Modulation detection via accidentals (replaces/augments confidence-gap method)

Instead of (or in addition to) the KL-divergence confidence-gap method already in
Stage 4, treat out-of-scale notes as **hard evidence** for modulation:
- One accidental (e.g. F# while nominally in C major) → candidate key is G major.
- Two consistent accidentals (F#, C#) → candidate key is D major.
- General rule: infer the minimal key change that explains the new accidental(s).

Longer-term extension: allow a fully chromatic "scale" (12-tone) as an escape valve
for passages that don't fit any diatonic key — treat out-of-scale color tones as
first-class events with their own prior weight rather than forcing a key label.

## 4. Hierarchical n-gram priors, learned from data

Extend the existing scale-agnostic (interval, quality) chord priors to also include:
- **Scale-to-scale transition priors** (e.g. likelihood of modulating a 5th up,
  relative major/minor, etc.), learned empirically rather than hand-specified.
- **Chord n-gram priors**: bigram and trigram progressions (e.g. ii-V-I), stored as
  scale-agnostic interval/quality tuples as already designed. Skip 4-chord+ n-grams
  for Phase 1 — revisit later.
- Weight training examples by the confidence score of the chord detector that
  produced them, so weak/uncertain detections don't pollute the learned priors.

**Update 2026-07-03:** this is directly buildable right now at near-zero cost —
all 909 POP909 songs' `chord_midi.txt`/`beat_midi.txt`/`key_audio.txt` are
symbolic (no audio/Basic Pitch needed), and `harmonia/theory/duration_prior.py::
fit_duration_prior()` already established the exact pattern (`POP909Parser.
parse_all(require_audio=False)` across the full corpus). See item #9 below for
a concrete plan that reuses this same pattern for chord-progression n-grams.

## 5. Learnable timing-deviation model (rubato/swing)

Replace the current rigid beat-window averaging with a two-stage model:
1. Fixed tempo/beat grid from madmom (unchanged).
2. A learned **timing offset distribution** per player/style — how far ahead of or
   behind the beat notes tend to land, and whether that offset is systematic
   (e.g. consistently early) or high-variance. This is inferred, not hand-tuned.
3. Use this distribution to reweight frame probabilities around the nominal beat
   time before pooling into beat-level note probabilities, instead of a flat window
   average.

This is the most architecturally invasive change — expect it to interact with
everything else (more free parameters, harder joint convergence). Build it last,
after the rigid-grid version is working end-to-end.

## 6. Multi-resolution training data from personal recordings

Plan: record ~1 hour of solo piano improv (free playing, deliberately drifting
between key centers) plus synthetically generated MuseScore progressions (chord
sheet → full arrangement, randomized instrumentation) as augmentation. Use source
separation (e.g. Spleeter or similar) to isolate the piano track from any
augmented/multi-instrument versions.

Critical detail: **segment length must match what you're trying to learn**:
- **~20–30s windows** for learning key/scale stability and modulation patterns —
  short windows are too statistically noisy to infer a stable scale.
- Slice those same long segments into **shorter windows (2–5s)** for learning chord
  transition priors, emission probabilities, and timing deviations.
- One recording session, multiple derived granularities — don't record separately
  for each.

Held-out validation: keep a portion of personal recordings unseen during training;
also cross-check generalization against POP909 (acknowledging it's pop, not jazz,
so treat it as a sanity check rather than primary validation for jazz-specific
priors).

## 7. Compute plan

- MacBook M4 is fine for inference (Viterbi decoding, running the pipeline).
- Use cloud GPU (e.g. Colab Pro, ~$20–50/mo) for training the EM/joint inference
  over n-gram priors and timing deviations — local overnight training would take
  weeks to converge.
- Budget: ~$50–100/month is acceptable.

## 8. Known risk areas

- **Basic Pitch**: not trained specifically for polyphonic piano across full
  register range — expect degraded performance on low/sustained notes; the soft
  probability tensor should mitigate this somewhat since no hard thresholding
  decisions are made early.
- **madmom**: trained on pop/electronic music; likely to snap-to-grid on jazz
  ballad rubato. The timing-deviation model (Section 5) is the fix, applied as a
  post-process rather than replacing madmom itself.
- **Krumhansl-Schmuckler profiles**: from 1980s tonal-music research; expect
  misclassification on modal jazz / modal interchange passages. No fix specified
  yet — flagged for future work.
- **Joint convergence risk**: as more learnable levers are added (n-gram priors,
  timing deviations, modulation priors), joint EM convergence gets harder. Build
  incrementally — get the rigid/simple version working fully before adding each
  new learnable layer.

## 9. Beat-phase-aware harmonic rhythm prior (low-level structure, cheap, immediate)

2026-07-03 exploratory work (`scripts/plot_chord_change_correlates.py`) asked a
direct question for one song first: which observable per-beat signal actually
predicts a real chord change? Point-biserial correlations against the real
chord-change indicator, song 001 (POP909's own ground-truth beat/downbeat grid,
not our audio beat tracker — see the script for why): **is-downbeat r=+0.53**,
bass-pitch-class-changed r=+0.40, onset density r=+0.24, local chroma novelty
r=+0.22, bass-onset-present r=+0.16. `P(chord changed | beat-in-bar phase)` for
that song: beat 0 (downbeat) 97.3%, beat 1 37.0%, beat 2 69.9%, beat 3 1.4% —
chords in that song almost only ever change on the downbeat or halfway through
the bar, never on the last beat.

**Checked immediately against the full 909-song corpus (symbolic only, no
audio — same `parse_all(require_audio=False)` pattern as `duration_prior.py`)
to see if song 001 generalizes:** it doesn't fully. Corpus-wide `P(chord
change | beat phase)`: beat 0 (downbeat) 50.8%, beat 1 42.0%, beat 2 29.3%,
beat 3 22.8% (n≈74k beat-transitions per phase, pooled). **What "real but
softer" concretely means** — see
`docs/plots/structure_proposal/phase_variability_across_songs.png`, generated
by `scripts/plot_structure_proposal_illustrations.py`: those four pooled
numbers are *means over 909 songs*, and the per-song spread around each mean
is huge. Concretely, define a per-song "downbeat advantage" =
`P(change|beat 0) − P(change|beat 3)` for that song alone: **corpus mean
+0.28, std 0.53, ranging from −0.96 to +1.00.** Song 001's value is **+0.99 —
the single most extreme song in the entire 909-song corpus** for this
statistic (100th percentile). So the earlier single-song plot wasn't wrong,
it just happened to pick close to the most metrically rigid song available;
a random other song could show almost no downbeat effect, or even a slight
*negative* one. The corpus-wide direction (downbeat > later beats) is real
and consistent, but a single fixed weight for "how much to trust this" would
be systematically wrong for a large fraction of songs — this is itself an
argument for making the weight per-song-adaptive (e.g. re-estimated from
that song's own harmonic-rhythm regularity) rather than a single global
constant, more so than most of the other priors in this document.

**What IS fully corpus-representative, and IS a real distribution, not a
point estimate:** `harmonia/theory/duration_prior.py::fit_duration_prior()`
(built session 5, already returns a full PMF — it just hadn't been plotted
before) — see
`docs/plots/structure_proposal/chord_duration_distribution.png`. Full shape,
909 songs / 120,069 chord events: **P(d=1 beat)=15.0%, P(d=2)=49.2%,
P(d=3)=9.0%, P(d=4)=25.6%**, essentially zero mass beyond 6 beats (mean=2.49,
mode=2, median=2). This is not a geometric/memoryless shape — a geometric
distribution is maximised at its minimum (d=1) and decays monotonically;
this one peaks at d=2, which is exactly the empirical argument (already used
to justify Candidate B's semi-Markov decoder in `docs/known_issues.md` issue
#1) for why `self_transition_boost`'s implied geometric shape is structurally
the wrong family, not just mistuned. This full PMF (not a summary statistic)
is what should be plugged into any duration-aware decoding — `duration_prior`
already accepts exactly this dict shape as an argument to
`ChordInferrer`/`HarmoniaPipeline`.

**Proposed use:** a per-beat multiplicative bias on the HMM's self-transition
probability in `harmonia/models/chord_hmm.py`, keyed on beat-in-bar phase
(needs real downbeat detection — currently `_track_beats_librosa()` always
returns `downbeat_times=np.array([])`, so this requires either enabling
madmom's downbeat tracker or a comparably reliable audio-based downbeat
estimate; POP909's own `beat_midi.txt` markers were used as ground truth for
this analysis and won't be available for other audio). Given the average
corpus signal is real but soft — and, per the per-song variability finding
above, unevenly distributed across songs — treat this as one weighted term
among several (same "priors regularize, don't override" principle as item
#1), with a per-song-adaptive weight rather than a single hardcoded constant.

## 10. Learned chord-progression n-grams from the full corpus (low-level structure)

**2026-07-03 update: see `docs/scale_taxonomy_2026-07-03.md` for the full,
detailed follow-up** — an atomic scale taxonomy (major-family covers all 7
"church modes" from one table; harmonic-minor-family added and *mostly*
found to be mathematically redundant with parallel-mode borrowing, not a
separate category as first assumed), a validated mode-agnostic parent-scale
identifier (95.3% agreement with GT using chord content alone, no key
lookup), fully pooled "atomic" bigrams, cross-scale transition tracking, and
an empirical finding that the specific 7th type (`dom7` vs `maj7`) predicts
functional-dominant behaviour better than scale position does. The
walkthrough below is the original (less general) version of this idea, kept
for the historical record of how the analysis got there.

Concrete build-out of item #4's chord n-gram idea, now that item #2's
bass-anchored root inference is validated and the required infra
(`RelativeChord`/`PROGRESSIONS`/`build_relative_transition_matrix` in
`harmonia/theory/jazz_priors.py`) already exists for *hand-specified*
progressions:

1. For each of the 909 songs, convert its `chord_midi.txt` (root, quality)
   sequence into a scale-degree-relative sequence using that song's real
   `key_audio.txt` key (e.g. `F#:maj → I`, `C#:maj → V`, `Bb:min → iii`,
   `Eb:min → vi` for a song in F# major) — exactly the `RelativeChord`
   representation `jazz_priors.py` already uses for its hand-written
   progressions like `ii_V_I_major`.
2. Fit empirical bigram/trigram transition frequencies over these relative
   sequences across all 909 songs (mirrors `fit_duration_prior()`'s pattern
   exactly — pure text parsing, no audio, minutes not hours to build).
3. Use the fitted frequencies as data-driven weights on top of (or in place
   of) `PROGRESSIONS`' current hand-specified ones, keeping the existing
   `style`-conditioned structure (`STYLE_PRIORS`) — pop programming will
   look different from the hand-written jazz progressions already coded, and
   this makes that difference explicit and measured rather than assumed.
4. Modulation frequency/direction (item #3) falls out of the same fit almost
   for free — `key_audio.txt` already gives real per-song key spans, so
   scale-to-scale transition frequencies (item #4's other bullet) can be
   fit from the same pass over the corpus.

Caveat carried over from item #4's original text: weight or filter training
examples by how reliable the source labels are. Session 8 found ~10-18% of
chords per song carry a bass-inversion marker (`docs/known_issues.md #1`) that
`POP909Parser` currently discards — worth preserving for this specific
purpose (inversions are structurally informative for voice-leading n-grams,
e.g. `I → V/vi → vi` cadential patterns) even though the current bass-root
scorer treats them as a labelling mismatch to route around.

**Worked example, already run** (`scripts/plot_structure_proposal_illustrations.
py::illustrate_ngrams()`, steps 1-2 above, no audio, ~106k chord-to-chord
transitions pooled from all 909 songs) — see
`docs/plots/structure_proposal/ngram_illustration.png`. Top scale-degree
bigrams, quality collapsed to maj/min/other so the table is readable (the
script also computes the quality-aware version, sparser and printed to
console but not plotted):

| bigram | share of all transitions |
|---|---|
| V → I | 9.58% |
| IV → V | 5.21% |
| I → IV | 4.56% |
| I → I (root stays put, quality changes — see below, this is real, not an artifact) | 4.27% |
| I → V | 3.70% |
| II → V | 3.57% |
| bVI → bVII | 2.92% |
| IV → I | 2.79% |
| bVII → I | 2.71% |
| III → VI | 2.56% |

This is immediately, directly interpretable: `V → I` (the authentic cadence)
alone accounts for nearly 1 in 10 of *all* chord-to-chord transitions in the
corpus — a far stronger, more concentrated signal than the hand-specified
`PROGRESSIONS` dict currently gives any weight to, simply because it wasn't
measured before. `bVI → bVII` and `bVII → I` being this frequent (ranks 7 and
9) reflects POP909's pop/Mixolydian-leaning vocabulary specifically (a
"borrowed" `bVII → I` cadence is common in pop but not in the hand-written
jazz `PROGRESSIONS`) — exactly the kind of style-specific pattern step 3
above says should be measured rather than assumed.

The `I → I` row was double-checked before writing it down here, since same-
root "transitions" are a plausible annotation artifact (a held chord split
across two adjacent lines in `chord_midi.txt`) — that case is explicitly
excluded already (`if a.root == b.root and a.quality == b.quality: continue`,
in `illustrate_ngrams()`, applied before either table is tallied). Breaking
down what's actually left in the "same root, different quality" bucket
(10,921 events) confirms it's genuine musical content, not a filtering gap:
`sus2 → maj` (1379), `maj → sus2` (1155), `sus4 → maj` (978), `maj → maj7`
(642), `maj → min` (544, i.e. real modal mixture / parallel-key borrowing) —
suspension resolutions and quality extensions, exactly the kind of
progression-adjacent information a quality-aware n-gram model should keep
distinct from genuine root motion, not collapse away.

## 11. High-level song FORM (AABA / verse-chorus), harder to validate

Distinct problem from items #9-10: those are about **local** harmonic rhythm
and progression; this is about **global** repeated-section structure — does
this song follow AABA, verse-chorus-verse-chorus-bridge-chorus, etc., and
where are the section boundaries in *bars*, not beats.

**What already exists:** `harmonia/models/structure.py` builds an SSM and a
checkerboard novelty curve, and `harmonia/models/periodicity.py::
score_periods()` finds repeating period lengths from the SSM's off-diagonal
averages (already found a clean 32-beat/8-bar period for song 001, score
0.82 — see `docs/plots/inference/pop909_001/ssm_periodicity.png` from the
issue #1 investigation). **What's missing:** both of these only find
*boundaries* or *periods* — neither assigns section *identity* (knowing
section 3 is "the same as" section 1, i.e. actually labelling the sequence
`A A B A` rather than just cutting the song into four unlabelled pieces).

**Proposed approach:**
1. Use bar-aligned windows (needs real downbeat detection — same
   prerequisite as item #9) instead of arbitrary beat-count windows, so
   candidate section boundaries always land on a downbeat.
2. Slice the song into candidate sections using `score_periods()`'s detected
   period(s) as the window length.
3. Cluster the resulting section-level chroma/harmonic summaries (cosine
   similarity, reusing `build_ssm`'s machinery but at section granularity
   instead of beat granularity) into repeated-section groups — a label
   sequence like `A A B A` is the *output* of this clustering step, not an
   assumption fed into it.
4. Render the result as a lead-sheet-style annotation (section labels +
   inferred progression per section) rather than folding it back into the
   HMM decoder — this is presentation/output-shaping, not something that
   obviously improves chord-recognition accuracy on its own.

**Why this is flagged as harder, lower-priority work:** POP909 has no
ground-truth section-label annotations (no verse/chorus/bridge markers
anywhere in the dataset) — items #9 and #10 above can be validated directly
against real labels; this one can only be checked by ear/eye (does the
clustering's `A A B A` match what a listener would actually call the song's
form?) or against a different, smaller, differently-annotated dataset
(e.g. SALAMI or Isophonics have structural-segmentation ground truth POP909
doesn't). Recommend doing #9 and #10 first — they're cheaper, directly
measurable, and more likely to move chord-recognition accuracy; treat this
as a "richer output" feature to revisit once those land.

**A real (not mocked) prototype was run** to check the approach isn't
vacuous before investing further — `illustrate_form_clustering()` in
`scripts/plot_structure_proposal_illustrations.py`: greedy nearest-centroid
clustering (cosine similarity, threshold 0.85) over `score_periods()`-length
windows, using librosa-derived beats (not POP909's ground-truth beat file,
since a real deployment won't have that either).

- **Song 002** (detected period 32 beats, periodicity score 0.56 — a middling
  score, i.e. real but not overwhelming repetition): produced
  `A B B B B A B B B B B B B A` — see
  `docs/plots/structure_proposal/form_clustering_song002.png`, which shows
  the SSM directly above the label strip so the block-structure driving the
  clustering is visible. The isolated "A" windows landing at the very start,
  once in the middle (~80-95s), and the very end is a musically plausible
  pattern (a recurring intro/interlude figure bookending and briefly
  interrupting the main verse/chorus body) — genuinely a nontrivial, useful
  answer, not noise.
- **Song 001** (detected period 16 beats, periodicity score 0.91 — much
  higher self-similarity): produced `A A A A A A A A A A A A A A A A A`, i.e.
  the whole song as one section — see
  `docs/plots/structure_proposal/form_clustering_song001.png`. This matches
  everything already known about song 001 (the same `F#maj → B → C#maj →
  Bbmin → Ebmin`-family loop repeating for the entire track, see
  `docs/known_issues.md`'s song-001 discussion throughout) — the algorithm
  correctly declined to manufacture a section distinction that isn't really
  there, rather than a failure to find one.
- Side note on the period numbers themselves: this run found 16 beats for
  song 001 (not the 32 beats / score 0.82 reported for the same song in the
  issue #1 investigation's `ssm_periodicity.png`). Both are real peaks in the
  same periodicity profile (16 is a harmonic subdivision of 32); which one
  `score_periods(top_k=1)` returns as *the* top candidate is sensitive to
  exactly which stage-1/beat-tracking parameters were used to build
  `beat_probs` at the time — worth being aware of if period selection
  becomes load-bearing for section-length decisions, since it isn't fully
  stable across runs yet.

## 12. Backlog locked 2026-07-06 (after the stem/beat investigation, in order)

Three tasks parked while the chord-change engine (see below) is built:

1. **Repetition-SSM structure detection on harmonic content.** AABA lives in
   repetition (A returns as a diagonal stripe at the section-length lag), not
   local contrast — plain Foote novelty scored only F=0.25 on symbolic chords
   because jazz ii-V churn out-novelties the sections. Attack with a time-lag /
   diagonally-enhanced SSM. Cheapest, testable on current data. WIP scaffold:
   `scripts/structure_repetition_ssm.py`.
2. **Regenerate the DB with per-section MMA grooves + fills.** MMA currently
   renders one groove for the whole tune (identical drum voices A vs B, flat
   density) so the drum-fill-marks-section phenomenon is absent — the
   drum-structure prior is untestable (known_issues #9). Section-varied grooves
   make it real and the DB more realistic. Bigger lift.
3. **Segmentation + emission on real evidence** — the confirmed end-to-end
   bottleneck (beat tracking ruled out at F≈0.87 clean/degraded; oracle
   boundaries → 86.8% root vs detected ~67%). This is what the chord-change
   engine below is ultimately serving.

**In progress now: the chord-change engine.** Scaffold on GT structure →
estimate harmonic-rhythm period per section → merge at that period →
same-or-different fill → zoom per-track for missed transitions. Foundation
validated 2026-07-06: merging beats into 2-beat blocks makes change-vs-hold
separable at AUC 0.962 (vs 0.643 per-beat) — merging is the load-bearing lever.

## Suggested build order

**Updated 2026-07-03** — items #2 and #9/#10's groundwork moved from
"proposed" to "partially validated" since this doc was first written;
reordered to build on that momentum rather than the original sequence:

1. ~~Bass-anchored root inference~~ — **done, validated** (item #2): 86.8%
   root accuracy at oracle chord-change timing.
2. **Chord-*change* detection** (the remaining half of item #2's original
   goal): combine item #9's beat-phase prior with the bass-change/chroma-
   novelty signals already measured, decode segment boundaries, then apply
   item #2's validated scoring formula per segment. Needs real downbeat
   detection first (madmom, or an equivalent audio-based estimate — item
   #9's analysis leaned on POP909's own ground-truth beat file, not
   available for new audio).
3. **Learned chord-progression n-grams from the full 909-song corpus**
   (item #10) — cheap (symbolic only), reuses `duration_prior.py`'s exact
   pattern, directly measurable against real GT.
4. Accidental-based modulation detection (item #3) — modulation-frequency
   stats can piggyback on item #10's corpus pass almost for free.
5. Record + generate personal training data at multi-resolution (item #6),
   once the corpus-learnable priors (2-4 above) are exhausted.
6. Learn timing-deviation distribution; integrate into beat-level pooling
   (item #5).
7. High-level FORM structure (item #11) — lower priority, harder to
   validate (no ground truth in POP909), better attempted once 2-4 give a
   stable per-segment chord signal to cluster on.
8. Revisit Krumhansl-Schmuckler suitability for modal jazz once the above
   is stable.

---

## Iterative chord-graph refinement pass (next to implement — 2026-07-07)

### Motivation

Current pipeline ceiling: bass-novelty segmentation at grid positions catches
~7/9 GT changes (4 bars of Anthropology). The two misses are **silent bass**
positions — the bass note doesn't change even though the chord does (e.g. Dm7→G7
where the walking bass stays on G). No threshold adjustment can recover these.

Post-hoc observation: if we classify the merged segment we get a chord that is
locally implausible (e.g. a wide maj7 surrounded by dom7s in an A section). The
chord-graph pass exploits this.

### Hard rules (apply before any learned model)

1. **Semitone UP in bass = likely chord change.** A semitone upward step (mod 12,
   e.g. Bb→B) strongly suggests a chord change; a walking bass almost never steps
   up by a minor second. Apply as a high-weight soft cue (not a hard override).
   **Semitone DOWN is ambiguous** — could be the maj7 of the current chord
   (e.g. Bb→A inside Bbmaj7), so do NOT force a cut on downward semitones.
2. **Octave bass movement = same chord.** If the bass leaps by 12 semitones (mod
   octave = 0) the root PC is unchanged — do not cut.

### LTAS normalisation for root model retraining

LTAS-normalised CQT chroma (divide each pitch-class row by its long-term mean)
is visually and informationally superior for bass note detection: Bb-register
dominance is suppressed, making weaker upper-register notes equally readable.
The root model (`root_model.npz`) should be retrained with LTAS-normalised CQT
chroma features replacing the raw Basic Pitch onset/note chroma. Expected gain:
the model currently struggles on degraded audio partly because BP onset chroma
is noisy in the bass register; LTAS-CQT is computed directly from the waveform
and degrades more gracefully.

### The iterative pass

After initial segmentation + classification, run one refinement loop:

```
for each inferred chord c:
    score(c) = log P(chroma | root_c, fam_c)  # template log-likelihood
    for each neighbour n in chord_graph(c):
        score(n) = log P(chroma | root_n, fam_n)
    if max(score(neighbours)) > score(c) + margin:
        switch hypothesis to argmax neighbour
        recompute root from new hypothesis
repeat until no hypothesis changes (typically 1-2 iterations)
```

This is how a jazz musician hears: start with the most plausible reading of the
bass, check if the full chroma is consistent, if not look at adjacent chords in
the graph and see if one fits better.

### Chord proximity graph

**What "close" means** (in order of priority):
1. **Root motion by 4th/5th** (ii→V→I, dominant cycle) — interval 5 or 7
2. **Tritone substitution** — interval 6 (bII7 substitutes V7)
3. **Stepwise** — intervals 1, 2 (chromatic approach, passing chord)
4. **Parallel quality** — same root, family changes (e.g. C7→Cm7)

Bootstrap the graph from `db.jsonl` bigrams: build a 12×12 root-transition count
matrix (transposition-invariant, keyed on interval), then add circle-of-fifths
distance as a prior for unseen transitions.

### Splitting on implausibility

If `score(c)` is below a minimum likelihood threshold AND the segment spans ≥
`grid_step` beats, split it at the midpoint and re-classify both halves. This
handles the "silent bass = merged ii-V" case: the merged chroma is a blend of
two chords, neither fits well, so we split first and re-score.

Split condition: `log P(chroma | best hypothesis) < -H_threshold` where
`H_threshold` is set empirically (start: top-1 template dot-product < 0.5).

### Implementation plan

1. **`harmonia/models/chord_graph.py`** — build and cache the proximity graph
   from `db.jsonl`; expose `neighbours(root_pc, fam_idx, k=6)`.
2. **`harmonia/models/chord_scorer.py`** — per-segment log-likelihood under
   a chord hypothesis; uses the 12 chord templates + chroma vector.
3. **Extend `infer_blind()`** — after initial classification, run 2 iterations
   of the graph pass; add split-on-implausibility step; re-run beat_seq_model
   on new sub-segments.
4. **Hard-rule pre-pass** — semitone bass detection before grid segmentation.

**1-hour falsifiable test:** run the hard-rule semitone pre-pass alone on
Anthropology and count how many additional GT boundaries are recovered. If
≥ 1 new correct cut (expected: the Bb→G7 transition at bar 1 beat 3), the
rule is validated and worth keeping regardless of the graph pass outcome.

## 13. Professional annotator tool — human-in-the-loop correction, not passive
review (spec started 2026-07-12, more detail to come)

**Core stance:** the model proposes, the annotator disposes. Harmonia's
inference is an assistive draft, not a finished label — the tool's job is to
make correcting it fast enough that a professional annotator would actually
choose to use it over typing chords into a spreadsheet, *and* to turn every
correction into training signal instead of a one-off fix.

### 1. Chord correction — nested nested drill-down rotors

Reuses the existing rotor mechanic (drag-to-spin, detent + haptic click,
`docs/logo`-style circular UI already shipped for transpose) as the base
interaction, extended into a **telescoping stack of wheels** opened one at a
time from a tapped chord:

1. Tap a chord cell (shows current guess + its certainty, same colour scale
   already used on the chart) → root-note wheel opens (12 positions, same
   geometry as the transpose rotor).
2. Confirm/change root → a second wheel opens for **quality family**
   (maj / min / dom / dim / aug / sus), replacing or nesting below the first.
3. Confirm family → **extension wheels** open progressively as needed: 7th
   present? which 7th? add a 9th? natural/b9/#9? 11th, 13th, alterations —
   each wheel only appears once the previous one is confirmed, so a plain
   triad never has to scroll past extension wheels it doesn't need.
4. **Live scale/mode feedback**: as extensions accumulate, show the implied
   scale/mode inline (Ionian, Dorian, Mixolydian, Locrian, altered, etc.) —
   this is score-relevant information a professional annotator wants
   *while* building the chord, not after, since it's often how they're
   thinking about the sonority in the first place (reading the accumulated
   interval set against `harmonia/theory/chord_vocabulary.py`'s templates
   and reporting the nearest scale match).
5. Each edit is a structured correction (`{bar, beat, old, new, annotator,
   timestamp}`), not just an overwrite — needed for #3 below.

### 1b. Chord-suggestion mode (noted 2026-07-12, **list view shipped** 2026-07-12)

A second correction mode alongside the drill-down rotor: instead of dialing
in a chord from scratch, show a ranked list of the model's *alternative*
hypotheses for that slot with their probabilities. **Shipped as a flat
ranked list**: `chord_pipeline_v1._top_chord_suggestions` keeps the top-5
joint (root x q5-quality) candidates from the two posteriors the pipeline
already computed and discarded after argmax (12-way beat-sequence root
posterior, 5-way family/seventh classifier posterior over
maj/min/dom/hdim/dim) — baked into the chart JSON as `P.chords[i].sug`.
The chord-edit modal's "Suggestions" tab renders them as rows with a
probability bar, a temperature slider (client-side `p_i^(1/T)`
renormalization — reshapes display spread only, never reorders), tap to
preview (arpeggio, see below) and select.

**Next requested step (2026-07-13, not yet implemented): a circular/radial
layout, color-coded by probability**, instead of (or alongside) the flat
list. This is the visual-chord-proximity idea below, now with an explicit
ask — still needs the research pass on chord-embedding/chord-space layouts
before picking a concrete geometry (see open questions just below); the
color-coding-by-probability part is straightforward regardless of layout
(e.g. probability → saturation/lightness on each candidate's existing
family-color, reusing `chart_interactive.py`'s `FAMILY_COLOR`/`motifColor()`
scheme so it stays visually consistent with the rest of the chart).

Likely needs a **visual chord-proximity representation** to be maximally
usable: a flat ranked list of "Cm7 (62%), Cm (18%), C7 (11%), ..." doesn't
give an annotator any sense of *why* those are the alternatives or how they
relate to each other. Some kind of 2D/radial layout where nearby chords in
the display are harmonically/acoustically close (voice-leading distance?
shared-tone count? a learned embedding from the training data? simplest:
just the circle of fifths, root angle = root pc, ring = quality family,
already half-built by the existing rotor geometry) would let an annotator
recognize the right answer by its neighborhood rather than reading a
list — worth a research pass on existing chord-embedding/chord-space
visualization work before designing this, not just inventing a layout.

### 2. Manual motif/section merging — the annotator supplies structure the
model had to guess at

Given two spans the annotator judges to be "the same section" (e.g. two A
sections in an AABA form that the model's own structure-detection wasn't
confident enough to merge), let them mark and merge them directly on the
chart — a lightweight extension of the existing motif-bracket UI
(`motifState`/`getMotifColor`/manual "Jazzify" tagging already in
`chart_interactive.py`, which already does draggable-range tagging with
rename/ungroup/delete — the merge action is new but the interaction
substrate already exists).

**Why this matters beyond convenience:** a merge is a strong prior fed back
into inference, not just a label. If section A-take-1 was uncertain between
`Am7` and `Am` at bar 12, and section A-take-2 (independently, from a
different acoustic instance — different voicing, different octave, maybe
different confidence) leaned `Am7` more strongly at the *equivalent* bar,
merging the sections means those two observations get pooled: same slot,
two independent pieces of acoustic evidence, effectively doubling the
sample and sharpening the posterior instead of treating each occurrence as
an isolated guess. This is exactly the kind of structural fact (`these two
spans are the same underlying material`) that's cheap for a human to
supply and expensive for the model to infer reliably — offloading it is a
strict win, and it's the same "soft hierarchy" principle as §1: a human
merge is the strongest possible prior, but if the acoustic evidence
actively disagrees at a specific bar, that should still be visible/
overridable, not silently forced.

### Decisions (resolved 2026-07-12, ready to implement)

- **Persistence**: per-song JSON sidecar, same pattern as `.yt_video_ids.json`
  / `.yt_audio_meta.json` in `scripts/harmonia_server.py` (e.g.
  `.annotations/<chart>.json`). One annotator per song — no multi-annotator
  agreement/audit-trail schema needed; the sidecar just holds the current
  best label per bar plus the merge groups, not a revision history.
- **Multi-annotator**: out of scope. One annotator per song simplifies the
  whole data model — a correction just overwrites the current value, no
  identity/conflict tracking required.
- **Merge UI**: explicit two-step selection — tap section A, tap section B,
  a "Merge" button appears/activates. Not drag-and-drop (unreliable on a
  small touch target for this kind of range-to-range gesture).
- **Re-inference on merge**: local re-score only. Recompute confidence for
  the bars in both merged spans against their pooled chroma observations —
  no full re-download/re-inference pass. Keeps the interaction fast enough
  for an annotator to use repeatedly without a multi-second wait each time.
- **UI substrate**: extend `chart_interactive.py` (existing rotor + motif-
  bracket/tagging system) with an "annotate" mode, rather than building a
  second UI surface from scratch.
- **Target device, phased**: build and validate the *mobile* (iPhone PWA)
  annotation flow first — same surface as everything else in this session.
  A desktop/mouse-and-keyboard variant comes after the mobile interaction
  is validated, once it's clear what the mobile version actually needs
  (don't design the desktop version blind before the mobile one is used).

- **Annotator identity**: a plain name string, entered once client-side,
  stored in `localStorage`, and replayed into the sidecar on every write —
  no accounts, no auth.

### Resolved (delegated 2026-07-12, both closed)

- **Sidecar schema**: full design in `docs/annotation_sidecar_schema.md`
  (Opus subagent). Chord corrections keyed on `(bar, beat)` — not array
  index `i`, which isn't a stable id — one sidecar per chart at
  `docs/plots/annotations/<filename>.json`, merge groups reuse the existing
  manual-motif `bars: [[s,e],...]` shape rather than inventing a new one.
  Flags real friction to resolve before implementation (§5 of that doc):
  no defined behaviour for how a human correction should interact with the
  "Sure ≥" confidence-gate slider; `(bar,beat)` matching needs tolerance,
  not exact float equality.
- **Chroma persistence** (Explore subagent's finding, then fixed): the
  cache `.npz` survived on disk but was practically unreachable — keyed by
  the *downloaded temp file's* path+mtime, deleted every job. Fixed in
  `scripts/harmonia_server.py` (`PITCH_CACHE_DIR`): activations now also
  save to `data/cache/pitch/<slug>.npz`, addressable by the same slug as
  the chart and audio file, no manifest needed. Verified directly (not
  just read): re-extracting the same audio after the pipeline already ran
  is a cache hit (0.06s vs. the original 15.96s), and the save/reload
  round-trip is correct.
