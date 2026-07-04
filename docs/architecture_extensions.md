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
