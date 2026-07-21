# Production Model Audit — 2026-07-21

User-directed mission pivot (mid-session): stop scaling `aligned_corpus`
(left at 5306 rows / 148 songs — see `docs/known_issues.md`), audit the
CURRENT LIVE production pipeline scope-by-scope with real numbers instead.

## 0. What is actually live (verified, not assumed)

`ps eww <pid>` on every running `scripts/harmonia_server.py` process on this
machine (ports 7771-7774) shows **no** `HARMONIA_ANALYZE_{FRONTEND,BASS,
QUALITY,SEGSOURCE}` or `HARMONIA_BEAT_PERIOD_MODE` overrides set anywhere
(one instance sets `HARMONIA_OCCAM_POSTPASS=1`, which is also the code
default — a no-op). So the code defaults in `scripts/harmonia_server.py`
(lines 259-276) ARE the live configuration:

| knob | live value | meaning |
|---|---|---|
| `feature_frontend` | `nnls24` | NNLS-Chroma 24-d features (not BP48/Basic Pitch) |
| `bass_frontend` | `musx` | music-x-lab's own bass, not the in-house NNLS-24 head |
| `quality_frontend` | `musx` | music-x-lab's own quality, not the in-house head |
| `segment_source` | `nnls` | chord-CHANGE times from per-beat NNLS root-argmax flips — **NOT** `musx` (see §2, a built-but-off lever) |
| `beat_period_mode` | `bestfit` | whole-song LSQ beat period (not raw librosa scalar) |
| beat backend | `beatthis` (function default) | Beat This! (ISMIR 2024) neural beat/downbeat tracker |

This is the exact call `harmonia_server.py`'s analyze route makes:
`infer_chords_v1(audio_path, feature_frontend="nnls24", bass_frontend="musx",
quality_frontend="musx", segment_source="nnls", beat_period_mode="bestfit")`.

## 1. ROOT / QUALITY / BASS accuracy — measured fresh this session

**Method**: `scratchpad/production_audit_root_quality_bass.py` (new). Runs
the exact live call above end-to-end on real re-downloaded YouTube audio for
a random sample of `aligned_corpus.npz` songs (non-circular iReal GT — see
`docs/known_issues.md` "aligned_corpus" entries), scores predicted vs GT
root/quality/bass at each corpus row's time span midpoint.

**A real bug found and fixed before trusting any number** (CLAUDE.md rule
#1): the first run gave root_acc=0.094 — traced to a self-inflicted
stem-keyed cache collision (every re-downloaded song reused the literal
temp filename `audio.wav`, and both `nnls_features.extract_bothchroma` and
`musx_bass.musx_labels` cache purely by filename stem, so every song after
the first silently scored against the FIRST song's cached features). Fixed
by giving each song a unique stem; polluted cache files deleted. Full
root-cause in `docs/known_issues.md`. This also means the number below is
the FIRST trustworthy full-pipeline (not just heads-in-isolation) audit of
this system on real, non-circular held-out audio.

**Result, 13 songs / 564 rows** (2 songs skipped: YouTube download failures,
not scoring failures):

| scope | metric | value |
|---|---|---|
| ROOT | exact pitch-class accuracy | **0.525** |
| QUALITY | exact 7-way (maj/min/dom/hdim/dim/aug/sus) | **0.532** |
| QUALITY | family-level partial credit (maj-ish/min-ish/dim-ish) | **0.713** |
| BASS | sounding-bass pc (project's 2026-07-16 target) | **0.520** |
| JOINT | root AND quality both exact | **0.410** |

**Headline finding**: this is a LARGE gap below every RWC-based bake-off
number this project has previously reported for the same components
(musx root/quality vs RWC: 0.7-0.9 range; in-house nnls24 heads similar).
**RWC bake-off numbers do not generalize to messy real-world YouTube
pop/jazz audio** — RWC is professionally mixed/mastered J-pop/pop with
clean single-take studio audio; `aligned_corpus` audio is whatever
`ytsearch1` happens to return (live performances, different arrangements,
lo-fi transfers, tribute covers). This is the single most important
audit finding: **the production system's real-world accuracy on the kind
of audio actual users bring it is roughly half what the RWC numbers in
`known_issues.md` would suggest**, not a subtle regression.

**Per-song breakdown (bimodal, not uniform)**:
```
Don't You Worry 'Bout A Thing  n=83  root=0.82 qual=0.58 bass=0.82
Brown Eyed Girl                n=59  root=0.75 qual=0.68 bass=0.75
Always                         n=34  root=0.71 qual=0.56 bass=0.71
Bellarosa                      n=72  root=0.60 qual=0.57 bass=0.60
A Child Is Born                n=30  root=0.60 qual=0.57 bass=0.50
Bernie's Tune                  n=26  root=0.58 qual=0.62 bass=0.58
Can't Buy Me Love              n=79  root=0.47 qual=0.70 bass=0.47
A Weaver Of Dreams             n=34  root=0.41 qual=0.29 bass=0.41
Ablution                       n=16  root=0.31 qual=0.62 bass=0.31
Always And Forever             n=82  root=0.30 qual=0.43 bass=0.29
All The Things You Are         n=39  root=0.08 qual=0.18 bass=0.10
Angel                          n=8   root=0.00 qual=0.00 bass=0.00
And What If I Don't            n=2   root=0.00 qual=1.00 bass=0.00
```
Pop tunes (Brown Eyed Girl, Always, Don't You Worry) cluster around
0.7-0.8 root; jazz standards (All The Things You Are, Angel, A Weaver Of
Dreams, Ablution) cluster around 0.0-0.4. **Not decomposed further this
session** — plausible confounds not yet separated: (a) jazz's denser
chord vocabulary is intrinsically harder for a 7-way quality head, (b)
jazz recordings pulled by bare-title `ytsearch1` may be a WRONG/different
arrangement more often (same title-ambiguity hazard documented in
`known_issues.md`'s cache-bug entry, partially mitigated by adding composer
to the query but not eliminated — no video-id verification exists), (c)
the shippable-priorities memory notes this project is deliberately
jazz-heavy for its frozen benchmark, so this gap is exactly the thing
that benchmark exists to catch. ❓ QUESTION FOR LOUIS: is the jazz-vs-pop
accuracy split here (0.0-0.4 vs 0.6-0.8) consistent with your own
listening impressions of the production tool on jazz standards, or does
it look more like a wrong-recording artifact to your ear on a couple of
these (Angel in particular, 0/8 rows, is worth a manual listen)?

**n is small (13 songs) — this is a first real measurement, not a final
verdict.** `scratchpad/production_audit_root_quality_bass.py` is reusable
for a larger run whenever there's time/network budget.

**Error analysis on the saved rows (disk/network-free, no new downloads)**:

Root errors are NOT random — they cluster exactly where music theory
predicts, not uniformly across all 11 wrong pitch classes:
```
+7 semitones (perfect 5th, V-for-I confusion):  25.4% of all root errors
+5 semitones (perfect 4th, IV-for-I):           14.2%
+4 / +3 semitones (relative maj/min 3rd):       11.6% / 10.1%
+10 semitones:                                  10.1%
```
This is a musically sane failure mode (confusing functionally-related
chords a 4th/5th/3rd away), not garbage — reassuring about WHERE the
system's attention is, even though the overall rate (§1) is weak on this
audio domain.

Quality confusion matrix (rows=true, cols=pred, 7-way):
```
true\pred    maj   min   dom  hdim   dim   aug   sus
      maj     88    25    25     4     0     0     0
      min     38   106    27     6     0     1     0
      dom     70    27   104    11     0     0     1
     hdim      4     6     4     2     0     0     0
      dim      1     0     1     0     0     0     1
      aug      0     0     0     0     0     0     0
      sus      4     5     2     1     0     0     0
```
True-quality distribution in this sample: maj 142, min 178, dom 213,
hdim 16, dim 3, aug 0, sus 12. **Rare qualities are essentially never
predicted** (dim: 3 true instances, 0 ever predicted as dim anywhere in
564 rows; sus: 12 true, only 2 predicted) — a textbook class-imbalance
symptom, the exact failure mode ChordFormer (§4 literature scan) targets
with a reweighted loss. maj/min/dom are heavily cross-confused with each
other too (dom→maj 70 times, maj→dom 25, min→maj 38) — on real (not
studio-clean) audio the tonic-vs-dominant-function distinction is
evidently much harder than on RWC.

## 2. SEGMENT/STRUCTURE — boundary quality of the LIVE mechanism is UNKNOWN; a better one exists but is OFF

The live `segment_source="nnls"` mechanism (chord-change points from
per-beat NNLS root-argmax flips) has **no directly-measured boundary-F1
on the current pipeline** — the last attempt (`docs/known_issues.md`,
"boundary-accuracy" entry) was blocked by disk space and fell back to a
documented-but-stale Billboard-path number (F1 0.01-0.67, different
corpus/frontend, not trustworthy as a current number).

What IS measured, on the same 100-song RWC set, is the **alternative**
`segment_source="musx"` lever (`_musx_boundary_segs()`, already built,
already in `infer_chords_v1`'s signature, **default OFF**): boundary-F1
**0.868 @0.25s / 0.901 @0.5s**, over-segmentation ratio 0.96 (does not
chatter). This is a real, already-implemented, already-tested improvement
sitting behind a flag that has never been flipped on.

**Recommendation (screen before implementing further, CLAUDE.md rule #2)**:
before any new structure work, cheaply test `segment_source="musx"` against
the SAME `aligned_corpus` sample used in §1 (root/quality accuracy should be
roughly unchanged since only segmentation timing changes, chord identity
comes from the same heads) — if boundary quality genuinely improves with no
identity regression, this is a much higher-leverage, lower-risk change than
any new model. **Not done this session** (time-boxed away from corpus work
into this audit) — flagged as the top candidate for the next work block.

Section-level structure (A/B/loop detection, distinct from chord-boundary
timing): `docs/known_issues.md` "chord-tone distance" entry (2026-07-21,
the current shipped section-clustering path, `HARMONIA_SECTION_REPR=
chordtone`) reports 53/58 corpus-exact matches — a different metric/thread
from the below, not re-run this session.

**Correction on the OTHER structure thread** (the symbolic learned-
similarity work `docs/handoff_2026_07_18_structure_detection.md` describes
as "CURRENT WINNING APPROACH", +0.010 V-measure over flat block8): a
LATER same-day entry in `known_issues.md` ("Task 3 (multi-seed
re-validation)... DOES NOT REPRODUCE at fresh seeds") found 9/9 fresh
seed-runs across 3 encoder variants gave margins of −0.002 to −0.007
(negative, not the originally-reported +0.010) — **the honest current
status is flat block8 and the learned encoder are statistically tied**,
not a validated win. The handoff doc itself is silent on this correction
(written before it landed) — worth knowing before picking that thread
back up. Flat block8 (V_F 0.68-0.70, zero ML dependency) remains the
practical baseline; the "adaptive agglomerative hierarchy" extension the
handoff describes as in-progress was ALSO already run and decisively
rejected (`known_issues.md` "Task 2... DECISIVELY re-falsifies the
adaptive agglomerative merge" — the greedy bottom-up merge itself
over-commits to spurious long-range matches regardless of similarity
source, tested with 2 different embedding types). **Net: this whole
symbolic-structure thread is currently at a dead end for beating block8**;
the literature scan's Buisson-et-al. suggestion (§4) is a genuinely
different angle (audio-native self-supervised, not symbolic-chord
metric-learning) rather than a variant of what's already been tried and
rejected here.

## 3. ALIGNMENT / TIMING — bar-grid phase, real measurement exists, not re-run

`beat_period_mode="bestfit"` (live default) was cross-validated against
madmom's independent RNN+DBN tracker on 14 cached real songs
(`docs/known_issues.md`, "BESTFIT beat period" entry, 2026-07-19):
**11/14 songs** land closer to madmom than the raw librosa scalar; mean
implied end-of-song drift falls **2.37 → 0.97 bars**. 6/14 songs show a
~2x octave-lock disagreement between trackers (a real, currently
unflagged-to-the-user failure mode). Not re-measured this session (already
a real, fairly recent, honestly-caveated number); flagged here only as
"still the best available evidence," not re-verified.

## Net summary

| scope | live default | measured this session? | headline number |
|---|---|---|---|
| root | nnls24 feat + musx bass | **yes, fresh** | 0.525 (13 songs, real audio) |
| quality | musx | **yes, fresh** | 0.532 exact / 0.713 family |
| bass/inversion | musx | **yes, fresh** | 0.520 |
| segment boundaries | nnls (NOT musx) | no (blocked by disk previously) | unknown; musx alternative measures 0.87-0.90 F1 but is OFF |
| structure (A/B/loop) | — | cited from prior session | 53/58 corpus-exact (different eval, not re-run) |
| alignment/bar-phase | bestfit | cited from prior session | 11/14 songs improved vs madmom, 6/14 octave-lock risk |

**Biggest actionable gap surfaced**: production's chord-CHANGE segmentation
uses the WORSE of two already-built mechanisms — `segment_source="musx"`
measures 0.87-0.90 boundary-F1 on RWC but sits opt-in/OFF while the live
default (`nnls`) has no current trustworthy F1 number at all. Testing and
potentially flipping this default is the clearest next step, independent
of anything corpus-scale-related.

**Biggest honesty-bar finding**: real-world (non-RWC) root/quality/bass
accuracy is roughly HALF the RWC bake-off numbers this project has
historically reported and compared variants against. Any future variant
comparison that only uses RWC risks optimizing for a benchmark that
doesn't represent what users actually get.

## 4. Literature scan — 2024-2025 ACR / structure-segmentation SOTA vs. our approach

Web search + a few paper fetches (WebSearch/WebFetch, not a systematic
survey — time-boxed). Comparing against what's actually live: `musx` is
**ISMIR2019** ("Large-Vocabulary Chord Recognition" / Chord Structure
Decomposition) — a 6-7 year old model by 2026. Worth knowing what's moved
since.

### Audio chord recognition
- **ChordFormer** (2025, conformer-based: CNN + transformer blocks,
  arxiv 2502.11840) reports **+2pp frame-wise / +6pp class-wise** accuracy
  over prior large-vocabulary baselines, specifically targeting the
  class-imbalance problem (rare qualities like aug/sus/hdim under-predicted)
  — the SAME failure mode this project's own `train_aligned_corpus_heads.py`
  already hand-tunes for via inverse-frequency `cw` class weights. Worth a
  read for whether their reweighting scheme beats naive inverse-frequency.
- **2025 MIREX Audio Chord Estimation leaderboard** (top systems YK1,
  MD1, wu-ensemble): **root ≈81-82%, majmin ≈78-81%, sevenths ≈62-66%** on
  Billboard2013/YAMAHA_Balanced/RWC-Popular. Two things this puts in
  perspective: (a) even 2025 SOTA on CLEAN benchmark audio sits well below
  100%, i.e. this task remains genuinely hard, not just under-invested-in
  here; (b) our own §1 fresh number (root=0.525 on messy real YouTube
  audio) is a fair distance below even THIS harder-than-RWC comparison —
  the gap is real, not just "RWC is easy." musx's own historical RWC
  numbers (this project's bake-offs, 0.7-0.9 depending on component) are
  roughly competitive with 2025 MIREX SOTA on similar clean benchmarks —
  **musx itself is not obviously stale on clean audio**; the gap this audit
  found is specifically a clean-vs-real-world generalization gap, not
  musx-vs-newer-model.
- **BACHI** (2025, arxiv 2510.06528, boundary-aware SYMBOLIC chord
  recognition via masked iterative decoding, evaluated on POP909/When-in-
  Rome/DCML) — directly targets the SAME oversegmentation problem as our
  §2 finding (live `segment_source="nnls"` vs the better-but-off `"musx"`),
  but for symbolic/MIDI input, not audio. Not directly portable (we work
  from audio), but the core idea — explicitly modeling boundary
  *locations* as a first-class decoding variable rather than deriving them
  as a side-effect of frame-level root changes — is exactly the shape of
  fix our own segmentation gap wants. Worth reading in more depth as a
  design reference for a from-scratch NNLS-side segmenter, if
  `segment_source="musx"` alone doesn't close the gap once tested.

### Structure segmentation
- **Buisson, McFee, Essid, Crayencour, "Self-Supervised Learning of
  Multi-level Audio Representations for Music Segmentation"** (IEEE/ACM
  TASLP 2024) — contrastive self-supervised learning of representations at
  MULTIPLE time scales simultaneously (not one fixed block size), applied
  to both boundary detection and section grouping. This is closely related
  to our own `docs/handoff_2026_07_18_structure_detection.md` "learned
  similarity" approach (item 4, the current winning symbolic-chord
  variant) but operates on raw AUDIO features via self-supervision rather
  than symbolic chord blocks via metric-learning on iReal labels — i.e. it
  solves a similar problem one representation-layer earlier. Genuinely
  worth trying: our structure work has so far assumed clean symbolic chord
  input is available (see handoff's "known unresolved gap" — nothing
  tested on real noisy predicted chords yet); an audio-native
  self-supervised representation sidesteps the noisy-symbolic-transfer
  problem entirely rather than needing it solved as a prerequisite.
- **EDMFormer** (2025, genre-specific self-supervised structure model) —
  narrower (electronic dance music, drops/buildups) than our jazz/pop
  scope, less directly relevant; noted for completeness only.
- **SSM-Net / joint SSM-loss + novelty-loss** (2023-2024 lineage,
  arxiv 2309.02243) — combines a self-similarity-matrix-based loss with a
  novelty-curve loss for boundary detection. Conceptually adjacent to what
  our `hcdf_boundary_probe.py` / `section_boundary_features.py` already
  explore (harmonic-change-detection-function-style novelty), but framed
  as a joint LEARNED loss rather than a hand-designed detection function —
  a natural next step if the hand-designed features plateau.

### Bottom line for this scan
Nothing found demands an immediate rewrite. Two genuinely actionable
leads, in priority order:
1. **Test `segment_source="musx"` for real** (§2) — already built,
   already measured well on RWC, zero new modeling risk, purely a
   configuration test. Should happen before any of the below.
2. **Buisson et al.'s multi-level self-supervised audio representations**
   as a way to unblock the structure-detection handoff's stated blocker
   (untested on real noisy audio) by not depending on clean symbolic
   chords as an input representation at all. This is a real research
   investment (weeks, not hours) — flag for a future dedicated session,
   not this one.
ChordFormer's reweighting scheme is a cheap thing to skim for the existing
class-imbalance handling but not clearly better than what's already done
here (informed guess, not verified against the paper's actual method
section in depth).

Sources:
- [ChordFormer: A Conformer-Based Architecture for Large-Vocabulary Audio Chord Recognition](https://arxiv.org/abs/2502.11840)
- [2025 MIREX Audio Chord Estimation Results](https://music-ir.org/mirex/wiki/2025:Audio_Chord_Estimation_Results)
- [BACHI: Boundary-Aware Symbolic Chord Recognition](https://arxiv.org/pdf/2510.06528)
- [Self-Supervised Learning of Multi-level Audio Representations for Music Segmentation (HAL)](https://hal.science/hal-04485065)
- [EDMFormer: Genre-Specific Self-Supervised Learning for Music Structure Segmentation](https://arxiv.org/html/2603.08759v1)
- [Self-Similarity-Based and Novelty-based loss for music structure analysis](https://arxiv.org/abs/2309.02243)
