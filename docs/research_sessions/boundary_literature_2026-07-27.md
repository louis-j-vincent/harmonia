# How the field solves chord BOUNDARY placement — literature review, 2026-07-27

Research-only session (no pipeline changes, no commits). Brief: our error analysis
(`docs/research_sessions/confusion_and_temporal_support_2026-07-26.md`) says 70.5% of
wrong-root duration is an *adjacent* chord's root, with a 2:1 bias toward holding the
previous chord; an oracle re-cut of our own labels on GT boundaries is **+6.56pp root**,
of which **+5.45pp** (≈83%) is recoverable by a jitter-only snap. So: right labels,
wrong places. What does the literature do about that?

Citation discipline: **[PAPER]** = claim reported by the source. **[OURS]** = my
inference or my reading of code on our disk. Every external claim has a URL + year +
venue.

> **Cross-session note (checked 2026-07-27, per CLAUDE.md delegation rules).** Two other
> sessions are live on this same problem and started ~10:16–10:19 CEST:
> `docs/research_sessions/musx_frame_posteriors_2026-07-27.md` (musx frame posteriors as
> the boundary lever) and `docs/research_sessions/dbn_segmentation_hybrid_2026-07-27.md`
> (fusion DBN as placer × musx as namer). The posteriors session has **independently
> found the same structural fact** ("`chord_recognition.py` computes `probs` … and then
> discards it"). What is *additional* here and not in either log: the decoder's unused
> **`use_beats` / `use_downbeats` / `beat_trans_penalty` path**, the **`layer_decode`**
> path, and the literature grounding for why a *re-decode* (rather than a novelty curve
> derived from the posteriors) is the move. Also note both sessions declare the vendored
> clone off-limits for edits — **no edit is needed**: with `cwd` set to the clone,
> `ChordNet` / `NetworkInterface` / `CQTV2` / `XHMMDecoder` are all importable, so
> posterior dumping and re-decoding can live entirely in a Harmonia-side module. This
> doc adds nothing to `harmonia/`; it is research only.

---

## PLAIN-LANGUAGE SUMMARY (read this first)

**Jargon key.** *Frame* = one short analysis slice of audio (ours ≈ 46 ms; the external
music-x-lab model's ≈ 23 ms). *Frame-level decoding* = decide a chord for every slice
independently-ish, then clean up. *Segment-level / event-level decoding* = decide
"a chord starts here and lasts this long, and it is X" as one joint decision.
*Self-transition penalty* = a fixed cost the decoder pays whenever it changes chord —
the knob that trades "jittery" against "sticky". *Duration model* = a learned
distribution over how long chords last, replacing that fixed cost. *Semi-Markov /
hidden semi-Markov (HSMM) / semi-CRF* = the standard machinery for "segment-level
decoding with an explicit duration model" — you search over `(start, end, label)`
triples instead of per-frame labels. *Segmentation Quality (SQ) / OverSeg / UnderSeg* =
the field's boundary metrics, computed from **directional Hamming distance**: for each
reference segment, find the estimated segment that overlaps it most and count the
non-overlap; do it once in each direction. One direction catches "you chopped one chord
into many" (over-segmentation), the other catches "you merged several chords into one"
(under-segmentation).

**Five things the literature actually says.**

1. **Our failure mode is a named, documented one, and its cause is architectural, not
   acoustic.** Korzeniowski & Widmer state it directly: a temporal model that must
   predict a symbol *at every audio frame* "focus[es] on short-term smoothing, and
   neglect[s] longer-term musical relations between chords, because, most of the time,
   the chord in the next audio frame is the same as in the current one"
   ([ISMIR 2018](https://arxiv.org/abs/1808.05335)). Their earlier paper is titled
   *On the Futility of Learning Complex Frame-Level Language Models for Chord
   Recognition* ([AES 2017](https://arxiv.org/abs/1702.00178)) — the finding is that at
   ~10 fps even an RNN degenerates into a smoother. A constant per-frame change penalty
   is mathematically a **geometric** duration prior (memoryless: the chance of changing
   is the same whether the chord has lasted 1 beat or 8), which is a bad model of real
   chord durations and systematically kills short chords. That is exactly our
   0.19-accuracy-under-1-beat cliff and our 2:1 hold-the-previous bias. **[OURS: the
   geometric-prior framing is standard HMM theory applied to our case; the "only
   smooths" finding is theirs.]**

2. **The remedy the field converged on is to decode at the SEGMENT level with explicit
   boundary variables** — not to add another novelty curve to the front end. The
   cleanest quantitative demonstration is brand new: Kim & Park
   ([ICASSP 2026, arXiv 2604.24386](https://arxiv.org/abs/2604.24386)) take the *same*
   transformer encoder and only change the output formulation from frame-wise to
   `(time-token, chord-token)` event sequences. On 471 pop songs, 5-fold: root
   **81.5 → 85.6** and over-segmentation score **81.4 → 92.4** [PAPER]. Same features,
   same encoder, +4.1pp root purely from how boundaries are represented at decode time.

3. **Making a model *predict* the boundary and then *condition* the label on it helps,
   and it is a small architectural change, not a new model.** Harmony Transformer
   (Chen & Su, [ISMIR 2019](https://zenodo.org/records/3527794)) predicts chord changes
   in the encoder and lets the decoder attend to the segmentation-informed sequence;
   the follow-up ([TISMIR 2021](https://transactions.ismir.net/articles/10.5334/tismir.65))
   reports SQ 71.93% → 75.50% for their improved variant and notes that "the worst HT
   variant even outperforms all the BTC variants in terms of chord segmentation
   quality" [PAPER]. BACHI ([arXiv 2510.06528](https://arxiv.org/abs/2510.06528), 2025)
   does the same in two explicit stages — a boundary head, then FiLM-conditioning the
   chord decoder on the predicted boundary; removing boundary detection + iterative
   decoding costs 2.0pp full-chord accuracy [PAPER].

4. **Nobody in ACR fixed boundaries with a better novelty curve.** Harte & Sandler's
   HCDF — the canonical harmonic-change detector — reports F ≈ 64.9% at a ±278 ms
   tolerance ([OFAI TR 2006](https://ofai.at/papers/oefai-tr-2006-13.pdf)). That is the
   ceiling of the whole family we already measured and refuted (chroma novelty,
   onset strength, HPSS). Our own 0.547 three-way beat-choice score is in the same
   regime. **This is confirmation that our refutations were correct, not bad luck** —
   we were competing in the category the field abandoned.

5. **The single biggest actionable finding is in our own tree, not in a paper.** The
   music-x-lab model we vendor already contains a decoder with a **beat- and
   downbeat-aware transition prior that we never switch on**, and its frame posteriors
   are computed and thrown away. See the next section — this is recommendation #1.

---

## ★ FINDING: what our vendored ISMIR-2019 model exposes that we do not use

Source read on disk:
`/Users/vincente/Documents/Projets Perso/Code/harmonia/harmonia/third_party/ISMIR2019-Large-Vocabulary-Chord-Recognition/extractors/xhmm_ismir.py`
and `.../chord_recognition.py`, `.../chordnet_ismir_naive.py`, `.../settings.py`,
`.../io_new/beatlab_io.py`. All **[OURS]** unless marked.

**Frame rate.** `DEFAULT_SR=22050`, `DEFAULT_HOP_LENGTH=512` → **43.07 fps, 23.2 ms
hop**. (Not 10 fps — worth noting, because it means the flat change penalty is applied
43 times a second.)

**Posteriors exist and are discarded.** `ChordNet.inference()` returns *six* softmax
arrays per frame — `(prob_triad, prob_bass, prob_7, prob_9, prob_11, prob_13)` —
averaged over a 5-model ensemble in `chord_recognition.py`. They are handed straight to
`XHMMDecoder.decode_to_chordlab(...)` and never written to disk. We read only the
argmax `.lab`. Dumping them is a `np.savez` one-liner in a file **we have already
patched once** (the fold-progress reveal, 2026-07-20).

**The decoder's transition prior, verbatim:**

```python
class XHMMDecoder():
    def __init__(self, diff_trans_penalty=30.0, beat_trans_penalty=(15.0, 45.0, 100.0),
                 template_file=..., use_bass=True, use_7=True, use_extended=True):
```

and in `decode(...)`:

```python
if beat_arr[t]:
    diff_trans = dp[t-1, dp_max_at[t-1]] - (self.diff_trans_penalty if beat_arr[t] == 1
                                            else self.beat_trans_penalty[beat_arr[t]-2])
    ...
else:
    dp[t, :] = same_trans + result_logprob[t, :]   # change FORBIDDEN at this frame
```

`beat_arr` comes from `__get_beat_arr(entry, length, use_beats, use_downbeats)`:

| `beat_arr[t]` | when | cost to change chord |
|---|---|---|
| `0` | frame is not on a beat (only if `use_beats=True`) | **∞ — change forbidden** |
| `1` | default everywhere when `use_beats=False` | `diff_trans_penalty` = **30.0** |
| `2` | downbeat (`beat_num == 1`), needs `use_downbeats=True` | `beat_trans_penalty[0]` = **15.0** |
| `3` | mid-bar beat (`beat_num == n_per_bar//2 + 1`, even meters only) | `beat_trans_penalty[1]` = **45.0** |
| `4` | any other beat | `beat_trans_penalty[2]` = **100.0** |

**`chord_recognition.py` calls `hmm.decode_to_chordlab(entry, probs, False)`** — i.e.
`use_beats=False, use_downbeats=False`. So `beat_arr` is **all ones**: chord changes are
allowed at *every 23 ms frame* at a *flat* cost of 30 nats, with no musical structure at
all. That is precisely the frame-level-smoothing regime Korzeniowski & Widmer describe
[PAPER], and it is the segmentation we inherit whenever we read musx's `.lab`.

Three unused surfaces follow directly:

- **`use_beats` / `use_downbeats`** turn the same decoder into a **beat-synchronous,
  metrically-graded** one. `io_new/beatlab_io.py` reads a plain TSV where column 0 is
  the time and column 2 is the beat-number-within-bar — so our Beat This! beat +
  downbeat grid can be fed in as a file with no code change to the third-party repo
  (`entry.append_file(path, BeatLabIO, 'beat')`).
- **`diff_trans_penalty` / `beat_trans_penalty` are free hyperparameters** we have never
  swept. Lowering them makes short chords survivable — the direct lever on our
  hold-the-previous bias.
- **`layer_decode()`** (decode triad+bass first, then decode extensions *constrained* to
  the chosen triad) is implemented and unused (`use_layer_decode=False` is hard-coded).
  Orthogonal to boundaries but directly aimed at our dominant quality failure (we drop
  the 7th, 61% of quality loss).

**Caveat [OURS]:** the shipped `beat_trans_penalty` ordering (downbeat 15 < mid-bar 45 <
other beats 100) is a *strong* metrical prior — on jazz with frequent beat-2/beat-4
changes, 100 may be too stiff. Treat all four numbers as sweep parameters, not as
constants. Also, the mid-bar tier only activates when `num_beat_per_bar` is even, and
`num_beat_per_bar` is inferred as `max(beat_num)` over the file — fragile on songs with
a pickup or a meter change.

---

## RANKED CANDIDATES FOR US

Each: **(a)** what it is · **(b)** evidence it helps boundaries · **(c)** cost for us ·
**(d)** interaction with what we already refuted.

### 1. Re-decode musx's own frame posteriors on OUR beat/downbeat grid, with the change penalty swept

**(a)** Dump `probs` (6 arrays, 43.07 fps) from `chord_recognition.py`; write our Beat
This! beats+downbeats as a BeatLab TSV; call
`XHMMDecoder.decode_to_chordlab(entry, probs, use_layer_decode=False, use_beats=True,
use_downbeats=True)` with `(diff_trans_penalty, beat_trans_penalty)` swept. The Viterbi
then chooses **boundary and label jointly**, on our grid, from the acoustic model's own
posteriors. This *is* joint segmentation+recognition — just done with machinery we
already own, with zero training.

**(b)** Direct: Kim & Park [PAPER] show +4.1pp root and +11pp over-seg from changing
only the boundary formulation on a fixed encoder
([arXiv 2604.24386](https://arxiv.org/abs/2604.24386), ICASSP 2026). Korzeniowski &
Widmer [PAPER] show that the frame-level flat prior is the thing that "only smooths"
([arXiv 1808.05335](https://arxiv.org/abs/1808.05335), ISMIR 2018). Masada & Bunescu
[PAPER] show that a *metrical-accent* feature is worth a lot to segment F-measure —
removing it drops F from 77.6% to 71.2% on BaCh
([TISMIR 2019](https://transactions.ismir.net/articles/10.5334/tismir.18)) — which is
what `use_downbeats` buys, as a soft prior inside the decode.

**(c)** Cheap. Patch = a `np.savez` of `probs` (file already patched once) + a
BeatLabIO-format writer + a sweep script. No training, no new data. Decode cost is
already measured in-tree at **~0.13 s/fold** for a full song, so a 5×5 penalty grid over
7 songs is minutes, not hours. Posteriors for a 4-min song ≈ 10 000 frames × ~250
classes float32 ≈ 10 MB — cache with the existing `data/cache/musx_infer/` discipline
and watch the disk (a disk-full incident is on record).

**(d) Not covered by any refutation, and specifically dodges the mechanism that nulled
the last attempt.** (i) "Taking the external model's own segmentation" (−1.97pp) used
musx's *default* decode = flat penalty 30 at every 23 ms frame with no beat information;
this is a different decode of the same posteriors. (ii) "Snapping to the metrical grid"
(−1.49pp raw) moved *already-decided* boundaries; here the grid restricts the *search
space* so the label can change when the boundary moves. (iii) Most important:
`_coalesce_labeled` erased recovered boundaries because both sides carried the same
label — a re-decode **cannot** produce that, because it emits a boundary only where the
decoded tag changes. **Recommended premise-screen before building (CLAUDE.md rule #2):
dump posteriors for ONE frozen song, decode at `use_beats=True/False` and 3 penalties,
and score boundary F1 + `mir_eval.chord.seg` against the frozen GT. If no setting beats
0.679 boundary F1, stop.**

### 2. Replace the flat change penalty with an explicit DURATION model over beats (semi-Markov)

**(a)** Korzeniowski & Widmer's factorisation: a **duration model** `P_D(change | history)`
that says only *whether* a chord ends, plus a **chord-level language model**
`P_L(next chord | previous chords)` that says *which* chord follows — decoded together
over a chord-time lattice (they use hashed beam search, `N_b=25` paths, hash over the
last 5 chord symbols so paths differing only in timing are pooled) [PAPER,
[ISMIR 2018](https://arxiv.org/abs/1808.05335)]. Equivalent formulation: a hidden
semi-Markov / semi-CRF decode on the **beat grid**, scoring `(start_beat, end_beat,
label)` triples.

**(b)** [PAPER] K&W report root 0.8123 → 0.8211, maj/min 0.7955 → 0.8047 and — the
relevant column — **Segmentation 0.8041 → 0.8139** on their best neural
duration+language pair vs a 2-gram + negative-binomial baseline. Modest (+1pp), but note
their acoustic model was already strong and frame-synchronous; our oracle says +6.56pp
is on the table. Masada & Bunescu [PAPER] show the segment-level formulation itself is
worth a lot on symbolic data (semi-CRF 83.2% vs HMM-perceptron 77.2% event accuracy on
BaCh, [TISMIR 2019](https://transactions.ismir.net/articles/10.5334/tismir.18)).

**(c)** Moderate but tractable, **and we have an unusually good prior source**: 139
iReal charts give an empirical distribution of chord durations **in beats** (jazz/pop
durations are extremely peaked at 2/4/8 beats — nothing like geometric). Turn it into a
per-beat hazard `h(d) = P(change | held d beats)` and it drops into the existing DP as a
duration-dependent penalty. On the beat grid the semi-Markov DP is `O(T · C · L)` with
T ≈ 400 beats, C ≈ 200 labels, L ≈ 32 max beats — ~2.6M ops, milliseconds. **[OURS: the
"use iReal charts as the duration prior" idea is ours; the factorisation is theirs.]**
Estimating the duration prior needs *no* audio alignment — chord durations in beats are
readable straight off the charts — so it is not blocked by our 7-song limit.

**(d)** Orthogonal to every refuted item. Note it *subsumes* the "snap to grid"
refutation properly: a hazard function makes a change at beat 4 of a 4-beat chord cheap
and a change at beat 1 of a 1-beat chord expensive, which is a *soft, learned* version
of the hard snap that failed.

### 3. Add `layer_decode` (triad-first, then constrained extensions) — quality lever, ships with #1

**(a)** Already implemented in `XHMMDecoder`: decode triad+bass with its own Viterbi,
then decode the full chord with `triad_restriction` forcing consistency. **[OURS]**

**(b)** No paper number isolates it, but the *decomposition* principle is exactly BACHI's
"iterative ranking of root, quality, bass" and Kim & Park's SPLIT tokenisation, both of
which [PAPER] report gains on rare/complex qualities
([arXiv 2510.06528](https://arxiv.org/abs/2510.06528);
[arXiv 2604.24386](https://arxiv.org/abs/2604.24386)). Kim & Park's SPLIT beats MERGE on
sevenths, 75.7 → 77.1 [PAPER].

**(c)** Near-zero: flip one boolean in the call, re-run. Do it inside the #1 sweep.

**(d)** Independent of boundaries; touches the "we drop the 7th" 61%-of-quality-loss
problem and the dormant `seventh_upgrade` brick. Must be measured with the brick OFF to
avoid double-counting.

### 4. A supervised beat-level BOUNDARY head with a *regression* target, not a classifier

**(a)** For each beat, predict (i) `P(boundary)` and (ii) a **continuous offset** to the
true boundary within the beat. Two tricks worth importing: Ullrich/Schlüter/Grill smear
the boundary target over neighbouring frames rather than using a one-hot target, and
peak-pick with a moving threshold ([ISMIR 2014](https://grrrr.org/pub/ullrich_schlueter_grill-2014-ismir.pdf));
Kong et al. regress **onset/offset times** instead of classifying frames, which gives
sub-frame resolution *and* — their key claim — robustness to misaligned labels
([TASLP 2021, arXiv 2010.01815](https://arxiv.org/abs/2010.01815)) [PAPER].

**(b)** [PAPER] Kong et al.'s regression targets are what let them beat frame-classification
baselines on MAESTRO at sub-frame resolution; label-misalignment robustness is stated
explicitly. **[OURS]** That robustness property is the reason this is the *right* shape
for our 139-song weakly-aligned set: the targets there are systematically off by a bit,
and a regression head with smeared targets tolerates that where a hard classifier does
not.

**(c)** Highest cost of the four. Needs training data we mostly lack: 7 hand-verified
songs is not a training set, so this is **flagged as data-blocked** unless the 139-song
weakly-aligned iReal/YouTube set is first turned into usable beat-level targets. The
smart order is: do #1 and #2 first, then use the improved decode to *self-label* the 139
(pseudo-labelling; see the table entry for arXiv 2602.19778, where a BTC teacher over
1000 h of unlabelled audio gets a student to 98% of teacher performance [PAPER]).
Features would be cheap: beat chroma, `Δ`chroma across the beat, metrical position,
musx posterior novelty `1 − ⟨p_t, p_{t+1}⟩`, harmonic-rhythm phase.

**(d)** This is the "supervised beat-level boundary classifier" our own 2026-07-26 log
already flagged as one of two live candidates — the literature contribution here is
*regression + target smearing* instead of hard classification, which is the part we had
not specified. It does **not** re-litigate onset-strength/HPSS: those were tested as
*unsupervised novelty curves*, not as features into a supervised head.

### 5. (Deferred) Swap the whole chord model for a jointly-segmenting one

Harmony Transformer ([repo](https://github.com/Tsung-Ping/Harmony-Transformer)), the
Kim & Park seq2seq ([repo](https://github.com/KimLeekyung/ACR_seq2seq)), ChordFormer
([arXiv 2502.11840](https://arxiv.org/html/2502.11840)). **Not recommended now.** Our
own measurement says our whole chord stage is worth **+0.2pp root over printing musx's
`.lab`** — the labels are not the bottleneck, the placement is, and #1–#3 attack
placement without a model swap. ChordFormer releases **no weights** [PAPER: no repo URL
in the paper]; HT ships code but its audio path is Billboard-preprocessing-specific;
ACR_seq2seq is brand new (ICASSP 2026) and unvalidated by us. Also CLAUDE.md rule #6:
a component swap of this size changes everything downstream at once. Revisit only if
#1–#3 leave a large gap.

---

## MEASUREMENT: how the field scores boundaries (adopt this)

We currently report boundary F1 against change-times. The field's standard for chord
segmentation is **directional Hamming distance**, exposed in `mir_eval` — and
**`mir_eval` 0.8.2 is already installed in our venv** with
`mir_eval.chord.seg / overseg / underseg` available [OURS, verified].

- `overseg` — 1 minus the directional Hamming distance reference→estimate. Low = we
  chopped one true chord into several.
- `underseg` — the other direction. Low = we merged several true chords into one.
- `seg` — `min(overseg, underseg)`, the MIREX `MeanSeg` score, so a good score in one
  direction can't hide a bad one in the other.
  ([mir_eval docs](https://mir-eval.readthedocs.io/latest/api/chord.html);
  [MIREX 2019 ACE task](https://www.music-ir.org/mirex/wiki/2019:Audio_Chord_Estimation))

**Why this matters for us specifically [OURS]:** `seg` is duration-weighted and fairly
insensitive to small jitter, while boundary-F1 at a tight tolerance is *very* sensitive
to it. Reporting **both** cleanly separates our two error components — the ~1pp of
genuinely missing boundaries (shows up in `underseg`) from the ~5.45pp of misplaced ones
(shows up in boundary-F1 at ±150 ms but barely in `seg`). That is the natural instrument
for our 83/17 split, and it lets us compare against published numbers.

**Reference values to place ourselves against** [PAPER]: Kim & Park report Under/Over/Mean
of 90.1 / 85.9 / 84.6 for BTC and 89.8 / 92.9 / 88.6 for their best model on 471 pop
songs ([arXiv 2604.24386](https://arxiv.org/abs/2604.24386)); Korzeniowski & Widmer
report Segmentation 0.8041 → 0.8139 on Billboard
([arXiv 1808.05335](https://arxiv.org/abs/1808.05335)); Chen & Su report SQ 71.9–75.5%
on BPS-FH ([TISMIR 2021](https://transactions.ismir.net/articles/10.5334/tismir.65)).
Note these are on *different* corpora and vocabularies — use them for order of magnitude,
not as a leaderboard.

**Tolerances used in the literature** [PAPER]: Harte & Sandler's HCDF evaluation used
±3 frames ≈ **278 ms** ([OFAI TR 2006](https://ofai.at/papers/oefai-tr-2006-13.pdf));
music-structure boundary detection standardised on **±0.5 s** and ±3 s
([ISMIR 2014](https://grrrr.org/pub/ullrich_schlueter_grill-2014-ismir.pdf)). **[OURS]**
For us a *beat-relative* tolerance (±0.5 beat) is more meaningful than a fixed
millisecond window, because our songs span 65–160 BPM — but report a fixed window too so
the numbers are comparable to published ones.

---

## PAPERS & REPOS

| Work | Venue / year | Link | Why it matters to us |
|---|---|---|---|
| Kim & Park, *Event-based Sequence Modeling … Oversegmentation Minimization* | ICASSP 2026 (arXiv 2604.24386) | [abs](https://arxiv.org/abs/2604.24386) · [code](https://github.com/KimLeekyung/ACR_seq2seq) | **Best single number in this review**: frame→event formulation alone, same encoder, root 81.5→85.6, OverSeg 81.4→92.4 |
| Korzeniowski & Widmer, *Improved Chord Recognition by Combining Duration and Harmonic Language Models* | ISMIR 2018 | [abs](https://arxiv.org/abs/1808.05335) · [pdf](http://ismir2018.ircam.fr/doc/pdfs/300_Paper.pdf) | The duration-model recipe (#2); names our exact failure mechanism; reports a Segmentation column |
| Korzeniowski & Widmer, *On the Futility of Learning Complex Frame-Level Language Models* | AES Semantic Audio 2017 | [abs](https://arxiv.org/abs/1702.00178) | Why frame-level temporal models degenerate into smoothers |
| Chen & Su, *Harmony Transformer* | ISMIR 2019 | [zenodo](https://zenodo.org/records/3527794) · [code](https://github.com/Tsung-Ping/Harmony-Transformer) | Chord-change head feeding a segmentation-informed decoder |
| Chen & Su, *Attend to Chords* (HT/HT\*) | TISMIR 2021 | [article](https://transactions.ismir.net/articles/10.5334/tismir.65) | SQ (directional Hamming) numbers 71.9→75.5; HT > BTC on segmentation quality |
| Park et al., *BTC: Bi-directional Transformer for Chord Recognition* | ISMIR 2019 | [abs](https://arxiv.org/abs/1907.02698) · [code](https://github.com/jayg996/BTC-ISMIR19) | Already in our scratchpad; CQT hop **2048** (≈10.8 fps), 10 s windows / 5 s overlap; no mandatory post-processing |
| BACHI, *Boundary-Aware Symbolic Chord Recognition* | arXiv 2510.06528 (2025) | [abs](https://arxiv.org/abs/2510.06528) · [html](https://arxiv.org/html/2510.06528v1) | Explicit boundary head → FiLM-conditions the label decoder; ablation −2.0pp without it. Symbolic (MIDI) only |
| Masada & Bunescu, *Chord Recognition in Symbolic Music: A Segmental CRF Model* | TISMIR 2019 | [article](https://transactions.ismir.net/articles/10.5334/tismir.18) · [arXiv](https://arxiv.org/abs/1810.10002) · [code](https://github.com/kristenmasada/chord_recognition_semi_crf) | Semi-CRF joint segmentation+labelling; segment-level features (purity, coverage, bass, bigram, **metrical accent** — removing accent: F 77.6→71.2) |
| Kong et al., *High-Resolution Piano Transcription by Regressing Onset and Offset Times* | TASLP 2021 (arXiv 2010.01815) | [abs](https://arxiv.org/abs/2010.01815) | Regression targets for boundary times → sub-frame resolution + robustness to misaligned labels (→ our 139 weak songs) |
| Ullrich, Schlüter & Grill, *Boundary Detection in Music Structure Analysis using CNNs* | ISMIR 2014 | [pdf](https://grrrr.org/pub/ullrich_schlueter_grill-2014-ismir.pdf) | Target smearing + moving-threshold peak-picking; the ±0.5 s tolerance convention |
| Harte & Sandler, *Detecting Harmonic Change in Musical Audio* | AMCMM 2006 | [pdf](https://ofai.at/papers/oefai-tr-2006-13.pdf) | HCDF ceiling F ≈ 64.9% @ ±278 ms — bounds the whole novelty-curve family we refuted |
| Akram et al., *ChordFormer* | arXiv 2502.11840 (2025) | [html](https://arxiv.org/html/2502.11840) | Conformer, same 6-way chord decomposition as musx; root 84.69 vs Jiang 2019's 83.39; **no weights released**; no segmentation metric |
| Phan et al., *Enhancing ACR via Pseudo-Labeling and Knowledge Distillation* | arXiv 2602.19778 (2026) | [abs](https://arxiv.org/abs/2602.19778) · [html](https://arxiv.org/html/2602.19778) | BTC teacher → student on 1000 h unlabelled reaches 98% of teacher — the template for exploiting our 139 weak songs |
| Micchi et al., *SERENADE: Human-in-the-loop ACE* | IEEE 2023 (arXiv 2310.11165) | [abs](https://arxiv.org/abs/2310.11165) | Autoregressive in-painting of sparse human corrections; ROI 1.18 labels fixed per correction — matches our "corrections → automation" ledger |
| Kim & Nam, *All-In-One Metrical and Functional Structure Analysis* | WASPAA 2023 (best student paper) | [MAC lab](https://mac.kaist.ac.kr/pubs.html) | Joint beat/downbeat/structure on demixed audio — alternative structure-boundary source if we revisit section priors |
| `mir_eval.chord` (`seg`/`overseg`/`underseg`) | — | [docs](https://mir-eval.readthedocs.io/latest/api/chord.html) · [MIREX ACE](https://www.music-ir.org/mirex/wiki/2019:Audio_Chord_Estimation) | The metric to adopt; already installed (0.8.2) |
| music-x-lab ISMIR-2019 model (ours) | ISMIR 2019 | [repo](https://github.com/music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition) | The vendored model — see the unused-surfaces section above |

---

## NOT APPLICABLE / ALREADY REFUTED — do not re-litigate

**Refuted on our data (from `confusion_and_temporal_support_2026-07-26.md`,
`chord_identity_final_2026-07-24.md`, `chord_change_timing_2026-07-20.md`,
`boundary_snap_2026-07-20.md`, `chord_beat_grid_unification_2026-07-22.md`) — the
literature is consistent with these refutations, it does not overturn them:**

- Chroma-novelty curves for boundary choice (0.547 three-way) — HCDF's own published
  ceiling is F ≈ 64.9%, so this family is *expected* to be weak. Confirmed dead.
- Onset strength (0.367) / HPSS-percussive onsets (0.370) / onset+chroma (0.482).
  Onset energy is blind to harmonic change. Dead **as unsupervised novelty curves**;
  they remain admissible as *features* into a supervised head (#4).
- Majority / duration-majority voting inside a span (dead on premise — the midpoint
  label already covers 89% of its span).
- Snapping decoded boundaries to the metrical grid (−1.49pp raw, +0.27pp gated);
  per-bar pooling (−20.59pp), half-bar (−2.39pp), strong-only (−1.62pp). **Note:** #1
  and #2 are *not* this — they put the metrical information *inside* the decode as a
  soft prior over the search space, which is what the semi-CRF metrical-accent feature
  does (and it is worth 6.4pp of segment F there [PAPER]).
- Taking musx's own segmentation (−1.97pp) and union(NNLS, musx) (−1.97pp) — both nulled
  by `_coalesce_labeled` erasing same-label cuts. **Note:** this was musx's *default*
  (flat-penalty, 23 ms-frame, beat-blind) decode; #1 is a different decode of the same
  posteriors, and by construction cannot hit the `_coalesce_labeled` failure.
- Within-beat frame trimming / shifting / weighting (≈ 0).
- musx onset-hint retiming (−0.44pp); gated under-segmentation repair (−0.34pp);
  DeepChroma peak-snap (falsified 2026-07-20 — root cause was fold-reconstruction drift).
- Global time-shift (optimum at 0 s — the grid is already aligned).

**Not applicable from the literature:**

- **BACHI** and **Masada & Bunescu semi-CRF** are **symbolic-only** (MIDI piano roll /
  MusicXML). Their *architectures* transfer; their *models* do not. BACHI's code was
  "to be released" as of the preprint.
- **ChordFormer** — no pretrained weights released; retraining needs the 1217-song
  Humphrey & Bello collection. Data-blocked.
- **Harmony Transformer / BTC / ACR_seq2seq as replacements** — deferred (#5). Our chord
  stage is only +0.2pp over raw musx, so a label-model swap is not where our loss is.
  BTC is already in our scratchpad and was measured as a boundary-*recall* lever
  (+15pp change-recall) but null end-to-end.
- **Anything requiring supervised training at scale** (#4's classifier, HT/ChordFormer
  fine-tuning, K&W's neural duration/language models) — 7 hand-verified songs is an
  evaluation set, not a training set. The 139-song weakly-aligned set is the only path,
  and pseudo-labelling ([arXiv 2602.19778](https://arxiv.org/abs/2602.19778)) is the
  documented way to use it. Flag any such proposal as data-blocked until that set has
  usable beat-level targets.
- **Novelty/SSM music-structure boundary detectors** (Ullrich et al., All-In-One) target
  *section* boundaries at 10–60 s scale, not chord boundaries at 1–4 beats. Only their
  *tricks* (target smearing, offset regression, moving-threshold peak-picking) transfer;
  the detectors themselves do not.

---

## SUGGESTED ORDER (with stopping criteria)

1. **Premise-screen #1 on ONE frozen song** (~1 h): dump musx posteriors, decode at
   `use_beats ∈ {False, True}` × `use_downbeats ∈ {False, True}` ×
   `diff_trans_penalty ∈ {10, 20, 30}`. Metric: boundary F1 vs frozen GT + `mir_eval`
   `seg`/`overseg`/`underseg`. **Continue only if some setting beats the current
   final-chart boundary F1 of ~0.679.**
2. If yes → sweep on all 7 frozen songs, scored with `harmonia/eval/accuracy_score.py`
   against the **raw-musx baseline** (root 0.7347), not only our own 0.7367.
   **Ship criterion: ≥ +2pp pooled root with zero per-song regressions** (the Brick-A
   bar). Add `layer_decode=True` as a second axis in the same sweep.
3. If #1 saturates below the +5.45pp jitter prize → build the beat-grid duration prior
   from the 139 iReal charts (#2), which needs no audio and no alignment.
4. Only then consider #4 (supervised boundary regression head), and only after the 139
   set has pseudo-labels.

**Watch out for the Occam amplifier**: our own log records that a 0.25-beat change to
the pooling window swings `close_to_you` by −11.5pp *through the Occam post-pass alone*.
Run every boundary experiment with `HARMONIA_OCCAM_POSTPASS=0` as a control alongside
the default, or the measurement is uninterpretable.
