# Music STRUCTURE (section) analysis — literature review + code trials, 2026-08-04

Brief: our sections have the right *content* but the wrong *starts* (Louis: « tous les
blocs sont bons, le souci c'est que des fois on ne commence pas les blocs au début »).
Measured: 64% of same-letter occurrence pairs need a 1–4 **bar** shift to line up
(n=162, 13 songs), run coverage 53–72%, and the ±2-bar failsafe in
`harmonia_min/sections.py` cannot reach a 3–4 bar error.

Citation discipline: **[PAPER]** = claim reported by the source. **[OURS]** = measured
here or read off code on our disk.

---

## SUMMARY (read this, the rest is the audit trail)

**Recommendation: place section starts from the CHORD sequence, not the chroma.** Measured
on 694 real section starts (80 Billboard tracks, annotators' own bar grid and chroma):

| cue | lands on the exact right bar |
|---|---|
| what we run today (chroma checkerboard novelty) | **34.7%** |
| chord-string repeat matching | **75.2%** |

And it does not need our chords to be correct — with 45% of chords mis-heard *the same way
every time* (which is how recognisers actually fail) it still scores 78%. It only asks
"does this 8-bar chord string come back later", never "is it right".

Three other things worth knowing:

1. **We have a free 890-track bar-level evaluation set and nobody noticed.** Billboard
   ships precomputed NNLS `bothchroma` (our exact substrate) *and* bar-notated chord
   annotations. No audio, no download. Every structure measurement in this repo so far was
   on 6–13 songs.
2. **Nobody in the field has solved our problem.** Since Goto's RefraiD (2006), no method
   enforces that repeated occurrences of a section agree on where they start — and
   `mir_eval` has no metric for it either. This is a real hole, not a lookup failure.
3. **The best off-the-shelf candidate is `as-seg` / CBM** (TISMIR 2023) — the only
   published segmenter that works natively on a bar grid, which is exactly what we need
   and what we can feed it. It still has no repeat-consistency constraint.

Deprioritise All-In-One: despite predicting downbeats itself, it does not snap sections to
them — it would reproduce our exact bug.

---

## 0. What is NEW versus the repo's prior literature note

The prior note is `docs/research_sessions/boundary_literature_2026-07-27.md`. It is about
**chord** boundaries (1–4 beats), and it explicitly parked this topic:

> "Novelty/SSM music-structure boundary detectors (Ullrich et al., All-In-One) target
> *section* boundaries at 10–60 s scale, not chord boundaries at 1–4 beats. Only their
> *tricks* transfer; the detectors themselves do not."

So section-level structure was **never reviewed** — the only prior traces are two
one-line mentions inside `docs/known_issues.md` (Buisson et al. 2024 self-supervised
multi-level representations, "flagged for a future session"; Kim & Nam All-In-One 2023,
"alternative structure-boundary source if we revisit section priors"). Neither was
followed up, no code was ever tried.

Everything below is new. The two things I would most want Louis to read are §1 (a
measurement that changes the ranking) and §2 (what actually runs).

---

## 1. ★ THE MEASUREMENT THAT DECIDES THE RANKING [OURS]

Before reading any paper I screened the premise (CLAUDE.md rule #2) on ground truth.

### 1.1 A new, free, 890-track bar-level evaluation set

`mirdata.initialize('billboard')` gives 890 McGill Billboard tracks. Two facts I had not
seen recorded anywhere in this repo:

1. **Every track ships precomputed NNLS `bothchroma`** (`t.chroma`, shape `(T, 25)` =
   time + 12 bass + 12 treble) — *the exact substrate `harmonia_min/sections.py` runs
   on*. 890/890 present on disk. **No audio download, no feature extraction.**
2. **The `salami_chords.txt` annotation is BAR-notated.** Lines look like
   `46.537891156 <TAB> B, bridge, | G:min7 | C:maj/5 | Bb:maj/5 | F:maj | G:maj |`, plus
   a `# metre:` header and `xN` repeat marks. Counting the `|` gives the bar count of
   each line; linear interpolation between line timestamps gives a **bar grid**.
   **112/120 sampled tracks yield a self-consistent grid** (>80% of lines within 12% of
   the song's median bar length). Median bar 2.05 s.

Net: we can measure *bar-accurate section-start placement* on ~830 real tracks, today,
for free. Parser: `scripts/`-ready code in the scratchpad
(`bb_bars.py`, ~40 lines). This is the missing instrument — every prior structure
measurement in this repo was on 6–13 of our own songs.

### 1.2 Checkerboard novelty cannot place a section start to the bar — even on GT

694 section starts, 80 tracks, using the **annotators' own bar grid** and the
**annotators' own chroma**, half-bar pooling + σ=1.5 blur + 16-half-bar checkerboard —
i.e. `sections.py`'s novelty path with every upstream error removed:

| cue | exact bar | within ±1 bar | median abs error |
|---|---|---|---|
| checkerboard novelty on NNLS chroma | **34.7%** | 42.7% | 1.50 bars |
| naive chroma lag/repeat cue (my quick version) | 17.1% | 29.1% | 3.00 bars |
| **chord-string repeat matching (symbolic)** | **75.2%** | **80.8%** | **0.00 bars** |

The novelty error histogram is bimodal: a peak at 0 and heavy mass saturating the ±4-bar
search window — "attracted to the neighbouring hypermetric position", which is exactly
Louis's complaint, reproduced under oracle conditions.

**Read this as a ceiling.** 34.7% is what the chroma-novelty family gets with a perfect
bar grid and perfect GT. No amount of peak-picking, post-pass or threshold tuning on that
substrate reaches bar-accurate starts. That kills "add a better novelty curve" as a
direction — the same conclusion the 2026-07-27 note reached for chord boundaries, now
measured for section boundaries.

*Caveat:* my bar grid is linearly interpolated inside each annotation line, so a track
with a rit. or a pickup contributes some GT error. That penalises all three cues equally,
so the *ranking* is safe; the absolute 75% is probably a slight over-estimate.

### 1.3 ★★ The symbolic path is immune to chord-recognition errors

The 75.2% used **oracle** chords, so I degraded them two ways.

| noise model | 15% | 30% | 45% |
|---|---|---|---|
| **iid** per-bar corruption (each bar independently replaced) | 45.4% | 38.6% | 37.6% |
| **systematic** per-chord confusion (a chord is mis-heard the *same way* everywhere) | **75.9%** | **76.8%** | **78.1%** |

Systematic error costs **nothing** — it even drifts up (noise, n=694). The reason is
structural: the method asks *"does this 8-bar string repeat elsewhere in the song?"*, never
*"is this string correct?"*. A consistent mistake is a perfectly good landmark.

Real recogniser errors are overwhelmingly of the systematic kind — musx hears the same
passage the same way on every chorus. **The iid row is the wrong noise model** and is the
pessimistic bracket only. Practically: the symbolic path does not need our chord accuracy
to improve, and would work today on musx posteriors.

**This is the single most decision-relevant number in the review**: our best acoustic cue
gets 35% of section starts on the right bar; a chord-repeat cue gets ~75%, and does not
care that our chords are wrong.

### 1.4 Confound checked: it is not an artefact of the annotation format

Billboard's chord labels and its section starts come from the *same* file, so chord changes
sit exactly on bar lines by construction — whereas our chords come from audio and are
misaligned to the bar. I simulated that by sampling each bar's chord at a jittered time
instead of the bar midpoint:

| sampling jitter | ±0 (midpoint) | ±0.25 bar | ±0.5 bar (anywhere in the bar) |
|---|---|---|---|
| exact-bar placement | 75.2% | 72.3% | **70.6%** |

Destroying the chord-to-bar alignment entirely costs 4.6 pp. The effect is real, not an
annotation artefact.

Scripts (scratchpad, not committed): `bb_bars.py`, `bb_novelty_placement.py`,
`bb_symbolic_placement.py`, `bb_symbolic_noise.py`, `bb_symbolic_syserr.py`.

---

## 2. WHAT THE FIELD HAS, AND THE ONE STRUCTURAL GAP

**The gap, stated once:** no paper found targets our defect. Every benchmark in music
structure analysis scores boundary *detection* — "is there a boundary within ±0.5 s (or
±3 s) of time T". None scores *"does occurrence 2 of the chorus start at the same point
in the loop as occurrence 1"*. `mir_eval.segment` has no repeat-consistency metric
either. So: we are not going to buy this off the shelf, and any candidate must be judged
on whether its *shape* fits the problem, not on its published F-measure.

Two consequences. (i) A method reporting HR.5F ≈ 0.70 is reporting hits inside a
half-second window; at a 2.05 s median bar that window is a quarter of a bar in the best
case and, at 3 s tolerance, **more than a whole bar** — so ±3 s numbers say nothing about
us. (ii) The repeat-consistency constraint is ours to add.

### The bar-native exception

**CBM — Marmoret, Cohen & Bimbot, "Barwise Music Structure Analysis with the Correlation
Block-Matching Segmentation Algorithm", TISMIR 6(1) 2023**
([arXiv:2311.18604](https://arxiv.org/abs/2311.18604) ·
[TISMIR](https://transactions.ismir.net/articles/10.5334/tismir.167) · code on GitLab
`a23marmo/autosimilarity_segmentation`, pip `as-seg`). The only method found that
**samples time at the bar** — it takes downbeats as input and emits bar-indexed
boundaries by construction, instead of emitting seconds that get rounded to bars later.
That is our exact shape, and we already have SOTA downbeats. Reported F0.5s ≈ 39.7% /
43.3% and F3s ≈ 60.4% / 65.0% (SALAMI-test / RWC-Pop) [PAPER] — *competitive with, not
better than*, supervised SOTA; the paper's own claim is "as good, using only bar
positions". No phase-consistency enforcement in it [checked, absent].

Supporting evidence for going bar-native: **Marmoret 2026, "Unsupervised Evaluation of
Deep Audio Embeddings for Music Structure Analysis"**
([arXiv:2603.27218](https://arxiv.org/abs/2603.27218)) benchmarks 9 pretrained embeddings
× 3 segmenters (Foote / spectral clustering / CBM) on RWC-Pop, SALAMI, Harmonix.
**CBM beats spectral clustering in 33 of 36 conditions** [PAPER], and nearly a third of
the deep embeddings *underperformed* a hand-crafted baseline. It also warns that
annotation "trimming" conventions swing reported scores by >10 pp — do not compare across
papers casually.

### The current supervised best

**SongFormer — Hong et al., "Scaling Music Structure Analysis with Heterogeneous
Supervision"** ([arXiv:2510.02797](https://arxiv.org/abs/2510.02797) ·
[github.com/ASLP-lab/SongFormer](https://github.com/ASLP-lab/SongFormer) ·
[weights on HF](https://huggingface.co/ASLP-lab/SongFormer)). Fuses MuQ + MusicFM
self-supervised embeddings into a 4-layer transformer. Harmonix HR.5F **0.703** vs
LinkSeg 0.630 vs All-In-One 0.596; RWC-Pop (unseen) HR.5F 0.650 / HR3F 0.800 [PAPER].
Code, weights, a 14k-song training set (SongFormDB) and a 300-song expert benchmark
(SongFormBench) are all released. This is the genuine 2025 successor to Ullrich/Grill &
Schlüter, and the strongest verified number in the review. It emits **seconds**, not bars.

**All-In-One — Kim & Nam, WASPAA 2023**
([arXiv:2307.16425](https://arxiv.org/abs/2307.16425) ·
[github.com/mir-aidj/all-in-one](https://github.com/mir-aidj/all-in-one)). Jointly
predicts beats, downbeats and sections on demixed audio; weights released, pip
installable; Harmonix HR.5F 0.660 [PAPER]. **Important negative:** despite predicting
downbeats in the same model, it does **not** snap section boundaries to them — segments
come from peak-picking a 24 s sliding window independently of the beat branch. So *it does
not fix our phase problem by construction*. Last code push 2024-05-09 — frozen.

### Everything else, briefly

- **Spectral clustering on the SSM (McFee & Ellis, ISMIR 2014,
  [PDF](https://brianmcfee.net/papers/ismir2014_spectral.pdf))** — the `Scluster`
  baseline everywhere. Still fine, but 2026 evidence says it is specifically weak *at bar
  resolution*, and CBM beats it 33/36 (above). Ships inside `librosa` as a recipe.
- **Serrà's Structure Features / time-lag matrix (IEEE TMM 2014)** — the ancestor of our
  own tiling-run idea. Maintained-ish reimplementation
  [wayne391/sf_segmenter](https://github.com/wayne391/sf_segmenter), last push 2023-02.
  **Searched specifically for post-2020 improvements to the time-lag representation:
  none exist.** The lineage was absorbed as one *input channel* to CNNs (that is literally
  what Grill & Schlüter 2015 do) rather than continuing on its own.
- **Buisson et al., TASLP 2024, self-supervised multi-level representations**
  ([DOI 10.1109/TASLP.2024.3379894](https://ieeexplore.ieee.org/document/10477022/)) —
  the lead flagged in `known_issues.md` and never followed up. Real paper; **no code repo
  found**. Same group's **LinkSeg** (ISMIR 2024, MSA as pairwise same-section link
  prediction + graph attention) — also **no repo found**, though others benchmark it.
  Both unverified for reproducibility.
- **`msaf`** — 555 stars, **not archived**, last release v0.1.80 (2023-06) but a PR "Fix
  installations for Python 3.12" merged **2025-07-09** and 3.11/3.12 added to the test
  matrix 2025-06. So the assumption "msaf is dead on 3.12" is **wrong** — it is
  low-activity but current. No new algorithms will land in it.
- **Toyama et al., ICASSP 2026** ([arXiv:2512.17209](https://arxiv.org/abs/2512.17209)) —
  masked-LM-style SSL encoders are the effective ones for structure. **Zhang et al.,
  WASPAA 2025** ([arXiv:2507.13572](https://arxiv.org/abs/2507.13572)) — widening
  foundation-model context for single-pass full-song inference.

**Could not verify** (flagged honestly): result tables for Wang et al. ISMIR 2021
supervised metric learning ([arXiv:2110.09000](https://arxiv.org/abs/2110.09000)) — PDF
text extraction failed; code/weight release for Buisson TASLP 2024 and LinkSeg; Grill &
Schlüter 2015's exact headline (~0.54 F0.5s, secondhand only).

---

## 3. ★ THE GAP IN THE LITERATURE (searched for specifically, twice, independently)

Two agents searched this angle separately and reached the same answer.

**Goto's RefraiD** (*A Chorus-Section Detection Method for Musical Audio Signals*,
[IEEE TASLP 2006](https://staff.aist.go.jp/m.goto/PAPER/IEEETASLP200609goto.pdf), orig.
ICASSP 2003) is the **only** method found that explicitly estimates *both ends* of every
repeated section by cross-checking the relations between different repeat-pairs — i.e.
that treats "all occurrences of this section must agree" as a constraint. It is 20 years
old and no successor kept that property:

| method | boundary decision | cross-occurrence constraint |
|---|---|---|
| RefraiD (2006) | lag matrix + repetition judgement | **yes**, explicit |
| Serrà SF (2012/2014) | per-location structure features | no |
| McFee & Ellis (2014) | spectral clustering on SSM | no |
| Ullrich/Grill & Schlüter (2014/15) | per-frame CNN saliency | no |
| All-In-One (2023) | peak-pick a 24 s window | no (and ignores its own downbeats) |
| CBM (2023) | bar-grid block matching | no — only a per-segment 4/8/12/16-bar *length* prior |
| SongFormer (2025) | transformer over SSL embeddings | no |

**And there is no metric for it either.** `mir_eval.segment` has exactly four families —
`detection` (hit-rate at ±0.5/±3 s), `pairwise`, `nce`, Rand index. None scores "do the
two A sections start at the same point in the loop". The closest formal analogue is the
MIREX *Discovery of Repeated Themes & Sections* family (establishment / occurrence /
three-layer P-R-F, Collins & Meredith), which lives entirely in the symbolic point-set
world and has never been applied to audio chord charts.

(Aside, [mir_eval issue #226](https://github.com/mir-evaluation/mir_eval/issues/226),
read directly: the maintainers themselves flag `nce`'s normalisation as arguably wrong and
unstable under padding. Closed, unfixed. Don't lean on NCE.)

**Conclusion.** Our defect is not a solved problem we failed to look up. It is a real hole
in the field, and §1.3 says the cheap fix — chord-string repeats — is already in our hands.
That is an unusually good position: strong measured evidence, no competing shelf product.

---

## 4. SYMBOLIC-PATH REFERENCES (we have SOTA chords; this is the road less travelled)

- **Mauch, Noland & Dixon, ISMIR 2009**, *Using Musical Structure to Enhance Automatic
  Chord Transcription* — the classic; averages chroma across repeat occurrences to clean
  chord labels. **Note the direction: it assumes structure is known and uses it to fix
  chords. We would be doing the reverse.** No code found.
- **Eldeeb & Malandro, ISMIR 2025**, *Barwise Section Boundary Detection in Symbolic Music
  Using CNNs* ([arXiv:2509.16566](https://arxiv.org/abs/2509.16566)) — bar-grid-native
  boundary CNN on symbolic input; releases 6134 annotated MIDI files curated from Lakh.
  Closest published thing to a symbolic bar-level detector. Still a per-bar classifier —
  no phase constraint.
- **Pitchclass2vec** (2023, [arXiv:2303.15306](https://arxiv.org/pdf/2303.15306)) —
  chord-symbol embeddings + LSTM segmentation *directly from chord annotations*. Exactly
  the "you already have good chord labels" premise. Not yet read in depth.
- **Sidorov, Jones & Marshall, ISMIR 2014**, *Music Analysis as a Smallest Grammar
  Problem* ([PDF](https://archives.ismir.net/ismir2014/paper/000226.pdf)) — grammar
  induction over a symbolic sequence; repeats become grammar rules. This is a principled
  version of "under-fold, never over-fold" and of our folding layer.
- **Meredith's SIA/SIATEC/COSIATEC** family — MIREX repeated-pattern discovery winners.
  Code: [OMNISIA](https://github.com/chromamorph/omnisia-recursia-rrt-mml-2019),
  [ostinato](https://github.com/pauldhein/ostinato) (Python).
- **Hanna, Robine & Rocher, JCDL 2009** — Smith-Waterman local alignment on chord strings.
  Built for cover-song *retrieval*. **Unverified** (PDF would not extract).

---

## 5. CODE TRIALS — what actually runs [OURS, all run today]

Test song `docs/audio/carpenters_close_to_you.m4a` (223.7 s). Throwaway venvs only —
nothing was installed into the project `.venv`, `harmonia_min/` untouched.

| package | version | licence | upstream last commit | py3.12 install | imports | ran? |
|---|---|---|---|---|---|---|
| `as-seg` (CBM) | 0.1.12 | **BSD** | 2026-07-17 | **clean, ~2 s** | **yes** | **yes** |
| `librosa` Laplacian recipe | librosa 0.11.0 | ISC | 2026-07-31 | already present | yes | **yes** |
| `msaf` | 0.1.80 | MIT | 2026-03-04 | installs | **NO — see below** | yes, after a 1-line patch |
| `allin1` | — | MIT | **2023-10-10** | installs (natten builds!) | **NO** | no |

### `as-seg` / CBM — the one that fits (RECOMMENDED TO TRY)

Installs clean on 3.12 in ~2 s. **Does not pull madmom** (it is lazily imported inside one
unused helper). Pure-python wheel; deps are all things we already have.

Entry points, verbatim from the package source:

```python
as_seg.CBM_algorithm.compute_cbm(autosimilarity, min_size=1, max_size=32,
        penalty_weight=1, penalty_func="modulo8", bands_number=None, ...)
        -> (segments: list[(start_bar, end_bar)], score)
as_seg.barwise_input.barwise_TF_matrix(spectrogram, bars, hop_length_seconds, subdivision)
as_seg.autosimilarity_computation.switch_autosimilarity(barwise_TF, "cosine"|"covariance"|"RBF")
as_seg.data_manipulation.segments_from_bar_to_time(segments, bars)
```

**`compute_cbm` takes an autosimilarity matrix directly** — so we can feed it *our own*
bar-level SSM and *our own* Beat This! downbeats. That makes it a genuine drop-in for the
cut-source in `sections.py`, not a rewrite.

Output on Close To You (82 bars, from the repo's cached Beat This! grid, 4 beats/bar):
`(0,8) (8,16) (16,24) (24,32) (32,40) (40,50) (50,58) (58,66) (66,74) (74,82)` — 10 clean
bar-indexed segments. Runtime 9.7 s, of which the CBM dynamic program itself is **0.01 s**;
the rest is log-mel extraction we would not need.

**Then I scored it properly** — same 694 Billboard section starts, same bar grid, same
chroma as §1.2, sweeping its hyperparameters:

| setting | exact bar | within ±1 bar | segments/song |
|---|---|---|---|
| defaults (`pw=1, bands=None, max=32`) | 16.0% | 23.1% | 5.8 |
| **best found** (`pw=0.5, bands=7, max=16`) | **35.7%** | **50.7%** | 14.7 |
| *our current novelty path, for reference* | *34.7%* | *42.7%* | — |
| *symbolic chord-repeat, for reference* | *75.2%* | *80.8%* | — |

**Honest verdict: CBM tuned ties our current novelty on exact-bar placement and beats it
within ±1 bar — but it gets there by emitting ~2× too many segments** (14.7 vs GT's ~7).
It is a respectable, cheap, bar-native cut source. It is **not** the answer to our defect,
and it is nowhere near the symbolic path. `bands_number=7` is the single setting that
matters — the default `None` is much worse.

### `librosa` Laplacian (McFee & Ellis 2014) — runs, no install

Implemented the gallery recipe (CQT→beat-sync chroma recurrence graph + MFCC path graph →
combined affinity → normalised Laplacian → eigen-embedding → KMeans), substituting the
repo's cached Beat This! grid (`data/cache/raw_beat_times_v3/…`, 326 beats,
`"source": "beat_this"`) for librosa's beat tracker. Ran clean, 3.35 s at K=6.

Emitted **33 segments** on a 224 s song — label-toggling over-segmentation, which is the
bare recipe's known behaviour without extra smoothing. Boundaries in seconds:
`0.37, 13.35, 21.54, 33.22, 44.13, 52.29, 60.40, 66.46, 72.52, 88.19, …`

Useful note found on the way: **that beat cache carries no downbeat phase** (`bar_sec:
null`) — periodic beat grid only. Anything needing bar phase must go to
`data/cache/bar_ref_downbeats/`.

### `msaf` — the crisp negative, and it is fixable

`uv pip install msaf` succeeds on 3.12. **`import msaf` then fails immediately:**

```
File ".../msaf/pymf/sivm_search.py", line 19, in <module>
    from scipy import inf
ImportError: cannot import name 'inf' from 'scipy'
```

msaf pins `scipy>=0.13.0` with no upper bound; modern scipy dropped the `scipy.inf` alias
(tried 1.18.0 and 1.13.1, both fail). The 2025 "Python 3.12 support" PR fixed packaging,
**not** this. With a one-line patch in the throwaway venv (`from numpy import inf`),
`msaf.process()` runs end to end:

| boundaries_id | labels_id | runtime | #bounds |
|---|---|---|---|
| `sf` (Serrà) | scluster | 8.29 s | 12 |
| `cnmf` | scluster | 0.75 s | 12 |
| `olda` | — | 1.27 s | 11 |
| `scluster` | scluster | 0.53 s | 19 |
| `foote` | — | **still broken** | — |

`foote` hits a *second*, independent scipy break: `scipy.signal.gaussian` moved to
`scipy.signal.windows.gaussian`. **Verdict: msaf is usable as a bakeoff harness after two
one-line patches, but ships broken on any current scipy.**

### `allin1` — blocked, and not worth unblocking

`pip install allin1` surprisingly succeeds (even `natten` builds from source on ARM), but
`import allin1` fails: `ModuleNotFoundError: No module named 'madmom'` — `allin1/
spectrogram.py` imports madmom unconditionally while the PyPI metadata does not declare
it. A real packaging bug. `pip install madmom` then fails to build (`No module named
'Cython'`, legacy setup.py + build isolation). Upstream last commit **2023-10-10**.
Combined with §2's finding that it does not snap sections to its own downbeats, this was
dropped.

---

## 6. RANKED SHORTLIST

| # | candidate | what it computes, plainly | code that runs | needs | fixes our bar-phase defect? | cost |
|---|---|---|---|---|---|---|
| **1** | **Chord-string repeat matching + repeat-consistency constraint** (ours; ancestors: RefraiD 2006, Pitchclass2vec 2023) | Write the song as one chord symbol per bar. For each candidate section start, ask *"does the 8-bar chord string starting here occur again elsewhere?"* Take the start that repeats best. Then force all occurrences of a letter to share phase and length. | none needed — ~150 lines on caches we already have | bar grid (have), per-bar chord labels (have) | **Yes — directly. 75.2% exact-bar vs 34.7% for our current cue [OURS, 694 starts]. Immune to systematic chord error.** | **8–12 h** |
| **2** | **CBM / `as-seg`** (Marmoret et al., TISMIR 2023) | Slides a block-shaped kernel along the SSM *indexed by bars*, scoring each candidate block by how self-similar it is, plus a bonus for 4/8/12/16-bar lengths. Bar-native by construction. | **RUN — BSD, installs clean on 3.12 in 2 s, takes our own SSM directly** | downbeats (have), our SSM (have) | **Partly, and less than hoped. Measured by me: 35.7% exact-bar vs our 34.7% — a tie — and 50.7% vs 42.7% within ±1 bar, but at 2× too many segments.** Bar-native so it cannot make a sub-bar error; **no cross-occurrence phase constraint** (confirmed absent). | **3–4 h** to wire as an alternative cut source |
| **3** | **SongFormer** (Hong et al., 2025) | Big self-supervised audio encoders (MuQ + MusicFM) feeding a 4-layer transformer trained on 14k songs; predicts boundaries end-to-end. | GitHub + HF weights, released | audio, GPU-ish | **No, not as a placer** — emits seconds, no bar alignment, no repeat constraint. Valuable as a **second opinion**: a strong independent boundary prior to arbitrate our tiling runs where coverage is low. HR.5F 0.703 Harmonix, 0.650 RWC-Pop [PAPER]. | 6–10 h (inference plumbing + weights) |
| **4** | **Spectral clustering on the SSM** (McFee & Ellis 2014) | Builds a recurrence graph over frames, takes the smallest eigenvectors of its Laplacian, and clusters — repeats end up in the same cluster at several granularities at once. | **RUN — zero install, librosa recipe, 3.3 s/song** | audio or our chroma | **No for placement** — emitted 33 segments on a 224 s song, and the 2026 benchmark has it losing to CBM in 33/36 conditions, specifically at bar scale. **Yes for the other half** — a better *letter-assignment* engine than our `LABEL_COS` ratio, which has no length or order term. | 3–4 h if used for labels only |
| **5** | **`msaf`** | A toolkit, not a method: one API over foote / scluster / sf / olda / cnmf + several labellers. | **RUN, but broken on arrival** — `from scipy import inf` ImportError; 4/5 algorithms work after a 1-line patch, `foote` needs a second | audio | **No.** Its value is as a **bakeoff harness** to score five published algorithms on our new Billboard bar-GT in an afternoon, so we stop guessing. | 2–3 h for a bakeoff |
| **6** | **All-In-One** (Kim & Nam 2023) | One model predicting beats, downbeats and sections together on demixed stems. | pip, weights released; **frozen since 2024-05** | audio | **No — and this is the trap.** Despite predicting downbeats itself, it does **not** snap sections to them; segments come from peak-picking a 24 s window. It would reproduce our exact bug. Deprioritise. | — |

**Explicitly NOT recommended:** Serrà's Structure Features / time-lag matrix. We searched
specifically for post-2020 improvements to the time-lag representation and **there are
none** — the lineage was absorbed as a CNN input channel. Our `tiling_runs` already *is* a
crude lag method; there is no better version of it to import.

---

## 7. RECOMMENDATION

**Do #1. Build the symbolic repeat-consistency placer; keep #2 as the fallback for
through-composed songs.**

Reasoning, in order of weight:

1. **It is the only candidate with a measured 2× on our exact metric.** 75.2% vs 34.7%
   exact-bar placement, on 694 real section starts, on ground truth. Every other candidate
   is judged on a ±0.5 s hit-rate that does not translate to bars.
2. **It does not depend on our chords getting better.** Systematic chord errors — the kind
   recognisers actually make — cost literally nothing (75.9% at 15% error, 78.1% at 45%).
   The method asks whether a string *repeats*, never whether it is *right*. This decouples
   the structure roadmap from the chord roadmap.
3. **It fixes the defect Louis named, not a neighbouring one.** "Les blocs sont bons, on ne
   commence pas au début" is a phase error. A phase error is fixed by a constraint that
   ties occurrences together. Every shipped candidate decides each boundary independently,
   which is why our five hand post-passes exist and why the ±2-bar failsafe cannot reach a
   3–4 bar error.
4. **It has zero install risk and no new dependency**, on caches we already hold.
5. **The field has left it open.** Nobody since Goto 2006 enforces cross-occurrence
   consistency; there is not even a metric for it. We would not be reimplementing a
   solved thing.

**Order of work.** (a) Build the Billboard bar-GT harness from §1.1 — it is 40 lines and it
is the instrument every later decision needs; nothing in this repo has ever measured
structure on more than 13 songs. (b) Implement the repeat placer and score it against the
current detector on the same 694 starts. Ship criterion: **≥60% exact-bar on our own
predicted chords**, versus 34.7% today. (c) Only if it stalls, install `as-seg` and try
CBM as the cut source for the low-coverage songs.

**Honest caveat.** §1's numbers are Billboard-only. I could **not** re-run the symbolic
test on our own six charts: the stored charts keep only the *folded* bars (Norah's letter
A stores 8 bars, not its 62), and `data/cache/musx_probs/` is keyed by a video id the
chart JSONs no longer carry. Reproducing the unfolded per-bar chord chain for one of our
songs is ~30 min of work and should be step (b0) — it is the one thing that could still
overturn this recommendation.
