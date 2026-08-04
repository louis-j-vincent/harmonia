# Segment → section GROUPING literature review, 2026-08-05

Question: once a song is cut into small chord-repetition segments (2–8 bars), how do we
GROUP them into real sections (verse/chorus/bridge, ideally a form string like AABA)?
This is the half of music-structure-analysis (MSA) the 2026-08-04 review
(`docs/research_sessions/structure_literature_2026-08-04.md`) did not cover — that review
was about WHERE segments start (boundary phase). This one is about which segments are
THE SAME THING.

Citation discipline: **[PAPER]** = claim reported by the source. **[OURS]** = verified here,
by fetching the PDF/repo myself. Unverified claims are labelled "unverified", not asserted.

Our current mechanism, for reference (`harmonia_min/sections.py:338-368`, `LABEL_COS=0.96`):
build a segment×segment matrix `M[i,j]` = mean blurred-SSM cross-block similarity, then
greedy first-fit: put segment `i` in the first existing group whose members all score
`M[i,j]/sqrt(M[i,i]*M[j,j]) > 0.96` against it, else start a new group. No length term, no
order term, no form-level model — it is a single hand-set threshold.

---

## The 5 approaches

### 1. Multi-resolution community detection ("MSCOM") — hierarchical + community-detection angle

de Berardinis, Vamvakaris, Cangelosi & Coutinho, **"Unveiling the Hierarchical Structure of
Music by Multi-Resolution Community Detection"**, TISMIR 3(1), 82–97, 2020.
[DOI 10.5334/tismir.41](https://transactions.ismir.net/articles/10.5334/tismir.41) ·
[PDF](https://transactions.ismir.net/articles/41/files/submission/proof/41-1-1232-2-10-20200625.pdf)

| field | value |
|---|---|
| **Mechanism** | Build a graph where nodes are music events (their implementation: beat-synchronous chroma+MFCC frames) and edge weight is feature similarity, plus a chain of edges linking neighbours in time. Run **community detection by modularity maximisation** (the same family as Louvain) not once but at ~90 different "resolution" settings — a knob that trades off "one giant community" against "every node its own community". Reading off communities at each resolution gives a **nested hierarchy of groupings for free**, without deciding boundaries and labels in two separate passes the way we do. **[PAPER]** |
| **Code** | GitHub `jonnybluesman/mscom` **[OURS, fetched 2026-08-05]**: contains only 3 evaluation Jupyter notebooks (`comparison.ipynb`, `monotonicity_study.ipynb`, `statistical_tests.ipynb`) and data — README says the algorithm itself is referenced, not that its source is in this repo. **No LICENSE file. Last push 2019-10-24** (before the 2020 publication — this is the pre-print working repo). **Not a runnable package**: no `requirements.txt`/`setup.py`, no clear top-level module to import. Practically: **no usable code found**, only the evaluation harness. |
| **Input needed** | A pairwise similarity/graph over whatever you choose as nodes — we would substitute **segments** (not their frames) as nodes, reusing our existing `M[i,j]` cross-block matrix from `sections.py` as the edge-weight graph directly (that matrix already IS a segment-level affinity graph). |
| **Output** | A **tree** — nested communities across resolutions. Cutting it at one resolution gives a flat label strip; walking multiple resolutions gives verse-inside-section-inside-song-half style nesting. Not a form STRING by itself — would need a separate pass to pick "the" resolution and read off AABA. |
| **Hours to prototype** | **4–6h**, because the awkward part (a segment-affinity matrix) already exists in `sections.py` as `M`. Swap the greedy-threshold loop for `python-louvain`/`networkx` modularity communities at 3–5 resolution values (`networkx.algorithms.community.louvain_communities(G, resolution=r)`), compare against the `LABEL_COS` baseline on our songs. |

**Adaptation note [OURS]**: this is also the answer to the "spectral clustering of the
SEGMENT graph, not the frame SSM" angle — the paper runs modularity-based community
detection rather than eigenvector spectral clustering, but the *object* it clusters
(a graph, one node per unit) is exactly the shape asked for. Re-pointing it at our
already-built segment-level `M` matrix instead of a frame graph is a **direct, cheap
substitution**, not a re-implementation of their whole pipeline.

---

### 2. Segment-level 2D-Fourier-magnitude clustering — the classical "stage 2" reference

Nieto & Bello, **"Music Segment Similarity Using 2D-Fourier Magnitude Coefficients"**,
ICASSP 2014.
([PDF](https://www.researchgate.net/publication/269295575_Music_segment_similarity_using_2D-Fourier_Magnitude_Coefficients))
— 2014, older than the 2020+ preference, flagged explicitly: this is included because it
**is** the standard reference for "cluster already-found segments", the exact sub-problem
asked about, and nothing newer has displaced it as *the* baseline (`msaf`'s `scluster`
labeller still ships a 2D-FMC option today).

| field | value |
|---|---|
| **Mechanism** | Take each segment's chroma patch (varying length), pad/resample it to a fixed size, take its **2D Fourier transform, keep only the magnitude**. This throws away phase, and phase is what encodes *where in time and at what pitch* the pattern sits — so two occurrences of the same 8-bar phrase transposed a fourth up, or starting 2 bars later inside a longer take, land near each other in this representation (key-invariant, shift-invariant by construction). Cluster the resulting fixed-size vectors with **adaptive k-means** (k chosen by a stability criterion, not fixed). **[PAPER]** |
| **Code** | Shipped inside `msaf` (`urinieto/msaf`, MIT, pushed 2026-05-13 **[OURS, verified in the 08-04 review]** — already confirmed installable on 3.12 after the 1-line scipy patch, per the prior session) as the `2dfmc` feature + `scluster`/`fmc2d` labeller. **Already runnable in our environment today**, no new install. |
| **Input needed** | Segment list `[(b0,b1)]` (have) + a per-bar or per-half-bar chroma-like feature (have: NNLS bothchroma). Key/shift-invariance means it does not need our chords to be right, only the chroma texture to repeat — similar robustness argument to the symbolic chord-repeat method in the 08-04 review, but on chroma instead of decoded chords. |
| **Output** | A **flat label strip** (cluster id per segment) — no hierarchy, no form string, no functional name. |
| **Hours to prototype** | **2–3h** — it is already inside a package we have running; the work is wiring our segment list + bothchroma into `msaf`'s 2D-FMC feature extractor instead of full boundary+label `msaf.process()`, and comparing cluster ids against `LABEL_COS`. |

---

### 3. Grammar induction → an actual form string (SEQUITUR / REPAIR / LONGEST-FIRST / MOST-COMPRESSIVE)

Perkins & Ventura, **"Musical Phrase Segmentation via Grammatical Induction"**, IJCAI 2024.
[arXiv:2405.18742](https://arxiv.org/abs/2405.18742) ·
[IJCAI proceedings](https://www.ijcai.org/proceedings/2024/855)

| field | value |
|---|---|
| **Mechanism** | Treat the song as a **string of symbols** (for us: one symbol per bar, or per segment). A grammar-induction algorithm repeatedly finds a repeated pair/run of symbols, replaces every occurrence with a new single symbol (a "rule"), and keeps going until nothing repeats — exactly like naming a repeated riff "X" everywhere it occurs, then noticing "X X Y X" itself repeats and naming that "Z". The paper benchmarks 5 variants: **SEQUITUR** (online, one pass, replaces digrams as soon as they repeat — the classic 1997 algorithm), and 3 **offline** variants sharing one framework (IRR): **RE-PAIR** (always merge the *most frequent* repeated pair) **[PAPER]**, **LONGEST-FIRST** (always merge the *longest* repeated run first) **[PAPER]**, and **MOST-COMPRESSIVE** (merge whichever repeat shrinks the total grammar size the most, i.e. an explicit MDL objective) **[PAPER]**. On their 3 datasets, **LONGEST-FIRST won** on F1 **[PAPER]**. The final grammar's top rule, read left to right, *is* a form string like `A A B A` once symbols are relabelled by rule identity. |
| **Code** | `reedperkins/grammatical-induction-phrase-segmentation` **[OURS, fetched 2026-08-05]** — pure Python (`sequitur.py`, `irr.py`, `grammar.py`, `levenshtein.py`, numpy only). **No LICENSE file.** Last push 2024-05-10. Small enough to read end to end; no exotic deps seen (numpy) → **plausibly installs on 3.12**, not independently verified by running it. |
| **Input needed** | Any symbol sequence — for us, the per-bar decoded chord symbols directly (no pre-segmentation needed at all: the grammar induction step subsumes the segmentation step), or, more conservatively, one symbol per already-found segment if we want to keep our current boundaries fixed and only solve grouping. |
| **Output** | A **context-free grammar**, i.e. genuinely a **form string + a tree** — the top rule is the flat AABA-style reading, and its right-hand-side non-terminals expand into the nested sub-phrase structure (a chorus's internal 2-bar cell, say). This is the only approach of the 5 that produces the literal artifact Louis asked for ("a form string like AABA") as its direct output, not a derived reading. |
| **Hours to prototype** | **5–8h**: map our per-bar chord signature (already computed in `sections.py._sig`) to a token alphabet, run SEQUITUR + LONGEST-FIRST over it, read off the grammar. The grammar induction itself is <100 lines; most of the time is choosing the token granularity (raw chord vs bar-signature-with-fam) and turning grammar output back into `[{b0,b1,label}]`. |

*Note*: the 08-04 review already covers Sidorov, Jones & Marshall (ISMIR 2014, smallest-grammar
problem) for this same angle — not repeated here. Perkins & Ventura 2024 is the newer,
code-available alternative with a cleaner algorithm description and 4 induction variants
benchmarked against each other, which the 2014 paper does not do.

---

### 4. Sequence-alignment based grouping (Smith-Waterman / Qmax family)

Foundational reference: Serra, Gómez & Herrera, **"Chroma Binary Similarity and Local
Alignment Applied to Cover Song Identification"**, IEEE TASLP 16(6), 2008 — the paper that
defines **Qmax**, a Smith-Waterman-style local-alignment score over a **binarised**
chroma recurrence matrix, built for cross-song cover detection. The within-song, chord-symbol
variant is Hanna, Robine & Rocher, JCDL 2009 (**already flagged in the 08-04 review as
unverified — PDF would not extract**; still unverified here, no new attempt made).

| field | value |
|---|---|
| **Mechanism** | Instead of a fixed-width block comparison (our `M[i,j]`) or a fixed-lag line (RefraiD, below), run **Smith-Waterman local alignment** between two chord/chroma strings: it finds the best-scoring *subsequence* match allowing insertions/deletions, so a chorus that gains an extra pickup bar on its second pass still aligns cleanly — length or bar-count mismatches between two occurrences of "the same" section do not break the match the way a fixed-length block-cosine does. Two segments are "the same letter" when their optimal local alignment score, normalised by length, clears a threshold. **[PAPER, mechanism as used for cover-song Qmax; applying it segment-to-segment within one song is the natural, undocumented extension]** |
| **Code** | **No code found** for Hanna et al. 2009. Qmax itself: no canonical maintained repo found in this search; `essentia` (C++/Python, MIT) ships a **different** cover-song aligner (`ChromaCrossSimilarity`/`CoverSongSimilarity`, "Qmax"-inspired per its own docs) — not independently verified here for API fit to a within-song use case. |
| **Input needed** | Per-bar chord symbols (have) or NNLS bothchroma (have) turned into a binary self-similarity matrix, one column pair per candidate segment pair. |
| **Output** | A **flat label strip** — pairwise alignment scores, then a clustering pass (see below) to turn "these two align well" into groups. No form string, no hierarchy by itself. |
| **Hours to prototype** | **6–8h**: implement Smith-Waterman on chord-symbol strings (small, textbook DP — no external dependency needed, we already have a chord-vocabulary distance from the pipeline for the substitution cost), run all-pairs on our segments, then re-use the `LABEL_COS`-style greedy grouping already in `sections.py`, but keyed on alignment score instead of block-cosine ratio. |

**Grouping-after-alignment is a solved sub-problem inside RefraiD itself** — see §IV.D
below, "Integrate Repeated Sections": once you have a set of pairwise matches, you still
need an explicit step to merge `(A,B)` and `(B,C)` matches into one 3-member group `{A,B,C}`,
handle a section appearing 3+ times, and prune spurious matches. RefraiD's grouping
algorithm is a better-specified version of the clustering step this approach would still
need than anything found elsewhere in the search — see the RefraiD section below for the
detail, since it was read in full for question (b).

---

### 5. Functional labelling (verse/chorus/bridge, not just letters)

Primary: Wang, Hung & Smith (ByteDance), **"To Catch a Chorus, Verse, Intro, or Anything
Else: Analyzing a Song with Structural Functions"**, ICASSP 2022.
[arXiv:2205.14700](https://arxiv.org/abs/2205.14700)

| field | value |
|---|---|
| **Mechanism** | A 7-class taxonomy (intro/verse/chorus/bridge/outro/instrumental/silence) **[PAPER]**. A spectral-temporal Transformer (SpecTNT) is trained to output, for every time frame, a "chorusness"/"verseness"/... curve rather than one label per segment — closer to object detection than classification. Trained with an extra **connectionty-temporal-localization (CTL) loss** that lets it learn from data where only *some* structural boundaries/functions are annotated, not the whole song. **[PAPER]** Sister paper **MuSFA** (Wang, Smith & Hung, [arXiv:2211.15787](https://arxiv.org/abs/2211.15787)) repurposes the 18k-excerpt HookTheory Lead Sheet Dataset as weak/partial labels to boost this. **[PAPER]** |
| **Code** | **No code or weights found** for either paper (ByteDance did not release; confirms the same gap the 08-04 review found for their earlier metric-learning paper). |
| **Code-available alternatives (lower quality, but they run):** | **DeepChorus** (He, Sun, Yu & Li, [arXiv:2202.06338](https://arxiv.org/abs/2202.06338)): binary chorus/non-chorus only, not the full taxonomy. Repo `qqi-he/deepchorus`, **no LICENSE**, pushed 2026-04-28 **[OURS]** — actively touched, but its pinned stack is `python==3.6.2, tensorflow==2.1.0, madmom==0.16.1` **[OURS, from repo's own README]** — TensorFlow 2.1 predates Python 3.9 support entirely; **implausible on 3.12** without a substantial rewrite, same class of problem the 08-04 review hit with `allin1`'s madmom dependency. — **`beantowel/chorus-from-music-structure`**: a hand-written heuristic (salience/repetition-count style, not learned) that reuses `msaf`'s boundaries and picks the chorus by frequency+position+energy. Repo has **no LICENSE**, last push 2023-02-02 **[OURS]**, needs `python 3.7` and an external melody-extraction repo (JDC) — **not a clean 3.12 install**, but the *idea* (score each of our own `LABEL_COS` groups by repeat-count, mean position in song, mean loudness, then call the top-scoring one "chorus") is trivial to reimplement from scratch, ignoring the repo entirely. |
| **Input needed** | For the heuristic route: our own segment groups + counts (have, it's the output of step 1–4 above) + loudness (would need one more feature, e.g. RMS per bar — cheap). For the learned route: raw audio/spectrogram + a functional-label training set we do not have (SALAMI has some functional labels; HookTheory does not ship publicly at this scale). |
| **Output** | Real names (**verse/chorus/bridge**) attached to whichever flat/hierarchical label strip the earlier steps produced — this is a layer ON TOP of grouping, not a substitute for it. |
| **Hours to prototype** | Heuristic route: **2–3h**, directly on our own existing letter groups. Learned route: **15–20h+** (data assembly, no weights to start from) — not recommended as a first move. |

---

## Angle coverage

| angle asked for | covered by |
|---|---|
| Hierarchical / multi-level | Approach 1 (MSCOM, tree/dendrogram) |
| Spectral clustering / community detection of the **segment graph** | Approach 1 (modularity, adapted to our existing segment matrix `M`); Approach 2 (2D-FMC + adaptive k-means is feature clustering, not graph-spectral, included because it is the standard "stage 2" baseline) |
| Grammar / compression / MDL → form string | Approach 3 (Perkins & Ventura 2024) |
| Sequence-alignment (edit distance / Smith-Waterman) | Approach 4 (Qmax/Smith-Waterman ancestry + RefraiD's own grouping algorithm) |
| Functional label (verse/chorus/bridge), not just a letter | Approach 5 |

---

## (a) Is there a published "form string correctness" metric?

**Not found. Report this as a real gap, not a lookup failure** — the same conclusion the
08-04 review reached independently for boundary-phase, now checked again for form strings
specifically.

- `mir_eval.segment` only has `detection` (hit-rate), `pairwise`, `nce`, Rand index — all
  operate on **frame-level label agreement over time**, not on a compact letter string. Two
  songs with identical form `AABA` but slightly different segment *lengths* score less than
  perfectly on all four, which is the wrong sensitivity for "did we get the FORM right".
  **[OURS, confirmed against the mir_eval docs during this search]**
- MIREX **Discovery of Repeated Themes & Sections** (Collins & Meredith) has a 9-metric
  family (establishment/occurrence/three-layer × P/R/F) **[PAPER, confirmed via MIREX wiki]**
  — but it operates on **symbolic point-sets** (note onsets in a score), scoring whether a
  *pattern's notes* were found, not whether a *label sequence* like `AABA` matches a
  reference label sequence. Never applied to audio chord charts, per the 08-04 review, and
  nothing found here changes that.
- Generic **string edit distance (Levenshtein)** between a predicted and a reference letter
  string is the obvious tool, and it turns up as **ad hoc internal tooling**: the Perkins &
  Ventura 2024 grammar-induction repo ships its own `levenshtein.py` (banded edit distance)
  to score predicted vs. reference **boundary** sequences for their (different) phrase-
  segmentation task **[OURS, read the source]**. That confirms people reach for edit distance
  when they need this kind of comparison — but it is not a published, named, standardized
  MSA metric anyone else cites or that mir_eval implements.

**Conclusion**: if we want to report "form string correctness" (predicted `AABA` vs. GT
`AABA`), we would be defining our own metric (e.g. a length-normalised edit distance between
letter strings, ignoring absolute bar positions) — there is no existing citation to lean on
for it.

---

## (b) Goto's RefraiD — is "runs along a sub-diagonal at lag k" exactly its time-lag matrix?

Fetched and read the full PDF **[OURS]**
(https://staff.aist.go.jp/m.goto/PAPER/IEEETASLP200609goto.pdf, IEEE TASLP 14(5):1783–1794,
Sept 2006). Answering the three sub-questions with direct quotes.

### (i) Continuous or thresholded/binary similarity?

**Continuous, not binary — at the base level.** The similarity `r(t,l)` between chroma
vectors is a normalised-Euclidean-distance score:

> "Since the denominator ... is the length of the diagonal line of a 12-dimensional
> hypercube with edge length 1, `r(t,l)` satisfies `0 ≤ r(t,l) ≤ 1`." **[PAPER, quoted]**

Thresholding happens **downstream, at three separate, per-song-ADAPTIVE stages**, never as
one global binarisation of the raw similarity matrix:

1. Picking which lags are "high peaks" in the derived possibility function `P(t,l)` (an
   integral of locally-mean-subtracted `r` along the lag axis) — threshold chosen per song
   by Otsu's discriminant criterion **[PAPER]**: *"Because this threshold is closely related
   to the repetition-judgment criterion which should be adjusted for each song, we use an
   automatic threshold selection method based on a discriminant criterion."*
2. Deciding how far a line segment extends in time, on the smoothed 1-D curve
   `r(t, l0)` at one candidate lag `l0` — a **second**, separately-computed Otsu threshold
   *"adjusted using the above automatic threshold selection method... instead of
   dichotomizing peak heights, the method selects the top five peak heights."* **[PAPER]**
3. Three more heuristic pruning thresholds when grouping (equally-spaced-peak removal,
   high-deviation-peak removal, overlap removal), each with its own adaptive/SD-based cutoff
   **[PAPER]**.

So: **the base representation is continuous**; RefraiD is not built on a binarised matrix.
It IS heavily threshold-driven, but every threshold is derived per-song, at a late stage, on
a locally-normalised statistic — the opposite of one fixed global binary cutoff.

### (ii) Does it detect line segments (runs) in the lag dimension?

**Yes — this part of the claim is confirmed, precisely.** Quoted directly:

> "the method finds line segments that are parallel to the horizontal time axis and that
> indicate consecutive regions with high `r(t,l)`... each horizontal line segment in the
> time-lag triangle indicates a repeated-section pair. We, therefore, need to detect all
> horizontal line segments in the time-lag triangle." **[PAPER, quoted]**

A "horizontal line at fixed lag `l` in the `(t,l)` time-lag plane" is the same object as "a
diagonal run at offset `l` in the ordinary `(t1,t2)` self-similarity matrix" — same
geometry, rotated 45°. So: **the geometric ancestor claim is correct.** The paper even
footnotes this is literally a 1-D Hough transform for horizontal lines.

### (iii) How does it handle a section repeating at several lags?

**Via an explicit, separate integration algorithm — not just "reading runs".** Section
IV.D, *"Integrate Repeated Sections"*, quoted:

> "Since each line segment indicates just a pair of repeated sections, it is necessary to
> organize into a group the line segments that have common sections. Suppose a section is
> repeated `N` times, the number of line segments to be grouped together should
> theoretically be `NC2` if all of them are found." **[PAPER, quoted]**

Concretely: line segments sharing (nearly) the same start/end are merged into a group
`Γ_k = ([t_b, t_e], Ω_k)` where `Ω_k` is a **set of lags** **[PAPER]**. On top of that
bottom-up pass, RefraiD runs a **top-down re-detection** step that infers missing pairs by
transitivity — its own worked example: in a song shaped `A B C B C C`, the two long
`(B,C)`-repetition line segments imply the 1st↔3rd and 2nd↔4th `C` repetitions even if those
weaker pairs were not found directly **[PAPER]**. Finally three heuristic filters prune
spurious peaks (accompaniment-loop artefacts at equally-spaced lags, peaks whose alignment
quality is uneven along its own length, peaks too close to a stronger neighbour) **[PAPER]**.
This is a non-trivial, multi-stage algorithm — well beyond "read off the runs".

### Verdict: **PARTIAL — confirm the geometry, refute "binary" and "exactly"**

| sub-claim | verdict |
|---|---|
| RefraiD works in a time-lag representation, and repeats show up as runs along a fixed-lag line | **CONFIRMED**, quoted directly — this is the real ancestor relationship |
| That representation is a **binary** similarity | **REFUTED** — base `r(t,l)` is continuous `[0,1]`; every threshold used is late, adaptive, per-song, and applied to a *derived* statistic, not the raw matrix |
| Reading the runs is **all** RefraiD does to handle multi-lag repeats | **REFUTED** — Section IV.D is an explicit, separately-described grouping/integration algorithm (set-of-lags per group, top-down transitive re-detection, 3 pruning heuristics), not a byproduct of run-reading |

**For citation purposes**: RefraiD is a legitimate, citable ancestor of "read runs along a
lag" as a geometric idea, and of "you still need a separate step to merge repeats across
lags" as a design lesson (§IV.D is worth reusing as a template for approach 4's grouping
step, above). It is **not** support for calling a binary bi-bar lag matrix "exactly"
RefraiD's method — the continuous-similarity-plus-adaptive-threshold design is a
substantive difference, not a detail.

---

## Recommendation

**Try #3 (grammar induction) first, #4/RefraiD's §IV.D grouping trick as the fallback
clustering step for #1 or our current `LABEL_COS`.**

1. #3 is the only approach whose *direct output* is the literal artifact asked for (a form
   string, with a tree underneath) rather than a flat strip requiring a second interpretive
   pass — and it can run on the per-bar chord sequence we already decode, skipping the
   pre-segmentation step entirely if desired.
2. Its code is small, pure Python, and readable end to end in under an hour — lowest
   integration risk of the five.
3. #1 (community detection) is the second most promising — it is a near-zero-cost swap
   for the existing greedy `LABEL_COS` loop, since the segment-affinity matrix it needs
   (`M[i,j]`) is already computed in `sections.py`. Worth an afternoon regardless of what we
   do with #3.
4. #2 and #5-heuristic are cheap, low-risk sanity baselines (`msaf` already installs; the
   chorus-heuristic is a few hours on data we already have) — good for a bakeoff, not a
   primary bet.
5. #4's *code* is unverified/absent, but its *idea* (RefraiD §IV.D's grouping-after-matching
   algorithm) is worth stealing regardless of which pairwise-similarity source feeds it —
   including for grouping the output of #1 or #3.
