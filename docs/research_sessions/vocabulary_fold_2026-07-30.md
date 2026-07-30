# Vocabulary-based folding of chord observations — 2026-07-30

**Idea (Louis).** Once sections are identified, every occurrence of a section is
another *observation* of the same music. Average the **observations** (the
per-beat note activations) across occurrences **before** decoding — not the
decoded labels. Label voting ("5 of 8 said G") throws away the confidence
geometry: it can never represent "the 7th was weakly present in every pass".

**Verdict in one line.** The fold works and it is a large win where the decoder
actually reads the folded evidence (**+13.7 pp strict, +15.9 pp partial-credit**
pooled on 2 Brick-0 songs, pure-NNLS config) — but it is **exactly 0.00 pp on the
shipped config**, and on This Love it **defers entirely**, so it changes no
chart today. It stays opt-in.

---

## 1. What was built

| Piece | Where |
|---|---|
| `fold_by_vocabulary(beat_probs, beat_times, sections, bar_bounds_sec)` | `harmonia/models/periodicity.py` |
| `_bar_beat_index` (bar → absolute beats, by TIME) | same |
| `_vocab_fold_enabled()` / `_provisional_chords()` / `_vocab_fold_arrays()` | `harmonia/models/chord_pipeline_v1.py` |
| live wiring (folds the NNLS-24 per-beat frame, re-runs the root head) | `harmonia/stages/chord_head.py::run_full` |
| wiring on the bp48 branch (step 5·V, next to the user-merge pooling) | `harmonia/models/chord_pipeline_v1.py::infer_chords_v1` |
| 19 tests | `tests/test_vocabulary_fold.py` |

Flag: `HARMONIA_VOCAB_FOLD=1`. Default OFF ⇒ identity.

### Correction to the brief: the integration point was wrong

The brief pointed at `harmonia/pipeline.py` + `ChordInferrer.infer(folded_views=…)`.
That consumer exists but **nothing shipped goes through it**. The live path is
`chord_pipeline_v1.infer_chords_v1` → (early return on `feature_frontend="nnls24"`)
→ `harmonia/stages/chord_head.py::NNLS24ChordHead.run_full`, which builds its own
`log_emission` and never constructs a `ChordInferrer`. Wiring only into
`pipeline.py` would have produced a feature with no effect on any number Louis
looks at. The fold was therefore put where the observation actually lives: the
NNLS-24 per-beat 24-d C-frame (`feat`), folded immediately after
`extract_features`, with the trained root head re-run on the folded rows so
segmentation **and** per-segment labelling both see the denoised evidence.

Related: `harmonia/models/user_constraints.py::pool_beat_evidence` is the same
mechanism already shipped, driven by *user-asserted* section merges. The
vocabulary fold is that, with a *learned* grouping. Two deliberate differences —
it is a **mean** (not a sum, so the folded array stays on the raw scale) and it is
opt-in until corpus-validated.

`harmonia/models/block_fold.py` is the **rejected** approach (voting on decoded
labels). Left untouched.

---

## 2. The alignment decision (and the ragged-count policy)

The vocabulary is indexed in **bars**; the evidence is indexed by **absolute
beat**. Two choices, both load-bearing.

**Bar → beat, purely by TIME.** Bar `b` owns every beat whose time falls in
`[bounds[b], bounds[b+1])`. No `beat` field is read from anywhere. Reason:
`rigid_grid.py` backs every bar edge off 0.15 bar and never compensates it in the
`beat` label it writes (`known_issues.md` OPEN #3 — This Love's downbeat histogram
is `{1:75, 2:1, 3:43}`, so **0/34 downbeats sit on beat 0**). Any map that trusted
that label would be off by one on nearly every bar. A time-based map inherits only
the grid's *constant* phase, which shifts which beats a bar owns identically for
every bar, leaving slot alignment between occurrences intact. Verified with a
dedicated test.

**Slot key = `(item label, bar index WITHIN the item, ordinal beat within that
bar)`.** Occurrences of one item routinely disagree on beat count (the bar grid is
rigid; beat *detection* is not). Policies considered:

| Policy | Verdict |
|---|---|
| truncate to the minimum / zip | **rejected** — one extra beat early in a pass shifts *every later beat of that pass* by a slot, so the whole occurrence averages against the wrong musical positions while the output still looks plausibly smoothed. CLAUDE.md error pattern #1. |
| resample onto a common slot count | **rejected** — interpolates across chord changes, inventing evidence exactly at the boundary the decoder most needs sharp. |
| skip the whole occurrence | **rejected** — a ±1-beat disagreement somewhere is the common case, not the exception (`pool_beat_evidence` measured almost every real merge group dying to a single off-by-one), so this discards most of the evidence. |
| **adopted: per-bar ordinal keying, ragged tail folded only across the occurrences that actually supply it** | exact when passes agree; a wobble is contained inside the one bar where it happened; a slot supplied by one occurrence folds to itself (mean of one = identity), never zipped against a different musical position. |

Every value is read from an immutable snapshot, so overlapping sections cannot
compound through an already-folded value (the order-independence rule
`pool_beat_evidence` had to learn the hard way).

Also deferred-by-design: too few chords, `rigid_grid_for` declining,
`vocab_sections` declining, `< 8` bars → **inputs returned unchanged**.

---

## 3. The three verifications

**19/19 pass** (`tests/test_vocabulary_fold.py`). Full suite: 1140 pass (one
pre-existing unrelated failure in `harmonic_texture.py:175`).

### 3.1 The averaging is literally the mean of the right cells

Synthetic song, `beat_probs[i, :] = i`, so every beat carries its own absolute
index and the expected fold of a slot is the mean of its beat indices — an
off-by-one is arithmetically visible. 12 bars, A×3 of a 4-bar item → occurrences
start at beats 0/16/32, so slot *s* must fold to **exactly** `(s + 16+s + 32+s)/3
= 16+s` at all three positions. It does. Neighbouring slots differ by exactly 1,
so a one-beat shift cannot accidentally satisfy the assertion.

Also pinned: two items fold independently; shape/dtype match `fold_beat_probs`;
input not mutated; a single-occurrence item is the identity; a section shorter
than its own item passes through unfolded; a pickup before the grid and beats past
the last edge pass through; empty/degenerate inputs defer.

**Confirmed on real audio too:** at all 8 This Love verse slots the folded value
equals the arithmetic mean of the 8 raw passes to full precision (table in §4).

### 3.2 A 1st/2nd ending is NOT smeared

Toy form `B×3 C  B×3 C  B×3 E` (24 bars, 2-bar items) — This Love's chorus tail
`C` and outro tail `E`. C's two occurrences sit 8 bars apart and E sits 8 bars
after the second C, so the fixed period is 8 bars = 32 beats.

| | C's marker | E's marker |
|---|---|---|
| `fold_by_vocabulary` at C | **1.000** | **0.000** |
| `fold_by_vocabulary` at E | 0.000 | 1.000 |
| `fold_beat_probs(period=32)` at C | 0.667 | 0.333 |
| `fold_beat_probs(period=32)` at E | 0.667 | 0.333 |

The old folder destroys both tails; the vocabulary folder keeps them apart. And
not by refusing to fold: a note present in only C's **first** pass still averages
to 0.5 across C's two occurrences, while E never sees it.

### 3.3 Ragged occurrences are never silently zipped

A×3 of a 2-bar item where bar 2 received a **5th** beat. Item-bar 1 / ordinal 0 =
beats 4, 13, 21 — all three still fold together, even though occurrence 2 carries
an extra beat *before* them. The extra beat (index 12) folds to itself, untouched.
Matched ordinals inside the ragged bar still fold. A bar with zero beats is
harmless.

---

## 4. Louis's question, with a number

**This Love, verse item `A`, 8 occurrences, slot 0 (the verse downbeat).** Live
NNLS-24 treble C-frame, chart-grade vocabulary
(`A×4 B×3 C A×3 B×3 C A D B×3 E B×3 E B×3 E`, bar = 2.5240 s over 80 bars).
Ground truth: **G** ("the A bars always open on G").

| | t (s) | that pass decoded | **F (pc 5)** | G (pc 7) | B (pc 11) | D (pc 2) | F/G |
|---|---|---|---|---|---|---|---|
| occ0 | 1.081 | G | 0.2218 | 0.2734 | 0.5910 | 0.1200 | 0.81 |
| occ1 | 11.181 | G7 | 0.1522 | 0.1959 | 0.7436 | 0.1460 | 0.78 |
| occ2 | 21.280 | G | 0.1230 | 0.1192 | 0.6954 | 0.0789 | 1.03 |
| occ3 | 31.379 | G7 | 0.1827 | 0.1261 | 0.8256 | 0.0753 | 1.45 |
| occ4 | 61.678 | G7 | 0.0805 | 0.1247 | 0.3914 | 0.3057 | 0.65 |
| occ5 | 71.777 | **Bb^7** | 0.1722 | 0.1335 | 0.4828 | 0.2012 | 1.29 |
| occ6 | 81.877 | **Ab-7** | 0.1234 | 0.0780 | 0.7115 | 0.3089 | 1.58 |
| occ7 | 112.175 | G7 | 0.1980 | 0.3260 | 0.1527 | 0.5006 | 0.61 |
| **RAW mean** | | | **0.1567** | 0.1721 | 0.5742 | 0.2171 | 0.91 |
| **FOLDED** | | | **0.1567** | 0.1721 | 0.5742 | 0.2171 | 0.91 |

### Does it sharpen G-vs-G7? **No.**

* The 7th's absolute activation **falls 29 %** at the first verse (0.2218 →
  0.1567) — but the root G falls **37 %** (0.2734 → 0.1721). Relatively the 7th
  gets **stronger**: F/G at that slot goes **0.81 → 0.91**. The fold moves the
  G-vs-G7 call the wrong way at the slot he asked about.
* **The decoded chord does not change.** Not at any of the 8 slots, under either
  vocabulary. Under the pipeline's own vocabulary it cannot change, because on This
  Love the fold **DEFERS** (see §6). Hand-fed the chart-grade vocabulary, the
  per-segment labels at those 8 times are byte-identical ON vs OFF
  (`G#:min / G:maj7 / G#:maj / G:maj7 / G#:maj7 / G#:maj7 / G:maj7`).

### Yes, the mis-decoded passes pollute the average — they are the whole problem

occ5 (bar 28, `Bb^7`) and occ6 (bar 32, `Ab-7`) are the two passes
`section_vocab`'s docstring already flags as mis-decoded. They are **exactly the
high-F passes**:

| group | F/G |
|---|---|
| the 6 clean passes only | **0.822** |
| the 2 polluted passes alone | **1.397** |
| all 8 (what the fold uses) | 0.911 |

An unweighted mean over 8 lets 2 bad passes drag the slot from 0.822 to 0.911 —
i.e. **without them the folded slot would be *better* than the first pass already
was (0.81)**. The obvious fix is a robust/weighted combiner (median, trimmed mean,
or confidence weighting), not a plain mean. Not implemented.

### What the fold *did* fix, and it is not small

The trained NNLS root head's own per-beat verdict at those same 8 slots:

| | verdict at the 8 verse downbeats | mean P(root = G) |
|---|---|---|
| OFF | `G G B B G B B G` — **4/8 wrong** (B for G) | 0.3944 |
| ON | `G G G G G G G G` — **8/8 right** | **0.5647** (+17.0 pp) |

So folding fixes the **root** at the verse downbeat decisively, while leaving the
**quality** (G vs G7) question untouched. Louis asked about the quality; the win
landed on the root.

---

## 5. End-to-end, flag ON vs OFF (Brick-0, `stand_by_me` + `bein_green`)

Cache-warm only; free disk checked before each run (52 GiB, `data/cache` 4.4 G).

### Shipped config (`segment_source=musx_redecode`, `quality/bass=musx`)

| song | root | majmin | 7ths | partial | strict | bass |
|---|---|---|---|---|---|---|
| stand_by_me OFF **and** ON | 0.852 | 0.852 | 0.731 | 0.731 | 0.731 | 0.852 |
| bein_green OFF **and** ON | 0.665 | 0.665 | 0.560 | 0.655 | 0.471 | 0.718 |
| **POOLED OFF and ON** | 0.7607 | 0.7605 | 0.6471 | **0.6938** | **0.6033** | 0.7864 |

**Δ = +0.00 pp on every metric.** The fold *fires* (352/355 and 193/224 beats
folded) yet nothing moves: under the shipped config music-x-lab supplies the
segmentation **and** the root, quality and bass, so the NNLS observation the fold
denoises is barely read. This also means the default-off guarantee is trivially
satisfied here — and it is confirmed separately: flag OFF reproduces the committed
baseline byte-for-byte.

### Pure-NNLS config (`segment_source=nnls`, `quality/bass=nnls24`) — where the folded evidence *is* decoded

| song | root | majmin | 7ths | partial | strict | bass |
|---|---|---|---|---|---|---|
| stand_by_me OFF | 0.988 | 0.741 | 0.619 | 0.619 | 0.619 | 0.988 |
| stand_by_me **ON** | 0.988 | **0.988** | **0.866** | **0.866** | **0.866** | 0.988 |
| bein_green OFF | 0.581 | 0.571 | 0.119 | 0.398 | 0.056 | 0.648 |
| bein_green **ON** | 0.585 | 0.534 | **0.180** | **0.466** | **0.079** | 0.587 |
| POOLED OFF | 0.7882 | 0.6576 | 0.3742 | 0.5107 | 0.3432 | 0.8215 |
| **POOLED ON** | 0.7905 | 0.7653 | 0.5298 | 0.6699 | 0.4804 | 0.7913 |
| **Δ (pp)** | **+0.23** | **+10.77** | **+15.56** | **+15.92** | **+13.72** | **−3.02** |

Read this as: **root barely moves (+0.2), quality moves a lot (+10.8 majmin,
+15.6 sevenths).** That is Louis's mechanism working — the root was already
mostly right, and averaging N observations sharpened the *quality* call. Two
caveats: **bass regresses 3.0 pp** (the fold averages the bass half of the
C-frame too, destroying each occurrence's own inversion), and `bein_green`'s
majmin *falls* 3.7 pp because its vocabulary is junk (§6). n = 2 songs — a
hypothesis, not a validated setting (rule #5).

---

## 6. What this does NOT solve

1. **On This Love the fold DEFERS — it changes nothing today.** The in-pipeline
   vocabulary is built from `_provisional_chords`: per-beat root argmax, runs
   coalesced, quality left blank (a G and a G7 read as identical). That chain is
   too noisy on This Love for `rigid_grid_for`/`vocab_sections` to agree on a
   grid, so the fold declines. Every promising This Love number in §4 used the
   **chart-grade** vocabulary, hand-fed — which the pipeline cannot see, because
   it only exists *after* the full decode. Closing that gap needs a real
   two-pass (decode → vocabulary → refold → re-decode), i.e. 2× runtime.
2. **The shipped config cannot benefit at all** (§5). To move the live numbers the
   pooling has to happen on **music-x-lab's frame posteriors**, not on the NNLS
   frame. That is the single highest-value follow-up.
3. **A wrong vocabulary is applied confidently.** `bein_green` got 12 items over
   57 bars — meaningless for a through-composed ballad, and its majmin fell. There
   is no internal confidence gate; `vocab_sections` deferring is the only guard.
4. **It inherits the grid octave.** `known_issues.md` OPEN #1: the bar is 2× wrong
   on ~⅓ of the corpus, and OPEN #2 says `rigid_grid_for` never actually defers.
   A wrong octave folds genuinely different music together.
5. **Unweighted mean, outlier-sensitive.** Quantified in §4: 2 of 8 passes move the
   slot from 0.822 to 0.911. Median / trimmed mean / confidence weighting is the
   obvious next lever.
6. **Bass should probably not be folded** (−3.02 pp). Folding treble-only
   (`feat[:, :12]`) is a one-line experiment.
7. **No occurrence reliability weighting** — a masked or badly-mixed pass counts as
   much as a clean one.

### Ranked next steps

1. Fold the treble half only — recover the −3.02 pp bass regression. (minutes)
2. Median / trimmed mean instead of the plain mean — §4 says this is worth ~0.09
   of F/G at the one slot measured. (minutes)
3. Pool music-x-lab's frame posteriors, so the shipped config can move. (the real
   lever)
4. Real two-pass so the pipeline sees the chart-grade vocabulary instead of the
   provisional chain. (2× runtime, gated)
5. A vocabulary-confidence gate before folding at all (`bein_green`).

---

## 7. What has to be re-run for the iPhone app to see any of this

**Nothing improves on its own.** The app loads a **baked** `const P = {…}` payload
out of `docs/plots/inferred_<slug>.html`; it never re-runs inference. Those files
are written by the server's analyze path
(`harmonia/serving/analysis.py`, `out = PLOTS_DIR / f"inferred_{slug}.html"` →
`render_interactive(...)`).

So to see the fold on the phone, for each song:

1. Set `HARMONIA_VOCAB_FOLD=1` **in the server process's own environment** —
   `harmonia_server.py` runs with no reloader, so **restart it**; a stale process
   silently served old code through an entire debugging session before.
2. Re-analyse the song through the analyze endpoint so
   `docs/plots/inferred_<slug>.html` is re-baked.
3. Reload the chart in the app.

And on the current default (shipped) config that re-bake produces a
**byte-identical** payload (§5) — plus This Love defers (§6). There is no user-
visible change to ship yet. This work is the primitive plus its evidence, not a
shippable improvement.
