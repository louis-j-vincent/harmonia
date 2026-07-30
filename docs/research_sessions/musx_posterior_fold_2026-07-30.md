# Two-pass music-x-lab decode: vocabulary fold on ITS OWN frame posteriors — 2026-07-30

**Verdict in one line.** Folding music-x-lab's frame posteriors by the section
vocabulary and decoding a second time is worth **+2.12 pp partial-credit /
+1.43 pp strict on the SHIPPED config** over the 7 frozen Brick-0 songs
(**+1.87 pp / +1.17 pp leave-one-song-out**), no song regresses on
partial-credit, and it costs one extra Viterbi decode. **Recommend default ON.**

This is the follow-up the NNLS-24 fold's writeup named as "the single
highest-value follow-up" (`vocabulary_fold_2026-07-30.md` §6.2), and it lands
where that one could not: **upstream of music-x-lab's decoder.**

---

## 0. The two folds are mirror images — that is the whole point

| | pure-NNLS config | SHIPPED config |
|---|---|---|
| `HARMONIA_VOCAB_FOLD` (NNLS-24 per-beat frame) | **+15.92 pp** partial | **+0.00 pp** |
| `HARMONIA_MUSX_FOLD` (musx frame posteriors) | **+0.00 pp** | **+2.12 pp** partial |

Each fold is inert in the config that does not read the evidence it denoises.
The NNLS fold is inert on the shipped config because music-x-lab supplies
segmentation, root, quality and bass there. This fold is inert on the pure-NNLS
config for the exactly symmetric reason: that config sets
`segment_source="nnls"`, so the `musx_redecode` branch never runs and the fold is
never reached. **They are complementary, not competing.**

---

## 1. What was built

| Piece | Where |
|---|---|
| `enabled()`, `agree_min_from_env()`, `chords_from_labels()`, `vocab_from_chords()`, `_bar_frame_index()`, `_slot_map()`, `_agreement()`, `fold_frame_posteriors()`, `two_pass_redecode()` | `harmonia/models/musx_posterior_fold.py` (NEW) |
| live wiring (one `if` inside the existing `musx_redecode` branch) | `harmonia/stages/chord_head.py::NNLS24ChordHead.label_stage` |
| 28 tests | `tests/test_musx_posterior_fold.py` (NEW) |
| guard calibration | `scratchpad/musx_fold_calibrate.py` |
| Brick-0 A/B + threshold sweep | `scratchpad/musx_fold_ab.py` |
| LOSO on the one fitted knob | `scratchpad/musx_fold_loso.py` |
| This Love before/after | `scratchpad/musx_fold_this_love.py` |

Flags: `HARMONIA_MUSX_FOLD=1` (default **OFF**), `HARMONIA_MUSX_FOLD_AGREE`
(guard threshold, default 0.60).

The loop:

```
pass 1   redecode(our beat grid, RAW posteriors)                -> chords
vocab    chords -> rigid_grid_for -> apply_rigid_grid -> vocab_sections
fold     average the posteriors across occurrences, ON THE FRAME GRID
pass 2   redecode(our beat grid, FOLDED posteriors)             -> final chords
```

`frame_posteriors` is cached stem-keyed to `data/cache/musx_probs`, so **pass 2
pays only the Viterbi decode — no second neural inference.** The vendored
`chord_recognition.py` was never touched; it only accepts an audio path and emits
hard labels, which is exactly why folding *its output* was never an option.

Note: `harmonia/stages/chord_head.py` in the working tree also carries **another
session's** treble-only change to the *NNLS* fold (~L964). Untouched here.

---

## 2. The alignment decision

**Slot key = `(item label, bar index WITHIN the item, ordinal frame within that
bar)`.** Frames come from the bar edges by TIME:
`round(bounds[b] / FRAME_DT) .. round(bounds[b+1] / FRAME_DT)`. No `beat` field
is read from anywhere — `rigid_grid` backs every bar edge off 0.15 bar and never
compensates it in the `beat` label it writes (`known_issues.md` OPEN #3), so a
label-based map would be off by one on nearly every bar. The backoff is a
*constant* phase shift: it moves which frames every bar owns identically, leaving
slot alignment between occurrences intact.

Frames are the **easy** case compared to beats. `rigid_grid._build_bounds` emits
exactly uniform bars, and a frame cannot be "missed" the way a beat can, so the
ragged-count problem that dominated `periodicity.fold_by_vocabulary` reduces to a
±1-frame rounding difference in the last slot of a bar. Per-bar keying (rather
than offset-from-occurrence-start) is kept anyway, so that difference stays
contained in the bar where it happened instead of shifting every later slot.

Only **complete** occurrences fold; a slot supplied by one occurrence is left
raw. Every value is read from the immutable input, so overlapping sections cannot
compound through an already-folded value.

**Two streams-level decisions:**

* **The bass stream is NOT folded.** Each occurrence has its own inversion, and
  the project's root target *is* the sounding bass pitch class. Folding the bass
  cost 3.0 pp of bass accuracy in the NNLS experiment. Folded:
  `triad(73), s7, s9, s11, s13`. Pinned by a test.
* Each folded row is **renormalised** to sum to 1. A mean of distributions
  already does, but the vendored decoder takes `log` of these values, so the
  invariant is pinned rather than assumed.

---

## 3. The disagreement guard, and why the histogram lied

Folding **amplifies** a grouping error: mis-group two sections and the mean is a
*confidently wrong* chord for every bar in the group. So a slot folds only if its
occurrences agree — **mean pairwise cosine of the 73-d triad posterior ≥ 0.60**.
Closed form used: with unit-normalised rows, `mean pw cos = (‖Σu‖² − n)/(n(n−1))`.

**Calibration, in two steps that disagreed.**

1. **The distribution** (18 052 multi-occurrence slots, 7 frozen songs + This
   Love): bimodal, median **0.904**, a distinct low mode below 0.05, histogram
   **trough at 0.25–0.30** (223 slots in that bin vs 7 632 in the top bin).
2. **End-to-end, the trough is the WRONG operating point.** Sweeping it on the
   7-song benchmark:

| `agree_min` | partial | Δ pp | strict | Δ pp |
|---|---|---|---|---|
| OFF | 0.6947 | — | 0.5181 | — |
| 0.30 (the trough) | 0.7067 | +1.20 | 0.5230 | +0.49 |
| 0.45 | 0.7139 | +1.92 | 0.5302 | +1.22 |
| **0.60** | **0.7159** | **+2.12** | **0.5323** | **+1.43** |
| 0.75 | 0.7073 | +1.26 | 0.5248 | +0.67 |
| 0.90 | 0.6994 | +0.47 | 0.5152 | −0.28 |

Clean unimodal curve peaking at 0.60 — not a spike, which is what a
noise-fitted knob looks like.

**Why 0.60 rather than the trough:** the statistic scales with the number of
occurrences. Three occurrences of which one is mis-decoded score ≈1/3 and are
refused at 0.60 — correctly, because with n=3 one bad pass is a third of the
evidence. Eight occurrences with one bad pass score ≈0.75 and still fold. **The
guard is strict when the evidence is thin and tolerant when it is thick.** Both
regimes are pinned by tests.

**Firing rate at 0.60** (fraction of multi-occurrence slots refused):

| song | fires | song | fires |
|---|---|---|---|
| stand_by_me | 2 % | close_to_you | 36 % |
| blue_bossa_backing | 2 % | blue_bossa | 37 % |
| bein_green | 16 % | georgia_on_my_mind | 64 % |
| every_breath_you_take | 24 % | This Love | 4 % |

So: neither inert nor firing constantly. georgia is the extreme — it folds only
250 of 9 365 frames and moves **exactly 0.0000 on every metric**, which is the
guard working as designed on a song whose "`A×2 B`" vocabulary is meaningless.

---

## 4. The four verifications

**28/28 pass** (`tests/test_musx_posterior_fold.py`). Full suite: 1171 pass, 2
fail — both pre-existing and unrelated (`harmonic_texture.py:175` inline-CQT
guard, already noted in the NNLS fold's writeup; and one
`test_render_youtube_chart` bar-phase test that **passes in isolation** and comes
from another session's WIP in that file).

### 4.1 The fold is the arithmetic mean of the right frames

Synthetic song with `bar = exactly 43 frames`, so the 4-bar item's three
occurrences start at frames **0 / 172 / 344** and the expectation can be written
down without reusing the module's own bar→frame map. Every frame carries its own
index (`triad[f,0] = f/n_frame`, col 1 the complement), so slot *s* must fold to
`mean(x[s], x[172+s], x[344+s])` — asserted at all three positions to **1e-12**.

The off-by-one check is explicit: the *shifted* expectation (slot *s* folded from
frames `o+s+1`) reproduces the output at **0 of 171 slots**. A frame slip cannot
pass this test.

Also pinned: rows still sum to 1; input not mutated; single-occurrence item is
the identity; no sections / no bars / empty posteriors defer.

### 4.2 A 1st/2nd ending is not smeared

*Synthetic* (`B B B C | B B B C | B B B E`, This Love's chorus tail vs outro
tail): C's marker reads **1.000** at C's frames and **0.000** at E's; E's the
reverse. And not by refusing to fold — a column present in only C's **first**
pass folds to exactly **0.5** across C's two occurrences, while E never sees it.

*Real audio* (This Love, production beat grid, form
`A×4 B×3 C A×3 B×3 C A D B×3 C B×3 E B×3 F`): items A (3 478 frames), B (3 260)
and C (654) fold, and **no two items share a single folded frame** — checked
exhaustively over every pair. At item C's last bar all 3 occurrences read
`G:maj` → folded `G:maj`; at item A's last bar the 5 occurrences read
`Bb:maj ×4, D:dim` → folded **`Bb:maj`**, i.e. the one odd pass was outvoted.

### 4.3 Flag OFF is byte-identical

Two independent proofs:

1. `two_pass_redecode` was monkeypatched to **raise** and the flag-OFF Brick-0
   run completed unchanged — metrics *and* per-song chord counts reproduce
   exactly. The entry point is never reached with the flag unset.
2. The flag-OFF numbers reproduce the **committed baseline** exactly:
   stand_by_me `root=0.852 majmin=0.852 7ths=0.731 | partial=0.731 strict=0.731
   bass=0.852`, 41 chords (`docs/known_issues.md:21136`).

### 4.4 Brick-0, ON vs OFF, SHIPPED config

Cache-warm; free disk checked before each run (51–52 GiB, `data/cache` 4.4 G).
No cache-cold run on the long songs.

**All 7 verified songs, pooled** (`agree_min=0.60`):

| | root | majmin | 7ths | **partial** | **strict** | bass |
|---|---|---|---|---|---|---|
| OFF | 0.7510 | 0.7318 | 0.5622 | **0.6947** | **0.5181** | 0.7582 |
| **ON** | 0.7679 | 0.7537 | 0.5756 | **0.7159** | **0.5323** | 0.7740 |
| **Δ pp** | **+1.69** | **+2.19** | **+1.34** | **+2.12** | **+1.43** | **+1.58** |
| **LOSO** | 0.7664 | 0.7515 | 0.5730 | **0.7134** | **0.5297** | 0.7732 |
| **Δ LOSO pp** | **+1.54** | **+1.97** | **+1.08** | **+1.87** | **+1.17** | **+1.50** |

LOSO = the threshold is chosen on the other 6 songs and applied to the held-out
one (same protocol `musx_redecode` used for its change penalty). **6 of 7 folds
pick 0.60.** The knob survives being unfitted.

**Per song, partial-credit / strict:**

| song | partial OFF → ON | strict OFF → ON |
|---|---|---|
| stand_by_me | 0.730 → **0.742** | 0.731 → **0.743** |
| bein_green | 0.656 → **0.661** | 0.471 → **0.487** |
| blue_bossa | 0.593 → **0.637** | 0.569 → **0.629** |
| blue_bossa_backing | 0.883 → **0.910** | 0.617 → **0.586** |
| close_to_you | 0.746 → **0.751** | 0.257 → **0.246** |
| every_breath_you_take | 0.735 → **0.740** | 0.508 → **0.513** |
| georgia_on_my_mind | 0.530 → 0.530 | 0.309 → 0.309 |

**No song regresses on partial-credit.** Two strict regressions:
blue_bossa_backing −3.1 pp (its 7ths fall the same amount) and close_to_you
−1.1 pp. blue_bossa is the big winner (+4.4 partial / +6.0 strict).

**Fast 2-song pair asked for in the brief** (pooled): partial 0.6938 → **0.7025**
(+0.87), strict 0.6033 → **0.6174** (+1.41).

**Pure-NNLS config, all 7 songs: exactly +0.00 pp on every metric.** Not a
measurement artefact — that config sets `segment_source="nnls"`, so the fold is
unreachable by construction. §0.

---

## 5. This Love, before and after

**The form does not change.** Both ways the chart reads
`A×4 B×3 C A×3 B×3 C A D B×3 E B×3 E B×3 E`; the target spec has `C` in the 10th
slot. So the section layout Louis sees is the same. **What changes is the chord
content of 19 of 80 bars**, all toward what the repeated passes agree on:

| | verse item `A` (target: `G Cm Fm7 Dø`) |
|---|---|
| BEFORE | `G G C- C- F-7 F-7 **Do Fo**` |
| AFTER | `G G C- C- F- F- **Dh7 Bb**` |

`Dh7` is **Dø** (half-diminished) — the fold moved the verse's 4th chord from
D-dim to exactly what Louis's hand-written lead sheet says
(`docs/this_love_target_spec.md`).

The two most legible repairs are the two passes `section_vocab`'s own docstring
already flags as mis-decoded ("This Love's 3rd and 4th verse passes, bars 28 and
32, where `G` decoded as `C` and `Bm`"):

| bar | before | after |
|---|---|---|
| 28 | **`N.C.`** (a whole bar of nothing) | **`G7`** |
| 32 | **`B-`** | **`G7`** |
| 3 | `Do\|Fo` | `Dh7\|Bb` |
| 47 | `Do` | `Dh7` |
| 68 | `F-7` | `C-\|F-` (chorus 2-chord bar restored) |
| 64/66/72/74/76 | `C-\|F-7` | `C-\|F-` (matches item B's `C- F- Bb Eb`) |

**And the gap this closes.** On This Love the *NNLS* fold **defers entirely** —
its vocabulary comes from a per-beat root-argmax chain with blank qualities, too
noisy for `rigid_grid_for`/`vocab_sections` to agree. Fed music-x-lab's pass-1
labels instead, the vocabulary **exists**: 6 items, 80 bars @ 2.524 s. That is
the whole reason the design is two-pass.

**Caveats, stated plainly.**

* `data/cache/ltas_family_dist.npz` **is present** (3 968 B), so the ctx family
  model is enabled and this decode is *not* degraded on that account. (An earlier
  version of this check reported it missing — `REPO.rglob` does not follow the
  `data/` symlink. Corrected.)
* There is **no ground truth for This Love**, so none of the above is scored. It
  is inspection, and it is consistent with the direction of the Brick-0 numbers.
* **The app does not improve on its own.** It serves a *baked* `const P = {…}`
  payload out of `docs/plots/inferred_<slug>.html`. To see this on the phone the
  server must be restarted with `HARMONIA_MUSX_FOLD=1` (`harmonia_server.py` runs
  with **no reloader** — a stale process silently served old code through an
  entire debugging session before) and each song re-analysed.
* The standalone ending check originally recomputed beats with
  `extract_beat_features`, which uses **librosa**, while `SHIPPED_CONFIG` uses
  `beat_backend="beatthis"` — a different grid, hence a different vocabulary.
  It now spies on the fold during a real shipped-config decode instead.

---

## 6. What the fold actually fixes

**Gross root/identity error on repeated material — not quality refinement.**

* root **+1.69** and majmin **+2.19** move more than sevenths **+1.34**.
* The This Love diffs are gross repairs (`N.C.`→`G7`, `Bm`→`G7`, `Ddim`→`Dø`),
  not 7th-degree adjustments.
* At the verse downbeat the maj-vs-dom7 margin moves by **0.000** — the competing
  note is present in *every* pass. **Averaging kills random error, not consistent
  bias.** Anything the front-end gets wrong the same way every time survives the
  fold untouched.

---

## 7. What this does NOT solve (CLAUDE.md rule #4)

1. **It inherits the bar grid wholesale.** `known_issues.md` OPEN #1: the bar is
   metrically 2× wrong on ~⅓ of the corpus, and OPEN #2 says `rigid_grid_for`
   never actually defers. A wrong octave folds genuinely different music
   together, and the guard is the only defence — per *slot*, never per *item*.
2. **Unweighted mean.** A masked or badly-mixed occurrence counts as much as a
   clean one. Median / trimmed mean / reliability weighting is the obvious next
   lever and is not implemented.
3. **The guard gates on the triad stream only** and applies that one decision to
   all folded streams — deliberately, so the streams stay mutually consistent,
   but a slot whose triads agree and whose 7ths disagree still folds.
4. **`function_family` still reads the RAW posteriors.**
   `harmonia/models/function_family.py` calls `frame_posteriors` independently,
   so the function-family head (on in `SHIPPED_CONFIG`) never sees the folded
   evidence. Cheap follow-up.
5. **No third pass.** The vocabulary is never re-derived from pass 2's chords.
6. **The threshold is fitted on these 7 songs.** LOSO says it survives
   (+1.87 pp), but n=7 is still a hypothesis (rule #5).
7. **Two strict regressions** (blue_bossa_backing −3.1, close_to_you −1.1) and
   one song that moves not at all (georgia). Not a universal win.
8. **It does not change This Love's form**, only 19 bars of its content.

---

## 8. Recommendation

**Ship it default ON**, with `HARMONIA_MUSX_FOLD=0` as the kill switch.

The number that justifies it: **+2.12 pp partial-credit in-sample, +1.87 pp
LOSO**, on the same 7-song frozen benchmark on which `musx_redecode` itself was
flipped default-ON at +2.20 pp — and unlike that flip, this one needs no
companion gate and regresses no song on partial-credit. Cost is one extra Viterbi
decode per candidate latency on cached posteriors.

The code as left here is **default OFF** (as briefed). Flipping it is a one-line
change to `enabled()`, exactly like `musx_redecode.enabled()`.

### Reproduce

```bash
.venv/bin/python -m pytest tests/test_musx_posterior_fold.py -q          # 28 tests
.venv/bin/python scratchpad/musx_fold_calibrate.py                       # guard distribution
.venv/bin/python scratchpad/musx_fold_ab.py --all --config shipped \
    --agree 0.3 0.45 0.6 0.75 0.9 --json sweep_shipped.json              # ~12 min warm
.venv/bin/python scratchpad/musx_fold_loso.py sweep_shipped.json         # LOSO
.venv/bin/python scratchpad/musx_fold_this_love.py                       # before/after
```

Always pass `**SHIPPED_CONFIG`: `infer_chords_v1`'s bare defaults are
`feature_frontend=bp48` + `nnls24` quality/bass, a different and much worse
pipeline.

### Ranked next steps

1. Fold `function_family`'s posteriors too (§7.4) — it reads the same cached
   array and is currently blind to the fold.
2. Median / trimmed mean instead of the plain mean (§7.2).
3. Per-occurrence reliability weighting, so a masked pass counts less.
4. Widen the benchmark past n=7 before trusting `agree_min=0.60` as a constant.
