# Bass-informed root discrimination — premise verified, repair refuted

Mission (Louis, 2026-07-30): half of Let It Be's missing chords are the decoder
swallowing a passing chord that shares tones with its neighbour — all 13 `D-7`
come out as `F:maj`, because `Dm7 = D F A C` contains `F A C` and only the D in
the bass tells them apart. Recover them with the bass at decode time.

**Verdict: the premise is real and narrow; the repair is dead.** The bass
evidence exists (8/13, specific), the mechanism is Let It Be's alone (0/46 on
five other songs), and on a **fresh decode of the shipped pipeline the repair
recovers exactly zero misses while inventing seven chords**.

The bigger finding is incidental: re-decoding the songs showed that
**`docs/ug_score_report.md` ranks error classes on charts baked by a pipeline
that no longer exists** — see §5, which is the part worth acting on.

Scripts: `scratchpad/bass_premise_check.py`, `scratchpad/bass_soft_evidence.py`,
`scratchpad/bass_root_recovery.py`, `scratchpad/bass_recovery_fresh.py`,
`scratchpad/musx_absorption_split.py`.

## 1. A calibration bug nearly killed the premise

The first premise run said **0/13 on every bass source** and I was one commit
away from reporting the premise dead. It was my bug. I retyped the NNLS
bothchroma rotation as `-9`; `nnls_features._ROLL_TO_C` is `+9`. Those differ by
18 ≡ 6 (mod 12) — a **tritone** — so every reading was six semitones off and D
was printed as Ab.

This is error-pattern #1 with nothing else changed: a low-level constant error
producing entirely plausible numbers (F chords "reading B", "Ab" — I even wrote a
sentence explaining it as NNLS partial junk, which is a *known real phenomenon*,
so the wrong answer had a ready-made story). The constant is now imported from
its owner and unit-tested against a synthetic one-hot before use.

## 2. Premise check — the evidence is there, and it is specific

Let It Be, 38 MISSED, four bass sources at the aligned spans:

| class | n | meaning |
|---|---|---|
| PRESENT | 14 | music-x-lab already decoded this root — lost *downstream*, no bass work reaches it |
| **BASS** | **10** | a bass source names the root decisively where musx's label does not |
| BASS-WEAK | 2 | right pitch class, margin under the 1.5× guard |
| NO-EVIDENCE | 12 | no bass source names it at all |

The 14 PRESENT reproduce the falsification report's 14 exactly, by an independent
path — that is the harness's own sanity check.

On the 13 `D-7` spots specifically:

| source | hit | decisive (≥1.5×) | fires on plain `F:maj` |
|---|---|---|---|
| **NNLS-24 bass argmax** | **8/13** | **7/13** | 4/71 guarded windows (5.6 %) |
| music-x-lab bass posterior | 0/13 | 0/13 | 0/100 |
| music-x-lab `.lab` bass | 0/13 | — | — |

**The two bass sources are complementary and the useful one is NNLS.**
music-x-lab's bass head reads F at mass 0.50, margin 2.5× — it is smoothed toward
its own chord decode and cannot see a 0.2 s passing note. NNLS is independent and
hears it. The reverse holds for the three `G` absorbed by `C:maj/5`, which only
musx's posterior carries (margin 4.4–10.3×).

This is the sanctioned use of the bass under `docs/harmonic_key_investigation.md`
— the bass as a **decisive NOTE** with a margin guard, never as chroma mass. No
mass from the bass half enters any decision here.

### The negative half, run before building

A sweep on music-x-lab's *secondary* posterior mass (`bass_soft_evidence.py`,
13 positives vs 224 null windows inside `F:maj`) tops out at **precision 0.47**
at any threshold. So the recovery had to run on NNLS; "the D is in there
somewhere" was not enough.

## 3. Cross-song — Let It Be's mechanism does not replicate

Replicating the musx-present/absent split (CLAUDE.md rule #5):

| song | MISSED | PRESENT | ABSENT | ABSENT that are shared-tone absorption |
|---|---|---|---|---|
| Let It Be | 38 | 15 | 23 | **11** (`D-7` swallowed by `F:maj`) |
| Close To You | 25 | 0 | 25 | 0 |
| Every Breath You Take | 6 | 1 | 5 | 0 |
| Hot N Cold | 10 | 3 | 7 | 0 |
| This Love | 3 | 2 | 1 | 0 |
| Stand By Me | 10 | 1 | 8 | 0 |

**Strict shared-tone absorption is 0/46 on every other song combined.** Those 46
split into 15 where musx also decodes explicit no-chord (Stand By Me's intro,
already logged), 24 with *partial* overlap that never clears the subset bar (Hot
N Cold's `A- [C] G`: musx holds `A:min` through the passing C, 2 of 3 notes
shared), and 7 with no shared pitch class at all. Close To You's 25 are a
different artifact entirely — a real Ab/Db coda our chart tracks and the tab does
not.

Correction to the falsification report: the D-7s decode as `F:maj` **11/13**, not
13/13; the other two are `C:maj`.

## 4. The repair — built, measured, refused

`bass_root_recovery.py` audits the baked chart, and where a decisive sounding
bass is a tone the host chord does not contain, splits the span and relabels the
sub-span as the minimal chord rooted on that bass.

The substitute set is **derived, not tabulated** — a hand-written list of
"relative substitute pairs" invites exactly the errors it is meant to fix. For a
host with pitch classes `H` rooted at `R` and a decisive bass `B`, a relabel is
proposed only when (1) `B ∉ H`, (2) some vocabulary quality `S` satisfies
`pcs(B,S) == H ∪ {B}`, and (3) the substitute is rooted on `B`. Condition 1 alone
throws out every inversion: `C/E` and `C/G` can never fire, which is what
protects Let It Be's three `C` chords whose sounding bass really is E.

Run over the vocabulary this yields exactly ten branches. Only one has measured
evidence — major triad + bass a minor third below → `min7` — and firing all ten
is net negative (ADDED +6 / ROOT +3 against MISSED −5). Enumerating tells you
what *could* fire; evidence decides what *should*.

### Result, all 7 songs, best configuration

Validated by replaying `ug_score.py`'s own scorer (imported, never edited) on the
stored sequences. The replay reproduces the published counts exactly on 6 of 7
songs and is flagged unfaithful on the seventh.

| song | fired | MISSED | ADDED | ROOT | QUALITY |
|---|---|---|---|---|---|
| Let It Be | 9 | 37 → **34** | 5 → 9 | 1 → 1 | 0 → 0 |
| Chain Of Fools | 9 | 0 → 0 | 57 → 58 | 1 → **0** | 19 → **16** |
| Stand By Me | 3 | 7 → 7 | 3 → 4 | 0 → 0 | 0 → 0 |
| Close To You | 1 | 25 → 25 | 3 → 4 | 0 → 0 | 2 → 2 |
| This Love | 2 | 3 → 4 | 0 → 1 | 1 → 1 | 0 → 0 |
| Every Breath You Take | 0 | 6 → 6 | 1 → 1 | 3 → 3 | 0 → 0 |
| Hot N Cold *(replay unfaithful)* | 2 | 11 → 11 | 18 → 20 | 0 → 0 | 2 → 2 |
| **total** | 24 | **89 → 87** | **87 → 97** | 6 → 5 | 23 → 20 |

The brief's acceptance test was: MISSED must drop on Let It Be **without** new
ADDED/ROOT on any song. It drops by 3 and costs 4 on Let It Be alone, plus one
new ADDED on four other songs. **The test fails in every configuration tried** —
late-only, tail-absorb, margin 1.5/2.0/3.0, either-source and both-sources. At
margin 2.0 the MISSED benefit disappears entirely while the ADDED cost remains.

### Superseded by §5 — these hosts come from a stale chart

The repair audits `our_seq` as stored in the `ug_score_*.json`, i.e. the baked
`docs/plots/inferred_*.html`. Commit `957971d` established that Let It Be's baked
chart is from 2026-07-21 and predates the music-x-lab `.lab` it was compared
against by 17 hours — it was rendered by a bare `infer_chords_v1` with
`segment_source="nnls"`. So the **host chords the repair splits are not the ones
the shipped config would produce today.** The premise check in §2 is unaffected
(it reads the `.lab`, the frame posteriors and the NNLS cache directly, all
current), but the §4 table should be read as "what this rule does to the chart
the UG report scored", not "what it would do live. Re-running it against
`SHIPPED_CONFIG` output is the obvious follow-up and was not done here.

### It is not the ruler wobbling

Inserting events into an ordinal diff can reshuffle pairings far from the
insertion, which would make the scorer an unfair judge here. Measured: of the 8
new ADDED/ROOT errors, **7 are at a firing and 1 is elsewhere**. The cost is real,
not an artifact.

### But some of the cost is the tab, not us

Of Let It Be's 9 firings, 5 land on a UG `D-7` and 4 do not — and those 4
(130.5 s, 144.4 s, 200.4 s, 231.8 s) are all the *same* F–E–D–C piano fill, at
bass-D mass 0.33–0.35, in the solo and choruses where the tab writes plain
`F | C`. The tab writes that fill as `Dm7` in the verses and omits it elsewhere.
So those four ADDED are the tab's inconsistency (rule #3: ground truth is a
measurement too). They are reported, not smoothed — but note that even
crediting all four, the tally is MISSED −3 against ADDED +0 on Let It Be and pure
cost on the other six songs.

## 5. What "the baked chart is stale" means, and what it costs

### Which pipeline made the chart

`docs/plots/inferred_let_it_be_remastered_2009.html` was last written
**2026-07-21 17:27** by `render_youtube_chart.py` at commit `09d99d7`, which
called `infer_chords_v1(audio_path, seventh_gate=0.0, cache_dir=…)` — **bare**.
`infer_chords_v1`'s defaults at that commit were `feature_frontend="bp48"`,
`bass_frontend="nnls24"`, `quality_frontend="nnls24"`,
`segment_source="nnls"`. Today's `SHIPPED_CONFIG` is `nnls24` features with
**bass, quality and segmentation all from music-x-lab** (`segment_source=
"musx_redecode"`). These are different models end to end. (Established by the
concurrent trace, `docs/postmusx_segment_loss.md`; re-verified here by decoding.)

The physical clincher from that trace: `data/cache/musx_infer/
let_it_be_remastered_2009_submission.lab` is dated **Jul 22 10:10**, the chart
**Jul 21 17:27**. The chart predates the artifact it was compared against by 17
hours.

### What that does to the MISSED attribution

The falsification test in `ug_score_report.md` split Let It Be's 38 MISSED into
"24 absent from musx ⇒ decoder absorption" and "14 present in musx ⇒ lost
downstream". Neither label survives:

| the report said | what it actually is |
|---|---|
| 14 "lost between the musx decode and the chart" | **not lost — never carried.** The chart's pipeline had no musx in it. Of the 14, 8 the chart does write and the ordinal diff failed to pair; 3 are genuine losses under the *shipped* config, and those die in `_split_collapsed_bars_via_musx` |
| 24 "decoder-level absorption" | absorption by the **bp48/nnls24** decoder of 2026-07-21, not by the decoder that ships. On a fresh decode, 10 of Let It Be's misses are simply gone |

"Lost downstream" and "the chart predates the decode" are not the same claim,
and only the second one is true. It was never a plumbing bug on that song.

### The cost: the report's error ranking is measured on a dead pipeline

Same scorer, same UG alignment, only the chart re-decoded under `SHIPPED_CONFIG`:

| | MISSED | ADDED | ROOT | QUALITY |
|---|---|---|---|---|
| baked charts (replayed) | 89 | 87 | 6 | 23 |
| fresh, shipped pipeline | 76 | 30 | 8 | 4 |
| baked, **excluding Chain Of Fools** | 89 | 30 | 5 | 4 |
| fresh, **excluding Chain Of Fools** | **73** | **30** | 8 | 2 |

**Read the bottom two rows, not the top two.** The headline "ADDED 87 → 30" is
entirely one song, and it is not an improvement — it is a collapse. Chain Of
Fools' fresh decode is **5 spans for 169 seconds**: `N`, `C:7` held for 79 s,
`N`, `C:7` held for 61 s, `N`. A chart with two chords in it cannot accumulate
ADDED errors. No warning fired and the pipeline ran normally; the song is a
one-chord modal vamp so a held `C7` is not absurd, but going 100 chords → 2 is a
pathology in its own right and must not be sold as a fix. (It is also why its
QUALITY falls 19 → 2: only two chords remain to be wrong about.)

With that song set aside, the honest comparison is:

* **MISSED 89 → 73** — a real gain, and where it comes from is checkable:
  Let It Be 37 → 27 and Hot N Cold 11 → 6. This is the shipped musx path
  writing passing chords the 2026-07-21 bp48/nnls24 chart never had.
* **ADDED 30 → 30** — flat.
* ROOT 5 → 8 and QUALITY 4 → 2 — small, and within the noise of a comparison
  that reuses a fixed `t_intro` and a reconstructed UG support value.

So the report's *conclusion* — MISSED is the defect class to work on — is
strengthened, not weakened: on the shipped pipeline MISSED is 73 against ADDED
30, i.e. **2.4× the next class**, where the report had them "level overall".
What does not survive is the report's per-song ranking and its absolute counts,
and in particular the framing of Chain Of Fools as "57 ADDED, the one song whose
harmony cannot time itself" — that song's chart no longer exists in that form.

## 6. Fresh-decode validation — the repair recovers nothing

Run: re-decode each song with `SHIPPED_CONFIG`, re-score against the same UG
alignment, then apply the recovery (`bass_recovery_fresh.py`).

| song | chords baked → fresh | fired | MISSED | ADDED | ROOT | QUALITY |
|---|---|---|---|---|---|---|
| Let It Be | 109 → 118 | 12 | 27 → 27 | 2 → **6** | 1 → 1 | 0 → 0 |
| Close To You | 52 → 65 | 2 | 25 → **24** | 2 → 3 | 2 → 4 | 1 → 1 |
| Hot N Cold | 114 → 122 | 2 | 6 → 7 | 22 → 23 | 2 → 2 | 1 → 1 |
| This Love | 121 → 121 | 1 | 3 → 3 | 0 → 1 | 1 → 1 | 0 → 0 |
| Chain Of Fools | 100 → 5 | 0 | 3 → 3 | 0 → 0 | 0 → 0 | 2 → 2 |
| Stand By Me | 41 → 42 | 0 | 6 → 6 | 3 → 3 | 0 → 0 | 0 → 0 |
| Every Breath You Take | 70 → 69 | 0 | 6 → 6 | 1 → 1 | 2 → 2 | 0 → 0 |
| **total** | | **17** | **76 → 76** | **30 → 37** | **8 → 10** | 4 → 4 |

**Zero net misses recovered, seven chords invented, two new root errors.** On
Let It Be — the song the whole mission was built on — twelve firings buy nothing
at all, because the shipped decode *already writes* the D-7s that the 2026-07-21
chart absorbed. The acceptance test ("MISSED down, no new ADDED/ROOT on any
song") fails on the only measurement that counts.

The §4 result (MISSED −3 for ADDED +4) was the best this idea ever looked, and
it was an artifact of auditing a four-model-generations-old chart.

## 7. The Chain Of Fools surprise — and why it also dies

The rule fires 9 times on Chain Of Fools and **9/9 land on a UG `C-`/`C-7`**, the
highest precision of any song. It nets −3 errors there (QUALITY 19 → 16, ROOT
1 → 0, ADDED 57 → 58) on the worst-scoring song in the report.

What it is doing there is *not* recovering a passing chord. It is reading a
decisive C bass under our `Eb` and concluding `C-7` — `Eb = Eb G Bb ⊂ C Eb G Bb`.
That is exactly the report's own `Eb → C-7` root error and its 14 `dom → min` /
8 `maj → min` quality errors, i.e. **the QUALITY and ROOT classes, not MISSED.**

So the bass discriminator's real value looked like **relabelling a chord we
already wrote, not inserting one we did not** — no ordinal-diff perturbation, no
invented events, and a 29-error class instead of a 12-error one.

**The fresh decode kills this lead too, for a boring reason: the errors it aimed
at no longer exist.** Chain Of Fools' 57 ADDED and 19 QUALITY were properties of
the 2026-07-21 chart. Re-decoded, that song has 2 chords, 0 ADDED and 2 QUALITY,
and the rule fires **0 times** on it. The whole target evaporated.

The idea is not refuted — it was never tested against a real target. If the
QUALITY/ROOT classes are worth attacking, the first step is to re-measure them
on fresh decodes and find out whether they still exist anywhere. On the current
7-song set they total **12 errors**, which is not worth a brick.

## What this does NOT solve (CLAUDE.md rule #4)

* **The 14 PRESENT misses.** Already traced by a concurrent session (commit
  `957971d`, landed on this branch while this work was running) — the answer is
  **3 genuine losses, not 14**, and `_split_collapsed_bars_via_musx` is the
  culprit. That trace also found something that bears on the numbers below: the
  **baked Let It Be chart predates music-x-lab by 17 hours** and was produced by
  a bare `infer_chords_v1` with `segment_source="nnls"`. See the caveat under
  §4.
* **The 12 NO-EVIDENCE misses.** Three of Let It Be's are `C` where the sounding
  bass genuinely is E (`C/E` in the descending line): the bass is right and the
  tab's root is not the bass, so a bass-rooted rule can never name them. A
  functional-grammar prior, not a bass argmax, is what that class needs.
* **The three `G` absorbed by `C:maj/5`.** Condition 1 refuses them by design (G
  is in the C triad). Whether `C/G` should have been `G` is a fifth-inversion
  question and belongs to `harmonia/models/fifth_discriminator.py`.
* **Anything live.** Nothing here touches `harmonia/**`.
* **Chain Of Fools' fresh decode**, which is 2 chords over 169 s. Whether that is
  defensible on a one-chord vamp or a real failure needs an ear, not a metric —
  but until someone listens, every number involving that song (in this document
  *and* in `ug_score_report.md`) should be treated as unusable rather than good.
* **Generalisation.** Every threshold here (1.5× margin, 0.30 s window, 0.25 s
  minimum run) is a single number on ≤7 songs — a hypothesis, not a law.
* **Hot N Cold's replay** is not faithful (11 vs the published 10 MISSED),
  because the stored JSON does not carry the UG side's audio-support value and it
  is reconstructed as neutral. Its row is reported and excluded from conclusions.

## Recommendation

1. **Drop bass-informed root discrimination.** It recovers zero misses on the
   shipped pipeline and invents seven chords. The premise was true and the
   evidence was real; the chords it was built to recover are already being
   written.
2. **Re-bake the 7 benchmark charts and re-run `ug_score.py --report` before any
   further work is aimed at that report.** This is the highest-value item here
   and it is cheap — ~30 s of decode per song. Everything in
   `docs/ug_score_report.md` (the class ranking, the per-song table, "MISSED and
   ADDED are level", the whole Chain Of Fools narrative) describes charts from
   2026-07-21. Two sessions have now each spent a day attacking defects measured
   on them.
3. **Listen to Chain Of Fools' fresh decode** (2 chords / 169 s) and decide
   whether it is correct-but-sparse or a collapse. It swings the benchmark's
   headline number by itself.
4. ~~Trace the 14 PRESENT-but-lost misses.~~ **Done concurrently** (`957971d`):
   8 the chart does write and the ordinal diff failed to pair, 3 are genuine, and
   `_split_collapsed_bars_via_musx` deletes 8 one-beat chords it then declines to
   re-emit. Closed as a research question, open as a fix.
5. **Add a staleness guard to the benchmark.** Both traps that cost time here
   were "a warning plus a plausible output": a chart older than the artifacts it
   is scored against, a worktree with no trained heads returning one chord for a
   whole song, a clone without `data/` silently disabling the re-decode. The
   scorer should refuse a payload whose mtime predates the caches it is compared
   with, the way `bass_recovery_fresh.py` now refuses a degraded decode.
