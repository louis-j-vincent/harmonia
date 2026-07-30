# Bass-informed root discrimination — premise verified, repair refuted

Mission (Louis, 2026-07-30): half of Let It Be's missing chords are the decoder
swallowing a passing chord that shares tones with its neighbour — all 13 `D-7`
come out as `F:maj`, because `Dm7 = D F A C` contains `F A C` and only the D in
the bass tells them apart. Recover them with the bass at decode time.

**Verdict: the premise is real and narrow; the repair does not pay.** The bass
evidence exists (8/13, specific), the mechanism is Let It Be's alone (0/46 on
five other songs), and every configuration of the repair costs more wrong chords
than it saves missing ones. One unexpected lead came out of it and is worth more
than the thing I was asked to build — see *The Chain Of Fools surprise*.

Scripts: `scratchpad/bass_premise_check.py`, `scratchpad/bass_soft_evidence.py`,
`scratchpad/bass_root_recovery.py`, `scratchpad/musx_absorption_split.py`.

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

## 5. The Chain Of Fools surprise — the lead worth following

The rule fires 9 times on Chain Of Fools and **9/9 land on a UG `C-`/`C-7`**, the
highest precision of any song. It nets −3 errors there (QUALITY 19 → 16, ROOT
1 → 0, ADDED 57 → 58) on the worst-scoring song in the report.

What it is doing there is *not* recovering a passing chord. It is reading a
decisive C bass under our `Eb` and concluding `C-7` — `Eb = Eb G Bb ⊂ C Eb G Bb`.
That is exactly the report's own `Eb → C-7` root error and its 14 `dom → min` /
8 `maj → min` quality errors, i.e. **the QUALITY and ROOT classes, not MISSED.**

So the bass discriminator's real value may be **relabelling a chord we already
wrote, not inserting one we did not**. Relabelling does not perturb the ordinal
diff, does not invent events, and aims at a 29-error class instead of a 12-error
one. That is a different experiment and it has not been run.

## What this does NOT solve (CLAUDE.md rule #4)

* **The 14 PRESENT misses.** They are in music-x-lab's decode already and are
  discarded between it and the baked chart — including two segments of 3.4 s,
  which is 4+ beats and cannot be a sub-beat grain problem. No bass work touches
  them. This is still the single largest identified bucket and it needs a
  stage-by-stage trace, not a model change.
* **The 12 NO-EVIDENCE misses.** Three of Let It Be's are `C` where the sounding
  bass genuinely is E (`C/E` in the descending line): the bass is right and the
  tab's root is not the bass, so a bass-rooted rule can never name them. A
  functional-grammar prior, not a bass argmax, is what that class needs.
* **The three `G` absorbed by `C:maj/5`.** Condition 1 refuses them by design (G
  is in the C triad). Whether `C/G` should have been `G` is a fifth-inversion
  question and belongs to `harmonia/models/fifth_discriminator.py`.
* **Anything live.** Nothing here touches `harmonia/**`. The measurement is the
  *ceiling* of the idea on the baked chart; the shipped chart layer never emits a
  chord under ~0.8 beats, so a live version would need that floor lifted first.
* **Generalisation.** Every threshold here (1.5× margin, 0.30 s window, 0.25 s
  minimum run) is a single number on ≤7 songs — a hypothesis, not a law.
* **Hot N Cold's replay** is not faithful (11 vs the published 10 MISSED),
  because the stored JSON does not carry the UG side's audio-support value and it
  is reconstructed as neutral. Its row is reported and excluded from conclusions.

## Recommendation

1. **Do not ship the split/insert repair.** It fails its own acceptance test.
2. **Run the relabel-only variant on the QUALITY/ROOT classes** (the Chain Of
   Fools mechanism): where a decisive sounding bass makes the written chord a
   strict subset of a chord rooted on that bass, change the label in place — no
   split, no insertion. Target: Chain Of Fools' 19 QUALITY + the report's 6 ROOT.
3. **Trace the 14 PRESENT-but-lost misses** through the post-musx chart layer.
   That is a plumbing bug with a 3.4 s smoking gun and no modelling risk, and it
   is a bigger bucket than anything the bass can reach.
