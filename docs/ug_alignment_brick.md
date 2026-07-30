# UG-tab → audio aligner (brick)

**What it does in one line.** Takes an Ultimate Guitar tab and our audio and
gives every chord in the tab a start/end time in that audio, so a >4.7★ human
chart becomes *time-bearing reference ground truth* for the songs we have no
iReal source for.

Runnable: `.venv/bin/python scratchpad/ug_align.py <slug> <ug_url_or_cached_html>`
(add `--asr` for phase 2). Outputs `scratchpad/ug_align_<slug>.{json,png}`.

---

## Why a forced aligner and not "match our chords to the tab"

The tab knows **which** chords and in **what order**; it says nothing about
**when**. Our model knows when things change but is the thing under test. So
the tab's chord *sequence* is held fixed and only its *timing* is searched: cut
the audio into N contiguous spans, one per tab chord, in tab order, scored by
chord-tone/chroma agreement. Nothing in the alignment reads our inferred
chords — they appear only in the diagnostic plot — so the output is not
circular with the model it is meant to judge.

## Method (phase 1 — harmony only)

| step | what | notes |
|---|---|---|
| 1 | parse tab | `js-store` → `wiki_tab.content`; `[ch]` chords, `[Section]` headers, `xN` repeats expanded, `(No music)` stretches remembered |
| 2 | **capo → sounding pitch** | applied at parse time, once. Everything downstream is sounding pitch. |
| 3 | chord → 24-d template | treble 12 = `local_key.chord_pcs` weights (**chord-tone**, so Bb sits nearer Gm than F — never root-only); bass 12 = slash-bass/root + fifth |
| 4 | audio → 24-d features | NNLS bothchroma (cached), A-first → C-first, log-compressed, per-half L2 |
| 5 | segmental Viterbi | one span per chord, in order, log-ratio duration prior against the tab's own char-span weights, free head/tail |
| 6 | identifiability audit | can harmony time this tab **at all**? |

**Duration prior.** A chord's relative length comes from the tab itself: the
character span between it and the next chord over the lyric line beneath it.
Bare chord lines (no lyrics under them) get one bar each instead — their
spacing is cosmetic. The prior is `beta * log(d / d_expected)²`, soft enough
that the audio can overrule the tab, strong enough that 161 chords do not
collapse onto one bar.

**Free head/tail (the intro problem).** Costs are centred on
`c_skip = mean(C)` before the DP, so an *uncovered* frame costs exactly the
same as a frame covered by an average-fitting chord. An intro, a fade, or an
outro the tab never wrote is therefore skipped for free, and the tab is only
stretched over audio it explains **better than average**. This is the
order-prior-with-leeway idea in its cheapest form: order is enforced by
monotonicity, leeway lives in the free ends and in the softness of `beta`.

## Identifiability audit — the part that must not be hidden

A plausible-looking alignment on a one-chord vamp is not an alignment. Two
numbers decide whether to believe the output:

- **`boundary_contrast`** — mean chord-tone distance between *consecutive* tab
  chords. If the tab is a vamp this is ~0 and no chroma method can time it.
- **`cost_contrast`** — how much worse 200 *random* monotone segmentations are
  than the fitted one, in units of their own spread (σ). This is the honest
  question: is the printed path special, or is the cost surface flat?

Verdict is `harmony-underdetermined` if `boundary_contrast < 0.12` or
`cost_contrast < 1.0σ`; `weak` under 2.5σ; else `ok`.

| song | Δtpl | contrast | verdict |
|---|---|---|---|
| This Love | 0.755 | **14.6σ** | ok |
| Close to You | 0.381 | **11.2σ** | ok |
| Chain of Fools | **0.000** | **−0.9σ** | **harmony-underdetermined** |

Chain of Fools' −0.9σ means the fitted path is, on average, *worse than a
random cut* — the cost surface is flat and the printed timings are one of
millions of equally good ones. The script says so out loud and tells you to
use `--asr`. (Its Δtpl is exactly 0.000 because `chord_pcs` gives a bare minor
triad a b7, so `Cm` and `Cm7` are literally the same template — see
"limitations".)

---

## Phase 1 landmark table

Reported as measured. Misses are misses.

| song | landmark | target | got | err | |
|---|---|---|---|---|---|
| This Love | chorus-tail F #1 | 57.9 | **57.7** | −0.2 | HIT |
| This Love | chorus-tail F #2 | 108.4 | **108.2** | −0.2 | HIT |
| This Love | chorus-tail F #3 | 158.9 | **158.7** | −0.2 | HIT |
| This Love | intro+V1 G7s inside 0–40 s | — | 4× G-rooted in **0.8–31.2 s** | — | HIT |
| Close to You | Db-section start (±3) | 98.0 | **97.0** | −1.0 | HIT |
| Chain of Fools | "(No music)" verse-2 passage in ≈75–95 s | 85.0 | 78.5 | −6.5 | HIT **but void** |

**Chain of Fools' "hit" does not count and is not claimed as one.** With
contrast −0.9σ the alignment carries no information; landing in the window is
luck. Phase 2 is what makes that row mean something.

Root agreement with our own chart (0.5 s frames, over the aligned span):
This Love **82.8%**, Close to You **72.5%**, Chain of Fools 61.4% (void).

---

## Four real bugs this found, all caught by landmarks

1. **The DP objective shrank with span length.** With a raw (non-centred) cost
   the total is a sum over covered frames, so the cheapest solution is to cram
   every chord into the shortest legal span. All 119 This Love chords landed in
   the last 40 s (chord #0 at 162.3 s) while every intermediate number looked
   plausible — cost, contrast, verdict `ok`. Only the landmark table exposed it.
   Textbook error-pattern #1. Fixed by centring on `mean(C)`; contrast on This
   Love went 3.4σ → 14.6σ.
2. **`[Chorus]` was not being read as a section header** — a guard meant to
   skip the `[ch]` markup token also swallowed every label starting "ch", so
   chorus chords inherited the previous section. It silently widened the
   "verse G7" landmark from 4 chords to 6 and out of its window.
3. **Hard anchor windows made the problem infeasible.** With 40 anchors on
   This Love every DP path hit `INF`, `argmin` returned frame 0, and the
   aligner emitted an all-zeros alignment *while still printing a cost, a
   contrast and a verdict*. Fixed by making anchors a quadratic pull instead
   of a mask, plus an explicit infeasibility raise.
4. **`difflib` matched the first chorus to the last chorus** (+101 s, and
   monotone, so the monotonicity check passed it). See "rate consistency".

## What does NOT work

- **Harmony alone cannot time a static-harmony song.** Chain of Fools is one
  Cm/Cm7 vamp end to end. This is detected, not papered over, but phase 1 has
  no answer for it.
- **`Cm` and `Cm7` are the same template.** `chord_pcs` adds a b7 to bare minor
  triads (`ivs.setdefault(10, 0.8)` on any `-`-quality token), so minor-triad
  vs minor-7 distinctions are invisible to this aligner. Fine for roots,
  useless for adjudicating m vs m7.
- **A capo is not optional.** This Love's best tab (4.86★/2412) is written in
  Am shapes with capo 3 and *sounds* in Cm. Without applying the capo the tab
  is a minor third off and the alignment is nonsense — so the tonality
  cross-check (`tab tonality` vs `infer_key` on our chroma) runs on every song
  and prints MATCH/MISMATCH before anything else.
- **Unwritten repeats stretch.** If the recording repeats a chorus the tab
  wrote once, the DP spreads that chorus's chords over the extra time rather
  than reporting a structural gap. Not handled; see "next".
- **Tabs contain material the recording does not.** Alternate endings,
  simplified appendices, "here it is again without the modulation". The DP
  will place them somewhere. Only the per-chord `support` number catches this,
  and only after the fact.
- **`infer_key` cannot arbitrate mode.** On Close to You it answers "C minor,
  conf 1.00" for a C-major song. The cross-check therefore compares **tonic
  only**, never mode.

---

## Phase 2 — lyric anchors (ASR)

Whisper (`openai-whisper`, already installed in `.venv` — no new dependency)
transcribes the audio with word timestamps. Each tab chord carries the lyric
fragment written under it; a fragment is promoted to a **hard anchor** only if
a run of ≥4 normalised words appears **exactly once** in the remaining ASR
stream. Uniqueness is the whole point: a repeated hook line anchors nothing.

Anchors become **soft** windows — `chord j should start in [t−4 s, t+1 s]`,
enforced by a quadratic pull, not a mask (see "bugs" below) — and the same
segmental Viterbi runs between them.

**Cost.** `openai-whisper` `small`, CPU, ~40 s per song (0.2× realtime).
Nothing installed. Whisper's *defaults* are unusable on music: they declared
the first 32 s of Chain of Fools non-speech and returned 135 words for a 169 s
song. Disabling the no-speech / logprob / compression gates and
`condition_on_previous_text` gives 590 words over the full span, same runtime.

**Two anchor filters, both load-bearing.**

1. *Rate consistency.* Repeated lyrics make `difflib` lock onto the wrong
   repetition. On This Love it matched the **first** chorus's words to the
   **last** chorus's audio: 11 of 40 anchors were +101 s out — and *monotone*,
   so monotonicity did not catch them. The test that does is on the implied
   pace: consecutive kept anchors must imply between 0.05× and 6× the song's
   mean seconds-per-chord. The bad gap implied 52 s/chord, the next 0.02
   s/chord. Longest valid chain by DP; 10 anchors dropped, all 4 landmarks
   restored.
2. *"(No music)" spans bound the next chord.* The tab says one chord holds
   across the a-cappella passage, so chord *j+1* cannot start before it ends.
   Without this the duration prior cut Chain of Fools' chord 20 at 79.2 s
   though the passage it owns runs to 96.0 s.

### Phase-2 landmark table

| song | landmark | target | phase 1 | phase 2 | |
|---|---|---|---|---|---|
| Chain | "(No music)" verse-2, **ASR-timed** | 75–95 s | — | **82.2–96.0 s**, 20 words | HIT |
| Chain | "(No music)" verse-2, **via alignment** | 85 ±10 | 74.8 (void) | **83.3** (span 70.7–96.0) | HIT |
| This Love | 3× chorus-tail F | 57.9/108.4/158.9 | 57.7/108.2/158.7 | **unchanged** | HIT |
| This Love | intro+V1 G7s in 0–40 s | — | 0.8–31.2 s | **unchanged** | HIT |
| Close | Db-section start | 98.0 ±3 | 97.0 | **97.0** | HIT |

Root agreement vs our chart: This Love 82.8% (unchanged), Close **72.5 →
75.3%**, Chain 61.4%.

**Chain of Fools is cracked.** Its harmony still says nothing —
`cost_contrast` is 0.83σ and the verdict stays
`harmony-underdetermined/ASR-anchored`, deliberately: the *timing* is now
pinned by lyrics, the *harmony* is still mute, and those are different claims
that must not be merged into one "ok".

### What the Chain of Fools plot shows about our own model

With the tab aligned, our chart's excursions to Eb / F / A between ~85 and
100 s sit **exactly inside** the a-cappella passage where the tab says nothing
is playing. That is the chord-vs-no-chord failure predicted in
`docs/harmonic_key_second_song.md`, now with timestamps instead of a guess.

---

## Audio support: catching tab material that is not in the recording

Global numbers can all look fine while a chunk of the alignment is fiction.
The 4.83★ Close to You tab appends an **alternate all-C ending** after the Db
modulation — 146 lines, the tail is a no-modulation version for players who
skip the key change. The DP stretched it over the real Db outro and nothing
complained.

So each chord now reports `support` = how much better than a neutral chord the
audio backs it over its own span. Runs of contradicted chords are printed:

    [support] 201.4-208.7s  chords 120-134 (15)

`unsupported_frac` (0.217 on Close, 0.229 on Chain) belongs in the same gate as
`cost_contrast`.

---

## Bonus adjudication: Close to You's late section — Db, not Ab

The v7b doc flagged a plagal ambiguity in the modulated section (Db, Db6, Bbm,
Eb, Abmaj7): tonic Db or tonic Ab?

Counting diatonic membership argues for **Ab**, and convincingly: over 97–197 s
the aligned chords are Db (IV), Cm (iii), Fm (vi), Bbm (ii), Eb (V), Abmaj7
(I△7) — all diatonic to Ab — with C7 → Fm and F7 → Bbm reading as textbook
V/vi and V/ii. Under Db, three of those (Cm, Eb, Abmaj7) are non-diatonic;
Abmaj7 contains G natural, which is #4 of Db.

**That argument is a trap, and the alignment is what exposes it.** The late
section is the earlier section transposed up one semitone — not approximately,
literally:

    early (+1 semitone):  Db C7 C- F- Db Ab Db C7 C- F- Db Ab Db6 Db Db6 …
    late               :  Db C7 C- F- Db Ab Db6 Db Db6 Db Db6 Db C- F F7 …

(`difflib` ratio 0.754 on chord-change sequences of length 33 vs 36; the
mismatch is one extra internal repeat, not different harmony.)

So the identical diatonic-counting argument applied to the **early** section
would make it G major, not C — same shape, same subdominant-heavy vocabulary,
same maj7-on-the-dominant (the tab writes **Gmaj7** there, exactly where the
late section writes Abmaj7). The tab is rated 4.83★ over 1104 votes and
declares tonality **C**. The plagal reading is therefore an artifact of the
progression's shape, available equally in both sections, and wrong in both.

**Verdict: the late section's tonic is Db** (Ab is its V), because it is the
C-major section a semitone up. v7b's 98 s modulation boundary is confirmed at
97.0 s. This is the kind of question the aligner is *for*: it settles a
tonic dispute by showing two passages are the same music, which no amount of
chord-vocabulary counting could do.

---

## How the colour / audit layer should consume this

The colour tracker's open problem (`docs/harmonic_key_second_song.md`) is that
it has no external reference: wrong tonic, wrong mode and genuinely wrong
chords all present as "audit storm". This brick supplies the reference.

1. **Gate on the audit block first.** `verdict == "harmony-underdetermined"`
   means *do not use these timings for anything*. Consume `cost_contrast` as a
   per-song confidence, not a boolean.
2. **Adjudicate per chord, not per song.** For each of our chart chords, look
   up the UG chord covering its midpoint. Root mismatch on a >10σ song is
   evidence against *us*; on a <2.5σ song it is evidence about nothing.
3. **Mode/key audit before chord audit.** The aligned tab gives a *time-varying*
   chord reference, so the v7b tonic track can be scored against it directly
   instead of against its own audit-proposal histogram.
4. **No-chord gating.** `(No music)` stretches in the tab are the ground truth
   for "no instruments are playing" — exactly the chord-vs-no-chord
   discrimination that keeps failing. Once phase 2 pins them, they are a
   labelled negative set, corpus-wide and free.
5. Do **not** feed alignment output back into the chord model's own emissions —
   that is the circularity this design was built to avoid. It is an evaluation
   and audit signal.

## Next (ranked)

1. Structural gap detection: flag sections whose aligned duration exceeds their
   duration prior by >2× — that is an unwritten repeat, not a slow chorus.
2. Scale to the corpus: for every `docs/plots/inferred_*.html` with audio, find
   the top-rated UG tab, run the aligner, keep only `verdict == "ok"` songs.
   That set is the reference benchmark.
3. Beat-sync the features (currently a flat 0.1 s grid) so boundaries snap to
   downbeats rather than to arbitrary frames.
