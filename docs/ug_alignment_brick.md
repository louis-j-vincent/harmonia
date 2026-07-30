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

## Two real bugs this found, both caught by landmarks

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

Anchors become interval constraints `chord j must start in [t−4 s, t+1 s]`
(a word starts *inside* its chord's span, not at its onset), and the same
segmental Viterbi runs between them.

Results and cost: see the phase-2 section below.

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
