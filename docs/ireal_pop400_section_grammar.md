# iReal Pop 400 — section architecture & grammar

Empirical priors on section structure, mined from 345 human-written lead
sheets (iReal Pro "Pop 400" playlist, `data/ireal/pop400.txt`). Purpose: give
the section-inference stage (boundary detection, folding N occurrences of a
repeated section into one canonical template) numbers to design against.
Reproduce with `.venv/bin/python scratchpad/pop400_section_grammar.py`
(needs `PYTHONPATH=.`).

Vocabulary: an **occurrence** is a maximal run of consecutive bars sharing
one section label (e.g. the form `i A B A B C B` has 7 occurrences). This is
what gets "folded" downstream, so most numbers below are occurrence-level,
not bar-level.

## Actionable conclusions

1. **Length prior should peak at 8 and 16 bars, not be flat.** 45.7% of all
   occurrences are exactly 8 or 16 bars (32.4% + 13.3%); 66.4% are a
   multiple of 4; 58.9% are a power of two. A boundary search with a uniform
   or continuous length prior is throwing away most of the signal.
2. **Fold-to-canonical (stack all occurrences of a label at one length) is
   safe only ~47% of the time.** Only 46.8% of (tune, label) groups with
   ≥2 occurrences share exactly one bar length corpus-wide. The
   occurrence-merge work (`fold_letter_groups`) needs a per-occurrence
   length-compatibility gate, not a rigid stack — this is the corpus-level
   confirmation behind Levier 1 of `docs/handoff_2026-08-08_merge_occurrences.md`.
3. **When lengths disagree, the deviant is almost always the LAST
   occurrence (92.5% of inconsistent groups).** Length drift is a tail
   phenomenon, not scattered across the form. Don't spend alignment budget
   equally on every occurrence — the last one specifically needs the
   phase-search/reject-if-no-match treatment.
4. **The tail deviation is usually an extension, not a truncation.** Of
   tunes where the final label recurs, 70.8% end LONGER than that label's
   in-tune canonical length, 15.6% end shorter, 13.7% match exactly. When
   folding, expect the last occurrence to carry a tag/turnaround/outro —
   truncate it toward the template rather than stretching the template to
   fit it.
5. **"…X X end" (immediate same-label repeat at the very end) cannot occur
   in this representation and isn't a corpus finding** — an occurrence is
   *defined* as a maximal same-label run, so consecutive occurrences never
   share a label (self-transition rate is 0% by construction, at every
   position in the form). Don't build a detector looking for this; it's
   ruled out by the representation, not by the music.
6. **B, not A, is the section that closes the song.** B is the final
   occurrence in 40.6% of tunes vs. 16.8% for A (tied with D) and 25.5% for
   C. If a section-inference stage needs a cheap "which letter is probably
   the chorus" prior from end-position alone, bet on B over A by >2×.
7. **The grammar is verse→chorus alternation, not AABA.** A's single most
   likely exit is B (83.6% of A→X transitions); the single most common
   occurrence-string is `iABAB` (10.7% of tunes alone), and the `iAB...`
   family dominates the top-20 list. A literal `ABA` return-to-verse
   substring appears in 51.9% of tunes, full `ABAB` alternation in 41.4%.
   Classic 32-bar AABA is not visible at the occurrence level by
   construction (see #5) — this corpus's charts read as pop verse/chorus
   form, not jazz-standard AABA.
8. **Intros are short and unusually power-of-two-locked.** Label `i` has
   modal length 4 bars (36.1% of its occurrences), median 6, and 80.8%
   power-of-two lengths — a much tighter distribution than any lettered
   section (A/B/C/D are 49–63% power-of-two). An intro-length prior can be
   narrower than a verse/chorus prior.
9. **Most charts use 3 or 4 distinct labels, not more.** 71.4% of tunes use
   exactly 3 or 4 distinct labels (32.8% + 38.6%); only 0.6% use 1 and 20.6%
   use 5 (the observed max). A section-inference run that discovers >5
   distinct sections on a pop song is very likely over-segmenting relative
   to how these charts are actually written.
10. **Clean half/double relationships explain a minority of length
    disagreements.** Of deviant occurrence lengths, 22.4% are exactly half
    or double the modal length, but 54.3% are irregular deltas with a
    median absolute delta of 6 bars. A folding heuristic that only tries
    {modal, modal/2, modal×2} as candidate lengths will catch under a
    quarter of real-world deviations — expect to need a more general
    length-search, not a fixed small candidate set.

## Method

`sectionized_measures(tune)` (`harmonia/data/ireal_corpus.py`) returns one
`(label, chord-string)` per bar in expanded (repeats-filled) order. Bars are
grouped into occurrences by run-length-encoding consecutive equal labels.
All numbers below are from this pipeline, corpus-wide (345 tunes), unless a
row says "4/4 only".

## 0. Coverage

The file is named "pop400" but contains 346 song segments (not ~400) after
splitting its single `irealb://` URL on `===`; all 346 are non-empty. Of
those, **345 parse and 1 fails**: "Tequila" (The Champs) raises a hard
`RuntimeError` in `pyRealParser._fill_codas` — it has 4 coda (`Q`) markers,
and that function only handles 0, 1, or 2 — and is silently dropped by
`load_playlist`'s try/except. This is the same coda edge case flagged as a
"non-blocking warning" in the task brief, but for this one tune it is a full
parse failure, not a warning. No other coverage gap exists.

## 1. Corpus overview

| | value |
|---|---|
| tunes loaded | 345 |
| time signature | 4/4: 90.4% (312), 3/4: 3.2%, 6/8: 3.2%, others (1/2, 6/4, 5/4, 2/2, 9/8, 2/4): ≤1.2% each |
| expanded form length (bars) | median 70, Q1 57, Q3 84, range 24–192 |
| occurrences per tune | median 6, Q1 5, Q3 7, range 1–28 |

Distinct labels per tune:

| distinct labels | tunes | % |
|---|---|---|
| 1 | 2 | 0.6% |
| 2 | 26 | 7.5% |
| 3 | 113 | 32.8% |
| 4 | 133 | 38.6% |
| 5 | 71 | 20.6% |

Label totals (bars / occurrences) across the corpus — A and B dominate, C is
roughly half of B, D is rare:

| label | bars | occurrences |
|---|---|---|
| A | 9929 | 709 |
| B | 8247 | 690 |
| C | 3891 | 326 |
| i | 1995 | 313 |
| D | 1270 | 106 |

## 2. Occurrence lengths

Share of occurrences at each bar count, plus aggregate stats (`mult4` =
% multiple of 4, `pow2` = % power of two, `odd` = % odd bar count). All time
signatures:

| label | n | median | mult4 | pow2 | odd | 2 | 4 | 8 | 12 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ALL | 2144 | 8.0 | 66.4% | 58.9% | 18.3% | 1.4% | 10.1% | 32.4% | 5.2% | 13.3% | 1.7% | 1.1% |
| A | 709 | 10.0 | 72.6% | 63.0% | 14.8% | 0.1% | 2.0% | 36.1% | 3.9% | 22.0% | 2.4% | 2.5% |
| B | 690 | 9.0 | 61.3% | 50.4% | 21.3% | 0.6% | 8.1% | 30.3% | 7.0% | 10.6% | 1.7% | 0.6% |
| C | 326 | 9.0 | 58.3% | 49.7% | 23.6% | 1.2% | 8.0% | 29.4% | 5.8% | 10.4% | 1.8% | 0.3% |
| i | 313 | 6.0 | 74.1% | 80.8% | 11.2% | 6.1% | 36.1% | 33.5% | 1.6% | 2.6% | 0.0% | 0.0% |
| D | 106 | 9.0 | 60.4% | 49.1% | 26.4% | 1.9% | 6.6% | 26.4% | 11.3% | 14.2% | 0.9% | 0.0% |

4/4 only (90.4% of tunes) — materially unchanged, so 3/4 and 6/8 tunes are
not skewing the picture:

| label | n | median | mult4 | pow2 | odd |
|---|---|---|---|---|---|
| ALL | 1946 | 8.0 | 66.8% | 58.8% | 18.2% |
| A | 640 | 10.0 | 73.9% | 63.9% | 13.9% |
| B | 635 | 9.0 | 60.6% | 49.4% | 22.2% |
| C | 294 | 9.0 | 58.5% | 49.7% | 24.8% |
| i | 284 | 6.0 | 73.9% | 80.3% | 10.9% |
| D | 93 | 9.0 | 64.5% | 51.6% | 22.6% |

Note: the median for A (10) sits between its two dominant modes (8 bars,
36.1%; 16 bars, 22.0%) — read the bucket table, not the median, as the
shape. The single most common exact length, every label included, is 8 bars
(694/2144 = 32.4% of all occurrences); 16 bars is second (286/2144 = 13.3%).

## 3. Within-tune same-label length consistency

598 (tune, label) groups have ≥2 occurrences.

| | count | % |
|---|---|---|
| all occurrences share one length | 280 | 46.8% |
| ≥1 occurrence deviates | 318 | 53.2% |

Of the 318 inconsistent groups, position of the deviant:

| | count | % |
|---|---|---|
| last occurrence is (a) deviant | 294 | 92.5% |
| deviant is earlier, last occurrence matches modal | 24 | 7.5% |

Shape of the deviation (383 deviant instances total; a group can contribute
more than one):

| pattern | count | % |
|---|---|---|
| double (2× modal) | 38 | 9.9% |
| half (0.5× modal) | 48 | 12.5% |
| +2 bars | 25 | 6.5% |
| −2 bars | 14 | 3.7% |
| +4 bars | 34 | 8.9% |
| −4 bars | 16 | 4.2% |
| other | 208 | 54.3% |

Even within "other", most deltas are not huge: 41.3% of all deviant
instances are within ±4 bars of modal, 19.3% within ±2 — but the median
absolute delta is 6 bars, so the typical disagreement is a real structural
difference (an extra half-chorus, a cut verse), not a rounding artifact.

**Spot check** (half/double cases, to confirm the pattern is real and not a
labeling bug): "Baker Street" A = [16, 8] — a shortened final verse.
"Blue Suede Shoes" A = [12, 24] — one A occurrence is a literal double
length, i.e. that verse plays through twice with no other label in between
(so it stays one occurrence by the run-length definition, §method) while
another instance of A elsewhere in the tune is a single pass. "I'm Yours"
A = [8, 9, 8], B = [8, 8, 9] — a 1-bar tag on the middle occurrence of each,
consistent with a live turnaround, not a parse error.

## 4. Tail behavior

Final occurrence's label (n=345 tunes):

| label | count | % |
|---|---|---|
| B | 140 | 40.6% |
| C | 88 | 25.5% |
| A | 58 | 16.8% |
| D | 58 | 16.8% |
| i | 1 | 0.3% |

| | count | % |
|---|---|---|
| final label already appeared earlier in the tune | 212 | 61.4% |
| final label is new material (first appearance) | 133 | 38.6% |
| form ends "…X X" (immediate same-label repeat) | 0 | 0.0% (impossible by construction, §method) |

Of the 212 tunes where the final label recurs, its length relative to that
label's in-tune canonical (modal) length:

| | count | % |
|---|---|---|
| at canonical length | 29 | 13.7% |
| truncated (shorter) | 33 | 15.6% |
| extended (longer) | 150 | 70.8% |

**Spot check**: the one tune ending on `i` is "I Can't Make You Love Me" —
`i(8) A(11) B(30) i(14)`, a 14-bar outro reprise of the intro material,
which is a real and sensible pop-ballad ending, not a mislabel.

## 5. Sequence language

Top occurrence-level label strings (collapsing immediate duplicates changes
nothing here — see §method, it's impossible by construction — so the two
views the brief asked for are identical):

| form | count | % |
|---|---|---|
| iABAB | 37 | 10.7% |
| iAB | 17 | 4.9% |
| iABABCB | 16 | 4.6% |
| iABC | 15 | 4.3% |
| iABA | 12 | 3.5% |
| iABABC | 10 | 2.9% |
| iA | 9 | 2.6% |
| iABABAB | 8 | 2.3% |
| iABCD | 8 | 2.3% |
| iABABA | 8 | 2.3% |
| iABCABC | 7 | 2.0% |
| iABABCD | 7 | 2.0% |

Bigram transition matrix, row-normalized % (row = from, col = to; blank/0
cells omitted for space, see script for the full START/END matrix):

| from \ to | i | A | B | C | D | END |
|---|---|---|---|---|---|---|
| START | 85.2% | 13.6% | 0.6% | 0.6% | 0.0% | — |
| i | 0.0% | 95.2% | 4.5% | 0.0% | 0.0% | 0.3% |
| A | 0.7% | 0.0% | 83.6% | 6.1% | 1.4% | 8.2% |
| B | 1.2% | 37.7% | 0.0% | 38.0% | 2.9% | 20.3% |
| C | 1.5% | 27.6% | 20.6% | 0.0% | 23.3% | 27.0% |
| D | 0.9% | 13.2% | 13.2% | 17.9% | 0.0% | 54.7% |

(Diagonal is 0% everywhere — impossible by the occurrence definition, §method;
this is not a music fact and the brief's "self-transition by position in the
form" analysis is correspondingly ill-defined at the occurrence level, so it
is omitted here rather than reported as a 0% "finding".)

Verdict-support numbers:

| statement | number |
|---|---|
| B is the final occurrence | 140/345 = 40.6% |
| A is the final occurrence | 58/345 = 16.8% |
| tune's labels are a subset of {i, A, B} only | 121/345 = 35.1% |
| occurrence string contains literal "ABA" | 179/345 = 51.9% |
| occurrence string contains literal "ABAB" | 143/345 = 41.4% |
| A's most likely exit is B | 83.6% of A's transitions |
| B's exits split almost evenly A / C / END | 37.7% / 38.0% / 20.3% |

**Verdict**: verse–chorus alternation, not AABA. A almost always leads into
B (83.6%); B is the section most likely to close the song (40.6% of tunes,
2.4× A's rate) but also the section most likely to hand back to A (37.7%) —
B is the structural pivot, not A. AABA cannot even be represented as a
literal occurrence-string pattern here (self-transitions are impossible by
construction), which itself is evidence this corpus's grammar is pop
verse/chorus, not jazz-standard 32-bar AABA form (where the repeated A
sections are the norm this representation would need to show self-transitions
for, if sections were annotated per-repeat instead of per-run — they are not,
see caveats).

## Caveats

**What pyRealParser expands** (verified by reading
`.venv/lib/python3.12/site-packages/pyRealParser/pyRealParser.py`):
simple `{...}` repeats; repeats with first/second (`N1`/`N2`) endings,
including chained multi-ending repeats; one- and two-bar repeat symbols
(`x`, `r`); slash/repeat-previous-chord (`p`); and D.S./D.C. al Coda when
there are exactly 0, 1, or 2 `Q` (coda) markers — the 2-marker case copies
the segno-to-first-coda span verbatim into the second coda's position.

**What it does not expand / cannot handle**:
- Free-text performance instructions inside `<...>` comments (e.g. "play 4x",
  "repeat and fade") are stripped as comments, not interpreted — the
  expanded output reflects the single written iteration, so a chart's
  written form can *undercount* a performed arrangement's true length. This
  is exactly the caveat the task brief asked to verify, confirmed by code
  reading rather than corpus measurement (too rare/free-text to detect
  reliably in bulk).
- Charts with >2 coda (`Q`) markers hard-fail rather than expand (§0,
  "Tequila" is the corpus's one instance).
- This project's own monkey-patch (`_fill_long_repeats_fixed` in
  `harmonia/data/ireal_corpus.py`) already fixes a bar-eating bug in
  upstream `_fill_long_repeats`'s simple-repeat branch (confirmed on
  "Autumn Leaves"); that fix is in effect for every number in this report.

**iReal charts are lead sheets, not performed arrangements.** A chart's
expanded form is what a musician would read off the page; a real recording
can add/drop bars (extra turnaround, cut verse, DJ edit) that the chart
never encodes. Treat every number here as a prior over *how charts are
written*, not a ground truth for *how songs are performed* — consistent with
this project's existing trust order (iReal > tabs > model output) being
about chord content, not necessarily exact performed bar-counts.

**Self-transition / "…X X" analyses are ill-defined at the occurrence
level, by construction, not by corpus fact** — an occurrence is a maximal
run of one label, so two consecutive occurrences can never share a label.
Anywhere the brief asked for this (self-transition rate, position-in-form
self-transition, "ends with immediate duplication"), the true answer is a
representational 0%/impossible, not a music-language finding, and is
reported as such rather than presented as a discovered rate.

**Time signature**: 90.4% of the corpus is 4/4; restricting every length
stat to 4/4-only moves the headline numbers by ≤2 points (§2), so the
mixed-time-signature tunes are not distorting the length-prior conclusions.
