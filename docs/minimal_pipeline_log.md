# HARMONIA-MIN — minimal pipeline rebuild log

Branch `feat/minimal-pipeline` (from `feat/chord-context-prior`).
Mission: rebuild a working audio → interactive chord chart path from the few
good bricks, ~300 lines of new orchestration, no old-pipeline imports.

## 2026-07-31 — Milestone 1 DONE: chart rendering end-to-end on :7772

`python -m harmonia_min.server` → http://localhost:7772 — the app shell
renders, search finds local audio, Analyse runs the thin pipeline (~4 s on
cached musx posteriors), the chart opens, audio plays, the playhead tracks
the right bar (verified in headless Chrome, screenshots in session scratchpad;
bar-label arithmetic checked against the grid by hand).

### Branch decisions

Skipped ALL ahead branches (none touches a copied brick — verified by diff):
- `feat/harmonic-key` (14 ahead, ACTIVE yesterday): docs + scratchpad UG-scoring
  analysis only. Merging an active session's branch = cross-session conflict risk.
- `feat/downbeat-section-monolith` / `feat/section-cnn*` (11 ahead): section
  detection — out of milestone-1 scope, and sections get scoped with Louis later.

Port note: a leftover `python3 serve.py 7772` (static preview of
`scratchpad/iphone_view`, started 2026-07-30 18:15) was squatting the assigned
port; killed. Restart it on another port if still wanted.

### What was KEPT (copied bricks, each verified against raw data before use)

| brick | copied to | verification |
|---|---|---|
| `musx_redecode.py` | `harmonia_min/musx.py` | posterior layout (T,73)/(T,13)/… rowsums=1.0, T·23.22ms = ffprobe duration ±0.03s |
| beatthis wrapper (re-extracted, ~100 L) | `harmonia_min/beats.py` | This Love 93.75 BPM (true ~95, right octave), 3.94 beats/bar |
| `key_profiles.py` | verbatim | C-triad chroma → "C major" conf 0.91 |
| `nnls_features.py` | copy, scratchpad MLP inlined | checkpoint loads, 7 qualities |
| `span_rescore.py` + `chord_context_prior.py` | copies, QUAL5/SEMITONE_NAMES inlined | import + prior npz loads (NOT yet wired to a route — lock UX is milestone 2) |
| `app_shell.html` | verbatim | renders against the new server |

New code: `labels.py` (musx→iReal-tail mapping, closed 26-label vocab, raises
on unknown — no silent maj default), `pipeline.py` (~230 L orchestration),
`server.py` (~200 L Flask).

### What was CUT

`chord_pipeline_v1` (4,779 L), `chord_head.py` post-passes (Occam, vocab fold,
section arbiter), `chord_hmm`, `section_structure`, repeat folding/endings,
ProgressionEncoder, LLM priors, `/api/reinfer`, annotations, iReal import,
jam/record modes, billboard corpus. Server routes for these return 404; the UI
degrades as designed.

### Deliberate DIVERGENCES from the old pipeline

1. **Bar grid = Beat This!'s real downbeats**, not a synthetic constant-tempo
   lattice, not chord-onset anchors (known_issues "BOTH CLOCKS ARE SYNTHETIC").
   Synthetic bar lines only at the edges (pickup/tail), median-length.
2. **No librosa anywhere, no silent fallback**: beat tracking failure raises
   `BeatTrackingError`. m4a is ffmpeg-transcoded to wav before Beat This!
   (the 2026-07-30 lesson, baked in).
3. **Labels straight from the beat-grid re-decode** — no post-passes at all.
4. **Confidence is acoustic** (mean musx triad posterior over the segment's
   own frames), not the repetition heuristic (whose AUC was 0.480).
5. **Key from decoded chords** (duration-weighted chord tones → KS profiles),
   not a separate chroma pass.
6. **One unfolded section**; `barSpans[r] = [[grid[r], grid[r+1]]]` — the
   playhead map is trivially correct by construction.

### Diffs vs the old pipeline's charts — SIGNAL for Louis, not failures

3 songs analyzed (cached musx posteriors): This Love, Let It Be, Stand By Me.

| song | old chart | new chart | reading |
|---|---|---|---|
| This Love | 81 bars, bar 0 `C G` (roots only at my extraction level), later `D Bb` | 81 bars, `G/B C- | F-7 | % | F-/Ab G/B`, `Dø7 G/B` | new = the actual changes incl. inversions; old/new disagree on bar-0 content and several roots (D/Bb vs Dø7) — worth an ear pass |
| Let It Be | 72 bars, bar 0 crams `C G A` (3 chords/bar) | 77 bars, `C G | A- F | C F | C` | old bar grid glued the intro; new opens on the real changes. 72 vs 77 bars = different grids, not an error flag per se |
| Stand By Me | 89 bars, bar 0 labeled `C` (in A major!) | 93 bars, bar 0 `N.C.` then held | the old "C" was a mislabeled no-chord; new shows the honest N.C. — but 15 intro bars of bass riff stay unlabeled by musx (chord-vs-no-chord discrimination, the known top priority) |

Other observations:
- All 3 songs selected **musx latency 0 ms** (the old 7-song study measured
  60–340 ms optima with per-song selection). Different beat backend feeding the
  grid may explain it; unverified — worth a look before trusting the latency knob.
- Small fixes made during the build: leading-pickup coverage (a chord sounding
  before the first detected downbeat got dropped — Let It Be's opening C was
  20 ms early), leading-N.C. rule (an N segment > half a bar = real played
  intro, shorter = silence).

### How to run

```
.venv/bin/python -m harmonia_min.server        # port 7772, NEVER 7771
# charts persist in harmonia_min/state/charts/*.json
# analyze from the UI (search a local song → Analyse) or pre-seed via pipeline.analyze()
```

### Next milestones (to scope with Louis)

lock/annotation UX (span_rescore + context prior are copied and load — need
`/api/context_rescore` wired), iReal import, sections/folding.

## 2026-07-31 (later) — Grid arrangement REDONE after Louis's review

Louis checked This Love on :7772: musx chords right, **arrangement onto the
grid wrong**. Root cause: my v1 assigned chords to bars by TIME CONTAINMENT
(which bar's [t0,t1) holds the onset). Redecode boundaries sit on beats only
up to the 23.22 ms musx frame grid, so a chord changing ON a bar line lands a
few ms either side → wrong bar → every bar-opening chord rendered as the tail
of the previous bar + a held "%". The live app already solved exactly this;
v2 is a minimal copy of its method (render_youtube_chart.py::
chart_to_interactive_inputs):

- chord onset → nearest detected-beat INDEX; bar/beat = integer arithmetic
  ((idx − off) // bpb, % bpb) — no time containment anywhere;
- bar phase `off` = modal residue of the tracker's downbeat indices, then the
  **harmonic re-anchor** (live thresholds 55%/15%) lets the chords out-vote
  the tracker when they overwhelmingly agree on a non-zero beat-in-bar;
- bar time spans = real beat times at the boundary indices (playhead map).

Result (browser-verified): This Love `G/B | Cm | Fm7 | Dø7 Fm/Ab` one chord
per bar, 4 held bars (was: chords straddling bars + % everywhere). Let It Be
`C G | Am F | C G | F C` (the real two-per-bar form, 0 held). Stand By Me
N.C. intro + E at bar 12. Re-anchor fired on none of the three — the downbeat
vote alone was right; it stays as the safety net it is in live.

## 2026-07-31 — Two one-beat-early chords (Louis, This Love 1:56 / 2:52): diagnosed + fixed

Not a layout bug — the layout put chords exactly where the decode put them
(all onsets within ±11 ms of a beat, all 81 downbeats on one lattice). At the
two phrase turns the raw musx frames go AMBIGUOUS for ~a beat (conf 0.3–0.6
vs 0.9+ elsewhere; band fill), and the flat change penalty made last-beat vs
next-downbeat equal cost — noise picked the early beat. Fix: pass Beat This!
downbeats into the re-decode → the vendored decoder's downbeat-graded
transition costs (15/45/100) break exactly that tie toward the downbeat.
Verified: Fm 116.61→117.21 (own bar), Cm 172.13→172.78 (joins Fm7);
beat-in-bar distribution goes 73/1/38/7 → 76 on beat 0, 41 on beat 2, zero
stragglers. Old study had graded at −0.17 pp label overlap (a wash) — kept ON
here because chart placement is the product. Let It Be / Stand By Me unchanged.

## 2026-07-31 — Dormant-route sweep + brick schema

Audit of everything that could switch chart-grid behavior. Found and removed:
- `redecode(downbeat_times=None)` default — a future caller would silently get
  the flat-penalty tie back. Now a REQUIRED keyword (TypeError without it).
- `redecode_audio()` convenience wrapper (unrouted second call site) — deleted.
- `key_profiles.activations_to_chroma` / `detect_modulations` — tied to the
  cut Basic Pitch front-end — deleted.
- Stale "downbeat grading not recommended" note in musx.py replaced with the
  current placement rationale.
Verified: all 3 charts byte-identical before/after the sweep. Env flags in the
chart path: ZERO (only HARMONIA_MUSX_DIR, clone dir resolution). span_rescore/
chord_context_prior/nnls_features stay as unrouted milestone-2 bricks — no
server route reaches them.

Brick schema: docs/harmonia_min_schema.png.

## 2026-07-31 — Harmonic key analysis added + Louis's two bar rules

`harmonia_min/harmonic_key.py` (~330 L): minimal port of the v7c colour pipe
(scratchpad/colour_hmm_song.py::decode_segments, feat/harmonic-key). Kept: the
five stages (chord chroma → tonic track → mode audit → colours → feedback) and
every v4.1 calibration constant. Cut: structure fold (needs the cut section
machinery; the script treats no-fold as graceful fallback), the global MODE
mutation (now a parameter), the third duplicate quality table (labels.py is
the vocabulary). Wired into pipeline.analyze best-effort: global key = longest
tonic segment; keySegments in the payload; colour/inflect/flag/sug per chord.
Port verified faithful: identical segments/mode/suspects to the reference
script run on the same input.

Bar rules (Louis): (1) a bar lists all chords sounding in it — carried chord
re-written at beat 0 when a new chord arrives mid-bar (carry-marked, excluded
from repetition counts); held bars keep "%". (2) bar cells are a bpb-quarter
grid; each chord sits at its beat quarter. Cap 2→bpb chords/bar; pickups
display at beat 0.

Verified in browser: Let It Be `C·G|Am·F` quarters + carries; This Love
mid-bar G at its real beat 2; Close to You modulation segmented at 98.0s.

FLAGS for the harmonic-key session (not fixed here — their science):
* v7b naming names Close to You seg-2 **Ab**; the v7b commit claims Db and UG
  adjudication (3787cca) says Db. Reproduced on the v7b-era script itself —
  the commit's 3/3 claim does not reproduce. Krumhansl's saturated argmax
  decides between candidates; duration itself favours Db (53s vs 38s).
* colour_chart_song.py::beat_grid reads data/cache/raw_beat_times_v2 — the
  cache serving/audio.py documents as 100% stale librosa beats (v3 superseded).

## 2026-07-31 — v7b.1 naming mirrored from 19d7841

Both flagged bugs fixed upstream (harmonic-key session, 19d7841). Mirrored the
new centre-naming rule in harmonia_min/harmonic_key.py: duration decides when
decisive (>=1.25x runner-up); Krumhansl breaks near-ties in log domain.
Verified: Close to You now C[0-98] → Db[98-224] (global key Db major, matches
UG adjudication); This Love / Let It Be / Stand By Me unchanged. The
raw_beat_times_v2 fix is display-side in their chart script — harmonia_min
never read that cache (its beats come from state/beats, Beat This! only).

## 2026-07-31 — Splitter-fix report integrated as design lessons

The fix itself (c602a93, _split_collapsed_bars_via_musx) has NO code target
here: harmonia_min has no delete-then-reemit stage (musx is read directly).
The three lessons were audited against harmonia_min and two changes made:
1. The ≤bpb chords/bar cap used to shed the SHORTEST chords on overflow —
   received content. Now it sheds only clamped bar-0 pickups (artifacts of
   our own clamping, the only possible overflow) and RAISES if a bar still
   overflows (invariant broken upstream ≠ something to hide).
2. The harmonic-key stage was try/except best-effort — a silent fallback
   (the Occam-gate trap verbatim). Now it fails loudly, same doctrine as
   beats.py. Both changes verified as no-ops on the 4-song corpus today.
3. Slash-bass survival: nothing rewrites chords in harmonia_min today, but
   NOTE FOR THE CHALLENGE-AUTOAPPLY AGENT (Task 1 brief): challenge
   alternatives are (root, q) only — when you replace a written chord,
   decide the /bass fate EXPLICITLY (inversions must survive rewrites).

## 2026-07-31 — Auto-apply "challenge" corrections: REFUTED at the premise check

Task was: sweep the 16 cached songs, then wire `kind == "challenge"` to
overwrite the rendered chord. **Nothing was wired.** The premise screen killed
it, and killed the upstream-prior follow-up with it. No file under
`harmonia_min/` changed this session.

### The sweep

16 songs, 1688 chords: **148 challenges (8.8%)**, 122 suspects, 52 inflections.
Not rare, and not noise — a systematic artefact of the challenge *scorer*.

| observation | number |
|---|---|
| flagged chords that are SEVENTH chords (`7`/`^7`/`-7`/`h7`) | 144/148 (97%) |
| top alternatives with FEWER notes than the written chord | 138/148 (93%) |
| top alternatives with MORE notes | 0/148 (0%) |
| musx backs the WRITTEN chord more than the alternative | 126/148 (85%) |
| alternative's musx posterior < 0.05 (invisible to the decoder) | 80/148 (54%) |

Median musx posterior: written **0.651**, alternative **0.037**.

### Why — verified, not inferred

`_challenges.score(pcs) = mean(chroma over pcs) * (0.7 + 0.3 * scale_fit)`.
Both factors fall monotonically as you add a note whose chroma mass is below
the chord's own mean and which sits outside the scale — and that is *exactly
and only* the condition that got the chord flagged in the first place. So the
scorer is a **seventh-stripper**: it flags a chord for having an out-of-scale
7th, then rewards deleting it. 0% of 148 alternatives add a note. In 19 cases
the "alternative" is the same triad with the 7th removed (`Ab7` → `Ab`), where
written and alternative have *identical* musx support by construction.

### Adjudicated against iReal Pro (trust order, CLAUDE.md rule 3)

5 of the 16 songs have an iReal chart in `docs/plots/.ireal_urls.json` (decoded
with `harmonia.irealb_fetcher`), covering **92 of the 148 challenges**. This is
form-agnostic — no chart↔audio alignment — so it answers "is this chord in the
tune at all", which is enough:

| | pooled, 92 challenges |
|---|---|
| written chord IS in the tune (**false alarm**) | 56 (61%) |
| written root right, quality wrong (**flag is right**) | 26 (28%) |
| written root not in the tune at all | 10 (11%) |
| **proposed alternative is a chord the tune contains** | **4 (4.3%)** |

Auto-applying would have rewritten 148 chords, ~96% of them into chords the
song never plays.

Blue Bossa is exact rather than form-agnostic (16-bar loop, one chord per root,
100 of the 148 challenges): 35/64 false alarms on `Ab7`/`Eb-7`/`Db^7` — the
tune's own bridge ii-V-I into Db — and **0 of the 24 genuine quality errors get
the iReal chord as their top alternative**.

`CHALLENGE_MARGIN = 1.25` is **not** the problem, so retuning it is not the fix:

| margin | flags | false alarms | real errors | correctly repaired |
|---|---|---|---|---|
| 1.25 | 64 | 35 | 24 | **0** |
| 2.00 | 12 | 6 | 5 | **0** |
| 3.50 | 1 | 0 | 1 | **0** |

The ranking is wrong, not the cut. (The threshold was calibrated on This Love +
Close to You, which between them produce 3 of the 148 challenges — those two
songs could not have exposed this.)

### Task 2 — upstream prior: stopped at 2a, then at a sharpened 2b

**2a as briefed: fails.** The alternative the challenge layer names carries a
median 0.037 musx posterior and is usually not a chord in the tune. An
emission-level prior toward it cannot help; it would push the decoder toward
readings the acoustic model correctly rejects.

**Sharpened 2b, then stopped too.** The one real residual is Blue Bossa's
`Db7` where the tune is `Db^7` — and that confusion lives entirely in musx's
*seventh* head (`s7`), not the 73-wide triad plane, since `Db7` and `Db^7`
share a Db-major triad. Measured `s7` over the flagged spans:

| chord | take 1 (maj7 / b7) | take 2 (maj7 / b7) | iReal |
|---|---|---|---|
| `Db7` (should flip) | 0.200 / 0.594 | 0.073 / 0.536 | `Db^7` |
| `Db^7` (decoder got it) | 0.592 / 0.304 | 0.461 / 0.184 | `Db^7` |
| `Ab7` (must NOT flip) | 0.077 / 0.737 | 0.025 / 0.653 | `Ab7` |

In C natural minor `Db7→Db^7` and `Ab7→Ab^7` each gain **exactly one** in-scale
pitch class, so any scale bonus shifts both by the same amount and flips
whichever has the smaller `s7` margin first. The margins overlap: **41 of 47
must-not-flip chords sit below the largest should-flip chord**. The best
possible single constant fixes 28/31 and breaks 12/47 — and that constant is
fitted on the same data it is scored on, from ONE tune. Corpus-wide the
root-right/quality-wrong signal is 24 Blue Bossa + 2 Close to You out of 26:
a one-tune phenomenon (rule #5). Not built.

`span_rescore.py` + `chord_context_prior.py` were read (Task 2c): `pool_span_musx`
+ `acoustic_logp_musx` are a clean per-span (n,60) log-posterior over the same
cached posteriors and would be the right foundation *if* there were something to
build. There isn't, so they stay unrouted.

### What this does NOT settle

* The challenge *detector* has real signal — 28% of flags land on a genuinely
  wrong chord. Only the *corrector* is refuted. A repair that stays on the
  written root and re-decides the seventh is the shape that the evidence
  supports; nothing here validates it.
* "Chord is in the tune" ≠ "chord is right at this instant" (no alignment).
  The 61% false-alarm figure is therefore approximate; the 4.3% figure (the
  alternative is not in the tune at all) needs no alignment and is exact.
* 56 of the 148 challenges are on songs with no iReal chart and were not
  adjudicated at all.
* `suspect`-grade was never in scope and remains advisory.
* Untouched, but noticed and NOT fixed here (they belong to whoever owns that
  code): (a) `pipeline.analyze` runs the harmonic layer on the flat chord list
  *after* carry-copies are inserted, so a carried chord's span is scored twice
  and overlaps its own original — 31 of the 148 challenges are on carry copies;
  (b) `redecode` selected latency 0 ms on 15 of 16 songs, where the original
  study measured +46…+289 ms — `path_loglik` tags the frames past the end of
  the shifted labelling as `N`, which penalises larger latencies structurally.

### Verified green

`docs/harmonic_challenge_earcheck.md` — the timestamps to judge by ear.
Milestone-1 rendering re-checked on :7772 in headless Chrome (system Chrome
channel; the playwright browser cache is empty on this box): This Love 129
chord glyphs, Let It Be 149, clock advances 0:00→0:03 on play, playhead lands
on bar 2, zero JS errors on both.

## 2026-07-31 — Sections milestone 1: label strip (branch feat/minimal-sections)

Scoped with Louis: bandeau only (detection + A/B/C labels, reps=1, NO
folding), minimal in-house detector, acceptance = ear-check on the 4 library
songs. New brick harmonia_min/sections.py (~140 L): chord-tone bar features
(never root-only; sounding bass at half weight; held bars inherit), cosine
SSM, blur-then-refine checkerboard novelty (no fixed section-length prior,
only a 2-bar degenerate guard), average-linkage letters. Wired into
pipeline.py stage 7: the ChartModel now carries the detected sections; form
strip + badges render in the untouched UI.

Screen results (boundaries, for Louis's ear):
  This Love     A[0:01-0:59] A[0:59-1:06] B[1:06-1:34] B[1:34-1:49]
                A[1:49-1:57] B[1:57-2:04] B[2:04-3:23]
  Let It Be     A[0:00-3:58]   ← ONE section, see limit below
  Stand By Me   A[0:01-0:44] B[0:44-2:56]
  Close to You  A[0:00-0:14] B[0:14-1:38] C[1:38-2:29] C[2:29-3:42]
                 ← C opens exactly on the Db modulation bar

MEASURED LIMIT (stated per rule #4): harmony-only novelty cannot cut a song
whose progression never changes — Let It Be loops C-G-Am-F through every
section and yields one segment. A cheap arrangement cue (raw chroma-mass
delta, 50/50 novelty blend) was tried and REFUTED on the spot: it fragmented
3/4 songs and lost Close's modulation cut; removed entirely (no dormant
lane). Cutting constant-harmony forms needs real arrangement features — the
section-detection branch's territory. This Love's bridge (~2:29) is also not
cut (its harmony stays in the same family). Folding, under-fold doctrine and
endings: next milestone.

## 2026-07-31 — Sections v2: NNLS half-bar SSM (Louis's corrections)

v1 (chord-tone bar features) read This Love as noise — Louis: « le SSM a une
structure, mais tu la lis mal ». v2 follows his two directives:
* grain = HALF-BAR, substrate = RAW NNLS bothchroma (texture: voicings, bass,
  harmonic rhythm), blurred checkerboard on that — the blur turns the fast
  alternation into section-scale blocks (clearly visible on the diagnostic
  scratchpad/ssm_nnls_this_love.png);
* two reader bugs found by LOOKING at the numbers, not the labels:
  (1) edge half-bars see a truncated checkerboard kernel; their artifact
  values inflated the mean+z·σ threshold past every real peak (thr 20.8 vs
  real peaks 11–13, zero cuts found). Fixed: mask kernel edges, threshold =
  fraction of strongest interior peak.
  (2) letters by segment-mean chroma cosine merge everything (all C-minor
  material). Fixed: letters from the OFF-diagonal repetition blocks — the
  cross/self block-mean ratio of the blurred SSM. Measured gap on This Love:
  same-type pairs 0.98–1.00, verse-vs-chorus 0.91–0.94 → threshold 0.96.

Result: This Love A[0:01] B[0:41] A[1:01] B[1:31] A[1:49] C[2:02] D[2:17]
B[2:22] — verse/chorus alternation + bridge + breakdown + final chorus; the
A/B split matches the 1-vs-2 chords/bar regimes on the rendered chart.
Close to You: A/B at the 1:38 modulation. Stand By Me: over-cut (9 segments,
letters plausible). Let It Be: 6 segments all-A (its harmony IS the same
everywhere — letters honest, boundaries unverified).

HYPOTHESES (rule #5): KERNEL 8 bars, PEAK_FRAC 0.5, LABEL ratio 0.96 are
calibrated on This Love + eyeballed on 3 others — a 4-song hypothesis, not a
validated setting. Ear-check by Louis pending; folding still out of scope.

## 2026-07-31 — Boundary bar mapping fixed (Louis: B's last bar bled into A)

Root cause, two layers, on This Love's B→A boundary (bar 23.5):
1. Mid-bar novelty peaks (odd half-bar — 4/7 cuts on This Love) were snapped
   with Python round() = BANKER'S rounding: a literal coin flip on .5. Now
   deterministic ceil — the change happens DURING bar b, so the new section's
   first full bar is b+1.
2. The chorus tail "Ab G | %" — a section never OPENS on a held bar; the
   hold belongs to the closing phrase. Cuts advance past held bars (≤2).
Bars are the reference unit for sections (Louis's doctrine): detection stays
at half-bar grain, but every boundary decision is now expressed as "which
bar joins which side" on the chart's own bars. A content-similarity vote was
tried first and REMOVED: a cadence bar (Ab G) pitch-matches the verse
(Dø7 Ab) better than its own chorus — pitch cannot classify transition bars.
This Love now: B[16–24] ends on the held cadence, A[25–35] opens Cm | Fm.
REMAINING ±1 AMBIGUITY (for Louis's ear): A3 opens on the attack-cadence
"Ab G" (bar 43) and final B opens "Bb Eb" (bar 57, one bar into the phrase?)
— resolving these needs letter-aware phrase alignment (each section's opening
snapped to match its group-mates), which is exactly the machinery the folding
milestone needs; deferred there.

## 2026-07-31 — Louis's three bar-level section rules + the "se recoupent" failsafe

Rules landed (sections.py), in final priority order after several measured
regressions in both directions:
1. CELLS (hard): never cut inside a recurring 2-bar cell. Cell = pair of
   attack bars whose signature recurs with occurrences ≤4 bars apart
   (tiling). The gap matters: verse-end→chorus-start pairs ALSO recur but 20
   bars apart — treating those as cells swallowed whole choruses (measured).
2. No section opens on a held bar; cadence-tail openings ("X | %") penalized.
3. EVENNESS (tiebreak only): lengths round to multiples of 2 on the
   EFFECTIVE length (trailing holds don't count). Evenness-first moved
   Close's tonally-exact Db cut by +2 bars (measured) — the peak distance
   now dominates, parity only arbitrates equally-near candidates.
4. FAILSAFE (Louis: same-letter sections must "se recouper"): after
   lettering, a section whose opening strictly equals no sibling's opening
   (2-bar signature prefix) may shift ±3 to a position that does. Fraction-
   based agreement scoring shifted sections on noise twice (measured);
   strict prefix equality is the shipped gate. Residual disagreement is
   logged as a warning — the failsafe signal itself.

This Love final: A[16b] B[9b] A[25:C-|F-] B[9b] A[45:C-|F-] C D B[56:C-7 F-7]
— all B's open on the chorus cell, both interior A's open identically ✓.
Close: modulation cut exact (1:38). Let It Be: 3 segments, letters honest.
Stand By Me: still over-cut (9 segments) — flagged, not hidden.

## 2026-07-31 — "%" removed; hold ownership settled (le G finit B)

Louis's two calls: (1) drop the "%" simile entirely for now — every bar
WRITES its sounding chord (carry-marked, 0.72 opacity in the UI); simile
returns later as a pure rendering overlay. Killing the empty-bar state
removed the held-bar special cases from layout AND detection. (2) Hold
ownership: the held G after "Ab G" ENDS the chorus (B), the verse starts on
Cm — A1's opening G/B is the same role as an anacrusis. Written section
lengths may therefore be odd (B = 9 written); the multiples-of-2 rule stays
a tiebreak on effective length.
Migration fallout, both found by measurement and fixed: held bars now sign
as their sounding chord, so (a) "G(held)|Cm" matched the verse's tiling
"G/B|Cm" cell and the cell rule forbade the validated cut — the cell test
is now location-aware (a held bar never binds a cell); (b) two cadence-tail
openings matched EACH OTHER in the failsafe — held bars no longer count as
prefix evidence and a cadence-tail opening is never a shift target.
This Love final: B ends (G), A opens Cm — both pairs identical; Close's
modulation cut exact; zero "%" cells rendered.

## 2026-07-31 — Multiples of 2 made RIGID; A opens on the G

Louis's correction: multiples-of-2 is not a preference, it is THE base rule —
once a cut location is decided, snap to the nearest multiple of 2 bars. And
A opens ON the held G (same role as A1's opening G/B). Implementation: every
cut snaps to an even bar index (section grid anchored at bar 0); ties between
the two nearest even bars broken by the novelty curve; cells stay hard; the
"never open on a held bar" rule and the cadence-tail penalty are DELETED
(both were artifacts of the "%" era and contradicted the validated
structure); failsafe shifts are ±2 only (parity-preserving).
Result: ALL sections even on all 4 songs. This Love: A16 B8 A12 B8 A4 C6 D2
B24 — all three A's open on G, B's open on the chorus cell and end on Ab G.
Close: 36/46, modulation exact. Simpler rule set than yesterday's: three
rules and a failsafe, no soft penalties left.
