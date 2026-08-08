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

## 2026-08-02 — Audit du sélecteur de latence musx : bug confirmé, mais pas coupable

L'audit (`docs/harmonia_min_audit_questions.md`) accusait `path_loglik` de
biaiser systématiquement vers 0 ms : les frames laissées hors des segments
décalés garderaient le tag 0 = "premier accord du vocabulaire" (`C:min/b7`),
payant une pénalité artificielle. Vérifié en lisant `xhmm_ismir.py` : le tag 0
n'est pas `C:min/b7`, c'est le "N" (no-chord) que le décodeur préfixe toujours
au vocabulaire, indépendamment du fichier chargé — l'audit s'est trompé de
tag. Le mécanisme lui-même (un `t0` négatif clampé à 0 dans `_tags_to_lab`,
puis `path_loglik` qui recalcule un index de frame trop tardif en ré-ajoutant
la latence) est réel et je l'ai reproduit précisément.

Mesuré sur les 5 chansons du corpus (posteriors + grille de battements en
cache) : cette fuite ne représente que 0.2–1.3 % de l'écart total de
log-vraisemblance entre candidats — 100 à 500× trop petit pour expliquer que
les 5/5 chansons choisissent 0 ms. La vraie cause : décaler la grille de
transitions légales force le Viterbi à garder l'ancien accord plus longtemps
à CHAQUE transition (pas juste la première), et avec la grille de battements
beatthis (déjà bien callée), ça dégrade le fit partout, de façon monotone —
cohérent avec la note du 31/07 ci-dessus ("different beat backend feeding the
grid may explain it").

Fix appliqué quand même (`harmonia_min/musx.py::path_loglik` + `redecode`) :
un `guard_frames` commun (13 frames ≈ 302 ms, dérivé de `max(latency_grid)`)
exclu des DEUX bords pour TOUS les candidats, y compris 0 ms — comparaison
enfin équitable. Validé sur un cas jouet que le fix corrige bien un écart de
15 nats artificiels. Mesuré sur les 5 vraies chansons : **latence choisie
identique avant/après (0 ms partout), 0 accord déplacé, 0 label changé** —
le fix ne change RIEN aux 5 charts actuels. This Love et She Will Be Loved
(chansons vedettes demandées) : aucun diff musical, rien n'a bougé.

Pas touché : `harmonia/models/musx_redecode.py` (ancien pipeline) a le même
bug, vérifié identique ligne à ligne — hors scope, non corrigé là-bas. Serveur
:7772 non redémarré (le fix est dans le code, pas encore chargé par un
process déjà tournant). Rapport complet : `docs/musx_latency_ab.md`.

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

## 2026-07-31 — REPLI phase 1: observation stacking + second musx pass

Louis's spec implemented in harmonia_min/folding.py (~230 L): per letter
group, detect the internal loop period (bar-feature autocorrelation, smallest
P in {2,4,8} scoring >=0.80 — This Love A=4, B=2 exactly as predicted; Close's
through-composed sections score 0.65-0.77 and stay unfolded), stack the bars
at each loop position across every occurrence, AVERAGE their musx frame
posteriors (noise ~1/√n), re-decode the averaged template ONCE (tiled ×3 to
kill Viterbi edge effects), and write the template chords back on every
contributing bar. The transition exception is data-driven: any member <0.85
cos from its stack centroid keeps its first-pass decode (This Love's
"Cm F7 | Ab G" cells land at 0.74 → variants, 5+1 of them).

THE MERGE-SAFETY STUDY (Louis: "il y aura un seuil ou une étude"):
member gate alone was NOT enough — a bimodal stack (Let It Be's A = a
verse+chorus composite under one letter) centres its centroid between modes,
everyone passes, and the fold rewrote real content ("Am F" -> "F C",
measured). Mean-to-centroid coherence was tautological after the member gate
(measured). The separating statistic is the per-position MEDIAN PAIRWISE cos:
This Love 0.88-0.94 vs composites 0.74-0.81. Gate: every position >=0.85 or
the letter does not fold. Result: This Love folds (A obs [7,8,8,8] — Louis's
"7 observations" on the nose — B obs [20,15]; 23 bars cleaned: chorus cells
unified to Cm Fm | Bb Eb, Fm7/Fm and Dø7 unified, stray Bb7/Ab-7/Bo removed);
Let It Be, Stand By Me, Close honestly refuse with measured reasons.

model["fold"] carries the full report (periods, n_obs, variants, changed
bars) — the data feed for the sub-section validation UI ("interface
ludique") where Louis will confirm/reject proposed sub-splits of composite
letters; not built yet. Display folding (write-once ×N) also still to come.

## 2026-08-01 — Singleton fusion + display fold + rendering audit

1. Adjacent NEVER-repeated letters fuse (This Love C6+D2 → one 8-bar bridge).
   Guard measured on Close to You: raw singleton fusion swallowed its two
   ≥8-bar key-halves (the 1:38 modulation vanished into one 82-bar blob) —
   only fragments (<8 bars) glue. Two-case calibration = hypothesis.
2. Display fold (folding.display_fold): same-letter, SAME-length sections
   agreeing on all but the last ≤2 bars are written once ×N — reps/spans/
   barRanges/barSpans multi-pass + endings variants, the exact ChartModel
   contract app_shell already renders. Under-fold doctrine: This Love B8+B8
   fold ×2, the 24-bar final B stays written out; A 16/12/4 stay split.
3. RENDERING AUDIT (verified): emit-order contract asserted on every section
   (bars vs barSpans rows, incl. the endings layout); multi-pass playhead map
   checked computationally (t=45s → B row 1 slot 0, t=95s → same row slot 1);
   JSON all plain floats; zero page/console errors; badge ×2 + form strip
   A B A B A C B correct. KNOWN RISKS (documented, not hit): client-side
   chord spans in folded sections are rigid translations (tap-to-seek in a
   later pass can drift vs the real tempo; the playhead itself runs on the
   server's real-time barSpans); annotate-mode (bar,beat) keys address the
   representative pass only; the endings path (tail>0) is code-complete but
   UNEXERCISED by the current 4 songs — needs a real 1st/2nd-endings song
   before trusting it visually.

## 2026-08-01 — Cross-pass stacking on display-folded sections (Louis's (b) fix)

When same-letter sections display-fold ×N, their musx frame posteriors are
now stacked position-by-position and re-decoded (same _template_chords core
as the loop repli, factored out) — the folded block shows a CONSENSUS of all
passes, not pass 0's chords, which dissolves the "representative pass"
problem. Skipped when the letter already loop-folded in phase 1 (its pool is
broader: This Love's B cells pooled across B1+B2+the 24-bar final). The
endings tail keeps per-pass decodes. Verified: Stand By Me B×3 stacked (own
musx pass, latency 80ms) → consensus CONFIRMED the first decode (0 bars
rewritten, three passes agreed: D | E | A — the IV-V-I turnaround); This
Love byte-identical (skip path); refactor regression-checked.
NOTE for Louis's mental model: musx eats its own CQT frame posteriors — the
NNLS chroma feeds the harmonic-key layer — so the stack averages those
posteriors; same principle as "empiler les chromas".

## 2026-08-01 — Les règles de référence dictées + audit + métrique variance

docs/harmonia_min_rules.md créé : la recette de base d'harmonia_min (source
= audio ; granularité = LA BARRE ; ordre: BPM/grille Beat This! → accords
musx snappés → propagation → répétitions SSM → sections → test de merge par
variance normalisée du chroma → squash + empilement + 2e passe musx). Audit
rule-by-rule dedans ; deux écarts identifiés : (1) le re-décodage autorise
les changements au quart de barre (la règle dit demi-barre) — de fait 100%
des accords de This Love tombent sur barre/demi-barre, mais pas structurel;
(2) le test de merge actuel est en cosinus, pas la variance/moyenne dictée.
Graphiques produits pour que Louis fixe le seuil de la métrique variance:
scratchpad/fold_variance_study.png (bonnes piles This Love 0.05-0.13 hors
variantes; piles mélangées 0.08-0.34; la métrique DÉNONCE d'elle-même la
contamination des variantes du B). Démo _template_chords:
scratchpad/template_chords_demo_this_love.png (pos 4 du A: 6 variantes de
1ère passe → consensus Dø7 propre).

## 2026-08-01 — Variantes par écart individuel vs collectif (règle raffinée)

Louis's refined rule replaces both the absolute cosine member gate and the
VAR_MAX position skip: each stack member's deviation from the centroid is
compared to the stack's own norm (median+MAD); robust z > 3 = variant, keeps
its first-pass decode. Measured separations: true variants z=5.7–38 vs
normals ≤1.7. This Love now auto-detects the pre-bridge D° bar (47) and the
interior Ab G cadence cells of the final B (62/63/70/71) as variants —
sections, letters and the A template unchanged. She Will Be Loved honestly
refuses to fold (pairwise coherence 0.59) until the alignment bug under
investigation is fixed — with the absolute gate gone, the phase disorder now
shows up in the coherence check instead of being silently excluded bar by
bar. Also cleaned a docstring corruption in folding.py (the morning's
constants edit had matched inside the module docstring). Prod restarted.

## 2026-08-01 — She Will Be Loved alignment bug: ROOT-CAUSED (sections open on duplicated bars)

Louis's report: at the A→B transition the last bar of A is duplicated into
B's first bar, everything after shifts. Investigated end to end; every claim
below is measured (diagnostic: scratchpad/swbl_boundary_diagnostic.png).

REFERENCE STRUCTURE (established BEFORE reading our output; source: two
agreeing UG tabs — 4.84★/567 votes capo-1 and 4.56★ "1 step up", both map to
concert C minor): verse vamp |Cm|Bb| ×8, chorus |Eb|Bb|Cm|Ab| ×3, bridge on
the verse vamp ending Ab. One chord per bar everywhere.

THE AUDIO FACT nobody modelled: chorus 1 is 12 bars + ONE extra bar of held
Ab (bar 32, 87.24–89.58s — musx decodes a full-bar Ab:maj7 there; "…she will
be loved" rings on). Verse 2 therefore starts at bar 33 — and every true
boundary after it is ODD (33/49/65/75/79). The bar-b-vs-b+2 similarity jumps
0.35–0.63 → 0.90+ exactly at bar 33 (the 2-bar-phase flip the parity scan
predicted). Beat grid itself is CLEAN (107 uniform bars, 2.34–2.38s, no
insertion) — the tracker is innocent.

THREE-LAYER DIVERGENCE (first-pass decode matches UG almost perfectly —
Eb Bb C-7 Bb|Eb Bb C-7 Ab|Eb Bb C-7 Ab — all corruption is downstream):

1. FOLD (already neutralised by a3fbe68 today): the old chart's fold pooled
   the whole all-'A' song at P=4 phased from composite-section starts; stack
   position 1 = bars {21,25,29} (first-pass Bb, chorus 1 = phase 1 of S0) +
   {49,53,57,61,79,83,87,91,95,99} (first-pass Eb, choruses 2/3 = phase 1 of
   S1/S2). 10 Eb vs 3 Bb → template Eb OVERWROTE the three correct Bb bars
   (fold report "changed: [21,25,29]") — the "Eb Eb" doubling Louis saw in
   chorus 1. Current build refuses the fold (coherence 0.59) so the symptom
   is gone, but the lesson stands: stacking phase MUST come from the real
   cell tiling, not from section starts (feeds the finest-confident-cell
   rework).

2. CELL RULE SATURATION (the actual section bug): in a two-chord vamp song
   EVERY adjacent bar pair is a "recurring pair ≤4 apart" — Cm|Bb and Bb|Cm
   tile at gap 2, the chorus 4-bar loop puts Eb|Bb, Bb|Cm, Cm|Ab, Ab|Eb at
   gap 4, and the verse→chorus pair Bb|Eb coincides with the chorus's own
   wrap pair. Traced cut loop: candidates 20.0/26.5/30.5/34.0/49.0/55.5/
   65.0/73.0/76.5/78.5 → ALL true boundaries rejected "SPLITS CELL"; the
   only two legal positions in the whole song were the duplicate seams
   themselves — bar 32 (Ab|Ab^7: pair occurs once → not "recurring") and
   bar 74 (Ab|Ab-carry: held bars can't bind cells). The cuts landed there
   BECAUSE those bars are duplicated. Sections then open on a copy of the
   previous bar and all content sits +1 bar into the section — exactly
   Louis's symptom.

3. EVEN-SNAP GLOBAL ANCHOR (co-conspirator): cuts snap to even ABSOLUTE bar
   indices (anchored bar 0). The real odd seam bar makes every later true
   boundary odd, so even with a fixed cell rule the snap re-forbids
   33/49/65/75/79. The pre-2026-07-31 doctrine ("multiples of 2 on
   EFFECTIVE length, trailing holds don't count — le G finit B") handled
   this: chorus1+Ab-hold = 13 written / 12 effective. The rigid absolute-
   index version erased that and This Love (all-even boundaries) couldn't
   show the difference. Also: the failsafe validated the wrong cuts — both
   sections open on the SAME kind of Ab seam, and consistently-wrong passes
   a consistency check.

NOT FIXED HERE (deliberate — rules #4/#6 + active concurrent rework of the
cell/stacking machinery): options, in preference order —
  A. Rebuild cut placement bottom-up from periodic-tiling RUNS (the
     rules-doc order: repetitions → sections): maximal sig-runs at P=2/P=4,
     P2-coverage wins overlaps (kills the verse-tail↔chorus-head
     coincidence that breaks pair-identity logic at bar 49), run edges =
     candidate cuts, novelty selects among edges, orphan seam bars attach
     to the closing section as tails. Paper-verified on SWBL (yields
     20/33/49/65/75/79) and consistent with This Love's validated cuts.
     Bonus: run starts give the fold its cell PHASE for free.
  B. Patch set on current rules (phase-aware cell test + parity re-anchored
     to previous cut on effective length + fix the single-option quirk when
     base is exactly even). Documented dead end: the P4 run back-extension
     coincidence still forbids bar 49 unless P2-coverage-wins is added — at
     which point B has become A.
  C. Tiny honest-failure guard: when all candidate cuts die to the cell
     rule or a cut opens on a bar signing identically to the closing bar,
     flag "sections suspect" in chart meta. No behaviour change.
Out of scope, untouched: Bb7-vs-Bbmaj7 quality (harmonic prior, later);
bar 23 Bb-vs-Ab in chorus cell 1 (musx call, plausible per record).

## 2026-08-01 — Demi-barre structurelle + réponses Q3/Q4 + étude normalisations

Louis's rulings applied: (1) « les demi-barres = premier niveau de fiabilité »
— chord transitions are now STRUCTURALLY restricted to bar/half-bar in the
musx decode (quarter beats zeroed in make_beat_arr, not just expensive);
the quarter-bar refinement is a FUTURE pass, to run once the half-bar level
is validated. Effect: Let It Be's 3-chords-in-a-bar (G F C) becomes G C —
the F waits for the quarter pass. (2) Q4: N.C. colliding with a chord on the
same (bar,beat) slot is dropped (merge en jetant le N.C.). (3) Q3 study
(scratchpad/variance_normalizations_study.png): var/mean SCALES WITH VOLUME
(×2 gain pushed good stacks into the bad range, separation 0.45×);
var/mean² and std/mean are volume-invariant; std/mean (CV) separates best
(0.94×). Recommendation: CV for any future variance gate.
Residual oddity noted: Stand By Me shows 2-3 onsets at beats 2/3 — pickup
clamps at the song edges, small, to revisit with the run-based cut placement
(option A, pending Louis's go).

## 2026-08-01 — CV gate calibré (<5% faux merges) + décisions Q2/Q4/Q5

CV (std/mean, sans dimension — validé par Louis) branché comme vérificateur
de squash par demi-barre : CV_MAX=0.51 = P5 de 400 piles volontairement
mélangées sur les 5 chansons → taux de faux merges 5.0% par construction.
Corpus-wide 52% des piles même-accord passent (les répétitions bruitées sont
refusées — conservateur voulu) ; TOUS les plis actuellement validés passent
(This Love CV 0.31-0.48, aucune position refusée en live). Rulings: Q2 non
(squash sections reste sur égalité des labels), Q4 N.C. jeté en collision
(implémenté), Q5 pas de plafond de variantes — consensus à la barre, la
barre qui diffère saute seule (= la règle OUTLIER_Z en place). En attente:
option A (coupes par runs), A/B du biais de latence.

## 2026-08-01 — OPTION A livrée : coupes par runs de tuilage

Louis's go + his framing ("les règles n'étaient pas en conflit — on
commençait mal la section"). sections.py: tiling_runs() — a bar belongs to a
P-run when its chroma matches ±P bars (P2 priority); cuts = run edges;
sections start ON their cell's first bar, so multiples-of-2 holds locally
(multiples of the CELL from the section start, not parity from bar 0).
Novelty cuts stay active inside no-run zones; below 50% run coverage the
whole song falls back to the novelty path (Close, Let It Be, Stand By Me).
Orphan fragments shorter than the closing run's period attach LEFT (the held
Ab). New chart-native rule: a section never opens on (attack, held) when +2
bars gives (attack, attack) — the cadence tail rolls into the closing
section (fixed This Love A2/A3 and She Will Be Loved's final B in one rule).

RESULT — She Will Be Loved: cuts 20/33/49/65/75 (the ODD boundaries the old
parity forbade), verse folds at P2 with 16-18 observations per cell bar (the
fine stacking Louis asked for), bridge and outro separated. This Love: keeps
its validated form (A 16b | B×2 | A opens Cm | A opens Cm | B final ×3 —
the 24-bar final B now folds ×3, better than before).

REMAINS (rule #4): This Love's bridge is no longer a separate letter (the
43-55 no-run zone kept no novelty cut at 48 — absorbed into the 3rd A), and
consequently its A letter-group refuses to fold (bridge pollutes the
stacks). To fix next: in-gap novelty thresholding relative to the gap, not
the song. Stand By Me slightly fragmented under the fallback. SWBL's final
B opens on the outro vamp (Cm|Bb7) — plausible against the record, Louis's
ear to judge.

## 2026-08-01 — Chaîne causale du mauvais départ de B (audit) + fix du code mort

Causal audit (independent agent, replayed the OLD code on today's inputs —
reproduces the old wrong cuts {32,74} exactly). Chain: TRIGGER = a real
musical fact (chorus 1 = 12 bars + 1 held Ab bar → all later true boundaries
ODD; beat grid innocent; the SSM SAW the truth — novelty peaks at 20/33/49/
65/76+); AMPLIFIER 1 = the cell veto saturates on a 2-chord vamp (95/106
junctions forbidden, incl. the true chorus start); AMPLIFIER 2 = rigid
parity implemented GLOBALLY (even indices from bar 0) where the doctrine was
local; FAILED VALIDATOR = the failsafe passed two identically-wrong seam
openings (consistent error passes a consistency test), and its ±2 shifts
could never reach an odd truth. Every rule was calibrated on This Love —
all-even boundaries, non-saturating cells (rule #5 made flesh).
Option A verified STRUCTURAL against this class (synthetic inserted-bar
test: run edges follow content). Fixed now: the in-gap novelty DEAD CODE
(prev=cuts[-1] compared candidates to the LAST run edge; now nearest cut
below) — This Love's bridge C[48-55] restored, other songs byte-identical.
MEASURED REMAINING RISKS: TILE_MIN=0.80 knife-edge (Let It Be's loop tiles
at 0.775 → treated as through-composed; This Love needs ≥0.78, SWBL's tag
needs ≤0.76 — no single threshold fits, adaptive needed); the 50% coverage
cliff (Stand By Me at 48%); THE OLD BUG CLASS STILL LIVES IN THE FALLBACK
PATH (global parity + cell veto — an odd-seam song under 50% coverage would
reproduce SWBL's bug); SWBL cut 55 mid-chorus-2 (variant bar breaks tiling);
SWBL B@75 CONFIRMED CORRECT by chroma (4-bar vamp tag before final chorus —
missing internal cut at 79 only). SBM C@66 opens on a carry (rule gap:
cadence-tail catches (attack,held), not (held,attack)) — ambiguous vs the
This Love "A opens on held G" ruling, left flagged.

## 2026-08-01 — Rapports pipeline interactifs (This Love + She Will Be Loved)

scratchpad/build_pipeline_report.py génère un rapport HTML par chanson
(servi via la nouvelle route /reports/<stem>.html du serveur :7772, audio
écoutable dedans) : les 7 étapes — audio, BPM/barres (Beat This!), accords
1ʳᵉ passe musx (latence, demi-barres), répétitions (courbes de tuilage +
SSM + runs), sections (matrice des lettres, coupes, forme), empilement +
2ᵉ passe (obs/position, variantes, avant/après), chart final — chacune avec
ses métriques, ses figures et des boutons ▶ (frontières ±2 barres,
occurrences empilées d'une même cellule, barres corrigées par le consensus).
Regénérable après tout changement de pipeline en relançant le script.

## 2026-08-01 — « B finit au 23 » : la tenue frontière rejoint la section qu'elle OUVRE

Louis: This Love's B ends at 23 — the held-G bar 24 opens A (like A1's G/B).
Root mechanism of the miss: bar 24's +P4 twin (bar 28) is an N.C. gap, so
24 never joined the verse run and the orphan rule glued it left. New
deterministic bar-level rule (post-letters): a HELD bar ending section X
whose sounding sig equals the OPENING sig of any same-letter sibling of the
NEXT section moves to open the next section. This Love: held G (G) = A1's
G/B opening → opens A2/A3; counter-case verified: She Will Be Loved's held
Ab matches no verse opening → stays with its chorus. Side effect: all five
8-bar B blocks now display-fold as B×5 (tails ≤2 as endings). Reports
regenerated.

## 2026-08-02 — Vue minimaliste (proposition 1 + twists de Louis)

harmonia_min/minimal_view.py + route /min/<file> : la représentation la plus
compacte — règle d'or « aucune info écrite deux fois ». Un bloc par lettre :
la cellule du repli (complétée à ≥4 barres) ; si les queues des passes
divergent : passes de longueur égale → le bloc couvre toute la passe jusqu'à
la queue (B de This Love = 8 barres, 3 cellules + cadence) ; passes
inégales → cellule + cellule de queue divergente (A de This Love = 8 :
G/B|Cm|Fm|Dø7 + …|D°, « 8 suffisent largement »). Timeline chronologique en
chips, comptée en unités de cellule (A×4 B A×3 B A C B B B — l'exemple
dicté au mot près) ; taper un chip saute l'audio au passage. This Love : 24
barres écrites au lieu de 80. Trois itérations mesurées sur la définition de
« queue qui varie » (passe entière → fins alignées → cellule+queue) toutes
guidées par les retours visuels de Louis.

## 2026-08-02 — Le repliement minimal DANS l'UI d'origine (correction Louis)

Louis: la logique de repli est bonne, mais l'ancienne UI reste LA
représentation — /min ne devait pas remplacer le chart. minimal_fold()
(folding.py) produit maintenant le ChartModel directement : UNE section par
lettre (bloc validé : cellule ≥4 barres / passe entière si queues égales en
longueur / cellule+queue divergente sinon), reps = toutes les passes,
barSpans proportionnels par passe (le contrat multi-passes que l'app_shell
rendait déjà — passes de longueurs différentes incluses). display_fold
remplacé dans le pipeline ; la page /min reste comme vue d'appoint.
This Love dans l'app : strip A B A B A C B×3, A×3 (8 barres écrites),
B×5 (8), C (8) — 24 barres écrites au lieu de 80, typographie d'origine,
playhead vérifié sur deux passes (contrat rows/slots asserté 5/5 chansons).

## 2026-08-02 — Playhead du pli minimal : carte PAR CONTENU

Louis: décalage du surlignage vs la réalité. Cause: la carte proportionnelle
étirait le bloc de 8 sur la passe de 16 — dès la 2e répétition de cellule le
highlight était sur les mauvaises lignes. Fix: barSpans par CONTENU — une
ligne de cellule reçoit UNE fenêtre temporelle par répétition réelle de la
cellule dans chaque passe (le tspans du client accepte n fenêtres par
ligne); les lignes de queue ne s'allument que sur la passe dont la fin
matche la queue divergente. Vérifié: ligne 0 allumée aux barres réelles
0/4/8/12, cellule cyclée en lecture (bar 1→2→3→4→1→2...), queue D° → ligne
7 seule. Prod relancée.

## 2026-08-02 — Dernières retouches cosmétiques (Louis)

(1) Le bloc d'une lettre = SA CELLULE, l'échelle de répétition minimale
(This Love A = 4 barres — le « cellule + queue divergente » de la veille
retiré sur son retour) ; les passes entières restent pour les queues à
longueur égale (B = 8). (2) Les ×N n'apparaissent QUE dans la strip FORM
résumée, plus sur le chart. (3) Les badges de section vivent AU-DESSUS des
rangées (gap secGap entre sections) — les accords ne sont plus décalés par
le badge. Playhead re-vérifié en lecture (cycle 1→2→3→4→1×2). This Love :
20 barres écrites au lieu de 80, dans l'UI lead-sheet d'origine.

## 2026-08-02 — PWA + règle d'or « barres uniformes »

(1) Harmonia_min s'installe comme une vraie app (manifest + apple-touch-icon
+ meta standalone réutilisés de docs/pwa, route /pwa/, zoom verrouillé) —
ré-ajouter à l'écran d'accueil pour activer le mode standalone. (2) Règle
d'or de Louis « toutes les barres de taille uniforme » : deux causes
corrigées — les rangées incomplètes de fin de section sont complétées par
des cellules vides bordées + bordure basse de fermeture (le gap
inter-sections détachait la border-top suivante), et la grille INTERNE des
quarts passait par repeat(1fr) dont le plancher min-content laissait une
barre à 2 accords élargir sa colonne → minmax(0,1fr). Preuve : toutes les
cellules mesurent exactement 98 px sur This Love et She Will Be Loved.

## 2026-08-02 — l'audio muet sur iPhone : root cause trouvée (app installée)

Symptôme (iPhone de Louis, iOS 18.3.2, app installée sur l'écran
d'accueil) : play accepté, durée connue, mais `buffered` vide pour
toujours ; côté serveur une tempête de 206 sains que WebKit jette.

Triage en 3 étapes :
1. Serveur mis octet-pour-octet identique à l'ancienne app (:7771) —
   Content-Type audio/mp4 (Python devinait audio/mp4a-latm) + ACAO sur
   chaque 206. N'a PAS suffi.
2. Découverte : le « fix » de juillet (564026b) n'avait été vérifié que
   sur un iPhone ÉMULÉ ; le log de l'ancienne app ne montre AUCUNE
   lecture audio réelle depuis le téléphone. Ce bug n'a jamais été
   résolu sur l'appareil — pas une régression harmonia_min.
3. Page /audiotest sur l'appareil réel : dans un ONGLET Safari tout
   joue (fetch 512 Ko OK, <audio controls> natif tamponne 205 s en 2 s,
   notre motif new Audio()+load()+play() atteint rs=4, ct avance).
   Seul le mode INSTALLÉ bloque → défaut du chargeur média de WebKit
   standalone, ni serveur, ni Tailscale, ni notre JS.

Contournement livré (e899352) : en mode installé, fetch() télécharge le
fichier (fetch marche très bien en standalone) et le lecteur reçoit un
blob: local — son chargeur réseau cassé n'est plus jamais sollicité.
Position/lecture préservées au swap. Reste à confirmer à l'oreille sur
l'appareil. Bonus : /audio logge désormais chaque Range demandé/servi.

## 2026-08-02 (après-midi) — New Chord UX livré dans app_shell (handoff du 21/07)

Le "handoff 3" analysé le 29/07 était un doublon du 13/07, déjà en prod.
Le VRAI drop le plus récent est `Harmonia.zip` → `design_handoff_new_chord_ux/`
(21/07, 6 features + notation compacte) — zéro trace intégrée jusqu'à
aujourd'hui. Implémenté ce jour dans `harmonia_min/app_shell.html`, algorithmes
réconciliés avec `docs/pedagogical_mode_design_2026_07_21.md` (§8 voicings
jazz réels, §10.1 le style choisi est honoré à tout niveau, §10.2 cartes
multiples dédupliquées par pitch-set, §11 invariance de basse, exception
rootless documentée). Vérifié par screenshots CDP 390×844 sur :7772.

- **Fiche accord** (tap en Read/Analyse) : piano C4=60, basse accentuée,
  cartes Close/Shell/Rootless-A/Drop-2 (Levine ; 13-pour-5 sur dominantes),
  arpège au tap. Une triade → 1 carte, Ebmaj7 → 4 (dédup).
- **Learn L1/L2/L3** : pill Advanced⇄Learn + échelle ; reduceQ pur (L1
  triades, L2 les trois 7e courantes, L3 tout) ; slash bass = L3 seulement ;
  tag ambre "simpler" si label OU basse simplifiés ; chip flottant en plein
  écran (tap = cycle) ; captions par niveau.
- **Voicing coach** (🎹 dans le transport) : clavier live au-dessus de la
  barre, repeint PAR CHANGEMENT D'ACCORD (hook timeupdate), sélecteur de
  style, fallback close gracieux.
- **Notation compacte + thèmes** (bouton Aa) : glyphe compact (altération +
  qualité empilées, △/−/ø/°, slot d'altération réservé), thème sombre
  complet (palette DARK du handoff, mutation de T + rebuild du chrome,
  l'audio n'est PAS interrompu), couleurs de papier, pref "Key colours".
- **Lentilles de tonalité** : Analyse = Function | Local keys | Global key.
  Bandes PLEINE CELLULE (le look approuvé du handoff — les strips 4px du
  21/07 sont remplacés), jointives entre cellules, étiquettes aux débuts de
  run, thème-aware. Vérifié musicalement sur Don't Know Why : Bb7 se colore
  en Eb majeur (V de Eb), D7 en sol mineur.
- **Déjà présents, rien à porter** : mode immersif (plein écran + poignée +
  swipe, 21/07) ; voltas (folding.py:445 émet `endings`, buildIReal rend les
  crochets 1./2. — aucun chart actuel n'a de tails divergents, la capacité
  attend ses données).

- **Voice-leading lissé** (spec §9), 5e style « Smooth » du coach : chaque
  accord est voicé pour minimiser le mouvement de main depuis le voicing
  réellement choisi pour le précédent (matching biparti de coût minimal,
  notes communes tenues à coût 0, VOICE_PEN=7 par voix non appariée). Les
  candidats sont styles × registres × re-registration d'UNE voix supérieure.
  **§11 prime sur §9** : aucun candidat ne change la basse sonnante (donc pas
  d'inversions ; drop-2 reçoit la basse sous son étalement ; rootless reste
  l'exception documentée). Les deux totaux sont affichés — Don't Know Why
  34 contre 148 demi-tons, This Love 251 contre 336 (les grilles à triades et
  basses slash épinglent la basse, elles gagnent donc moins : attendu).

Deux bugs trouvés en TESTANT L3 contre L1 au lieu de l'affirmer (règle 1) :
`vlTotals()` lisait son cache avant que `vlTrack()` puisse l'invalider — un
changement de niveau renvoyait les chiffres du niveau précédent, identiques ;
et le coach vit dans le transport, que `go()` ne reconstruit jamais — il
gardait le voicing de l'ancien niveau jusqu'au prochain mouvement de tête de
lecture. Corrigés tous les deux ; le coach suit désormais aussi le scrub à
l'arrêt.

- **Extensions diatoniques vs altérées** (spec §10.3) dans la fiche accord,
  au L3 et seulement pour les accords à septième : 9/11/13 naturelles lues
  contre la gamme de la tonalité — dans la gamme = couleur sûre (vert), hors
  gamme = on colle au degré voisin et C'EST l'altération que la tonalité veut
  (♭9/♯9/♯11/♭13, ambre), 11 naturelle sur tierce majeure = note à éviter
  (gris), montée en ♯11. Chaque tension est épelée par SA propre altération
  (un ♯11 affiché « D♭ » se contredisait). Vérifié à la main : B♭maj7 en Si♭
  majeur → 9 do diatonique, ♯11 mi, 13 sol diatonique ; G7 en do mineur →
  ♭9 la♭, ♯11 do♯, ♭13 mi♭ — le motif exact de l'exemple travaillé de la spec
  (D7 en Si♭ → ♭9, éviter, ♭13), transposé.

Non-résolu (règle 4) : la cascade est calculée en ordre de
LECTURE (le chart écrit), pas en ordre joué : une reprise repart du voicing
écrit au lieu de continuer la position de main de la passe précédente. Le
coût est le mouvement L1 total avec pénalité de voix — il ne modélise ni
l'empan de la main, ni le doigté, ni la ligne de soprano (mêmes non-solves
que la spec §9). Voltas jamais vues en vrai faute de données. Session serveur concurrente (P1 annotations, P2
context_rescore) : voir docs/handoff_2026-08-02_min_server_gaps.md.

## P1 — persistance des annotations (session serveur, 2026-08-02)

`POST /api/annotations/<file>` existait côté shell mais tombait dans le 404
attrape-tout : chaque accord verrouillé était perdu au rechargement, en
silence (le shell avale l'erreur). Nouveau module `harmonia_min/annotations.py`
(écriture atomique tmp+`os.replace`, schéma 1 comme le sidecar historique) et
réhydratation **côté serveur** dans `/api/chart-model/<file>` — le shell ne
GET jamais les annotations, l'overlay doit donc se faire là.

Piège évité, mesuré : le shell envoie **`bass:-1` en dur** (app_shell.html
`saveAnnotations`). Appliquer la règle de l'ancienne app (« le fix gagne s'il
n'est pas None ») aurait effacé la basse de tout accord slash dès sa
confirmation — This Love mesure 1 est `G/B`. Ici `-1` = « pas d'avis », on
garde la basse du modèle ; un test le verrouille.

Vérifié en vrai (pas seulement curl) : sidecar écrit → rechargement complet de
la page dans Chromium → la case affiche `Dm7/B` au lieu de `G/B`, le `/B`
survit. `merges` est désormais renvoyé dans le modèle (il valait toujours
`[]`, donc chaque sauvegarde les écrasait). `DELETE /api/chart/<file>` supprime
aussi le sidecar. 13 tests red-first (`tests/test_harmonia_min_annotations.py`),
sur de vrais charts.

Non-résolu (règle 4) : `(bar, beat)` n'est pas unique — les sections repliées
rejouent la même identité de mesure (Norah Jones `LA`, Stand By Me `LB`). Un
verrou est appliqué à **toutes** les copies, délibérément (même matériel
replié) ; on ne peut donc pas faire diverger deux passes repliées. Pas de
fusion de deux écrivains concurrents, pas d'historique/undo. Les `merges` sont
stockés et renvoyés tels quels, rien ne les interprète.

**Serveur redémarré** après la modif (pas de reloader).

## P2 — propagation des verrous : `/api/context_rescore/<file>` (2026-08-02)

Prémisse screené AVANT de coder (règle 2), sur les 6 charts sauvegardés :
zéro verrou → zéro changement partout (c'est structurel, `differential_rescore`
rejoue le même treillis deux fois et ne garde que ce qui est imputable au
verrou) ; un verrou volontairement FAUX propage 4 voisins sur This Love et 1
sur Norah Jones. Prémisse vivant → endpoint construit.
`harmonia_min/context_rescore.py` + route (alias `/api/reinfer/<file>`, que le
chemin « fusion » du shell appelle encore : il reçoit désormais une réponse
no-op correcte au lieu d'un 404 avalé). 6 tests red-first.

**Test d'acceptation, résultat honnête.** Deux corrections *justes* et
vérifiées :

| cas | verrou | propagé |
|---|---|---|
| This Love, le `B°` de 127.3s (ton oreille, 2026-07-31 : « c'est un G ») | B° → G | **0** |
| Close to You, le `B` de 14.0s (tab UG 4.83★ : B7) | B → B7 | **0** |
| sonde de vivacité : verrou volontairement FAUX (A♯ → F, 54.1s) | — | **4** |

Le verrou est honoré dans les trois cas (il revient à l'identique). La
propagation nulle sur les deux vraies corrections n'est pas un câblage mort —
la sonde le prouve — c'est que **les voisins de ces deux accords sont déjà
justes et très confiants** (0.93 / 0.90 autour du B° ; B-7 et E-7 à 0.83/0.97
autour du B). Le prior n'a rien à corriger là. Le `B°` lui-même est à c=0.391,
la confiance la plus basse du morceau : l'erreur était isolée, pas contagieuse.

À surveiller : sur la sonde fausse, la propagation fait *glisser* la boucle
(D♯→A♯, G♯→F, G→G♯) — le prior ré-aligne la progression répétée autour du
faux point d'ancrage. C'est cohérent, mais ça montre qu'un verrou erroné peut
corrompre ses voisins ; le `margin_gate` existe pour ça et est à 0 (défaut).

Non-résolu (règle 4) : un span modifié perd sa septième (le treillis décode
QUAL5 maj/min/dom/hdim/dim, donc un `C-7` rescoré revient `C:min` → affiché
`Cm`). Les spans inchangés ne sont jamais dans le diff et gardent leur
qualité, donc ça ne mord qu'où le modèle a bougé — mais ça mord vraiment là.
Les frontières ne bougent jamais (par conception) ; les fusions gardent donc
besoin de l'ancien re-décodage complet. Cache musx froid → repli silencieux
sur les têtes NNLS-24 (backend acoustique plus faible). Prior poolé, pas de
conditionnement par genre.

**Serveur redémarré** après la modif.

## P3 — dégradation propre (2026-08-02)

`/api/library` expose désormais `capabilities` (`["annotations","reinfer"]`).
Une capacité n'est déclarée que si la route existe **et** que sa dépendance
est réellement là : `reinfer` est conditionné à la table du prior entraînée,
parce que sans elle `span_rescore` retombe silencieusement sur un scorer
uniforme qui ne peut jamais changer un argmax — soit un bouton qui a l'air
vivant et ne fait rien. Le shell décide quoi afficher (côté client, pas à
moi) : voir « Client asks » du handoff.

Deux culs-de-sac supprimés, tous deux atteignables depuis la bibliothèque :
`/debug/section-merge-game` renvoyait un 404 Flask nu (page pleine, aucun
retour possible) → page brève avec un bouton « Back to the library » ;
`/api/section-merge-verdict` renvoyait 404 **à chaque chargement de page**
(deux requêtes en échec par visite, mesuré au navigateur) → renvoie
maintenant `{total:0, merge:0, keep:0}`, ce qui est la vérité de ce build.
Vérifié : plus aucune requête en échec au chargement.

**Serveur redémarré.**

## Screen — vocabulaire LM étendu (2026-08-02)

Avant de coder quoi que ce soit (règle 2), mesuré sur le corpus poolé complet,
même ensemble de labels accepté dans les trois bras, seule la granularité des
classes change (`scratchpad/lm_vocab_coverage.py`) :

| bras | classes | trigrammes | clés distinctes | ≥5 obs | ≥10 | ≥20 |
|---|---|---|---|---|---|---|
| QUAL5 (aujourd'hui) | 5 | 330 985 | 6 659 | 97.9% | 95.7% | 92.9% |
| Q8 (+7èmes, sus) | 9 | 341 509 | 12 761 | 95.8% | 91.9% | 86.9% |
| Q16 (+6èmes, 9èmes) | 18 | 342 929 | 16 438 | 94.4% | 89.6% | 83.7% |

Prémisse vivant : passer de 5 à 18 classes ne coûte que 92.9% → 83.7%
d'occurrences dans des clés bien observées, et l'espace de clés ne croît que
×2.5 (pas |Q|³ — la musique réelle n'occupe qu'un coin de l'espace
combinatoire). Backoff nécessaire pour la queue des ~16%, pas pour le gros.

Trouvaille secondaire, et c'est le meilleur argument : le nombre de trigrammes
**augmente** avec le vocabulaire fin (330 985 → 342 929, +3.6%). Les accords
consécutifs identiques sont dédupliqués — donc sous QUAL5 un vrai `C → C7`
était effacé comme « même accord ». Le coarsening ne floutait pas seulement
les étiquettes, il **supprimait ~12 000 changements d'accord** du signal
d'entraînement.

Suite confiée à une session dédiée : `docs/handoff_2026-08-02_extended_chord_lm.md`.

## 2026-08-07 — les candidats de l'éditeur d'annotation sont maintenant ceux de musx

Louis : « on devrait mettre les accords prédits par musx, j'ai l'impression
qu'on met les accords qu'on prédit autrement ». Vérifié : vrai pour la liste
de candidats, faux pour l'accord principal.

**Ce qui s'affichait avant** dans le Compass/Guide (« candidates the model
considered ») :

* sur ~9 % des accords (les `flag`és) : les alternatives de
  `harmonic_key._challenges` — masse de chroma NNLS × bonus diatonique,
  rendue comme un pourcentage alors que ce n'en est pas un. Réfutées au
  premise-check du 2026-07-31 (ci-dessus) : musx soutient l'accord ÉCRIT
  contre l'alternative 126/148 fois (85 %), posterior médian de
  l'alternative 0.037.
* sur les ~91 % restants : le fallback **inventé** de `candList`
  (app_shell) — l'accord courant + root+7 en « 7 » à 30 % + root+2 en
  « -7 » à 22 %. Aucun modèle derrière.

L'accord principal (le hub du Compass), lui, était déjà le décodage musx.

**Le fix** (`fix/annotation-musx-chords`) :

* `span_rescore.musx_suggestions(probs, chords)` : top-3 des 60 cellules
  (racine × QUAL5) du posterior musx poolé sur la fenêtre de CHAQUE accord —
  la brique `pool_span_musx` + `acoustic_logp_musx` que la session du
  2026-07-31 avait désignée comme « the right foundation » sans la router.
  Le `c` affiché est un vrai posterior (sachant « un accord sonne » ;
  masse N exclue). La cellule de l'accord écrit garde sa queue fine
  (`A-7` reste `A-7`, pas `A-`) : taper dessus re-choisit le même accord au
  lieu de lui retirer sa 7e.
* `pipeline.py` étape 8 : le `flag` de `_challenges` reste (28 % de ses
  flags tombent sur une vraie erreur), ses `alts` disparaissent ; `sug`
  vient de musx pour TOUS les accords → le fallback inventé du shell est
  mort (inatteignable).
* `scripts/backfill_musx_sug.py` : 43/43 charts de `state/charts/`
  backfillés depuis le cache `musx_probs` (0 skip, aucun run musx frais).
* Tests : `tests/test_musx_suggestions.py` (4, verts) — ranking attendu sur
  posteriors synthétiques, préservation de la queue écrite, N.C. sautés,
  vocabulaire des queues = celui du shell.

**Vérifié dans l'app rendue** (Playwright 390px, Bein Green, A7 mesure 13) :
Guide = A7 « top pick » 67 %, A 21 %, F7 2 % — le `sug` musx du chart,
plus le triple inventé.

**Non résolu** : le goulot cinq-familles (les alternatives sont QUAL5 — une
alternative ne distingue pas `-7` de `-9`) ; la latence de décodage reste
ignorée au pooling (convention partagée avec `label_confidence` et
`compute_acoustic_logp`) ; l'app legacy (`harmonia/output/`, port 7771)
garde l'ancien comportement.

## 2026-08-08 — Annotate affiche les accords du chart ; Compass réparé (retours de Louis)

Trois retours sur la session d'hier, tous dans `app_shell.html` :

1. **Plus de strip en mode Annotate.** L'échelle de profondeur (`depthOf` :
   c<0.42 → famille seule, c<0.66 → 7e) réécrivait les accords douteux —
   Louis voyait « la version strippée des 7èmes » au lieu du chart.
   `chordDepth` rend maintenant toujours "exact" (grille ET entête de
   l'éditeur) ; le doute reste porté par la teinte `confColor` + le « ? ».
   Bannière reformulée en conséquence.
2. **Compass : l'accord courant ne tourne plus sur le pourtour** (il est
   déjà le hub) — seules les vraies alternatives orbitent. Tailles : rayon
   sur toute la bande [Sz·0.07, prMax] en √proba (aire ∝ proba) au lieu de
   l'ancienne formule additive plafonnée.
3. **Le hub est cliquable** : taper la bulle centrale sélectionne l'accord
   courant (même flux `onPick` qu'une orbe — preview + « Lock A7 » armé).

**Bug préexistant trouvé en vérifiant** (mesures DOM vs endpoints des
lignes SVG) : les keyframes `ap-orb` (`fill-mode both`) terminaient sur un
`transform:scale(1)` nu qui ÉCRASAIT le `translate(-50%,-50%)` du bouton —
chaque orbe rendue décalée de +demi-taille vers le bas-droite, d'où les
orbes collées au hub/à la couronne. C'est ce décalage, pas les tailles, qui
faisait le gros de l'effet « toutes petites ». Fix : le translate vit dans
les keyframes.

Vérifié rendu (Playwright 390px, Bein Green) : grille Annotate = mêmes
accords que Read (Bbmaj7, F7sus4, Gø7…) ; orbes A 21 % / F7 2 % à 80 px du
centre exactement ; clic hub → « Lock A7 », clic orbe → « Lock A ».

### Addendum 2026-08-08 — orbes agrandies (2e retour de Louis)

« Elles prennent très peu de place » : trois causes traitées dans
`buildCompass`. (1) Bande élargie — `centerClear` colle au rayon visuel du
hub (Sz·0.125), la couronne devient la limite extérieure, plancher
`prMin=Sz·0.095`. (2) La cause principale : deux candidats sur des rayons
VOISINS du cercle des quintes (un accord et sa quinte — le cas courant,
F/B♭ ici) déclenchaient la boucle de rétrécissement tant qu'on exigeait la
séparation complète dans un petit anneau. Les orbes peuvent maintenant se
chevaucher (centres à 68 % de la distance de contact) ; la plus petite est
dessinée après, donc tapable au-dessus. Mesuré : orbes 63/59 px contre
43/41 px avant (hub 69 px). Défaut connu accepté : sur une paire voisine,
la petite orbe peut recouvrir une partie du « % » de la grande.

### Addendum 2 — 2026-08-08 : jamais de chevauchement, décalage radial

Louis sur le recouvrement des « % » : « ils ne doivent jamais se chevaucher,
il faut laisser un tout petit interstice ». Le layout de compromis (centres
à 62 % de la distance de contact) est REMPLACÉ par un placement radial :
deux candidats sur des rayons voisins (≤45°) alternent — le plus gros
contre le hub, le suivant poussé vers la couronne (légèrement au-delà, les
noms de notes restent dégagés) ; interstice minimal Sz·0.008 ; une orbe
isolée garde le siège mi-bande à taille pleine ; si une paire décalée ne
passe toujours pas, tout rétrécit ensemble (géométrie, pas choix). Mesuré
(F7 de Bein Green, paire F/B♭ voisine) : 51/48 px SANS chevauchement, les
deux % lisibles — contre 63/59 chevauchés, 43/41 à l'origine.

Correction du commit 34761b7 : son message dit « centres à 68 % » mais le
code committé porte 0.62 — mon édit 0.68 (non committé) a été écrasé par le
refresh UI de l'autre session (8c9e12b) entre l'édit et le commit. Sans
conséquence : les deux valeurs sont mortes, place() les remplace.
