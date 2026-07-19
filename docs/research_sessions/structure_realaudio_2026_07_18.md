# BAR-GRID vs REAL-MUSIC DRIFT (the user's "une bar ne correspond pas toujours à la même unité de temps" report) — CONFIRMED REAL, QUANTIFIED, ROOT-CAUSED as a SYSTEMATIC CONSTANT-TEMPO MISCALIBRATION (librosa's global tempo = LOCAL-MEDIAN spacing, misses the WHOLE-SONG AVERAGE), NOT rubato and NOT missed beats. Uniform grid slips up to ~4 bars from real downbeats over a song (abba −0.7→+16.4 beats). SAFE FIX = recalibrate `period` to the whole-song best-fit slope (STILL uniform); real-per-beat grid NOT recommended. NO PRODUCTION CHANGE (deferred to staged flagged rollout; file has 173 lines concurrent uncommitted work; no bar-precise real-audio GT exists to VERIFY correctness). Full writeup: `docs/known_issues.md` top entry "BAR-GRID vs REAL-MUSIC DRIFT..." — 2026-07-19, budget up to 4h (Opus), ~1.5h used — READ THIS FIRST

One-paragraph verdict: Investigation of the DEEP version of tonight's earlier
grid-phase findings — the beat-grid `bt` itself (`chord_pipeline_v1.py` ~L2923-2927),
shared by every backend path. **Fresh-eyes correction of Part A**: its "offset never
exceeds 0.5 beat" was a nearest-index-lookup artifact (bounded by 0.5·period *by
construction*); the honest beat-NUMBER-matched offset DOES accumulate (abba slips to
+16.4 beats ≈ 4 bars over the song — exactly the user's report). **Decomposition
(`scratchpad/beatgrid_decompose.py`, 6 real songs)**: 67–97% of the offset variance
is LINEAR — a systematic 0.5–2.3% error between librosa's `tempo_bpm` scalar and the
beats' actual whole-song average spacing (drift −1.1 to −7.3 bars by song end);
rubato residual (the only part a real-per-beat grid buys) is <1 bar; missed/spurious
beats 0/6. **Non-circular held-out test (`beatgrid_heldout.py`, fit period on 10–50%,
predict 50–90%)**: whole-song best-fit period beats librosa 4/6 songs, dramatically
on worst-drift ones (abba 3221ms→812ms, commodores 2333→725ms RMSE), tied on 2
low-drift songs. **Two key negatives**: (a) NO grid variant moves the "crammed bar"
symptom (that's decode churn + render, already mitigated); (b) "median-IBI period" is
BYTE-IDENTICAL to stock — librosa's tempo already IS the local-median spacing, so the
current choice is locally-accurate-but-accumulating, not arbitrary. **Recommendation**:
recalibrate the constant period (safe, uniform, low-blast-radius) rather than switch
to real beats (high blast radius, unquantified dropout risk); stage behind a default-
off `beat_period_mode` flag, verify on the 3 songs with playwright before touching the
default, and ideally source bar-precise downbeat GT (or madmom-DBN cross-ref) first
since no real-audio GT currently exists to verify the "more correct" claim end-to-end.
Artifacts: `scratchpad/beatgrid_{decompose,ab_decode,heldout}.py`(+`_results.json`),
`beatgrid_offset_trace.json`. No production/UI/test files touched, no reinfer calls,
no commits.

---

# PART A IMPLEMENTED — SINGLE-QUANTIZED DURATION COUNT IN `_span_to_beats` — CLOSES THE PART A / PART B LOOP: `round((t1−t0)/period)` REPLACES `round(t1)−round(t0)`; REAL-AUDIO WHOLE-BLOCK SPEC 21 → 4 SPANS EXCLUDED, 1 → 7 FULL POOLS, 10/10 GROUPS APPLIED; ±1-BEAT EXCLUSIONS NOW RARE (GENUINE OUTLIERS ONLY) — SHIPPED IN SOURCE + 3 RED-FIRST TESTS + LIVE `/api/reinfer` (2026-07-19, budget up to 2h stated) — READ THIS FIRST

This is the implementation of the fix PART A (entry two below) root-caused and
deferred, done INSIDE the same function PART B (entry immediately below) extended
— it explicitly closes the loop those two entries opened. Full writeup:
`docs/known_issues.md`, entry "Span→beat-count 'drift' FIXED (Part A
IMPLEMENTED...)" (top of file). One-paragraph verdict:

Part A's recommended fix is now live in `_span_to_beats`: `b0` stays the
nearest-beat START anchor (no over/under-run into a neighbour — the subtlety
Part A flagged), but the COUNT becomes `b1 = b0 + round((t1−t0)/period)`
(single-quantized duration, `period = median(diff(bt))`) instead of the
double-quantized `argmin(t1) − argmin(t0)`. `b1` feeds ONLY the count/precondition
(the offset loop uses `b0+off` with its own bounds check), so it is deliberately
NOT clipped — a near-end span keeps its honest count. **Verification, all real:**
(1) 3 red-first tests from the ACTUAL `drift_rootcause_check_results.json` data
(aretha vamp cluster B `[32,33,32,32,32]`→`[32,32,32,32,32]` full pool; cluster A's
real 43-beat block still excluded) — confirmed RED on pre-fix code, GREEN after;
full suite 473→476. (2) Three-state re-run of Part B's OWN
`group_pool_graceful_verify.py` on all 3 songs
(`scratchpad/partA_three_state_comparison.json`): OLD all-or-nothing 1/10 applied
→ Part B alone 9/10 (21 excluded) → **this fix 10/10 (4 excluded, 7 full, 0
unpoolable)** — exclusions became RARE, exactly as both prior agents predicted;
the 3 residual partials are the genuine length-outliers (aretha 43-beat, autumn
20-beat) Part B is meant to catch. (3) Live server :7771 restarted with the fix,
EVERY route re-verified (`/`, `/library`, `/classic`, all 9 `/debug/*` → 200;
`/chart/<f>` → 302 → 200; candidates API → 200); real HTTP `/api/reinfer` for
aretha cluster B now returns `partial=[]`/`rejected=[]`/n_changed=29 (clean full
pool, was partial-with-1-excluded under Part B alone), cluster A correctly returns
a `partial` excluding the 43-beat (delta +11) and 31-beat (delta −1) outliers.
Files changed: `harmonia/models/user_constraints.py` (`_span_to_beats` +
docstring), `tests/test_user_constraints.py` (3 new tests). Did NOT touch
`chart_interactive.py`/`app_shell.html`. Does NOT change `_span_to_segments`
(joint_decode segment-level pooling, outside Part A's diagnosed scope — rule #4).
No commits. Artifacts: `scratchpad/partA_three_state_comparison.json`,
`scratchpad/group_pool_partA_fix_verify_results.json`.

---

# WEAK-LINK GRACEFUL EXCLUSION IN `pool_beat_evidence` (PART B of the drift + weak-link brief) — ONE MISMATCHED SPAN NO LONGER BREAKS THE WHOLE MERGE GROUP; POOL THE MODE, EXCLUDE THE REST EXPLICITLY; REAL-AUDIO 1/10 → 9/10 GROUPS APPLIED — SHIPPED IN SOURCE + TESTS + `/api/reinfer` (2026-07-19, budget up to 5h stated)

Scope: split-brief, PART B ONLY — confined to `pool_beat_evidence`'s own
validation/pooling logic; did NOT touch `_span_to_beats` or beat-grid
construction (Part A's territory, entry immediately below). Full writeup:
`docs/known_issues.md`, entry "`pool_beat_evidence` graceful weak-link
exclusion..." (top of file). One-paragraph verdict:

The user's rule was broken because `pool_beat_evidence` required EVERY span in
a merge group to share the identical beat count — one weak link (a span
quantized ±1 beat off by the double-quantization noise Part A root-caused)
killed the ENTIRE group, and for N-way section clusters that meant almost every
real group died (aretha 0/2, abba 0/3, autumn 1/5 groups applied). **Fix**: on
unequal beat counts, pool the MODE (majority) beat count's spans and EXCLUDE the
mismatched ones EXPLICITLY (new `pooled_report` out-param + enriched `rejected`
+ a new `/api/reinfer` `partial` response field) — never force-align by
truncating/padding (a far-miss span is different music, excluding is safe,
forcing corrupts). Subtleties handled + justified in the docstring: no-majority
tiebreak (deterministic toward larger beat count), `MIN_POOL_SPANS=2` floor
(a group that collapses to 1 surviving span is UNPOOLABLE and reported, not a
silent no-op), near/far-miss magnitude reported, order-independence preserved
(reads the immutable `orig` snapshot), all-bad batch still hard-raises.
**Verification**: full suite 467→473 (6 new red-first tests, confirmed failing
on pre-fix code first); re-ran the EXACT failing whole-block cluster spec (the
non-workaround `group_pool_section_clusters.py` encoding) through the production
in-process path AND real HTTP `/api/reinfer` (isolated server, port 7799) on all
3 songs — aretha 0→2/2, abba 0→3/3, autumn 1→4/5 groups applied (1 genuinely
unpoolable [16,15], reported), 21 spans excluded honestly, HTTP n_changed
56/130/82, all routes re-checked 200. **Cross-validation with Part A**: it
predicted the residual genuine length-outliers (aretha ~43-beat block, autumn
20-beat block) that "Part B's exclusion SHOULD drop" — this fix excluded exactly
those (aretha +12, autumn +4). Complementary: Part A's duration-count fix would
make the ±1 near-miss exclusions rare, leaving this as the outlier safety-net —
recommended as its own focused next call (not entangled here). Artifacts:
`scratchpad/group_pool_graceful_verify.py`(+`_results.json`). Files changed:
`user_constraints.py`, `chord_pipeline_v1.py`, `harmonia_server.py`,
`tests/test_user_constraints.py`. No commits. Did not touch
`chart_interactive.py`/`app_shell.html` (mtime-verified untouched).

---

# SPAN→BEAT-COUNT "DRIFT" ROOT-CAUSED (PART A of the drift + weak-link brief) — IT IS DOUBLE-QUANTIZATION NOISE, NOT TEMPO DRIFT; REAL-BEAT COUNTING DOESN'T FIX IT; DURATION-BASED COUNT DOES (6/8 clusters) — NO SOURCE CODE CHANGED (correctly deferred to Part B) (2026-07-19, budget up to 5h stated, ~1h used)

Scope: split-brief, PART A ONLY (root-cause the span→beat-count drift; Part B
— `pool_beat_evidence` graceful weak-link exclusion — is a separate parallel
agent, whose file `user_constraints.py` I did NOT touch). Full writeup:
`docs/known_issues.md`, entry "Span→beat-count 'drift' ROOT-CAUSED..." (top of
file). One-paragraph verdict:

**The brief's Part-A mental model (accumulating tempo/phase drift over a long
song, fixable by counting real detected beats instead of constant-tempo
arithmetic) is falsified by direct measurement on the 3 real songs' actual
failing clusters.** The conversion lives in
`user_constraints.py::_span_to_beats`: `count = argmin(t1) − argmin(t0)` over
the production **constant-tempo uniform grid** `bt` (`chord_pipeline_v1.py`
~L2926). It IS constant-tempo arithmetic, but the bug is not "drift" — it's
that differencing two *independent* nearest-index lookups sums each endpoint's
±0.5-beat rounding error into a ±1-beat scatter (`round(a)−round(b)` noise 1.0
vs `round(a−b)` noise 0.5). Evidence: (1) uniform-vs-real-beat offset never
exceeds 0.5 beat on ANY song (max 0.490/0.467/0.500 — no accumulation, so no
"1-beat slip at bar N" exists; the error is equally likely at bar 5 or 300);
(2) `corr(|count−mode|, block_time)` weak/mixed (−0.38..+0.54) — position-
independent; (3) real-beat counting recovers the SAME 0/2,0/3,1/5 equal-count
clusters as the current method (no better). **The fix that works**:
`round((t1−t0)/period)` — single-quantized duration — recovers equal counts on
**6 of 8** previously-failing real clusters (aretha 0→1/2, abba 0→3/3, autumn
1→3/5). The 3 residual failures are all GENUINE length-outliers (aretha's
43-beat block, autumn D's 20-beat block) that Part B's exclusion logic SHOULD
drop. **Not implemented**: the duration-count change isn't isolated —
`_span_to_beats` returns `(b0,b1)` used by `pool_beat_evidence` for BOTH the
precondition AND its offset-alignment loop, so changing it requires editing
that function (Part B's exclusive territory, parallel agent live in the file).
Correctly deferred rather than forced. The two fixes are complementary: a
future duration-count switch INSIDE `pool_beat_evidence` makes precondition
mismatches rare, leaving Part B's weak-link exclusion as the intended
safety-net for real length-outliers. Artifacts:
`scratchpad/drift_rootcause_check.py`(+`_results.json`). No production/UI/test
files touched (verified via `git diff`); no `/api/reinfer` calls (Part A
needed none — measurement was direct beat-grid reconstruction); no commits.

---

# LEARNED grain=4/8 AUTO-TIER CLASSIFIER — CIRCULAR WIN CAUGHT AND ABLATED, HONEST 4th NEGATIVE RESULT (2026-07-19, budget up to 3h stated) — READ THIS FIRST

User's ask (translated): at grain=4 or grain=8, find the automatic-fusion
decision boundary with a LEARNED classifier (not a hand threshold),
optimizing for near-zero false positives. Full writeup:
`docs/known_issues.md`, entry "LEARNED (not threshold) grain=4/8 auto-tier
classifier..." (top of file). One-paragraph verdict:

Trained directly on REAL AUDIO via leave-one-song-out CV (the brief's
explicit real-audio-first instruction, a genuinely different angle from
every prior symbolic-corpus-trained attempt tonight), with 5 features
including a neighborhood-noise proxy (`local_variance`) and an audio/
symbolic disagreement feature (`abs_diff`). **The apparent big win (logreg
recall 0.601 @ FPR=0.011 vs threshold's 0.104, grain=4) was caught as
circular before being reported**: `symbolic_sim`/`abs_diff` are derived from
the same baseline-decode buckets that the pseudo-GT label itself is defined
from. Audio-only ablation collapses it (logreg 0.601→0.018, GBM 0.828→0.055
at FPR<=0.01) — worse than the plain threshold. Grain=8 has ~1 positive
label in the entire 1244-pair real pooled census — a label-scarcity
ceiling, not a modeling gap. `external_gt_match`'s GT-design gap (only ever
1 or None, zero confirmed negatives) was also caught before its vacuous
precision=1.0 numbers were reported as a finding. **Verdict: do not ship —
this is the 4th independent negative result tonight on this exact problem,
and it replicates `merge_criterion.py`'s bar-level finding (logreg loses to
threshold) at a new grain with a genuinely different, real-audio-native
training approach — "more complex classifier ≠ better" holds up under a
real, honest attempt, not an assumption.** Concrete next prerequisite:
~10-20 externally-CONFIRMED-FALSE real-audio block pairs (symmetric to the
confirmed-true set that already exists) — the missing ingredient is
negative-labeled real data, not a better model. Artifacts:
`scratchpad/learned_autotier_grain48.py`,
`scratchpad/learned_autotier_grain48_results.json`. No production/UI files
touched, no `/api/reinfer` calls, no commits.

---

# SECTION-GRAIN (4 vs 8) AUTO-TIER FOR AUTOMATIC FUSION — BUILT AT BOTH GRAINS, NEITHER SAFE ON REAL AUDIO (2026-07-19, budget 2.5h stated) — READ THIS FIRST

User's question: "il faut trouver le niveau k de fusion qui marche le mieux:
4 ou 8?" for AUTOMATIC fusion (merge without human confirmation) — distinct
from the grain=8-wins ranking-quality result already in `docs/known_issues.md`
(that's about ordering human-reviewed suggestions; this is about whether
either grain can be trusted to merge silently). Full writeup:
`docs/known_issues.md`, "Section-grain (4-bar vs 8-bar) AUTO-tier for
automatic fusion..." (top of file). One-paragraph verdict:

Built the block-grain equivalent of `tau_auto_search.py`'s bar-level
auto-tier methodology at BOTH grain=4 and grain=8
(`scratchpad/section_tau_auto_search.py`, full 2399-tune iReal corpus,
nested train/val/blind-test, GT = block-level chord identity not
same-section-letter — a stricter, more mechanistically-correct redefinition
than the existing section-suggestion tool's GT). Had to fix the GT mid-call
(rule #4): literal all-bar-match was unusable on real audio (pseudo-GT=1 for
0/165 real pairs, including pairs INSIDE aretha's own externally-confirmed
vamp region — per-bar decode flicker compounds across the block), so
switched to "allow-one-mismatch" (matches the section tool's own worked
example, autumn_leaves' known-true A/A repeat scoring 7/8=0.875). Symbolic
corpus tau_auto: grain=4 0.9665, grain=8 0.9583, both nested-CV validated
<=2.2% blind-test error. **Real-audio validation (pseudo-GT AND external
non-circular GT reused from tonight's earlier aretha/autumn_leaves external
checks) finds the SAME real-to-symbolic transfer failure bar-level
tau_auto=0.96 had, worse in degree**: the direct threshold port recovers
just 0.9% (grain=4) / 0% (grain=8) of externally-confirmed true block pairs;
the joint audio+symbolic gate (same fix that worked at bar level) restores
~100% measured precision but collapses the pooled candidate count to n=1-2
across all 3 real songs combined — not a usable auto-tier at either grain.
**Verdict: neither grain=4 nor grain=8 is safe for automatic section-level
fusion at this data scale — stick to human-reviewed suggestions (already
shipped).** No production/UI files touched, no `/api/reinfer` calls, no
commits (scope-guarded alongside two concurrently-running agents this
session). Artifacts: `scratchpad/section_tau_auto_search.py`(+`_results.json`),
`scratchpad/section_realaudio_autotier.py`(+`_results.json`),
`scratchpad/auto_tier_grain_comparison_results.json`.

---

# N-WAY SECTION-CLUSTER GROUP POOLING — MECHANICALLY BROKEN AS SPECIFIED, MITIGATION WORKS BUT REGRESSION RISK COMPARABLE-TO-WORSE THAN PAIRWISE (2026-07-19, budget 2.5h stated) — READ THIS FIRST

Direct extension of the PAIRWISE `pool_beat_evidence` validation
(2026-07-18 ★ CHORD-ROBUSTNESS / BAR-MERGE entries below — the
order-independence fix and the tau_auto=0.96 real-audio measurement, 61%
pooled regression rate baseline this call compares against). Full
writeup: `docs/known_issues.md`, entry "N-WAY section-cluster group
pooling..." (top of ★ CHORD-ROBUSTNESS / BAR-MERGE / STRUCTURE). One-
paragraph verdict:

**Finding 1 (found before any accuracy question): the literal spec
("submit a whole cluster's blocks as ONE merge, spans = whole-8-bar-block
time ranges") is MECHANICALLY BROKEN for almost every real cluster.**
`_span_to_beats` maps each ~32-beat block span independently off the real
beat grid; small per-block quantization drift compounds over 32 beats, so
the probability ALL N cluster members share the exact same beat count
collapses fast with N — and because a fully-malformed batch still raises
(all-or-nothing), this took down the ENTIRE request for 2/3 songs: aretha
0/2 groups applied, abba 0/3 applied, autumn_leaves 1/5 applied (the one
2-block "cluster" that was really just a pair). **Mitigation**: decompose
each cluster into up to 8 per-bar-OFFSET merges (bar 0 of every member
pooled together, bar 1 of every member pooled together, ...) — shorter
spans drift less, 16-40 of the possible merges per song now actually
apply. All accuracy numbers below use this mitigation (the ONLY encoding
that exercises the mechanism at group scale).

**Finding 2: pooled regression rate 63.4% (399/629 bars) vs the pairwise
baseline's 61% (162/267)** — comparable on 2/3 songs (aretha even
slightly BETTER, 57.3% vs 68%; autumn_leaves close, 60.9% vs 62%) but a
real +11pp WORSE on abba (69.2% vs 58%), and the average confidence-loss
magnitude is ~66% larger pooled (-0.0900 vs -0.0542) — the hypothesized
"more can go wrong in a bigger group" direction is real but modest, not
catastrophic, and song-dependent enough to argue against a blanket
"always safe at k<=5" policy.

**Finding 3 (coordinator-flagged mid-task, confirmed real): within-
cluster pairwise-similarity outlier detection found a genuine "two
variations under one letter" case** — abba's 12-block "A" cluster flags
block 0 (the song's literal opening 8 bars, similarity 0.592 to the rest
vs cluster mean 0.694±0.088) as an outlier relative to its 11 mid/late-
song siblings; autumn_leaves' 14-block "D" cluster similarly flags a late
block (34, bars 272-280). **But the tested stricter gate (drop flagged
outliers, re-pool) only closes ~15% of the regression gap** (63.4%→62.8%
pooled) — aretha's numbers are byte-IDENTICAL gated vs ungated (no
outlier existed to drop) — because the dominant risk is diffuse weak-
signal disagreement spread across most pairs in most clusters, not
concentrated in single bad-apple blocks a mean-1std filter can isolate.
Consistent with the already-known grain=8 detector AUC=0.674 (vs bar-
level's 0.99) and the k-selection rule's own 54.1% exact-match ceiling.

**Spot-checks, method stated explicitly per cluster**: aretha's largest
cluster (B, n=5) — POSITIVE, externally corroborated by reusing the
2026-07-18 external-GT-check's landmarks (Cm-vamp chart + RMS no-chord-
bridge location): all 5 members fall cleanly outside the bridge, which
the clustering independently isolated into its own singleton letter.
abba's largest cluster (A, n=12) — MIXED, internal-consistency method
only (11/12 mutually consistent, block 0 flagged, per Finding 3) — no
fresh listening pass, explicitly time-boxed per the brief's own fallback
allowance. autumn_leaves' largest clusters (C/D, n=14 each) — NOT
externally verifiable, a known pre-existing ceiling (bar-precise iReal
alignment only reliable for t<22s, no cluster member falls there) — not
re-attempted, per rule #4.

**Recommendation: premature to ship as a "pool this whole cluster" UI
action**, for three independent reasons — (1) the literal spec doesn't
run at all without the per-bar-position re-encoding, itself unshipped;
(2) even working, measured regression risk is comparable-to-worse than
the pairwise tool, which itself was never cleared for auto-apply
(tau_auto=0.96 real-audio precision only 39.4%); (3) the proposed
stricter all-pairs gate is real but only closes ~15% of the gap — the
bottleneck is the underlying grain=8 similarity signal, not the pooling
mechanism itself. If revisited: surface the within-cluster similarity
matrix as a USER-FACING signal (let a human exclude an odd-one-out before
confirming) rather than an automatic filter, and treat a real grain=8
detector improvement as the actual prerequisite. Artifacts:
`scratchpad/group_pool_section_clusters.py`,
`scratchpad/group_pool_per_bar_position_run.py`,
`scratchpad/group_pool_gated_perbar_run.py`,
`scratchpad/group_pool_section_clusters_results.json`. No production/UI
files touched, no commits.

---

# MATRIX-INTRINSIC k-SELECTION (EIGENGAP / GAP STATISTIC / SVD KNEE) — CONFIRMED NEGATIVE RESULT (2026-07-19, budget 2.5h stated) — READ THIS FIRST

Direct follow-up to the PRINCIPLED k-SELECTION entry immediately below.
User's question: "et si on prend la matrice de similarité à granularité 8
ou 16 on peut trouver un critère pour trouver le k?" — can the section
similarity matrix ITSELF, per song, supply a k signal to complement (not
replace) the corpus length prior. Full writeup:
`docs/known_issues.md`, entry "Matrix-intrinsic k-selection (eigengap,
gap statistic, SVD knee) tested properly at corpus scale...". One-
paragraph verdict:

**Genuinely re-tested rather than assumed to fail**: the project's one
prior spectral-clustering negative result (`clustering_bakeoff_results.json`,
`spectral_eigengap`, FPR floor 0.34-0.39) was on tiny bar-merge graphs
(m~4-15 blocks) for a different task (pairwise merge-candidate generation,
no reject option) — grain=8 section graphs are comparably-or-larger
(autumn_leaves=41 blocks), so whether the finding transfers was a real
open question. **It transfers, and is confirmed at genuine corpus scale
(1873-1943 tunes, same eval protocol as the length-prior work) —
eigengap (51.5% exact-match), SVD-knee (50.8%), and gap statistic (45.9%)
ALL underperform both the existing prior-alone (53.6%) and the trivial
"always predict corpus mode" baseline (52.6%)** — gap statistic is a
clear loser (-6.7pp vs. trivial baseline). Grain=16 makes all three
methods WORSE (37.8-38.5% exact-match), not better — larger blocks just
throw away resolution on typical-length iReal tunes. **Combining the best
matrix signal (eigengap) with the existing length prior is flat-to-
negative** (54.4% exact / 91.7% within-1 at best weight, vs. the deployed
prior+silhouette's 54.1%/93.4% — a REGRESSION in within-1). On the 3 real
songs, eigengap and SVD-knee both **collapse to k=2 for all three songs**
(one dominant "this-block-vs-rest" spectral gap drowns out the finer
splits the prior correctly infers at k=5/4/3); gap statistic shows some
real per-song sensitivity (k=4 for autumn_leaves/abba) but is the
corpus-scale loser of the three, so not trustworthy alone. **Verdict,
stated as a legitimate finding**: matrix-intrinsic signals add nothing
here — this is direct, now doubly-confirmed evidence for the length-prior
entry's own honest "the prior does almost all the work" finding.
**No production change** — prior+silhouette stays deployed as-is.
Artifacts: `scratchpad/matrix_intrinsic_k.py`,
`scratchpad/matrix_intrinsic_k_results.json`. No production/UI files
touched, no reinfer calls.

---

# PRINCIPLED k-SELECTION: EMPIRICAL PRIOR + SILHOUETTE (2026-07-18, budget 2.5h stated) — READ THIS FIRST

Direct follow-up to the k<=5 CONSTRAINT call immediately below, replacing
its stopgap `clip(round(n_blocks/8),3,5)` adaptive-k formula ("a reasonable
heuristic, not fit to data") with a data-fit rule, per the user's ask to
learn P(k|song_length) from real data and combine it with the existing
silhouette signal. Full writeup: `docs/known_issues.md`, entry "Principled
k-selection: empirical P(k | song_length) prior fit from the FULL iReal
corpus (1992 tunes)...". One-paragraph verdict:

**The iReal corpus (2401 tunes parsed, 1992 with >=2 real sections) gives
genuine ground-truth k, and the length dependency is real but moderate**:
Pearson r=0.496 (n_bars vs k), p<1e-120, mean k rises 2.0 (<=16 bars) to
3.78 (129-160 bars), corpus-wide k is bounded [2,5] everywhere (never 6+,
an independent corpus confirmation of the user's own k<=5 rule). Fit both
a binned histogram AND a log-linear regression (`k=-1.126+1.014*ln(n_bars)`,
R^2=0.259) because the histogram has almost no data past 160 bars while 2
of the 3 real songs (abba 232, autumn_leaves 328 bars) sit past that —
flagged the extrapolation/genre-domain caveat explicitly rather than
hiding it. **Combined rule `score(k)=log P(k|n_bars)+weight*silhouette(k)`
reproduces the old ad hoc heuristic's exact k on all 3 real songs at every
weight tested** (autumn_leaves 5, abba 4, aretha 3) — a genuine
convergence, not built to match by construction. **Corpus-scale validation
(full 1873-tune eval, no subsampling) is the most important honest
finding: the LENGTH PRIOR alone (53.6% exact-match, 94.9% within-1) does
almost all the work** — combined rule (54.1%/93.4%) barely beats prior-
alone, while silhouette-alone (49.9%/82.0%) is barely better than a
trivial "always predict the corpus mode" baseline (52.6%). Kept
silhouette in the formula anyway (it's the only signal with any
song-specific sensitivity, and the only one available at all on real
audio), but reported the "prior does most of the work" finding
transparently rather than oversell the combination. Weight sweep
(0.5-20) confirms the rule isn't weight-sensitive in any practical way.
Artifacts: `scratchpad/build_k_prior.py`, `k_prior_corpus_extract.json`,
`scratchpad/k_prior_selection.py`, `scratchpad/k_prior_results.json`. No
production/UI files touched.

---

# k<=5 CONSTRAINT + INSPECTABLE MATCHING CRITERIA (2026-07-18, budget 3h stated)

Direct follow-up to the DUAL-MATRIX + CLUSTERING call immediately below —
that call built the dual audio+symbolic matrices (Mantel-validated,
p<=0.019 all 3 songs) and a complete-linkage clustering that correctly
groups autumn_leaves' block0/block1, but reported it at k=10 (10 distinct
letters A-J), which the user flagged as wrong on its face — real songs
essentially never have more than ~5 distinct section TYPES even with many
repeats. Full writeup: `docs/known_issues.md`, entry "k<=5 section-label
constraint enforced... 3 inspectable matching criteria compared...", top
of file. One-paragraph verdict:

**Premise check confirmed k<=5 is musically correct, not just a
constraint to satisfy**: autumn_leaves n_blocks=41 (~328 bars) vs
canonical AABC 32-bar form ≈ 10.25 chorus repeats of a 4-section
vocabulary — k=10 was over-fragmentation, not a valid finer-grained
answer. Built 3 candidate matching criteria on the SAME validated
complete-linkage algorithm, differing only in the distance formula:
`blend_0.6_0.4` (existing default), `symbolic_primary_audio_gate`
(AND-style, audio must clear a p40 floor before symbolic_sim is trusted),
`mutual_topK_rank_bonus` (rewards pairs independently top-20% in BOTH
matrices — a direct operationalization of the Mantel-test logic). Swept
k=3,4,5 for all 3 real songs (`scratchpad/section_matching_criteria.py` →
`section_matching_criteria_results.json`). **Candidate 2 REJECTED**: root-
caused (not just observed) a degenerate all-in-one-cluster collapse for
autumn_leaves and abba at every k tested — the 1.5 penalty ceiling creates
a mass of tied distances that complete-linkage chains through, the SAME
failure family as the union-find chaining collapse already documented in
the entry below (second independent confirmation hard AND-gates are wrong
here). **Candidate 3 is operationally IDENTICAL to candidate 1 at k<=5**
for all 3 songs (its own silhouette-suggested k=12 shows the extra signal
only shows up at finer grain than this constraint permits) — no reason to
prefer it here. **Recommended and deployed**: `blend_0.6_0.4` with an
ADAPTIVE k = clip(round(n_blocks/8), 3, 5) rather than one fixed k for all
songs (forcing aretha, the shortest song at 10 blocks, to k=5 produces a
5-distinct-letter string across only 10 blocks — worse fit than k=3).
Result: autumn_leaves k=5 (`AABCDCDCCDCECDEBCCDECEDDCCECDADECEDCEDDDD`,
block0==block1 still ✓, the one song with a known-correct answer), abba
k=4, aretha k=3 — all pass <=5, none degenerate. Corpus-scale precision/
recall does NOT transfer to this task (it's a fixed-k partition, not a
pairwise threshold decision) — reported each criterion's own inspectable
gate/coverage stats instead of forcing an inapplicable metric.
`section_structure_clusters_grain8.json` updated in place (same schema,
additive `all_candidates` key for a future side-by-side view) — data only,
no production/UI files touched.

---

# DUAL-MATRIX + CLUSTERING FOLLOW-UP TO THE SECTION-SUGGESTION TOOL (2026-07-18, budget 3h stated) — READ THIS FIRST

Direct response to the user's report "the first 8 bars repeat twice in a
row [Autumn Leaves] and you should be able to detect that." Full writeup:
`docs/known_issues.md`, entry "Section-suggestion ranking failure
diagnosed... + a NON-ranking fix that actually works" (★ STRUCTURE /
SEGMENTATION, top of file). One-paragraph verdict:

**Confirmed the ranking bug is real and diagnosed its exact cause**
(deployed `symbolic_sim` reads the model's own noisy real-audio chord
decode, not the clean iReal chart the premise-check used — code-level
confirmed, `section_merge_candidates.py:52,103` vs
`section_premise_check.py`'s `load_playlist`/`tune_to_mma`). **Tried six
different ways to fix the pairwise RANKING with a joint audio+symbolic
criterion — all failed to move the user's own block0/1 pair above its
audio-only rank of 42/60**, even though the same joint criterion
demonstrably raises aggregate corpus precision (0.38→0.48-1.0,
`section_realaudio_check_results.json`) — an honest, explicable tension:
the corpus-average fix leans on exactly the noisy signal that's most
degraded for this one example. **The real fix is a change of algorithm,
not formula: complete-linkage agglomerative clustering on the joint
grain=8 similarity matrix robustly places autumn_leaves' block0 and
block1 in the same cluster at every k tested (4-12), insensitive to the
exact audio/symbolic weighting** — first attempt (union-find) failed
badly via single-linkage chaining (31/41 blocks collapsed into one blob),
documented as a negative result. Built the dual-matrix (audio SSM +
symbolic SSM, grain=8) the user explicitly asked to see, for all 3 real
songs, as JSON data (not a rendered chart, per tonight's division of
labor). Session log:
`scratchpad/dual_matrix_grain8.py` → `dual_matrix_grain8_results.json`;
clustering → `section_structure_clusters_grain8.json`. No production/UI
files touched (out of scope this call). No candidate JSON regenerated —
no verified ranking improvement was found, so the honest choice was to
document rather than force a change.

---

# SECTION-LEVEL (8-BAR) SUGGESTION UI FOLLOW-UP (2026-07-18, budget 2.5h stated)

Direct continuation of the "SECTION-LEVEL (8-BAR) REPEAT-DETECTION
SUGGESTION TOOL" entry immediately below — that call built the
algorithm/data layer (candidate JSON, ROC calibration, real-audio
validation) and explicitly deferred the UI because the relevant files
(`app_shell.html`, `chart_interactive.py`, `harmonia_server.py`) were
scope-guarded/locked by a parallel agent. This call builds the UI now that
the files are free. Full writeup in `docs/known_issues.md` (★ UI /
NAVIGATION, "SECTION-level (8-bar) suggestion UI — the FOLLOW-UP... is now
built, live, and playwright-verified"), one-paragraph version here:

Added `GET /api/section-merge-candidates/<filename>` (thin passthrough,
`scripts/harmonia_server.py`, strips both `inferred_` prefix and `.html`
suffix to find `scratchpad/section_merge_candidates_<slug>_grain8.json`)
and a new client-side mode in `app_shell.html`: "🧩 Section suggestions"
(violet, mutually exclusive with the existing bar-merge and pool-two-passes
modes), rendering the top 15 rank-ordered candidates as faint dashed-
outline tints over their 8-bar spans + a small tappable badge on each
span's first bar (not a full per-cell tint or per-bar badge — an 8-bar
span is 2 full grid rows, so a lighter-weight treatment was needed to stay
legible). Confirm sheet copy is deliberately softer/more hedged than the
bar-level tool's ("much weaker signal... worth a listen, not a strong bet
... use your ear") given the corpus-validated ~45-55% precision vs the bar
tool's ~84-92%. Confirm reuses `/api/reinfer`'s existing `merges` mechanism
unchanged. Verified with real playwright screenshots (chromium confirmed
working) on all 3 real songs (autumn_leaves 15/60 shown, abba_chiquitita
15/42, aretha_chain_of_fools 13/13) — toggle → badges render legibly →
tap badge → confirm sheet shows correct hedged copy → tap confirm → real
`/api/reinfer` round-trip completes and chart updates, zero console/page
errors at every step. Regression-verified the pre-existing "💡 Bar
suggestions" tool and the "★ CHART / BAR-GRID" bar-assignment fix both
still work after this call's additive changes. Screenshots in
`scratchpad/screenshots/` (`04b_scroll_*.png`, `07_confirm_sheet.png`,
`08_after_confirm.png`, `regress_inferred_*.png`).

---

# SECTION-LEVEL (8-BAR) REPEAT-DETECTION SUGGESTION TOOL (2026-07-18, budget 2.5h stated) — READ THIS FIRST, THEN THE AUTUMN-LEAVES ENTRY BELOW

SCOPE-GUARDED call (parallel agent held `chart_interactive.py`/
`app_shell.html`/`harmonia_server.py` for the whole call — none touched;
algorithm/data layer only). Full writeup:
`docs/known_issues.md` "SECTION-level (8-bar) repeat-detection suggestion
tool" (★ STRUCTURE / SEGMENTATION, top of file).

**One-paragraph verdict**: the user's own Autumn Leaves worked example
(A-section repeat, bars 1-8 vs 9-16) checks out — symbolic sim 0.875,
real-audio sim 0.789, both clearly elevated — but corpus-scale (full
iReal, nested train/val/test) the block-level chord-similarity signal for
"are these two 8-bar spans the same recurring section" is MUCH weaker than
the analogous bar-level "same chord identity" signal used for the bar-merge
tool (ROC-AUC 0.674 best case (jazz1460-only, grain=8) vs 0.99 for bar-
level chord-identity), with no cheap-recall regime — precision sits around
0.45-0.55 across the whole 70-95% recall range, degrading smoothly rather
than having a sharp knee. Real audio shows the SAME real-to-symbolic
transfer gap already diagnosed for the bar-level tau_auto (audio-only
agree_rate tops out ~0.31-0.33 even at very high similarity, tiny sample
sizes). Shipped as a RANK-ORDERED, suggest-tier-ONLY tool (no auto-tier,
no hard threshold) with an explicit lower-confidence caveat baked into the
candidate JSON's `meta` — grain=8 beats grain=4 at every recall target,
confirming the user's own "8-bar as our standard" intuition is the right
choice even though the underlying signal is weaker than the bar tool's.

**A real methodology bug was caught and fixed mid-call**: the pair-pool
builder initially excluded ADJACENT blocks (`MIN_GAP_BLOCKS=1`, a naive
port of the bar-level `MIN_GAP=4` convention) — which silently excluded
the user's own worked example (adjacent A-section blocks) from the entire
calibration and candidate pool. Caught by cross-checking generated
candidates against the premise check, fixed to `MIN_GAP_BLOCKS=0`, all
reported numbers are post-fix. Logged per CLAUDE.md rule #4/#6.

**UI follow-up (NOT built this call, files locked)**: candidate JSON format,
recommended badge treatment (visually distinct/lower-confidence than the
bar tool, reuse its confirm-sheet interaction pattern), and next validation
step are spelled out in known_issues.md's entry — read that before starting
the UI work.

Session log, in order run:
1. `scratchpad/section_premise_check.py` — Autumn Leaves premise check,
   PASS (see verdict above). `section_premise_check_results.json`.
2. `scratchpad/section_pairs.py` — corpus pair-pool builder (reuses
   `tau_auto_search.load_corpus_bar_chords` verbatim); found+fixed the
   O(n²)-per-pair prefix-sum perf bug (4min+ for a smoke test -> 5.7s) AND
   (later) the MIN_GAP_BLOCKS adjacency bug above.
3. `scratchpad/section_roc_suggest.py` — full 7-playlist corpus nested
   ROC/recall-target retune, grain 8 and 4. `section_roc_suggest_results.json`.
4. `scratchpad/section_roc_jazz_only.py` — follow-up diagnostic, triggered
   by the surprisingly weak full-corpus AUC: restricting to jazz1460-only
   recovered +0.08-0.10 AUC at both grains, confirming the "vamp-based
   genre negatives dilute the signal" hypothesis. `section_roc_jazz_only_results.json`.
5. `scratchpad/section_realaudio_check.py` — 3-real-song audio-only +
   joint-gate transfer check, reproduces (worse, smaller-sample) the
   bar-level tau_auto real-audio transfer failure. `section_realaudio_check_results.json`.
6. `scratchpad/section_merge_candidates.py` — final candidate generator,
   rank-ordered top-60, `section_merge_candidates_<slug>_grain{8,4}.json`
   (6 files). Verified the premise-check example survives at rank 41/60.

---

# AUTUMN LEAVES GAP-CLOSING CALL (2026-07-18, budget 1.5h stated, ~1h used) — READ THIS FIRST

Direct continuation of the REAL (NON-CIRCULAR) EXTERNAL GT CHECK call
immediately below (read that first) — closes its explicit
"autumn_leaves: NOT attempted, time-boxed, lowest priority" gap. Full
writeup: `docs/known_issues.md`, entry titled "autumn_leaves external GT
check: 0/14 auto-tier candidates confirmable..." (top of file, ★
CHORD-ROBUSTNESS / BAR-MERGE).

**Headline, stated plainly up front: unlike aretha (11/11 confirmed) and
abba (2/2 confirmed), autumn_leaves' real-audio structure defeats
bar-precise external verification entirely — 0 of 14 current auto-tier
candidates could be confidently checked against the iReal chart within
budget.** This is an honest "cannot verify" result, not a forced pass or
fail, and it is the correct outcome given the evidence, not a shortfall of
effort (see below for what was tried).

**Tune identity**: `data/ireal/jazz1460.txt` has exactly one "Autumn
Leaves" entry (Kosma/Joseph, key G-, Medium Swing, 4/4, standard 32-bar
AABC form) — unambiguous, no arrangement-picking judgment call needed.

**Unplanned but load-bearing finding: `docs/audio/autumn_leaves.m4a`
(422.3s) is NOT the video its filename/corpus entry implies.**
`scripts/build_yt_corpus.py` maps this song to YouTube video id
`YVedK1VUfLM`, titled "Autumn Leaves (Remastered)" by Nat King Cole —
independently confirmed via `yt-dlp --print duration` to be **160s**, not
422s. This is the SAME mismatch already flagged in `docs/known_issues.md`
issue #35 ("autumn_leaves GT span 160s vs inferred 422s, 2.64×") — this
call adds independent confirmation of the true short video's real duration
(via yt-dlp, not just the existing GT-chart-timeline argument) and eliminates
"ffprobe read the wrong stream" as an explanation. The 422s file is a
real, different (likely big-band/combo, many-chorus, solo-heavy) recording
of the same tune, not a corrupted read of the short vocal track — not
solved this call, just corroborated with a second, independent method.

**Alignment method, honestly reported as insufficient**: reused
`docs/plots/annotations/irealb_autumn_leaves_sectionwise.json`, EXISTING
prior work (`docs/known_issues.md` #37, 2026-07-14) that already did
real-seconds structural alignment via chord-proxy matching against the
chart — more rigorous than a fresh RMS pass could achieve in this budget,
so reused rather than re-derived (CLAUDE.md rule #6 spirit: don't rebuild
what already exists and works). It confidently covers ONLY chart bars 0–15
(both A-sections), t=0.65–22.0s, match_score 0.46–0.73; every section after
that (through t=142s, as far as that prior effort went) is explicitly
flagged `is_vamp=True`, match_score 0.17–0.39 — i.e. even a dedicated prior
alignment attempt distrusts its own output past t=22s. A fresh
`librosa.feature.rms` pass this call found NO sharp silences (unlike
aretha's clear a cappella bridge) and NO periodicity peak near the chart's
expected ~41s chorus length via autocorrelation — this is a continuously-
played arrangement with no RMS-detectable chorus boundaries, confirming
the prior effort's difficulty rather than beating it. (RMS tail: a smooth
~9s exponential fade-out from ~t=412s to ~420s — a studio fade, not a live
cutoff — but not clearly enough tied to a specific chart chord to use as
an anchor.)

**All 14 current auto-tier candidates fall entirely outside the one
confidently-aligned window.** Used `scratchpad/bar_merge_full_census_
autumn_leaves.json` (14 auto after the joint gate — the capped UI file
only shows 7; the brief's "36" figure is stale, superseded by the gate).
Earliest candidate bar starts at t=42.0s, 20s past the reliable boundary.
Zero of 14 could be bar-precisely checked.

**Supplementary (explicitly weaker, NOT counted as verification) signal
reported anyway, per the brief's "quality over quantity, report what you
can"**: 13 of 14 candidates' self-decode majority labels are BOTH bars
`C:7` (1/14 is `G#:7`/`G#:7`) — i.e. `symbolic_sim=1.0` for all 14 is
driven by collision on the single most common label in the whole song.
Quantified the base rate: `C:7` is the majority label on **25.5% of all
330 bars** (next most common, `C#:maj7`, is 11.8%) — a much more skewed
distribution than aretha or abba had, meaning symbolic-gate agreement is a
weaker discriminator for this song specifically (naive independence
estimate: two random bars would collide on `C:7` ~6.5% of the time by
chance alone). Under the already-established (#37) systematic +2-semitone
transpose, `C:7`→`D:7`, which IS a real, frequently-recurring chart chord
(the `Ah7 D7b13` ii-V cadence recurs 5×/32-bar chorus, resolving to the
tonic) — directionally favorable, but explicitly NOT bar-precise: it shows
the shared label is a plausible chart chord, not that these SPECIFIC bars
land on a D7 position. A rigid bar-index-mod-32 extrapolation was tried and
explicitly discarded as too fragile to use per-candidate (no drift/rubato
correction, and the one prior alignment effort already distrusts its own
tempo model past t=22s — extending it 6+ choruses further is not
defensible).

**Decision: no gate-logic change, no auto-apply wiring** (verification
call, same as its predecessor). No files modified besides
`docs/known_issues.md` and this file.
`scratchpad/autumn_leaves_gt_check.py` written to
`/private/tmp/claude-501/.../scratchpad/` (session scratch dir, not the
repo) — not saved into the repo since it's a one-off diagnostic, same
convention as the prior call's un-saved one-off snippets.

**NEXT STEP**: none of the three songs now have a full corpus-scale
external validation; autumn_leaves specifically would need either (a) a
timestamped lyric/solo-order annotation of the real 422s recording to
locate chorus boundaries, or (b) accepting that this recording's structure
(long, continuous, no silence landmarks, heavy solo content) may not be
externally verifiable at bar precision without manual listening — a
genuine structural ceiling, not a shortfall of this call's effort.

---

# REAL (NON-CIRCULAR) EXTERNAL GROUND-TRUTH CHECK CALL (2026-07-18, budget 2h stated, ~1.75h used) — READ THIS FIRST

Direct continuation of the ARETHA DIAGNOSIS call immediately below (read that
first) and the JOINT AUDIO+SYMBOLIC GATE call below that — this call finally
does the thing both flagged and neither did: **REAL, externally-sourced
ground truth (not the model's own baseline decode) for the disputed
bar-pairs**, per CLAUDE.md rule #3's trust order. Full writeup:
`docs/known_issues.md`, entry titled "REAL (non-circular) external GT check
on aretha's 11 auto-tier candidates..." (top of file, ★ CHORD-ROBUSTNESS /
BAR-MERGE).

**Sourcing**: "Chain Of Fools" and "Autumn Leaves" were both already in the
local iReal corpus (`data/ireal/pop400.txt`, `data/ireal/jazz1460.txt` —
checked via grep before assuming external search was needed, per the
brief), parsed with the existing `harmonia.data.ireal_corpus` parser (no new
decoder built). "Chiquitita" is NOT in the local corpus — sourced via
WebSearch/WebFetch, 2 independent guitar-chord sites cross-referenced
against each other. For Chain Of Fools, ALSO cross-referenced a 2nd
independent guitar-chord site against the iReal chart.

**Headline result: aretha's 11 auto-tier candidates are 11/11 (100%)
externally confirmed as real chord matches** — a real, non-circular
verification, not just self-consistency. Mechanism: "Chain Of Fools" is
essentially a single Cm/Cm7 vamp for its ENTIRE body (both iReal and an
independent guitar chart agree), so the model's own wildly-varying labels
across the 11 candidate bars (`C:7`, `D#:maj7`, `C:dim7`/`C:hdim7`, etc.) are
mostly noise/reharmonization-style misreadings of one static chord, not
real harmonic movement — verified this isn't just an assumption by locating
the song's one genuine no-chord bridge (83.7–100s, via RMS energy, audio-
derived and model-independent) and confirming none of the 11 candidates
fall inside it, plus a chroma-based (also model-independent) cross-check on
the one ambiguous edge case (bars 54/78, spanning the song's abrupt ending).

**Bonus finding: the "known false positive" aretha 13/17 is probably ALSO a
real match** — its FALSE POS label was itself derived from the model's own
self-decode disagreement, never independently verified; external sources
say the real chord doesn't change there either. The gate's conservative
demotion to suggest-tier was still right (self-decode noise justifies human
review regardless), but the number used to call it a "miss" was circular.

**ABBA**: known FP (32/64) CONFIRMED real (quality genuinely differs, and
the disputed chord `C#m` is externally confirmed rare/localized — implausible
both bars hit it). Known TP (206/222) PLAUSIBLY confirmed (directionally
consistent with 2 external chart sources, but not bar-precise — no
timestamped transcript available). Other 51 abba auto-tier candidates: NOT
attempted (abba has real chord changes throughout, unlike aretha's static
vamp — RMS-section alignment alone isn't precise enough, and no timestamped
transcript was available within budget).

**Unplanned, corpus-relevant finding: abba's detected tempo (170.45 BPM) is
a ~2x-tempo-octave lock** — 3+ independent BPM databases agree real
"Chiquitita" tempo is 83–84 BPM ("double-time 168" per one source). Same
failure mode as CLAUDE.md rule #1's POP909 song 002 example, now confirmed
on a real-audio pipeline input too. Doesn't invalidate this call's
real-seconds-based comparisons but means abba's "audio bar N" is really a
half-bar at the song's true tempo.

**Autumn Leaves: NOT attempted** (explicitly lowest priority, time-boxed
out) — tune IS in the local iReal corpus for a future call's head start;
light-touch spot check only (8/14 auto candidates decode identically both
sides — self-consistency, not external verification, explicitly not
counted as such).

**Honest limits, stated plainly**: this is a handful of hand-checked pairs
(aretha 11 + 1 bonus, abba 2), NOT a corpus-scale validation. It's real,
non-circular evidence in favor of the gate's real-audio precision being
genuinely high — strongest exactly where the project's own numbers looked
weakest (aretha) — but does not replace a proper corpus-scale, timestamped-
transcript validation if auto-apply is ever reconsidered.

**Decision: no gate-logic change, no auto-apply wiring** (verification call,
not a fix call). No scratchpad scripts added (all analysis was one-off
snippets against already-cached data + fresh librosa RMS/chroma calls). No
files modified besides `docs/known_issues.md` and this file. No commits, no
server/UI files touched.

---

# ARETHA DIAGNOSIS + V3/ENSEMBLE SWEEP CALL (2026-07-18, budget 2.5h stated, ~1h used) — READ THIS FIRST

Direct continuation of the JOINT AUDIO+SYMBOLIC AUTO-TIER GATE CALL
immediately below (read that first). Full writeup:
`docs/known_issues.md`, entry titled "Aretha's joint-gate 54.5% precision
gap is a PSEUDO-GT MEASUREMENT ARTIFACT..." (top of file, ★
CHORD-ROBUSTNESS / BAR-MERGE). Picked up that call's explicit "NEXT STEP
(a)/(b)" handoff: (a) diagnose why aretha's joint-gate precision (54.5%,
n=11) lags autumn_leaves/abba (94–100%); (b) corpus-sweep V3_tiv as an
alternative/ensemble symbolic feature (previously only premise-checked on
4 pairs).

**(a) Result: hand-inspected all 11 of aretha's joint-gate-passed
candidates individually.** 5/11 counted as pseudo-GT "disagreements" by
the existing methodology — but that methodology's pseudo-GT samples a
single MIDPOINT timestamp per bar while `symbolic_sim` itself uses a
MAJORITY-VOTE label over the whole bar. For 4/5 "disagreements" (incl.
bars 53/69, one of the two ALREADY hand-validated true positives from two
calls ago), the majority labels agree exactly (`symbolic_sim=1.0`) but the
midpoint sample landed on a different passing chord. Quantified: aretha's
baseline decode has a 26.5% majority-vs-midpoint "flicker rate" (22/83
bars) and 3.11 avg chord segments/bar, vs abba's 9.9%/2.03 and
autumn_leaves' 0.0%/2.04 — aretha's decode genuinely churns more
sub-bar (matches "shorter song, less redundancy" hypothesis), but the
effect channel is measurement-timing sensitivity, not majority-label
instability. Recomputing pseudo-GT with the SAME majority-vote convention
`symbolic_sim` uses: aretha 54.5%→**90.9%** (10/11), abba
94.2%→**100%**, autumn_leaves unchanged (100%, zero flicker), pooled
89.6%→**98.7%**. The one genuine remaining miss (aretha 54/78, `C:dim7`
vs `C:hdim7`) IS real vocabulary aliasing under V1_binary's 6-way bucket
(both map to the same `[0,3,6]` triad family) — confirms hypothesis 2 but
as a minor (1/77), not dominant, driver. **Caveat stated explicitly, not
glossed over**: this corrected number is MORE circular than the original
(both sides now derive from the identical majority-vote label, differing
only in bucket-scheme boundaries) — real hand-verified GT is still the
only way to get a true non-circular precision estimate. What's robust to
which convention is used: aretha's gap vs the other 2 songs is
overwhelmingly a pseudo-GT measurement-timing artifact, not a real
symbolic-gate failure.

**(b) Result: V3_tiv alone + AND/OR/AVG(V1,V3) ensembles, corpus-scale,
both pseudo-GT conventions** (`scratchpad/symbolic_v3_ensemble_search.py`).
Premise check on the 4 known pairs passed (V3/AND/AVG all separate
cleanly at tau=0.90; OR is measurably riskier at looser tau, re-admitting
`aretha_13_17_FALSE_POS` at tau=0.80 via V1 alone — a real, documented
failure mode of "either signal is enough"). **Corpus-scale: at the
deployed tau_symbolic=0.90, all 5 schemes (V1, V3, AND, OR, AVG) are
BYTE-IDENTICAL** — same n=77 pooled, same precision, every song including
aretha, under both pseudo-GT conventions. Honest negative result: V3_tiv
does NOT do better than V1_binary on aretha's richer harmonic vocabulary,
and no ensemble rule catches anything V1 alone misses at this threshold —
positively rules out "wrong symbolic feature" as aretha's problem,
reinforcing (a)'s finding.

**Decision: no gate-logic change shipped.** `apply_symbolic_gate()`
already used `bar_chord_majority` correctly — only the EVALUATION
scripts' pseudo-GT had the midpoint/majority inconsistency.
`TAU_SYMBOLIC=0.90` and V1_binary stand unchanged (no improvement from
V3/ensemble to justify the added complexity). No candidate JSON
regenerated (deployed gate output is unaffected). Same standing caution:
**auto-apply still NOT wired into any default path** — this call raises
confidence the gate's true precision is closer to 98.7%/90.9%-100% than
the previously reported 89.6%/54.5%, but that confidence still rests on
self-decode pseudo-GT, not independent ground truth.

**NEXT STEP**: real hand-verified ground truth (at minimum the 11 aretha
candidates + a matched abba/autumn_leaves subset) remains the single
highest-value next action — flagged by both this call and the prior one,
neither has done it. Smaller idea: `symbolic_v3_ensemble_search.py`
already implements majority-vote pseudo-GT side-by-side with the original
midpoint version as a reusable reference; a small patch to
`joint_threshold_search.py`/`realaudio_threshold_check.py` to default to
majority-vote sampling would save future calls from rediscovering this.

**Files this call**: `scratchpad/symbolic_v3_ensemble_search.py` (new),
`scratchpad/symbolic_v3_ensemble_search_results.json` (new). No files
modified — diagnosis-and-negative-result call. No server/UI files
touched. No commits.

---

# JOINT AUDIO+SYMBOLIC AUTO-TIER GATE CALL (2026-07-18, budget 3h stated) — READ THIS FIRST

Direct continuation of the DUAL-MATRIX call immediately below (read that
first — this call reuses its matrices, doesn't rebuild them). Full
writeup: `docs/known_issues.md`, entry titled "Joint audio+symbolic
auto-tier gate: raises real-audio auto-tier precision 39.4%→89.6%
pooled..." (top of file, ★ CHORD-ROBUSTNESS / BAR-MERGE). Four tasks per
the brief, in order, ~2h of the 3h budget used (task 4 optional/time-
permitting — see below for whether it ran).

**1. Premise check (cheap, first): symbolic chord-tone similarity (V1
binary, `chord_distance.py`) DOES separate the 2 known real-audio false
positives from the 2 known true positives, where audio similarity alone
does not** — FP max V1=0.87, TP=1.0 exactly, vs audio_sim sitting at
0.978-0.994 for BOTH classes. V2_weighted compresses the gap too much
(0.95 vs 1.0) to trust; excluded from further sweeps. Premise PASSED,
proceeded to corpus scale.

**2. Corpus-scale joint threshold** (`scratchpad/joint_threshold_search.py`,
all 3 songs' full candidate census, n=389 pairs, same pseudo-GT
methodology as `realaudio_threshold_check.py`, reproduced its 39.4%
number exactly as a sanity check first): audio_sim>=0.96 AND
symbolic_sim(V1)>=0.90 raises pooled precision 39.4%→**89.6%** (n=77 of
the 180-pair audio-only pool retained, 42.8% recall of that pool).
Consistent DIRECTION across all 3 songs (not a single-song artifact) but
wildly inconsistent MAGNITUDE: aretha 24.2%→54.5% (still far short of a
"never false positive" bar), autumn_leaves 38.9%→100%, abba 44.1%→94.2%.
Honest headline: real, corpus-validated improvement, but NOT sufficient
on its own for "never a false positive" — aretha specifically stays weak.

**3. Wired into `bar_merge_candidates.py`** as a purely-additive, opt-in
post-processing function (`apply_symbolic_gate`, `TAU_SYMBOLIC=0.90`) —
`candidate_groups()` itself, and every existing caller of it, is
unchanged; the gate only runs when a caller explicitly invokes it with a
baseline chord decode. Demotes (never deletes) failing auto-tier
candidates to suggest-tier; `symbolic_sim`/`tier_reason` fields added for
auditing. Regenerated all 3 songs' UI-capped and full-census candidate
JSON (`scratchpad/regen_candidates_with_symbolic_gate.py`) — auto-tier
counts drop substantially (e.g. abba full-census 111→52, aretha 33→11,
autumn_leaves 36→14), spot-checked against the known-good abba 206/222
pair (stays auto, symbolic_sim=1.0) and a known-bad high-audio-sim abba
pair (correctly demoted, symbolic_sim=0.29).

**Decision: still DO NOT wire silent auto-apply into any default/
user-facing path** — same caution as every prior call. This gate makes
the SERVED "auto" LABEL meaningfully more trustworthy (worth keeping) but
54.5%-100% per-song precision (aretha's floor) is not "never wrong."
**NEXT STEP**: diagnose why aretha specifically underperforms
(shorter song / noisier baseline decode / richer harmonic vocabulary
aliasing under V1_binary's 6-way bucket — not yet tested); try V3_tiv as
an alternative/ensemble symbolic feature at corpus scale (only
premise-checked, not corpus-swept, this call); real (not pseudo-)
ground truth is still the only way to get a non-circular precision
estimate. No server/UI files touched. No commits.

---

# DUAL-MATRIX CROSS-VALIDATION CALL (2026-07-18, budget 2.5h stated) — READ THIS FIRST

Full writeup: `docs/known_issues.md`, entry titled "Dual-matrix
cross-validation for real-audio structure..." (top of file, ★ STRUCTURE /
DUAL-MATRIX). One-paragraph summary:

User's idea, in response to the tau_auto=0.96 symbolic-to-real-audio
transfer failure (see the entry directly below this one): instead of
porting a threshold calibrated on one corpus/feature-space to another
(the exact failure mode that already burned a call tonight), build TWO
independent 1-bar SSMs from the SAME real song via two different signal
pipelines — a symbolic matrix (production `infer_chords_v1` decode ->
`chord_distance.py` V1/V2/V3 chord-tone cosine similarity) and the
existing untrained audio matrix (`rawchroma.py` bt_concat, already built
tonight) — and check whether they agree with each other on that one song,
never crossing corpora. Built `scratchpad/dual_matrix_correlation.py` for
the 3 real songs (aretha_chain_of_fools n=83 bars, autumn_leaves n=330,
abba_chiquitita n=232). **Premise check passed**: a proper Mantel-style
permutation test (999 joint row/col shuffles of the audio matrix, not a
naive Pearson which SSM block-structure would inflate) shows both matrices
agree above chance in all 3 songs x all 3 chord-tone schemes (z=1.9-10.3,
p=0.001-0.037) — genuine signal, not noise. Sanity-checked against 2
already-validated true positives from earlier tonight (aretha 53/69, abba
206/222) — both show PERFECT agreement between matrices. **Boundary/
padding analysis (K=4/8/16 vs matched-size random interior-window control)
is inconsistent across the 3 songs, an honest mixed/negative result, not a
clean win**: abba supports "boundary bars show lower cross-matrix agreement
= padding signature" (z=-2.3 to -2.9 at K=8/16); aretha shows the OPPOSITE
direction (z=+7.4 to +14.9) despite both matrices individually agreeing
boundary bars are unusually dissimilar to the rest of the song (a plausible
"static intro riff reads as uniformly different in both signals" mechanism,
not disagreement); autumn_leaves shows no effect either direction. **Net
finding: the two-matrix agreement signal does not clearly add value over
each matrix's own single-matrix "boundary bars are dissimilar to the rest
of the song" signature** — 2/3 songs show that single-matrix signature in
the expected direction, but the cross-matrix statistic itself moves in
opposite directions between those same two songs. n=3 is too few to settle
whether abba-vs-aretha is a real distinction (e.g. static-vamp vs.
harmonically-active intro) or small-n noise — flagged as the concrete next
step, not resolved this call. Full matrices + stats in
`scratchpad/dual_matrix_correlation_results.json`; no chart built (per
brief, orchestrating session's job). No commits.

---

# OVERNIGHT AUTONOMOUS CALL (2026-07-18, budget 3h stated) — SUGGEST-TIER
ROC/AUC RETUNE + AUTO-APPLY WIRED AND MEASURED — READ THIS FIRST

Direct continuation of the tau_auto entry immediately below (read that
first). Full writeup: `docs/known_issues.md`, entry titled "AUTO-tier
auto-apply WIRED and MEASURED — mechanism works, but real-audio
measurement found the tau_auto=0.96 threshold does NOT transfer to real
audio... — 2026-07-18 ★ CHORD-ROBUSTNESS / BAR-MERGE" (top of file). Three
tasks, in the brief's stated order, budget spent roughly 20%/30%/50%:

1. **SUGGEST-tier ROC/AUC retune** (`scratchpad/roc_suggest_tier.py`):
   re-derived tau_suggest around a recall target instead of FPR<=0.05 (the
   old operating point only surfaced ~22% of real merge opportunities to
   the human). 5-fold blind-test ROC-AUC=0.9885±0.0016,
   PR-AUC=0.9590±0.0047. Picked tau_suggest=0.80 (85% recall target,
   FPR~3.5-3.9%) — FPR/precision degrade much faster past 85% than up to
   it. Shipped into `bar_merge_candidates.py` (`DEFAULT_TAU` 0.93→0.80),
   regenerated all 3 songs' candidate JSON (both the UI-capped file and a
   new uncapped full-census file for tasks 2/3's own use).

2. **Auto-apply mechanism built** (`scratchpad/auto_apply_merges.py`): a
   one-time batch script (option (a) from the brief — deliberately NOT
   wired into `/api/analyze`, and, given finding 3, NOT surfaced as a UI
   button either), conservative pairs→groups resolution (union-find capped
   at 8 bars, empirically never needed since real components topped out at
   6). **Found and fixed a real production bug along the way**: sending a
   song's full auto-tier batch (17-54 groups) in one `/api/reinfer`
   request hit 100% rejection every time — one malformed merge (unequal
   beat count from tempo-grid drift) was aborting the WHOLE batch inside
   `pool_beat_evidence`, not just itself. Fixed with a per-merge
   skip-and-report (`rejected` out-param), red-first regression tests
   added, full suite 467/467 passing, server restarted + all routes
   re-verified 200.

3. **Measurement found a serious problem, not a clean win.** With the
   batch bug fixed, real auto-tier merges DID apply — and 61% of the 267
   touched bars (aggregated, 3 songs) had confidence go DOWN, 36% flipped
   LABEL entirely, directly contradicting tau_auto=0.96's "never a false
   positive" design intent. **Root-caused**: tau_auto was calibrated on
   iReal's SYMBOLIC proxy features (clean one-hot chord vectors — iReal has
   no audio) and the resulting number (0.96) was ported unchanged onto
   `bar_merge_candidates.py`'s real-audio `rawchroma.bt_concat` feature — a
   different feature space the threshold was never validated against.
   Confirmed via 2 hand-inspected real examples (abba 32↔64: sim=0.979 but
   baseline-decoded C#:maj7 vs C#:min; aretha 13↔17: sim=0.978 but
   baseline-decoded E:maj7 vs C:maj7 — different ROOT) AND a corpus-scale
   pseudo-GT check (`scratchpad/realaudio_threshold_check.py`, using the
   model's own unconstrained decode as noisy ground truth across the full
   3-song candidate census): pooled agreement rate only **39.4% at
   tau=0.96**, rising to just **62.5% at tau=0.99** — nowhere near
   98-99%. Honestly caveated: this pseudo-GT metric is a conservative
   underestimate (checked against the project's own 2 known-good validated
   pairs — one agrees, one is a case of pooling correctly overriding a
   noisy baseline read that this metric can't distinguish from a bad
   merge) — but the gap is far too large for that alone to explain.
   **Decision: auto-apply is NOT wired into any default or user-facing
   path.** Mechanism works; the threshold it was told to trust does not
   transfer to real audio. Next step: real-audio-native recalibration
   (real ground truth, not the symbolic proxy) before any future
   auto-apply attempt.

No commits (per instructions).

---

# SCOPE-GUARDED PARALLEL CALL (2026-07-18, budget 2h stated) — TWO-TIER
AUTO/SUGGEST BAR-MERGE THRESHOLD (tau_auto=0.96), READ THIS FIRST

Ran CONCURRENTLY with the 8th call below (which owns
`chart_interactive.py`/`app_shell.html`/`harmonia_server.py`'s
`serve_chart` — explicitly out of scope for this call, never touched).
Scope: find tau_auto, the similarity threshold above which a bar-merge
candidate can auto-apply with no human tap (existing tau_suggest=0.93
unchanged), and tag the candidate JSON with a `tier` field. Full writeup:
`docs/known_issues.md`, entry titled "Two-tier AUTO/SUGGEST bar-merge
threshold: tau_auto=0.96 found... — 2026-07-18 ★ CHORD-ROBUSTNESS /
BAR-MERGE" (top of file). One-paragraph summary:

1. **User's original ask (French): a threshold with LITERALLY zero false
   positives, ever.** Mid-task the coordinator relaxed this to an
   explicitly-acceptable 1-2% auto-tier error rate — both searches are in
   the known_issues.md entry, in that order, because the strict search's
   own result (a real, non-zero ~0.3% floor even at bar-pair similarity
   EXACTLY 1.0, from feature-representation aliasing between distinct
   chords) is what justifies the relaxation as reasonable rather than
   arbitrary.
2. **Premise check caught a real GT-label bug before trusting anything**:
   reusing the rest of the bar-merge thread's "same GT section" label (from
   merge_criterion.py/clustering_bakeoff.py) at BAR level (not their
   grain=8 blocks) gave ~50% error EVEN AT SIM==1.0 — a single bar's
   harmony doesn't determine section identity (the same one-bar chord
   recurs across verse/chorus/bridge constantly). Corrected GT: label=1
   iff the two bars' majority chord identity (root_pc, qbucket) matches —
   the actually-correct target for a chord-robustness POOLING decision
   (pooling two same-chord bars from different sections is the intended
   behavior, not a false positive).
3. **A real single-split overfitting failure mode found and fixed**: naive
   lowest-tau-meeting-2%-on-one-pool selection chose tau≈0.933 in every
   seed, then blew past the target on that SAME seed's held-out fold in
   2/5 seeds (up to 10.6% observed) — the corpus has a steep, tune-
   heterogeneous error cliff just above tau_suggest (1.92% at tau=0.94,
   9.22% at tau=0.93). Fixed with proper nested train→val-escalation→blind-
   test selection, then a final fixed-threshold cross-check against all 5
   folds' blind test data landed on **tau_auto=0.96**, verified CP-upper-
   95% 0.86%-1.54% across all 5 independently-held-out folds (never above
   ~1.6% in the worst observed fold).
4. **Real-audio tier census, full pairwise scan on the 3 real songs**: auto
   tier is usefully non-empty on all 3 (aretha 197, autumn_leaves 76, abba
   835 pairs at full census; 46/45/128 under the deployed k-NN edge-
   selection algorithm specifically) — not a corner case that rarely fires.
   Notable side-finding: because the shipped UI ranks by similarity and
   caps at 20, **today's top-20 shortlist for all 3 songs is already 100%
   auto-tier** — the suggest tier exists in the data but the current UI cap
   means it's never actually surfaced yet.
5. **Sanity check**: both already-validated real true positives (aretha
   53/69 sim 0.989, abba 206/222 sim 0.994) land in AUTO tier — consistent
   with what's already known about them.
6. **Shipped**: `scratchpad/bar_merge_candidates.py` (`TAU_AUTO=0.96`
   constant, `tier` field per candidate, additive-only to the existing
   `{candidates:[{bars,spans,confidence,n_bars}]}` contract
   `scripts/harmonia_server.py`'s `api_bar_merge_candidates` and
   `/debug/bar-merge-game` read — verified read-only, not touched),
   regenerated all 3 songs' candidate JSON + `bar_merge_game_data.json` via
   the existing `rebuild_bar_merge_game_data.py` driver. New:
   `scratchpad/tau_auto_search.py`, `scratchpad/tau_auto_search_results.json`.
   **No server/UI files touched, no server restart** (nothing server-side
   changed — only static JSON the existing routes already read fresh).
7. **NOT done (explicit next step, blocked on the 8th call's files)**:
   wiring AUTO-tier candidates to actually auto-apply via `/api/reinfer`
   with no human tap — needs `chart_interactive.py`/`app_shell.html`/
   `harmonia_server.py`, all locked this call. Full handoff spec in the
   known_issues.md entry's "NEXT STEP" paragraph.

---

# 8th call (2026-07-18, continuation, budget 2h stated) — BAR-MERGE SUGGESTIONS
OVERLAY PORTED INTO THE SPA + `/chart/<file>` EXEMPTION REMOVED, READ THIS FIRST

Direct continuation of the 7th call's UI thread (see below): user sent a
SECOND screenshot, same "Toujours l'interface dégueu" complaint, still
landing on the old chooser page for `inferred_autumn_leaves.html` — the one
song the 7th call's `/chart/<file>` redirect fix deliberately EXEMPTED
because it alone carried the bar-merge-suggestions overlay (💡 badges,
tap→confirm→pool-and-reinfer) string-patched into its baked static HTML,
with no equivalent in the SPA's own JS-driven chart renderer
(`harmonia/output/app_shell.html`). User's priority stated explicitly: no
ugly page anywhere, even at the cost of temporarily losing the overlay
while it's ported properly.

**Full writeup: `docs/known_issues.md`, entry titled "Bar-merge SUGGESTIONS
overlay ported into the SPA (`app_shell.html`) + `/chart/<file>` exemption
removed... — DONE (live) — 2026-07-18 ★ UI / NAVIGATION"** (appended at the
end of that file, following this repo's append-at-end convention for new
entries). One-paragraph summary:

Read both sides end-to-end first (the baked overlay in `chart_interactive
.py`'s `_TEMPLATE` and the SPA's existing chart renderer/modal patterns in
`app_shell.html`) before writing anything, per the task brief. Ported the
overlay into `app_shell.html` using the SPA's OWN established bottom-sheet
modal pattern (`overlay()`/`sheetBox()`, same one `openMerge`/`openRotor`
use) rather than the baked template's custom popover, for visual
consistency. New state (`S.suggMode`/`suggCandidates`/`suggDismissed`/
`suggConfirmed`), a "💡 Bar suggestions" toggle in Annotate mode (mutually
exclusive with the existing "⋈ Pool two passes" free-select tool), badges
painted onto `S._cells` at the end of `buildIReal()` (mirrors the existing
`paintSpans()` call site), and a confirm flow that REUSES `runFlow()` (the
same preview-only `/api/reinfer/<file>` round-trip the existing pool tool
already uses) rather than a parallel fetch — required one additive change
to `runFlow()` (now returns `true`/`false` for success, previously void;
verified existing callers ignore the return value). Both existing endpoints
(`/api/bar-merge-candidates/<file>`, `/api/reinfer/<file>`) reused verbatim,
not reinvented, per the task brief.

Verified for real, not assumed: `node --check` on the extracted inline
script (PASS, both before and after the server restart), `py_compile` on
the server, data-contract trace by reading both endpoints' Python source
against what the new JS consumes, and full `/api/reinfer` round-trips using
REAL candidate spans fetched from the live API for TWO of the three
candidate-bearing songs (autumn_leaves, abba_chiquitita — both `200,
n_changed:0, rejected:[]`, well-formed non-error previews). Only once that
held did the `serve_chart` exemption get removed: it now redirects
UNCONDITIONALLY (existence check only, no content sniffing), and ~110 lines
of now-fully-dead baked-HTML-serving logic were deleted rather than left as
cruft (docstring says restore from git history if ever needed again).
Post-restart, re-verified the full route sweep tonight's other work
deployed (all 200) plus `/chart/inferred_autumn_leaves.html` now redirecting
like every other song, plus the reinfer round-trip again.

Two honest, explicitly-flagged gaps (no scope creep to close them
silently): (1) no headless browser, so the actual on-device tap→badge→
sheet→confirm interaction is unverified — everything checkable without one
was checked for real. (2) bar-index alignment between the SPA's fold-
collapsed `S.bars` and the baked template's raw unfolded bar numbering is
UNVERIFIED for songs with `reps>1` (folded/repeated sections) — all 3
currently candidate-bearing songs happen to be single-section/`reps:1` so
this never got exercised; if it ever manifests it's cosmetic only (badge on
the wrong visual cell), not a correctness bug, since the actual merge
request always uses real timestamps (`cand.spans`), never a bar index.

Not done, matching the baked version's own already-logged scope cuts:
rejections still page-session-only (not persisted server-side), no
pending-count badge on the toggle button, no live DOM-patching of confirmed
merges into the grid (shows a diff banner instead, same as every other
reinfer flow in this app).

No commits made (per instructions).

# 7th call (2026-07-18, continuation, budget 2.5h stated) — THREE FOLLOW-UPS:
ENSEMBLE (no win), MULTI-MERGE BUG (found+fixed), STRUCTURE RECHECK (worse),
READ THIS FIRST

Picked up the 3 follow-ups the 6th call's brief left open, in priority
order. Full writeup: `docs/known_issues.md`, entry titled "Three follow-ups
(ensemble bakeoff, multi-merge testing, structure recheck) ... — 2026-07-18
★ CHORD-ROBUSTNESS / BAR-MERGE" (top of file, above the 6th call's entry).
One-paragraph summary:

1. **Ensemble/union of k-NN + agglomerative_complete does NOT beat k-NN
   alone** (`scratchpad/ensemble_bakeoff.py`, harness sanity-checked against
   the published knn_solo baseline first). Naive union of two
   independently-safe operating points blows the FPR budget (4/5 seeds); a
   properly re-derived JOINT operating point gets recall 0.220±0.033 vs
   k-NN alone's 0.217±0.031 — a +0.003 non-improvement. Root cause measured
   directly: mean Jaccard overlap of the two algorithms' own true positives
   is 0.84 (0-6.2% of agglomerative's TPs aren't already in k-NN's) — they
   find almost entirely the SAME positives, not complementary ones. No
   ensemble deployed; k-NN alone remains the shipped default.
2. **Multi-merge-per-request testing found and fixed a real bug**:
   independent non-overlapping merge groups in one request work correctly,
   and a single 3-span merge group works correctly (and revealed pooling
   can correctly DECREASE confidence on a genuine outlier, not just
   increase it — a useful non-bug finding). But two merge groups that SHARE
   a bar produced ORDER-DEPENDENT results (same 2 merges, reversed request
   order, different confidences — reproduced on 2 real songs) because
   `pool_beat_evidence` read/wrote one shared mutable array across the
   merge loop, so a later merge silently pooled an earlier merge's already-
   pooled output instead of that beat's original evidence. Fixed by
   snapshotting the original arrays once and having every merge read from
   that snapshot (last-write-wins on genuinely contested beats, matching
   this project's existing confirms convention, but no more compounding).
   Red-first regression test added
   (`tests/test_user_constraints.py::test_pool_beat_evidence_merges_read_original_evidence_not_each_others_output`),
   re-verified on the live production endpoint post-restart on both songs.
3. **Structure V-measure recheck: k-NN candidates make it WORSE, not
   better** (`scratchpad/full_pipeline_eval_knn.py`) — V_F=0.6590±0.0048 vs
   block8's 0.6798±0.0094 (delta -0.0208, a real loss), vs the
   threshold-based pipeline's +0.0050 tie. The chord-robustness k-NN
   operating point (precision-first, tuned for sparse high-confidence
   merge *suggestions*) under-clusters when used directly as a structure
   decoder — do not swap the deployed structure pipeline's clustering knob
   to it.

Deploy: no candidate-generation code changed (items 1 and 3 didn't win), so
`/debug/bar-merge-game` untouched; the `user_constraints.py` bugfix (item 2)
required a server restart (PID 68727), all 8 routes curl-verified 200
after. No commits.

---

# 6th call (2026-07-18, ~2:14-2:45pm CEST region, budget 3h stated, ~30min
used this pass) — MULTI-ALGORITHM BAR-MERGE BAKEOFF, READ THIS FIRST

Picked up the "not attempted this call" item flagged at the bottom of the
5th call's summary below: item 3 of that entry, "multi-algorithm candidate
bakeoff (only pairs+threshold tried)". Full writeup:
`docs/known_issues.md`, entry titled "Multi-algorithm bar-merge
candidate-generation BAKEOFF ... — DEPLOYED — 2026-07-18". One-paragraph
summary:

1. **Ran the full 5-algorithm bakeoff the brief asked for** (threshold+pairs,
   k-NN+connected-components, agglomerative x3 linkages, DBSCAN, spectral+
   eigengap), corpus-scale (900 iReal tunes), 5 seeds, matched FPR<=0.05
   operating points, reusing merge_criterion.py's exact protocol.
   `scratchpad/clustering_bakeoff.py` / `_results.json`. NMF (item 6) not
   attempted — explicitly "if time allows" in the brief, deprioritized for
   the mandatory real-audio check instead.
2. **k-NN(k=1, floor~0.9) is a real but MODEST winner** (recall
   0.217±0.031 vs 0.187-0.192 for everything else at ~the same precision/
   FPR) — a ~13% relative recall gain, not a knockout. **Spectral+eigengap
   is a clear loser** for the low-FPR priority (FPR floor ~0.34-0.39, can't
   be tuned lower — eigengap on tiny per-tune graphs is too noisy).
   Threshold/agglomerative/DBSCAN are all numerically near-identical in
   this small-block-count regime (expected, not a bug). DBSCAN's
   noise-rejection feature never activated (min_samples=1 always won on
   val) — a specific, useful negative result if DBSCAN gets re-proposed
   for that reason later.
3. **Real-audio check reproduced the ALREADY-DOCUMENTED over-merge
   collapse quantitatively** (threshold+full-transitive-closure at
   tau=0.93 gives 71/117/183-bar single components on aretha/autumn_leaves/
   abba) and found k-NN's connected-components groups, while far more
   collapse-resistant (7/21/14-bar largest components), still don't fit
   this project's pairs-only UI format.
4. **Shipped a deliberately narrower change**: k-NN's per-bar
   top-1-above-floor edge SELECTION rule (the actual source of its
   corpus-validated edge), WITHOUT the connected-components closure —
   preserves the exact pairs-only JSON format, avoids the collapse risk of
   shipping real transitive groups. `scratchpad/bar_merge_candidates.py`
   now defaults to `algo="knn"` (old behavior kept as `algo="threshold"`).
   Regenerated candidates for all 3 real songs (still 20/song, same
   schema) via new `scratchpad/rebuild_bar_merge_game_data.py` (no such
   driver script existed before — the prior call built the JSON ad-hoc).
5. **Deployed, no restart needed** (`/debug/bar-merge-game` reads its data
   file fresh per request) — curl-verified 200 + content-checked
   (`meta.algo=="knn"` present in the served page's embedded DATA) plus
   200 on all other `/debug/*` routes, `/`, `/library`. Re-ran the one
   concretely real-audio-validated pair from the 5th call (aretha bars
   53/69) through the live `/api/reinfer` post-deploy — still ranked #1
   candidate, same confidence, `rejected: []`, pooled re-decode still
   works. The new default did not regress the prior call's validated
   result.
6. **Still open**: NMF arm, multi-bar-group UI support, GT-scored
   real-audio error analysis (blocked on the well-established fact — see
   below — that no real-audio section/bar GT exists anywhere in this
   repo), multi-merge-per-request testing.

Budget note: this pass used well under the stated 3h; stopped here because
the bakeoff + real-audio check + deploy + this log entry is a complete,
self-contained unit of work and a natural checkpoint, not because the
budget ran out. A fresh continuation call could pick up any of the "still
open" items above, or move to the secondary structure-detection
V-measure question per the standing task brief's stated priority order.

---

# AFTERNOON UPDATE (2026-07-18, 5th call) — reframes everything below (see
new 6th-call section above for the latest pickup on this same thread)

**The user reframed the whole thread's goal: structure detection (V-measure,
everything the "MORNING SUMMARY" below is about) is now SECONDARY. The
primary goal became CHORD-RECOGNITION ROBUSTNESS** — using the same 1-bar
SSM this thread built to find harmonically-identical bar pairs, then POOL
their per-beat evidence (`pool_beat_evidence`, already implemented) to
denoise per-bar chord predictions, "stack frames to denoise" style. Full
writeup: `docs/known_issues.md` "★ CHORD-ROBUSTNESS / BAR-MERGE" section
(top of file). One-paragraph summary:

1. **Premise check (mandatory, done first): pooling helps under any
   realistic per-bar noise model and is never harmful even in the
   adversarial worst case** (pure systematic bias, no randomness — delta
   exactly 0.0000, not negative; any random component present gives a real,
   growing benefit, delta +0.006 to +0.12 across sigma 0.05-0.40).
2. **Real deployment blocker found and fixed**: `/api/reinfer` always
   preferred the Billboard backend (checkpoint present in prod), which has
   NO pooling support and silently rejected every `merges` request — so
   `pool_beat_evidence`, despite being fully implemented, was UNREACHABLE
   from the endpoint for any real-audio song before this call. Fixed:
   merges now route around Billboard to the pooling-capable
   `infer_chords_v1` backend.
3. **Built and deployed `/debug/bar-merge-game`** — a new, separate
   Candy-Crash-style tap-to-confirm UI (did NOT touch `chart_interactive.py`
   per instruction), wired to a candidate generator
   (`scratchpad/bar_merge_candidates.py`, pairs not transitive groups — see
   known_issues.md for why) and the fixed `/api/reinfer`.
4. **Validated end-to-end on real production code path, 2 real songs**:
   pooling took two bars that DISAGREED despite the SSM scoring them >0.98
   similar and reconciled them to the same, higher-confidence label
   (aretha: E:maj7→D#:maj7 conf 0.63→0.65; abba: A:7→A:maj7 conf 0.48→0.53).
   Effect is properly localized (1 real label change per merge, verified by
   TIME-matched comparison — an earlier position-matched comparison wrongly
   suggested a 49-chord cascade; that was own-script bug, corrected before
   logging, see known_issues.md for the full story).
5. **Not done this call** (open for next pickup): multi-algorithm candidate
   bakeoff (only pairs+threshold tried), multi-merge-per-request testing,
   secondary structure V-measure re-check.

---

# MORNING SUMMARY (rewritten after 4th continuation/consolidation call,
2026-07-18) — structure-detection findings below now SECONDARY per the
reframe above, still accurate for what they cover

## TL;DR — the answer to "what's the best approach"

**On iReal (where ground truth exists): the full recommended pipeline
(intro-trim + FPR=0.10 bar-merge criterion) statistically TIES flat block8
— V_F=0.6847±0.0078 vs 0.6798±0.0094 (5 seeds, 1989 tunes, delta +0.005).**
Not a win. This matches the pattern that recurred all night across every
structure-detection variant tried (learned encoders, grammar induction,
hierarchical extensions, chord-only similarity thresholds): none of them
reliably beat flat 8-bar block matching on chord-only input. That ceiling
looks structural, not a tuning gap — see "what's still open" below for what
kind of signal would be needed to actually clear it.

**On real audio (no GT, 3 songs): the deployed pipeline is (adaptive
per-song threshold) + (recursive local re-split), NOT the FPR=0.10
threshold above.** This call explicitly tested whether the night's
best-tuned iReal threshold (FPR=0.10, tau=0.7759) would reduce the need for
the real-audio-specific adaptive-percentile patch — **it does not; if
anything it makes the patch more necessary**, because 0.7759 sits almost
exactly at the already-known over-merge collapse point (aretha collapses to
1 section, autumn_leaves reaches only 4). Real audio's off-diagonal
similarity floor (0.67-0.89, song-dependent) sits structurally above the
range where any single fixed iReal-tuned threshold works, no matter how
well that threshold is chosen ON iReal. Deployed at
`http://100.89.209.63:7771/debug/merge-criterion`, row (d).

## What's deployed right now

- `/debug/merge-criterion`: 5 rows per song — (a) V-measure-optimal tau
  [failure-case context], (b) old low-FP tau=0.05 [failure-case context],
  (e) NEW FPR=0.10 tau=0.7759 [context: shows it fails the same way as (a)],
  (c) adaptive per-song percentile, **(d) adaptive + recursive local
  re-split — the actual recommendation**. A green summary box at the top of
  the page states both the iReal-tie number and the real-audio patch
  status in one place. curl-verified 200 + content-checked on all 7 routes
  this call.
- The corpus-scale iReal number (full pipeline vs block8, +0.005 tie) is
  NOT deployed anywhere audio-facing — it's a symbolic/iReal-only eval
  result, reported here and in known_issues.md as the honest answer to
  "does this approach actually beat the baseline."

## What's still genuinely open / unsolved (stated plainly, not papered over)

1. **The chord-only-similarity structure-detection ceiling is real and
   looks structural, not a threshold/algorithm tuning gap.** Every variant
   tried across the whole night (this call's full pipeline, the section
   detector alone, the learned encoder from two calls ago, grammar
   induction, hierarchical block8 extensions) lands in the same 0.68-0.70
   V_F band on iReal. Nothing has beaten it decisively and reproducibly
   across multi-seed checks. If someone wants to actually clear this
   ceiling, chord/root-only input is probably the wrong lever to keep
   pulling — needs a different signal class entirely (melody contour,
   lyric-line boundaries, or a learned audio-native embedding), not another
   similarity metric or clustering rule on the same chord features.
2. **Autumn Leaves' 80-bar residual run (bars 136-216) is now DECISIVELY
   characterized as unsolvable at this feature resolution, not just
   unsolved.** This call ran two targeted checks (both negative,
   documented in known_issues.md): (a) sweeping the local re-split
   threshold from P5-P95 recovers no clean structure below P80, and even
   then only peels off 1-2 outlier blocks; (b) checking specifically for
   32-bar (4-block) lag periodicity — since this span is ~2.5 choruses of a
   32-bar standard — found NO periodicity bump at all (flat 0.76-0.80
   across every lag). The chord/root features in this span carry no
   detectable chorus-repeat signal, full stop. Root cause: Autumn Leaves'
   chorus repeats plausibly have near-identical harmony by construction (a
   32-bar standard's whole point); what differentiates choruses lives in
   melody/lyrics, not chords. **Do not re-attempt this with more
   threshold or clustering variants** — it needs a non-harmonic signal this
   pipeline doesn't have.
3. **Real-audio adaptive-percentile + recursive-resplit is still validated
   only qualitatively (no GT exists for real audio in this repo).** It's
   the best available real-audio pipeline by inspection, not a scored
   result. If real-audio section GT ever becomes available (e.g. manual
   annotation of the 3 songs), this is the first thing to actually score.
4. **The FPR-gate frontier (target_fpr knob) and the real-audio adaptive
   threshold are now confirmed to be solving genuinely different problems**
   (frontier = aggregate iReal bar-pair precision/recall tradeoff;
   adaptive = per-song floor variance on real audio) — this is now a closed
   question (tested explicitly this call), not open, but worth remembering
   so nobody re-proposes "just use the better iReal threshold" for real
   audio again without re-reading this.

## Next steps if someone picks this up fresh

1. If chasing the iReal V-measure ceiling further: try a genuinely
   different input signal (not another chord-similarity variant) — the
   whole night's evidence says chord-only inputs are saturated around
   0.68-0.70 regardless of algorithm.
2. If real-audio structure quality actually matters for a user-facing
   feature: get even minimal manual section-boundary annotations on the 3
   real songs so the adaptive-percentile pipeline can finally be scored,
   not just eyeballed.
3. Don't re-open: floor-blend noise calibration for merge-criterion
   training (self-cancelling, closed), FPR=0.10-style fixed thresholds for
   real audio (structurally can't work, closed this call), or more
   threshold tuning on Autumn Leaves' 80-bar span (decisively ruled out
   this call).

---

# Call 3 (2026-07-18 ~04:21-05:35 CEST) — 3rd continuation call

Executed the 3 concrete follow-ups Call 2's summary proposed, plus the
mandatory error-analysis loop on whatever ended up deployed.

## Call 3 — what was tried, in order

1. **Follow-up 1 (does floor-blend-CALIBRATED iReal training help real-audio
   transfer?): NO — negative, root-caused.** Trained Step 2's merge
   threshold on alpha=0.40 floor-blended iReal (train+val blended, test
   clean — the proper design, unlike Call 2's same-call "sanity pass"
   which blended everything together). The blend-trained threshold (0.988)
   came out HIGHER/more conservative than the clean-trained one (0.973),
   and still failed to rescue autumn_leaves/aretha on real audio, and made
   abba_chiquitita marginally worse. Mechanism: floor-blending raises
   negative-pair similarity in lockstep with positive-pair similarity in
   TRAINING data, so the FPR-gated threshold rises to compensate —
   self-cancelling. This is evidence FOR keeping the adaptive-percentile
   fix (a single global alpha can't encode real audio's PER-SONG-varying
   floor: aretha 0.891 vs abba 0.672). `scratchpad/blend_transfer_test.py`.
2. **Follow-up 2 (map the full FPR-gate frontier): DONE — found an interior
   optimum, not a monotonic tradeoff.** Swept target_fpr in
   {0.02,0.05,0.10,0.15,0.20,0.30}, 5 seeds, corpus-scale (1989 tunes).
   V_F peaks at target_fpr=0.10 (0.6851 ± 0.0151), matching the
   separately-tuned "V-measure-optimal" tau=0.78 result (0.682) within
   noise — useful cross-check the two protocols agree. Currently-deployed
   low-FP point (0.05) costs 0.037 V_F vs this optimum for higher bar-pair
   precision (0.687 vs 0.627) — now an actual quantified knob, not a
   binary choice. `scratchpad/fpr_frontier_sweep.py`.
3. **Follow-up 3 (does the adaptive-percentile fix hurt iReal, where GT
   exists?): YES, clearly — confirms it's a real-audio-specific patch, not
   a generally-better strategy.** Swept percentile 50-98 on iReal test
   songs (5 seeds, corpus-scale): best case (P=98) V_F=0.603, ~0.08 below
   the best fixed-tau operating point (0.685), plus a residual 4.2%
   degenerate rate that never disappears. As hypothesized: iReal's own
   similarity floor is lower and more uniform across songs than real
   audio's, so a per-song relative threshold is a much noisier, worse idea
   there. Verdict stated plainly: keep this fix SCOPED to real audio only.
   `scratchpad/adaptive_percentile_on_ireal.py`.
4. **Mandatory error-analysis loop on the deployed adaptive-percentile fix
   (row c): genuine PARTIAL win, deployed.** Inspected row (c)'s actual
   failure cases directly: 2 of 3 songs have very long (48-80 bar)
   single-cluster runs that plausibly hide internal chorus-repeat
   structure (autumn_leaves is a known 32-bar jazz-standard form; an
   80-bar run = ~2.5 choruses collapsed with no internal boundary
   recovered). Fix tried: recursive LOCAL re-split — any run >=32 bars
   gets its own local (within-run-only) P75 threshold and is re-clustered
   independently. Result: autumn_leaves 12→16 sections, abba_chiquitita
   10→14 sections, no new degeneracy anywhere — but one 80-bar run on
   autumn_leaves still resists even local splitting (characterized as a
   genuine remainder needing a non-harmonic signal, not another threshold
   tweak — logged per rule #4, don't re-attempt with more threshold
   tuning). `scratchpad/error_analysis_recursive_split.py`.

## What's deployed right now (after Call 3)

**`http://100.89.209.63:7771/debug/merge-criterion`** now shows 4 rows per
song (was 3 after Call 2): (a) V-measure-optimal ceiling [failure-case
context], (b) Step 2 low-FP [failure-case context], (c) Step 6 adaptive
percentile, (d) NEW — (c) + recursive local re-split (RECOMMENDED, the
best available row). Static-file rebuild only
(`scratchpad/build_real_transfer_viz.py` now also reads
`error_analysis_recursive_split_results.json`) — **no server restart was
needed**, the existing Flask route already serves the file fresh off disk.
curl-verified 200 on ALL 7 routes (`/`, `/library`, `/debug/ssm-multigrain`,
`/debug/ssm`, `/debug/structure`, `/debug/metric-artifact`,
`/debug/merge-criterion`) + content-checked (13 `level-label` divs found =
4 rows × 3 songs + 1 CSS selector, matches per-song terminal output
exactly).

## What's still open / next-step recommendation (updated after Call 3)

1. **The autumn_leaves 80-bar residual run (bar 136-216) is now a
   characterized, not-solved remainder** — local re-split at P75 didn't
   touch it either, meaning that span is genuinely harmonically flat at
   this similarity resolution. Recovering it needs a non-chord signal
   (melody contour, lyric/vocal onset timing — none available in this
   pipeline currently) or a fundamentally different similarity feature,
   not another threshold variant. Don't re-attempt with threshold tuning.
2. **Follow-up 2's frontier (target_fpr=0.10 optimum) has NOT been
   propagated to the real-audio adaptive-percentile deployment** — Steps
   5/6 and this call's error-analysis loop all still use Step 2's original
   FPR=0.05-derived threshold as the "clean-trained" reference point where
   one is needed. Someone could re-run Step 5/6's real-audio transfer using
   the FPR=0.10 threshold as the new "clean-trained" baseline before
   applying the adaptive-percentile correction on top — untested, since
   the adaptive fix operates on each song's OWN distribution and doesn't
   actually consume the FPR=0.05 threshold directly (only Follow-up 1's
   comparison did), so this is lower priority than it sounds.
3. **All 3 follow-ups from Call 2's summary are now closed** — two negative
   (Follow-up 1: floor-blend training doesn't help; Follow-up 3: adaptive-
   percentile shouldn't generalize to iReal), one positive deliverable
   (Follow-up 2: the frontier itself). No further follow-up work was
   implied by the brief beyond the mandatory error-analysis loop, which is
   also done. Remaining budget (~45 min of Call 3's 2h) not required to hit
   any stated target — stopping here is a legitimate outcome per doctrine,
   not premature.
4. **No commits made** (per process convention). New files this call:
   `scratchpad/blend_transfer_test.py`, `scratchpad/blend_transfer_test_results.json`,
   `scratchpad/fpr_frontier_sweep.py`, `scratchpad/fpr_frontier_sweep_results.json`,
   `scratchpad/adaptive_percentile_on_ireal.py`,
   `scratchpad/adaptive_percentile_on_ireal_results.json`,
   `scratchpad/error_analysis_recursive_split.py`,
   `scratchpad/error_analysis_recursive_split_results.json`. Modified:
   `scratchpad/build_real_transfer_viz.py` (adds row (d), reads the new
   JSON), `scratchpad/real_transfer_viz.html` (regenerated). `scripts/
   harmonia_server.py` NOT touched this call (no route change needed).
   All findings logged to `docs/known_issues.md` (search "Follow-up 1",
   "Follow-up 2", "Follow-up 3", "Mandatory error-analysis loop", all
   dated 2026-07-18, ★ STRUCTURE / SEGMENTATION).

---

# Call 2 (continuation call, 2026-07-18 ~04:00-04:20 CEST)

Picked up exactly where the prior call's "NEW SESSION SECTION" (bottom of
this file) left off: Step 1's original additive-noise premise had just been
falsified, and Steps 2-6 of the 6-part brief were unstarted. This call
finished all 6 steps inside ~25 minutes of a 2h budget (see "why so fast"
notes throughout this file for the recurring pattern — reusing
already-validated infra from prior calls, not re-deriving it, is most of
the speed).

## What was tried, in order

1. **Step 1 RETRY (floor-blend noise model)**: additive Gaussian noise
   (previous call) could only ever LOWER the off-diagonal-similarity floor
   statistic (`stat_B`/`mean_p90`), moving away from real audio's elevated
   target (0.832 combined). Tried the recommended fix instead: blend each
   bar's vector toward a shared corpus-mean "generic chord" vector
   (`v' = (1-alpha)*v + alpha*generic`). **Converges cleanly**: alpha=0.40
   combined-register gives stat_B=0.8365 vs target 0.8322 (gap 0.0043),
   and — importantly — barely degrades true label-discriminability (stat_A
   AUC 0.779→0.776 across alpha 0→0.6), unlike additive noise which
   necessarily degrades both together. This is a genuine positive result,
   the first real win on the noise-calibration question across all of
   tonight's calls. `scratchpad/noise_calibrate_floor.py`.
2. **Step 2 (mandatory, learned bar-merge criterion on clean iReal)**:
   5-d feature logistic regression vs a simple threshold on combined
   bass+treble similarity, both FPR-gated at the user's stated priority
   (target_fpr=0.05), 900-tune corpus, 5 seeds. **Honest negative-ish
   result**: logreg does NOT beat the simple threshold (recall 0.126 vs
   0.187 at matched ~5% FPR) — the extra features (block distance, size
   ratio) don't carry independent signal beyond the similarity score
   itself. **Recommendation: deploy threshold-only.** Also: recall at the
   low-FP operating point is inherently low (13-19%) — stated plainly as
   the real cost of the user's own precision-first priority, not hidden
   behind a precision-only report. `scratchpad/merge_criterion.py`.
3. **Step 3 (intro detector)**: initial hypothesis ("intro = low
   similarity to rest of song") was checked with a direct AUC test BEFORE
   building anything on it — AUC=0.388, i.e. the OPPOSITE of what was
   assumed. Root-caused (iReal intros are short static vamps, spuriously
   similar to lots of unrelated material) and flipped. Real, positive,
   corpus-validated result after the flip: 2-bar edge, FPR=0.05 →
   precision 0.553 vs base rate 0.220 (2.5x lift). `scratchpad/intro_outro.py`.
4. **Step 4 (section detector on Step 2's criterion)**: full 1989-tune
   corpus. V-measure-optimal tau ties flat block8 (0.6815, matches the
   established 0.68-0.70 range). Actually deploying Step 2's low-FP tau
   instead costs 0.044 V_F (0.638) — the first time this specific tradeoff
   was quantified rather than just asserted. `scratchpad/section_detector.py`.
5. **Step 5 (real-audio transfer, no GT) + Step 6 (error analysis,
   mandatory iteration loop)**: transferring BOTH of Step 4's fixed iReal-
   calibrated tau values to the 3 real songs produces the cleanest failure
   demo of the whole night: tau=0.78 collapses aretha_chain_of_fools to 1
   section (over-merge); tau=0.973 leaves autumn_leaves at 41/41 sections
   (zero merges). Diagnosed cause: real audio's similarity floor is not
   just elevated, it's elevated by a SONG-DEPENDENT amount (offdiag mean
   0.891 for aretha vs 0.672 for abba — no fixed constant can straddle
   both). **Fix**: per-song adaptive threshold = the 90th percentile of
   that song's own off-diagonal similarity distribution. Non-degenerate on
   all 3 songs (5/12/10 sections vs collapse-to-1 or 41/41). Honest caveat:
   doesn't exactly match the earlier (also-unvalidated) learned-encoder
   reference counts, logged as a real compromise, not a perfect fix.
   `scratchpad/real_transfer.py`.

## What's deployed right now

**`http://100.89.209.63:7771/debug/merge-criterion`** — NEW this call.
Shows all 3 real songs, 3 operating points each (V-measure-optimal /
low-FP / adaptive-fix), docked audio, intro badges. curl-verified 200
(new PID 56820) alongside every pre-existing route (`/`, `/library`,
`/debug/structure`, `/debug/ssm`, `/debug/ssm-multigrain`,
`/debug/metric-artifact`) — nothing else broken. Response body
content-checked, not just status code.

**Recommendation: look at row (c) "adaptive percentile" on the new debug
page first** — rows (a)/(b) are shown as failure-case context, not the
deployable answer.

## What's still open / next-step recommendation

1. **The adaptive-percentile fix (Step 6) is itself unvalidated beyond
   "non-degenerate + roughly plausible vs an unvalidated reference."** If
   someone wants to push this further: the natural next move is exactly
   what the earlier "Call 3" section of this file already flagged as
   highest-EV and never got to — sweep the percentile per-song using
   SOME proxy signal that doesn't require GT (e.g. a stability criterion:
   pick the percentile that makes the resulting cluster COUNT most stable
   under a small perturbation of the similarity matrix, or under the
   Step 1 floor-blend noise transform applied at a few alpha levels).
2. **Step 1's floor-blend recipe (alpha=0.40) was validated on the
   STATISTIC match only** — nobody has yet actually retrained Step 2's
   merge criterion on floor-blended iReal data and checked whether IT
   (not just the raw similarity statistic) transfers better to real audio
   than the clean-iReal-trained version. The cheap sanity pass done this
   call (blending train+val+test together) was NOT that check — a proper
   train-clean/eval-blended (or train-blended/eval-real, impossible
   without GT) comparison is still open.
3. **Logreg's failure to beat a simple threshold (Step 2) suggests the
   5-d feature vector is under-powered, not that no learned model could
   help** — a feature that actually captures something the raw similarity
   doesn't (e.g. a same-key/different-key indicator, or a local chord-
   density feature) might do better than block distance / size ratio.
   Untested this call.
4. **No commits made** (per process convention — a separate coordinator
   handles git). New/modified files this call: `scratchpad/
   noise_calibrate_floor.py`, `scratchpad/noise_calibrate_floor_results.json`,
   `scratchpad/merge_criterion.py`, `scratchpad/merge_criterion_results.json`,
   `scratchpad/intro_outro.py`, `scratchpad/section_detector.py`,
   `scratchpad/section_detector_results.json`, `scratchpad/real_transfer.py`,
   `scratchpad/real_transfer_results.json`,
   `scratchpad/build_real_transfer_viz.py`, `scratchpad/real_transfer_viz.html`.
   Modified: `scripts/harmonia_server.py` (`/debug/merge-criterion` route
   only, additive). All findings also logged to `docs/known_issues.md`
   ("Step 1 RETRY", "Step 2", "Step 3", "Step 4", "Step 5+6", all dated
   2026-07-18, ★ STRUCTURE / SEGMENTATION — read in that order for full
   detail, this summary is not a substitute).

---

# MILESTONE — Stage B real-audio structure checkpoint DEPLOYED and curl-verified

_Status: DONE for this call (Call 1 of 2). Landed well inside the 2h budget
(~15 min of a 2h window) — see "why so fast" note at the end of this section._

- **Deployed**: yes, new debug route `/debug/structure` added to
  `scripts/harmonia_server.py` (server restarted, PID 48947, existing routes
  `/`, `/chart/<file>`, `/library` all re-verified 200 after restart — no
  existing chart-serving path touched).
- **URL**: `http://100.89.209.63:7771/debug/structure` — curl-verified 200,
  content confirmed to contain all 3 songs' section timelines and working
  `/audio/<file>` playback links (also curl-verified 200).
- **Songs**: `autumn_leaves`, `abba_chiquitita`, `aretha_chain_of_fools` (all
  3 had audio on disk in `docs/audio/`; `anthropology.m4a` was NOT present,
  skipped rather than substituted silently).
- **Stage A finding (root-only vs full-chord, 2-bar nuclear, keynorm)**:
  root-only loses real ground on the raw pairwise frontier (recall 0.295 vs
  0.356 @ equal precision) but is a STATISTICAL TIE with full-chord at the
  actually-deployable 8-bar union scale (V_F 0.698 vs 0.692, single seed) —
  good news for Stage B, since real audio's root posterior is much more
  trustworthy than its quality posterior.
- **Stage B3a finding (synthetic noise robustness)**: the probabilistic-
  input model beats a hard-label baseline fed the SAME noise across the
  whole realistic range (root-confusion rate 0-0.17, calibrated to
  music-x-lab-level accuracy per `symstruct_robust.py`) — e.g. V_F 0.606 vs
  0.588 at the realistic 0.17 point — and degrades to parity (not below)
  the hard baseline only at implausibly high noise (>=0.30). Approach
  validated cheaply before real-audio wiring, as required.
- **Stage B3b**: real-audio structure segmentation is QUALITATIVE ONLY (no
  section GT exists for real audio in this repo, confirmed rather than
  assumed) — all 3 songs produced non-degenerate segmentations (not one
  giant cluster, not all singletons); see the Stage B3b section below for
  the per-song read. The user should judge this by looking/listening at the
  URL above, not by a number.
- **Not done in this call (by design, deferred to Call 2 per the brief)**:
  hierarchical/multi-level clustering refinement, grid-phase-misalignment
  fix, multi-seed validation of Stage A/B headline numbers, real key
  detection to replace the `estimate_tonic_pc` heuristic.
- **Why so fast**: most of the wall-clock cost in a task like this is
  usually model training, but every training run here is CPU, ~10-25s
  (small BiLSTM, iReal corpus is small/pre-cached) — the actual bottleneck
  budget-wise was reading/verifying, not compute. Time budget was NOT the
  binding constraint this call; correctness/verification was prioritized
  instead (every claim above is curl- or file-verified, not asserted).

---

# Session log — structure detection on real audio (Call 1 of 2)

Start: 2026-07-17 20:04 CEST. Budget: 2h (stop ~22:04 CEST).

## Context recap (not re-deriving, see docs/handoff_2026_07_18_structure_detection.md)

- Deployable baseline: learned key-norm union at fixed 8-bar blocks, V_F≈0.70
  vs flat block8 0.68-0.695. 2-bar nuclear now the mandated default per user
  correction (2-bar beats hard-matching by the widest P/R margin once merged
  via learned similarity; old "4-bar beats 2-bar" was a hard-match artifact).
- Grid phase misalignment is the dominant remaining per-bar V_F loss source
  (oracle phase correction: 0.679->0.738, 40% of songs have nonzero optimal
  phase) — not addressed in this call, that's tracked separately.
- Nothing yet validated on real predicted chords from real audio — that's
  this call's actual goal (Stage B).
- Server confirmed up before starting: `lsof -i :7771` -> PID 46782 LISTEN.

## Plan for this call

1. Stage A: root-only vs root+quality token, learned key-norm encoder,
   2-bar nuclear, clean iReal/V-measure.
2. Stage B: adapt BlockEncoder to take a 13-dim root-probability vector
   instead of a discrete token; wire to chord_pipeline_v1.py's root_proba.
3. Stage B3a: synthetic noise stress test (reuse symstruct_robust.py's
   corruption approach) before touching real audio.
4. Stage B3b: qualitative real-audio structure segmentation on 2-3 known-good
   songs.
5. Deploy debug route, curl-verify.

Working log starts below.

## Stage A — root-only vs root+quality token, learned key-norm encoder

Ran `scratchpad/symstruct_learned.py` (now supports `--rootonly`, added this
session — `qual_mode="none"` drops the quality embedding entirely, root pc
0-11 or NC token only). Rule #1 reproduction check: 2-bar full-chord pairwise
frontier reproduced the exact number already in known_issues.md ("2-bar
nuclear now BEATS hard-matching by the widest margin, recall 0.356 vs hard
0.245 @ equal precision") — baseline confirmed live before trusting anything
new.

### Raw block-pair P/R frontier (2-bar nuclear, keynorm, test, seed0)
| variant | HARD P/R | LEARNED recall @P>=hard.P | LEARNED precision @R>=hard.R |
|---|---|---|---|
| full chord (root+qual) | 0.532/0.245 | **0.356** | 0.559 |
| root-only | 0.532/0.245 | **0.295** | 0.549 |

Root-only loses a real, non-trivial chunk on the raw pairwise frontier at
fine (2-bar) granularity (recall 0.356->0.295 @ equal precision) — dropping
quality does cost something when blocks are this short (2 bars = little
context to disambiguate by root pattern alone).

### Downstream V-measure at the actually-deployable scale (8-bar union, keynorm, seed0)
| variant | TEST V_F |
|---|---|
| full chord (root+qual) | 0.692 |
| **root-only** | **0.698** |
| flat block8 (ref) | 0.695 |

**Finding: at the deployable 8-bar union scale, root-only is statistically
indistinguishable from full-chord (both ~ties block8, single-seed, +/-0.006
apart — within noise).** The pairwise-frontier loss from dropping quality
does not survive into the downstream clustering metric at 8-bar granularity
— hypothesis: root pc is already doing most of the section-discrimination
work; quality mostly disambiguates ties that get washed out once the union
threshold is tuned per-representation on val. This is a single-seed result
(not yet 5-seed validated per project convention) but directly answers the
Stage A brief's question: **root-only is NOT a meaningful sacrifice at
deployment granularity** — good news for Stage B, since real audio's root
posterior is far more trustworthy than its quality posterior.

Caveat: not multi-seed validated (time-boxed within a 2h budget shared with
Stage B); if this becomes a headline claim later it needs the same 5-seed
protocol as the original key-norm result.

## Stage B1 — per-bar root softmax from real audio, end to end

`scratchpad/real_root_proba.py`. Reuses `nnls_features.get_heads()`,
`extract_bothchroma`, `pool_beats` and `heads.root_proba(feat)` (the EXACT
function chord_pipeline_v1._infer_nnls24 calls internally at its own line
~2501) — no re-derivation of the model. Beat grid duplicated from
`infer_chords_v1`'s own librosa beat_track + circular-mean phase-correction
logic (not exposed as a return value from the pipeline function, so this is
a copy of ~6 lines of grid arithmetic, not new model logic). Bars = fixed
groups of 4 beats from beat 0 (inherits the known GRID PHASE MISALIGNMENT
limitation as-is — out of scope for this call).

**Premise check before trusting the extraction**: ran it on
`docs/audio/autumn_leaves.m4a` and compared `n_bars` (330) against the
ALREADY-DEPLOYED chart's embedded `nBars` value in
`docs/plots/inferred_autumn_leaves.html` (329) — matches within rounding.
Confirms this script's beat/bar grid reproduces what's already live for this
song before trusting anything built on top of it (CLAUDE.md rule #1).

Note: `infer_chords_v1` itself can't load `.m4a` directly (`sf.read` doesn't
support AAC/ALAC; only `librosa.load`'s audioread fallback does, which this
script uses) — a pre-existing loader gap unrelated to this work, not fixed
here (production presumably gets around it via yt-dlp's raw download format,
not investigated further).

Per-bar output looks sane on inspection: entropy 0.8-1.95 nats/bar (vs log(12)
=2.48 max), i.e. genuinely peaked-but-uncertain distributions, not near-
uniform (which would indicate a broken extraction, error pattern #1 style).

## Stage B2 — BlockEncoder adapted to soft probability input

`scratchpad/symstruct_proba.py`. `BlockEncoder` (symstruct_learned.py) now
takes `root_mode="proba"` (13-d softmax -> `nn.Linear(13,d_tok)`, replacing
the `nn.Embedding` lookup) and `qual_mode="none"` (root-only, per Stage A).
Trained on the iReal corpus by representing clean chord labels as one-hot
13-d vectors (a valid point in the same probability simplex a real softmax
lives in) — no natural probability-labeled iReal data exists, so this is the
correct fallback per the brief.

**Sanity check**: CLEAN downstream V_F for the proba-input model =
**0.698** (tau=0.85, size=8, keynorm) vs flat block8 **0.695** — matches
Stage A's discrete root-only-token result (0.698) almost exactly, confirming
the soft-projection reparameterization doesn't lose information relative to
the discrete embedding when the input actually IS one-hot. Good — the
architecture swap itself is not the source of any degradation.

## Stage B3a — synthetic noise stress test (the required cheap validation before real audio)

`corrupt_proba()`: corrupts one-hot root vectors into smoothed distributions
— confidence drawn per bar in [0.35,0.85] (never a hard 0/1, matching the
entropy range measured on real `autumn_leaves.m4a` audio in Stage B1) plus a
`p_wrong` chance of re-centering on a music-theoretically confusable root
(5th/4th/relative-minor/3rd — same confusion flavor as `symstruct_robust.py`'s
existing hard-corruption baseline). Compared the proba-input learned model
against a HARD-LABEL baseline (argmax the SAME corrupted vector -> discrete
flat block-union, i.e. what the OLD discrete-token pipeline would do fed the
same noisy signal), TEST set, size=8:

| p_wrong (root confusion rate) | proba-model V_F | hard-argmax-union V_F |
|---|---|---|
| 0.00 (confidence noise only) | **0.694** | 0.643 |
| 0.10 | **0.625** | 0.590 |
| 0.17 (music-x-lab-level, per symstruct_robust.py calibration) | **0.606** | 0.588 |
| 0.30 | 0.581 | 0.579 |
| 0.45 | 0.580 | 0.579 |

**Confirms the required premise**: the probabilistic-input model degrades
gracefully and clearly beats the hard-label baseline across the realistic
noise range (0-0.17 root-confusion rate, matching real music-x-lab-level
accuracy) — margin +0.05 at clean/low noise, +0.018 at the realistic 0.17
point. The margin collapses to ~0 only at implausibly high noise (>=0.30
wrong-root rate), where both methods bottom out near the same floor — i.e.
the soft-probability approach doesn't help once there's no real signal left
to be soft about, which is the expected/sane failure mode, not a red flag.
**Approach validated cheaply before spending budget on real audio wiring —
proceeding to Stage B3b.**

## Stage B3b — real audio, end to end, QUALITATIVE checkpoint (no GT, no V_F)

`scratchpad/run_real_structure.py`: Stage B1 extraction -> Stage B2 model
(`scratchpad/keynorm_proba_rootonly_s8.pt`, trained clean, size=8, root-only,
proba input) -> union-find clustering directly on real per-bar softmax, on
3 songs with existing rendered charts (`docs/plots/inferred_autumn_leaves
.html`, `inferred_abba_chiquitita_official_lyric_video.html`,
`inferred_aretha_franklin_chain_of_fools_official_lyric_video.html`) —
picked from the prompt's candidate list, all three have audio on disk
(`docs/audio/`); `anthropology.m4a` did not (checked, not present, skipped).

Confirmed via `docs/known_issues.md`/session docs: **no real-audio corpus in
this repo has section-structure ground truth at scale** — iReal has GT
sections but no audio; real audio here has no section GT. This is why
Stage B3b is explicitly qualitative, per the task brief — no V_F reported,
no fake metric manufactured.

Key-normalization on real audio has no iReal `key` field to read, so used a
CHEAP HEURISTIC proxy (`estimate_tonic_pc`: the pc with the most total
root-probability mass across the song) rather than skipping keynorm
entirely — flagged explicitly as a heuristic, NOT a validated key detector.

Results (full run-length segmentation in `scratchpad/real_structure_results
.json`):

| song | tempo | n_bars | n_sections found | qualitative read |
|---|---|---|---|---|
| autumn_leaves | 187.5 bpm | 330 | 18 | S0 recurs 11x — plausible A-section recognition on a 32-bar AABA jazz standard playing multiple choruses; matches the tune's known form structurally (dominant repeated block + several one-off variants). |
| abba_chiquitita | 170.5 bpm | 232 | 17 | Clear repeats (S2@16-24 & 48-56; S4/S5/S6/S7 recurring across 32-176) consistent with verse/chorus pop form, not degenerate (not all-one-cluster, not all-singletons). |
| aretha_chain_of_fools | 117.2 bpm | 83 | 2 | Mostly one dominant section (S0, 75/83 bars) with one 8-bar contrasting block (S1@40-48) — plausible for a vamp-heavy soul groove tune with a short bridge; the LOW section count itself is a plausible read for this song's actual form, not obviously a failure (would need to actually listen against the audio to be sure — flagged as the honest confidence level for a qualitative check). |

None of the three collapsed to one giant cluster or fragmented into all-
singletons (the two known failure modes seen throughout tonight's clean-
iReal work) — a reasonable non-degenerate first real-audio result. **This is
a qualitative checkpoint only, not a validated metric** — the user should
inspect/listen via the deployed debug page, not trust a number.

Known caveats carried into this checkpoint, not resolved here: (1) tempo
inherits the pre-existing beat-tracker/grid-phase limitations already
tracked in known_issues.md (autumn_leaves' 187.5bpm/4-beat-bar grid matches
what's ALREADY deployed for this song, so not a new regression, but not
fixed either); (2) `estimate_tonic_pc` is a first-pass heuristic, not a real
key detector — a wrong tonic guess feeds a consistent-but-wrong rotation
into the model (within-song relative structure is still preserved, per the
reasoning in symstruct_learned.py, but absolute-key-dependent pattern
recognition inside the model may be degraded); (3) root-only input inherits
whatever the nnls24 root head's real accuracy is on these specific
recordings — not independently re-verified in this call, taken from the
capstone doc's corpus-level numbers.



---

# MILESTONE — Call 2 of 2: grid-phase fix ruled out (2 principled attempts,
# both negative), adaptive hierarchy re-falsified with a NEW proba+varspan
# encoder, multi-seed re-audit CORRECTS the "learned beats block8" claim to a
# statistical tie, 3-level (phrase/section/form) structure DEPLOYED and
# curl-verified.

**Bottom line**: spent the full budget on real, principled experiments; most
came back negative, and are logged as closed/negative so they aren't re-tried.
The one clearly positive, load-bearing outcome is an HONESTY correction: the
previously-reported "+0.010 learned-encoder margin over block8, 5/5 seeds
positive" does NOT reproduce across 9 fresh seed-runs (3 encoder variants x 3
seeds each) — mean margin is -0.002 to -0.007 in all three, i.e. statistically
tied with a training-free baseline. This matters more than any single
metric win: it changes what claim is safe to make about the "deployable
learned structure model" going forward (see Task 3 below).

**What's deployed right now**: `http://100.89.209.63:7771/debug/structure`
renders 3 nested levels (phrase/2-bar, section/8-bar, form/coarse) per song,
curl-verified 200, server restarted cleanly (PID 49597), `/` and `/library`
re-verified 200 too.

**Task-by-task**:
1. **Grid-phase fix — NEGATIVE, closed.** Two unsupervised phase-selectors
   (repeat_clarity-style: 0.689->0.627; chord-change-boundary-alignment:
   0.689->0.672, even with a conservative margin gate) both make things
   WORSE than doing nothing. A phase-free reformulation (novelty-curve
   segmentation) is even more decisively negative: its own per-song ORACLE
   over the full hyperparameter grid (0.670) stays BELOW the deployed
   baseline (0.689) — not a tuning problem, a formulation ceiling. The
   +0.078 oracle gap (0.689->0.767) is real but not currently reachable.
2. **Hierarchical/multi-level clustering.** Trained a NEW encoder combining
   two previously-separate pieces (probabilistic-root input + variable-span
   training) — `scratchpad/keynorm_proba_varspan.pt`. Confirmed root-only-vs-
   full-chord tie extends to this setting (0.696 vs 0.689 at scale=8). Then
   re-ran the adaptive agglomerative merge with THIS similarity source:
   **still 0.448, statistically identical to the token encoder's 0.460
   from Call 1's handoff** — this DECISIVELY closes "maybe a better
   similarity rescues the adaptive merge" (tested and rejected across 2
   independent encoder types now) — the bottleneck is the greedy merge
   DECISION procedure, not the embedding. Delivered "multiple structure
   levels" instead as a validated 3-tier STACK of the same fixed-grid method
   (phrase/section/form) — deployed, see above.
3. **Multi-seed validation — the important correction.** Re-ran the
   deployable scale=8 comparison at 3 fresh seeds each for 3 encoder variants
   (fixed-8 non-varspan, token-varspan, proba-varspan): **9/9 runs land
   within +-0.014 of parity with block8, mean margin negative in all 3
   variants.** This does not match the previously-logged 5-seed/+0.010/
   sign-test-p=0.03 result. Revised recommendation: don't cite "learned
   beats block8" going forward — treat them as tied. The learned approach's
   real, still-valid value is elsewhere (P/R frontier improvement, real-audio
   noise robustness — both are RELATIVE comparisons unaffected by this
   correction).
4. **Real key detection — deprioritized, as the brief authorized.** Found
   `local_key_model.py`/`local_key_heuristic.py` but neither is a drop-in
   replacement for `estimate_tonic_pc` (no trained checkpoint, discrete-token
   input requirement, solves per-section local key not global tonic) —
   documented why rather than rushing a mismatched integration.

**Full detail for all 4 tasks is in `docs/known_issues.md`** (search "Task 1",
"Task 2", "Task 3", "Task 4", all dated 2026-07-18, ★ STRUCTURE / SEGMENTATION)
— that's the authoritative record; this file is the session narrative.

**New files this call**: `scratchpad/phase_fix.py`, `scratchpad/phase_fix2.py`,
`scratchpad/novelty_seg.py`, `scratchpad/symstruct_proba_varspan.py`,
`scratchpad/adaptive_proba.py`, `scratchpad/hierarchy_real.py`,
`scratchpad/run_real_structure_multilevel.py`,
`scratchpad/keynorm_proba_varspan{,_s1,_s2}.pt`,
`scratchpad/keynorm_varspan_s{1,2}.pt`, `scratchpad/real_structure_multilevel.json`.
Modified: `scripts/harmonia_server.py` (`/debug/structure` route only).

**Next-step recommendation for whoever picks this up**: the flat-matching
ceiling (chord-sequence-only, ~0.69-0.70) now looks genuinely hard to move
with the tools tried so far (hard match, learned similarity, adaptive merge,
phase correction, novelty curves — 4 independent hierarchy/phase directions
now closed-negative across two sessions). The highest-EV remaining direction
per Call 1's handoff is still queued and untouched: **learned harmonic
equivalence** (tritone subs / reharmonizations treated as "the same" for
matching) — a genuinely different signal than anything tried in this call,
not another spin on phase or merge-decision variants.

---

# Call 2 of 2 — Grid-phase fix, hierarchical clustering, multi-seed validation

Start: 2026-07-17 20:20 CEST (Call 1 ended well under budget). Budget: up to 3h,
stop ~23:20 CEST. Mandate (user's words): "find our best model, don't forget to
cluster at different structure levels, use the harmonia research grills and don't
quit until you have something that works great." 2-bar nuclear mandated default.

Reading order followed: `docs/handoff_2026_07_18_structure_detection.md`,
known_issues.md CORRECTION + GRID PHASE entries, Call 1's log above,
`scratchpad/symstruct_adaptive.py`/`symstruct_adaptive_scale.py`.

## Premise check / baseline reproduction (doctrine Phase 0)

Reproduced `symstruct_adaptive_scale.py --enc keynorm_varspan.pt --tau 0.75`:
fixed learned-union scale=8 V_F=0.689 (doc says 0.696, within noise), GT-oracle
scale=0.768 (doc 0.770), block8=0.695 (doc 0.695). Close enough — baseline
reproduces, no environment drift. Also reproduced `symstruct_adaptive.py
--nuclear 2`: free-form recurrence merge V_F=0.460 (doc table: 0.46). Confirmed.

**Confirmed the handoff's flagged-unconfirmed question**: `nuclear_spans()` is
called directly (phase-locked, first block always starts at bar 0) by BOTH
`symstruct_adaptive.py:adaptive_segment()` (line 100) and
`symstruct_adaptive_scale.py:learned_union_labels()` (line 33). Only
`build_varspan_blocks()` (the *training*-time representation) is phase-agnostic
(stride-2, all start positions) — that property was never carried into the
deployed inference path. So yes, the deployed adaptive/scale-selector code was
leaving the phase gain on the table, exactly as flagged.

## Task 1: grid-phase fix attempts — BOTH candidate approaches tried, BOTH negative

Full writeup in known_issues.md ("Call 2 follow-up..." and "Approach 1b..."
entries) — summary here:

1. **1a: unsupervised phase selection.** Confirmed the oracle gap is real and
   even bigger for the deployed learned method than the old hard-match number:
   0.689 (phase=0) -> 0.767 (oracle-of-8-phases), 40.3% of songs affected
   (`scratchpad/phase_fix.py`). Tried two selectors: `repeat_clarity()` (the
   SAME heuristic that worked for scale-selection) -> 0.627, WORSE than doing
   nothing (confound: it scores clustering OUTPUT, which a bad phase can fake).
   Chord-change boundary-alignment signal (content-only, no clustering-output
   confound) -> 0.672, still worse than baseline, even with a conservative
   margin-gated switch (0.671-0.674 across margins 0.05-0.20)
   (`scratchpad/phase_fix2.py`). Diagnosis: chord changes are too frequent
   (every 1-2 bars typically) to discriminate true section boundaries from
   arbitrary bar positions. **Closed, negative.**
2. **1b: phase-free novelty-curve segmentation**, sidestepping phase-selection
   structurally instead of solving it (`scratchpad/novelty_seg.py`): boundary
   novelty = 1-cosine(embed(before-window), embed(after-window)) per bar,
   peak-picked, then union-find labeling on the resulting variable segments.
   Grid search on val -> test V_F=0.557, well below baseline. Verified this
   isn't a segment-COUNT problem (predicted 3.58 vs GT 3.31 mean segments/song,
   60-song sample) before concluding it's a placement problem. Then ran a
   per-song ORACLE over the full hyperparameter grid (w, thresh, tau_label,
   n=100): **mean V_F=0.670 — still below the deployed 0.689 baseline even with
   unlimited per-song hindsight tuning.** This is decisive, not a tuning gap:
   the block-similarity encoder was never trained for boundary/novelty
   detection, only for same-section block similarity, and that mismatch caps
   this formulation below the fixed-grid approach regardless of hyperparameters.
   **Closed, negative.**

**Net result: Task 1 spent real, principled effort (4 independent tested
hypotheses across 1a/1b) and found the +0.078 oracle gap genuinely
inaccessible with current tools.** Deployable recommendation is UNCHANGED:
fixed-phase (phase=0) learned-union at scale=8, V_F≈0.689-0.696. Moving to
Task 2 with this as the confirmed best single-level base.

---

# Call 3 — V4 raw real-chroma dot product on real audio, V1/V2/V3 multi-seed
# validation, full bar-to-bar self-similarity matrices (SSM)

Start: 2026-07-18, continuing directly from the "Hand-crafted CHORD-TONE-
DISTANCE" entry in known_issues.md (V1/V2/V3 hand-crafted similarity, single-
split only, V4 not yet implemented). Budget: up to 2h. Reading order:
`docs/handoff_2026_07_18_structure_detection.md`, the CHORD-TONE-DISTANCE
entry, this file's Call 1/2 sections (for the real-song qualitative baseline
to compare V4 against and to avoid re-litigating settled findings — grid
phase, adaptive merge, multi-seed correction on the LEARNED encoder are all
already closed).

## Premise / environment check

Reproduced `chord_distance_eval.py`'s single-seed baseline exactly before
changing anything: V1=0.682/tau*=0.84, V2=0.683/tau*=0.81, V3=0.675/tau*=0.75
at size=8 — matches known_issues.md verbatim. No environment drift.

## Task 1: V4 (raw real-chroma dot product) on real audio

`scratchpad/chord_distance_v4_real.py`. Reused `real_root_proba.py` (Call
1/2's extraction) and `symstruct_proba.estimate_tonic_pc` (same heuristic
key-norm proxy) unchanged — only the SIMILARITY is new, position-aligned
cosine on the real 12-d per-bar root-softmax instead of an idealized chord
template, same `block_sim`/union-find machinery as V1-V3 (imported from
`chord_distance_eval.py`, not reimplemented).

Reused V1's val-tuned tau*=0.84 (no real-audio GT exists to tune tau
against — same reuse-tau spirit as `run_real_structure.py`'s reuse of
`tau_star` from training).

| song | tempo | n_bars | V4 n_sections (tau=0.84) | learned-encoder n_sections (Call 1/2, same song) |
|---|---|---|---|---|
| autumn_leaves | 187.5bpm | 330 | **41 (degenerate — ~zero merges)** | 18 |
| abba_chiquitita | 170.5bpm | 232 | 17 | 17 |
| aretha_chain_of_fools | 117.2bpm | 83 | 2 | 2 |

2/3 songs: V4 matches the learned encoder almost exactly with zero training
— strong support for "the encoder isn't earning its complexity" theme.
1/3 (autumn_leaves): V4 degenerates badly at the reused tau.

**Root-caused the autumn_leaves failure** (doctrine: diagnose before
concluding "doesn't work") via a tau sweep (0.55-0.84, logged per-song in
the JSON's `tau_sensitivity_n_sections`): tau=0.65 fixes autumn_leaves
(18 sections, matches the learned encoder) but the SAME tau=0.65 collapses
aretha_chain_of_fools to n_sections=1. No tau in the swept range is
simultaneously non-degenerate on all 3 songs. Diagnosis: on clean iReal, one
global tau (0.84) generalizes across 1992 tunes because idealized chord
templates have a fixed geometry; real per-bar softmax vectors vary in
peakiness/entropy from song to song (different recording/arrangement
"noisiness"), so a fixed cosine threshold is mis-calibrated per-song in a
way it isn't on symbolic data. **This is the first real-audio result this
session that clearly favors the learned encoder over the training-free
schemes** — not on accuracy where they're tied, but on cross-song threshold
robustness with a SINGLE fixed operating point. Reported both halves
plainly, per the brief, rather than picking a side.

## Task 2: V1/V2/V3 multi-seed validation (mandatory, Honesty-bar rule)

`scratchpad/chord_distance_multiseed.py`, 7 independent split seeds
(0-6), size=8, identical val-tau-sweep -> test-report protocol as the
single-seed run, same corpus loaded once and reused across seeds (only the
train/val split shuffle changes — these schemes have no trained parameters,
so there is no other kind of "seed").

| scheme | mean | std | range |
|---|---|---|---|
| V1_binary | 0.6834 | 0.0020 | [0.681, 0.687] |
| V2_weighted | 0.6832 | 0.0018 | [0.682, 0.687] |
| V3_tiv | 0.6788 | 0.0026 | [0.675, 0.684] |

Pairwise (seed-paired, same splits): V1-V2 mean diff = +0.0001±0.0009,
sign flips 4/7 — **not a real ranking, noise**. V1-V3 and V2-V3: mean diff
+0.0044/+0.0045, sign-consistent 7/7 seeds, diff ≈4.5x the seed std —
**a real, if small, effect**: V3 (the approximate-TIV scheme) is reliably
worse than V1/V2 by about half a point of V_F. Revised honest ranking:
V1≈V2 > V3 (small but real gap), all three still land inside flat block8's
0.68-0.70 range and remain tied with the learned encoder given Call 2's own
multi-seed correction on that side.

## Task 3: full bar-to-bar self-similarity matrix (SSM)

`scratchpad/bar_distance_matrix.py`. Two examples, saved as JSON (not
visualized — orchestrating session's job per the brief):

1. **Clean iReal, "All Of Me"** (jazz1460.txt), 32 bars, AABC form
   (A:0-7, B:8-15, A:16-23, C:24-31) — legible textbook form, GT labels
   included. Scheme: V1 binary (the interpretable default; Task 2 found
   V1≈V2 tied and V3 reliably worse, so V1 is the right pick when scores are
   close, per the brief's own tie-break guidance).
   -> `scratchpad/bar_distance_matrix_all_of_me.json`.
2. **Real audio, "aretha_chain_of_fools"**, 83 bars — small enough for a
   readable matrix, and Task 1 already characterized its structure (S0
   dominant 75/83 bars + one 8-bar bridge S1@40-48) as a cross-check.
   Scheme: V4 raw chroma (key-normalized via `estimate_tonic_pc`).
   -> `scratchpad/bar_distance_matrix_aretha_chain_of_fools.json`.

**Verification before trusting either matrix** (doctrine: verify a small
worked example before trusting a broad result):
- Diagonal = 1.0 exactly (every bar perfectly self-similar) on BOTH matrices.
- All Of Me: naive "same GT label anywhere" check is a trap — bar 0 (start
  of an A-run) vs bar 23 (END of a different A-run) scored 0.000, which
  looked like a bug until re-checked: those are different PHRASE POSITIONS
  within the section (a "the" turnaround chord vs a "1" downbeat chord),
  not a repeat of each other. Fixed the check to compare POSITION-MATCHED
  bars across two occurrences of the same section (bar 4-of-run1 vs
  bar 20-of-run1, both position 4 within an A-run) -> sim=1.000, as
  expected. Cross-label (A vs B) sim=0.289, correctly lower.
- aretha_chain_of_fools (no GT, so no label-based check): within-S0 mean
  similarity (bars in Task 1's S0 clusters, 0-40 and 48-83) = 0.787 >
  S0-vs-S1-bridge cross mean = 0.671 — directionally consistent with Task
  1's independently-derived clustering, even though individual real-audio
  bar pairs are noisier than the clean example (e.g. within-S1 bridge
  self-similarity mean only 0.762, close to the cross-section number —
  the bridge itself is short/less internally homogeneous, plausible for
  an 8-bar transitional block).

Both matrices verified correct and ready for the orchestrating session's
visualization pass.

## Session summary

All 3 tasks completed within budget with verified, saved artifacts. Key
findings, most to least novel: (1) V4 raw real-chroma dot product ties the
trained encoder's qualitative read on 2/3 real songs with zero training, but
is measurably LESS robust than the encoder to per-song threshold
mis-calibration (autumn_leaves needs a different tau than the other two
songs) — first result this session that gives the learned encoder a concrete
edge over training-free schemes on real audio specifically; (2) V1≈V2 tied,
V3 reliably ~0.0044 worse, multi-seed confirmed — a small but real
distinction that wasn't visible in the single-seed number; (3) full n_bars
SSMs built and verified for one clean/GT and one real/noisy song, ready for
visualization. Nothing cut for time.

Next-step recommendation: if V4's threshold-robustness gap is worth closing,
a per-song ADAPTIVE tau (e.g. picked from the similarity distribution's own
statistics — top-k percentile of off-diagonal values — rather than one fixed
global number) is the natural next experiment; untested this call, flagged
as the highest-EV follow-up specifically for the V4 track.

---

# NEW SESSION SECTION (2026-07-18, later call) — learned merge criterion on top of the multigrain SSMs

Budget: up to 3h foreground. Brief (5 linked pieces from the user): (1)
learned/calibrated merge criterion, low-false-positive priority, (2) noise-
calibrate iReal to match real-audio SSM statistics, (3) intro/outro via
1-2 bar matrix, (4) section detection via ML on 8/16-bar matrices, (5)
compositional hierarchy check. Read `docs/handoff_2026_07_18_structure_detection.md`
and all 2026-07-18 ★ STRUCTURE / SEGMENTATION known_issues.md entries first
(doctrine: don't re-derive project history).

**Mid-session coordinator update**: user reviewed the bass/treble/combined
3-row comparison live and wants bass_sim/treble_sim kept as a 2-D feature
pair, not collapsed to the `bt_concat` averaged scalar, for the merge
criterion and section detector. Threaded through everything below.

## Step 0 — compositional hierarchy shortcut: VERIFIED

`scratchpad/hierarchy_shortcut.py`. Built the 1-bar raw dot-product Gram
matrix G1 once per song per register (bass_only, treble_only, kept
separate), derived grains {2,4,8,16} via diagonal-prefix-sum lookups (the
"raw dot product composes, cosine doesn't" identity — block_sim's numerator
is already a position-aligned sum of per-bar dot products, so it's exactly
a diagonal-band sum of G1). Checked against the EXISTING full-recompute
matrices already saved in `bar_ssm_rawchroma_<song>.json` (`grains_bass`/
`grains_treble`, computed independently by an earlier script).

**Result: max abs diff = 2.1e-14 across 3 songs (aretha n=83, autumn_leaves
n=330, abba_chiquitita n=232) x 2 registers x 4 derived grains.** Matches to
float noise — confirmed correct, safe to build on. Full numbers in
`scratchpad/hierarchy_shortcut_verify.json`. Logged to known_issues.md.

Next: Step 1, noise calibration.

**Design-decision correction, mid-session (logged per coordinator's explicit
request, so it isn't confusing later):** an earlier coordinator message said
the user wants bass_sim/treble_sim kept as a 2-D pair ("le modèle Bass et
treble est clairement le plus adapté" read as "keep them separate"). A
follow-up correction reversed this: the intended reading was "fuse/combine
bass and treble" ("fusionne Bass et treble"), i.e. use the COMBINED
bt_concat-equivalent scalar as originally specified in the brief, not a 2-D
feature pair. Reverted to the combined scalar as PRIMARY for Step 1's
calibration target and everything downstream (Step 2, Step 4). Step 0's
per-register (bass/treble) substrate was NOT redone/discarded — it's kept as
a diagnostic breakdown alongside the combined number in
`scratchpad/noise_calibrate.py`, and the combined signal is computed as
`(sim_bass+sim_treble)/2`, which is exactly `cosine(bt_concat)` per the
already-proven independent-unit-norm identity, so no extra computation path
was needed to support both views.

## Step 1 — noise calibration: PREMISE FALSIFIED (full details in known_issues.md)

Summary (see known_issues.md "Step 1 noise calibration..." entry for full
numbers): defined `mean_p90` off-diagonal-similarity statistic, validated it
correlates with a label-based AUC on clean iReal (r=0.50, n=78, after an
initial "gap" formulation failed validation at r=-0.23 — logged as a dead
end, don't retry the gap formulation). Swept additive-Gaussian noise
sigma 0-2.0 on iReal per-bar vectors, 3 seeds, 300-tune sample, grain=8,
bass/treble/combined. **Real audio's target stat_B (0.83 combined) sits
ABOVE clean iReal's zero-noise baseline (0.66)** — every tested noise level
moves further from the target, not closer. Confirmed via raw off-diagonal
similarity distributions: real audio's minimum pairwise similarity (0.75 for
aretha_chain_of_fools) exceeds clean iReal's MEDIAN (0.60) — an elevated
floor, not added scatter. Additive noise cannot reproduce this regime by
construction.

**Stopping here for this call.** Per the brief's own priority ("a correct,
verified calibration is worth more than five half-checked ideas") and the
research-loop doctrine (report negative results plainly, don't build Step
2/4 on a calibration known not to transfer), NOT proceeding to Steps 2-5
(learned merge criterion, intro/outro, section detector, deploy) this call —
they all depend on noise-calibrated training data that doesn't yet exist in
a valid form.

## Recommended next steps (for whoever picks this up)

1. Fix the noise model first: try a multiplicative/floor-blend model (e.g.
   `noisy = (1-alpha)*clean + alpha*floor_vector` for some fixed or
   per-song floor vector, rather than additive Gaussian) — target is to
   raise the LOWER BOUND of off-diagonal similarity, not just add variance
   around the existing mean. Re-run the same `mean_p90` validation before
   trusting it.
2. Worth checking cheaply first: is the elevated floor specific to the
   UNTRAINED V4 raw-chroma scheme (no learned noise suppression at all), or
   does the trained-head `root_proba` V4 (from the "PROBABILISTIC root-only"
   entry, real_root_proba.py) show the same elevated-floor pattern? If the
   trained head's floor is closer to clean iReal, that's a strong argument
   for building the merge criterion on trained-head chroma, not the
   genuinely-untrained scheme this call worked with — a cheap check
   (compute the same off-diagonal-mean table) before redesigning the noise
   model.
3. Steps 2 (learned merge criterion) and 4 (section detector) are ready to
   proceed AS SOON AS a validated noise model exists — Step 0's
   compositional-hierarchy substrate and the `mean_p90` validated statistic
   are both reusable as-is, only the noise-injection function needs
   replacing.
4. Step 3 (intro/outro) does not depend on the noise calibration at all
   (it's a direct real-GT validation against iReal's `i`-labeled intros) —
   could be picked up independently/in parallel with item 1 above if
   someone wants a non-blocked task.

---

# NEW SESSION SECTION — 3rd continuation call, ~04:21 CEST, 2h budget

Following the 3 concrete follow-ups from the prior call's morning summary
(scratchpad artifacts all read first, per doctrine — none re-derived).

## Follow-up 1: floor-blend-trained merge criterion — DONE, negative result

`scratchpad/blend_transfer_test.py`. Trained Step 2's threshold on
alpha=0.40 floor-blended iReal (TRAIN+VAL blended, TEST clean — the proper
asymmetric design the prior call's "sanity pass" was NOT). Result: does
NOT help real-audio transfer. Blend-trained threshold (0.988) is actually
HIGHER/more conservative than clean-trained (0.973), and still fails to
rescue autumn_leaves (41/41) or aretha (10/10) on real audio, and makes
abba_chiquitita marginally worse (28->29, newly degenerate). Root cause:
floor-blend raises negative-pair similarity in lockstep with positive-pair
similarity, so the FPR-gated threshold rises to compensate — self-
cancelling. This is evidence FOR keeping the per-song adaptive-percentile
fix (Step 6) rather than trying to fix it with better absolute-threshold
calibration; the problem is structurally about per-song floor variance
(aretha 0.891 vs abba 0.672), which a single global alpha can't encode.
Logged to known_issues.md in full. Time: ~15 min.

Next: Follow-up 2 (FPR-gate frontier sweep).

## Follow-up 2: FPR-gate frontier sweep — DONE, real finding

`scratchpad/fpr_frontier_sweep.py`. Swept target_fpr in
{0.02,0.05,0.10,0.15,0.20,0.30}, 5 seeds, corpus-scale. Found an interior
optimum at target_fpr=0.10 (V_F=0.6851 +- 0.0151), matching the
separately-tuned V-measure-optimal tau=0.78 result (0.682) within noise —
useful cross-check. Currently-deployed low-FP point (target_fpr=0.05) costs
0.037 V_F relative to this optimum in exchange for higher bar-pair
precision (0.687 vs 0.627). Full table logged to known_issues.md. Time: ~20
min.

## Follow-up 3: adaptive-percentile stress test on iReal — DONE, negative
(as hypothesized)

`scratchpad/adaptive_percentile_on_ireal.py`. Swept percentile in
{50..98} corpus-scale (5 seeds) on iReal test songs, using the SAME
adaptive-percentile mechanism deployed for real audio. Result: clearly
HURTS iReal at every percentile — best case (P=98) V_F=0.603 vs fixed-tau
best 0.685, plus a residual 4.2% degenerate rate that never fully
disappears. Confirms the adaptive-percentile fix is a REAL-AUDIO-SPECIFIC
patch (compensating for real audio's uniquely elevated, song-dependent SSM
floor), not a generally-better thresholding strategy — should not be
proposed as a replacement for fixed-tau on iReal-native eval. Logged in
full to known_issues.md. Time: ~15 min.

Next: error-analysis loop on the 3 real songs' current best deployment
(adaptive-percentile, still the best available for real audio per Follow-up
1's negative result and Follow-up 3's confirmation it's appropriately
scoped).

## Error-analysis loop (mandatory per brief) — DONE, partial win, DEPLOYED

`scratchpad/error_analysis_recursive_split.py`. Inspected adaptive-
percentile's (row c) actual failure cases on the 3 real songs directly
(qualitative, no GT): 2 of 3 songs have very long (48-80 bar) single-
cluster runs that plausibly hide internal chorus-repeat structure
(autumn_leaves is a known 32-bar form; an 80-bar run = ~2.5 choruses
collapsed). Hypothesis: global per-song tau is calibrated for the whole
song, missing internal structure within its own high-similarity residue.
Fix: recursive local re-split — runs >=32 bars get a LOCAL P75 threshold
and local re-clustering, applied once. Result: genuine partial win
(autumn_leaves 12->16 sections, abba_chiquitita 10->14 sections, no new
degeneracy anywhere) but NOT complete — one 80-bar run on autumn_leaves
still resists splitting even locally, characterized as a real remainder
needing a non-harmonic signal, not another threshold tweak (logged per
rule #4, do not re-attempt with more threshold tuning).

DEPLOYED as new row (d) on `/debug/merge-criterion` (static-file rebuild
only, no server restart needed). curl-verified 200 on ALL 7 routes (/,
/library, /debug/ssm-multigrain, /debug/ssm, /debug/structure,
/debug/metric-artifact, /debug/merge-criterion) + content-checked (13
level-label divs = 4 rows x 3 songs + 1 CSS selector, matches terminal
output). Time: ~25 min.

Total call time so far: ~75 min of the 2h budget. All 3 follow-ups +
mandatory error-analysis loop complete. Updating MORNING SUMMARY next per
instructions (merge into existing, don't leave two competing summaries).

## Call: on-chart bar-merge suggestions overlay (chart_interactive.py, sanctioned edit #2)

Task: add bar-level algorithmic merge SUGGESTIONS rendered directly on the
real chart, additive to the existing free-select section-merge UI, per
user's explicit ask ("il faudrait que ce soit directement sur le chart pour
qu'on puisse voir où tombe les accords") — not the abstract card-list
`/debug/bar-merge-game` page built earlier tonight. This is the SECOND-ever
sanctioned edit to `harmonia/output/chart_interactive.py` (first was the
"Label sections" feature, commit `318474a`); an earlier instruction tonight
said not to touch this file for the debug-page work, explicitly lifted for
this request only.

Read the full existing merge flow first (per CLAUDE.md's log-before-change
rule for this file): `#merge-mode-btn` / `mergeSelectActive` / `mergeSel` /
`#mergeConfirmModal` is section-granularity free manual selection that
persists to `store.merges` via `/api/annotations` — it does NOT call
`/api/reinfer` at all (confirmed by grep, not assumed). The reinfer-pooling
pattern I needed to reuse was actually already fully worked out in
`scratchpad/bar_merge_game.html` / `/debug/bar-merge-game` (built earlier
tonight): POST `{confirms:[], merges:[{spans}]}` to `/api/reinfer/<file>`,
preview-only re-decode, nothing persisted. Full details + verification in
`docs/known_issues.md` ("REFRAME: on-chart bar-merge SUGGESTIONS overlay").

Built: new `#suggest-mode-btn` toggle, own IIFE (zero shared state with the
existing merge flow), gold badge overlay on candidate bars (`.measure[data-
bar]`), tap → popover → confirm calls `/api/reinfer` same as the debug page,
dismiss is page-session-only (not persisted). New server route `GET /api/
bar-merge-candidates/<filename>` — thin passthrough over `scratchpad/
bar_merge_candidates_<stem>.json`, 200+empty (not 404) for unscoped songs,
loosely-coupled data contract so the parallel clustering-bakeoff session's
eventual better generator can swap in without a UI change.

One real deviation from the brief: preferred target song was aretha/abba,
but both their BAKED `docs/plots/*.html` files predate the "Label sections"
commit (no matching anchor strings for a safe mechanical patch), and a full
re-bake via `/api/analyze` was judged too risky for a same-night additive
task (touches acoustic backend selection — CLAUDE.md rule #6 territory).
Used `autumn_leaves` instead (baked same day, after the anchor commit,
candidate JSON + cached audio both present) — same validation strength,
lower risk. Logged as a scope decision, not a silent substitution.

Verification: `py_compile` both Python files OK; `node --check` on the
FULL embedded `<script>` extracted from `_TEMPLATE` (old + new code paths
together) OK; `node --check` on the actually-patched `autumn_leaves.html`'s
real embedded script OK; server restarted (PID 68258); all 8 routes named
in the brief curl to 200; new candidate API returns real data for
autumn_leaves and graceful empty JSON for an unscoped song; full `/api/
reinfer` round-trip using a REAL candidate span from the API → 200,
well-formed `n_changed`/`rejected` fields; 3 other chart pages (including
the untouched aretha/abba files) still 200 after restart. Honest gap: no
headless browser here, so the actual tap→popover interaction on a phone is
unverified — flagged explicitly rather than claimed.

Time: ~70 min of this call's 2h budget. Stopping here — target delivered,
verified to the extent possible without a browser, logged.

---

## Follow-up call: `/chart/<file>` "terrible interface" complaint — structural fix (redirect to SPA), NOT another serve_chart patch — 2026-07-18

User (direct, screenshot): followed `/chart/inferred_autumn_leaves.html`
(the link the bar-merge-suggestions work above had them use tonight) and
landed on "l'interface terrible" — the plain baked chart page's own
Read/Analyse/Annotate control, structurally a different, plainer render
than the SPA's chart view even though both show that same 3-mode control.
Explicit ask: stop `/chart/<file>` from ever serving that page.

This is the THIRD complaint about this exact route this project — see
known_issues.md's two prior "old UI" entries (both 2026-07-17), both of
which patched *within* `serve_chart` (bar-1 offset re-derivation, back-
button destination) without touching the actual structural problem their
own diagnosis named: `/chart/<file>` (`serve_chart`, reads baked HTML off
disk) and `/` (the SPA, renders via `/api/chart-model`) are two genuinely
separate code paths for the same content, so every fix to one leaves the
other's presentation untouched. Patching a third cosmetic detail into
`serve_chart` would have hit the same wall again.

**Fix implemented (durable, not another patch):** `serve_chart` now issues
a 302 to `/?open=<file>`. New deep-link support added to `app_shell.html`'s
`API.build()`: reads `?open=` from the query string and calls
`openChart(file)` directly instead of defaulting to `go("library")` — the
SPA already had everything else needed (`openChart` fetches `/api/chart-
model/<file>` and renders in place); it just never had an entry point that
bypassed the library screen. So any `/chart/<file>` link (bookmarks, the
align tool's `backHref`, swipe-nav, the classic `/library` list, shared
links) now opens directly into the polished SPA showing that exact song.

**Scope guard — did NOT silently break tonight's bar-merge-suggestions
overlay.** That feature (logged above, "REFRAME: on-chart bar-merge
SUGGESTIONS overlay") is baked directly into `docs/plots/inferred_autumn_
leaves.html`'s static HTML and has no SPA-side equivalent (the SPA's chart
view is a from-scratch JS renderer, not a share of `chart_interactive.py`'s
template) — redirecting it away would have made the overlay unreachable
minutes after it shipped. Checked whether porting it into `app_shell.html`
was cheap: no — `app_shell.html`'s `renderChart()` is a ~900-line from-
scratch JS renderer with its own bar-cell/section model, not a shared
surface with `_TEMPLATE`'s baked JS; porting a tap→badge→popover→reinfer
flow into it correctly (plus verifying it) would have been the bulk of this
call's budget on its own. Went with the smaller, safe option instead:
`serve_chart` checks the baked file's own content for the overlay's marker
string (`suggest-mode-btn`) before redirecting — if present, falls through
to the historic full-HTML serving path unchanged (offset re-derivation,
back-button injection, swipe-nav, everything). This is a *content* check,
not a filename allowlist, so it automatically stays correct as more charts
get baked with the overlay by default (`chart_interactive.py`'s live
`_TEMPLATE` already includes it) — no code will need touching again when
that happens.

**Verified (server restarted, PID 69006):**
- `curl -D- /chart/inferred_autumn_leaves.html` → 200, NOT redirected (has
  `suggest-mode-btn`) — old full-HTML path confirmed still serving,
  `node --check` on all 10 inline `<script>` blocks in the actually-served
  page → all OK, `#harm-back` present pointing at `/`.
- `curl -D- /chart/inferred_blue_bossa.html` → `302 Location: /?open=
  inferred_blue_bossa.html`; `curl -L` on the same URL lands on
  `<title>Harmonia</title>` (the SPA), page contains the new `URLSearchParams
  (location.search).get("open")` deep-link code.
- Same 302→SPA behavior confirmed for 3 more untouched songs (abba
  chiquitita, adele hello, satin doll) — fix is general, not scoped to one
  song, by deliberate design (content-check, not filename check).
- `/api/chart-model/inferred_blue_bossa.html` (what `openChart` fetches)
  → 200, real payload (title "Blue Bossa", 10 sections).
- `node -e "new Function(...)"` on `app_shell.html`'s full inline script
  → syntax OK.
- Full deployed-tonight route sweep, all 200: `/`, `/library`,
  `/debug/structure`, `/debug/ssm-multigrain`, `/debug/ssm`, `/debug/
  metric-artifact`, `/debug/merge-criterion`, `/debug/bar-merge-game`,
  `/api/bar-merge-candidates/inferred_autumn_leaves.html`.
- `curl /chart/does_not_exist.html` → still 404 (existence check runs
  before the redirect branch, unaffected).
- Grepped for any `fetch()`/`XHR` consumer of `/chart/<file>` that would
  silently receive a redirect body instead of chart HTML (would be a real
  regression) — none found; every reference is a navigation (`<a href>`,
  `location.href`, `window.open`), all of which transparently follow a 302.

**Not verified (honest gap, no headless browser):** the actual on-device
tap flow (open a link on iPhone → land in SPA → chart renders correctly
end-to-end visually). Verified the server contract and JS syntax; the
render itself was already the SPA's existing, previously-verified
`openChart`/`loadModel` path (unchanged by this fix), so risk here is low
but not zero.

**What this does NOT solve:** the bar-merge-suggestions overlay is still
only reachable via the old baked-HTML path for the one song that has it —
it has not been ported into the SPA's chart renderer. Next time a song
other than `autumn_leaves` gets that overlay baked in (automatic on any
future re-bake, since `_TEMPLATE` carries it now), it will correctly stay
un-redirected too, but the underlying UI duality (two renderers for one
feature) persists until the overlay is ported into `app_shell.html`
properly — flagged as a real follow-up, not resolved here.

Time: ~45 min of the 2h budget for this call.

---

## Bar-grid "content overflow" complaint — root-caused + fixed: near-boundary onset mis-binning, not fast harmonic rhythm — 2026-07-18 ★ CHART / BAR-GRID

**Brief**: user reported the chart's bar grid isn't "square" — content overflows one cell into the next, guessed at "the API level" and possibly the song's beginning. Orchestrating session had already screenshotted `autumn_leaves` and found bar 5 (0-indexed bar 4) showing two chords ("Gø7 Cm7") crammed together, flagged as genuinely ambiguous ((a) real fast harmonic rhythm — Autumn Leaves does have ii-V compressions — vs (b) a mis-binning bug) pending evidence. Mid-call, the user supplied a fresh screenshot + explicit intent ("I want Gø7 on its own bar and Cm7 on the very next bar") that settled (a) vs (b) as (b), a real bug, and redirected the ask to root-cause + fix.

### Root cause

Production real-audio bar assignment (`chart_to_interactive_inputs` in `scripts/render_youtube_chart.py`, and its live re-derivation `_apply_bar1_offset_to_payload` in `scripts/harmonia_server.py`) assigns each chord onset to a bar via `bar = floor(abs_beat / bpb)` off a rigid, artificially-uniform beat grid (`chord_pipeline_v1.py`'s `bt = np.arange(phase, duration_s+period, period)`, one constant `period = 60/tempo_bpm` for the WHOLE track). Chord events are a *sparse "start of change" list* — a held chord gets no entry of its own in bars it merely sustains through, rendered client-side as a bare "%" (simile) glyph.

Consequence: whenever a chord's onset timestamp floors into the tail of bar *b* (e.g. beat 2–3 of a 4-beat bar) while the chord itself musically belongs to (and mostly sounds through) bar *b+1*, floor-division still assigns it to bar *b* — stacking it visually alongside bar *b*'s existing onset — and bar *b+1* renders as an empty "%" even though it's really where that second chord's harmony belongs. This reads exactly like "content overflowing into the next cell," but the actual overflow direction is backward (the *next* bar's content bleeds into the *current* cell), and the label pairing is real content, correctly detected — just filed under the wrong bar index.

### Corpus check (within-song, all 328 bars of `autumn_leaves`, not just the reported example)

Verified this is systematic, not a one-off, by counting bars with ≥2 onsets whose following bar is empty (the exact "% masking a real second chord" signature), at every bar-1 phase offset from -3 to +3:

| offset | crowded bars | of which next-bar-empty |
|---|---|---|
| **0 (baked default)** | 11 / 329 | 7 |
| 1 | 10 / 328 | 7 |
| **3 (user's saved "Set bar 1" fix)** | 17 / 328 | 11 |

Two findings: (1) the bug is present even at the untouched offset=0 baseline — 11 instances scattered from bar 4 to bar 317, i.e. throughout the whole 7-minute song, not localized to the intro; (2) the user's own global bar-1 phase correction (saved 2026-07-18T15:23:22, offset_beats=3, applied via the "Set bar 1" tool to fix the *song-start* alignment) measurably **worsened** the mid-song symptom (11→17 crowded bars) — a single global phase constant cannot simultaneously be correct for the intro and for every later passage, the same structural limitation already characterized in this file's "GRID PHASE MISALIGNMENT" entry above, now confirmed to be the SAME underlying phenomenon surfacing as a visible chart-UI defect, not just a downstream structure-metric artifact.

This is NOT the abba-style 2×-tempo-octave lock (checked and ruled out as the mechanism here): `autumn_leaves`'s beat tracker converges to 184.57 BPM at native 44.1kHz vs 92.29 BPM when the same exact `librosa.beat.beat_track(y,sr)` call is run on the same audio resampled to 22.05kHz (a real, reproducible, sample-rate-dependent octave-lock artifact of librosa's fixed `hop_length=512` default — worth flagging separately, see below) — but this tempo ambiguity is a *separate, unresolved* question (which octave is truly correct wasn't settled) and is orthogonal to the onset-mis-binning bug: the mis-binning reproduces identically regardless of which tempo octave is "true," because it's a floor-division artifact of the bar-assignment step, not a beat-tracking error.

### Fix (implemented, verified, live)

Added `rebalance_near_boundary_onsets(chord_dicts, bpb)` in `scripts/render_youtube_chart.py`: when a bar ends up with ≥2 onsets AND the immediately following bar has none, and the last onset in that bar sits in its back half (`beat >= bpb/2`), re-anchor it as the lead (and only) onset of the next bar instead (`bar += 1, beat = 0` — not `beat - bpb`, which goes negative since `beat` is already reduced mod `bpb`; caught this in my own first pass via the offline verification script before touching the live server). Deliberately narrow: bars with ≥2 onsets where the next bar *also* has content are left untouched — that's ordinary fast harmonic rhythm (real ii-V compressions etc.), not this failure mode.

Wired into both production call sites:
- `chart_to_interactive_inputs` (fresh bake, offset=0 baseline) — fixes new analyses going forward.
- `_apply_bar1_offset_to_payload` (server-side live re-derivation under any saved bar-1 offset) — the early-return-on-`offset_beats==0` fast path was changed to still run the rebalance (previously a pure no-op there, which is exactly why the offset=0 baseline case wasn't being fixed at all), and the rebalance is re-applied *after* any non-zero offset shift too, since (per the table above) the offset shift itself can re-trigger new instances of the pattern.

**Result on autumn_leaves at the user's current offset=3**: crowded bars 17→7, next-bar-empty instances 11→1. The originally-reported bar: bar 4 now shows only "Gø7", bar 5 now shows only "Cm7" — exactly the layout the user asked for. Verified with `playwright` against the live server (restarted, PID 75118) at `http://127.0.0.1:7771/chart/inferred_autumn_leaves.html` (redirects to the SPA `/?open=...` view, the actual production surface) — DOM `data-bar` inspection before/after, plus a full-page screenshot (`scratchpad/autumn_bars_after_fix.png`) as the required inspectable artifact. Regression-checked 3 other live songs (abba chiquitita, aretha chain of fools, blue bossa) post-restart: all render, zero JS console errors, bar counts sane. Server log clean, no exceptions.

**What this does NOT solve**: the remaining 1 empty-next instance at offset≥1 (a case the single-pass heuristic doesn't reach, likely 3+ onsets stacked in one bar or a chain of consecutive near-boundary onsets — not investigated further this call, low volume). Does NOT touch or resolve the underlying rigid-constant-tempo-grid limitation itself (the actual per-song, per-passage phase drift) — this is a downstream mitigation on the bar-assignment/rendering layer, not a beat-tracking fix; the already-open "GRID PHASE MISALIGNMENT" entry's oracle-phase-selection / adaptive-hierarchy follow-ups remain the real long-term fix for the underlying grid. Also does NOT resolve the sample-rate-dependent tempo-octave ambiguity noted above (92.29 vs 184.57 BPM for this song) — flagged, not diagnosed further; whichever octave is true, this fix is octave-agnostic.

**Files changed**: `scripts/render_youtube_chart.py` (new `rebalance_near_boundary_onsets`, wired into `chart_to_interactive_inputs`), `scripts/harmonia_server.py` (`_apply_bar1_offset_to_payload` now imports and calls it on both the offset==0 and offset!=0 paths). No commits (per session convention). Live server restarted and verified (PID 75118, was 73124).

Time: ~70 min of the 2.5h budget for this call (investigation + corpus-scale-within-song verification + fix + playwright verification + regression check + this write-up).
