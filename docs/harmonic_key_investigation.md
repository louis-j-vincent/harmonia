# Harmonic-key investigation (branch `feat/harmonic-key`)

Goal: given NNLS chroma (+ section folding later), detect which *scale* the
song is in at each moment — e.g. This Love alternating C harmonic minor
(B natural, on the G7 bars) vs C natural minor (Bb).

## v1 — binary scale masks vs per-bar chroma (2026-07-30)

Script: `scratchpad/key_scale_dotprod.py` →
plot `scratchpad/key_scale_dotprod_this_love.png`.

Setup: per-beat NNLS treble chroma (C-first, live `pool_beats`), pooled to
4-beat bars (anchor chosen to align bar lines with the baked chart's chord
onsets: offset 2, 74/117 onsets within 120 ms). For each C-minor variant
(natural / harmonic / melodic / dorian), score = share of the bar's chroma
mass inside the 7-pc mask.

Findings:

1. **The alternation Louis hears is clearly recoverable** — but the signal
   lives almost entirely in the two discriminating pitch classes, not the
   full mask. Per-bar Bb-vs-B share flips exactly with the chart's G/G7
   bars (B ≈ 0.20–0.29 there, ≈ 0.02 elsewhere); verse 1 is solidly
   leading-tone (harmonic-minor colour), the Bb-chord sections solidly
   natural. B > Bb in 40% of bars; A > Ab in 32% (F-7/C-7 bars).
2. **The full 7-pc mask dot product is diluted**: variants share 6 of 7 pcs,
   so mean shares sit within 0.05 of each other (natural .768, harmonic
   .747, dorian .735, melodic .714) and the top-panel curves overlap. A
   detector should score the *contrast pcs* between hypotheses (pairwise
   likelihood-ratio), not the whole mask.
3. **Noise floor**: ~11.6% of chroma mass is outside *every* C-minor variant
   (Db+E+Gb) — NNLS bleed + vocals. Any threshold must clear this.
4. Cosine (Louis's normalized-dot formulation) vs L1 mass share: r = 0.65–0.91
   per variant — same information, L1 share easier to read as "% in scale".

Design calls (Louis, 2026-07-30): key stays C minor; track the *colour* per
chord; include the bass half; sticky HMM first, section folding after.

## v2 — per-chord colour via sticky HMM (2026-07-30)

Script: `scratchpad/colour_hmm_this_love.py` →
plot `scratchpad/colour_hmm_this_love.png`.

Model: states = {natural, harmonic, dorian, melodic} = joint setting of the
two contrast degrees (Ab/A × Bb/B). Per chord span (chart t0/t1): chroma =
L1(treble) + L1(bass), evidence per degree = (mass on the pair, raised
share), Bernoulli-style emission (Q=0.85) weighted by GAIN=25 × mass,
Viterbi with STAY=0.92. Chords with no contrast-pc mass are carried by
stickiness ("hold until forced", same philosophy as local key).

Results (120 chords): natural 25, harmonic 40, dorian 55, melodic 0;
11 switches; 6/7 G-root chords get the raised 7th.

1. **The 7th-degree axis is crisp and matches the ear**: verses decode
   harmonic (B share 0.7–0.93, big evidence), flat-7 sections decode
   clean. This axis is reliable.
2. **The 6th-degree axis is the weak one**: through the chorus the A share
   hovers 0.55–0.65 with modest mass, so the HMM calls **dorian**, not
   natural. Two readings: (a) the chorus really has F major/F7 (A natural)
   and the chart's `F-` label is wrong — then this is a genuine catch; or
   (b) A is harmonic-series bleed (A = 3rd partial of D, and D is in every
   Bb chord). Needs Louis's ear / iReal to adjudicate.
3. **The bass half matters a lot**: 44/120 chords change colour without it;
   it is what pushes the chorus to dorian.

Open: make 6th-degree evidence chord-aware (only count it when the sounding
chord can contain a 6th degree) or raise its evidence bar; then fold
evidence by section group.

## v2b — ear check + bass-as-sounding-note fix (2026-07-30)

**Ear GT (Louis)**: chorus F at ~42.7s is definitely **F minor** → the v2
dorian chorus was a false positive. But the *last 4 bars* of the B section
do have **F major** → dorian there is real. (Chart agrees: it writes `F^7`
at 57.9 / 108.4 / 158.9 s — the end of each B occurrence — and plain `F` at
179.1 / 199.3 s.)

**Mechanism found** (diagnostic in the script): on the F- chord the *bass
half's* top pitch classes are A (0.14) and E (0.14) — partial-series junk
(A is the 5th partial of F) — while the treble correctly has Ab ≥ A. Adding
the bass half as a mass blob (v2 `l1` mode) injects a fake A natural under
every F bass. **Fix**: bass enters as the *sounding-bass pc only* (argmax
one-hot, +0.3 mass) — the same way the live pipeline uses the bass half
(`nnls_features.py`: "UNTRAINED argmax"). Ledger note: ear-correction →
general rule, *never use the NNLS bass half as mass; argmax only*.

Decode comparison (120 chords):

| colour   | argmax (new) | l1 (v2) | treble-only |
|----------|-----|-----|------|
| natural  | 78  | 25  | 69   |
| harmonic | 27  | 40  | 36   |
| dorian   | 11  | 55  | 15   |
| melodic  | 4   | 0   | 0    |

Chorus now decodes natural (matches ear); verses stay harmonic.

Remaining gaps:
1. Of the 5 chart F-major chords, only #102 (179.1s) flips to dorian — a
   single-chord inflection loses to stickiness (2 switch penalties ≈ 7).
   Design question: should a one-chord borrow flip the sticky track, or be
   a per-chord *inflection flag* layered on the prevailing colour?
2. A 4-chord **melodic** block appeared near the bridge (around Ab→G7) —
   raised 6+7 simultaneously; check real vs bleed.
3. Treble-half bleed remains on Bb chords (A = partial of D; #22: treble
   A=.050 vs Ab=.016) — feeds the pre-chorus dorian patch. Chord-aware
   evidence is the likely guard, but the 6th-degree pair is genuinely
   quality-entangled on Bb roots (Bb7 has Ab, Bbmaj7 has A).

**Ear GT addition (Louis)**: the chart's `B-` chords (bars 33 and 51) are
chart errors — the sounding chord is **G**. This is exactly the class of
error the harmonic prior should eventually repair (B minor is nonsensical
in C minor; the colour tracker reads raised-7 there, consistent with G).

## v3 — chord-gated evidence + relax-to-natural + inflection layer (2026-07-30)

Same scripts. Three changes, in Louis's priority order:

- **Fix 2 — chord-gated evidence**: a degree's chroma pair is only counted
  when the chord symbol *voices* that degree (quality templates; F/Dh7/Ab
  chords gate the 6th, G/Bb/Eb/C-7 chords gate the 7th). Plus a **bass
  decisiveness guard** (`BASS_MARGIN=1.5`): the sounding-bass one-hot is
  only injected when the bass half's top pc clearly beats the runner-up —
  on This Love's F- the bass half reads A/E/F nearly tied and argmax
  picked A, re-injecting the v2 fake-A bug through a pinhole.
- **Fix 1 — relax-to-natural prior**: each raised degree costs
  LAMBDA=0.25 per chord (emission prior) and the transition matrix is
  asymmetric (raised→natural 0.12 cheap, natural→raised 0.02 expensive).
  Raised notes are guests of the chord that brings them.
- **Fix 3 — inflection layer**: chords whose own gated evidence clearly
  contradicts the prevailing colour (weight ≥1, share margin ≥0.15) get a
  borrowed-colour flag; drawn as a thick underline on the lead-sheet.

Results: natural 97 / harmonic 22 / dorian 1 / melodic 0; 13 switches.
The melodic latch is dead (the Ab chord's gated flat-6 turns the Ab→G7
zone harmonic, which then decays). Chorus starts are natural. The single
prevailing-dorian chord is the real F major at 179.1s. Flags: the three
F^7s (57.9/108.4/158.9s) → dorian, G7 at 137.4s → harmonic, and F- at
93.2s → dorian (**suspicious — ear-check candidate**).

Still open: F at 199.3s (outro, weak evidence) gets neither flip nor flag;
gate quality-templates trust the chart's chord quality (circular if the
quality is wrong — acceptable for now, the B-→G repair is future work).

## v3.1 + v4 — Louis's session directives, implemented (2026-07-30)

Directives (in his words, paraphrased):
- No bass in colour evidence unless it clearly helps (bassists go
  off-diatonic). → Ablation: 0–2/120 decisions changed, none better.
  **BASS_MODE="none" is the default.**
- Fix 2 must not let the inferred chord supersede the harmonic prior —
  the endpoint is the prior correcting wrong chords. → The challenger
  audits chords non-diatonic to every colour on RAW (ungated) chroma;
  circularity broken by alternation, not softening the gate.
- Fix 1 hold-vs-decay is a *zone de flou* (his ear holds the last colour);
  keep the mild decay for now.
- The chord model's confidence must be part of the story: when it is
  unsure, section structure / harmonic priors supersede (F- at 93.2s is
  真 F minor, context carries it; outro F at 199.3s is inaudible in the
  fadeout, structure must supply it).

v4 = structure fold: chord-level colour evidence and raw chroma pooled
across same-slot occurrences of `section_vocab.vocab_sections` items
(the detector designed with Louis on this song). Form recovered:
`A×4 B×3 C A×3 B×3 C A D B×3 E B×3 E B×3 E` — the target spec modulo one
C/E swap near 158.9s. 119/120 chords slotted.

Results vs his three asks:
1. **#50 F- (93.2s) false dorian flag: gone** — pooled with its 15-strong
   chorus-slot sisters.
2. **#117 outro F (199.3s): dorian flag appears** — inherits the E-slot
   evidence from 179.1s. All five F-major moments now marked dorian
   (2 prevailing + 3 flags).
3. **B- (81.9s) audit sharpened: top-3 = G 0.163, G^7, G7** vs written
   0.137 — still "suspect" (1.19× < the 1.25 auto-challenge bar).
   B- (127.3s) stays: the audio there genuinely has B share 0.42 and the
   chart's own confidence peaks (0.58) — needs an ear-check, may be a real
   B-rooted sonority in the bridge.
4. Chart confidence (`lv.exact.c`) is in the audit report but does NOT
   separate the false flag from real ones on this song (all 0.39) — the
   structure fold is what does. Confidence-scaled challenge thresholds
   remain open.

Bonus: Ab-7 (80.6s) challenged →Bb after folding (proposal machinery is
crude — target spec suggests the slot is really Dø; mean-per-tone template
scoring favours triads and needs work before trusting proposals beyond
"this chord is wrong").

Next: generalise beyond This Love (corpus premise check, CLAUDE.md rule
#5), and decide where this lives in the live pipeline (post-decode audit
pass feeding chord corrections + colour track to the chart).

## Bridge B- verdict: the chroma is honest, the G is *implied* (2026-07-30)

Louis ear-checked: the B- at 127.3s is "definitely a G". Diagnostic
(`scratchpad/b_minor_bridge_diag.py` + heatmap PNG): during 127.3–129.8s
the signal is **B 0.45 + F#(Gb) 0.21 in the bass half, B 0.25 + F# 0.16
in the treble — and G is nearly absent from both** (0.06). musx
independently labels the same span `B:dim`. So both transcribers root it
at B because B (plus its 3rd-partial/power-chord fifth F#) is *all that
is sounded* — a bare chromatic bass B walking Eb^7 → **B** → C-7. The G
Louis hears is the **function**: bass B a semitone under the C- landing,
in harmonic colour, is V6 (G/B) with the G supplied by the ear, not the
band.

Consequence: no chroma-scoring fix can output G here — G isn't in the
audio. The repair needs a **functional grammar prior**: (sounded bass pc,
current colour, resolution target) → chord function; bass-B resolving to
C- in C minor ⇒ relabel G/B. Also refines the confidence rule: the
chart's 0.58 confidence was confidence in the literal notes (which were
right); label confidence ≠ functional correctness.

## v4.1 — centre-weighted span pooling (Louis's transition rule, 2026-07-30)

Louis (voice note): distrust the bass for chord identity (the bridge
bassist played B and its fifth F# — confirmed excluded from both colour
evidence and audits); and weight the chroma toward the middle of each
chord's time window, because the edges carry transition notes.

Measured before adopting: Eb^7's last third holds the *next* chord's B at
0.11 (vs 0.00 first third); Dh7's F jumps 0.07→0.17 toward the coming
F-7; F-7's G 0.03→0.11 toward Eb^7. Edge contamination is real. Hann
window adopted (`CENTRE_POOL`).

Effects: **all five F-major chords now decode prevailing dorian** (was
2+3 flags); verse Bb^7s become natural-inflection *flags* over a harmonic
verse (musically the right reading of bVII colour); Ab-7→Bb challenge
sharpens to 2.2×; B- 81.9s →G steady (1.19×). The bridge B- (127.3s) is
untouched **by design**: B/F# hold all three thirds of its span, G absent
even mid-frame — centre-weighting cannot arbitrate toward a note that was
never played; only the functional rule (bass B resolving to C- ⇒ G/B)
can relabel it. Its suspect proposal is now B° (mid-span F counts more).

## v5 — functional grammar rule, V6→i (2026-07-30)

`functional_repairs()` in the colour script: when a chord (a) fits no
C-minor colour, (b) has a *decisive* sounding-bass = the leading tone
(bass-half argmax at ≥1.5× runner-up — the bass NOTE is allowed here even
though bass MASS is banned; the walkup is the object), and (c) resolves to
the tonic chord, relabel it V in first inversion. The V template is scored
WITHOUT its root (the un-played note the ear supplies), and only **parity**
(≥0.9× the written chord's score) is required — B- vs G/B differ
spectrally only by the bass note's own partial (F#), so chroma can never
win that comparison; the functional pattern is the tiebreaker (Louis's
confidence-supersedes rule).

Result: **both B- chords repaired — 81.9s → G/B, 127.3s → G7/B** —
matching Louis's ear ("peut-être que c'est un G sur B"). Rendered as solid
→G/B frames on the lead-sheet.

NOT solved by v1 of the rule: only the V6→i pattern fires (next root ==
tonic). Bass walkdowns to bVI, ii–V bass motion, deceptive resolutions —
none handled. The Ab-7→Bb challenge proposal also remains crude
(template-mean favours triads; the slot is likely Dø per the target spec).

## v6 — mode audit before chord audit (2026-07-30, generic script)

`mode_audit()` in `scratchpad/colour_hmm_song.py`: measure the 3rd-degree
contrast (b3 vs 3) on the spans of **tonic-rooted chords** (their third IS
the mode; a modulation section can't poison them — the broad all-chords
gate read Close to You at 0.46 because Db-land floods rel-pc 3 with
Eb = 2nd-of-Db). Fallback to the broad gate only below mass 1.5
(measured: Close's 12 tonic chords carry mass 2.47 with a clean 0.75).
The winning mode re-homes the relax prior (minor→natural, major→melodic
= the major scale) and the colour-scale third everywhere.

Results, 3/3 correct: This Love minor (0.14), Chain minor (0.44 — the
blues-third colour is real but sub-majority, matching UG's Cm), **Close
to You major (0.75) — contradicting its own baked payload**, exactly the
auditor known_issues asked for.

Honest remainder: major-mode decode of Close is still dominated by the
un-modeled **modulation** (the Db section saturates the b6 axis → 26
"harmonic-maj" chords; audit eligibility barely moves, 36→35, because
Db-land chords are genuinely non-diatonic *of C* — they're a key change,
not chart errors). v7 priority is a tonic track, not more states.

NOTE — baseline shift: the concurrent session re-baked This Love (and
Misery) at 18:32 (commit d3f0bae + uncommitted changes; new chords, new
times, includes a 2-chords-per-bar chorus fix). All v4.x/v5 numbers in
this doc refer to the OLD payload; current This Love folded counts under
the same model are natural=104 harmonic=12 dorian=5. My v6 edits were
A/B-verified behaviour-neutral on the committed script before rebaselining.

## v7a — tonic track: hold-until-forced from chroma (2026-07-30)

`tonic_track()` in `colour_hmm_song.py`. Two-stage design, each stage
doing the one thing it is sharp at:

1. **Boundaries** by CUSUM on the *forbidden-mass* contrast — energy on
   the two pcs outside every colour of a tonic (b2, #4). Rival tonics
   accumulate advantage; a switch fires only after a sustained > TT_PEN
   run (hold-until-forced, causal, adapts the
   `continuity_scale_track_v2` doctrine to chroma — that function needs
   trustworthy TOKENS, which are exactly what's under audit here).
2. **Labels** by full Krumhansl profile on each segment's raw summed
   chroma (the 2-pc signal finds boundaries exactly but is too thin to
   name a tonic — it anchored This Love on Eb). Adjacent same-label
   segments merge.

Results vs the stated criteria — all three pass:
- This Love: flat C (0 false switches) ✓
- Chain of Fools: flat **C** — self-corrects the payload's wrong A ✓
- Close to You: exactly one switch at **98s** ✓; per-segment mode audit
  says major (x3 0.86/0.85); **audit eligibility 36 → 1** (target <8) ✓

Honest limitation: Close's segments label **G→Ab where the ear says
C→Db**. The verse's B7/Bm chords flood F# and starve F, so the
Krumhansl-preferred *collection* is the F#-one — G major and C major
differ by exactly that pc. For diatonicity judgments (v7's purpose) the
collection is what matters and the audits collapse correctly; naming the
*tonal centre* within a collection (C, not G) is a separate step —
`local_key._label_collection` already embodies that logic and is the
pointer for v7b. Not solved either: per-segment re-decode of colours/
flags/audits (currently only eligibility is recomputed per segment).

The "sliver" hypothesis was wrong at the bake layer — the real defect:
`scripts/render_youtube_chart.py` anchored bars one beat late vs the
harmony (beat_this downbeat anchor claims k≡2, harmony changes at k≡1),
so 75/120 This Love chords sat on beat 3 and every bar-opening chord
rendered as the tail of the previous bar. Fix: `harmonic_phase_correction`
— rotate grid phase when ≥55% of chords agree on one non-zero beat residue
and ≤15% sit on beat 0 (This Love: 62.5% vs 6.7%); kill-switch
`HARMONIA_HARMONIC_REANCHOR=0`. Red-first tests in
`tests/test_render_youtube_chart.py` (7/7). Corpus scan: fires on This
Love, Misery, one Let It Be demo only. NOT solved: baked charts need
re-analysis + server restart; section chips still one bar late
(fix belongs in chart_model.py, untouched — another session has a large
uncommitted diff there); upstream beat_this phase error remains.
