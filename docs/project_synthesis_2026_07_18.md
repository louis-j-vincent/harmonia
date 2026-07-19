# Harmonia — full-project synthesis: every approach tried, by category

*2026-07-18. Compiled from `known_issues.md` (838 KB ledger), the 2026-07-13
four-agent audit (`handoff_2026-07-13_project_overview_and_bayesian_plan.md`),
`SESSION_PRESENTATION_2026_07_17.md`, `COMMERCIAL_LICENSING_AUDIT_2026_07_17.md`,
the blog series (20 posts), and git history. Purpose: the single map for the
"experimental repo → trusted bricks" transition. Numbers cited are the
post-correction ones; where a headline was later invalidated, the invalidation
is what's recorded.*

---

## 0. What the app is

A collaborative chord-chart app: user supplies a song (YouTube URL), the
pipeline produces an interactive, playable chord chart (PWA served by
`scripts/harmonia_server.py`, rendered by `harmonia/output/chart_interactive.py`),
with per-chord confidence, section structure, play-along sync, and user
corrections that feed back into inference (chord-confirm, section-merge,
hand-drawn section labels). The differentiating premise: **honest uncertainty +
user-as-evidence** — the model shows where it's unsure and user assertions
propagate through the probabilistic model, not just the display.

## 1. The three eras

| Era | Corpus | Emissions | Decoder | Status |
|---|---|---|---|---|
| **Gen-1** (to 07-04) | POP909 renders | fixed chord templates | coherent semi-Markov HMM (`chord_hmm.py`) | FROZEN — emissions couldn't discriminate qualities (#1, #5) |
| **Gen-2** (07-04→07-13) | synthetic iReal→MMA→FluidSynth ("jazz1460") | trained heads (root/family/7th, BP48 features) | greedy rerankers → then joint root×quality semi-Markov Viterbi (#27) | Decoder shipped; **all pre-07-09 numbers are synthetic-only** |
| **Real-audio era** (07-09→now) | YouTube+iReal, Billboard, RWC, JAAH, GuitarSet | NNLS-24 chroma heads → **music-x-lab pretrained (zero-shot)** | musx root/quality/bass live default on `/api/analyze` | Current production |

Key inflection points:
- **07-09**: domain gap discovered — synth-trained models fail on real audio
  (#19). Every earlier 88–96% number is synthetic-only.
- **07-13**: the audit found the "dream model" (coherent Bayesian decoder) was
  already built and frozen; Gen-2 had regressed to stacked rerankers. Steps 1–2
  of the re-coupling plan shipped (calibration, joint decode, semi-Markov).
- **07-16**: **target redefinition** — root/bass = sounding bass pitch class
  (`sounding_bass_pc()`), not functional root (+4.4pp overall / +23pp on
  inversions just from asking the acoustically answerable question).
- **07-17**: NNLS-24 beats BP48 (+17pp root); then **music-x-lab zero-shot
  beats in-house NNLS-24** in a fair bake-off (+7.3pp root, +13.9pp joint raw);
  full replacement deployed, cascades all lost to it.

## 2. Approaches by category

### A. Feature front-ends (audio → features)
| Approach | Verdict |
|---|---|
| Basic Pitch BP48 (48-dim pitch activations) | Long-time default; beaten on every axis by NNLS-24; onset smear caps boundary F1 |
| **NNLS-Chroma 24-dim (bass⊕treble)** | +17.3pp root, +20pp quality-bal, +38.5pp bass-on-inversions vs BP48 (RWC, confound-clean CV). GPL plugin (licensing §5) |
| **music-x-lab ISMIR2019 (pretrained, zero-shot)** | Beats trained NNLS-24: root 0.874, quality raw 0.824, joint 0.780 on RWC. Now live default. Weak only on rare dim/aug/sus |
| BTC-ISMIR19 | Ran in bake-off, lost to music-x-lab |
| pYIN low-pass bass tracker | Corroborator: NNLS∩pYIN agreement → bass acc 0.906; disagreement → 0.407. Confidence gate, not primary |
| Bass stem separation (Demucs) | REJECTED — no better than full-mix low register; weights are non-commercial anyway |
| FluidSynth synthetic training data | REJECTED for quality (below majority floor — synth too clean), PARTIAL for root (84% of real baseline) |

### B. Chord labeling heads (features → root/quality/bass)
| Approach | Verdict |
|---|---|
| beat_seq root models v1→v4 | v4 shipped (93.3% per-beat CV, synth) |
| ctx entropy-gated MLP (family), seventh head | Shipped Gen-2; superseded on real audio |
| Root-relative rotation for quality | The single biggest quality lever (+15.6pp bal oracle-frame); erodes under predicted root; **top-k root marginalization** recovers it (0.719 vs 0.735 oracle) |
| Learned trigram context (neighbor root-posteriors as features) | Helps Billboard (0.714→0.735), **hurts RWC (−7.9pp)** — the recurring "LM prior dead-to-negative on real audio" pattern |
| maj/min cascade vs flat class-weighted head | Cascade +8.1pp raw / −7pp balanced. Deployment-dependent choice, documented in `quality_model_selection.md` |
| Key-relative ctx feature (commit 736b57c) | +4.3pp family (bootstrap upper bound; the one diatonic-flavored idea that worked — as an input feature, not a prior) |
| Extensions (9/11/13), \|DFT\| quality head, joint single-softmax | All rejected (rarity, passing-tone flooding, lost to simpler splits) |
| Bass→root correction rules | Bass READING solved (0.77–0.80); bass-as-root-corrector consistently net-negative (inversions only ~8–12% → most disagreements are root errors). 5 rejected variants documented 07-16 |

### C. Decoding, priors, sequence models
| Approach | Verdict |
|---|---|
| Greedy reranker stack (diatonic, local-key, progression) | The Gen-2 regression; progression reranker's +1.0pp was a bypass-harness artifact, **−3.6pp on the real path** (#25) — default OFF |
| **Joint root×quality segment Viterbi (#27)** | GATE PASSED, default ON (+2.2pp majmin jazz) |
| **Per-beat semi-Markov (explicit duration) decode** | GATE PASSED, default ON (all metrics up) — duration is the live lever |
| Grammar/transition factors (key-local bigram, encoder shallow fusion, density-ratio) | ALL dead ends, wired default-OFF; the transition slot stays empty |
| Diatonic prior | PASS premise on POP909 (93% diatonic), FAIL on jazz (49%); even on POP909 explicit prior ≈ neutral-harmful |
| Local-key GRU | Wired as reranker + ctx feature; jazz diatonicity limits it |
| LLM priors (Mission 5) | Glue wired; non-circular eval infeasible on this corpus |
| Boundary detection (learned MLP, F1 0.45→0.78) | Real sub-task win, **~0 predicted downstream payoff** (oracle boundaries don't move end-to-end accuracy — labeling is the limiter) |

### D. Ground truth, corpora, alignment (the recurring burn)
- **POP909**: discards `/bass` (10–18% of lines); functional-root only; research-only license.
- **jazz1460** (synthetic iReal): perfect GT by construction, wrong domain.
- **Billboard**: "81.7% quality" was a collapsed-GT artifact (#31 — head trained on majmin-collapsed labels); corrected full-vocab balanced acc 0.41–0.44; root 0.84 real. McGill NNLS ≠ our audio (0.890 vs 0.379 — the "integrity incident" conflation, caught by the user).
- **RWC**: current workhorse (bundled audio, 13204 chords, alignment-clean).
- **JAAH**: jazz; within-corpus NNLS>BP48 confirmed (+8pp root).
- **GuitarSet**: OOD probe — trained NNLS-24 head generalizes (0.955), untrained bass-argmax doesn't (0.583).
- **AAM** (isolated bass stems, perfect alignment): highest-value next corpus, **blocked on disk** (~44 GB needed, 7 GiB free).
- **Alignment validator (Mission 6)**: repeat-consistency z-score validator, 91% slip recall @ 4% FP synthetic; display banner shipped; training-filter awaits real-audio FP check (#30).
- **Beat-grid iReal→audio alignment** (Mission 1): FAILED the ±150ms gate (#20) — still no large trusted real-audio GT benchmark.
- Trust order when sources disagree: iReal Pro > tabs > model output.

### E. Structure / segmentation (the 07-17→18 campaign)
The stubborn result: **flat 8-bar blocking ("block8") statistically ties every
learned/clever alternative** — learned section-similarity encoders (whole-song
key normalization was the one real unlock, then multi-seed re-validation
demoted the win to a tie), grammar induction (RePair/Sequitur), hierarchical
multi-scale (+0.05 oracle ceiling exists but no unsupervised selector reaches
it), symbolic chord-only structure (bar-drift kills transfer), hand-crafted
chord-tone-distance similarity (also ties). **Grid phase misalignment** (bar-1
offset), not content matching, is the dominant V-measure loss — user-caught,
corpus-confirmed; two phase-recovery heuristics failed.
Real-audio SSMs have a structurally **elevated similarity floor** (not "noisy
iReal") — floor/multiplicative noise model converges, additive doesn't;
per-song adaptive-percentile threshold deployed for real audio only (costs
V_F on clean iReal — domain-specific patch, not a general win).
**Reframe that stuck: bar-merge as CHORD-ROBUSTNESS pooling** (√N denoising of
emissions across tied bars), not structure detection — real-audio pooling gave
+10pp quality on repeats (#28); two-tier AUTO/SUGGEST threshold τ_auto=0.96
nested-CV validated (~0.3% FP floor from feature aliasing — "never wrong" is
unachievable); k-NN candidate generation won the bake-off; a multi-merge
order-dependence bug was found+fixed. Intro detector: real positive (2.5× base
precision). Not yet wired to auto-apply (scope-guarded vs a parallel session).

### F. Calibration & confidence (the app's core promise)
Was: raw max-softmax, quality-only, stale after rerank, synth-fitted (#26).
Now: fused root×quality isotonic calibration, two-domain maps (synth ECE 0.037,
real ECE 0.007 — but the real map collapses confidence to ~base-rate 0.44,
honest and uninformative), stale-conf bug fixed. Remaining: real map is
root-blind (#29, display-only); refit on production confidence pending a
trusted real benchmark.

### G. User-in-the-loop
Constraint factors shipped (Mission 3): chord-confirm + section-merge as
factors with re-decode (`user_constraints.py`, `/api/reinfer`); annotator UI
v3/v4 (waveform beat-grid editor → chord editing); gt-align draggable
correction tools; correction log endpoint; hand-drawn section labels (07-17).
The silent-rejection bug (#33 — merge rejected but reported success) was the
cautionary tale. √N pooling on user-merged spans is validated on real audio.

### H. Beat / tempo
Beat tracking is NOT the accuracy bottleneck (#9), but the **2× tempo
octave-lock is unsolved and unsolvable blind** (blind disambiguator 3/8 vs
oracle 8/8; madmom doesn't fix it; both trackers land in-range and pick the
wrong multiple). Needs an external prior: style-conditioned tempo range, user
tap, or lead-sheet metadata. madmom itself is a licensing blocker (§5).

## 3. Cross-cutting negative-result patterns (do not re-try)
1. **Priors/LMs on real audio ≈ dead-to-negative** (bigram gate, trigram-RWC,
   diatonic, transition-slot factors, progression reranker). Evidence quality
   beats prior sophistication every time; the exceptions are *duration* and
   *key-relative input features*.
2. **Oracle wins don't survive prediction** (root-relative rotation, inversion
   degree, neighbor context, hierarchical structure ceiling) — unless
   uncertainty is marginalized (top-k root marg is the one success).
3. **Harness bypasses lie** (#25 reranker reversal, #11 GT-source mismatch,
   #31 collapsed GT, the 0.379/0.890 conflation) — only real-path, real-GT,
   multi-seed numbers count.
4. **Boundaries aren't the bottleneck; labels are** (oracle-boundary tests, twice).
5. **Structure: simple flat baselines tie clever methods**; phase, not content,
   is the loss.

## 4. Current production stack (as deployed 07-17/18)
`/api/analyze` default: audio → NNLS-24 extraction + **music-x-lab
root/quality** (zero-shot, durable clone in `harmonia/third_party/`), musx bass
opt-in, beat grid (madmom, librosa fallback), joint/semi-Markov decode wired for
the in-house path, section fallback + adaptive-percentile merge threshold on
real audio, two-domain calibrated confidence, PWA chart with section chips,
play-along, GT pills, hand-drawn labels.

## 5. Shippability blockers (ranked)
1. **Trust consolidation** — no single frozen "production truth": which model,
   which numbers, which eval. known_issues is an append-only lab notebook
   (838 KB), not a spec. Several headline numbers were later invalidated;
   everything downstream of an invalidated number needs re-derivation or a
   tombstone.
2. **No stable real-audio benchmark** — Mission 1 failed its gate; RWC is the
   de-facto eval but is pop, bundled-audio, oracle-boundary. The app's actual
   input (YouTube) has no trusted held-out set → every "it works" claim on the
   product path is qualitative.
3. **Licensing** — madmom CC BY-NC-SA (hard blocker, replaceable), NNLS GPL
   (SaaS-shape-dependent), music-x-lab weights training-data provenance,
   in-house weights trained on POP909/YouTube (research-only / ToS).
4. **Monolith architecture** — 9312-line Flask server mixing product routes
   with ~30 debug/research routes; 3562-line `chord_pipeline_v1`; ~4000-line
   HTML-template PWA; live-path files that were untracked; concurrent-session
   clobber risk. No brick boundaries, thin tests on the live path (#3).
5. **Strategic model question** — in-house line just lost to an off-the-shelf
   pretrained model. What Harmonia *owns* needs re-deciding: the user-evidence
   Bayesian layer, sounding-bass output, jazz vocab, and the app — or a
   renewed in-house emission effort (distillation, AAM bass training).
6. **Unsolved UX-level model gaps** — tempo octave-lock (needs user tap or
   style prior), structure phase misalignment, rare-quality coverage under
   musx (dim/aug/sus recall 0.14–0.37).

## 6. Open-bug ledger + restart-clean candidates (added 2026-07-19)

Restore point: tag `prod-workable-2026-07-19` (= `f9e70e0`, pushed).

### Concrete open/deferred fixes (from known_issues, verified current)
1. **Bar-grid tempo miscalibration** (07-19, deferred by design) — the single
   uniform grid (`chord_pipeline_v1.py` ~L2923) slips up to ~4 bars/song;
   67–97% of drift is a linear tempo-scalar error (librosa median-local ≠
   whole-song average). Safe staged fix specified: `beat_period_mode="bestfit"`
   behind a flag; blocked on bar-precise GT to verify.
2. **Real-audio confidence map is root-blind** (#29) — display shows
   quality-only confidence on the default path; refit machinery exists
   (`score_kind="fused"`), blocked on a real benchmark.
3. **Time signature hard-assumed 4/4** — silently wrong on every waltz/3-4
   jazz standard; no issue number, found in the 07-17 timing profile.
4. **Tempo octave-lock** — unsolvable blind (proven); needs style-conditioned
   prior, user tap, or lead-sheet metadata. Product decision pending.
5. **Auto-apply bar merges unsafe** — τ_auto=0.96 calibrated on symbolic data
   gives ~39% real-audio agreement (not ~98%); opt-in joint gate reaches 89.6%
   pooled precision, short of the 98–99% bar. Correctly not shipped.
6. **Structure k-selection & phase** — k≤5 length-prior heuristic (three
   matrix-intrinsic methods all lost to it); grid-phase misalignment still the
   dominant V_F loss; "two variations in one cluster letter" case confirmed.
7. **Crammed-bar rendering** (aretha 92% bars ≥2 chords) — NNLS decode churn +
   rendering, NOT the beat grid (measured); partially mitigated.
8. **Perf/UX** — musx 10–18 s cold is the long pole; key inference could
   surface at ~4 s with a reorder; progressive-analysis screen scoped, unbuilt.
9. **Zero tests on `pipeline.py`/`mirex_eval.py`** (#3); write-only Basic-Pitch
   `.npz` persist (no reader in repo); #15 disk-blocked cache regen.

### Underlying issues only scratched — restart-clean candidates
- **A. The time model (STRONGEST candidate).** Tempo scalar, octave lock, grid
  phase, bar-1 offset, 4/4 assumption, and beat-drift pooling failures are all
  symptoms of one un-designed brick: a single `np.arange` line that every chart
  consumes. Rebuild as a proper time brick (tempo w/ prior or tap, best-fit or
  piecewise period, downbeat phase, meter) with its own GT and tests.
- **B. No bar-precise real-audio GT.** The bar-grid fix is *unverifiable* today
  — Brick 0 must include downbeat/bar annotations (tap tool or madmom-DBN
  cross-check), not just chord labels, or chart-level bugs stay unfalsifiable.
- **C. Symbolic→real calibration transfer.** Every threshold calibrated on
  clean/symbolic data has failed to transfer (auto-tier, noise models,
  adaptive percentile). Process rule: decision thresholds are only ever fitted
  on real-audio data against the frozen benchmark.
- **D. Confidence pipeline.** Root-blind real map + base-rate-collapsed honest
  confidence = the app's core promise currently unbacked on its default path;
  rebuild inside the calibration brick, not as another patch.
