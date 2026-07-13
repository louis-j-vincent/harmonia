# Harmonia — full project overview, bias audit & the road to the principled Bayesian model

*Synthesis handoff, 2026-07-13. Produced autonomously from a 4-agent audit
(architecture/redundancy, history/dead-ends/bias-contamination, Bayesian-readiness
+ music-theory research, empirical bias tests). This is the "where are we really,
and do we have the ingredients for the model I'm dreaming of" document.*

Companion artifact: an HTML overview of the same material (browsable), published
separately. This markdown is the authoritative, git-tracked version.

---

## 0. TL;DR (read this first)

1. **The single most important finding.** You already *built* the principled
   model and then froze it. `harmonia/models/chord_hmm.py` is a coherent
   generative semi-Markov HMM — emission matrix, key prior, transition matrix,
   **explicit-duration Viterbi (`viterbi_duration_aware`)**, and forward–backward
   posteriors. It was frozen not because the *structure* was wrong but because
   its *emissions* (fixed chord templates) couldn't discriminate similar
   qualities (issue #1). Gen-2 correctly fixed the emissions (trained
   classifiers) — but in doing so replaced the probabilistic backbone with a
   **stack of greedy, hand-weighted rerankers**. **The dream is not a from-scratch
   build; it is re-coupling Gen-2's good emissions into the coherent decoder you
   already have.** The gap is ~1–2 weeks of focused work, not a rewrite.

2. **Ingredient readiness: PARTIAL-but-strong.** Every likelihood and every prior
   the Bayesian model needs already exists at good maturity. What's missing is
   (g) *joint* inference, (h) an *applied* calibrator, and (i) user-input-as-evidence.
   See §6.

3. **Calibration — the honest-confidence vision is currently unbacked.** Nothing
   in the pipeline applies a calibrator. The confidence the app shows is a raw
   max-softmax, measured only on clean synthetic audio, and — a real bug — it is
   **stale**: the rerankers flip the chord's quality but carry over the
   *pre-rerank* confidence, so the number displayed doesn't describe the decision
   displayed. See §7. This is the cheapest high-value fix and it directly serves
   the "show where it's unsure" UX.

4. **Biggest measurement caveat.** Every headline accuracy number from before
   2026-07-09 (the 88–96% figures) is **on synthetic MMA audio only**. On real
   YouTube audio the synth-trained classifiers fail badly — dominant-7th recall
   is literally **0%** (every dom7 collapses to major). The domain gap (#19), not
   the decoder, is the largest standing accuracy gap. See §5.

5. **Biggest *unaudited* risk.** Issue #1's entire A/B/C investigation ran on
   audio that was silently missing its second half (`BASIC_PITCH_FRAME_RATE` 2×
   bug) and, for two of three candidates, a mislabeled low-fidelity soundfont.
   Those sweeps were **never re-run at scope** after the fixes. The qualitative
   conclusion ("emission discriminability is the binding constraint") is
   independently corroborated and probably survives, but the *numbers* behind it
   are suspect and it is cited downstream as settled. See §4.

6. **Immediate safety item (do this regardless of anything else):**
   `harmonia/irealb_aligner.py` and `harmonia/irealb_fetcher.py` are **on the
   live server path but untracked in git.** Any branch/stash/worktree operation
   can destroy them. Commit them. See §8.

---

## 1. The two-generation story (the spine of the project)

| | **Gen-1 (FROZEN)** | **Gen-2 (CURRENT)** |
|---|---|---|
| Corpus | POP909 (real piano, rendered) | Synthetic iReal Pro charts → MMA accompaniment → FluidSynth (perfect GT by construction) |
| Chord model | Fixed chord templates, `beat_probs @ E.T` dot product | Trained classifiers (root / family / seventh heads) |
| Decoder | Coherent generative HMM: `chord_hmm.py`, incl. semi-Markov `viterbi_duration_aware` + forward–backward | Greedy classifier → stacked rerankers (diatonic, local-key, progression), argmax-and-overwrite |
| Entry point | `pipeline.py::HarmoniaPipeline` | `chord_pipeline_v1.py::infer_chords_v1` |
| Issues | #0–#8 | #9 onward |
| Best POP909 numbers | root ~33% / majmin ~29% | root 60.5% / majmin 39.1% |

The pivot (2026-07-04) was correct: template emissions were the binding
constraint, and a generative synthetic corpus with perfect labels was the right
way to train better emissions. **The unintended casualty of the pivot was the
probabilistic backbone** — Gen-2 never re-acquired a joint decoder, it accreted
rerankers instead.

---

## 2. Live architecture & import graph (what actually runs)

Live entry points, all routing to `chord_pipeline_v1.infer_chords_v1`:
`scripts/harmonia_server.py` (Flask, the production PWA backend),
`scripts/render_youtube_chart.py` (CLI), `scripts/evaluate.py` (offline eval).

Live inference stack (lazy-imported by `chord_pipeline_v1`):
`stage1_pitch` → `theory.key_profiles` (`infer_key`) → beat-seq root model
(`beat_seq_model_v4.npz`) → family/seventh heads + `ctx_v2.npz` (entropy-gated
MLP) → `theory.local_key` + `LocalKeySeqGRU` (`local_key_seq_gru.pt`) →
`progression_encoder` (`progression_encoder.pt`) → `section_structure`.
Output via `output.chart_interactive` (the ~3200-line HTML/CSS/JS PWA template).

Two dead-weight facts:
- Gen-2 imports `harmonia.pipeline` **only to reuse the `ChordChart` dataclass**,
  which drags the entire frozen Gen-1 stack (`chord_hmm`, `rhythm`, `structure`,
  `jazz_priors`) into every inference process. A one-line `ChordChart`-only shim
  would sever this. (Low priority — it's loaded, not run.)
- `harmonia/priors/` was an empty package (now deleted, see §3).

Live model artifacts (all ≤840 KB — disk is not a constraint here):
`beat_seq_model_v4.npz`, `beat_seq_model_v3.npz` (separate quality role),
`root_model.npz`, `ctx_v2.npz`, `progression_encoder.pt`,
`data/cache/local_key_seq_gru.pt`, `data/cache/ltas_family_dist.npz`,
`data/cache/audio_chord_features.npz`.

---

## 3. Redundancy & cleanup

**A crucial nuance before deleting anything:** the ~45 `experiment_*` /
`diag_*` / `bakeoff_*` / `check_*_premise` scripts are **not clutter** — each maps
1:1 to a dated, written-up finding in `known_issues.md`. That is the project's
documented "screen the premise cheaply, log immediately" working style. Deleting
them would erase the audit trail. Leave them.

### Executed (this session) — both verified zero-importer, git-tracked, reversible
- **Deleted** `harmonia/models/chord_pipeline_v0.py` — superseded by v1, zero
  importers anywhere. Recover with `git checkout -- harmonia/models/chord_pipeline_v0.py`.
- **Deleted** `harmonia/priors/__init__.py` (+ empty dir) — empty dead package.

Confirmed `chord_pipeline_v1` still imports cleanly after both removals.

### Proposed (NOT executed — needs your call)
| Candidate | Action | Why held |
|---|---|---|
| `migrate_*.py` (15 one-off UI-migration scripts) | **Archive** to `scripts/archive/`, don't delete | CLAUDE.md flags `chart_interactive.py` as having no other recovery mechanism for lost UI intent; keep them findable |
| `local_key_model.py` + `local_key_gru.pt` | Keep for now | Superseded on the inference path, **but still imported by a test + `train_local_key_model.py` + `eval_local_key_baselines.py`** — not pure dead code |
| `beat_seq_model.py` (v1) + npz | Delete candidate (med conf) | Unreachable fallback behind v4→v2; verify test coverage first |
| Orphaned artifacts (`chroma_root_template.npz`, `family_hard_oracle_model.npz`, `family_ltas_model.npz`, `local_key_gru.pt`, `change_detector.json`) | Keep | Tiny; several are intentionally-kept ablation evidence cited in known_issues #18 |
| `yt_chord_model*.npz`, `yt_online.npz` + trainers (untracked) | **Do not touch** | Tied to OPEN issue #19 (domain gap) — almost certainly current active work |
| `data/cache/accomp_varied/` (~347 MB stale, issue #15) | Your call | Real disk item (16 GB free). Blocked-regen cache; deleting is plausibly correct but is a data decision — left for you |

Not redundant despite appearances: `tab_aligner.py` vs `irealb_aligner.py`
(different GT sources per the trust hierarchy); `chart_render.py` vs
`chart_interactive.py` (static PNG vs interactive PWA).

---

## 4. Bias-contamination audit — have past bugs poisoned past conclusions?

Eight calibration/measurement bugs were found, several *late*. For each: were
earlier conclusions computed before the fix, and were they re-verified after?

| # | Bug | Conclusions computed before fix | Re-run after fix? | Verdict |
|---|---|---|---|---|
| 1 | `BASIC_PITCH_FRAME_RATE` 2× (dropped 2nd half of every song) | Issue #1 candidates **A, B, C in full** | Single-song spot check only | **Silently invalidated (unverified).** Numbers suspect; qualitative verdict probably survives |
| 2 | Mislabeled "Vintage Dreams" soundfont posing as GeneralUser | Session-4 baseline; A & B baselines | C re-ran; **A/B never did** | **Invalidated for A/B**, safe for C |
| 3 | Song 002/005 2× tempo-octave lock | Any metric on the audio beat grid for those songs | Not systematically corrected | Bounded/minor, diluted across the 5-song pool |
| 4 | Key posterior near-uniform (correlation used as log-likelihood) | Only `.confidence`, which no consumer read | **Explicitly re-checked** | **Safe** — model example of re-verification |
| 5 | Unnormalized (duration-scaling) family-emission features | Early Gen-2 majmin (~39–59%) | Fix immediate; new baseline | **Safe going forward**, superseded numbers documented |
| 6 | GT-source mismatch in the chord-change harness | "Oracle boundaries don't help", ~75% oracle-root | **Re-run, conclusion reversed** (true ceiling ~91%, segmentation costs ~17pp) | **Best self-correction in the project** |
| 7 | MMA-synth → real-audio domain gap | **Every Gen-2 number before 07-09** (88–96%) | Only a 50-song pilot | **Largest gap, openly flagged** — read pre-07-09 "%" as "synthetic only" |
| 8 | POP909 GT discards `/bass` inversions (10–18% of lines) | All POP909 root numbers (unfiltered labels) | Only the oracle sprint used `--exclude-slash` | **Known, un-retrofitted** — uncorrected penalty of unknown size in every POP909 table |

**What to actually do about it:** the highest-value re-verification is a single
clean re-run of issue #1's isolated per-beat and boundary-F metrics on the fixed
frame-rate + correct soundfont, to convert "probably still true" into "confirmed."
It is cheap (the harness exists: `scripts/experiment_issue1.py`) and it de-risks a
diagnosis the whole Gen-2 design leans on.

---

## 5. Where & why we're bad (hard numbers)

- **The accuracy cascade** (jazz1460, in-domain synthetic): root 88.7% →
  majmin 84.0% (−4.7pp) → **7ths 58.6% (a further −25.4pp cliff)**. POP909: root
  78.6% → majmin 73.6% → 7ths 41.8%. *Family credit is cheap; getting the seventh
  exactly right is the wall.* Keep reporting partial-credit alongside strict.
- **5th-apart root confusion is real but rescuable.** P4/P5 errors are ~46–51% of
  all root errors (worse on jazz walking bass). But the true root is in the
  model's own top-2 for 85.9%, top-3 for 95.2%, and recoverable via adjacent-beat
  argmax for 82.4% → **92.5% jointly rescuable**. The progression reranker only
  realizes **+1.0pp** because the acoustic prior fed to it is a confidence-gated
  one-hot that pins near-degenerate above conf ≈ 0.65 (issue #21). *There is a
  large gap between the rescuable ceiling and the realized gain — this is a
  fusion problem, not an evidence problem.*
- **Domain gap is severe and quality-specific** (issue #19): synth-trained
  classifier on real YouTube audio = 41.0% strict / 60.3% lenient quality
  accuracy, with **dom7 recall = 0%** (dominants collapse to major), min 46%,
  maj 87%. A 15–40pp absolute drop, non-uniform.
- **Diatonic bias:** GT diatonicity is 93.3% (POP909) vs 49.4% (jazz1460) — jazz
  reharmonization is genuinely out-of-prior. An *explicit* diatonic prior is
  neutral-to-harmful even on POP909 (inferred local key is too noisy to snap to).
  The one lever that clearly helped (+4.3pp family) is a key-relative *input
  feature* (commit 736b57c), currently a bootstrap upper bound (GT context at
  train time), not yet realized in production.
- **Not yet measured (worth a cheap follow-up):** a fresh Gen-2 quality-confusion
  matrix (the near-degenerate-template finding, issue #5, is about the *frozen*
  template scorer, not Gen-2's learned classifier); a real confidence
  reliability diagram; the non-P4/P5 interval error breakdown.

---

## 6. Ingredient audit for the Bayesian model

Target: a generative `P(chords, key, structure | audio)` factorized the way the
project already discovered — chord = (root, quality); root ≈ bass evidence;
quality ≈ chroma-template evidence; priors from key/diatonic, progression grammar,
duration, and repetition.

| Ingredient | Exists / maturity | Currently combined how | Principled? |
|---|---|---|---|
| a. root ← bass emission | **Yes** — BeatSeqModelV4, 93.3% per-beat CV, real (n,12) posterior | sum→argmax, independent of quality | factorized-then-argmax; no shared normalizer with quality |
| b. quality ← chroma emission | **Yes** — family LR + ctx MLP + seventh head; real q5 logprob | base+ctx via entropy gate | leaf logprob real; fusion heuristic |
| c. key + local-key | **Yes** — `infer_key` (proper multinomial post-#0), `LocalKeySeqGRU` | 2-candidate log compare, boost=4.0 hand-tuned | log-linear but w=4 is an unnormalized potential |
| d. progression transition | **Yes** — bigram JSON + `ProgressionEncoder` | post-hoc quality-only reranker, w=2.0 | greedy; not a transition factor in a joint decode |
| e. duration (semi-Markov) | **Yes but ORPHANED** — `duration_prior.py` + `viterbi_duration_aware` | **not used by Gen-2 at all** | correct and idle |
| f. structure / repetition | **Partial** — `section_structure`, `motif`, `block_fold` | labels sections; doesn't *tie* repeated slots | fold is decision-level |
| g. coherent joint inference | **Absent in Gen-2** (frozen in `chord_hmm`) | classifier → 3 rerankers in series, each overwriting the last argmax | **the central gap** |
| h. calibration / UQ | **Partial** — `plot_calibration.py` measures ECE on OOF | nothing *applies* a calibrator | see §7 |
| i. user-input-as-prior | **Absent as a model mechanism** — annotator UI exists | edits override the *display*, not the model | terminates at display layer |

**Verdict: you have every component; you are missing the coupling (g), the applied
calibrator (h), and the user-evidence channel (i).** None of the three is a new
research problem — all three reuse parts already in the repo.

---

## 7. Calibration verdict — NOT met (measurement exists, honest confidence doesn't)

Five concrete problems, all fixable:
1. **Nothing applies a calibrator.** The ECE/reliability scripts *measure* but no
   saved temperature/isotonic map is *consumed* at inference.
2. Displayed `conf` is a **raw max-softmax**.
3. It is **stale** — the rerank loop writes the flipped quality but carries the
   pre-rerank confidence (`chord_pipeline_v1.py` around L2085/2102). *The number
   shown does not describe the decision shown.* This is a real bug.
4. It is **quality-only** — root and quality confidences are never fused into a
   single per-chord posterior.
5. It is measured on **clean MMA**, so almost certainly overconfident on real
   audio (issue #19).

This is load-bearing for the entire collaborative-app premise ("the AI tells you
where it's sure and unsure so you can trust it"). Fixing it is the cheapest,
highest-leverage item and can ship independently of the joint-decoder work.

---

## 8. The clean model — design

**Generative story.**
```
K            ~ P(K)                              global key
R(form)                                          repetition/section structure
for each slot i:
    (r_i, q_i) ~ P(C_i | C_{i-1}, L_i)           progression transition, key-relative
    d_i        ~ P(d | q_i)                       duration (semi-Markov)
    L_i        ~ P(L_i | K, chords)              local key / tonicization
emissions:
    bass_i     ~ P(bass | r_i)                   root ← bass
    chroma_i   ~ P(chroma | r_i, q_i)            quality ← chroma
tied slots (repeats / user-merged) SHARE their emission likelihoods (pooling).
```

**Inference.** One semi-Markov Viterbi over the **root × q5 joint state** (≈60
states), reusing `chord_hmm.viterbi_duration_aware`, with every factor entering
at weight 1 **once each head is calibrated**. Per-chord confidence = the
normalized forward–backward **max-marginal**, isotonic-mapped — and it should
drive the display *depth* (family → seventh → exact) so the sheet literally shows
less detail where the model knows less.

**User inputs are factors in the same graph** (this is the payoff of joint over
stacked):
- **section-merge / "these are the same part"** → sum the tied slots' emission
  log-likelihoods (more observations per slot). This is the principled version of
  "push the merge feature further."
- **chord-confirm / edit** → clamp `C_i` with a delta/strong prior that
  **propagates to neighbours through the transition factor** — confirming one
  chord should sharpen the ones around it. A stacked reranker cannot do this; a
  joint decode does it for free.
- **set-key** → clamp `K` / `L_i`.

**The visualization *is* the factor graph.** For each chord, show two needles —
bass→root and chroma→quality — plus the prior's expected chord, colored by the
joint marginal; on an edit, highlight the neighbouring chords whose posterior
moved. The deferred radial/circle-of-fifths suggestion view
(`architecture_extensions.md` §1b) is the natural home for the per-chord marginal.
This makes "the visuals directly reflect how the model is built," as you asked.

**External validation of the target** (research pass, nothing installed):
Mauch & Dixon's DBN (one Bayesian net over metric-position + key + chord + **bass**)
is almost exactly this factorization already realized jointly. Masada & Bunescu's
semi-CRF (joint segmentation + labelling) is the upgrade path for the orphaned
`viterbi_duration_aware`. For calibration, temperature scaling (accuracy-preserving,
−45–66% ECE) applied to the forward–backward **marginals** (Kuleshov & Liang,
NIPS'15), not the raw softmax. `music21` is **already installed** — use
`roman.RomanNumeral` to turn the hand-built `_DIA_*` dicts into real functional
conditional tables for free. Candidate emission front-ends worth a *gated* trial
for the real-audio gap (issue #1 / #19): Chordino/NNLS-chroma and `crema`
(602-class, with bass tracking) — but both need installs, so behind a disk +
premise check per CLAUDE.md rule #2.

---

## 8.5 The unifying principles (from the 2026-07-13 note) — what ties it all together

The design in §8 is the skeleton. These five principles are the *nervous system* —
they say how the levels should relate. The encouraging finding: **each is already
partially built somewhere in the repo**; the work is to make them global and to let
the levels talk, not to invent them.

**P1 — Transposition-equivariance everywhere (the only thing that changes between
keys is the transposition).** Chroma shapes, bass walks, progression n-grams, and
structural repetition are *identical across keys up to a pitch-class shift*.
Represent every latent variable and every piece of evidence in a **key-relative
(scale-degree / pitch-class-relative) frame**, and learn in the quotient space mod
the Z₁₂ transposition group. A single `ii–V–I` seen in C then teaches every key —
a large data-efficiency and generalization win.
- *Already built:* `progression_prior.py` states are `((root−tonic)%12, family)`
  — scale-relative. `progression_encoder.py` is root-relative "transpose-invariant
  by construction." Classifiers use root-shift augmentation. The key-relative ctx
  feature (commit 736b57c) gave +4.3pp family.
- *Gap:* it's applied **per-component with inconsistent references** (some to the
  global tonic, some to local key, chroma/bass emission and structure not in a
  key-relative frame at all). Make the **local key the single transposition
  reference at every level**; the *only* key-dependent object left is the one map
  from key-relative degree back to absolute pitch for display.

**P2 — Cross-level message passing (the levels talk both ways).** Four levels:
acoustic (chroma + bass) → chord (root × quality, hierarchical) → progression
sentence (scale-degree n-gram grammar + local key) → form/structure (repetition +
global key). Bottom-up, evidence sharpens chords, chords score progression
likelihood, progression consistency scores structure. **Top-down**, structure
identifies repeats and pools/constrains chords, the progression grammar constrains
which roots/qualities are plausible at each slot, and the local key constrains the
degree. This is loopy belief propagation / hierarchical structured inference over
the whole song — the concrete form of gap (g), with the multi-scale structure made
explicit. It replaces the current one-way `classifier → reranker → reranker`
pipeline, in which information can only ever flow forward.

**P3 — Parallelism as denoising (the √N win).** When repetition is detected (the
SSM) *or* asserted by the user (a merge), **tie the corresponding chord slots and
pool their chroma/bass observations** — the same chord inferred from N snippets,
variance down ~1/N. This is the single mechanism behind both auto-repeat and the
user's "these two parts are the same." *Critical lesson already paid for:*
Candidate C (blind cross-repeat averaging) **hurt** `majmin`, because it averaged
genuinely-different reharmonized repeats. The new framing fixes this: **pool only
where cross-repeat agreement — or a user assertion — confirms the slots are truly
the same.** Parallelism is a *gated, confirmed* tie, never an unconditional average.
The fold machinery (`periodicity.py`, `block_fold.py`, `motif.py`) is built and
tested; it just needs to be gated instead of blind, and wired to the tie.

**P4 — N-grams at every level → the minimal logical sentence.** Reduce a song to
its canonical harmonic skeleton: dedup repeated sections into one representative
progression, express it as a **short scale-degree n-gram "sentence" of chord
shapes**, infer on that compressed sentence (more evidence per token, shortest
sequence, grammar strongest), then expand back to the full timeline. Bi/tri-gram
grammars already exist at the chord level (`progression_prior`, the trigram
section-phase model); the extension is a grammar **at the section level too** (a
language over functional phrases / section types), so the same "logical sentence"
idea operates at every scale.

**P5 — Chords as a hierarchy, conditioned on their key.** `chord_tree.py`'s
family → seventh → extension is the complexity axis; the quality prior is a
**function of scale degree in the local key** (a ii tends min7, a V tends dom7, a I
tends maj/maj7). Inference is coarse-to-fine: commit to the family where the
evidence is strong, descend to seventh/extension only where the *calibrated
marginal* supports it — which is exactly what should set the **display depth**
(a bare family symbol where the model is unsure, the full altered chord where it's
sure). This unifies the complexity hierarchy, the key-conditioning, and calibration
(§7) into one rule — "report at the level the evidence supports" — which is
`chord_tree.py`'s stated motivation but is **not yet wired into the Gen-2 path**.

**How this changes the build order:** P1–P5 are *cross-cutting invariants*, not
separate steps. Step 2 (the joint decode) becomes explicitly **multi-level with
top-down messages**; the decode runs in the **key-relative frame** (P1); the
duration/emission pooling for tied slots (P3) is *the same code path* as the
user-merge factor in step 4 — auto-parallelism and user-parallelism differ only in
what opens the gate. Step 5's chord-tree depth (P5) is where calibration meets the
display. Practically: enforce P1 as you touch each factor in step 2 (cheap, mostly
already there), and build P3's gated tie once — it serves both structure denoising
and the collaborative merge.

---

## 9. Ranked build order (with try-order and stopping criteria)

1. **Calibrate + pipe the real posterior to the app. (½–1 day; do first.)**
   Fit temperature/isotonic on the OOF preds the reliability infra already
   produces; display `P(root)·P(q5)` normalized; **fix the stale-confidence bug
   (§7.3).** *Stop when* displayed ECE < 0.05 on held-out. Delivers "show where
   it's unsure" on its own, independent of everything below.
2. **Collapse the 3 rerankers into one joint semi-Markov decode. (2–4 days.)**
   Gen-2 emissions → `log_emission`; progression + diatonic → `log_transition`;
   duration prior → the semi-Markov term; run `viterbi_duration_aware` over
   root×q5. *Gate (CLAUDE.md rule #6):* diff all intermediates and require
   end-to-end ≥ current on the irealb held-out before switching the default.
3. **Learn the factor weights. (1–2 days.)** Max-likelihood on the **audio-free**
   symbolic GT chord sequences (music21 for functional tables). *Premise-check
   first:* does a learned bigram-CRF beat the hand-tuned rerankers? If not, stop.
4. **User-input factors. (3–5 days.)** chord-confirm = clamped emission + re-decode;
   section-merge = pooled emissions; wire to the annotator UI; visualize the
   posterior propagation. This is the collaborative-app core.
5. **Close the calibration loop on real audio. (ongoing.)** Re-fit temperature on
   a small real-recording set (issue #19); add reliability to the nightly run.

Parallel, independent: the issue #1 re-verification (§4) and the real-audio
domain-gap corpus (issue #19) can proceed on their own tracks.

---

## 10. Immediate action items

- [ ] **Commit `harmonia/irealb_aligner.py` and `harmonia/irealb_fetcher.py`** —
  live-path files, currently untracked, one git op away from being lost.
- [ ] Decide on the `migrate_*.py` archive move and the `accomp_varied/` cache (§3).
- [ ] Green-light build-order step 1 (calibration + stale-conf fix) — highest
  value per hour, ships independently.
- [ ] Optional: the cheap issue #1 re-run to close the largest unaudited-bias risk.

---

## Appendix — catalogue of dead ends (so they aren't re-tried)

Issue #1 candidates A (L1-norm — provably inert; sqrt/log1p helped isolated
metric, hurt pipeline), B (explicit-duration HSMM — worse at every blend), C
(periodicity folding — cleanest-signal song regressed most); harmonic-rhythm
period-per-section (falsified corpus-wide, 92% of sections have no clean period);
four zoom-to-beat segmentation schemes (perfect boundaries didn't move accuracy);
drum-structure prior (untestable on single-groove MMA); bass-stem separation (no
better than full-mix low register); motif decision-level voting (null→negative);
global bigram progression prior (premise MARGINAL 63.8% < 70% gate); isolation-gap
bass detector (no usable threshold); 9th/11th/13th extensions (only 10.7% of
chords carry them, passing-tone energy floods those PCs anyway); local-key GRU as
a jazz diatonic reranker (jazz is ~49% diatonic); v3 dominant-chain consolidation
as a classifier feature (erased the functional cue); |DFT|-magnitude quality head
and joint root×quality single-softmax (both lost to simpler splits);
GarageBand/iReal Pro as renderers (no programmatic interface).
