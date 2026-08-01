# When should the chord LM overrule the pipeline? — audit + rule

*2026-08-01, branch `feat/chord-lm`*

Louis: *"j'aimerais bien qu'il puisse intervenir quand on n'est pas sûr de
l'accord qu'on met"* — so we need to quantify how sure the pipeline is, how sure
the LM is, and where the crossover sits.

Short version:

1. **The pipeline's confidence field `c` is not one quantity.** On a folded song
   it is a repetition count wearing an acoustic posterior's clothes. This blocks
   the plan until fixed.
2. **The LM's confidence is trustworthy** (ECE 0.033, AUC 0.88; at p≥0.9 it is
   96.5% right).
3. **A top-3 shortlist is worth far more than a single override** — 92.8% vs
   79.5% containment.
4. **The ratio rule needs the pipeline's confidence to reach AUC ≥ 0.70** before
   it beats simply thresholding the LM. Below that it actively hurts.

---

## 1. Audit: what "confidence" means in `harmonia_min` today

Three different things, all written into the same `c` field:

| source | what it actually measures |
|---|---|
| `pipeline._segment_confidence` | mean musx triad-plane posterior over the chord's own frames — a real acoustic quantity, uncalibrated |
| `folding.py` (2 sites) | `min(0.97, 0.5 + 0.08 × n_obs)` — **a count of how many times the section repeated.** No audio involved |
| `span_rescore` margins | top-1-minus-runner-up under the lattice; documented in-code as "a display-only confidence proxy, not a scored quantity" |

Measured over the five charts in `harmonia_min/state/charts/`:

| chart | chords | carrying `n_obs` | `c` equal to a folding-formula value |
|---|---|---|---|
| Stand By Me | 77 | 6% | 6% |
| Close to You | 106 | 0% | 2% |
| Let It Be | 140 | 0% | 1% |
| She Will Be Loved | 108 | 0% | 1% |
| **This Love** | 102 | **79%** | **79%** |

On This Love, **79% of chords report `c = 0.97`** — the saturation value of the
folding formula, reached by any section seen ≥ 6 times. A gate on `c` would never
fire on that song, no matter how wrong the chords are. The audio never got a vote.

**A second blocker: there is no verified ground truth to calibrate against.**
`data/real_audio_benchmark/brick0_batch1.json` is `verified: false` — machine
proposals awaiting an ear. So the pipeline's confidence cannot be scored today,
only structurally audited.

## 2. The LM's own confidence IS meaningful

Held-out iReal charts, one slot masked at a time, binned by the model's max
probability:

| confidence bin | n | mean conf | actual accuracy |
|---|---|---|---|
| 0.2–0.3 | 394 | 0.257 | 0.254 |
| 0.4–0.5 | 999 | 0.453 | 0.414 |
| 0.6–0.7 | 1131 | 0.650 | 0.577 |
| 0.8–0.9 | 1799 | 0.854 | 0.783 |
| **0.9–1.0** | **10405** | **0.980** | **0.965** |

**ECE 0.033. AUC 0.880.** Slightly overconfident in the middle, essentially
honest at the top. 58% of all slots land in the top bin, where it is right 96.5%
of the time. This is directly usable as a gate.

## 3. Top-k: propose a shortlist, do not silently override

| | top-1 | top-2 | top-3 | top-5 |
|---|---|---|---|---|
| clean context | 79.47% | 89.28% | **92.77%** | 95.70% |
| 20%-corrupted context | 68.63% | 80.21% | **85.53%** | 90.45% |

**When the LM's top-1 is wrong, the truth is still in its top-3 64.8% of the
time** (53.9% with corrupted context). A UI that offers three candidates
recovers most of what a single override throws away, and it never regresses a
chord without a human saying so.

## 4. The deployment condition nobody had measured

Every earlier number gave the LM a perfect chart minus one slot. In deployment
its neighbours are the pipeline's output. Corrupting a fraction of slots to
simulate that:

| pipeline error rate | repair rate | damage rate | net@0.99 | net@0.95 | net@0.9 | net@0.8 |
|---|---|---|---|---|---|---|
| **0%** | — | 0.205 | **−28** | **−180** | **−368** | **−759** |
| 3% | 0.782 | 0.222 | +106 | +63 | −87 | −439 |
| 5% | 0.790 | 0.230 | +209 | +225 | +117 | −185 |
| 10% | 0.739 | 0.261 | +305 | +497 | +469 | +236 |
| 20% | 0.697 | 0.316 | +346 | +770 | +940 | +951 |

*repair* = true chord proposed at a corrupted slot; *damage* = something else
proposed at an already-correct slot; *net* = chords fixed − chords broken.

**The 0% row is the important one.** On a chart that is already right, applying
the LM at p≥0.9 breaks 368 more chords than it fixes. The LM disagrees with 20.5%
of correct slots — that is its false-alarm floor, and it does not go away.

So the intervention is only safe where the pipeline is *actually* likely to be
wrong. Break-even at p≥0.95 is around a **3–5% error rate**; at p≥0.8 it needs
~8%. This is exactly why the gate cannot be one-sided.

## 5. The rule, and the spec it implies

Louis's proposal — a ratio between the two confidences, no calibration required,
just monotone meaning. Tested by simulating a pipeline confidence with a
*controlled* discriminative power (AUC), tuning thresholds on val, reporting on
test, 20% simulated error rate:

| pipeline conf AUC | lm-only net | **ratio net** | two-sided net | ratio precision |
|---|---|---|---|---|
| 0.50 (worthless) | +990 | **+185** | +983 | 0.54 |
| 0.60 | +990 | **+583** | +992 | 0.61 |
| **0.70** | +990 | **+970** | +1008 | 0.68 |
| 0.80 | +990 | **+1293** | +1137 | 0.77 |
| 0.90 | +990 | **+1627** | +1409 | 0.84 |

Read this as a specification:

* **Below AUC 0.70 the ratio rule is worse than ignoring the pipeline's
  confidence entirely.** Dividing by a noisy number destroys signal.
* **At AUC 0.70 it breaks even.** That is the bar to clear.
* **At 0.80 it is worth +31%** over lm-only, at 0.90 **+64%**, and precision
  rises from 0.72 to 0.84 — meaning 84% of the chords we change are genuine
  improvements.
* The **two-sided** rule (`LM ≥ t1 AND pipeline ≤ t2`) is the safer form: it
  degenerates gracefully to lm-only when the pipeline's confidence is useless,
  and never underperforms it at any AUC tested.

## 6. What has to happen next, in order

1. **Make `c` mean one thing.** The folding formula must stop overwriting the
   acoustic posterior; carry repetition support as a *separate* field
   (`n_obs` already exists) instead of laundering it into confidence.
2. **Get a verified ground truth** so `c`'s AUC can be measured rather than
   assumed. Cheapest path: the Ultimate Guitar tabs already in `data/ug_tabs.db`
   (≥4.7★ is trusted per project convention) for the five state charts, or a
   short ear pass from Louis on a fixed slot sample.
3. **Then** place the current `c` on the AUC axis above and pick the rule.
4. Ship the **top-3 shortlist** first regardless — it needs no pipeline
   confidence at all, cannot regress a chord without a human click, and recovers
   65% of the cases where the LM's own first guess is wrong.

## Caveats

* The pipeline error model is a **simulation**, not a fitted error distribution;
  there is no verified GT to fit one from. The AUC axis is a spec, not a
  measurement of where we stand.
* Corruption is applied i.i.d. per slot. Real pipeline errors cluster (a bad
  section, a wrong key), which the LM's neighbours-based repair would find
  harder. The repair rates here are therefore optimistic.
* All LM numbers are on iReal charts — jazz-weighted symbolic material, not our
  own audio-derived charts.

---

# Addendum (same day): `c` measured against verified ground truth

Two follow-ups from Louis: (1) how is a chord actually re-inferred when a section
repeats, and (2) drop the UG-tab idea, use a properly verified dataset.

## A. What folding actually does (read from `folding.py`, not from its prose)

`_template_chords` — the chord is **re-inferred acoustically**, not voted on:

1. For each position *k* of the detected period (This Love's A = a 4-bar loop),
   gather every bar sitting at that position across every occurrence.
2. **Average their musx frame posteriors** — `np.mean` over members, per frame.
   This is the actual noise reduction: *n* independent noisy observations of the
   same harmony, averaged, so noise falls ≈ √n.
3. Concatenate the *P*-bar template and **tile it ×3** so Viterbi edge effects
   land on the copies, not the middle.
4. **Re-run the musx beat-grid decode** (`_musx.redecode`) on the averaged
   posteriors, with half-bar transitions costing the same as bar transitions
   (15, 15, 100).
5. Keep the middle copy's events, write those chords onto every contributing bar.

The important consequence: **a chord that no single occurrence decoded correctly
can still emerge from the average.** It is not a majority vote over labels — it
is a fresh decode of better evidence. So `0.5 + 0.08·n_obs` is a defensible
*proxy* for that √n gain, which is why Louis is right that it can stay.

Guards before any of this fires: period score ≥ 0.80, member outlier test
(z > 3.0 vs the median+MAD of all members' deviations → kept unfolded as a
variant), stack coherence ≥ 0.85 median pairwise cosine, and CV ≤ 0.51 per
half-bar (tuned to a 5% false-merge rate).

## B. Which verified corpus we actually have

| corpus | expert GT | audio on disk | usable |
|---|---|---|---|
| **GuitarSet** | 360 JAMS, manually verified | **360 wav** | **yes** |
| McGill-Billboard | 890 expert annotations | **none** (chroma only) | no — musx needs audio |
| JAAH | 115 label files | audio dir **empty** | no |
| RWC | — | audio dir **0 bytes** | no |
| CHOCO | partitions | audio dir empty | no |

GuitarSet ships audio and absolute-timestamp GT *together*, so there is no
alignment step and therefore no circularity — the cleanest possible test of `c`.

## C. The measurement — `c` clears the bar, but only one half of it does

59 comping excerpts, 879 decoded chord spans, `harmonia_min.pipeline.analyze`:

    root accuracy   0.890        root + family   0.753

| | n | root acc | **AUC (root)** | AUC (root+family) |
|---|---|---|---|---|
| **all spans** | 879 | 0.890 | **0.800** | 0.758 |
| acoustic branch (`_segment_confidence`) | 712 | 0.875 | **0.819** | 0.804 |
| folding branch (`0.5 + 0.08·n_obs`) | 167 | **0.952** | **0.019** | 0.024 |

Reliability of the acoustic branch is cleanly monotone — exactly "meaningful but
not calibrated", which is all Louis asked for:

| `c` bin | n | mean `c` | actual root accuracy |
|---|---|---|---|
| 0.15–0.25 | 15 | 0.215 | 0.533 |
| 0.36–0.46 | 72 | 0.413 | 0.639 |
| 0.57–0.67 | 260 | 0.648 | 0.885 |
| 0.78–0.88 | 105 | 0.836 | 0.943 |
| 0.88–0.99 | 256 | 0.939 | 0.992 |

**Verdict: AUC 0.80 on the acoustic branch — the ratio rule is worth using.**
Against the spec table, AUC 0.80 buys **+31% net chords fixed** over an LM-only
threshold and lifts precision from 0.72 to 0.77.

**But the two branches must not share a field.** Folded spans are the *most*
accurate in the whole set (95.2% root vs 87.5%) and yet carry a *lower*
confidence (0.66 vs a 0.784 acoustic median). The folding value is not wrong as
an idea — it is on a different, unaligned scale, and pooling them costs real
discriminative power:

    acoustic branch alone   AUC 0.819
    both pooled into `c`    AUC 0.800

So the fix is not to delete the folding formula (Louis is right to keep it) but
to stop laundering it into the same number: keep the acoustic posterior in `c`,
carry repetition support alongside (`n_obs` is already written), and let the
intervention rule read both. That recovers ~2 points of AUC for free and makes
the folded-chord confidence stop lying downward.

## D. Revised order of work

1. Split `c`: acoustic posterior stays in `c`, folding support stays in `n_obs`.
   (AUC 0.800 → 0.819, no modelling change.)
2. Ship the **top-3 shortlist** — no pipeline confidence needed, cannot regress a
   chord without a human click, recovers 65% of the LM's top-1 misses.
3. Then the two-sided / ratio rule, now that `c` is known to sit at AUC ≈ 0.80.

Caveat carried forward: GuitarSet is solo acoustic guitar in 30 s excerpts, not
our production domain. 19% of spans still came from folding despite the short
excerpts, so both branches are represented, but the folding sample (167 spans,
2 distinct confidence values) is thin.
