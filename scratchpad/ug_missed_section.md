## MISSED characterization

MISSED was the #1 defect class. Characterized before anyone tries to fix it — and characterizing it removed most of it.

### Two measurement artifacts found first

**1. Crammed tab material (−72 of 160).** The aligner gives a chord it has no room for the minimum legal duration (0.30 s). Every Breath You Take's tab writes out the entire fade-out loop — **73 chords, every one at the floor, all inside 205–229 s** — and each was being scored as a chord we missed, making it the second-worst song in the report. Same family as the Close to You alternate-ending problem already logged; `unsupported_frac` was flagging it at 0.275 and I did not act on it until the duration histogram made it unmissable. These are now class `CRAMMED`. Every Breath: **53 → 6 misses**.

**2. Diff slop (−16 more).** In 16 of 82 remaining cases (20%) our chart *does* write that root within ±2 s — the ordinal diff simply failed to pair them. Per the doctrine (UG timing is hand-made), those are not misses.

**160 reported → 66 real misses.** Everything below describes the 82 post-CRAMMED rows, with the slop share marked.

### 1. Duration — these are passing chords, not dropped bars

Median **0.60 s = 1.04 beats**. **58% last one beat or less**, 85% two beats or less, and only 9% are a full bar. We are not dropping structural changes; we are dropping ornaments and approach chords.

| song | n | median s | median beats | ≤1 beat |
|---|---|---|---|---|
| Let It Be | 37 | 0.60 | 0.68 | 68% |
| Close To You | 24 | 0.60 | 0.79 | 76% |
| Hot N Cold | 9 | 1.00 | 2.19 | 0% |
| Stand By Me | 5 | 1.40 | 2.80 | 20% |
| Every Breath You Take | 5 | 0.60 | 1.17 | 80% |
| This Love | 2 | 1.00 | 1.58 | 0% |

### 2. Position — NOT a downbeat/mid-bar effect

82 located on the payload bar grid. Counts by beat: beat 1 = 28, beat 2 = 17, beat 3 = 21, beat 4 = 16. That is close to uniform, so the missed chords are **not** concentrated off the downbeat. A bar-grid *phase* problem would show a strong mid-bar bias; it does not. This hypothesis is rejected by the data.

### 3. What we show instead

| | n | share |
|---|---|---|
| a third chord | 37 | 45% |
| held PREVIOUS | 18 | 22% |
| WE DO WRITE IT (timing slop) | 16 | 20% |
| no-chord | 6 | 7% |
| held NEXT | 5 | 6% |

Only ~22% is “we held the previous chord through the change”, so this is not mainly a decoder-stickiness story. The largest real bucket is *a third chord entirely* — consistent with our chart writing one chord for a whole bar that the tab fills with three.

### 4. Function — approach chords

Root motion out of the missed chord is a fourth/fifth (33%) or a step (27%): these sit **between** two structural chords and resolve into the next one. Top recurring shapes (prev → [missed] → next):

| n | shape | song |
|---|---|---|
| 12 | C [D-7] C | Let It Be |
| 7 | G [F] C | Let It Be |
| 6 | A- [C] G | Hot N Cold |
| 4 | G [C] G | Close To You |
| 4 | A- [G] F | Let It Be |
| 3 | C [G] C | Close To You |
| 3 | C [G] F | Let It Be |
| 3 | F [C] D-7 | Let It Be |

### 5. Are the big two one repeating pattern each?

**Partly, and it matters.**

- **Let It Be** (37): top shape `C [D-7] C` 12/37 (32%)
- **Close To You** (24): top shape `G [C] G` 4/24 (17%)
- **Hot N Cold** (9): top shape `A- [C] G` 6/9 (67%)
- **Stand By Me** (5): top shape `E [A] Gb-` 2/5 (40%)
- **Every Breath You Take** (5): top shape `Eb [Ab] F-` 1/5 (20%)
- **This Love** (2): top shape `Ab [G] C-` 2/2 (100%)

Let It Be is one third a single repeated shape and Hot N Cold two thirds; Close to You is genuinely diverse. So a fix aimed at one progression would clear a third of Let It Be and most of Hot N Cold, and nothing on Close to You.

### 6. Screening test — already run, and it confirms

The premise “our chart cannot express a short chord” is checkable with data already on disk, so I ran it rather than proposing it (project rule: screen the premise cheaply).

| | median | min | under 1 beat | under 2 beats |
|---|---|---|---|---|
| **our chart** | 4.00 beats | 0.80 | **0%** | 2% |
| **UG tab** | 2.22 beats | 0.09 | 14% | 40% |

**Our chart never emits a chord shorter than about one beat** (min 0.80 beats over 488 chords; 0% under a beat). Per song it is starker: Stand By Me, Hot N Cold and Every Breath are locked to *exactly* 4.00 beats — one chord per bar, no exceptions. Let It Be and This Love run on a half-bar grid (median 2.00). Close to You sits at 8 beats, two whole bars.

And the miss counts track the **tab's** sub-beat content almost perfectly: Let It Be 35% of tab chords under a beat → 37 misses; Close to You 22% → 24; every other song ~0–2% → 5–9 misses.

### Ranked hypotheses

| # | hypothesis | verdict |
|---|---|---|
| 1 | **Our chart is quantized to a bar / half-bar grid and cannot represent a sub-beat chord at all.** The tab puts 40% of its chords under two beats; we put 2%. | **Confirmed** by the screening test above, and the per-song miss counts follow the tab's sub-beat share |
| 2 | Beat-grid *phase* — mid-bar changes merged into the downbeat | **Rejected** — position is near-uniform across the four beats |
| 3 | Decoder stickiness — the HMM holds the previous chord | **Minor** — only ~22% show “held previous” |
| 4 | Genuine harmonic error (we hear a different chord) | **Residual** — the “third chord” bucket, largely explained by (1): one chord written over a bar the tab fills with three |

### The one cheapest test that would falsify hypothesis 1

Hypothesis 1 says the chords exist in the model but are destroyed by quantization. So: **dump the decoder's pre-quantization chord sequence for Let It Be and check whether the 12 `C [Dm7] C` and 7 `G [F] C` events are present as sub-bar segments before the bar grid is applied.**

- If they are there → quantization is the defect; the fix is grain, not harmony, and it is cheap.
- If they are absent pre-quantization → hypothesis 1 is falsified, the chords never existed, and the defect is upstream in the emission or the decoder's duration prior. That would redirect the whole effort.

One song, one intermediate dump, no retraining, no sweep. Run it before touching anything.

*(Figures: `scratchpad/ug_missed_duration.png`, `scratchpad/ug_missed_position.png`.)*
