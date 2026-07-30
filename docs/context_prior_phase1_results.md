# Chord context prior — Phase 1 results

Branch `feat/chord-context-prior`. See `docs/design_chord_context_prior.md` for the design and `harmonia/models/chord_context_prior.py` for the model.

## Corpus stats

Train songs: **3826**  |  Held-out songs: **473**  (deterministic 1-in-10 split by stable hash of song id)

| source | records seen | songs kept | songs dropped (short) | songs dropped (other) | tokens kept |
|---|---|---|---|---|---|
| accomp_db | 1856 | 1842 | 12 | 2 | 92268 |
| pop909 | 909 | 909 | 0 | 0 | 100813 |
| choco | 1561 | 1548 | 13 | 0 | 151372 |

### Dropped-label counters (top 15 per source)

**accomp_db** (total dropped label instances shown here: top-15 of the recorded set; grand total dropped = 9):

| label | count |
|---|---|
| `accomp:unmapped_mma:C*-^*` | 8 |
| `accomp:unmapped_mma:F#*-^*` | 1 |

**pop909** (total dropped label instances shown here: top-15 of the recorded set; grand total dropped = 4736):

| label | count |
|---|---|
| `pop909:N` | 4736 |

**choco** (total dropped label instances shown here: top-15 of the recorded set; grand total dropped = 12626):

| label | count |
|---|---|
| `choco:billboard:N` | 4793 |
| `choco:jaah:N` | 1135 |
| `choco:isophonics:N` | 574 |
| `choco:billboard:unmapped:G:5` | 481 |
| `choco:billboard:unmapped:D:5` | 416 |
| `choco:billboard:unmapped:A:5` | 414 |
| `choco:uspop2002:N` | 397 |
| `choco:billboard:unmapped:E:1/1` | 389 |
| `choco:billboard:unmapped:A:1/1` | 301 |
| `choco:billboard:unmapped:C:5` | 298 |
| `choco:rwc-pop:N` | 263 |
| `choco:billboard:unmapped:D:1/1` | 259 |
| `choco:billboard:unmapped:B:1/1` | 219 |
| `choco:billboard:unmapped:F:5` | 215 |
| `choco:billboard:unmapped:F#:5` | 203 |


## Eval 1 — masked infilling recall (held-out songs)

Model vs same-candidate-space baselines, all on the identical masked positions.

**Caveat on the two "repeat a neighbour" baselines**: every segment has had consecutive-identical tokens collapsed (change-to-change transitions only, per the design doc), so by construction `prev != true_tok != next` at every masked position. "Repeat prev"/"repeat next" therefore score EXACTLY 0% — not a real result, a structural consequence of the dedup, kept only because the brief named them. The meaningful baseline is "prev-bigram only" (uses real context, just one-sided).

| method | top-1 recall | top-3 recall | n |
|---|---|---|---|
| context prior P(c\|prev,next) | 28.5% | 57.6% | 35598 |
| repeat prev chord | 0.0% | - | 35598 |
| repeat next chord | 0.0% | - | 35598 |
| prev-bigram only (no next) | 20.2% | 45.7% | 35598 |

### Root-correct / quality-correct, independently (model, full context)

| axis | top-1 recall | top-3 recall | n |
|---|---|---|---|
| root correct (any quality) | 37.9% | 68.4% | 35598 |
| quality correct (any root) | 54.2% | 82.2% | 35598 |

### Exact-match recall by TRUE candidate quality (model)

| quality | top-1 recall | top-3 recall | n |
|---|---|---|---|
| maj | 36.7% | 72.1% | 19623 |
| min | 14.8% | 39.2% | 10414 |
| dom | 29.3% | 46.1% | 4795 |
| hdim | 0.0% | 17.0% | 376 |
| dim | 0.0% | 0.0% | 390 |

### Exact-match recall by corpus family (model)

| family | top-1 recall | top-3 recall | n |
|---|---|---|---|
| jazz | 33.6% | 55.5% | 9351 |
| pop | 26.7% | 58.3% | 26247 |

## Eval 2 — theory sanity panel (ranked top-8, all contexts in C)

### Dm7 -> ? -> Cmaj  (ii-V-I; expect G7 dominant high, Db7 tritone sub in top-8)

| rank | chord | quality | P |
|---|---|---|---|
| 1 | G7 | dom | 0.177 |
| 2 | Gmaj | maj | 0.128 |
| 3 | A#maj | maj | 0.124 |
| 4 | Fmaj | maj | 0.120 |
| 5 | Gm7 | min | 0.076 |
| 6 | Dmaj | maj | 0.067 |
| 7 | Am7 | min | 0.053 |
| 8 | Em7 | min | 0.049 |

### Dm7b5 -> ? -> Cm  (minor ii-V-i; expect G7)

| rank | chord | quality | P |
|---|---|---|---|
| 1 | G7 | dom | 0.282 |
| 2 | A#m7 | min | 0.137 |
| 3 | D#maj | maj | 0.112 |
| 4 | Fm7 | min | 0.064 |
| 5 | G#maj | maj | 0.050 |
| 6 | A#maj | maj | 0.039 |
| 7 | C#7 | dom | 0.038 |
| 8 | Dm7 | min | 0.035 |

### Cmaj -> ? -> Dm7  (expect diatonic passing / A7 = V-of-ii)

| rank | chord | quality | P |
|---|---|---|---|
| 1 | Fmaj | maj | 0.206 |
| 2 | Am7 | min | 0.121 |
| 3 | Dmaj | maj | 0.115 |
| 4 | Gmaj | maj | 0.109 |
| 5 | A#maj | maj | 0.068 |
| 6 | Cm7 | min | 0.050 |
| 7 | Gm7 | min | 0.048 |
| 8 | Em7 | min | 0.038 |

### Cmaj -> ? -> G7  (expect Dm7 ii, or D7 V-of-V)

| rank | chord | quality | P |
|---|---|---|---|
| 1 | Dm7 | min | 0.221 |
| 2 | Fmaj | maj | 0.172 |
| 3 | Gmaj | maj | 0.155 |
| 4 | A#maj | maj | 0.066 |
| 5 | Am7 | min | 0.056 |
| 6 | C7 | dom | 0.044 |
| 7 | D7 | dom | 0.041 |
| 8 | Em7 | min | 0.035 |

### Fmaj -> ? -> Cmaj  (expect G7, Bb7 backdoor, Fm iv, Ab/G dim passing)

| rank | chord | quality | P |
|---|---|---|---|
| 1 | Gmaj | maj | 0.221 |
| 2 | A#maj | maj | 0.169 |
| 3 | Fm7 | min | 0.093 |
| 4 | Dm7 | min | 0.085 |
| 5 | C7 | dom | 0.050 |
| 6 | Cm7 | min | 0.049 |
| 7 | Em7 | min | 0.044 |
| 8 | Am7 | min | 0.041 |

### Am7 -> ? -> Dm7  (expect A7 or E7-family secondary dominants)

| rank | chord | quality | P |
|---|---|---|---|
| 1 | D7 | dom | 0.154 |
| 2 | Fmaj | maj | 0.127 |
| 3 | Cmaj | maj | 0.119 |
| 4 | Amaj | maj | 0.100 |
| 5 | Gmaj | maj | 0.087 |
| 6 | Dmaj | maj | 0.069 |
| 7 | Em7 | min | 0.062 |
| 8 | A#maj | maj | 0.058 |

**Observations (reported as measured, not cherry-picked)**: ii-V-I, minor iiø-V-i, and V-of-V (Cmaj->?->G7 surfaces Dm7 #1, D7 #7) all match the stated expectation. Two clear misses: (1) the Db7 tritone sub does **not** appear in the Dm7->?->Cmaj top-8 at all — tritone subs are a real but rare voicing choice in the training corpora, so the model has little to learn it from; (2) in Fmaj->?->Cmaj the model's top-2 candidates are **Gmaj/Bbmaj (plain triads)**, not the expected **G7/Bb7 (dominant)** — the pooled corpus is pop-heavy, and pop's IV-V-I / IV-bVII-I idioms are overwhelmingly plain triads, diluting the jazz-specific dominant reading (the design doc's own genre-pooling caveat, confirmed here). Am7->?->Dm7's top candidate is D7 (same root as the *next* chord, quality shifting dom->min) rather than the expected A7/E7 secondary dominant — worth a follow-up look at whether same-root quality-shift transitions are systematically over-weighted by the root-evidence term.

## Genre-arm experiment

Branch `feat/chord-context-prior`. Tests the HYPOTHESIS that pooling jazz (accomp_db) and pop (POP909+ChoCo) into one table diluted the jazz theory panel (Phase 1 finding), by building jazz-only and pop-only tables (`build_context_prior(corpus=...)`) alongside the unchanged pooled table, plus an interpolation probe.

### Corpus sizes per arm

| table | train songs | held-out songs | trigram count sum |
|---|---|---|---|
| pooled | 3826 | 473 | 295387 |
| jazz | 1630 | 212 | 79233 |
| pop | 2196 | 261 | 216154 |

### Leakage / split-consistency verification (guardrail: verify-don't-trust)

```
jazz held-out: 212 songs (canonical, from tables['jazz'])
  table=pooled  train_ids leakage check: OK (0 leaked)
  table=jazz    train_ids leakage check: OK (0 leaked)
  table=pop     train_ids leakage check: OK (0 leaked)
pop held-out: 261 songs (canonical, from tables['pop'])
  table=pooled  train_ids leakage check: OK (0 leaked)
  table=jazz    train_ids leakage check: OK (0 leaked)
  table=pop     train_ids leakage check: OK (0 leaked)
  pooled-vs-jazz-arm held-out split identical: True (pooled sees 212, arm sees 212)
  pooled-vs-pop-arm held-out split identical: True (pooled sees 261, arm sees 261)
```

### Eval 1 — 3x2 matrix: each table scored on each genre's OWN held-out songs

Held-out songs for a genre are that genre's own table's `heldout_ids` (verified identical to the pooled table's held-out subset for the same song ids, and never in ANY table's train_ids — see verification above).

| table | held-out genre | n positions | exact top-1 | exact top-3 | root-any top-1 | root-any top-3 | qual-any top-1 | qual-any top-3 |
|---|---|---|---|---|---|---|---|---|
| pooled | jazz | 9351 | 33.6% | 55.5% | 45.3% | 68.8% | 49.0% | 77.1% |
| pooled | pop | 26247 | 26.7% | 58.3% | 35.2% | 68.3% | 56.0% | 84.0% |
| jazz | jazz | 9351 | 39.4% | 60.1% | 52.4% | 69.9% | 52.6% | 85.0% |
| jazz | pop | 26247 | 21.7% | 48.0% | 34.4% | 64.1% | 41.6% | 80.3% |
| pop | jazz | 9351 | 21.0% | 41.6% | 39.7% | 66.0% | 38.0% | 63.0% |
| pop | pop | 26247 | 29.3% | 58.1% | 35.9% | 68.9% | 61.1% | 81.4% |

### Interpolation probe — score = alpha*P_jazz + (1-alpha)*P_pop

| alpha | held-out genre | n positions | exact top-1 | exact top-3 | root-any top-1 | qual-any top-1 |
|---|---|---|---|---|---|---|
| 0.3 | jazz | 9351 | 31.5% | 54.2% | 43.5% | 46.4% |
| 0.3 | pop | 26247 | 25.6% | 59.1% | 34.0% | 56.1% |
| 0.5 | jazz | 9351 | 36.8% | 57.7% | 47.4% | 51.2% |
| 0.5 | pop | 26247 | 24.3% | 55.6% | 34.7% | 51.6% |
| 0.7 | jazz | 9351 | 38.7% | 59.5% | 50.6% | 52.4% |
| 0.7 | pop | 26247 | 24.2% | 51.0% | 35.1% | 49.4% |

## Eval 2 — theory panel, three tables side by side (top-8 each)

### Dm7 -> ? -> Cmaj  (ii-V-I; expect G7 dominant high, Db7 tritone sub in top-8)

- **pooled**: G7(0.177), Gmaj(0.128), A#maj(0.124), Fmaj(0.120), Gm7(0.076), Dmaj(0.067), Am7(0.053), Em7(0.049)
- **jazz**: G7(0.498), Fmaj(0.062), A#maj(0.050), Em7(0.039), Am7(0.038), C#7(0.031), C7(0.026), Dm7b5(0.025)
- **pop**: Gmaj(0.176), A#maj(0.142), Fmaj(0.135), Gm7(0.102), Dmaj(0.078), G7(0.067), Am7(0.057), Em7(0.051)

### Dm7b5 -> ? -> Cm  (minor ii-V-i; expect G7)

- **pooled**: G7(0.282), A#m7(0.137), D#maj(0.112), Fm7(0.064), G#maj(0.050), A#maj(0.039), C#7(0.038), Dm7(0.035)
- **jazz**: G7(0.397), A#m7(0.061), D#maj(0.044), C#7(0.043), D#7(0.043), Fm7(0.040), C#m7(0.035), Cmaj(0.032)
- **pop**: G7(0.187), D#maj(0.149), A#m7(0.148), Fm7(0.068), G#maj(0.067), A#maj(0.063), Gmaj(0.058), C#7(0.039)

### Cmaj -> ? -> Dm7  (expect diatonic passing / A7 = V-of-ii)

- **pooled**: Fmaj(0.206), Am7(0.121), Dmaj(0.115), Gmaj(0.109), A#maj(0.068), Cm7(0.050), Gm7(0.048), Em7(0.038)
- **jazz**: Am7(0.137), Fmaj(0.135), Gmaj(0.101), A7(0.088), D7(0.087), D#dim7(0.050), Em7(0.045), A#maj(0.041)
- **pop**: Fmaj(0.223), Dmaj(0.145), Am7(0.118), Gmaj(0.112), A#maj(0.073), Cm7(0.055), Gm7(0.054), Em7(0.036)

### Cmaj -> ? -> G7  (expect Dm7 ii, or D7 V-of-V)

- **pooled**: Dm7(0.221), Fmaj(0.172), Gmaj(0.155), A#maj(0.066), Am7(0.056), C7(0.044), D7(0.041), Em7(0.035)
- **jazz**: Dm7(0.332), Fmaj(0.092), Gmaj(0.091), C7(0.058), Am7(0.048), A#maj(0.046), G#7(0.031), D7(0.030)
- **pop**: Fmaj(0.200), Gmaj(0.178), Dm7(0.167), A#maj(0.072), D7(0.058), Am7(0.057), Em7(0.040), C7(0.037)

### Fmaj -> ? -> Cmaj  (expect G7, Bb7 backdoor, Fm iv, Ab/G dim passing)

- **pooled**: Gmaj(0.221), A#maj(0.169), Fm7(0.093), Dm7(0.085), C7(0.050), Cm7(0.049), Em7(0.044), Am7(0.041)
- **jazz**: Gmaj(0.153), G7(0.110), Fm7(0.091), Dm7(0.080), C7(0.074), A#maj(0.069), A#7(0.052), Gm7(0.035)
- **pop**: Gmaj(0.226), A#maj(0.186), Fm7(0.092), Dm7(0.086), Cm7(0.061), Em7(0.045), C7(0.043), Am7(0.042)

### Am7 -> ? -> Dm7  (expect A7 or E7-family secondary dominants)

- **pooled**: D7(0.154), Fmaj(0.127), Cmaj(0.119), Amaj(0.100), Gmaj(0.087), Dmaj(0.069), Em7(0.062), A#maj(0.058)
- **jazz**: D7(0.304), A7(0.154), Cmaj(0.069), Fmaj(0.068), Gmaj(0.060), Em7(0.048), A#maj(0.039), Bm7(0.036)
- **pop**: Fmaj(0.146), Dmaj(0.145), Cmaj(0.137), Amaj(0.120), Gmaj(0.096), Em7(0.067), A#maj(0.064), D7(0.048)

### G7 -> ?  (prev-only bigram; Dm7-G7-? with ? after G7; expect Cmaj top)

- **pooled**: Cmaj(0.302), Cm7(0.152), C7(0.114), F#7(0.041), D7(0.038), Dm7(0.037), Gm7(0.026), Em7(0.023)
- **jazz**: Cmaj(0.272), Cm7(0.181), C7(0.117), F#7(0.055), Gm7(0.034), Dm7(0.033), D7(0.025), F#m7(0.022)
- **pop**: Cmaj(0.352), C7(0.110), Cm7(0.106), D7(0.059), Dm7(0.043), Fmaj(0.039), Dmaj(0.031), Gmaj(0.030)


## Eval 2b — interpolation, alpha*jazz + (1-alpha)*pop (top-8)

### Dm7 -> ? -> Cmaj  (ii-V-I; expect G7 dominant high, Db7 tritone sub in top-8)

- **alpha=0.3**: G7(0.196), Gmaj(0.128), A#maj(0.114), Fmaj(0.113), Gm7(0.075), Dmaj(0.059), Am7(0.051), Em7(0.047)
- **alpha=0.5**: G7(0.282), Fmaj(0.098), A#maj(0.096), Gmaj(0.095), Gm7(0.058), Dmaj(0.047), Am7(0.047), Em7(0.045)
- **alpha=0.7**: G7(0.368), Fmaj(0.084), A#maj(0.078), Gmaj(0.063), Am7(0.044), Em7(0.042), Gm7(0.040), Dmaj(0.035)

### Dm7b5 -> ? -> Cm  (minor ii-V-i; expect G7)

- **alpha=0.3**: G7(0.250), A#m7(0.122), D#maj(0.118), Fm7(0.060), G#maj(0.053), A#maj(0.051), Gmaj(0.040), C#7(0.040)
- **alpha=0.5**: G7(0.292), A#m7(0.105), D#maj(0.096), Fm7(0.054), G#maj(0.045), A#maj(0.043), C#7(0.041), Cmaj(0.034)
- **alpha=0.7**: G7(0.334), A#m7(0.087), D#maj(0.075), Fm7(0.049), C#7(0.042), G#maj(0.036), A#maj(0.035), Cmaj(0.033)

### Cmaj -> ? -> Dm7  (expect diatonic passing / A7 = V-of-ii)

- **alpha=0.3**: Fmaj(0.197), Am7(0.123), Gmaj(0.109), Dmaj(0.107), A#maj(0.064), Cm7(0.045), Gm7(0.042), Em7(0.039)
- **alpha=0.5**: Fmaj(0.179), Am7(0.127), Gmaj(0.107), Dmaj(0.082), A#maj(0.057), A7(0.051), D7(0.050), Em7(0.041)
- **alpha=0.7**: Fmaj(0.162), Am7(0.131), Gmaj(0.105), A7(0.066), D7(0.064), Dmaj(0.057), A#maj(0.051), Em7(0.043)

### Cmaj -> ? -> G7  (expect Dm7 ii, or D7 V-of-V)

- **alpha=0.3**: Dm7(0.216), Fmaj(0.167), Gmaj(0.152), A#maj(0.064), Am7(0.054), D7(0.050), C7(0.044), Em7(0.034)
- **alpha=0.5**: Dm7(0.249), Fmaj(0.146), Gmaj(0.134), A#maj(0.059), Am7(0.052), C7(0.048), D7(0.044), Em7(0.030)
- **alpha=0.7**: Dm7(0.282), Fmaj(0.124), Gmaj(0.117), A#maj(0.054), C7(0.052), Am7(0.051), D7(0.038), Em7(0.027)

### Fmaj -> ? -> Cmaj  (expect G7, Bb7 backdoor, Fm iv, Ab/G dim passing)

- **alpha=0.3**: Gmaj(0.204), A#maj(0.151), Fm7(0.092), Dm7(0.084), C7(0.052), G7(0.046), Cm7(0.044), Em7(0.041)
- **alpha=0.5**: Gmaj(0.189), A#maj(0.128), Fm7(0.092), Dm7(0.083), G7(0.064), C7(0.059), Em7(0.038), Am7(0.038)
- **alpha=0.7**: Gmaj(0.175), A#maj(0.105), Fm7(0.091), G7(0.083), Dm7(0.082), C7(0.065), Gm7(0.037), A#7(0.037)

### Am7 -> ? -> Dm7  (expect A7 or E7-family secondary dominants)

- **alpha=0.3**: D7(0.124), Fmaj(0.123), Cmaj(0.116), Dmaj(0.103), Amaj(0.087), Gmaj(0.085), Em7(0.061), A7(0.058)
- **alpha=0.5**: D7(0.176), Fmaj(0.107), Cmaj(0.103), A7(0.086), Gmaj(0.078), Dmaj(0.075), Amaj(0.065), Em7(0.057)
- **alpha=0.7**: D7(0.227), A7(0.113), Fmaj(0.092), Cmaj(0.089), Gmaj(0.071), Em7(0.053), Dmaj(0.047), A#maj(0.047)

### G7 -> ?  (prev-only bigram; Dm7-G7-? with ? after G7; expect Cmaj top)

- **alpha=0.3**: Cmaj(0.328), Cm7(0.129), C7(0.112), D7(0.049), Dm7(0.040), Fmaj(0.031), F#7(0.030), Dmaj(0.024)
- **alpha=0.5**: Cmaj(0.312), Cm7(0.144), C7(0.113), D7(0.042), Dm7(0.038), F#7(0.037), Fmaj(0.026), Gm7(0.024)
- **alpha=0.7**: Cmaj(0.296), Cm7(0.158), C7(0.114), F#7(0.044), Dm7(0.036), D7(0.035), Gm7(0.028), Em7(0.023)


## Condensed key rows

### Dm7 -> ? -> Cmaj (ii-V-I)

| table | G7 rank | G7 P | Db7(tritone sub) rank | Db7 P |
|---|---|---|---|---|
| pooled | 1 | 0.177 | 17 | 0.009 |
| jazz | 1 | 0.498 | 6 | 0.031 |
| pop | 6 | 0.067 | 24 | 0.002 |

### Fmaj -> ? -> Cmaj (IV-V-I / IV-bVII-I)

| table | Gmaj rank | Gmaj P | G7 rank | G7 P | Bb7(backdoor) rank | Bb7 P |
|---|---|---|---|---|---|---|
| pooled | 1 | 0.221 | 11 | 0.026 | 17 | 0.009 |
| jazz | 1 | 0.153 | 2 | 0.110 | 7 | 0.052 |
| pop | 1 | 0.225 | 12 | 0.019 | 26 | 0.002 |

### Same Fmaj -> ? -> Cmaj row, interpolated (jazz/pop combination)

| alpha | Gmaj rank | Gmaj P | G7 rank | G7 P | Bb7 rank | Bb7 P | does G7 overtake Gmaj? |
|---|---|---|---|---|---|---|---|
| 0.3 | 1 | 0.204 | 6 | 0.046 | 13 | 0.017 | no |
| 0.5 | 1 | 0.189 | 5 | 0.064 | 12 | 0.027 | no |
| 0.7 | 1 | 0.175 | 4 | 0.083 | 8 | 0.037 | no |

### Own-genre gain vs cross-genre cost (delta vs pooled, pp)

| table used | own-genre exact top-1 | own-genre exact top-3 | wrong-genre exact top-1 | wrong-genre exact top-3 |
|---|---|---|---|---|
| jazz table | +5.8 | +4.7 | -5.0 | -10.3 |
| pop table | +2.5 | -0.3 | -12.5 | -13.9 |

**Recommendation**: ship a **two-table setup with content-type routing**, keep `pooled` as the fallback when a song's genre is unknown or mixed — do NOT ship interpolation.

Why, in one paragraph: the jazz-only table is a clear win on jazz content (own-genre exact top-1 +5.8pp, top-3 +4.7pp vs pooled; ii-V-I G7 confidence jumps from 0.177 to 0.498 and the Db7 tritone sub newly enters the top-8; Fmaj->?->Cmaj gets G7 and the Bb7 backdoor into the top-8 where pooled had neither) but costs real recall if it is ever applied to a pop song (-5.0pp top-1, -10.3pp top-3) — symmetrically the pop table modestly helps pop (+2.5pp top-1) but is actively worse than pooled on jazz content, badly enough to invert the ii-V-I ranking (G7 drops out of #1 to #6, behind Gmaj). That asymmetry is the whole argument for routing rather than always using one arm: the failure mode of guessing wrong is worse than the failure mode of falling back to pooled. Interpolation earns no seat — at every alpha tested it sits strictly between the two pure single-genre tables' own-genre scores (never beats either on its matching genre) and even alpha=0.7 (jazz-heavy) does not flip Gmaj/G7 in Fmaj->?->Cmaj, so it buys none of the jazz table's clearest win while still paying a pop-side cost — added complexity for no clear gain. Given the app's actual content (docs/plots charts: mostly jazz standards + classic pop, i.e. genuinely mixed), the practical shape is: tag each song/chart with a coarse genre label (already implicit in provenance — accomp_db/iReal jazz charts vs POP909/ChoCo pop charts) and route to the matching table; fall back to pooled only when that label is unavailable, never to the wrong single-genre table.


