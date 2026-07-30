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

