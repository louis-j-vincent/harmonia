# Beat This! (CPJKU, arXiv 2407.21658) premise-check — 2026-07-21

Transformer beat+DOWNBEAT tracker, NO DBN postprocessing, **MIT license** (code + weights).
Candidate for (1) replacing madmom's CC-BY-NC-SA beat role (COMMERCIAL_LICENSING_AUDIT
blocker), (2) fixing bar-1/downbeat anchoring at the SOURCE (this session's core bug).

## Setup
`beat_this` 1.1.0 ALREADY INSTALLED (torch 2.12.1) — NO install cost, no env disruption.
API: `File2Beats(checkpoint_path='final0', device='cpu', dbn=False)(wav) -> (beats, downbeats)`.
Inference **3–4 s CPU** for a full ~4-min song — acceptable for live analyze (≈ musx cost).

## Findings (matched-set songs)
| song | BeatThis 1st downbeat | true 1st onset | BeatThis tempo | librosa tempo | note |
|---|---|---|---|---|---|
| This Love | **1.08 s** | 1.045 | 95 | 95.7 | ✓ nails bar-1 natively (the bug we patched); downbeats align to chord changes |
| Let It Be | **0.04 s** | 0.05 | 69.8 | 143.6 (2×) | ✓ anchor; picks HALF octave |
| abba | **0.14 s** | 0.07 | 85.7 | 172.3 (2×) | ✓ anchor; picks HALF octave |
| SWBL | 11.94 s | 1.42 | 103.4 | 103.4 | ✗ SKIPS the free/rubato guitar intro (beats start at band entry 11.4 s) |

**Verdict — PROMISING, worth a gated front-end swap (its own rollout, NOT this round):**
1. **Solves bar-1/downbeat at the source** for steady-beat songs — This Love/Let It Be/abba
   all correct natively, no `_attach_musx_onset_hints`/flux-anchor/structure-anchor patches.
2. **MIT** → removes the madmom NC licensing blocker for the beat-tracking role.
3. Fast enough for live.

**Caveats to validate before a swap:**
- **Tempo-octave choice differs**: BeatThis picks the SLOWER octave (½ of librosa) on ballads.
  Avoids librosa's 2× lock but is its OWN choice — cascades into bar counts / chord-per-bar
  density / the whole bestfit+flux grid. Needs a corpus octave-accuracy check vs GT tempo
  (POP909 downbeat GT is available — `is_downbeat`/`downbeat_times`, known_issues env gotcha).
- **Rubato intros skipped** (SWBL) — a chart-vs-performance-meter mismatch to handle.
- Downbeat accuracy validated on only 4 songs here; needs the octave-lock corpus (known_issues
  #1) + POP909 downbeat-GT for a real accuracy number.

## Recommendation
Scope a `beat_backend="beatthis"` option (alongside librosa/madmom) feeding the raw-beat +
DOWNBEAT step of bestfit/flux — a foundational swap addressing BOTH the licensing blocker and
the downbeat-anchor problem. Gate like bestfit/flux (corpus octave/downbeat accuracy + matched-
set 2-run + no-regression). Do NOT wire live without explicit go-ahead. (Also directly relevant
to §-discrimination thread: better SOURCE downbeats → better section boundaries, the ckpt-8 lever.)

## FULL VALIDATION (2026-07-21) — octave PASSES, tempo-default-flip does NOT
Rendered POP909 songs from MIDI (fluidsynth + MuseScore_General.sf2), compared Beat This!
vs librosa against beat_midi.txt GT (beats + downbeats). `scratchpad/bt_full.py`.
**46-song sample:**
| metric | Beat This! | librosa |
|---|---|---|
| tempo-octave correct | **78%** | 65% |
| — HARD subset (GT<80 / >170 BPM, n=32) | **88%** | 66% |
| error mode | ½× (slow) 20% | 2× (fast) 22% |
| beat F1 (±70ms) | **0.85** | 0.77 |
| downbeat F1 (±70ms) | 0.68 (0.69 octave-correct) | — (no downbeats) |
**Octave criterion PASSES**: 88% on the hard subset where the documented blind ceiling was
~38% — Beat This! genuinely solves the "unsolvable" octave-lock on ballads/bebop, AND beats
raw librosa (65%). (The 20-song first cut showed a 70-70 tie — that was easy-song sampling.)

**BUT the default-flip FAILS no-regression** (matched-set, `scratchpad/gate_bt.py`): Beat This!
gives the CORRECT slower octave, which CHANGES the bar interpretation on ~half the set:
Let It Be 140→70 BPM (GT 72 ✓), abba 169→95, Easy 133→70, Bein' Green 150→75 (all ½×);
Stand By Me 79→120. 5 songs identical (This Love, Billie Jean, aretha, henny, just-aint, SWBL).
The whole downstream (bestfit period, `condense` which folds 2×-FAST down but NOT ½×-slow up,
and the user-validated forms) is tuned to librosa's tempo octave. Adopting Beat This!'s octave
is a FOUNDATIONAL cascade, not a clean drop-in — condense would need a ½×-expand path and every
form re-validated. **→ shipped OPT-IN (`beat_backend="beatthis"`), default stays librosa.**

**⚠ CROSS-SESSION COLLISION (flag for coordinator):** a concurrent session is ALREADY wiring
Beat This! for the DOWNBEAT ANCHOR (`harmonia/models/downbeat_anchor.py`, `sota_downbeat_phase`,
`HARMONIA_GRID_ANCHOR_SOTA=on` DEFAULT, in `_infer_nnls24`) — the bar-1 fix, using Beat This!
downbeats, default-on. My work is the complementary TEMPO role + this octave VALIDATION (which
SUPPORTS their adoption). The two Beat This! integrations should be MERGED: their downbeats for
bar-1 (default) + my validated tempo option (opt-in until condense is re-tuned for ½×). Staged
surgically (my 4 hunks only; their _fifth_corrected_quality + sota_downbeat WIP untouched).

## Licensing (resolves the blocker for the beat role)
Beat This! is **MIT** (code + weights) → the CC-BY-NC-SA madmom dependency is no longer needed
for beat/downbeat tracking. (NNLS-Chroma's GPL status is separate and unaffected.)
