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

## DEFAULT FLIPPED to beatthis (2026-07-21) — full acceptance gate PASSED
User authorized the flip. Both acceptance criteria (correct bar-1 automatically + no drift):
**1. NO DRIFT (the key new bar) — beatthis IMPROVES it.** Non-circular measure vs POP909 GT
beats (`scratchpad/p909d/griddrift.py`), mean|offset| START/MID/END:
| song | librosa bestfit-grid vs GT | beatthis beats vs GT |
|---|---|---|
| 001 | 58/44/29 ms | **17/18/17 ms** |
| 003 | 8/33/**65** ms (DRIFTS) | **15/14/16 ms** (flat) |
| 007 | 110/111/110 ms | **9/13/17 ms** |
beatthis is 9-18 ms FLAT (no growth) everywhere; librosa drifts (003: 8→65) or sits high
(007: 110). beatthis's per-beat detections track the audio without the uniform-bestfit-grid
accumulation. Matched-set circular check (vs librosa beats) also flat: Let It Be 49/33/38,
This Love 32/37/30 — all <150 ms, no growth.
**2. CORRECT BAR-1 automatically** — live /api/analyze, side port, 2-run STABLE, 6 songs:
| song | nBars | bar0 t0 | form | note |
|---|---|---|---|---|
| Let It Be | 72 | **0.0** | A | 70 BPM = GT 72 (was librosa 140/142) |
| This Love | 80 | **1.08** | A×2·B·A·B×6 | unchanged |
| abba | 58 | **0.14** | A | 95 BPM (was 169) |
| Stand By Me | 89 | **0.4** | Intro·A×10 | 120 BPM (was 79) |
| SWBL | 115 | **0.147** | Intro·A×3·B·A×8·C·B | richer form + the Eb B section |
| Billie Jean | 72 | **1.2** | Intro·A·B·A | unchanged |
All bar-1 land at the song start/first onset; all 2-run identical. Forms that changed are the
CORRECT octave (verified vs known tempos), not regressions. Combined with the concurrent
`downbeat_anchor.py` (Beat This! downbeats for bar-1 phase, default-on), the pipeline now
detects the right first beat out of the box AND stays aligned start-to-finish.
Test suite: 67 passed. **beat_backend="librosa" is instant rollback.**
**⚠ RESTART of live 7771 required to pick up the flip (this is a "your restart" item).**

## MODULARITY refactor — DESIGN (deferred: 3-file concurrent-WIP collision)
The user asked for a modularity pass ("il faut que notre code soit très modulaire"). The
time/grid logic is scattered across `chord_pipeline_v1.py` (`_bestfit_beat_period`,
`_flux_downbeat_phase`, `_structure_anchor_phase`, `_attach_musx_onset_hints`, the
librosa/madmom/beatthis dispatch) + the concurrent session's `downbeat_anchor.py`, coupled via
module-level functions and env flags. Target design — `harmonia/models/beat_grid.py`:
```
@dataclass
class TimeGrid:
    tempo_bpm: float; period: float
    beat_times: np.ndarray          # raw per-beat detections (source backend)
    downbeat_phase: int             # beats offset of bar-1 downbeat
    bar_times: np.ndarray           # derived bar boundaries
    backend: str

class BeatBackend(Protocol):        # interchangeable strategies, one interface
    def detect(self, audio_path) -> tuple[beat_times, tempo, downbeats|None]: ...
# LibrosaBackend / MadmomBackend / BeatThisBackend  (each ~10 lines, no if/elif chains)

def time_grid(audio_path, backend="beatthis") -> TimeGrid:      # single entry point
    beats, tempo, downbeats = _BACKENDS[backend].detect(audio_path)
    tg = _dejitter_bestfit(beats, tempo)        # composable refinement steps,
    tg = _downbeat_phase(tg, downbeats, audio)  # each individually testable:
    return tg                                    # bestfit / flux / structure-anchor / musx-hint / SOTA-downbeat
```
`downbeat_anchor.sota_downbeat_phase` becomes one `_downbeat_phase` strategy. Every existing
kill-switch (HARMONIA_GRID_ANCHOR, _SOTA, _BEAT_PERIOD_MODE, MUSX_ONSET_HINT…) preserved as
params/env on the composable steps. **DEFERRED because** `chord_pipeline_v1.py`, `render_youtube_chart.py`,
AND `harmonia_server.py` are all under active concurrent WIP right now (_fifth_corrected_quality,
downbeat_anchor integration, _snap fix, jam-mode, iReal-import) — an in-place extraction (which
must DELETE the old scattered functions) would clobber. Execute once that WIP lands; behavior-
preserving, verified by diffing matched-set outputs before/after (target: zero diff).
