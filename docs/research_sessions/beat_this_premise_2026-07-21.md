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
