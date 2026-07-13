# Tempo-octave disambiguator — results (2026-07-14)

**Verdict: UNSOLVABLE from audio-internal signals.** The true octave is always
present among the candidates (**oracle 8/8**), but no self-referential signal
selects it. Best blind strategy = **3/8 (38%)**, far below the 8/10 target. The
failure is *structural*, not a tuning problem — the audio-internal cues are
octave-symmetric, and the only cue that isn't (an absolute tempo prior) is
external information that provably breaks on tunes whose true tempo is far from
its centre.

- Script: `scripts/disambiguate_octave.py` (self-contained; reruns everything)
- Data: `docs/octave_disambiguator_data.json`
- Plot: `docs/plots/octave_accuracy_per_song.png`

## Setup

For each song we take librosa's base detection, build octave-related candidates
(`base × {1/3,1/2,2/3,3/4,1,4/3,3/2,2,3}`, clamped to 40–300 BPM — this set
always contains the true octave), and score each candidate with four independent
signals. "Correct octave" = pick within ±32% of GT (`|log2(pick/GT)| < 0.4`),
lenient enough for the soft GTs but tight enough to reject the nearest wrong
candidate (3/4 and 4/3 sit at |log2|=0.415).

**Ground truth** (rule #3: GT is a measurement too):

| trust | songs | source |
|---|---|---|
| hard | blue_bossa_150 (150), adele hello (79), let it be (72) | filename / documented |
| soft | autumn leaves (120), ghost (65), airegin (225), a foggy day (150), blue bossa (155) | style tempo, good to ~±30% (factor-2 error still unambiguous) |
| none | nina simone feeling good, kermit being green | genuinely unknown — reported, **excluded** from accuracy |

Accuracy is over the **8 hard+soft songs**.

## Per-version results

| version | signal | accuracy | why it fails |
|---|---|---:|---|
| **V1 harmonic rhythm** | beats-per-chord near a musical integer (chord changes from beat-independent chroma novelty) | **1/8 (12%)** | **octave-symmetric by construction** — if 1× gives bpc≈2, then 2× gives ≈4 and ½× gives ≈1; *all* land on musical integers, so the score ties across octaves. Also collapses on static-harmony bossas (novelty finds no changes → ICI ~45 s). |
| **V2 chroma stability** | (folded into V1) | — | reduces to the same beat-grid-independent novelty curve; a chord boundary is a chord boundary regardless of how many beats you slice it into. Same symmetry as V1. |
| onset ACF | autocorrelation of onset envelope at the beat lag | 3/8 (38%) | classic tempo salience — but has peaks at the true period **and every multiple**, and systematically prefers the **faster** (2×) octave (shorter lag → more repetitions). On 2 of 3 hard songs it actively favours 2×GT. |
| metrical alternation | strong/weak onset alternation (fires when a grid is 2× too fast) | — (near-zero everywhere, ~0.01–0.08) | the one audio cue that *could* break the 2× symmetry, but full-band mixes have continuous comping/bass filling every subdivision, so off-beats are as loud as beats. No usable contrast. |
| tempo prior | log-normal centred 120 BPM | 3/8 (38%) | not really a "signal" — a bias. Right only when true tempo ∈ ~[95,160]; wrong on every ballad (65/72/79) and the bebop head (225). |
| **combined** (ACF×prior×(1−alt)) | best blind version | **3/8 (38%)** | inherits the prior's ceiling; ACF drags it toward the fast octave. |
| **oracle** | closest candidate to GT | **8/8 (100%)** | proves the information is in the candidate set — the problem is *selection*, not *coverage*. |

## The ambiguity, made explicit (theoretical test)

For the 3 hard-GT songs, compare the true octave against 2×GT on every
audio-internal signal:

| song | signal | @GT | @2×GT | separable? |
|---|---|---:|---:|---|
| blue_bossa_150 | acf | 0.391 | 0.412 | → prefers 2×GT (**wrong**) |
| blue_bossa_150 | hr / alt | 0.000 / 0.006 | 0.000 / 0.004 | tie |
| adele hello | acf | 0.241 | 0.256 | tie (→ slightly wrong) |
| adele hello | hr / alt | 0.832 / 0.017 | 0.832 / 0.006 | tie |
| let it be | acf | 0.288 | 0.388 | → prefers 2×GT (**wrong**) |
| let it be | hr | 0.936 | 0.938 | tie |
| let it be | alt | 0.077 | 0.026 | → GT (only weak win in the whole table) |

The true octave and its double are acoustically **indistinguishable** on the
harmonic-rhythm and alternation signals, and onset-ACF *prefers the wrong one*.
This matches the human-perception literature (McKinney & Moelants): listeners
themselves disagree by a factor of two on a large fraction of music — tempo
octave is genuinely bimodal, and the disambiguating information is not in the
signal, it's in the listener's learned genre/tempo prior.

## Why the prior can't reach 8/10 either

A single log-normal prior can only be centred once. Our corpus true tempos span
**65 → 225 BPM** (a factor of 3.5). Centre it at 120 and you correctly pull
autumn/foggy/bossa into place but wrongly pull the ballads (Hello 79, Let It Be
72, Ghost 65) *up* to ~120 and the bebop (Airegin 225) *down*. No fixed centre
wins more than ~half. A per-song prior would need to know the tempo — circular.

## Conclusion / recommendation

- **Do not build a blind audio-only octave disambiguator** — the premise (a
  self-referential signal that ranks the true octave first) is falsified. This
  is CLAUDE.md rule #2 (screen the premise cheaply) paying off: 4 signals, one
  afternoon, clear negative.
- **The lever that actually works is external tempo info**, in order of value:
  1. **Style/genre prior** — the pipeline already infers style
     (`infer_style_posteriors`); a ballad→[50–90], bossa→[120–160],
     bebop→[180–260] band would fix the ballads and bebop the fixed 120-prior
     misses. This is the recommended next step (turns the prior from
     single-centre into style-conditioned).
  2. **Human confirmation** — a one-tap "is this the beat?" in the UI. Cheap,
     exact, and honest about the fundamental ambiguity.
  3. **iReal/lead-sheet metadata** when a chart is attached (many carry a tempo
     marking) — highest trust, per the GT hierarchy.
- **The tracker choice is irrelevant** (confirms known_issues #9 + the madmom
  finding): both librosa and madmom land in [55,215] and pick a wrong multiple;
  a downstream chooser is the only fix, and a *blind* one caps at ~38%.
