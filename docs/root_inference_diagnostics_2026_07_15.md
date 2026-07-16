# Root-inference diagnostics + source-separation screen (2026-07-15)

Two-part session on the Billboard BP48 real-audio root problem (current best
root acc 54–56% test, `billboard_bp48_60_rollaug_v1`). Part 1 makes the
"muddy pitch-activation" story *visible*; Part 2 screens source separation as
a pre-processing step and reports a clear negative.

Predictions in all plots come from the **read-only shipped checkpoint**
`data/models/billboard_bp48_60_rollaug_v1.pt` (not retrained here). The four
songs are all **train** songs — chosen deliberately so generalization is not a
confound: on the hard cases the model has *seen* the audio and still fails,
which sharpens the "intrinsic audio difficulty" reading.

## Part 1 — what the plots show

Songs (all from the 58-song `billboard_bp48_60_fixed_beatgrid.npz`):

| song | artist / title | model root acc | inv frac | role |
|------|----------------|----------------|----------|------|
| bb_1111 | Chris Kenner – Land of 1000 Dances | 0.99 | 0% | clean control |
| bb_362  | Wednesday – Last Kiss | 0.10 | 0% | pure muddy / fifth-confusion |
| bb_887  | De La Soul – Me Myself and I | 0.70 | 44% | inversion story |
| bb_1027 | Greg Kihn – Lucky | 0.32 | 49% | inv-heavy + low acc |

Two PNGs per song in `docs/plots/`:
- `root_diag_A_chroma_<sid>.png` — waveform + per-beat **bass** chroma (0–52
  MIDI register) + full **note** chroma heatmaps, with GT chord roots overlaid
  (teal = root-position, pink = inversion).
- `root_diag_B_rootvsgt_<sid>.png` — waveform + GT root (thick, teal/pink) vs
  model-predicted root (thin black, dashed when wrong); orange spans mark
  **P4/P5 (fifth) errors**; a pink ▼ marks a prediction that equals the
  sounding slash-bass PC.

### New visual insight (beyond the existing numbers)

1. **The easy-vs-hard difference lives almost entirely in the BASS panel, not
   the note panel.** In every song the *full note chroma* (bottom panel) is a
   near-uniform purple wash — Basic Pitch's `note_probs` folded to 12 PC carry
   almost no root information (this is the known "note_probs is near-constant"
   property, now visible). Discrimination is carried by the bass-register
   onset chroma. On bb_1111 (clean) that bass panel is a single dominant bright
   row (D♯, with A♯ = the fifth also lit) → root argmax is trivial and the
   model scores 99%. On bb_362 (hard) the bass energy is **smeared across E,
   C♯ and others while the GT root hops** — the bass parks on a pedal-ish PC
   while the harmony moves, which is exactly the fifth-confusion mechanism seen
   as a picture.

2. **Fifth-confusion is legible as "prediction sticks to the fifth of the
   bass."** In `root_diag_B_*` the orange P4/P5 spans dominate the wrong
   predictions on bb_887 (De La Soul is an E↔A two-chord sample loop; errors
   are the E↔A perfect-fifth swap) — confirming the corpus-level ~0.36–0.42
   P4/P5 share as a per-song visual, not just an aggregate.

3. **Inversion errors here do NOT mostly land on the sounding bass PC.** Across
   bb_887 / bb_1027, the "pred = slash-bass" marker almost never fires
   (`pred_eq_bass_share_of_inv_err ≈ 0`). This is *weaker* than the earlier
   corpus finding (#3: "36% of inversion errors predict the sounding bass").
   Interpretation: on these particular songs the inversion errors are ordinary
   fifth/муddy errors, not clean "heard-the-bass-reported-it-as-root" cases —
   a reminder that the 36% is a corpus average, not a per-song rule. bb_1027's
   inversion accuracy is catastrophic (0.086) and is dominated by generic
   muddiness, not a bass-vs-functional-root story.

## Part 2 — source separation as a pre-processing step

### Literature (does prior work do this?)

Yes, this is established ground — new to *this project's* notes but not new to
MIR:
- **HPSS as chroma cleanup**: classic pre-processor — suppress percussion,
  extract chroma from the harmonic component, for chord estimation / HMM
  pipelines (Ono et al.; AudioLabs Erlangen HPSS course notes).
- **Demucs/HTDemucs stems for chord recognition**: APSIPA 2025 "Accuracy
  Improvement of Automatic Chord Recognition…" uses HTDemucs to make
  drum-removed / drum-and-vocal-removed / **isolated-bass** stems, runs chord
  recognition on each, and for the **bass stem retains only root/bass-note
  info** — essentially the exact hypothesis here. Daniel Ko, "Automatic Chord
  Recognition by Music Source Separation." A Sep-2025 LLM-CoT chord paper
  (arXiv 2509.18700) also uses HTDemucs for per-instrument volume balancing
  before recognition.

So the idea is sound in principle; the question is whether it helps *this*
front-end (Demucs stem → Basic Pitch → onset-chroma argmax).

### Feasibility

- No Demucs/Spleeter installed, but `torchaudio.pipelines.HDEMUCS_HIGH_MUSDB`
  needs **no pip install** — just a 319 MB weight download (deleted after the
  screen; disk held ~2.1–2.4 GB free throughout).
- CPU inference ~30–60 s per 2.5-min song. Stems + 44.1 k WAVs deleted
  immediately (disk discipline).

### Cheap screen (mandated: isolated bass stem, 1–2 songs, controlled)

Per-chord `onset`-chroma **argmax→root** on **root-position chords only**, same
songs, same GT intervals. `sharp` = peak/mean of the 12-vector (higher =
less muddy). Repro: `scratchpad/bass_stem_screen.py`,
`scratchpad/bass_stem_screen_results.json`.

| song | mix bass acc | mix sharp | **isolated-bass acc** | bass sharp | drums-removed acc | nodrum sharp |
|------|-------------:|----------:|----------------------:|-----------:|------------------:|-------------:|
| bb_362 (hard) | 0.072 | 2.44 | **0.085** | 4.38 | 0.084 | 3.87 |
| bb_1111 (clean) | 0.977 | 4.43 | **0.188** | 4.54 | 0.988 | 4.76 |

### Result — NEGATIVE (isolated bass) / NEUTRAL (drums-removed)

- **Isolated Demucs bass stem is a net negative.** It *does* sharpen the chroma
  (bb_362 peak/mean 2.44 → 4.38) but the peak sits on the **wrong** PC, so
  accuracy is unchanged on the hard song (0.072 → 0.085) and **collapses** on
  the clean song (0.977 → 0.188). Cause: **Basic Pitch cannot reliably
  transcribe a solo bass stem** — with no harmonic context and very-low-frequency
  content, it makes octave/PC errors. A cleaner *source* is not a cleaner
  *root* through this transcriber.
- **Drums-removed mix is essentially neutral** (bb_362 0.072 → 0.084; bb_1111
  0.977 → 0.988; both within noise). Critically it does **not** rescue the hard
  song — so the muddiness on bb_362 is **not** percussion interference; it is
  tonal/harmonic ambiguity in the bass content itself, which separation cannot
  add information to.

This corroborates finding #4 ("a sharper bass *feature* alone… didn't clear
the SNR wall") and #1 (genuine audio-domain difficulty) with a second,
independent lever: cleaning the *audio source* (not just re-weighting the
feature) also fails.

### Recommendation: ABANDON (for this front-end); one bounded avenue to revisit

Do **not** spend budget on a full separated-source BP48 rebuild + retrain. The
mandated cheap screen is strongly negative and the milder recipe is neutral.

The one honest caveat: the failure is specifically **Demucs-stem → Basic
Pitch**. The supporting literature runs a *dedicated chord/bass model* on
stems, not Basic Pitch. So a cheap, no-retrain follow-up worth flagging for
later is: run a **monophonic pitch tracker (pYIN or CREPE) on the Demucs bass
stem** to read the bass note directly, bypassing Basic Pitch's weak
low-register polyphonic transcription — this is the only route by which
separation might still help root, and it does not require rebuilding the
corpus. Not attempted this session (out of screen scope; the isolated-bass
premise already failed for the pipeline as-is).
