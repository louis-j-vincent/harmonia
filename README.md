# Harmonia

**Hierarchical Bayesian chord and structure inference from audio.**

Harmonia transcribes any audio recording into a chord chart — tempo, key, modulations, and chord symbols — using a multi-level probabilistic model grounded in jazz and music theory.

Unlike frame-level neural chord recognisers, Harmonia reasons at multiple timescales simultaneously:

```
Audio
  ↓
P(note, frame)          ← Basic Pitch: soft piano-roll, not hard MIDI
  ↓
Beat grid               ← madmom: tempo-aware beat tracking
  ↓
Structural segments     ← Bayesian changepoint detection
  ↓
Key per segment         ← Krumhansl-Schmuckler + Dirichlet prior
  ↓
Chord sequence          ← HMM: P(chord | notes, key, prev_chord, style)
  ↓
Chord chart             ← JSON / PDF output
```

The key insight: **chord inference is conditioned on inferred key**. A Bb in a C major context is almost certainly not a Bbmaj chord. Scale-agnostic jazz priors (II-V-I, tritone subs, cycle of fifths) further constrain the posterior — encoded once in relative form and instantiated at inference time for any key.

---

## Why this is different from Chordify

| | Chordify | Harmonia |
|---|---|---|
| Model type | Frame-level CNN | Hierarchical Bayesian |
| Key conditioning | No | Yes — per structural segment |
| Jazz theory priors | None | II-V-I, tritone sub, cycle of 5ths, modal interchange |
| Chord vocabulary | Major/minor/7th | Triads → 7ths → 9ths → 11ths → 13ths (phased) |
| Sus chords | Partial | Full (sus2, sus4, 7sus4, 9sus4) |
| Modulation detection | No | Yes — Bayesian changepoint |
| Uncertainty | None | Full posterior over chord vocabulary |
| Style-aware tempo prior | No | Yes — ballad / swing / bebop / blues / modal |

---

## Installation

```bash
# Clone
git clone https://github.com/yourusername/harmonia.git
cd harmonia

# Install (Python 3.10+)
pip install -e ".[dev]"

# For MIDI rendering (data pipeline only)
brew install fluidsynth        # macOS
# apt install fluidsynth       # Linux
pip install -e ".[render]"
```

---

## Quick start

```python
from pathlib import Path
from harmonia.models.stage1_pitch import PitchExtractor
from harmonia.theory.key_profiles import infer_key, activations_to_chroma

# Extract soft note activations from any audio file
extractor = PitchExtractor(cache_dir=Path("data/cache"))
activations = extractor.extract(Path("my_recording.wav"))

# Infer key from the full track chroma
chroma = activations.chroma()
key = infer_key(chroma)
print(f"Detected key: {key.key_name}  (confidence: {key.confidence:.2f})")

# Full inference pipeline (chord chart) — coming in v0.2
# from harmonia.pipeline import HarmoniaPipeline
# chart = HarmoniaPipeline().run(Path("my_recording.wav"))
# chart.print()
```

---

## Data pipeline

### 1. Download POP909 (labelled training data — free)

```bash
git clone https://github.com/music-x-lab/POP909-Dataset data/pop909
```

### 2. Parse and inspect

```python
from harmonia.data.pop909_parser import POP909Parser

parser = POP909Parser("data/pop909")
songs = parser.parse_all(max_songs=50)
print(f"Loaded {len(songs)} songs")
print(parser.chord_statistics(songs))
```

### 3. Render your own piano recordings

Record yourself playing through chord progressions and export MIDI from your DAW, then:

```python
from harmonia.data.midi_renderer import MIDIRenderer

renderer = MIDIRenderer(soundfont_dir="data/soundfonts")
variants = renderer.render_variants(
    midi_path="my_recording.mid",
    output_dir="data/renders/",
    n_variants=8,    # 8 timbral variants from one MIDI
)
```

Download free soundfonts:
- [Salamander Grand Piano](https://freepats.zenvoid.org/Piano/) — best piano
- [GeneralUser GS](https://schristiancollins.com/generaluser.php) — full GM

---

## Chord vocabulary

Phase 1 (current): **121 chord types** — triads, suspended, all 7th qualities  
Phase 2: + 9ths and altered dominants (7b9, 7#9, 9sus4)  
Phase 3: + 11ths (maj9#11 Lydian, min11, 7#11)  
Phase 4: + 13ths  

All encoded as interval sets with soft acoustic weights. The model reasons about chords as pitch-class probability distributions, not binary note sets — which is why it handles ambiguous voicings and missing 5ths correctly.

---

## Jazz priors (scale-agnostic)

All progressions are encoded **relative to the tonic** as (semitone_interval, quality) pairs. They are instantiated at inference time for whatever key the model inferred. One prior definition covers all 12 transpositions.

| Prior | Relative form | Weight |
|---|---|---|
| II-V-I major | iim7 → V7 → Imaj7 | 1.0 |
| II-V-I minor | iiø7 → V7 → im7 | 0.85 |
| Tritone sub | iim7 → bII7 → Imaj7 | 0.7 |
| Rhythm changes bridge | III7 → VI7 → II7 → V7 | 0.6 |
| Cycle of 5ths (×4) | vim7 → iim7 → V7 → Imaj7 | 0.75 |
| Borrowed bVII | bVII → I | 0.5 |
| 12-bar blues | I7 × 4 → IV7 × 2 → I7 × 2 → V7 → IV7 → I7 → V7 | 0.7 |

Style priors further condition on tempo: a II-V-I at 280 BPM (bebop) spans 2 beats; the same at 55 BPM (ballad) spans 8 bars.

---

## Project structure

```
harmonia/
├── theory/
│   ├── chord_vocabulary.py   # chord types, interval templates, phase hierarchy
│   ├── jazz_priors.py        # scale-agnostic progressions, style priors
│   └── key_profiles.py       # Krumhansl-Schmuckler, Bayesian key inference
├── models/
│   ├── stage1_pitch.py       # Basic Pitch wrapper → P(note, frame)
│   ├── rhythm.py             # madmom beat tracking (coming)
│   ├── structure.py          # Bayesian changepoint / segmentation (coming)
│   └── chord_hmm.py          # Hierarchical chord HMM (coming)
├── data/
│   ├── midi_renderer.py      # FluidSynth MIDI → audio, timbral augmentation
│   └── pop909_parser.py      # POP909 chord annotation parser
└── eval/
    └── mirex_eval.py         # MIREX chord evaluation metrics (coming)
```

---

## Roadmap

- [x] Chord vocabulary with phase hierarchy (triads → 13ths)
- [x] Scale-agnostic jazz priors with tempo-aware style weights
- [x] Krumhansl-Schmuckler Bayesian key inference
- [x] Basic Pitch Stage 1 wrapper with caching
- [x] FluidSynth MIDI renderer with timbral augmentation
- [x] POP909 dataset parser
- [ ] madmom beat tracking integration
- [ ] Bayesian structural segmentation
- [ ] Chord HMM with NumPyro
- [ ] MIREX evaluation harness
- [ ] CLI: `harmonia infer my_song.mp3`
- [ ] PDF / MusicXML chord chart export
- [ ] 9th / 11th / 13th vocabulary extension

---

## References

- Krumhansl, C.L. (1990). *Cognitive Foundations of Musical Pitch*. Oxford.
- Bitteur et al. (2022). [Basic Pitch](https://arxiv.org/abs/2203.06616). ICASSP.
- Wang et al. (2020). [POP909](https://arxiv.org/abs/2008.07142). ISMIR.
- Böck et al. (2016). [madmom](https://arxiv.org/abs/1605.07008).

---

## License

MIT
