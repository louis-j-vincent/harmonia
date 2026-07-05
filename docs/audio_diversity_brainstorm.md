# Audio rendering diversity — everything we can vary to make training audio richer

Brainstorm (2026-07-06) for maximizing diversity + realistic noise in the rendered
training set. Goal: cover the variation a real recording has, so the model doesn't
overfit the clean synthetic distribution. Grouped by axis; ✅ = already implemented.

## 1. Instrumentation & arrangement (the SOURCE)
- Soundfont per stem (grand/upright piano, EP/Rhodes, guitar, organ, strings, synth) ✅ (2 SFs)
- Different MMA grooves / styles (swing, bossa, samba, rock, ballad, funk, blues…) ✅ (per-song)
- Ensemble size: solo / trio / quartet / big-band (add or drop stems)
- GM program per stem (vary the comping & bass instrument)
- Comping density: sparse vs busy grooves; "Plus"/"Sus" MMA variants
- Voicing variation per section occurrence (drop tones, inversions, octaves) ✅
- Lead melody, varying instrument (flute/sax/trumpet/voice) ✅
- Bass style: walking / root-only / pedal / slap; octave doubling
- Doubling: unison/octave instruments, section unisons

## 2. Performance (the PLAYING)
- Timing humanization: swing ratio, laid-back/on-top feel, per-note jitter ✅ (jitter)
- Velocity/dynamics variation, crescendos ✅ (per-note)
- Tempo drift / rubato (accelerando, ritardando, human wobble)
- Pitch: slight detuning per instrument, out-of-tune ensemble, tape-speed drift
- Ornaments / passing tones in the melody ✅ (some)
- Independent take per repeat (each chorus played differently) — the key one for FOLDING

## 3. Acoustic / recording chain (the CAPTURE)
- Room reverb: size, damping, early reflections ✅ (on/off)
- Mic distance / placement: EQ tilt, level, off-axis coloration
- Band-limiting: phone (150 Hz–6 kHz), AM radio, old 78, laptop speaker ✅ (phone)
- Mains hum 50/60 Hz + harmonics ✅
- Broadband hiss / tape hiss ✅
- Codec artifacts: low-bitrate MP3/AAC, packet loss/dropouts
- Vinyl: crackle, pops, wow & flutter; Tape: saturation, wow, print-through
- Clipping / overdrive / AGC pumping ✅ (soft-clip)
- Bit-crush / sample-rate reduction ✅
- Compression / limiting (radio-style), multiband

## 4. Environment / background (the ROOM)
- Café / restaurant chatter, street traffic, crowd, applause
- TV/other music bleed, air-con hum, fridge, footsteps
- Wind / handling noise (phone in hand), pops
- Distance-varying background level

## 5. Mix balance (the FADERS)
- Loud drums / masking lead melody / quiet comping ✅ (scenarios)
- Panning, mono collapse, stereo width
- Per-section balance changes (verse quieter than chorus)

## 6. Dynamics over time (NON-UNIFORM — critical for structure folding)
- Time-varying gain / noise / muffling (mic movement, distance) ✅ (`time_varying_degrade`)
- Fades in/out, dropouts, glitches, level automation
- **Degrade each section occurrence INDEPENDENTLY** so repeats differ → the only way
  audio-fold helps (learned this the hard way: uniform degradation leaves repeats
  correlated → folding useless).

## 7. Musical / global
- Corpus/style: jazz, pop, blues, country, dixieland ✅ (5 corpora, symbolic; jazz+pop audio)
- Transposition / key coverage ✅
- Tempo range, time signatures (3/4, 5/4, 12/8)
- Reharmonization / chord substitutions between repeats

## Priorities to add next (highest realism-per-effort)
1. **Independent-per-repeat degradation** (unlocks folding) — apply §6 per section.
2. **Codec (low-bitrate MP3) + vinyl/tape** — very common real-world, distinct artifacts.
3. **Background ambience** (café/street) — cheap (mix a noise bed), big realism gain.
4. **Detuning / tempo drift** — breaks the "perfectly-in-tune, perfectly-on-grid" tell.
5. **More soundfonts / ensembles** — the source itself is currently only 2 SFs.
