"""Synthetic chord-progression MIDI generator + fluidsynth renderer.

Generates 'songs' of chord progressions with EXACT known ground truth
(root, 7-way quality, inversion/sounding-bass), matching RWC-Popular's
quality vocabulary and inversion distribution (~87.6% root / 12.4% inverted).

Renders to WAV with fluidsynth using a real quality soundfont, with
instrument/velocity/tempo/voicing diversity so the audio is not a single
sterile timbre.

Harte labels emitted are exactly those the project's parse_jaah understands:
  maj -> C:maj   min -> C:min   dom -> C:7   hdim -> C:hdim7
  dim -> C:dim7  aug -> C:aug   sus -> C:sus4
Inversions as Harte degree tokens (/3 /5 /b3 /b7 ...), rendered so the
sounding bass note matches the label.
"""
from __future__ import annotations
import subprocess, tempfile, os
from pathlib import Path
import numpy as np
import pretty_midi

NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

# 7-way quality -> (shorthand tail, chord-tone semitones above root)
QUAL = {
    "maj":  ("maj",   (0, 4, 7)),
    "min":  ("min",   (0, 3, 7)),
    "dom":  ("7",     (0, 4, 7, 10)),
    "hdim": ("hdim7", (0, 3, 6, 10)),
    "dim":  ("dim7",  (0, 3, 6, 9)),
    "aug":  ("aug",   (0, 4, 8)),
    "sus":  ("sus4",  (0, 5, 7)),
}
# RWC-Popular observed quality frequencies (dominated by maj/min, some dom).
QUAL_W = {"maj": 0.50, "min": 0.28, "dom": 0.14, "sus": 0.04,
          "hdim": 0.02, "dim": 0.015, "aug": 0.005}

# Bass degree token -> semitones above root. Only tokens that are real chord
# tones for the given quality are used (see valid_inversions()).
DEG_SEMI = {"3": 4, "b3": 3, "5": 7, "b7": 10, "7": 11, "2": 2, "4": 5,
            "6": 9, "b6": 8, "b5": 6}
# RWC inversion-degree weights (from corpus_schema note: /3 707 /5 491 /b3 108
# /2 175 /b7 95 /4 38 /7 14 ...). Applied only when a chord is chosen inverted.
INV_DEG_W = {"3": 0.42, "5": 0.29, "2": 0.10, "b3": 0.07, "b7": 0.06,
             "4": 0.03, "7": 0.02, "6": 0.01}
INV_FRAC = 0.124  # target fraction of chords that are inverted

# Chord (comping) GM programs and bass GM programs — diversity of timbre.
CHORD_PROGRAMS = [0, 4, 24, 25, 48, 19, 26, 11]   # piano, epiano, gtrs, strings, organ, jazz gtr, vibes
BASS_PROGRAMS  = [32, 33, 34, 43]                  # acoustic/finger/pick bass, contrabass


def _valid_inv_degrees(qual: str) -> list[str]:
    tones = set(QUAL[qual][1])
    out = []
    for tok, semi in DEG_SEMI.items():
        if semi in tones and semi != 0:
            out.append(tok)
    return out


def _mk_label(root_pc: int, qual: str, inv_tok: str | None) -> str:
    tail = QUAL[qual][0]
    lab = f"{NOTE_NAMES[root_pc]}:{tail}"
    if inv_tok:
        lab += f"/{inv_tok}"
    return lab


def gen_progression(rng: np.random.RandomState, n_chords: int) -> list[tuple]:
    """Return list of (root_pc, qual, inv_tok_or_None, beats)."""
    quals = list(QUAL_W); qw = np.array([QUAL_W[q] for q in quals]); qw /= qw.sum()
    prog = []
    root = rng.randint(12)
    # diatonic-ish walk: move by common intervals (4th/5th/2nd) for musicality
    steps = [5, 7, 2, 10, 9, 3]
    for _ in range(n_chords):
        qual = rng.choice(quals, p=qw)
        inv_tok = None
        if rng.rand() < INV_FRAC:
            valid = _valid_inv_degrees(qual)
            if valid:
                w = np.array([INV_DEG_W.get(t, 0.01) for t in valid]); w /= w.sum()
                inv_tok = rng.choice(valid, p=w)
        beats = int(rng.choice([2, 2, 4, 4, 4, 1, 8]))
        prog.append((root, qual, inv_tok, beats))
        root = (root + rng.choice(steps)) % 12
    return prog


def build_midi(prog, tempo, rng, chord_prog, bass_prog):
    """Render a progression to a pretty_midi object + list of (t0,t1,label)."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=float(tempo))
    ci = pretty_midi.Instrument(program=chord_prog)
    bi = pretty_midi.Instrument(program=bass_prog)
    spb = 60.0 / tempo
    t = 0.0
    ann = []
    for root, qual, inv_tok, beats in prog:
        dur = beats * spb
        tones = QUAL[qual][1]
        bass_semi = DEG_SEMI[inv_tok] if inv_tok else 0
        bass_pc = (root + bass_semi) % 12
        # Bass note: octave 2 (MIDI ~ 36-47)
        bass_midi = 36 + bass_pc
        vel_b = int(rng.randint(70, 100))
        bi.notes.append(pretty_midi.Note(vel_b, bass_midi, t + 0.005, t + dur - 0.01))
        # Chord voicing in octave 4-5, root position of tones (bass carries inv)
        base = 60  # C4
        vel_c = int(rng.randint(55, 90))
        # slight strum / voicing variation
        for k, semi in enumerate(tones):
            note = base + ((root + semi) % 12)
            if rng.rand() < 0.3:
                note += 12  # spread voicing
            onset = t + (0.0 if rng.rand() < 0.7 else rng.uniform(0, 0.04))
            ci.notes.append(pretty_midi.Note(vel_c + rng.randint(-8, 8),
                                             note, onset + 0.002, t + dur - 0.01))
        ann.append((t, t + dur, _mk_label(root, qual, inv_tok)))
        t += dur
    pm.instruments.append(ci)
    pm.instruments.append(bi)
    return pm, ann


def add_melody(pm, prog, tempo, rng, program=73):
    """Add a busy melody/arpeggio layer (chord tones + occasional passing tones)
    to raise spectral density toward real multi-instrument audio."""
    mi = pretty_midi.Instrument(program=program)  # 73 flute / lead
    spb = 60.0 / tempo
    t = 0.0
    for root, qual, inv_tok, beats in prog:
        tones = list(QUAL[qual][1])
        dur = beats * spb
        nsub = max(1, int(beats * 2))  # eighth-note motion
        sd = dur / nsub
        for k in range(nsub):
            if rng.rand() < 0.25:  # rests for phrasing
                continue
            semi = rng.choice(tones)
            if rng.rand() < 0.2:
                semi = (semi + rng.choice([1, 2, -1, -2])) % 12  # passing tone
            note = 72 + ((root + semi) % 12)  # octave 5
            on = t + k * sd
            mi.notes.append(pretty_midi.Note(int(rng.randint(50, 80)), note,
                                             on + 0.003, on + sd * 0.9))
        t += dur
    pm.instruments.append(mi)


def add_noise_wav(path: str, snr_db: float, rng):
    """Inject band-limited broadband noise post-render (mimics mix/mastering
    floor + percussion spread that real audio has and clean synth lacks)."""
    import soundfile as sf
    x, sr = sf.read(path)
    if x.ndim > 1:
        x = x.mean(1)
    rms = np.sqrt((x ** 2).mean()) + 1e-9
    noise = rng.randn(len(x)).astype(np.float32)
    # mild low-pass so it's colored (pinkish), not pure white
    from numpy import convolve
    k = np.ones(8) / 8
    noise = convolve(noise, k, mode="same")
    noise *= (rms / (np.sqrt((noise ** 2).mean()) + 1e-9)) * (10 ** (-snr_db / 20))
    y = (x + noise).astype(np.float32)
    y /= (np.abs(y).max() + 1e-9)
    sf.write(path, y, sr)


def render_wav(pm, sf2: str, out_wav: str, sr: int = 22050):
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
        midi_path = f.name
    pm.write(midi_path)
    cmd = ["fluidsynth", "-ni", "-g", "0.8", "-r", str(sr), "-F", out_wav, sf2, midi_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    os.unlink(midi_path)
    if not os.path.exists(out_wav) or os.path.getsize(out_wav) < 1000:
        raise RuntimeError(f"fluidsynth failed: {r.stderr[-400:]}")
    return out_wav
