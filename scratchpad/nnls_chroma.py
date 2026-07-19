"""nnls_chroma.py — from-scratch NNLS Chroma (Mauch & Dixon, ISMIR 2010).

Pipeline (faithful to the VAMP NNLS-Chroma reference, simplified):
  1. Constant-Q log-frequency spectrogram, 3 bins/semitone, A0 (27.5 Hz) up.
  2. Spectral whitening: subtract running mean over ~1 octave, half-wave rectify.
  3. NNLS approximate transcription: fit a dictionary of harmonic note-profiles
     (geometric harmonic decay s^(h-1), s=0.7) to each whitened frame via
     scipy.optimize.nnls -> per-note activations. THIS is the overtone-
     suppression step: a note's harmonics are modelled explicitly, so energy at
     (e.g.) a root's 3rd/5th harmonic is explained by the root note, not the
     neighbouring pitch class.
  4. Fold note activations to 12-pc chroma (full, bass-register, treble-register).

Index convention: chroma[0] = C (unlike McGill's VAMP output which starts at A,
hence the '+9 shift' seen in the corpus features).
"""
from __future__ import annotations
import numpy as np
import librosa
from scipy.optimize import nnls
from scipy.ndimage import uniform_filter1d

SR = 22050
BPS = 3                     # bins per semitone
BPO = 12 * BPS             # 36 bins/octave
FMIN = librosa.note_to_hz("A0")   # 27.5 Hz, MIDI 21
N_OCT = 8
N_BINS = BPO * N_OCT       # 288 CQT bins
HOP = 1024                 # ~21.5 fps

MIDI_LO, MIDI_HI = 21, 105          # dictionary note fundamentals (A0..~A7)
N_HARM = 20
S_DECAY = 0.7

# CQT bin center frequencies and their log positions (bins)
_cqt_freqs = FMIN * 2.0 ** (np.arange(N_BINS) / BPO)


def _bin_of_freq(f):
    """Continuous CQT-bin index of frequency f."""
    return BPO * np.log2(f / FMIN)


def build_dictionary():
    """(N_BINS, n_notes) harmonic note-profile matrix E and note MIDI list."""
    notes = np.arange(MIDI_LO, MIDI_HI + 1)
    E = np.zeros((N_BINS, len(notes)))
    for j, m in enumerate(notes):
        f0 = librosa.midi_to_hz(m)
        for h in range(1, N_HARM + 1):
            fh = h * f0
            if fh >= _cqt_freqs[-1]:
                break
            center = _bin_of_freq(fh)
            amp = S_DECAY ** (h - 1)
            # triangular kernel, +/- 1 semitone (BPS bins) around center
            lo = int(np.floor(center - BPS)); hi = int(np.ceil(center + BPS))
            for b in range(max(0, lo), min(N_BINS, hi + 1)):
                w = max(0.0, 1.0 - abs(b - center) / BPS)
                E[b, j] += amp * w
    # normalise each note profile to unit energy
    E /= (np.linalg.norm(E, axis=0, keepdims=True) + 1e-9)
    return E, notes


def whiten(logspec):
    """Running-mean subtraction over ~1 octave along log-freq, half-wave rect."""
    bg = uniform_filter1d(logspec, size=BPO, axis=0, mode="nearest")
    return np.maximum(logspec - bg, 0.0)


def extract(wav_path):
    """Return dict of per-frame data: note activations (n_notes,T), frame times,
    and the dictionary note MIDI list."""
    y, sr = librosa.load(str(wav_path), sr=SR, mono=True)
    C = np.abs(librosa.cqt(y, sr=SR, hop_length=HOP, fmin=FMIN,
                           n_bins=N_BINS, bins_per_octave=BPO))
    C = whiten(C)
    E, notes = build_dictionary()
    T = C.shape[1]
    A = np.zeros((len(notes), T))
    for t in range(T):
        col = C[:, t]
        if col.max() < 1e-6:
            continue
        act, _ = nnls(E, col)
        A[:, t] = act
    times = librosa.frames_to_time(np.arange(T), sr=SR, hop_length=HOP)
    return {"act": A, "notes": notes, "times": times}


def note_pc_maps(notes):
    """Boolean masks folding notes->pc for full/bass/treble registers.
    bass = MIDI < 52 (matches _reg_raw bass 0-52 upper bound); treble = MIDI>=52."""
    pc = notes % 12
    return pc, notes < 52, notes >= 52


def block_chroma(A, notes, times, t0, t1):
    """Sum activations over [t0,t1), fold to (full12, bass12, treble12) chroma."""
    pc, bmask, tmask = note_pc_maps(notes)
    m = (times >= t0) & (times < t1)
    if m.sum() == 0:
        j = np.argmin(np.abs(times - 0.5 * (t0 + t1)))
        m = np.zeros(len(times), bool); m[j] = True
    seg = A[:, m].sum(1)              # (n_notes,)
    full = np.zeros(12); bass = np.zeros(12); treb = np.zeros(12)
    for k, p in enumerate(pc):
        full[p] += seg[k]
        if bmask[k]: bass[p] += seg[k]
        if tmask[k]: treb[p] += seg[k]
    return full, bass, treb
