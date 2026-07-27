"""STATED vs IMPLIED harmony — is a chord actually being *played* right now?

WHY THIS EXISTS
---------------
2026-07-27, Louis's ear, on the two worst-scoring slices of the whole frozen
benchmark (Blue Bossa's contrabass solo, ~28.6 s marked ``N`` by the model; and
passages of Stand By Me, ~10 s):

    "For stand by me also only the bass is present — it is enough to deduce the
    chord from the general harmony but no chord is being played, same as the
    blue bossa live bass solo section."

Both slices had been filed as a chord-vs-no-chord *bug*: the model emits ``N``,
the reference asserts a chord.  They are not a bug.  **The model is acoustically
correct — no chord is played.**  The reference asserts the *implied* harmony,
which a chart legitimately does and an acoustic chord detector legitimately
cannot hear.  This also retroactively explains why every per-instant acoustic
signal tried on those regions (chroma novelty, onset strength, HPSS) was
refuted: there was nothing to hear.

Regions therefore need a **STATED / IMPLIED** label before anything else can be
said about them, and — per the repo's "corrections → automation" rule — that
label must come from a general detector, not from hand-marking two songs.

THE DETECTOR
------------
One number per frame: mid-register harmonic energy, relative to the song's own
median.

    y                -> HPSS harmonic component (drop percussion)
    CQT(y_harm)      -> semitone bins, MIDI 28..95
    mid(t)           -> sum of bins MIDI 55..95  (above bass/contrabass range)
    smooth 2 s       -> moving average
    rel(t)           -> smooth(mid) / median(mid over the analysed span)
    IMPLIED where rel(t) < 0.45, for at least 3 s

MIDI 55 (G3) is the floor because an upright bass tops out around G3 and an
electric bass around C4; energy above it is comping/melody, not the bass line.
Dividing by the song's own median makes it loudness-invariant, so a *quiet but
comped* passage is not misread as implied — the failure mode of raw energy.

CALIBRATION — PRE-REGISTERED, AND DELIBERATELY NOT TUNED TO ANY SCORE
--------------------------------------------------------------------
The threshold was fixed on Louis's ear-anchors **before any accuracy number was
computed with it**, and has not been moved since:

    IMPLIED anchors   stand_by_me intro / 28–38 s, blue_bossa 399–427 s
                      rel = 0.13 … 0.33
    STATED  anchors   blue_bossa head / solo / post-solo, the 150 bpm backing
                      track, close_to_you, georgia, bein_green, every_breath
                      rel = 0.67 … 1.52

The gap between the two clusters is 0.33 → 0.67; ``IMPLIED_REL_THR = 0.45`` sits
in it.  **Do not re-fit this constant on the benchmark.**  Its whole evidentiary
value is that it was set from the ear and then measured, in that order.  If a
future session wants a different operating point, add a *named* second threshold
and report both — do not move this one.

Unaided validation: on Blue Bossa the detector located the contrabass solo at
397–440 s with no hint of where to look, matching both Louis's ear and the
model's own 28.6 s ``N`` span.

*Reading those anchor numbers back.*  They were computed as
``median(raw mid over the window) / median(raw mid over the whole file)``.  This
module instead reports ``median`` of the already-smoothed, span-normalised
``rel``, which is not the same summary: on the one anchor whose texture changes
*inside* the window (stand_by_me 28–38 s) it reads 0.454 rather than 0.315,
because the 2 s smoother pulls in the thickening that starts around 31 s.  Every
other anchor moves by <0.1, and swapping the normaliser from whole-file to
analysed-span moves the divisor by ≤8.6% on all 7 songs.  **The detector itself
is unchanged**: the per-frame decision and therefore the emitted spans are
bit-identical to the prototype that fixed the threshold (verified span-by-span,
max endpoint difference 0.000000000 s on all 7 songs).  The discrepancy is in how
a *window* is summarised, not in what the detector does — and the one window it
affects is precisely the gradual-fade-in case of limitation 2 below.

WHAT THIS DOES **NOT** SOLVE  (CLAUDE.md rule #4)
-------------------------------------------------
1. **It is song-relative, so a wholly bass-only track defeats it.**  The median
   is taken over the analysed span, which assumes the song is *mostly* STATED.
   A solo-bass recording, a walking-bass etude, or a track that is bass-only
   from end to end has its own bass level as the median and will be reported
   100% STATED.  There is no absolute-loudness fallback here; if you need one,
   `mid_register_energy` returns the raw un-normalised curve.
2. **It is conservative on gradual fade-ins, by construction.**  Stand By Me's
   texture thickens bar by bar rather than switching on: the pre-registered 0.45
   flags 11.3 s of what the ear calls bass-only, where 0.60 would flag 23.7 s and
   0.75 would flag 33.7 s.  That under-call is a *known, accepted* cost of not
   tuning the threshold.  It means IMPLIED spans should be read as
   high-precision / low-recall: what it flags is implied, but it does not flag
   everything that is.  Measured against the shipped model's own ``N`` emissions
   on the 7 frozen songs (an independent instrument — the model never sees this
   detector), the 0.45 operating point gives **precision 0.630 / recall 0.619**
   over 64.6 s of model-``N``; on Blue Bossa, where the passage is unambiguous,
   **0.828 / 0.831**.  Total flagged time by threshold, all 7 songs:
   0.35 → 20.4 s, **0.45 → 65.8 s**, 0.60 → 130.8 s, 0.75 → 251.8 s.
3. **It says "no chord is sounding", not "no chord exists".**  An implied region
   still has harmony; the detector only reports that the acoustic evidence for
   it is absent.  Naming the chord there requires form / repetition / bass
   inference, which this module does not do and which does not yet exist in the
   package (finding #9: "repetition/bass is the only honest path for them").
4. **It does not separate IMPLIED from genuinely CHORDLESS.**  Silence, a drum
   break, a spoken intro and a bass solo all read as low mid-register energy.
   IMPLIED here means "the harmonic register is empty", nothing more.  Any
   downstream policy that assumes "implied ⇒ a chord exists and we may state it"
   must gate on something else as well.
5. **It has no opinion about *which* chord, and none about time.**  Spans are
   quantised only by the 2 s smoother and the 3 s minimum-span rule; boundaries
   are soft to roughly ±1 s.  Do not use them as chord boundaries.
6. **A big dynamic arc can fool it.**  Loudness-invariance is only relative to
   the song median; a genuinely quiet-but-comped bridge in a very loud song can
   fall under 0.45.  None of the 7 frozen songs does, but the set is 7 songs.
7. **Cost.**  HPSS + CQT at 22.05 kHz mono is ~1–3 s per song, single-threaded.
   Not free; not cached here.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "IMPLIED_REL_THR", "TextureCurve", "mid_register_energy", "texture_curve",
    "implied_spans", "implied_mask", "spans_from_env",
]

# ── analysis constants (see CALIBRATION above; do not re-fit on a score) ──────
SR = 22050
HOP = 512
BAND_LO_MIDI = 28          # CQT floor (E1) — low enough to include the bass
BASS_TOP_MIDI = 55         # G3 — above an upright bass's practical top
BAND_HI_MIDI = 95          # B6 — above this is mostly noise/cymbal wash
SMOOTH_S = 2.0
MIN_SPAN_S = 3.0
IMPLIED_REL_THR = 0.45     # PRE-REGISTERED on ear anchors. Do not tune.


@dataclass(frozen=True)
class TextureCurve:
    """Per-frame harmonic-texture readout for one song.

    Attributes:
        times:  (T,) frame times, seconds.
        rel:    (T,) smoothed mid-register energy / the span's median.  ``< 1``
                means "thinner above the bass than this song's typical bar".
        raw:    (T,) the same energy before smoothing and normalisation.
        median: the divisor actually used (raw units), for auditability.
    """

    times: np.ndarray
    rel: np.ndarray
    raw: np.ndarray
    median: float

    def rel_at(self, t: float) -> float:
        """``rel`` at wall-clock time ``t`` (nearest frame)."""
        return float(self.rel[int(np.clip(np.searchsorted(self.times, t), 0,
                                          len(self.times) - 1))])


def mid_register_energy(audio_path: str | Path, *, sr: int = SR,
                        hop: int = HOP) -> tuple[np.ndarray, np.ndarray]:
    """Raw mid-register harmonic energy envelope.

    Returns ``(times, energy)``: the CQT magnitude of the HPSS *harmonic*
    component summed over MIDI ``BASS_TOP_MIDI..BAND_HI_MIDI``.  Un-smoothed and
    un-normalised — this is the only absolute-scale output of this module.
    """
    import librosa

    y, _ = librosa.load(str(audio_path), sr=sr, mono=True)
    y_harm, _ = librosa.effects.hpss(y)
    cqt = np.abs(librosa.cqt(y_harm, sr=sr, hop_length=hop,
                             fmin=librosa.midi_to_hz(BAND_LO_MIDI),
                             n_bins=BAND_HI_MIDI - BAND_LO_MIDI,
                             bins_per_octave=12))
    times = librosa.frames_to_time(np.arange(cqt.shape[1]), sr=sr,
                                   hop_length=hop)
    energy = cqt[BASS_TOP_MIDI - BAND_LO_MIDI:].sum(0)
    return times, energy


def _moving_average(x: np.ndarray, times: np.ndarray, win_s: float) -> np.ndarray:
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 1.0
    w = max(3, int(round(win_s / dt)) | 1)
    pad = w // 2
    return np.convolve(np.pad(x, pad, mode="edge"),
                       np.ones(w) / w, "valid")[:len(x)]


def texture_curve(audio_path: str | Path, t0: float | None = None,
                  t1: float | None = None, *, smooth_s: float = SMOOTH_S,
                  ) -> TextureCurve:
    """Song-relative harmonic-texture curve.

    ``t0``/``t1`` restrict the span the **median is taken over** (not the frames
    returned) — pass the scored/annotated span so a long silent tail cannot drag
    the normaliser down.  See limitation 1: the median assumes a mostly-STATED
    song.
    """
    times, raw = mid_register_energy(audio_path)
    smooth = _moving_average(raw, times, smooth_s)
    sel = np.ones(len(times), dtype=bool)
    if t0 is not None:
        sel &= times >= t0
    if t1 is not None:
        sel &= times <= t1
    if not sel.any():
        sel[:] = True
    median = float(np.median(raw[sel]))
    return TextureCurve(times=times, rel=smooth / (median + 1e-9), raw=raw,
                        median=median)


def implied_spans(curve: TextureCurve, t0: float | None = None,
                  t1: float | None = None, *,
                  threshold: float = IMPLIED_REL_THR,
                  min_span_s: float = MIN_SPAN_S) -> list[list[float]]:
    """Contiguous ``[start, end]`` spans where the harmony is only IMPLIED.

    Spans shorter than ``min_span_s`` are dropped (not merged): a 1 s dip
    between two comped chords is a gap in the comping, not an implied passage.
    Boundaries are soft to roughly ±1 s — never use them as chord boundaries.
    """
    t, rel = curve.times, curve.rel
    sel = np.ones(len(t), dtype=bool)
    if t0 is not None:
        sel &= t >= t0
    if t1 is not None:
        sel &= t <= t1
    if not sel.any():
        return []
    tt, mm = t[sel], (rel < threshold)[sel]
    spans: list[list[float]] = []
    cur, start = bool(mm[0]), float(tt[0])
    for i in range(1, len(mm)):
        if bool(mm[i]) != cur:
            if cur:
                spans.append([start, float(tt[i])])
            cur, start = bool(mm[i]), float(tt[i])
    if cur:
        spans.append([start, float(tt[-1])])
    return [s for s in spans if (s[1] - s[0]) >= min_span_s]


def implied_mask(times: np.ndarray, spans: list[list[float]]) -> np.ndarray:
    """(len(times),) bool — True inside any span.  ``times`` may be any grid."""
    times = np.asarray(times, float)
    out = np.zeros(len(times), dtype=bool)
    for a, b in spans:
        out |= (times >= a) & (times < b)
    return out


def spans_from_env(default: float = IMPLIED_REL_THR) -> float:
    """``HARMONIA_IMPLIED_THR`` override, for *diagnostics only*.

    Provided so an experiment can sweep the operating point without editing the
    pre-registered constant.  Production must use ``IMPLIED_REL_THR``; a swept
    value is not a calibrated value (see CALIBRATION).
    """
    raw = os.environ.get("HARMONIA_IMPLIED_THR")
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default
