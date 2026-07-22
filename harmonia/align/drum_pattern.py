"""Stage 1 of the Bayesian fusion aligner — the DRUM BEAT / TEMPO tracker.

This is the standalone instrument that Stage 0 validated (see
``docs/fusion_aligner_design.md`` for the full spec + the Stage-0 result). It
tracks the BEAT PHASE, the local TEMPO, and the strong-beat PAIR from the drum
kit's percussive onset envelope, holding the grid through solos / dense mixes
where harmonic agreement collapses. It becomes the drum observation term
(stream #2) of the Stage-2 fusion DBN.

WHAT IT IS NOT.  This is **not a downbeat detector**. Stage 0 proved drums alone
cannot give beat 1: the backbeat's 2-beat symmetry (kick 1&3 / snare 2&4) is
real, but which member of the pair is "1" is not recoverable from drums
(Autumn's beat 1 is actually *softer* than 2&4). So the tracker outputs the
strong-beat PAIR (the half-tempo 1&3-vs-2&4 grid) and explicitly leaves beat-1
disambiguation to the fusion (chart/form prior + bass + harmonic rhythm).

THREE STAGE-0 REQUIREMENTS, and how they are met here:

1. **Octave-anchored tempo** — naive drum-envelope autocorrelation octave-slips
   on 2/4 measures (subdivision / 3-beat). We take a tempo-octave prior from
   Beat This! (median beat period) — or the head — and LOCK the octave: PLP is
   run with a tempo band of only ``+/- octave_tol`` around the prior (default
   +/-18%), which brackets the true beat but excludes 2x / 0.5x. The beat
   PERIOD is then refined *from the drum onset envelope* within that band
   (`estimate_period_in_octave`), so drums set the period but can never slip.

2. **Learn where DRUMS are steady, not where harmony is confident** — the
   harmony-confident region can be drum-less (Let It Be's clean intro is solo
   piano). `select_drum_anchor` auto-selects the window of highest, most
   regular percussive onset density to seed phase (and to phase-anchor the grid
   to the prior, removing the onset-envelope group delay). `drum_reliability`
   returns the per-time steadiness/energy curve = the DBN's weight driver
   w_drum(t) (HIGH in solos, LOW in the piano intro).

3. **Track beat phase through the whole song** — PLP (predominant local pulse,
   a comb-filter over the Fourier tempogram) on the drum onset envelope,
   octave-locked, self-organises the phase through solos/dense mixes; beats are
   its local maxima.

OUTPUT API (`DrumBeatTrack`, all documented on the dataclass):
  * ``beat_times``            — tracked beat times (s).
  * ``local_tempo(t)``        — local tempo (BPM) at time t; ``local_period(t)``
                                is the same in seconds (= 60 / tempo).
  * ``beat_likelihood_at(t)`` — the per-candidate-beat DRUM BEAT-LIKELIHOOD: how
                                well time t sits on the drum grid, in [0, 1].
                                THIS is the DBN observation term L_drum(t). Also
                                available pre-sampled per tracked beat as
                                ``beat_likelihood``.
  * ``reliability_at(t)``     — the DBN weight driver w_drum(t): drum
                                energy x pulse steadiness, in [0, 1].
  * ``strong_beat_times`` + ``strong_beat_parity`` + ``strong_beat_confidence``
                                — the strong-beat PAIR (half-tempo 1&3 grid),
                                explicitly NOT the downbeat (see above).

NON-CIRCULARITY.  Nothing here touches the model's chord decode. The only
external input is Beat This! (an independent NN beat tracker) used purely for
the tempo-octave prior and a single global phase anchor in the drum-steady
window; the whole through-the-song track comes from the raw drum onset
envelope. The tracker also runs with no Beat This! at all (audio-free unit
tests): pass ``prior_period`` directly and it uses the drum grid's own phase.

Feature recipe (frozen from the Stage-0 premise check, drum_premise2.py):
HPSS-percussive (margin 2.0) -> mel spectrogram (128 mels) -> librosa onset
strength, broadband + low(<155 Hz, kick) / mid / high(>1600 Hz, snare+hat)
bands, at SR=22050, HOP=256 (~86.1 fps).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np

log = logging.getLogger("harmonia.align.drum_pattern")

# ── frozen feature-extraction constants (Stage-0 recipe) ─────────────────────
SR = 22050
HOP = 256
_HPSS_MARGIN = 2.0
_N_MELS = 128
_LOW_HZ = 155.0        # kick band ceiling
_HIGH_HZ = 1600.0      # snare/hat band floor
_FPS = SR / HOP        # ~86.13 onset frames / s

# ── tracker defaults ─────────────────────────────────────────────────────────
_OCTAVE_TOL = 0.18     # PLP tempo band = prior * (1 +/- this). 2x/0.5x excluded.
_PLP_WIN = 384         # PLP tempogram window (frames) ~ 4.5 s local-tempo scope
_ANCHOR_WIN_S = 8.0    # drum-steady anchor-window length
_ANCHOR_HOP_S = 2.0
_REL_WIN_S = 4.0       # reliability (steadiness) window
_REL_HOP_S = 0.5
_PHASE_SNAP_FRAC = 0.5  # phase anchor: match prior beats within +/- this * period


# ═════════════════════════════════════════════════════════════════════════════
# Feature extraction
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class OnsetEnvelopes:
    """HPSS-percussive onset-strength envelopes (Stage-0 recipe).

    ``times`` are frame times (s). ``broad`` is the broadband envelope used for
    beat tracking; ``low``/``mid``/``high`` are the <155 / 155-1600 / >1600 Hz
    band envelopes (kick / body / snare+hat) used for the strong-beat pair.
    """
    times: np.ndarray
    broad: np.ndarray
    low: np.ndarray
    mid: np.ndarray
    high: np.ndarray
    sr: int = SR
    hop: int = HOP


def percussive_onset_envelopes(y: np.ndarray, sr: int = SR,
                               hop: int = HOP) -> OnsetEnvelopes:
    """Compute the frozen Stage-0 percussive onset envelopes from a mono signal.

    HPSS (margin 2.0) isolates percussion; a 128-mel dB spectrogram of the
    percussive part feeds ``librosa.onset.onset_strength`` (broadband) and
    ``onset_strength_multi`` split at 155 / 1600 Hz.
    """
    import librosa

    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        sr = SR
    y = np.ascontiguousarray(y, dtype=np.float32)
    _, y_p = librosa.effects.hpss(y, margin=_HPSS_MARGIN)
    S = librosa.feature.melspectrogram(y=y_p, sr=sr, hop_length=hop, n_mels=_N_MELS)
    Sdb = librosa.power_to_db(S)
    mel_f = librosa.mel_frequencies(n_mels=_N_MELS, fmin=0.0, fmax=sr / 2)
    lo = int(np.searchsorted(mel_f, _LOW_HZ))
    hi = int(np.searchsorted(mel_f, _HIGH_HZ))
    broad = librosa.onset.onset_strength(S=Sdb, sr=sr, hop_length=hop)
    multi = librosa.onset.onset_strength_multi(
        S=Sdb, sr=sr, hop_length=hop, channels=[0, lo, hi, _N_MELS])
    times = librosa.frames_to_time(np.arange(len(broad)), sr=sr, hop_length=hop)
    return OnsetEnvelopes(times=times, broad=np.asarray(broad, float),
                          low=np.asarray(multi[0], float),
                          mid=np.asarray(multi[1], float),
                          high=np.asarray(multi[2], float), sr=sr, hop=hop)


# ═════════════════════════════════════════════════════════════════════════════
# Octave-anchored period + drum-steady anchor window + reliability
# ═════════════════════════════════════════════════════════════════════════════

def _autocorr(x: np.ndarray) -> np.ndarray:
    x = x - x.mean()
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    return ac


def estimate_period_in_octave(broad: np.ndarray, prior_period: float,
                              sr: int = SR, hop: int = HOP,
                              octave_tol: float = _OCTAVE_TOL,
                              span: Optional[tuple[float, float]] = None,
                              ) -> float:
    """Refine the beat PERIOD *from the drum onset envelope*, locked to the
    prior's octave.

    The autocorrelation of ``broad`` is peak-picked ONLY within the lag band
    ``[prior/(1+tol), prior*(1+tol)]`` (seconds). Because 2x / 0.5x fall well
    outside an 18% band, the refined period can drift with the drums but can
    never octave-slip — the Stage-0 calibration caveat. ``span`` (t0, t1) limits
    the estimate to a window (e.g. the anchor window); default = whole song.
    Falls back to ``prior_period`` if the band is empty.
    """
    times = np.arange(len(broad)) * (hop / sr)
    if span is not None:
        m = (times >= span[0]) & (times <= span[1])
        seg = broad[m]
    else:
        seg = broad
    if len(seg) < 8 or prior_period <= 0:
        return prior_period
    ac = _autocorr(seg)
    lo = int(np.floor(prior_period / (1 + octave_tol) * sr / hop))
    hi = int(np.ceil(prior_period * (1 + octave_tol) * sr / hop))
    lo = max(1, lo)
    hi = min(len(ac) - 1, hi)
    if hi <= lo:
        return prior_period
    band = ac[lo:hi + 1]
    lag = lo + int(np.argmax(band))
    # parabolic interpolation around the integer peak for sub-frame precision
    if 0 < lag < len(ac) - 1:
        y0, y1, y2 = ac[lag - 1], ac[lag], ac[lag + 1]
        denom = (y0 - 2 * y1 + y2)
        shift = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
        shift = float(np.clip(shift, -0.5, 0.5))
    else:
        shift = 0.0
    return (lag + shift) * hop / sr


@dataclass
class AnchorWindow:
    """The auto-selected drum-steady window used to seed phase / period.

    ``score`` = normalized percussive density x pulse clarity (the max over all
    candidate windows). ``density`` is the mean onset strength; ``clarity`` is
    the beat-lag autocorrelation ratio (0..1). Fields ``win_times``/``win_score``
    hold the full per-window curves for inspection/plots.
    """
    t0: float
    t1: float
    score: float
    density: float
    clarity: float
    win_times: np.ndarray = field(default_factory=lambda: np.array([]))
    win_score: np.ndarray = field(default_factory=lambda: np.array([]))


def _pulse_clarity(seg: np.ndarray, lag: int) -> float:
    """Autocorr(seg)[lag] / autocorr(seg)[0], clipped to [0, 1] — how strongly a
    window pulses at the beat period (the Stage-0 'drum beat-clarity')."""
    if len(seg) <= lag or lag <= 0:
        return 0.0
    ac = _autocorr(seg)
    if ac[0] <= 0:
        return 0.0
    return float(np.clip(ac[lag] / ac[0], 0.0, 1.0))


def select_drum_anchor(broad: np.ndarray, period: float, sr: int = SR,
                       hop: int = HOP, win_s: float = _ANCHOR_WIN_S,
                       hop_s: float = _ANCHOR_HOP_S) -> AnchorWindow:
    """Auto-select the drum-steady anchor window (requirement #2).

    Slides a ``win_s`` window (hop ``hop_s``) over the broadband envelope and
    scores each by (density x clarity), where density = mean onset strength
    (z-scored to a 0..1 rank over windows) and clarity = beat-lag autocorrelation
    ratio. The best window seeds phase/period and is where drums are steadiest —
    which need NOT be where harmony is confident (Stage-0 scoping catch).
    """
    fps = sr / hop
    w = max(1, int(win_s * fps))
    step = max(1, int(hop_s * fps))
    lag = int(round(period * fps))
    starts = list(range(0, max(1, len(broad) - w), step))
    dens = np.array([broad[s:s + w].mean() for s in starts])
    clar = np.array([_pulse_clarity(broad[s:s + w], lag) for s in starts])
    # rank-normalise density to 0..1 so it is comparable to clarity
    if np.ptp(dens) > 0:
        dens_n = (dens - dens.min()) / np.ptp(dens)
    else:
        dens_n = np.ones_like(dens)
    score = dens_n * clar
    win_times = np.array([(s + w / 2) / fps for s in starts])
    if len(score) == 0:
        return AnchorWindow(0.0, len(broad) / fps, 0.0, 0.0, 0.0)
    b = int(np.argmax(score))
    t0 = starts[b] / fps
    return AnchorWindow(t0=float(t0), t1=float(t0 + win_s), score=float(score[b]),
                        density=float(dens[b]), clarity=float(clar[b]),
                        win_times=win_times, win_score=score)


def drum_reliability(broad: np.ndarray, period: float, sr: int = SR,
                     hop: int = HOP, win_s: float = _REL_WIN_S,
                     hop_s: float = _REL_HOP_S) -> tuple[np.ndarray, np.ndarray]:
    """The DBN weight driver w_drum(t): local drum energy x pulse steadiness.

    Returns ``(rel_times, rel)`` in [0, 1]. HIGH where the kit drives a steady
    pulse (solos), LOW where drums are absent/erratic (Let It Be's piano intro).
    This is the reliability the fusion uses to up-weight the drum stream exactly
    where harmonic agreement fails.
    """
    fps = sr / hop
    w = max(1, int(win_s * fps))
    step = max(1, int(hop_s * fps))
    lag = int(round(period * fps))
    starts = list(range(0, max(1, len(broad) - w), step))
    dens = np.array([broad[s:s + w].mean() for s in starts])
    clar = np.array([_pulse_clarity(broad[s:s + w], lag) for s in starts])
    dens_n = dens / (np.percentile(dens, 95) + 1e-9) if len(dens) else dens
    dens_n = np.clip(dens_n, 0.0, 1.0)
    rel = dens_n * clar
    rel_times = np.array([(s + w / 2) / fps for s in starts])
    return rel_times, rel


# ═════════════════════════════════════════════════════════════════════════════
# The tracker
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class DrumBeatTrack:
    """Output of the drum beat/tempo tracker — the Stage-2 DBN's stream #2.

    Attributes
    ----------
    beat_times : (K,) float
        Tracked beat times (s), octave-locked, phase-anchored.
    beat_likelihood : (K,) float in [0, 1]
        Per-tracked-beat drum grid-fit strength (the PLP pulse at each beat).
    period : float
        Global octave-locked beat period (s), refined from the drum envelope.
    tempo_bpm : float
        60 / period.
    local_periods : (K,) float
        Local beat period (s) at each tracked beat (smoothed inter-beat spacing,
        clamped to the octave band) — tracks slow drift (e.g. Blue Bossa).
    pulse : (F,) float / pulse_times : (F,) float
        The frame-level PLP pulse curve and its (phase-corrected) times — the raw
        drum beat-likelihood observation the DBN samples at arbitrary times.
    reliability / reliability_times : (R,) float
        The DBN weight driver w_drum(t) in [0, 1] (see `drum_reliability`).
    strong_beat_mask : (K,) bool
        Per-beat metrical class (parity-flip-immune): True where the tracked beat
        is on the STRONG (1&3) grid, False on the backbeat (2&4) grid. This is
        the authoritative strong-beat-pair output. NOT a downbeat.
    strong_beat_parity : int (0 or 1)
        Index parity of the strong grid in the drum-steady anchor window — a
        SEED/reference only (a single global parity is unreliable because the
        tracker can insert/drop a beat; use ``strong_beat_mask``).
    strong_beat_times : (~K/2,) float
        The strong-beat PAIR grid = ``beat_times[strong_beat_mask]`` (the
        half-tempo 1&3 grid). ``backbeat_times`` is the complementary 2&4 grid.
    strong_beat_confidence : float in [0, 1]
        Kick/snare parity asymmetry — how cleanly the pair is resolved.
    anchor : AnchorWindow
        The drum-steady window used to seed phase/period.
    octave_locked : bool
        Sanity flag: refined period within the octave band of the prior (no slip).
    prior_period : float
        The tempo-octave prior period (s) the octave was locked to.
    phase_offset_s : float
        Global phase shift applied to align to the prior in the anchor window
        (removes the onset-envelope group delay); 0 when no prior beats given.
    """
    beat_times: np.ndarray
    beat_likelihood: np.ndarray
    period: float
    tempo_bpm: float
    local_periods: np.ndarray
    pulse: np.ndarray
    pulse_times: np.ndarray
    reliability: np.ndarray
    reliability_times: np.ndarray
    strong_beat_parity: int
    strong_beat_times: np.ndarray
    backbeat_times: np.ndarray
    strong_beat_confidence: float
    anchor: AnchorWindow
    octave_locked: bool
    prior_period: float
    strong_beat_mask: np.ndarray = field(default_factory=lambda: np.array([], bool))
    phase_offset_s: float = 0.0

    # ── DBN-facing query methods ────────────────────────────────────────────
    def beat_likelihood_at(self, t: Union[float, np.ndarray]
                           ) -> Union[float, np.ndarray]:
        """Drum beat-likelihood L_drum(t) at arbitrary time(s): the PLP pulse
        interpolated at t, in [0, 1]. This is the DBN observation term — high on
        the drum grid, low between beats."""
        return np.interp(t, self.pulse_times, self.pulse, left=0.0, right=0.0)

    def reliability_at(self, t: Union[float, np.ndarray]
                       ) -> Union[float, np.ndarray]:
        """Drum reliability weight w_drum(t) at arbitrary time(s), in [0, 1]."""
        if len(self.reliability_times) == 0:
            return np.zeros_like(t) if np.ndim(t) else 0.0
        return np.interp(t, self.reliability_times, self.reliability,
                         left=self.reliability[0], right=self.reliability[-1])

    def local_period(self, t: Union[float, np.ndarray]
                     ) -> Union[float, np.ndarray]:
        """Local beat period (s) at time(s) t (interpolated, octave-clamped)."""
        if len(self.beat_times) == 0:
            return np.full_like(t, self.period) if np.ndim(t) else self.period
        return np.interp(t, self.beat_times, self.local_periods,
                         left=self.local_periods[0], right=self.local_periods[-1])

    def local_tempo(self, t: Union[float, np.ndarray]
                    ) -> Union[float, np.ndarray]:
        """Local tempo (BPM) at time(s) t = 60 / local_period(t)."""
        return 60.0 / self.local_period(t)

    def strong_beat_phase_at(self, t: Union[float, np.ndarray]):
        """For time(s) t, 0 if the nearest tracked beat is on the STRONG (1&3)
        pair, else 1 (the 2&4 backbeat pair). NOT a downbeat. Uses the per-beat
        ``strong_beat_mask`` (parity-flip-immune)."""
        bt = self.beat_times
        if len(bt) == 0 or len(self.strong_beat_mask) != len(bt):
            return np.zeros_like(t, dtype=int) if np.ndim(t) else 0
        idx = np.searchsorted(bt, t)
        idx = np.clip(idx, 1, len(bt) - 1)
        left = idx - 1
        pick = np.where(np.abs(bt[idx] - t) < np.abs(bt[left] - t), idx, left)
        phase = np.where(self.strong_beat_mask[pick], 0, 1)
        return int(phase) if np.ndim(t) == 0 else phase.astype(int)


def _local_periods(beat_times: np.ndarray, prior_period: float,
                   octave_tol: float, k: int = 2) -> np.ndarray:
    """Per-beat local period = median inter-beat interval over +/-k beats,
    clamped to the octave band (so a single dropped/extra beat can't halve/double
    the local tempo)."""
    if len(beat_times) < 2:
        return np.full(len(beat_times), prior_period)
    d = np.diff(beat_times)
    d = np.concatenate([d, d[-1:]])                 # length K
    out = np.empty(len(beat_times))
    for i in range(len(beat_times)):
        lo, hi = max(0, i - k), min(len(d), i + k + 1)
        out[i] = np.median(d[lo:hi])
    lo_b, hi_b = prior_period / (1 + octave_tol), prior_period * (1 + octave_tol)
    return np.clip(out, lo_b, hi_b)


def _sample_env_at(env: np.ndarray, times: np.ndarray, t: np.ndarray,
                   half_w: float) -> np.ndarray:
    """Max of ``env`` within +/- half_w of each time in ``t``."""
    out = np.zeros(len(t))
    for i, ti in enumerate(t):
        lo = int(np.searchsorted(times, ti - half_w))
        hi = int(np.searchsorted(times, ti + half_w))
        if hi > lo:
            out[i] = env[lo:hi].max()
        elif 0 <= lo < len(env):
            out[i] = env[min(lo, len(env) - 1)]
    return out


_SB_WIN = 4     # +/- beats of the local window that classifies each beat's grid


def _strong_beat_pair(beat_times: np.ndarray, low: np.ndarray, high: np.ndarray,
                      times: np.ndarray, period: float,
                      anchor: "AnchorWindow"):
    """Resolve the strong-beat PAIR (strong 1&3 / backbeat 2&4) — NOT the
    downbeat.

    Per-beat and PARITY-FLIP-IMMUNE: a single global index parity washes out
    because the beat tracker occasionally inserts/drops a beat (flipping every
    downstream parity). Instead each beat is classified LOCALLY — over a
    +/-``_SB_WIN``-beat neighbourhood, is its own metrical parity the one that
    carries the snare/backbeat (high-band) accent? The snare (2&4) grid is the
    clear accent (Stage 0: 'swing accents 2&4'); the STRONG pair (1&3) is its
    complement. Returns:
      * ``strong_mask`` (K,) bool — True where the beat is on the STRONG (1&3)
        grid, False on the backbeat (2&4) grid.
      * ``confidence`` in [0, 1] — snare-accent separation between the two
        classified grids (high => a clean backbeat).
      * ``anchor_parity`` — the strong-grid index parity in the drum-steady
        anchor window (a seed/reference; NOT globally reliable by itself).
    """
    K = len(beat_times)
    if K < 4:
        return np.ones(K, bool), 0.0, 0
    half_w = 0.12 * period
    hi_b = _sample_env_at(high, times, beat_times, half_w)   # snare / backbeat
    W = _SB_WIN
    is_backbeat = np.zeros(K, bool)
    for i in range(K):
        lo, hi = max(0, i - W), min(K, i + W + 1)
        seg = hi_b[lo:hi]
        par = (np.arange(lo, hi) - i) % 2          # 0 = same parity as beat i
        m_same = seg[par == 0].mean() if np.any(par == 0) else 0.0
        m_other = seg[par == 1].mean() if np.any(par == 1) else 0.0
        is_backbeat[i] = m_same > m_other          # beat i sits on the snare grid
    strong_mask = ~is_backbeat
    eps = 1e-9
    mb = hi_b[is_backbeat].mean() if np.any(is_backbeat) else 0.0
    ms = hi_b[strong_mask].mean() if np.any(strong_mask) else 0.0
    conf = float(abs(mb - ms) / (mb + ms + eps))
    # anchor-window seed parity of the STRONG grid (reference only)
    aw = (beat_times >= anchor.t0) & (beat_times <= anchor.t1)
    if np.any(aw & strong_mask):
        anchor_parity = int(np.median(np.flatnonzero(aw & strong_mask)) % 2)
    else:
        anchor_parity = 0
    return strong_mask, conf, anchor_parity


def track_drum_beats(broad: np.ndarray, prior_period: float,
                     low: Optional[np.ndarray] = None,
                     high: Optional[np.ndarray] = None,
                     beat_this_beats: Optional[np.ndarray] = None,
                     sr: int = SR, hop: int = HOP,
                     octave_tol: float = _OCTAVE_TOL,
                     plp_win: int = _PLP_WIN) -> DrumBeatTrack:
    """Track drum beats from the percussive onset envelope, octave-locked.

    Parameters
    ----------
    broad : (F,) float
        Broadband percussive onset envelope (`percussive_onset_envelopes`).
    prior_period : float
        Tempo-octave prior beat period (s), e.g. median Beat This! beat spacing.
        LOCKS the octave; drums refine the period within +/- ``octave_tol``.
    low, high : (F,) float, optional
        Kick / snare band envelopes for the strong-beat pair. If omitted the
        broadband envelope is used for both (weaker pair confidence).
    beat_this_beats : (N,) float, optional
        Independent prior beat times. Used ONLY to (a) confirm the octave and
        (b) set the global phase in the drum-steady anchor window (remove the
        onset-envelope group delay). The through-song track is drums-only. If
        omitted, the tracker keeps the PLP grid's own phase.

    Returns
    -------
    DrumBeatTrack
    """
    import librosa

    fps = sr / hop
    # 1) octave-anchored period, refined from the drums within the prior's octave
    period = estimate_period_in_octave(broad, prior_period, sr, hop, octave_tol)
    bpm0 = 60.0 / period
    octave_locked = (prior_period / (1 + octave_tol) - 1e-6
                     <= period <= prior_period * (1 + octave_tol) + 1e-6)

    # 2) drum-steady anchor window (seeds phase; where drums are steadiest)
    anchor = select_drum_anchor(broad, period, sr, hop)

    # 3) octave-locked PLP -> pulse curve + beats (holds through solos/dense mix)
    tempo_min = 60.0 / (prior_period * (1 + octave_tol))
    tempo_max = 60.0 / (prior_period / (1 + octave_tol))
    pulse = librosa.beat.plp(onset_envelope=np.asarray(broad, float), sr=sr,
                             hop_length=hop, win_length=plp_win,
                             tempo_min=tempo_min, tempo_max=tempo_max)
    beat_frames = np.flatnonzero(librosa.util.localmax(pulse))
    frame_times = np.arange(len(pulse)) * (hop / sr)
    beat_times = frame_times[beat_frames].astype(float)

    # 4) global phase anchor to the prior (single constant shift = the
    #    onset-envelope group delay). Estimated as the ROBUST MEDIAN over ALL
    #    matched beats, not one short window: the offset is a global constant, so
    #    a whole-song median averages out local micro-timing (an 8 s window
    #    over-fits — e.g. Let It Be). Only beats with a prior beat within
    #    +/- _PHASE_SNAP_FRAC * period count (rejects drum-less stretches). The
    #    THROUGH-SONG track stays 100% drums; only this one scalar uses the prior.
    phase_offset = 0.0
    if beat_this_beats is not None and len(beat_this_beats) and len(beat_times):
        bt = np.asarray(beat_this_beats, float)
        offs = []
        for x in bt:
            j = int(np.argmin(np.abs(beat_times - x)))
            d = beat_times[j] - x
            if abs(d) <= _PHASE_SNAP_FRAC * period:
                offs.append(d)
        if offs:
            phase_offset = float(np.median(offs))
    beat_times = beat_times - phase_offset
    pulse_times = frame_times - phase_offset

    # 5) per-beat likelihood (pulse at each beat) + local period
    beat_likelihood = np.interp(beat_times, pulse_times, pulse, left=0.0, right=0.0)
    if pulse.max() > 0:
        pulse = pulse / pulse.max()
        beat_likelihood = beat_likelihood / max(beat_likelihood.max(), 1e-9)
    local_periods = _local_periods(beat_times, prior_period, octave_tol)

    # 6) reliability weight curve (DBN weight driver)
    rel_t, rel = drum_reliability(broad, period, sr, hop)

    # 7) strong-beat PAIR (strong 1&3 / backbeat 2&4) — NOT the downbeat
    lo_env = np.asarray(low if low is not None else broad, float)
    hi_env = np.asarray(high if high is not None else broad, float)
    strong_mask, sb_conf, anchor_parity = _strong_beat_pair(
        beat_times, lo_env, hi_env, pulse_times, period, anchor)
    strong_beat_times = beat_times[strong_mask]
    backbeat_times = beat_times[~strong_mask]

    return DrumBeatTrack(
        beat_times=beat_times, beat_likelihood=beat_likelihood, period=float(period),
        tempo_bpm=float(bpm0), local_periods=local_periods, pulse=pulse,
        pulse_times=pulse_times, reliability=rel, reliability_times=rel_t,
        strong_beat_parity=int(anchor_parity), strong_beat_mask=strong_mask,
        strong_beat_times=strong_beat_times, backbeat_times=backbeat_times,
        strong_beat_confidence=float(sb_conf), anchor=anchor,
        octave_locked=bool(octave_locked), prior_period=float(prior_period),
        phase_offset_s=float(phase_offset))


# ═════════════════════════════════════════════════════════════════════════════
# Convenience: end-to-end from audio (used by the real-audio smoke test)
# ═════════════════════════════════════════════════════════════════════════════

def _beat_this_prior(wav_path: str) -> Optional[np.ndarray]:
    """Independent Beat This! beat times, or None if the backend is unavailable."""
    try:
        from beat_this.inference import load_audio
        from harmonia.models import downbeat_anchor as da
        a2f, pp = da._get_beat_this()
        signal, sr = load_audio(wav_path)
        beat_logits, downbeat_logits = a2f(signal, sr)
        beats, _ = pp(beat_logits, downbeat_logits)
        return np.asarray(beats, dtype=float)
    except Exception as e:  # pragma: no cover - backend/optional
        log.warning("Beat This! prior unavailable (%s); using drum-only phase", e)
        return None


def track_from_audio(source: Union[str, np.ndarray], sr: int = SR,
                     prior_period: Optional[float] = None,
                     prior_bpm: Optional[float] = None,
                     beat_this_beats: Optional[np.ndarray] = None,
                     run_beat_this: bool = True) -> DrumBeatTrack:
    """End-to-end: audio -> percussive onset envelopes -> drum beat track.

    ``source`` is a wav path or a mono signal array. The tempo-octave prior is
    resolved as: explicit ``prior_period`` / ``prior_bpm`` > ``beat_this_beats``
    median > (if ``run_beat_this``) a live Beat This! pass > a self octave from
    the drum autocorrelation as a last resort.
    """
    import librosa

    if isinstance(source, str):
        y, _sr = librosa.load(source, sr=SR, mono=True)
        wav_path: Optional[str] = source
    else:
        y = np.asarray(source, float)
        wav_path = None
    env = percussive_onset_envelopes(y, sr=SR, hop=HOP)

    if beat_this_beats is None and wav_path is not None and run_beat_this \
            and prior_period is None and prior_bpm is None:
        beat_this_beats = _beat_this_prior(wav_path)

    if prior_period is None:
        if prior_bpm is not None:
            prior_period = 60.0 / prior_bpm
        elif beat_this_beats is not None and len(beat_this_beats) > 1:
            prior_period = float(np.median(np.diff(np.asarray(beat_this_beats, float))))
        else:
            # last resort: self octave from the global autocorrelation peak in a
            # plausible beat band (0.25-1.0 s => 60-240 BPM)
            prior_period = estimate_period_in_octave(
                env.broad, 0.5, SR, HOP, octave_tol=1.0)

    return track_drum_beats(env.broad, prior_period, low=env.low, high=env.high,
                            beat_this_beats=beat_this_beats, sr=SR, hop=HOP)
