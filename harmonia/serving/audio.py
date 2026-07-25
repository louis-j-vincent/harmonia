"""harmonia/serving/audio.py — audio-envelope / beat-grid compute-and-cache
leaves for the serving layer (Phase 6c architectural pass).

Home of the three server-side audio helpers that decode a song's cached m4a and
memoise a small JSON artifact to disk:

  * ``_waveform_peaks``       — normalised RMS amplitude envelope (annotator ruler)
  * ``_beat_grid_for``        — beat/downbeat grid for the snap ruler
  * ``_raw_beat_times_cached``— raw detected beat times (boundary-snap consumer)

MOVED VERBATIM here — ``_waveform_peaks`` / ``_beat_grid_for`` out of
``harmonia.serving.state`` (where Phase 6c batch-3 had temporarily parked them),
``_raw_beat_times_cached`` out of ``scripts/harmonia_server.py`` (its last
server-owned dependency of ``harmonia.serving.render._chart_model_for``, whose
removal from the server lets the render↔server lazy back-import be deleted). All
three are PURE LEAVES: config paths + a disk cache + lazy in-body
numpy/librosa/chord_pipeline_v1 only — no server-module state and no other server
helper — so importing this module is cheap and creates no cycle (api/render/the
server import these names; nothing here imports api or the server).
"""

from __future__ import annotations

import json
import logging

from harmonia.serving.config import AUDIO_DIR, BEATGRID_CACHE, WAVEFORM_CACHE, _BEAT_TIMES_CACHE

log = logging.getLogger(__name__)


def _waveform_peaks(slug: str, n_cols: int = 1800) -> dict | None:
    """Server-side waveform peaks for <slug> as a normalised RMS array.

    Decoding the audio in the browser (Web Audio `decodeAudioData` on an m4a/
    AAC blob) is the load-bearing fragility of the v1/v2 annotators on iPhone:
    iOS Safari decodes AAC unreliably and a multi-minute buffer + a >16k-px
    canvas can silently paint nothing. Decoding once here with librosa and
    shipping ~1.8k floats sidesteps all of that — the client just draws bars.
    Cached to disk keyed by slug (+n_cols); librosa load of an m4a is ~1–2 s.
    """
    audio_path = AUDIO_DIR / f"{slug}.m4a"
    if not audio_path.exists():
        return None
    WAVEFORM_CACHE.mkdir(parents=True, exist_ok=True)
    cache = WAVEFORM_CACHE / f"{slug}.{n_cols}.json"
    if cache.exists():
        try:
            d = json.loads(cache.read_text(encoding="utf-8"))
            if d.get("peaks"):
                return d
        except ValueError:
            pass
    try:
        import numpy as np
        import librosa
        # 8 kHz mono is plenty for an amplitude envelope and keeps decode fast.
        y, sr = librosa.load(str(audio_path), sr=8000, mono=True)
        dur = float(len(y) / sr) if sr else 0.0
        if len(y) == 0:
            return None
        edges = np.linspace(0, len(y), n_cols + 1).astype(int)
        peaks = np.empty(n_cols, dtype=np.float64)
        for i in range(n_cols):
            seg = y[edges[i]:edges[i + 1]]
            peaks[i] = np.sqrt(np.mean(seg * seg)) if seg.size else 0.0
        pmax = float(peaks.max()) or 1e-6
        # gentle gamma so quiet intros/outros stay visible
        peaks = np.power(np.clip(peaks / pmax, 0.0, 1.0), 0.7)
        result = {"peaks": [round(float(p), 3) for p in peaks],
                  "duration": round(dur, 3), "n": n_cols}
    except Exception as e:
        log.warning("waveform peak extraction failed for %s (%s)", slug, e)
        return None
    try:
        cache.write_text(json.dumps(result), encoding="utf-8")
    except OSError:
        pass
    return result


def _beat_grid_for(slug: str, audio_path, tempo: float, duration: float) -> dict:
    """Beat/downbeat grid for the snap ruler. Prefers Mission-1's
    extract_beat_grid() on the real audio (cached to disk — librosa beat
    tracking is a few seconds); falls back to a uniform grid from the chart
    tempo if audio/librosa is unavailable."""
    BEATGRID_CACHE.mkdir(parents=True, exist_ok=True)
    cache = BEATGRID_CACHE / f"{slug}.json"
    if cache.exists():
        try:
            d = json.loads(cache.read_text(encoding="utf-8"))
            if d.get("beats"):
                return d
        except ValueError:
            pass
    result = None
    if audio_path is not None and audio_path.exists():
        try:
            from mission_1_build_benchmark import extract_beat_grid  # scripts/ on sys.path
            bg = extract_beat_grid(audio_path, bpm_hint=tempo)
            result = {
                "beats": [round(float(x), 4) for x in list(bg.beat_times)],
                "downbeats": [round(float(x), 4) for x in list(bg.downbeat_times)],
                "bpm": float(bg.bpm), "source": "extract_beat_grid",
            }
        except Exception as e:  # librosa/soundfile/pyRealParser missing, decode error…
            log.warning("extract_beat_grid failed for %s (%s) — uniform fallback", slug, e)
    if result is None:
        step = 60.0 / (tempo or 120.0)
        n = int((duration or 0.0) / step) + 4
        beats = [round(i * step, 4) for i in range(n)]
        result = {"beats": beats, "downbeats": beats[::4],
                  "bpm": float(tempo or 120.0), "source": "uniform"}
    try:
        cache.write_text(json.dumps(result), encoding="utf-8")
    except OSError:
        pass
    return result


def _raw_beat_times_cached(slug: str) -> list | None:
    """Real detected beat times for <slug>, disk-cached — SAME backend
    (Beat This!, falling back to librosa) as the production decode's default
    ``beat_backend="beatthis"`` (chord_pipeline_v1.infer_chords_v1).

    2026-07-21 bug found and fixed: this function used to hard-code
    ``librosa.beat.beat_track`` regardless of which backend actually built
    the chart's bar-grid — a genuine two-different-clocks bug (CLAUDE.md
    rule #6, "component swaps change more than the target metric"), not
    just a display nicety. The DISPLAY SNAP (below) is supposed to correct
    the playhead onto the real beat the chart's own grid is built from; with
    a mismatched backend it was instead correcting onto an UNRELATED
    tracker's beats, which can disagree by more than the wobble it was
    trying to fix (librosa is validated worse on tempo-octave: 65% vs Beat
    This!'s 78%, see docs/known_issues.md). User report, confirmed live on
    "Happy": a bar between two "real" (old, librosa) beats was only 3 beats
    long where the surrounding tempo says 4 — a genuine missed detection.
    Cache moved to a new directory (``raw_beat_times_v2``) rather than
    invalidating the old one in place, so a stale entry can never silently
    survive this fix by matching on slug alone.

    Production consumer for the boundary-placement fix (2026-07-20,
    docs/research_sessions/boundary_snap_2026-07-20.md): a folded A×N
    section's reconstructed onsets phase-wobble vs the real beat (corpus
    mean 84ms wrapped-std) because one representative phrase is offset onto
    every repeat's span, and repeats aren't identically timed (intra-phrase
    rubato). Snapping each reconstructed onset to the nearest of THESE
    beats (app_shell.html's loadModel, gated on ``m.beatTimes``) measured
    84->27ms (-68%) on the matched set. Opt-in (HARMONIA_BOUNDARY_SNAP=1,
    default off — zero cost/risk when unset, matches every other flag in
    this file)."""
    audio_path = AUDIO_DIR / f"{slug}.m4a"
    if not audio_path.exists():
        return None
    _BEAT_TIMES_CACHE.mkdir(parents=True, exist_ok=True)
    cache = _BEAT_TIMES_CACHE / f"{slug}.json"
    if cache.exists():
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except ValueError:
            pass
    try:
        import librosa
        beat_times_raw = None
        try:
            from harmonia.models.chord_pipeline_v1 import _get_beatthis
            _f2b = _get_beatthis()
            if _f2b is not None:
                _bts, _dbs = _f2b(str(audio_path))
                _bts = [float(t) for t in _bts]
                if len(_bts) >= 4:
                    beat_times_raw = _bts
        except Exception as exc:  # noqa: BLE001 — never break the snap over an opt-in
            log.warning("raw-beat-times beatthis backend failed for %s (%s); librosa fallback", slug, exc)
        if beat_times_raw is None:
            import librosa.beat
            y, sr = librosa.load(str(audio_path), mono=True, sr=None)
            _tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
            beat_times_raw = librosa.frames_to_time(beat_frames, sr=sr)
        times = [round(float(t), 4) for t in beat_times_raw]
    except Exception as e:
        log.warning("raw-beat-times extraction failed for %s (%s)", slug, e)
        return None
    try:
        cache.write_text(json.dumps(times), encoding="utf-8")
    except OSError:
        pass
    return times
