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

from harmonia.serving.config import (
    AUDIO_DIR, BAR_REF_CACHE, BEATGRID_CACHE, WAVEFORM_CACHE, _BEAT_TIMES_CACHE,
)

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

    **2026-07-30: the 2026-07-21 fix did not hold, and never had.** Beat This!
    reads audio through torchaudio / soundfile / madmom, and on this box all
    three refuse ``.m4a`` (no torchcodec; libsndfile has no AAC; madmom is
    py3.12-broken). So ``_get_beatthis()(m4a)`` threw on EVERY song, the
    ``except`` swallowed it, and the librosa branch ran every time — the exact
    two-different-clocks bug described above, reintroduced by an unrelated
    environment gap and hidden by a warning nobody read. All 48 ``v2`` cache
    entries were librosa's beats. Now: transcode with ffmpeg first (via
    ``_beats_and_downbeats``), no librosa fallback at all, cache bumped to v3.

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
    # `_beats_and_downbeats` transcodes to wav with ffmpeg before calling Beat
    # This!, which is the whole point: calling `_get_beatthis()` directly on an
    # .m4a throws on this box, and the old code caught that and fell through to
    # librosa — for EVERY song, silently (2026-07-30, see the note above).
    bd = _beats_and_downbeats(audio_path)
    beat_times_raw = [float(t) for t in bd[0]] if bd else None
    if beat_times_raw is not None and len(beat_times_raw) < 4:
        beat_times_raw = None
    if beat_times_raw is None:
        # No librosa fallback. Beat This! is the official tracker (Louis,
        # 2026-07-30) precisely because librosa locks 2x tempo octaves, and
        # snapping the playhead onto an UNRELATED tracker's beats is worse than
        # not snapping at all — that is the two-different-clocks bug this
        # docstring already claimed to have fixed once. Returning None disables
        # the opt-in snap, which is the honest degradation.
        log.warning("raw-beat-times: Beat This! unavailable for %s — snap "
                    "disabled (NOT falling back to librosa)", slug)
        return None
    times = [round(float(t), 4) for t in beat_times_raw]
    try:
        cache.write_text(json.dumps(times), encoding="utf-8")
    except OSError:
        pass
    return times


def _beats_and_downbeats(audio_path) -> "tuple[list, list] | None":
    """Beat This! beats AND native downbeats for one audio file, or ``None``.

    Beat This! reads audio through torchaudio / soundfile / madmom, and in this
    environment ALL THREE refuse ``.m4a`` (no torchcodec; libsndfile has no AAC;
    madmom is py3.12-broken).  So the m4a is transcoded to a temporary wav with
    ffmpeg first.  Discovered 2026-07-30 while building the octave cue — the
    same blindness makes ``_raw_beat_times_cached`` above fall through to its
    librosa branch for every m4a on this box, i.e. the display beat-snap has
    been running on librosa's beats, not the chart's own backend.  That is the
    exact "two different clocks" bug its docstring says was fixed in 2026-07-21;
    logged in docs/known_issues.md rather than fixed here (different surface).
    """
    import subprocess
    import tempfile

    from harmonia.models.chord_pipeline_v1 import _get_beatthis

    f2b = _get_beatthis()
    if f2b is None:
        return None
    try:
        bts, dbs = f2b(str(audio_path))
        return list(bts), list(dbs)
    except Exception:  # noqa: BLE001 — almost always "cannot decode m4a"
        pass
    with tempfile.TemporaryDirectory() as td:
        wav = f"{td}/a.wav"
        try:
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(audio_path),
                            "-ac", "1", "-ar", "22050", wav],
                           check=True, timeout=300)
            bts, dbs = f2b(wav)
            return list(bts), list(dbs)
        except Exception as exc:  # noqa: BLE001
            log.warning("beat_this decode failed for %s (%s)", audio_path, exc)
            return None


def bar_ref_for_slug(slug: str) -> "float | None":
    """Bar length in seconds for <slug> from Beat This!'s NATIVE downbeats — the
    external accent cue that breaks the rigid grid's 2x metrical octave.

    ``None`` means "no opinion": no audio, the tracker failed, or its downbeats
    did not pass ``bar_len_from_downbeats``'s self-consistency gate.  Every
    consumer treats ``None`` as "keep the onsets-only answer", so an abstention
    is always a no-op rather than a guess.  Disk-cached (including the ``None``),
    since it costs a full beat-tracking pass.
    """
    audio_path = AUDIO_DIR / f"{slug}.m4a"
    if not slug or not audio_path.exists():
        return None
    BAR_REF_CACHE.mkdir(parents=True, exist_ok=True)
    cache = BAR_REF_CACHE / f"{slug}.json"
    if cache.exists():
        try:
            return json.loads(cache.read_text(encoding="utf-8")).get("bar_sec")
        except ValueError:
            pass
    bar = None
    got = _beats_and_downbeats(audio_path)
    if got is not None:
        from harmonia.models.rigid_grid import bar_len_from_downbeats
        bts, dbs = got
        bar = bar_len_from_downbeats(dbs, bts)
        if bar is not None:
            bar = round(float(bar), 4)
    try:
        cache.write_text(json.dumps({"bar_sec": bar, "source": "beat_this"}),
                         encoding="utf-8")
    except OSError:
        pass
    return bar
