"""harmonia_min/beats.py — Beat This! beats + downbeats. The ONLY beat tracker.

Re-extracted 2026-07-30 from chord_pipeline_v1._get_beatthis (~15 lines) +
harmonia/serving/audio.py::_beats_and_downbeats (the m4a lesson), for the
minimal rebuild. Two hard-won rules are load-bearing here:

1. **librosa is banned.** It locks 2x tempo octaves (validated 2026-07-21:
   tempo-octave 65% vs Beat This! 78% on POP909; song 002 doubled to ~129 BPM
   against three agreeing GT annotations at ~64). The old code silently fell
   back to librosa whenever Beat This! threw — which was EVERY m4a on this box
   (see 2), so the fallback was the code path. Here a failure raises.

2. **Beat This! must get a wav.** It reads audio via torchaudio / soundfile /
   madmom and on this box all three refuse .m4a (no torchcodec; libsndfile has
   no AAC; madmom is py3.12-broken). Discovered 2026-07-30 after the librosa
   fallback had silently served every song. So: try the file as-is, and on any
   decode error transcode to a temp wav with ffmpeg and retry. Only if THAT
   fails do we raise.

Results are disk-cached to harmonia_min/state/beats/<stem>.json (stem-keyed,
same rationale as the musx cache: fresh downloads of the same video get new
mtimes, the stem is the stable key).
"""
from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent / "state" / "beats"

_f2b = None  # loaded once per process (~2 s model load)


def _get_beatthis():
    global _f2b
    if _f2b is None:
        from beat_this.inference import File2Beats
        _f2b = File2Beats(device="cpu", dbn=False)
        logger.info("beats: loaded Beat This! (MIT) beat+downbeat tracker")
    return _f2b


class BeatTrackingError(RuntimeError):
    """Beat This! could not produce a usable beat grid. No fallback exists —
    librosa is banned (2x tempo-octave lock), so the caller must surface this."""


def track(audio_path: str | Path, *, use_cache: bool = True) -> dict:
    """Beats + downbeats for one audio file.

    Returns {"beats": [s...], "downbeats": [s...], "bpm": float}.
    Raises BeatTrackingError instead of ever falling back to another tracker.
    """
    audio_path = Path(audio_path)
    cache = CACHE_DIR / f"{audio_path.stem}.json"
    if use_cache and cache.exists():
        try:
            d = json.loads(cache.read_text(encoding="utf-8"))
            if len(d.get("beats", [])) >= 4:
                return d
        except ValueError:
            pass

    f2b = _get_beatthis()
    try:
        bts, dbs = f2b(str(audio_path))
    except Exception:
        # Almost always "cannot decode m4a" — transcode and retry (lesson 2).
        with tempfile.TemporaryDirectory() as td:
            wav = f"{td}/a.wav"
            try:
                subprocess.run(
                    ["ffmpeg", "-v", "error", "-y", "-i", str(audio_path),
                     "-ac", "1", "-ar", "22050", wav],
                    check=True, timeout=300)
                bts, dbs = f2b(wav)
            except Exception as exc:
                raise BeatTrackingError(
                    f"Beat This! failed on {audio_path.name} even after ffmpeg "
                    f"transcode ({exc}). No librosa fallback — fix the input."
                ) from exc

    beats = [round(float(t), 4) for t in bts]
    downbeats = [round(float(t), 4) for t in dbs]
    if len(beats) < 4:
        raise BeatTrackingError(
            f"Beat This! returned only {len(beats)} beats for {audio_path.name}")
    import numpy as np
    bpm = round(60.0 / float(np.median(np.diff(beats))), 2)
    out = {"beats": beats, "downbeats": downbeats, "bpm": bpm}
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(out), encoding="utf-8")
    return out
