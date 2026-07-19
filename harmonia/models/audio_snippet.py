"""Exact-[t0,t1) audio-clip extraction, reusing the bleed-fixed convention.

This is the same frame-clipped (NOT beat-grid-snapped) audio slicing established
as correct in the 2026-07-16 boundary-bleed work and used by
``docs/error_report_wrong_root_2026_07_16/fetch_clips.py`` and
``scratchpad/bleed_verify.py``: ffmpeg trim to exact sample boundaries with ZERO
padding, so the returned clip's duration matches ``t1 - t0`` to sub-millisecond
precision (ffprobe-verifiable, same standard as ``docs/bleed_verification_2026_07_16``).

Powers the Annotate-tab "listen to the real audio of this chord" button so a
human correcting a chord hears the exact span the model pooled, not a bar-snapped
approximation.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


def extract_snippet_wav(audio_path: str | Path, t0: float, t1: float) -> bytes:
    """Return the WAV bytes for the exact ``[t0, t1)`` span of ``audio_path``.

    Uses ffmpeg input-seek + transcode to 16-bit PCM (``-accurate_seek`` is on by
    default when transcoding, so the seek is sample-accurate even for AAC/.m4a
    sources). ZERO padding on either side — the clip is exactly ``t1 - t0`` long,
    matching the bleed-fixed frame-clip pooling convention.

    Raises ``ValueError`` for a non-positive / out-of-order span and
    ``RuntimeError`` if ffmpeg fails or the source file is missing.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise RuntimeError(f"audio not found: {audio_path}")
    t0 = max(0.0, float(t0))
    t1 = float(t1)
    dur = t1 - t0
    if dur <= 0:
        raise ValueError(f"non-positive span [{t0}, {t1})")

    # Write to a real (seekable) temp file rather than a pipe: ffmpeg can then
    # backfill the RIFF/data size fields in the WAV header. A piped WAV leaves
    # those as placeholders, which Safari on iOS (the target device) refuses to
    # play. The file is deleted before we return — nothing persists on disk.
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        cmd = [
            "ffmpeg", "-nostdin", "-y", "-loglevel", "error",
            "-ss", f"{t0:.6f}", "-i", str(audio_path),
            "-t", f"{dur:.6f}",
            "-ac", "1",            # mono — a chord snippet needs no stereo image
            "-c:a", "pcm_s16le", "-f", "wav",
            tmp.name,
        ]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0:
            tail = proc.stderr.decode("utf-8", "replace")[-400:]
            raise RuntimeError(f"ffmpeg snippet extraction failed: {tail}")
        data = Path(tmp.name).read_bytes()
    if not data:
        raise RuntimeError("ffmpeg produced an empty snippet")
    return data
