"""Compare librosa vs madmom beat-tracker tempo across the docs/audio corpus.

Load-bearing check for docs/known_issues.md #1 (tempo octave-lock): does
madmom's RNN+DBN tracker avoid the 2x octave the librosa DP tracker locks onto
on ballads/swing?  We measure both engines on every song, and compare to a
reference BPM where one is known (backing-track filename, or hand annotation).

Outputs JSON to docs/tempo_comparison_madmom.json for the plot + report.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# The editable install maps `harmonia` → the STALE clone at ~/harmonia (see
# CLAUDE.md). When run as a file, scripts/ is sys.path[0] and cwd is not on the
# path, so imports resolve to the stale clone and miss edits in THIS repo.
# Force the canonical repo root ahead of the editable finder.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_WAV_CACHE = Path(tempfile.gettempdir()) / "harmonia_reinfer_wav"


def _to_wav(path: Path) -> Path:
    """Match infer_chords_v1's input: a 44.1k mono wav read via soundfile."""
    _WAV_CACHE.mkdir(parents=True, exist_ok=True)
    out = _WAV_CACHE / (path.stem + ".wav")
    if not out.exists():
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path),
                        "-ac", "1", "-ar", "44100", str(out)], check=True)
    return out

import librosa
import numpy as np
import soundfile as sf

from harmonia.models.rhythm import RhythmAnalyser

AUDIO_DIR = Path("docs/audio")
OUT_JSON = Path("docs/tempo_comparison_madmom.json")

# Reference tempos.  Only include values we can actually justify — a filename
# that states BPM, or a well-known tempo for the standard.  `None` = unknown.
# Sources noted so the report can cite the trust level (rule #3: GT is a
# measurement too).
REFERENCE_BPM: dict[str, dict] = {
    "blue_bossa_150bpm_backing_track": {"bpm": 150.0, "src": "filename (backing track)"},
    # Common performance tempos for the standards (approximate, for octave
    # diagnosis only — a factor-of-2 error is unambiguous even against a loose
    # reference).  Marked low-trust.
    "ghost_of_a_chance":  {"bpm": 60.0,  "src": "ballad ~55-65 (approx)"},
    "autumn_leaves":      {"bpm": 120.0, "src": "medium swing ~120 (approx)"},
    "blue_bossa":         {"bpm": 150.0, "src": "bossa ~150 (approx)"},
    "a_foggy_day":        {"bpm": 150.0, "src": "medium swing (approx)"},
    "airegin":            {"bpm": 220.0, "src": "up-tempo bebop (approx)"},
}


def librosa_tempo(wav: Path) -> float:
    # EXACT production path: infer_chords_v1 does sf.read(wav) at native sr,
    # then librosa.beat.beat_track(y, sr). librosa's octave choice is sr/hop
    # sensitive, so we must match production to compare fairly.
    y, sr = sf.read(wav)
    y = (y.mean(1) if y.ndim > 1 else y).astype("float32")
    tempo_arr, _ = librosa.beat.beat_track(y=y, sr=sr)
    return float(np.atleast_1d(tempo_arr)[0])


def octave_ratio(bpm: float, ref: float) -> float:
    """How far bpm is from ref in octaves (log2). ~+1 = doubled, ~-1 = halved."""
    return float(np.log2(bpm / ref)) if (bpm > 0 and ref > 0) else float("nan")


def main() -> None:
    analyser = RhythmAnalyser(prefer_madmom=True)
    avail = analyser._check_madmom()
    print("madmom available:", avail)
    if not avail:
        raise SystemExit("madmom not available — aborting (would silently "
                         "produce a librosa-only comparison).")

    rows = []
    for path in sorted(AUDIO_DIR.glob("*.m4a")):
        name = path.stem
        wav = _to_wav(path)
        t0 = time.time()
        lib = librosa_tempo(wav)
        t_lib = time.time() - t0

        t0 = time.time()
        grid = analyser.analyse(str(wav))
        t_mad = time.time() - t0
        mad = float(grid.tempo_bpm)

        ref = REFERENCE_BPM.get(name)
        ref_bpm = ref["bpm"] if ref else None

        row = {
            "song": name,
            "librosa_bpm": round(lib, 1),
            "madmom_bpm": round(mad, 1),
            "madmom_backend": grid.backend,
            "madmom_nbeats": int(grid.n_beats),
            "madmom_ndownbeats": int(len(grid.downbeat_times)),
            "ref_bpm": ref_bpm,
            "ref_src": ref["src"] if ref else None,
            "librosa_oct_vs_ref": round(octave_ratio(lib, ref_bpm), 3) if ref_bpm else None,
            "madmom_oct_vs_ref": round(octave_ratio(mad, ref_bpm), 3) if ref_bpm else None,
            "librosa_vs_madmom_oct": round(octave_ratio(lib, mad), 3),
            "t_librosa_s": round(t_lib, 1),
            "t_madmom_s": round(t_mad, 1),
        }
        rows.append(row)
        print(f"{name:52s} lib={lib:6.1f}  mad={mad:6.1f}  "
              f"ref={ref_bpm}  lib/mad_oct={row['librosa_vs_madmom_oct']:+.2f}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {OUT_JSON}  ({len(rows)} songs)")


if __name__ == "__main__":
    main()
