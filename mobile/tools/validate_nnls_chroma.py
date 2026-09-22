"""P0-T2 parity check: `harmonia_mobile.nnls_chroma` (numpy) vs the real
VAMP plugin references in `mobile/tools/nnls_ref/*.npz` (made by
`gen_nnls_ref.py` on the Mac, using the actual `vamp.collect` call).

Per song, per frame: max abs diff, per-frame Pearson correlation, and the
metric that actually matters musically -- bass-half argmax agreement, since
that's what `harmonia/nnls_features.py`'s `bass_pc_onset` /
`_write_sounding_bass` reads off this feature. Also times a 240s (4 min)
excerpt to answer the T5-adjacent "is this remotely fast enough" question
(desktop only here; the real phone-speed gate is T5).

Run with the worktree's `.venv` (needs librosa to decode the reference
audio; the numpy port itself needs only numpy):

    .venv/bin/python mobile/tools/validate_nnls_chroma.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # mobile/
from harmonia_mobile.nnls_chroma import NNLSChromaNumpy  # noqa: E402

REF_DIR = Path(__file__).resolve().parent / "nnls_ref"
AUDIO_DIR = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia/docs/audio")
SR = 44100

SONGS = {
    "maroon_5_this_love": AUDIO_DIR / "maroon_5_this_love.m4a",
    "ben_e_king_stand_by_me_audio": AUDIO_DIR / "ben_e_king_stand_by_me_audio.m4a",
    "autumn_leaves": AUDIO_DIR / "autumn_leaves.m4a",
}


def compare(a: np.ndarray, r: np.ndarray):
    T = min(len(a), len(r))
    a, r = a[:T].astype(np.float64), r[:T].astype(np.float64)
    diff = np.abs(a - r)
    corrs = np.array([
        np.corrcoef(a[i], r[i])[0, 1] if a[i].std() > 1e-9 and r[i].std() > 1e-9 else np.nan
        for i in range(T)
    ])
    bass_match = (a[:, :12].argmax(1) == r[:, :12].argmax(1)).mean()
    return dict(T=T, max_abs_diff=float(diff.max()), mean_abs_diff=float(diff.mean()),
                mean_corr=float(np.nanmean(corrs)), bass_argmax_match=float(bass_match))


def main() -> None:
    import librosa  # only needed here, to decode the same audio the ref was made from

    eng = NNLSChromaNumpy()
    for name, path in SONGS.items():
        ref = np.load(REF_DIR / f"{name}.npz")
        y, _ = librosa.load(str(path), sr=SR, mono=True)
        t0 = time.time()
        arr, times = eng.extract_bothchroma(y, SR)
        t1 = time.time()
        stats = compare(arr, ref["arr"])
        print(f"{name}: dur={len(y)/SR:.1f}s process={t1-t0:.1f}s "
              f"({(len(y)/SR)/(t1-t0):.1f}x realtime) -- {stats}")

    # dedicated 4-minute timing (separate from the parity numbers above)
    path4 = AUDIO_DIR / "autumn_leaves.m4a"
    y4, _ = librosa.load(str(path4), sr=SR, mono=True, duration=240.0)
    t0 = time.time()
    eng.extract_bothchroma(y4, SR)
    t1 = time.time()
    print(f"4-min timing: {t1-t0:.2f}s for {len(y4)/SR:.1f}s of audio "
          f"-> {(len(y4)/SR)/(t1-t0):.2f}x realtime (desktop numpy, not phone)")


if __name__ == "__main__":
    main()
