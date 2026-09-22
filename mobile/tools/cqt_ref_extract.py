"""P2b — dump the EXACT tensor musx's ChordNet receives, for 3 reference songs.

Calls the real vendored clone (via harmonia.musx, which chdirs into
third_party/musx_ismir2019 exactly like the live pipeline) — no numpy port
involved here, this is the ground truth to port against.

Saves, per song, mobile/tools/cqt_ref/<stem>.npz with:
  y            — float32, the mono 22050 Hz signal MusicIO.read() produced
  cqt          — float32 (n_frames, 288), abs(hybrid_cqt).T — CQTV2's own output
  net_input    — float32 (n_frames, 252), cqt[:, 18:270] — what ChordNet.inference
                 actually slices and feeds to the CNN (see chordnet_ismir_naive.py:180-181)
  tuning       — scalar, librosa.estimate_tuning(y, sr=22050, bins_per_octave=36)
                 (hybrid_cqt is called with tuning=None in cqt.py:CQTV2, so this
                 IS exercised — not the librosa default tuning=0.0)
  sr, hop      — 22050, 512 (MUSX_SR, MUSX_HOP)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from harmonia import musx  # noqa: E402

AUDIO_DIR = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia/docs/audio")
SONGS = {
    "maroon_5_this_love": AUDIO_DIR / "maroon_5_this_love.m4a",
    "ben_e_king_stand_by_me_audio": AUDIO_DIR / "ben_e_king_stand_by_me_audio.m4a",
    "autumn_leaves": AUDIO_DIR / "autumn_leaves.m4a",
}

OUT_DIR = Path(__file__).resolve().parent / "cqt_ref"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SHIFT_HIGH = 6
SHIFT_STEP = 3
SPEC_DIM = 252
SLICE_LO = SHIFT_HIGH * SHIFT_STEP  # 18
SLICE_HI = SLICE_LO + SPEC_DIM  # 270


def main() -> None:
    for stem, path in SONGS.items():
        if not path.exists():
            print(f"MISSING {path}")
            continue
        t0 = time.time()
        with musx._InMusxDir():
            import librosa
            from mir import io, DataEntry
            from extractors.cqt import CQTV2

            entry = DataEntry()
            entry.prop.set('sr', musx.MUSX_SR)
            entry.prop.set('hop_length', musx.MUSX_HOP)
            entry.append_file(str(path), io.MusicIO, 'music')
            y = np.asarray(entry.music, dtype=np.float32)
            entry.append_extractor(CQTV2, 'cqt')
            cqt = np.asarray(entry.cqt, dtype=np.float32)
            tuning = float(
                librosa.estimate_tuning(y=y, sr=musx.MUSX_SR, bins_per_octave=36)
            )
        dt = time.time() - t0
        net_input = cqt[:, SLICE_LO:SLICE_HI]
        out = OUT_DIR / f"{stem}.npz"
        np.savez_compressed(
            out,
            y=y,
            cqt=cqt,
            net_input=net_input,
            tuning=np.float32(tuning),
            sr=np.int64(musx.MUSX_SR),
            hop=np.int64(musx.MUSX_HOP),
        )
        print(
            f"{stem}: y={y.shape} dur={len(y)/musx.MUSX_SR:.1f}s "
            f"cqt={cqt.shape} net_input={net_input.shape} tuning={tuning:.5f} "
            f"extract_time={dt:.1f}s -> {out}"
        )


if __name__ == "__main__":
    main()
