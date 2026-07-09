"""Two cheap checks:
  (1) v2 root error-interval distribution — is 52% of error really +5/+7 (P4/P5)?
  (2) v2 accuracy on the POP909 GT beat grid (beat_midi.txt) instead of the
      librosa tempo grid — does that reproduce the task's interior=93.6% / beat-1=62.2%?
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from train_beat_seq_model_v3 import _tempo_grid_beats, HARTE_TO_PC
from harmonia.models.chord_pipeline_v1 import _BeatSeqModel, _pool_beats, MODELS
from harmonia.models.stage1_pitch import PitchExtractor
from harmonia.data.pop909_parser import POP909Parser

POP = REPO / "data" / "pop909" / "POP909"
SONGS = ("001", "002", "003", "004", "005")


def run(grid: str):
    """grid = 'librosa' (tempo grid) or 'gt' (POP909 beat_midi grid)."""
    parser = POP909Parser(POP)
    ex = PitchExtractor(cache_dir=REPO / "data" / "cache")
    v2 = _BeatSeqModel(MODELS / "beat_seq_model_v2.npz")

    GT = []; PR = []; POSg = []; BND = []
    for sid in SONGS:
        wav = REPO / "data" / "renders" / "pop909" / sid / f"{sid}_v005_musescoregeneral.wav"
        if not wav.exists():
            continue
        y, sr = sf.read(wav)
        y = (y.mean(1) if y.ndim > 1 else y).astype("float32")
        song = parser.parse_song(sid)
        if grid == "librosa":
            bt, _ = _tempo_grid_beats(y, sr)
            downpos = None
        else:  # gt grid
            gb = song.beat_times
            bt = np.append(gb, gb[-1] + (gb[-1] - gb[-2]))
            downpos = song.is_downbeat
        acts = ex.extract(wav)
        onset_b = _pool_beats(acts.frame_times, acts.onset_probs, bt)
        note_b = _pool_beats(acts.frame_times, acts.note_probs, bt)
        pred = v2.predict_proba(onset_b, note_b).argmax(1)

        n = min(len(bt) - 1, len(pred))
        prev = None; db_since = 0
        for b in range(n):
            ev = song.chord_at_time(0.5 * (bt[b] + bt[b + 1]))
            if grid == "gt" and b < len(downpos) and downpos[b]:
                db_since = 0
            elif grid == "gt":
                db_since += 1
            if ev is None:
                prev = None
                continue
            rs = ev.label.split(":")[0].split("/")[0]
            if rs not in HARTE_TO_PC:
                prev = None
                continue
            gt = HARTE_TO_PC[rs]
            GT.append(gt); PR.append(int(pred[b]))
            POSg.append(db_since % 4 if grid == "gt" else b % 4)
            BND.append(prev is not None and gt != prev)
            prev = gt
    GT = np.array(GT); PR = np.array(PR); POSg = np.array(POSg); BND = np.array(BND)
    ok = GT == PR
    print(f"\n=== grid={grid}  (n={len(GT)}) ===")
    print(f"  overall  {ok.mean():.1%}   interior {ok[~BND].mean():.1%}   boundary {ok[BND].mean():.1%}")
    for p in range(4):
        m = POSg == p
        print(f"  pos {p}: {ok[m].mean():.1%} (n={m.sum()})", end="")
    print()
    # error-interval distribution
    err = ~ok
    iv = (PR[err] - GT[err]) % 12
    print("  error-interval dist (semitones up from GT):")
    counts = np.bincount(iv, minlength=12)
    for s in np.argsort(counts)[::-1][:5]:
        print(f"    +{s:2d}: {counts[s]/err.sum():.1%}", end="")
    print()
    p57 = (counts[5] + counts[7]) / err.sum()
    print(f"  +5/+7 (P4/P5) share of errors: {p57:.1%}")


if __name__ == "__main__":
    run("librosa")
    run("gt")
