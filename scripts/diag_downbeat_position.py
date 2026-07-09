"""Test architecture-A's premise properly: is root accuracy worst on the DOWNBEAT?

The task claims beat-1 (downbeat) = 62.2% (worst), beats 2&4 = 86%. My librosa-grid
`b%4` check reversed that. Confound: the librosa tempo grid's phase is arbitrary.
POP909 ships GROUND-TRUTH downbeats (beat_midi.txt col 3, known_issues #7). So here
we map each librosa grid beat to the nearest GT beat and read its real bar position
(beats since last GT downbeat, mod 4), then break accuracy down by TRUE bar position.
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

from train_beat_seq_model_v3 import V3Model, _tempo_grid_beats, HARTE_TO_PC
from harmonia.models.chord_pipeline_v1 import _BeatSeqModel, _pool_beats, MODELS
from harmonia.models.stage1_pitch import PitchExtractor
from harmonia.data.pop909_parser import POP909Parser

POP = REPO / "data" / "pop909" / "POP909"


def gt_bar_position(beat_center: float, gt_beats: np.ndarray, gt_down: np.ndarray) -> int:
    """Beats since the last GT downbeat at/before beat_center, mod 4."""
    j = int(np.searchsorted(gt_beats, beat_center, side="right") - 1)
    j = max(0, min(j, len(gt_beats) - 1))
    # walk back to nearest downbeat index
    db_idx = np.where(gt_down[: j + 1])[0]
    if len(db_idx) == 0:
        return -1
    return (j - db_idx[-1]) % 4


def collect(songs=("001", "002", "003", "004", "005")):
    parser = POP909Parser(POP)
    ex = PitchExtractor(cache_dir=REPO / "data" / "cache")
    v2 = _BeatSeqModel(MODELS / "beat_seq_model_v2.npz")

    rec = {k: [] for k in ("gt", "v2", "gtpos", "song")}
    for sid in songs:
        wav = REPO / "data" / "renders" / "pop909" / sid / f"{sid}_v005_musescoregeneral.wav"
        if not wav.exists():
            continue
        y, sr = sf.read(wav)
        y = (y.mean(1) if y.ndim > 1 else y).astype("float32")
        bt, _ = _tempo_grid_beats(y, sr)
        acts = ex.extract(wav)
        onset_b = _pool_beats(acts.frame_times, acts.onset_probs, bt)
        note_b = _pool_beats(acts.frame_times, acts.note_probs, bt)
        v2_pred = v2.predict_proba(onset_b, note_b).argmax(1)

        song = parser.parse_song(sid)
        gt_beats = song.beat_times
        gt_down = song.is_downbeat
        n = min(len(bt) - 1, len(v2_pred))
        for b in range(n):
            center = 0.5 * (bt[b] + bt[b + 1])
            ev = song.chord_at_time(center)
            if ev is None:
                continue
            root_str = ev.label.split(":")[0].split("/")[0]
            if root_str not in HARTE_TO_PC:
                continue
            pos = gt_bar_position(center, gt_beats, gt_down)
            if pos < 0:
                continue
            rec["gt"].append(HARTE_TO_PC[root_str])
            rec["v2"].append(int(v2_pred[b]))
            rec["gtpos"].append(pos)
            rec["song"].append(sid)
    return {k: np.array(v) for k, v in rec.items()}


if __name__ == "__main__":
    rec = collect()
    gt = rec["gt"]; pred = rec["v2"]; pos = rec["gtpos"]
    ok = pred == gt
    print(f"v2 root accuracy by TRUE (GT-downbeat-aligned) bar position:")
    print(f"  overall     {ok.mean():6.1%}  (n={len(gt)})")
    for p in range(4):
        m = pos == p
        tag = " (downbeat)" if p == 0 else ""
        print(f"  bar pos {p}   {ok[m].mean():6.1%}  (n={m.sum()}){tag}")
