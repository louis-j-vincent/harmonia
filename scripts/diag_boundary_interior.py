"""Premise check: reproduce the boundary/interior/beat-1 root-accuracy breakdown
for the EXISTING beat_seq_model_v2 (LR 240d) and beat_seq_model_v3 (canonical).

Renders (data/renders/pop909/00X/) and BP extraction cache (data/cache/) both
exist, so this is fast — no re-render, no re-extract.

A "boundary beat" = its GT root differs from the previous valid beat's GT root.
Beat position in bar = tempo-grid beat index % 4 (as the diagnostic spec says:
"Estimated from the detected beat grid, beat index mod 4").
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


def collect(songs=("001", "002", "003", "004", "005")):
    """Return dict of per-beat arrays: gt_root, pos, boundary, song + model preds."""
    parser = POP909Parser(POP)
    ex = PitchExtractor(cache_dir=REPO / "data" / "cache")
    v2 = _BeatSeqModel(MODELS / "beat_seq_model_v2.npz")
    v3 = V3Model(MODELS / "beat_seq_model_v3.npz")

    rec = {k: [] for k in ("gt", "v2", "v3", "pos", "bnd", "song")}
    for sid in songs:
        wav = REPO / "data" / "renders" / "pop909" / sid / f"{sid}_v005_musescoregeneral.wav"
        if not wav.exists():
            print(f"  skip {sid}: no render"); continue
        y, sr = sf.read(wav)
        y = (y.mean(1) if y.ndim > 1 else y).astype("float32")
        bt, _ = _tempo_grid_beats(y, sr)
        acts = ex.extract(wav)
        onset_b = _pool_beats(acts.frame_times, acts.onset_probs, bt)
        note_b = _pool_beats(acts.frame_times, acts.note_probs, bt)
        v2_pred = v2.predict_proba(onset_b, note_b).argmax(1)
        v3_pred = v3.predict_proba(onset_b, note_b)[0].argmax(1)

        song = parser.parse_song(sid)
        n = min(len(bt) - 1, len(v2_pred))
        prev_gt = None
        for b in range(n):
            ev = song.chord_at_time(0.5 * (bt[b] + bt[b + 1]))
            if ev is None:
                prev_gt = None
                continue
            root_str = ev.label.split(":")[0].split("/")[0]
            if root_str not in HARTE_TO_PC:
                prev_gt = None
                continue
            gt = HARTE_TO_PC[root_str]
            rec["gt"].append(gt)
            rec["v2"].append(int(v2_pred[b]))
            rec["v3"].append(int(v3_pred[b]))
            rec["pos"].append(b % 4)
            rec["bnd"].append(prev_gt is not None and gt != prev_gt)
            rec["song"].append(sid)
            prev_gt = gt
    return {k: np.array(v) for k, v in rec.items()}


def report(rec):
    gt = rec["gt"]; pos = rec["pos"]; bnd = rec["bnd"]; song = rec["song"]
    for model in ("v2", "v3"):
        pred = rec[model]
        ok = pred == gt
        print(f"\n=== {model} ===")
        print(f"  overall       {ok.mean():6.1%}  (n={len(gt)})")
        print(f"  interior      {ok[~bnd].mean():6.1%}  (n={(~bnd).sum()})")
        print(f"  boundary      {ok[bnd].mean():6.1%}  (n={bnd.sum()})")
        for p in range(4):
            m = pos == p
            print(f"  beat pos {p}    {ok[m].mean():6.1%}  (n={m.sum()})")
        print("  per-song:", end="")
        for sid in ("001", "002", "003", "004", "005"):
            m = song == sid
            if m.any():
                print(f"  {sid}={ok[m].mean():.1%}", end="")
        print()


if __name__ == "__main__":
    report(collect())
