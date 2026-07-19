"""Cache per-beat root/quality softmax posteriors for the 4 boundary-diag songs,
so p_self can be swept offline without re-downloading audio / re-running Basic
Pitch each time. Read-only w.r.t. chord_pipeline_v1.py — duplicates the
per-beat feature-extraction lines (steps 1-4 of infer_chords_billboard_v1)
directly rather than importing private internals.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

REPO = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia")
sys.path.insert(0, str(REPO))
from harmonia.models.chord_pipeline_v1 import extract_beat_features, _get_billboard_model
from harmonia.data.yt_chord_corpus import seg_feature, seg_feature_abs, download_audio

CACHE = REPO / "data/cache/billboard_60"
AUDIO = CACHE / "audio"; BP = CACHE / "bp_cache"; OUT = REPO / "scratchpad" / "beat_posteriors"
OUT.mkdir(parents=True, exist_ok=True)

SONGS = {
    "1111": "3joI5VtuNV0",
    "887":  "NerII_Zgd5I",
    "1027": "qNHWpi7CKgU",
    "362":  "JRiAMe1zsQ0",
}

ckpt = _get_billboard_model()
root_model, root_mean, root_std = ckpt["root_model"], ckpt["root_mean"], ckpt["root_std"]
qual_model, qual_mean, qual_std = ckpt["quality_model"], ckpt["quality_mean"], ckpt["quality_std"]
qualities = ckpt["qualities"]

for tid, vid in SONGS.items():
    outp = OUT / f"bb_{tid}.npz"
    if outp.exists():
        print("skip (cached)", tid)
        continue
    print("processing", tid, vid, flush=True)
    wav = download_audio(vid, AUDIO)
    try:
        bf = extract_beat_features(wav, cache_dir=BP)
    finally:
        wav.unlink(missing_ok=True)
    onset_b, note_b, bt, tempo_bpm = bf.onset_b, bf.note_b, bf.beat_times, bf.tempo_bpm
    n_beats = len(bt) - 1
    root_ps = np.empty((n_beats, len(root_mean) and 12), dtype=np.float32)
    # determine root dim properly
    root_ps = []
    qual_ps = []
    with torch.no_grad():
        for i in range(n_beats):
            fabs = seg_feature_abs(onset_b, note_b, i, i + 1)
            x_root = torch.tensor(((fabs - root_mean) / root_std)[None], dtype=torch.float32)
            root_p = torch.softmax(root_model(x_root)[0], dim=0).numpy()
            root = int(root_p.argmax())
            frel = seg_feature(onset_b, note_b, i, i + 1, root)
            x_q = torch.tensor(((frel - qual_mean) / qual_std)[None], dtype=torch.float32)
            q_p = torch.softmax(qual_model(x_q)[0], dim=0).numpy()
            root_ps.append(root_p)
            qual_ps.append(q_p)
    root_ps = np.array(root_ps, dtype=np.float32)
    qual_ps = np.array(qual_ps, dtype=np.float32)
    np.savez(outp, root_p=root_ps, qual_p=qual_ps, bt=bt, tempo_bpm=tempo_bpm,
             qualities=np.array(qualities))
    print("saved", outp, root_ps.shape, qual_ps.shape, flush=True)

print("DONE")
