"""Train root+quality heads on the FIXED (frame-clipped) RWC corpus, single
seed, --roll methodology (matches train_jaah_cv.py's one_split exactly for
seed=0). Saves the checkpoint + full per-record predictions/probs for the
held-out test split so the error-report tool can pick wrong-root examples
without re-running inference.

Read-only on rwc_bp48_fixed.npz. Writes only to data/models/ (new filename)
and docs/error_report_wrong_root_2026_07_16/ (new dir, durable — scratchpad
proved unreliable this session).
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import torch

REPO = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from harmonia.data.corpus_schema import MatchQuality, filter_by_match, load_corpus
from train_real_audio_final import (
    _train_head, _eval, _augment_root_by_roll, QUALITIES, _standardize,
)

SEED = 0
CORPUS = REPO / "data/cache/rwc/rwc_bp48_fixed.npz"
OUT_MODEL = REPO / "data/models/_eval_only_rwc_bp48_fixed_root_2026_07_16.pt"
OUT_DIR = REPO / "docs/error_report_wrong_root_2026_07_16"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_MANIFEST = OUT_DIR / "fixed_test_predictions.json"

device = "mps" if torch.backends.mps.is_available() else "cpu"

d = load_corpus(CORPUS)
keep = filter_by_match(d["match"], minimum=MatchQuality.EXACT)
feat48 = d["feat48"][keep]; feat48_abs = d["feat48_abs"][keep]
quality_idx = d["quality_idx"].astype(int)[keep]
roots = d["root"].astype(int)[keep]
song_id = d["song_id"][keep]
labels = d["labels"][keep]
t0 = d["t0"][keep]; t1 = d["t1"][keep]
quality_str = d["quality"][keep]

songs = sorted(set(song_id.tolist()))
rng = np.random.RandomState(SEED); rng.shuffle(songs)
n_test = max(1, int(round(0.2 * len(songs))))
test_songs = set(songs[:n_test])
tr = np.array([s not in test_songs for s in song_id])
te = ~tr
print(f"train {tr.sum()} / test {te.sum()} over {len(test_songs)} test songs / {len(songs)} total", flush=True)

# --- root head (feat48_abs, roll augment) ---
Xtr, ytr = feat48_abs[tr], roots[tr]
Xtr, ytr = _augment_root_by_roll(Xtr, ytr)
root_model, root_mean, root_std = _train_head(
    Xtr, ytr, 12, epochs=60, lr=3e-4, batch=64, device=device, head_name="root")
root_acc, root_recall, root_preds_te = _eval(feat48_abs[te], roots[te], root_model, root_mean, root_std, device)
print(f"ROOT held-out acc = {root_acc:.4f}", flush=True)

Xte_n = ((feat48_abs[te] - root_mean) / root_std).astype(np.float32)
with torch.no_grad():
    logits = root_model(torch.tensor(Xte_n, device=device))
    root_probs_te = torch.softmax(logits, dim=1).cpu().numpy()

# --- quality head (feat48, root-relative) ---
qual_model, qual_mean, qual_std = _train_head(
    feat48[tr], quality_idx[tr], 7, epochs=60, lr=3e-4, batch=64, device=device, head_name="qual")
qual_acc, qual_recall, qual_preds_te = _eval(feat48[te], quality_idx[te], qual_model, qual_mean, qual_std, device)
print(f"QUALITY held-out acc = {qual_acc:.4f}", flush=True)

Xte_qn = ((feat48[te] - qual_mean) / qual_std).astype(np.float32)
with torch.no_grad():
    logits_q = qual_model(torch.tensor(Xte_qn, device=device))
    qual_probs_te = torch.softmax(logits_q, dim=1).cpu().numpy()

bass_block_te = feat48_abs[te][:, 24:36]
bass_argmax_te = bass_block_te.argmax(1)

torch.save({
    "root_model_state": root_model.state_dict(),
    "root_mean": root_mean, "root_std": root_std,
    "qual_model_state": qual_model.state_dict(),
    "qual_mean": qual_mean, "qual_std": qual_std,
    "seed": SEED, "corpus": str(CORPUS), "roll": True,
    "root_held_out_acc": root_acc, "qual_held_out_acc": qual_acc,
    "test_songs": sorted(test_songs),
    "note": "trained by error_report_wrong_root tool 2026-07-16 on FIXED (frame-clipped, zero-bleed) RWC corpus, single seed=0, --roll methodology matching train_jaah_cv.py",
}, OUT_MODEL)
print(f"saved model -> {OUT_MODEL}", flush=True)

idx_te = np.nonzero(te)[0]
records = []
for i, gi in enumerate(idx_te):
    records.append({
        "song_id": str(song_id[gi]),
        "t0": float(t0[gi]), "t1": float(t1[gi]),
        "label": str(labels[gi]),
        "gt_root": int(roots[gi]),
        "gt_quality": str(quality_str[gi]),
        "gt_quality_idx": int(quality_idx[gi]),
        "pred_root": int(root_preds_te[i]),
        "pred_root_prob": float(root_probs_te[i, root_preds_te[i]]),
        "root_probs": root_probs_te[i].tolist(),
        "pred_quality_idx": int(qual_preds_te[i]),
        "pred_quality": QUALITIES[int(qual_preds_te[i])],
        "pred_quality_prob": float(qual_probs_te[i, qual_preds_te[i]]),
        "bass_argmax_pc": int(bass_argmax_te[i]),
    })

with open(OUT_MANIFEST, "w") as f:
    json.dump({
        "qualities": QUALITIES,
        "root_held_out_acc": root_acc,
        "qual_held_out_acc": qual_acc,
        "test_songs": sorted(test_songs),
        "records": records,
    }, f, indent=1)
print(f"saved manifest -> {OUT_MANIFEST} ({len(records)} test records)", flush=True)

n_wrong = sum(1 for r in records if r["pred_root"] != r["gt_root"])
print(f"wrong-root count in test = {n_wrong} / {len(records)} ({n_wrong/len(records):.1%})", flush=True)
print("DONE", flush=True)
