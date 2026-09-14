"""train_nnls24_heads.py — train the SHIPPED NNLS-24 production inference heads.

Produces harmonia/models/nnls24_heads.npz, the checkpoint consumed by the opt-in
NNLS-24 front-end of harmonia.models.chord_pipeline_v1.infer_chords_v1
(feature_frontend="nnls24").  This is the production counterpart of the
oracle-boundary CV harness scripts/rwc_nnls_multihead_cv.py — same recipe, same
features (data/cache/rwc/rwc_nnls24.npz, real Mauch NNLS-Chroma VAMP, C-frame,
L2-per-half; bass=[:12], treble=[12:]), but trained on ALL RWC rows for
deployment (no held-out) after a single-split sanity check is logged.

Two heads (bass is untrained argmax on the NNLS bass half — no weights):
  * root head    : MLP(24 -> 128 -> 64 -> 12) on the ABSOLUTE nnls24 vector
                   (predicts the C-frame root pitch-class).  RWC 5-seed CV: 0.789.
  * quality head : the DEPLOYABLE cascade — rotate bass|treble by the PREDICTED
                   root (candidate root -> index 0), MLP(24 -> 128 -> 64 -> 7),
                   class-weighted.  Rotation-only, NO trigram context: on real
                   audio (RWC) the learned trigram HURTS quality by ~7.9pp
                   (SESSION_PRESENTATION_2026_07_17 §4 row 7), so the shipped
                   recipe is rotation-only.

Serialisation: every torch state_dict tensor is stored as an npz array keyed
`root__<k>` / `qual__<k>`, plus scalar metadata, so the checkpoint loads with a
plain np.load + a small MLP reconstruction (see NNLS24Heads in
harmonia/models/nnls_features.py).  No torch object pickling.

Usage:  .venv/bin/python scripts/train_nnls24_heads.py [--seed 0]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scratchpad"))

# Verified recipe (imported, not reconstructed) — same functions the CV harness
# scripts/rwc_nnls_multihead_cv.py uses.
from multihead_training import MLP, train_clf, predict_proba, rotate_by_root  # noqa: E402

QUALITIES = ["maj", "min", "dom", "hdim", "dim", "aug", "sus"]
KQ = 7
NPZ = REPO / "data" / "cache" / "rwc" / "rwc_nnls24.npz"
OUT = REPO / "harmonia" / "models" / "nnls24_heads.npz"


def _state_arrays(model, prefix: str) -> dict[str, np.ndarray]:
    return {f"{prefix}__{k}": v.detach().cpu().numpy()
            for k, v in model.state_dict().items()}


def song_split(sid, seed, test_frac=0.2):
    songs = np.unique(sid)
    rng = np.random.RandomState(seed)
    rng.shuffle(songs)
    nte = max(1, int(round(test_frac * len(songs))))
    te = np.isin(sid, songs[:nte])
    return ~te, te


def train_heads(nn24, roots, quals, sid, tr_mask, va_mask):
    """Train root + quality-cascade heads on rows selected by tr_mask.

    va_mask supplies early-stopping validation (a slice of the training pool).
    Returns (root_model, qual_model, root_proba_full) where root_proba_full is
    over ALL rows (needed to rotate the quality features by predicted root).
    """
    bass, treb = nn24[:, :12], nn24[:, 12:]
    cnt = np.bincount(quals, minlength=KQ)
    cw = (cnt.sum() / (KQ * np.maximum(cnt, 1))).astype(np.float32)

    # root head — absolute nnls24
    rm = train_clf(nn24[tr_mask], roots[tr_mask], nn24[va_mask], roots[va_mask],
                   24, 12, hid=(128, 64), epochs=50)
    root_proba = predict_proba(rm, nn24)          # (N, 12)
    pred_root = root_proba.argmax(1)

    # quality cascade — rotate by PREDICTED root (deployable), rotation-only
    Xc = np.concatenate([rotate_by_root(bass, pred_root),
                         rotate_by_root(treb, pred_root)], 1)
    qm = train_clf(Xc[tr_mask], quals[tr_mask], Xc[va_mask], quals[va_mask],
                   24, KQ, hid=(128, 64), epochs=60, cw=cw)
    return rm, qm, root_proba


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    d = np.load(NPZ, allow_pickle=True)
    nn24 = d["nnls24"].astype(np.float32)
    roots = d["root"].astype(np.int64) % 12
    quals = d["quality_idx"].astype(np.int64)
    sid = d["song_id"]
    assert list(map(str, d["qualities"])) == QUALITIES, "quality order mismatch"
    print(f"rows={len(nn24)} songs={len(np.unique(sid))}  "
          f"qual dist={dict(zip(QUALITIES, np.bincount(quals, minlength=KQ).tolist()))}",
          flush=True)

    # ── 1. single-split sanity (held-out songs) ──────────────────────────────
    tr, te = song_split(sid, a.seed)
    # carve a small val slice out of the train pool for early stopping
    rng = np.random.RandomState(a.seed + 100)
    tr_idx = np.where(tr)[0]
    va_pick = rng.choice(tr_idx, size=max(1, len(tr_idx) // 8), replace=False)
    va = np.zeros(len(nn24), bool); va[va_pick] = True
    trs = tr & ~va
    rm, qm, rp = train_heads(nn24, roots, quals, sid, trs, va)
    root_acc = float((rp[te].argmax(1) == roots[te]).mean())
    bass_arg = nn24[:, :12].argmax(1)
    bass_acc = float((bass_arg[te] == roots[te]).mean())   # untrained bass->root proxy
    Xc = np.concatenate([rotate_by_root(nn24[:, :12], rp.argmax(1)),
                         rotate_by_root(nn24[:, 12:], rp.argmax(1))], 1)
    qpred = predict_proba(qm, Xc[te]).argmax(1)
    q_raw = float((qpred == quals[te]).mean())
    q_bal = float(np.nanmean([
        (qpred[quals[te] == c] == c).mean() if (quals[te] == c).any() else np.nan
        for c in range(KQ)]))
    print(f"[sanity seed={a.seed}]  root_acc={root_acc:.3f}  "
          f"qual_raw={q_raw:.3f}  qual_bal={q_bal:.3f}  "
          f"bass-argmax->root(untrained)={bass_acc:.3f}  "
          f"(train {trs.sum()} / test {te.sum()})", flush=True)

    # ── 2. SHIPPED heads — train on ALL rows (small val slice for early stop) ─
    rng2 = np.random.RandomState(a.seed + 200)
    va_pick2 = rng2.choice(len(nn24), size=max(1, len(nn24) // 8), replace=False)
    va_all = np.zeros(len(nn24), bool); va_all[va_pick2] = True
    tr_all = ~va_all
    rm_f, qm_f, _ = train_heads(nn24, roots, quals, sid, tr_all, va_all)

    arrs: dict[str, np.ndarray] = {}
    arrs.update(_state_arrays(rm_f, "root"))
    arrs.update(_state_arrays(qm_f, "qual"))
    arrs["root_din"] = np.array([24]); arrs["root_dout"] = np.array([12])
    arrs["qual_din"] = np.array([24]); arrs["qual_dout"] = np.array([KQ])
    arrs["hid"] = np.array([128, 64])
    arrs["qualities"] = np.array(QUALITIES)
    arrs["frame"] = np.array(["C"])          # nnls24 is already C-frame, L2-per-half
    arrs["sanity"] = np.array([root_acc, q_raw, q_bal, bass_acc])
    np.savez(OUT, **arrs)
    print(f"saved {OUT}  ({OUT.stat().st_size} bytes)", flush=True)


if __name__ == "__main__":
    main()
