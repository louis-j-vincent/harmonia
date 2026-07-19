"""FAIR oracle-boundary root/quality comparison: music-x-lab vs in-house NNLS-24.

Follow-up to docs/known_issues.md "★ PIVOT LEAD (getting chords right)".  That
lead compared music-x-lab (oracle-boundary, zero-shot, RAW accuracy) against the
NNLS-24 heads (song-grouped CV, BALANCED macro-recall) — two confounds at once
(boundary condition AND metric).  This script removes both:

  * BOUNDARY: both systems scored on the IDENTICAL GT chord segments.  The
    NNLS-24 features in rwc_nnls24.npz are pooled per GT [t0,t1) block, i.e.
    already oracle-boundary; music-x-lab is read at each block's midpoint.  Same
    rows for both (filled NNLS row AND a parseable musx label).
  * METRIC: root raw-acc, quality RAW-acc, quality BALANCED macro-recall,
    per-family recall, and JOINT (root&quality) reported for BOTH systems.

music-x-lab = zero-shot pretrained (cached .lab, midpoint lookup, parse_jaah
family).  NNLS-24 = song-grouped K-fold OOF, multi-seed (deployable recipe:
root MLP + quality cascade rotated by PREDICTED root, rotation-only — the shipped
train_nnls24_heads recipe).  Every row gets an out-of-fold NNLS prediction, so
both systems are scored on exactly the same row set.

Usage: .venv/bin/python scratchpad/chordfair_oracle_compare.py [--seeds 5] [--folds 5] [--smoke]
"""
from __future__ import annotations
import sys, argparse, json
from pathlib import Path
import numpy as np

REPO = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scratchpad"))

import torch
from multihead_training import train_clf, predict_proba, rotate_by_root, balanced_recall
from scripts.build_jaah_corpus import parse_jaah as parse_harte

QUALITIES = ['maj', 'min', 'dom', 'hdim', 'dim', 'aug', 'sus']
QIDX = {q: i for i, q in enumerate(QUALITIES)}
KQ = 7
NPZ = REPO / "data/cache/rwc/rwc_nnls24.npz"
MUSX_OUT = Path("/private/tmp/claude-501/-Users-vincente-Documents-Projets-Perso-Code-"
                "harmonia/a6a4757d-a729-450a-a5be-b9970a43c412/scratchpad/musx_out")


def load_lab(path: Path):
    out = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        p = line.split()
        if len(p) < 3:
            continue
        try:
            out.append((float(p[0]), float(p[1]), p[2]))
        except ValueError:
            continue
    return out


def label_at(intervals, t):
    for t0, t1, lab in intervals:
        if t0 <= t < t1:
            return lab
    return None


def kfold_song(sid, seed, folds):
    songs = np.unique(sid)
    rng = np.random.RandomState(seed)
    rng.shuffle(songs)
    assign = {s: (i % folds) for i, s in enumerate(songs)}
    fold_of = np.array([assign[s] for s in sid])
    return fold_of


def nnls_oof(nn24, roots, quals, sid, seed, folds):
    """Song-grouped K-fold out-of-fold NNLS-24 predictions for ALL rows.

    Deployable recipe (train_nnls24_heads): root MLP(24->128->64->12) on absolute
    nn24; quality cascade = rotate bass|treble by PREDICTED root, MLP(24->..->7),
    class-weighted, rotation-only.  Returns (root_pred, qual_pred) over all rows.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    fold_of = kfold_song(sid, seed, folds)
    N = len(nn24)
    root_pred = np.full(N, -1, np.int64)
    qual_pred = np.full(N, -1, np.int64)
    bass, treb = nn24[:, :12], nn24[:, 12:]
    cnt = np.bincount(quals, minlength=KQ)
    cw = (cnt.sum() / (KQ * np.maximum(cnt, 1))).astype(np.float32)

    for f in range(folds):
        te = fold_of == f
        trpool = ~te
        # val slice out of train pool for early stopping
        rng = np.random.RandomState(seed * 100 + f)
        tp_idx = np.where(trpool)[0]
        va_pick = rng.choice(tp_idx, size=max(1, len(tp_idx) // 8), replace=False)
        va = np.zeros(N, bool); va[va_pick] = True
        tr = trpool & ~va

        # root head
        rm = train_clf(nn24[tr], roots[tr], nn24[va], roots[va], 24, 12,
                       hid=(128, 64), epochs=50)
        rproba = predict_proba(rm, nn24)          # in-sample for tr, OOF for te
        prroot = rproba.argmax(1)
        root_pred[te] = prroot[te]

        # quality cascade — rotate by PREDICTED root
        Xc = np.concatenate([rotate_by_root(bass, prroot),
                             rotate_by_root(treb, prroot)], 1)
        qm = train_clf(Xc[tr], quals[tr], Xc[va], quals[va], 24, KQ,
                       hid=(128, 64), epochs=60, cw=cw)
        qpred_all = predict_proba(qm, Xc).argmax(1)
        qual_pred[te] = qpred_all[te]
    return root_pred, qual_pred


def score(name, root_p, qual_p, gt_root, gt_qual):
    """Return dict of root raw, qual raw, qual balanced, joint, per-family recall."""
    root_raw = float((root_p == gt_root).mean())
    qual_raw = float((qual_p == gt_qual).mean())
    rec = balanced_recall(qual_p, gt_qual, KQ)
    qual_bal = float(np.nanmean(rec))
    joint = float(((root_p == gt_root) & (qual_p == gt_qual)).mean())
    return dict(root_raw=root_raw, qual_raw=qual_raw, qual_bal=qual_bal,
                joint=joint, rec=[float(x) for x in rec])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.seeds, a.folds = 1, 3

    d = np.load(NPZ, allow_pickle=True)
    nn24 = d["nnls24"].astype(np.float32)
    roots = d["root"].astype(np.int64) % 12
    quals = d["quality_idx"].astype(np.int64)
    labels = d["labels"]
    sid = np.array([str(s) for s in d["song_id"]])
    t0 = d["t0"].astype(float); t1 = d["t1"].astype(float)
    assert list(map(str, d["qualities"])) == QUALITIES
    filled = np.abs(nn24).sum(1) > 0

    # --- music-x-lab prediction per row (midpoint lookup, zero-shot) ---
    N = len(roots)
    mx_root = np.full(N, -1, np.int64)
    mx_qual = np.full(N, -1, np.int64)
    lab_cache: dict[str, list] = {}
    for i in range(N):
        song = sid[i].replace("rwc_", "")
        if song not in lab_cache:
            lab_cache[song] = load_lab(MUSX_OUT / f"{song}.lab")
        lab = label_at(lab_cache[song], 0.5 * (t0[i] + t1[i]))
        if lab is None:
            continue
        r, fam, _ = parse_harte(lab)
        if r is None or fam is None:
            continue
        mx_root[i] = r % 12
        mx_qual[i] = QIDX[fam]

    keep = filled & (mx_root >= 0) & (mx_qual >= 0)
    gt_root = roots[keep]; gt_qual = quals[keep]
    print(f"rows kept={keep.sum()} / {N}  songs={len(np.unique(sid[keep]))}", flush=True)
    print(f"GT quality dist: {dict(zip(QUALITIES, np.bincount(gt_qual, minlength=KQ).tolist()))}",
          flush=True)

    # --- music-x-lab scores (deterministic) ---
    mx = score("musx", mx_root[keep], mx_qual[keep], gt_root, gt_qual)
    print("\n=== music-x-lab (zero-shot, oracle-boundary) ===")
    print(f"  root_raw={mx['root_raw']:.4f}  qual_raw={mx['qual_raw']:.4f}  "
          f"qual_bal={mx['qual_bal']:.4f}  joint={mx['joint']:.4f}")
    print(f"  per-family recall {dict(zip(QUALITIES, np.round(mx['rec'],3).tolist()))}")

    # --- NNLS-24 multi-seed OOF ---
    nn_runs = []
    nn_preds = []  # (rp, qp) per seed, reused by the cascade
    for s in range(a.seeds):
        rp, qp = nnls_oof(nn24, roots, quals, sid, s, a.folds)
        sc = score("nnls", rp[keep], qp[keep], gt_root, gt_qual)
        nn_runs.append(sc)
        nn_preds.append((rp, qp))
        print(f"[nnls seed {s}] root_raw={sc['root_raw']:.4f} qual_raw={sc['qual_raw']:.4f} "
              f"qual_bal={sc['qual_bal']:.4f} joint={sc['joint']:.4f}", flush=True)

    def ms(k):
        v = np.array([r[k] for r in nn_runs]); return float(v.mean()), float(v.std())
    nn_rec = np.array([r['rec'] for r in nn_runs])
    nn_rec_m = np.nanmean(nn_rec, 0)
    print("\n=== NNLS-24 (deployable cascade, song-grouped OOF, "
          f"{a.seeds}-seed mean±std) ===")
    for k in ("root_raw", "qual_raw", "qual_bal", "joint"):
        m, sd = ms(k); print(f"  {k:9s} = {m:.4f} ± {sd:.4f}")
    print(f"  per-family recall {dict(zip(QUALITIES, np.round(nn_rec_m,3).tolist()))}")

    # --- head-to-head per-family recall table ---
    print("\n=== per-family recall: musx vs nnls (Δ = musx − nnls) ===")
    for i, q in enumerate(QUALITIES):
        n_q = int((gt_qual == i).sum())
        dm = mx['rec'][i] - nn_rec_m[i]
        print(f"  {q:5s} (n={n_q:5d})  musx={mx['rec'][i]:.3f}  nnls={nn_rec_m[i]:.3f}  Δ={dm:+.3f}")

    # --- CASCADE (a): musx common-chord front-end, NNLS rare-quality residual ---
    # Root always musx (stronger). Quality: musx if musx∈{maj,min} else NNLS.
    # Also a full-replacement baseline (musx root+qual). Per-seed (NNLS varies).
    print("\n=== INTEGRATION STRATEGIES (joint root&quality, "
          f"{a.seeds}-seed for NNLS-dependent) ===")
    casc_runs = []
    for s in range(a.seeds):
        rp, qp = nn_preds[s]
        rp_k, qp_k = rp[keep], qp[keep]
        mxr, mxq = mx_root[keep], mx_qual[keep]
        common = np.isin(mxq, [QIDX['maj'], QIDX['min']])
        # cascade A: root=musx; qual = musx if common(maj/min) else nnls
        cq = np.where(common, mxq, qp_k)
        cr = mxr  # root from musx
        jointA = float(((cr == gt_root) & (cq == gt_qual)).mean())
        # cascade B: root=musx common, else nnls root+qual on residual
        cr2 = np.where(common, mxr, rp_k)
        jointB = float(((cr2 == gt_root) & (cq == gt_qual)).mean())
        # cascade C: SURGICAL — root=musx; quality=NNLS only where musx predicts a
        #   family NNLS is measurably better at (dim/aug/sus), else keep musx.
        surg = np.isin(mxq, [QIDX['dim'], QIDX['aug'], QIDX['sus']])
        cqC = np.where(surg, qp_k, mxq)
        jointC = float(((mxr == gt_root) & (cqC == gt_qual)).mean())
        recC = balanced_recall(cqC, gt_qual, KQ)
        casc_runs.append((jointA, jointB, jointC, float(np.nanmean(recC))))
    jA = np.array([c[0] for c in casc_runs]); jB = np.array([c[1] for c in casc_runs])
    jC = np.array([c[2] for c in casc_runs]); balC = np.array([c[3] for c in casc_runs])
    mx_full_joint = mx['joint']
    nn_joint_m, nn_joint_sd = ms('joint')
    print(f"  full-replacement (musx root+qual)           joint={mx_full_joint:.4f}  (qual_bal={mx['qual_bal']:.4f})")
    print(f"  NNLS-24 alone                               joint={nn_joint_m:.4f} ± {nn_joint_sd:.4f}")
    print(f"  cascade A (musx root; nnls qual off maj/min)joint={jA.mean():.4f} ± {jA.std():.4f}")
    print(f"  cascade B (musx common; nnls root+qual res) joint={jB.mean():.4f} ± {jB.std():.4f}")
    print(f"  cascade C (SURGICAL nnls on dim/aug/sus)    joint={jC.mean():.4f} ± {jC.std():.4f}  "
          f"(qual_bal={balC.mean():.4f} ± {balC.std():.4f})")

    out = dict(
        n_rows=int(keep.sum()), n_songs=int(len(np.unique(sid[keep]))), seeds=a.seeds, folds=a.folds,
        gt_qual_dist=dict(zip(QUALITIES, np.bincount(gt_qual, minlength=KQ).tolist())),
        musx=mx,
        nnls={k: ms(k) for k in ("root_raw", "qual_raw", "qual_bal", "joint")},
        nnls_rec=nn_rec_m.tolist(),
        integration=dict(
            full_replacement_joint=mx_full_joint,
            nnls_alone_joint=[nn_joint_m, nn_joint_sd],
            cascadeA_joint=[float(jA.mean()), float(jA.std())],
            cascadeB_joint=[float(jB.mean()), float(jB.std())],
            cascadeC_surgical_joint=[float(jC.mean()), float(jC.std())],
            cascadeC_surgical_qualbal=[float(balC.mean()), float(balC.std())],
        ),
    )
    json.dump(out, open(REPO / "scratchpad/chordfair_oracle_result.json", "w"), indent=2)
    print("\nsaved scratchpad/chordfair_oracle_result.json")


if __name__ == "__main__":
    main()
