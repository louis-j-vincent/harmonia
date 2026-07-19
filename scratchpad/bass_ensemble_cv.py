"""STEP 2 — Calibrated ensemble / stacking for sounding-bass pc (RWC).

Premise check FIRST (CLAUDE.md rule 2): oracle best-of-N ceiling over the base
estimators {NNLS argmax, trained full24 head, trained bass12 head, pYIN}. If the
oracle (pick the estimator that is right, per chord) can't reach 0.95, no stacker can.

Then a real STACKING meta-classifier (leak-free 3-way song split: base head trains on
TRAIN songs, meta trains on VAL songs using head preds it never saw, eval on TEST songs).
Meta features: head-full24 proba(12) + head-bass12 proba(12) + argmax one-hot(12) +
pYIN one-hot(12) + pYIN conf + pYIN reliable flag. Meta = multinomial logistic regression.

Two coverage regimes reported:
  (A) pYIN-covered test rows only  (full ensemble incl. pYIN)
  (B) ALL test rows                (head+argmax ensemble, no pYIN — 100-song coverage)
Every number from a completed run; pooled over TEST rows across seeds.
"""
from __future__ import annotations
import sys, json, argparse
from pathlib import Path
import numpy as np

REPO = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scratchpad")); sys.path.insert(0, str(REPO / "scripts"))
from harmonia.data.corpus_schema import load_corpus, sounding_bass_pc
from multihead_training import train_clf, predict_proba
from rwc_nnls_multihead_cv import song_split
from sklearn.linear_model import LogisticRegression


def oh(idx, k=12):
    idx = np.asarray(idx); out = np.zeros((len(idx), k), np.float32)
    ok = idx >= 0; out[np.arange(len(idx))[ok], idx[ok]] = 1.0
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    nn = load_corpus(REPO / "data/cache/rwc/rwc_nnls24.npz")
    bp = load_corpus(REPO / "data/cache/rwc/rwc_bp48_fixed.npz")
    py = np.load(REPO / "scratchpad/pyin_bass_cache.npz", allow_pickle=True)
    assert (nn["song_id"] == bp["song_id"]).all()

    nn24 = nn["nnls24"].astype(np.float32)
    keep = np.abs(nn24).sum(1) > 0
    nn24 = nn24[keep]
    roots = bp["root"].astype(np.int64)[keep] % 12
    sid = bp["song_id"][keep]
    labels = bp["labels"][keep]
    py_bass = py["bass_pc"][keep].astype(np.int64)
    py_conf = np.nan_to_num(py["conf"][keep].astype(np.float32))
    py_rel = py["reliable"][keep].astype(np.float32)

    gb = np.full(len(roots), -1, np.int64)
    for i in range(len(roots)):
        v = sounding_bass_pc(str(labels[i]), int(roots[i]))
        if v is not None:
            gb[i] = v % 12
    valid = gb >= 0
    inv = (gb != roots) & valid
    argmax = nn24[:, :12].argmax(1)
    has_py = py_bass >= 0

    print(f"rows={len(roots)} valid={valid.sum()} pYIN-covered={has_py.sum()} "
          f"inversions={int(inv.sum())}", flush=True)

    # pooled TEST collectors
    C = {k: [] for k in ["gb", "inv", "haspy", "argmax", "pyin", "pyconf",
                         "headf", "headb", "stackA", "stackB", "sel_conf"]}
    for seed in range(a.seeds):
        tr, va, te = song_split(sid, seed)
        trv = tr & valid
        # base heads (train on TRAIN songs only)
        mf = train_clf(nn24[trv], gb[trv], nn24[va & valid], gb[va & valid], 24, 12, hid=(128, 64), epochs=50)
        mb = train_clf(nn24[trv][:, :12], gb[trv], nn24[va & valid][:, :12], gb[va & valid], 12, 12, hid=(128, 64), epochs=50)
        pf = predict_proba(mf, nn24); pb = predict_proba(mb, nn24[:, :12])

        # meta features
        def meta_X(mask):
            return np.concatenate([pf[mask], pb[mask], oh(argmax[mask]),
                                   oh(np.where(has_py[mask], py_bass[mask], -1)),
                                   py_conf[mask, None], py_rel[mask, None]], 1)
        # stack A: full (incl pYIN), trained on VAL (leak-free), eval TEST
        vav = va & valid; tev = te & valid
        metaA = LogisticRegression(max_iter=2000, C=1.0)
        metaA.fit(meta_X(vav), gb[vav])
        stackA = metaA.predict(meta_X(tev))
        # stack B: head+argmax only (no pYIN) — same recipe minus pYIN cols
        def meta_Xb(mask):
            return np.concatenate([pf[mask], pb[mask], oh(argmax[mask])], 1)
        metaB = LogisticRegression(max_iter=2000, C=1.0)
        metaB.fit(meta_Xb(vav), gb[vav])
        stackB = metaB.predict(meta_Xb(tev))

        C["gb"] += gb[tev].tolist(); C["inv"] += inv[tev].tolist(); C["haspy"] += has_py[tev].tolist()
        C["argmax"] += argmax[tev].tolist(); C["pyin"] += py_bass[tev].tolist(); C["pyconf"] += py_conf[tev].tolist()
        C["headf"] += pf[tev].argmax(1).tolist(); C["headb"] += pb[tev].argmax(1).tolist()
        C["stackA"] += stackA.tolist(); C["stackB"] += stackB.tolist()
        print(f"[seed {seed}] te={tev.sum()} headf={np.mean(pf[tev].argmax(1)==gb[tev]):.3f} "
              f"stackB={np.mean(stackB==gb[tev]):.3f}", flush=True)

    A = {k: np.array(v) for k, v in C.items()}
    GT = A["gb"]; INV = A["inv"] == 1; HP = A["haspy"] == 1; n = len(GT)
    def acc(pred, mask):
        return float((pred[mask] == GT[mask]).mean()) if mask.sum() else float('nan')
    allm = np.ones(n, bool)

    print("\n" + "=" * 70)
    print(f"ENSEMBLE — {a.seeds}-seed song-grouped CV, pooled {n} test chords ({INV.sum()} inv)")
    print("=" * 70)

    # ---- ORACLE CEILING (premise check) ----
    print("\n[PREMISE CHECK] Oracle best-of-N ceiling (per-chord: is ANY estimator correct?)")
    est_all = {"argmax": A["argmax"], "headf": A["headf"], "headb": A["headb"]}
    est_py = dict(est_all); est_py["pyin"] = A["pyin"]
    def oracle(ests, mask):
        correct = np.zeros(mask.sum(), bool)
        for e in ests.values():
            correct |= (e[mask] == GT[mask])
        return float(correct.mean())
    print(f"   {'{argmax,headf,headb}':32s} all={oracle(est_all, allm):.3f}  inv={oracle(est_all, INV):.3f}")
    print(f"   {'{+pyin}, pYIN-covered rows':32s} all={oracle(est_py, HP):.3f}  inv={oracle(est_py, HP & INV):.3f}")

    # ---- base + stack numbers ----
    print("\n[RESULTS] all / inversions / rootpos")
    rows = [("argmax (untrained)", A["argmax"], allm),
            ("head full24", A["headf"], allm),
            ("head bass12", A["headb"], allm),
            ("STACK B (head+argmax, 100-song)", A["stackB"], allm),
            ("STACK A (+pYIN, pYIN rows)", A["stackA"], HP),
            ("argmax (pYIN rows)", A["argmax"], HP),
            ("pYIN raw (pYIN rows)", A["pyin"], HP)]
    res = {}
    for lbl, pred, mask in rows:
        a_all = acc(pred, mask); a_inv = acc(pred, mask & INV); a_rp = acc(pred, mask & ~INV)
        print(f"   {lbl:34s} all={a_all:.3f}  inv={a_inv:.3f}  rootpos={a_rp:.3f}  (n={mask.sum()})")
        res[lbl] = [a_all, a_inv, a_rp, int(mask.sum())]

    # ---- agreement gate (confidence) ----
    agree = (A["argmax"] == A["pyin"]) & HP
    print(f"\n[AGREEMENT GATE] argmax==pyin on {agree.sum()}/{HP.sum()} pYIN rows "
          f"({agree.sum()/max(HP.sum(),1):.3f}); where agree acc={acc(A['argmax'], agree):.3f}, "
          f"disagree acc={acc(A['argmax'], HP & ~agree):.3f}")

    json.dump({"seeds": a.seeds, "n": n, "n_inv": int(INV.sum()),
               "oracle_all_noPy": oracle(est_all, allm), "oracle_all_Py": oracle(est_py, HP),
               "results": res}, open(REPO / "scratchpad/bass_ensemble_result.json", "w"), indent=2)
    print("\nsaved scratchpad/bass_ensemble_result.json")


if __name__ == "__main__":
    main()
