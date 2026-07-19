"""NNLS-24 vs BP48 quality analysis on the SAME RWC 5-seed CV splits as
scripts/rwc_nnls_multihead_cv.py. Two deliverables (both oracle-root frame, the
fair front-end comparison; NNLS uses the winning rotation-only recipe, BP48 uses
its established root-relative feat48):

(1) 3rd-vs-7th confusion breakdown — does NNLS's treble sharpness concentrate its
    quality gain on 3rd (maj<->min) and 7th (dom<->maj etc.) discrimination?
    Pooled 7x7 confusion over all 5 seeds' test predictions, per-true-class
    normalized; extract the 3rd-axis and 7th-axis cells for NNLS and BP48.

(2) Maj/min cascade test — fraction of chords that are easy maj/min, and each
    front-end's accuracy on that easy subset (binary maj-vs-min) vs the harder
    residual (dom/hdim/dim/aug/sus, 5-way balanced acc).

All numbers printed come from completed training runs. Log: nnls_quality_breakdown.log
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np

REPO = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scratchpad"))
from harmonia.data.corpus_schema import load_corpus
from multihead_training import train_clf, predict_proba, rotate_by_root, balanced_recall

QUALITIES = ['maj', 'min', 'dom', 'hdim', 'dim', 'aug', 'sus']
KQ = 7
SEEDS = 5


def song_split(sid, seed, test_frac=0.2, val_frac=0.1):
    songs = np.unique(sid); rng = np.random.RandomState(seed); rng.shuffle(songs)
    n = len(songs)
    nte = max(1, int(round(test_frac * n))); nva = max(1, int(round(val_frac * n)))
    te = np.isin(sid, list(songs[:nte])); va = np.isin(sid, list(songs[nte:nte + nva]))
    return ~(te | va), va, te


def main():
    nn = load_corpus(REPO / "data/cache/rwc/rwc_nnls24.npz")
    bp = load_corpus(REPO / "data/cache/rwc/rwc_bp48_fixed.npz")
    assert (nn["song_id"] == bp["song_id"]).all() and (nn["root"] == bp["root"]).all()

    roots = bp["root"].astype(np.int64) % 12
    quals = bp["quality_idx"].astype(np.int64)
    sid = bp["song_id"]
    labs = bp["labels"]
    nn_bass, nn_treb = nn["nnls24"][:, :12].astype(np.float32), nn["nnls24"][:, 12:].astype(np.float32)
    f48rr = bp["feat48"].astype(np.float32)         # BP48 root-relative (oracle frame)

    # oracle-root-frame NNLS feature (rotation-only, the winning recipe)
    Xnn = np.concatenate([rotate_by_root(nn_bass, roots), rotate_by_root(nn_treb, roots)], 1)
    Xbp = f48rr

    cw = (np.bincount(quals, minlength=KQ).sum() /
          (KQ * np.maximum(np.bincount(quals, minlength=KQ), 1))).astype(np.float32)

    # ---- (1) pooled confusion over 5 seeds ----
    conf = {"NNLS": np.zeros((KQ, KQ)), "BP48": np.zeros((KQ, KQ))}
    for seed in range(SEEDS):
        tr, va, te = song_split(sid, seed)
        for name, X in [("NNLS", Xnn), ("BP48", Xbp)]:
            m = train_clf(X[tr], quals[tr], X[va], quals[va], X.shape[1], KQ,
                          hid=(128, 64), epochs=60, cw=cw)
            pr = predict_proba(m, X[te]).argmax(1)
            for t, p in zip(quals[te], pr):
                conf[name][t, p] += 1
        print(f"  [confusion] seed {seed} done", flush=True)

    def rownorm(C):
        return C / np.maximum(C.sum(1, keepdims=True), 1)

    print("\n=== (1) 3rd-vs-7th CONFUSION (pooled 5 seeds, per-true-class rate) ===")
    out = {"confusion": {}, "axes": {}}
    for name in ("NNLS", "BP48"):
        R = rownorm(conf[name])
        out["confusion"][name] = R.tolist()
        mi = {q: i for i, q in enumerate(QUALITIES)}
        # 3rd axis: maj<->min
        maj_min = R[mi['maj'], mi['min']]; min_maj = R[mi['min'], mi['maj']]
        # 7th axis: dom<->maj (dom = maj triad + b7); hdim<->min ; hdim<->dim
        dom_maj = R[mi['dom'], mi['maj']]; maj_dom = R[mi['maj'], mi['dom']]
        dom_min = R[mi['dom'], mi['min']]
        hdim_min = R[mi['hdim'], mi['min']]; hdim_dim = R[mi['hdim'], mi['dim']]
        rec = {q: float(R[mi[q], mi[q]]) for q in QUALITIES}
        ax = dict(third_maj2min=float(maj_min), third_min2maj=float(min_maj),
                  seventh_dom2maj=float(dom_maj), seventh_maj2dom=float(maj_dom),
                  seventh_dom2min=float(dom_min), seventh_hdim2min=float(hdim_min),
                  seventh_hdim2dim=float(hdim_dim), recall=rec)
        out["axes"][name] = ax
        print(f"\n {name}: diag recall = " + " ".join(f"{q}={rec[q]:.2f}" for q in QUALITIES))
        print(f"   3rd-axis conf : maj->min={maj_min:.3f}  min->maj={min_maj:.3f}")
        print(f"   7th-axis conf : dom->maj={dom_maj:.3f}  maj->dom={maj_dom:.3f}  "
              f"dom->min={dom_min:.3f}  hdim->min={hdim_min:.3f}  hdim->dim={hdim_dim:.3f}")

    # ---- (2) maj/min cascade ----
    print("\n=== (2) MAJ/MIN CASCADE ===")
    def qtok(l):
        l = str(l); return l.split(':', 1)[1].split('/', 1)[0] if ':' in l else l
    pure = np.array([qtok(l) in ('maj', 'min') for l in labs])
    fam = np.isin(quals, [0, 1])                      # collapsed maj/min family
    print(f"  pure maj/min triad (Harte exactly maj|min): {pure.sum()}/{len(pure)} = {pure.mean():.3f}")
    print(f"  collapsed maj/min family (quality_idx 0/1): {fam.sum()}/{len(fam)} = {fam.mean():.3f}")
    resid = ['dom', 'hdim', 'dim', 'aug', 'sus']
    residmap = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4}

    casc = {"NNLS": {"bin": [], "resid": []}, "BP48": {"bin": [], "resid": []}}
    for seed in range(SEEDS):
        tr, va, te = song_split(sid, seed)
        for name, X in [("NNLS", Xnn), ("BP48", Xbp)]:
            # binary maj-vs-min on the collapsed family subset
            trf = tr & fam; vaf = va & fam; tef = te & fam
            yb = quals.copy()                          # 0=maj,1=min already
            mb = train_clf(X[trf], yb[trf], X[vaf], yb[vaf], X.shape[1], 2, hid=(64, 32), epochs=50)
            acc_bin = float((predict_proba(mb, X[tef]).argmax(1) == yb[tef]).mean())
            casc[name]["bin"].append(acc_bin)
            # residual 5-way balanced acc on non-family subset
            rmask = ~fam
            trr = tr & rmask; var = va & rmask; ter = te & rmask
            yr = np.array([residmap.get(q, -1) for q in quals])
            cwr = (np.bincount(yr[trr], minlength=5).sum() /
                   (5 * np.maximum(np.bincount(yr[trr], minlength=5), 1))).astype(np.float32)
            mr = train_clf(X[trr], yr[trr], X[var], yr[var], X.shape[1], 5, hid=(64, 32),
                           epochs=60, cw=cwr)
            prr = predict_proba(mr, X[ter]).argmax(1)
            balr = float(np.nanmean(balanced_recall(prr, yr[ter], 5)))
            casc[name]["resid"].append(balr)
        print(f"  [cascade] seed {seed} done", flush=True)

    out["cascade"] = {"frac_pure_triad": float(pure.mean()), "frac_family": float(fam.mean())}
    for name in ("NNLS", "BP48"):
        b = np.array(casc[name]["bin"]); r = np.array(casc[name]["resid"])
        out["cascade"][name] = dict(bin_acc=[float(b.mean()), float(b.std())],
                                    resid_bal=[float(r.mean()), float(r.std())])
        print(f"  {name}: binary maj-vs-min acc = {b.mean():.3f}±{b.std():.3f} | "
              f"residual 5-way bal acc = {r.mean():.3f}±{r.std():.3f}")

    json.dump(out, open(REPO / "scratchpad/nnls_quality_breakdown.json", "w"), indent=2)
    print("\nsaved scratchpad/nnls_quality_breakdown.json")


if __name__ == "__main__":
    main()
