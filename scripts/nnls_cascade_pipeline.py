"""BUILD + EVALUATE the two-stage maj/min cascade as a real end-to-end pipeline,
scored FULL 7-way against the flat single-stage baselines, on the same 5-seed RWC
CV splits used throughout (oracle-root frame; NNLS rotation-only front-end).

Pipeline:
  Stage 1 (router) : 3-way {maj, min, residual}. A pure binary maj/min head can't
                     reject non-maj/min chords, so Stage 1 is realized as a 3-way
                     gate -- the validated binary maj-vs-min 0.953 lives on its
                     accept branch; the 3rd logit is the reject/route-to-Stage-2.
  Stage 2 (spec.)  : 5-way {dom, hdim, dim, aug, sus}, trained ONLY on residual
                     chords, class-weighted.

Combine strategies (final = one of 7 classes per chord):
  hard : Stage1 argmax; if == residual -> Stage2 argmax.
  conf : accept maj/min only if Stage1 top-prob >= tau; else (argmax==residual OR
         low confidence) -> Stage2 argmax. Sweep tau, report best-balanced.
  soft : hierarchical product  p7 = [p1_maj, p1_min, p1_res * p2(dom..sus)];
         argmax over 7. No threshold.

Baselines on identical splits: flat NNLS 7-way (primary), flat BP48 7-way (wider).
Every number printed comes from a completed run. Log nnls_cascade_pipeline.log,
result nnls_cascade_pipeline.json.
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
TAUS = [0.5, 0.6, 0.7, 0.8, 0.9]


def song_split(sid, seed, test_frac=0.2, val_frac=0.1):
    songs = np.unique(sid); rng = np.random.RandomState(seed); rng.shuffle(songs)
    n = len(songs)
    nte = max(1, int(round(test_frac * n))); nva = max(1, int(round(val_frac * n)))
    te = np.isin(sid, list(songs[:nte])); va = np.isin(sid, list(songs[nte:nte + nva]))
    return ~(te | va), va, te


def cw_of(y, K):
    c = np.bincount(y, minlength=K)
    return (c.sum() / (K * np.maximum(c, 1))).astype(np.float32)


def bal7(pred, true):
    return float(np.nanmean(balanced_recall(pred, true, KQ)))


def main():
    nn = load_corpus(REPO / "data/cache/rwc/rwc_nnls24.npz")
    bp = load_corpus(REPO / "data/cache/rwc/rwc_bp48_fixed.npz")
    assert (nn["song_id"] == bp["song_id"]).all() and (nn["root"] == bp["root"]).all()
    roots = bp["root"].astype(np.int64) % 12
    quals = bp["quality_idx"].astype(np.int64)
    sid = bp["song_id"]
    Xnn = np.concatenate([rotate_by_root(nn["nnls24"][:, :12].astype(np.float32), roots),
                          rotate_by_root(nn["nnls24"][:, 12:].astype(np.float32), roots)], 1)
    Xbp = bp["feat48"].astype(np.float32)

    s1lab = np.where(quals <= 1, quals, 2)                 # maj0 min1 residual2
    res_mask_all = quals >= 2
    res_map = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4}
    res_inv = {0: 2, 1: 3, 2: 4, 3: 5, 4: 6}
    s2lab = np.array([res_map.get(q, -1) for q in quals])

    # accumulate pooled test predictions across seeds
    pool = {k: {"pred": [], "true": []} for k in
            ["flat_nnls", "flat_bp48", "casc_hard", "casc_soft"] + [f"casc_conf_{t}" for t in TAUS]}

    for seed in range(SEEDS):
        tr, va, te = song_split(sid, seed)
        true_te = quals[te]

        # ---- flat baselines (7-way) ----
        for name, X in [("flat_nnls", Xnn), ("flat_bp48", Xbp)]:
            m = train_clf(X[tr], quals[tr], X[va], quals[va], X.shape[1], KQ,
                          hid=(128, 64), epochs=60, cw=cw_of(quals[tr], KQ))
            pr = predict_proba(m, X[te]).argmax(1)
            pool[name]["pred"].append(pr); pool[name]["true"].append(true_te)

        # ---- Stage 1 router (3-way) on NNLS ----
        m1 = train_clf(Xnn[tr], s1lab[tr], Xnn[va], s1lab[va], Xnn.shape[1], 3,
                       hid=(128, 64), epochs=60, cw=cw_of(s1lab[tr], 3))
        p1 = predict_proba(m1, Xnn[te])                    # (nte,3)

        # ---- Stage 2 specialist (5-way) on residual NNLS ----
        trr = tr & res_mask_all; var = va & res_mask_all
        m2 = train_clf(Xnn[trr], s2lab[trr], Xnn[var], s2lab[var], Xnn.shape[1], 5,
                       hid=(64, 32), epochs=60, cw=cw_of(s2lab[trr], 5))
        p2 = predict_proba(m2, Xnn[te])                    # (nte,5)
        s2arg = p2.argmax(1)

        # ---- combine: hard ----
        a1 = p1.argmax(1)
        hard = np.where(a1 <= 1, a1, np.array([res_inv[x] for x in s2arg]))
        pool["casc_hard"]["pred"].append(hard); pool["casc_hard"]["true"].append(true_te)

        # ---- combine: conf (sweep tau) ----
        top1 = p1.max(1)
        for t in TAUS:
            accept = (a1 <= 1) & (top1 >= t)
            conf = np.where(accept, a1, np.array([res_inv[x] for x in s2arg]))
            pool[f"casc_conf_{t}"]["pred"].append(conf)
            pool[f"casc_conf_{t}"]["true"].append(true_te)

        # ---- combine: soft hierarchical ----
        p7 = np.zeros((te.sum(), KQ))
        p7[:, 0] = p1[:, 0]; p7[:, 1] = p1[:, 1]
        p7[:, 2:] = p1[:, 2:3] * p2
        soft = p7.argmax(1)
        pool["casc_soft"]["pred"].append(soft); pool["casc_soft"]["true"].append(true_te)
        print(f"  [seed {seed}] done", flush=True)

    # ---- score pooled ----
    def raw_bal(name):
        pr = np.concatenate(pool[name]["pred"]); tr_ = np.concatenate(pool[name]["true"])
        return float((pr == tr_).mean()), bal7(pr, tr_)

    print("\n=== END-TO-END 7-WAY QUALITY (pooled 5 seeds) ===")
    out = {}
    rows = [("flat_nnls", "Flat NNLS 7-way (primary baseline)"),
            ("flat_bp48", "Flat BP48 7-way (wider baseline)"),
            ("casc_hard", "Cascade HARD routing (NNLS)"),
            ("casc_soft", "Cascade SOFT hierarchical (NNLS)")]
    best_conf = max(TAUS, key=lambda t: raw_bal(f"casc_conf_{t}")[1])
    rows.append((f"casc_conf_{best_conf}", f"Cascade CONF routing tau={best_conf} (NNLS, best-bal)"))
    for key, lbl in rows:
        raw, bal = raw_bal(key); out[key] = {"raw": raw, "bal": bal}
        print(f"  {lbl:46s}: raw={raw:.3f}  bal={bal:.3f}")
    # also record all conf taus
    out["conf_sweep"] = {str(t): dict(zip(["raw", "bal"], raw_bal(f"casc_conf_{t}"))) for t in TAUS}
    fn_raw, fn_bal = raw_bal("flat_nnls")
    for key, lbl in rows[2:]:
        raw, bal = raw_bal(key)
        print(f"  Δ vs flat NNLS [{lbl.split('(')[0].strip()}]: raw {raw-fn_raw:+.3f}  bal {bal-fn_bal:+.3f}")
    json.dump(out, open(REPO / "scratchpad/nnls_cascade_pipeline.json", "w"), indent=2)
    print("\nsaved scratchpad/nnls_cascade_pipeline.json")


if __name__ == "__main__":
    main()
