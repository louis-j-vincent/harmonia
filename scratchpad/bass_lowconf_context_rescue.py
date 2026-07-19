"""Confidence-GATED context rescue for sounding-bass (2026-07-16).

Re-poses the REJECTED windowed context-MLP test. The prior test evaluated
context/fusion on POOLED accuracy and concluded it doesn't beat the 0.654
chroma-only baseline. User clarifies the real question: context was never meant
to beat chroma OVERALL, only to help on the subset where the chroma-only model
is ALREADY UNSURE (per its own calibrated confidence). This is a different test.

Pipeline (reuses machinery from bass_context_mlp_cv.py + bass_argmax_anchor_confidence.py):

  1. Chroma-only bass-12 MLP(64,32) on feat48_abs[:,24:36]. Its own calibrated
     confidence (temperature-scaled max-softmax; T fit on held-out train slice).
     LOW-CONFIDENCE SUBSET = bottom {20,30,40}% of TEST rows by chroma confidence.
     (Percentile cuts are rank-based => temperature-invariant; we also print the
      calibrated confidence VALUE at each cut.)

  2. On exactly that subset, compare (REALISTIC predicted neighbours, not oracle):
       - chroma-alone acc  (the number being rescued FROM)
       - context-alone acc (context-MLP, neighbours only, NO chroma input)
       - gated system:  conf<thr -> use context/fusion ; conf>=thr -> chroma
         reported as NET acc on FULL test corpus AND acc on the low-conf subset.
     Plus the ORACLE-neighbour ceiling of context-alone on the same subset.

  3. Root-position vs inversion split WITHIN the low-conf subset, and the overlap
     check: is the low-conf subset mostly inversions?

Leakage control identical to bass_context_mlp_cv.py: chroma preds used as
neighbour features / fusion inputs are OOF on train, out-of-sample on test.
5-seed song-grouped 80/20 CV.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from harmonia.data.corpus_schema import sounding_bass_pc

RNG_HID = (64, 32)
PCTS = [0.20, 0.30, 0.40]


def fit_proba(Xtr, ytr, Xte, seed, hidden=RNG_HID):
    sc = StandardScaler().fit(Xtr)
    clf = MLPClassifier(hidden, max_iter=300, random_state=seed, early_stopping=True)
    clf.fit(sc.transform(Xtr), ytr)
    P = np.zeros((len(Xte), 12))
    P[:, clf.classes_] = clf.predict_proba(sc.transform(Xte))
    return P


def softmax_T(logits, T):
    z = logits / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_temperature(proba, y_idx):
    logits = np.log(np.clip(proba, 1e-8, 1))
    n = len(y_idx)
    def nll(T):
        p = softmax_T(logits, T)
        return -np.log(np.clip(p[np.arange(n), y_idx], 1e-12, 1)).mean()
    r = minimize_scalar(nll, bounds=(0.3, 5.0), method="bounded")
    return r.x


def oof_and_test(X, y, song, train_mask, test_mask, seed, inner=5):
    P = np.zeros((len(y), 12))
    tr_songs = np.array(sorted(set(song[train_mask].tolist())))
    rng = np.random.RandomState(seed + 777)
    rng.shuffle(tr_songs)
    for f in np.array_split(tr_songs, inner):
        held = set(f.tolist())
        in_hold = np.array([s in held for s in song]) & train_mask
        in_fit = train_mask & ~in_hold
        if in_hold.sum() == 0:
            continue
        P[in_hold] = fit_proba(X[in_fit], y[in_fit], X[in_hold], seed)
    P[test_mask] = fit_proba(X[train_mask], y[train_mask], X[test_mask], seed)
    return P


def build_neighbors(song, t0, N, k=4):
    neigh = np.full((N, 2 * k), -1, int)
    for s in sorted(set(song.tolist())):
        idx = np.where(song == s)[0]
        idx = idx[np.argsort(t0[idx])]
        for pos, gi in enumerate(idx):
            slots = [pos + off for off in range(-k, 0)] + [pos + off for off in range(1, k + 1)]
            for sl, p in enumerate(slots):
                if 0 <= p < len(idx):
                    neigh[gi, sl] = idx[p]
    return neigh


def ctx_features(neigh, rep_vec):
    N, S = neigh.shape
    X = np.zeros((N, S * 13), np.float32)
    for sl in range(S):
        j = neigh[:, sl]
        present = j >= 0
        X[present, sl * 13] = 1.0
        X[present, sl * 13 + 1: sl * 13 + 13] = rep_vec[j[present]]
    return X


def acc(pred, y, m):
    return float((pred[m] == y[m]).mean()) if m.sum() else float("nan")


def one_seed(seed, fabs, bass_true, inv, song, neigh):
    N = len(bass_true)
    songs = np.array(sorted(set(song.tolist())))
    rng = np.random.RandomState(seed)
    sh = songs.copy(); rng.shuffle(sh)
    cut = int(0.8 * len(sh)); tr_songs = set(sh[:cut].tolist())
    train_mask = np.array([s in tr_songs for s in song])
    test_mask = ~train_mask
    te = test_mask
    yte = bass_true[te]
    inv_te = inv[te]

    bass_chroma = fabs[:, 24:36].astype(np.float32)

    # ---- chroma-only baseline (OOF on train, full-train on test) ----
    Pc = oof_and_test(bass_chroma, bass_true, song, train_mask, test_mask, seed)
    chroma_pred = Pc.argmax(1)

    # temperature: fit on held-out slice of train songs (last 20% of train songs)
    cal_songs = set(sh[int(0.6 * len(sh)):cut].tolist())
    calm = np.array([s in cal_songs for s in song])
    T = fit_temperature(Pc[calm], bass_true[calm])
    conf_te = softmax_T(np.log(np.clip(Pc[te], 1e-8, 1)), T).max(1)

    # ---- context-MLP: realistic (predprob) + oracle neighbour reps ----
    onehot = np.eye(12, dtype=np.float32)
    Xctx_real = ctx_features(neigh, Pc)          # deployable soft neighbours
    Xctx_orac = ctx_features(neigh, onehot[bass_true])  # ceiling
    Pctx_real = fit_proba(Xctx_real[train_mask], bass_true[train_mask], Xctx_real[te], seed)
    Pctx_orac = fit_proba(Xctx_orac[train_mask], bass_true[train_mask], Xctx_orac[te], seed)
    ctx_real_pred = Pctx_real.argmax(1)
    ctx_orac_pred = Pctx_orac.argmax(1)

    # ---- learned fusion (realistic): MLP on concat(p_chroma, p_ctx) ----
    Pctx_oof = np.zeros((N, 12))
    tr_arr = np.array(sorted(tr_songs)); rng2 = np.random.RandomState(seed + 999)
    rng2.shuffle(tr_arr)
    for f in np.array_split(tr_arr, 5):
        held = set(f.tolist())
        in_hold = np.array([s in held for s in song]) & train_mask
        in_fit = train_mask & ~in_hold
        if in_hold.sum() == 0:
            continue
        Pctx_oof[in_hold] = fit_proba(Xctx_real[in_fit], bass_true[in_fit], Xctx_real[in_hold], seed)
    Ztr = np.concatenate([Pc[train_mask], Pctx_oof[train_mask]], axis=1)
    Zte = np.concatenate([Pc[te], Pctx_real], axis=1)
    Pfuse = fit_proba(Ztr, bass_true[train_mask], Zte, seed, hidden=(32,))
    fuse_pred = Pfuse.argmax(1)

    chroma_pred_te = chroma_pred[te]

    out = {"seed": seed, "n_test": int(te.sum()), "T": float(T),
           "inv_rate_all": float(inv_te.mean())}

    for p in PCTS:
        thr = np.quantile(conf_te, p)
        low = conf_te <= thr            # low-confidence subset
        high = ~low
        key = f"p{int(p*100)}"

        # gated systems: below thr use context/fusion, else chroma
        gated_ctx = np.where(low, ctx_real_pred, chroma_pred_te)
        gated_fuse = np.where(low, fuse_pred, chroma_pred_te)

        allm = np.ones(len(yte), bool)
        rp_low = low & ~inv_te
        iv_low = low & inv_te

        out[key] = {
            "thr_conf": float(thr),
            "n_low": int(low.sum()),
            "inv_rate_low": float(inv_te[low].mean()) if low.sum() else float("nan"),
            "frac_low_that_are_inv": float(inv_te[low].mean()) if low.sum() else float("nan"),
            # --- accuracy ON the low-conf subset ---
            "sub_chroma": acc(chroma_pred_te, yte, low),
            "sub_ctx_real": acc(ctx_real_pred, yte, low),
            "sub_ctx_oracle": acc(ctx_orac_pred, yte, low),
            "sub_fuse": acc(fuse_pred, yte, low),
            # root/inv split within subset
            "sub_chroma_rp": acc(chroma_pred_te, yte, rp_low),
            "sub_chroma_iv": acc(chroma_pred_te, yte, iv_low),
            "sub_ctx_real_rp": acc(ctx_real_pred, yte, rp_low),
            "sub_ctx_real_iv": acc(ctx_real_pred, yte, iv_low),
            "sub_ctx_oracle_rp": acc(ctx_orac_pred, yte, rp_low),
            "sub_ctx_oracle_iv": acc(ctx_orac_pred, yte, iv_low),
            "sub_fuse_rp": acc(fuse_pred, yte, rp_low),
            "sub_fuse_iv": acc(fuse_pred, yte, iv_low),
            "n_rp_low": int(rp_low.sum()), "n_iv_low": int(iv_low.sum()),
            # --- NET accuracy on FULL test corpus ---
            "net_chroma": acc(chroma_pred_te, yte, allm),
            "net_gated_ctx": acc(gated_ctx, yte, allm),
            "net_gated_fuse": acc(gated_fuse, yte, allm),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=REPO / "data/cache/rwc/rwc_bp48_fixed.npz")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", type=Path, default=REPO / "scratchpad/bass_lowconf_context_rescue.json")
    a = ap.parse_args()

    d = np.load(a.corpus, allow_pickle=True)
    labels = d["labels"]; root = d["root"].astype(int) % 12
    song = d["song_id"]; fabs = d["feat48_abs"].astype(np.float32); t0 = d["t0"]
    N = len(labels)
    bass_true = np.array([sounding_bass_pc(l, int(root[i])) for i, l in enumerate(labels)]).astype(int)
    inv = np.array(["/" in str(l) for l in labels])

    neigh = build_neighbors(song, t0, N, k=4)
    print(f"[corpus] N={N} songs={len(set(song.tolist()))} inv={inv.sum()} ({100*inv.mean():.1f}%)")

    runs = [one_seed(s, fabs, bass_true, inv, song, neigh) for s in range(a.seeds)]
    a.out.write_text(json.dumps(runs, indent=2))

    def agg(key, field):
        v = np.array([r[key][field] for r in runs], float)
        v = v[~np.isnan(v)]
        return (float(v.mean()), float(v.std())) if len(v) else (float("nan"), 0.0)

    print(f"\n{'='*94}\nCONFIDENCE-GATED CONTEXT RESCUE  ({a.seeds} song-grouped seeds)\n{'='*94}")
    for p in PCTS:
        key = f"p{int(p*100)}"
        print(f"\n--- LOW-CONF SUBSET = bottom {int(p*100)}% by chroma confidence "
              f"(conf<= {agg(key,'thr_conf')[0]:.3f}, n_low={agg(key,'n_low')[0]:.0f}/seed) ---")
        print(f"  inversion rate:  overall test = {np.mean([r['inv_rate_all'] for r in runs]):.3f}   "
              f"within low-conf subset = {agg(key,'inv_rate_low')[0]:.3f}")
        print(f"  {'ON LOW-CONF SUBSET':28s} {'acc':>13s} {'root-pos':>13s} {'inversion':>13s}")
        rows = [("chroma-alone (rescue FROM)", "sub_chroma", "sub_chroma_rp", "sub_chroma_iv"),
                ("context-alone (realistic)", "sub_ctx_real", "sub_ctx_real_rp", "sub_ctx_real_iv"),
                ("context-alone (ORACLE ceil)", "sub_ctx_oracle", "sub_ctx_oracle_rp", "sub_ctx_oracle_iv"),
                ("learned fusion (realistic)", "sub_fuse", "sub_fuse_rp", "sub_fuse_iv")]
        for name, fa, frp, fiv in rows:
            m, s = agg(key, fa); mrp, _ = agg(key, frp); miv, _ = agg(key, fiv)
            print(f"  {name:28s} {m:.3f}±{s:.3f} {mrp:>13.3f} {miv:>13.3f}")
        print(f"  {'NET on FULL corpus':28s} {'acc':>13s}")
        for name, fa in [("chroma-only (baseline)", "net_chroma"),
                         ("gated: ctx if low-conf", "net_gated_ctx"),
                         ("gated: fusion if low-conf", "net_gated_fuse")]:
            m, s = agg(key, fa)
            print(f"  {name:28s} {m:.3f}±{s:.3f}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
