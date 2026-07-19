"""Windowed context-MLP for SOUNDING-BASS prediction (2026-07-16).

User proposal (supersedes the REJECTED marginal 12x12 HMM, which was swamped
by the 87.6% root-position majority class):

  Predict the sounding-bass pc of the MIDDLE chord from a symmetric +/-4
  chord-INDEX window of its 8 neighbours (NOT the middle chord itself, NOT a
  time-based window). This is a direct conditional/local model, so it should
  sidestep the majority-class-swamping that killed the marginal HMM.

Neighbour input representation (answers "hard label vs soft vector?"):
  ORACLE   : one-hot(GT sounding-bass pc)         <- teacher-forced upper bound
  PRED-HARD: one-hot(chroma-model argmax bass pc) <- deployable, hard
  PRED-PROB: chroma-model 12-d proba vector       <- deployable, soft
  RAW      : neighbour bass-12 chroma (fabs[:,24:36]) <- deployable, no model

Each of 8 slots = [present-mask(1) | 12-d rep] -> 8*13 = 104 input dims.
Edge (first/last 4 chords of a song): missing slots zero-padded, mask=0.
Reported separately: WINDOWED (all 8 present) vs EDGE (>=1 missing).

Leakage control: chroma-model predictions used as neighbour features and for
fusion are OUT-OF-FOLD on train (inner 5-fold CV over train songs) and
out-of-sample on test (full-train model). Test chords' neighbours are same-song
=> also test => always out-of-sample. ORACLE deliberately uses GT everywhere
(labelled as an unrealistic ceiling).

Fusion of context-MLP (PRED-PROB) with the chroma-only baseline:
  (a) simple probability average
  (b) confidence-weighted average (weight = each model's max-prob)
  (c) small learned fusion MLP on concat(p_chroma, p_ctx) [24->12], trained on
      OOF proba pairs.

Baseline = the established 0.654 chroma-only model: MLP(64,32) on fabs[:,24:36].
5-seed song-grouped 80/20 CV. Reports pooled / root-pos / inversion acc + ECE.
Read-only on corpus npz; writes one JSON.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from harmonia.data.corpus_schema import sounding_bass_pc

RNG_HID = (64, 32)


def ece(conf, correct, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    e, M = 0.0, len(conf)
    if M == 0:
        return float("nan")
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() == 0:
            continue
        e += (m.sum() / M) * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def fit_proba(Xtr, ytr, Xte, seed, hidden=RNG_HID):
    sc = StandardScaler().fit(Xtr)
    clf = MLPClassifier(hidden, max_iter=300, random_state=seed, early_stopping=True)
    clf.fit(sc.transform(Xtr), ytr)
    P = np.zeros((len(Xte), 12))
    P[:, clf.classes_] = clf.predict_proba(sc.transform(Xte))
    return P


def oof_and_test(X, y, song, train_mask, test_mask, seed, inner=5):
    """OOF proba on train rows (inner CV over train songs) + full-train proba on
    test rows. Returns proba for every row (train via OOF, test via full)."""
    P = np.zeros((len(y), 12))
    tr_songs = np.array(sorted(set(song[train_mask].tolist())))
    rng = np.random.RandomState(seed + 777)
    rng.shuffle(tr_songs)
    folds = np.array_split(tr_songs, inner)
    for f in folds:
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
    """rep_vec: (N,12) representation for each row. Build (N, 8*13)."""
    N, S = neigh.shape
    X = np.zeros((N, S * 13), np.float32)
    for sl in range(S):
        j = neigh[:, sl]
        present = j >= 0
        X[present, sl * 13] = 1.0
        X[present, sl * 13 + 1: sl * 13 + 13] = rep_vec[j[present]]
    return X


def subset_metrics(pred, proba, y, masks):
    out = {}
    conf = proba.max(1)
    correct = (pred == y).astype(float)
    for name, m in masks.items():
        if m.sum() == 0:
            out[f"acc_{name}"] = float("nan"); out[f"ece_{name}"] = float("nan"); out[f"n_{name}"] = 0
            continue
        out[f"acc_{name}"] = float(correct[m].mean())
        out[f"ece_{name}"] = ece(conf[m], correct[m])
        out[f"n_{name}"] = int(m.sum())
    return out


def one_seed(seed, fabs, bass_true, inv, song, neigh, n_present):
    N = len(bass_true)
    songs = np.array(sorted(set(song.tolist())))
    rng = np.random.RandomState(seed)
    sh = songs.copy(); rng.shuffle(sh)
    cut = int(0.8 * len(sh)); tr_songs = set(sh[:cut].tolist())
    train_mask = np.array([s in tr_songs for s in song])
    test_mask = ~train_mask

    bass_chroma = fabs[:, 24:36].astype(np.float32)

    # ---- chroma-only baseline w/ OOF (for neighbour features + fusion) ----
    Pc = oof_and_test(bass_chroma, bass_true, song, train_mask, test_mask, seed)
    chroma_pred = Pc.argmax(1)

    # test-subset masks
    te = test_mask
    masks = {
        "all": np.ones(te.sum(), bool),
        "rootpos": ~inv[te],
        "inv": inv[te],
        "windowed": n_present[te] == neigh.shape[1],
        "edge": n_present[te] < neigh.shape[1],
    }
    yte = bass_true[te]

    res = {"seed": seed, "n_test": int(te.sum())}
    res["baseline"] = subset_metrics(chroma_pred[te], Pc[te], yte, masks)

    # ---- neighbour representations ----
    onehot = np.eye(12, dtype=np.float32)
    reps = {
        "oracle": onehot[bass_true],           # GT everywhere (ceiling)
        "predhard": onehot[chroma_pred],       # deployable hard
        "predprob": Pc,                        # deployable soft
        "raw": bass_chroma,                    # deployable raw chroma
    }

    ctx_test_proba = {}
    for rname, rvec in reps.items():
        X = ctx_features(neigh, rvec)
        Pctx = fit_proba(X[train_mask], bass_true[train_mask], X[te], seed)
        ctx_test_proba[rname] = Pctx
        res[f"ctx_{rname}"] = subset_metrics(Pctx.argmax(1), Pctx, yte, masks)

    # ---- fusion: chroma baseline (+) context PRED-PROB ----
    Pchroma_te = Pc[te]
    Pctx_te = ctx_test_proba["predprob"]

    # (a) simple average
    Pavg = 0.5 * (Pchroma_te + Pctx_te)
    res["fuse_avg"] = subset_metrics(Pavg.argmax(1), Pavg, yte, masks)

    # (b) confidence-weighted average
    wc = Pchroma_te.max(1, keepdims=True); wx = Pctx_te.max(1, keepdims=True)
    Pcw = (wc * Pchroma_te + wx * Pctx_te) / (wc + wx)
    res["fuse_confw"] = subset_metrics(Pcw.argmax(1), Pcw, yte, masks)

    # (c) learned fusion MLP on OOF proba pairs
    #   need OOF ctx-predprob on train: inner CV of the context model.
    Xctx_pp = ctx_features(neigh, reps["predprob"])
    Pctx_oof = np.zeros((N, 12))
    tr_songs_arr = np.array(sorted(tr_songs)); rng2 = np.random.RandomState(seed + 999)
    rng2.shuffle(tr_songs_arr)
    for f in np.array_split(tr_songs_arr, 5):
        held = set(f.tolist())
        in_hold = np.array([s in held for s in song]) & train_mask
        in_fit = train_mask & ~in_hold
        if in_hold.sum() == 0:
            continue
        Pctx_oof[in_hold] = fit_proba(Xctx_pp[in_fit], bass_true[in_fit], Xctx_pp[in_hold], seed)
    Ztr = np.concatenate([Pc[train_mask], Pctx_oof[train_mask]], axis=1)
    Zte = np.concatenate([Pchroma_te, Pctx_te], axis=1)
    Pfl = fit_proba(Ztr, bass_true[train_mask], Zte, seed, hidden=(32,))
    res["fuse_learned"] = subset_metrics(Pfl.argmax(1), Pfl, yte, masks)

    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=REPO / "data/cache/rwc/rwc_bp48_fixed.npz")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", type=Path, default=REPO / "scratchpad/bass_context_mlp_result.json")
    a = ap.parse_args()

    d = np.load(a.corpus, allow_pickle=True)
    labels = d["labels"]; root = d["root"].astype(int) % 12
    song = d["song_id"]; fabs = d["feat48_abs"].astype(np.float32); t0 = d["t0"]
    N = len(labels)
    bass_true = np.array([sounding_bass_pc(l, int(root[i])) for i, l in enumerate(labels)])
    assert not any(b is None for b in bass_true), "unexpected None bass target"
    bass_true = bass_true.astype(int)
    inv = np.array(["/" in str(l) for l in labels])

    neigh = build_neighbors(song, t0, N, k=4)
    n_present = (neigh >= 0).sum(1)
    print(f"[corpus] N={N} songs={len(set(song.tolist()))} inv={inv.sum()} "
          f"({100*inv.mean():.1f}%)  full-window rows={(n_present==8).sum()} "
          f"edge rows={(n_present<8).sum()}")

    runs = [one_seed(s, fabs, bass_true, inv, song, neigh, n_present) for s in range(a.seeds)]

    keys = [k for k in runs[0] if isinstance(runs[0][k], dict)]

    def ms(model, metric):
        v = np.array([r[model][metric] for r in runs if metric in r[model]])
        v = v[~np.isnan(v)]
        return (float(v.mean()), float(v.std())) if len(v) else (float("nan"), 0.0)

    summary = {model: {metric: ms(model, metric) for metric in runs[0][model]} for model in keys}
    a.out.write_text(json.dumps({"summary": summary, "runs": runs, "seeds": a.seeds}, indent=2))

    print("\n" + "=" * 90)
    print(f"WINDOWED CONTEXT-MLP for sounding-bass, {a.seeds} song-grouped seeds")
    order = ["baseline", "ctx_oracle", "ctx_predhard", "ctx_predprob", "ctx_raw",
             "fuse_avg", "fuse_confw", "fuse_learned"]
    hdr = f"{'model':14s} | {'all':>13s} {'rootpos':>13s} {'inv':>13s} {'windowed':>13s} {'edge':>13s} | {'ECEall':>7s} {'ECEinv':>7s}"
    print(hdr); print("-" * len(hdr))
    for model in order:
        if model not in summary:
            continue
        def cell(metric):
            m, sd = ms(model, metric); return f"{m:.3f}±{sd:.3f}"
        s = summary[model]
        print(f"{model:14s} | {cell('acc_all'):>13s} {cell('acc_rootpos'):>13s} "
              f"{cell('acc_inv'):>13s} {cell('acc_windowed'):>13s} {cell('acc_edge'):>13s} | "
              f"{ms(model,'ece_all')[0]:>7.3f} {ms(model,'ece_inv')[0]:>7.3f}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
