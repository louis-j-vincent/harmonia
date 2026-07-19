"""Interval-aware root context via a Viterbi HMM over each song's root
sequence.  Multi-seed song-grouped CV.

Emission  = root-head softmax posterior (BP48 feat48_abs, roll-aug MLP).
Transition= learned 12x12 root-to-root motion matrix from TRAIN true roots
            (Laplace).  This is the CORRECT interval-aware context model:
            it encodes the P4/P5/M2 voice-leading the premise check found,
            and propagates information globally (not one-neighbour local).

Baselines compared, all on the SAME per-split test set:
  S0  argmax posterior  (no context)
  S1  local one-neighbour log-linear combine (the prior probe's method)
  S2  Viterbi HMM w/ learned transition (sweep gamma = transition weight)

Reports acc on ALL test chords and on the bottom-25%-confidence LOW-CONF
subset (the population the rescue is meant to help).  min 3 seeds.

Reuses scripts/train_real_audio_final._train_head + _augment_root_by_roll
and the exact RWC BP48 corpus, per project convention (no rebuild).
Read-only on corpus.  Writes a JSON summary.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from harmonia.data.corpus_schema import MatchQuality, filter_by_match, load_corpus
from train_real_audio_final import _train_head, _augment_root_by_roll


def learn_transition(roots, sids, t0, laplace=1.0):
    Tm = np.full((12, 12), laplace)
    for s in sorted(set(sids.tolist())):
        idx = np.where(sids == s)[0]
        idx = idx[np.argsort(t0[idx])]
        r = roots[idx]
        for a, b in zip(r[:-1], r[1:]):
            Tm[a, b] += 1
    Tm /= Tm.sum(1, keepdims=True)
    return Tm


def viterbi(logemit, logtrans):
    """logemit (T,12), logtrans (12,12). Returns best path (T,)."""
    T = logemit.shape[0]
    dp = np.full((T, 12), -np.inf); bp = np.zeros((T, 12), int)
    dp[0] = logemit[0]
    for t in range(1, T):
        # score[j,k] = dp[t-1,j] + logtrans[j,k]
        scores = dp[t-1][:, None] + logtrans
        bp[t] = scores.argmax(0)
        dp[t] = scores.max(0) + logemit[t]
    path = np.zeros(T, int); path[-1] = dp[-1].argmax()
    for t in range(T-1, 0, -1):
        path[t-1] = bp[t, path[t]]
    return path


def _softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def one_split(feat, roots, sids, t0, seed, epochs, lr, batch, device,
              gammas, lams, lowconf_q=0.25):
    songs = sorted(set(sids.tolist()))
    rng = np.random.RandomState(seed); rng.shuffle(songs)
    n_test = max(1, int(round(0.2 * len(songs))))
    test_songs = set(songs[:n_test])
    tr = np.array([s not in test_songs for s in sids]); te = ~tr

    import torch
    Xtr, ytr = _augment_root_by_roll(feat[tr], roots[tr])
    model, mean, std = _train_head(Xtr, ytr, 12, epochs=epochs, lr=lr, batch=batch,
                                   device=device, head_name="root")
    with torch.no_grad():
        Xn = ((feat[te] - mean) / std).astype(np.float32)
        logits = model(torch.tensor(Xn, device=device)).cpu().numpy()
    post = _softmax(logits)
    pred0 = post.argmax(1)
    root_te = roots[te]; sid_te = sids[te]; t0_te = t0[te]
    conf = post.max(1)
    thr = np.quantile(conf, lowconf_q)
    lc = conf <= thr

    Tm = learn_transition(roots[tr], sids[tr], t0[tr])
    logTm = np.log(Tm)
    logpost = np.log(post + 1e-9)

    res = {"n_te": int(te.sum()), "n_lc": int(lc.sum())}
    res["s0_all"] = float((pred0 == root_te).mean())
    res["s0_lc"] = float((pred0[lc] == root_te[lc]).mean())

    # ---- S1 local one-neighbour log-linear combine (prior probe method) ----
    # build neighbour context P(cur|prev) + P(cur|next) using PREDICTED posteriors
    order_idx = {}
    ctx = np.zeros((te.sum(), 12))
    te_local = np.where(te)[0]
    # map global test index -> position in te array
    g2l = {g: l for l, g in enumerate(te_local)}
    for s in sorted(set(sid_te.tolist())):
        loc = np.where(sid_te == s)[0]
        loc = loc[np.argsort(t0_te[loc])]
        for k, li in enumerate(loc):
            cc = np.zeros(12)
            if k > 0:
                cc += post[loc[k-1]] @ Tm
            if k < len(loc)-1:
                cc += post[loc[k+1]] @ Tm.T
            ctx[li] = cc / cc.sum() if cc.sum() > 0 else np.ones(12)/12
    logc = np.log(ctx + 1e-9)
    for lam in lams:
        comb = (logpost + lam * logc).argmax(1)
        res[f"s1_all_l{lam}"] = float((comb == root_te).mean())
        res[f"s1_lc_l{lam}"] = float((comb[lc] == root_te[lc]).mean())

    # ---- S2 Viterbi HMM per song ----
    for gamma in gammas:
        pred2 = pred0.copy()
        for s in sorted(set(sid_te.tolist())):
            loc = np.where(sid_te == s)[0]
            loc = loc[np.argsort(t0_te[loc])]
            path = viterbi(logpost[loc], gamma * logTm)
            pred2[loc] = path
        res[f"s2_all_g{gamma}"] = float((pred2 == root_te).mean())
        res[f"s2_lc_g{gamma}"] = float((pred2[lc] == root_te[lc]).mean())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=REPO / "data/cache/rwc/rwc_bp48_fixed.npz")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, default=REPO / "scratchpad/root_context_hmm_result.json")
    a = ap.parse_args()

    dev = a.device
    if dev is None:
        try:
            import torch; dev = "mps" if torch.backends.mps.is_available() else "cpu"
        except Exception:
            dev = "cpu"

    d = load_corpus(a.corpus)
    keep = filter_by_match(d["match"], minimum=MatchQuality.EXACT)
    feat = d["feat48_abs"][keep].astype(np.float32)
    roots = d["root"][keep].astype(int)
    sids = d["song_id"][keep]; t0 = d["t0"][keep]
    print(f"Corpus {a.corpus.name}: EXACT {keep.sum()} chords, {len(set(sids.tolist()))} songs, dev={dev}")

    gammas = [0.25, 0.5, 1.0, 1.5, 2.0]
    lams = [0.25, 0.5, 1.0]
    runs = []
    for s in range(a.seeds):
        print(f"--- seed {s} ---", flush=True)
        r = one_split(feat, roots, sids, t0, s, a.epochs, a.lr, a.batch, dev, gammas, lams)
        runs.append(r)
        best_g = max(gammas, key=lambda g: r[f"s2_all_g{g}"])
        print(f"  S0 all={r['s0_all']:.3f} lc={r['s0_lc']:.3f} | "
              f"S2(g={best_g}) all={r[f's2_all_g{best_g}']:.3f} lc={r[f's2_lc_g{best_g}']:.3f}",
              flush=True)

    def ms(k):
        v = np.array([r[k] for r in runs if k in r])
        return (float(v.mean()), float(v.std())) if len(v) else (float('nan'), 0.0)

    summary = {k: ms(k) for k in runs[0].keys()}
    a.out.write_text(json.dumps({"summary": summary, "runs": runs, "gammas": gammas,
                                 "lams": lams, "seeds": a.seeds}, indent=2))

    print("\n" + "=" * 72)
    print(f"RWC root-context HMM, {a.seeds} song-grouped seeds")
    print(f"  mean test chords {np.mean([r['n_te'] for r in runs]):.0f}, "
          f"low-conf {np.mean([r['n_lc'] for r in runs]):.0f}")
    def line(lbl, k):
        if k not in runs[0]: return
        m, sd = ms(k); print(f"  {lbl:44s}: {m:.4f} +/- {sd:.4f}")
    print("\n  --- ALL test chords ---")
    line("S0 baseline (no context)", "s0_all")
    for lam in lams: line(f"S1 local-neighbour lam={lam}", f"s1_all_l{lam}")
    for g in gammas: line(f"S2 Viterbi HMM gamma={g}", f"s2_all_g{g}")
    print("\n  --- LOW-CONF subset (bottom 25% conf) ---")
    line("S0 baseline (no context)", "s0_lc")
    for lam in lams: line(f"S1 local-neighbour lam={lam}", f"s1_lc_l{lam}")
    for g in gammas: line(f"S2 Viterbi HMM gamma={g}", f"s2_lc_g{g}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
