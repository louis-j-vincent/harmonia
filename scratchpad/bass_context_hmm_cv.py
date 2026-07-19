"""Interval-aware context on the NEW target = SOUNDING BASS pitch class.

Emission  = a per-chord 12-way bass-pc head (feat48_abs, roll-aug MLP) --
            same architecture/pipeline as the root head, target = absolute
            sounding bass pc (derive_bass_target).
Transition= learned 12x12 absolute bass-pc motion matrix from TRAIN bass pcs
            (Laplace).  Encodes the chromatic/stepwise bass voice-leading the
            premise check found (esp. on inversions).
Decode    = per-song Viterbi (sweep gamma).  Compared vs S0 argmax and the
            S1 local one-neighbour combine.

ALL absolute-PC / probability space: emission is a distribution over absolute
bass pcs, Tm is over absolute (prev_bass, cur_bass).  No chroma rotation, no
shift-back -- the degenerate collapse-to-C artifact cannot occur.

Multi-seed song-grouped CV.  Reports acc on all + inversions + low-conf.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from harmonia.data.corpus_schema import MatchQuality, filter_by_match, load_corpus
from train_real_audio_final import _train_head, _augment_root_by_roll
from bass_simple_cv import derive_bass_target


def learn_transition(seq, sids, t0, laplace=1.0):
    Tm = np.full((12, 12), laplace)
    for s in sorted(set(sids.tolist())):
        idx = np.where(sids == s)[0]; idx = idx[np.argsort(t0[idx])]
        r = seq[idx]
        for a, b in zip(r[:-1], r[1:]): Tm[a, b] += 1
    Tm /= Tm.sum(1, keepdims=True)
    return Tm


def viterbi(logemit, logtrans):
    T = logemit.shape[0]
    dp = np.full((T, 12), -np.inf); bp = np.zeros((T, 12), int)
    dp[0] = logemit[0]
    for t in range(1, T):
        sc = dp[t-1][:, None] + logtrans
        bp[t] = sc.argmax(0); dp[t] = sc.max(0) + logemit[t]
    path = np.zeros(T, int); path[-1] = dp[-1].argmax()
    for t in range(T-1, 0, -1): path[t-1] = bp[t, path[t]]
    return path


def _softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def one_split(feat, bass, is_inv, sids, t0, seed, epochs, lr, batch, device, gammas, lams):
    import torch
    songs = sorted(set(sids.tolist()))
    rng = np.random.RandomState(seed); rng.shuffle(songs)
    n_test = max(1, int(round(0.2 * len(songs))))
    test = set(songs[:n_test])
    tr = np.array([s not in test for s in sids]); te = ~tr

    # roll-aug for absolute bass pc (global transpose preserves bass class + k)
    Xtr, ytr = _augment_root_by_roll(feat[tr], bass[tr])
    model, mean, std = _train_head(Xtr, ytr, 12, epochs=epochs, lr=lr, batch=batch,
                                   device=device, head_name="bass")
    with torch.no_grad():
        Xn = ((feat[te] - mean) / std).astype(np.float32)
        logits = model(torch.tensor(Xn, device=device)).cpu().numpy()
    post = _softmax(logits); pred0 = post.argmax(1)
    b_te = bass[te]; sid_te = sids[te]; t0_te = t0[te]; inv_te = is_inv[te]
    conf = post.max(1); thr = np.quantile(conf, 0.25); lc = conf <= thr
    is_iv = inv_te == 1

    Tm = learn_transition(bass[tr], sids[tr], t0[tr]); logTm = np.log(Tm)
    logpost = np.log(post + 1e-9)

    def a(pred, m=None): return float((pred[m] == b_te[m]).mean()) if m is not None else float((pred == b_te).mean())
    res = {"n_te": int(te.sum()), "n_lc": int(lc.sum()), "n_inv": int(is_iv.sum())}
    res["s0_all"] = a(pred0); res["s0_lc"] = a(pred0, lc); res["s0_inv"] = a(pred0, is_iv)

    # S1 local neighbour
    ctx = np.zeros((te.sum(), 12))
    for s in sorted(set(sid_te.tolist())):
        loc = np.where(sid_te == s)[0]; loc = loc[np.argsort(t0_te[loc])]
        for k, li in enumerate(loc):
            cc = np.zeros(12)
            if k > 0: cc += post[loc[k-1]] @ Tm
            if k < len(loc)-1: cc += post[loc[k+1]] @ Tm.T
            ctx[li] = cc / cc.sum() if cc.sum() > 0 else np.ones(12)/12
    logc = np.log(ctx + 1e-9)
    for lam in lams:
        comb = (logpost + lam * logc).argmax(1)
        res[f"s1_all_l{lam}"] = a(comb); res[f"s1_lc_l{lam}"] = a(comb, lc); res[f"s1_inv_l{lam}"] = a(comb, is_iv)

    # S2 Viterbi
    for g in gammas:
        pred2 = pred0.copy()
        for s in sorted(set(sid_te.tolist())):
            loc = np.where(sid_te == s)[0]; loc = loc[np.argsort(t0_te[loc])]
            pred2[loc] = viterbi(logpost[loc], g * logTm)
        res[f"s2_all_g{g}"] = a(pred2); res[f"s2_lc_g{g}"] = a(pred2, lc); res[f"s2_inv_g{g}"] = a(pred2, is_iv)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=REPO / "data/cache/rwc/rwc_bp48_fixed.npz")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, default=REPO / "scratchpad/bass_context_hmm_result.json")
    a = ap.parse_args()
    dev = a.device
    if dev is None:
        try:
            import torch; dev = "mps" if torch.backends.mps.is_available() else "cpu"
        except Exception: dev = "cpu"

    d = load_corpus(a.corpus)
    keep = filter_by_match(d["match"], minimum=MatchQuality.EXACT)
    feat = d["feat48_abs"][keep].astype(np.float32)
    root = d["root"][keep].astype(int); labels = d["labels"][keep]
    sids = d["song_id"][keep]; t0 = d["t0"][keep]
    bass = np.zeros(len(labels), int); is_inv = np.zeros(len(labels), int)
    for i, lab in enumerate(labels):
        iv, b = derive_bass_target(lab, root[i]); is_inv[i] = iv; bass[i] = b
    print(f"Corpus: EXACT {keep.sum()} chords, inv {is_inv.sum()} ({100*is_inv.mean():.1f}%), dev={dev}")

    gammas = [0.25, 0.5, 1.0, 1.5]; lams = [0.25, 0.5, 1.0]
    runs = []
    for s in range(a.seeds):
        print(f"--- seed {s} ---", flush=True)
        r = one_split(feat, bass, is_inv, sids, t0, s, a.epochs, a.lr, a.batch, dev, gammas, lams)
        runs.append(r)
        bg = max(gammas, key=lambda g: r[f"s2_all_g{g}"])
        print(f"  S0 all={r['s0_all']:.3f} inv={r['s0_inv']:.3f} lc={r['s0_lc']:.3f} | "
              f"S2(g{bg}) all={r[f's2_all_g{bg}']:.3f} inv={r[f's2_inv_g{bg}']:.3f} lc={r[f's2_lc_g{bg}']:.3f}", flush=True)

    def ms(k):
        v = np.array([r[k] for r in runs if k in r]); return (float(v.mean()), float(v.std())) if len(v) else (float('nan'),0.0)
    a.out.write_text(json.dumps({"summary": {k: ms(k) for k in runs[0]}, "runs": runs,
                                 "gammas": gammas, "lams": lams}, indent=2))
    print("\n" + "="*72 + f"\nBASS-PC context HMM, {a.seeds} seeds (target=SOUNDING BASS)")
    def line(l,k):
        if k in runs[0]: m,sd=ms(k); print(f"  {l:40s}: {m:.4f} +/- {sd:.4f}")
    for grp, suf in [("ALL","all"),("INVERSIONS","inv"),("LOW-CONF","lc")]:
        print(f"\n  --- {grp} ---")
        line("S0 baseline", f"s0_{suf}")
        for lam in lams: line(f"S1 local lam={lam}", f"s1_{suf}_l{lam}")
        for g in gammas: line(f"S2 Viterbi g={g}", f"s2_{suf}_g{g}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
