"""LLM-correction experiment — STAGE 1: base (deployable) quality predictions.

Reuses the CONFIRMED deployable recipe (matches scratchpad/rwc_structured.py's
cascade path, i.e. the ~52% balanced / ~52% dom baseline) — does NOT invent a
new/stronger base classifier. One song-held-out split (fixed seed). Produces,
for every held-out segment: GT root/quality, predicted root, predicted quality
(top-k root-marginalised), plus per-song key/mode (Krumhansl-Schmuckler).

Writes: scratchpad/llm_correction_base.npz + a compact per-song JSON the LLM
correction stage consumes. Read-only on the shared corpus.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))

from harmonia.data.corpus_schema import MatchQuality, filter_by_match, load_corpus
from harmonia.theory.key_profiles import infer_key
from train_real_audio_final import _train_head, QUALITIES

DOM = QUALITIES.index("dom")
PC_NAMES = ["C","Db","D","Eb","E","F","Gb","G","Ab","A","Bb","B"]
TEST_FRAC = 0.20
SEED = 0


def _softmax(model, Xn, device):
    import torch
    with torch.no_grad():
        return torch.softmax(model(torch.tensor(Xn.astype(np.float32), device=device)), 1).cpu().numpy()

def _block_roll(f48, shift):
    N = f48.shape[0]; r = f48.reshape(N, 4, 12)
    if np.isscalar(shift):
        out = np.roll(r, shift, 2)
    else:
        out = np.stack([np.roll(r[i], int(shift[i]), 2) for i in range(N)])
    return out.reshape(N, 48)

def _fold12(f48):
    v = f48.reshape(-1, 4, 12).mean(1)
    return (v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)).astype(np.float32)

def _roll_aug(X, target, bw=12, n_shifts=12):
    n, d = X.shape; nb = d // bw
    Xs, ys = [X], [target]
    for k in range(1, n_shifts):
        Xs.append(np.roll(X.reshape(n, nb, bw), k, 2).reshape(n, d))
        ys.append((target + k) % 12)
    return np.concatenate(Xs), np.concatenate(ys)

def _bal_dom(preds, y, n=7):
    rec = {c: (float((preds[y == c] == c).mean()) if (y == c).sum() else 0.0) for c in range(n)}
    return float(np.mean([rec[c] for c in range(n)])), rec[DOM], rec


def song_keys(d):
    sid = d["song_id"]; fold = _fold12(d["feat48_abs"])
    out = {}
    for s in sorted(set(sid.tolist())):
        m = sid == s
        kp = infer_key(fold[m].sum(0))
        out[s] = (kp.tonic, kp.mode)
    return out


def main():
    import torch
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    d = dict(load_corpus(REPO / "data/cache/rwc/rwc_bp48.npz"))
    keep = filter_by_match(d["match"], minimum=MatchQuality.EXACT)
    f48 = d["feat48"][keep]; fa = d["feat48_abs"][keep]
    qidx = d["quality_idx"].astype(int)[keep]
    roots = d["root"].astype(int)[keep]; sid = d["song_id"][keep]
    t0 = d["t0"][keep]; t1 = d["t1"][keep]
    keymap = song_keys(d)

    songs = sorted(set(sid.tolist()))
    rng = np.random.RandomState(SEED); rng.shuffle(songs)
    n_test = int(round(TEST_FRAC * len(songs)))
    test_songs = set(songs[:n_test])
    tr = np.array([s not in test_songs for s in sid]); te = ~tr
    print(f"split: {len(songs)-n_test} train / {n_test} test songs; "
          f"{tr.sum()} train / {te.sum()} test segments", flush=True)

    # root head (absolute + roll aug) -> deployable predicted root
    Xa, ya = _roll_aug(fa[tr], roots[tr])
    rm, rmean, rstd = _train_head(Xa, ya, 12, epochs=60, lr=3e-4, batch=64,
                                  device=dev, head_name="root")
    root_probs = _softmax(rm, (fa[te] - rmean) / rstd, dev)
    pred_root = root_probs.argmax(1)
    print(f"root acc (test): {(pred_root==roots[te]).mean():.3f}", flush=True)

    # quality head (oracle root-relative training frame) = project's current-best
    # quality model (#31). Predictions are oracle-root-relative on the test set —
    # this is the CONFIRMED STRONG baseline (~52% bal), NOT the eroded predicted-
    # root cascade. Isolates the quality question and gives the LLM true roots.
    qm, qmean, qstd = _train_head(f48[tr], qidx[tr], 7, epochs=60, lr=3e-4, batch=64,
                                  device=dev, head_name="qual")
    q_probs = _softmax(qm, (f48[te] - qmean) / qstd, dev)   # f48 = oracle root-rel
    pred_q = q_probs.argmax(1)

    # deployable cascade (predicted root) — reported for context only
    per_root = np.stack([_softmax(qm, (_block_roll(fa[te], -r) - qmean) / qstd, dev)
                         for r in range(12)])
    topk = np.argsort(-root_probs, 1)[:, :3]
    w = np.take_along_axis(root_probs, topk, 1); w /= w.sum(1, keepdims=True)
    qmix = np.zeros((te.sum(), 7), np.float32)
    for j in range(3):
        qmix += w[:, j:j+1] * per_root[topk[:, j], np.arange(te.sum())]

    yq = qidx[te]
    bal, dom, rec = _bal_dom(pred_q, yq)
    cbal, cdom, _ = _bal_dom(qmix.argmax(1), yq)
    print(f"\nBASE (oracle-root-relative quality head = confirmed strong baseline):")
    print(f"  balanced acc {bal:.3f}   dom recall {dom:.3f}   raw acc {(pred_q==yq).mean():.3f}")
    print("  per-class recall:", {QUALITIES[c]: round(rec[c],3) for c in range(7)})
    print(f"  [context] deployable predicted-root cascade: bal {cbal:.3f} dom {cdom:.3f}")

    # persist for the LLM stage
    te_sid = sid[te]
    np.savez(REPO / "scratchpad/llm_correction_base.npz",
             pred_root=pred_root, pred_q=pred_q, qmix=qmix,
             gt_root=roots[te], gt_q=yq, song_id=te_sid, t0=t0[te], t1=t1[te])

    # compact per-song JSON for the LLM. Roots shown = TRUE roots (the oracle-rel
    # quality head assumes correct root; the prompt tells the LLM root is fixed).
    gr = roots[te]
    songs_json = {}
    for s in sorted(set(te_sid.tolist())):
        m = te_sid == s
        order = np.argsort(t0[te][m])
        pr = gr[m][order]; pq = pred_q[m][order]
        seq = [{"i": int(i), "root": PC_NAMES[int(pr[i])], "q": QUALITIES[int(pq[i])]}
               for i in range(len(order))]
        tonic, mode = keymap[s]
        songs_json[s] = {"key": PC_NAMES[tonic], "mode": mode, "seq": seq}
    (REPO / "scratchpad/llm_correction_songs.json").write_text(json.dumps(songs_json))
    print(f"\nwrote {len(songs_json)} test songs to llm_correction_songs.json")


if __name__ == "__main__":
    main()
