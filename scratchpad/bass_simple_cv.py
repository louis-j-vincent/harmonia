"""SIMPLE unconditional bass-PC head on RWC BP48 (2026-07-16).

Supersedes the pooled-gated (v1) and temporal-GRU (v2) attempts. Per user:
the bass model is "bete et mechant" — predict the ACTUAL SOUNDING bass pc
UNCONDITIONALLY (root-position C -> C; inversion A/E -> E). No inversion gate.

Features: 9 POOLED feat48_abs vectors = current chord + 4 before + 4 after
(within song, zero-padded at boundaries) -> 9*48 = 432 dims. NOT frame-level.

Variants:
  (a) RAW      : absolute chroma; target = absolute bass pc (12-way).
                 roll-augmented (global transpose is label-preserving).
  (b) RENORM   : the literal "anchor rotation to CURRENT chord's bass" makes
                 the current-chord target trivially 0 (degenerate for a bass
                 CLASSIFIER). We instead anchor to the current chord's
                 FUNCTIONAL ROOT — the quantity a cascade would have from the
                 root head — and predict bass-RELATIVE-TO-ROOT (0 = root pos).
                 This is the "bass-vs-root delta" the known_issues recommends,
                 and it directly answers "is the normalized frame more
                 learnable?". No aug (already root-anchored).
  (c) CONT     : optional — additionally append a coarse continuous chroma
                 window over the context span (root-anchored) alongside (b).

Combination test: does an unconditional bass head help ROOT accuracy without
the net-negative root-position regression the v1 hard-gate caused?
  S0 baseline root; S1 soft bass-class penalty (sweep beta);
  S2 learned ensemble (root head trained on feat48_abs (+) bass logits).

Read-only on corpus npz. Writes only a small JSON + a plot.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from harmonia.data.corpus_schema import MatchQuality, filter_by_match, load_corpus
from train_real_audio_final import _train_head, _augment_root_by_roll

BASS_SEMI = {"b2":1,"2":2,"b3":3,"3":4,"4":5,"b5":6,"5":7,"b6":8,"6":9,"b7":10,"7":11,"b9":1,"9":2}


def derive_bass_target(label, root):
    """(is_inversion, absolute_sounding_bass_pc). Root-pos/unresolvable -> (0, root)."""
    label = str(label).strip()
    if "/" not in label or label in ("N", "X", ""):
        return 0, root % 12
    b = label.split("/", 1)[1].strip()
    if b not in BASS_SEMI:
        return 0, root % 12  # absolute-note bass we can't resolve -> treat root-pos
    return 1, (root + BASS_SEMI[b]) % 12


def build_context(feat48_abs, song_id, t0, k=4):
    """9 pooled vectors per chord (current +/- k), within song, zero-padded.
    Returns (n, 9*48). Order: [i-4,...,i-1, i, i+1,...,i+4]."""
    n, d = feat48_abs.shape
    out = np.zeros((n, (2*k+1) * d), np.float32)
    # order chords within each song by t0
    for s in sorted(set(song_id.tolist())):
        idx = np.where(song_id == s)[0]
        idx = idx[np.argsort(t0[idx])]
        for pos, gi in enumerate(idx):
            for off in range(-k, k+1):
                p = pos + off
                slot = off + k
                if 0 <= p < len(idx):
                    out[gi, slot*d:(slot+1)*d] = feat48_abs[idx[p]]
    return out


def roll_blocks(X, k, block=12):
    """Roll every 12-wide block of X by k (global transpose)."""
    n, d = X.shape
    return np.roll(X.reshape(n, d//block, block), shift=k, axis=2).reshape(n, d)


def anchor_to_root(Xctx, roots, block=12):
    """Rotate all blocks of each row by -root[i] so functional root -> pc 0."""
    out = np.empty_like(Xctx)
    for i in range(len(Xctx)):
        out[i] = roll_blocks(Xctx[i:i+1], -int(roots[i]) % 12, block)[0]
    return out


def aug_bass(Xctx, y_bass, n_shifts=12, block=12):
    """Roll-aug for absolute bass head: roll all blocks by k, bass += k."""
    n, d = Xctx.shape
    Xs, ys = [Xctx], [y_bass]
    for k in range(1, n_shifts):
        Xs.append(roll_blocks(Xctx, k, block)); ys.append((y_bass + k) % 12)
    return np.concatenate(Xs), np.concatenate(ys)


def _logits(X, model, mean, std, device):
    import torch
    Xn = ((X - mean) / std).astype(np.float32)
    with torch.no_grad():
        return model(torch.tensor(Xn, device=device)).cpu().numpy()


def _softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def one_split(d, seed, is_inv, bass_abs, Xctx, roots_all, *, epochs, lr, batch, device,
              do_cont=False, cont=None, test_frac=0.2):
    keep = filter_by_match(d["match"], minimum=MatchQuality.EXACT)
    feat48_abs = d["feat48_abs"][keep]
    roots = roots_all[keep]
    song_id = d["song_id"][keep]
    inv = is_inv[keep]; babs = bass_abs[keep]
    Xc = Xctx[keep]

    songs = sorted(set(song_id.tolist()))
    rng = np.random.RandomState(seed); rng.shuffle(songs)
    n_test = max(1, int(round(test_frac * len(songs))))
    test_songs = set(songs[:n_test])
    tr = np.array([s not in test_songs for s in song_id]); te = ~tr

    res = {}
    inv_te = inv[te]; is_rp = inv_te == 0; is_iv = inv_te == 1
    res["n_inv_te"] = int(is_iv.sum()); res["n_rp_te"] = int(is_rp.sum())

    # ---------- ROOT head (baseline arm) ----------
    Xtr, ytr = _augment_root_by_roll(feat48_abs[tr], roots[tr])
    rm, rmn, rsd = _train_head(Xtr, ytr, 12, epochs=epochs, lr=lr, batch=batch,
                               device=device, head_name="root")
    root_logits = _logits(feat48_abs[te], rm, rmn, rsd, device)
    root_pred = root_logits.argmax(1); root_te = roots[te]
    res["root_acc_all"] = float((root_pred == root_te).mean())
    res["root_acc_rootpos"] = float((root_pred[is_rp] == root_te[is_rp]).mean())
    res["root_acc_inv"] = float((root_pred[is_iv] == root_te[is_iv]).mean()) if is_iv.sum() else 0.0

    # ---------- (a) RAW absolute bass head ----------
    Xa, ya = aug_bass(Xc[tr], babs[tr])
    ba, amn, asd = _train_head(Xa, ya, 12, epochs=epochs, lr=lr, batch=batch,
                               device=device, head_name="bassRAW")
    bass_logits = _logits(Xc[te], ba, amn, asd, device)
    bass_pred = bass_logits.argmax(1); bass_te = babs[te]
    res["raw_bass_acc_all"] = float((bass_pred == bass_te).mean())
    res["raw_bass_acc_rootpos"] = float((bass_pred[is_rp] == bass_te[is_rp]).mean())
    res["raw_bass_acc_inv"] = float((bass_pred[is_iv] == bass_te[is_iv]).mean()) if is_iv.sum() else 0.0

    # ---------- (b) RENORM (anchor to root) -> predict bass-minus-root ----------
    Xc_rr = anchor_to_root(Xc, roots)
    y_rel = (babs - roots) % 12          # 0 = root position
    br, rmn2, rsd2 = _train_head(Xc_rr[tr], y_rel[tr], 12, epochs=epochs, lr=lr,
                                 batch=batch, device=device, head_name="bassRR")
    rel_pred = _logits(Xc_rr[te], br, rmn2, rsd2, device).argmax(1)
    rel_te = y_rel[te]
    # reconstruct absolute bass from relative + root (using GT root for a clean
    # "is the frame learnable" read; combination test below uses predicted root)
    rr_abs_pred = (rel_pred + root_te) % 12
    res["rr_bass_acc_all"] = float((rr_abs_pred == bass_te).mean())
    res["rr_bass_acc_rootpos"] = float((rr_abs_pred[is_rp] == bass_te[is_rp]).mean())
    res["rr_bass_acc_inv"] = float((rr_abs_pred[is_iv] == bass_te[is_iv]).mean()) if is_iv.sum() else 0.0
    res["rr_rel_acc_inv"] = float((rel_pred[is_iv] == rel_te[is_iv]).mean()) if is_iv.sum() else 0.0

    # ---------- (c) optional continuous window ----------
    if do_cont and cont is not None:
        Ck = cont[keep]
        Xcont = np.concatenate([Xc_rr, Ck], axis=1)  # renorm context + cont window
        bc, cmn, csd = _train_head(Xcont[tr], y_rel[tr], 12, epochs=epochs, lr=lr,
                                   batch=batch, device=device, head_name="bassCONT")
        cont_pred = _logits(Xcont[te], bc, cmn, csd, device).argmax(1)
        cont_abs = (cont_pred + root_te) % 12
        res["cont_bass_acc_all"] = float((cont_abs == bass_te).mean())
        res["cont_bass_acc_inv"] = float((cont_abs[is_iv] == bass_te[is_iv]).mean()) if is_iv.sum() else 0.0

    # ---------- COMBINATION: does bass head help ROOT acc? ----------
    # S1 soft penalty: penalize the predicted-bass class in root logits.
    rsm = _softmax(root_logits); bsm = _softmax(bass_logits)
    for beta in (0.25, 0.5, 1.0):
        comb = rsm - beta * bsm
        cp = comb.argmax(1)
        res[f"s1_root_all_b{beta}"] = float((cp == root_te).mean())
        res[f"s1_root_rp_b{beta}"] = float((cp[is_rp] == root_te[is_rp]).mean())
        res[f"s1_root_inv_b{beta}"] = float((cp[is_iv] == root_te[is_iv]).mean()) if is_iv.sum() else 0.0

    # S2 learned ensemble: retrain root head on feat48_abs (+) bass logits.
    # bass logits on train fold from the RAW bass head (trained on train only).
    bass_logits_tr = _logits(Xc[tr], ba, amn, asd, device)
    Xe_tr = np.concatenate([feat48_abs[tr], bass_logits_tr], axis=1)
    Xe_te = np.concatenate([feat48_abs[te], bass_logits], axis=1)
    em, emn, esd = _train_head(Xe_tr, roots[tr], 12, epochs=epochs, lr=lr,
                               batch=batch, device=device, head_name="rootENS")
    ep = _logits(Xe_te, em, emn, esd, device).argmax(1)
    res["s2_root_all"] = float((ep == root_te).mean())
    res["s2_root_rp"] = float((ep[is_rp] == root_te[is_rp]).mean())
    res["s2_root_inv"] = float((ep[is_iv] == root_te[is_iv]).mean()) if is_iv.sum() else 0.0
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=REPO / "data/cache/rwc/rwc_bp48_fixed.npz")
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--cont", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, default=REPO / "scratchpad/bass_simple_result.json")
    a = ap.parse_args()

    dev = a.device
    if dev is None:
        try:
            import torch; dev = "mps" if torch.backends.mps.is_available() else "cpu"
        except Exception:
            dev = "cpu"

    d = load_corpus(a.corpus)
    labels = d["labels"]; roots_all = d["root"].astype(int)
    is_inv = np.zeros(len(labels), int); bass_abs = np.zeros(len(labels), int)
    for i, lab in enumerate(labels):
        iv, b = derive_bass_target(lab, roots_all[i])
        is_inv[i] = iv; bass_abs[i] = b
    Xctx = build_context(d["feat48_abs"], d["song_id"], d["t0"], k=4)

    keep = filter_by_match(d["match"], minimum=MatchQuality.EXACT)
    print(f"Corpus {a.corpus.name}: {len(labels)} recs, EXACT {keep.sum()}, "
          f"inversions {is_inv[keep].sum()} ({100*is_inv[keep].mean():.1f}%), dev={dev}")

    runs = []
    for s in range(a.seeds):
        print(f"--- seed {s} ---", flush=True)
        r = one_split(d, s, is_inv, bass_abs, Xctx, roots_all, epochs=a.epochs,
                      lr=a.lr, batch=a.batch, device=dev, do_cont=a.cont)
        runs.append(r)
        print(f"  RAW bass all={r['raw_bass_acc_all']:.3f} rp={r['raw_bass_acc_rootpos']:.3f} "
              f"inv={r['raw_bass_acc_inv']:.3f} | RR bass all={r['rr_bass_acc_all']:.3f} "
              f"inv={r['rr_bass_acc_inv']:.3f} | root base={r['root_acc_all']:.3f} "
              f"S2={r['s2_root_all']:.3f}", flush=True)

    def ms(k):
        v = np.array([r[k] for r in runs if k in r]); return (float(v.mean()), float(v.std())) if len(v) else (float('nan'),0.0)

    summary = {k: ms(k) for k in runs[0].keys()}
    a.out.write_text(json.dumps({"summary": summary, "runs": runs,
                                 "corpus": a.corpus.name, "seeds": a.seeds}, indent=2))

    print("\n" + "="*74)
    print(f"RWC SIMPLE unconditional bass-PC, {a.seeds} song-stratified seeds ({a.corpus.name})")
    print(f"  mean test chords/split: inv {np.mean([r['n_inv_te'] for r in runs]):.0f}  "
          f"root-pos {np.mean([r['n_rp_te'] for r in runs]):.0f}")
    rows = [
        ("=== BASS-PC accuracy (unconditional 12-way, chance .083) ===", None),
        ("RAW  bass acc (all)", "raw_bass_acc_all"),
        ("RAW  bass acc (root-position, bass==root)", "raw_bass_acc_rootpos"),
        ("RAW  bass acc (INVERSIONS)", "raw_bass_acc_inv"),
        ("RENORM(root) bass acc (all)", "rr_bass_acc_all"),
        ("RENORM(root) bass acc (root-position)", "rr_bass_acc_rootpos"),
        ("RENORM(root) bass acc (INVERSIONS)", "rr_bass_acc_inv"),
        ("  RENORM rel-to-root acc on inversions", "rr_rel_acc_inv"),
        ("CONT bass acc (all)", "cont_bass_acc_all"),
        ("CONT bass acc (inversions)", "cont_bass_acc_inv"),
        ("=== ROOT accuracy: does bass head help? ===", None),
        ("ROOT baseline (all)", "root_acc_all"),
        ("ROOT baseline (root-position)", "root_acc_rootpos"),
        ("ROOT baseline (inversions)", "root_acc_inv"),
        ("S1 b0.25 root (all)", "s1_root_all_b0.25"),
        ("S1 b0.5  root (all)", "s1_root_all_b0.5"),
        ("S1 b1.0  root (all)", "s1_root_all_b1.0"),
        ("S1 b0.5  root (root-position, regression check)", "s1_root_rp_b0.5"),
        ("S1 b0.5  root (inversions)", "s1_root_inv_b0.5"),
        ("S2 ensemble root (all)", "s2_root_all"),
        ("S2 ensemble root (root-position, regression check)", "s2_root_rp"),
        ("S2 ensemble root (inversions)", "s2_root_inv"),
    ]
    for lbl, k in rows:
        if k is None:
            print(f"\n  {lbl}"); continue
        if k not in runs[0]:
            continue
        m, sd = ms(k); print(f"  {lbl:52s}: {m:.3f} +/- {sd:.3f}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
