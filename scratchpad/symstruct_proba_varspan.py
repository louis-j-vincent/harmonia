"""symstruct_proba_varspan.py — Task 2 prerequisite: a VARIABLE-SPAN,
PROBABILISTIC-root encoder, combining two Call-1/Call-2 pieces that existed
separately but never together:
  - symstruct_proba.py's soft 13-d root-probability input (root_mode="proba"),
    validated Stage A/B: root-only ties full-chord downstream, and the
    probabilistic input beats a hard-label baseline under realistic noise.
  - symstruct_learned.py's build_varspan_blocks() variable-length training
    (stride-2 windows at ALL start positions, lengths 2-16 bars), which is
    what symstruct_adaptive.py's agglomerative merge needs (arbitrary-length
    span embedding) -- and is itself a partial phase-fix (Call 2 known_issues
    entry: training on all start positions makes the encoder phase-agnostic
    as a SIMILARITY function, even though downstream nuclear_spans() calls
    are still phase-locked as a SEGMENTATION scheme, per the "Call 2
    follow-up" finding).

Neither piece alone had this combination trained. This script trains it, so
the mandated hierarchical-merge experiment (brief Task 2: "use the
PROBABILISTIC-root encoder... as the merge-similarity source" for the
adaptive agglomerative hierarchy) can actually run on a same-length-family
similarity source instead of falling back to the token encoder.
"""
from __future__ import annotations
import sys, collections, argparse, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from symstruct import load_corpus, vmeasure, predict_blockmatch
from symstruct_learned import (BlockEncoder, key_pc, nt_xent, NC_ROOT, MAXSPAN)
from symstruct_proba import (onehot13, make_pos_index, eval_pairwise, prf)


def build_varspan_blocks_proba(corpus, keynorm=True, lengths=(2, 4, 6, 8, 12, 16),
                                stride=2, purity=0.8):
    """Like symstruct_learned.build_varspan_blocks but each bar is a 13-d
    (one-hot, clean-data) root-probability vector instead of a discrete
    token, root-only (matches Stage A's validated root-only finding)."""
    blocks = []
    for si, c in enumerate(corpus):
        feat, gt = c["feat"], c["labels"]
        n = len(feat)
        shift = (-key_pc(c.get("key")) % 12) if keynorm else 0
        for L in lengths:
            if L > n:
                continue
            for s in range(0, n - L + 1, stride):
                e = s + L
                cnt = collections.Counter(gt[s:e])
                sec, top = cnt.most_common(1)[0]
                if top / L < purity:
                    continue
                roots = []
                for i in range(s, e):
                    from symstruct_learned import bar_tokens
                    r, _ = bar_tokens(feat[i])
                    if shift and r < NC_ROOT:
                        r = (r + shift) % 12
                    roots.append(r)
                vecs = np.stack([onehot13(r) for r in roots])
                padded = np.zeros((MAXSPAN, 13), np.float32)
                padded[:len(vecs)] = vecs[:MAXSPAN]
                blocks.append({"song": si, "sec": (si, sec), "roots": padded,
                               "len": min(len(vecs), MAXSPAN), "span": (s, e)})
    return blocks


def encode_blocks(model, blocks, bs=1024):
    model.eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(blocks), bs):
            chunk = blocks[i:i + bs]
            roots = np.stack([b["roots"] for b in chunk]).astype(np.float32)
            lens = torch.tensor([b["len"] for b in chunk])
            z = model(torch.tensor(roots), None, lens)
            embs.append(z.cpu())
    return torch.cat(embs)


def sample_batch(blocks, pos_groups, group_keys, n_pairs, rng):
    keys = rng.choice(len(group_keys), size=min(n_pairs, len(group_keys)),
                      replace=False)
    a_idx, p_idx, gids = [], [], []
    for gi in keys:
        idxs = pos_groups[group_keys[gi]]
        i, j = rng.choice(len(idxs), size=2, replace=False)
        a_idx.append(idxs[i]); p_idx.append(idxs[j]); gids.append(gi)
    allb = a_idx + p_idx
    gids2 = gids + gids
    roots = np.stack([blocks[i]["roots"] for i in allb]).astype(np.float32)
    lens = np.array([blocks[i]["len"] for i in allb])
    return roots, lens, np.array(gids2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--npairs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default="scratchpad/keynorm_proba_varspan.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    print("loading corpus...", file=sys.stderr)
    corpus = [c for c in load_corpus() if len(set(c["labels"])) >= 2]
    ids = list(range(len(corpus)))
    rng.shuffle(ids)
    n = len(ids)
    ntr, nval = int(0.7 * n), int(0.15 * n)
    train_ids, val_ids = set(ids[:ntr]), set(ids[ntr:ntr + nval])
    test_ids = set(ids[ntr + nval:])
    sub = lambda s: [corpus[i] for i in sorted(s)]
    tr_c, val_c, te_c = sub(train_ids), sub(val_ids), sub(test_ids)

    tr_blocks = build_varspan_blocks_proba(tr_c, keynorm=True)
    val_blocks = build_varspan_blocks_proba(val_c, keynorm=True)
    te_blocks = build_varspan_blocks_proba(te_c, keynorm=True)
    print("varspan-proba blocks: train=%d val=%d test=%d" %
          (len(tr_blocks), len(val_blocks), len(te_blocks)), file=sys.stderr)

    model = BlockEncoder(root_mode="proba", qual_mode="none")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    pos_groups = make_pos_index(tr_blocks)
    group_keys = list(pos_groups.keys())
    print("train pos-groups=%d params=%d" %
          (len(group_keys), sum(p.numel() for p in model.parameters())), file=sys.stderr)

    t0 = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        roots, lens, gids = sample_batch(tr_blocks, pos_groups, group_keys,
                                          args.npairs, rng)
        z = model(torch.tensor(roots), None, torch.tensor(lens))
        loss = nt_xent(z, gids, tau=0.2)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0 or step == args.steps:
            print("  step %4d loss=%.3f (%.0fs)" % (step, loss.item(), time.time() - t0),
                  file=sys.stderr)

    taus = np.round(np.arange(0.30, 0.96, 0.05), 2)
    Ev = encode_blocks(model, val_blocks)
    sv, samev = eval_pairwise(Ev, val_blocks)
    best_tau = max(taus, key=lambda t: prf(sv, samev, t)[2])
    Et = encode_blocks(model, te_blocks)
    st, samet = eval_pairwise(Et, te_blocks)
    P, R, F1 = prf(st, samet, best_tau)
    print("=== CLEAN varspan-pair (proba, root-only) tau*=%.2f P=%.3f R=%.3f F1=%.3f ==="
          % (best_tau, P, R, F1))

    if args.save:
        torch.save({"model": model.state_dict(),
                    "args": {"hidden": 32, "emb": 32, "arch": "lstm",
                             "root_mode": "proba", "qual_mode": "none",
                             "keynorm": True},
                    "val_ids": sorted(val_ids), "test_ids": sorted(test_ids)},
                   args.save)
        print("saved ->", args.save)


if __name__ == "__main__":
    main()
