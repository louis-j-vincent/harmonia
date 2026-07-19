"""symstruct_proba.py — Stage B of docs/research_sessions/structure_realaudio_
2026_07_18.md: adapt the learned key-norm block encoder to accept a per-bar
12-dim (13 w/ no-chord) root SOFTMAX vector instead of a hard-decoded root
token, so it can consume the real pipeline's actual `root_proba` output
directly instead of an argmax'd discrete chord label.

Root-only (no quality token) per Stage A's finding: at the deployable 8-bar
union scale, root-only ties full-chord (0.698 vs 0.692, TEST, single seed) —
so dropping quality is not a meaningful sacrifice, and it sidesteps needing a
trustworthy quality posterior (real audio's is far noisier than root's, per
docs/session_2026_07_17_bass_root_capstone.md).

Training data: iReal symbolic corpus has no natural probability vectors, so
CLEAN blocks are represented as one-hot 13-d vectors (mathematically a
degenerate softmax) — this is a valid training signal for the soft-input
projection (nn.Linear(13,d_tok)) since a one-hot vector is inside the same
input space a real softmax lives in, just at a vertex of the simplex.

Stage B3a (this file's __main__): synthetic noise stress test — corrupt
clean one-hot root vectors into realistic smoothed distributions (confusable-
root confusion + confidence attenuation, calibrated to roughly match the
entropy range measured on real audio via scratchpad/real_root_proba.py,
~0.8-1.9 nats) and confirm the probabilistic model degrades gracefully vs a
HARD-LABEL baseline (argmax the same noisy vectors -> discrete flat block8).

No audio required for this stress test (reuses the iReal corpus + synthetic
corruption, same spirit as symstruct_robust.py). No commits. Server/UI
untouched.
"""
from __future__ import annotations
import sys, collections, argparse, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from symstruct import load_corpus, vmeasure, predict_blockmatch
from symstruct_learned import (BlockEncoder, nuclear_spans, bar_tokens, key_pc,
                                nt_xent, NC_ROOT, N_ROOT)

MAXLEN = None  # set from --size at runtime (2*size, matches build_blocks convention)


# ── proba-vector blocks ────────────────────────────────────────────────────
def onehot13(root_pc):
    v = np.zeros(13, np.float32)
    v[root_pc] = 1.0   # root_pc in [0..11] real pc, or 12 = NC
    return v


def build_blocks_proba(corpus, size, keynorm=False, noise_fn=None, rng=None):
    """Like symstruct_learned.build_blocks but each bar -> 13-d float vector
    (one-hot, or corrupted soft vector if noise_fn given) instead of a
    discrete (root,qual) token pair. Quality dropped (root-only, Stage A)."""
    span_len = 2 * size
    blocks = []
    for si, c in enumerate(corpus):
        feat, gt = c["feat"], c["labels"]
        n = len(feat)
        shift = (-key_pc(c.get("key")) % 12) if keynorm else 0
        for (s, e) in nuclear_spans(n, size):
            roots = []
            for i in range(s, e):
                r, _ = bar_tokens(feat[i])
                if shift and r < NC_ROOT:
                    r = (r + shift) % 12
                roots.append(r)
            vecs = np.stack([onehot13(r) for r in roots])
            if noise_fn is not None:
                vecs = noise_fn(vecs, rng)
            L = len(vecs)
            padded = np.zeros((span_len, 13), np.float32)
            padded[:L] = vecs[:span_len]
            sec = collections.Counter(gt[s:e]).most_common(1)[0][0]
            blocks.append({"song": si, "sec": (si, sec), "roots": padded,
                           "len": min(L, span_len), "span": (s, e)})
    return blocks


def corrupt_proba(vecs, rng, p_wrong=0.17, conf_lo=0.35, conf_hi=0.85):
    """Corrupt an array of one-hot (T,13) vectors into smoothed, sometimes
    mis-centered distributions. Calibrated against real_root_proba.py's
    measured entropy range (~0.8-1.9 nats/bar on autumn_leaves.m4a):
      - p_wrong: probability the distribution gets re-centered on a
        music-theoretically confusable root (5th/4th/relative-minor/3rd),
        same confusion set flavor as symstruct_robust.py's hard corruption.
      - confidence in [conf_lo, conf_hi] drawn per bar (peaked but not
        one-hot), remaining mass spread uniformly over the other 11 pcs.
      NC (index 12) bars pass through unchanged (no chord = no ambiguity to
      model here)."""
    out = vecs.copy()
    T = len(vecs)
    for t in range(T):
        if vecs[t, 12] > 0.5:      # NC bar, leave alone
            continue
        root = int(vecs[t, :12].argmax())
        center = root
        if rng.random() < p_wrong:
            offset = rng.choice([7, 5, 9, 4, -5, -7])
            center = (root + offset) % 12
        conf = rng.uniform(conf_lo, conf_hi)
        v = np.full(12, (1 - conf) / 12, np.float32)
        v[center] += conf
        v /= v.sum()
        out[t, :12] = v
        out[t, 12] = 0.0
    return out


def encode_blocks_proba(model, blocks, device, bs=1024):
    model.eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(blocks), bs):
            chunk = blocks[i:i + bs]
            roots = np.stack([b["roots"] for b in chunk]).astype(np.float32)
            lens = torch.tensor([b["len"] for b in chunk], device=device)
            z = model(torch.tensor(roots, device=device), None, lens)
            embs.append(z.cpu())
    return torch.cat(embs)


def make_pos_index(blocks):
    groups = collections.defaultdict(list)
    for i, b in enumerate(blocks):
        groups[b["sec"]].append(i)
    return {k: v for k, v in groups.items() if len(v) >= 2}


def sample_batch_proba(blocks, pos_groups, group_keys, n_pairs, rng):
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


def eval_pairwise(embs, blocks):
    by_song = collections.defaultdict(list)
    for i, b in enumerate(blocks):
        by_song[b["song"]].append(i)
    sims, same = [], []
    for song, idxs in by_song.items():
        if len(idxs) < 2:
            continue
        E = embs[idxs]
        S = (E @ E.t()).numpy()
        secs = [blocks[i]["sec"] for i in idxs]
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                sims.append(S[a, b]); same.append(secs[a] == secs[b])
    return np.array(sims), np.array(same, bool)


def prf(sims, same, tau):
    pred = sims >= tau
    tp = int((pred & same).sum()); fp = int((pred & ~same).sum())
    fn = int((~pred & same).sum())
    P = tp / (tp + fp) if tp + fp else float("nan")
    R = tp / (tp + fn) if tp + fn else float("nan")
    F1 = 2 * P * R / (P + R) if P and R and P + R else 0.0
    return P, R, F1


# ── downstream union-find clustering, proba blocks -> V-measure ───────────
def predict_learned_union_proba(feat, model, device, size, tau, keystr=None,
                                 keynorm=False, corrupt=None, rng=None):
    n = len(feat)
    if n < size:
        return ["A"] * n
    shift = (-key_pc(keystr) % 12) if keynorm else 0
    spans = nuclear_spans(n, size)
    blocks = []
    for (s, e) in spans:
        roots = []
        for i in range(s, e):
            r, _ = bar_tokens(feat[i])
            if shift and r < NC_ROOT:
                r = (r + shift) % 12
            roots.append(r)
        vecs = np.stack([onehot13(r) for r in roots])
        if corrupt is not None:
            vecs = corrupt(vecs, rng)
        L = len(vecs)
        padded = np.zeros((2 * size, 13), np.float32)
        padded[:L] = vecs[:2 * size]
        blocks.append({"roots": padded, "len": min(L, 2 * size)})
    E = encode_blocks_proba(model, blocks, device)
    S = (E @ E.t()).numpy()
    m = len(spans)
    parent = list(range(m))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[max(ra, rb)] = min(ra, rb)
    for i in range(m):
        for j in range(i + 1, m):
            if S[i, j] >= tau:
                union(i, j)
    remap = {}; lab = ["A"] * n
    for k, (s, e) in enumerate(spans):
        r = find(k)
        if r not in remap: remap[r] = len(remap)
        for t in range(s, e):
            lab[t] = "S%d" % remap[r]
    return lab


def rotate13(vecs, shift):
    """Rotate the 12 real-pc dims of a (T,13) proba array by `shift`
    semitones (index 12 = no-chord, untouched)."""
    if not shift:
        return vecs
    out = vecs.copy()
    out[:, :12] = np.roll(vecs[:, :12], shift, axis=1)
    return out


def estimate_tonic_pc(bar_proba):
    """Cheap real-audio key proxy (NOT a real key detector): the pc with the
    highest total root-probability mass across the whole song, on the
    heuristic that the tonic triad is usually the most harmonically visited
    chord in tonal pop/jazz. Documented as heuristic, not validated against
    GT key labels — good enough for a qualitative Stage B3b checkpoint, not
    for a claimed metric."""
    return int(bar_proba[:, :12].sum(0).argmax())


def predict_learned_union_proba_bars(bar_proba, model, device, size, tau,
                                     keynorm_shift=0):
    """Downstream union-find clustering directly on a REAL-AUDIO per-bar
    proba array (n_bars,13) from real_root_proba.py — no symbolic `feat`
    involved, this is the real deployment path. Returns a list of length
    n_bars with cluster labels 'S0','S1',..."""
    n = len(bar_proba)
    if n < size:
        return ["S0"] * n
    vecs_all = rotate13(bar_proba, keynorm_shift)
    spans = nuclear_spans(n, size)
    blocks = []
    for (s, e) in spans:
        vecs = vecs_all[s:e]
        L = len(vecs)
        padded = np.zeros((2 * size, 13), np.float32)
        padded[:L] = vecs[:2 * size]
        blocks.append({"roots": padded, "len": min(L, 2 * size)})
    E = encode_blocks_proba(model, blocks, device)
    S = (E @ E.t()).numpy()
    m = len(spans)
    parent = list(range(m))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[max(ra, rb)] = min(ra, rb)
    for i in range(m):
        for j in range(i + 1, m):
            if S[i, j] >= tau:
                union(i, j)
    remap = {}; lab = ["S0"] * n
    for k, (s, e) in enumerate(spans):
        r = find(k)
        if r not in remap: remap[r] = len(remap)
        for t in range(s, e):
            lab[t] = "S%d" % remap[r]
    return lab


def hard_argmax_labels(vecs):
    """Argmax a corrupted (T,13) proba block down to a hard token sequence
    (for the hard-label baseline: same corrupted input, no soft modeling)."""
    return vecs.argmax(1)


def predict_blockmatch_from_proba(feat, size, corrupt_fn, rng, keystr=None,
                                   keynorm=False):
    """HARD-LABEL baseline under the SAME noise: argmax the corrupted proba
    vector per bar -> discrete root token -> flat block-match union (root-
    only exact match within block, transposition-naive since already
    keynorm'd) at `size`-bar blocks. This is the thing the probabilistic
    model needs to beat under noise."""
    n = len(feat)
    shift = (-key_pc(keystr) % 12) if keynorm else 0
    roots_hard = []
    for i in range(n):
        r, _ = bar_tokens(feat[i])
        if shift and r < NC_ROOT:
            r = (r + shift) % 12
        roots_hard.append(r)
    vecs = np.stack([onehot13(r) for r in roots_hard])
    vecs = corrupt_fn(vecs, rng)
    hard = vecs.argmax(1)  # (n,) corrupted hard root per bar
    spans = nuclear_spans(n, size)
    sigs = [tuple(hard[s:e].tolist()) for (s, e) in spans]
    m = len(spans)
    parent = list(range(m))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[max(ra, rb)] = min(ra, rb)
    for i in range(m):
        for j in range(i + 1, m):
            if sigs[i] == sigs[j]:
                union(i, j)
    remap = {}; lab = ["A"] * n
    for k, (s, e) in enumerate(spans):
        r = find(k)
        if r not in remap: remap[r] = len(remap)
        for t in range(s, e):
            lab[t] = "S%d" % remap[r]
    return lab


# ── main: train on clean one-hot, then noise-stress-test ──────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=8)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--npairs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cpu"

    print("loading corpus...", file=sys.stderr)
    corpus = [c for c in load_corpus() if len(set(c["labels"])) >= 2]
    ids = list(range(len(corpus)))
    rng.shuffle(ids)
    n = len(ids)
    ntr, nval = int(0.7 * n), int(0.15 * n)
    train_ids = set(ids[:ntr]); val_ids = set(ids[ntr:ntr + nval])
    test_ids = set(ids[ntr + nval:])
    def sub(idset):
        return [corpus[i] for i in sorted(idset)]
    tr_c, val_c, te_c = sub(train_ids), sub(val_ids), sub(test_ids)

    tr_blocks = build_blocks_proba(tr_c, args.size, keynorm=True)
    val_blocks = build_blocks_proba(val_c, args.size, keynorm=True)
    te_blocks = build_blocks_proba(te_c, args.size, keynorm=True)
    print("blocks: train=%d val=%d test=%d" %
          (len(tr_blocks), len(val_blocks), len(te_blocks)), file=sys.stderr)

    model = BlockEncoder(root_mode="proba", qual_mode="none").to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    pos_groups = make_pos_index(tr_blocks)
    group_keys = list(pos_groups.keys())
    print("train pos-groups=%d params=%d" %
          (len(group_keys), sum(p.numel() for p in model.parameters())),
          file=sys.stderr)

    t0 = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        roots, lens, gids = sample_batch_proba(tr_blocks, pos_groups,
                                               group_keys, args.npairs, rng)
        z = model(torch.tensor(roots, device=device), None,
                  torch.tensor(lens, device=device))
        loss = nt_xent(z, gids, tau=0.2)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 250 == 0 or step == args.steps:
            print("  step %4d loss=%.3f (%.0fs)" % (step, loss.item(),
                  time.time() - t0), file=sys.stderr)

    # ---- CLEAN sanity check (should ~match root-only token result) ----
    taus = np.round(np.arange(0.30, 0.96, 0.05), 2)
    Ev = encode_blocks_proba(model, val_blocks, device)
    sv, samev = eval_pairwise(Ev, val_blocks)
    best_tau = max(taus, key=lambda t: prf(sv, samev, t)[2])
    Et = encode_blocks_proba(model, te_blocks, device)
    st, samet = eval_pairwise(Et, te_blocks)
    P, R, F1 = prf(st, samet, best_tau)
    print("=== CLEAN pairwise (proba model, size=%d) tau*=%.2f P=%.3f R=%.3f F1=%.3f ==="
          % (args.size, best_tau, P, R, F1))

    dtaus = np.round(np.arange(0.50, 0.951, 0.05), 2)
    best_dtau, best_dv = None, -1
    for tau in dtaus:
        vs = [vmeasure(c["labels"], predict_learned_union_proba(
            c["feat"], model, device, args.size, tau, c.get("key"), True))[0]
            for c in val_c]
        mv = np.mean(vs)
        if mv > best_dv:
            best_dv, best_dtau = mv, tau
    vlearn = [vmeasure(c["labels"], predict_learned_union_proba(
        c["feat"], model, device, args.size, best_dtau, c.get("key"), True))[0]
        for c in te_c]
    vb8 = [vmeasure(c["labels"], predict_blockmatch(c["feat"], base_bars=8))[0]
           for c in te_c]
    print("=== CLEAN downstream V_F: proba-model=%.3f (tau=%.2f)  block8=%.3f ==="
          % (np.mean(vlearn), best_dtau, np.mean(vb8)))

    if args.save:
        torch.save({"model": model.state_dict(), "args": vars(args),
                    "tau_star": float(best_dtau),
                    "val_ids": sorted(val_ids), "test_ids": sorted(test_ids)},
                   args.save)
        print("saved ->", args.save)

    # ---- Stage B3a: synthetic noise stress test ----
    print("\n=== STAGE B3a: synthetic noise stress test (TEST, size=%d) ===" % args.size)
    print("%-10s %14s %14s %14s" % ("p_wrong", "proba-model", "hard-argmax-union",
                                     "flat block8(noisy)"))
    noise_rng = np.random.default_rng(1000 + args.seed)
    for p_wrong in [0.0, 0.10, 0.17, 0.30, 0.45]:
        def cfn(vecs, r, pw=p_wrong):
            return corrupt_proba(vecs, r, p_wrong=pw)
        v_proba, v_hard, v_b8 = [], [], []
        for c in te_c:
            v_proba.append(vmeasure(c["labels"], predict_learned_union_proba(
                c["feat"], model, device, args.size, best_dtau, c.get("key"),
                True, corrupt=cfn, rng=noise_rng))[0])
            v_hard.append(vmeasure(c["labels"], predict_blockmatch_from_proba(
                c["feat"], args.size, cfn, noise_rng, c.get("key"), True))[0])
            # flat block8 on hard-argmax'd noisy roots too (size fixed 8, ref)
            v_b8.append(vmeasure(c["labels"], predict_blockmatch_from_proba(
                c["feat"], 8, cfn, noise_rng, c.get("key"), True))[0])
        print("%-10.2f %14.3f %14.3f %14.3f" % (
            p_wrong, np.mean(v_proba), np.mean(v_hard), np.mean(v_b8)))


if __name__ == "__main__":
    main()
