"""symstruct_learned.py — LEARNED section-similarity encoder (metric learning).

Replaces the hard/exact transposition-invariant chord-block matching (whose
clean-GT block-pair ceiling is precision 0.590 / recall 0.328, docs/known_issues.md
2026-07-17) with a small neural encoder trained so that blocks from the SAME GT
section land close in embedding space and blocks from DIFFERENT sections land far
apart — tolerating fills / turnarounds / reharms that break exact matching.

Data:  1992 multi-section iReal tunes (reuses scratchpad/symstruct.load_corpus).
Block: fixed `size`-bar nuclear block; each bar -> (root_pc 0..12, qual 0..6).
Split: song-level 70/15/15 (no song in two splits).
Aug:   random per-block transposition (root shift) — bakes in transposition
       tolerance via data rather than hand-coded normalization.
Loss:  NT-Xent / InfoNCE with in-batch negatives, same-(song,section) masked out.
Eval:  within-song block-pair precision/recall vs the 0.590/0.328 hard baseline,
       plus downstream union-find V-measure vs flat block8 (0.68) / oracle (0.732).

No audio. No commits. Server / UI untouched.
"""
from __future__ import annotations
import sys, collections, random, argparse, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from symstruct import load_corpus, vmeasure, _bar_sig, _block_sim, predict_blockmatch

NC_ROOT, NC_QUAL = 12, 6          # "no chord" indices
N_ROOT, N_QUAL = 13, 7


# ── nuclear blocks ────────────────────────────────────────────────────────────
def nuclear_spans(n, size):
    sp = [(s, min(s + size, n)) for s in range(0, n, size)]
    if len(sp) >= 2 and (sp[-1][1] - sp[-1][0]) < size / 2:
        s, e = sp.pop()
        sp[-1] = (sp[-1][0], e)
    return sp


def bar_tokens(feat_row):
    r, q = _bar_sig(feat_row)          # (-1..11, -1..5)
    return (r if r >= 0 else NC_ROOT, q if q >= 0 else NC_QUAL)


def key_pc(keystr):
    """iReal key string -> tonic pitch class (0=C). Minor keys carry a trailing
    '-' (e.g. 'C-', 'Bb-'); strip it, the tonic pc is the same letter."""
    from harmonia.data.ireal_corpus import chord_root_pc
    if not keystr:
        return 0
    k = keystr.rstrip("-")
    pc = chord_root_pc(k)
    return pc if pc is not None else 0


def build_blocks(corpus, size, keynorm=False):
    """Return list of dicts: {song, sec, roots[np int], quals[np int], span, n}.
    roots/quals are padded to `2*size` with NC (pad-mask via length).
    keynorm: transpose the WHOLE song so its tonic maps to C (pc 0) before
    windowing — one rigid global shift per song, preserving all within-song
    relative structure while canonicalizing cross-song key variance."""
    blocks = []
    for si, c in enumerate(corpus):
        feat, gt = c["feat"], c["labels"]
        n = len(feat)
        shift = (-key_pc(c.get("key")) % 12) if keynorm else 0
        for (s, e) in nuclear_spans(n, size):
            toks = [bar_tokens(feat[i]) for i in range(s, e)][:2 * size]
            if shift:
                toks = [((r + shift) % 12 if r < NC_ROOT else r, q) for r, q in toks]
            L = len(toks)
            roots = np.full(2 * size, NC_ROOT, np.int64)
            quals = np.full(2 * size, NC_QUAL, np.int64)
            for k, (r, q) in enumerate(toks):
                roots[k] = r
                quals[k] = q
            sec = collections.Counter(gt[s:e]).most_common(1)[0][0]
            blocks.append({"song": si, "sec": (si, sec), "roots": roots,
                           "quals": quals, "len": L, "span": (s, e)})
    return blocks


MAXSPAN = 16  # widest span (bars) the variable-span encoder embeds


def build_varspan_blocks(corpus, keynorm=False, lengths=(2, 4, 6, 8, 12, 16),
                         stride=2, purity=0.8):
    """Blocks of VARYING bar-length (for the adaptive hierarchy encoder, which
    must embed arbitrary-length spans in-distribution). For each song and each
    length L, slide a window (stride 2 bars); keep only windows that are >=purity
    one GT section (so the label is clean). Group key = (song, section) as before;
    same-section spans of any length are positives -> the encoder learns a
    length-robust, section-consistent embedding."""
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
                toks = [bar_tokens(feat[i]) for i in range(s, e)][:MAXSPAN]
                if shift:
                    toks = [((r + shift) % 12 if r < NC_ROOT else r, q)
                            for r, q in toks]
                roots = np.full(MAXSPAN, NC_ROOT, np.int64)
                quals = np.full(MAXSPAN, NC_QUAL, np.int64)
                for k, (r, q) in enumerate(toks):
                    roots[k] = r; quals[k] = q
                blocks.append({"song": si, "sec": (si, sec), "roots": roots,
                               "quals": quals, "len": len(toks), "span": (s, e)})
    return blocks


# ── model ─────────────────────────────────────────────────────────────────────
class BlockEncoder(nn.Module):
    """root_mode: 'token' (discrete embedding lookup, default) or 'proba'
    (soft 13-dim root-probability vector projected via nn.Linear — used by
    Stage B to accept a real pipeline's softmax root posterior instead of a
    hard-decoded root pc). qual_mode: 'token' or 'none' (root-only, no
    quality dimension at all — Stage A's root-only variant)."""
    def __init__(self, d_tok=24, hidden=32, emb=32, arch="lstm",
                 root_mode="token", qual_mode="token"):
        super().__init__()
        self.root_mode = root_mode
        self.qual_mode = qual_mode
        if root_mode == "token":
            self.root_emb = nn.Embedding(N_ROOT, d_tok)
        else:  # "proba": 13-dim softmax (12 pc + no-chord mass) -> d_tok
            self.root_proj = nn.Linear(N_ROOT, d_tok)
        if qual_mode == "token":
            self.qual_emb = nn.Embedding(N_QUAL, d_tok)
        in_dim = d_tok if qual_mode == "none" else 2 * d_tok
        self.arch = arch
        if arch == "lstm":
            self.rnn = nn.LSTM(in_dim, hidden, batch_first=True,
                               bidirectional=True)
            self.head = nn.Linear(2 * hidden, emb)
        else:  # mean-pool MLP
            self.mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                     nn.Linear(hidden, emb))

    def forward(self, roots, quals, lengths):
        """roots: LongTensor(B,T) of root-pc token ids if root_mode='token',
        or FloatTensor(B,T,13) softmax vectors if root_mode='proba'.
        quals: LongTensor(B,T) qual ids, ignored if qual_mode='none'."""
        if self.root_mode == "token":
            r_emb = self.root_emb(roots)
        else:
            r_emb = self.root_proj(roots)
        if self.qual_mode == "none":
            x = r_emb
        else:
            x = torch.cat([r_emb, self.qual_emb(quals)], dim=-1)
        if self.arch == "lstm":
            out, _ = self.rnn(x)                     # (B,T,2H)
            # mean over valid positions
            mask = (torch.arange(x.size(1), device=x.device)[None, :]
                    < lengths[:, None]).float().unsqueeze(-1)
            z = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
            z = self.head(z)
        else:
            mask = (torch.arange(x.size(1), device=x.device)[None, :]
                    < lengths[:, None]).float().unsqueeze(-1)
            z = (x * mask).sum(1) / mask.sum(1).clamp(min=1)
            z = self.mlp(z)
        return F.normalize(z, dim=-1)


# ── batch construction ────────────────────────────────────────────────────────
def transpose_block(roots, shift):
    out = roots.copy()
    m = out < NC_ROOT
    out[m] = (out[m] + shift) % 12
    return out


def encode_blocks(model, blocks, device, augment=False, rng=None, bs=1024):
    """Encode a list of blocks -> (N, emb) tensor (no grad)."""
    model.eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(blocks), bs):
            chunk = blocks[i:i + bs]
            roots = np.stack([b["roots"] for b in chunk])
            quals = np.stack([b["quals"] for b in chunk])
            if augment and rng is not None:
                for j in range(len(chunk)):
                    roots[j] = transpose_block(roots[j], int(rng.integers(12)))
            lens = torch.tensor([b["len"] for b in chunk], device=device)
            z = model(torch.tensor(roots, device=device),
                      torch.tensor(quals, device=device), lens)
            embs.append(z.cpu())
    return torch.cat(embs)


def make_pos_index(blocks):
    """group_key -> list of block indices (only groups with >=2)."""
    groups = collections.defaultdict(list)
    for i, b in enumerate(blocks):
        groups[b["sec"]].append(i)
    return {k: v for k, v in groups.items() if len(v) >= 2}


def sample_batch(blocks, pos_groups, group_keys, n_pairs, rng, size, aug="perpair"):
    """Sample n_pairs anchor/positive pairs; return tensors + group ids.

    aug: 'none'     no transposition (absolute pitch)
         'perblock' independent random shift per block (full transp-invariance)
         'perpair'  SAME random shift for an anchor & its positive (keeps the
                    within-song key relationship intact while seeing all keys)."""
    keys = rng.choice(len(group_keys), size=min(n_pairs, len(group_keys)),
                      replace=False)
    a_idx, p_idx, gids = [], [], []
    for gi in keys:
        idxs = pos_groups[group_keys[gi]]
        i, j = rng.choice(len(idxs), size=2, replace=False)
        a_idx.append(idxs[i])
        p_idx.append(idxs[j])
        gids.append(gi)
    allb = a_idx + p_idx
    gids2 = gids + gids
    roots = np.stack([blocks[i]["roots"] for i in allb])
    quals = np.stack([blocks[i]["quals"] for i in allb])
    lens = np.array([blocks[i]["len"] for i in allb])
    if aug == "perblock":
        for j in range(len(allb)):
            roots[j] = transpose_block(roots[j], int(rng.integers(12)))
    elif aug == "perpair":
        npair = len(a_idx)
        for k in range(npair):
            sh = int(rng.integers(12))
            roots[k] = transpose_block(roots[k], sh)          # anchor
            roots[k + npair] = transpose_block(roots[k + npair], sh)  # positive
    return roots, quals, lens, np.array(gids2)


def nt_xent(z, gids, tau=0.2):
    """InfoNCE with in-batch negatives; same-group (other than the paired anchor)
    still counts as positive-ish, so we mask same-group entries from the denom
    except the designated positive. Simpler: treat every same-gid pair as positive
    (supervised contrastive)."""
    B = z.size(0)
    sim = z @ z.t() / tau
    sim.fill_diagonal_(-1e9)
    gids_t = torch.tensor(gids, device=z.device)
    pos_mask = (gids_t[:, None] == gids_t[None, :]).float()
    pos_mask.fill_diagonal_(0)
    logits = sim
    logsumexp = torch.logsumexp(logits, dim=1, keepdim=True)
    log_prob = logits - logsumexp
    denom = pos_mask.sum(1).clamp(min=1)
    loss = -(pos_mask * log_prob).sum(1) / denom
    return loss.mean()


# ── evaluation: within-song block-pair precision/recall ───────────────────────
def eval_pairwise(embs, blocks, song_of, taus):
    """Pool over within-song block pairs. Return dict tau -> (P,R,F1) and the
    similarity/label arrays for curve analysis."""
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
                sims.append(S[a, b])
                same.append(secs[a] == secs[b])
    sims = np.array(sims); same = np.array(same, bool)
    out = {}
    for tau in taus:
        pred = sims >= tau
        tp = int((pred & same).sum()); fp = int((pred & ~same).sum())
        fn = int((~pred & same).sum())
        P = tp / (tp + fp) if tp + fp else float("nan")
        R = tp / (tp + fn) if tp + fn else float("nan")
        F1 = 2 * P * R / (P + R) if P and R and P + R else 0.0
        out[tau] = (P, R, F1)
    return out, sims, same


def hard_pairwise(blocks, sim_threshold=0.75):
    """Baseline: raw pairwise hard transposition-match P/R (within-song)."""
    by_song = collections.defaultdict(list)
    for i, b in enumerate(blocks):
        by_song[b["song"]].append(i)
    sigs = [list(zip(b["roots"].tolist(), b["quals"].tolist())) for b in blocks]
    # convert NC back to -1 so _block_sim's "both none" logic works
    def sig_of(b):
        return [(-1 if r == NC_ROOT else r, -1 if q == NC_QUAL else q)
                for r, q in zip(blocks[b]["roots"][:blocks[b]["len"]],
                                blocks[b]["quals"][:blocks[b]["len"]])]
    tp = fp = fn = 0
    for song, idxs in by_song.items():
        if len(idxs) < 2:
            continue
        S = [sig_of(i) for i in idxs]
        secs = [blocks[i]["sec"] for i in idxs]
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                pred = _block_sim(S[a], S[b]) >= sim_threshold
                sm = secs[a] == secs[b]
                if pred and sm: tp += 1
                elif pred and not sm: fp += 1
                elif not pred and sm: fn += 1
    P = tp / (tp + fp) if tp + fp else float("nan")
    R = tp / (tp + fn) if tp + fn else float("nan")
    return P, R


# ── downstream: union-find clustering with learned similarity -> V-measure ────
def predict_learned_union(feat, model, device, size, tau, keystr=None, keynorm=False):
    n = len(feat)
    if n < size:
        return ["A"] * n
    shift = (-key_pc(keystr) % 12) if keynorm else 0
    spans = nuclear_spans(n, size)
    blocks = []
    for (s, e) in spans:
        toks = [bar_tokens(feat[i]) for i in range(s, e)][:2 * size]
        if shift:
            toks = [((r + shift) % 12 if r < NC_ROOT else r, q) for r, q in toks]
        roots = np.full(2 * size, NC_ROOT, np.int64)
        quals = np.full(2 * size, NC_QUAL, np.int64)
        for k, (r, q) in enumerate(toks):
            roots[k] = r; quals[k] = q
        blocks.append({"roots": roots, "quals": quals, "len": len(toks)})
    E = encode_blocks(model, blocks, device)
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


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=4)
    ap.add_argument("--emb", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--arch", default="lstm", choices=["lstm", "meanpool"])
    ap.add_argument("--tau", type=float, default=0.2, help="InfoNCE temperature")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--npairs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--aug", default="none", choices=["none", "perblock", "perpair"])
    ap.add_argument("--keynorm", action="store_true",
                    help="whole-song key normalization (tonic->C) as fixed input")
    ap.add_argument("--varspan", action="store_true",
                    help="train on variable-length spans (for adaptive hierarchy)")
    ap.add_argument("--rootonly", action="store_true",
                    help="Stage A: drop quality entirely, token = root pc only")
    ap.add_argument("--save", default="")
    ap.add_argument("--downstream", action="store_true")
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
    print("split: train=%d val=%d test=%d songs" %
          (len(train_ids), len(val_ids), len(test_ids)), file=sys.stderr)

    def sub(idset):
        return [corpus[i] for i in sorted(idset)]
    if args.varspan:
        tr_blocks = build_varspan_blocks(sub(train_ids), keynorm=args.keynorm)
        val_blocks = build_varspan_blocks(sub(val_ids), keynorm=args.keynorm)
        te_blocks = build_varspan_blocks(sub(test_ids), keynorm=args.keynorm)
    else:
        tr_blocks = build_blocks(sub(train_ids), args.size, keynorm=args.keynorm)
        val_blocks = build_blocks(sub(val_ids), args.size, keynorm=args.keynorm)
        te_blocks = build_blocks(sub(test_ids), args.size, keynorm=args.keynorm)
    print("blocks: train=%d val=%d test=%d" %
          (len(tr_blocks), len(val_blocks), len(te_blocks)), file=sys.stderr)

    # baseline hard-matching pairwise on TEST
    hb_p, hb_r = hard_pairwise(te_blocks)
    print("HARD-MATCH baseline (test, pairwise): P=%.3f R=%.3f" % (hb_p, hb_r))

    qual_mode = "none" if args.rootonly else "token"
    model = BlockEncoder(hidden=args.hidden, emb=args.emb, arch=args.arch,
                         qual_mode=qual_mode).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    pos_groups = make_pos_index(tr_blocks)
    group_keys = list(pos_groups.keys())
    print("train pos-groups=%d, params=%d" %
          (len(group_keys), sum(p.numel() for p in model.parameters())),
          file=sys.stderr)

    taus = np.round(np.arange(0.30, 0.96, 0.02), 2)
    t0 = time.time()
    best_val_f1 = -1
    for step in range(1, args.steps + 1):
        model.train()
        roots, quals, lens, gids = sample_batch(
            tr_blocks, pos_groups, group_keys, args.npairs, rng, args.size,
            aug=args.aug)
        z = model(torch.tensor(roots, device=device),
                  torch.tensor(quals, device=device),
                  torch.tensor(lens, device=device))
        loss = nt_xent(z, gids, tau=args.tau)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 250 == 0 or step == args.steps:
            E = encode_blocks(model, val_blocks, device)
            song_of = [b["song"] for b in val_blocks]
            res, _, _ = eval_pairwise(E, val_blocks, song_of, taus)
            # pick tau maximizing F1 on val
            bt = max(res, key=lambda t: res[t][2])
            P, R, F1 = res[bt]
            # also recall@precision>=0.59 point
            print("  step %4d loss=%.3f  val bestF1 tau=%.2f P=%.3f R=%.3f F1=%.3f  (%.0fs)"
                  % (step, loss.item(), bt, P, R, F1, time.time() - t0))
            best_val_f1 = max(best_val_f1, F1)

    # final: choose tau on VAL, report on TEST
    Ev = encode_blocks(model, val_blocks, device)
    resv, _, _ = eval_pairwise(Ev, val_blocks, [b["song"] for b in val_blocks], taus)
    tau_star = max(resv, key=lambda t: resv[t][2])
    Et = encode_blocks(model, te_blocks, device)
    rest, sims, same = eval_pairwise(Et, te_blocks, [b["song"] for b in te_blocks], taus)
    P, R, F1 = rest[tau_star]
    print("\n=== TEST (tau*=%.2f chosen on val) ===" % tau_star)
    print("  LEARNED : P=%.3f R=%.3f F1=%.3f" % (P, R, F1))
    print("  HARD    : P=%.3f R=%.3f" % (hb_p, hb_r))
    # Pareto points: recall at P>=hard_P, precision at R>=hard_R
    r_at_p = max([rest[t][1] for t in taus if rest[t][0] >= hb_p] or [float("nan")])
    p_at_r = max([rest[t][0] for t in taus if rest[t][1] >= hb_r] or [float("nan")])
    print("  LEARNED recall @P>=%.2f: %.3f  |  precision @R>=%.2f: %.3f"
          % (hb_p, r_at_p, hb_r, p_at_r))
    # full curve dump (coarse)
    print("  curve (tau: P/R):", " ".join(
        "%.2f:%.2f/%.2f" % (t, rest[t][0], rest[t][1]) for t in taus[::3]))

    if args.save:
        torch.save({"model": model.state_dict(), "args": vars(args),
                    "tau_star": float(tau_star),
                    "val_ids": sorted(val_ids), "test_ids": sorted(test_ids),
                    "train_ids": sorted(train_ids)}, args.save)
        print("saved ->", args.save)

    if args.downstream:
        run_downstream(model, device, sub(val_ids), sub(test_ids), args.size,
                       keynorm=args.keynorm)


def run_downstream(model, device, val_corpus, test_corpus, size, keynorm=False):
    """Sweep the union threshold on VAL (V-measure-optimal != F1-optimal), then
    report TEST V-measure vs flat block8."""
    print("\n=== DOWNSTREAM V-measure ===")
    taus = np.round(np.arange(0.50, 0.951, 0.05), 2)
    # choose tau on val
    best_tau, best_v = None, -1
    for tau in taus:
        vs = [vmeasure(c["labels"], predict_learned_union(
            c["feat"], model, device, size, tau, c.get("key"), keynorm))[0]
            for c in val_corpus]
        mv = np.mean(vs)
        if mv > best_v:
            best_v, best_tau = mv, tau
    print("  val tau-sweep best: tau=%.2f V_F=%.3f" % (best_tau, best_v))
    vlearn, vb8, vb4 = [], [], []
    for c in test_corpus:
        gt = c["labels"]
        vlearn.append(vmeasure(gt, predict_learned_union(
            c["feat"], model, device, size, best_tau, c.get("key"), keynorm))[0])
        vb8.append(vmeasure(gt, predict_blockmatch(c["feat"], base_bars=8))[0])
        vb4.append(vmeasure(gt, predict_blockmatch(c["feat"], base_bars=4))[0])
    print("  TEST learned union (size=%d, tau*=%.2f) V_F=%.3f" %
          (size, best_tau, np.mean(vlearn)))
    print("  TEST flat block8 (ref)                 V_F=%.3f" % np.mean(vb8))
    print("  TEST flat block4 (ref)                 V_F=%.3f" % np.mean(vb4))
    print("  [full-corpus refs: block8=0.681, fixed-scale oracle=0.732]")


if __name__ == "__main__":
    main()
