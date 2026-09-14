"""Train + evaluate the half-bar chord LM against n-gram baselines.

    .venv/bin/python scripts/chord_lm_train.py --epochs 60

Reports one table. Column meanings (see harmonia_min/chord_lm/evaluate.py):
    tok%      next-slot token accuracy over the full 90-token vocabulary
    chg%      "does the harmony change on this half-bar" accuracy
    chgF1     F1 on the change class (the minority-ish, informative one)
    id%       exact chord right at slots where a change actually happens
    id|chg%   same, but told a change happens (argmax over chord tokens only)
    root%     root right, given a change
    fam%      family right, given a change
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harmonia_min.chord_lm import corpus, data, evaluate, model as M, vocab

OUT_DIR = Path("data/models")


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


# ── baselines ───────────────────────────────────────────────────────────────
def eval_ngram(lm, seqs: list[list[int]]) -> evaluate.Metrics:
    """Next-slot prediction, scored on positions 1.. so it matches the causal LM."""
    acc = evaluate.MetricAccumulator()
    for s in seqs:
        if len(s) < 2:
            continue
        lp = lm.logprobs(s)[1:]
        acc.add(lp, np.array(s[1:], dtype=np.int64))
    return acc.result()


# ── training ────────────────────────────────────────────────────────────────
def train(net: M.ChordLM, ds_tr: data.ChordDataset, ds_va: data.ChordDataset, *,
          epochs: int, lr: float, batch_size: int, device: str, tag: str,
          log_every: int = 10) -> M.ChordLM:
    net.to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01,
                            betas=(0.9, 0.95))
    n_steps = max(epochs * math.ceil(len(ds_tr) / batch_size), 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=n_steps,
                                                pct_start=0.1)
    best = (float("inf"), None)
    step = 0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        net.train()
        tot, ntok = 0.0, 0
        for b in ds_tr.batches(batch_size):
            b = b.to(device)
            logits = net(b.x, b.slot_ix)
            loss = F.cross_entropy(logits.reshape(-1, vocab.VOCAB_SIZE),
                                   b.y.reshape(-1), ignore_index=vocab.PAD)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            n = int((b.y != vocab.PAD).sum())
            tot += loss.item() * n
            ntok += n
        va = validate(net, ds_va, batch_size, device)
        if va < best[0]:
            best = (va, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()})
        if ep % log_every == 0 or ep == 1 or ep == epochs:
            print(f"    [{tag}] epoch {ep:3d}  train ppl {math.exp(tot/max(ntok,1)):7.3f}"
                  f"   val ppl {math.exp(va):7.3f}   ({time.time()-t0:5.1f}s)")
    if best[1] is not None:
        net.load_state_dict(best[1])
    print(f"    [{tag}] best val ppl {math.exp(best[0]):.3f}")
    return net


@torch.no_grad()
def validate(net: M.ChordLM, ds: data.ChordDataset, batch_size: int, device: str) -> float:
    net.eval()
    tot, ntok = 0.0, 0
    for b in ds.batches(batch_size, shuffle=False):
        b = b.to(device)
        logits = net(b.x, b.slot_ix)
        loss = F.cross_entropy(logits.reshape(-1, vocab.VOCAB_SIZE),
                               b.y.reshape(-1), ignore_index=vocab.PAD,
                               reduction="sum")
        n = int((b.y != vocab.PAD).sum())
        tot += float(loss)
        ntok += n
    return tot / max(ntok, 1)


@torch.no_grad()
def eval_causal(net: M.ChordLM, seqs: list[list[int]], device: str,
                batch_size: int = 32) -> evaluate.Metrics:
    net.eval()
    acc = evaluate.MetricAccumulator()
    for i in range(0, len(seqs), batch_size):
        chunk = seqs[i:i + batch_size]
        T = max(len(s) for s in chunk)
        x = torch.full((len(chunk), T), vocab.PAD, dtype=torch.long)
        for j, s in enumerate(chunk):
            x[j, :len(s)] = torch.tensor(s)
        x = x.to(device)
        slot_ix = (torch.arange(T, device=device) % net.cfg.slots_per_bar
                   ).expand(len(chunk), T)
        lp = F.log_softmax(net(x, slot_ix), dim=-1).cpu().numpy()
        for j, s in enumerate(chunk):
            if len(s) < 2:
                continue
            acc.add(lp[j, :len(s) - 1], np.array(s[1:], dtype=np.int64))
    return acc.result()


@torch.no_grad()
def eval_cloze(net: M.ChordLM, seqs: list[list[int]], device: str,
               max_batch: int = 128) -> evaluate.Metrics:
    """Mask exactly ONE slot, keep every other slot true, fill it in.

    This is the number that matters for the pipeline: it measures how well pure
    harmonic grammar can name a slot when both its past and its future are
    known — i.e. the ceiling on what the LM can contribute where the acoustic
    model is unsure. Every slot of every test sequence is masked in turn.
    """
    net.eval()
    acc = evaluate.MetricAccumulator()
    for s in seqs:
        T = len(s)
        if T < 4:
            continue
        base = torch.tensor(s, dtype=torch.long)
        # start at 1, not 0: eval_causal cannot score position 0 (nothing
        # precedes it), and comparing the two on different position sets
        # would confound the information asymmetry we are measuring.
        for start in range(1, T, max_batch):
            positions = list(range(start, min(start + max_batch, T)))
            x = base.unsqueeze(0).repeat(len(positions), 1).clone()
            for r, p in enumerate(positions):
                x[r, p] = vocab.MASK
            x = x.to(device)
            slot_ix = (torch.arange(T, device=device) % net.cfg.slots_per_bar
                       ).expand(len(positions), T)
            lp = F.log_softmax(net(x, slot_ix), dim=-1)
            rows = torch.arange(len(positions), device=device)
            picked = lp[rows, torch.tensor(positions, device=device)].cpu().numpy()
            acc.add(picked, np.array([s[p] for p in positions], dtype=np.int64))
    return acc.result()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--causal-epochs", type=int, default=None)
    # The masked model gets a loss on only ~16% of slots per epoch (the masked
    # ones); the causal model gets one on every slot. Equal epochs = ~6x less
    # gradient signal, so it needs proportionally more of them.
    ap.add_argument("--masked-epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--slots-per-bar", type=int, default=2)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--only", choices=["causal", "masked", "both"], default="both")
    ap.add_argument("--no-rope", action="store_true",
                    help="learned absolute positions instead of rotary (relative)")
    ap.add_argument("--out", type=str, default="chord_lm")
    args = ap.parse_args()

    device = pick_device()
    print(f"device: {device}   slots/bar: {args.slots_per_bar}")

    splits = corpus.load_splits(slots_per_bar=args.slots_per_bar)
    print(splits.summary())

    tr_seqs = [c for ch in splits.train for c in data.chunk(ch, args.max_len)]
    va_seqs = [c for ch in splits.val for c in data.chunk(ch, args.max_len)]
    te_seqs = [c for ch in splits.test for c in data.chunk(ch, args.max_len)]
    print(f"sequences: train {len(tr_seqs)}  val {len(va_seqs)}  test {len(te_seqs)}")

    rows: list[tuple[str, evaluate.Metrics]] = []

    if not args.skip_baselines:
        print("\nbaselines...")
        rows.append(("always-REPEAT", eval_ngram(evaluate.AlwaysRepeat(), te_seqs)))
        uni = evaluate.UnigramLM().fit(tr_seqs)
        rows.append(("unigram", eval_ngram(uni, te_seqs)))
        for order, name in ((2, "bigram"), (3, "trigram"), (4, "4-gram")):
            t0 = time.time()
            ng = evaluate.NGramLM(order=order).fit(tr_seqs)
            ng.tune_lambdas(va_seqs[:120], n_iter=12)
            rows.append((f"{name} (12-key counts)", eval_ngram(ng, te_seqs)))
            print(f"    {name}: lambdas {np.round(ng.lambdas, 3)}  ({time.time()-t0:.0f}s)")

    cfg_common = dict(d_model=args.d_model, n_layers=args.layers, n_heads=args.heads,
                      d_ff=4 * args.d_model, max_len=args.max_len,
                      slots_per_bar=args.slots_per_bar, rope=not args.no_rope)
    print(f"positions: {'rotary (relative)' if not args.no_rope else 'learned absolute'}")

    cfg_c = cfg_m = net_c = net_m = None

    # ── causal (GPT-style) ──────────────────────────────────────────────────
    if args.only in ("causal", "both"):
        print("\ncausal transformer...")
        cfg_c = M.LMConfig(causal=True, **cfg_common)
        net_c = M.ChordLM(cfg_c)
        print(f"    params: {net_c.n_params()/1e6:.2f}M")
        ds_tr = data.ChordDataset(splits.train, max_len=args.max_len, causal=True,
                                  slots_per_bar=args.slots_per_bar,
                                  augment=not args.no_augment, seed=0)
        ds_va = data.ChordDataset(splits.val, max_len=args.max_len, causal=True,
                                  slots_per_bar=args.slots_per_bar, augment=False,
                                  seed=1)
        net_c = train(net_c, ds_tr, ds_va,
                      epochs=args.causal_epochs or args.epochs, lr=args.lr,
                      batch_size=args.batch_size, device=device, tag="causal")
        rows.append(("transformer causal", eval_causal(net_c, te_seqs, device)))

    # ── masked (BERT-style) ─────────────────────────────────────────────────
    if args.only in ("masked", "both"):
        print("\nmasked transformer...")
        cfg_m = M.LMConfig(causal=False, **cfg_common)
        net_m = M.ChordLM(cfg_m)
        ds_trm = data.ChordDataset(splits.train, max_len=args.max_len, causal=False,
                                   slots_per_bar=args.slots_per_bar,
                                   augment=not args.no_augment, seed=2)
        # fixed rate for validation: a random rate would make val ppl jump between
        # epochs and pick the best checkpoint by luck rather than by fit
        ds_vam = data.ChordDataset(splits.val, max_len=args.max_len, causal=False,
                                   slots_per_bar=args.slots_per_bar, augment=False,
                                   mask_prob=0.15, min_mask_prob=0.15, seed=3)
        net_m = train(net_m, ds_trm, ds_vam,
                      epochs=args.masked_epochs or args.epochs, lr=args.lr,
                      batch_size=args.batch_size, device=device, tag="masked")
        if not args.skip_baselines:
            cb = evaluate.ClozeBigram().fit(tr_seqs)
            acc = evaluate.MetricAccumulator()
            for s in te_seqs:
                if len(s) >= 4:
                    acc.add(cb.logprobs_cloze(s), np.array(s, dtype=np.int64))
            rows.append(("cloze bigram (both sides)", acc.result()))
        rows.append(("transformer masked (cloze)", eval_cloze(net_m, te_seqs, device)))

    print("\n" + "=" * 100)
    print("TEST SET — next-slot prediction (all rows) except the masked row,")
    print("which is single-slot cloze with both-sided context.")
    print("=" * 100)
    print(evaluate.HEADER)
    print("-" * 100)
    for name, m in rows:
        print(m.table_row(name))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if net_c is not None:
        torch.save({"cfg": cfg_c.__dict__, "state": net_c.state_dict()},
                   OUT_DIR / f"{args.out}_causal.pt")
    if net_m is not None:
        torch.save({"cfg": cfg_m.__dict__, "state": net_m.state_dict()},
                   OUT_DIR / f"{args.out}_masked.pt")
    (OUT_DIR / f"{args.out}_results.json").write_text(json.dumps(
        {name: m.__dict__ for name, m in rows}, indent=2))
    print(f"\nsaved -> {OUT_DIR}/{args.out}_*.pt")


if __name__ == "__main__":
    main()
