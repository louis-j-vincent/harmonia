"""Train + evaluate the per-chord local-key SEQUENCE model (#20/#23).

Distils the rule-based heuristic ``continuity_scale_track_v2`` (NOT the section
oracle — the user's explicit choice) into a many-to-many bi-GRU tagger that
predicts a key at every chord, with whole-song context so it can smooth
secondary-dominant chains that the heuristic's 2-chord lookahead cannot.

Reports (per the user's brief):
  1. per-position key accuracy, pop-like (pop400+blues50) vs jazz1460 separately;
  2. collection-change "churn" per 100 chords, model vs raw heuristic — the model
     should be *less* noisy than its own teacher on dominant chains without
     missing genuine collection changes;
  3. the "A Beautiful Friendship" bridge case side by side (heuristic vs model).

Usage:
    python scripts/train_local_key_seq_model.py [--epochs 60] [--device mps]
"""
from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from harmonia.models.local_key_seq_data import (
    DEFAULT_DB,
    JAZZ_CORPORA,
    POP_CORPORA,
    build_seq_examples,
    collection_of,
    count_collection_changes,
    rel_to_abs_key,
    split_seq_examples,
    tokens_to_rel_example,
)
from harmonia.models.local_key_seq_model import (
    PAD_KEY,
    LocalKeySeqGRU,
    collate,
    predict_sequence,
)
from harmonia.theory.local_key import key_name

POP_LIKE = POP_CORPORA | JAZZ_CORPORA  # for labelling only
CKPT = DEFAULT_DB.parent.parent / "cache" / "local_key_seq_gru.pt"

# "A Beautiful Friendship" section B (context: C major) — the user's canonical case.
ABF_TOKENS = ["G-7", "C7", "F^7", "Bb7", "E-7", "A7", "D7", "G7#5"]
ABF_HOME = (0, "major")


def _corpus_group(corpus: str) -> str:
    return "jazz" if corpus in JAZZ_CORPORA else "pop"


@torch.no_grad()
def evaluate(model, examples, device):
    """Per-position accuracy + churn, split by corpus group ('pop' | 'jazz').

    Returns {group: {acc, n_chord, model_changes, heur_changes, n_seq}}.
    """
    model.eval()
    agg = {g: dict(correct=0, n_chord=0, model_ch=0, heur_ch=0, n_seq=0)
           for g in ("pop", "jazz")}
    for i in range(0, len(examples), 128):
        chunk = examples[i:i + 128]
        root, qual, lengths, targets = collate(
            [(e["seq"], e["y"]) for e in chunk], device)
        pred = model(root, qual, lengths).argmax(-1).cpu().numpy()  # (B,T)
        for j, e in enumerate(chunk):
            g = _corpus_group(e["corpus"])
            n = len(e["seq"])
            p = pred[j, :n].tolist()
            y = e["y"]
            agg[g]["correct"] += sum(int(a == b) for a, b in zip(p, y))
            agg[g]["n_chord"] += n
            agg[g]["model_ch"] += count_collection_changes(p)
            agg[g]["heur_ch"] += count_collection_changes(y)
            agg[g]["n_seq"] += 1
    out = {}
    for g, a in agg.items():
        nc = max(a["n_chord"], 1)
        out[g] = {
            "acc": a["correct"] / nc,
            "n_chord": a["n_chord"],
            "n_seq": a["n_seq"],
            "model_churn": 100 * a["model_ch"] / nc,
            "heur_churn": 100 * a["heur_ch"] / nc,
        }
    return out


def _knm(idx):
    return key_name(idx % 12, "major" if idx < 12 else "minor")


def _predict_abs(model, tokens, gt, gmode, device):
    """Heuristic + model per-chord ABSOLUTE key idx for a raw token stream, via
    the relative encoding (roots/targets relative to the global tonic ``gt``)."""
    seq, y_rel = tokens_to_rel_example(tokens, gt, gmode)
    pred_rel = predict_sequence(model, seq, device)
    heur = [rel_to_abs_key(r, gt) for r in y_rel]
    pred = [rel_to_abs_key(r, gt) for r in pred_rel]
    return heur, pred, y_rel, pred_rel


def _show_case(model, device):
    gt, gmode = ABF_HOME
    heur, pred, _, _ = _predict_abs(model, ABF_TOKENS, gt, gmode, device)
    print("\n── 'A Beautiful Friendship' section B (home C major) ──")
    print(f"  {'chord':<7} {'heuristic (raw)':<16} {'model':<16}")
    for tok, h, p in zip(ABF_TOKENS, heur, pred):
        print(f"  {tok:<7} {_knm(h):<16} {_knm(p):<16}")
    print(f"  collection changes:  heuristic={count_collection_changes(heur)}   "
          f"model={count_collection_changes(pred)}")

    # transpose-equivariance demonstration: SAME motif seeded in E major (+4).
    from harmonia.theory.local_key import transpose_token
    abf_e = [transpose_token(t, 4, flats=False) for t in ABF_TOKENS]
    _, pred_e, pr_c, pr_e = _predict_abs(model, abf_e, 4, "major", device)
    print("\n── equivariance check: same motif in E major (+4) ──")
    print(f"  relative preds identical across keys: {pr_c == pr_e}")
    print(f"  C-major model:  {[_knm(p) for p in pred]}")
    print(f"  E-major model:  {[_knm(p) for p in pred_e]}")
    return heur, pred


def _collection_map(device: str) -> torch.Tensor:
    """(24,12) 0/1 matrix folding a key-idx softmax into collection mass."""
    M = torch.zeros(24, 12)
    for k in range(24):
        M[k, collection_of(k)] = 1.0
    return M.to(device)


def _churn_penalty(logits, targets, coll_map):
    """Soft expected collection-change rate of the predicted sequence.

    Folds the per-position softmax into a 12-way collection distribution q_t and
    penalises ``1 - <q_{t-1}, q_t>`` across valid adjacent pairs. Minimised when
    consecutive chords agree on their collection — so it removes *weakly
    supported* flips (a fleeting secondary dominant) first, while a
    strongly-supported jump (Gm7's Bb in C major) survives because the
    cross-entropy term resists moving it. Weight 0 ⇒ pure distillation.
    """
    p = logits.softmax(-1)                       # (B,T,24)
    q = p @ coll_map                             # (B,T,12)
    valid = (targets != PAD_KEY)                 # (B,T)
    pair = valid[:, 1:] & valid[:, :-1]          # (B,T-1)
    agree = (q[:, 1:] * q[:, :-1]).sum(-1)       # (B,T-1) = <q_t, q_{t-1}>
    change = (1.0 - agree) * pair
    denom = pair.sum().clamp(min=1)
    return change.sum() / denom


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--churn-weight", type=float, default=0.0,
                    help="weight on the soft collection-churn penalty (0 = pure "
                         "distillation; ~0.5-1.5 smooths dominant chains)")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    print("Building per-chord heuristic-distillation dataset...")
    ex = build_seq_examples(DEFAULT_DB)
    train, val = split_seq_examples(ex)
    n_chords = sum(len(e["seq"]) for e in ex)
    print(f"  songs={len(ex)}  chords={n_chords}  "
          f"train={len(train)} songs / val={len(val)} songs")
    for g in ("pop", "jazz"):
        gv = [e for e in val if _corpus_group(e["corpus"]) == g]
        print(f"    val {g}: {len(gv)} songs, "
              f"{sum(len(e['seq']) for e in gv)} chords")

    device = args.device
    model = LocalKeySeqGRU().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    lossf = torch.nn.CrossEntropyLoss(ignore_index=PAD_KEY)
    coll_map = _collection_map(device)
    print(f"churn-weight (smoothing) = {args.churn_weight}")

    best, best_state = -1.0, None
    for ep in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train)
        tot_loss = 0.0
        for i in range(0, len(train), 64):
            chunk = train[i:i + 64]
            # No transpose augmentation: the relative-to-global encoding
            # (local_key_seq_data.tokens_to_rel_example) is transpose-equivariant
            # by construction, so augmentation would be a literal no-op.
            batch = [(e["seq"], e["y"]) for e in chunk]
            root, qual, lengths, targets = collate(batch, device)
            logits = model(root, qual, lengths)              # (B,T,24)
            loss = lossf(logits.reshape(-1, 24), targets.reshape(-1))
            if args.churn_weight > 0:
                loss = loss + args.churn_weight * _churn_penalty(
                    logits, targets, coll_map)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += loss.item() * len(chunk)
        res = evaluate(model, val, device)
        macro = (res["pop"]["acc"] + res["jazz"]["acc"]) / 2
        if macro > best:
            best = macro
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 5 == 0 or ep == 1:
            print(f"  ep {ep:>3}  loss {tot_loss/max(len(train),1):.3f}  "
                  f"pop acc {res['pop']['acc']:.1%}  jazz acc {res['jazz']['acc']:.1%}")

    # With a churn penalty the smoothing is the point, so a best-by-accuracy
    # pick would just select the least-smoothed epoch; use the converged final
    # weights instead. Pure distillation (weight 0) keeps best-by-accuracy.
    if args.churn_weight == 0:
        model.load_state_dict(best_state)
    res = evaluate(model, val, device)
    print("\n=== BEST MODEL (per-position key accuracy vs heuristic teacher) ===")
    for g in ("pop", "jazz"):
        r = res[g]
        print(f"  {g:<5}  acc {r['acc']:.1%}  (n_chord={r['n_chord']}, "
              f"{r['n_seq']} songs)")
    print("\n=== CHURN: collection changes / 100 chords (val) ===")
    for g in ("pop", "jazz"):
        r = res[g]
        delta = r["model_churn"] - r["heur_churn"]
        print(f"  {g:<5}  heuristic {r['heur_churn']:.2f}  →  model {r['model_churn']:.2f}  "
              f"({delta:+.2f})")

    _show_case(model, device)

    if not args.no_save:
        CKPT.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "hparams": {}}, CKPT)
        print(f"\nsaved -> {CKPT}")


if __name__ == "__main__":
    main()
