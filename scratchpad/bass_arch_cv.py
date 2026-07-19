"""Architecture sweep for the SIMPLE unconditional bass-PC head (2026-07-16).

Same 9-vector aggregated context input as bass_simple_cv.py (current chord + 4
before + 4 after; each 48-dim pooled feat48_abs) — ONLY the model that
processes it varies. NOT raw frame sequences (that was the failed v2). The 9
slots form a short sequence of length 9, feature dim 48, in chord order
[-4..-1, cur, +1..+4].

Architectures (all kept SMALL — bigger capacity hurts on this corpus size):
  MLP  : flatten 432 -> 128 -> 64 -> 12   (reference, matches bass_simple)
  CNN  : Conv1d over the 9-position axis (48 channels) -> global pool -> 12
  LSTM : 1-layer LSTM over 9 timesteps (48-dim) -> last hidden -> 12
  RNN  : 1-layer vanilla RNN (tanh) over 9 timesteps -> last hidden -> 12

Normalization variants (reuse bass_simple helpers):
  RAW    : absolute chroma, target = absolute bass pc (roll-augmented)
  RENORM : root-anchored context, target = bass-relative-to-root (no aug)

Read-only corpus. Small JSON out.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from harmonia.data.corpus_schema import MatchQuality, filter_by_match, load_corpus
from bass_simple_cv import (derive_bass_target, build_context, anchor_to_root, aug_bass)

N_SLOT, FEAT = 9, 48


def make_model(kind, n_classes=12):
    import torch.nn as nn
    if kind == "mlp":
        return nn.Sequential(
            nn.Linear(N_SLOT*FEAT, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(64, n_classes))

    class CNN(nn.Module):
        def __init__(s):
            super().__init__()
            s.c1 = nn.Conv1d(FEAT, 32, 3, padding=1); s.c2 = nn.Conv1d(32, 32, 3, padding=1)
            s.ln = nn.LayerNorm(32); s.do = nn.Dropout(0.3); s.fc = nn.Linear(32, n_classes)
            s.act = nn.GELU()
        def forward(s, x):
            x = x.view(-1, N_SLOT, FEAT).transpose(1, 2)      # (b, 48, 9)
            x = s.act(s.c1(x)); x = s.act(s.c2(x))
            x = x.mean(-1)                                    # global avg pool over positions
            return s.fc(s.do(s.act(s.ln(x))))

    class Seq(nn.Module):
        def __init__(s, cell):
            super().__init__()
            rnn = nn.LSTM if cell == "lstm" else nn.RNN
            s.rnn = rnn(FEAT, 32, batch_first=True)
            s.ln = nn.LayerNorm(32); s.do = nn.Dropout(0.3); s.fc = nn.Linear(32, n_classes)
        def forward(s, x):
            x = x.view(-1, N_SLOT, FEAT)                      # (b, 9, 48)
            out, _ = s.rnn(x)
            h = out[:, -1, :]                                 # last timestep (current+4)
            return s.fc(s.do(s.ln(h)))

    if kind == "cnn":  return CNN()
    if kind == "lstm": return Seq("lstm")
    if kind == "rnn":  return Seq("rnn")
    raise ValueError(kind)


def train_head(X, y, kind, *, epochs, lr, batch, device, n_classes=12):
    import torch, torch.nn as nn
    mean = X.mean(0).astype(np.float32); std = (X.std(0) + 1e-9).astype(np.float32)
    Xn = ((X - mean) / std).astype(np.float32)
    Xt = torch.tensor(Xn, device=device); yt = torch.tensor(y, dtype=torch.long, device=device)
    counts = np.bincount(y, minlength=n_classes).astype(float)
    w = 1.0/(counts+1.0); w /= w.sum(); w *= n_classes
    wt = torch.tensor(w, dtype=torch.float32, device=device)
    model = make_model(kind, n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss(weight=wt)
    n = len(Xt)
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n, device=device)
        for i in range(0, n, batch):
            idx = perm[i:i+batch]; opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx]); loss.backward(); opt.step()
        sched.step()
    model.eval()
    return model, mean, std


def predict(X, model, mean, std, device):
    import torch
    Xn = ((X - mean) / std).astype(np.float32)
    with torch.no_grad():
        return model(torch.tensor(Xn, device=device)).argmax(1).cpu().numpy()


def run(d, is_inv, bass_abs, Xctx, roots_all, *, kinds, variant, seeds, epochs, lr, batch, device):
    keep = filter_by_match(d["match"], minimum=MatchQuality.EXACT)
    roots = roots_all[keep]; song_id = d["song_id"][keep]
    inv = is_inv[keep]; babs = bass_abs[keep]; Xc = Xctx[keep]
    if variant == "renorm":
        Xc = anchor_to_root(Xc, roots); y = (babs - roots) % 12
    else:
        y = babs.copy()

    out = {k: {"all": [], "rp": [], "inv": []} for k in kinds}
    for s in range(seeds):
        songs = sorted(set(song_id.tolist()))
        rng = np.random.RandomState(s); rng.shuffle(songs)
        n_test = max(1, int(round(0.2*len(songs)))); test_songs = set(songs[:n_test])
        tr = np.array([x not in test_songs for x in song_id]); te = ~tr
        inv_te = inv[te]; is_rp = inv_te == 0; is_iv = inv_te == 1
        root_te = roots[te]; bass_te = babs[te]
        if variant == "raw":
            Xtr, ytr = aug_bass(Xc[tr], y[tr])
        else:
            Xtr, ytr = Xc[tr], y[tr]
        for k in kinds:
            m, mn, sd = train_head(Xtr, ytr, k, epochs=epochs, lr=lr, batch=batch, device=device)
            pred = predict(Xc[te], m, mn, sd, device)
            abs_pred = (pred + root_te) % 12 if variant == "renorm" else pred
            out[k]["all"].append(float((abs_pred == bass_te).mean()))
            out[k]["rp"].append(float((abs_pred[is_rp] == bass_te[is_rp]).mean()))
            out[k]["inv"].append(float((abs_pred[is_iv] == bass_te[is_iv]).mean()) if is_iv.sum() else 0.0)
        print(f"  [{variant} seed {s}] " + " ".join(
            f"{k}:inv={out[k]['inv'][-1]:.3f}" for k in kinds), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=REPO / "data/cache/rwc/rwc_bp48_fixed.npz")
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--variants", default="renorm,raw")
    ap.add_argument("--kinds", default="mlp,cnn,lstm,rnn")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, default=REPO / "scratchpad/bass_arch_result.json")
    a = ap.parse_args()
    dev = a.device
    if dev is None:
        import torch; dev = "mps" if torch.backends.mps.is_available() else "cpu"

    d = load_corpus(a.corpus); labels = d["labels"]; roots_all = d["root"].astype(int)
    is_inv = np.zeros(len(labels), int); bass_abs = np.zeros(len(labels), int)
    for i, lab in enumerate(labels):
        iv, b = derive_bass_target(lab, roots_all[i]); is_inv[i] = iv; bass_abs[i] = b
    Xctx = build_context(d["feat48_abs"], d["song_id"], d["t0"], k=4)
    kinds = a.kinds.split(","); variants = a.variants.split(",")
    print(f"arch sweep: kinds={kinds} variants={variants} seeds={a.seeds} dev={dev}", flush=True)

    results = {}
    for v in variants:
        print(f"=== variant {v} ===", flush=True)
        results[v] = run(d, is_inv, bass_abs, Xctx, roots_all, kinds=kinds, variant=v,
                         seeds=a.seeds, epochs=a.epochs, lr=a.lr, batch=a.batch, device=dev)

    def ms(l): a_=np.array(l); return [float(a_.mean()), float(a_.std())]
    summary = {v: {k: {g: ms(results[v][k][g]) for g in ("all","rp","inv")} for k in kinds} for v in variants}
    a.out.write_text(json.dumps({"summary": summary, "seeds": a.seeds, "corpus": a.corpus.name}, indent=2))

    print("\n" + "="*70)
    print(f"ARCH SWEEP bass-PC acc, {a.seeds}-seed CV ({a.corpus.name})")
    for v in variants:
        print(f"\n--- {v} ---   {'arch':6s} {'all':>14s} {'root-pos':>14s} {'inversions':>16s}")
        for k in kinds:
            r = summary[v][k]
            print(f"  {k:6s}  {r['all'][0]:.3f}+/-{r['all'][1]:.3f}  "
                  f"{r['rp'][0]:.3f}+/-{r['rp'][1]:.3f}  {r['inv'][0]:.3f}+/-{r['inv'][1]:.3f}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
