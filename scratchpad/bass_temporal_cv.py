"""Temporal bass/inversion model on RWC frame-level BP48 vs the pooled baseline.

Prior pooled result (docs/known_issues.md, full 100-song corpus):
  bass-PC acc on true inversions 66.4%+/-7.1% (12-way, chance 8.3%)
  inversion-detector precision 20.4%, recall 40.0%  <- THE bottleneck
  oracle-inv redirect +13.5pp on inversion root acc, but BLIND redirect net-loss
  because the 20% precision detector fires on root-position chords.

This script tests whether FRAME-LEVEL temporal input (bass note trajectory over
and around the chord span) beats a single pooled snapshot -- specifically for
INVERSION DETECTION precision, the identified bottleneck.

Arms, same song-stratified CV splits & seeds:
  ROOT head      : pooled feat48_abs, roll-augmented (baseline root arm, the gate's root)
  POOLED inv/bass: MLP on pooled bass-block (reproduces the prior baseline on THIS subset)
  TEMPORAL inv/bass: small bi-GRU over the per-frame chroma sequence in [t0-ctx,t1+ctx)

Then the root-redirection GATE (redirect root off the predicted bass-pc when the
inversion head fires) is re-run using the TEMPORAL detector; we report whether a
better detector makes the blind gate net-positive (prior: net-negative).

Read-only on scratchpad/bass_temporal/bass_temporal_frames.npz. Writes nothing shared.
"""
from __future__ import annotations
import argparse, sys, json
from pathlib import Path
import numpy as np

REPO = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from train_real_audio_final import _train_head, _augment_root_by_roll  # noqa

BASS_SEMI = {"b2":1,"2":2,"b3":3,"3":4,"4":5,"b5":6,"5":7,"b6":8,"6":9,"b7":10,"7":11,"b9":1,"9":2}


def derive_bass(label, root):
    label = label.strip()
    if "/" not in label or label in ("N","X",""):
        return 0, -1
    b = label.split("/",1)[1].strip()
    if b not in BASS_SEMI:
        return 0, -1
    return 1, (root + BASS_SEMI[b]) % 12


def norm_blocks(x):
    y = x.reshape(*x.shape[:-1], x.shape[-1]//12, 12)
    n = np.linalg.norm(y, axis=-1, keepdims=True)
    return (y/(n+1e-9)).reshape(x.shape)


def _logits(X, model, mean, std, device):
    import torch
    Xn = ((X-mean)/std).astype(np.float32)
    with torch.no_grad():
        return model(torch.tensor(Xn, device=device)).cpu().numpy()


# ---------------- temporal GRU ----------------
def make_gru(in_dim, hidden):
    import torch, torch.nn as nn
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(in_dim, hidden, batch_first=True, bidirectional=True)
            self.drop = nn.Dropout(0.3)
            self.inv = nn.Linear(4*hidden, 2)
            self.bass = nn.Linear(4*hidden, 12)
        def forward(self, x, lengths):
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = self.gru(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)  # (B,T,2H)
            mask = (torch.arange(out.size(1), device=x.device)[None,:] < lengths[:,None]).float().unsqueeze(-1)
            summ = (out*mask).sum(1)/mask.sum(1).clamp(min=1)          # mean pool
            mx = out.masked_fill(mask==0, -1e9).max(1).values          # max pool
            h = self.drop(torch.cat([summ, mx], -1))
            return self.inv(h), self.bass(h)
    return M()


def collate(seqs, device):
    import torch
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    T = int(lengths.max())
    B = len(seqs); D = seqs[0].shape[1]
    x = np.zeros((B, T, D), np.float32)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = s
    return torch.tensor(x, device=device), lengths.to(device)


def train_temporal(seqs, inv_y, bass_y, *, hidden, epochs, lr, batch, device):
    import torch, torch.nn as nn
    model = make_gru(seqs[0].shape[1], hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    # balanced inv weights
    cnt = np.bincount(inv_y, minlength=2).astype(float); w = 1.0/(cnt+1.0); w/=w.sum(); w*=2
    wt = torch.tensor(w, dtype=torch.float32, device=device)
    inv_t = torch.tensor(inv_y, dtype=torch.long, device=device)
    bass_t = torch.tensor(np.where(bass_y>=0, bass_y, 0), dtype=torch.long, device=device)
    bass_mask = torch.tensor(inv_y==1, dtype=torch.bool, device=device)
    n = len(seqs)
    for ep in range(epochs):
        model.train(); perm = np.random.permutation(n)
        for i in range(0, n, batch):
            idx = perm[i:i+batch]
            x, lens = collate([seqs[j] for j in idx], device)
            li, lb = model(x, lens)
            loss = nn.functional.cross_entropy(li, inv_t[idx], weight=wt)
            bm = bass_mask[idx]
            if bm.any():
                loss = loss + nn.functional.cross_entropy(lb[bm], bass_t[idx][bm])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    model.eval()
    return model


def predict_temporal(model, seqs, device, batch=128):
    import torch
    inv_p, bass_p = [], []
    with torch.no_grad():
        for i in range(0, len(seqs), batch):
            x, lens = collate(seqs[i:i+batch], device)
            li, lb = model(x, lens)
            inv_p.append(li.argmax(1).cpu().numpy()); bass_p.append(lb.argmax(1).cpu().numpy())
    return np.concatenate(inv_p), np.concatenate(bass_p)


# ---------------- data ----------------
def _downsample(seg, ds):
    """Mean-pool non-overlapping windows of ds frames -> ~86/ds Hz. Denoises BP
    onset jitter and shortens the GRU sequence (bass movement is slow)."""
    if ds <= 1 or len(seg) < ds:
        return seg
    n = (len(seg)//ds)*ds
    return seg[:n].reshape(-1, ds, seg.shape[1]).mean(1)


def build_chord_data(npz, ctx, feat_slice, max_frames, ds=4):
    d = np.load(npz, allow_pickle=True)
    ft = d["frame_times"]; fe = d["frame_feats"].astype(np.float32)
    sids = d["song_ids"]; off = d["song_off"]
    soff = {s: (int(a), int(b)) for s, (a, b) in zip(sids, off)}
    c_song = d["c_song"]; c_t0 = d["c_t0"]; c_t1 = d["c_t1"]; c_root = d["c_root"].astype(int)
    c_label = d["c_label"]
    inv = np.zeros(len(c_song), int); bpc = np.full(len(c_song), -1, int)
    for i, lab in enumerate(c_label):
        iv, b = derive_bass(str(lab), c_root[i]); inv[i] = iv; bpc[i] = b
    seqs = []; pooled = []; keep = np.ones(len(c_song), bool)
    for i in range(len(c_song)):
        a, b = soff[c_song[i]]
        fts = ft[a:b]; fes = fe[a:b]
        lo = np.searchsorted(fts, c_t0[i]-ctx, "left")
        hi = np.searchsorted(fts, c_t1[i]+ctx, "left")
        # in-span (no ctx) for pooled feature = match fixed corpus pooling
        slo = np.searchsorted(fts, c_t0[i], "left"); shi = np.searchsorted(fts, c_t1[i], "left")
        if shi <= slo:  # floor to >=1 frame around midpoint
            c = np.searchsorted(fts, 0.5*(c_t0[i]+c_t1[i]), "left"); slo = max(0, c); shi = slo+1
        pooled.append(norm_blocks(fes[slo:shi].sum(0)))
        seg = fes[lo:hi] if hi > lo else fes[slo:shi]
        if len(seg) == 0:
            keep[i] = False; seqs.append(np.zeros((1, 48), np.float32)); continue
        seg = _downsample(seg, ds)
        if len(seg) > max_frames:
            sel = np.linspace(0, len(seg)-1, max_frames).round().astype(int); seg = seg[sel]
        # per-frame L2-norm each 12-block so bass trajectory is a clean chroma path
        seg = norm_blocks(seg)[:, feat_slice]
        seqs.append(seg.astype(np.float32))
    pooled = np.stack(pooled).astype(np.float32)
    return dict(seqs=seqs, pooled=pooled, song=c_song, root=c_root, inv=inv,
                bpc=bpc, keep=keep)


def metrics_inv(inv_pred, inv_true):
    tp = ((inv_pred==1)&(inv_true==1)).sum(); fp=((inv_pred==1)&(inv_true==0)).sum()
    fn = ((inv_pred==0)&(inv_true==1)).sum()
    rec = tp/(tp+fn) if (tp+fn) else 0.0; pre = tp/(tp+fp) if (tp+fp) else 0.0
    return float(rec), float(pre)


def one_split(D, seed, *, epochs, lr, batch, device, hidden, feat_dim):
    keep = D["keep"]
    idx_all = np.where(keep)[0]
    song = D["song"]; songs = sorted(set(song[keep].tolist()))
    rng = np.random.RandomState(seed); rng.shuffle(songs)
    n_test = max(1, int(round(0.2*len(songs)))); test_songs = set(songs[:n_test])
    tr = np.array([keep[i] and song[i] not in test_songs for i in range(len(song))])
    te = np.array([keep[i] and song[i] in test_songs for i in range(len(song))])
    tr_i = np.where(tr)[0]; te_i = np.where(te)[0]
    root = D["root"]; inv = D["inv"]; bpc = np.where(D["bpc"]>=0, D["bpc"], 0)
    pooled = D["pooled"]; seqs = D["seqs"]

    res = {}
    # ROOT head (pooled feat48_abs, roll aug) -- for the gate
    Xtr, ytr = _augment_root_by_roll(pooled[tr_i], root[tr_i])
    rm, rmean, rstd = _train_head(Xtr, ytr, 12, epochs=epochs, lr=lr, batch=batch, device=device, head_name="root")
    root_logits = _logits(pooled[te_i], rm, rmean, rstd, device); root_pred = root_logits.argmax(1)

    # POOLED inv + bass heads (bass block 24:36) -- baseline reference on this subset
    Xb = pooled[:, 24:36]
    pim, pm, ps = _train_head(Xb[tr_i], inv[tr_i], 2, epochs=epochs, lr=lr, batch=batch, device=device, head_name="pinv")
    pinv_pred = _logits(Xb[te_i], pim, pm, ps, device).argmax(1)
    inv_tr = tr_i[inv[tr_i]==1]
    pbm, pbmean, pbstd = _train_head(Xb[inv_tr], bpc[inv_tr], 12, epochs=epochs, lr=lr, batch=batch, device=device, head_name="pbass")
    pbass_pred = _logits(Xb[te_i], pbm, pbmean, pbstd, device).argmax(1)

    # TEMPORAL inv + bass (shared GRU, multitask)
    tm = train_temporal([seqs[i] for i in tr_i], inv[tr_i], D["bpc"][tr_i],
                        hidden=hidden, epochs=epochs, lr=lr, batch=batch, device=device)
    tinv_pred, tbass_pred = predict_temporal(tm, [seqs[i] for i in te_i], device)

    root_te = root[te_i]; inv_te = inv[te_i]; bpc_te = bpc[te_i]
    is_iv = inv_te==1; is_rp = inv_te==0

    res["n_inv_te"] = int(is_iv.sum()); res["n_rp_te"] = int(is_rp.sum())
    res["root_acc_inv"] = float((root_pred[is_iv]==root_te[is_iv]).mean()) if is_iv.sum() else 0.0
    res["root_acc_rootpos"] = float((root_pred[is_rp]==root_te[is_rp]).mean()) if is_rp.sum() else 0.0
    inv_err = is_iv & (root_pred!=root_te)
    res["err_on_bass"] = float((root_pred[inv_err]==bpc_te[inv_err]).mean()) if inv_err.sum() else 0.0

    # pooled detector + bass
    res["pool_inv_rec"], res["pool_inv_pre"] = metrics_inv(pinv_pred, inv_te)
    res["pool_bass_acc"] = float((pbass_pred[is_iv]==bpc_te[is_iv]).mean()) if is_iv.sum() else 0.0
    # temporal detector + bass
    res["tmp_inv_rec"], res["tmp_inv_pre"] = metrics_inv(tinv_pred, inv_te)
    res["tmp_bass_acc"] = float((tbass_pred[is_iv]==bpc_te[is_iv]).mean()) if is_iv.sum() else 0.0
    res["tmp_bass_eq_root"] = float((tbass_pred[is_iv]==root_te[is_iv]).mean()) if is_iv.sum() else 0.0

    # ---- GATE re-test with TEMPORAL detector+bass (blind) ----
    def gate(det_pred, bass_pred, oracle=False):
        corrected = root_pred.copy()
        fire = ((inv_te==1) if oracle else (det_pred==1)) & (root_pred==bass_pred)
        if fire.sum():
            rows = np.where(fire)[0]; masked = root_logits.copy()
            masked[rows, bass_pred[rows]] = -1e9; corrected[rows] = masked[rows].argmax(1)
        ai = float((corrected[is_iv]==root_te[is_iv]).mean()) if is_iv.sum() else 0.0
        arp = float((corrected[is_rp]==root_te[is_rp]).mean()) if is_rp.sum() else 0.0
        aall = float((corrected==root_te).mean())
        return ai, arp, aall, int(fire.sum()), int((fire&is_iv).sum())

    res["base_root_all"] = float((root_pred==root_te).mean())
    ai,arp,aall,nf,nfi = gate(tinv_pred, tbass_pred, oracle=False)
    res["tmp_gate_inv"]=ai; res["tmp_gate_rp"]=arp; res["tmp_gate_all"]=aall; res["tmp_fire"]=nf; res["tmp_fire_inv"]=nfi
    aio,_,_,_,_ = gate(tbass_pred, tbass_pred, oracle=True)  # oracle inv gate w/ temporal bass
    # oracle uses inv_te internally; det_pred arg ignored
    ao_i, ao_rp, ao_all, _, _ = gate(tinv_pred, tbass_pred, oracle=True)
    res["tmp_gate_inv_oracle"]=ao_i
    # pooled blind gate (reference)
    pi,prp,pall,pnf,pnfi = gate(pinv_pred, pbass_pred, oracle=False)
    res["pool_gate_inv"]=pi; res["pool_gate_rp"]=prp; res["pool_gate_all"]=pall
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=str(REPO/"scratchpad/bass_temporal/bass_temporal_frames.npz"))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=24)
    ap.add_argument("--ctx", type=float, default=0.4, help="seconds of cross-chord context each side")
    ap.add_argument("--feat", choices=["full","bassnote","bass"], default="full")
    ap.add_argument("--max-frames", type=int, default=120)
    ap.add_argument("--ds", type=int, default=4, help="frame downsample factor (mean-pool)")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    dev = a.device
    if dev is None:
        try:
            import torch; dev = "mps" if torch.backends.mps.is_available() else "cpu"
        except Exception:
            dev = "cpu"

    fslice = {"full": slice(0,48), "bassnote": np.r_[12:24,24:36], "bass": slice(24,36)}[a.feat]
    feat_dim = 48 if a.feat=="full" else (24 if a.feat=="bassnote" else 12)

    D = build_chord_data(a.npz, a.ctx, fslice, a.max_frames, ds=a.ds)
    nk = D["keep"].sum(); ninv = D["inv"][D["keep"]].sum()
    print(f"corpus: {nk} chords kept, {ninv} inversions ({100*ninv/nk:.1f}%), "
          f"feat={a.feat}({feat_dim}d) ctx={a.ctx}s hidden={a.hidden} dev={dev}", flush=True)

    runs = []
    for s in range(a.seeds):
        print(f"--- seed {s} ---", flush=True)
        r = one_split(D, s, epochs=a.epochs, lr=a.lr, batch=a.batch, device=dev,
                      hidden=a.hidden, feat_dim=feat_dim)
        runs.append(r)
        print(f"  POOLED inv P/R={r['pool_inv_pre']:.3f}/{r['pool_inv_rec']:.3f} bass={r['pool_bass_acc']:.3f} "
              f"| TEMPORAL inv P/R={r['tmp_inv_pre']:.3f}/{r['tmp_inv_rec']:.3f} bass={r['tmp_bass_acc']:.3f}", flush=True)
        print(f"    root_all base={r['base_root_all']:.3f} | TMP gate: inv {r['root_acc_inv']:.3f}->{r['tmp_gate_inv']:.3f} "
              f"rp {r['root_acc_rootpos']:.3f}->{r['tmp_gate_rp']:.3f} all {r['base_root_all']:.3f}->{r['tmp_gate_all']:.3f} "
              f"(oracle inv {r['tmp_gate_inv_oracle']:.3f})", flush=True)

    def ms(k):
        v = np.array([r[k] for r in runs]); return v.mean(), v.std()
    print("\n"+"="*74)
    print(f"RWC TEMPORAL bass/inversion, {a.seeds} seeds, feat={a.feat}, ctx={a.ctx}s")
    print(f"  mean inv/rp test chords: {np.mean([r['n_inv_te'] for r in runs]):.0f}/{np.mean([r['n_rp_te'] for r in runs]):.0f}")
    rows = [
      ("POOLED inv precision (prior 0.20)", "pool_inv_pre"),
      ("POOLED inv recall   (prior 0.40)", "pool_inv_rec"),
      ("POOLED bass-pc acc  (prior 0.664)", "pool_bass_acc"),
      ("TEMPORAL inv precision", "tmp_inv_pre"),
      ("TEMPORAL inv recall", "tmp_inv_rec"),
      ("TEMPORAL bass-pc acc", "tmp_bass_acc"),
      ("  (temporal bass==root sanity, ~chance)", "tmp_bass_eq_root"),
      ("root acc all (base)", "base_root_all"),
      ("root acc inversion (base)", "root_acc_inv"),
      ("root acc inversion AFTER temporal blind gate", "tmp_gate_inv"),
      ("root acc inversion AFTER oracle-inv gate", "tmp_gate_inv_oracle"),
      ("root acc root-pos AFTER temporal gate (must not drop)", "tmp_gate_rp"),
      ("root acc ALL AFTER temporal gate (net)", "tmp_gate_all"),
      ("root acc ALL AFTER pooled gate (ref)", "pool_gate_all"),
    ]
    for lbl, k in rows:
        m, sd = ms(k); print(f"  {lbl:52s}: {m:.3f} +/- {sd:.3f}")
    print(f"  temporal gate fires: {np.mean([r['tmp_fire'] for r in runs]):.0f}/split "
          f"({np.mean([r['tmp_fire_inv'] for r in runs]):.0f} on true inversions)")
    # save
    out = REPO/"scratchpad/bass_temporal"/f"cv_result_{a.feat}_ctx{a.ctx}.json"
    out.write_text(json.dumps({k:[float(r[k]) for r in runs] for k in runs[0]}, indent=1))
    print(f"saved {out.name}")


if __name__ == "__main__":
    main()
