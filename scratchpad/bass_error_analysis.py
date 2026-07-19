"""Systematic ERROR ANALYSIS of the current bass-pc / root / quality heads on RWC BP48.

Read-only on data/cache/rwc/rwc_bp48.npz. Reproduces the three heads from
bass_inversion_cv.py (byte-compatible training via train_real_audio_final helpers),
pools TEST predictions over N seeds, and computes structured error breakdowns to
find PATTERNS (not a diffuse rate):

  BASS-PC head (12-way, inversion chords only):
    - 12x12 confusion (true sounding-bass pc -> predicted)
    - error INTERVAL histogram (pred-true mod 12): fifth? third? semitone?
    - where errors land relative to FUNCTIONAL ROOT (drift-to-root check)
    - error rate by chord DURATION quartile (short/fast vs long/sustained)
    - error rate by LOCAL DENSITY (chords-per-10s in the song)
    - error rate per true bass pc
    - per-song error rate (data-quality vs model)

  ROOT head (12-way, all chords): error interval histogram on inversions vs
    root-position; bass-landing fraction (reproduce + characterise the rest).

  QUALITY head (7-way family): confusion matrix + per-class recall.
"""
from __future__ import annotations
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from harmonia.data.corpus_schema import MatchQuality, filter_by_match, load_corpus
from train_real_audio_final import _train_head, _augment_root_by_roll, QUALITIES

BASS_SEMI = {"b2":1,"2":2,"b3":3,"3":4,"4":5,"b5":6,"5":7,"b6":8,"6":9,"b7":10,"7":11,"b9":1,"9":2}
NOTE = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]


def derive_bass(label, root):
    label = label.strip()
    if "/" not in label or label in ("N","X",""):
        return 0, -1
    b = label.split("/",1)[1].strip()
    if b not in BASS_SEMI:
        return 0, -1
    return 1, (root + BASS_SEMI[b]) % 12


def _logits(X, model, mean, std, device):
    import torch
    Xn = ((X - mean)/std).astype(np.float32)
    with torch.no_grad():
        return model(torch.tensor(Xn, device=device)).cpu().numpy()


def main():
    import os
    dev = "cpu"  # small MLP + per-batch .item() sync -> CPU beats MPS here
    SEEDS = int(os.environ.get("SEEDS", "2"))
    EP = int(os.environ.get("EP", "30")); BS = int(os.environ.get("BS", "512"))
    d = load_corpus(REPO/"data/cache/rwc/rwc_bp48.npz")
    labels = d["labels"]; roots_all = d["root"].astype(int)
    is_inv = np.zeros(len(labels), int); bass_pc = np.full(len(labels), -1, int)
    for i, lab in enumerate(labels):
        iv, b = derive_bass(str(lab), roots_all[i])
        is_inv[i] = iv; bass_pc[i] = b if b >= 0 else 0

    keep = filter_by_match(d["match"], minimum=MatchQuality.EXACT)
    feat = d["feat48_abs"][keep]; featrel = d["feat48"][keep]
    roots = roots_all[keep]; qidx = d["quality_idx"].astype(int)[keep]
    song = d["song_id"][keep]; inv = is_inv[keep]; bpc = bass_pc[keep]
    t0 = d["t0"][keep]; t1 = d["t1"][keep]; dur = t1 - t0

    # local density: chords per 10s within each song
    dens = np.zeros(len(song))
    for s in set(song.tolist()):
        m = song == s
        span = max(t1[m].max() - t0[m].min(), 1e-3)
        dens[m] = m.sum() / span * 10.0

    songs = sorted(set(song.tolist()))
    print(f"RWC: {len(labels)} recs, EXACT {keep.sum()}, inversions {inv.sum()} ({100*inv.mean():.1f}%), dev={dev}")

    # pooled test collectors
    B = defaultdict(list)  # bass-pc head on true-inversion test chords
    Rinv = defaultdict(list)  # root head, inversion test chords
    Rrp = defaultdict(list)   # root head, root-pos test chords
    Q = defaultdict(list)     # quality head

    for seed in range(SEEDS):
        rng = np.random.RandomState(seed); sh = list(songs); rng.shuffle(sh)
        n_test = max(1, round(0.2*len(sh))); test = set(sh[:n_test])
        tr = np.array([s not in test for s in song]); te = ~tr

        # root head (roll-aug, abs)
        rm, rmu, rsd = _train_head(feat[tr], roots[tr], 12, epochs=EP, lr=3e-4, batch=BS, device=dev, head_name="root")
        rpred = _logits(feat[te], rm, rmu, rsd, dev).argmax(1)
        # quality head (root-relative)
        qm, qmu, qsd = _train_head(featrel[tr], qidx[tr], 7, epochs=EP, lr=3e-4, batch=BS, device=dev, head_name="qual")
        qpred = _logits(featrel[te], qm, qmu, qsd, dev).argmax(1)
        # bass-pc head (inversion-only train, abs)
        itr = tr & (inv == 1)
        bm, bmu, bsd = _train_head(feat[itr], bpc[itr], 12, epochs=EP, lr=3e-4, batch=BS, device=dev, head_name="basspc")
        bpred = _logits(feat[te], bm, bmu, bsd, dev).argmax(1)

        te_idx = np.where(te)[0]
        iv_te = inv[te] == 1
        # bass-pc: only true inversions
        for j, gi in enumerate(te_idx):
            if iv_te[j]:
                B["true"].append(bpc[te][j]); B["pred"].append(bpred[j]); B["root"].append(roots[te][j])
                B["dur"].append(dur[te][j]); B["dens"].append(dens[te][j]); B["song"].append(song[te][j])
        # root head
        for j in range(len(te_idx)):
            tgt = "inv" if iv_te[j] else "rp"
            (Rinv if iv_te[j] else Rrp)["true"].append(roots[te][j])
            (Rinv if iv_te[j] else Rrp)["pred"].append(rpred[j])
            if iv_te[j]:
                Rinv["bass"].append(bpc[te][j])
        # quality
        Q["true"].extend(qidx[te].tolist()); Q["pred"].extend(qpred.tolist())
        print(f"  seed {seed}: bass-pc n={iv_te.sum()} acc={(bpred[iv_te]==bpc[te][iv_te]).mean():.3f}")

    # ---------- BASS-PC ANALYSIS ----------
    bt = np.array(B["true"]); bp = np.array(B["pred"]); br = np.array(B["root"])
    bd = np.array(B["dur"]); bde = np.array(B["dens"]); bs = np.array(B["song"])
    ok = bt == bp
    print(f"\n{'='*72}\nBASS-PC head: pooled {len(bt)} true-inversion test chords, acc={ok.mean():.3f}")
    # error interval hist
    err = ~ok
    ivl = (bp[err] - bt[err]) % 12
    print("Error INTERVAL (pred-true mod12) distribution:")
    for s in range(12):
        c = (ivl == s).sum()
        if c: print(f"   +{s:2d} ({NOTE[s]:>2} above): {c:4d}  {100*c/err.sum():5.1f}%")
    # relation to functional root
    print(f"  bass-pc errors that land on FUNCTIONAL ROOT: {100*(bp[err]==br[err]).mean():.1f}%")
    print(f"  bass-pc errors on ROOT+7 (fifth of root):    {100*(bp[err]==(br[err]+7)%12).mean():.1f}%")
    # by duration quartile
    qs = np.quantile(bd, [0,.25,.5,.75,1.0])
    print("Error rate by DURATION quartile:")
    for i in range(4):
        m = (bd>=qs[i]) & (bd<=qs[i+1]) if i==3 else (bd>=qs[i]) & (bd<qs[i+1])
        print(f"   Q{i+1} [{qs[i]:.2f},{qs[i+1]:.2f}]s n={m.sum():4d} err={100*(~ok[m]).mean():5.1f}%")
    # by density
    ds = np.quantile(bde, [0,.5,1.0])
    for lo,hi,lbl in [(ds[0],ds[1],"low-density (sustained)"),(ds[1],ds[2]+1,"high-density (fast)")]:
        m = (bde>=lo)&(bde<hi)
        print(f"   {lbl:26s} n={m.sum():4d} err={100*(~ok[m]).mean():5.1f}%")
    # by true bass pc
    print("Error rate by TRUE bass pc:")
    for pc in range(12):
        m = bt==pc
        if m.sum(): print(f"   {NOTE[pc]:>2}: n={m.sum():4d} err={100*(~ok[m]).mean():5.1f}%")
    # per-song worst
    perr = []
    for s in set(bs.tolist()):
        m = bs==s
        if m.sum()>=8: perr.append((100*(~ok[m]).mean(), m.sum(), s))
    perr.sort(reverse=True)
    print("Worst 6 songs (>=8 inv chords) by bass-pc err:")
    for e,n,s in perr[:6]: print(f"   {s}: n={n} err={e:.1f}%")
    print(f"   [{len(perr)} songs; median song err {np.median([e for e,_,_ in perr]):.1f}%]")

    # ---------- ROOT ANALYSIS ----------
    for name, R in [("INVERSION", Rinv), ("ROOT-POS", Rrp)]:
        rt = np.array(R["true"]); rp_ = np.array(R["pred"]); e = rt != rp_
        print(f"\nROOT head on {name} test chords: n={len(rt)} acc={ (rt==rp_).mean():.3f} err={e.sum()}")
        ivl = (rp_[e]-rt[e])%12
        top = sorted([(100*(ivl==s).sum()/max(e.sum(),1), s) for s in range(12)], reverse=True)[:4]
        print("   top error intervals: " + ", ".join(f"+{s}({NOTE[s]}) {p:.0f}%" for p,s in top))
        if name=="INVERSION":
            bass = np.array(R["bass"])
            print(f"   root errors landing on SOUNDING BASS: {100*(rp_[e]==bass[e]).mean():.1f}%")

    # ---------- QUALITY ANALYSIS ----------
    qt = np.array(Q["true"]); qp = np.array(Q["pred"])
    print(f"\nQUALITY head: n={len(qt)} acc={(qt==qp).mean():.3f}")
    print("   per-class recall:")
    for i,q in enumerate(QUALITIES):
        m = qt==i
        if m.sum():
            # top confusion target
            wrong = qp[m & (qp!=qt)] if False else qp[(qt==i)&(qp!=i)]
            tgt = ""
            if len(wrong): tgt = QUALITIES[np.bincount(wrong,minlength=7).argmax()]
            print(f"     {q:>4}: n={m.sum():5d} recall={100*(qp[m]==i).mean():5.1f}%  ->{tgt}")


if __name__ == "__main__":
    main()
