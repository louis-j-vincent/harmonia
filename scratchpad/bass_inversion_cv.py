"""Dedicated bass/inversion head on RWC BP48, ADDITIVE to root+quality heads.

Task: "Fix le /bass pour que le root modele puisse correctement les classifier."

Design (per task brief, NOT a factored replacement -- the ChordFormer 6-slot
factoring caused a dom-recall collapse; this is narrow + additive):
  * ROOT head: byte-identical to the confirmed baseline (feat48_abs, roll aug,
    12-way). This is the current-production root arm; it is the BEFORE.
  * INVERSION head: binary "is this chord an inversion (has a /bass != root)".
  * BASS-PC head: 12-way absolute sounding-bass pitch-class, trained on
    inversion chords only (conditional on inversion=yes).
  * Both new heads on feat48_abs (absolute frame) -- bass PC is an absolute
    quantity, so the root-relative feat48 is the wrong frame. Ablation: also
    try the BASS BLOCK only (dims 24:36 = onset MIDI<52 register, the most
    bass-informative single block per docs/feature_domain_bridge...).

Hypothesis under test: the root head's errors on inversion chords land
disproportionately on the SOUNDING BASS pc (Billboard: 36%). Does an explicit
bass output let us REDIRECT those errors? Interaction mechanism (blind, no
oracle): if inversion-head fires AND root-argmax == bass-pc-head's prediction,
take the root head's best class != that bass pc. Report root acc on
inversion-only chords BEFORE vs AFTER, and the bass-landing fraction.

Read-only on data/cache/rwc/rwc_bp48.npz. Writes nothing there.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from harmonia.data.corpus_schema import MatchQuality, filter_by_match, load_corpus
from train_real_audio_final import _train_head, _augment_root_by_roll, _standardize

# root-relative bass degree token -> semitones above root
BASS_SEMI = {
    "b2": 1, "2": 2, "b3": 3, "3": 4, "4": 5, "b5": 6,
    "5": 7, "b6": 8, "6": 9, "b7": 10, "7": 11, "b9": 1, "9": 2,
}


def derive_bass(label: str, root: int):
    """(is_inversion, sounding_bass_pc or -1) from a Harte label + functional root."""
    label = label.strip()
    if "/" not in label or label in ("N", "X", ""):
        return 0, -1
    b = label.split("/", 1)[1].strip()
    if b not in BASS_SEMI:
        return 0, -1  # absolute-note bass we can't resolve -> treat as root-pos
    return 1, (root + BASS_SEMI[b]) % 12


def _logits(X, model, mean, std, device):
    import torch
    Xn = ((X - mean) / std).astype(np.float32)
    with torch.no_grad():
        return model(torch.tensor(Xn, device=device)).cpu().numpy()


def one_split(d, seed, is_inv, bass_pc, *, epochs, lr, batch, device, bass_feat, test_frac=0.2):
    keep = filter_by_match(d["match"], minimum=MatchQuality.EXACT)
    feat48_abs = d["feat48_abs"][keep]
    roots = d["root"].astype(int)[keep]
    song_id = d["song_id"][keep]
    inv = is_inv[keep]
    bpc = bass_pc[keep]

    # bass-feature selector for the new heads
    if bass_feat == "block":
        Xbass = feat48_abs[:, 24:36]   # bass block (onset MIDI<52)
    else:
        Xbass = feat48_abs             # full 48-dim absolute

    songs = sorted(set(song_id.tolist()))
    rng = np.random.RandomState(seed); rng.shuffle(songs)
    n_test = max(1, int(round(test_frac * len(songs))))
    test_songs = set(songs[:n_test])
    tr = np.array([s not in test_songs for s in song_id]); te = ~tr

    # ---- ROOT head (baseline arm, roll-augmented feat48_abs) ----
    Xtr, ytr = _augment_root_by_roll(feat48_abs[tr], roots[tr])
    rm, rmean, rstd = _train_head(Xtr, ytr, 12, epochs=epochs, lr=lr, batch=batch,
                                  device=device, head_name="root")
    root_logits = _logits(feat48_abs[te], rm, rmean, rstd, device)
    root_pred = root_logits.argmax(1)

    # ---- INVERSION head (binary) ----
    im, imean, istd = _train_head(Xbass[tr], inv[tr], 2, epochs=epochs, lr=lr,
                                  batch=batch, device=device, head_name="inv")
    inv_pred = _logits(Xbass[te], im, imean, istd, device).argmax(1)

    # ---- BASS-PC head (12-way, trained on inversion chords only) ----
    inv_tr = tr & (is_inv == 1)
    bm, bmean, bstd = _train_head(Xbass[inv_tr], bpc[inv_tr], 12, epochs=epochs,
                                  lr=lr, batch=batch, device=device, head_name="basspc")
    basspc_pred = _logits(Xbass[te], bm, bmean, bstd, device).argmax(1)

    # ---- evaluation ----
    root_te = roots[te]; inv_te = inv[te]; bpc_te = bpc[te]
    rp = root_pred
    is_rp = inv_te == 0           # root-position test chords
    is_iv = inv_te == 1           # inversion test chords

    res = {}
    res["root_acc_all"] = float((rp == root_te).mean())
    res["root_acc_rootpos"] = float((rp[is_rp] == root_te[is_rp]).mean())
    res["root_acc_inv"] = float((rp[is_iv] == root_te[is_iv]).mean()) if is_iv.sum() else 0.0

    # fraction of root ERRORS on inversion chords that land on the sounding bass pc
    inv_err = is_iv & (rp != root_te)
    if inv_err.sum():
        res["inv_err_on_bass_before"] = float((rp[inv_err] == bpc_te[inv_err]).mean())
    else:
        res["inv_err_on_bass_before"] = 0.0

    # ---- bass/inversion head own accuracy ----
    res["inv_acc"] = float((inv_pred == inv_te).mean())
    # inversion detection recall/precision (positive = inversion)
    tp = ((inv_pred == 1) & (inv_te == 1)).sum()
    fp = ((inv_pred == 1) & (inv_te == 0)).sum()
    fn = ((inv_pred == 0) & (inv_te == 1)).sum()
    res["inv_recall"] = float(tp / (tp + fn)) if (tp + fn) else 0.0
    res["inv_prec"] = float(tp / (tp + fp)) if (tp + fp) else 0.0
    # bass-pc accuracy on TRUE inversions (oracle inversion gate)
    res["basspc_acc_oninv"] = float((basspc_pred[is_iv] == bpc_te[is_iv]).mean()) if is_iv.sum() else 0.0
    # top of chance for 12-way = 1/12
    # bass-pc accuracy relative to functional root (is it just predicting root?)
    res["basspc_eq_root_oninv"] = float((basspc_pred[is_iv] == root_te[is_iv]).mean()) if is_iv.sum() else 0.0

    # ---- INTERACTION mechanism (blind AFTER) ----
    # If inv-head fires AND root-argmax == predicted bass pc -> the root head is
    # likely reporting the bass; take root head's best class != that bass pc.
    corrected = rp.copy()
    fire = (inv_pred == 1) & (rp == basspc_pred)
    if fire.sum():
        masked = root_logits.copy()
        # forbid the predicted-bass class for firing rows
        rows = np.where(fire)[0]
        masked[rows, basspc_pred[rows]] = -1e9
        corrected[rows] = masked[rows].argmax(1)
    res["n_fire"] = int(fire.sum())
    res["n_fire_on_inv"] = int((fire & is_iv).sum())
    res["root_acc_inv_after"] = float((corrected[is_iv] == root_te[is_iv]).mean()) if is_iv.sum() else 0.0
    res["root_acc_rootpos_after"] = float((corrected[is_rp] == root_te[is_rp]).mean()) if is_rp.sum() else 0.0
    inv_err_a = is_iv & (corrected != root_te)
    res["inv_err_on_bass_after"] = float((corrected[inv_err_a] == bpc_te[inv_err_a]).mean()) if inv_err_a.sum() else 0.0

    # ORACLE-gate variant: apply the same redirect but gate on TRUE inversion
    # (isolates "does bass-pc head know the bass" from "does inv head fire")
    corrected_o = rp.copy()
    fire_o = (inv_te == 1) & (rp == basspc_pred)
    if fire_o.sum():
        rows = np.where(fire_o)[0]
        masked = root_logits.copy(); masked[rows, basspc_pred[rows]] = -1e9
        corrected_o[rows] = masked[rows].argmax(1)
    res["root_acc_inv_after_oracle"] = float((corrected_o[is_iv] == root_te[is_iv]).mean()) if is_iv.sum() else 0.0

    res["n_inv_te"] = int(is_iv.sum()); res["n_rp_te"] = int(is_rp.sum())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=REPO / "data/cache/rwc/rwc_bp48.npz")
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--bass-feat", choices=["full", "block"], default="full")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    dev = a.device
    if dev is None:
        try:
            import torch; dev = "mps" if torch.backends.mps.is_available() else "cpu"
        except Exception:
            dev = "cpu"

    d = load_corpus(a.corpus)
    labels = d["labels"]; roots_all = d["root"].astype(int)
    is_inv = np.zeros(len(labels), int); bass_pc = np.full(len(labels), -1, int)
    for i, lab in enumerate(labels):
        iv, b = derive_bass(str(lab), roots_all[i])
        is_inv[i] = iv; bass_pc[i] = b if b >= 0 else 0

    keep = filter_by_match(d["match"], minimum=MatchQuality.EXACT)
    print(f"Corpus {a.corpus.name}: {len(labels)} records, EXACT-match {keep.sum()}, "
          f"inversions {is_inv[keep].sum()} ({100*is_inv[keep].mean():.1f}%), "
          f"bass_feat={a.bass_feat}, device={dev}")

    runs = []
    for s in range(a.seeds):
        print(f"--- seed {s} ---", flush=True)
        r = one_split(d, s, is_inv, bass_pc, epochs=a.epochs, lr=a.lr, batch=a.batch,
                      device=dev, bass_feat=a.bass_feat)
        runs.append(r)
        print(f"  root all={r['root_acc_all']:.3f} rootpos={r['root_acc_rootpos']:.3f} "
              f"inv={r['root_acc_inv']:.3f} | err->bass {r['inv_err_on_bass_before']:.3f} | "
              f"inv_head acc={r['inv_acc']:.3f} R={r['inv_recall']:.3f} P={r['inv_prec']:.3f} | "
              f"basspc_oninv={r['basspc_acc_oninv']:.3f} | "
              f"AFTER inv={r['root_acc_inv_after']:.3f} (oracle {r['root_acc_inv_after_oracle']:.3f}) "
              f"err->bass {r['inv_err_on_bass_after']:.3f} rp_after={r['root_acc_rootpos_after']:.3f}", flush=True)

    def ms(k):
        v = np.array([r[k] for r in runs]); return v.mean(), v.std()

    print("\n" + "=" * 72)
    print(f"RWC bass/inversion CV, {a.seeds} song-stratified seeds (bass_feat={a.bass_feat})")
    print(f"  mean inv test chords/split: {np.mean([r['n_inv_te'] for r in runs]):.0f}  "
          f"root-pos: {np.mean([r['n_rp_te'] for r in runs]):.0f}")
    rows = [
        ("ROOT acc (all)", "root_acc_all"),
        ("ROOT acc (root-position)", "root_acc_rootpos"),
        ("ROOT acc (inversion)", "root_acc_inv"),
        ("  -> root err on inversions landing on sounding bass (BEFORE)", "inv_err_on_bass_before"),
        ("INVERSION head acc", "inv_acc"),
        ("INVERSION head recall (find inversions)", "inv_recall"),
        ("INVERSION head precision", "inv_prec"),
        ("BASS-PC head acc on true inversions (12-way, chance .083)", "basspc_acc_oninv"),
        ("  (bass-pc head == functional root, sanity)", "basspc_eq_root_oninv"),
        ("ROOT acc (inversion) AFTER blind redirect", "root_acc_inv_after"),
        ("ROOT acc (inversion) AFTER oracle-inv redirect", "root_acc_inv_after_oracle"),
        ("  -> err on bass AFTER", "inv_err_on_bass_after"),
        ("ROOT acc (root-position) AFTER (must not drop)", "root_acc_rootpos_after"),
    ]
    for lbl, k in rows:
        m, sd = ms(k); print(f"  {lbl:58s}: {m:.3f} +/- {sd:.3f}")
    print(f"  redirect fires: {np.mean([r['n_fire'] for r in runs]):.0f}/split "
          f"({np.mean([r['n_fire_on_inv'] for r in runs]):.0f} on true inversions)")


if __name__ == "__main__":
    main()
