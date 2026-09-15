"""ChordFormer-style FACTORED chord-slot model vs the flat baseline on RWC-Popular.

Implements the ChordFormer (arXiv 2502.11840) structured-chord representation as
a factored output over shared-backbone heads, and evaluates it head-to-head
against the project's confirmed flat baseline (root 64.0%+/-2.0%, quality
balanced 52.2%+/-4.0%) using the *identical* song-stratified multi-seed CV
methodology as scripts/train_jaah_cv.py.

ChordFormer's 6-slot schema (confirmed from arxiv 2502.11840v1, the source):
  slot 1  root+triad : root {C..B} x triad {maj,min,sus4,sus2,dim,aug}
  slot 2  bass       : {N, C..B}  (inversion, as scale-degree here)
  slot 3  7th        : {N, 7, b7, bb7}
  slot 4  9th        : {N, 9, #9, b9}
  slot 5  11th       : {N, 11, #11}
  slot 6  13th       : {N, 13, b13}
  loss = per-slot weighted cross-entropy; "N" is an explicit class per slot, so
  no masking is needed for absent elements (an absent 9th IS the label N).

ADAPTATION for this project (documented, deliberate):
  * The project already factors ROOT out of the features: feat48 is ROOT-RELATIVE
    (rotated by the true root), feat48_abs is absolute. Root therefore CANNOT be
    predicted from feat48 and MUST use feat48_abs -- exactly as the baseline does.
    So we split ChordFormer's combined "root+triad" slot 1 into an independent
    root head (feat48_abs, +roll augment -- byte-identical to the baseline) and a
    triad slot on the shared root-relative backbone. Root is thus held FIXED
    across both arms; the factoring only changes the quality/extension side.
  * Slots 2-6 are all defined RELATIVE TO ROOT, so they live on the feat48
    (root-relative) backbone together with triad. This is where ChordFormer's
    claimed long-tail win must show up, if anywhere.
  * bass slot uses scale-degree tokens (root-relative) rather than absolute pc,
    since the backbone is root-relative -- equivalent information.

Capacity discipline (CLAUDE.md 'bigger MLP hurt on Billboard'): the shared
backbone is 48->128->64 (the SAME trunk as the flat quality MLP); the 6 slot
heads are tiny 64->n_slot linears. Total params ~ the flat baseline's, not
bigger.

Read-only on data/cache/rwc/rwc_bp48.npz. Writes nothing to that dir.
"""
from __future__ import annotations
import argparse, sys, re
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from harmonia.data.corpus_schema import MatchQuality, filter_by_match, load_corpus
from train_real_audio_final import (
    _make_mlp, _train_head, _eval, _augment_root_by_roll, _standardize, QUALITIES,
)

# ---------------------------------------------------------------------------
# Harte label -> ChordFormer slots
# ---------------------------------------------------------------------------
TRIADS   = ["maj", "min", "sus", "dim", "aug"]          # sus2/sus4 folded -> sus
SEVENTHS = ["N", "7", "b7", "bb7"]                       # natural / dominant / dim
NINTHS   = ["N", "9", "#9", "b9"]
ELEVENTHS = ["N", "11", "#11"]
THIRTEENTHS = ["N", "13", "b13"]
BASSES   = ["N", "b2", "2", "b3", "3", "4", "b5", "5", "b6", "6", "b7", "7"]  # root-rel degree
TRIAD_IDX = {t: i for i, t in enumerate(TRIADS)}
S7_IDX = {t: i for i, t in enumerate(SEVENTHS)}
S9_IDX = {t: i for i, t in enumerate(NINTHS)}
S11_IDX = {t: i for i, t in enumerate(ELEVENTHS)}
S13_IDX = {t: i for i, t in enumerate(THIRTEENTHS)}
BASS_IDX = {t: i for i, t in enumerate(BASSES)}

# shorthand -> degree-token set (Harte semantics)
_SHORTHAND = {
    "maj": {"3", "5"}, "min": {"b3", "5"}, "dim": {"b3", "b5"}, "aug": {"3", "#5"},
    "maj7": {"3", "5", "7"}, "min7": {"b3", "5", "b7"}, "7": {"3", "5", "b7"},
    "dim7": {"b3", "b5", "bb7"}, "hdim7": {"b3", "b5", "b7"}, "minmaj7": {"b3", "5", "7"},
    "maj6": {"3", "5", "6"}, "min6": {"b3", "5", "6"},
    "9": {"3", "5", "b7", "9"}, "maj9": {"3", "5", "7", "9"}, "min9": {"b3", "5", "b7", "9"},
    "11": {"3", "5", "b7", "9", "11"}, "min11": {"b3", "5", "b7", "9", "11"},
    "13": {"3", "5", "b7", "9", "11", "13"},
    "sus4": {"4", "5"}, "sus2": {"2", "5"}, "min7b5": {"b3", "b5", "b7"},
    "": {"3", "5"},  # bare root -> maj triad
}


def parse_slots(label: str):
    """Harte label -> (triad, bass, s7, s9, s11, s13) index tuple, or None for N/X."""
    label = label.strip()
    if label in ("N", "X", ""):
        return None
    # bass
    bass = "N"
    core = label
    if "/" in label:
        core, b = label.split("/", 1)
        bass = b.strip()
        if bass not in BASS_IDX:
            bass = "N"  # unknown/absolute-note bass -> treat as root position
    # split root:tail
    if ":" in core:
        tail = core.split(":", 1)[1]
    else:
        tail = ""  # bare root note = maj triad
    # base shorthand + parenthetical modifiers
    m = re.match(r"^([a-zA-Z0-9#b]*)(\(([^)]*)\))?$", tail)
    if m:
        base = m.group(1)
        mods = m.group(3)
    else:
        base, mods = tail, None
    degs = set(_SHORTHAND.get(base, None) if base in _SHORTHAND else set())
    if base not in _SHORTHAND:
        # interval-list-only form like "(3,5)" or unknown base: start empty
        degs = set()
    if mods:
        for tok in (t.strip() for t in mods.split(",") if t.strip()):
            if tok.startswith("*"):
                degs.discard(tok[1:])
            else:
                degs.add(tok)
    if not degs:
        degs = {"3", "5"}  # last-resort: assume maj triad

    # ---- triad ----
    has_b3, has_3 = "b3" in degs, "3" in degs
    has_b5, has_5, has_s5 = "b5" in degs, "5" in degs, "#5" in degs
    has_4 = "4" in degs
    has_2 = "2" in degs
    if not has_3 and not has_b3 and (has_4 or has_2):
        triad = "sus"
    elif has_b3:
        triad = "dim" if has_b5 else "min"
    elif has_3:
        triad = "aug" if has_s5 else "maj"
    else:
        triad = "maj"  # third omitted (e.g. maj(*3)) -> treat as maj

    # ---- 7th ---- (order: bb7 > b7 > 7)
    s7 = "bb7" if "bb7" in degs else ("b7" if "b7" in degs else ("7" if "7" in degs else "N"))
    # ---- 9th ----
    s9 = "b9" if "b9" in degs else ("#9" if "#9" in degs else ("9" if "9" in degs else "N"))
    # ---- 11th ----
    s11 = "#11" if "#11" in degs else ("11" if "11" in degs else "N")
    # ---- 13th ----
    s13 = "b13" if "b13" in degs else ("13" if "13" in degs else "N")

    return (TRIAD_IDX[triad], BASS_IDX[bass], S7_IDX[s7], S9_IDX[s9], S11_IDX[s11], S13_IDX[s13])


# reconstruct the project's 7-way family from (triad, 7th) so we can compare
# directly to the flat quality head's balanced-acc / dom-recall numbers.
def slots_to_family(triad_i, s7_i):
    triad = TRIADS[triad_i]; s7 = SEVENTHS[s7_i]
    if triad == "sus":
        return "sus"
    if triad == "aug":
        return "aug"
    if triad == "min":
        return "min"
    if triad == "dim":
        return "hdim" if s7 == "b7" else "dim"   # dim triad + b7 = half-dim
    # maj triad
    return "dom" if s7 == "b7" else "maj"


FAM_IDX = {q: i for i, q in enumerate(QUALITIES)}

SLOT_NAMES = ["triad", "bass", "7th", "9th", "11th", "13th"]
SLOT_SIZES = [len(TRIADS), len(BASSES), len(SEVENTHS), len(NINTHS), len(ELEVENTHS), len(THIRTEENTHS)]


# ---------------------------------------------------------------------------
# Factored model: shared backbone + per-slot heads
# ---------------------------------------------------------------------------
def _make_factored(in_dim, slot_sizes):
    import torch.nn as nn

    class Factored(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(in_dim, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.3),
                nn.Linear(128, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.3),
            )
            self.heads = nn.ModuleList([nn.Linear(64, s) for s in slot_sizes])

        def forward(self, x):
            h = self.trunk(x)
            return [head(h) for head in self.heads]

    return Factored()


def _train_factored(X, Y, slot_sizes, *, epochs, lr, batch, device):
    """Y: (N, n_slots) int. Per-slot balanced-weighted CE, summed. Returns (model,mean,std)."""
    import torch, torch.nn as nn

    Xn, mean, std = _standardize(X)
    Xt = torch.tensor(Xn, dtype=torch.float32, device=device)
    Yt = torch.tensor(Y, dtype=torch.long, device=device)

    # per-slot balanced weights (same recipe as the flat baseline head)
    weights = []
    for j, ncls in enumerate(slot_sizes):
        counts = np.bincount(Y[:, j], minlength=ncls).astype(float)
        w = 1.0 / (counts + 1.0); w /= w.sum(); w *= ncls
        weights.append(torch.tensor(w, dtype=torch.float32, device=device))
    loss_fns = [nn.CrossEntropyLoss(weight=weights[j]) for j in range(len(slot_sizes))]

    model = _make_factored(X.shape[1], slot_sizes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = len(Xt)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            logits = model(Xt[idx])
            loss = sum(loss_fns[j](logits[j], Yt[idx, j]) for j in range(len(slot_sizes)))
            loss.backward(); opt.step()
        sched.step()
    model.eval()
    return model, mean, std


def _eval_factored(X, Y, model, mean, std, device):
    import torch
    Xn = ((X - mean) / std).astype(np.float32)
    with torch.no_grad():
        logits = model(torch.tensor(Xn, device=device))
        preds = np.stack([l.argmax(1).cpu().numpy() for l in logits], axis=1)  # (N, n_slots)
    return preds


def _balanced_recall(y_true, y_pred, n_classes):
    recs = []
    for c in range(n_classes):
        m = y_true == c
        if m.sum() > 0:
            recs.append((y_pred[m] == c).mean())
    return float(np.mean(recs)) if recs else 0.0


def one_split(d, seed, slots, families, *, roll, epochs, lr, batch, device, test_frac=0.2):
    keep = filter_by_match(d["match"], minimum=MatchQuality.EXACT)
    feat48 = d["feat48"][keep]; feat48_abs = d["feat48_abs"][keep]
    roots = d["root"].astype(int)[keep]
    quality_idx = d["quality_idx"].astype(int)[keep]
    song_id = d["song_id"][keep]
    S = slots[keep]              # (N, 6)
    fam = families[keep]         # (N,) 7-way family idx (reconstructed from GT slots)

    songs = sorted(set(song_id.tolist()))
    rng = np.random.RandomState(seed); rng.shuffle(songs)
    n_test = max(1, int(round(test_frac * len(songs))))
    test_songs = set(songs[:n_test])
    tr = np.array([s not in test_songs for s in song_id]); te = ~tr

    # ---- ROOT head: identical in both arms (feat48_abs, roll augment) ----
    Xtr, ytr = feat48_abs[tr], roots[tr]
    if roll:
        Xtr, ytr = _augment_root_by_roll(Xtr, ytr)
    rm, rmean, rstd = _train_head(Xtr, ytr, 12, epochs=epochs, lr=lr, batch=batch,
                                  device=device, head_name="root")
    root_acc, _, _ = _eval(feat48_abs[te], roots[te], rm, rmean, rstd, device)

    # ---- BASELINE arm: flat 7-way quality head on feat48 ----
    qm, qmean, qstd = _train_head(feat48[tr], quality_idx[tr], 7, epochs=epochs, lr=lr,
                                  batch=batch, device=device, head_name="qual")
    q_acc, q_recall, _ = _eval(feat48[te], quality_idx[te], qm, qmean, qstd, device)
    base_bal = float(np.mean([q_recall.get(i, 0.0) for i in range(7)]))
    base_dom = q_recall.get(FAM_IDX["dom"], 0.0)

    # ---- FACTORED arm: shared backbone + 6 slot heads on feat48 ----
    fm, fmean, fstd = _train_factored(feat48[tr], S[tr], SLOT_SIZES, epochs=epochs,
                                      lr=lr, batch=batch, device=device)
    P = _eval_factored(feat48[te], S[te], fm, fmean, fstd, device)  # (n_te, 6)

    # per-slot accuracy + balanced recall
    slot_acc, slot_bal = {}, {}
    for j, name in enumerate(SLOT_NAMES):
        slot_acc[name] = float((P[:, j] == S[te][:, j]).mean())
        slot_bal[name] = _balanced_recall(S[te][:, j], P[:, j], SLOT_SIZES[j])

    # reconstruct 7-way family from predicted (triad, 7th) and score like the flat head.
    # true_fam is the STORED quality_idx -- byte-identical GT to the flat arm, so the
    # comparison is apples-to-apples (any parser/stored mismatch counts AGAINST factored).
    pred_fam = np.array([FAM_IDX[slots_to_family(P[i, 0], P[i, 2])] for i in range(len(P))])
    true_fam = quality_idx[te]
    fac_qacc = float((pred_fam == true_fam).mean())
    fac_bal = _balanced_recall(true_fam, pred_fam, 7)
    fac_dom = float((pred_fam[true_fam == FAM_IDX["dom"]] == FAM_IDX["dom"]).mean()) \
        if (true_fam == FAM_IDX["dom"]).sum() else 0.0

    return {
        "root": root_acc,
        "base_qbal": base_bal, "base_qacc": q_acc, "base_dom": base_dom,
        "fac_qbal": fac_bal, "fac_qacc": fac_qacc, "fac_dom": fac_dom,
        "slot_acc": slot_acc, "slot_bal": slot_bal,
        "n_train": int(tr.sum()), "n_test": int(te.sum()), "n_test_songs": len(test_songs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=REPO / "data/cache/rwc/rwc_bp48.npz")
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--roll", action="store_true")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    dev = a.device
    if dev is None:
        try:
            import torch; dev = "mps" if torch.backends.mps.is_available() else "cpu"
        except Exception:
            dev = "cpu"

    d = load_corpus(a.corpus)
    labels = d["labels"]
    # parse slots for every record
    slots = np.full((len(labels), 6), -1, dtype=int)
    n_bad = 0
    for i, lab in enumerate(labels):
        s = parse_slots(str(lab))
        if s is None:
            n_bad += 1
            slots[i] = (0, 0, 0, 0, 0, 0)
        else:
            slots[i] = s
    families = np.array([FAM_IDX[slots_to_family(slots[i, 0], slots[i, 2])] for i in range(len(labels))])

    print(f"Corpus {a.corpus.name}: {len(labels)} records, "
          f"{len(set(d['song_id'].tolist()))} songs, device={dev}, unparsed_slots={n_bad}")
    for j, name in enumerate(SLOT_NAMES):
        counts = np.bincount(slots[:, j], minlength=SLOT_SIZES[j])
        vocab = [TRIADS, BASSES, SEVENTHS, NINTHS, ELEVENTHS, THIRTEENTHS][j]
        print(f"  slot {name:5s}: " + ", ".join(f"{vocab[k]}={counts[k]}" for k in range(len(vocab))))
    # cross-check reconstructed family vs stored quality
    stored = d["quality_idx"].astype(int)
    agree = (families == stored).mean()
    print(f"  reconstructed-family vs stored quality_idx agreement: {agree:.3f}\n")

    runs = []
    for s in range(a.seeds):
        print(f"--- seed {s} ---", flush=True)
        r = one_split(d, s, slots, families, roll=a.roll, epochs=a.epochs,
                      lr=a.lr, batch=a.batch, device=dev)
        runs.append(r)
        print(f"  root={r['root']:.3f} | FLAT qbal={r['base_qbal']:.3f} dom={r['base_dom']:.3f}"
              f" | FACTORED qbal={r['fac_qbal']:.3f} dom={r['fac_dom']:.3f}"
              f" triad_bal={r['slot_bal']['triad']:.3f} 7th_bal={r['slot_bal']['7th']:.3f}\n",
              flush=True)

    def ms(k):
        v = np.array([r[k] for r in runs]); return v.mean(), v.std()

    print("=" * 70)
    print(f"RWC-Popular CV over {a.seeds} song-stratified splits (roll={a.roll})")
    print(f"  {'Root acc (shared, both arms)':38s}: {ms('root')[0]:.1%} +/- {ms('root')[1]:.1%}")
    print("  --- QUALITY (7-way family), FLAT baseline vs FACTORED reconstruction ---")
    for k, lbl in [("base_qbal", "FLAT  quality balanced acc"),
                   ("fac_qbal", "FACT  quality balanced acc"),
                   ("base_qacc", "FLAT  quality raw acc"),
                   ("fac_qacc", "FACT  quality raw acc"),
                   ("base_dom", "FLAT  dom recall"),
                   ("fac_dom", "FACT  dom recall")]:
        m, sd = ms(k)
        print(f"  {lbl:38s}: {m:.1%} +/- {sd:.1%}")
    print("  --- per-slot balanced recall (factored heads) ---")
    for name in SLOT_NAMES:
        v = np.array([r["slot_bal"][name] for r in runs])
        va = np.array([r["slot_acc"][name] for r in runs])
        print(f"    {name:6s}: bal {v.mean():.1%}+/-{v.std():.1%}   acc {va.mean():.1%}")
    print("\nConfirmed flat baseline: root 64.0%+/-2.0%, quality balanced 52.2%+/-4.0%, dom 52.4%")


if __name__ == "__main__":
    main()
