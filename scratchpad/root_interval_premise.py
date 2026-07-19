"""Premise check (correct version): is root-to-root motion non-uniform?

The prior probe measured only "true root == a neighbour's PREDICTED root"
(31%) -> an IDENTITY test that by construction cannot see voice-leading
(bass rarely repeats pitch). Correct test: the empirical distribution of
(true_root - neighbour_root) mod 12 vs a uniform null over 12 offsets.

Computes for previous & next neighbour, using both GT roots and the root
model's PREDICTED roots, on (a) the full EXACT-match test population and
(b) the bottom-25% calibrated-confidence low-conf subset (same split as
scripts/calibration_root_gate_probe.py).

Read-only. Writes a JSON + a histogram plot.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from scipy.stats import chisquare

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from harmonia.data.corpus_schema import MatchQuality, filter_by_match

CKPT = REPO / "data/models/_eval_only_rwc_bp48_fixed_root_2026_07_16.pt"
CORPUS = REPO / "data/cache/rwc/rwc_bp48_fixed.npz"

# music-theory prior ranking of |offset| by expected frequency (semitones)
THEORY_ORDER = [0, 5, 7, 2, 10, 3, 9, 4, 8, 1, 11, 6]
NAMES = {0:"unison",1:"m2",2:"M2",3:"m3",4:"M3",5:"P4",6:"tritone",
         7:"P5",8:"m6",9:"M6",10:"m7",11:"M7"}


def make_mlp(in_dim, n):
    return nn.Sequential(
        nn.Linear(in_dim, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.3),
        nn.Linear(128, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.3),
        nn.Linear(64, n))


def hist12(offsets):
    h = np.bincount(np.asarray(offsets) % 12, minlength=12).astype(float)
    return h


def report(name, offsets, out):
    h = hist12(offsets)
    n = h.sum()
    p = h / n
    chi, pval = chisquare(h)  # vs uniform expectation n/12
    # top-3 offsets by empirical mass
    order = np.argsort(-h)
    top = [(int(o), NAMES[int(o)], float(p[int(o)])) for o in order[:4]]
    print(f"\n=== {name}  (n={int(n)}) ===")
    print(f"  chi-square vs uniform: chi2={chi:.1f}  p={pval:.2e}  "
          f"({'NON-UNIFORM' if pval < 1e-3 else 'uniform-ish'})")
    print("  offset(semi)  name     count    frac    (uniform=%.3f)" % (1/12))
    for o in THEORY_ORDER:
        bar = "#" * int(round(p[o] * 120))
        print(f"   {o:2d}  {NAMES[o]:8s} {int(h[o]):6d}  {p[o]:.3f}  {bar}")
    print(f"  top offsets: {top}")
    # fraction on the 4 dominant harmonic intervals (P4/P5/M2/m7 = fifth-related + step)
    frac_fifths = float(p[5] + p[7])
    frac_steps = float(p[1] + p[2] + p[10] + p[11])
    frac_thirds = float(p[3] + p[4] + p[8] + p[9])
    print(f"  frac P4|P5={frac_fifths:.3f}  steps(m2/M2/m7/M7)={frac_steps:.3f}  "
          f"thirds(m3/M3/m6/M6)={frac_thirds:.3f}  unison={p[0]:.3f}")
    out[name] = {"hist": h.tolist(), "frac": p.tolist(), "chi2": float(chi),
                 "pval": float(pval), "n": int(n),
                 "frac_fifths": frac_fifths, "frac_steps": frac_steps,
                 "frac_thirds": frac_thirds, "frac_unison": float(p[0]),
                 "top": top}
    return h


def main():
    d = torch.load(CKPT, map_location="cpu", weights_only=False)
    c = np.load(CORPUS, allow_pickle=True)
    keep = filter_by_match(c["match"], minimum=MatchQuality.EXACT)
    X = c["feat48_abs"].astype(np.float32)[keep]
    root = c["root"].astype(int)[keep]
    sid = c["song_id"][keep]; t0 = c["t0"][keep]
    test_songs = set(d["test_songs"])
    test_mask = np.isin(sid, list(test_songs))
    train_mask = ~test_mask

    model = make_mlp(48, 12); model.load_state_dict(d["root_model_state"]); model.eval()
    mean = d["root_mean"]; std = d["root_std"]
    with torch.no_grad():
        logits = model(torch.tensor(((X - mean) / std).astype(np.float32))).numpy()
    z = logits - logits.max(1, keepdims=True); e = np.exp(z); post = e / e.sum(1, keepdims=True)
    pred = post.argmax(1); conf = post.max(1)

    out = {}

    # ---- (0) CORPUS-WIDE GT->GT motion over adjacent chord CHANGES (all songs) ----
    prev_off_all, next_off_all = [], []
    prev_off_change, next_off_change = [], []
    for s in sorted(set(sid.tolist())):
        idx = np.where(sid == s)[0]
        idx = idx[np.argsort(t0[idx])]
        r = root[idx]
        for j in range(len(r)):
            if j > 0:
                o = (r[j] - r[j-1]) % 12
                prev_off_all.append(o)
                if o != 0: prev_off_change.append(o)
            if j < len(r)-1:
                o = (r[j] - r[j+1]) % 12
                next_off_all.append(o)
                if o != 0: next_off_change.append(o)
    report("A_GTprev_all_transitions", prev_off_all, out)
    report("A_GTprev_CHANGES_only", prev_off_change, out)
    report("A_GTnext_CHANGES_only", next_off_change, out)

    # ---- (1) TEST-set neighbour offsets, GT neighbour roots ----
    idx_map = {}  # (song ordered) -> local index arrays
    prev_gt, next_gt, prev_pr, next_pr = [], [], [], []
    prev_gt_lc, next_gt_lc, prev_pr_lc, next_pr_lc = [], [], [], []
    # low-conf threshold on test set (bottom 25%)
    thr = np.quantile(conf[test_mask], 0.25)
    for s in sorted(set(sid[test_mask].tolist())):
        idx = np.where(sid == s)[0]
        idx = idx[np.argsort(t0[idx])]
        r = root[idx]; pr = pred[idx]; cf = conf[idx]
        for j in range(len(idx)):
            lc = cf[j] <= thr
            if j > 0:
                prev_gt.append((r[j]-r[j-1]) % 12); prev_pr.append((r[j]-pr[j-1]) % 12)
                if lc: prev_gt_lc.append((r[j]-r[j-1]) % 12); prev_pr_lc.append((r[j]-pr[j-1]) % 12)
            if j < len(idx)-1:
                next_gt.append((r[j]-r[j+1]) % 12); next_pr.append((r[j]-pr[j+1]) % 12)
                if lc: next_gt_lc.append((r[j]-r[j+1]) % 12); next_pr_lc.append((r[j]-pr[j+1]) % 12)

    report("B_TEST_prev_GTneighbour_ALL", prev_gt, out)
    report("C_TEST_prev_PREDneighbour_ALL", prev_pr, out)
    report("D_TEST_prev_GTneighbour_LOWCONF", prev_gt_lc, out)
    report("E_TEST_prev_PREDneighbour_LOWCONF", prev_pr_lc, out)
    report("F_TEST_next_GTneighbour_LOWCONF", next_gt_lc, out)
    report("G_TEST_next_PREDneighbour_LOWCONF", next_pr_lc, out)

    # identity match (reproduce prior number) for sanity
    id_prev = np.mean([1 if o == 0 else 0 for o in prev_pr_lc]) if prev_pr_lc else 0
    id_next = np.mean([1 if o == 0 else 0 for o in next_pr_lc]) if next_pr_lc else 0
    print(f"\n[sanity] identity (offset==0) low-conf: prev_pred={id_prev:.3f} next_pred={id_next:.3f} "
          f"(prior reported 0.31 for prev|next union)")

    (REPO / "scratchpad/root_interval_premise.json").write_text(json.dumps(out, indent=2))
    print("\nwrote scratchpad/root_interval_premise.json")

    # plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        keys = ["A_GTprev_CHANGES_only", "B_TEST_prev_GTneighbour_ALL",
                "D_TEST_prev_GTneighbour_LOWCONF", "E_TEST_prev_PREDneighbour_LOWCONF"]
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        for ax, k in zip(axes.ravel(), keys):
            fr = np.array(out[k]["frac"])
            colors = ["#2a9d8f" if o in (0,5,7,2,10) else "#8888aa" for o in range(12)]
            ax.bar(range(12), fr, color=colors)
            ax.axhline(1/12, ls="--", c="crimson", lw=1, label="uniform 1/12")
            ax.set_title(f"{k}\nn={out[k]['n']}  chi2 p={out[k]['pval']:.1e}", fontsize=9)
            ax.set_xticks(range(12)); ax.set_xticklabels([NAMES[o] for o in range(12)], rotation=45, fontsize=7)
            ax.set_ylabel("P(offset)")
            ax.legend(fontsize=7)
        fig.suptitle("Root-to-root interval distribution: (true_root - neighbour_root) mod 12", fontsize=12)
        fig.tight_layout()
        fig.savefig(REPO / "scratchpad/root_interval_premise.png", dpi=110)
        print("wrote scratchpad/root_interval_premise.png")
    except Exception as ex:
        print("plot skipped:", ex)


if __name__ == "__main__":
    main()
