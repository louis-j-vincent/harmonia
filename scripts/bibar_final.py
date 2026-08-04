"""FINAL numbers: Louis's binary bi-bar lag matrix vs everything else, on the
Billboard bar-GT harness, with the tie-break artefact removed.

THE ONE THING TO READ. `scripts/section_start_placement_screen.py` (and the
lit review's 75.2%) break placement ties by PREFERRING THE SMALLER SHIFT. Any
cue that is CONSTANT across the +-4-bar window then scores 100% — including the
cue "never move at all". Inside a repeated section the 8-bar chord string
matches perfectly at every offset, so the chord-repeat cue is exactly that
constant, and its 75.2% is mostly the tie-break, not the cue. Ties are broken
at RANDOM below; both conventions are printed side by side so the gap is
visible rather than argued.

    python scripts/bibar_final.py [n_tracks]
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
from bibar_binary_ssm import (bibar, bin_ssm, bin_ssm_rot, binarise,  # noqa: E402
                              boundary_prf, chordstring_score, chordtone_binary,
                              cont_ssm, density, lag_runs, novelty_score,
                              placement, run_edge_score)
from bibar_sweep import load  # noqa: E402

CACHE = Path("/private/tmp/claude-501/-Users-vincente-Documents-Projets-Perso-"
             "Code-harmonia/29e6c8ff-69c3-4685-aae3-2c46f129bcde/scratchpad")
BEST = dict(mode="topk", kb=2, kt=4, join="concat", tau=0.8, reader="edgesum")


def best_ssm(s, rot=False):
    X = bibar(binarise(s["Cb"], BEST["mode"], BEST["kb"], BEST["kt"]),
              BEST["join"])
    return (bin_ssm_rot if rot else bin_ssm)(X, BEST["tau"])


def placement_biased(score, starts, nb, win=4):
    """The OLD convention, kept only to quantify what it was worth."""
    out = []
    tie = 1e-6 * np.abs(np.arange(-win, win + 1))
    for sb in starts:
        if sb < 8 or sb > nb - 9:
            continue
        v = np.array([score[c] if 0 <= c < len(score) else -9.0
                      for c in range(sb - win, sb + win + 1)], float)
        out.append(int(np.argmax(v - tie)) - win)
    return out


def main(n=None):
    songs = load(n)
    ns = sum(len(s["starts"]) for s in songs)
    print(f"{len(songs)} Billboard tracks · {ns} annotated section starts\n")
    rows = []

    def cue(name, fn):
        er, eb = [], []
        for s in songs:
            sc = fn(s)
            er += placement(sc, s["starts"], s["nb"])
            eb += placement_biased(sc, s["starts"], s["nb"])
        er, eb = np.array(er), np.array(eb)
        r = dict(name=name, exact=float((er == 0).mean() * 100),
                 pm1=float((np.abs(er) <= 1).mean() * 100),
                 med=float(np.median(np.abs(er))),
                 biased=float((eb == 0).mean() * 100), n=int(len(er)))
        rows.append(r)
        print(f"  {name:<44s} exact {r['exact']:5.1f}%  |e|<=1 {r['pm1']:5.1f}%"
              f"  med {r['med']:.1f}   [old tie-break: {r['biased']:5.1f}%]")
        return r

    print("PLACEMENT — exact-bar, +-4 bar window   (random ties | distance ties)")
    cue("trivial: never move (constant score)", lambda s: np.zeros(s["nb"]))
    cue("chroma checkerboard novelty (SHIPPED)", lambda s: novelty_score(s["Hb"]))
    cue("chord-string 8-bar repeat (lit review)",
        lambda s: chordstring_score(s["sym"]))
    cue("BINARY bi-bar lag, edgesum (LOUIS)",
        lambda s: run_edge_score(best_ssm(s), "edgesum"))
    cue("  ... same, reader=diff (capped, coarse)",
        lambda s: run_edge_score(best_ssm(s), "diff"))
    cue("  ... same, TRANSPOSITION-INVARIANT (12 rot)",
        lambda s: run_edge_score(best_ssm(s, rot=True), "edgesum"))
    cue("CONTROL: CONTINUOUS cosine, same reader",
        lambda s: run_edge_score(cont_ssm(s["Cb"], BEST["join"], 0.93),
                                 "edgesum"))
    cue("CONTROL: binary on CHORD TONES, same reader",
        lambda s: run_edge_score(
            bin_ssm(bibar(chordtone_binary(s["sym"]), BEST["join"]), 0.9),
            "edgesum"))
    print("\n  threshold sweep on the two lag-run cues (the reader is the same)")
    for t in (0.85, 0.88, 0.90, 0.93, 0.95, 0.97):
        cue(f"    continuous cosine, tau={t}",
            lambda s, t=t: run_edge_score(cont_ssm(s["Cb"], BEST["join"], t),
                                          "edgesum"))
    for t in (0.6, 0.7, 0.8, 0.9):
        cue(f"    binary bi-bar, tau={t}",
            lambda s, t=t: run_edge_score(
                bin_ssm(bibar(binarise(s["Cb"], BEST["mode"], BEST["kb"],
                                       BEST["kt"]), BEST["join"]), t),
                "edgesum"))
    cue("FUSION: continuous lag-run + novelty (z-summed)",
        lambda s: _z(run_edge_score(cont_ssm(s["Cb"], BEST["join"], 0.93),
                                    "edgesum"))
        + _z(novelty_score(s["Hb"]))[:s["nb"] - 1])

    # ── the hypothesis Louis's idea rests on, tested directly ───────────────
    print("\nHYPOTHESIS TEST — do repeats differ in ARRANGEMENT, so that a")
    print("binary 'same notes present' separates same-section pairs better")
    print("than a continuous cosine?  AUC over GT same-letter vs different.")
    ab, ac = [], []
    for s in songs:
        st, le = s["starts"], s["letters"]
        if len(st) < 3:
            continue
        Bx = bibar(binarise(s["Cb"], BEST["mode"], BEST["kb"], BEST["kt"]),
                   BEST["join"]).astype(np.float32)
        Bn = Bx / np.maximum(np.linalg.norm(Bx, axis=1, keepdims=True), 1e-9)
        V = s["Cb"] / np.maximum(np.linalg.norm(s["Cb"], axis=1, keepdims=True), 1e-9)
        Cx = np.concatenate([V[:-1], V[1:]], 1)
        Cn = Cx / np.maximum(np.linalg.norm(Cx, axis=1, keepdims=True), 1e-9)
        pos_b, neg_b, pos_c, neg_c = [], [], [], []
        for i in range(len(st)):
            for j in range(i + 1, len(st)):
                a, b = st[i], st[j]
                if a >= len(Bn) or b >= len(Bn):
                    continue
                sb_, sc_ = float(Bn[a] @ Bn[b]), float(Cn[a] @ Cn[b])
                (pos_b if le[i] == le[j] else neg_b).append(sb_)
                (pos_c if le[i] == le[j] else neg_c).append(sc_)
        for P, N, acc in ((pos_b, neg_b, ab), (pos_c, neg_c, ac)):
            if P and N:
                acc.append(np.mean([(p > q) + 0.5 * (p == q)
                                    for p in P for q in N]))
    print(f"  binary bi-bar dot     AUC {np.mean(ab):.3f}  (n={len(ab)} songs)")
    print(f"  continuous cosine     AUC {np.mean(ac):.3f}")
    print("  -> " + ("BINARY separates better — hypothesis SUPPORTED"
                     if np.mean(ab) > np.mean(ac) + 0.01 else
                     "no advantage for binarising — hypothesis NOT supported"))
    auc = dict(binary=float(np.mean(ab)), continuous=float(np.mean(ac)))

    # ── transposition invariance: the false-merge counterweight ─────────────
    print("\nTRANSPOSITION INVARIANCE — same-letter decision, GT letters")
    tr = {}
    for tag, rot in (("raw chroma", False), ("max over 12 rotations", True)):
        tp = fp = fn = 0
        for s in songs:
            st, le = s["starts"], s["letters"]
            Bx = bibar(binarise(s["Cb"], BEST["mode"], BEST["kb"], BEST["kt"]),
                       BEST["join"]).astype(np.float32)
            Bn = Bx / np.maximum(np.linalg.norm(Bx, axis=1, keepdims=True), 1e-9)
            for i in range(len(st)):
                for j in range(i + 1, len(st)):
                    a, b = st[i], st[j]
                    if a + 4 >= len(Bn) or b + 4 >= len(Bn):
                        continue
                    if rot:
                        v = max(float((Bn[a:a + 4] * np.roll(
                            Bn[b:b + 4].reshape(4, -1, 12), r, axis=2
                        ).reshape(4, -1)).sum(1).mean()) for r in range(12))
                    else:
                        v = float((Bn[a:a + 4] * Bn[b:b + 4]).sum(1).mean())
                    pred, gt = v >= 0.80, le[i] == le[j]
                    tp += pred and gt
                    fp += pred and not gt
                    fn += (not pred) and gt
        P = tp / max(1, tp + fp)
        R = tp / max(1, tp + fn)
        tr[tag] = dict(P=P, R=R, F=2 * P * R / max(1e-9, P + R))
        print(f"  {tag:<26s} P {P:.3f}  R {R:.3f}  F {2*P*R/max(1e-9,P+R):.3f}")

    # ── unconditioned boundary detection ────────────────────────────────────
    print("\nBOUNDARY DETECTION (unconditioned) — bar tolerance")
    bd = {}
    for tol in (0, 1, 2):
        f_l, f_n = [], []
        for s in songs:
            M = best_ssm(s)
            pred = sorted({r[0] for r in lag_runs(M, 4)}
                          | {r[0] + r[1] for r in lag_runs(M, 4)})
            pred = [p for p in pred if 0 < p < s["nb"]]
            f_l.append(boundary_prf(pred, list(s["starts"]), tol)[2])
            nv = novelty_score(s["Hb"])
            th = nv[8:-8].mean() + nv[8:-8].std() if s["nb"] > 20 else nv.mean()
            pk = [i for i in range(8, s["nb"] - 8)
                  if nv[i] == nv[max(0, i - 2):i + 3].max() and nv[i] >= th]
            f_n.append(boundary_prf(pk, list(s["starts"]), tol)[2])
        bd[tol] = dict(binary=float(np.mean(f_l)), novelty=float(np.mean(f_n)))
        print(f"  +-{tol} bar   binary bi-bar runs F {np.mean(f_l):.3f}   "
              f"novelty peaks F {np.mean(f_n):.3f}")

    dens = float(np.mean([density(best_ssm(s)) for s in songs]))
    print(f"\nSSM off-band density at the chosen setting: {dens*100:.1f}% "
          "(a saturated matrix would be near 100% and mean nothing)")
    json.dump(dict(rows=rows, auc=auc, transpose=tr, boundary=bd,
                   density=dens, n_tracks=len(songs), n_starts=ns, best=BEST),
              open(CACHE / "bibar_final.json", "w"), indent=1)
    print("saved", CACHE / "bibar_final.json")


def _z(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / max(v.std(), 1e-9)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
