"""Falsifiable test of the user's prediction: is root-motion interval
concentration STRONGER for the SOUNDING-BASS target than for FUNCTIONAL ROOT?

Same (true - prev_neighbour) mod 12 histogram under both target definitions,
changes-only and all.  Concentration measured by:
  - Shannon entropy H (bits); lower = more concentrated (uniform=log2(12)=3.585)
  - Total-variation distance from uniform (higher = more non-uniform)
  - normalised chi2 (chi2 / n); scale-free non-uniformity
  - mass on the 'common voice-leading' set {P4,P5,M2,m7,m2,M7} (fifths+steps)
Reported plainly whether it confirms the prediction or not.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from scipy.stats import chisquare

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from harmonia.data.corpus_schema import MatchQuality, filter_by_match
from bass_simple_cv import derive_bass_target

CORPUS = REPO / "data/cache/rwc/rwc_bp48_fixed.npz"


def offsets(seq, sid, t0, changes_only):
    out = []
    for s in sorted(set(sid.tolist())):
        idx = np.where(sid == s)[0]; idx = idx[np.argsort(t0[idx])]
        r = seq[idx]
        for j in range(1, len(r)):
            o = (r[j] - r[j-1]) % 12
            if changes_only and o == 0: continue
            out.append(o)
    return np.array(out)


def stats(offs):
    h = np.bincount(offs % 12, minlength=12).astype(float); n = h.sum(); p = h / n
    H = -(p[p > 0] * np.log2(p[p > 0])).sum()
    tv = 0.5 * np.abs(p - 1/12).sum()
    chi, _ = chisquare(h)
    common = p[5] + p[7] + p[2] + p[10] + p[1] + p[11]  # P4 P5 M2 m7 m2 M7
    return dict(n=int(n), H=H, tv=tv, chi2_over_n=chi/n, common=common, p=p)


def main():
    c = np.load(CORPUS, allow_pickle=True)
    keep = filter_by_match(c["match"], minimum=MatchQuality.EXACT)
    labels = c["labels"][keep]; root = c["root"][keep].astype(int)
    sid = c["song_id"][keep]; t0 = c["t0"][keep]
    bass = np.array([derive_bass_target(l, root[i])[1] for i, l in enumerate(labels)])

    print(f"{'target':16s} {'set':13s} {'n':>6s} {'H(bits)':>8s} {'TVfromU':>8s} "
          f"{'chi2/n':>8s} {'common%':>8s}")
    print("  (uniform baseline: H=3.585 bits, TV=0, chi2/n=0, common=50.0%)")
    rows = []
    for tgt, seq in [("functional-root", root), ("sounding-bass", bass)]:
        for co, tag in [(False, "all"), (True, "changes-only")]:
            s = stats(offsets(seq, sid, t0, co))
            rows.append((tgt, tag, s))
            print(f"{tgt:16s} {tag:13s} {s['n']:6d} {s['H']:8.3f} {s['tv']:8.3f} "
                  f"{s['chi2_over_n']:8.3f} {100*s['common']:7.1f}%")

    # verdict on changes-only (the cleanest comparison)
    rr = next(s for t, tag, s in rows if t == "functional-root" and tag == "changes-only")
    bb = next(s for t, tag, s in rows if t == "sounding-bass" and tag == "changes-only")
    print("\n--- VERDICT (changes-only) ---")
    print(f"  entropy:   root {rr['H']:.3f} vs bass {bb['H']:.3f} bits  "
          f"({'bass MORE concentrated' if bb['H'] < rr['H'] else 'ROOT more concentrated'})")
    print(f"  TV-from-U: root {rr['tv']:.3f} vs bass {bb['tv']:.3f}  "
          f"({'bass MORE non-uniform' if bb['tv'] > rr['tv'] else 'ROOT more non-uniform'})")
    print(f"  chi2/n:    root {rr['chi2_over_n']:.3f} vs bass {bb['chi2_over_n']:.3f}  "
          f"({'bass' if bb['chi2_over_n'] > rr['chi2_over_n'] else 'ROOT'} more non-uniform)")
    print(f"  common voice-leading mass: root {100*rr['common']:.1f}% vs bass {100*bb['common']:.1f}%")
    print("  breakdown P4/P5/M2/m7/m2/M7 (root -> bass):")
    for o, nm in [(5,"P4"),(7,"P5"),(2,"M2"),(10,"m7"),(1,"m2"),(11,"M7"),(3,"m3"),(4,"M3")]:
        print(f"    {nm:3s}: {100*rr['p'][o]:5.1f}% -> {100*bb['p'][o]:5.1f}%")


if __name__ == "__main__":
    main()
