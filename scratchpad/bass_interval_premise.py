"""Premise check under the NEW project target = SOUNDING BASS pitch class.

Reuses derive_bass_target (validated, from bass_simple_cv) to resolve the
absolute sounding bass pc from Harte labels (root-pos C->C, inversion C/E->E).
Computes (bass_pc[i] - bass_pc[neighbour]) mod 12 -- the actual sounding
BASS-LINE motion, which is what voice-leading conventions literally describe.

Pure GT statistic (no model, no chroma rotation, no shift-back). Reports the
full 12-bin histogram vs uniform for: all transitions, changes-only, and the
inversion-involving subset (where bass_pc != functional root at either end).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from scipy.stats import chisquare

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from harmonia.data.corpus_schema import MatchQuality, filter_by_match
from bass_simple_cv import derive_bass_target  # validated resolver

CORPUS = REPO / "data/cache/rwc/rwc_bp48_fixed.npz"
NAMES = {0:"unison",1:"m2",2:"M2",3:"m3",4:"M3",5:"P4",6:"tritone",
         7:"P5",8:"m6",9:"M6",10:"m7",11:"M7"}
THEORY = [0,5,7,2,10,1,11,3,9,4,8,6]


def report(name, offs):
    h = np.bincount(np.array(offs) % 12, minlength=12).astype(float)
    n = h.sum(); p = h / n
    chi, pv = chisquare(h)
    print(f"\n=== {name}  (n={int(n)})  chi2={chi:.1f} p={pv:.1e} "
          f"{'NON-UNIFORM' if pv<1e-3 else 'uniform'} ===")
    for o in THEORY:
        print(f"   {o:2d} {NAMES[o]:8s} {int(h[o]):6d}  {p[o]:.3f}  {'#'*int(p[o]*120)}")
    print(f"  P4|P5={p[5]+p[7]:.3f}  steps(m2/M2/m7/M7)={p[1]+p[2]+p[10]+p[11]:.3f} "
          f"chromatic(m2/M7)={p[1]+p[11]:.3f}  unison={p[0]:.3f}")
    return p


def main():
    c = np.load(CORPUS, allow_pickle=True)
    keep = filter_by_match(c["match"], minimum=MatchQuality.EXACT)
    labels = c["labels"][keep]; root = c["root"][keep].astype(int)
    sid = c["song_id"][keep]; t0 = c["t0"][keep]
    is_inv = np.zeros(len(labels), int); bass = np.zeros(len(labels), int)
    for i, lab in enumerate(labels):
        iv, b = derive_bass_target(lab, root[i]); is_inv[i] = iv; bass[i] = b
    print(f"EXACT {keep.sum()} chords, inversions {is_inv.sum()} ({100*is_inv.mean():.1f}%)")

    all_off, chg_off, inv_off = [], [], []
    for s in sorted(set(sid.tolist())):
        idx = np.where(sid == s)[0]; idx = idx[np.argsort(t0[idx])]
        b = bass[idx]; iv = is_inv[idx]
        for j in range(1, len(b)):
            o = (b[j] - b[j-1]) % 12
            all_off.append(o)
            if o != 0: chg_off.append(o)
            if iv[j] or iv[j-1]:  # transition touching an inversion
                inv_off.append(o)
    report("BASS motion, ALL transitions", all_off)
    report("BASS motion, CHANGES only", chg_off)
    report("BASS motion, INVERSION-touching transitions", inv_off)

    # compare to functional-root motion changes-only (baseline reference)
    root_chg = []
    for s in sorted(set(sid.tolist())):
        idx = np.where(sid == s)[0]; idx = idx[np.argsort(t0[idx])]
        r = root[idx]
        for j in range(1, len(r)):
            o = (r[j]-r[j-1]) % 12
            if o != 0: root_chg.append(o)
    report("FUNCTIONAL-ROOT motion, CHANGES only (reference)", root_chg)


if __name__ == "__main__":
    main()
