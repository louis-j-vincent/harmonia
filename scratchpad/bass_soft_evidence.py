"""bass_soft_evidence.py — the premise's second chance: is the D there at all?

``bass_premise_check.py`` answered the ARGMAX question and got 0/13: at the 13
Let It Be D-7 spots no bass source's top pitch class is D.  That kills the
simple discriminator but not the premise — a repair could still work off
*secondary* mass if the D is present-but-losing.  This script asks the softer
question and, crucially, asks it against a NULL.

Two statistics per spot, both peaked over a +-``WIN`` s window (UG timing is
hand-made, so an exact-span reading under-reads by construction):

* ``bassD``   — music-x-lab's frame bass posterior on the substitute's root pc.
* ``triadD``  — the same model's 73-way triad posterior on ``<root>:min``.

The null is every 0.3 s window inside a music-x-lab ``F:maj`` segment that is
**not** within 1.5 s of any UG D-7 — i.e. the spans a recovery rule must leave
alone.  A statistic is only useful if the 13 true spots separate from that null;
"the D is visible" means nothing if D is equally visible on 700 windows of plain
F.  Reported as a full precision/recall sweep over thresholds, not a single
number, so the operating point is chosen with its cost in view.

WHAT THIS DOES NOT ANSWER
-------------------------
It measures separability of the EVIDENCE, not the value of a repair.  Even a
perfectly separating statistic would still have to survive the UG scorer without
minting ADDED/ROOT errors elsewhere, which is a separate experiment.  And the
null is drawn from ONE song: a threshold tuned here is a hypothesis about Let It
Be, not about the corpus (CLAUDE.md rule #5).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from harmonia.models.musx_bass import _parse_root  # noqa: E402
from harmonia.models.musx_redecode import FRAME_DT  # noqa: E402

PC = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
TRIADS = ["maj", "min", "sus4", "sus2", "dim", "aug"]
WIN = 0.6          # peak search half-width around the UG span
NULL_W = 0.3       # null window length
GUARD = 1.5        # keep the null this far from any true spot


def _load(slug: str):
    z = np.load(REPO / "data" / "cache" / "musx_probs" / f"{slug}.npz")
    lab = []
    for line in (REPO / "data" / "cache" / "musx_infer"
                 / f"{slug}_submission.lab").read_text().splitlines():
        f = line.split()
        if len(f) >= 3:
            lab.append((float(f[0]), float(f[1]), f[2]))
    sc = json.loads((REPO / "scratchpad" / f"ug_score_{slug}.json").read_text())
    return z["triad"], z["bass"], lab, sc


def _peak(pbass, ptriad, t0, t1, pc, step=0.05):
    """Max (bass mass, min-triad mass) over sliding sub-windows of the span."""
    bb = tt = 0.0
    t = t0
    while t <= t1 - 1e-9:
        a = max(0, int(round(t / FRAME_DT)))
        b = min(pbass.shape[0], max(a + 1, int(round((t + NULL_W) / FRAME_DT))))
        bb = max(bb, float(pbass[a:b, 1 + pc].mean()))
        tt = max(tt, float(ptriad[a:b, 1 + 12 + pc].mean()))
        t += step
    return bb, tt


def main() -> None:
    slug = "let_it_be_remastered_2009"
    sub_name, host_root, host_kind = "D-7", 5, "maj"      # D-7 absorbed by F:maj
    sub_pc = _parse_root(sub_name)
    ptriad, pbass, lab, sc = _load(slug)

    spots = [(float(e["t0"]), float(e["t1"])) for e in sc["errors"]
             if e["cls"] == "MISSED" and e.get("ug") == sub_name]

    pos = []
    for t0, t1 in spots:
        bb, tt = _peak(pbass, ptriad, t0 - WIN, t1 + WIN, sub_pc)
        pos.append((t0, bb, tt))

    neg = []
    for t0, t1, s in lab:
        r = _parse_root(s.split("/")[0].split(":")[0]) if s not in ("N", "X") else None
        q = s.split(":")[1].split("/")[0] if ":" in s else "maj"
        if r != host_root or q != host_kind:
            continue
        t = t0
        while t + NULL_W <= t1:
            if all(not (t0s - GUARD <= t <= t1s + GUARD) for t0s, t1s in spots):
                a = int(round(t / FRAME_DT))
                b = int(round((t + NULL_W) / FRAME_DT))
                neg.append((t, float(pbass[a:b, 1 + sub_pc].mean()),
                            float(ptriad[a:b, 1 + 12 + sub_pc].mean())))
            t += 0.15

    print(f"=== {slug}: is the {PC[sub_pc]} evidence separable from plain "
          f"{PC[host_root]}:{host_kind}? ===")
    print(f"  positives: {len(pos)} UG {sub_name} spots (peak over +-{WIN}s)")
    print(f"  null:      {len(neg)} {NULL_W}s windows inside {PC[host_root]}:"
          f"{host_kind} segments, >= {GUARD}s from any spot\n")

    print("  spot     peak bass-D   peak triad-Dmin")
    for t0, bb, tt in pos:
        print(f"  {t0:7.1f}      {bb:.3f}          {tt:.3f}")
    pb = np.array([p[1] for p in pos]); pt = np.array([p[2] for p in pos])
    nb = np.array([n[1] for n in neg]); nt = np.array([n[2] for n in neg])
    print(f"\n  positives  bass-D  median {np.median(pb):.3f}  max {pb.max():.3f}")
    print(f"  null       bass-D  median {np.median(nb):.3f}  p95 "
          f"{np.percentile(nb, 95):.3f}  max {nb.max():.3f}")
    print(f"  positives  triadDm median {np.median(pt):.3f}  max {pt.max():.3f}")
    print(f"  null       triadDm median {np.median(nt):.3f}  p95 "
          f"{np.percentile(nt, 95):.3f}  max {nt.max():.3f}")

    for name, p, n in (("bass-D", pb, nb), ("triad-Dmin", pt, nt)):
        print(f"\n  --- sweep on {name} ---")
        print("   thresh   recall(13)   null windows fired   precision")
        for th in (0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40):
            r = int((p >= th).sum()); f = int((n >= th).sum())
            prec = r / (r + f) if (r + f) else 0.0
            print(f"   {th:.2f}     {r:2d}/13        {f:4d}/{len(n)}"
                  f"            {prec:.2f}")

    out = {"slug": slug, "sub": sub_name, "host": f"{PC[host_root]}:{host_kind}",
           "positives": [{"t0": t, "bassD": b, "triadDmin": c} for t, b, c in pos],
           "null_n": len(neg),
           "null_bassD_p50_p95_max": [float(np.median(nb)),
                                      float(np.percentile(nb, 95)), float(nb.max())],
           "null_triadDmin_p50_p95_max": [float(np.median(nt)),
                                          float(np.percentile(nt, 95)), float(nt.max())]}
    (REPO / "scratchpad" / "bass_soft_evidence_letitbe.json").write_text(
        json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
