"""Build the ear-labelling deck: pairs of passages for Louis to judge.

Why this exists (2026-08-02): the criteria study fitted "same SALAMI letter",
but ~38% of Billboard's same-letter pairs disagree in length by >5% — so that
label TOLERATES exactly the length mismatch Louis's rule refuses. A model fitted
to it is fitted against us. The target has to be relabelled to the decision we
actually take: *should these two passages be written as ONE block?*

Design decisions that keep the labels honest:
  * Cards carry AUDIO, bar ranges and durations only — NOT the written chords.
    The chart writes ONE folded block per letter, so both sides would print the
    same chords by construction and anchor the listener toward "same". The ear
    is the instrument here.
  * Metric values are computed (for ordering) but hidden until after the verdict,
    so the number cannot anchor the judgement.
  * Ordering is by INFORMATIVENESS, not by song. Tier A = "sounds alike but the
    lengths differ" (align>=0.80, len<0.95) — the disputed class where Billboard
    and Louis's rule actively disagree, and where a label is worth most.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "scripts"))
from fold_criteria_billboard import slot_cos          # noqa: E402
from fold_criteria_ourcharts import chroma_span       # noqa: E402

CHARTS = Path(HERE, "harmonia_min/state/charts")
AUDIO = Path(HERE, "docs/audio")
OUT = Path(HERE, "harmonia_min/state/reports/fold_label_deck.json")


def main():
    from harmonia_min.nnls_features import extract_bothchroma
    cards = []
    for f in sorted(CHARTS.glob("min_*.json")):
        ch = json.load(open(f))
        stem = f.stem[4:]
        ap = AUDIO / f"{stem}.m4a"
        if not ap.exists():
            continue
        occ = [(s["label"], tuple(r)) for s in ch["sections"]
               for r in (s.get("barRanges") or [])]
        if len(occ) < 2:
            continue
        arr, times = extract_bothchroma(ap)
        arr = np.asarray(arr)
        X = arr[:, 12:24] + arr[:, :12] if arr.shape[1] >= 24 else arr[:, :12]
        X = X / np.maximum(X.sum(1, keepdims=True), 1e-9)
        dt = float(np.median(np.diff(times)))
        grid = np.array(ch["barGrid"], float)
        tspan = lambda a, b: (float(grid[a]), float(grid[min(b + 1, len(grid) - 1)]))
        cs = {}
        for L, (a, b) in occ:
            cs[(L, a, b)] = chroma_span(X, dt, *tspan(a, b))
        # Phase search (the study's finding, 2026-08-02): OUR occurrences are
        # out of phase far more often than Billboard's — sliding one span by a
        # few bars rescues 24.4% of our refused pairs vs 2.4% of Billboard's,
        # and costs nothing on the corpus. Without it every align here is
        # depressed and the tiering below is measuring our bar-phase error
        # instead of whether the music repeats.
        bar = float(np.median(np.diff(grid))) if len(grid) > 2 else 2.0
        SHIFTS = [k * bar for k in (0, 1, -1, 2, -2, 3, -3, 4, -4)]

        def best_align(v1, t2a, t2b):
            best, arg = -1.0, 0.0
            for d in SHIFTS:
                w = chroma_span(X, dt, t2a + d, t2b + d)
                if w is None:
                    continue
                s = float(slot_cos(v1, w))
                if s > best:
                    best, arg = s, d
            return best, arg

        for i in range(len(occ)):
            for j in range(i + 1, len(occ)):
                (L1, (a1, b1)), (L2, (a2, b2)) = occ[i], occ[j]
                v1, v2 = cs[(L1, a1, b1)], cs[(L2, a2, b2)]
                if v1 is None or v2 is None:
                    continue
                n1, n2 = b1 - a1 + 1, b2 - a2 + 1
                # A pair is only an EAR question if both sides are holdable in
                # the head. blue_bossa's detector emitted a 90-bar and a 58-bar
                # "occurrence" (3m43 vs 1m20) — comparing those is not a merge
                # decision, it is a segmentation failure upstream, and asking
                # for a verdict on it spends the listener for nothing.
                if not (2 <= n1 <= 24 and 2 <= n2 <= 24):
                    continue
                al_raw = float(slot_cos(v1, v2))
                al, shift = best_align(v1, *tspan(a2, b2))
                la = min(n1, n2) / max(n1, n2)
                same = L1 == L2
                # tier = how much a human label here is worth
                if same and al >= .80 and la < .95:
                    tier, why = "A", "sonne pareil, longueurs differentes"
                elif abs(al - .85) < .08:
                    tier, why = "B", "juste sur le seuil"
                elif same:
                    tier, why = "C", "meme lettre, cas net"
                else:
                    tier, why = "D", "lettres differentes (controle)"
                t1a, t1b = tspan(a1, b1)
                t2a, t2b = tspan(a2, b2)
                cards.append({
                    "id": f"{stem}|{L1}{a1}-{b1}|{L2}{a2}-{b2}",
                    "song": stem, "title": ch.get("title") or stem,
                    "audio": f"/audio/{stem}.m4a",
                    "a": {"lab": L1, "b0": a1, "b1": b1, "n": n1, "t0": t1a, "t1": t1b},
                    "b": {"lab": L2, "b0": a2, "b1": b2, "n": n2, "t0": t2a, "t1": t2b},
                    "align": round(al, 3), "align_raw": round(al_raw, 3),
                    "shift_bars": round(shift / bar) if bar else 0,
                    "len": round(la, 3),
                    "same_letter": same, "tier": tier, "why": why})
        print(f"  {stem[:44]:<46} {len(occ)} occurrences", flush=True)

    # Tier D (different letters) is the easy class — keep a handful as
    # attention checks rather than spending Louis's ears on them.
    order = {"A": 0, "B": 1, "C": 2, "D": 3}
    cards.sort(key=lambda c: (order[c["tier"]], -c["align"]))
    # interleave songs so he isn't stuck on one tune for 20 cards
    by_tier = {}
    for c in cards:
        by_tier.setdefault(c["tier"], []).append(c)
    by_tier["D"] = sorted(by_tier.get("D", []), key=lambda c: -c["align"])[:24]
    final, seen = [], set()
    for t in "ABCD":
        pool = by_tier.get(t, [])
        while pool:
            rest = []
            used = set()
            for c in pool:
                if c["song"] in used:
                    rest.append(c)
                else:
                    used.add(c["song"]); final.append(c)
            pool = rest
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(final, open(OUT, "w"))
    from collections import Counter
    print("\ncards:", len(final), Counter(c["tier"] for c in final))
    print("songs:", len({c['song'] for c in final}), "->", OUT)


if __name__ == "__main__":
    main()
