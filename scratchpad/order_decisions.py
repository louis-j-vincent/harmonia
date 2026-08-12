"""Le jeu de décisions « poser ce bloc de 4 maintenant, ou le différer ».

Mesuré en E6/E8 : le parcours unique où l'on diffère TOUT reproduit exactement la
prod (0,789 · Blue Lights 0,696) ; celui où l'on pose TOUT donne 0,795 mais casse
The Walk. Le plafond de cet espace est 0,795, et il ne demande **qu'un bit par
morceau** (Blue Lights : le 4e point de décision, la famille B de la mesure 13).

Ce fichier construit le jeu : un point de décision = une ancre où seul le bloc de
4 existe. Pour chacun, des descripteurs et une étiquette = le gain marginal
(poser celui-là seul, tout le reste différé, contre tout différer).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
sys.path.insert(0, str(HERE / "scratchpad"))

import order_bundle                       # noqa: E402
import order_search as OS                 # noqa: E402
import order_mixed as OM                  # noqa: E402
import harmonia_min.voice_sections as VS  # noqa: E402
from ssm_zoo import SONGS                 # noqa: E402


def feats(c4, ctx):
    """Les descripteurs d'un point de décision, tous calculables en ligne."""
    b, cur, hard = ctx["b"], ctx["cursor"], ctx["hard"]
    n = b["n"]
    fam = sorted([c4["b0"]] + c4["occ"])
    sc = c4["sc"]
    d = np.diff(fam) if len(fam) > 1 else np.array([0.0])
    bnds = sorted({x for c in fam for x in (c, c + 4) if 0 < x < n})
    sup = (sum(1 for x in bnds if any(abs(x - h) <= 1 for h in hard))
           / max(1, len(bnds)))
    # le bloc de 8 à la même ancre, en IGNORANT le veto des pics : existe-t-il ?
    free8 = VS._peaks(_sc(b, cur, 8), cur, n, VS.THR8, 8, ctx["claimed"])
    return {
        "n_occ": len(fam),
        "couv": len(fam) * 4 / n,
        "sc_moy": float(np.mean([sc[p] for p in c4["occ"]])),
        "sc_min": float(np.min([sc[p] for p in c4["occ"]])),
        "reg": float(np.std(d) / max(1e-6, np.mean(d))) if len(fam) > 2 else 1.0,
        "per16": float(np.mean(d % 8 == 0)) if len(fam) > 1 else 0.0,
        "appui": sup,
        "pic_ici": float(any(abs(cur - h) <= 1 for h in hard)),
        "pic_fin": float(any(abs(cur + 4 - h) <= 1 for h in hard)),
        "n8_libre": len(free8),
        "bloque_par_pic": float(any(cur + 1 < h < cur + 7 for h in hard)),
        "pos": cur / n,
    }


def _sc(b, p, L):
    n, S, M, mute = b["n"], b["S"], b["M"], b["mute"]
    return VS.block_score(VS._slide(M, p, L, n), VS._slide(S, p, L, n),
                          p, n, mute=mute, block=L)


def collect(stem):
    """[(feats, gain)] pour un morceau."""
    b = order_bundle.get(stem)
    n = b["n"]
    base = OS.score(OM.sections(b, OM.decide_only4(()), stride=4), stem, n)

    rec = []

    def probe_rec(c8, c4, ctx):
        if c8 is not None:
            return c8
        rec.append(feats(c4, ctx))
        return None

    OM.sections(b, probe_rec, stride=4)
    out = []
    for i in range(len(rec)):
        bits = tuple(1 if j == i else 0 for j in range(len(rec)))
        s = OS.score(OM.sections(b, OM.decide_only4(bits), stride=4), stem, n)
        out.append((rec[i], s - base, s, base))
    return out


def main():
    import json
    data = []
    for stem, title in SONGS:
        rows = collect(stem)
        print(f"{title}  ({len(rows)} points, base {rows[0][3]:.3f})"
              if rows else f"{title}  (aucun point)")
        for f, g, s, _ in rows:
            data.append({"stem": stem, "titre": title, "gain": g, "score": s,
                         **f})
            flag = "  <<<" if g > 0.005 else ("   --" if g < -0.005 else "")
            print(f"    gain {g:+.3f} -> {s:.3f}  n_occ={f['n_occ']} "
                  f"couv={f['couv']:.2f} sc={f['sc_moy']:.2f}/{f['sc_min']:.2f} "
                  f"reg={f['reg']:.2f} per16={f['per16']:.2f} "
                  f"appui={f['appui']:.2f} n8={f['n8_libre']} "
                  f"bloq={f['bloque_par_pic']:.0f} pos={f['pos']:.2f}{flag}")
    (HERE / "scratchpad" / "order_decisions.json").write_text(json.dumps(data))
    g = np.array([d["gain"] for d in data])
    print(f"\n{len(data)} points · {int((g > 0.005).sum())} gagnants · "
          f"{int((g < -0.005).sum())} perdants · {int((abs(g) <= 0.005).sum())} neutres")


if __name__ == "__main__":
    main()
