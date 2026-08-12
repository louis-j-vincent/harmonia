"""Est-ce que ça régresse ? La question posée avant chaque mise en prod.

    .venv/bin/python scripts/bench.py --quick    # les 4 morceaux chauds, ~10 s
    .venv/bin/python scripts/bench.py            # TOUT ce que Louis a annoté
    .venv/bin/python scripts/bench.py --save     # fige la référence actuelle

Sort en code 1 dès qu'un morceau recule de plus de TOL — donc utilisable comme
garde-fou dans `/ship`.

POURQUOI CE FICHIER EXISTE. L'audit du 2026-08-12 (docs/session_2026-08-12_
diagnostic_boucle.md) a montré qu'une poignée de morceaux porte tout le
débogage — This Love 12 jours, Stand By Me 6, She Will Be Loved 6, Blue Lights —
et qu'ils étaient redécouverts à chaque fois au lieu d'être un banc nommé. Les
voilà nommés.

LE PIÈGE QU'IL FERME AUSSI. La référence retient l'empreinte de
`section_metric.py`. Quand la métrique change, les scores changent sans que le
modèle ait bougé d'un pouce — c'est arrivé le 2026-08-12 (fusion de même lettre
rendue gratuite : +0,028 de moyenne, zéro ligne de modèle touchée). Le banc
refuse alors de comparer et le dit, au lieu d'annoncer un gain.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
sys.path.insert(0, str(HERE / "scratchpad"))

import numpy as np                                              # noqa: E402

from ssm_zoo import SONGS, gt_sections                          # noqa: E402
from section_metric import compare                              # noqa: E402
from order_lab import new_sections, HOLDOUT                     # noqa: E402
import order_bundle                                             # noqa: E402

REF = HERE / "docs" / "bench_sections.json"
GT_DIR = HERE / "harmonia_min" / "state" / "sections"
AUDIO_DIR = HERE / "docs" / "audio"
TOL = 0.005

# Les morceaux sur lesquels Louis revient, mesurés sur six semaines d'échanges.
HOT = [
    "maroon_5_this_love",
    "ben_e_king_stand_by_me_audio",
    "maroon_5_she_will_be_loved_official_music_video",
    "jorja_smith_blue_lights_a_colors_show",
]

TITLES = dict(list(SONGS) + HOLDOUT)
HOLD = {s for s, _ in HOLDOUT}


def annotated() -> list[str]:
    """TOUS les morceaux que Louis a annotés, pas une liste figée.

    Le 2026-08-12 il a annoté un morceau de plus pendant qu'on travaillait et
    le banc, qui lisait une liste écrite à la main, ne l'aurait jamais vu. Une
    étiquette qui n'entre pas dans la mesure ne sert à rien.
    """
    if not GT_DIR.exists():
        return []
    out = []
    for p in sorted(GT_DIR.glob("*.json")):
        if (AUDIO_DIR / f"{p.stem}.m4a").exists():
            out.append(p.stem)
    return out


def title(stem: str) -> str:
    return TITLES.get(stem) or stem.replace("_", " ")[:26]


def metric_fingerprint() -> str:
    """De quoi refuser une comparaison quand c'est la règle qui a bougé."""
    h = hashlib.sha256()
    for f in ("section_metric.py",):
        h.update((HERE / "scripts" / f).read_bytes())
    return h.hexdigest()[:12]


def run(stems) -> dict:
    out = {}
    for stem in stems:
        gt = gt_sections(stem)
        if not gt:
            continue
        b = order_bundle.get(stem)
        out[stem] = compare(new_sections(b)[0], gt["sections"], b["n"])["score"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="les 4 morceaux chauds")
    ap.add_argument("--save", action="store_true", help="fige la référence")
    a = ap.parse_args()

    stems = HOT if a.quick else annotated()
    now = run(stems)
    fp = metric_fingerprint()

    if a.save:
        old = json.loads(REF.read_text()) if REF.exists() else {"scores": {}}
        scores = {**old.get("scores", {}), **{k: round(v, 4) for k, v in now.items()}}
        REF.write_text(json.dumps({"metric": fp, "scores": scores}, indent=1,
                                  sort_keys=True) + "\n")
        print(f"référence figée : {len(scores)} morceaux, métrique {fp}")
        return 0

    if not REF.exists():
        print("pas de référence — lance `--save` une fois.")
        for s, v in sorted(now.items(), key=lambda x: -x[1]):
            print(f"  {title(s):26s} {v:.3f}")
        return 0

    ref = json.loads(REF.read_text())
    stale = ref.get("metric") != fp
    old = ref.get("scores", {})

    rows = [(title(s), s, old.get(s), v) for s, v in now.items()]
    rows.sort(key=lambda r: (r[2] is None, (r[3] - r[2]) if r[2] is not None else 0))

    print(f"{'morceau':28s} {'réf':>6s} {'maintenant':>11s} {'écart':>7s}")
    down = []
    for name, stem, o, v in rows:
        tag = " ·HC" if stem in HOLD else ""
        if o is None:
            print(f"{name + tag:28s} {'—':>6s} {v:11.3f} {'nouveau':>7s}")
            continue
        d = v - o
        if d < -TOL:
            down.append((name, d))
        print(f"{name + tag:28s} {o:6.3f} {v:11.3f} {d:+7.3f}")

    known = [(o, v) for _, s, o, v in rows if o is not None]
    if known:
        O = np.array([x[0] for x in known])
        V = np.array([x[1] for x in known])
        print(f"\n{len(known)} morceaux : moyenne {O.mean():.3f} → {V.mean():.3f}"
              f"   médiane {np.median(O):.3f} → {np.median(V):.3f}")

    if stale:
        print(f"\n⚠ LA MÉTRIQUE A CHANGÉ (réf {ref.get('metric')} → {fp}).")
        print("  Les écarts ci-dessus ne mesurent PAS le modèle. Relance --save,")
        print("  et dis-le dans le message de commit.")
        return 0

    if down:
        print("\nRECUL :")
        for t, d in down:
            print(f"  {t} {d:+.3f}")
        return 1
    print("\nzéro recul.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
