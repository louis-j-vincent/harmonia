"""La matrice SSM d'un morceau, cliquable, avec tête de lecture.

    .venv/bin/python scripts/ssm_playhead.py                 # les charts annotés
    .venv/bin/python scripts/ssm_playhead.py <stem> [<stem>…]
    .venv/bin/python scripts/ssm_playhead.py --tous          # toute la bibliothèque

    ->  /plots/ssm_<stem>.html   (une page par morceau)
    ->  /plots/ssm_playhead.html (l'index)

Louis, 2026-08-16 : « je veux une matrice ssm avec playhead cliquable ».

Le moteur est `harmonia_min/ssm_page.py` — le même que celui qu'une route du
serveur pourrait servir en direct. Ce script ne fait qu'écrire des pages
statiques dans `docs/plots/`, servies par `/plots/<nom>` : c'est la même voie
que `ssm_zoo.py` et `soudure_pages.py`, et elle ne touche pas à `server.py`.

Coût : la grille et les postérieurs musx sont déjà en cache pour la prod, donc
~1 s par morceau à chaud. À froid, c'est l'inférence musx qui domine (~20 s).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))

from harmonia_min.ssm_page import page_html            # noqa: E402

CHARTS = HERE / "harmonia_min" / "state" / "charts"
AUDIO = HERE / "docs" / "audio"
ANNOTES = HERE / "harmonia_min" / "state" / "sections"
SORTIE = HERE / "docs" / "plots"

#: Les liens doivent être cliquables depuis son téléphone : `file://` ne marche
#: pas pour lui (2026-08-07), tout passe par le serveur en Tailscale.
BASE = "http://100.89.209.63:7772"


def _charts() -> dict[str, Path]:
    """{stem audio -> chemin du chart} pour tout ce qui a un audio sur disque."""
    out = {}
    for p in sorted(CHARTS.glob("*.json")):
        try:
            stem = Path(json.loads(p.read_text(encoding="utf-8"))
                        .get("audio_url") or "").stem
        except (OSError, ValueError):
            continue
        if stem and (AUDIO / f"{stem}.m4a").exists():
            out[stem] = p
    return out


def _index(faits: list[tuple[str, str, bool]]) -> str:
    lignes = "\n".join(
        '<li><a href="ssm_{s}.html">{t}</a>{a}</li>'.format(
            s=stem, t=titre, a=' <span class="a">· tes annotations</span>' if ann else "")
        for stem, titre, ann in faits)
    return f"""<!doctype html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Matrices SSM cliquables</title><style>
body{{font:15px/1.6 -apple-system,sans-serif;background:#f7f3e9;color:#1c1c1c;
max-width:640px;margin:0 auto;padding:24px 18px}}
h1{{font:italic 600 24px Georgia,serif;margin:0 0 4px}}
p{{color:#6f6a60;font:italic 14px Georgia,serif;margin:0 0 18px}}
ul{{list-style:none;padding:0;margin:0}}
li{{padding:11px 0;border-bottom:1px solid #ddd5c4}}
a{{color:#8a2b2b;text-decoration:none;font-weight:600}}
.a{{color:#6f6a60;font-weight:400;font-size:13px}}
@media (prefers-color-scheme:dark){{body{{background:#17171a;color:#ece8e0}}
li{{border-color:#33323a}}a{{color:#d4735e}}}}
</style></head><body>
<h1>Matrices SSM cliquables</h1>
<p>La matrice chord-tone de la prod, au grain de la demi-mesure. Clique dedans :
le son y saute. {len(faits)} morceaux.</p>
<ul>
{lignes}
</ul></body></html>
"""


def main(argv: list[str]) -> int:
    dispo = _charts()
    if "--tous" in argv:
        stems = list(dispo)
    elif [a for a in argv if not a.startswith("-")]:
        stems = [a for a in argv if not a.startswith("-")]
    else:
        stems = [s for s in dispo if (ANNOTES / f"{s}.json").exists()]
        if not stems:
            stems = list(dispo)[:6]

    SORTIE.mkdir(parents=True, exist_ok=True)
    faits, rates = [], []
    for stem in stems:
        p = dispo.get(stem)
        if p is None:
            rates.append((stem, "pas de chart, ou pas d'audio sur disque"))
            continue
        chart = json.loads(p.read_text(encoding="utf-8"))
        html = page_html(chart, audio_dir=AUDIO, base_url=BASE)
        if html is None:
            rates.append((stem, "chart trop court pour une matrice"))
            continue
        (SORTIE / f"ssm_{stem}.html").write_text(html, encoding="utf-8")
        titre = chart.get("title") or stem
        faits.append((stem, titre, (ANNOTES / f"{stem}.json").exists()))
        print(f"  {BASE}/plots/ssm_{stem}.html   {titre}")

    if faits:
        (SORTIE / "ssm_playhead.html").write_text(_index(faits), encoding="utf-8")
        print(f"\nindex : {BASE}/plots/ssm_playhead.html   ({len(faits)} morceaux)")
    for stem, pourquoi in rates:
        print(f"  SAUTÉ {stem} : {pourquoi}")
    return 0 if faits else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
