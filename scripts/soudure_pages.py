"""La page Soudure, servie sur les vrais morceaux.

    .venv/bin/python scripts/soudure_pages.py [<stem> ...]
        -> /plots/soudure_<stem>.html (une par morceau) + /plots/soudure.html

`docs/plots/soudure.html` est la page autonome du brief : un seul fichier, tout
en ligne, aucune ressource externe, et un jeu d'exemple en dur. Elle lit
`window.SONG` au chargement s'il existe — c'est le seul point d'entrée prévu
pour de vraies données. Ce script écrit donc une copie par morceau avec le
`window.SONG` du morceau devant, et rien d'autre de changé : chaque page reste
un fichier unique qu'on peut ouvrir seul.

Le mot est celui de `vote_fill.fill` — une lettre par bi-mesure, la même
matière que `scripts/bpe_lab.py`. La différence est ce qu'on en fait : bpe_lab
DÉROULE l'agglomération et la donne à lire ; ici c'est Louis qui soude, à
l'oreille, et la machine ne fait que proposer la paire la plus fréquente.

CE QUE ÇA NE RÉSOUT PAS : rien du critère d'arrêt. La page ne sait toujours pas
quand il faut s'arrêter de souder — elle déplace la décision vers l'oreille au
lieu de l'automatiser, et c'est exactement l'intention.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
sys.path.insert(0, str(HERE / "scratchpad"))

from ssm_zoo import SONGS, AUDIO          # noqa: E402
from vote_fill import fill                # noqa: E402
import order_bundle                       # noqa: E402

OUTDIR = HERE / "docs" / "plots"
GABARIT = OUTDIR / "soudure.html"


def song_json(stem: str, titre: str) -> dict:
    """La chanson au format que la page attend."""
    b = order_bundle.get(stem)
    grid = b["grid"]                       # bornes de mesures, en secondes
    R = fill(b, stem)
    mot, x0 = R["mot"], R["x0"]

    # Les jetons sont des bi-mesures, sauf là où la chanson porte une mesure
    # impaire : sur She Will Be Loved un jeton fait 3 mesures. On envoie donc
    # les bornes exactes plutôt que de supposer un pas constant — sans ça la
    # page décale tout l'audio après la mesure fautive.
    if len(x0) != len(mot) + 1:
        raise ValueError(f"{stem}: {len(x0)} bornes pour {len(mot)} jetons")

    nm = int(x0[-1])                       # mesures réellement couvertes par le mot
    return {
        "titre": titre,
        "n_mesures": nm,
        "temps_par_mesure": [round(grid[i + 1] - grid[i], 4) for i in range(nm)],
        "mot": mot,
        "jetons": [int(v) for v in x0],
        # Le mot de recherche lit la BASSE (commit c3802b1) ; celui que l'app
        # sert sur /soudure/<file> lit l'harmonie complète. Deux mots, deux
        # découpages — la page le dit, pour qu'on sache lequel on juge.
        "mot_source": "basse",
        "t0": round(grid[0], 4),
        "audio_url": f"/audio/{stem}.m4a",
    }


def page(gabarit: str, song: dict) -> str:
    """Le gabarit, avec le window.SONG du morceau posé devant."""
    tete = ("<script>window.SONG = "
            + json.dumps(song, ensure_ascii=False, separators=(",", ":"))
            + ";</script>\n")
    if "<body>" not in gabarit:
        raise ValueError("gabarit inattendu : pas de <body>")
    out = gabarit.replace("<body>", "<body>\n" + tete, 1)
    return out.replace("<title>Soudure</title>",
                       f"<title>Soudure — {song['titre']}</title>", 1)


INDEX = """<!doctype html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Soudure</title><style>
body{{margin:0 auto;max-width:520px;padding:22px 16px calc(24px + env(safe-area-inset-bottom));
 background:#f7f3e9;color:#1c1c1c;
 font:16px/1.45 -apple-system,BlinkMacSystemFont,system-ui,sans-serif}}
h1{{font:italic 400 27px/1.15 Georgia,serif;margin:0 0 4px}}
p{{color:#8a8371;font-size:14px;margin:0 0 20px}}
a{{display:flex;align-items:center;min-height:56px;padding:0 16px;margin-bottom:8px;
 background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;
 color:#1c1c1c;text-decoration:none;font-family:Georgia,serif;font-style:italic;
 font-size:18px}}
a small{{margin-left:auto;color:#8a8371;font-family:-apple-system,system-ui,sans-serif;
 font-style:normal;font-size:13px}}
</style></head><body>
<h1>Soudure</h1>
<p>La chanson est une bande de jetons. Tu soudes deux voisins, et la soudure
s’applique partout à la fois.</p>
{liens}
</body></html>
"""


def main():
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    todo = [(s, t) for s, t in SONGS if s in args] or (
        [(s, s) for s in args] if args else list(SONGS))
    gabarit = GABARIT.read_text(encoding="utf-8")

    vex = hashlib.sha1(gabarit.encode('utf-8')).hexdigest()[:8]
    liens = [f'<a href="soudure.html?v={vex}">Exemple<small>84 mesures</small></a>']
    for stem, titre in todo:
        if not (AUDIO / f"{stem}.m4a").exists():
            print(f"  ?? pas d'audio {stem}")
            continue
        try:
            song = song_json(stem, titre)
        except Exception as e:                       # noqa: BLE001
            print(f"  !! {stem}: {e}")
            continue
        html = page(gabarit, song)
        (OUTDIR / f"soudure_{stem}.html").write_text(html, encoding="utf-8")
        # L'empreinte de la page dans le lien : Safari sur iPhone garde les
        # pages en cache avec entêtement, et une correction de son invisible
        # parce qu'on relit l'ancienne page coûte un aller-retour pour rien.
        v = hashlib.sha1(html.encode("utf-8")).hexdigest()[:8]
        liens.append(f'<a href="soudure_{stem}.html?v={v}">{titre}'
                     f'<small>{song["n_mesures"]} mesures · '
                     f'{len(song["mot"])} jetons</small></a>')
        print(f"  ok {titre}: {song['n_mesures']} mes., mot {song['mot']}")

    (OUTDIR / "soudure_lab.html").write_text(
        INDEX.format(liens="\n".join(liens)), encoding="utf-8")
    print(f"wrote docs/plots/soudure_lab.html + {len(liens) - 1} pages")


if __name__ == "__main__":
    main()
