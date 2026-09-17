"""Ce qu'une seule section annotée fait au découpage automatique.

Louis, 2026-09-17, sur Don't Want My Love : « ça a changé les sections qui
étaient détectées automatiquement et qui étaient pas mal », puis « je veux une
visualisation explicite de tout ça avec le chart brut ».

La page met côte à côte, mesure par mesure et sur le même axe :

  * le chart BRUT, les accords tels que le chart les porte ;
  * ce que la machine trouve TOUTE SEULE (SongFormer, puis le repli) ;
  * ce que rend l'inférence quand on lui donne UN trait, à deux endroits
    différents — dont un placé exactement là où la machine avait déjà mis sa
    section, pour montrer que le dégât ne vient pas du trait.

CE QU'ELLE MONTRE, et c'est le diagnostic : `sections_inferer` ne lit JAMAIS
les sections du chart. Elle repart du mot de bi-mesures et relance
l'algorithme des quatre mots sur tout le morceau, en ne figeant que les traits
reçus. Le découpage automatique n'est pas une entrée — il est jeté.

CE QU'ELLE NE FAIT PAS : corriger quoi que ce soit. Louis, 2026-09-17 :
« n'essaye pas de fix, aide-moi juste à établir le diagnostic, c'est moi qui
dirai comment fix. »

    python -m tools.annotation_degats                       # Don't Want My Love
    python -m tools.annotation_degats --chart min_xxx --trait A:15-22
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from harmonia.settings import SETTINGS

NOMS = "C C# D Eb E F F# G Ab A Bb B".split()
#: une teinte stable par étiquette, pour que l'œil suive une lettre d'une
#: bande à l'autre — c'est toute la lisibilité de la page.
TEINTES = ("#8a2b2b", "#2f5fa8", "#1f7a6b", "#b06a1f", "#7b4ea3", "#4a7c3f",
           "#a8336a", "#556b2f", "#8a6d3b", "#3b6f8a")


def couleur(label: str) -> str:
    lab = (label or "?").strip().lower()
    if lab in ("intro", "outro", "silence"):
        return "#8a8371"                       # le décor, pas une lettre
    return TEINTES[sum(ord(c) for c in lab) % len(TEINTES)]


def nom_accord(c: dict) -> str:
    if not c:
        return "·"
    if c.get("nc"):
        return "N.C."
    q = {"": "", "min": "m", "dom": "7", "hdim": "ø", "dim": "°"}.get(
        c.get("q"), c.get("q") or "")
    s = NOMS[int(c.get("root", 0)) % 12] + q
    b = c.get("bass", -1)
    if b is not None and b >= 0 and b != c.get("root"):
        s += "/" + NOMS[int(b) % 12]
    return s


def par_mesure(sections: list[dict], n: int) -> list[str]:
    """[étiquette par mesure] depuis une liste de sections à `barRanges`."""
    out = ["?"] * n
    for s in sections or []:
        for a, b in (s.get("barRanges") or []):
            for i in range(max(0, a), min(n, b + 1)):
                out[i] = s.get("label") or "?"
    return out


def depuis_inferer(sections: list[dict], n: int) -> tuple[list[str], list[str]]:
    """(étiquette, source) par mesure, depuis la réponse de l'inférence."""
    lab, src = ["?"] * n, [""] * n
    for s in sections or []:
        for i in range(max(0, s["mesure_debut"] - 1), min(n, s["mesure_fin"])):
            lab[i] = s.get("label") or "?"
            src[i] = s.get("source") or ""
    return lab, src


def trace(cle: str, label: str, b0: int, b1: int) -> dict:
    """Le déroulé de l'algorithme sur un trait donné, étape par étape.

    On refait à la main ce que fait `sections_inferer`, uniquement pour
    pouvoir montrer les valeurs intermédiaires — le mot, chaque soudure, le
    coût de chaque hypothèse. Aucune décision n'est prise ici.
    """
    from harmonia.phrases4 import cout, grouper_restes, merges4, nommer, phrases
    from harmonia.soudure import mot_sur_traits, song_du_chart, traits_propres

    chart = json.loads((SETTINGS.charts_dir / f"{cle}.json")
                       .read_text(encoding="utf-8"))
    song = song_du_chart(chart, audio_dir=SETTINGS.audio_dir)
    gardes, _ = traits_propres(
        [{"label": label, "mesure_debut": b0, "mesure_fin": b1}],
        song["n_mesures"])
    bornes, mot, src = mot_sur_traits(
        chart, [(x, y) for x, y, _ in gardes], audio_dir=SETTINGS.audio_dir)
    idx = {b: j for j, b in enumerate(bornes)}
    fixes = [(idx[x], idx[y + 1] - 1, lab) for x, y, lab in gardes]
    depart, j = [], 0
    for j0, j1, _ in fixes:
        while j < j0:
            depart.append((j, j + 1, mot[j])); j += 1
        depart.append((j0, j1 + 1, mot[j0:j1 + 1])); j = j1 + 1
    while j < len(mot):
        depart.append((j, j + 1, mot[j])); j += 1
    geles = {(j0, j1 + 1) for j0, j1, _ in fixes}

    hypos = []
    for cible in (4, 6):
        steps = merges4(mot, cible=cible, depart=depart, geles=geles)
        blocs = grouper_restes(nommer(steps[-1]["jetons"], cible, geles=geles),
                               mot, cible)
        hypos.append({
            "cible": cible, "cout": cout(blocs, cible),
            "soudures": [{"paire": s_["paire"], "compte": s_["compte"],
                          "reste": len(s_["jetons"])} for s_ in steps[1:]],
            "blocs": [{"m0": bornes[b["j0"]] + 1, "m1": bornes[b["j1"]],
                       "label": b["label"] + ("′" if b["prime"] else ""),
                       "type": b["type"], "queue": bool(b["queue"])}
                      for b in blocs]})
    _secs, info = phrases(mot, depart=depart, geles=geles)
    return {"mot": mot, "source_mot": src, "n_jetons": len(bornes) - 1,
            "trait": (label, b0, b1, fixes[0][0], fixes[0][1]),
            "contenu_trait": mot[fixes[0][0]:fixes[0][1] + 1],
            "bornes": [b + 1 for b in bornes],
            "hypos": hypos, "retenue": info["cible"], "couts": info["couts"]}


def collecte(cle: str, traits: list[tuple]) -> dict:
    from harmonia.pipeline import analyze
    from harmonia.server.app import create_app
    from harmonia.soudure import accords_par_mesure

    chart = json.loads((SETTINGS.charts_dir / f"{cle}.json")
                       .read_text(encoding="utf-8"))
    n = int(chart.get("nBars") or 0)
    grid = [float(t) for t in (chart.get("barGrid") or [])]
    stem = Path(chart.get("audio_url") or "").stem

    # 1. la machine toute seule — on la REJOUE, parce que la version d'origine
    #    a été écrasée par la validation ; la chaîne est déterministe à caches
    #    chauds, donc c'est bien ce qu'elle avait trouvé.
    auto = analyze(SETTINGS.audio_dir / f"{stem}.m4a",
                   title=chart.get("title") or stem, file_key=cle,
                   audio_url=f"/audio/{stem}.m4a")

    # 2. ce que donne l'inférence avec UN trait, à chaque endroit demandé
    client = create_app().test_client()
    apres = []
    for label, b0, b1 in traits:
        r = client.post(f"/api/sections/inferer/{cle}", json={"humain": [
            {"label": label, "mesure_debut": b0, "mesure_fin": b1}]}).get_json()
        lab, src = depuis_inferer(r.get("sections") or [], n)
        apres.append({"titre": f"ton trait {label} sur {b0}-{b1}",
                      "lab": lab, "src": src,
                      "ecartes": r.get("ecartes") or []})

    return {
        "cle": cle, "titre": chart.get("title") or stem, "n": n,
        "audio": chart.get("audio_url") or f"/audio/{stem}.m4a",
        "grid": [round(t, 3) for t in grid],
        "accords": [" ".join(nom_accord(c) for c in (b or [])) or "·"
                    for b in accords_par_mesure(chart)][:n],
        "auto": par_mesure(auto.get("sections"), n),
        "chart": par_mesure(chart.get("sections"), n),
        "apres": apres,
        "trace": trace(cle, *traits[-1]),
    }


CSS = """
*{box-sizing:border-box}
body{margin:0 auto;padding:14px 12px 40px;background:#faf6ec;color:#2c2820;
 font:15px/1.45 system-ui,-apple-system,sans-serif;max-width:900px}
h1{font-size:19px;margin:0 0 4px}
h2{font-size:15.5px;margin:22px 0 6px}
.note{color:#8a8371;font-size:12.5px}
.lede{background:#fff7df;border:1px solid #e8d9a8;border-radius:10px;
 padding:10px 12px;font-size:14px;margin:10px 0}
.bandes{border:1px solid #ddd3b8;border-radius:10px;background:#fffdf7;
 padding:9px 10px;margin:10px 0}
.ligne{display:flex;align-items:center;gap:8px;margin:3px 0}
/* Les quatre bandes doivent tenir dans la largeur SANS scroll : leur seul
   intérêt est d'être comparées d'un coup d'œil, et une comparaison qu'il faut
   faire défiler n'en est plus une. Les blocs les plus étroits perdent leur
   étiquette, pas leur place — l'infobulle et le tableau plus bas la donnent. */
.nom{flex:0 0 76px;font:600 10.5px system-ui;color:#2c2820;text-align:right;
 line-height:1.25}
.bande{display:flex;flex:1 1 auto;min-width:0;height:24px;border-radius:4px;
 overflow:hidden}
.bloc{display:flex;align-items:center;justify-content:center;color:#fff;
 font:700 10px system-ui;border-right:1px solid #fffdf7;cursor:pointer;
 overflow:hidden;white-space:nowrap}
.bloc.humain{box-shadow:inset 0 0 0 2px #2c2820}
table{border-collapse:collapse;width:100%;font-size:13px;margin-top:6px}
th,td{padding:4px 6px;border-bottom:1px solid #eee3c8;text-align:left;
 white-space:nowrap}
th{font:600 11.5px system-ui;color:#8a8371;position:sticky;top:0;background:#faf6ec}
td.m{color:#8a8371;font-variant-numeric:tabular-nums;width:38px}
td.a{font:600 13px ui-monospace,monospace}
.et{display:inline-block;min-width:44px;padding:1px 5px;border-radius:4px;
 color:#fff;font:700 10.5px system-ui;text-align:center}
.et.vide{background:transparent;color:#c9bb93}
.chg{background:#fff2f2}
.tab{overflow-x:auto;border:1px solid #ddd3b8;border-radius:10px;background:#fffdf7;
 padding:4px 8px}
.etape{border-left:3px solid #ddd3b8;padding:2px 0 2px 10px;margin:10px 0;
 font-size:13.5px}
.etape ol{margin:6px 0 0;padding-left:20px}
.etape li{margin:2px 0}
.mot{display:flex;flex-wrap:wrap;gap:2px;margin:7px 0 4px}
.mot span{flex:0 0 auto;width:20px;height:22px;border-radius:3px;color:#fff;
 font:700 11px ui-monospace,monospace;display:flex;align-items:center;
 justify-content:center}
.hypo{border:1px solid #ddd3b8;border-radius:9px;padding:7px 10px;margin:8px 0;
 background:#fffdf7;font-size:13px}
.hypo.gagne{border-color:#8a2b2b;background:#fffaf5}
.hypo ol{margin:4px 0;padding-left:20px;font-size:12.5px;color:#5c5647}
.hypo table{margin-top:4px}
button{border:1px solid #d8cfb4;border-radius:9px;background:#fff;padding:7px 11px;
 font:600 13px system-ui;cursor:pointer;min-height:40px;color:#2c2820}
"""

JS = r"""
const au=document.getElementById('au');let stop=null;
au.addEventListener('timeupdate',()=>{if(stop!=null&&au.currentTime>=stop){au.pause();stop=null;}});
function jouer(src,t0,t1){
  const go=()=>{try{au.currentTime=t0;}catch(e){}stop=t1;au.play().catch(()=>{});};
  if(au.getAttribute('src')!==src){au.setAttribute('src',src);
    au.addEventListener('loadedmetadata',go,{once:true});au.load();}
  else go();
}
"""


def bandes(d: dict) -> str:
    """Les bandes comparées, une ligne par lecture du morceau."""
    lignes = [("la machine toute seule", d["auto"], None),
              ("le chart d'aujourd'hui", d["chart"], None)]
    for a in d["apres"]:
        lignes.append((a["titre"], a["lab"], a["src"]))
    out = ["<div class=bandes>"]
    for nom, lab, src in lignes:
        out.append(f"<div class=ligne><div class=nom>{html.escape(nom)}</div>"
                   "<div class=bande>")
        i = 0
        while i < d["n"]:
            j = i
            while j + 1 < d["n"] and lab[j + 1] == lab[i] and \
                    (src is None or src[j + 1] == src[i]):
                j += 1
            larg = 100.0 * (j - i + 1) / d["n"]
            t0 = d["grid"][i] if i < len(d["grid"]) else 0
            t1 = d["grid"][min(j + 1, len(d["grid"]) - 1)]
            sien = " humain" if src and src[i] == "humain" else ""
            titre = f"mesures {i+1}-{j+1}" + (f" · {src[i]}" if src and src[i] else "")
            out.append(
                f"<div class='bloc{sien}' style=\"width:{larg:.3f}%;"
                f"background:{couleur(lab[i])}\" title=\"{titre}\" "
                f"onclick=\"jouer('{d['audio']}',{t0:.2f},{t1:.2f})\">"
                f"{html.escape(lab[i]) if larg > 6 else ''}</div>")
            i = j + 1
        out.append("</div></div>")
    out.append("<div class=note style='margin-top:6px'>Touche une bande pour "
               "l'écouter. Le liseré noir marque le bloc que tu as tracé.</div>")
    out.append("</div>")
    return "".join(out)


def tableau(d: dict) -> str:
    cols = ["la machine", "le chart"] + [a["titre"] for a in d["apres"]]
    out = ["<div class=tab><table><tr><th>mes.</th><th>accords</th>"]
    out += [f"<th>{html.escape(c)}</th>" for c in cols]
    out.append("</tr>")
    for i in range(d["n"]):
        vals = [d["auto"][i], d["chart"][i]] + [a["lab"][i] for a in d["apres"]]
        chg = " class=chg" if len({v for v in vals}) > 1 else ""
        out.append(f"<tr{chg}><td class=m>{i+1}</td>"
                   f"<td class=a>{html.escape(d['accords'][i] if i < len(d['accords']) else '·')}</td>")
        for k, v in enumerate(vals):
            src = d["apres"][k - 2]["src"][i] if k >= 2 else ""
            bord = ";box-shadow:inset 0 0 0 2px #2c2820" if src == "humain" else ""
            out.append(f"<td><span class='et' style=\"background:{couleur(v)}{bord}\" "
                       f"title=\"{html.escape(src)}\">{html.escape(v)}</span></td>")
        out.append("</tr>")
    out.append("</table></div>")
    return "".join(out)


def bloc_trace(d: dict) -> str:
    """Le déroulé lisible de l'algorithme, avec les vraies valeurs."""
    t = d["trace"]
    lab, b0, b1, j0, j1 = t["trait"]
    mot = t["mot"]
    lettres = sorted(set(mot))
    part = max(mot.count(c) for c in lettres) / max(1, len(mot))
    B = ["<h2>Comment l'algorithme s'y prend, pas à pas</h2>",
         "<div class=etape><b>1. La grille de jetons.</b> Le morceau est "
         f"découpé en <b>{t['n_jetons']} jetons</b> de deux mesures, dont les "
         "bords épousent ton trait. Ton trait "
         f"<b>{html.escape(lab)} sur les mesures {b0}-{b1}</b> occupe les "
         f"jetons {j0} à {j1}.</div>",
         "<div class=etape><b>2. Le mot.</b> Chaque jeton reçoit une lettre "
         "selon ce qu'il contient — deux jetons qui sonnent pareil reçoivent la "
         f"même. Source : {html.escape(t['source_mot'])}.<div class=mot>"
         + "".join(f"<span style=\"background:{couleur(c)}\">{c}</span>"
                   for c in mot) + "</div>"
         + f"<span class=note>{len(lettres)} lettre(s) distincte(s) pour "
         f"{len(mot)} jetons — <b>{part:.0%} du morceau porte la même</b>.</span>"
         "</div>",
         "<div class=etape><b>3. Le point de départ.</b> Ton trait entre comme "
         f"UNE unité gelée, de contenu « {html.escape(t['contenu_trait'])} ». "
         "Elle ne pourra plus être soudée à personne. Tout le reste entre en "
         "unités d'un seul jeton.</div>",
         "<div class=etape><b>4. L'agglomération.</b> On soude la paire "
         "adjacente la plus fréquente, sans jamais dépasser la taille cible, et "
         "on s'arrête dès qu'aucune paire soudable ne se répète. Deux tailles "
         "cibles sont essayées.</div>"]
    for h in t["hypos"]:
        gagne = h["cible"] == t["retenue"]
        B.append(f"<div class='hypo{' gagne' if gagne else ''}'>"
                 f"<b>cible {h['cible']} bi-mesures</b> — coût {h['cout']}"
                 + (" · <b>retenue</b>" if gagne else " · écartée") + "<ol>")
        for s_ in h["soudures"]:
            B.append(f"<li>souder « {html.escape(s_['paire'][0])} » + "
                     f"« {html.escape(s_['paire'][1])} », vu {s_['compte']}× "
                     f"→ {s_['reste']} unités</li>")
        B.append("</ol><table>")
        for b in h["blocs"]:
            B.append(f"<tr><td class=m>{b['m0']}-{b['m1']}</td>"
                     f"<td><span class=et style=\"background:{couleur(b['label'])}\">"
                     f"{html.escape(b['label'])}</span></td>"
                     f"<td class=a>{html.escape(b['type'])}</td>"
                     f"<td class=note>{'trop court, recollé' if b['queue'] else ''}</td></tr>")
        B.append("</table></div>")
    B.append("<div class=etape><b>5. On garde l'hypothèse la moins chère</b> — "
             f"coûts {html.escape(json.dumps(t['couts']))}, donc la cible "
             f"{t['retenue']}.</div>")
    B.append("<div class=etape><b>6. Le nommage, dans cet ordre exact.</b> "
             "Chaque bloc reçoit son étiquette par la PREMIÈRE règle qui "
             "s'applique :<ol>"
             "<li><b>humain</b> — le bloc EST ton trait : il garde ton nom ;</li>"
             "<li><b>propage</b> — son contenu est <i>exactement</i> celui d'un "
             "de tes traits : il prend ton nom ;</li>"
             "<li><b>ressemble</b> — il ressemble assez à un de tes traits "
             "(ressemblance d'accords ≥ 0,75) : il prend ton nom ;</li>"
             "<li><b>algo</b> — sinon, une lettre neuve que tu n'as pas "
             "utilisée.</li></ol>"
             f"C'est la règle 2 qui fait les dégâts ici : ton trait a pour "
             f"contenu « {html.escape(t['contenu_trait'])} », et "
             f"<b>trois autres blocs ont exactement le même contenu</b>. Ils "
             "prennent donc tous ton nom, sans que rien ne les ait comparés "
             "musicalement.</div>")
    return "".join(B)


def page(d: dict) -> str:
    perdu = sum(1 for i in range(d["n"])
                if d["auto"][i] != d["apres"][-1]["lab"][i])
    B = [f"<h1>{html.escape(d['titre'])} — ce qu'un seul trait change</h1>",
         "<div class=lede>Tu as tracé <b>une</b> section et tout le découpage a "
         "changé. Voici pourquoi.<br><br>"
         "<b>L'inférence ne lit jamais les sections du chart.</b> Elle repart du "
         "mot de bi-mesures et relance l'algorithme des quatre mots sur tout le "
         "morceau, en ne figeant que les traits reçus. Ce que SongFormer avait "
         "trouvé n'est pas une entrée : il est jeté, puis remplacé par ce que "
         "l'autre algorithme sait faire.<br><br>"
         "La dernière bande est le contrôle : le trait y est posé <b>exactement "
         "là où la machine avait déjà mis sa section</b>. On ne lui apprend donc "
         "rien — et le reste change quand même. Le dégât ne vient pas de ton "
         "trait.</div>",
         f"<p class=note>{d['n']} mesures · "
         f"<b>{perdu} mesures sur {d['n']}</b> changent d'étiquette entre la "
         "machine seule et l'annotation de contrôle.</p>",
         bandes(d),
         "<h2>Mesure par mesure</h2>",
         "<p class=note>Les lignes rosées sont celles où les lectures ne "
         "s'accordent pas. Le cadre noir marque le bloc tracé à la main.</p>",
         tableau(d),
         bloc_trace(d)]
    for a in d["apres"]:
        if a["ecartes"]:
            B.append("<p class=note>⚠ traits écartés pour « "
                     + html.escape(a["titre"]) + " » : "
                     + html.escape(json.dumps(a["ecartes"], ensure_ascii=False))
                     + "</p>")
    return ("<!-- tools/annotation_degats.py -->"
            "<meta charset=utf-8>"
            "<meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>{html.escape(d['titre'])} — un trait, tout change</title>"
            "<style>" + CSS + "</style><body>" + "".join(B)
            + "<audio id=au preload=auto playsinline></audio>"
            "<script>" + JS + "</script>")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chart", default="min_B6AHb9W_LkM")
    ap.add_argument("--titre", default=None,
                    help="remplace le titre du chart, souvent "
                         "l'identifiant YouTube brut")
    ap.add_argument("--trait", action="append", default=None,
                    help="LABEL:debut-fin, en mesures 1-indexées ; répétable")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    traits = []
    for t in (a.trait or ["A:15-22", "A:7-14"]):
        lab, plage = t.split(":")
        b0, b1 = plage.split("-")
        traits.append((lab, int(b0), int(b1)))
    d = collecte(a.chart, traits)
    if a.titre:
        d['titre'] = a.titre
    out = a.out or (SETTINGS.reports_dir / f"annotation_degats_{a.chart}.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page(d), encoding="utf-8")
    perdu = sum(1 for i in range(d["n"]) if d["auto"][i] != d["apres"][-1]["lab"][i])
    print(f"→ {out}\n   {d['n']} mesures · {perdu} changent d'étiquette "
          "entre la machine seule et le contrôle")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
