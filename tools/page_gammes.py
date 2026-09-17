"""La couleur dit la GAMME — celle qui règne, ou celle vers laquelle on sort.

Louis, 2026-09-17, en deux temps. D'abord « j'aimerais bien voir ce que ça
donne si à chaque couleur on associait une gamme harmonique ». Puis, après une
première version qui donnait à chaque accord SA gamme :

    « je veux une couleur par gamme, mais pour les accords ils sont par défaut
    de la couleur de la gamme dans laquelle cette partie de la chanson est,
    sauf si l'accord lui même est en dehors de cette gamme auquel cas il est
    de la couleur de la gamme dans laquelle il projette via les notes qui
    sortent de la gamme, donc comme l'outil local keys dans analyse »

CE QUE ÇA CHANGE PAR RAPPORT À LA PREMIÈRE VERSION. Donner à chaque accord sa
propre gamme faisait sept couleurs sur huit accords : joli, et muet. Ici une
RÉGION est d'une seule couleur, et seul l'accord qui sort du jeu de notes
change de teinte — celle de la gamme vers laquelle il pointe. La page ne
montre plus des étiquettes, elle montre les sorties.

CE QU'EST « UNE GAMME » ICI, et c'est la décision qui porte tout le reste :
SEPT NOTES. Pas une tonique, pas un mode — un jeu de notes. La règle de Louis
travaille sur « les notes qui sortent de la gamme », donc l'objet coloré doit
être ce jeu-là. Conséquence assumée : si♭ majeur et sol mineur sont LA MÊME
gamme et portent la même couleur. C'est exact — ce sont les mêmes sept notes —
et c'est ce qui permet à un morceau en mineur de ne pas clignoter à chaque
emprunt à sa relative.

LA TEINTE EST LA POSITION DE LA GAMME SUR LE CERCLE DES QUINTES, via le
`rootHue` de l'app : la roue des couleurs est encore la roue des quintes,
comme dans le compas. Deux gammes voisines d'une quinte sont donc voisines de
30° — et l'écart de teinte entre un accord et sa région EST sa distance
harmonique, en quintes, lisible sans rien compter.

LA PROJECTION. Un accord qui sort prend la gamme LA PLUS PROCHE (en quintes)
qui le contient en entier. Si aucune ne le contient — un diminué, un altéré —
on prend celle qui en contient le plus, départage par la proximité. Sur Autumn
Leaves en sol mineur : 22 accords sur 26 sont dedans, le `D7` sort par son
fa♯ et projette 3 quintes plus loin (la gamme de sol), le `Ab^7` sort par son
la♭ et projette d'une quinte (mi♭), le `A-7` sort par son mi et projette d'une
quinte (fa).

CE QUE ÇA NE RÉSOUT PAS. (1) Les douze gammes sont les douze jeux
DIATONIQUES ; la mineure harmonique n'en est pas un, donc le `D7` de sol
mineur — qui est le V le plus ordinaire du monde — est lu comme une sortie
vers sol, et pas comme la sensible de sa propre tonalité. C'est exact quant
aux notes, discutable quant à la fonction, et c'est le premier arbitrage à
rendre. (2) La région vient de `keySegments`, donc d'un seul segment sur un
morceau qui ne module pas : la démonstration « une région = une couleur » n'est
pas mise à l'épreuve ici. (3) Les tensions écrites (♭9, ♯11) comptent comme
des notes de l'accord, ce qui est voulu, mais un `13` sans 11e écrite ne dit
rien de sa 11e.

    .venv/bin/python -m tools.page_gammes [clé_du_chart]
"""
from __future__ import annotations

import html
import json
import sys

from harmonia.harmonic_key import template
from harmonia.settings import SETTINGS

DEFAUT = "min_autumn_leaves_easy_jazz_piano_piano_cover_sheets"
OUT = SETTINGS.repo / "docs" / "plots" / "gammes.html"
NOMS_B = "C Db D Eb E F Gb G Ab A Bb B".split()
NOMS_D = "C C# D D# E F F# G G# A A# B".split()
#: mêmes majeurs bémolisés que `kit.js::FLAT_MAJ`.
FLAT_MAJ = {0, 1, 3, 5, 6, 8, 10}
#: les degrés d'un jeu diatonique depuis sa tonique majeure.
DIATONIQUE = (0, 2, 4, 5, 7, 9, 11)


def collection(maj: int) -> frozenset[int]:
    """Les sept notes de la gamme dont la tonique MAJEURE est `maj`."""
    return frozenset((maj + i) % 12 for i in DIATONIQUE)


def quintes(pc: int) -> int:
    """La place d'une note sur le cercle des quintes — `kit.js::fifthsIndex`."""
    return (pc * 7) % 12


def ecart(a: int, b: int) -> int:
    """Combien de quintes séparent deux gammes, dans le sens le plus court."""
    d = (quintes(a) - quintes(b)) % 12
    return min(d, 12 - d)


def projette(pcs: frozenset[int], regne: int) -> tuple[int, bool]:
    """Vers quelle gamme cet accord pointe-t-il ? -> (tonique majeure, entier ?)

    La plus proche EN QUINTES qui le contient tout entier. Si aucune ne le
    contient (diminué, altéré), celle qui en contient le plus — départagée par
    la proximité, pour ne pas envoyer un accord à l'autre bout du cercle quand
    deux gammes le servent aussi mal.
    """
    plein = [k for k in range(12) if pcs <= collection(k)]
    if plein:
        return min(plein, key=lambda k: (ecart(k, regne), k)), True
    return max(range(12), key=lambda k: (len(pcs & collection(k)), -ecart(k, regne))), False


def collecte(cle: str, n_mesures: int = 24) -> dict:
    m = json.loads((SETTINGS.charts_dir / f"{cle}.json").read_text(encoding="utf-8"))
    segs = m.get("keySegments") or [{"t0": 0.0, "t1": 1e9,
                                     "tonic": m["key"]["tonic"], "mode": m["key"]["mode"]}]
    maj0 = (m["key"]["tonic"] + 3) % 12 if m["key"]["mode"] == "minor" else m["key"]["tonic"]
    noms = NOMS_B if maj0 % 12 in FLAT_MAJ else NOMS_D

    def regne(t: float) -> tuple[int, str]:
        for s in segs:
            if s["t0"] <= t < s["t1"]:
                return s["tonic"], s["mode"]
        return segs[-1]["tonic"], segs[-1]["mode"]

    mesures: list[dict] = []
    vues: set[int] = set()
    for sec in m["sections"]:
        b0 = sec["barRanges"][0][0]
        for bi, bar in enumerate(sec["bars"]):
            k = b0 + bi
            if k >= n_mesures:
                break
            accs = []
            for c in bar:
                if c.get("nc"):
                    continue
                q = c.get("q") or ""
                ivs = template(q)
                root = int(c["root"]) % 12
                t0 = float(c["t0"])
                tonic, mode = regne(t0)
                rmaj = (tonic + 3) % 12 if mode == "minor" else tonic % 12
                if ivs is None:                       # queue inconnue : la triade
                    ivs = (0, 4, 7) if not q.startswith("-") else (0, 3, 7)
                pcs = frozenset((root + i) % 12 for i in ivs)
                dedans = pcs <= collection(rmaj)
                cible, entier = (rmaj, True) if dedans else projette(pcs, rmaj)
                sortantes = sorted(pcs - collection(rmaj))
                vues.add(cible)
                vues.add(rmaj)
                accs.append({
                    "root": root, "q": q, "ecrit": noms[root] + q,
                    "t0": round(t0, 3), "t1": round(float(c["t1"]), 3),
                    "notes": sorted(pcs), "regne": rmaj, "cible": cible,
                    "dedans": dedans, "entier": entier, "sortantes": sortantes,
                    "ecart": ecart(cible, rmaj),
                    "cle": f"{noms[tonic % 12]} {'mineur' if mode == 'minor' else 'majeur'}",
                })
            mesures.append({"n": k + 1, "sec": sec["label"], "accords": accs})
        if len(mesures) >= n_mesures:
            break
    mesures.sort(key=lambda b: b["n"])
    stem = (m.get("audio_url") or "").rsplit("/", 1)[-1] or f"{cle}.m4a"
    return {"titre": m.get("title") or cle, "tonalite": m.get("keyName"),
            "audio": "../audio/" + stem, "noms": noms, "mesures": mesures,
            "gammes": sorted(vues, key=quintes),
            "collections": {str(k): sorted(collection(k)) for k in range(12)}}


def page(D: dict) -> str:
    e = html.escape
    N = D["noms"]
    grille = "".join(
        f"""<div class="mes" data-i="{i}"><span class="num">{b['n']}</span>"""
        + "".join(
            f"""<button class="ac" type="button" data-i="{i}" data-j="{j}"
                 ><span class="sq">{e(a['ecrit'])}</span></button>"""
            for j, a in enumerate(b["accords"]))
        + "</div>"
        for i, b in enumerate(D["mesures"]))

    return f"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Les Gammes En Couleur</title>
<style>
/* la palette de l'app (`harmonia/static/ui/kit.js`), valeur pour valeur */
 :root{{--bg:#e7e0d0;--paper:#f7f3e9;--card:#fffdf6;--ink:#1c1c1c;--rule:#b9b09a;
  --faint:#8a8371;--line:#e5dcc6;--deep:#2a2622;
  --accent:#8a2b2b;--green:#1f8a5b;--amber:#c58a2e;--blue:#2a6fb0;}}
 @media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{
  --bg:#151210;--paper:#211c17;--card:#2a241d;--ink:#f2ebde;--rule:#5b5344;
  --faint:#a99f8c;--line:#39332a;--deep:#0e0c0a;}}}}
 :root[data-theme="dark"]{{--bg:#151210;--paper:#211c17;--card:#2a241d;--ink:#f2ebde;
  --rule:#5b5344;--faint:#a99f8c;--line:#39332a;--deep:#0e0c0a;}}
 *{{box-sizing:border-box}}
 body{{background:var(--bg);color:var(--ink);margin:0 auto;max-width:560px;
  font:400 15px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;
  padding:16px 14px 40px}}
 h1{{font:600 22px/1.2 Georgia,'Times New Roman',serif;margin:0;letter-spacing:-.01em}}
 .kick{{font:italic 13px/1.5 Georgia,'Times New Roman',serif;color:var(--faint);margin:3px 0 0}}
 .eyebrow{{font:600 10.5px/1 -apple-system,system-ui,sans-serif;letter-spacing:.1em;
  text-transform:uppercase;color:var(--faint);margin-bottom:7px}}
 button{{font:600 13px/1 -apple-system,system-ui,sans-serif;border-radius:9px;
  padding:9px 12px;min-height:40px;cursor:pointer;border:1px solid var(--rule);
  background:var(--paper);color:var(--ink);-webkit-tap-highlight-color:transparent}}
 button:active{{transform:translateY(1px)}}
 button:focus-visible{{outline:2px solid var(--accent);outline-offset:2px}}

 .feuille{{background:var(--card);border-radius:22px;padding:14px 14px 16px;
  margin:10px 0 4px;box-shadow:0 8px 28px -18px rgba(50,35,20,.55)}}
 .bascule{{display:inline-flex;background:var(--line);border-radius:10px;
  padding:3px;gap:2px;margin-bottom:12px}}
 .bascule button{{border:none;border-radius:8px;padding:7px 12px;min-height:36px;
  background:transparent;color:var(--faint);font-weight:600;font-size:12.5px}}
 .bascule button.on{{background:var(--card);color:var(--ink);
  box-shadow:0 1px 2px rgba(0,0,0,.12)}}

 .grille{{display:grid;grid-template-columns:repeat(4,1fr);gap:5px}}
 .mes{{position:relative;min-height:62px;border:1px solid var(--line);
  border-radius:10px;padding:14px 4px 5px;display:flex;flex-wrap:wrap;gap:3px;
  align-content:flex-start;background:var(--paper)}}
 .mes.ici{{border-color:var(--accent);box-shadow:0 0 0 2px rgba(138,43,43,.18)}}
 .num{{position:absolute;top:3px;left:6px;font:600 9px/1 -apple-system,system-ui,sans-serif;
  color:var(--faint)}}
 .ac{{flex:1 1 auto;min-width:0;min-height:30px;padding:5px 4px;
  border:1.5px solid rgba(0,0,0,.14);border-radius:7px;
  display:flex;align-items:center;justify-content:center;color:#1c1c1c}}
 /* un accord qui SORT de la gamme de sa région : le trait pointillé le dit
    même pour qui ne lit pas la nuance de teinte, et le dit en noir et blanc. */
 .ac.sort{{border-style:dashed;border-width:2px;border-color:rgba(0,0,0,.45)}}
 .ac.sel{{outline:2px solid var(--accent);outline-offset:1px}}
 .ac .sq{{font:italic 600 13px/1 Georgia,serif;white-space:nowrap}}

 .fiche{{margin-top:12px;border-top:1px solid var(--line);padding-top:10px;min-height:122px}}
 .fiche .tete{{display:flex;align-items:baseline;gap:9px;flex-wrap:wrap}}
 .fiche .nom{{font:italic 600 19px/1 Georgia,serif}}
 .fiche .g{{font:600 12.5px/1 -apple-system,system-ui,sans-serif;padding:4px 9px;
  border-radius:20px;color:#1c1c1c}}
 .fiche .pq{{font:italic 12.5px/1.5 Georgia,serif;color:var(--faint);margin:6px 0 9px}}
 .fiche .pq b{{color:var(--ink);font-style:normal;font-weight:600}}
 .clavier{{display:grid;grid-template-columns:repeat(12,1fr);gap:2px}}
 .clavier div{{height:40px;border-radius:5px;background:var(--line);
  display:flex;flex-direction:column;align-items:center;justify-content:flex-end;
  padding-bottom:3px;gap:2px;
  font:600 8.5px/1 -apple-system,system-ui,sans-serif;color:var(--faint)}}
 .clavier div.gamme{{color:#1c1c1c}}
 .clavier div b{{font-size:9px;line-height:1}}
 .clavier div.horsgamme{{outline:2px solid var(--accent);outline-offset:-2px}}

 .roue{{display:flex;justify-content:center;padding:2px 0 0}}
 .lgs{{display:grid;grid-template-columns:1fr;gap:4px;margin-top:8px}}
 .lg{{display:flex;align-items:center;gap:8px;padding:5px 7px;border-radius:8px;
  font:500 12px/1.3 -apple-system,system-ui,sans-serif;color:var(--faint)}}
 .lg.vif{{background:var(--paper)}}
 .lg i{{width:16px;height:16px;border-radius:5px;flex:0 0 auto;
  border:1px solid rgba(0,0,0,.15)}}
 .lg b{{color:var(--ink);font-weight:600}}
 .lg span{{margin-left:auto;font-size:10.5px;opacity:.8;text-align:right}}

 .barre{{display:flex;gap:10px;align-items:center;margin-top:10px}}
 .barre .av{{font:500 11.5px/1 -apple-system,system-ui,sans-serif;color:var(--faint)}}
 .barre .sp{{flex:1}}
 .note{{background:var(--card);border:1px solid var(--line);border-radius:12px;
  padding:12px 14px;margin:16px 0}}
 .note p{{margin:7px 0;font-size:12.5px;color:var(--faint)}}
 .note b{{color:var(--ink)}}
 code{{font:500 11.5px/1 ui-monospace,Menlo,monospace;background:var(--bg);
  border-radius:4px;padding:1px 4px}}
 footer{{margin-top:20px;padding-top:12px;border-top:1px solid var(--line);
  font:500 11px/1.8 -apple-system,system-ui,sans-serif;color:var(--faint)}}
</style>

<div class="eyebrow">{e(D['titre'])} &middot; {e(D['tonalite'] or '')}</div>
<h1>Les gammes en couleur</h1>
<p class="kick">une région, une couleur — et les accords qui en sortent prennent celle de leur destination</p>

<div class="feuille">
  <div class="bascule" id="bascule">
    <button type="button" data-m="gamme" class="on">couleur&nbsp;= la gamme</button>
    <button type="button" data-m="note">couleur&nbsp;= la note</button>
  </div>
  <div class="grille" id="grille">{grille}</div>
  <div class="fiche" id="fiche"></div>
  <div class="barre">
    <span class="av" id="av">touche un accord</span>
    <span class="sp"></span>
    <button id="play" type="button">&#9658; écouter</button>
  </div>
</div>

<div class="feuille">
  <div class="eyebrow">les gammes du morceau, sur le cercle des quintes</div>
  <div class="roue"><svg id="roue" width="300" height="300" viewBox="0 0 300 300"></svg></div>
  <div class="lgs" id="lgs"></div>
</div>

<div class="note">
  <p><b>Une gamme, ici, c'est sept notes.</b> Pas une tonique, pas un
  mode&nbsp;: un jeu de notes. La règle travaille sur «&nbsp;les notes qui
  sortent&nbsp;», donc l'objet coloré doit être ce jeu-là. Conséquence assumée,
  et il faut la dire&nbsp;: si♭ majeur et sol mineur sont LA MÊME gamme et
  portent la même couleur. C'est exact — ce sont les mêmes sept notes — et
  c'est ce qui permet à un morceau en mineur de ne pas clignoter à chaque
  emprunt à sa relative.</p>
  <p><b>La teinte est la place de la gamme sur le cercle des quintes</b>, par
  le <code>rootHue</code> de l'app&nbsp;: la roue des couleurs reste la roue
  des quintes, comme dans le compas. Deux gammes voisines d'une quinte sont
  voisines de 30°, donc l'écart de teinte entre un accord et sa région EST sa
  distance harmonique, sans rien compter.</p>
  <p><b>Un accord qui sort prend la gamme la plus proche qui le contient en
  entier.</b> Si aucune ne le contient — un diminué, un altéré — celle qui en
  contient le plus, départagée par la proximité. Le pointillé le redit en noir
  et blanc&nbsp;: pas besoin de lire la nuance pour voir qu'il sort.</p>
  <p><b>Le premier arbitrage à rendre.</b> Les douze gammes sont les douze
  jeux DIATONIQUES, et la mineure harmonique n'en est pas un. Donc le
  <code>D7</code> de sol mineur — le V le plus ordinaire du monde — est lu
  comme une sortie de 3 quintes vers la gamme de sol, et pas comme la sensible
  de sa propre tonalité. C'est exact quant aux notes, discutable quant à la
  fonction. Si tu préfères l'autre lecture, il faut ajouter les mineures
  harmoniques aux gammes et la cadence redevient muette&nbsp;— on ne peut pas
  avoir les deux.</p>
  <p><b>Ce que la démo ne met pas à l'épreuve.</b> Ce morceau ne module pas
  (<code>keySegments</code> n'a qu'un segment), donc «&nbsp;une région, une
  couleur&nbsp;» n'est pas encore éprouvé ici&nbsp;— il faudra un morceau qui
  change de tonalité.</p>
</div>

<footer>
  gammes&nbsp;: les douze jeux diatoniques &middot; région&nbsp;:
  <code>keySegments</code> (<code>harmonic_key</code>) &middot; teinte&nbsp;:
  <code>kit.js::rootHue</code> &middot;
  <a href="/plots/compas_da.html" style="color:var(--accent)">le compas</a>
</footer>

<script>
const D = {json.dumps(D, ensure_ascii=False, separators=(',', ':'))};
const N = D.noms, COLL = D.collections;
const mod = (n, m) => ((n % m) + m) % m;
const quintes = pc => mod(pc * 7, 12);
// LA TEINTE D'UNE GAMME = sa place sur le cercle des quintes. La saturation et
// la clarté sont figées : rien d'autre que l'identité n'entre dans la couleur
// (même règle que le compas depuis le 2026-09-17).
const gammeFill = maj => `hsl(${{Math.round(quintes(maj) / 12 * 360)}} 55% 72%)`;
const gammeEdge = maj => `hsl(${{Math.round(quintes(maj) / 12 * 360)}} 59% 52%)`;
// la couleur-note d'aujourd'hui, pour la comparaison
const noteFill = pc => `hsl(${{Math.round(quintes(pc) / 12 * 360)}} 55% 72%)`;
// une gamme se nomme par sa majeure ET sa relative mineure : c'est le même
// jeu de notes, et le dire évite de croire qu'on a changé de monde.
const nomGamme = maj => N[maj] + ' / ' + N[mod(maj + 9, 12)] + '-';

let mode = 'gamme', sel = null, audio = null, stopAt = null;
const accordDe = (i, j) => D.mesures[i].accords[j];

function peins(){{
  document.querySelectorAll('.ac').forEach(b => {{
    const a = accordDe(+b.dataset.i, +b.dataset.j);
    b.style.background = mode === 'gamme' ? gammeFill(a.cible) : noteFill(a.root);
    // l'encre posée sur une pastille claire ne suit pas le thème : ces fonds
    // sont clairs par construction dans les deux (même règle que les pétales
    // du compas, cf. known_issues 2026-09-17).
    b.style.color = '#1c1c1c';
    b.classList.toggle('sort', mode === 'gamme' && !a.dedans);
  }});
  document.querySelectorAll('.lg').forEach(l => {{
    l.classList.toggle('vif', !!(sel && +l.dataset.g === sel.cible));
  }});
}}

function fiche(a){{
  const f = document.getElementById('fiche');
  f.innerHTML = '';
  if (!a) {{
    const d = document.createElement('div'); d.className = 'pq';
    d.textContent = "touche un accord pour voir sa gamme, et ce qui en sort";
    f.appendChild(d); return;
  }}
  const tete = document.createElement('div'); tete.className = 'tete';
  const nom = document.createElement('span'); nom.className = 'nom'; nom.textContent = a.ecrit;
  const g = document.createElement('span'); g.className = 'g';
  g.style.background = gammeFill(a.cible);
  g.textContent = nomGamme(a.cible);
  tete.appendChild(nom); tete.appendChild(g);
  f.appendChild(tete);

  const pq = document.createElement('div'); pq.className = 'pq';
  if (a.dedans) {{
    pq.innerHTML = 'toutes ses notes sont dans la gamme qui règne ici (<b>'
      + nomGamme(a.regne) + '</b>) — il en prend la couleur';
  }} else {{
    const s = a.sortantes.map(p => N[p]).join(', ');
    pq.innerHTML = '<b>' + s + '</b> sort de ' + nomGamme(a.regne)
      + ' — il projette vers <b>' + nomGamme(a.cible) + '</b>, à '
      + a.ecart + (a.ecart > 1 ? ' quintes' : ' quinte') + ' d\\'ici'
      + (a.entier ? '' : ' (aucune gamme ne le contient en entier : la plus proche qui en prend le plus)');
  }}
  f.appendChild(pq);

  // LES DOUZE DEMI-TONS. Allumés : les notes de la gamme visée. Cerclées
  // d'accent : les notes de l'accord qui sortent de la gamme de la RÉGION —
  // c'est-à-dire exactement ce qui a décidé de la couleur.
  const cl = document.createElement('div'); cl.className = 'clavier';
  const dedans = new Set(COLL[String(a.cible)]);
  const notes = new Set(a.notes), sortantes = new Set(a.sortantes);
  for (let i = 0; i < 12; i++) {{
    const pc = mod(a.regne + i, 12);
    const d = document.createElement('div');
    if (dedans.has(pc)) {{ d.className = 'gamme'; d.style.background = gammeFill(a.cible); }}
    if (sortantes.has(pc)) d.classList.add('horsgamme');
    if (notes.has(pc)) {{ const b = document.createElement('b'); b.textContent = '●'; d.appendChild(b); }}
    const t = document.createElement('span'); t.textContent = N[pc]; d.appendChild(t);
    cl.appendChild(d);
  }}
  f.appendChild(cl);
}}

// ── LA ROUE DES GAMMES : les douze jeux diatoniques à leur place de quintes,
// ceux du morceau allumés, la région cerclée d'accent. C'est la légende ET la
// carte : on y voit d'un coup à quelle distance chaque sortie emmène.
function roue(){{
  const svg = document.getElementById('roue');
  svg.innerHTML = '';
  const NS = 'http://www.w3.org/2000/svg';
  const mk = (t, a) => {{ const n = document.createElementNS(NS, t);
    for (const k in a) n.setAttribute(k, a[k]); return n; }};
  const cx = 150, cy = 150, R = 108;
  const par = {{}};
  D.mesures.forEach(m => m.accords.forEach(a => {{
    par[a.cible] = (par[a.cible] || 0) + 1; }}));
  const regne = D.mesures.length && D.mesures[0].accords.length
    ? D.mesures[0].accords[0].regne : 0;
  svg.appendChild(mk('circle', {{cx, cy, r: R, fill: 'none',
    stroke: getComputedStyle(document.documentElement).getPropertyValue('--line').trim(),
    'stroke-width': 1.5}}));
  for (let i = 0; i < 12; i++) {{
    const maj = mod(i * 7, 12);                    // i-ème quinte depuis do
    const ang = (-90 + i * 30) * Math.PI / 180;
    const x = cx + R * Math.cos(ang), y = cy + R * Math.sin(ang);
    const n = par[maj] || 0;
    const r = n ? 15 + Math.min(11, n) : 7;
    svg.appendChild(mk('circle', {{cx: x, cy: y, r,
      fill: n ? gammeFill(maj) : 'transparent',
      stroke: n ? gammeEdge(maj) : 'var(--rule)',
      'stroke-width': maj === regne ? 3 : 1.25,
      'stroke-opacity': n ? 1 : 0.45}}));
    const t = mk('text', {{x, y, 'text-anchor': 'middle', 'dominant-baseline': 'central',
      'font-family': 'Georgia,serif', 'font-style': 'italic', 'font-weight': 600,
      'font-size': n ? 13 : 10,
      fill: n ? '#1c1c1c' : 'var(--faint)'}});
    t.textContent = N[maj];
    svg.appendChild(t);
    if (n) {{
      const c = mk('text', {{x, y: y + r + 9, 'text-anchor': 'middle',
        'font-family': '-apple-system,system-ui,sans-serif', 'font-size': 9,
        'font-weight': 600, fill: 'var(--faint)'}});
      c.textContent = n + (maj === regne ? ' · la région' : '');
      svg.appendChild(c);
    }}
  }}
  // la légende en liste, pour les gammes réellement vues
  const lgs = document.getElementById('lgs');
  lgs.innerHTML = '';
  Object.keys(par).map(Number).sort((a, b) => par[b] - par[a]).forEach(maj => {{
    const d = document.createElement('div'); d.className = 'lg'; d.dataset.g = maj;
    const i = document.createElement('i'); i.style.background = gammeFill(maj);
    const b = document.createElement('b'); b.textContent = nomGamme(maj);
    const s = document.createElement('span');
    s.textContent = par[maj] + (par[maj] > 1 ? ' accords' : ' accord')
      + (maj === regne ? ' · la région'
         : ' · ' + ecartDe(maj, regne) + (ecartDe(maj, regne) > 1 ? ' quintes' : ' quinte'));
    d.appendChild(i); d.appendChild(b); d.appendChild(s);
    lgs.appendChild(d);
  }});
}}
function ecartDe(a, b){{ const d = mod(quintes(a) - quintes(b), 12); return Math.min(d, 12 - d); }}

document.querySelectorAll('.ac').forEach(b => b.addEventListener('click', () => {{
  document.querySelectorAll('.ac').forEach(x => x.classList.remove('sel'));
  b.classList.add('sel');
  sel = accordDe(+b.dataset.i, +b.dataset.j);
  fiche(sel); peins();
  document.getElementById('av').textContent = sel.ecrit + ' · ' + nomGamme(sel.cible);
  if (audio) {{ stopAt = sel.t1 + 0.15; audio.currentTime = Math.max(0, sel.t0);
    audio.play().catch(() => {{}}); }}
}}));

document.querySelectorAll('#bascule button').forEach(b => b.addEventListener('click', () => {{
  mode = b.dataset.m;
  document.querySelectorAll('#bascule button').forEach(x => x.classList.toggle('on', x === b));
  peins();
}}));

// l'audio : fetch + blob, jamais de boucle rAF — les deux pièges iOS déjà
// payés (`reference_ios_audio_traps`).
fetch(D.audio).then(r => r.blob()).then(b => {{
  audio = new Audio(URL.createObjectURL(b));
  audio.addEventListener('timeupdate', () => {{
    if (stopAt != null && audio.currentTime >= stopAt) {{ audio.pause(); stopAt = null; return; }}
    const t = audio.currentTime;
    D.mesures.forEach((m, i) => {{
      const dedans = m.accords.length && t >= m.accords[0].t0
        && t < m.accords[m.accords.length - 1].t1;
      document.querySelectorAll(`.mes[data-i="${{i}}"]`)
        .forEach(el => el.classList.toggle('ici', dedans));
    }});
  }});
}}).catch(() => {{}});

document.getElementById('play').addEventListener('click', () => {{
  if (!audio) return;
  if (!audio.paused) {{ audio.pause(); return; }}
  stopAt = null;
  if (!sel && D.mesures[0].accords.length) audio.currentTime = D.mesures[0].accords[0].t0;
  audio.play().catch(() => {{}});
}});

peins(); fiche(null); roue();
</script>
"""


def main() -> int:
    cle = sys.argv[1] if len(sys.argv) > 1 else DEFAUT
    D = collecte(cle)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page(D), encoding="utf-8")
    N = D["noms"]
    tout = [a for b in D["mesures"] for a in b["accords"]]
    dedans = [a for a in tout if a["dedans"]]
    print(f"→ {OUT}  ({len(D['mesures'])} mesures, {len(tout)} accords)")
    print(f"   dedans : {len(dedans)}/{len(tout)}")
    for a in tout:
        if not a["dedans"]:
            s = ", ".join(N[p] for p in a["sortantes"])
            print(f"   {a['ecrit']:9s} sort par {s:6s} → {N[a['cible']]} "
                  f"({a['ecart']} quinte{'s' if a['ecart'] > 1 else ''})"
                  f"{'' if a['entier'] else '  [partiel]'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
