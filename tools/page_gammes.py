"""Et si la couleur disait la GAMME plutôt que la note ?

Louis, 2026-09-17 : « j'aimerais bien voir ce que ça donne si à chaque couleur
on associait une gamme harmonique mais fais moi juste une démo pour voir si
j'aime bien ».

CE QUE ÇA CHANGE, en une phrase : aujourd'hui la couleur dit *quelle
fondamentale*, donc deux accords de la même couleur partagent une note ; si
elle dit *quelle gamme*, deux accords de la même couleur se jouent avec les
MÊMES NOTES. Pour un instrumentiste ce n'est pas la même information — la
première nomme, la seconde dit quoi jouer.

Et surtout, la couleur-gamme montre une chose que la couleur-note ne peut pas
montrer : les RÉGIONS du morceau. Sur Autumn Leaves, les six premiers accords
sont tous dans si♭ majeur (six teintes différentes en couleur-note, une seule
famille chaude en couleur-gamme), puis la cadence mineure bascule d'un coup
dans les froides et le rouge. On voit la forme, pas les lettres.

LE TABLEAU ACCORD → GAMME. C'est la table classique des chord-scales, pas une
invention : on prend le degré de la fondamentale dans la tonalité qui règne
ICI (`keySegments`, la clé locale, pas la clé du morceau) et la famille de
l'accord.

  majeur sur le I        -> ionien          majeur sur le IV   -> lydien
  dominante sur le V     -> mixolydien      dominante vers un mineur -> altéré
  mineur sur le ii       -> dorien          sur le iii         -> phrygien
  mineur sur le vi (ou i)-> éolien          demi-diminué       -> locrien
  diminué                -> demi-ton/ton    augmenté           -> par tons

CE QUE ÇA NE RÉSOUT PAS. Un accord peut appeler deux gammes également
défendables (un mineur sur le i : dorien ou éolien, l'oreille tranche) et la
table en choisit une. Les tensions écrites (♭9, ♯11) ne sont pas lues : elles
devraient pouvoir forcer la gamme. C'est une DÉMO pour décider si l'idée
mérite d'aller dans l'app, pas une règle arrêtée.

    .venv/bin/python -m tools.page_gammes [clé_du_chart]
"""
from __future__ import annotations

import html
import json
import sys

from harmonia.settings import SETTINGS

DEFAUT = "min_autumn_leaves_easy_jazz_piano_piano_cover_sheets"
OUT = SETTINGS.repo / "docs" / "plots" / "gammes.html"
NOMS_B = "C Db D Eb E F Gb G Ab A Bb B".split()
NOMS_D = "C C# D D# E F F# G G# A A# B".split()
#: mêmes majeurs bémolisés que `kit.js::FLAT_MAJ` — la page doit écrire dans
#: la tonalité du morceau, sinon elle est illisible pour lui (2026-09-17).
FLAT_MAJ = {0, 1, 3, 5, 6, 8, 10}

# ── LES GAMMES ─────────────────────────────────────────────────────────────
# Rangées par CLARTÉ MODALE : c'est l'ordre classique, du mode qui a le plus
# d'altérations ascendantes vers celui qui en a le plus de descendantes
# (lydien +1 dièse, ionien 0, mixolydien -1, dorien -2, éolien -3, phrygien
# -4, locrien -5). Cet axe est réel, il ne vient pas d'un goût : il compte les
# altérations par rapport à la majeure de même fondamentale.
#
# LA COULEUR SUIT DEUX CHOSES, et c'est tout :
#   * chaud / froid  = la tierce est majeure / mineure. La rupture entre
#     mixolydien et dorien n'est donc pas un saut arbitraire dans la roue,
#     c'est LE changement de mode de l'accord ;
#   * clair / foncé  = le rang de clarté à l'intérieur de sa famille.
# Les gammes qui ne sont pas des modes de la majeure sortent de la rampe, dans
# les rouges : « hors de la tonalité » se voit comme hors de la rampe.
GAMMES = {
    "lydien":      dict(deg=[0, 2, 4, 6, 7, 9, 11], c="hsl(48 72% 70%)",  rang=1, fam="majeur"),
    "ionien":      dict(deg=[0, 2, 4, 5, 7, 9, 11], c="hsl(34 70% 64%)",  rang=2, fam="majeur"),
    "mixolydien":  dict(deg=[0, 2, 4, 5, 7, 9, 10], c="hsl(19 68% 59%)",  rang=3, fam="majeur"),
    "dorien":      dict(deg=[0, 2, 3, 5, 7, 9, 10], c="hsl(158 44% 58%)", rang=4, fam="mineur"),
    "éolien":      dict(deg=[0, 2, 3, 5, 7, 8, 10], c="hsl(203 48% 60%)", rang=5, fam="mineur"),
    "phrygien":    dict(deg=[0, 1, 3, 5, 7, 8, 10], c="hsl(248 40% 62%)", rang=6, fam="mineur"),
    "locrien":     dict(deg=[0, 1, 3, 5, 6, 8, 10], c="hsl(280 34% 54%)", rang=7, fam="mineur"),
    "altéré":      dict(deg=[0, 1, 3, 4, 6, 8, 10], c="hsl(352 64% 58%)", rang=8, fam="hors rampe"),
    "demi-ton/ton": dict(deg=[0, 1, 3, 4, 6, 7, 9, 10], c="hsl(322 54% 60%)", rang=9, fam="hors rampe"),
    "par tons":    dict(deg=[0, 2, 4, 6, 8, 10],    c="hsl(300 32% 64%)", rang=10, fam="hors rampe"),
}


def _famille(q: str) -> str:
    """La famille d'un accord d'après sa queue, comme l'app l'écrit."""
    q = q or ""
    if q.startswith("o") or q.startswith("dim"):
        return "dim"
    if q.startswith("-7b5") or q.startswith("h"):
        return "hdim"
    if q.startswith("+"):
        return "aug"
    if q.startswith("-"):
        return "min"                                 # `-`, `-7`, `-^7`, `-9`…
    if q.startswith("^") or q in ("", "6", "69") or q.startswith("add"):
        return "maj"
    if q.startswith("sus"):
        return "maj"
    return "dom"                                     # 7, 9, 13, 7sus4, …


def gamme_de(root: int, q: str, tonic: int, mode: str) -> tuple[str, str]:
    """(accord, tonalité qui règne ici) -> (nom de gamme, pourquoi).

    ``tonic``/``mode`` viennent de la clé LOCALE. Le degré se compte dans la
    majeure : une tonalité mineure passe par sa relative majeure, parce que la
    table des chord-scales est la même des deux côtés.
    """
    maj = (tonic + 3) % 12 if mode == "minor" else tonic % 12
    d = (root - maj) % 12
    fam = _famille(q)
    rom = {0: "I", 2: "ii", 4: "iii", 5: "IV", 7: "V", 9: "vi", 11: "vii"}.get(d)
    ou = f"le {rom}" if rom else "hors gamme"
    if fam == "maj":
        if d == 0:
            return "ionien", "majeur sur le I"
        if d == 5:
            return "lydien", "majeur sur le IV"
        # UNE TRIADE MAJEURE SUR LE V EST UNE DOMINANTE, septième écrite ou
        # non : le `F` d'Autumn Leaves va sur si♭, il appelle le mixolydien et
        # pas le lydien. C'est l'erreur qu'a montrée la première exécution.
        if d == 7:
            return "mixolydien", "majeur sur le V — c'est une dominante"
        return "lydien", f"majeur sur {ou}"
    if fam == "dom":
        # une dominante qui vise le mineur prend l'altérée : c'est la quinte du
        # ton mineur, et c'est exactement le D7 d'Autumn Leaves.
        if mode == "minor" and d == (tonic + 7 - maj) % 12:
            return "altéré", "dominante de la tonique mineure"
        return "mixolydien", f"dominante sur {ou}"
    if fam == "min":
        if d == 2:
            return "dorien", "mineur sur le ii"
        if d == 4:
            return "phrygien", "mineur sur le iii"
        if d == 9:
            return "éolien", "mineur sur le vi"
        if mode == "minor" and d == (tonic - maj) % 12:
            return "éolien", "la tonique mineure"
        return "dorien", f"mineur sur {ou}"
    if fam == "hdim":
        return "locrien", f"demi-diminué sur {ou}"
    if fam == "dim":
        return "demi-ton/ton", "diminué"
    return "par tons", "augmenté"


def collecte(cle: str, n_mesures: int = 16) -> dict:
    m = json.loads((SETTINGS.charts_dir / f"{cle}.json").read_text(encoding="utf-8"))
    segs = m.get("keySegments") or [{"t0": 0.0, "t1": 1e9,
                                     "tonic": m["key"]["tonic"], "mode": m["key"]["mode"]}]
    maj = (m["key"]["tonic"] + 3) % 12 if m["key"]["mode"] == "minor" else m["key"]["tonic"]
    noms = NOMS_B if maj % 12 in FLAT_MAJ else NOMS_D

    def regne(t: float) -> tuple[int, str]:
        for s in segs:
            if s["t0"] <= t < s["t1"]:
                return s["tonic"], s["mode"]
        return segs[-1]["tonic"], segs[-1]["mode"]

    mesures: list[dict] = []
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
                t0 = float(c["t0"])
                tonic, mode = regne(t0)
                g, pourquoi = gamme_de(int(c["root"]) % 12, c.get("q") or "", tonic, mode)
                accs.append({
                    "root": int(c["root"]) % 12, "q": c.get("q") or "",
                    "ecrit": noms[int(c["root"]) % 12] + (c.get("q") or ""),
                    "t0": round(t0, 3), "t1": round(float(c["t1"]), 3),
                    "gamme": g, "pourquoi": pourquoi,
                    "cle": f"{noms[tonic % 12]} {'mineur' if mode == 'minor' else 'majeur'}",
                })
            mesures.append({"n": k + 1, "sec": sec["label"], "accords": accs})
        if len(mesures) >= n_mesures:
            break
    mesures.sort(key=lambda b: b["n"])
    stem = (m.get("audio_url") or "").rsplit("/", 1)[-1] or f"{cle}.m4a"
    return {"titre": m.get("title") or cle, "tonalite": m.get("keyName"),
            "audio": "../audio/" + stem, "noms": noms, "mesures": mesures,
            "gammes": GAMMES}


def page(D: dict) -> str:
    e = html.escape
    grille = "".join(
        f"""<div class="mes" data-i="{i}"><span class="num">{b['n']}</span>"""
        + "".join(
            f"""<button class="ac" type="button" data-i="{i}" data-j="{j}"
                 data-t0="{a['t0']}" data-t1="{a['t1']}"
                 data-pc="{a['root']}" data-g="{e(a['gamme'])}"
                 ><span class="sq">{e(a['ecrit'])}</span></button>"""
            for j, a in enumerate(b["accords"]))
        + "</div>"
        for i, b in enumerate(D["mesures"]))

    legende = "".join(
        f"""<div class="lg" data-g="{e(g)}"><i style="background:{v['c']}"></i>
            <b>{e(g)}</b><span>{e(v['fam'])}</span></div>"""
        for g, v in sorted(D["gammes"].items(), key=lambda kv: kv[1]["rang"]))

    return f"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>La Couleur Dit La Gamme</title>
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

 /* l'outil vit dans une feuille, comme le compas dans l'app */
 .feuille{{background:var(--card);border-radius:22px;padding:14px 14px 16px;
  margin:10px 0 4px;box-shadow:0 8px 28px -18px rgba(50,35,20,.55)}}
 .bascule{{display:inline-flex;background:var(--line);border-radius:10px;
  padding:3px;gap:2px;margin-bottom:12px}}
 .bascule button{{border:none;border-radius:8px;padding:7px 13px;min-height:36px;
  background:transparent;color:var(--faint);font-weight:600}}
 .bascule button.on{{background:var(--card);color:var(--ink);
  box-shadow:0 1px 2px rgba(0,0,0,.12)}}

 .grille{{display:grid;grid-template-columns:repeat(4,1fr);gap:5px}}
 .mes{{position:relative;min-height:62px;border:1px solid var(--line);
  border-radius:10px;padding:14px 4px 5px;display:flex;flex-wrap:wrap;gap:3px;
  align-content:flex-start;background:var(--paper)}}
 .mes.ici{{border-color:var(--accent);box-shadow:0 0 0 2px rgba(138,43,43,.18)}}
 .num{{position:absolute;top:3px;left:6px;font:600 9px/1 -apple-system,system-ui,sans-serif;
  color:var(--faint)}}
 .ac{{flex:1 1 auto;min-width:0;min-height:30px;padding:5px 4px;border:1.5px solid transparent;
  border-radius:7px;display:flex;align-items:center;justify-content:center}}
 .ac .sq{{font:italic 600 13px/1 Georgia,serif;white-space:nowrap}}
 .ac.sel{{outline:2px solid var(--accent);outline-offset:1px}}

 .fiche{{margin-top:12px;border-top:1px solid var(--line);padding-top:10px;min-height:96px}}
 .fiche .tete{{display:flex;align-items:baseline;gap:9px;flex-wrap:wrap}}
 .fiche .nom{{font:italic 600 19px/1 Georgia,serif}}
 .fiche .g{{font:600 12.5px/1 -apple-system,system-ui,sans-serif;padding:4px 9px;
  border-radius:20px;color:#1c1c1c}}
 .fiche .pq{{font:italic 12.5px/1.5 Georgia,serif;color:var(--faint);margin:6px 0 9px}}
 .clavier{{display:grid;grid-template-columns:repeat(12,1fr);gap:2px}}
 .clavier div{{height:38px;border-radius:5px;background:var(--line);
  display:flex;align-items:flex-end;justify-content:center;padding-bottom:3px;
  font:600 8.5px/1 -apple-system,system-ui,sans-serif;color:var(--faint)}}
 .clavier div.in{{color:#1c1c1c}}
 .clavier div.fond{{outline:2px solid var(--accent);outline-offset:-2px}}

 .lgs{{display:grid;grid-template-columns:repeat(2,1fr);gap:4px;margin-top:10px}}
 .lg{{display:flex;align-items:center;gap:7px;padding:4px 6px;border-radius:7px;
  font:500 11.5px/1 -apple-system,system-ui,sans-serif;color:var(--faint)}}
 .lg.vif{{background:var(--paper)}}
 .lg i{{width:15px;height:15px;border-radius:4px;flex:0 0 auto}}
 .lg b{{color:var(--ink);font-weight:600}}
 .lg span{{margin-left:auto;font-size:10px;opacity:.75}}

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
<h1>La couleur dit la gamme</h1>
<p class="kick">deux accords de la même couleur se jouent avec les mêmes notes</p>

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
  <div class="eyebrow">les gammes, de la plus claire à la plus sombre</div>
  <div class="lgs" id="lgs">{legende}</div>
</div>

<div class="note">
  <p><b>La rampe n'est pas un dégradé décoratif.</b> Les sept modes sont rangés
  par clarté modale&nbsp;— le compte d'altérations par rapport à la majeure de
  même fondamentale, de&nbsp;+1 (lydien) à&nbsp;−5 (locrien). C'est un axe
  réel, pas un goût.</p>
  <p><b>Chaud ou froid&nbsp;= la tierce.</b> La rupture entre le mixolydien et
  le dorien n'est donc pas un saut arbitraire dans la roue&nbsp;: c'est le
  passage du mode majeur au mode mineur. À l'intérieur de chaque famille, plus
  clair&nbsp;= plus brillant. Et les gammes qui ne sont pas des modes de la
  majeure sortent de la rampe, dans les rouges&nbsp;: «&nbsp;hors de la
  tonalité&nbsp;» se voit comme hors de la rampe.</p>
  <p><b>Ce que la couleur-note ne peut pas montrer.</b> Bascule&nbsp;: les six
  premiers accords d'Autumn Leaves sont six teintes différentes en
  couleur-note, et une seule famille chaude en couleur-gamme&nbsp;— ils sont
  tous dans si♭ majeur. Puis la cadence mineure bascule d'un coup dans les
  froides et le rouge. On voit la FORME, pas les lettres.</p>
  <p><b>Ce que ça ne résout pas.</b> Un accord peut appeler deux gammes
  également défendables (un mineur sur le i&nbsp;: dorien ou éolien, l'oreille
  tranche) et la table en choisit une. Les tensions écrites (♭9, ♯11) ne sont
  pas lues&nbsp;— elles devraient pouvoir forcer la gamme. La clé locale
  (<code>keySegments</code>) règne, pas la clé du morceau&nbsp;: sur un morceau
  qui module, c'est elle qui fait le travail.</p>
</div>

<footer>
  table des chord-scales &middot; clé locale&nbsp;: <code>harmonic_key</code>
  &middot; palette et feuille&nbsp;: <code>ui/kit.js</code> &middot;
  le compas&nbsp;: <a href="/plots/compas_da.html" style="color:var(--accent)">cascade</a>
</footer>

<script>
const D = {json.dumps(D, ensure_ascii=False, separators=(',', ':'))};
const N = D.noms, G = D.gammes;
const mod = (n, m) => ((n % m) + m) % m;
// la couleur-note de l'app : teinte au cercle des quintes, saturation et
// clarté figées (c'est la règle posée le 2026-09-17 sur le compas).
const fifths = pc => mod(pc * 7, 12);
const parNote = pc => `hsl(${{Math.round(fifths(pc) / 12 * 360)}} 55% 72%)`;
const parGamme = g => (G[g] || {{}}).c || 'var(--line)';

let mode = 'gamme', sel = null, audio = null, stopAt = null;

function accordDe(i, j){{ return D.mesures[i].accords[j]; }}

function peins(){{
  document.querySelectorAll('.ac').forEach(b => {{
    const a = accordDe(+b.dataset.i, +b.dataset.j);
    b.style.background = mode === 'gamme' ? parGamme(a.gamme) : parNote(a.root);
    // l'encre posée sur une pastille claire ne suit pas le thème : ces fonds
    // sont clairs par construction dans les deux (même règle que les pétales
    // du compas, cf. known_issues 2026-09-17).
    b.style.color = '#1c1c1c';
    b.style.borderColor = 'rgba(0,0,0,.14)';
  }});
  document.querySelectorAll('.lg').forEach(l => {{
    const vif = mode === 'gamme' && sel && sel.gamme === l.dataset.g;
    l.classList.toggle('vif', !!vif);
  }});
}}

function fiche(a){{
  const f = document.getElementById('fiche');
  f.innerHTML = '';
  if (!a) {{ f.innerHTML = '<div class="pq" style="font:italic 12.5px Georgia,serif;color:var(--faint)">touche un accord pour voir sa gamme</div>'; return; }}
  const tete = document.createElement('div'); tete.className = 'tete';
  const nom = document.createElement('span'); nom.className = 'nom'; nom.textContent = a.ecrit;
  const g = document.createElement('span'); g.className = 'g';
  g.style.background = parGamme(a.gamme);
  g.textContent = N[mod(a.root, 12)] + ' ' + a.gamme;
  tete.appendChild(nom); tete.appendChild(g);
  f.appendChild(tete);
  const pq = document.createElement('div'); pq.className = 'pq';
  pq.textContent = a.pourquoi + ' — la tonalité qui règne ici est ' + a.cle;
  f.appendChild(pq);
  // LES DOUZE DEMI-TONS, ceux de la gamme allumés dans sa couleur : c'est la
  // phrase « mêmes notes » rendue vérifiable d'un coup d'œil.
  const cl = document.createElement('div'); cl.className = 'clavier';
  const deg = new Set((G[a.gamme] || {{deg: []}}).deg);
  for (let i = 0; i < 12; i++) {{
    const pc = mod(a.root + i, 12);
    const d = document.createElement('div');
    if (deg.has(i)) {{ d.className = 'in'; d.style.background = parGamme(a.gamme); }}
    if (i === 0) d.classList.add('fond');
    d.textContent = N[pc];
    cl.appendChild(d);
  }}
  f.appendChild(cl);
}}

document.querySelectorAll('.ac').forEach(b => b.addEventListener('click', () => {{
  document.querySelectorAll('.ac').forEach(x => x.classList.remove('sel'));
  b.classList.add('sel');
  sel = accordDe(+b.dataset.i, +b.dataset.j);
  fiche(sel); peins();
  document.getElementById('av').textContent = sel.ecrit + ' · ' + sel.gamme;
  if (audio) {{ stopAt = sel.t1 + 0.15; audio.currentTime = Math.max(0, sel.t0);
    audio.play().catch(() => {{}}); }}
}}));

document.querySelectorAll('#bascule button').forEach(b => b.addEventListener('click', () => {{
  mode = b.dataset.m;
  document.querySelectorAll('#bascule button').forEach(x => x.classList.toggle('on', x === b));
  document.getElementById('lgs').parentElement.style.opacity = mode === 'gamme' ? '1' : '.45';
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
  if (!sel) audio.currentTime = D.mesures[0].accords.length ? D.mesures[0].accords[0].t0 : 0;
  audio.play().catch(() => {{}});
}});

peins(); fiche(null);
</script>
"""


def main() -> int:
    cle = sys.argv[1] if len(sys.argv) > 1 else DEFAUT
    D = collecte(cle)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page(D), encoding="utf-8")
    n = sum(len(b["accords"]) for b in D["mesures"])
    vues = {}
    for b in D["mesures"]:
        for a in b["accords"]:
            vues[a["gamme"]] = vues.get(a["gamme"], 0) + 1
    print(f"→ {OUT}  ({len(D['mesures'])} mesures, {n} accords)")
    for g, k in sorted(vues.items(), key=lambda kv: -kv[1]):
        print(f"   {g:14s} {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
