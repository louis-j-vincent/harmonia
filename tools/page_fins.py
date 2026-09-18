"""« 1re/2e fin » ou « la fin habituelle et son exception » ? — la page qui tranche.

Louis, 2026-09-18, en regardant This Love : « 3 des 4 fins sont les mêmes, ça
sert à rien de les écrire 3 fois ». Sa proposition : noter la fin classique
`B` et l'autre `B alt`, et marquer pareil sur la bande de forme en tête de
grille — ce qui distingue ce cas de celui où le morceau joue systématiquement
deux passes dont les fins diffèrent, qui est ce que `1.` / `2.` veut dire.

CE QUE LA MESURE DIT (les 22 charts de la bibliothèque qui portent des fins) :

    une fin = une passe, dans l'ordre   14 / 21 sections   `1.` `2.` est JUSTE
    une fin couvre plusieurs passes      6 / 21 sections   `1.` `2.` MENT
    mélange des deux (The Lazy Song)     1 / 21 sections   ni l'un ni l'autre

Le critère n'est donc pas « combien de fins » mais « est-ce qu'une fin couvre
plusieurs passes ». Dès qu'elle en couvre plusieurs, il n'y a plus de
« deuxième fois » et le numéro raconte une histoire fausse.

CE QUE CETTE PAGE NE FAIT PAS : décider. Elle pose les trois morceaux dans les
deux écritures, côte à côte, chaque fin écoutable à sa vraie place dans
l'audio, pour que Louis arbitre LA question ouverte — « classique » veut-il
dire la fin la plus fréquente, ou la première jouée ? Les deux règles ne sont
en désaccord que sur un des trois morceaux, et c'est le sien.

    python -m tools.page_fins          # → docs/plots/fins_alt_3morceaux.html
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from harmonia.settings import SETTINGS

CHARTS = SETTINGS.repo / "state" / "cache" / "charts"
SORTIE = SETTINGS.repo / "docs" / "plots" / "fins_alt_3morceaux.html"

FLAT = ["C", "D♭", "D", "E♭", "E", "F", "G♭", "G", "A♭", "A", "B♭", "B"]
SHARP = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"]
FLAT_MAJ = {0, 1, 3, 5, 6, 8, 10}

# Les mêmes queues que `ui/kit.js` (notation normale), pour que la page et
# l'app écrivent le même accord — une page d'arbitrage qui orthographie
# autrement que le chart ne vaut rien.
TOK = {"": "", "6": "6", "^7": "maj7", "^": "maj7", "7": "7", "-": "m", "-7": "m7",
       "-^7": "mMaj7", "-6": "m6", "o": "dim", "o7": "dim7", "-7b5": "m7♭5",
       "h7": "ø7", "h": "ø", "9": "9", "-9": "m9", "^9": "maj9", "13": "13",
       "7b9": "7♭9", "7#9": "7♯9", "7b5": "7♭5", "7#11": "7♯11", "sus": "sus",
       "7sus": "7sus", "sus4": "sus4", "7sus4": "7sus4", "+": "+", "+7": "+7",
       "69": "6/9"}

MORCEAUX = [
    ("min_maroon_5_this_love", "C'est ton cas : les deux règles ne sont PAS d'accord ici."),
    ("min_maroon_5_she_will_be_loved_official_music_video",
     "Les deux règles tombent d'accord — et l'exception est INTERCALÉE."),
    ("old_ben_e_king_stand_by_me_audio",
     "Les deux règles tombent d'accord — mais l'« exception » est un N.C."),
]


# ── lecture du modèle ────────────────────────────────────────────────────
def charge(stem: str) -> dict:
    p = CHARTS / f"{stem}.json"
    if not p.exists():
        raise SystemExit(f"chart introuvable : {p}")
    return json.loads(p.read_text())


def notes_du_ton(cle: dict) -> list[str]:
    tonic = int((cle or {}).get("tonic") or 0)
    maj = (tonic + 3) % 12 if (cle or {}).get("mode") == "minor" else tonic % 12
    return FLAT if maj in FLAT_MAJ else SHARP


def accord(c: dict, notes: list[str]) -> str:
    if c.get("nc"):
        return "N.C."
    q = c.get("q") or ""
    s = notes[c["root"] % 12] + TOK.get(q, q)
    b = c.get("bass")
    if b is not None and b >= 0 and b % 12 != c["root"] % 12:
        s += "/" + notes[b % 12]
    return s


def mesure(bar, notes) -> str:
    return " ".join(accord(c, notes) for c in bar) or "%"


def cle_racine(bars) -> tuple:
    """Deux fins sont « la même » quand elles posent les mêmes FONDAMENTALES,
    mesure par mesure. La qualité (`F-` / `F7` / `F`) et un N.C. terminal sont
    du bruit de modèle : c'est exactement ce qui faisait écrire trois fois la
    même fin sur This Love. Mesuré : sur les 48 fins du corpus, cette fusion
    n'en retire que 2 — et les deux sont sur ce morceau. Ce n'est donc pas un
    levier général, c'est ce qui rend CE chart lisible."""
    return tuple(tuple(c["root"] % 12 for c in bar if not c.get("nc")) for bar in bars)


def groupes(sec) -> list[dict]:
    """Les fins écrites, fusionnées sur la fondamentale. Chaque groupe garde
    la première variante comme écriture de référence et l'union des passes."""
    out: list[dict] = []
    for v in sec["endings"]["variants"]:
        k = cle_racine(v["bars"])
        g = next((x for x in out if x["k"] == k), None)
        if g is None:
            out.append({"k": k, "bars": v["bars"], "passes": list(v["passes"]),
                        "variantes": [v]})
        else:
            g["passes"] += list(v["passes"])
            g["variantes"].append(v)
    for g in out:
        g["passes"].sort()
    return out


def spans_de_fin(sec, gi_variante: int) -> list[list[float]]:
    """L'audio de la fin `gi_variante`, passe par passe. `barSpans` est listé
    dans l'ORDRE D'ÉMISSION du chart — les mesures du préfixe, puis la queue
    de chaque variante — donc on y marche avec le même curseur que `chart.js`,
    jamais en recalculant des temps."""
    tail = sec["endings"]["tail"]
    pref = len(sec["bars"]) - tail
    bs = sec.get("barSpans") or []
    debut = pref + gi_variante * tail
    lignes = bs[debut:debut + tail]
    if not lignes:
        return []
    n = len(lignes[0])
    return [[lignes[0][i][0], lignes[-1][i][1]] for i in range(n)]


# ── rendu ────────────────────────────────────────────────────────────────
def e(s) -> str:
    return html.escape(str(s))


def grille(rangees: list[dict]) -> str:
    """Une petite grille 4 colonnes, même treillis que l'app : barre noire
    continue à gauche, cellules bordées partout, la fin en boîte collée sous
    la rangée du dessus quand elle ne porte pas la ligne entière."""
    out = ['<div class="gr">']
    for r in rangees:
        cls = "row" + (" bare" if r.get("bare") else "")
        out.append(f'<div class="{cls}">')
        for c in r["cells"]:
            if c is None:
                out.append('<div class="cell vide"></div>')
                continue
            crochet = ""
            if c.get("label"):
                crochet = (f'<span class="brk" style="width:calc({c.get("span",1)*100}% - 6px)"></span>'
                           f'<span class="tick"></span>'
                           f'<span class="num">{e(c["label"])}</span>')
            pad = " haut" if c.get("label") or r.get("bare") else ""
            out.append(f'<div class="cell{pad}">{crochet}<span class="ch">{e(c["txt"])}</span></div>')
        out.append("</div>")
    out.append("</div>")
    return "".join(out)


def bande(passes: list[str], stem: str, sec_id: str, temps: list[list[float]]) -> str:
    """La bande de forme, une cellule par passe — c'est elle qui porte l'ORDRE
    quand le crochet ne le porte plus (Louis : « noter pareil sur le haut de
    la grille qui permet de suivre où on en est »). Chaque cellule joue sa
    propre passe : la bande n'est pas une légende, elle s'écoute."""
    out = ['<div class="bande">']
    for i, lab in enumerate(passes):
        alt = " alt" if lab.endswith("alt") else ""
        t = temps[i] if i < len(temps) else None
        onclick = (f' onclick="joue(\'{stem}\',{t[0]:.3f},{t[1]:.3f},this)"' if t else "")
        out.append(f'<button class="pas{alt}"{onclick} title="passe {i+1}">{e(lab)}</button>')
    out.append("</div>")
    return "".join(out)


def carte(stem: str, note_louis: str, regle: str) -> str:
    m = charge(stem)
    notes = notes_du_ton(m.get("key") or {})
    sec = next(s for s in m["sections"] if s.get("endings"))
    tail = sec["endings"]["tail"]
    pref = sec["bars"][:len(sec["bars"]) - tail]
    npass = len(sec.get("spans") or [])
    gs = groupes(sec)
    lettre = sec["label"]
    audio = (m.get("audio_url") or "").lstrip("/")

    # quel groupe est « classique » ?
    if regle == "frequence":
        idx_cl = max(range(len(gs)), key=lambda i: (len(gs[i]["passes"]), -gs[i]["passes"][0]))
    else:
        idx_cl = min(range(len(gs)), key=lambda i: gs[i]["passes"][0])

    # ── la grille d'AUJOURD'HUI, exactement comme l'app la pose ─────────
    # La 1re fin CONTINUE la rangée du préfixe — « c'est la suite du A »
    # (correction de Louis, 2026-07-21) ; seules les fins suivantes prennent
    # leur propre rangée. Le compter autrement gonflerait le coût d'avant et
    # ferait passer la proposition pour meilleure qu'elle n'est.
    ecrites = sec["endings"]["variants"]
    rangs_av: list[dict] = []
    col = len(pref) % 4
    cells = [{"txt": mesure(b, notes)} for b in pref]
    for j, b in enumerate(ecrites[0]["bars"]):
        if col + j < 4:
            cells.append({"txt": mesure(b, notes),
                          "label": "1." if j == 0 else None,
                          "span": len(ecrites[0]["bars"])})
    while len(cells) % 4:
        cells.append(None)
    for i in range(0, len(cells), 4):
        rangs_av.append({"cells": cells[i:i + 4]})
    for vi, v in enumerate(ecrites[1:], 2):
        ligne = [None] * 4
        for j, b in enumerate(v["bars"]):
            if col + j < 4:
                ligne[col + j] = {"txt": mesure(b, notes),
                                  "label": f"{vi}." if j == 0 else None,
                                  "span": len(v["bars"])}
        rangs_av.append({"cells": ligne, "bare": True})

    # ── la grille PROPOSÉE : la classique EN LIGNE, l'alt dessous ───────
    rangs_ap: list[dict] = []
    cl = gs[idx_cl]
    cells = [{"txt": mesure(b, notes)} for b in pref] + \
            [{"txt": mesure(b, notes)} for b in cl["bars"]]
    for j, b in enumerate(cl["bars"]):
        if j == 0:
            cells[len(pref)]["label"] = lettre
            cells[len(pref)]["span"] = len(cl["bars"])
    while len(cells) % 4:
        cells.append(None)
    for i in range(0, len(cells), 4):
        rangs_ap.append({"cells": cells[i:i + 4]})
    col_alt = len(pref) % 4
    for gi, g in enumerate(gs):
        if gi == idx_cl:
            continue
        suffixe = "alt" if len(gs) == 2 else f"alt {gi + 1}"
        ligne = [None] * 4
        for j, b in enumerate(g["bars"]):
            if col_alt + j < 4:
                ligne[col_alt + j] = {"txt": mesure(b, notes),
                                      "label": f"{lettre} {suffixe}" if j == 0 else None,
                                      "span": len(g["bars"])}
        rangs_ap.append({"cells": ligne, "bare": True})

    # ── la bande de forme, et l'audio de chaque passe ───────────────────
    etiq = ["?"] * npass
    for gi, g in enumerate(gs):
        lab = lettre if gi == idx_cl else (lettre + " alt")
        for p in g["passes"]:
            if p < npass:
                etiq[p] = lab
    temps = [None] * npass
    for vi, v in enumerate(ecrites):
        for k, p in enumerate(v["passes"]):
            sp = spans_de_fin(sec, vi)
            if p < npass and k < len(sp):
                # une mesure d'élan avant la fin : une fin ne se juge pas
                # décollée de ce qui l'amène
                t0, t1 = sp[k]
                temps[p] = [max(0.0, t0 - (t1 - t0) / max(1, tail)), t1]

    # ── ce que ça coûte en rangées ──────────────────────────────────────
    av_rows, ap_rows = len(rangs_av), len(rangs_ap)

    lignes_av = [f'{vi}. passes {", ".join(str(p+1) for p in v["passes"])} — '
                 f'{" | ".join(mesure(b, notes) for b in v["bars"])}'
                 for vi, v in enumerate(ecrites, 1)]
    ordre = [idx_cl] + [i for i in range(len(gs)) if i != idx_cl]   # la classique d'abord
    lignes_ap = [f'{lettre if gi==idx_cl else lettre+" alt"} : passes '
                 f'{", ".join(str(p+1) for p in gs[gi]["passes"])} — '
                 f'{" | ".join(mesure(b, notes) for b in gs[gi]["bars"])}'
                 for gi in ordre]

    return f"""
<section class="carte">
  <h2>{e(m.get('title') or stem)} <span class="sec">section {e(lettre)}, jouée {npass} fois</span></h2>
  <div class="deux">
    <div class="col">
      <h3>aujourd'hui <span class="cout">{av_rows} rangées</span></h3>
      {grille(rangs_av)}
      <ul class="det">{''.join(f'<li>{e(x)}</li>' for x in lignes_av)}</ul>
    </div>
    <div class="col">
      <h3>proposé <span class="cout {'ok' if ap_rows<av_rows else ''}">{ap_rows} rangées{' — même place' if ap_rows==av_rows else ''}</span></h3>
      {grille(rangs_ap)}
      <ul class="det">{''.join(f'<li>{e(x)}</li>' for x in lignes_ap)}</ul>
    </div>
  </div>
  <div class="ecoute">
    <div class="lg">la bande de forme — clique une passe pour entendre SA fin</div>
    {bande(etiq, audio, sec['id'], temps)}
  </div>
  <p class="dit"><b>Ce que ça te dit :</b> {e(note_louis)}</p>
</section>"""


PAGE = """<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Fins : « 1. / 2. » ou « B / B alt » ?</title>
<style>
 :root{--pap:#f7f3e9;--ink:#1c1c1c;--rule:#b9b09a;--faint:#8a8371;--line:#e5dcc6;--acc:#8a2b2b;--bg:#e7e0d0}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--ink);
      font:15px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif}
 .wrap{max-width:1000px;margin:0 auto;padding:18px 16px 60px}
 h1{font:600 italic 26px Georgia,serif;margin:0 0 6px}
 .chapo{color:var(--faint);margin:0 0 18px;max-width:70ch}
 table.cnt{border-collapse:collapse;margin:0 0 20px;background:var(--pap);border-radius:8px;overflow:hidden}
 table.cnt td,table.cnt th{padding:7px 12px;border-bottom:1px solid var(--line);text-align:left;font-size:14px}
 table.cnt th{font-weight:600;background:#efe9db}
 table.cnt td.n{font-variant-numeric:tabular-nums;font-weight:600;text-align:right}
 .bar{display:flex;gap:8px;align-items:center;margin:0 0 22px;flex-wrap:wrap}
 .bar b{font-weight:600}
 .bar button{font:600 13px inherit;padding:8px 14px;border:1px solid var(--rule);
   background:var(--pap);border-radius:999px;cursor:pointer;min-height:40px}
 .bar button[aria-pressed=true]{background:var(--acc);color:#f7f3e9;border-color:var(--acc)}
 .carte{background:var(--pap);border:1px solid var(--line);border-radius:12px;
   padding:16px;margin:0 0 20px}
 .carte h2{font:600 italic 20px Georgia,serif;margin:0 0 14px}
 .carte h2 .sec{font:500 13px -apple-system,system-ui,sans-serif;color:var(--faint);margin-left:8px}
 .deux{display:flex;gap:18px;flex-wrap:wrap}
 .col{flex:1 1 320px;min-width:0}
 .col h3{font:600 13px inherit;margin:0 0 8px;color:var(--faint);text-transform:uppercase;letter-spacing:.04em}
 .cout{float:right;text-transform:none;letter-spacing:0;color:var(--faint);font-weight:500}
 .cout.ok{color:#1f8a5b;font-weight:700}
 /* le treillis, même dessin que l'app */
 .gr{position:relative;background:var(--pap);border:1px solid var(--line);border-radius:6px;overflow:hidden}
 .gr::before{content:"";position:absolute;left:0;top:0;bottom:0;width:2px;background:var(--ink);z-index:2}
 .row{display:grid;grid-template-columns:repeat(4,minmax(0,1fr))}
 .cell{position:relative;min-height:46px;display:flex;align-items:center;padding:0 4px 0 7px;
   border-left:1px solid var(--rule);border-top:1px solid var(--rule);min-width:0}
 .row:first-child .cell{border-top:none}
 .cell:first-child{border-left:none}
 .cell.haut{padding-top:11px}
 .row.bare .cell{border:none;background:transparent}
 .row.bare .cell:not(.vide){border-left:1px solid var(--rule);border-right:1px solid var(--rule);
   border-bottom:1px solid var(--rule);min-height:38px}
 .row.bare .cell:not(.vide):first-child{border-left:none}
 .ch{font:600 italic 17px Georgia,serif;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
 .vide .ch{display:none}
 .brk{position:absolute;left:2px;top:1px;height:2px;background:var(--acc);z-index:3}
 .tick{position:absolute;left:2px;top:1px;width:2px;height:13px;background:var(--acc);z-index:3}
 .num{position:absolute;left:7px;top:4px;font:800 9px -apple-system,system-ui,sans-serif;
   line-height:1;color:var(--acc);z-index:3;white-space:nowrap}
 ul.det{list-style:none;padding:0;margin:8px 0 0;font-size:12.5px;color:var(--faint)}
 ul.det li{padding:2px 0;font-variant-numeric:tabular-nums}
 .ecoute{margin-top:14px;border-top:1px solid var(--line);padding-top:12px}
 .lg{font-size:12.5px;color:var(--faint);margin-bottom:7px}
 .bande{display:flex;gap:4px;flex-wrap:wrap}
 .pas{flex:1 1 60px;min-width:56px;min-height:44px;font:700 13px inherit;cursor:pointer;
   background:#fffdf6;border:1px solid var(--line);border-radius:5px;color:var(--ink)}
 .pas.alt{border-color:var(--acc);color:var(--acc);
   background:repeating-linear-gradient(135deg,#fffdf6,#fffdf6 5px,#f4e9e6 5px,#f4e9e6 10px)}
 .pas.joue{background:var(--acc);color:#f7f3e9;border-color:var(--acc)}
 .dit{margin:14px 0 0;padding:10px 12px;background:#efe9db;border-radius:8px;font-size:14px}
 @media(max-width:560px){ .deux{flex-direction:column} }
</style></head><body><div class="wrap">
<h1>« 1. / 2. » ou « B / B alt » ?</h1>
<p class="chapo">Les 22 charts de la bibliothèque qui portent des fins alternées, relus.
Le critère n'est pas <i>combien</i> de fins, c'est : <b>est-ce qu'une fin couvre plusieurs
passes ?</b> Si chaque fin appartient à une seule passe, « 1re fois ceci, 2e fois cela » est
vrai. Dès qu'une fin en couvre plusieurs, il n'y a plus de deuxième fois — et le numéro ment.</p>
<table class="cnt">
 <tr><th>forme</th><th>ce que ça veut dire</th><th>sections</th><th>notation</th></tr>
 <tr><td>une fin = une passe</td><td>1re fois ceci, 2e fois cela</td><td class="n">14 / 21</td><td>1. 2. — juste</td></tr>
 <tr><td>une fin couvre plusieurs passes</td><td>la fin habituelle, et une exception</td><td class="n">6 / 21</td><td><b>ment</b></td></tr>
 <tr><td>mélange (The Lazy Song)</td><td>4 fins pour 6 passes, dont une qui en couvre 3</td><td class="n">1 / 21</td><td><b>ment</b></td></tr>
</table>
<div class="bar"><b>« classique » veut dire :</b>
  <button id="bf" aria-pressed="true"  onclick="regle('frequence')">la fin la plus fréquente</button>
  <button id="bp" aria-pressed="false" onclick="regle('premiere')">la première jouée</button>
  <span id="note" style="color:var(--faint);font-size:13px"></span>
</div>
<div id="frequence">__FREQ__</div>
<div id="premiere" hidden>__PREM__</div>
</div>
<script>
// Deux pièges iOS, tous les deux déjà payés sur ce projet :
//   * le fichier passe par fetch() puis un blob — un <audio src> distant ne
//     bufferise jamais dans l'app installée (tempête de 206) ;
//   * AUCUNE boucle requestAnimationFrame — elle empêche WebKit de démarrer
//     son moteur audio. L'arrêt de l'extrait est piloté par `timeupdate`.
const cache = {};
let cour = null;
async function el(src){
  if(cache[src]) return cache[src];
  const a = new Audio();
  try{
    const r = await fetch('/' + src);
    a.src = URL.createObjectURL(await r.blob());
  }catch(e){ a.src = '/' + src; }
  a.preload = 'auto';
  cache[src] = a;
  return a;
}
function coupe(){
  // RETIRER l'écouteur, pas seulement mettre en pause. Deux extraits d'affilée
  // sur le MÊME morceau partagent l'élément <audio> : l'ancien `timeupdate`
  // restait accroché, voyait `currentTime` sauté au nouvel extrait — donc
  // au-delà de SA borne de fin — et coupait aussitôt la lecture qui commençait.
  if(!cour) return;
  try{ cour.a.pause(); }catch(e){}
  cour.a.removeEventListener('timeupdate', cour.stop);
  cour.b.classList.remove('joue');
  cour = null;
}
async function joue(src, t0, t1, btn){
  const reclic = cour && cour.b === btn;
  coupe();
  if(reclic) return;                       // deuxième tap sur la même passe = stop
  const a = await el(src);
  const stop = () => { if(a.currentTime >= t1) coupe(); };
  a.addEventListener('timeupdate', stop);
  try{ a.currentTime = t0; }catch(e){}
  btn.classList.add('joue');
  cour = {a, b: btn, stop};
  a.play().catch(()=>{ btn.classList.remove('joue'); cour = null; });
}
function regle(r){
  document.getElementById('frequence').hidden = (r !== 'frequence');
  document.getElementById('premiere').hidden  = (r !== 'premiere');
  document.getElementById('bf').setAttribute('aria-pressed', r === 'frequence');
  document.getElementById('bp').setAttribute('aria-pressed', r === 'premiere');
  document.getElementById('note').textContent =
    (r === 'frequence') ? 'seul This Love change entre les deux règles'
                        : 'sur This Love, le A♭ G redevient la fin classique';
  coupe();
}
regle('frequence');
</script></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=SORTIE)
    a = ap.parse_args()
    freq = "".join(carte(s, n, "frequence") for s, n in MORCEAUX)
    prem = "".join(carte(s, n, "premiere") for s, n in MORCEAUX)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(PAGE.replace("__FREQ__", freq).replace("__PREM__", prem))
    print(f"→ {a.out}")
    print(f"   http://100.89.209.63:7772/plots/{a.out.name}")


if __name__ == "__main__":
    main()
