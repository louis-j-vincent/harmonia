"""Six morceaux, leur vraie structure, et les boutons qui manquent pour l'obtenir.

Louis, 2026-09-18, après Fallin' : « je te laisse prendre quelques chansons dont
tu analyses et comprends toi-même la structure […] et voilà les knobs à bouger
pour que ces chansons soient bien découpées en sections par nous ».

CE QUE CETTE PAGE EST, ET CE QU'ELLE N'EST PAS. Je ne peux pas écouter l'audio.
Mon analyse part de l'harmonie ÉCRITE, des durées, et de ce que je connais de
ces morceaux. C'est ce qui a suffi pour Fallin', où la réponse était dans les
accords. Chaque lettre est donc jouable ici : la page existe pour que Louis
vérifie mon découpage à l'oreille, pas pour le lui annoncer.

LE RÉSULTAT PRINCIPAL, mesuré avec le vrai chemin de production
(`folding.merge_similar_letters`, centroïde des vecteurs de mesure, seuil
0.93) : il répare 5 morceaux sur 6 et en casse un — et AUCUN seuil ne répare
les six.

    pour fusionner Fallin'       il faut un seuil ≤ 0.961  (sa paire la plus basse)
    pour garder Let It Be intact il faut un seuil > 0.967  (couplet ↔ refrain)

La raison est structurelle, pas numérique : un centroïde est un SAC de notes,
aveugle à l'ORDRE. Le couplet de Let It Be (`C G | A- F | C G | F C`) et son
refrain (`A- G | F C | C G | F C`) sont faits des mêmes quatre accords dans un
ordre différent — leurs sacs sont les mêmes, leurs passages non.

    python -m tools.page_sections      # → docs/plots/sections_6morceaux.html
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np

from harmonia.folding import MERGE_LETTERS_COS, _bar_vecs
from harmonia.nnls_features import extract_bothchroma
from harmonia.sections.similarity import halfbar_features
from harmonia.settings import SETTINGS

CHARTS = SETTINGS.repo / "state" / "cache" / "charts"
SORTIE = SETTINGS.repo / "docs" / "plots" / "sections_6morceaux.html"
FLAT = ["C", "D♭", "D", "E♭", "E", "F", "G♭", "G", "A♭", "A", "B♭", "B"]
SHARP = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"]
FLAT_MAJ = {0, 1, 3, 5, 6, 8, 10}
TOK = {"": "", "6": "6", "^7": "maj7", "^": "maj7", "7": "7", "-": "m", "-7": "m7",
       "-^7": "mMaj7", "-6": "m6", "o": "dim", "o7": "dim7", "-7b5": "m7♭5",
       "h7": "ø7", "h": "ø", "9": "9", "-9": "m9", "^9": "maj9", "13": "13",
       "sus": "sus", "7sus": "7sus", "sus4": "sus4", "+": "+", "69": "6/9"}

# (chart, stem audio, ce que JE lis, sur quoi je me fonde, le verdict)
MORCEAUX = [
    ("min_Urdlvw0SSEc", "Urdlvw0SSEc", "Fallin'",
     "UNE boucle de 2 mesures, E−…B−, de la mesure 12 à la fin. Rien d'autre.",
     "L'harmonie : 97 mesures sur 97 alternent E− et B−. Les 8 lettres ne "
     "diffèrent que par la longueur (2, 4, 6, 8, 9 mesures), un B−7 au lieu "
     "d'un B−, et un E−/B au lieu d'un B− — trois façons d'écrire le même "
     "instant.",
     "SUR-DÉCOUPÉ : 8 lettres pour une boucle."),
    ("min_ben_e_king_stand_by_me_audio", "ben_e_king_stand_by_me_audio",
     "Stand By Me",
     "UNE boucle de 8 mesures (A…F♯m…D…E…A), jouée d'un bout à l'autre.",
     "Les mesures 5 à 8 de « A » et de « C » sont identiques au symbole près "
     "(F♯m·D | D·E | E·A | A). « A » ne diffère que par des N.C. là où seule "
     "la basse joue — le couplet nu du début.",
     "SUR-DÉCOUPÉ : deux lettres pour une boucle."),
    ("min_bobby_hebb_sunny_official_audio", "bobby_hebb_sunny_official_audio",
     "Sunny",
     "UNE forme de 8 mesures avec DEUX fins, rejouée un demi-ton plus haut.",
     "A et B partagent leurs 3 premières mesures (E− | G | C^7) et divergent "
     "sur la queue : c'est une 1re/2e fin, pas deux sections. C et D font "
     "exactement la même chose un demi-ton au-dessus (F− | A♭7 | D♭^7).",
     "SUR-DÉCOUPÉ : 4 lettres pour une forme à deux queues, plus sa "
     "transposition."),
    ("min_the_police_every_breath_you_take_official_music_video",
     "the_police_every_breath_you_take_official_music_video",
     "Every Breath You Take",
     "Couplet / refrain / pont central — trois sections VRAIMENT différentes, "
     "plus le couplet en trois longueurs.",
     "A, A′ et A″ sont le même couplet écrit sur 8, 10 et 7 mesures. B "
     "(D♭ D♭ A♭ A♭ B♭ B♭ E♭ E♭) est une autre musique. C module en mi/sol♭ — "
     "c'est le pont, il n'a rien à voir.",
     "PRESQUE JUSTE : seul le couplet est éclaté en trois."),
    ("min_yesterday_remastered_2009", "yesterday_remastered_2009", "Yesterday",
     "AABA, avec le A de 7 mesures — la forme irrégulière célèbre.",
     "A ×4 sur 7 mesures, B ×4 sur 4. C'est la forme du morceau, telle "
     "qu'elle est écrite partout.",
     "JUSTE. C'est le témoin : rien ne doit le casser."),
    ("min_let_it_be_remastered_2009", "let_it_be_remastered_2009", "Let It Be",
     "Couplet / refrain / pont, plus trois fragments de 2 mesures qui ne sont "
     "pas des sections.",
     "Le couplet est `C G | A− F | C G | F C`, le refrain `A− G | F C | C G | "
     "F C`. MÊMES quatre accords, ordre différent. B′ (2 mes.) est le début du "
     "refrain, C (2 mes.) sa queue, outro (2 mes.) la moitié du pont.",
     "C'EST LE CAS QUI CASSE LE LEVIER : le centroïde ne distingue pas le "
     "couplet du refrain."),
]

KNOBS = [
    ("<code>SETTINGS.merge_letters</code>", "False",
     "Fusionne les lettres dont les centroïdes chroma se confondent. Sa propre "
     "docstring le dit : « sans lui, deux occurrences identiques nommées "
     "différemment par le détecteur ne se rencontrent jamais ».",
     "Fallin', Stand By Me, Sunny, Every Breath"),
    ("<code>MERGE_LETTERS_COS</code>", "0.93",
     "Le seuil. Jamais mesuré sur le corpus — l'en-tête du code le dit : "
     "« hypothèse, un seul morceau ».",
     "aucun seuil ne sauve les six (voir ci-dessus)"),
    ("le groupage <code>(lettre, longueur)</code>", "dans <code>minimal_fold</code>",
     "Deux occurrences d'une même lettre à des longueurs différentes restent "
     "deux blocs écrits. C'est la règle « under-fold, never over-fold » du "
     "2026-07-30 — elle est juste en général, elle coûte ici.",
     "Fallin' (2/4/6/8/9 mes.), Every Breath (8/10/7)"),
    ("<code>_merge_coupe</code> refuse un bloc PLUS LONG", "en dur",
     "Un passage coupé rejoint le bloc long ; un passage plus LONG que le "
     "bloc écrit ne peut jamais le rejoindre (<code>len(blk) >= len(host)</code> "
     "→ refus). Le couplet de 10 mesures d'Every Breath est bloqué là.",
     "Every Breath (A′ de 10 mes.)"),
    ("la finesse de <code>_barsig</code>", "accord exact",
     "<code>B−</code>, <code>B−7</code> et <code>E−/B</code> comptent comme "
     "trois musiques différentes, donc la cascade leur donne des primes. Une "
     "signature à la FONDAMENTALE les confondrait.",
     "Fallin' (A′, B′, B″, B‴)"),
    ("aucune longueur minimale", "—",
     "Un bloc de 2 mesures est traité comme une section. Or une plage de 2 "
     "mesures est presque toujours une queue ou un élan.",
     "Let It Be (B′, C, outro)"),
    ("aucune transposition", "loi de production",
     "Le repli ne compare jamais deux passages à des hauteurs différentes — "
     "c'est écrit en tête de <code>fold_letter_groups</code>.",
     "Sunny (C/D = A/B un demi-ton plus haut)"),
]


def notes_du_ton(cle):
    tonic = int((cle or {}).get("tonic") or 0)
    maj = (tonic + 3) % 12 if (cle or {}).get("mode") == "minor" else tonic % 12
    return FLAT if maj in FLAT_MAJ else SHARP


def accord(c, notes):
    if c.get("nc"):
        return "N.C."
    q = c.get("q") or ""
    s = notes[c["root"] % 12] + TOK.get(q, q)
    b = c.get("bass")
    if b is not None and b >= 0 and b % 12 != c["root"] % 12:
        s += "/" + notes[b % 12]
    return s


def audio_for(stem):
    for d in (SETTINGS.repo / "docs" / "audio",
              Path.home() / "harmonia" / "data" / "audio"):
        for ext in (".m4a", ".mp3", ".wav"):
            p = d / (stem + ext)
            if p.exists():
                return p
    h = list((Path.home() / "harmonia").rglob(stem + ".m4a"))
    return h[0] if h else None


def cosinus(m, audio_stem):
    """La matrice que le vrai chemin de production calculerait — centroïdes des
    vecteurs de mesure BRUTS (`_bar_vecs`), comme `merge_similar_letters`."""
    ap = audio_for(audio_stem)
    if not ap:
        return None
    grid = m["barGrid"]
    arr, times = extract_bothchroma(ap)
    Vb = _bar_vecs(halfbar_features(grid, arr, times), len(grid) - 1)
    lettres = {}
    for s in m["sections"]:
        if str(s["label"]).lower() in ("intro", "outro"):
            continue
        lettres.setdefault(s["label"], []).extend(
            b for a, bb in s["barRanges"] for b in range(a, bb + 1)
            if b < len(Vb))
    noms = sorted(lettres)
    if len(noms) < 2:
        return None
    cen = {}
    for L in noms:
        v = Vb[lettres[L]].mean(0)
        n = float(np.linalg.norm(v))
        cen[L] = v / n if n else v
    return noms, [[float(cen[L] @ cen[M]) for M in noms] for L in noms]



def accords_cos(m):
    """L'AUTRE substrat : ce qu'on a DÉCODÉ, pas ce qu'on a entendu.

    Une mesure = le tuple de ses fondamentales (N.C. ignorés) ; deux lettres
    se comparent par la meilleure concordance CYCLIQUE sur la plus courte des
    deux. Aucun audio : tout est déjà dans le chart.

    C'est ce qui sépare là où la chroma échoue — la chroma porte la
    production (voix, ad-libs, mix) en plus de l'harmonie, donc deux passages
    aux mêmes accords mais à l'arrangement différent s'éloignent, et deux
    passages aux mêmes notes dans un ordre différent se confondent.
    """
    L = [x for x in m["sections"]
         if str(x["label"]).lower() not in ("intro", "outro")]
    if len(L) < 2:
        return None
    def sigs(sec):
        return [t for t in
                (tuple(c["root"] % 12 for c in bar if not c.get("nc"))
                 for bar in (sec.get("bars") or [])) if t]
    def sc(A, B):
        if not A or not B:
            return 0.0
        k = min(len(A), len(B))
        return max(sum(1 for j in range(k) if A[j % len(A)] == B[(j + r) % len(B)]) / k
                   for r in range(len(B)))
    noms = [x["label"] for x in L]
    S = {x["label"]: sigs(x) for x in L}
    return noms, [[sc(S[a], S[b]) for b in noms] for a in noms]


def e(x):
    return html.escape(str(x))


def carte(chart_stem, audio_stem, titre, lecture, fonde, verdict):
    m = json.loads((CHARTS / f"{chart_stem}.json").read_text())
    notes = notes_du_ton(m.get("key") or {})
    audio = (m.get("audio_url") or "").lstrip("/")
    lignes = []
    for s in m["sections"]:
        mes = " | ".join(" ".join(accord(c, notes) for c in bar) or "%"
                         for bar in (s.get("bars") or []))
        boutons = "".join(
            f'<button class="p" onclick="joue(\'{audio}\',{a:.2f},{b:.2f},this)">'
            f'{i + 1}</button>'
            for i, (a, b) in enumerate(s.get("spans") or []))
        lignes.append(
            f'<tr><td class="l">{e(s["label"])}</td>'
            f'<td class="n">×{s.get("reps")}</td>'
            f'<td class="n">{len(s.get("bars") or [])} mes.</td>'
            f'<td class="ac">{e(mes[:118])}</td>'
            f'<td class="pl">{boutons}</td></tr>')
    def matrice(res, seuil, fmt):
        if not res:
            return ""
        noms, mat = res
        t = ('<table class="cos"><tr><th></th>'
             + "".join(f"<th>{e(x)}</th>" for x in noms) + "</tr>")
        for i, L in enumerate(noms):
            t += f"<tr><th>{e(L)}</th>"
            for j in range(len(noms)):
                if i == j:
                    t += '<td class="d">—</td>'
                else:
                    v = mat[i][j]
                    t += (f'<td class="{"hi" if v >= seuil else ""}">'
                          + format(v, fmt) + "</td>")
            t += "</tr>"
        return t + "</table>"

    tc = (f'<div class="lg">1. <b>centroïde chroma</b> — ce que '
          f'<code>merge_letters</code> compare aujourd\'hui '
          f'(≥ {MERGE_LETTERS_COS} = fusion)</div>'
          + matrice(cosinus(m, audio_stem), MERGE_LETTERS_COS, ".3f")
          + '<div class="lg" style="margin-top:12px">2. <b>accords décodés</b> — '
            'la même question posée à ce qu\'on a ÉCRIT, pas à ce qu\'on a '
            'entendu (≥ 0.70 = même passage)</div>'
          + matrice(accords_cos(m), 0.70, ".2f"))
    return f"""
<section class="carte">
  <h2>{e(titre)}</h2>
  <p class="lis"><b>Ce que je lis :</b> {e(lecture)}</p>
  <p class="fond"><b>Sur quoi :</b> {e(fonde)}</p>
   <div class="tw"><table class="sec"><tr><th>lettre</th><th></th><th></th><th>accords écrits</th>
    <th>écouter</th></tr>{''.join(lignes)}</table></div>
  <div class="cosw">{tc}</div>
  <p class="verdict">{e(verdict)}</p>
</section>"""


PAGE = """<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Six morceaux, leur structure, et les boutons qui manquent</title>
<style>
 :root{--pap:#f7f3e9;--ink:#1c1c1c;--rule:#b9b09a;--faint:#8a8371;--line:#e5dcc6;--acc:#8a2b2b;--bg:#e7e0d0;--ok:#1f8a5b}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--ink);
      font:15px/1.55 -apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif}
 .wrap{max-width:1040px;margin:0 auto;padding:18px 16px 60px}
 h1{font:600 italic 26px Georgia,serif;margin:0 0 8px}
 .chapo{color:var(--faint);max-width:74ch;margin:0 0 6px}
 .cle{background:var(--pap);border-left:3px solid var(--acc);padding:12px 14px;
      border-radius:0 8px 8px 0;margin:16px 0 22px;max-width:74ch}
 .cle b{color:var(--acc)}
 .carte{background:var(--pap);border:1px solid var(--line);border-radius:12px;padding:16px;margin:0 0 18px}
 .carte h2{font:600 italic 21px Georgia,serif;margin:0 0 10px}
 .lis,.fond{margin:0 0 6px;font-size:14px} .fond{color:var(--faint)}
 table{border-collapse:collapse;width:100%;font-size:13px}
 table.sec{margin:12px 0 4px}
 table.sec th{text-align:left;font-weight:600;color:var(--faint);font-size:11.5px;
   text-transform:uppercase;letter-spacing:.04em;padding:0 8px 4px 0;border-bottom:1px solid var(--line)}
 table.sec td{padding:5px 8px 5px 0;border-bottom:1px solid var(--line);vertical-align:top}
 td.l{font:700 14px Georgia,serif;white-space:nowrap}
 td.n{color:var(--faint);white-space:nowrap;font-variant-numeric:tabular-nums}
 td.ac{font:italic 13px Georgia,serif}
 td.pl{white-space:nowrap;text-align:right}
 .p{min-width:26px;min-height:26px;margin-left:3px;border:1px solid var(--line);
    background:#fffdf6;border-radius:5px;font:600 11px inherit;cursor:pointer;color:var(--ink)}
 .p.joue{background:var(--acc);color:var(--pap);border-color:var(--acc)}
 .cosw{margin-top:12px} .lg{font-size:12px;color:var(--faint);margin-bottom:5px}
 table.cos{width:auto;font-variant-numeric:tabular-nums}
 table.cos th{font:600 11px inherit;color:var(--faint);padding:2px 7px}
 table.cos td{padding:2px 7px;text-align:right;border:1px solid var(--line)}
 table.cos td.hi{background:rgba(138,43,43,.14);color:var(--acc);font-weight:700}
 table.cos td.d{color:var(--line)}
 .verdict{margin:12px 0 0;padding:9px 12px;background:#efe9db;border-radius:8px;
   font-size:13.5px;font-weight:600}
 table.kn{background:var(--pap);border-radius:10px;overflow:hidden;margin-top:8px}
 table.kn th{background:#efe9db;text-align:left;padding:8px 10px;font-size:12px;
   text-transform:uppercase;letter-spacing:.04em;color:var(--faint)}
 table.kn td{padding:8px 10px;border-bottom:1px solid var(--line);vertical-align:top;font-size:13px}
 table.kn td:first-child{white-space:nowrap}
 code{font:12px ui-monospace,Menlo,monospace;background:#efe9db;padding:1px 4px;border-radius:3px}
 /* Les deux tableaux larges scrollent DANS leur boîte — le corps de la page
    ne scrolle jamais horizontalement (390px est la cible). */
 .tw,.cosw{overflow-x:auto;-webkit-overflow-scrolling:touch}
 table.kn{display:block;overflow-x:auto;white-space:normal}
 @media(max-width:600px){
   td.ac{display:none}
   table.sec{min-width:0}
   td.pl{white-space:normal;text-align:left}
 }
</style></head><body><div class="wrap">
<h1>Six morceaux, leur structure, et les boutons qui manquent</h1>
<p class="chapo"><b>Je ne peux pas écouter l'audio.</b> Ce qui suit part de
l'harmonie écrite, des durées, et de ce que je connais de ces morceaux. Chaque
lettre est jouable ici : la page existe pour que tu vérifies mon découpage à
l'oreille, pas pour te l'annoncer.</p>
<div class="cle"><b>Le résultat, mesuré avec le vrai chemin de production.</b>
Allumer <code>merge_letters</code> au seuil 0.93 répare <b>5 morceaux sur 6</b>
et en casse un. Et aucun seuil ne répare les six :<br>
pour fusionner Fallin' il faut un seuil <b>≤ 0.961</b> (sa paire la plus basse) ;
pour garder Let It Be intact il faut <b>&gt; 0.967</b> (couplet ↔ refrain).<br>
La raison n'est pas numérique : un centroïde est un <b>sac de notes, aveugle à
l'ordre</b>, et la chroma porte la <b>production</b> (voix, ad-libs, mix) en plus
de l'harmonie. Le couplet de Let It Be et son refrain sont faits des mêmes
quatre accords dans un ordre différent — même sac, deux passages.
<br><br><b>Ce qui sépare, lui.</b> Poser la même question aux accords DÉCODÉS
plutôt qu'à la chroma :
<table style="margin:8px 0 0;font-size:13px;width:auto">
<tr><th style="text-align:left;padding-right:14px"></th>
    <th style="padding:0 10px">Fallin' <span style="font-weight:400">(doit fusionner)</span></th>
    <th style="padding:0 10px">Let It Be <span style="font-weight:400">(ne doit pas)</span></th>
    <th style="padding:0 10px">écart</th></tr>
<tr><td style="padding-right:14px">centroïde chroma</td>
    <td style="text-align:center">min 0.961</td><td style="text-align:center">A–B 0.970</td>
    <td style="text-align:center;color:var(--acc);font-weight:700">−0.009</td></tr>
<tr><td style="padding-right:14px">accords décodés</td>
    <td style="text-align:center">min 0.75</td><td style="text-align:center">A–B 0.50</td>
    <td style="text-align:center;color:var(--ok);font-weight:700">+0.25</td></tr>
</table>
<br>Un essai intermédiaire — comparer les SÉQUENCES de chroma au lieu de leur
moyenne, pour rendre la mesure sensible à l'ordre — a été mesuré et <b>échoue
aussi</b> (Fallin' min 0.839, Let It Be A–B 0.864 : l'ordre est encore inversé).
Le problème n'est pas la moyenne, c'est le substrat.</div>
__CARTES__
<h2 style="font:600 italic 22px Georgia,serif;margin:26px 0 4px">Les boutons</h2>
<table class="kn"><tr><th>bouton</th><th>valeur</th><th>ce qu'il fait</th><th>qui en a besoin</th></tr>
__KNOBS__
</table>
</div>
<script>
// Deux pièges iOS déjà payés : fetch()+blob (un <audio src> distant ne
// bufferise jamais dans l'app installée) et AUCUNE boucle rAF (elle empêche
// WebKit de démarrer son moteur audio). L'arrêt est piloté par `timeupdate`,
// et l'écouteur est RETIRÉ à chaque coupe — sinon le précédent tue l'extrait
// suivant sur le même morceau.
const cache = {};
let cour = null;
async function el(src){
  if(cache[src]) return cache[src];
  const a = new Audio();
  try{ const r = await fetch('/' + src); a.src = URL.createObjectURL(await r.blob()); }
  catch(e){ a.src = '/' + src; }
  a.preload = 'auto'; cache[src] = a; return a;
}
function coupe(){
  if(!cour) return;
  try{ cour.a.pause(); }catch(e){}
  cour.a.removeEventListener('timeupdate', cour.stop);
  cour.b.classList.remove('joue'); cour = null;
}
async function joue(src, t0, t1, btn){
  const reclic = cour && cour.b === btn;
  coupe();
  if(reclic) return;
  const a = await el(src);
  const stop = () => { if(a.currentTime >= t1) coupe(); };
  a.addEventListener('timeupdate', stop);
  try{ a.currentTime = t0; }catch(e){}
  btn.classList.add('joue');
  cour = {a, b: btn, stop};
  a.play().catch(()=>{ btn.classList.remove('joue'); cour = null; });
}
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=SORTIE)
    a = ap.parse_args()
    cartes = "".join(carte(*x) for x in MORCEAUX)
    knobs = "".join(
        f"<tr><td>{k}</td><td><code>{e(v)}</code></td><td>{d}</td>"
        f"<td style='color:var(--faint)'>{e(q)}</td></tr>"
        for k, v, d, q in KNOBS)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(PAGE.replace("__CARTES__", cartes).replace("__KNOBS__", knobs))
    print(f"→ {a.out}")
    print(f"   http://100.89.209.63:7772/plots/{a.out.name}")


if __name__ == "__main__":
    main()
