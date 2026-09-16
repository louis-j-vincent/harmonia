"""La page « comment un accord est décidé » : la chaîne, et un vrai morceau dedans.

Louis, 2026-09-16 : « ça me soûle d'arbitrer 10 mille règles comme ça, j'ai déjà
arbitré assez pour que tu puisses inférer des règles qui marchent, montres moi
actuellement les décisions qu'on prend sur les accords et comment on y arrive ».

Deux moitiés, dans cet ordre :

  1. UN VRAI MORCEAU. Les accords de Ready, chacun avec ce que musx a entendu,
     ce que la frontière a coûté, ce que la basse a été lue et quelle branche
     a tranché. C'est « comment on y arrive », montré et non raconté.
  2. LA CHAÎNE, et surtout la PROVENANCE de chaque nombre : arbitré par Louis /
     mesuré sur un corpus / hérité d'un modèle / posé sans justification.

Le tableau de provenance vient de `scratchpad/decisions_inventory.md` (audit du
2026-09-16, lecture du code et non des docstrings). Le tracé vient de
`scratchpad/trace_chords.py`, qui rejoue la chaîne réelle.

    .venv/bin/python -m tools.page_decisions
"""
from __future__ import annotations

import html
import json
from pathlib import Path

from harmonia.settings import SETTINGS

TRACE = SETTINGS.repo / "scratchpad" / "trace_chords.json"
HEAT = SETTINGS.repo / "scratchpad" / "heatmap.json"
OUT = SETTINGS.reports_dir / "decisions" / "accords.html"
NAMES = "C Db D Eb E F Gb G Ab A Bb B".split()

# ── la chaîne, condensée depuis l'inventaire ────────────────────────────────
# (nom, ce que l'étage décide, [(constante, valeur, provenance, note)])
CHAINE = [
    ("Les temps", "où tombe chaque temps, et si le morceau est jouable du tout", [
        ("Beat This!, jamais librosa", "—", "arbitré",
         "2026-07-21, verrou d'octave 2× mesuré : 65 % contre 78 % sur POP909"),
        ("DUP_TOL / LARGE_TOL", "0,25 / 0,45", "mesuré",
         "199 morceaux en cache ; c'est le bug de Ready d'hier"),
        ("HALF_TOL", "0,18", "mesuré", "66 morceaux, 7 modifiés, aucun verdict changé"),
        ("GRID_MIN_COVERAGE", "0,85", "mesuré",
         "le SEUL seuil du dépôt qu'on sait re-mesurer — mais le fossé fait deux morceaux"),
        ("GRID_MIN_BARS", "30", "posé", "sous 30 mesures on ne juge pas"),
    ]),
    ("Ce que le modèle entend", "une probabilité par accord, toutes les 23 ms", [
        ("les 5 folds ISMIR 2019", "—", "hérité", "le clone vendu, moyenné"),
        ("22050 / 512 = 23,22 ms", "—", "hérité", "erreur d'unité ici = tout faux en silence"),
    ]),
    ("Quel accord sur quel temps", "LE CŒUR — où les accords changent", [
        ("DEFAULT_PENALTY", "40,0", "posé",
         "INERTE, mesuré le 2026-09-16 : de 5 à 200, le morceau rend exactement les "
         "mêmes 86 accords. Le décodeur ne la lit que sur les images hors de la "
         "grille de temps (avant le premier temps, après le dernier) ; sur le corps "
         "du morceau, ce sont les coûts par temps qui s'appliquent. Son commentaire "
         "annonce « pooled optimum 40, LOSO-stable sur 7 morceaux » — aucune trace "
         "de cette étude, et elle porterait de toute façon sur un curseur mort."),
        ("coût de changement", "15 / 45 / 100", "arbitré",
         "temps fort / mi-mesure / autre temps. LE VRAI CURSEUR : 5/15/30 donne 92 "
         "accords, 40/80/200 en donne 78. La forme est arbitrée (This Love, "
         "2026-07-31) ; les trois nombres, eux, sont posés."),
        ("granularité", "tout temps", "arbitré",
         "Louis 2026-08-07 : « on ne met plus de restrictions sur la granularité »"),
    ]),
    ("Les barres de mesure", "où commence chaque mesure", [
        ("phase = résidu majoritaire", "un seul nombre pour tout le morceau", "hérité",
         "les temps forts du traceur votent ; la marque « Set bar 1 » les surclasse"),
        ("chute d'un N de tête", "2,0 × le temps", "posé",
         "mal réglé, ça déplace TOUT le morceau d'une mesure"),
        ("re-calage harmonique", "0,55 / 0,15", "posé",
         "MORT : ne se déclenche sur aucun des 44 morceaux"),
    ]),
    ("Les sections", "où commencent couplet, refrain, pont", [
        ("SongFormer", "—", "arbitré",
         "Louis sur /plots/songformer.html : « je suis d'accord avec lui partout, "
         "on le prend en prod ». Aucun repli."),
    ]),
    ("Le repli", "si une section garde son décodage ou prend le consensus de ses passages", [
        ("STACK_COHERENCE", "0,85", "posé",
         "le levier le plus violent du chart : au-dessus, une section entière est "
         "réécrite par son gabarit ; en dessous, aucune. Calibré sur 3 morceaux."),
        ("PERIOD_MIN_SCORE", "0,80", "posé", "2 à 5 morceaux"),
        ("OUTLIER_Z", "3,0", "posé", "quelle mesure échappe à la pile"),
        ("CV_MAX", "0,51", "posé",
         "le seul avec un taux d'erreur nommé (5 % de fausses fusions) — script et "
         "données absents du dépôt"),
        ("FIN_SECTION_HORS_PILE", "1", "posé",
         "le commentaire dit lui-même que Louis avait dit « souvent les 2 dernières »"),
        ("coût mi-mesure du gabarit", "15 au lieu de 45", "posé",
         "le 2ᵉ décodage peut couper une mesure là où le 1ᵉʳ ne le ferait pas. "
         "Justifié par UN accord observé."),
    ]),
    ("La tonalité et la couleur", "majeur/mineur, et l'orthographe ♭/♯ de tout le chart", [
        ("MODE_MASS_MIN", "1,5", "posé", "décide majeur/mineur, donc l'orthographe"),
        ("Q, GAIN, LAMBDA", "0,85 / 25 / 0,25", "posé",
         "en-tête : « byte-identical to v4.1, do not retune here » — l'inverse d'une justification"),
        ("1,25 décisivité de durée", "1,25", "posé", "le code dit lui-même : calibré sur deux morceaux"),
    ]),
    ("La basse du slash", "si on écrit /Bb après l'accord", [
        ("FLOOR", "30 %", "arbitré",
         "33 arbitrages à l'oreille dans state/human/bass_verdicts.json, rejoués par "
         "tests/test_bass_rules.py — le SEUL bloc du système où changer la règle "
         "fait rougir un test"),
        ("PLAUSIBLE", "{fond., 9e, 3ce m, 3ce M, 5te}", "arbitré", "idem"),
        ("UNTESTED traités comme impossibles", "{b5, b13, 6te, b7}", "posé",
         "jamais soumis à l'oreille — posé dans le bon sens (défaut = accord nu)"),
    ]),
]


# ── une illustration par étage : un cas RÉEL, mesuré, pas un dessin ─────────
# Toutes prises sur Ready (PJ Morton), le morceau tracé plus haut, pour qu'on
# puisse écouter la même musique d'un étage à l'autre.
# (titre, ce que l'étage change ici, [(gauche, droite)], [(libellé, t0, t1)])
ILLUS = {
 1: ("Un temps parasite déplaçait toutes les barres",
     "Beat This! avait posé deux marques à 180 ms d'écart à 13,9 s. Le filtre à "
     "jumeaux coupait juste au-dessus. Un temps de trop dans la liste, et toutes "
     "les barres suivantes tombaient un temps trop tôt.",
     [("mesure 9 à 20,900 s", "mesure 9 à 21,520 s"),
      ("mesure 10 à 23,540 s", "mesure 10 à 24,100 s")],
     [("la couture, comme c'était", 19.4, 23.2), ("le vrai départ de la mesure 9", 21.52, 24.10)]),
 2: ("Deux accords à deux points d'écart",
     "Le modèle ne rend pas un accord mais une probabilité par accord, toutes les "
     "23 ms. À la mesure 2, D♭ et B♭m se tiennent à deux points — c'est l'étage "
     "suivant qui tranche, pas celui-ci.",
     [("D♭ · 33 %", "B♭m · 31 %"), ("puis E♭7 · 12 %", "et le reste sous 5 %")],
     [("la mesure 2, où ça hésite", 3.90, 6.42), ("la mesure 3, où c'est net (E♭ 80 %)", 6.42, 9.06)]),
 3: ("La constante la plus documentée du système ne fait rien",
     "On fait varier ce qu'on croyait être le curseur principal, la pénalité de "
     "changement : de 5 à 200, le morceau rend exactement 86 accords. Elle ne "
     "s'applique qu'aux images HORS de la grille de temps — la seconde avant le "
     "premier temps et après le dernier. Ce sont les trois coûts par temps qui "
     "décident tout.",
     [("pénalité 5 → 86 accords", "pénalité 200 → 86 accords"),
      ("coûts 5/15/30 → 92 accords", "coûts 40/80/200 → 78 accords")],
     [("les 4 premières mesures", 1.38, 11.56)]),
 4: ("La phase des barres est UN nombre pour tout le morceau",
     "Les temps forts du traceur votent, et la marque « Set bar 1 » les surclasse. "
     "Un morceau qui gagnerait vraiment un temps au milieu reste inécrivable : "
     "c'est ce qui a rendu le bug de l'étage 1 si visible.",
     [("phase donnée par ta marque", "mesure 1 à 1,355 s"),
      ("un seul résidu, tout le morceau", "aucun changement de phase possible")],
     [("mesure 1, la marque", 1.38, 3.90), ("mesure 9, huit mesures plus loin", 21.52, 24.10)]),
 5: ("Une section devient six",
     "Avant cet étage, le chart brut est une seule section d'un bloc. SongFormer "
     "écoute le son et nomme les parties — c'est ce qui rend le chart lisible et "
     "ce qui rend le repli possible.",
     [("brut : 1 section", "final : intro · A · B · A′ · B′ · B″"),
      ("aucune répétition vue", "A revient en A′, B en B′ et B″")],
     [("intro", 1.38, 5.0), ("A", 21.52, 25.1), ("B", 52.28, 55.9),
      ("A′", 82.8, 86.4), ("B′", 102.86, 106.5), ("B″", 126.14, 129.7)]),
 6: ("82 accords écrits deviennent 74",
     "Une section jouée plusieurs fois n'est écrite qu'une fois, et ses passages "
     "sont empilés pour se mettre d'accord. C'est le levier le plus violent du "
     "chart : au-dessus du seuil de cohérence, TOUTES les mesures de la lettre "
     "sont réécrites par le gabarit commun.",
     [("brut : 82 accords écrits", "final : 74"),
      ("A écrite 1 fois", "jouée en A et A′")],
     [("A, mesure 9", 21.52, 26.66), ("A′, le même endroit", 82.8, 87.9)]),
 7: ("« A# minor » devient « B♭ minor »",
     "L'analyse harmonique nomme la tonalité, et ce nom décide l'orthographe de "
     "TOUT le chart : les mêmes notes s'écrivent en bémols ou en dièses. C'est "
     "exactement l'erreur que tu m'as signalée sur la page d'arbitrage.",
     [("brut : A# minor", "final : B♭ minor"),
      ("un G♭^7 en la# mineur", "s'écrirait F#maj7")],
     [("l'accord de tonique", 1.38, 3.90), ("le G♭^7 de la mesure 8", 18.96, 21.52)]),
 8: ("Le slash qu'on écrit, et la lecture à 25 % qui le pose",
     "La basse est lue à l'ATTAQUE de chaque temps, pas en moyenne. Ici la "
     "fondamentale D♭ n'est pas retrouvée au-dessus du plancher, donc la lecture "
     "du temps 1 est gardée — E♭, à 25 %. Tes verdicts r16/r17/r19 disent que "
     "c'est juste : un slash faible peut être vrai.",
     [("sans la règle : D♭", "avec : D♭/E♭"),
      ("lecture du temps 1 : E♭ 25 %", "intervalle +2, une 9e — jouable")],
     [("l'attaque, ce que la règle lit", 3.90, 4.80), ("la mesure entière", 3.90, 6.42)]),
}

PROV = {"arbitré": ("mark", "arbitré par toi"), "mesuré": ("acc", "mesuré"),
        "hérité": ("dim", "hérité"), "posé": ("warn", "posé sans justification")}


def e(x):
    return html.escape(str(x))


def page(tr: dict) -> str:
    ac = tr["accords"]
    audio_js = json.dumps("/audio/" + tr["key"] + ".m4a")
    H = tr.get("heat") or {}
    heat_json = json.dumps(H, ensure_ascii=False, separators=(",", ":"))
    heat_n = len(H.get("lignes", [{}])[0].get("p", [])) if H.get("lignes") else 0
    heat_k = len(H.get("lignes") or [])
    heat_reste = 73 - heat_k
    heat_pct = f"{(H.get('reste_moyen') or 0) * 100:.0f} %"
    heat_t0 = H.get("t0", 0.0)
    heat_t1 = H.get("t1", 0.0)
    heat_b2 = (H.get("barres") or [0, 0, 0])[0]
    heat_b3 = (H.get("barres") or [0, 0, 0])[1]
    heat_b3plus = heat_b3 + 0.9
    n_prov = {k: 0 for k in PROV}
    for _, _, cs in CHAINE:
        for c in cs:
            n_prov[c[2]] += 1

    def trace_carte(r):
        top = " ".join(
            f"<span class='cand{' w' if i == 0 else ''}'>{e(a)}<i>{b:.0f}%</i></span>"
            for i, (a, b) in enumerate(r["musx_top"][:3]))
        bl = " ".join(f"<span class='cand'>{e(NAMES[x['pc']])}<i>{x['share']:.0f}%</i></span>"
                      for x in r["basse_lue"][:4])
        cout = r["cout_transition"]
        cname = {15: "temps fort", 45: "mi-mesure", 100: "autre temps"}[cout]
        why = r["basse_decision"]["pourquoi"]
        sous = (r["bass"] >= 0 and r["basse_lue"]
                and r["basse_lue"][0]["share"] < tr["constantes"]["plancher_basse"])
        t0, t1 = r["t0"], r["t1"]
        # Trois écoutes, chacune répondant à une question différente :
        # le contexte dit si l'accord tombe au bon endroit ; l'accord seul dit
        # si l'étiquette est juste ; l'attaque est LITTÉRALEMENT ce que la règle
        # de basse lit (les 150 premières ms), étirée à ce que l'oreille peut
        # saisir.
        ec = (f"<button class='pl' data-t0='{max(0, t0 - 1.6):.3f}' data-t1='{t1:.3f}'"
              f" type='button'>&#9658; en contexte</button>"
              f"<button class='pl' data-t0='{t0:.3f}' data-t1='{t1:.3f}'"
              f" type='button'>&#9658; l'accord</button>"
              f"<button class='pl' data-t0='{t0:.3f}' data-t1='{t0 + 0.9:.3f}'"
              f" type='button'>&#9658; l'attaque</button>")
        return f"""
<div class="tr">
  <div class="trh"><b>{e(r['label'])}</b><span class="mes">mes. {r['bar']}.{r['beat']}</span>
    <span class="sec">{e(r['section'])}</span><span class="t">{r['t0']:.2f}s</span></div>
  <div class="acts">{ec}</div>
  <div class="tl"><span class="k">le modèle entend</span>{top}</div>
  <div class="tl"><span class="k">le changement coûte</span>
    <span class="cout c{cout}">{cout} &middot; {cname}</span></div>
  <div class="tl"><span class="k">la basse, temps par temps</span>{bl}</div>
  <div class="tl"><span class="k">la règle tranche</span>
    <span class="why">{e(why)}</span>
    {"<span class='flag'>lecture sous le plancher de 30 %</span>" if sous else ""}</div>
</div>"""

    def etage(i, nom, quoi, cs):
        badges = "".join(
            f"<tr class='r1'><td class='cst'>{e(c[0])}</td><td class='val'>{e(c[1])}</td>"
            f"<td class='bx'><span class='b b-{PROV[c[2]][0]}'>{e(c[2])}</span></td></tr>"
            f"<tr class='r2'><td class='note' colspan='3'>{e(c[3])}</td></tr>" for c in cs)
        il = ILLUS.get(i)
        bloc = ""
        if il:
            titre, texte, paires, ecoutes = il
            pp = "".join(
                f"<div class='pr'><span class='g'>{e(a)}</span>"
                f"<span class='fl'>&rarr;</span><span class='d'>{e(b)}</span></div>"
                for a, b in paires)
            bb = "".join(
                f"<button class='pl' data-t0='{t0:.3f}' data-t1='{t1:.3f}' "
                f"type='button'>&#9658; {e(lab)}</button>" for lab, t0, t1 in ecoutes)
            bloc = f"""
  <div class="illus">
    <div class="ith">{e(titre)}</div>
    <p class="itx">{e(texte)}</p>
    <div class="prs">{pp}</div>
    <div class="acts ill">{bb}</div>
  </div>"""
        return f"""
<div class="etage">
  <div class="eh"><span class="num">{i}</span><b>{e(nom)}</b><span class="quoi">{e(quoi)}</span></div>
  {bloc}
  <div class="wrap"><table>{badges}</table></div>
</div>"""

    return f"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Comment un accord est décidé</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&family=IBM+Plex+Mono:wght@400;500;600&family=Public+Sans:wght@400;500&display=swap">
<style>
 :root{{--bg:#F5F1E7;--surface:#fff;--surface-2:#FBF7EE;--rule:#E3DAC4;--rule-strong:#C9BE9F;
  --ink:#211C14;--ink-dim:#6F6555;--ink-faint:#A79C86;--accent:#B4632A;--accent-soft:#EFDFC5;
  --accent-ink:#5A3315;--mark:#1F6E5C;--mark-soft:#DCEDE7;--warn:#B23A33;--warn-soft:#F6E0DD;}}
 @media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{
  --bg:#14120D;--surface:#1B1812;--surface-2:#211D16;--rule:#332D22;--rule-strong:#4A4130;
  --ink:#ECE4D3;--ink-dim:#A89D89;--ink-faint:#726858;--accent:#DE9251;--accent-soft:#3B2C1B;
  --accent-ink:#F3D2AC;--mark:#59B39C;--mark-soft:#1B302B;--warn:#E1837B;--warn-soft:#3A2220;}}}}
 *{{box-sizing:border-box}}
 body{{background:var(--bg);color:var(--ink);margin:0 auto;max-width:940px;
  font-family:'Public Sans',-apple-system,sans-serif;padding:22px 16px 80px;font-size:14.5px;line-height:1.55}}
 h1{{font-family:'Fraunces',Georgia,serif;font-size:27px;font-weight:600;margin:0;text-wrap:balance}}
 h2{{font-family:'Fraunces',Georgia,serif;font-size:19px;margin:32px 0 4px}}
 .eyebrow{{font-family:'IBM Plex Mono',monospace;font-size:11px;letter-spacing:.09em;text-transform:uppercase;
  color:var(--accent-ink);background:var(--accent-soft);display:inline-block;padding:2px 7px;border-radius:3px;margin-bottom:8px}}
 .sub{{color:var(--ink-dim);font-size:13.5px;margin:5px 0 0}}
 .card{{background:var(--surface);border:1px solid var(--rule);border-radius:11px;padding:14px 16px;margin:14px 0}}
 code{{font-family:'IBM Plex Mono',monospace;font-size:12.5px;background:var(--surface-2);
  border:1px solid var(--rule);border-radius:4px;padding:1px 5px}}
 .wrap{{overflow-x:auto}}
 table{{border-collapse:collapse;width:100%;font-size:12.5px}}
 td{{padding:6px 7px;border-bottom:1px solid var(--rule);vertical-align:top}}
 td:first-child{{padding-left:0}} td:last-child{{padding-right:0}}
 .cst{{font-family:'IBM Plex Mono',monospace;font-weight:500;white-space:nowrap}}
 .val{{font-family:'IBM Plex Mono',monospace;color:var(--ink-dim);white-space:nowrap}}
 .note{{color:var(--ink-dim);font-size:12px;padding-top:0;padding-bottom:9px}}
 tr.r1 td{{border-bottom:none;padding-bottom:2px}}
 td.bx{{text-align:right;white-space:nowrap}}
 .b{{font-family:'IBM Plex Mono',monospace;font-size:10px;padding:2px 6px;border-radius:4px;white-space:nowrap}}
 .b-mark{{background:var(--mark-soft);color:var(--mark)}}
 .b-warn{{background:var(--warn-soft);color:var(--warn);font-weight:600}}
 .b-acc{{background:var(--accent-soft);color:var(--accent-ink)}}
 .b-dim{{background:var(--surface-2);color:var(--ink-faint);border:1px solid var(--rule)}}

 .split{{display:flex;gap:3px;margin:12px 0 6px;height:26px;border-radius:6px;overflow:hidden}}
 .split div{{display:flex;align-items:center;justify-content:center;overflow:hidden;
  font-family:'IBM Plex Mono',monospace;font-size:11.5px;color:#fff;font-weight:600}}
 .s-mark{{background:var(--mark)}} .s-warn{{background:var(--warn)}}
 .s-acc{{background:var(--accent)}} .s-dim{{background:var(--ink-faint)}}
 .leg{{font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--ink-dim);
  display:flex;gap:12px;flex-wrap:wrap;align-items:center}}
 .sw{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px}}

 .etage{{border:1px solid var(--rule);border-radius:11px;background:var(--surface);
  padding:12px 15px;margin-bottom:9px}}
 .eh{{display:flex;gap:9px;align-items:baseline;flex-wrap:wrap;margin-bottom:6px}}
 .num{{font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--accent-ink);
  background:var(--accent-soft);border-radius:4px;padding:1px 7px}}
 .quoi{{color:var(--ink-dim);font-size:12.5px}}
 .illus{{background:var(--surface-2);border:1px solid var(--rule);border-left:3px solid var(--accent);
  border-radius:8px;padding:11px 13px;margin:8px 0 11px}}
 .ith{{font-weight:600;font-size:13.5px;margin-bottom:5px}}
 .itx{{font-size:12.5px;color:var(--ink-dim);margin:0 0 9px}}
 .prs{{display:flex;flex-direction:column;gap:5px;margin-bottom:9px}}
 .pr{{display:flex;gap:8px;align-items:baseline;flex-wrap:wrap;
  font-family:'IBM Plex Mono',monospace;font-size:12px}}
 .pr .g{{color:var(--ink-dim)}} .pr .fl{{color:var(--ink-faint)}}
 .pr .d{{color:var(--accent-ink);font-weight:600}}
 .acts.ill{{border-bottom:none;padding-bottom:0;margin-bottom:0}}
 .hwrap{{overflow-x:auto;-webkit-overflow-scrolling:touch}}
 #heat{{display:block;min-width:660px;width:100%;height:auto;border-radius:6px;
  border:1px solid var(--rule);touch-action:pan-x}}
 .hlab{{font-family:'IBM Plex Mono',monospace;font-size:11.5px;color:var(--ink-dim);
  margin:7px 0 9px;min-height:1.4em}}
 .hlab b{{color:var(--ink)}}

 .tr{{border:1px solid var(--rule);border-radius:10px;background:var(--surface);
  padding:11px 13px;margin-bottom:7px}}
 .trh{{display:flex;gap:9px;align-items:baseline;flex-wrap:wrap;margin-bottom:7px}}
 .trh b{{font-family:'IBM Plex Mono',monospace;font-size:16px}}
 .mes,.t{{font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--ink-faint)}}
 .sec{{font-size:11px;color:var(--ink-dim);background:var(--surface-2);border:1px solid var(--rule);
  border-radius:4px;padding:1px 6px;margin-left:auto}}
 .tl{{display:flex;gap:7px;align-items:center;flex-wrap:wrap;margin-bottom:5px}}
 .k{{font-size:11px;color:var(--ink-faint);min-width:150px;font-family:'IBM Plex Mono',monospace}}
 .cand{{font-family:'IBM Plex Mono',monospace;font-size:12px;background:var(--surface-2);
  border:1px solid var(--rule);border-radius:5px;padding:1px 7px;color:var(--ink-dim)}}
 .cand i{{font-style:normal;color:var(--ink-faint);margin-left:5px;font-size:10.5px}}
 .cand.w{{background:var(--mark-soft);border-color:var(--mark);color:var(--mark);font-weight:600}}
 .cand.w i{{color:var(--mark)}}
 .cout{{font-family:'IBM Plex Mono',monospace;font-size:11.5px;border-radius:5px;padding:1px 7px}}
 .c15{{background:var(--mark-soft);color:var(--mark)}}
 .c45{{background:var(--accent-soft);color:var(--accent-ink)}}
 .c100{{background:var(--warn-soft);color:var(--warn)}}
 .why{{font-size:12.5px;color:var(--ink)}}
 .flag{{font-family:'IBM Plex Mono',monospace;font-size:10.5px;background:var(--warn-soft);
  color:var(--warn);border-radius:4px;padding:1px 7px}}
 .acts{{display:flex;gap:6px;flex-wrap:wrap;margin:0 0 9px;padding-bottom:9px;
  border-bottom:1px dashed var(--rule)}}
 button{{font-family:'IBM Plex Mono',monospace;font-size:11.5px;border-radius:6px;padding:5px 10px;
  cursor:pointer;border:1px solid var(--rule);background:var(--surface-2);color:var(--ink-dim)}}
 button:hover{{border-color:var(--accent);color:var(--accent-ink);background:var(--accent-soft)}}
 button:focus-visible{{outline:2px solid var(--accent);outline-offset:2px}}
 button.playing{{background:var(--accent);border-color:var(--accent);color:var(--surface)}}
 .bandeau{{position:sticky;top:0;z-index:20;background:var(--surface);border:1px solid var(--rule);
  border-radius:10px;padding:8px 12px;margin:14px 0;display:flex;gap:12px;flex-wrap:wrap;
  align-items:center;font-family:'IBM Plex Mono',monospace;font-size:12px;color:var(--ink-dim)}}
 .opt{{display:inline-flex;align-items:center;gap:5px;cursor:pointer}}
 .opt input{{accent-color:var(--accent);width:14px;height:14px}}
 footer{{margin-top:32px;padding-top:12px;border-top:1px solid var(--rule);font-size:11.5px;
  color:var(--ink-faint);font-family:'IBM Plex Mono',monospace;line-height:1.8}}
</style>

<span class="eyebrow">la chaîne d'accords &middot; audit du 2026-09-16</span>
<h1>Comment un accord est décidé</h1>
<p class="sub">64 décisions séparent le fichier audio d'un accord imprimé dans une mesure.
Voici lesquelles, et d'où vient chaque nombre.</p>

<div class="card">
  <div class="split">
    <div class="s-mark" style="flex:25">25</div>
    <div class="s-warn" style="flex:22">22</div>
    <div class="s-acc" style="flex:12">12</div>
    <div class="s-dim" style="flex:5">5</div>
  </div>
  <div class="leg">
    <span><i class="sw s-mark"></i>arbitrés par toi</span>
    <span><i class="sw s-warn"></i><b style="color:var(--warn)">posés sans justification</b></span>
    <span><i class="sw s-acc"></i>mesurés</span>
    <span><i class="sw s-dim"></i>hérités</span>
  </div>
  <p class="sub" style="margin-top:10px"><b>Ces chiffres flattent la réalité.</b>
  « Arbitré » veut presque toujours dire que la <i>forme</i> de la règle vient de toi
  et que son <i>nombre</i> a été posé. Un seul bloc est arbitré au sens fort —
  verdicts écrits sur disque et rejoués par un test qui rougit si la règle bouge :
  <b>les règles de basse</b>. Les 24 autres reposent sur une citation datée dans un
  commentaire : une vraie preuve, mais rien ne casse si on change la règle.
  Et <b>une seule mesure sur douze est reproductible</b> depuis le dépôt — les autres
  scripts ont disparu.</p>
</div>

<h2>Un vrai morceau, accord par accord</h2>
<div class="bandeau">
  <button id="stop" type="button">&#9632; stop</button>
  <label class="opt"><input type="checkbox" id="slow"> ralenti 60&nbsp;%</label>
  <label class="opt"><input type="checkbox" id="loop"> boucle</label>
  <span id="etat">chargement de l'audio&hellip;</span>
</div>
<p class="sub">{e(tr['titre'] or tr['key'])} &middot; {e(tr['tonalite'])} &middot;
les {len(ac)} accords des 16 premières mesures, rejoués par la chaîne réelle.</p>
{"".join(trace_carte(r) for r in ac)}

<h2>Les probabilités, image par image</h2>
<p class="sub">Mesures 2 et 3 &middot; {heat_n} images à 23&nbsp;ms &middot;
les {heat_k} accords qui dépassent 4&nbsp;% une fois ; les {heat_reste} autres pèsent
{heat_pct} en moyenne.</p>

<div class="card">
  <div class="hwrap"><canvas id="heat"></canvas></div>
  <div class="hlab" id="hread">survole ou touche la carte</div>
  <div class="acts" style="border-bottom:none;padding-bottom:0">
    <button class="pl" data-t0="{heat_t0:.3f}" data-t1="{heat_t1:.3f}" type="button">&#9658; les deux mesures</button>
    <button class="pl" data-t0="{heat_b2:.3f}" data-t1="{heat_b3:.3f}" type="button">&#9658; mesure 2</button>
    <button class="pl" data-t0="{heat_b3:.3f}" data-t1="{heat_t1:.3f}" type="button">&#9658; mesure 3</button>
  </div>
</div>

<div class="card">
  <div class="ith">Ce que la carte montre, et ce qu'elle ne peut pas montrer</div>
  <p class="itx">Mesure 2, le modèle hésite : <b>D♭ 33 %</b> contre <b>B♭m 30 %</b> sur le
  premier créneau. Mesure 3, il ne doute plus du tout : <b>E♭ 80 %</b>.</p>
  <p class="itx">Sauf que sur cette mesure 3, tu as tranché <b>A♭^7</b> à l'oreille &mdash;
  et <code>Ab</code> n'est qu'à 7,5&nbsp;%. Le modèle n'a pas tort de peu : un
  <b>A♭^9 sans tierce, c'est un triton E♭ posé sur un A♭</b>. Ce qui sonne au-dessus
  est littéralement un accord de E♭. <b>Aucune quantité de probabilités d'accord ne
  peut les distinguer</b> &mdash; seule la basse le peut.</p>
  <p class="itx">Et la basse le dit : à l'attaque de cette mesure, la lecture donne
  <b>A♭ à 47&nbsp;%</b>, plus fort que E♭. Mais la règle de basse cherche d'abord la
  fondamentale de l'accord ÉCRIT, trouve E♭ au-dessus du plancher, et conclut
  « pas de slash ». <b>La règle de basse ne peut être juste que si l'accord l'est</b> :
  ici l'étiquette fausse a fait taire le seul témoin qui savait.</p>
  <div class="acts ill">
    <button class="pl" data-t0="{heat_b3:.3f}" data-t1="{heat_b3plus:.3f}" type="button">&#9658; l'attaque de la mesure 3</button>
    <button class="pl" data-t0="{heat_b3:.3f}" data-t1="{heat_t1:.3f}" type="button">&#9658; la mesure 3 entière</button>
  </div>
</div>

<h2>La chaîne, étage par étage</h2>
<p class="sub" style="margin-bottom:12px">Pour chaque nombre : d'où il vient.</p>
{"".join(etage(i, n, q, cs) for i, (n, q, cs) in enumerate(CHAINE, 1))}

<h2>Ce que tes arbitrages règlent déjà</h2>
<div class="card">
  <p>Sur la trace ci-dessus, deux des trois slashes de Ready sont posés sur une lecture
  de basse <b>sous le plancher de 30 %</b> — le plancher ne s'applique qu'à la recherche
  de la fondamentale, jamais au slash écrit. Ça ressemble à un oubli.</p>
  <p><b>Ce n'en est pas un, et c'est ta donnée qui le dit.</b> Au 2ᵉ tour d'arbitrage,
  une version de test retirait les slashes posés sur une lecture faible :</p>
  <div class="wrap"><table>
   <tr><td class="cst">r16</td><td>Let It Be mes. 36</td><td class="val">F/G → F</td>
       <td class="val">27,3 %</td><td><span class="b b-warn">ko</span> garde le slash</td></tr>
   <tr><td class="cst">r17</td><td>Ready mes. 2</td><td class="val">D♭/E♭ → D♭</td>
       <td class="val">24,7 %</td><td><span class="b b-warn">ko</span> garde le slash</td></tr>
   <tr><td class="cst">r19</td><td>Stand By Me mes. 18</td><td class="val">E/G♭ → E</td>
       <td class="val">27,9 %</td><td><span class="b b-warn">ko</span> garde le slash</td></tr>
   <tr><td class="cst">r20</td><td>This Love mes. 7</td><td class="val">F-7/G♭ → F-7</td>
       <td class="val">32,4 %</td><td><span class="b b-mark">ok</span> retire-le</td></tr>
  </table></div>
  <p style="margin-top:10px">Le seul retrait que tu valides est <b>le plus confiant des
  quatre</b>, et il se justifie par l'intervalle (G♭ sur F = une b9, écartée), pas par la
  confiance. Mettre un plancher sur le slash casserait trois de tes verdicts.
  <b>C'est déjà la bonne règle</b> — et le <code>D♭/E♭</code> de la mesure 2 ci-dessus
  est littéralement le cas r17.</p>
</div>

<h2>Ce qui est mort sans que personne le sache</h2>
<div class="card">
  <p><b>Le re-calage harmonique de la phase</b> ne se déclenche sur aucun des 44 morceaux —
  mais pas pour la raison écrite dans le code. C'est le seuil <code>PHASE_BEAT0_MAX</code>
  qui bloque, pas le vote : les accords se répartissent structurellement entre temps fort
  et mi-mesure, donc la part du temps 0 ne descend jamais sous 0,24 face à un plafond de
  0,15. Il était vivant jusqu'au 15/09 à 15h24 — il est mort en <b>effet de bord</b> du
  retrait de la recherche de latence, pas sur décision.</p>
  <p><b>Cinq constantes de <code>settings.py</code> ne sont lues nulle part</b>
  (<code>sections</code>, <code>merge</code>, <code>fold_loop</code>, <code>fold_gate</code>,
  <code>fold_transpose</code>) : elles ressemblent à des réglages et documentent des lois
  codées en dur ailleurs. Les changer ne fait rien.</p>
  <p><b>L'ancienne grille rigide a emporté un garde-fou.</b> <code>grille_rigide</code>
  vérifiait que la grille rigide colle vraiment aux temps du traceur — c'est ce qui
  empêchait deux morceaux de recevoir « une grille qui déplace la musique au lieu de la
  décrire ». Le chemin vivant n'a pas d'équivalent.</p>
  <p><b>Et l'écran Jam était mort depuis hier</b> : un troisième appelant de
  <code>redecode</code> n'avait pas suivi le changement de signature du 15/09. Corrigé,
  avec le test qui lit les trois sites d'appel.</p>
</div>

<footer>
  inventaire : <code>scratchpad/decisions_inventory.md</code> (lecture du code, pas des docstrings)
  &middot; tracé : <code>scratchpad/trace_chords.py</code>, chaîne réelle rejouée<br>
  constantes du tracé : pénalité musx {tr['constantes']['penalite_musx']:.0f} &middot;
  coûts {tr['constantes']['cout_transition']} &middot;
  plancher basse {tr['constantes']['plancher_basse']:.0f} % &middot;
  intervalles jouables {tr['constantes']['intervalles_jouables']}
</footer>

<script>
// Le fichier est chargé en entier avant de jouer : le lecteur média de Safari
// ne bufferise pas les extraits servis en 206 (piège iOS connu du projet).
// Pas de boucle rAF non plus — sur iOS elle empêche le moteur audio de démarrer.
const AUDIO = {audio_js};
let audio = null, playing = null, stopAt = null;
const etat = document.getElementById('etat');
fetch(AUDIO).then(r => r.blob()).then(b => {{
  audio = new Audio(URL.createObjectURL(b));
  audio.preservesPitch = true;
  audio.addEventListener('timeupdate', () => {{
    if (stopAt == null) return;
    if (audio.currentTime >= stopAt) {{
      if (document.getElementById('loop').checked && playing) {{
        audio.currentTime = parseFloat(playing.dataset.t0); return;
      }}
      audio.pause(); if (playing) playing.classList.remove('playing');
      playing = null; stopAt = null;
    }}
  }});
  etat.textContent = 'audio prêt';
}}).catch(() => {{ etat.textContent = 'audio indisponible'; }});

function jouer(btn){{
  if (!audio) return;
  const t0 = parseFloat(btn.dataset.t0), t1 = parseFloat(btn.dataset.t1);
  if (playing) playing.classList.remove('playing');
  if (playing === btn) {{ audio.pause(); playing = null; stopAt = null; return; }}
  audio.playbackRate = document.getElementById('slow').checked ? 0.6 : 1;
  stopAt = t1;
  audio.currentTime = Math.max(0, t0);
  audio.play().catch(() => {{}});
  btn.classList.add('playing'); playing = btn;
  etat.textContent = t0.toFixed(2).replace('.', ',') + ' s \u2192 ' + t1.toFixed(2).replace('.', ',') + ' s';
}}
document.querySelectorAll('button.pl').forEach(b => b.addEventListener('click', () => jouer(b)));

// ── la carte de chaleur ────────────────────────────────────────────────────
// Rampe SÉQUENTIELLE à teinte unique, clair -> foncé : une probabilité est une
// magnitude, pas une catégorie — donc jamais d'arc-en-ciel. Deux jeux d'arrêts,
// un par thème, choisis sur le fond réellement peint.
const H = {heat_json};
(function(){{
  const cv = document.getElementById('heat'), read = document.getElementById('hread');
  if (!cv || !H.lignes || !H.lignes.length) return;
  const sombre = () => {{
    const t = document.documentElement.getAttribute('data-theme');
    if (t === 'dark') return true;
    if (t === 'light') return false;
    return matchMedia('(prefers-color-scheme: dark)').matches;
  }};
  const STOPS_CLAIR = [[251,247,238],[239,223,197],[222,146,81],[180,99,42],[90,51,21]];
  const STOPS_SOMBRE = [[27,24,18],[59,44,27],[180,99,42],[222,146,81],[243,210,172]];
  const LIGNES = H.lignes.length, COLS = H.lignes[0].p.length;
  const LAB_W = 62, TOP = 16, ROW = 17, PAD_B = 18;

  function couleur(v, stops){{
    const x = Math.max(0, Math.min(1, Math.pow(v, 0.55)));   // gamma : les petites
    const n = stops.length - 1, i = Math.min(n - 1, Math.floor(x * n));
    const f = x * n - i, a = stops[i], b = stops[i + 1];
    return `rgb(${{Math.round(a[0]+(b[0]-a[0])*f)}},${{Math.round(a[1]+(b[1]-a[1])*f)}},${{Math.round(a[2]+(b[2]-a[2])*f)}})`;
  }}

  let cellW = 0;
  function dessine(){{
    const dark = sombre();
    const stops = dark ? STOPS_SOMBRE : STOPS_CLAIR;
    const encre = dark ? '#A89D89' : '#6F6555';
    const trait = dark ? '#4A4130' : '#C9BE9F';
    const dispW = Math.max(660, cv.parentNode.clientWidth);
    const W = dispW, Hh = TOP + LIGNES * ROW + PAD_B;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    cv.width = W * dpr; cv.height = Hh * dpr;
    cv.style.width = W + 'px'; cv.style.height = Hh + 'px';
    const g = cv.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, Hh);
    cellW = (W - LAB_W - 6) / COLS;

    g.font = '10px "IBM Plex Mono", monospace';
    g.textBaseline = 'middle';
    H.lignes.forEach((l, r) => {{
      const y = TOP + r * ROW;
      for (let c = 0; c < COLS; c++) {{
        g.fillStyle = couleur(l.p[c], stops);
        g.fillRect(LAB_W + c * cellW, y, Math.ceil(cellW) + 0.5, ROW - 1.5);
      }}
      g.fillStyle = encre; g.textAlign = 'right';
      g.fillText(l.lab, LAB_W - 7, y + (ROW - 1.5) / 2);
    }});

    // les barres de mesure, et les frontières d'accord écrites
    const x = t => LAB_W + (t - H.t0) / H.dt * cellW;
    g.textAlign = 'center';
    (H.barres || []).forEach((t, i) => {{
      g.strokeStyle = trait; g.lineWidth = 2; g.beginPath();
      g.moveTo(x(t), TOP - 3); g.lineTo(x(t), TOP + LIGNES * ROW); g.stroke();
      g.fillStyle = encre;
      g.fillText('mes. ' + (i + 2), Math.min(W - 24, x(t) + 24), TOP - 8);
    }});
    (H.ecrits || []).forEach(c => {{
      g.strokeStyle = dark ? '#E1837B' : '#B23A33'; g.lineWidth = 1.5;
      g.setLineDash([3, 3]); g.beginPath();
      g.moveTo(x(c.t0), TOP); g.lineTo(x(c.t0), TOP + LIGNES * ROW); g.stroke();
      g.setLineDash([]);
      g.fillStyle = dark ? '#E1837B' : '#B23A33';
      g.textAlign = 'left';
      g.fillText(c.lab, x(c.t0) + 4, TOP + LIGNES * ROW + 9);
    }});
  }}

  function lire(ev){{
    const rect = cv.getBoundingClientRect();
    const px = (ev.touches ? ev.touches[0].clientX : ev.clientX) - rect.left;
    const py = (ev.touches ? ev.touches[0].clientY : ev.clientY) - rect.top;
    const c = Math.floor((px - LAB_W) / cellW), r = Math.floor((py - TOP) / ROW);
    if (c < 0 || c >= COLS || r < 0 || r >= LIGNES) {{ read.textContent = 'survole ou touche la carte'; return; }}
    const t = H.t0 + c * H.dt, v = H.lignes[r].p[c];
    read.innerHTML = `<b>${{H.lignes[r].lab}}</b> &middot; ${{(v*100).toFixed(1)}} % &middot; ${{t.toFixed(2).replace('.', ',')}} s`;
  }}
  cv.addEventListener('mousemove', lire);
  cv.addEventListener('touchstart', e => {{ lire(e); }}, {{passive: true}});
  cv.addEventListener('touchmove', e => {{ lire(e); }}, {{passive: true}});
  cv.addEventListener('mouseleave', () => read.textContent = 'survole ou touche la carte');
  dessine();
  addEventListener('resize', dessine);
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', dessine);
}})();
document.getElementById('stop').addEventListener('click', () => {{
  if (!audio) return;
  audio.pause(); if (playing) playing.classList.remove('playing');
  playing = null; stopAt = null; etat.textContent = 'audio prêt';
}});
document.getElementById('slow').addEventListener('change', e => {{
  if (audio) audio.playbackRate = e.target.checked ? 0.6 : 1;
}});
</script>
"""


def main() -> int:
    tr = json.loads(TRACE.read_text(encoding="utf-8"))
    tr["heat"] = json.loads(HEAT.read_text(encoding="utf-8")) if HEAT.exists() else None
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page(tr), encoding="utf-8")
    print(f"→ {OUT}  ({len(tr['accords'])} accords tracés)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
