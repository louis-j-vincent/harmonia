"""Ancres par VOTE DE SIGNAUX : peu de frontières, presque jamais fausses.

    .venv/bin/python scripts/ancres.py --table          # le réglage, sur les 18
    .venv/bin/python scripts/ancres.py [<stem> ...]     # -> /plots/ancres.html

COLLISION DE NOMS, RÉSOLUE (2026-08-14). Ce fichier a d'abord écrasé un autre
`ancres.py` — les ancres par diagonales de voix, d'une session concurrente. Rien
n'a été perdu (c'était committé, 32a88a4) et cette session-là a repris son code
sous `scripts/zones_voix.py`, en laissant le nom ici. Les deux sont
COMPLÉMENTAIRES : là-bas une ancre est une REPRISE claire du chant, qui fixe deux
sections d'un coup et donne leur ÉTENDUE ; ici c'est une FRONTIÈRE sur laquelle
plusieurs signaux indépendants tombent d'accord, ce qui donne la PHASE dont les
zones ont besoin. En aval, même règle pour les deux : `mots4` n'a pas le droit de
traverser une ancre.

Louis, 2026-08-14 :

  « Je veux un outil qui trouve QUELQUES frontières, pas besoin de les trouver
    toutes, juste les plus évidentes → je veux qu'il y ait très peu de faux
    positifs. Ces frontières servent à trancher lorsqu'il y a une ambiguïté sur
    où on coupe ; une fois qu'on est bons on peut utiliser nos critères de
    phrases de 4 lettres, avec la règle qu'on ne peut pas couper ces frontières.
    Le vote de signaux je suis chaud. Évite de prendre des pics de signaux qui
    crient au loup tout le temps. »

DONC LA MÉTRIQUE N'EST PAS LE RAPPEL, C'EST LA PRÉCISION. Une ancre fausse
FORCE une coupure au mauvais endroit et casse tout ce qui vient après ; une
ancre manquée ne coûte rien, les critères de phrases s'en chargeront. Tout est
réglé là-dessus.

LES TROIS RÈGLES, dans l'ordre où elles s'appliquent.

1. **Un critère ne parle que de ses moments EXCEPTIONNELS.** Pas ses k plus
   hauts pics — ses pics à plus de `Z` écarts absolus médians de sa propre
   médiane. Un critère dont la courbe ondule sans rien affirmer n'en a aucun, et
   se tait. C'est la règle « pas les signaux qui crient au loup », prise à la
   lettre : on ne compare pas les critères entre eux, on demande à chacun s'il
   est surpris.

2. **Et s'il est surpris trop souvent, il est disqualifié.** Au-delà d'un pic
   exceptionnel toutes les `1/DENSITE` mesures, le critère est retiré POUR CE
   MORCEAU. Un critère bavard n'est pas informatif, il est bruyant : sa présence
   dans le vote suffirait à faire élire n'importe quelle mesure.

3. **Le vote compte des SIGNAUX, pas des critères.** Les six variantes
   d'harmonie ne pèsent pas six fois ; chaque signal (voix, harmonie, basse,
   accords, batterie, timbre, intensité) a une voix et une seule, celle de son
   critère le plus surpris. Sans ça, le vote mesurerait le nombre de variantes
   qu'on a écrites par signal, pas l'accord entre sources indépendantes.

LE DÉCALAGE, la question de Louis : « comment mettre ça en pratique quand on ne
connaît pas le vrai début de la section ? Ma solution : regarder s'il y a un
décalage constant avec une majorité de pics qui sont d'accord. » C'est
exactement ce qui est fait, et sans jamais regarder les frontières : autour de
chaque ancre, les signaux qui votent ne piquent pas tous à la même fraction de
mesure (le fill de batterie est en avance, la voix est parfois en retard). On
prend la MÉDIANE de leurs positions comme position de l'ancre, puis la médiane
sur toutes les ancres de l'écart à la mesure la plus proche : c'est le décalage
du morceau. Il est affiché ; il n'est appliqué que s'il dépasse `DECAL_MIN`.

CE QUE ÇA NE FAIT PAS. Aucune section n'est proposée : une ancre dit « ici, ça
change, et sept signaux ne peuvent pas tous se tromper », rien de plus. Le
découpage en sections reste au travail des phrases. Et la grille de 4 mesures
n'entre PAS dans le détecteur — elle est vérifiée à part (`docs/known_issues.md`,
2026-08-14 : 82 % en RELATIF, 44 % en absolu), et c'est aux phrases de s'en
servir, en repartant de zéro à chaque ancre.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
sys.path.insert(0, str(HERE / "scratchpad"))

CACHE = HERE / "scratchpad" / "critere_cache"
OUT = HERE / "docs" / "plots" / "ancres.html"

Z = 3.5          # « exceptionnel » = à Z écarts absolus médians de sa médiane
DENSITE = 1 / 6  # au-delà d'un pic exceptionnel toutes les 6 mesures : disqualifié
TOL = 1.0        # deux pics à ≤ 1 mesure parlent du même endroit
K = 5            # nombre de SIGNAUX distincts qu'il faut pour faire une ancre
MINI = 2         # une ancre seule n'est corroborée par rien : le morceau se tait


def courbes(stem, rebuild=False):
    """{(signal, nom, res): courbe brute} — les 39 critères, en cache."""
    p = CACHE / f"{stem}_courbes.pkl"
    if p.exists() and not rebuild:
        with p.open("rb") as f:
            return pickle.load(f)
    from criteres_sections import signaux, criteres
    S = signaux(stem)
    C = criteres(S)
    d = {"n": S["n"], "grid": [float(x) for x in S["grid"]],
         "c": [(sig, nom, glose, res, np.asarray(c, float)) for sig, nom, glose, res, c in C]}
    with p.open("wb") as f:
        pickle.dump(d, f)
    return d


def pics_rares(c, res, n, z=Z, densite=DENSITE):
    """[(position en mesures, surprise)] — les moments exceptionnels d'un critère.

    Retourne une liste VIDE si le critère est surpris plus d'une fois toutes les
    `1/densite` mesures : il crie au loup, on ne l'écoute pas sur ce morceau.
    """
    from scipy.signal import find_peaks
    c = np.nan_to_num(np.asarray(c, float))
    med = float(np.median(c))
    mad = float(np.median(np.abs(c - med))) * 1.4826
    if mad <= 1e-12:
        return []
    idx, _ = find_peaks(c, prominence=mad)
    out = [(i / res, (c[i] - med) / mad) for i in idx if (c[i] - med) / mad >= z]
    if len(out) > densite * n:
        return []
    return out


def vote(stem, z=Z, tol=TOL, densite=DENSITE):
    """(vote par mesure, qui a voté, où exactement, les critères muets, n).

    `V[b]` = le nombre de SIGNAUX distincts dont un critère est exceptionnel à
    moins de `tol` mesures de b. `POS[b]` = où chacun de ces signaux pique
    exactement (en mesures, fractionnaire) — c'est de là que sort la position de
    l'ancre, et pas de `b`.

    Attention, c'est le piège de ce détecteur : avec `tol=1`, un pic fait voter
    TROIS mesures consécutives. `V` est donc un plateau, et choisir `b` au
    milieu du plateau est un tirage au sort qui coûte une mesure sur deux
    (mesuré : 10 % d'ancres exactes en prenant `b`, 51 % en prenant la médiane
    des `POS`). La position d'une ancre est celle où les signaux piquent, jamais
    celle de la case qui les compte.
    """
    d = courbes(stem)
    n = d["n"]
    par_signal: dict[str, list] = {}
    muets = []
    for sig, nom, _g, res, c in d["c"]:
        pk = pics_rares(c, res, n, z=z, densite=densite)
        if not pk:
            muets.append(f"{sig} · {nom}")
            continue
        par_signal.setdefault(sig, []).extend(pk)

    V = np.zeros(n, int)
    QUI: list[list[str]] = [[] for _ in range(n)]
    POS: list[list[float]] = [[] for _ in range(n)]
    for sig, pk in par_signal.items():
        for b in range(n):
            proches = [(p, s) for p, s in pk if abs(p - b) <= tol]
            if proches:
                V[b] += 1
                QUI[b].append(sig)
                POS[b].append(max(proches, key=lambda x: x[1])[0])
    return V, QUI, POS, muets, n


def _phase(A, poids=(1.0, .55, .25)):
    """La phase du réseau de 4 mesures que les ancres elles-mêmes désignent.

    Vérifié sur les 20 annotations (known_issues 2026-08-14) : l'écart entre
    deux frontières consécutives est un multiple de 4 dans 82 % des cas, mais la
    phase ABSOLUE est propre à chaque morceau (44 % seulement à phase 0). On ne
    peut donc pas imposer « multiple de 4 » ; on peut demander aux ancres d'être
    d'accord entre elles, ce qui ne coûte aucune annotation.
    """
    def d(b, p):
        x = (p - b) % 4
        return x if x <= 2 else x - 4
    return max(range(4), key=lambda p: sum(v * poids[abs(d(b, p))] for b, v, *_ in A))


def ancres(stem, k=K, z=Z, tol=TOL, densite=DENSITE, ecart=3.0, reseau=True):
    """[(mesure, nb de signaux, qui, dispersion)] — les ancres du morceau.

    Trois passes : (1) les mesures où au moins `k` signaux distincts sont
    exceptionnels, prises de la plus soutenue à la moins soutenue et espacées
    d'au moins `ecart` mesures ; (2) la position de chacune = la médiane de là
    où ses signaux piquent vraiment ; (3) si `reseau`, on ne garde que celles
    qui tombent sur la phase de 4 mesures que la majorité désigne — c'est la
    solution de Louis (« un décalage constant avec une majorité de pics qui sont
    d'accord ») appliquée aux ancres entre elles, sans jamais lire une frontière.

    Mesuré sur les 18 morceaux validés : 2,2 ancres par morceau, 75 % exactes.
    """
    V, QUI, POS, muets, n = vote(stem, z=z, tol=tol, densite=densite)
    cand = sorted([b for b in range(1, n) if V[b] >= k],
                  key=lambda b: (-V[b], float(np.median(np.abs(
                      np.array(POS[b]) - np.median(POS[b]))))))
    gardees: list[int] = []
    for b in cand:
        if all(abs(b - g) >= ecart for g in gardees):
            gardees.append(b)
    A = []
    for b in sorted(gardees):
        P = np.array(POS[b])
        pos = int(round(float(np.median(P))))
        disp = float(np.median(np.abs(P - np.median(P))))
        if 0 < pos < n:
            A.append((pos, int(V[b]), sorted(set(QUI[b])), disp))
    # deux ancres peuvent se retrouver sur la même mesure après recalage
    vus: dict[int, tuple] = {}
    for a in A:
        if a[0] not in vus or a[1] > vus[a[0]][1]:
            vus[a[0]] = a
    A = [vus[p] for p in sorted(vus)]
    ph = _phase(A) if (reseau and A) else None
    if ph is not None:
        A = [a for a in A if a[0] % 4 == ph]
    # une ancre toute seule n'est corroborée par rien d'autre qu'elle-même, et
    # la phase du réseau qu'elle désigne est alors un tirage au sort : sur les
    # 18 morceaux, les cas à une seule ancre survivante sont faux 3 fois sur 4.
    if reseau and len(A) < MINI:
        A = []
    return A, ph, muets, n, V


def gt_de(stem):
    p = HERE / "harmonia_min" / "state" / "sections" / f"{stem}.json"
    if not p.exists():
        return []
    d = json.loads(p.read_text())
    return sorted({int(s["b0"]) for s in d["sections"] if 0 < int(s["b0"]) < d["n"]})


VALIDES = None


def liste_validee():
    global VALIDES
    if VALIDES is None:
        P = HERE / "harmonia_min" / "state" / "sections"
        A = HERE / "docs" / "audio"
        VALIDES = sorted(f.stem for f in P.glob("*.json")
                         if json.loads(f.read_text()).get("validated")
                         and (A / f"{f.stem}.m4a").exists())
    return VALIDES


def table():
    """Le réglage : précision et nombre d'ancres, sur les 18 morceaux validés.

    Ce qu'on lit : à `k` signaux il reste peu d'ancres et presque aucune n'est
    fausse ; en descendant d'un cran on double le nombre d'ancres et on paie en
    faux positifs. C'est le seul arbitrage de la page.
    """
    stems = liste_validee()
    print(f"{'':>3} {'z':>4} {'k':>2} | {'ancres':>6} {'justes':>7} {'précision':>10} "
          f"| {'à ±0':>5} {'±1':>4} | morceaux sans aucune ancre")
    for z in (2.5, 3.0, 3.5):
        for k in (3, 4, 5):
            tot = bon = ex = pr = 0
            vides = 0
            for s in stems:
                A, _d, _m, _n, _V = ancres(s, k=k, z=z)
                gt = gt_de(s)
                if not A:
                    vides += 1
                for b, _v, _q, _f in A:
                    tot += 1
                    dd = min([abs(b - g) for g in gt], default=99)
                    if dd <= 1:
                        bon += 1
                    if dd == 0:
                        ex += 1
                    elif dd == 1:
                        pr += 1
            p = 100 * bon / max(1, tot)
            print(f"{'':>3} {z:>4} {k:>2} | {tot:>6} {bon:>7} {p:>9.0f}% "
                  f"| {ex:>5} {pr:>4} | {vides}/{len(stems)}")


def par_morceau(k=K, z=Z):
    stems = liste_validee()
    print(f"{'morceau':<34}{'anc':>4}{'justes':>7}{'prec':>6}  {'ph':>3}  ancres "
          f"(mesure ·nb signaux, ✗ = fausse, ~ = à une mesure)")
    T = B = E = 0
    for s in stems:
        A, ph, _m, _n, _V = ancres(s, k=k, z=z)
        gt = gt_de(s)

        def ec(b):
            return min([abs(b - g) for g in gt], default=99)
        bon = sum(1 for b, *_ in A if ec(b) <= 1)
        T += len(A); B += bon; E += sum(1 for b, *_ in A if ec(b) == 0)
        det = " ".join(f"{b + 1}·{v}" + ("" if ec(b) == 0 else ("~" if ec(b) == 1 else "✗"))
                       for b, v, _q, _d in A)
        pc = f"{100 * bon / len(A):.0f}%" if A else "—"
        print(f"{s[:33]:<34}{len(A):>4}{bon:>7}{pc:>6}  {str(ph):>3}  {det}")
    print(f"\nTOTAL {E}/{T} = {100 * E / max(1, T):.0f}% d'ancres EXACTES, "
          f"{B}/{T} = {100 * B / max(1, T):.0f}% à une mesure près, "
          f"{T / len(stems):.1f} ancres par morceau")


# ── la page ─────────────────────────────────────────────────────────────────

ARB = HERE / "scratchpad" / "ia_ancres"      # verdicts d'un LLM, s'il y en a
SIGNAUX = ["voix", "harmonie", "basse", "accords", "rythme", "timbre", "intensité"]


def _verdicts(stem):
    """({mesure 1-based: verdict}, [ajouts], note) — l'arbitrage d'un LLM.

    `scratchpad/ia_ancres/<stem>.json`. Le modèle ne voit QUE le résumé texte
    (`resume_texte.py`) et la liste des ancres proposées par les signaux : il
    n'entend pas l'audio et ne voit aucune frontière validée. Il peut garder,
    rejeter, ou ajouter — et doit dire pourquoi en une phrase, pour que Louis
    puisse juger le RAISONNEMENT et pas seulement le résultat.
    """
    p = ARB / f"{stem}.json"
    if not p.exists():
        return {}, [], ""
    d = json.loads(p.read_text())
    return ({int(v["mesure"]): v for v in d.get("verdicts", [])},
            d.get("ajouts", []), d.get("note", ""))


def page(stems):
    from criteres_sections import TEINTE, INK, GT_LINE, bande_png
    from ssm_zoo import SONGS
    titres = dict(SONGS)
    body = []
    for stem in stems:
        A, ph, muets, n, V = ancres(stem)
        d = courbes(stem)
        gt = gt_de(stem)
        grid = d["grid"]
        verd, ajouts, note = _verdicts(stem)

        # où chaque signal est exceptionnel — une ligne par signal
        par_sig: dict[str, list] = {}
        for sig, nom, _g, res, c in d["c"]:
            for p, s in pics_rares(c, res, n):
                par_sig.setdefault(sig, []).append((p, s, nom))

        lignes = []
        for sig in SIGNAUX:
            pk = par_sig.get(sig, [])
            t = "".join(
                f'<i class=p style="left:{p / n * 100:.4f}%" '
                f'title="{sig} · {nom} — surprise {s:.1f}"></i>' for p, s, nom in pk)
            etat = "" if pk else " muet"
            lignes.append(
                f'<div class=lane><div class=lab style="color:{TEINTE[sig]}">{sig}'
                f'<small>{len(pk) or "—"}</small></div>'
                f'<div class="strip{etat}">{t}</div></div>')

        vpng = bande_png(np.clip(V / 7.0, 0, 1))
        gtl = "".join(f'<i class=gt style="left:{g / n * 100:.4f}%"></i>' for g in gt)
        anc = ""
        for b, v, q, _dsp in A:
            ko = verd.get(b + 1, {}).get("garde") is False
            anc += (f'<i class="a{" ko" if ko else ""}" style="left:{b / n * 100:.4f}%" '
                    f'title="mesure {b + 1} · {v} signaux : {", ".join(q)}"></i>')
        for aj in ajouts:
            m = int(aj["mesure"]) - 1
            anc += (f'<i class="a ia" style="left:{m / n * 100:.4f}%" '
                    f'title="ajout du LLM · mesure {m + 1} : {aj.get("raison", "")}"></i>')
        blocs = ""
        p = HERE / "harmonia_min" / "state" / "sections" / f"{stem}.json"
        if p.exists():
            for s in json.loads(p.read_text())["sections"]:
                w = (s["b1"] - s["b0"] + 1) / n * 100
                blocs += (f'<i style="left:{s["b0"] / n * 100:.4f}%;width:{w:.4f}%">'
                          f'{s.get("label", "")}</i>')
        pas = 4 if n <= 60 else (8 if n <= 140 else 16)
        regle = "".join(f'<i style="left:{i / n * 100:.4f}%">{i + 1}</i>'
                        for i in range(0, n, pas))

        btns = ""
        for b, v, q, _dsp in A:
            e = min([abs(b - g) for g in gt], default=99)
            cls = "ok" if e == 0 else ("pres" if e <= 1 else "ko")
            ver = verd.get(b + 1)
            rai = f'<em>{ver["raison"]}</em>' if ver and ver.get("raison") else ""
            mort = " barre" if ver and ver.get("garde") is False else ""
            btns += (f'<button class="an {cls}{mort}" data-b="{b}">mes. <b>{b + 1}</b>'
                     f'<small>{v} signaux · {", ".join(q)}</small>{rai}</button>')
        for aj in ajouts:
            m = int(aj["mesure"]) - 1
            e = min([abs(m - g) for g in gt], default=99)
            cls = "ok" if e == 0 else ("pres" if e <= 1 else "ko")
            btns += (f'<button class="an ia {cls}" data-b="{m}">mes. <b>{m + 1}</b>'
                     f'<small>ajout du LLM</small><em>{aj.get("raison", "")}</em></button>')
        if not A:
            btns = ('<div class=rien>Aucune ancre : moins de deux mesures rassemblent '
                    f'{K} signaux exceptionnels, ou elles ne tombent pas sur la même '
                    'phase de 4 mesures. Le morceau se tait plutôt que de deviner.</div>')

        body.append(
            f'<section data-grid="{json.dumps(grid)}" data-n="{n}">'
            f'<div class=hd><h2>{titres.get(stem, stem.replace("_", " "))}</h2>'
            f'<span class=sub>{n} mesures · {len(A)} ancre(s) · '
            f'{len(gt)} frontières validées · {len(muets)}/39 critères muets</span></div>'
            f'<div class=bar><button class=pp>▶</button><span class=pos>mes. 1</span>'
            f'<span class=hint>clique la bande pour écouter · clique une ancre pour '
            f'l\'entendre en contexte (2 mesures avant → 2 après)</span></div>'
            f'<div class=stack>'
            f'<div class=lane><div class=lab></div><div class=secs>{blocs}</div></div>'
            f'<div class=lane><div class=lab></div><div class=rule>{regle}</div></div>'
            f'<div class=lane><div class=lab><b>vote</b></div>'
            f'<div class=strip><img src="data:image/png;base64,{vpng}" alt="">{anc}</div></div>'
            f'{"".join(lignes)}'
            f'<div class=ov>{gtl}<div class=cur></div></div></div>'
            f'<div class=lane2>{btns}</div>'
            + (f'<div class=note><b>Verdict du LLM</b> — {note}</div>' if note else '')
            + f'<audio preload=metadata playsinline src="../audio/{stem}.m4a"></audio>'
              f'</section>')
        print(f"  ok {titres.get(stem, stem)} — {len(A)} ancre(s), phase {ph}")

    css = f"""
body{{margin:0;background:#e7e0d0;font:15px/1.55 -apple-system,system-ui,sans-serif;color:{INK}}}
.wrap{{max-width:1240px;margin:0 auto;padding:22px 14px 80px}}
h1{{font:italic 600 26px Georgia,serif;margin:0 0 6px}}
.lede{{color:#6f6857;font-size:13.5px;margin-bottom:16px;max-width:980px}}
.lede b{{color:{INK}}}
section{{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;
  padding:10px 14px 14px;margin-bottom:14px;--lab:96px}}
.hd{{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap}}
h2{{font:700 18px system-ui;margin:0;color:#8a2b2b}}
.sub{{font:500 12px system-ui;color:#8a8371}}
.bar{{display:flex;align-items:center;gap:14px;margin:8px 0 10px;flex-wrap:wrap}}
.pp{{width:34px;height:34px;border-radius:50%;border:1px solid #d8cfb4;
  background:#f7f3e9;font-size:13px;cursor:pointer;flex:none}}
.pos{{font:600 12px ui-monospace,monospace;min-width:120px}}
.hint{{font:500 11.5px system-ui;color:#a89f8c}}
.stack{{position:relative}}
.lane{{display:grid;grid-template-columns:var(--lab) 1fr;align-items:center;height:20px}}
.lab{{font:600 11px system-ui;padding-right:8px;text-align:right;white-space:nowrap}}
.lab small{{color:#b5ab94;font-weight:500;margin-left:5px}}
.strip{{position:relative;height:14px;border-bottom:1px solid #f2ecdd}}
.strip img{{width:100%;height:14px;display:block;border-radius:3px;
  image-rendering:pixelated}}
.strip.muet{{background:repeating-linear-gradient(45deg,#faf6ec,#faf6ec 4px,#f2ecdd 4px,#f2ecdd 8px)}}
.p{{position:absolute;top:2px;width:3px;height:10px;border-radius:1px;
  background:#5c8fa8;transform:translateX(-1.5px)}}
.a{{position:absolute;top:-4px;width:3px;height:22px;background:#e8a33d;
  transform:translateX(-1.5px);border-radius:1px;box-shadow:0 0 0 1px rgba(255,255,255,.8)}}
.a.ko{{background:#c9c1ab}}
.a.ia{{background:#1f6b6b;top:-6px;height:26px}}
button.an.ia{{border-style:dashed;border-color:#1f6b6b}}
button.an.barre b{{text-decoration:line-through;opacity:.55}}
.secs{{position:relative;height:18px}}
.secs i{{position:absolute;top:0;height:18px;font:700 10px system-ui;color:#8a6a4a;
  background:#f4ecd9;border-left:1px solid #e0d3b6;box-sizing:border-box;
  padding-left:3px;line-height:18px;overflow:hidden;border-radius:2px}}
.rule{{position:relative;height:13px}}
.rule i{{position:absolute;top:0;font:500 9.5px ui-monospace,monospace;color:#a89f8c;
  transform:translateX(-50%)}}
.ov{{position:absolute;left:var(--lab);right:0;top:0;bottom:0;cursor:crosshair;z-index:4}}
.gt{{position:absolute;top:0;bottom:0;width:1px;background:{GT_LINE};opacity:.75}}
.cur{{position:absolute;top:0;bottom:0;width:2px;background:#c1121f;display:none;
  box-shadow:0 0 0 1px rgba(255,255,255,.7)}}
.lane2{{display:flex;flex-wrap:wrap;gap:6px;margin-top:12px}}
button.an{{border:2px solid #e0d7c2;background:#f7f3e9;border-radius:9px;padding:5px 9px;
  cursor:pointer;font:600 12.5px system-ui;color:#4a4438;text-align:left}}
button.an small{{display:block;font:500 10px system-ui;color:#a89f8c}}
button.an em{{display:block;font:500 10.5px system-ui;color:#6f6857;max-width:230px;
  font-style:normal;margin-top:2px}}
button.an.ok{{border-color:#3f7a4f}} button.an.pres{{border-color:#d9a441}}
button.an.ko{{border-color:#b4472c}}
button.an.on{{background:#0d2437;color:#fff}} button.an.on small{{color:#9fb8c6}}
.rien{{font:500 12px system-ui;color:#8a8371;background:#faf6ec;border-radius:9px;
  padding:8px 10px}}
.note{{font:500 12px system-ui;color:#6f6857;margin-top:10px;background:#f4f7f4;
  border-left:3px solid #1f6b6b;padding:7px 10px;border-radius:0 8px 8px 0}}
audio{{display:none}}
"""
    js = """
document.querySelectorAll("section").forEach(function(sec){
  var au=sec.querySelector("audio"); if(!au) return;
  var G=JSON.parse(sec.dataset.grid), n=+sec.dataset.n;
  var cur=sec.querySelector(".cur"), ov=sec.querySelector(".ov");
  var pp=sec.querySelector(".pp"), pos=sec.querySelector(".pos"), timer=null, stop=null;
  function b2t(f){ var i=Math.max(0,Math.min(n-1,Math.floor(f)));
    return G[i]+(f-i)*(G[i+1]-G[i]); }
  function t2b(t){ if(t<=G[0])return 0; if(t>=G[n])return n;
    var lo=0,hi=n; while(hi-lo>1){var m=(lo+hi)>>1; G[m]<=t?lo=m:hi=m;}
    return lo+(t-G[lo])/(G[lo+1]-G[lo]); }
  function fmt(s){ return Math.floor(s/60)+":"+String(Math.floor(s%60)).padStart(2,"0"); }
  function draw(){ var f=t2b(au.currentTime); cur.style.display="block";
    cur.style.left="calc("+(100*f/n)+"% - 1px)";
    pos.textContent="mes. "+(Math.floor(f)+1)+" · "+fmt(au.currentTime); }
  au.addEventListener("timeupdate", draw);
  au.addEventListener("play", function(){
    document.querySelectorAll("audio").forEach(function(a){ if(a!==au) a.pause(); });
    pp.textContent="❚❚"; clearInterval(timer); timer=setInterval(draw,90); });
  au.addEventListener("pause", function(){ pp.textContent="▶"; clearInterval(timer); draw(); });
  pp.onclick=function(){ clearInterval(stop);
    au.paused ? au.play().catch(function(){}) : au.pause(); };
  function aller(t, jusqua){
    clearInterval(stop);
    var go=function(){ try{ au.currentTime=t; }catch(e){} draw(); };
    if(au.readyState>=1) go(); else au.addEventListener("loadedmetadata",go,{once:true});
    au.play().catch(function(){});
    if(jusqua!=null) stop=setInterval(function(){
      if(au.currentTime>=jusqua||au.paused){ clearInterval(stop); au.pause(); } },60);
  }
  ov.onclick=function(e){ var r=ov.getBoundingClientRect();
    aller(b2t(n*(e.clientX-r.left)/r.width), null); };
  sec.querySelectorAll("button.an").forEach(function(btn){
    btn.onclick=function(){
      sec.querySelectorAll("button.an").forEach(function(x){x.classList.remove("on")});
      btn.classList.add("on");
      var b=+btn.dataset.b;
      aller(b2t(Math.max(0,b-2)), b2t(Math.min(n,b+2)));
    };
  });
});
"""
    OUT.write_text(f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Les ancres</title><style>{css}</style></head><body><div class=wrap>
<h1>Les ancres — peu, mais sûres</h1>
<div class=lede>Une <b>ancre</b> est une mesure où <b>au moins {K} des 7 signaux</b>
sont simultanément <i>exceptionnels</i> — pas « ont un pic », mais « piquent à plus
de {Z} écarts de leur propre médiane ». Un critère qui s'étonne plus d'une fois
toutes les 6 mesures est retiré du vote : il crie au loup.<br><br>
<b>Les traits ambre</b> sont les ancres, <b>les traits rouges</b> tes frontières
validées, <b>les petits traits bleus</b> disent où chaque signal s'est étonné.
La bande <b>vote</b> est sombre là où beaucoup de signaux sont d'accord.
<b>Clique une ancre pour l'entendre</b> : deux mesures avant, deux après.<br><br>
Le contour des boutons dit la comparaison à tes frontières —
<b style="color:#3f7a4f">vert</b> = pile dessus,
<b style="color:#d9a441">ambre</b> = à une mesure,
<b style="color:#b4472c">rouge</b> = ailleurs. Sur les 18 morceaux validés :
<b>2,1 ancres par morceau, 78 % exactes</b>, et 6 morceaux où l'outil préfère se
taire.</div>
{''.join(body)}</div><script>{js}</script></body></html>""")
    print(f"wrote {OUT}  ({OUT.stat().st_size // 1024} KB)")
    print("   http://100.89.209.63:7772/plots/ancres.html")


if __name__ == "__main__":
    if "--table" in sys.argv:
        table()
    elif "--songs" in sys.argv:
        par_morceau()
    else:
        args = [a for a in sys.argv[1:] if not a.startswith("--")]
        page(args or liste_validee())
