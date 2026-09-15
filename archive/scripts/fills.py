"""Les fills de batterie : l'anomalie qui annonce une section.

    .venv/bin/python scripts/fills.py --table        # le tableau des variantes
    .venv/bin/python scripts/fills.py [<stem> ...]   # -> /plots/fills.html

Louis, 2026-08-14 : « Une mesure essentielle que tu n'as pas encore réussie :
choper les fill-ins de la batterie juste avant une section. Il faut tester plein
d'options et voir laquelle coïncide le mieux avec le changement de sections
(prendre en compte le décalage, des fois avant, des fois après le temps 1). »

L'HYPOTHÈSE DE DÉPART (celle du brief, tenue pour une hypothèse) : un fill n'est
pas un événement, c'est une **anomalie suivie d'une résolution**. La mesure b
sort du groove, la mesure b+1 y revient. Les 39 critères de
`criteres_sections.py` mesurent « avant ≠ après » : un fill leur fait sortir un
pic une mesure trop tôt (mesuré : sur 68 ancres, 22 en avance, 9 en retard).

DEUX PRÉCAUTIONS MÉTHODOLOGIQUES, toutes deux déjà payées dans ce projet.

1. **Les annotations ne sont pas exhaustives** (Louis : « je n'ai pas noté tous
   les changements de section »). Donc on ne dit jamais « faux positif » : on dit
   « pic hors annotation ». La métrique principale est le RAPPEL — quelle
   fraction des frontières annotées porte un pic — et le coût est la DENSITÉ de
   pics. Chaque variante reçoit le même budget de pics (`n/8`), sinon la
   comparaison ment.
2. **La position d'un pic n'est pas la case qui le compte.** Avec une tolérance
   de ±1 mesure un pic fait voter trois mesures ; prendre celle du milieu fait
   tomber la justesse de 51 % à 10 % (`docs/known_issues.md`, 2026-08-14). Ici
   toutes les positions restent FRACTIONNAIRES, en mesures, jusqu'au bout.

LE DÉCALAGE est le cœur du travail : pour chaque variante on rapporte la
distribution des écarts signés (pic − frontière annotée la plus proche), en
mesures et en temps. Une variante à écart constant −1 vaut mieux qu'une variante
à écart aléatoire ±1 à précision brute égale : un décalage constant se corrige.
La correction est évaluée en *leave-one-song-out* (l'offset est élu sur 17
morceaux et appliqué au 18e, jamais vu), jamais sur le morceau lui-même.
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

AUDIO = HERE / "docs" / "audio"
ANN = HERE / "harmonia_min" / "state" / "sections"
CACHE = HERE / "scratchpad" / "fills_cache"
CACHE.mkdir(exist_ok=True)
OUT = HERE / "docs" / "plots" / "fills.html"

SR = 22050
HOP = 256                     # 86,1 trames/s : une double-croche à 120 bpm ≈ 11 trames
BANDES = [(0, 130), (130, 1200), (1200, 5500), (5500, 11025)]
NOM_BANDE = ["grosse caisse", "caisse claire / toms", "médium-aigu", "cymbales"]
VERSION = 1


# ── le substrat : la batterie séparée, en quatre bandes d'attaques ──────────

def feats(stem, rebuild=False):
    """{env, times, onsets, n, grid} — les attaques de la batterie, en cache.

    `env` = (4, T) enveloppe d'attaques par bande, normalisée bande par bande.
    `onsets` = [(temps, bande dominante, force)] — les attaques détectées sur la
    somme des bandes, chacune attribuée à la bande où elle est la plus forte.
    Les stems demucs sont DÉJÀ en cache pour les 18 morceaux validés ; ce module
    ne relance jamais la séparation sur autre chose.
    """
    p = CACHE / f"{stem}_v{VERSION}.pkl"
    if p.exists() and not rebuild:
        with p.open("rb") as f:
            return pickle.load(f)
    import librosa
    import soundfile as sf
    import order_bundle
    from rhythm_ssm import separate_drums

    b = order_bundle.get(stem)
    d = separate_drums(AUDIO / f"{stem}.m4a")
    if d is None:
        raise RuntimeError(f"pas de stem batterie en cache pour {stem}")
    y, sr = sf.read(str(d))
    y = y.mean(1) if y.ndim > 1 else y
    if sr != SR:
        y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=SR)
    y = np.asarray(y, np.float32)

    n_mels = 64
    fmels = librosa.mel_frequencies(n_mels=n_mels, fmin=0.0, fmax=SR / 2)
    canaux = [0] + [int(np.searchsorted(fmels, hi)) for _lo, hi in BANDES]
    canaux = sorted(set(min(n_mels, c) for c in canaux))
    env = librosa.onset.onset_strength_multi(
        y=y, sr=SR, hop_length=HOP, n_mels=n_mels, channels=canaux,
        aggregate=np.median)
    times = librosa.frames_to_time(np.arange(env.shape[1]), sr=SR, hop_length=HOP)
    env = env / np.clip(env.max(axis=1, keepdims=True), 1e-9, None)

    tot = env.sum(0)
    idx = librosa.util.peak_pick(tot, pre_max=3, post_max=3, pre_avg=8, post_avg=8,
                                 delta=0.08 * float(np.median(tot) + tot.std()),
                                 wait=2)
    onsets = [(float(times[i]), int(np.argmax(env[:, i])), float(tot[i])) for i in idx]

    out = {"env": env.astype(np.float32), "times": times.astype(np.float32),
           "onsets": onsets, "n": int(b["n"]),
           "grid": np.asarray(b["grid"], float)}
    with p.open("wb") as f:
        pickle.dump(out, f)
    return out


def bornes(grid, n, res):
    """Les bornes des cases : res = 1 (mesure), 2 (demi-mesure), 4 (temps)."""
    t = []
    for b in range(n):
        t0, t1 = grid[b], grid[b + 1]
        t += [t0 + (t1 - t0) * k / res for k in range(res)]
    t.append(grid[n])
    return np.asarray(t)


def patchs(F, res, sous=16):
    """(n_cases, 4*sous) — le motif d'attaques de chaque case, L2-normalisé.

    `sous` points par MESURE (donc sous/res par case) : à 16, une case de mesure
    est décrite au niveau de la double-croche.
    """
    env, times, grid, n = F["env"], F["times"], F["grid"], F["n"]
    e = bornes(grid, n, res)
    k = max(2, sous // res)
    P = np.zeros((len(e) - 1, env.shape[0] * k), float)
    for i in range(len(e) - 1):
        ts = e[i] + (np.arange(k) + 0.5) / k * (e[i + 1] - e[i])
        v = np.stack([np.interp(ts, times, env[b]) for b in range(env.shape[0])])
        P[i] = v.reshape(-1)
    return P, e


def brut(F, res, sous=16):
    """Les mêmes patchs, SANS normalisation — pour l'énergie et la densité."""
    env, times, grid, n = F["env"], F["times"], F["grid"], F["n"]
    e = bornes(grid, n, res)
    k = max(2, sous // res)
    P = np.zeros((len(e) - 1, env.shape[0], k), float)
    for i in range(len(e) - 1):
        ts = e[i] + (np.arange(k) + 0.5) / k * (e[i + 1] - e[i])
        for b in range(env.shape[0]):
            P[i, b] = np.interp(ts, times, env[b])
    return P, e


def _norm(P):
    return P / np.clip(np.linalg.norm(P, axis=1, keepdims=True), 1e-9, None)


def _ref(P, res, W):
    """M[i] = la NORME LOCALE de la case i : la médiane des W cases occupant la
    MÊME position dans les W mesures précédentes.

    Le calage de phase n'est pas cosmétique. À la demi-mesure ou au temps, un
    groove à contretemps rend chaque case différente de sa voisine ; comparer le
    temps 4 à la moyenne des temps 1-2-3 ferait de tout backbeat une anomalie.
    On compare le temps 4 aux temps 4 d'avant. À la mesure (res=1) ça se réduit
    à la médiane glissante des W mesures précédentes.
    """
    M = np.zeros_like(P)
    for i in range(len(P)):
        prev = [P[i - k * res] for k in range(1, W + 1) if i - k * res >= 0]
        M[i] = np.median(prev, axis=0) if prev else P[i]
    return M


def _cos(A, B):
    a = np.clip(np.linalg.norm(A, axis=1), 1e-9, None)
    b = np.clip(np.linalg.norm(B, axis=1), 1e-9, None)
    return np.clip(np.einsum("ij,ij->i", A, B) / (a * b), 0, 1)


def _decale(x, k):
    """x décalé de k cases vers la gauche : y[i] = x[i+k] (bord = dernière)."""
    y = np.empty_like(x)
    if k <= 0:
        return x.copy()
    y[:-k] = x[k:]
    y[-k:] = x[-1] if x.ndim == 1 else x[-1][None]
    return y


def _z(x):
    """z-score robuste (médiane / écart absolu médian) — l'échelle du morceau."""
    x = np.nan_to_num(np.asarray(x, float))
    m = float(np.median(x))
    s = float(np.median(np.abs(x - m))) * 1.4826
    return (x - m) / s if s > 1e-12 else x - m


# ── les variantes ───────────────────────────────────────────────────────────

def variantes(F, res=1, W=4):
    """{nom: (courbe, res)} — toutes les façons de dire « ici, un fill ».

    Convention, et elle est décisive pour lire les écarts : la courbe est
    indexée par la case où LE PHÉNOMÈNE SE PRODUIT, jamais par la frontière
    qu'on en déduit. Un fill dans la mesure b donne un pic à la position b, donc
    un écart de −1 mesure avec la frontière b+1. Aucun décalage n'est appliqué
    en douce ; c'est la mesure des écarts qui doit le révéler.
    """
    n = F["n"]
    P, e = patchs(F, res)              # motif normalisé (forme du groove)
    B, _ = brut(F, res)                # (cases, 4 bandes, sous-pas) énergie brute
    M = _ref(P, res, W)
    nrj = B.sum(axis=(1, 2))
    Mn = np.array([np.median([nrj[i - k * res] for k in range(1, W + 1) if i - k * res >= 0]
                             or [nrj[i]]) for i in range(len(nrj))])

    # proximité de chaque case à sa norme locale, et son retour une mesure après
    prox = _cos(P, M)
    prox_apres = _decale(prox, res)
    anom = 1.0 - prox
    retour = np.clip(prox_apres - prox, 0, None)

    # LE MOTIF CENTRÉ. Sur une enveloppe d'attaques toujours positive, le
    # cosinus est saturé par le SOCLE commun à toutes les mesures (charleston
    # sur chaque croche, kick sur 1 et 3) : deux mesures très différentes
    # restent à cos ≈ 0,95 et l'anomalie mesure surtout du bruit. C'est le même
    # diagnostic que `rhythm_ssm._mean_center_renormalise` (et que le plancher
    # DC du cosinus de chroma). On retire le motif MOYEN DU MORCEAU avant de
    # comparer : ce qui reste, ce sont les écarts au groove — c'est-à-dire les
    # fills.
    Pc = _norm(P - P.mean(0, keepdims=True))
    Mc = _ref(Pc, res, W)
    proxc = np.clip(np.einsum("ij,ij->i", Pc, _norm(Mc)), -1, 1)
    anomc = 1.0 - proxc
    retourc = np.clip(_decale(proxc, res) - proxc, 0, None)

    V = {}
    V["anomalie seule"] = anom
    V["retour seul"] = retour
    V["anomalie × retour"] = anom * retour
    V["anomalie centrée"] = anomc
    V["anomalie centrée × retour"] = anomc * retourc

    # densité d'attaques
    ons = np.array([t for t, _b, _s in F["onsets"]])
    cnt = np.histogram(ons, bins=e)[0].astype(float)
    Mc = np.array([np.median([cnt[i - k * res] for k in range(1, W + 1) if i - k * res >= 0]
                             or [cnt[i]]) for i in range(len(cnt))])
    dens = np.log1p(cnt) - np.log1p(Mc)
    V["densité d'attaques"] = np.clip(dens, 0, None)
    V["densité × retour"] = np.clip(dens, 0, None) * retour

    # déplacement de bande : le fill déserte kick+charley et charge le médium
    mix = B.sum(axis=2)
    mix = mix / np.clip(mix.sum(1, keepdims=True), 1e-9, None)
    Mm = _ref(mix, res, W)
    V["déplacement de bande"] = 1.0 - _cos(mix, Mm)
    med = (mix[:, 1] + 1e-6) / (mix[:, 0] + mix[:, 3] + 1e-6)
    Mr = np.array([np.median([med[i - k * res] for k in range(1, W + 1) if i - k * res >= 0]
                             or [med[i]]) for i in range(len(med))])
    V["médium / (kick+cymbales)"] = np.clip(np.log(med / np.clip(Mr, 1e-9, None)), 0, None)

    # sortie de grille : l'énergie va aux positions qui ne sont pas des croches
    k = B.shape[2]
    pas = max(1, k // (8 // res)) if res <= 8 else 1
    sur = np.zeros(len(B), bool)
    grille = np.zeros(k, bool)
    grille[::max(1, pas)] = True
    on = B[:, :, grille].sum(axis=(1, 2))
    off = B[:, :, ~grille].sum(axis=(1, 2)) if (~grille).any() else np.zeros(len(B))
    hg = off / np.clip(on + off, 1e-9, None)
    Mh = np.array([np.median([hg[i - kk * res] for kk in range(1, W + 1) if i - kk * res >= 0]
                             or [hg[i]]) for i in range(len(hg))])
    V["sortie de grille"] = np.clip(hg - Mh, 0, None)
    del sur

    # le break : l'anomalie est un TROU, et la batterie revient après
    trou = np.clip(1.0 - nrj / np.clip(Mn, 1e-9, None), 0, None)
    revient = np.clip(_decale(nrj, res) / np.clip(Mn, 1e-9, None) - nrj / np.clip(Mn, 1e-9, None),
                      0, None)
    V["break (trou × retour)"] = trou * revient

    # crash : transitoire de cymbale sur la PREMIÈRE croche de la case, z-scoré
    # sur toutes les cases de même phase. Le seul critère qui vise la frontière
    # elle-même et pas la mesure d'avant.
    tete = B[:, 3, :max(1, k // (8 // res))].max(axis=1) if k >= 2 else B[:, 3, 0]
    cr = np.zeros(len(tete))
    for ph in range(res):
        sel = np.arange(ph, len(tete), res)
        cr[sel] = _z(tete[sel])
    V["crash sur le temps 1"] = np.clip(cr, 0, None)

    # fill PUIS crash : l'anomalie en b et la cymbale en b+1, le motif complet
    V["fill puis crash"] = V["anomalie × retour"] * np.clip(_decale(cr, res), 0, None)
    V["densité puis crash"] = V["densité d'attaques"] * np.clip(_decale(cr, res), 0, None)

    # référence : le damier de Foote sur la matrice de batterie (un des 39).
    # Sur le motif CENTRÉ, pour la même raison que ci-dessus.
    S = Pc @ Pc.T
    L = max(2, 4 * res)
    a = np.arange(-L, L) + 0.5
    g = np.exp(-0.5 * (a / (0.6 * L)) ** 2)
    Wk = np.outer(g, g) * np.outer(np.sign(a), np.sign(a))
    fo = np.zeros(len(P))
    for i in range(L, len(P) - L + 1):
        fo[i] = float((S[i - L:i + L, i - L:i + L] * Wk).sum())
    # signe : le damier vaut (bloc avant + bloc après) − (blocs croisés), donc
    # il est HAUT à une frontière. Il était nié dans la première version : le
    # critère de référence pointait exactement à côté de ce qu'il détecte.
    V["damier batterie (référence)"] = fo / max(1e-9, np.abs(Wk).sum())

    return {k2: (np.nan_to_num(np.asarray(v, float)), res) for k2, v in V.items()}, n


# ── mesurer : rappel à budget imposé, et la DISTRIBUTION DES ÉCARTS ─────────

BUDGET = 8.0     # un pic toutes les 8 mesures au plus — le coût, fixé pour tous
TOL = 1.0        # « le pic marque la frontière » = à moins d'une mesure
PORTEE = 2.5     # au-delà, le pic ne parle plus de cette frontière


def pics(courbe, res, k, ecart=2.0):
    """[(position EN MESURES, hauteur)] — les k plus hauts maxima locaux.

    La position est `i / res`, c'est-à-dire le DÉBUT de la case où le critère
    pique, et elle reste fractionnaire. C'est le piège de `ancres.py` pris à
    l'envers : on ne renvoie jamais « la mesure qui compte le pic ».
    """
    c = np.asarray(courbe, float)
    K = len(c)
    sep = max(1, int(round(ecart * res)))
    loc = [i for i in range(1, K - 1) if c[i] >= c[i - 1] and c[i] > c[i + 1] and c[i] > 0]
    loc.sort(key=lambda i: -c[i])
    out = []
    for i in loc:
        if all(abs(i - j) >= sep for j in out):
            out.append(i)
        if len(out) >= k:
            break
    return sorted((i / res, float(c[i])) for i in out)


def frontieres(stem):
    p = ANN / f"{stem}.json"
    if not p.exists():
        return [], 0
    d = json.loads(p.read_text())
    return sorted({int(s["b0"]) for s in d["sections"] if 0 < int(s["b0"]) < d["n"]}), d["n"]


VALIDES = None


def liste_validee():
    global VALIDES
    if VALIDES is None:
        VALIDES = sorted(f.stem for f in ANN.glob("*.json")
                         if json.loads(f.read_text()).get("validated")
                         and (AUDIO / f"{f.stem}.m4a").exists())
    return VALIDES


SANS_BATTERIE = {"yesterday_remastered_2009", "bein_green"}


def ecarts(P, G, portee=PORTEE):
    """Pour chaque frontière, l'écart signé au pic le plus proche (ou None).

    Orienté FRONTIÈRE et pas pic, exprès : les annotations ne sont pas
    exhaustives, donc un pic sans frontière n'est pas une erreur et n'a pas
    d'écart à rapporter. Négatif = le pic est EN AVANCE sur la frontière.
    """
    out = []
    for g in G:
        d = [p - g for p, _v in P if abs(p - g) <= portee]
        out.append(min(d, key=abs) if d else None)
    return out


def score_morceau(courbe, res, n, G, budget=BUDGET, decal=0.0, tol=TOL):
    k = max(1, int(round(n / budget)))
    P = [(p + decal, v) for p, v in pics(courbe, res, k)]
    E = ecarts(P, G)
    touche = sum(1 for e in E if e is not None and abs(e) <= tol)
    return {"pics": P, "ecarts": E, "touche": touche, "ftr": len(G),
            "densite": n / max(1, len(P))}


def _mad(x):
    x = np.asarray(x, float)
    return float(np.median(np.abs(x - np.median(x)))) * 1.4826 if len(x) else float("nan")


def evalue(config, stems=None, budget=BUDGET):
    """{nom de variante: bilan corpus} pour une (res, W) donnée.

    Le bilan contient le rappel BRUT, la distribution des écarts, et le rappel
    après correction du décalage en *leave-one-song-out* : l'offset est la
    médiane des écarts des 17 AUTRES morceaux, appliqué au 18e. C'est la
    différence entre « je sais qu'il y a un décalage » et « je sais le
    corriger sur un morceau que je n'ai jamais vu ».
    """
    res, W = config
    stems = stems or liste_validee()
    par_stem = {}
    for s in stems:
        F = feats(s)
        V, n = variantes(F, res=res, W=W)
        G, _ = frontieres(s)
        par_stem[s] = (V, n, G)
    noms = list(next(iter(par_stem.values()))[0].keys())

    bilan = {}
    for nom in noms:
        brut_e, touche, ftr, npics, nbars = [], 0, 0, 0, 0
        par = {}
        for s in stems:
            V, n, G = par_stem[s]
            c, r = V[nom]
            sc = score_morceau(c, r, n, G, budget=budget)
            par[s] = sc
            touche += sc["touche"]; ftr += len(G)
            npics += len(sc["pics"]); nbars += n
            brut_e += [e for e in sc["ecarts"] if e is not None]
        # correction leave-one-song-out du décalage
        tl, tl05 = 0, 0
        offs = {}
        for s in stems:
            autres = [e for t in stems if t != s for e in par[t]["ecarts"] if e is not None]
            off = -float(np.median(autres)) if autres else 0.0
            offs[s] = off
            V, n, G = par_stem[s]
            c, r = V[nom]
            tl += score_morceau(c, r, n, G, budget=budget, decal=off)["touche"]
            tl05 += score_morceau(c, r, n, G, budget=budget, decal=off, tol=0.5)["touche"]
        # LA DISTRIBUTION DES ÉCARTS, et pourquoi ce n'est pas `ecarts()`.
        # `ecarts()` prend le pic LE PLUS PROCHE de chaque frontière : à densité
        # 1 pic / 8 mesures, il y a presque toujours exactement un pic dans la
        # fenêtre ±2,5, tiré presque uniformément — la médiane du plus proche
        # vaut donc ~0 quoi qu'il arrive. C'est un artefact d'estimateur, pas un
        # décalage nul. L'histogramme ci-dessous compte TOUTES les paires
        # (pic, frontière) à moins de 3 mesures : lui a un plancher plat, et son
        # excès au-dessus du plancher est le vrai décalage.
        H = np.zeros(13)
        for s in stems:
            _V, _n, G = par_stem[s]
            for p, _v in par[s]["pics"]:
                for g in G:
                    d = p - g
                    if abs(d) <= 3.0:
                        H[int(round(2 * d)) + 6] += 1
        bilan[nom] = {
            "histo": H,
            "pic_histo": (int(np.argmax(H)) - 6) / 2.0,
            "rappel": touche / max(1, ftr), "ftr": ftr, "touche": touche,
            "densite": nbars / max(1, npics),
            "ecart_med": float(np.median(brut_e)) if brut_e else float("nan"),
            "ecart_mad": _mad(brut_e), "ecart_std": float(np.std(brut_e)) if brut_e else float("nan"),
            "n_ecarts": len(brut_e),
            "rappel_loso": tl / max(1, ftr), "rappel_loso_05": tl05 / max(1, ftr),
            "offset": float(np.median(list(offs.values()))),
            "par": par,
        }
    return bilan


def hasard(n, G, budget=BUDGET, tol=TOL, tirages=400, rng=None):
    """Le rappel qu'un tirage uniforme obtient au même budget. Le plancher."""
    rng = rng or np.random.default_rng(0)
    k = max(1, int(round(n / budget)))
    tot = 0.0
    for _ in range(tirages):
        P = []
        for _t in range(200):
            if len(P) >= k:
                break
            x = float(rng.integers(1, max(2, n)))
            if all(abs(x - y) >= 2 for y in P):
                P.append(x)
        tot += sum(1 for g in G if any(abs(p - g) <= tol for p in P))
    return tot / tirages


def niveau_hasard(stems=None, budget=BUDGET):
    stems = stems or liste_validee()
    a = b = 0.0
    for s in stems:
        G, n = frontieres(s)
        a += hasard(n, G, budget=budget); b += len(G)
    return a / max(1, b)


CONFIGS = [(1, 2), (1, 4), (1, 8), (2, 4), (4, 4), (2, 8), (4, 8)]
RES_NOM = {1: "mesure", 2: "½ mesure", 4: "temps"}


def table(configs=CONFIGS, budget=BUDGET, stems=None):
    """Le tableau comparatif : rappel, densité, décalage, dispersion."""
    stems = stems or liste_validee()
    h = niveau_hasard(stems, budget)
    print(f"\n{len(stems)} morceaux · budget {budget:.0f} mesures/pic · tolérance ±{TOL:.0f} mesure")
    print(f"NIVEAU DU HASARD au même budget : rappel {100 * h:.0f} %  "
          f"— toute variante en dessous ne dit rien.\n")
    print(f"{'variante':<30}{'gran.':<10}{'W':>2} {'rappel':>7} {'dens.':>6} "
          f"{'écart méd':>10} {'MAD':>6} {'σ':>6} {'LOSO±1':>7} {'LOSO±½':>7}")
    lignes = []
    for res, W in configs:
        B = evalue((res, W), stems=stems, budget=budget)
        for nom, r in sorted(B.items(), key=lambda kv: -kv[1]["rappel"]):
            lignes.append(((res, W), nom, r))
            print(f"{nom[:29]:<30}{RES_NOM[res]:<10}{W:>2} "
                  f"{100 * r['rappel']:>6.0f}% {r['densite']:>6.1f} "
                  f"{r['ecart_med']:>+10.2f} {r['ecart_mad']:>6.2f} {r['ecart_std']:>6.2f} "
                  f"{100 * r['rappel_loso']:>6.0f}% {100 * r['rappel_loso_05']:>6.0f}%")
        print()
    lignes.sort(key=lambda x: -x[2]["rappel"])
    print("── les douze meilleures, tous réglages confondus ──")
    for (res, W), nom, r in lignes[:12]:
        print(f"{nom[:29]:<30}{RES_NOM[res]:<10}{W:>2} "
              f"{100 * r['rappel']:>6.0f}% {r['densite']:>6.1f} "
              f"{r['ecart_med']:>+10.2f} {r['ecart_mad']:>6.2f} {r['ecart_std']:>6.2f} "
              f"{100 * r['rappel_loso']:>6.0f}% {100 * r['rappel_loso_05']:>6.0f}%")
    print("\n── la DISTRIBUTION DES ÉCARTS (toutes les paires pic/frontière à ≤3 "
          "mesures), en demi-mesures de −3 à +3 ──")
    print(f"{'variante':<30}{'gran.':<10}" + "".join(f"{x / 2:>+6.1f}" for x in range(-6, 7)))
    for (res, W), nom, r in lignes[:12]:
        H = r["histo"]
        print(f"{nom[:29]:<30}{RES_NOM[res]:<10}" + "".join(f"{int(v):>6d}" for v in H))
    return lignes


def profil(res=1, W=4, stems=None, lags=None):
    """Le RANG MOYEN de chaque courbe autour des frontières — sans pics.

    Le pic-picking ajoute son propre bruit ; ce diagnostic-là mesure le signal
    brut : « à la frontière + L, la courbe est-elle haute dans sa propre
    distribution ? ». 0,50 = niveau de base, il n'y a rien.
    """
    stems = stems or liste_validee()
    lags = lags or (range(-3, 4) if res == 1 else range(-2 * res, 2 * res + 1))
    acc: dict = {}
    for s in stems:
        F = feats(s)
        V, n = variantes(F, res=res, W=W)
        G, _ = frontieres(s)
        for nom, (c, r) in V.items():
            x = np.nan_to_num(np.asarray(c, float))
            rg = np.argsort(np.argsort(x)) / max(1, len(x) - 1)
            for L in lags:
                acc.setdefault(nom, {}).setdefault(L, []).extend(
                    [rg[g * res + L] for g in G if 0 <= g * res + L < len(rg)])
    unite = "mesures" if res == 1 else ("demi-mesures" if res == 2 else "temps")
    print(f"\nrang moyen de la courbe à (frontière + décalage), décalage en {unite} "
          f"· granularité {RES_NOM[res]} · fenêtre {W}")
    print(f"{'variante':<30}" + "".join(f"{L:>6}" for L in lags))
    for nom, d in sorted(acc.items(), key=lambda kv: -max(np.mean(v) for v in kv[1].values())):
        print(f"{nom[:29]:<30}" + "".join(f"{np.mean(d[L]):>6.2f}" for L in lags))
    return acc


def par_morceau(nom, res, W, budget=BUDGET, stems=None):
    """Une variante, morceau par morceau : c'est là que se lit l'hétérogénéité."""
    stems = stems or liste_validee()
    print(f"\n« {nom} » · {RES_NOM[res]} · fenêtre {W} · budget {budget:.0f} mes./pic")
    print(f"{'morceau':<46}{'ftr':>4}{'pics':>5}{'marquées':>9}{'rappel':>7}"
          f"{'écart méd':>10}  batterie")
    T = M = 0
    for s in stems:
        F = feats(s)
        V, n = variantes(F, res=res, W=W)
        G, _ = frontieres(s)
        sc = score_morceau(V[nom][0], V[nom][1], n, G, budget=budget)
        E = [e for e in sc["ecarts"] if e is not None]
        T += len(G); M += sc["touche"]
        print(f"{s[:45]:<46}{len(G):>4}{len(sc['pics']):>5}{sc['touche']:>9}"
              f"{100 * sc['touche'] / max(1, len(G)):>6.0f}%"
              f"{(np.median(E) if E else float('nan')):>+10.2f}"
              f"  {'—' if s in SANS_BATTERIE else 'oui'}")
    print(f"{'TOTAL':<46}{T:>4}{'':>5}{M:>9}{100 * M / max(1, T):>6.0f}%")


# ── la page écoutable ───────────────────────────────────────────────────────

QUATRE = ["maroon_5_this_love", "let_it_be_remastered_2009",
          "pharrell_williams_happy_official_video",
          "the_ronettes_be_my_baby_music_video"]

# ce qu'on montre, dans l'ordre : les variantes du brief plus les deux témoins
MONTRE = [
    ("anomalie centrée", 1, 2, "le motif d'attaques de la mesure contre la médiane des "
     "2 précédentes, motif MOYEN du morceau retiré. La variante retenue."),
    ("anomalie × retour", 1, 2, "la formule du brief : anomalie en b MULTIPLIÉE par le "
     "retour au groove en b+1. Elle perd 4 points contre l'anomalie seule."),
    ("densité d'attaques", 1, 2, "combien de frappes dans la mesure, contre la norme "
     "locale. Un fill entasse 1,5 à 3× plus."),
    ("déplacement de bande", 1, 4, "la répartition kick / caisse claire / médium / "
     "cymbales change : le fill déserte le charley et charge les toms."),
    ("sortie de grille", 1, 2, "l'énergie part aux positions qui ne sont pas des croches "
     "— doubles-croches, triolets."),
    ("break (trou × retour)", 1, 4, "le dual du fill : la batterie s'arrête une mesure, "
     "puis revient."),
    ("crash sur le temps 1", 1, 2, "transitoire de cymbale sur la première croche, "
     "z-scoré sur toutes les mesures. Le seul qui vise la frontière elle-même."),
    ("fill puis crash", 1, 2, "le motif complet : l'anomalie en b et la cymbale en b+1."),
    ("damier batterie (référence)", 1, 4, "TÉMOIN, ce n'est pas un détecteur de fill : "
     "le damier de Foote sur la matrice de batterie, l'un des 39 critères existants."),
]
CHOISIE = ("anomalie centrée", 1, 2)


def page(stems=None):
    from criteres_sections import bande_png, INK, GT_LINE
    from ssm_zoo import SONGS
    stems = stems or QUATRE
    titres = dict(SONGS)
    body = []
    for stem in stems:
        F = feats(stem)
        G, _ = frontieres(stem)
        n, grid = F["n"], F["grid"]
        k = max(1, int(round(n / BUDGET)))
        cache: dict = {}

        def V(res, W):
            if (res, W) not in cache:
                cache[(res, W)] = variantes(F, res=res, W=W)[0]
            return cache[(res, W)]

        lignes = ""
        for nom, res, W, glose in MONTRE:
            c, r = V(res, W)[nom]
            rg = np.argsort(np.argsort(np.nan_to_num(c))) / max(1, len(c) - 1)
            png = bande_png(rg)
            P = pics(c, r, k)
            ticks = ""
            for p, _v in P:
                e = min([abs(p - g) for g in G], default=99)
                cl = "ok" if e <= 0.5 else ("pres" if e <= 1.0 else "hors")
                ticks += (f'<i class="pk {cl}" style="left:{p / n * 100:.4f}%" '
                          f'title="{nom} — mesure {p + 1:.2f}"></i>')
            marq = sum(1 for g in G if any(abs(p - g) <= 1.0 for p, _v in P))
            lignes += (f'<div class=lane><div class=lab title="{glose}">{nom}'
                       f'<small>{marq}/{len(G)}</small></div>'
                       f'<div class=strip><img src="data:image/png;base64,{png}" alt="">'
                       f'{ticks}</div></div>')

        nom, res, W = CHOISIE
        c, r = V(res, W)[nom]
        P = pics(c, r, k)
        btns = ""
        for p, _v in P:
            d = [p - g for g in G]
            e = min(d, key=abs) if d else 99
            cl = "ok" if abs(e) <= 0.5 else ("pres" if abs(e) <= 1.0 else "hors")
            eti = ("sur ta frontière" if abs(e) <= 0.5 else
                   (f"à {e:+.0f} mesure de ta frontière" if abs(e) <= 1.0
                    else "hors annotation"))
            t = grid[int(p)] + (p - int(p)) * (grid[min(n, int(p) + 1)] - grid[int(p)])
            btns += (f'<button class="fl {cl}" data-b="{p:.3f}">mes. <b>{p + 1:.0f}</b>'
                     f'<small>{eti} · {int(t // 60)}:{int(t % 60):02d}</small></button>')

        blocs = ""
        pann = ANN / f"{stem}.json"
        if pann.exists():
            for s in json.loads(pann.read_text())["sections"]:
                w = (s["b1"] - s["b0"] + 1) / n * 100
                blocs += (f'<i style="left:{s["b0"] / n * 100:.4f}%;width:{w:.4f}%">'
                          f'{s.get("label", "")}</i>')
        pas = 4 if n <= 60 else (8 if n <= 140 else 16)
        regle = "".join(f'<i style="left:{i / n * 100:.4f}%">{i + 1}</i>'
                        for i in range(0, n, pas))
        gtl = "".join(f'<i class=gt style="left:{g / n * 100:.4f}%"></i>' for g in G)

        body.append(
            f'<section data-grid="{json.dumps([float(x) for x in grid])}" data-n="{n}">'
            f'<div class=hd><h2>{titres.get(stem, stem.replace("_", " "))}</h2>'
            f'<span class=sub>{n} mesures · {len(G)} frontières validées · '
            f'{k} pics par variante (budget commun : une toutes les 8 mesures)</span></div>'
            f'<div class=bar><button class=pp>▶</button><span class=pos>mes. 1</span>'
            f'<span class=hint>clique la bande pour écouter · clique un fill pour '
            f"l'entendre en contexte (2 mesures avant → 2 après)</span></div>"
            f'<div class=stack>'
            f'<div class=lane><div class=lab></div><div class=secs>{blocs}</div></div>'
            f'<div class=lane><div class=lab></div><div class=rule>{regle}</div></div>'
            f'{lignes}<div class=ov>{gtl}<div class=cur></div></div></div>'
            f'<div class=lane2>{btns}</div>'
            f'<audio preload=metadata playsinline src="../audio/{stem}.m4a"></audio>'
            f'</section>')
        print(f"  ok {titres.get(stem, stem)} — {len(P)} fills, {len(G)} frontières")

    css = f"""
body{{margin:0;background:#e7e0d0;font:15px/1.55 -apple-system,system-ui,sans-serif;color:{INK}}}
.wrap{{max-width:1240px;margin:0 auto;padding:22px 14px 80px}}
h1{{font:italic 600 26px Georgia,serif;margin:0 0 6px}}
.lede{{color:#6f6857;font-size:13.5px;margin-bottom:16px;max-width:1000px}}
.lede b{{color:{INK}}}
section{{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;
  padding:10px 14px 14px;margin-bottom:14px;--lab:200px}}
.hd{{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap}}
h2{{font:700 18px system-ui;margin:0;color:#8a2b2b}}
.sub{{font:500 12px system-ui;color:#8a8371}}
.bar{{display:flex;align-items:center;gap:14px;margin:8px 0 10px;flex-wrap:wrap}}
.pp{{width:34px;height:34px;border-radius:50%;border:1px solid #d8cfb4;
  background:#f7f3e9;font-size:13px;cursor:pointer;flex:none}}
.pos{{font:600 12px ui-monospace,monospace;min-width:120px}}
.hint{{font:500 11.5px system-ui;color:#a89f8c}}
.stack{{position:relative}}
.lane{{display:grid;grid-template-columns:var(--lab) 1fr;align-items:center;height:22px}}
.lab{{font:600 11px system-ui;padding-right:8px;text-align:right;white-space:nowrap;
  overflow:hidden;text-overflow:ellipsis}}
.lab small{{color:#b5ab94;font-weight:500;margin-left:5px}}
.strip{{position:relative;height:15px;border-bottom:1px solid #f2ecdd}}
.strip img{{width:100%;height:15px;display:block;border-radius:3px;image-rendering:pixelated}}
.pk{{position:absolute;top:-2px;width:3px;height:19px;border-radius:1px;
  transform:translateX(-1.5px);box-shadow:0 0 0 1px rgba(255,255,255,.75)}}
.pk.ok{{background:#3f7a4f}} .pk.pres{{background:#d9a441}} .pk.hors{{background:#5c8fa8}}
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
button.fl{{border:2px solid #e0d7c2;background:#f7f3e9;border-radius:9px;padding:5px 9px;
  cursor:pointer;font:600 12.5px system-ui;color:#4a4438;text-align:left}}
button.fl small{{display:block;font:500 10px system-ui;color:#a89f8c}}
button.fl.ok{{border-color:#3f7a4f}} button.fl.pres{{border-color:#d9a441}}
button.fl.hors{{border-color:#5c8fa8;border-style:dashed}}
button.fl.on{{background:#0d2437;color:#fff}} button.fl.on small{{color:#9fb8c6}}
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
  sec.querySelectorAll("button.fl").forEach(function(btn){
    btn.onclick=function(){
      sec.querySelectorAll("button.fl").forEach(function(x){x.classList.remove("on")});
      btn.classList.add("on");
      var b=parseFloat(btn.dataset.b);
      aller(b2t(Math.max(0,b-2)), b2t(Math.min(n,b+2)));
    };
  });
});
"""
    OUT.write_text(f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Les fills de batterie</title><style>{css}</style></head><body><div class=wrap>
<h1>Les fills de batterie — neuf façons de les entendre venir</h1>
<div class=lede>Chaque ligne est une façon de dire « la batterie fait quelque chose
d'inhabituel ici ». Toutes ont <b>le même budget</b> : une proposition toutes les
8 mesures, ni plus ni moins — sans ça une ligne bruitée « trouverait » tout.
<b>Les traits rouges</b> sont tes frontières. Les marques sur les bandes sont les
propositions : <b style="color:#3f7a4f">vert</b> = sur une frontière,
<b style="color:#d9a441">ambre</b> = à une mesure,
<b style="color:#5c8fa8">bleu</b> = hors annotation — et hors annotation ne veut
pas dire faux, tu n'as pas noté tous tes changements : <b>ce sont ceux-là qu'il
faut écouter</b>.<br><br>
Les boutons du bas sont les fills de la variante retenue
(<b>anomalie centrée</b>) : clique, ça joue deux mesures avant → deux après.
Le chiffre à droite de chaque nom est le nombre de tes frontières que cette
ligne marque.<br><br>
<b>Ce que ça vaut, sur les 18 morceaux validés</b> : la meilleure variante marque
<b>50 % de tes frontières</b> contre <b>36 % pour un tirage au hasard</b> au même
budget. C'est réel, c'est faible, et c'est très inégal d'un morceau à l'autre
(70 % sur This Love et Grenade, 30 % sur Stand By Me).
<b>Multiplier l'anomalie par le retour au groove — la formule attendue — fait
perdre 4 points</b> ; l'anomalie seule est meilleure.</div>
{''.join(body)}</div><script>{js}</script></body></html>""")
    print(f"wrote {OUT}  ({OUT.stat().st_size // 1024} KB)")
    print("   http://100.89.209.63:7772/plots/fills.html")


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--table" in a:
        table()
    elif "--profil" in a:
        for res in (1, 2, 4):
            profil(res=res)
    elif "--morceau" in a:
        i = a.index("--morceau")
        par_morceau(a[i + 1], int(a[i + 2]), int(a[i + 3]))
    else:
        page([x for x in a if not x.startswith("--")] or QUATRE)
