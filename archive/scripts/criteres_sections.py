"""Trente critères de changement de section, côte à côte, avec tête de lecture.

    .venv/bin/python scripts/criteres_sections.py [<stem> ...]
        -> docs/plots/criteres_sections.html
        (http://100.89.209.63:7772/plots/criteres_sections.html)

Louis, 2026-08-14 :

  « Pour chaque signal qu'on a (voix, rythme, harmonie, intensité du son…),
    proposer plusieurs critères de différenciation — varier la granularité
    (temps, demi-barre, barre), le type de produit scalaire (cosinus,
    harmonique, pondération, intensité), matrice SSM ou juste profil de
    changement d'intensité. Puis voir lesquels, INDIVIDUELLEMENT, sont les mieux
    corrélés à des changements de section. Pas grave s'ils en loupent : mon
    hypothèse, c'est que chaque chanson, chaque style a son propre ensemble de
    signaux qui marquent les changements, et parfois c'est un peu avant ou un
    peu après. Fais-moi surtout un outil de visualisation avec tête de lecture
    pour que je puisse évaluer à la main lesquels marchent le mieux. »

CE QUE LA PAGE MONTRE. Une ligne par critère. Chaque ligne est une **courbe de
changement** : sombre = « ici, ça change ». Toutes les lignes partagent le même
axe de mesures, la tête de lecture les traverse toutes, un clic n'importe où
lance le son à cet endroit. Les traits rouges verticaux sont les frontières que
Louis a validées (`harmonia_min/state/sections/*.json`).

LE BUDGET DE PICS — la seule précaution qui rend la comparaison honnête. Un
critère qui pique partout tomberait juste partout. Donc chaque critère propose
EXACTEMENT autant de frontières qu'il y en a dans le morceau : ses plus hauts
pics, espacés d'au moins deux mesures. Les pastilles à gauche disent, frontière
par frontière, si ce critère l'a trouvée (pleine = à la mesure près, cercle =
à une mesure près, vide = ratée). Lire une ligne = compter ses pastilles pleines.

LES CINQ FAÇONS DE FABRIQUER UNE COURBE, à partir d'un même signal :

  damier        le critère de Foote, et c'est exactement la règle que Louis a
                posée pour les reprises, retournée pour les frontières : le
                carré AVANT se ressemble à lui-même, le carré APRÈS aussi, et
                les deux blocs croisés ne se ressemblent PAS. Le noyau est
                gaussien, sa demi-largeur L est dite en mesures.
  contraste     sans matrice : la moyenne des L cases d'avant contre celle des L
                cases d'après, distance cosinus. Plus grossier, plus robuste.
  rupture       la LIGNE de la matrice change : « à qui ressemble cette case
                dans tout le morceau » d'un côté et de l'autre. La bande locale
                est retirée, sinon la diagonale décide. C'est le seul critère
                qui voit une boucle CHANGER DE PARTENAIRE sans changer de
                contenu local.
  saut          aucune matrice, un scalaire par case (RMS, énergie batterie,
                densité de chant) et la valeur absolue de son saut. C'est le
                « profil de changement d'intensité » demandé.
  écart-type    la variabilité locale du scalaire — un fill de batterie, un
                arrêt net, une montée : ça bouge sans changer de niveau moyen.

LES QUATRE PRODUITS SCALAIRES, quand il y a une matrice :

  cos           cosinus des vecteurs normalisés. Le défaut du projet.
  harm          produit HARMONIQUE : un noyau 12×12 qui rapproche les hauteurs
                parentes (quinte 0,55 · tierce 0,40 · ton 0,15 · demi-ton 0,05).
                Sib et Gm se ressemblent, Sib et Fa beaucoup moins. C'est la
                règle « la SSM doit être chord-tone » appliquée au produit.
  corr          les colonnes centrées sur la moyenne du morceau avant le
                cosinus : le profil MOYEN du morceau n'est pas une information
                de section. Indispensable pour le timbre.
  pond          le cosinus MULTIPLIÉ par le rapport d'intensité des deux cases.
                Deux cases qui jouent les mêmes notes mais pas au même volume
                cessent d'être identiques — c'est l'entrée de l'intensité dans
                une matrice de contenu.

CE QUE ÇA NE FAIT PAS. Aucune fusion, aucun modèle, rien de branché sur la prod.
Un critère par ligne, chacun seul avec lui-même : la page sert à décider À
L'OREILLE lesquels méritent d'être fusionnés ensuite, et sur quel style.
"""
from __future__ import annotations

import base64
import io
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
sys.path.insert(0, str(HERE / "scratchpad"))

OUT = HERE / "docs" / "plots" / "criteres_sections.html"
AUDIO = HERE / "docs" / "audio"
ANN = HERE / "harmonia_min" / "state" / "sections"
CACHE = HERE / "scratchpad" / "critere_cache"
CACHE.mkdir(exist_ok=True)
IA = HERE / "scratchpad" / "ia_sections"          # lanes écrites par un LLM

VERSION = 4          # bump = les caches de courbes sont refaits

TROIS = ["maroon_5_this_love",
         "jorja_smith_blue_lights_a_colors_show",
         "bein_green"]

INK = "#3f3a2e"
GT_LINE = "#b4472c"
RAMP = ["#fbf7ec", "#cfe0ea", "#84b3cf", "#3d7fa6", "#1b4a6b", "#0d2437"]

# La couleur d'un signal : la même partout sur la page, pour lire un groupe d'un
# coup d'œil sans lire les libellés.
TEINTE = {
    "voix":      "#8a2b2b",
    "harmonie":  "#2f6d8a",
    "basse":     "#3f5a86",
    "accords":   "#4a6b3a",
    "rythme":    "#8a5a1f",
    "timbre":    "#6a4a8a",
    "intensité": "#8a7a1f",
    "IA":        "#1f6b6b",
}
RES_NOM = {1: "mesure", 2: "demi-mesure", 4: "temps"}


# ── outillage numérique ─────────────────────────────────────────────────────

def cases(grid, n, res):
    """Les bornes des cases : res = 1 (mesure), 2 (demi-mesure), 4 (temps)."""
    t = []
    for b in range(n):
        t0, t1 = grid[b], grid[b + 1]
        t += [t0 + (t1 - t0) * k / res for k in range(res)]
    t.append(grid[n])
    return np.asarray(t)


def pool(arr, times, edges):
    """Moyenne des trames dans chaque case ; la plus proche si la case est vide."""
    arr = np.atleast_2d(np.asarray(arr, float))
    if arr.shape[0] != len(times):
        arr = arr.T
    out = np.zeros((len(edges) - 1, arr.shape[1]))
    for i in range(len(edges) - 1):
        sel = (times >= edges[i]) & (times < edges[i + 1])
        if sel.any():
            out[i] = arr[sel].mean(0)
        elif len(times):
            out[i] = arr[int(np.argmin(np.abs(times - 0.5 * (edges[i] + edges[i + 1]))))]
    return out


def unit(V):
    return V / np.clip(np.linalg.norm(V, axis=1, keepdims=True), 1e-9, None)


def rank01(x):
    """La courbe ramenée à ses propres rangs, dans [0,1].

    Sans ça, une courbe qui vit entre 0,02 et 0,05 est invisible à côté d'une
    qui vit entre 0 et 1, et on comparerait des échelles au lieu de comparer des
    emplacements. Les rangs sont monotones : les pics ne bougent pas.
    """
    x = np.asarray(x, float)
    if x.size == 0 or not np.isfinite(x).any():
        return np.zeros_like(x)
    x = np.nan_to_num(x)
    o = np.argsort(np.argsort(x))
    return o / max(1, len(x) - 1)


def noyau_harmonique():
    """Le noyau 12×12 des parentés de hauteurs, projeté PSD.

    Poids par classe d'intervalle : unisson 1 · quinte/quarte 0,55 · tierces et
    sixtes 0,40 · triton 0,18 · tons 0,15 · demi-tons 0,05. Symétrique par
    inversion (une tierce majeure et une sixte mineure sont le même intervalle).
    La projection PSD garantit que `v·Kv >= 0`, donc que le cosinus harmonique
    reste un vrai cosinus (dans l'espace où K est la métrique).
    """
    w = np.array([1.0, .05, .15, .40, .40, .55, .18, .55, .40, .40, .15, .05])
    K = np.array([[w[(i - j) % 12] for j in range(12)] for i in range(12)])
    ev, U = np.linalg.eigh(K)
    return U @ np.diag(np.clip(ev, 0, None)) @ U.T


KH = noyau_harmonique()


def matrice(F, mode="cos", e=None):
    """La SSM d'un tableau de cases, selon le produit scalaire demandé."""
    F = np.nan_to_num(np.asarray(F, float))
    vide = np.linalg.norm(F, axis=1) <= 1e-9
    if mode == "corr":
        F = F - F.mean(0, keepdims=True)
    if mode == "harm" and F.shape[1] == 12:
        G = F @ KH
        d = np.sqrt(np.clip(np.einsum("ij,ij->i", F, G), 1e-12, None))
        V = G / d[:, None]
        S = np.clip(F @ V.T / d[:, None], -1, 1)
        S = 0.5 * (S + S.T)
    else:
        V = unit(F)
        S = np.clip(V @ V.T, -1, 1)
    if mode == "corr":
        S = (S + 1.0) / 2.0
    S = np.clip(S, 0, 1)
    if mode == "pond" and e is not None:
        a = np.clip(np.asarray(e, float), 1e-9, None)
        R = np.minimum.outer(a, a) / np.maximum.outer(a, a)
        S = S * R
    S[vide, :] = 0.0
    S[:, vide] = 0.0
    return S


def damier(S, L):
    """La nouveauté de Foote : deux carrés qui se tiennent, deux blocs croisés
    qui ne se ressemblent pas. `courbe[i]` = la frontière AVANT la case i."""
    K = S.shape[0]
    a = np.arange(-L, L) + 0.5
    g = np.exp(-0.5 * (a / (0.6 * L)) ** 2)
    sgn = np.outer(np.sign(a), np.sign(a))
    W = np.outer(g, g) * sgn
    out = np.zeros(K)
    for i in range(L, K - L + 1):
        out[i] = float((S[i - L:i + L, i - L:i + L] * W).sum())
    return out / max(1e-9, np.abs(W).sum())


def contraste(F, L):
    """Sans matrice : la moyenne des L cases d'avant contre celle d'après."""
    F = np.nan_to_num(np.asarray(F, float))
    K = F.shape[0]
    out = np.zeros(K)
    for i in range(L, K - L + 1):
        a, b = F[i - L:i].mean(0), F[i:i + L].mean(0)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na > 1e-9 and nb > 1e-9:
            out[i] = 1.0 - float(a @ b) / (na * nb)
    return out


def rupture(S, garde=2):
    """La LIGNE de la matrice change : à qui la case ressemble dans TOUT le
    morceau, avant et après. La bande locale (±`garde`) est retirée, sinon la
    diagonale — qui bouge à chaque case par construction — décide seule."""
    K = S.shape[0]
    out = np.zeros(K)
    for i in range(1, K):
        m = np.ones(K, bool)
        m[max(0, i - 1 - garde):min(K, i + 1 + garde)] = False
        u, v = S[i - 1][m], S[i][m]
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        if nu > 1e-9 and nv > 1e-9:
            out[i] = 1.0 - float(u @ v) / (nu * nv)
    return out


def saut(x, L):
    """Le profil de changement : |moyenne d'après − moyenne d'avant|."""
    x = np.nan_to_num(np.atleast_2d(np.asarray(x, float).T).T)
    if x.ndim == 1:
        x = x[:, None]
    K = x.shape[0]
    out = np.zeros(K)
    for i in range(L, K - L + 1):
        out[i] = float(np.linalg.norm(x[i:i + L].mean(0) - x[i - L:i].mean(0)))
    return out


def bouge(x, L):
    """L'écart-type local d'un scalaire : ça remue, sans changer de niveau."""
    x = np.nan_to_num(np.asarray(x, float))
    K = len(x)
    out = np.zeros(K)
    for i in range(L, K - L + 1):
        out[i] = float(x[i - L:i + L].std())
    return np.abs(np.diff(out, prepend=out[0]))


def pics(courbe, res, k, ecart_mes=2.0):
    """Les k plus hauts maxima locaux, espacés d'au moins `ecart_mes` mesures.

    LE BUDGET EST LE MÊME POUR TOUS LES CRITÈRES (k = le nombre de frontières du
    morceau). Sans ça, un critère bruité qui pique toutes les deux mesures
    « trouverait » toutes les frontières, et la page mentirait.
    """
    c = np.asarray(courbe, float)
    K = len(c)
    ecart = max(1, int(round(ecart_mes * res)))
    loc = [i for i in range(1, K - 1) if c[i] >= c[i - 1] and c[i] > c[i + 1] and c[i] > 0]
    loc.sort(key=lambda i: -c[i])
    out = []
    for i in loc:
        if all(abs(i - j) >= ecart for j in out):
            out.append(i)
        if len(out) >= k:
            break
    return sorted(out)


# ── les signaux ─────────────────────────────────────────────────────────────

def signaux(stem):
    """Tous les substrats d'un morceau, à trois granularités, plus les scalaires.

    Rien n'est calculé deux fois : la séparation demucs (voix, batterie), le
    NNLS et les postérieurs musx sont déjà en cache pour la prod ; ici on ne
    fait que les REDÉCOUPER sur trois grilles.
    """
    import order_bundle
    from ssm_zoo import capture
    import harmonia_min.harmonic_sections as HS
    from licks import notes_de
    import librosa

    b = order_bundle.get(stem)
    n, grid = b["n"], np.asarray(b["grid"], float)
    cap = capture(stem)
    notes = notes_de(stem)

    y, sr = librosa.load(str(AUDIO / f"{stem}.m4a"), sr=22050, mono=True)
    hop = 512
    tf = librosa.frames_to_time(np.arange(1 + len(y) // hop), sr=sr, hop_length=hop)
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    rms_db = librosa.amplitude_to_db(np.clip(rms, 1e-6, None))
    flux = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=48, hop_length=hop)
    mel_db = librosa.power_to_db(mel + 1e-9)
    bandes = np.stack([mel_db[:16].mean(0), mel_db[16:32].mean(0), mel_db[32:].mean(0)])
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=hop)

    env = tenv = None
    try:
        from rhythm_ssm import separate_drums, _drum_onset_bands
        import soundfile as sf
        d = separate_drums(AUDIO / f"{stem}.m4a")
        if d is not None:
            dy, dsr = sf.read(str(d))
            dy = dy.mean(1) if dy.ndim > 1 else dy
            if dsr != 22050:
                dy = librosa.resample(dy, orig_sr=dsr, target_sr=22050)
            env, tenv = _drum_onset_bands(dy.astype(np.float32), 22050)
    except Exception as exc:                                   # pragma: no cover
        print(f"    (batterie indisponible : {exc})")

    S = {"n": n, "grid": grid, "res": {}}
    for res in (1, 2, 4):
        e = cases(grid, n, res)
        m = len(e) - 1
        P = pool(cap["arr"], cap["times"], e)
        d = {}
        d["basse"] = P[:, :12]
        d["harmonie"] = P[:, 12:]
        d["accords"] = np.asarray(HS.harmonic_vectors(cap["triad"], list(e)), float)
        d["timbre"] = pool(mfcc.T, tf[:mfcc.shape[1]], e)

        # voix : trois lectures de la même mélodie
        cl = np.zeros((m, 12)); ha = np.zeros((m, 37)); at = np.zeros((m, 4))
        dens = np.zeros(m)
        lo = min([int(x[2]) for x in notes], default=48)
        for t0, dur, mid in notes:
            i0 = int(np.searchsorted(e, t0) - 1)
            i1 = int(np.searchsorted(e, t0 + dur) - 1)
            for i in range(max(0, i0), min(m, i1 + 1)):
                a, z = max(t0, e[i]), min(t0 + dur, e[i + 1])
                if z > a:
                    cl[i, int(mid) % 12] += (z - a)
                    dens[i] += (z - a) / max(1e-9, e[i + 1] - e[i])
                    p = int(mid) - lo
                    if 0 <= p < 37:
                        ha[i, p] += (z - a)
            if 0 <= i0 < m:
                f = (t0 - e[i0]) / max(1e-9, e[i0 + 1] - e[i0])
                at[i0, min(3, int(4 * f))] += 1.0
        d["voix_classe"], d["voix_hauteur"], d["voix_attaque"] = cl, ha, at

        # intensité
        d["bandes"] = pool(bandes.T, tf[:bandes.shape[1]], e)
        rr = pool(rms_db[:, None], tf[:len(rms_db)], e)[:, 0]
        ff = pool(flux[:, None], tf[:len(flux)], e)[:, 0]
        d["_rms"] = rr
        d["_flux"] = ff
        d["_dens"] = dens
        d["_nrj"] = np.clip(np.linalg.norm(P, axis=1), 1e-9, None)

        if env is not None:
            from rhythm_ssm import build_slot_patches
            V = build_slot_patches(env, tenv, list(e), sub_steps=8, shifts=(0.0,))[:, 0, :]
            d["rythme"] = np.asarray(V, float)
            d["_bat"] = pool(env.mean(0)[:, None], tenv, e)[:, 0]
        S["res"][res] = d
    return S


# ── les critères ────────────────────────────────────────────────────────────

def criteres(S):
    """[(signal, nom, glose, res, courbe)] — un critère par ligne de la page.

    L'ordre de la liste n'a pas d'importance : la page trie par nombre de
    frontières trouvées, ou par signal.
    """
    R = S["res"]
    C = []

    def add(sig, nom, glose, res, courbe):
        c = np.asarray(courbe, float)
        if np.isfinite(c).any() and float(np.nanmax(c) - np.nanmin(c)) > 1e-12:
            C.append((sig, nom, glose, res, np.nan_to_num(c)))

    # ── voix ────────────────────────────────────────────────────────────────
    add("voix", "damier · classes · ½mes · cos · L4",
        "les douze demi-tons chantés, pondérés par la durée. La matrice voix du "
        "projet, lue en frontières au lieu de reprises.",
        2, damier(matrice(R[2]["voix_classe"]), 8))
    add("voix", "damier · hauteur réelle · temps · cos · L2",
        "l'octave n'est plus jetée : un refrain chanté plus haut se sépare d'un "
        "couplet sur les mêmes notes.",
        4, damier(matrice(R[4]["voix_hauteur"]), 8))
    add("voix", "damier · classes · mes · harm · L4",
        "le produit HARMONIQUE au lieu du cosinus : deux phrases parentes (une "
        "quinte, une tierce d'écart) cessent d'être étrangères.",
        1, damier(matrice(R[1]["voix_classe"], "harm"), 4))
    add("voix", "damier · attaques seules · ½mes · cos · L4",
        "où tombent les attaques dans la case, SANS aucune hauteur — la figure "
        "rythmique du chant, qui change souvent au refrain.",
        2, damier(matrice(R[2]["voix_attaque"]), 8))
    add("voix", "rupture de ligne · classes · ½mes",
        "la case change de PARTENAIRES dans le morceau. Le seul critère qui voit "
        "une boucle identique se mettre à répondre à autre chose.",
        2, rupture(matrice(R[2]["voix_classe"])))
    add("voix", "saut · densité de chant · mes",
        "combien de temps on chante dans la mesure, et rien d'autre. Aucune "
        "matrice : c'est l'entrée et la sortie du chant qu'on lit ici.",
        1, saut(R[1]["_dens"], 4))
    add("voix", "saut · densité de chant · ½mes",
        "le même, deux fois plus fin — attrape le silence d'une demi-mesure "
        "avant un refrain.",
        2, saut(R[2]["_dens"], 4))
    add("voix", "contraste · hauteur réelle · mes · L4",
        "sans matrice : le registre moyen des 4 mesures d'avant contre celui des "
        "4 d'après. Grossier, mais insensible au détail.",
        1, contraste(R[1]["voix_hauteur"], 4))

    # ── harmonie (NNLS aigu) ────────────────────────────────────────────────
    add("harmonie", "damier · ½mes · cos · L4",
        "les 12 cases aiguës du NNLS : le VOICING, pas seulement l'accord.",
        2, damier(matrice(R[2]["harmonie"]), 8))
    add("harmonie", "damier · ½mes · harm · L4",
        "le même avec le produit harmonique — un Sib et un Gm ne sont plus deux "
        "vecteurs étrangers.",
        2, damier(matrice(R[2]["harmonie"], "harm"), 8))
    add("harmonie", "damier · mes · corr · L8",
        "colonnes centrées et grande échelle : ce qui reste quand on retire "
        "l'harmonie MOYENNE du morceau.",
        1, damier(matrice(R[1]["harmonie"], "corr"), 8))
    add("harmonie", "damier · ½mes · pond · L4",
        "le cosinus multiplié par le rapport d'intensité : mêmes accords joués "
        "doucement puis fort = ce n'est plus la même case.",
        2, damier(matrice(R[2]["harmonie"], "pond", R[2]["_nrj"]), 8))
    add("harmonie", "rupture de ligne · ½mes",
        "la boucle change de partenaire — le cas des morceaux où l'harmonie est "
        "la même partout mais l'ARRANGEMENT des reprises change.",
        2, rupture(matrice(R[2]["harmonie"])))
    add("harmonie", "damier · temps · cos · L2",
        "au temps : la granularité qui voit un changement d'accord isolé, donc "
        "beaucoup de faux pics — à confronter aux autres.",
        4, damier(matrice(R[4]["harmonie"]), 8))

    # ── basse ───────────────────────────────────────────────────────────────
    add("basse", "damier · ½mes · cos · L4",
        "les 12 cases graves du NNLS : quelle note est à la basse.",
        2, damier(matrice(R[2]["basse"]), 8))
    add("basse", "damier · mes · harm · L4",
        "la basse au produit harmonique : une pédale de dominante et sa tonique "
        "se répondent au lieu de s'opposer.",
        1, damier(matrice(R[1]["basse"], "harm"), 4))
    add("basse", "rupture de ligne · ½mes",
        "la ligne de basse change de partenaire dans le morceau — souvent le "
        "signe d'un pont.",
        2, rupture(matrice(R[2]["basse"])))
    add("basse", "contraste · ½mes · L4",
        "sans matrice : les 4 mesures de basse d'avant contre celles d'après.",
        2, contraste(R[2]["basse"], 8))

    # ── accords (postérieur musx, le substrat de la prod) ───────────────────
    add("accords", "damier · ½mes · cos · L4",
        "LE SUBSTRAT DE LA PROD : le postérieur musx projeté sur 12 hauteurs. "
        "C'est le point de comparaison de toute la page.",
        2, damier(matrice(R[2]["accords"]), 8))
    add("accords", "damier · ½mes · harm · L4",
        "le même en produit harmonique.",
        2, damier(matrice(R[2]["accords"], "harm"), 8))
    add("accords", "damier · mes · cos · L8",
        "à la mesure et à grande échelle : on cherche la SECTION, pas la boucle "
        "de deux mesures.",
        1, damier(matrice(R[1]["accords"]), 8))
    add("accords", "rupture de ligne · ½mes",
        "l'accord ne change pas, mais il ne répond plus aux mêmes mesures.",
        2, rupture(matrice(R[2]["accords"])))

    # ── rythme (batterie séparée) ───────────────────────────────────────────
    if "rythme" in R[2]:
        add("rythme", "damier · motif batterie · ½mes · cos · L4",
            "le motif d'attaques en 3 bandes (grave/caisse/cymbales) : deux "
            "sections peuvent partager les accords et pas le groove.",
            2, damier(matrice(R[2]["rythme"]), 8))
        add("rythme", "damier · motif batterie · mes · corr · L8",
            "centré et à grande échelle : le groove MOYEN du morceau retiré.",
            1, damier(matrice(R[1]["rythme"], "corr"), 8))
        add("rythme", "saut · énergie batterie · mes",
            "aucune matrice : la batterie entre, sort, ou double. C'est le "
            "marqueur de section le plus brutal de la pop.",
            1, saut(R[1]["_bat"], 4))
        add("rythme", "remue · fill de batterie · ½mes",
            "la variabilité locale, pas le niveau : un FILL annonce la frontière "
            "juste AVANT, pas dessus.",
            2, bouge(R[2]["_bat"], 4))
        add("rythme", "rupture de ligne · batterie · ½mes",
            "le groove change de partenaire dans le morceau.",
            2, rupture(matrice(R[2]["rythme"])))

    # ── timbre ──────────────────────────────────────────────────────────────
    add("timbre", "damier · MFCC · ½mes · corr · L4",
        "QUI joue, pas quoi. Le substrat classique du MIR pour couper "
        "couplet/refrain : un refrain ajoute des instruments.",
        2, damier(matrice(R[2]["timbre"], "corr"), 8))
    add("timbre", "damier · MFCC · mes · corr · L8",
        "le même à grande échelle : les grands pans d'arrangement.",
        1, damier(matrice(R[1]["timbre"], "corr"), 8))
    add("timbre", "damier · MFCC · temps · corr · L2",
        "au temps : voit les entrées d'instrument isolées.",
        4, damier(matrice(R[4]["timbre"], "corr"), 8))
    add("timbre", "saut · profil MFCC · mes",
        "sans matrice : la distance entre le timbre moyen d'avant et d'après.",
        1, saut(R[1]["timbre"], 4))
    add("timbre", "rupture de ligne · MFCC · ½mes",
        "l'arrangement change de partenaire — le cas des reprises de refrain de "
        "plus en plus fournies.",
        2, rupture(matrice(R[2]["timbre"], "corr")))

    # ── intensité ───────────────────────────────────────────────────────────
    add("intensité", "saut · RMS · mes",
        "le volume, en dB, et son saut. Le critère le plus bête de la page — et "
        "sur beaucoup de morceaux pop, le plus juste.",
        1, saut(R[1]["_rms"], 4))
    add("intensité", "saut · RMS · ½mes",
        "le même, deux fois plus fin.",
        2, saut(R[2]["_rms"], 4))
    add("intensité", "saut · RMS · temps",
        "au temps : attrape les arrêts nets d'une seule noire.",
        4, saut(R[4]["_rms"], 8))
    add("intensité", "saut · 3 bandes · ½mes",
        "grave / médium / aigu séparément : une entrée de basse et une entrée de "
        "cymbales ne se voient pas au même endroit du spectre.",
        2, saut(R[2]["bandes"], 4))
    add("intensité", "damier · 3 bandes · ½mes · corr · L4",
        "l'équilibre spectral en matrice : la couleur globale du mix.",
        2, damier(matrice(R[2]["bandes"], "corr"), 8))
    add("intensité", "saut · flux spectral · mes",
        "la densité d'attaques de TOUT le mix (pas seulement la batterie).",
        1, saut(R[1]["_flux"], 4))
    add("intensité", "remue · RMS · ½mes",
        "ça remue sans changer de niveau : les crescendos et les breaks.",
        2, bouge(R[2]["_rms"], 4))
    return C


# ── vérité terrain ──────────────────────────────────────────────────────────

def frontieres(stem, n):
    """Les frontières validées par Louis, en mesures (0 exclu)."""
    p = ANN / f"{stem}.json"
    if not p.exists():
        return [], []
    d = json.loads(p.read_text())
    secs = d.get("sections", [])
    b = sorted({int(s["b0"]) for s in secs if 0 < int(s["b0"]) < n})
    return b, secs


# ── rendu ───────────────────────────────────────────────────────────────────

def _rgb(t):
    """La rampe séquentielle du projet, échantillonnée en t ∈ [0,1]."""
    cols = [tuple(int(c[i:i + 2], 16) for i in (1, 3, 5)) for c in RAMP]
    x = np.clip(t, 0, 1) * (len(cols) - 1)
    i = np.clip(x.astype(int), 0, len(cols) - 2)
    f = (x - i)[:, None]
    a = np.array(cols)[i]
    b = np.array(cols)[i + 1]
    return (a * (1 - f) + b * f)


GAMMA = 3.5      # ne noircir que le HAUT de la courbe (voir bande_png)


def bande_png(courbe01):
    """La courbe en PNG d'une seule ligne de pixels — étiré en CSS, pixelisé.

    Une case = un pixel : la granularité affichée est EXACTEMENT celle du
    critère, sans interpolation qui mentirait sur sa finesse.

    Les rangs sont élevés à la puissance `GAMMA` AVANT la couleur. Sans ça, une
    courbe en rangs a la moitié de ses cases au-dessus de la médiane, donc la
    moitié de la bande est sombre par construction et toutes les lignes se
    ressemblent — essayé, illisible. Avec, seul le dernier quart d'une courbe
    marque, l'encre reste le même budget pour tout le monde, et on lit des pics.
    """
    from PIL import Image
    arr = _rgb(np.asarray(courbe01, float) ** GAMMA).astype(np.uint8)[None, :, :]
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def bundle(stem, rebuild=False):
    """Les courbes + les pics + les pastilles d'un morceau, en cache."""
    p = CACHE / f"{stem}_v{VERSION}.pkl"
    if p.exists() and not rebuild:
        with p.open("rb") as f:
            return pickle.load(f)
    t0 = time.time()
    S = signaux(stem)
    n = S["n"]
    C = criteres(S)
    gt, secs = frontieres(stem, n)
    k = max(2, len(gt))
    lignes = []
    for sig, nom, glose, res, courbe in C:
        c01 = rank01(courbe)
        pk = pics(courbe, res, k)
        pk_mes = [i / res for i in pk]
        etat = []
        for g in gt:
            d = min([abs(pm - g) for pm in pk_mes], default=99)
            etat.append(2 if d <= 0.5 else (1 if d <= 1.5 else 0))
        touche = [min([abs(pm - g) for g in gt], default=99) <= 1.5 for pm in pk_mes]
        lignes.append({"sig": sig, "nom": nom, "glose": glose, "res": res,
                       "png": bande_png(c01), "pics": pk_mes, "touche": touche,
                       "etat": etat,
                       "score": sum(1 for e in etat if e == 2) + 0.5 * sum(1 for e in etat if e == 1)})
    b = {"stem": stem, "n": n, "grid": [float(x) for x in S["grid"]],
         "gt": gt, "secs": secs, "lignes": lignes, "sec": time.time() - t0}
    with p.open("wb") as f:
        pickle.dump(b, f)
    return b


def lane_ia(stem, n, gt):
    """La ligne « IA » : des frontières proposées par un LLM, s'il y en a.

    Le fichier `scratchpad/ia_sections/<stem>.json` porte `{"bars": [...],
    "note": "..."}` — écrit à la main par un modèle qui n'a lu qu'un RÉSUMÉ
    TEXTE du morceau (accords par mesure, chant présent ou non, niveau sonore),
    jamais l'audio. C'est une ligne comme les autres, avec le même budget de
    pics, pour qu'elle soit comparable aux trente critères de signal.

    `bars` est en NUMÉROS DE MESURE (base 1), comme la règle de la page et comme
    `resume_texte.py` : c'est ce que le modèle a sous les yeux. Les frontières de
    Louis, elles, sont des INDICES (base 0). Sans le `-1` ici, les treize
    propositions tombaient toutes à une mesure près et pas une seule sur la
    frontière — un décalage constant qui ressemblait à une performance médiocre
    au lieu d'un bug de convention.
    """
    p = IA / f"{stem}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    bars = [int(b) - 1 for b in d.get("bars", []) if 0 < int(b) - 1 < n]
    if not bars:
        return None
    c = np.zeros(n)
    for b in bars:
        c[b] = 1.0
    etat = []
    for g in gt:
        dd = min([abs(b - g) for b in bars], default=99)
        etat.append(2 if dd <= 0.5 else (1 if dd <= 1.5 else 0))
    return {"sig": "IA", "nom": d.get("nom", "un LLM lit le résumé texte du morceau"),
            "glose": d.get("note", ""), "res": 1, "png": bande_png(c),
            "pics": [float(b) for b in bars],
            "touche": [min([abs(b - g) for g in gt], default=99) <= 1.5 for b in bars],
            "etat": etat,
            "score": sum(1 for e in etat if e == 2) + 0.5 * sum(1 for e in etat if e == 1)}


def section_html(b, titre):
    n, gt, lignes = b["n"], b["gt"], b["lignes"]
    ia = lane_ia(b["stem"], n, gt)
    if ia:
        lignes = lignes + [ia]
    lignes = sorted(lignes, key=lambda L: -L["score"])
    pas = 4 if n <= 60 else (8 if n <= 140 else 16)

    # la bande des sections validées, en haut de la pile
    blocs = []
    for s in b["secs"]:
        b0, b1 = int(s["b0"]), int(s["b1"])
        w = (b1 - b0 + 1) / n * 100
        blocs.append(f'<i style="left:{b0 / n * 100:.4f}%;width:{w:.4f}%">'
                     f'{s.get("label", "")}</i>')
    regle = "".join(f'<i style="left:{i / n * 100:.4f}%">{i + 1}</i>'
                    for i in range(0, n, pas))
    gtl = "".join(f'<i class=gt style="left:{g / n * 100:.4f}%"></i>' for g in gt)

    rows = []
    for L in lignes:
        dots = "".join(f'<i class="d e{e}"></i>' for e in L["etat"])
        pk = "".join(f'<i class="pk{" on" if t else ""}" '
                     f'style="left:{p / n * 100:.4f}%"></i>'
                     for p, t in zip(L["pics"], L["touche"]))
        gl = L["glose"].replace('"', "'")
        rows.append(
            f'<div class=lane data-sig="{L["sig"]}" data-score="{L["score"]}" '
            f'title="{L["sig"]} — {L["nom"]}&#10;{gl}">'
            f'<div class=lab><b style="color:{TEINTE.get(L["sig"], INK)}">{L["sig"]}</b>'
            f'<span>{L["nom"]}</span></div>'
            f'<div class=dots title="frontière par frontière : pleine = à la mesure,'
            f' cercle = à une mesure près, vide = ratée">{dots}</div>'
            f'<div class=strip><img src="data:image/png;base64,{L["png"]}" alt="">'
            f'{pk}</div>'
            f'<div class=glose>{L["glose"]}</div></div>')

    return (
        f'<section data-grid="{json.dumps(b["grid"])}" data-n="{n}">'
        f'<div class=hd><h2>{titre}</h2>'
        f'<span class=sub>{n} mesures · {len(gt)} frontières · '
        f'{len(lignes)} critères · budget {max(2, len(gt))} pics chacun</span></div>'
        f'<div class=bar><button class=pp>▶</button><span class=pos>mes. 1</span>'
        f'<label><input type=checkbox class=blind> aveugle '
        f'<small>(cache mes frontières)</small></label>'
        f'<label><input type=checkbox class=grp> grouper par signal</label>'
        f'<span class=hint>clique n\'importe où dans une bande → le son part de '
        f'cette mesure</span></div>'
        f'<div class=stack>'
        f'<div class=lane><div class=lab></div><div class=dots></div>'
        f'<div class=secs>{"".join(blocs)}</div><div class=glose></div></div>'
        f'<div class=lane><div class=lab></div><div class=dots></div>'
        f'<div class=rule>{regle}</div><div class=glose></div></div>'
        f'{"".join(rows)}'
        f'<div class=ov>{gtl}<div class=cur></div></div>'
        f'</div>'
        f'<audio preload=metadata playsinline src="../audio/{b["stem"]}.m4a"></audio>'
        f'</section>')


CSS = """
body{margin:0;background:#e7e0d0;font:15px/1.55 -apple-system,system-ui,sans-serif;color:INK}
.wrap{max-width:1500px;margin:0 auto;padding:22px 14px 80px}
h1{font:italic 600 26px Georgia,serif;margin:0 0 6px}
.lede{color:#6f6857;font-size:13.5px;margin-bottom:16px;max-width:1020px}
.lede b{color:INK} .lede code{background:#f2ede0;padding:1px 4px;border-radius:4px}
section{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;
  padding:10px 14px 14px;margin-bottom:14px;--lab:288px;--dots:104px;--glose:0px}
.hd{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap}
h2{font:700 18px system-ui;margin:0;color:#8a2b2b}
.sub{font:500 12px system-ui;color:#8a8371}
.bar{display:flex;align-items:center;gap:14px;margin:8px 0 10px;position:sticky;top:0;
  background:#fffdf6;padding:6px 0;z-index:6;border-bottom:1px solid #f0e9d8;flex-wrap:wrap}
.pp{width:34px;height:34px;border-radius:50%;border:1px solid #d8cfb4;background:#f7f3e9;
  font-size:13px;cursor:pointer;flex:none}
.pos{font:600 12px ui-monospace,monospace;min-width:130px}
.bar label{font:500 12px system-ui;color:#6f6857;cursor:pointer;user-select:none}
.bar small{color:#a89f8c}
.hint{font:500 11.5px system-ui;color:#a89f8c;margin-left:auto}
.stack{position:relative}
.lane{display:grid;grid-template-columns:var(--lab) var(--dots) 1fr;align-items:center;
  height:22px;border-bottom:1px solid #f6f1e4}
.lane:hover{background:#fbf7ea}
.lab{font:500 11px system-ui;color:#6f6857;padding-right:8px;white-space:nowrap;
  overflow:hidden;text-overflow:ellipsis}
.lab b{font:700 11px system-ui;margin-right:6px}
.dots{display:flex;gap:2px;align-items:center;padding-right:10px;overflow:hidden}
.d{width:6px;height:6px;border-radius:50%;flex:none;box-sizing:border-box}
.d.e2{background:#2f6d3a}
.d.e1{border:1.5px solid #2f6d3a}
.d.e0{border:1.5px solid #ddd5c0}
.strip{position:relative;height:17px}
.strip img{width:100%;height:17px;display:block;border-radius:3px;
  image-rendering:pixelated;image-rendering:crisp-edges}
.pk{position:absolute;bottom:0;width:0;height:0;transform:translateX(-4px);
  border-left:4px solid transparent;border-right:4px solid transparent;
  border-bottom:7px solid #b9ae94;opacity:.9}
.pk.on{border-bottom-color:#e8a33d;border-bottom-width:8px;opacity:1}
.secs{position:relative;height:18px}
.secs i{position:absolute;top:0;height:18px;font:700 10px system-ui;color:#8a6a4a;
  background:#f4ecd9;border-left:1px solid #e0d3b6;box-sizing:border-box;
  padding-left:3px;line-height:18px;overflow:hidden;border-radius:2px}
.rule{position:relative;height:14px}
.rule i{position:absolute;top:0;font:500 9.5px ui-monospace,monospace;color:#a89f8c;
  transform:translateX(-50%)}
.ov{position:absolute;left:calc(var(--lab) + var(--dots));right:0;top:0;bottom:0;
  cursor:crosshair;z-index:4}
.gt{position:absolute;top:0;bottom:0;width:1px;background:GTLINE;opacity:.75}
.cur{position:absolute;top:0;bottom:0;width:2px;background:#c1121f;display:none;
  box-shadow:0 0 0 1px rgba(255,255,255,.7)}
.stack.blind .gt,.stack.blind .dots,.stack.blind .secs i{visibility:hidden}
.glose{display:none;font:500 11px system-ui;color:#8a8371}
audio{display:none}
@media(max-width:900px){section{--lab:120px;--dots:0px}.dots{display:none}
  .lab{font-size:10px}.hint{display:none}}
"""

JS = """
document.querySelectorAll("section").forEach(function(sec){
  var au = sec.querySelector("audio"); if(!au) return;
  var G = JSON.parse(sec.dataset.grid), n = +sec.dataset.n;
  var stack = sec.querySelector(".stack"), ov = sec.querySelector(".ov");
  var cur = sec.querySelector(".cur"), pp = sec.querySelector(".pp");
  var pos = sec.querySelector(".pos"), timer = null;
  function b2t(f){ var i=Math.max(0,Math.min(n-1,Math.floor(f)));
    return G[i]+(f-i)*(G[i+1]-G[i]); }
  function t2b(t){ if(t<=G[0])return 0; if(t>=G[n])return n;
    var lo=0,hi=n; while(hi-lo>1){var m=(lo+hi)>>1; G[m]<=t?lo=m:hi=m;}
    return lo+(t-G[lo])/(G[lo+1]-G[lo]); }
  function fmt(s){ return Math.floor(s/60)+":"+String(Math.floor(s%60)).padStart(2,"0"); }
  // Pas de requestAnimationFrame ici : sur iPhone une boucle rAF empêche WebKit
  // de démarrer le moteur audio (piège déjà payé, cf. app_shell). timeupdate +
  // un intervalle court suffisent, et le trait reste lisse.
  function draw(){ var f=t2b(au.currentTime); cur.style.display="block";
    cur.style.left="calc("+(100*f/n)+"% - 1px)";
    pos.textContent="mes. "+(Math.floor(f)+1)+" · "+fmt(au.currentTime); }
  au.addEventListener("timeupdate", draw);
  au.addEventListener("play", function(){
    document.querySelectorAll("audio").forEach(function(a){ if(a!==au) a.pause(); });
    pp.textContent="❚❚"; clearInterval(timer); timer=setInterval(draw,90); });
  au.addEventListener("pause", function(){ pp.textContent="▶";
    clearInterval(timer); draw(); });
  pp.onclick=function(){ au.paused ? au.play().catch(function(){}) : au.pause(); };
  ov.onclick=function(e){
    var r=ov.getBoundingClientRect();
    var t=b2t(n*(e.clientX-r.left)/r.width);
    var seek=function(){ try{ au.currentTime=t; }catch(err){} draw(); };
    if(au.readyState>=1) seek(); else au.addEventListener("loadedmetadata",seek,{once:true});
    au.play().catch(function(){}); };
  sec.querySelector(".blind").onchange=function(){ stack.classList.toggle("blind", this.checked); };
  // Regrouper par signal, ou trier par nombre de frontières trouvées : les deux
  // lectures que Louis a demandées ("lesquels marchent" / "quel signal porte ce
  // morceau"). Les deux premières lignes (sections, règle) ne bougent jamais.
  var lanes = Array.prototype.slice.call(stack.querySelectorAll(".lane[data-sig]"));
  var ordreSig = ["voix","harmonie","basse","accords","rythme","timbre","intensité","IA"];
  sec.querySelector(".grp").onchange=function(){
    var g = this.checked;
    lanes.slice().sort(function(a,b){
      if(g){ var d = ordreSig.indexOf(a.dataset.sig) - ordreSig.indexOf(b.dataset.sig);
             if(d) return d; }
      return (+b.dataset.score) - (+a.dataset.score);
    }).forEach(function(l){ stack.insertBefore(l, ov); });
  };
});
"""


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    stems = args or TROIS
    rebuild = "--rebuild" in sys.argv
    from ssm_zoo import SONGS
    titres = dict(SONGS)
    body = []
    for stem in stems:
        if not (AUDIO / f"{stem}.m4a").exists():
            print(f"  ?? pas d'audio pour {stem}")
            continue
        b = bundle(stem, rebuild=rebuild)
        body.append(section_html(b, titres.get(stem, stem.replace("_", " "))))
        best = sorted(b["lignes"], key=lambda L: -L["score"])[:3]
        print(f"  ok {titres.get(stem, stem)} — {b['n']} mes., {len(b['gt'])} frontières, "
              f"{len(b['lignes'])} critères, {b['sec']:.0f} s")
        for L in best:
            print(f"       {L['score']:>4.1f}/{len(b['gt'])}  {L['sig']:<10} {L['nom']}")

    html = f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Les critères de changement de section</title>
<style>{CSS.replace('INK', INK).replace('GTLINE', GT_LINE)}</style></head>
<body><div class=wrap>
<h1>Qu'est-ce qui marque un changement de section&nbsp;?</h1>
<div class=lede>Une ligne = <b>un critère</b>. Sombre = « ici, ça change ».
Toutes les lignes partagent le même axe de mesures ; la tête de lecture les
traverse toutes ; <b>un clic n'importe où lance le son à cette mesure</b>.<br><br>
Chaque critère propose <b>exactement autant de frontières qu'il y en a</b> dans
le morceau (ses plus hauts pics, espacés d'au moins deux mesures) — sans ce
budget commun, un critère qui pique partout tomberait juste partout. Les
<b>triangles</b> sont ses propositions : ambre = elle tombe sur une des tiennes,
gris = non. Les <b>pastilles</b> à gauche disent, frontière par frontière :
pleine = trouvée à la mesure près, cercle = à une mesure près, vide = ratée.
<b>Lire une ligne = compter ses pastilles pleines.</b><br><br>
Les lignes sont triées par nombre de frontières trouvées. <b>« Grouper par
signal »</b> répond à l'autre question — <i>quel signal porte CE morceau</i> —
et <b>« aveugle »</b> cache tes frontières pour juger la courbe sans savoir la
réponse.<br><br>
<b>Ce n'est pas un modèle.</b> Aucune fusion, rien de branché : chaque critère
est seul avec lui-même, pour qu'on décide à l'oreille lesquels méritent d'être
combinés, et sur quel style.</div>
{''.join(body)}</div>
<script>{JS}</script></body></html>"""
    OUT.write_text(html)
    print(f"wrote {OUT}  ({OUT.stat().st_size // 1024} KB)")
    print("   http://100.89.209.63:7772/plots/criteres_sections.html")


if __name__ == "__main__":
    main()
