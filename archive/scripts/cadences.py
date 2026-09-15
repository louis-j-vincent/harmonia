"""La CADENCE : « la phrase finit sur quelque chose de cohérent », mesuré.

    .venv/bin/python scripts/cadences.py --table     # le tableau sur les 18
    .venv/bin/python scripts/cadences.py --premisse  # les taux bruts, sans détecteur
    .venv/bin/python scripts/cadences.py [<stem>…]   # -> /plots/cadences.html

Louis, 2026-08-14 : « Il y a une vraie logique harmonique : en général une
section a un sens mélodique, la phrase finit sur quelque chose de cohérent, je
ne sais pas comment l'expliquer. »

Ça porte un nom : la **cadence**. Une section ne s'arrête pas au milieu d'une
progression, elle se résout. Ce fichier transforme ça en courbes comparables.

CE QUI DISTINGUE UNE CADENCE D'UN CHANGEMENT. Les 39 critères de
`criteres_sections.py` répondent tous à « ici, ça change » : ils comparent
l'avant et l'après et cherchent une DIFFÉRENCE. Une cadence ne dit pas que ça
change, elle dit que ça **finit** — et elle peut arriver sans aucun changement
de timbre, de volume ni d'instrumentation. C'est une information de nature
différente, pas une variante de plus.

LE DÉCALAGE, mesuré et pas supposé (voir le test de prémisse ci-dessous) : la
résolution tombe sur le PREMIER TEMPS DE LA SECTION SUIVANTE, pas sur la
dernière mesure de celle qui finit. Le V est la dernière mesure, le I est la
frontière. Donc toutes les courbes d'ici sont indexées de sorte qu'un pic à la
mesure `b` prédit une frontière à la mesure `b` — décalage nul, rien à corriger.

LE BUG DE CALIBRATION QUI BLOQUAIT TOUT (trouvé le 2026-08-14). Le chroma NNLS a
l'index **0 = LA**, pas DO (`harmonia_min/nnls_features.py:11`). Aucun usage
précédent ne s'en apercevait : SSM, produits scalaires et noyau harmonique sont
tous invariants par transposition. Dès qu'on compare une basse à un nom d'accord
musx, ça compte : `racine_musx − argmax_basse ≡ 9` demi-tons sur 75 % des
mesures du corpus, et « la racine est la tonique locale » tombait à 2 % au lieu
de 43 %. Conversion : `pc = (index_nnls + NNLS_A) % 12`.

CE QUE ÇA NE FAIT PAS. Aucune fusion avec les 39 critères, rien de branché sur
la prod, et surtout : **le rappel est la métrique, pas la précision**. Les
annotations de Louis ne sont pas exhaustives (« je n'ai pas noté tous les
changements de section »), donc un pic hors annotation n'est PAS un faux
positif — il est peut-être une frontière qu'il n'a pas notée. Le coût est
compté en densité de pics (un pic toutes les combien de mesures), pas en
précision.
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

CACHE = HERE / "scratchpad" / "cadence_cache"
CACHE.mkdir(exist_ok=True)
OUT = HERE / "docs" / "plots" / "cadences.html"
AUDIO = HERE / "docs" / "audio"
ANN = HERE / "harmonia_min" / "state" / "sections"

NNLS_A = 9        # le chroma NNLS commence au LA — voir l'entête
NOMS = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
PC = {n: i for i, n in enumerate(NOMS)}

# Krumhansl–Kessler, les deux profils de tonalité
KMAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KMIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

QUATRE = ["maroon_5_this_love",
          "norah_jones_don_t_know_why",
          "the_police_every_breath_you_take_official_music_video",
          "bobby_hebb_sunny_official_audio"]


# ── les données ─────────────────────────────────────────────────────────────

def liste_validee():
    return sorted(f.stem for f in ANN.glob("*.json")
                  if json.loads(f.read_text()).get("validated")
                  and (AUDIO / f"{f.stem}.m4a").exists())


def gt_de(stem):
    d = json.loads((ANN / f"{stem}.json").read_text())
    return sorted({int(s["b0"]) for s in d["sections"] if 0 < int(s["b0"]) < d["n"]})


def brut(stem, rebuild=False):
    """Les substrats en cache : basse/harmonie/accords par case, noms, chant."""
    p = CACHE / f"{stem}.pkl"
    if p.exists() and not rebuild:
        with p.open("rb") as f:
            return pickle.load(f)
    import order_bundle
    from ssm_zoo import capture
    from criteres_sections import cases, pool
    from resume_texte import accord_par_mesure
    from licks import notes_de
    import harmonia_min.harmonic_sections as HS

    bd = order_bundle.get(stem)
    n, grid = bd["n"], np.asarray(bd["grid"], float)
    cap = capture(stem)
    d = {"n": n, "grid": grid}
    for res in (1, 2, 4):
        e = cases(grid, n, res)
        P = pool(cap["arr"], cap["times"], e)
        d[f"basse{res}"] = P[:, :12]
        d[f"harm{res}"] = P[:, 12:]
        d[f"acc{res}"] = np.asarray(HS.harmonic_vectors(cap["triad"], list(e)), float)
    d["noms"] = accord_par_mesure(cap["triad"], grid)
    d["noms2"] = accord_par_mesure(cap["triad"], cases(grid, n, 2))
    d["notes"] = [(float(a), float(b), float(c)) for a, b, c in notes_de(stem)]
    with p.open("wb") as f:
        pickle.dump(d, f)
    return d


def racine_nom(nm):
    """« Abm » -> 8. None pour « N » (pas d'accord) et « ? »."""
    if not nm or nm in ("?", "N"):
        return None
    for k in (2, 1):
        if nm[:k] in PC:
            return PC[nm[:k]]
    return None


def mineur_nom(nm):
    return bool(nm) and nm.endswith("m")


def chroma_pc(X):
    """Le chroma NNLS ramené en classes de hauteur (do = 0)."""
    return np.roll(np.asarray(X, float), NNLS_A, axis=1)


def tonique(chroma, fen=16):
    """[(tonique, mode)] par case — Krumhansl sur une fenêtre glissante.

    `fen=0` -> une seule tonalité pour tout le morceau (tonique globale). La clé
    LOCALE de ce projet (`harmonia/theory/local_key.py`) est une continuité
    causale sur des jetons d'accords ; ici on n'a que du chroma, et la fenêtre
    glissante suffit à ce qu'on lui demande — savoir où est le « chez soi » à ce
    moment du morceau, pas nommer la modulation.
    """
    n = len(chroma)
    M = (KMAJ - KMAJ.mean()) / KMAJ.std()
    m = (KMIN - KMIN.mean()) / KMIN.std()
    out = []
    for b in range(n):
        if fen:
            a, z = max(0, b - fen // 2), min(n, b + fen // 2 + 1)
        else:
            a, z = 0, n
        v = chroma[a:z].sum(0)
        if v.sum() <= 1e-9 or v.std() <= 1e-9:
            out.append((None, None)); continue
        v = (v - v.mean()) / v.std()
        sc = [(float(np.dot(np.roll(M, k), v)), k, "maj") for k in range(12)]
        sc += [(float(np.dot(np.roll(m, k), v)), k, "min") for k in range(12)]
        _s, k, mo = max(sc)
        out.append((k, mo))
    return out


# ── le modèle de progression, en leave-one-song-out ─────────────────────────

def jetons(stem, D, T):
    """La suite d'accords du morceau en DEGRÉS de la clé locale : (degré, m?).

    Transposition-invariant : un II–V–I en fa et un II–V–I en sol sont le même
    objet. C'est ce qui permet d'apprendre sur 17 morceaux et de lire le 18e.
    """
    d = D[stem]
    out = []
    for b in range(d["n"]):
        r = racine_nom(d["noms"][b]); k = T[stem][b][0]
        out.append(None if (r is None or k is None)
                   else ((r - k) % 12, mineur_nom(d["noms"][b])))
    return out


def bigramme(stems, J, alpha=0.5):
    """log P(suivant | précédent) sur les degrés, lissé — appris hors morceau."""
    from collections import Counter
    C = Counter(); M = Counter()
    for s in stems:
        seq = [x for x in J[s] if x is not None]
        for a, b in zip(seq, seq[1:]):
            C[(a, b)] += 1; M[a] += 1
    V = 24

    def lp(a, b):
        return float(np.log((C[(a, b)] + alpha) / (M[a] + alpha * V)))
    return lp


# ── les formulations ────────────────────────────────────────────────────────

def _pcmass(v, k):
    s = float(v.sum())
    return float(v[k]) / s if s > 1e-9 else 0.0


def formulations(stem, D, T16, T0, LP):
    """[(clé, groupe, nom, glose, courbe)] — un pic en `b` prédit une frontière en `b`."""
    d = D[stem]
    n = d["n"]
    B1 = chroma_pc(d["basse1"]); B2 = chroma_pc(d["basse2"])
    R = [racine_nom(x) for x in d["noms"]]
    R2 = [racine_nom(x) for x in d["noms2"]]
    K = [t[0] for t in T16[stem]]
    KG = T0[stem][0][0]
    F = []

    # ── 1. la basse ────────────────────────────────────────────────────────
    q = np.zeros(n)
    for b in range(1, n):
        q[b] = sum(_pcmass(B1[b - 1], (k + 7) % 12) * _pcmass(B1[b], k) for k in range(12))
    F.append(("q_nue", "basse", "quinte descendante nue",
              "la basse descend d'une quinte en passant la barre — sans savoir où "
              "est la tonique. C'est la piste 1 du brief.", q))

    q2 = np.zeros(n)
    for b in range(n):
        for j in (2 * b, 2 * b + 1):
            if 1 <= j < len(B2):
                q2[b] = max(q2[b], sum(_pcmass(B2[j - 1], (k + 7) % 12) * _pcmass(B2[j], k)
                                       for k in range(12)))
    F.append(("q_demi", "basse", "quinte descendante, demi-mesure",
              "la même, cherchée aussi à l'intérieur de la mesure (le V sur le 3e "
              "temps qui résout au temps 1 suivant).", q2))

    # ── 2. la tonique locale ───────────────────────────────────────────────
    ton = np.array([_pcmass(B1[b], K[b]) if K[b] is not None else 0.0 for b in range(n)])
    F.append(("ton", "tonique", "la basse est sur la tonique locale",
              "combien de la basse de cette mesure tombe sur la tonique du moment "
              "(clé estimée sur 16 mesures glissantes).", ton))

    arr = np.zeros(n)
    arr[1:] = np.clip(ton[1:] - ton[:-1], 0, None)
    F.append(("arr", "tonique", "ARRIVÉE sur la tonique",
              "la tonique gagne du terrain d'une mesure à l'autre. Ce qui compte "
              "n'est pas d'être sur I, c'est d'y ARRIVER — un vamp sur I ne "
              "cadence pas.", arr))

    vi = np.zeros(n)
    for b in range(1, n):
        if K[b] is None:
            continue
        k = K[b]
        dom = max(_pcmass(B1[b - 1], (k + 7) % 12), 0.8 * _pcmass(B1[b - 1], (k + 5) % 12))
        vi[b] = dom * _pcmass(B1[b], k)
    F.append(("vi", "tonique", "V→I (ou IV→I) vers la tonique locale",
              "la quinte descendante, mais qui ARRIVE sur la tonique. C'est la "
              "cadence parfaite ; la plagale IV→I compte 0,8.", vi))

    vim = np.zeros(n)
    for b in range(1, n):
        if K[b] is None or R[b] is None or R[b - 1] is None:
            continue
        k = K[b]
        if R[b] == k and R[b - 1] in ((k + 7) % 12, (k + 5) % 12, (k + 10) % 12):
            vim[b] = 1.0
    F.append(("vi_musx", "tonique", "V→I lu sur les noms d'accords musx",
              "la même cadence, mais décidée sur le nom d'accord du modèle plutôt "
              "que sur le chroma de basse. Compte aussi bVII→I (le rock).", vim))

    vg = np.zeros(n)
    for b in range(1, n):
        vg[b] = (max(_pcmass(B1[b - 1], (KG + 7) % 12), 0.8 * _pcmass(B1[b - 1], (KG + 5) % 12))
                 * _pcmass(B1[b], KG))
    F.append(("vi_glob", "tonique", "V→I vers la tonique GLOBALE",
              "la même, avec une seule tonalité pour tout le morceau — contrôle : "
              "la clé locale sert-elle vraiment à quelque chose ?", vg))

    # ── 3. le chant ────────────────────────────────────────────────────────
    notes = d["notes"]; grid = np.asarray(d["grid"], float)
    fin_tonique = np.zeros(n); fin_longue = np.zeros(n)
    descend = np.zeros(n); silence = np.zeros(n)
    for b in range(1, n):
        t = grid[b]
        av = [x for x in notes if x[0] + x[1] <= t + 1e-3]
        ap = [x for x in notes if x[0] >= t - 1e-3]
        if not av:
            continue
        der = av[-1]
        mes = max(1e-6, grid[min(n, b + 1)] - grid[b])
        # (a) la dernière note est la tonique ou la tierce de la clé locale
        if K[b] is not None:
            deg = (int(der[2]) - K[b]) % 12
            fin_tonique[b] = 1.0 if deg == 0 else (0.6 if deg in (3, 4) else 0.0)
        # (b) elle est plus longue que la moyenne de la phrase (8 mesures avant)
        t0 = grid[max(0, b - 8)]
        ph = [x for x in av if x[0] >= t0]
        if len(ph) >= 3:
            mo = float(np.mean([x[1] for x in ph[:-1]])) or 1e-6
            fin_longue[b] = float(np.clip(der[1] / mo, 0, 4))
        # (c) le contour descend sur les dernières notes
        if len(ph) >= 4:
            y = np.array([x[2] for x in ph[-4:]])
            pente = float(np.polyfit(np.arange(len(y)), y, 1)[0])
            descend[b] = max(0.0, -pente)
        # (d) un silence avant la phrase suivante
        fin = der[0] + der[1]
        deb = ap[0][0] if ap else grid[n]
        silence[b] = float(np.clip((deb - fin) / mes, 0, 4))
    F += [("c_ton", "chant", "(a) la phrase finit sur la tonique ou la tierce",
           "la dernière note chantée avant la mesure b, lue en degré de la clé "
           "locale. Tonique = 1, tierce = 0,6.", fin_tonique),
          ("c_long", "chant", "(b) la dernière note est plus LONGUE",
           "sa durée divisée par la durée moyenne des notes des 8 mesures "
           "précédentes. Une phrase qui se pose tient sa note.", fin_longue),
          ("c_desc", "chant", "(c) le contour DESCEND",
           "la pente des quatre dernières notes, en demi-tons par note ; on ne "
           "garde que la descente.", descend),
          ("c_sil", "chant", "(d) un SILENCE avant la phrase suivante",
           "la durée du trou entre la fin de la dernière note et l'attaque de la "
           "suivante, comptée en mesures.", silence)]

    ch = 0.45 * fin_tonique + 0.25 * np.tanh(fin_longue / 2) + \
        0.10 * np.tanh(descend) + 0.45 * np.tanh(silence)
    F.append(("chant", "chant", "les quatre réunies",
              "somme pondérée des quatre mesures de fermeture mélodique.", ch))

    # ── 4. la surprise d'un modèle de progression ──────────────────────────
    J = jetons(stem, D, T16)
    sur = np.zeros(n)
    for b in range(1, n):
        a, z = J[b - 1], J[b]
        sur[b] = -LP(a, z) if (a is not None and z is not None) else 0.0
    F.append(("sur", "progression", "surprise du modèle de progression",
              "moins le log de la probabilité de cet accord sachant le précédent, "
              "en degrés de la clé locale. Bigramme appris sur les 17 AUTRES "
              "morceaux (leave-one-song-out).", sur))

    for W in (4, 8):
        dl = np.zeros(n)
        for b in range(n):
            av = sur[max(1, b - W):b]
            dl[b] = sur[b] - (float(np.mean(av)) if len(av) else 0.0)
        F.append((f"sur_d{W}", "progression", f"saut de surprise (phrase de {W} mesures)",
                  "la surprise de cette mesure MOINS la surprise moyenne des "
                  f"{W} mesures d'avant. C'est la formulation exacte de Louis : "
                  "cohérent à l'intérieur de la phrase, rupture au passage à la "
                  "suivante.", dl))

    # ── 5. la cadence complète ─────────────────────────────────────────────
    z = lambda x: (x - x.mean()) / max(1e-9, x.std())
    F.append(("cad", "cadence", "CADENCE = la basse résout + le chant se ferme",
              "z(V→I sur la clé locale) + z(fermeture mélodique). Les deux moitiés "
              "de ce que décrit Louis : l'harmonie se résout ET la phrase se pose.",
              z(vi) + z(ch)))
    F.append(("cad3", "cadence", "CADENCE + saut de surprise",
              "la même, plus le saut de surprise du modèle de progression.",
              z(vi) + z(ch) + z(F[[f[0] for f in F].index("sur_d8")][4])))
    F.append(("cad_h", "cadence", "CADENCE harmonique seule",
              "z(V→I) + z(arrivée sur la tonique) — sans le chant, pour voir ce "
              "que la voix apporte vraiment.", z(vi) + z(arr)))
    return F


# ── le jeu de courbes, pour tout le corpus ──────────────────────────────────

def tout(stems=None):
    stems = stems or liste_validee()
    D = {s: brut(s) for s in stems}
    T16 = {s: tonique(chroma_pc(D[s]["harm1"]), 16) for s in stems}
    T0 = {s: tonique(chroma_pc(D[s]["harm1"]), 0) for s in stems}
    J = {s: jetons(s, D, T16) for s in stems}
    F = {}
    for s in stems:                       # LOSO : le modèle n'a jamais vu `s`
        LP = bigramme([x for x in stems if x != s], J)
        F[s] = formulations(s, D, T16, T0, LP)
    return D, F


# ── notation ────────────────────────────────────────────────────────────────

def pics(c, n, densite=1 / 6, ecart=3):
    """Les `n*densite` plus hauts pics, espacés d'au moins `ecart` mesures.

    Budget commun à toutes les formulations : sans lui, une courbe bruitée qui
    pique toutes les deux mesures « trouve » tout. Un pic toutes les 6 mesures,
    c'est le coût maximum que Louis accepte.
    """
    c = np.nan_to_num(np.asarray(c, float))
    if c.std() <= 1e-12:
        return []
    k = max(1, int(round(n * densite)))
    out = []
    for b in np.argsort(-c):
        b = int(b)
        if c[b] <= 0 and out:
            break
        if all(abs(b - g) >= ecart for g in out):
            out.append(b)
        if len(out) >= k:
            break
    return sorted(out)


def note(stems, F, densite=1 / 6, tol=1):
    """{clé: (rappel ±0, rappel ±tol, pics/morceau, médiane et MAD de l'écart)}."""
    res = {}
    for i, (cle, grp, nom, _g, _c) in enumerate(F[stems[0]]):
        tot = ex = pr = npic = 0
        ecarts = []
        for s in stems:
            c = F[s][i][4]
            n = len(c)
            P = pics(c, n, densite)
            gt = gt_de(s)
            npic += len(P)
            tot += len(gt)
            for g in gt:
                dd = min([abs(p - g) for p in P], default=99)
                if dd == 0:
                    ex += 1
                if dd <= tol:
                    pr += 1
            for p in P:
                if gt:
                    e = min(gt, key=lambda g: abs(p - g))
                    if abs(p - e) <= 4:
                        ecarts.append(p - e)
        m = float(np.median(ecarts)) if ecarts else float("nan")
        mad = float(np.median(np.abs(np.array(ecarts) - m))) if ecarts else float("nan")
        res[cle] = dict(grp=grp, nom=nom, ex=ex, pr=pr, tot=tot, npic=npic,
                        med=m, mad=mad, ecarts=ecarts)
    return res


def hasard(stems, densite=1 / 6, n_tir=200):
    """Le rappel qu'obtiendraient des pics TIRÉS AU SORT, au même budget.

    Indispensable pour lire le tableau : avec un pic toutes les 6 mesures et une
    frontière toutes les 8, le hasard touche déjà 43 % des frontières à ±1. Un
    rappel de 55 % n'est pas « plus d'une sur deux », c'est « +12 sur le hasard ».
    """
    rng = np.random.default_rng(0)
    h0 = h1 = 0.0
    for _ in range(n_tir):
        ex = pr = tot = 0
        for s in stems:
            n = brut(s)["n"]; gt = gt_de(s)
            P = sorted(rng.choice(n, max(1, int(round(n * densite))), replace=False))
            tot += len(gt)
            for g in gt:
                dd = min([abs(p - g) for p in P], default=99)
                ex += dd == 0; pr += dd <= 1
        h0 += 100 * ex / tot; h1 += 100 * pr / tot
    return h0 / n_tir, h1 / n_tir


def table():
    stems = liste_validee()
    D, F = tout(stems)
    nbar = sum(D[s]["n"] for s in stems)
    R = note(stems, F)
    print(f"{len(stems)} morceaux · {nbar} mesures · "
          f"{sum(len(gt_de(s)) for s in stems)} frontières annotées")
    print("Budget commun : un pic toutes les 6 mesures. « rappel » = part des "
          "frontières annotées\ntouchées par un pic. « écart » = position du pic "
          "− frontière la plus proche (médiane ± MAD).\n")
    print(f"{'formulation':<46}{'rappel ±0':>10}{'rappel ±1':>11}"
          f"{'pics/morceau':>14}{'écart':>14}")
    ordre = sorted(R, key=lambda k: -R[k]["pr"])
    for cle in ordre:
        r = R[cle]
        e = ("—" if np.isnan(r["med"]) else f"{r['med']:+.1f} ± {r['mad']:.1f}")
        print(f"{r['nom'][:45]:<46}{100 * r['ex'] / r['tot']:>9.0f}%"
              f"{100 * r['pr'] / r['tot']:>10.0f}%{r['npic'] / len(stems):>14.1f}{e:>14}")
    h0, h1 = hasard(stems)
    print(f"{'LE HASARD, au même budget de pics':<46}{h0:>9.0f}%{h1:>10.0f}%")
    print("\nSans cette dernière ligne le tableau ment : un pic toutes les 6 mesures "
          "et une\nfrontière toutes les 8, ça suffit à en toucher 43 % au hasard.")
    return R


def par_morceau(cle="cad"):
    stems = liste_validee()
    D, F = tout(stems)
    i = [f[0] for f in F[stems[0]]].index(cle)
    print(f"formulation « {F[stems[0]][i][2]} », par morceau\n")
    print(f"{'morceau':<40}{'mes':>5}{'ftr':>5}{'pics':>6}{'±0':>5}{'±1':>5}{'rappel':>8}")
    for s in stems:
        c = F[s][i][4]; n = len(c); P = pics(c, n); gt = gt_de(s)
        ex = sum(1 for g in gt if min([abs(p - g) for p in P], default=99) == 0)
        pr = sum(1 for g in gt if min([abs(p - g) for p in P], default=99) <= 1)
        print(f"{s[:39]:<40}{n:>5}{len(gt):>5}{len(P):>6}{ex:>5}{pr:>5}"
              f"{100 * pr / max(1, len(gt)):>7.0f}%")


# ── la page ─────────────────────────────────────────────────────────────────

TEINTE = {"basse": "#3f5a86", "tonique": "#2f6d8a", "chant": "#8a2b2b",
          "progression": "#4a6b3a", "cadence": "#8a5a1f"}
INK = "#3f3a2e"


def page(stems):
    from ssm_zoo import SONGS
    titres = dict(SONGS)
    D, F = tout(liste_validee())          # LOSO sur tout le corpus, affiché sur 4
    body = []
    for stem in stems:
        d = D[stem]; n = d["n"]; grid = [float(x) for x in d["grid"]]
        gt = gt_de(stem)
        noms = d["noms"]
        lignes = []
        boutons = []
        for cle, grp, nom, glose, c in F[stem]:
            P = pics(c, n)
            cc = np.nan_to_num(np.asarray(c, float))
            lo, hi = float(cc.min()), float(cc.max())
            u = (cc - lo) / max(1e-9, hi - lo)
            barres = "".join(
                f'<i class=v style="left:{b / n * 100:.4f}%;width:{100 / n:.4f}%;'
                f'opacity:{.10 + .85 * u[b]:.2f};background:{TEINTE[grp]}"></i>'
                for b in range(n))
            marks = ""
            for p in P:
                e = min([abs(p - g) for g in gt], default=99)
                k = "ok" if e == 0 else ("pres" if e <= 1 else "hors")
                marks += (f'<i class="pk {k}" style="left:{(p + .5) / n * 100:.4f}%" '
                          f'data-b="{p}" title="mesure {p + 1} · {nom}"></i>')
            ex = sum(1 for g in gt if min([abs(p - g) for p in P], default=99) <= 1)
            lignes.append(
                f'<div class=lane><div class=lab title="{glose}" '
                f'style="color:{TEINTE[grp]}">{nom}<small>{ex}/{len(gt)}</small></div>'
                f'<div class=strip>{barres}{marks}</div></div>')
            if cle in ("vi", "cad", "chant", "sur_d8"):
                for p in P:
                    e = min([abs(p - g) for g in gt], default=99)
                    k = "ok" if e == 0 else ("pres" if e <= 1 else "hors")
                    ac = " → ".join(noms[max(0, p - 2):min(n, p + 2)])
                    boutons.append(
                        f'<button class="cd {k}" data-b="{p}" data-cle="{cle}">'
                        f'<b>mes. {p + 1}</b><small>{nom}</small><em>{ac}</em></button>')

        blocs = ""
        for s in json.loads((ANN / f"{stem}.json").read_text())["sections"]:
            w = (s["b1"] - s["b0"] + 1) / n * 100
            blocs += (f'<i style="left:{s["b0"] / n * 100:.4f}%;width:{w:.4f}%">'
                      f'{s.get("label", "")}</i>')
        pas = 4 if n <= 60 else (8 if n <= 140 else 16)
        regle = "".join(f'<i style="left:{i / n * 100:.4f}%">{i + 1}</i>'
                        for i in range(0, n, pas))
        gtl = "".join(f'<i class=gt style="left:{g / n * 100:.4f}%"></i>' for g in gt)
        vus = set(); btn = []
        for b in boutons:
            if b not in vus:
                vus.add(b); btn.append(b)
        body.append(
            f'<section data-grid="{json.dumps(grid)}" data-n="{n}">'
            f'<div class=hd><h2>{titres.get(stem, stem.replace("_", " "))}</h2>'
            f'<span class=sub>{n} mesures · {len(gt)} frontières validées</span></div>'
            f'<div class=bar><button class=pp>▶</button><span class=pos>mes. 1</span>'
            f'<span class=hint>clique la bande pour écouter · clique un pic (ou un '
            f'bouton) pour entendre la cadence : 2 mesures avant → 2 après</span></div>'
            f'<div class=stack>'
            f'<div class=lane><div class=lab></div><div class=secs>{blocs}</div></div>'
            f'<div class=lane><div class=lab></div><div class=rule>{regle}</div></div>'
            f'{"".join(lignes)}'
            f'<div class=ov>{gtl}<div class=cur></div></div></div>'
            f'<div class=lane2>{"".join(btn)}</div>'
            f'<audio preload=metadata playsinline src="../audio/{stem}.m4a"></audio>'
            f'</section>')
        print(f"  ok {titres.get(stem, stem)}")

    R = note(liste_validee(), F)
    h0, h1 = hasard(liste_validee())
    ordre = sorted(R, key=lambda k: -R[k]["pr"])
    tb = "".join(
        f'<tr><td>{R[c]["nom"]}</td><td>{100 * R[c]["ex"] / R[c]["tot"]:.0f}%</td>'
        f'<td><b>{100 * R[c]["pr"] / R[c]["tot"]:.0f}%</b></td>'
        f'<td>{100 * R[c]["pr"] / R[c]["tot"] - h1:+.0f}</td>'
        f'<td>{R[c]["npic"] / 18:.1f}</td>'
        f'<td>{"—" if np.isnan(R[c]["med"]) else f"{R[c]['med']:+.1f} ± {R[c]['mad']:.1f}"}'
        f'</td></tr>' for c in ordre)
    tb += (f'<tr class=haz><td>le HASARD, au même budget de pics</td><td>{h0:.0f}%</td>'
           f'<td><b>{h1:.0f}%</b></td><td>—</td><td>14.2</td><td>—</td></tr>')

    css = f"""
body{{margin:0;background:#e7e0d0;font:15px/1.55 -apple-system,system-ui,sans-serif;color:{INK}}}
.wrap{{max-width:1240px;margin:0 auto;padding:22px 14px 80px}}
h1{{font:italic 600 26px Georgia,serif;margin:0 0 6px}}
.lede{{color:#6f6857;font-size:13.5px;margin-bottom:16px;max-width:1000px}}
.lede b{{color:{INK}}}
section{{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;
  padding:10px 14px 14px;margin-bottom:14px;--lab:250px}}
.hd{{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap}}
h2{{font:700 18px system-ui;margin:0;color:#8a2b2b}}
.sub{{font:500 12px system-ui;color:#8a8371}}
.bar{{display:flex;align-items:center;gap:14px;margin:8px 0 10px;flex-wrap:wrap}}
.pp{{width:34px;height:34px;border-radius:50%;border:1px solid #d8cfb4;
  background:#f7f3e9;font-size:13px;cursor:pointer;flex:none}}
.pos{{font:600 12px ui-monospace,monospace;min-width:120px}}
.hint{{font:500 11.5px system-ui;color:#a89f8c}}
.stack{{position:relative}}
.lane{{display:grid;grid-template-columns:var(--lab) 1fr;align-items:center;height:19px}}
.lab{{font:600 11px system-ui;padding-right:8px;text-align:right;white-space:nowrap;
  overflow:hidden;text-overflow:ellipsis;cursor:help}}
.lab small{{color:#b5ab94;font-weight:500;margin-left:5px}}
.strip{{position:relative;height:13px;border-bottom:1px solid #f6f1e4}}
.v{{position:absolute;top:0;height:13px}}
.pk{{position:absolute;top:-3px;width:0;height:0;border-left:4px solid transparent;
  border-right:4px solid transparent;border-top:7px solid #b4472c;cursor:pointer;
  transform:translateX(-4px);z-index:6}}
.pk.ok{{border-top-color:#2f7a45}} .pk.pres{{border-top-color:#d9a441}}
.pk.hors{{border-top-color:#7a6f5a;opacity:.75}}
.secs{{position:relative;height:18px}}
.secs i{{position:absolute;top:0;height:18px;font:700 10px system-ui;color:#8a6a4a;
  background:#f4ecd9;border-left:1px solid #e0d3b6;box-sizing:border-box;
  padding-left:3px;line-height:18px;overflow:hidden;border-radius:2px}}
.rule{{position:relative;height:13px}}
.rule i{{position:absolute;top:0;font:500 9.5px ui-monospace,monospace;color:#a89f8c;
  transform:translateX(-50%)}}
.ov{{position:absolute;left:var(--lab);right:0;top:0;bottom:0;cursor:crosshair;z-index:4}}
.gt{{position:absolute;top:0;bottom:0;width:1px;background:#b4472c;opacity:.8}}
.cur{{position:absolute;top:0;bottom:0;width:2px;background:#c1121f;display:none;
  box-shadow:0 0 0 1px rgba(255,255,255,.7)}}
.lane2{{display:flex;flex-wrap:wrap;gap:6px;margin-top:12px}}
button.cd{{border:2px solid #e0d7c2;background:#f7f3e9;border-radius:9px;padding:5px 9px;
  cursor:pointer;font:600 12.5px system-ui;color:#4a4438;text-align:left;max-width:260px}}
button.cd small{{display:block;font:500 10px system-ui;color:#a89f8c}}
button.cd em{{display:block;font:600 10.5px ui-monospace,monospace;color:#6f6857;
  font-style:normal;margin-top:2px}}
button.cd.ok{{border-color:#2f7a45}} button.cd.pres{{border-color:#d9a441}}
button.cd.hors{{border-color:#c9c1ab}}
button.cd.on{{background:#0d2437;color:#fff}} button.cd.on small,button.cd.on em{{color:#9fb8c6}}
table{{border-collapse:collapse;font:500 12.5px system-ui;background:#fffdf6;
  border-radius:12px;overflow:hidden;margin:8px 0 20px;width:100%;max-width:900px}}
th,td{{padding:5px 10px;text-align:right;border-bottom:1px solid #efe8d6}}
th:first-child,td:first-child{{text-align:left}}
th{{background:#f4ecd9;font-weight:700}}
tr.haz td{{background:#f2ede0;color:#8a8371;font-style:italic}}
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
  function cadence(b){ aller(b2t(Math.max(0,b-2)), b2t(Math.min(n,b+2))); }
  ov.onclick=function(e){ var r=ov.getBoundingClientRect();
    aller(b2t(n*(e.clientX-r.left)/r.width), null); };
  sec.querySelectorAll(".pk").forEach(function(p){
    p.onclick=function(ev){ ev.stopPropagation(); cadence(+p.dataset.b); }; });
  sec.querySelectorAll("button.cd").forEach(function(btn){
    btn.onclick=function(){
      sec.querySelectorAll("button.cd").forEach(function(x){x.classList.remove("on")});
      btn.classList.add("on"); cadence(+btn.dataset.b); }; });
});
"""
    OUT.write_text(f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Les cadences</title><style>{css}</style></head><body><div class=wrap>
<h1>La cadence — « la phrase finit sur quelque chose de cohérent »</h1>
<div class=lede>Une <b>cadence</b>, c'est une section qui ne s'arrête pas au
milieu d'une progression : elle se <b>résout</b>. Chaque ligne est une façon de
mesurer ça, mesure par mesure. Sombre = fort.
<b>Les triangles</b> sont les pics retenus — un pic toutes les 6 mesures, le même
budget pour toutes les lignes.
<b style="color:#2f7a45">vert</b> = pile sur une frontière que tu as validée,
<b style="color:#d9a441">ambre</b> = à une mesure,
<b style="color:#7a6f5a">gris</b> = ailleurs (et « ailleurs » ne veut pas dire
faux : tu n'as pas noté tous tes changements de section — écoute-les).
<b>Clique un triangle</b> pour entendre la cadence : deux mesures avant, deux
après.<br><br>
<b>Le décalage a été mesuré, pas supposé.</b> La résolution tombe sur le
<b>premier temps de la section suivante</b> : le V est la dernière mesure de la
section qui finit, le I est la frontière. Toutes les lignes sont donc indexées
« un pic en b = une frontière en b », sans correction.<br><br>
Sur les 18 morceaux validés, à budget égal (un pic toutes les 6 mesures) :
<table><tr><th>formulation</th><th>rappel ±0</th><th>rappel ±1</th><th>gain</th>
<th>pics / morceau</th><th>écart médian</th></tr>{tb}</table>
</div>
{''.join(body)}</div><script>{js}</script></body></html>""")
    print(f"wrote {OUT}  ({OUT.stat().st_size // 1024} KB)")
    print("   http://100.89.209.63:7772/plots/cadences.html")


if __name__ == "__main__":
    if "--table" in sys.argv:
        table()
    elif "--songs" in sys.argv:
        par_morceau([a for a in sys.argv[1:] if not a.startswith("--")][0]
                    if len([a for a in sys.argv[1:] if not a.startswith("--")]) else "cad")
    else:
        args = [a for a in sys.argv[1:] if not a.startswith("--")]
        page(args or QUATRE)
