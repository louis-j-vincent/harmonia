"""L'ordre de remplissage vient des VOIX ; la matière vient des BI-MESURES.

    .venv/bin/python scripts/vote_fill.py [--dump] [<stem> ...]
        -> /plots/vote_fill.html + une page par morceau (audio + tête de lecture)

Louis, 2026-08-12 :

  « Explique-moi comment tu fais l'ordre des sections pour privilégier comment le
    remplir. Je veux que ça se fasse par rapport au nombre de matrices qu'on a qui
    votent pour un pic en particulier, et tu peux reprendre les groupages par
    bi-barres que tu me montrais avant que j'aimais beaucoup. »

DEUX INGRÉDIENTS, DEUX RÔLES DISTINCTS.

  * **le mot de deux mesures** — chaque bi-mesure reçoit une lettre minuscule
    d'après la matrice d'accords : deux bi-mesures qui se ressemblent portent la
    même lettre. Le morceau devient une chaîne de caractères, et c'est là que la
    structure se lit à l'œil nu. Blue Lights : `aaaaaabc` cinq fois de suite — le
    `bc` est exactement chacun des B de Louis. This Love : `aaae` cinq fois, à
    chacun de ses B, et le pont sort comme le seul mot unique du morceau
    (`fdfg`). C'est la MATIÈRE : ce qui se répète, avant toute décision.

  * **les voix des matrices** — chaque pic du profil fusionné est proposé par un
    certain nombre des sept matrices. Ce nombre est une CONFIANCE, pas une
    position (mesuré : 1 voix → 9 % de pics justes, 5 voix → 58 %, 7 voix →
    60 % et 90 % à ±1 mesure). C'est l'ORDRE : on tranche d'abord là où
    l'accord est le plus large.

L'ORDRE DE REMPLISSAGE, EXACTEMENT — c'est la réponse à sa question.

  1. On garde les pics à `MIN_VOTES` voix ou plus, calés sur la grille de deux
     mesures (±1 mesure). Ils découpent le morceau en blocs.
  2. Les blocs sont traités **du plus voté au moins voté** — le score d'un bloc
     est le nombre de voix de sa frontière la mieux votée.
  3. Le premier bloc traité est celui dont on est le plus sûr : il DÉFINIT une
     lettre. Les blocs suivants ne peuvent que se comparer à ce qui existe déjà.
     C'est pour ça que l'ordre compte, et c'est tout ce qu'il fait.
  4. Chaque bloc est rempli en lisant son mot, dans cet ordre de préférence :
       b. le mot entier est déjà écrit ailleurs → il reprend sa lettre ;
       c. un début ou une fin du mot est connu → on coupe là et on recommence.
          C'est la règle qui écrit Blue Lights : `aaaaaa` puis `bc`, puis
          `aaaaaa`, puis `bc`… soit un A de 12 mesures et un B de 4, cinq fois ;
       a. le mot est une répétition (`u u`, `u u u`, unité ≥ 8 mesures) → on
          écrit deux ou trois sections identiques COLLÉES, pas une grande
          (« un A comme une succession de petits A », Louis 2026-08-10) ;
       d. le mot change de matière — sa fin n'emploie aucune lettre de son début
          → on coupe là ;
       e. le mot change de lettre pile sur la trame de 8 mesures calée sur le
          premier chant (77 % des frontières de Louis y sont) → on coupe là ;
       f. faute de mieux, on coupe à 8 mesures ; et si le bloc est court, c'est
          une lettre neuve.

CE QUE ÇA DONNE, sans rien mesurer — trois pages à regarder :
Blue Lights et Every Breath You Take sortent à la frontière près comme le
découpage de Louis, Chain of Fools aussi sauf la longueur de son B. Bein' Green
se décale de 4 mesures parce qu'un pic à 3 voix (mesure 9) tombe à côté ; le
défaut est dans le pic, pas dans le remplissage, et la page le montre.

CE QUE ÇA NE FAIT PAS. Sunny module d'un demi-ton à chaque reprise : ses
bi-mesures ne se ressemblent plus une fois transposées, le mot ne se répète
jamais (22 lettres pour 43 bi-mesures) et le remplissage n'a rien à quoi
s'accrocher après la moitié du morceau. Il faudrait une ressemblance invariante
par transposition ; ce n'est pas écrit. Stand By Me n'a AUCUN pic à 3 voix : son
morceau entier est un seul bloc, l'ordre n'a rien à ordonner, et tout repose
alors sur le mot seul.
"""
from __future__ import annotations

import base64
import io
import os
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
sys.path.insert(0, str(HERE / "scratchpad"))

from ssm_zoo import SONGS, AUDIO, GT_LINE, gt_sections     # noqa: E402
from peak_profile import page                              # noqa: E402
from hard_prior_sections import colourmap                  # noqa: E402
import order_bundle                                        # noqa: E402
import order_lab as OL                                     # noqa: E402

OUTDIR = HERE / "docs" / "plots"
VCACHE = HERE / "scratchpad" / "order_cache"
PLOT_L, PLOT_R = 0.155, 0.995
INK = "#1c1c1c"
HARD = "#0d2437"
SYM = "abcdefghijklmnopqrstuvwxyz"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

SYM_THR = 0.90     # deux bi-mesures portent la même lettre au-delà de ça
MIN_VOTES = 3      # en dessous, un pic est une hypothèse et ne coupe rien
SNAP = 1           # un pic se cale sur la grille de deux mesures à ±1 mesure
WORD_TOL = 0.75    # deux mots de même longueur sont « les mêmes » à partir de là
MIN_BI = 4         # 8 mesures : l'unité minimale d'une répétition
MAX_BI = 6         # 12 mesures : au-delà, un bloc DOIT se couper
MIN_PIECE = 2      # 4 mesures : rien de plus court ne s'écrit (sauf la queue)
PER_TOL = 0.85     # une période, elle, doit être franche : à 0,75 The Walk
                   # sortait une fausse période de 12 mesures (20 bi-mesures
                   # concordantes sur 26) qui écrasait ses A de 8 mesures.


# ── les voix ────────────────────────────────────────────────────────────────

def votes(stem: str, rebuild=False) -> list[dict]:
    """[{bar, votes, voters, retenu}] — chaque pic et qui l'a proposé.

    Mis en cache : le calcul refait les sept matrices (~10 s par morceau).
    """
    p = VCACHE / f"votes_{stem}.pkl"
    if p.exists() and not rebuild:
        with p.open("rb") as f:
            return pickle.load(f)
    from peak_profile import fused_profile, peak_votes
    P, kept, rest, lines, n, _extra = fused_profile(stem)
    m = len(P)
    out = []
    for v in peak_votes(kept + rest, lines):
        out.append({"bar": int(round(v["cut"] * n / m)), "votes": v["votes"],
                    "voters": v["voters"], "retenu": v["cut"] in kept,
                    "haut": float(P[v["cut"]])})
    out.sort(key=lambda d: d["bar"])
    with p.open("wb") as f:
        pickle.dump(out, f)
    return out


# ── le mot de deux mesures ──────────────────────────────────────────────────

def bibar_word(b, thr=SYM_THR):
    """(mot, ancre, J, sim) — une lettre minuscule par bi-mesure.

    L'ancre est la phase de la première mesure chantée : c'est là que les
    sections commencent (mesuré : 90 % des frontières de Louis sont sur cette
    grille de deux mesures).

    Le groupage est en lien complet, comme `bibar_stack_lab.bibar_clusters` que
    Louis a validé le 2026-08-08 pour l'empilement des répétitions : un membre
    n'entre dans un groupe que s'il ressemble à TOUS ses membres, ce qui évite
    la chaîne où a≈b, b≈c mais a≉c.
    """
    S, n, a = b["S"], b["n"], b["start"] % 2
    J = (n - a) // 2
    idx = [a + 2 * j for j in range(J)]
    B = np.zeros((J, J))
    for i, p in enumerate(idx):
        for k, q in enumerate(idx):
            B[i, k] = (S[p, q] + S[p + 1, q + 1]) / 2
    d = np.sqrt(np.clip(np.diag(B), 1e-9, None))
    B = B / np.outer(d, d)

    used, lab, k = set(), [-1] * J, 0
    for j in sorted(range(J), key=lambda j: -(B[j] >= thr).sum()):
        if j in used:
            continue
        mem = [j]
        for q in range(J):
            if q in used or q == j:
                continue
            if all(B[q, x] >= thr for x in mem):
                mem.append(q)
        for q in mem:
            lab[q] = k
        used.update(mem)
        k += 1
    # les lettres dans l'ordre d'apparition, pour que le mot se lise
    ren, k = {}, 0
    for j in range(J):
        if lab[j] not in ren:
            ren[lab[j]] = k; k += 1
    word = "".join(SYM[ren[lab[j]] % 26] for j in range(J))
    return word, a, J, B


def same(u: str, v: str, tol=WORD_TOL) -> bool:
    """Deux mots de même longueur, à quelques bi-mesures près."""
    if len(u) != len(v) or not u:
        return False
    eq = sum(1 for x, y in zip(u, v) if x == y)
    return eq >= max(1, int(np.ceil(tol * len(u))))


def periode(w: str, pmin=MIN_BI) -> int:
    """La plus petite longueur p ≥ `pmin` telle que le mot soit `u` répété.

    `pmin` = 4 bi-mesures = 8 mesures, et c'est la correction du 2026-08-12 :
    sans lui, Chain of Fools (`aaaa…`, 38 bi-mesures identiques) sort avec une
    période de 1 et se fait hacher en dix-neuf sections de deux mesures. Une
    boucle harmonique de deux ou quatre mesures est la PÉRIODE du morceau, pas
    l'ÉCHELLE de ses sections — c'est le constat qui revient depuis le début.
    La dernière copie a le droit d'être incomplète (un morceau finit rarement
    pile sur sa boucle).
    """
    L = len(w)
    for p in range(pmin, L // 2 + 1):
        eq = sum(1 for i in range(L - p) if w[i] == w[i + p])
        if eq >= PER_TOL * (L - p):
            return p
    return L


def contraste(w: str, mini=MIN_PIECE):
    """La position où le mot change de matière, ou None.

    Le cas de Blue Lights : `aaaaaabc` — la fin `bc` n'emploie AUCUN symbole du
    début. C'est la seule coupure interne qu'on s'autorise sans pic, parce
    qu'elle ne suppose rien : le morceau change littéralement de vocabulaire.
    """
    L = len(w)
    for k in range(L // 2, mini - 1, -1):      # la plus LONGUE queue contrastante
        if not (set(w[L - k:]) & set(w[:L - k])):
            return L - k
    for k in range(L // 2, mini - 1, -1):      # …ou la plus longue tête
        if not (set(w[:k]) & set(w[k:])):
            return k
    return None


# ── le remplissage, par ordre de voix ───────────────────────────────────────

def fill(b, stem, min_votes=MIN_VOTES, snap=SNAP):
    """{blocs, sections, coupures, mot, ancre} — le remplissage vote-ordonné.

    `blocs` garde la TRACE de l'ordre : chaque entrée dit combien de voix a porté
    le bloc, dans quel ordre il a été traité et par quelle règle il a été rempli.
    C'est ce que la page dessine.
    """
    word, a, J, B = bibar_word(b)
    n = b["n"]
    pk = [v for v in votes(stem) if v["votes"] >= min_votes]

    # 1 ── les pics calés sur la grille de deux mesures
    cuts = {}
    for v in pk:
        j = int(round((v["bar"] - a) / 2))
        if 0 < j < J and abs(v["bar"] - (a + 2 * j)) <= snap:
            cuts[j] = max(cuts.get(j, 0), v["votes"])
    bounds = sorted({0, J} | set(cuts))

    # 2 ── les blocs, du mieux voté au moins voté. Le score d'un bloc est le
    #      nombre de voix de sa frontière la mieux votée ; les BORDS du morceau
    #      ne votent pas (ils ne sont l'avis de personne).
    blocs = []
    for i in range(len(bounds) - 1):
        j0, j1 = bounds[i], bounds[i + 1]
        v = [cuts.get(j0, 0), cuts.get(j1, 0)]
        blocs.append({"j0": j0, "j1": j1, "voix": max(v), "somme": sum(v),
                      "mot": word[j0:j1]})
    order = sorted(range(len(blocs)),
                   key=lambda i: (-blocs[i]["voix"], -blocs[i]["somme"],
                                  blocs[i]["j0"]))

    # 3 ── on remplit dans cet ordre ; le premier bloc traité DÉFINIT sa lettre,
    #      les suivants ne peuvent que se comparer à ce qui est déjà écrit.
    lex: list[tuple[str, str]] = []          # (mot, lettre) déjà écrits
    pieces: list[dict] = []

    def lettre(w):
        for u, L in sorted(lex, key=lambda x: -len(x[0])):
            if same(u, w):
                return L
        L = LETTERS[len({x for _u, x in lex}) % len(LETTERS)]
        lex.append((w, L))
        return L

    def put(j0, j1, L, regle, rang):
        pieces.append({"j0": j0, "j1": j1, "label": L, "regle": regle,
                       "rang": rang, "etape": len(pieces)})

    def poser(j0, j1, rang, prof=0):
        w = word[j0:j1]
        if not w or prof > 24:
            return
        # b. le mot entier est déjà connu — c'est la même section qu'ailleurs
        for u, L in sorted(lex, key=lambda x: -len(x[0])):
            if same(u, w):
                put(j0, j1, L, "connu", rang); return
        # c. un début ou une fin est connu — on coupe là, du plus long au plus
        #    court. C'est la règle qui écrit Blue Lights : `aaaaaa` puis `bc`,
        #    puis `aaaaaa`, puis `bc`… soit A de 12 mesures et B de 4.
        for u, L in sorted(lex, key=lambda x: -len(x[0])):
            k = len(u)
            if not (MIN_PIECE <= k <= len(w) - MIN_PIECE):
                continue                     # jamais de moignon de deux mesures
            if same(w[:k], u):
                put(j0, j0 + k, L, "début connu", rang)
                poser(j0 + k, j1, rang, prof + 1); return
            if same(w[-k:], u):
                poser(j0, j1 - k, rang, prof + 1)
                put(j1 - k, j1, L, "fin connue", rang); return
        # a. une répétition : on écrit des petites sections identiques collées,
        #    pas une grande (Louis, 2026-08-10)
        p = periode(w)
        if p < len(w):
            for q in range(j0, j1, p):
                poser(q, min(q + p, j1), rang, prof + 1)
            return
        if len(w) > MAX_BI:
            k = contraste(w)                 # d. le morceau change de matière
            if k:
                poser(j0, j0 + k, rang, prof + 1)
                poser(j0 + k, j1, rang, prof + 1)
                return
            # e. le mot change de lettre PILE sur la trame de 8 mesures calée
            #    sur le premier chant — 77 % des frontières de Louis y sont.
            for j in range(j0 + 1, j1):
                if (word[j] != word[j - 1] and j % MIN_BI == 0
                        and j - j0 >= MIN_PIECE and j1 - j >= MIN_PIECE):
                    poser(j0, j, rang, prof + 1)
                    poser(j, j1, rang, prof + 1)
                    return
            cut = j0 + MIN_BI - (j0 % MIN_BI or 0)   # f. faute de mieux, 8 mes.
            cut = cut if j0 < cut < j1 else j0 + MIN_BI
            put(j0, cut, lettre(word[j0:cut]), "défaut 8 mes.", rang)
            poser(cut, j1, rang, prof + 1)
            return
        put(j0, j1, lettre(w), "neuf", rang)         # f. une lettre neuve

    for rang, i in enumerate(order):
        blocs[i]["rang"] = rang
        poser(blocs[i]["j0"], blocs[i]["j1"], rang)

    # 4 ── retour aux mesures, plus l'intro et la queue hors grille
    pieces.sort(key=lambda s: s["j0"])
    ren, k = {}, 0                    # les lettres dans l'ordre du morceau : le
    for s in pieces:                  # remplissage n'est pas chronologique, la
        if s["label"] not in ren:     # lecture, si.
            ren[s["label"]] = LETTERS[k % len(LETTERS)]; k += 1
    for s in pieces:
        s["label"] = ren[s["label"]]
    secs = []
    if a:
        secs.append({"b0": 0, "b1": a - 1, "label": "intro", "regle": "hors grille",
                     "rang": -1, "etape": -1})
    for s in pieces:
        secs.append({"b0": a + 2 * s["j0"], "b1": a + 2 * s["j1"] - 1,
                     "label": s["label"], "regle": s["regle"], "rang": s["rang"],
                     "etape": s["etape"]})
    if a + 2 * J < n:
        secs.append({"b0": a + 2 * J, "b1": n - 1, "label": "queue",
                     "regle": "hors grille", "rang": -1, "etape": -1})
    for s in secs:                            # tout ce qui précède le chant
        if s["b1"] < b["start"]:
            s["label"] = "intro"
    return {"sections": secs, "blocs": blocs, "ordre": order, "mot": word,
            "ancre": a, "J": J, "cuts": cuts, "pics": votes(stem), "sim": B}


# ── la page ─────────────────────────────────────────────────────────────────

def fig2b64_fixed(fig) -> str:
    bio = io.BytesIO()
    fig.savefig(bio, format="png", dpi=112)      # marges fixes : tête de lecture
    plt.close(fig)
    try:
        from PIL import Image
        bio.seek(0)
        im = Image.open(bio).convert("RGB").quantize(colors=128, method=2)
        s = io.BytesIO(); im.save(s, format="PNG", optimize=True)
        if s.tell() < bio.getbuffer().nbytes:
            bio = s
    except Exception:
        pass
    return base64.b64encode(bio.getvalue()).decode()


SYMCOL = ["#a8c8dc", "#e0c9a6", "#c4d8bf", "#dcc0c8", "#cdc6e0", "#e6d9a8",
          "#bcd3d8", "#e3cdb8", "#cfd6c0", "#d8c4d0"]


def matrices_png(R, n, gtb) -> str:
    """Les deux matrices que Louis demande, côte à côte.

    À GAUCHE, la CORRÉLATION entre toutes les bi-mesures — la matière brute d'où
    sort le mot. Un carré sombre hors diagonale = deux bi-mesures qui se
    ressemblent ; les damiers réguliers sont les couplets, les bandes qui coupent
    tout sont les ponts. Les traits bleus sont les sections qu'on écrit, les
    rouges les tiennes.

    À DROITE, la TRANSITION entre les lettres du mot : combien de fois la
    bi-mesure `x` est suivie de `y`. C'est la grammaire du morceau, et elle se
    lit vite — une diagonale forte veut dire « chaque bi-mesure se prolonge »
    (une boucle lente), un cycle `a→b→a` un balancement de deux mesures, et une
    case isolée un passage qui n'arrive qu'une fois (le pont).
    """
    from matplotlib.colors import LinearSegmentedColormap
    B, word, a, J = R["sim"], R["mot"], R["ancre"], R["J"]
    cmap = LinearSegmentedColormap.from_list("h", ["#faf6ec", "#9fc0d4", "#1d4d69"])

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13.0, 5.6), facecolor="#fffdf6",
        gridspec_kw={"width_ratios": [1.55, 1], "wspace": 0.16})

    lo = float(np.quantile(B, 0.15))
    ax1.imshow(B, cmap=cmap, vmin=lo, vmax=1.0, origin="upper",
               extent=[a, a + 2 * J, a + 2 * J, a], interpolation="nearest")
    for s in R["sections"][1:]:
        ax1.axvline(s["b0"], color="#0d2437", lw=0.8, alpha=0.75)
        ax1.axhline(s["b0"], color="#0d2437", lw=0.8, alpha=0.75)
    for g in gtb:
        ax1.axvline(g, color=GT_LINE, lw=0.8, alpha=0.7)
    step = 8 if n <= 120 else 16
    ax1.set_xticks(np.arange(0, n + 1, step)); ax1.set_yticks(np.arange(0, n + 1, step))
    ax1.set_xticklabels([str(i + 1) for i in np.arange(0, n + 1, step)], fontsize=8)
    ax1.set_yticklabels([str(i + 1) for i in np.arange(0, n + 1, step)], fontsize=8)
    ax1.tick_params(length=2, colors="#8a8371")
    ax1.set_title("corrélation entre toutes les bi-mesures", fontsize=10.5,
                  color="#4a4438", pad=8)

    syms = sorted(set(word))
    K = len(syms)
    T = np.zeros((K, K))
    for x, y in zip(word, word[1:]):
        T[syms.index(x), syms.index(y)] += 1
    ax2.imshow(T, cmap=cmap, vmin=0, vmax=max(1.0, T.max()), interpolation="nearest")
    for i in range(K):
        for j in range(K):
            if T[i, j]:
                ax2.text(j, i, f"{int(T[i, j])}", ha="center", va="center",
                         fontsize=8.5,
                         color="#fffdf6" if T[i, j] > 0.55 * T.max() else "#4a4438")
    ax2.set_xticks(range(K)); ax2.set_yticks(range(K))
    ax2.set_xticklabels(syms, fontsize=9); ax2.set_yticklabels(syms, fontsize=9)
    ax2.tick_params(length=0, colors="#4a4438")
    for i, t in enumerate(ax2.get_xticklabels() + ax2.get_yticklabels()):
        t.set_bbox(dict(facecolor=SYMCOL[(ord(syms[i % K]) - 97) % len(SYMCOL)],
                        edgecolor="none", boxstyle="round,pad=0.22"))
    ax2.set_xlabel("suivie de…", fontsize=9, color="#8a8371")
    ax2.set_ylabel("la bi-mesure…", fontsize=9, color="#8a8371")
    ax2.set_title("transition d'une bi-mesure à la suivante", fontsize=10.5,
                  color="#4a4438", pad=8)
    for ax in (ax1, ax2):
        for sp in ax.spines.values():
            sp.set_color("#e0d7c2")
    fig.subplots_adjust(left=0.05, right=0.99, top=0.93, bottom=0.07)
    return fig2b64_fixed(fig)


def song_page(stem: str, title: str) -> str:
    b = order_bundle.get(stem)
    n, grid = b["n"], b["grid"]
    R = fill(b, stem)
    word, a, J = R["mot"], R["ancre"], R["J"]
    gt = gt_sections(stem)
    today = OL.new_sections(b)[0]

    strips = [("le remplissage", R["sections"]), ("ce qu'on écrit", today)]
    if gt:
        strips.append(("toi", gt["sections"]))
    hist = sorted([s for s in R["sections"] if s["etape"] >= 0],
                  key=lambda s: s["etape"])
    rows = 3 + len(hist) + len(strips)         # mot · voix · ordre · pas · bandes
    fig, axs = plt.subplots(
        rows, 1, figsize=(13.0, 0.55 * 3 + 0.30 * len(hist) + 0.62 * len(strips) + 0.6),
        facecolor="#fffdf6",
        gridspec_kw={"height_ratios": [1.0, 1.25, 0.9] + [0.62] * len(hist)
                     + [1.15] * len(strips), "hspace": 0.0})
    gtb = [s["b0"] for s in gt["sections"][1:]] if gt else []

    def deco(ax, lab, colour=INK, last=False, hard=True):
        ax.set_xlim(0, n); ax.set_ylim(0, 1); ax.set_yticks([])
        ax.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=9.5,
                      color=colour)
        for s in ax.spines.values():
            s.set_visible(False)
        if hard:
            for j in R["cuts"]:
                ax.axvline(a + 2 * j, color=HARD, lw=1.5, alpha=0.9)
        if last:
            step = 8 if n <= 120 else 16
            ax.set_xticks(np.arange(0, n + 1, step))
            ax.set_xticklabels([str(i + 1) for i in np.arange(0, n + 1, step)],
                               fontsize=8, color="#8a8371")
            ax.tick_params(length=2, colors="#c9c0aa")
        else:
            ax.set_xticks([])

    # 1 ── le mot de deux mesures
    ax = axs[0]
    for j, ch in enumerate(word):
        ax.add_patch(plt.Rectangle((a + 2 * j, 0.15), 2, 0.7,
                                   facecolor=SYMCOL[(ord(ch) - 97) % len(SYMCOL)],
                                   edgecolor="#fffdf6", lw=0.9))
        if n <= 120:
            ax.text(a + 2 * j + 1, 0.5, ch, ha="center", va="center", fontsize=7.5,
                    color="#4a4438")
    deco(ax, "le mot\n(2 mesures)", "#4a4438", hard=False)

    # 2 ── les voix de chaque pic
    ax = axs[1]
    vmax = max([v["votes"] for v in R["pics"]] + [1])
    for v in R["pics"]:
        h = v["votes"] / vmax
        on = v["votes"] >= MIN_VOTES
        ax.plot([v["bar"], v["bar"]], [0, max(h, 0.06) * 0.82], lw=3.2,
                color=HARD if on else "#c3ccd2", solid_capstyle="butt")
        if v["votes"]:
            ax.text(v["bar"], h * 0.82 + 0.06, str(v["votes"]), ha="center",
                    fontsize=7.5, color=HARD if on else "#a8b0b6")
    deco(ax, "voix\n(sur 7)", "#4a4438", hard=False)

    # 3 ── l'ordre dans lequel les blocs ont été remplis
    ax = axs[2]
    for i, bl in enumerate(R["blocs"]):
        x0, x1 = a + 2 * bl["j0"], a + 2 * bl["j1"]
        g = 0.86 - 0.5 * (bl["rang"] / max(1, len(R["blocs"]) - 1))
        ax.add_patch(plt.Rectangle((x0, 0.2), x1 - x0, 0.6,
                                   facecolor=str(g), edgecolor="#fffdf6", lw=1.2))
        ax.text((x0 + x1) / 2, 0.5, f"{bl['rang'] + 1}", ha="center", va="center",
                fontsize=8.5, color="#fffdf6" if g < 0.6 else "#4a4438")
    deco(ax, "ordre de\nremplissage", "#4a4438")

    # 4 ── l'HISTORIQUE, section par section, dans l'ordre où on les écrit.
    #      Chaque ligne montre l'état de la partition après une pose : ce qui
    #      était déjà là en pâle, ce qu'on vient d'écrire en couleur. C'est la
    #      demande de Louis (« l'historique de comment les sections sont
    #      inférées de manière incrémentale, section par section ») et c'est la
    #      seule vue où l'on voit POURQUOI l'ordre compte : une section « connue »
    #      ne peut l'être que d'une ligne écrite plus haut.
    cm = colourmap()
    for k, ax in enumerate(axs[3:3 + len(hist)]):
        for s in hist[:k + 1]:
            new = s is hist[k]
            w = s["b1"] - s["b0"] + 1
            ax.add_patch(plt.Rectangle((s["b0"], 0.12), w, 0.76,
                                       facecolor=cm(s["label"]),
                                       alpha=1.0 if new else 0.28,
                                       edgecolor="#fffdf6", lw=0.9))
            if new and w >= max(2, n * 0.03):
                ax.text(s["b0"] + w / 2, 0.5, str(s["label"]), ha="center",
                        va="center", fontsize=7.5, color=INK)
        s = hist[k]
        deco(ax, f"{k + 1}. {s['label']} · {s['regle']}", "#6f6858", hard=False)

    # 5 ── les bandes de sections
    for k, (ax, (lab, ss)) in enumerate(zip(axs[3 + len(hist):], strips)):
        for s in ss:
            w = s["b1"] - s["b0"] + 1
            ax.add_patch(plt.Rectangle((s["b0"], 0.08), w, 0.84,
                                       facecolor=cm(s["label"]),
                                       edgecolor="#fffdf6", lw=1.1))
            if w >= max(4, n * 0.04):        # sinon les étiquettes se chevauchent
                ax.text(s["b0"] + w / 2, 0.5, str(s["label"]), ha="center",
                        va="center", fontsize=8.5, color=INK)
        for g in gtb:
            ax.axvline(g, color=GT_LINE, lw=0.7, alpha=0.55)
        deco(ax, lab, INK, last=(k == len(strips) - 1))

    fig.subplots_adjust(left=PLOT_L, right=PLOT_R, top=0.99, bottom=0.055)
    img = fig2b64_fixed(fig)
    mats = matrices_png(R, n, gtb)

    ordre = "".join(
        f'<button class=blk data-p="[{a + 2 * bl["j0"]},{a + 2 * bl["j1"]}]">'
        f'{bl["rang"] + 1}<small>{bl["voix"] if bl["voix"] < 99 else "bord"} voix · '
        f'{word[bl["j0"]:bl["j1"]]}</small></button>'
        for bl in sorted(R["blocs"], key=lambda x: x["rang"]))
    secs = "".join(
        f'<button class=blk data-p="[{s["b0"]},{s["b1"] + 1}]">{s["label"]}'
        f'<small>mes. {s["b0"] + 1}–{s["b1"] + 1} · {s["regle"]}</small></button>'
        for s in R["sections"])

    body = f"""<section><h2>{title} <span class=sub>{n} mesures · mot de
{J} bi-mesures</span></h2>
<div class=plot><img src="data:image/png;base64,{img}">
<div class=cur></div><div class=hit></div></div>
<div class=bar><button class=pp>▶</button><span class=pos>mes. 1 · 0:00</span>
<span class=hint>touche le graphique pour te déplacer</span></div>
<div class=lane><span class=lab>l'ordre</span>{ordre}</div>
<div class=lane><span class=lab>ce qu'on pose</span>{secs}</div>
<div class=votes><b>Le mot —</b> <code>{word}</code></div>
<img src="data:image/png;base64,{mats}" alt="matrices">
<div class=votes><b>À gauche</b>, la corrélation entre toutes les bi-mesures :
c'est la matière brute d'où sort le mot. Un carré sombre hors diagonale = deux
bi-mesures qui se ressemblent. Traits bleus, nos sections ; rouges, les
tiennes.<br><b>À droite</b>, la transition d'une bi-mesure à la suivante — la
grammaire du morceau. Une diagonale forte = une boucle lente ; un aller-retour
<code>a→b→a</code> = un balancement de deux mesures ; une case isolée = un
passage qui n'arrive qu'une fois.</div>
</section>
<audio id=au preload=metadata playsinline src="/audio/{stem}.m4a"></audio>
<script>window.GRID={[round(t, 3) for t in grid]};
window.PLOT=[{PLOT_L},{PLOT_R}]; window.U=1;</script>"""
    return page(f"{title} — voix et bi-mesures", body, back=True,
                lede_html=LEDE, back_href="vote_fill.html",
                back_label="tous les morceaux")


LEDE = f"""<div class=lede><b>Bande 1 — le mot de deux mesures.</b> Chaque
bi-mesure reçoit une lettre d'après la matrice d'accords : deux bi-mesures qui se
ressemblent portent la même lettre. Le morceau devient une chaîne, et la
structure s'y lit directement — Blue Lights donne <code>aaaaaabc</code> cinq fois
de suite, et le <code>bc</code> est exactement chacun de tes B.<br><br>
<b>Bande 2 — les voix.</b> La hauteur d'un trait est le nombre de matrices, sur
sept, qui proposent ce pic. En noir celles qu'on garde ({MIN_VOTES} voix ou
plus), en gris les hypothèses.<br><br>
<b>Bande 3 — l'ordre.</b> Les pics gardés découpent le morceau en blocs ; le
numéro est l'ordre dans lequel on les remplit, <b>du mieux voté au moins
voté</b>. Le premier bloc traité définit sa lettre ; les suivants ne peuvent que
se comparer à ce qui est déjà écrit. C'est tout ce que l'ordre fait — et c'est
pour ça qu'il change le résultat.<br><br>
<b>En dessous</b> : ce que le remplissage écrit, ce qu'on écrit aujourd'hui, et
ton découpage (traits rouges). <b>Touche un bloc ou une section pour
l'écouter.</b></div>"""


def dump():
    for stem, title in SONGS:
        b = order_bundle.get(stem)
        R = fill(b, stem)
        gt = gt_sections(stem)
        print(f"\n{title}  ({b['n']} mes., ancre {R['ancre']})")
        print(f"  mot   {R['mot']}")
        print("  voix  " + " ".join(f"{v['bar'] + 1}:{v['votes']}"
                                    for v in R["pics"] if v["votes"] >= 1))
        print("  ordre " + " ".join(
            f"[{bl['rang'] + 1}]{bl['j0']}-{bl['j1']}({bl['voix']})"
            for bl in sorted(R["blocs"], key=lambda x: x["rang"])))
        print("  nous  " + " ".join(f"{s['label']}{s['b0'] + 1}-{s['b1'] + 1}"
                                    for s in R["sections"]))
        if gt:
            print("  toi   " + " ".join(f"{s['label']}{s['b0'] + 1}-{s['b1'] + 1}"
                                        for s in gt["sections"]))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--dump" in sys.argv:
        dump(); return
    todo = [(s, t) for s, t in SONGS if s in args] or (
        [(s, s) for s in args] if args else list(SONGS))
    rows = []
    for stem, title in todo:
        if not (AUDIO / f"{stem}.m4a").exists():
            continue
        (OUTDIR / f"vote_{stem}.html").write_text(song_page(stem, title))
        rows.append(f'<tr><td><a href="vote_{stem}.html">{title}</a></td></tr>')
        print(f"  ok {title}")
    (OUTDIR / "vote_fill.html").write_text(page(
        "Voix et bi-mesures", "<section><table class=idx>" + "".join(rows)
        + "</table></section>", lede_html=LEDE, back_href="vote_fill.html"))
    print(f"wrote docs/plots/vote_fill.html + {len(rows)} pages")


if __name__ == "__main__":
    main()
