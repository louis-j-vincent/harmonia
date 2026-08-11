"""Toutes les matrices SSM d'un morceau, côte à côte, sur la même grille.

    .venv/bin/python scripts/ssm_zoo.py [<stem> ...]   ->  /plots/ssm_zoo.html

Louis, 2026-08-10 :

  « On a la matrice SSM accords qui donne un bon ancrage pour détecter des
    bi-mesures. Là où on pêche, c'est ensuite pour les assembler en sections
    cohérentes. Je pense que la réponse est dans les matrices SSM voix, SSM
    rythme, SSM harmonie (produit scalaire des valeurs NNLS par demi-barre, sur
    la basse et sur l'harmonie), et toute autre matrice utile — si ça nous
    permet de distinguer une ou deux grandes sections, c'est super. »

CE QUE LA PAGE MONTRE. Sept substrats, la MÊME grille de demi-mesures pour tous,
les mêmes morceaux. Aucun chiffre ne commande la page : ce qui décide, c'est de
voir si un bloc se détache à l'œil là où les sections changent. Les traits fins
sur chaque matrice sont les frontières de section que Louis a lui-même validées
(`harmonia_min/state/sections/*.json`) — donc « est-ce que ce substrat sépare
ses sections » se lit directement, sans métrique.

LES SEPT SUBSTRATS, et ce que chacun entend :

  accords          le postérieur d'accords musx projeté sur douze hauteurs.
                   C'EST CELUI DE LA PROD (`harmonic_sections.harmonic_vectors`)
                   — la référence, celle qui ancre déjà bien les bi-mesures.
  basse            les 12 cases graves du NNLS. Quelle note est à la basse.
  harmonie         les 12 cases aiguës du NNLS. Le voicing, pas juste l'accord.
  basse+harmonie   les 24 ensemble, chaque moitié normalisée à part — le
                   substrat du mode `HARMONIA_SECTIONS=chroma`.
  voix             ce que le chant pose comme hauteurs dans la demi-mesure
                   (demucs + pyin, déjà en cache pour le mode prod `voice`).
                   Une demi-mesure sans chant est mise à zéro : un instrumental
                   se voit comme une croix blanche, et c'est une information.
  rythme           le motif de BATTERIE (demucs drums + enveloppe d'attaques en
                   3 bandes, `scratchpad/rhythm_ssm.py`). Deux sections peuvent
                   partager les accords et pas le groove.
  timbre           les MFCC — QUI joue, pas quoi. C'est le substrat classique en
                   MIR pour couper couplet/refrain, parce qu'un refrain ajoute
                   des instruments. Il ne connaît rien à l'harmonie : c'est
                   justement pour ça qu'il est ajouté ici.
  fusion           la moyenne des sept, chacune ramenée à son propre rang (une
                   matrice qui vit à 0,9 partout ne doit pas écraser une qui vit
                   à 0,3). Illustration seulement — pas un modèle.

CE QUE ÇA NE FAIT PAS. Aucune détection de sections n'est lancée ici, aucun score
n'est calculé : c'est une page de lecture. Le pas d'après — lire ces matrices
ensemble pour poser des frontières — n'est pas écrit.

Coût mesuré (Mac M-series, caches chauds) : ~4 s par morceau. À froid, la
séparation demucs de la batterie domine (~40 s) ; tout le reste est déjà en cache
pour la prod.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
import numpy as np                                   # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
sys.path.insert(0, str(HERE / "scratchpad"))

OUT = HERE / "docs" / "plots" / "ssm_zoo.html"
ANN = HERE / "harmonia_min" / "state" / "sections"
AUDIO = HERE / "docs" / "audio"

# Les morceaux déjà bien étudiés : ceux dont Louis a validé le découpage.
SONGS = [
    ("maroon_5_this_love", "This Love"),
    ("norah_jones_don_t_know_why", "Don't Know Why"),
    ("bein_green", "Bein' Green"),
    ("let_it_be_remastered_2009", "Let It Be"),
    ("bobby_hebb_sunny_official_audio", "Sunny"),
    ("jorja_smith_blue_lights_a_colors_show", "Blue Lights"),
    ("ben_e_king_stand_by_me_audio", "Stand By Me"),
    ("maroon_5_she_will_be_loved_official_music_video", "She Will Be Loved"),
    ("the_police_every_breath_you_take_official_music_video", "Every Breath You Take"),
    ("aretha_franklin_chain_of_fools_official_lyric_video", "Chain of Fools"),
    ("mayer_hawthorne_the_walk", "The Walk"),
    ("bruno_mars_grenade_official_music_video", "Grenade"),
]

# Ce que je LIS sur chaque morceau, écrit après avoir regardé la page — une
# phrase, la lecture et rien d'autre. Vide = je n'ai pas encore regardé.
READINGS: dict[str, str] = {
    "maroon_5_this_love":
        "l'harmonie donne le grain fin, mais ses blocs se ressemblent tous ; "
        "le timbre coupe le morceau en deux grands pans que rien d'autre ne voit.",
    "norah_jones_don_t_know_why":
        "l'harmonie est la même boucle du début à la fin. Ce qui découpe ici, "
        "c'est la VOIX : sa bande blanche isole d'un coup le passage instrumental.",
    "bein_green":
        "batterie quasi muette (voix + piano) — la matrice rythme ne lit que du "
        "bruit de séparation, et c'est le timbre qui pose des bandes sur tes B.",
    "let_it_be_remastered_2009":
        "l'harmonie est PLATE : la même boucle partout, aucune information de "
        "section. Le rythme et le timbre portent tout, avec un seul pic net au milieu.",
    "bobby_hebb_sunny_official_audio":
        "le morceau monte d'un demi-ton à chaque reprise : les matrices "
        "harmoniques éclatent, le timbre et le rythme — aveugles à la "
        "transposition — gardent un bloc solide.",
    "jorja_smith_blue_lights_a_colors_show":
        "la matrice accords est un peigne parfaitement régulier : elle donne la "
        "PÉRIODE de la boucle, jamais l'ÉCHELLE de la section. Le rythme, lui, "
        "sépare le corps du morceau de l'intro et de l'outro.",
    "ben_e_king_stand_by_me_audio":
        "une seule boucle sur 86 mesures : les quatre matrices harmoniques sont "
        "un damier uniforme. Seul le timbre voit l'arrangement qui s'épaissit.",
    "maroon_5_she_will_be_loved_official_music_video":
        "le cas où le timbre fait tout le travail : gros blocs nets, dont les "
        "bords tombent sur tes frontières.",
    "the_police_every_breath_you_take_official_music_video":
        "le pont sort comme une croix claire dans TOUTES les matrices — le cas "
        "facile, où les sept substrats disent la même chose au même endroit.",
    "aretha_franklin_chain_of_fools_official_lyric_video":
        "un seul vamp du début à la fin, et les substrats s'accordent sur UN "
        "seul contraste : ton B du milieu. C'est peu, et c'est exactement ce "
        "qu'il faut — la prod en sortait cinq lettres.",
    "mayer_hawthorne_the_walk":
        "les quatre matrices harmoniques donnent la même grille de 4 mesures, "
        "très régulière ; les pics du timbre sont plus rares et plus hauts — le "
        "seul substrat qui propose une échelle plus grande que la boucle.",
    "bruno_mars_grenade_official_music_video":
        "le timbre découpe en grands pans (couplet nu / refrain plein) là où "
        "l'harmonie rejoue la même cadence partout.",
}

# Sequential, une seule teinte, clair -> foncé : une similarité est une
# MAGNITUDE. Un arc-en-ciel (RdYlBu) invente des frontières là où la valeur
# monte régulièrement, et c'est exactement ce qu'on cherche à lire ici.
CMAP = LinearSegmentedColormap.from_list(
    "harmonia_seq", ["#fbf7ec", "#cfe0ea", "#84b3cf", "#3d7fa6", "#1b4a6b", "#0d2437"])
GT_LINE = "#b4472c"     # les frontières validées par Louis


class _Stop(Exception):
    """Sort du pipeline dès que la grille est connue — inutile de détecter les
    sections pour dessiner des matrices."""


# ── 1. la grille de mesures + les features déjà calculées par la prod ────────

def capture(stem: str) -> dict:
    """`grid` (bornes de mesures), `arr`/`times` (NNLS bothchroma), `triad`
    (postérieurs musx), sans lancer la détection de sections.

    On espionne `sections.detect_sections` — le seul endroit du pipeline où ces
    quatre objets existent en même temps — et on interrompt là. Détecter les
    sections coûterait 96-98 % du temps (docstring de `pipeline.analyze_steps`)
    pour un résultat dont cette page n'a pas besoin.
    """
    from harmonia_min import pipeline as _pl
    from harmonia_min import sections as hs
    real = hs.detect_sections
    cap: dict = {}

    def spy(grid, arr, times, bars=None, **kw):
        cap.update(grid=list(grid), arr=np.asarray(arr), times=np.asarray(times),
                   triad=kw.get("triad"), bars=bars)
        raise _Stop

    hs.detect_sections = spy
    try:
        for _ in _pl.analyze_steps(AUDIO / f"{stem}.m4a", title="x",
                                   file_key="x", audio_url=""):
            pass
    except _Stop:
        pass
    finally:
        hs.detect_sections = real
    if "grid" not in cap:
        raise RuntimeError(f"{stem}: la grille n'a pas été capturée")
    return cap


def halfbar_edges(grid: list[float]) -> list[float]:
    """Les bornes de demi-mesures — la granularité que Louis a demandée."""
    e = []
    for b in range(len(grid) - 1):
        mid = 0.5 * (grid[b] + grid[b + 1])
        e += [grid[b], mid]
    e.append(grid[-1])
    return e


def _unit(V: np.ndarray) -> np.ndarray:
    return V / np.clip(np.linalg.norm(V, axis=1, keepdims=True), 1e-9, None)


def _pool(arr: np.ndarray, times: np.ndarray, edges: list[float]) -> np.ndarray:
    """Moyenne des trames dans chaque case de la grille."""
    out = np.zeros((len(edges) - 1, arr.shape[1]))
    for i in range(len(edges) - 1):
        sel = (times >= edges[i]) & (times < edges[i + 1])
        if sel.any():
            out[i] = arr[sel].mean(0)
        else:
            out[i] = arr[int(np.argmin(np.abs(times - 0.5 * (edges[i] + edges[i + 1]))))]
    return out


# ── 2. les substrats ────────────────────────────────────────────────────────

def sub_accords(cap, edges):
    """Le substrat de la PROD : postérieur musx -> 12 hauteurs."""
    import harmonia_min.harmonic_sections as HS
    V = HS.harmonic_vectors(cap["triad"], edges)
    return np.clip(V @ V.T, 0, 1)


def _nnls_halves(cap, edges):
    P = _pool(cap["arr"], cap["times"], edges)
    return _unit(P[:, :12]), _unit(P[:, 12:])


def sub_nnls(cap, edges):
    """basse / harmonie / les deux — produit scalaire des valeurs NNLS."""
    Vb, Vt = _nnls_halves(cap, edges)
    both = _unit(np.concatenate([Vb, Vt], axis=1))
    return (np.clip(Vb @ Vb.T, 0, 1), np.clip(Vt @ Vt.T, 0, 1),
            np.clip(both @ both.T, 0, 1))


def sub_voix(stem, edges):
    """Ce que le chant pose comme hauteurs, demi-mesure par demi-mesure.

    Même chaîne que le mode prod `voice` : demucs (piste vocale) -> pyin ->
    notes nettoyées. Les deux sont en cache dès qu'un morceau est passé en prod.
    """
    import vocal_anchor as VA
    import vocal_melody as VM
    voc = VA.separate_vocals(AUDIO / f"{stem}.m4a")
    tt, ff, vv, rr = VM.track_f0(voc)
    notes, _ = VM.melody_notes(tt, ff, vv, rr)
    notes, _ = VM.clean(notes)
    n = len(edges) - 1
    P = np.zeros((n, 12))
    for t0, d, m in notes:
        i = int(np.searchsorted(edges, t0) - 1)
        if 0 <= i < n:
            P[i, int(m) % 12] += d
    mute = P.sum(1) <= 0
    V = _unit(P)
    V[mute] = 0.0
    S = np.clip(V @ V.T, 0, 1)
    return S, mute


def sub_rythme(stem, grid):
    """Le motif de batterie, + le POIDS de la batterie dans le mix.

    `variant="smooth"` = le défaut faible de `rhythm_ssm` (aucune variante ne
    domine sur les 3 morceaux calibrés).

    Le poids n'est pas une décoration : sur Bein' Green la piste batterie sort à
    0,007 de RMS contre 0,105 sur This Love. La matrice existe quand même et
    ressemble à quelque chose — elle ne lit que du bruit de séparation. Sans ce
    chiffre affiché, c'est indistinguable d'un vrai groove uniforme.
    """
    from rhythm_ssm import rhythm_ssm, separate_drums
    S = np.asarray(rhythm_ssm(AUDIO / f"{stem}.m4a", grid, slots_per_bar=2,
                              variant="smooth"), dtype=float)
    rms = None
    d = separate_drums(AUDIO / f"{stem}.m4a")
    if d is not None:
        import soundfile as sf
        y, _ = sf.read(str(d))
        y = y.mean(1) if y.ndim > 1 else y
        rms = float(np.sqrt((y ** 2).mean()))
    return S, rms


def sub_timbre(stem, edges):
    """QUI joue : MFCC moyens par demi-mesure, centrés (corrélation), ramenés
    sur [0,1] comme les autres."""
    import librosa
    y, sr = librosa.load(str(AUDIO / f"{stem}.m4a"), sr=22050, mono=True)
    M = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=512)
    t = librosa.frames_to_time(np.arange(M.shape[1]), sr=sr, hop_length=512)
    P = _pool(M.T, t, edges)
    P = P - P.mean(0, keepdims=True)          # le timbre MOYEN du morceau n'est
    V = _unit(P)                              # pas une information de section
    return (np.clip(V @ V.T, -1, 1) + 1.0) / 2.0


def _rankify(S: np.ndarray) -> np.ndarray:
    """La matrice ramenée à ses propres rangs, dans [0,1]. Sans ça la fusion est
    décidée par le substrat au fond le plus haut, pas par le plus informatif."""
    flat = S.reshape(-1)
    order = np.argsort(np.argsort(flat))
    return (order / max(1, len(flat) - 1)).reshape(S.shape)


def substrates(stem: str) -> tuple[list[tuple], int, dict, dict]:
    """[(nom, glose, matrice)] + n_mesures + le chrono + les à-côtés.

    Les à-côtés (`extra`) portent ce dont les pages en aval ont besoin et qui
    n'est pas une matrice : la grille de mesures en secondes (pour la tête de
    lecture), le masque des demi-mesures sans chant (les instrumentaux), le RMS
    de la piste batterie.
    """
    T: dict[str, float] = {}
    t = time.time(); cap = capture(stem); T["grille + NNLS + musx"] = time.time() - t
    grid = cap["grid"]
    n = len(grid) - 1
    edges = halfbar_edges(grid)

    out = []
    t = time.time(); S = sub_accords(cap, edges); T["accords"] = time.time() - t
    out.append(("accords", "le postérieur musx sur 12 hauteurs — CELUI DE LA PROD", S))

    t = time.time(); Sb, St, Sbt = sub_nnls(cap, edges); T["NNLS ×3"] = time.time() - t
    out.append(("basse", "les 12 cases graves du NNLS", Sb))
    out.append(("harmonie", "les 12 cases aiguës du NNLS", St))
    out.append(("basse + harmonie", "les 24, chaque moitié normalisée à part", Sbt))

    t = time.time(); Sv, mute = sub_voix(stem, edges); T["voix"] = time.time() - t
    out.append(("voix", "les hauteurs chantées ; blanc = personne ne chante", Sv))

    t = time.time(); Sr, rms = sub_rythme(stem, grid); T["rythme"] = time.time() - t
    if Sr.shape[0] == len(edges) - 1:
        g = "le motif de batterie (demucs + attaques 3 bandes)"
        if rms is not None:
            g += (f" · piste batterie à {rms:.3f} de RMS"
                  + (" — <b>quasi muette, cette matrice ne lit que du bruit "
                     "de séparation</b>" if rms < 0.02 else ""))
        out.append(("rythme", g, Sr))

    t = time.time(); Sm = sub_timbre(stem, edges); T["timbre"] = time.time() - t
    out.append(("timbre", "MFCC — QUI joue, pas quoi", Sm))

    F = np.mean([_rankify(S) for _, _, S in out], axis=0)
    out.append(("fusion", "la moyenne des sept, chacune ramenée à ses rangs", F))
    extra = {"grid": grid, "mute": mute, "drums_rms": rms, "edges": edges,
             "triad": cap["triad"]}
    return out, n, T, extra


# ── 3. la page ──────────────────────────────────────────────────────────────

def _novelty(S: np.ndarray) -> np.ndarray:
    """La nouveauté en damier, normalisée à 1 — celle de la prod, pas une
    réécriture (`harmonia_min.sections._novelty`, noyau `KERNEL_HB` = 8 mesures).
    Négatif mis à zéro : une frontière est un creux de similarité croisée, et
    le signe négatif ne dit rien de plus qu'« ici c'est homogène ».

    Les `KERNEL_HB` cases de chaque bout sont MASQUÉES, comme dans la prod : le
    noyau y est tronqué et déséquilibré, ses valeurs sont des artefacts (et
    c'est ce qui avait fait rater tous les vrais pics de This Love).
    """
    from harmonia_min.sections import _novelty as prod_novelty, KERNEL_HB
    nov = np.clip(prod_novelty(S, KERNEL_HB), 0, None)
    kw, m = KERNEL_HB, len(nov)
    if m <= 2 * kw:
        return np.full(m, np.nan)
    out = np.full(m, np.nan)
    out[kw:m - kw] = nov[kw:m - kw] / max(1e-9, float(nov[kw:m - kw].max()))
    return out


def gt_sections(stem: str):
    p = ANN / f"{stem}.json"
    if not p.exists():
        return None
    return json.load(p.open())


def fig2b64(fig) -> str:
    import base64
    from io import BytesIO
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=112, bbox_inches="tight",
                facecolor="#fffdf6")
    plt.close(fig)
    # Palette 128 couleurs : la page fait 12 planches et se charge par Tailscale.
    # Une SSM n'a qu'une rampe de couleurs, donc la quantification ne coûte rien
    # de visible et divise le poids par ~4.
    try:
        from PIL import Image
        buf.seek(0)
        im = Image.open(buf).convert("RGB").quantize(colors=128, method=2)
        small = BytesIO()
        im.save(small, format="PNG", optimize=True)
        if small.tell() < buf.getbuffer().nbytes:
            buf = small
    except Exception:
        pass
    return base64.b64encode(buf.getvalue()).decode()


def song_html(stem: str, title: str) -> tuple[str, dict]:
    subs, n, T, _extra = substrates(stem)
    gt = gt_sections(stem)
    k = len(subs)
    ncol = 4                                   # deux rangées de quatre : à huit
    nrow = int(np.ceil(k / ncol))              # de front les blocs sont illisibles
    fig, axs = plt.subplots(2 * nrow, ncol, figsize=(4.2 * ncol, 5.15 * nrow),
                            facecolor="#fffdf6",
                            gridspec_kw={"height_ratios": [5, 1] * nrow,
                                         "hspace": 0.14, "wspace": 0.13})
    axs = np.atleast_2d(axs)
    cells = [(axs[2 * r, c], axs[2 * r + 1, c])
             for r in range(nrow) for c in range(ncol)]
    for a, b in cells[k:]:
        a.axis("off"); b.axis("off")
    for (ax, axn), (nm, _g, S) in zip(cells, subs):
        m = S.shape[0]
        lo, hi = np.percentile(S, 4), np.percentile(S, 99.5)
        ax.imshow(S, cmap=CMAP, origin="lower", vmin=lo, vmax=max(hi, lo + 1e-6),
                  interpolation="nearest")
        bounds = [sg["b0"] * m / max(1, n) for sg in gt["sections"][1:]] if gt else []
        for x in bounds:                        # les frontières validées
            ax.axhline(x, color=GT_LINE, lw=0.7, alpha=0.75)
            ax.axvline(x, color=GT_LINE, lw=0.7, alpha=0.75)
        ax.set_title(nm, fontsize=13, loc="left", pad=6,
                     color="#8a2b2b" if nm in ("accords", "fusion") else "#1c1c1c")
        ax.set_xticks([]); ax.set_yticks([])

        # LA MÊME LECTURE QUE LA PROD, sous chaque matrice : la nouveauté en
        # damier de `harmonia_min.sections._novelty`, noyau 8 mesures. Un pic sur
        # un trait rouge = cette frontière-là est trouvable dans ce substrat.
        nov = _novelty(S)
        axn.fill_between(np.arange(m), nov, color="#3d7fa6", lw=0, alpha=0.85)
        for x in bounds:
            axn.axvline(x, color=GT_LINE, lw=0.7, alpha=0.75)
        axn.set_xlim(0, m - 1); axn.set_ylim(0, 1.05)
        axn.set_xticks([]); axn.set_yticks([])
        for a in (ax, axn):
            for s in a.spines.values():
                s.set_color("#d8cfb8")
    img = fig2b64(fig)

    letters = ""
    if gt:
        letters = " · ".join(f"{sg['label']}<span class=bars>{sg['b1']-sg['b0']+1}</span>"
                             for sg in gt["sections"])
    gloss = "".join(f"<div class=cell><b>{nm}</b><br><span class=note>{g}</span></div>"
                    for nm, g, _S in subs)
    read = READINGS.get(stem, "")
    read = f"<div class=read><b>Ce que ça dit —</b> {read}</div>" if read else ""
    return (f"""<section id="{stem}"><h2>{title}
<span class=sub>{n} mesures · grille demi-mesure ({2*n} cases)</span></h2>
<div class=truth>Découpage validé par toi : {letters or "—"}
<span class=note>— les traits rouges sur les matrices</span></div>
<img src="data:image/png;base64,{img}" alt="matrices SSM de {title}">
{read}<div class=cells>{gloss}</div></section>""", T)


def main():
    stems = sys.argv[1:] or [s for s, _ in SONGS]
    todo = [(s, t) for s, t in SONGS if s in stems] or [(s, s) for s in stems]
    body, chrono = "", []
    for stem, title in todo:
        if not (AUDIO / f"{stem}.m4a").exists():
            print(f"  ?? pas d'audio pour {stem}")
            continue
        t0 = time.time()
        html, T = song_html(stem, title)
        body += html
        chrono.append((title, time.time() - t0, T))
        print(f"  ok {title:24s} {time.time()-t0:5.1f} s  " +
              " ".join(f"{k} {v:.1f}" for k, v in T.items()))

    keys = list(chrono[0][2]) if chrono else []
    rows = "".join(
        "<tr><td>" + t + "</td>" +
        "".join(f"<td>{T.get(k, 0):.1f}</td>" for k in keys) +
        f"<td><b>{tot:.1f}</b></td></tr>" for t, tot, T in chrono)
    head = "".join(f"<th>{k}</th>" for k in keys)
    table = (f"<table><tr><th>morceau</th>{head}<th>total</th></tr>{rows}</table>"
             if chrono else "")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Le zoo des matrices SSM</title><style>
body{{margin:0;background:#e7e0d0;font:15px/1.55 -apple-system,system-ui,sans-serif;color:#1c1c1c}}
.wrap{{max-width:1700px;margin:0 auto;padding:22px 15px 70px}}
h1{{font:italic 600 26px Georgia,serif;margin:0 0 6px}}
.lede{{color:#6f6857;font-size:13.5px;margin-bottom:20px;max-width:900px}}
.lede b{{color:#1c1c1c}}
section{{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;padding:16px 18px;margin-bottom:14px}}
h2{{font:700 18px system-ui;margin:0 0 4px;color:#8a2b2b}}
.sub{{font:500 12px system-ui;color:#8a8371}}
.truth{{font-size:12.5px;color:#1c1c1c;margin:0 0 10px}}
.truth .bars{{color:#8a8371;font-size:11px;vertical-align:super;margin-left:2px}}
img{{max-width:100%;border-radius:8px;display:block}}
.takeaway{{margin:6px 0 0;padding-left:20px;font-size:13.5px;max-width:980px}}
.takeaway li{{margin-bottom:7px}}
.read{{margin-top:11px;font-size:13.5px;background:#f4efe1;border-left:3px solid #b4472c;
border-radius:0 8px 8px 0;padding:8px 12px}}
.cells{{display:flex;gap:8px;margin-top:10px;flex-wrap:wrap}}
.cell{{flex:1;min-width:135px;font-size:11.5px;background:#f7f3e9;border-radius:8px;padding:6px 9px}}
.note{{color:#8a8371}}
table{{border-collapse:collapse;font-size:12.5px;margin-top:8px}}
th,td{{padding:4px 10px;text-align:right;border-bottom:1px solid #ece4d2}}
th:first-child,td:first-child{{text-align:left}}
th{{color:#8a8371;font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.04em}}
</style></head><body><div class=wrap>
<h1>Le zoo des matrices SSM</h1>
<div class=lede>Sept façons de mesurer « est-ce que ces deux demi-mesures se
ressemblent », le même morceau, la même grille. <b>Aucun chiffre ne commande
cette page</b> : les traits rouges sont les frontières de section que tu as
validées, donc « ce substrat sépare-t-il tes sections ? » se lit à l'œil.
L'échelle est étirée par matrice (4<sup>e</sup>–99,5<sup>e</sup> centile) — sans
ça, une matrice dont tout le fond est à 0,9 paraît vide.
<b>Ce que la page ne fait pas</b> : elle ne détecte rien et ne score rien.</div>
<section><h2>Ce que je lis, avant que tu regardes <span class=sub>à arbitrer, pas à croire</span></h2>
<ul class=takeaway>
<li><b>Les matrices harmoniques répondent à « où se répète le motif », donc à la
PÉRIODE</b> — 2, 4, 8 mesures. C'est l'ancrage qui marche déjà, et il marche ici
aussi.</li>
<li><b>Elles ne répondent presque jamais à « à quelle ÉCHELLE regrouper ».</b>
Sur une chanson bâtie sur une boucle — Stand By Me, Let It Be, Blue Lights,
Don't Know Why — les quatre matrices harmoniques sont un damier uniforme :
elles ne contiennent pas l'information que tu cherches.</li>
<li><b>Le TIMBRE est le seul substrat qui sort des gros blocs à l'échelle de la
section</b>, y compris là où l'harmonie est plate (Let It Be, Stand By Me) ou
cassée par une modulation (Sunny). Le rythme fait pareil quand la batterie est
vraiment jouée — sur Bein' Green elle est quasi muette et la matrice ne lit que
du bruit.</li>
<li><b>La voix ne dit pas la même chose que les autres</b> : elle marque les
TROUS (l'intro, l'instrumental de Don't Know Why), pas les répétitions.</li>
<li><b>D'où l'hypothèse à arbitrer</b> : les accords disent OÙ couper, le timbre
et le rythme disent À QUELLE ÉCHELLE grouper. Aucune des deux moitiés ne suffit
seule, et c'est peut-être pour ça que l'assemblage coince.</li>
</ul></section>
<section><h2>Combien de temps ça coûte <span class=sub>par morceau, en secondes</span></h2>
<div class=truth>Caches chauds — c'est-à-dire un morceau déjà passé en prod :
les postérieurs musx, le NNLS, la voix demucs et le pyin y sont déjà, la page ne
fait que les relire. <b>Le seul coût neuf est la séparation de la BATTERIE</b>
(demucs, une fois par morceau) : 52 à 60 s mesurées sur les quatre morceaux qui
ne l'avaient pas encore.
<span class=note>La première ligne porte en plus le chargement des
bibliothèques ; les colonnes à 0,0 sont des lectures de cache, pas des calculs
gratuits.</span></div>
{table}</section>
{body}</div></body></html>""")
    print(f"wrote {OUT.relative_to(HERE)} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
