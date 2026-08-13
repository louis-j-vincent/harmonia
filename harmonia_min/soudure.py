"""harmonia_min/soudure.py — le mot d'un chart, pour le jeu de soudure.

Louis, 2026-08-13 : « mets le moi comme une option sur chaque chanson dans le
chart, car c'est vraiment une interface hyper pratique. »

La page (`docs/plots/soudure.html`) attend une chanson sous forme de bande de
jetons : une lettre par bi-mesure, deux jetons de même lettre étant le même
endroit du morceau. Ce module fabrique cette bande à partir du chart que l'app
sert déjà — donc sans audio, sans analyse, et surtout **sans second détecteur
de similarité**.

C'est la règle que `section_tool.py` a posée et qui vaut ici mot pour mot :
écrire un second scorer de similarité créerait deux vérités divergentes pour la
même question, et l'outil pourrait contredire le chart qu'il annote. On ne
compare donc rien : deux bi-mesures portent la même lettre si et seulement si
elles jouent **exactement la même basse**, mesure par mesure. C'est de
l'égalité, pas de la ressemblance ; il n'y a ni seuil, ni réglage, ni modèle.

POURQUOI LA BASSE, ET PAR MESURE. Le mot du projet est passé aux accords à la
BASSE le 2026-08-12 (commit c3802b1) ; on suit. Le grain est la mesure, pas le
temps, et c'est mesuré : sur les 45 charts servis, la part de bi-mesures qui
appartiennent à une signature vue au moins deux fois est de

    basse par mesure   médiane 94 %   min 60 %    <- retenu
    basse par temps    médiane 62 %   min 13 %
    accord par temps   médiane 49 %   min  5 %

Au grain du temps, un morceau sur cinq n'a presque rien à souder — le jeu
n'aurait pas de matière. Au grain de la mesure, il en a partout.

CE QUE ÇA NE RÉSOUT PAS. L'égalité stricte SOUS-GROUPE : deux passages qui se
ressemblent sans être identiques reçoivent deux lettres différentes, et la
soudure « partout à la fois » ne les attrapera pas ensemble. C'est le sens de
la préférence de Louis (« under-fold, never over-fold ») et le jeu permet de
les souder à la main, mais il faut le savoir : ce mot-ci est plus bavard que
celui de `vote_fill.fill`, qui lui groupe par similarité de SSM.

Et sur un morceau à un seul accord (Chain of Fools), la basse ne dit rien : le
mot devient une seule lettre répétée. La page reste jouable, mais elle n'a plus
de structure à montrer — c'est honnête, ce n'est pas utile.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# 62 symboles, pas 26. Le maximum observé sur les 45 charts est 29 bi-mesures
# distinctes (H D3Vffhvs4, 106 mesures) : à 26 lettres, trois morceaux voyaient
# des bi-mesures DIFFÉRENTES recevoir la même lettre, donc se faire souder
# ensemble comme si c'était le même endroit. La page ne fait qu'identifier des
# symboles, elle ne les lit pas — le jeu de caractères peut être large.
LETTRES = ("abcdefghijklmnopqrstuvwxyz"
           "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
           "0123456789")


def _basse_par_mesure(chart: dict) -> list[tuple]:
    """Une signature par mesure : (classe de hauteur de la basse, silence).

    La basse d'une mesure est celle de l'accord qui y dure le plus longtemps.
    `bass` vaut -1 quand le chart n'a pas d'inversion : la fondamentale est
    alors la basse qui sonne, ce qui est la définition retenue par le projet
    (`corpus_schema.sounding_bass_pc`).
    """
    grid = chart.get("barGrid") or []
    accords = (chart.get("prompter") or {}).get("chords") or []
    out = []
    for b in range(len(grid) - 1):
        t0, t1 = grid[b], grid[b + 1]
        duree: dict = {}
        for c in accords:
            chevauche = min(t1, c.get("t1", 0.0)) - max(t0, c.get("t0", 0.0))
            if chevauche <= 0:
                continue
            bass = c.get("bass", -1)
            if bass is None or bass < 0:
                bass = c.get("root", -1)
            cle = (int(bass), bool(c.get("nc")))
            duree[cle] = duree.get(cle, 0.0) + chevauche
        out.append(max(duree, key=duree.get) if duree else (-1, True))
    return out


def _jetons(n_mesures: int) -> list[int]:
    """Les bornes des bi-mesures. La mesure orpheline d'un morceau impair est
    rattachée au dernier jeton plutôt que jetée — sinon la fin du morceau
    disparaît de la bande sans que rien ne le dise."""
    if n_mesures < 2:
        return [0, n_mesures]
    bornes = list(range(0, n_mesures - 1, 2))
    bornes.append(n_mesures)
    return bornes


def mot_du_chart(chart: dict) -> dict | None:
    """{mot, jetons, n_mesures, temps_par_mesure, t0} — ou None si le chart
    n'a pas de quoi faire une bande."""
    grid = chart.get("barGrid") or []
    n = len(grid) - 1
    if n < 2:
        return None
    sig = _basse_par_mesure(chart)
    bornes = _jetons(n)

    vus: dict = {}
    lettres = []
    for j in range(len(bornes) - 1):
        cle = tuple(sig[m] for m in range(bornes[j], bornes[j + 1]))
        if cle not in vus:
            if len(vus) < len(LETTRES):
                vus[cle] = LETTRES[len(vus)]
            else:
                # Au-delà de 62, deux bi-mesures DIFFÉRENTES se diraient
                # identiques et se feraient souder ensemble. Jamais vu (29 au
                # maximum), mais si ça arrive on le dit plutôt que de mentir.
                vus[cle] = LETTRES[-1]
                log.warning("soudure: plus de %d bi-mesures distinctes — les "
                            "dernières partagent un symbole", len(LETTRES))
        lettres.append(vus[cle])

    return {
        "n_mesures": n,
        "temps_par_mesure": [round(grid[i + 1] - grid[i], 4) for i in range(n)],
        "mot": "".join(lettres),
        "jetons": bornes,
        "t0": round(grid[0], 4),
    }


def _otsu(v, lo=0.30, hi=0.999, n=200, defaut=0.90) -> float:
    """Le seuil qui sépare le mieux la distribution du morceau en deux paquets.

    Repris tel quel de `scripts/vote_fill.otsu`, avec sa raison d'être :
    mesuré le 2026-08-12, la médiane des ressemblances entre bi-mesures vaut
    0,33 sur Sunny, 0,90 sur Let It Be et 0,99 sur Blue Lights. Un seuil fixe
    tombe donc au milieu de la distribution d'un morceau et au ras d'un autre.
    Otsu lit la FORME de la distribution, pas son niveau.
    """
    import numpy as np
    v = np.asarray(v, float)
    v = v[(v >= lo) & (v <= hi)]
    if v.size < 8:
        return defaut
    best, seuil = -1.0, defaut
    for t in np.linspace(v.min() + 1e-6, v.max() - 1e-6, n):
        a, c = v[v <= t], v[v > t]
        if a.size < 2 or c.size < 2:
            continue
        w = a.size * c.size * (a.mean() - c.mean()) ** 2
        if w > best:
            best, seuil = w, float(t)
    return seuil


def mot_par_ssm(grid, triad, bornes: list[int]) -> str | None:
    """Le mot par RESSEMBLANCE, sur le substrat que l'app utilise déjà.

    Toujours pas de second détecteur : la matrice est celle de
    `section_tool.substrates` — les vecteurs chord-tone de `harmonic_sections`
    sur les postérieures musx, c'est-à-dire le substrat sur lequel l'app
    répond déjà « où ce bloc se rejoue-t-il ? ». La comparaison bloc-à-bloc
    est `voice_sections._diag`, celle de la prod. Ne sont nouveaux ici que le
    grain (la bi-mesure) et le fait d'en tirer des lettres.

    Le groupage est en LIEN MOYEN, pas complet — Louis, 2026-08-12, sur
    Grenade : en lien complet une seule paire ratée sur six empêche le groupe
    et ses quatre couplets sortaient en `aaaa`, `babb`, `bfbd`, `gfge`.

    Différence assumée avec le mot de recherche (`vote_fill.bibar_word`) :
    celui-ci lit la BASSE, celui-là l'harmonie complète. Même règle de
    groupage, même seuil, substrat différent.
    """
    import numpy as np
    from harmonia_min import section_tool as st
    from harmonia_min import voice_sections as VS

    S, _V, _M, _mute, _ch = st.substrates(grid, triad)
    J = len(bornes) - 1
    if J < 2:
        return None
    B = np.zeros((J, J))
    for j in range(J):
        for k in range(J):
            L = min(bornes[j + 1] - bornes[j], bornes[k + 1] - bornes[k])
            B[j, k] = VS._diag(S, bornes[j], bornes[k], L)
    d = np.sqrt(np.clip(np.diag(B), 1e-9, None))
    B = B / np.outer(d, d)
    thr = _otsu(B[~np.eye(J, dtype=bool)])

    used, lab, k = set(), [-1] * J, 0
    for j in sorted(range(J), key=lambda j: -(B[j] >= thr).sum()):
        if j in used:
            continue
        mem = [j]
        for q in range(J):
            if q in used or q == j:
                continue
            if float(np.mean([B[q, x] for x in mem])) >= thr:
                mem.append(q)
        for q in mem:
            lab[q] = k
        used.update(mem)
        k += 1
    ren, k = {}, 0
    for j in range(J):
        if lab[j] not in ren:
            ren[lab[j]] = k
            k += 1
    return "".join(LETTRES[ren[lab[j]] % len(LETTRES)] for j in range(J))


def song_du_chart(chart: dict, audio_dir=None) -> dict | None:
    """La chanson complète attendue par la page, audio et retour compris.

    Le mot vient de la RESSEMBLANCE harmonique quand on peut la calculer, et
    de l'égalité stricte des basses sinon. Mesuré sur 14 charts (les autres
    donnent la même image) :

        mot                soudures   1re paire   jetons restants à la fin
        basse (égalité)        7         ×6              47 %
        harmonie (SSM)         6        ×12              19 %

    Autant de soudures, mais des paires deux fois plus fréquentes et un
    morceau qui se replie deux fois plus loin — l'égalité stricte sous-groupe,
    comme prévu. Le repli est donc un vrai repli, pas un choix : il est nommé
    dans `mot_source` et la page l'affiche, pour qu'on ne juge jamais un
    découpage sans savoir de quel mot il sort.
    """
    base = mot_du_chart(chart)
    if not base:
        return None
    base["mot_source"] = "basse"
    try:
        from pathlib import Path
        from harmonia_min import musx as _musx
        stem = Path(chart.get("audio_url") or "").stem
        audio = (audio_dir or Path("docs/audio")) / f"{stem}.m4a"
        if stem and audio.exists():
            m = mot_par_ssm(chart["barGrid"], _musx.frame_posteriors(audio)[0],
                            base["jetons"])
            if m:
                base["mot"] = m
                base["mot_source"] = "harmonie"
    except Exception:                                    # noqa: BLE001
        log.exception("soudure: mot par ressemblance indisponible, "
                      "repli sur la basse")

    base["titre"] = chart.get("title") or chart.get("file") or "Sans titre"
    base["audio_url"] = chart.get("audio_url") or None
    if chart.get("file"):
        base["retour"] = {"href": "/?open=" + chart["file"], "label": "le chart"}
    return base
