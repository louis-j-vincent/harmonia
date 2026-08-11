"""Les sections, en partant des pics comme PRIORS DURS. À écouter.

    .venv/bin/python scripts/hard_prior_sections.py [--prod] [<stem> ...]
        -> /plots/hard_sections.html + une page par morceau (audio + tête de lecture)

Louis, 2026-08-10 :

  « On commence nos recherches de sections en partant de ces pics, avec la règle
    claire qu'une section ne peut jamais traverser un pic (car chaque pic marque
    un nouveau début de section) -> c'est pour ça qu'il faut bien marquer un A
    comme une succession de petits A de longueur 8 barres par exemple, car un pic
    peut marquer une différence entre un A et le A suivant ! La matrice de voix
    nous donne un indicateur des sections de solo/bridge. »

L'ALGORITHME, en quatre temps, et rien de plus :

  1. **Les pics sont durs.** Le profil fusionné de `peak_profile.py` donne des
     pics ; ils deviennent des frontières que RIEN ne peut traverser. Le morceau
     est donc d'abord découpé en zones, avant toute recherche.
  2. **Une longueur de bloc, une seule, pour tout le morceau.** On la choisit
     parmi 4 / 8 / 16 mesures : celle dont la matrice d'accords dit qu'elle se
     répète le mieux (similarité moyenne entre la mesure b et la mesure b+L).
     C'est le « A comme une succession de petits A de 8 barres » — on n'écrit
     jamais un A de seize mesures, on écrit deux A de huit.
  3. **Chaque zone est pavée de blocs de cette longueur.** Le reste d'une zone
     rejoint le bloc précédent s'il fait moins d'une demi-longueur, sinon il vit
     comme un bloc court. Une zone plus courte qu'un bloc EST un bloc.
  4. **Les lettres viennent des accords**, pas des pics : deux blocs portent la
     même lettre si leur similarité croisée dépasse `SAME` fois leur
     auto-similarité. Un bloc où personne ne chante devient `solo` — c'est
     l'indicateur voix que Louis demande, et il court-circuite la lettre.

CE QUE ÇA NE FAIT PAS, et il faut le lire avant de juger la page (règle n°4) :

  * **Le rappel des pics est bas.** Le profil fusionné rend peu de frontières
     (quatre sur This Love, où Louis en a neuf) : les zones sont donc GRANDES, et
     tout le découpage fin est fait par le pavage régulier, pas par les pics. La
     page montre les deux couleurs séparément pour qu'on voie qui a fait quoi.
  * **Les lettres ne survivent pas à une modulation.** L'identité est lue sur la
     matrice d'accords sans rotation : sur Sunny, qui monte d'un demi-ton à
     chaque reprise, les reprises reçoivent des lettres différentes. La rotation
     existe (`voice_sections._rot_sim`) et n'est pas branchée ici.
  * **Rien n'est en prod.** `harmonia_min/sections.py` n'a pas changé.
"""
from __future__ import annotations

import json
import os
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

from ssm_zoo import SONGS, AUDIO, gt_sections, fig2b64        # noqa: E402
from peak_profile import fused_profile, page, PLOT_L, PLOT_R  # noqa: E402

OUTDIR = HERE / "docs" / "plots"
CACHE = HERE / "scratchpad" / "hard_prior_prod_sections.json"

BLOCKS = (4, 8, 16)    # les longueurs de bloc candidates, en mesures
SAME = 0.94            # au-dessus, deux blocs portent la même lettre. Réglé À
                       # L'ŒIL sur les douze morceaux : à 0,96 Chain of Fools
                       # (un seul vamp) sort cinq lettres, à 0,92 This Love colle
                       # son pont au couplet. Ce n'est pas un optimum mesuré.
MUTE_SOLO = 0.8        # part de demi-mesures sans chant au-delà de laquelle le
                       # bloc est un solo / un pont instrumental
LETTERS = "ABCDEFGHIJKLMNOP"


# ── l'algorithme ────────────────────────────────────────────────────────────

def _block_len(S_acc: np.ndarray, n: int, tol: float = 0.95) -> int:
    """La longueur de bloc du morceau : la PLUS GRANDE qui se répète presque
    aussi bien que la meilleure.

    Prendre simplement le maximum ne marche pas et c'est un piège d'échelle : un
    lag court est toujours plus similaire qu'un lag long (moins de temps pour
    dériver), donc l'argmax choisit systématiquement 4 mesures. Mesuré sur This
    Love : il sortait vingt blocs de 4 là où Louis en écrit huit de 8, et le
    score contre son découpage tombait à 0,652. On garde donc la plus longue
    dont le score reste à `tol` du meilleur — « un A de huit », pas deux de
    quatre.
    """
    m = S_acc.shape[0]
    hb = max(1, m // max(1, n))               # cases par mesure (2)
    sc = {}
    for L in BLOCKS:
        lag = L * hb
        if n < 2 * L or lag >= m:
            continue
        sc[L] = float(np.mean([S_acc[i, i + lag] for i in range(m - lag)]))
    if not sc:
        return BLOCKS[0]
    best = max(sc.values())
    return max(L for L, v in sc.items() if v >= tol * best)


def _tile(zones: list[tuple[int, int]], L: int) -> list[tuple[int, int]]:
    """Chaque zone pavée de blocs de L mesures ; le reste rejoint le bloc
    précédent s'il est plus court qu'une demi-longueur."""
    out = []
    for z0, z1 in zones:                       # z1 exclusif
        b = z0
        while b < z1:
            e = min(b + L, z1)
            if z1 - e and z1 - e < L / 2:      # queue trop courte : on l'absorbe
                e = z1
            out.append((b, e))
            b = e
    return out


def _letters(S_acc: np.ndarray, blocks, n: int, mute) -> list[str]:
    """Les lettres, lues sur les accords. Un bloc muet devient `solo`."""
    m = S_acc.shape[0]
    sc = m / max(1, n)

    def rect(a, b):
        A = slice(int(a[0] * sc), max(int(a[0] * sc) + 1, int(a[1] * sc)))
        B = slice(int(b[0] * sc), max(int(b[0] * sc) + 1, int(b[1] * sc)))
        r = S_acc[A, B]
        return float(r.mean()) if r.size else 0.0

    mu = np.asarray(mute, bool) if mute is not None else np.zeros(m, bool)

    def is_solo(bl):
        a, z = int(bl[0] * sc), max(int(bl[0] * sc) + 1, int(bl[1] * sc))
        seg = mu[a:min(z, len(mu))]
        return bool(len(seg)) and float(seg.mean()) >= MUTE_SOLO

    labs, reps = [], []
    sung = next((i for i, bl in enumerate(blocks) if not is_solo(bl)), len(blocks))
    for i, bl in enumerate(blocks):
        if is_solo(bl):
            # Muet AVANT le premier bloc chanté = l'intro ; muet au milieu = un
            # solo ou un pont instrumental. Les deux sont la même observation
            # (personne ne chante) et deux choses différentes pour un lead sheet.
            labs.append("intro" if i < sung else "solo")
            continue
        hit = None
        for k, ri in enumerate(reps):
            denom = (rect(bl, bl) * rect(blocks[ri], blocks[ri])) ** 0.5
            if denom > 0 and rect(bl, blocks[ri]) / denom >= SAME:
                hit = k
                break
        if hit is None:
            labs.append(LETTERS[len(reps) % len(LETTERS)])
            reps.append(i)
        else:
            labs.append(LETTERS[hit])
    return labs


def voice_with_hard(stem: str, hard: list[int], n: int, extra: dict,
                    triad=None) -> list[dict]:
    """La recherche de la PROD (`voice_sections`), avec les pics comme CONTRAINTE.

    C'est la lecture littérale de ce que Louis a demandé, et la correction de ma
    première tentative : « on commence nos recherches de sections EN PARTANT de
    ces pics, avec la règle qu'une section ne peut jamais traverser un pic ». Il
    a dit contrainte ; j'avais écrit un remplaçant (pavage régulier depuis chaque
    pic), qui rend 0,600 de médiane contre 0,769 pour la prod. Ici on garde la
    recherche qui marche — bloc de 8 glissé sur tout le morceau, toutes ses
    reprises verrouillées d'un coup — et on lui interdit trois choses :

      1. une ancre ne peut pas ENJAMBER un pic (elle saute au pic) ;
      2. une reprise trouvée qui enjambe un pic est rejetée ;
      3. à l'assemblage, une section est coupée sur chaque pic, même si les
         mesures des deux côtés appartiennent au même motif — c'est le « un A de
         seize s'écrit deux A de huit » de Louis.
    """
    import harmonia_min.voice_sections as VS
    from harmonia_min import harmonic_sections as HS
    VA, VM, MS, _B8 = VS._scripts()

    V = HS.harmonic_vectors(triad, extra["grid"])
    S = V @ V.T
    voc = VA.separate_vocals(AUDIO / f"{stem}.m4a")
    tt, ff, vv, rr = VM.track_f0(voc)
    notes, _ = VM.melody_notes(tt, ff, vv, rr)
    notes, _ = VM.clean(notes)
    M, mute = MS.melody_bars(notes, extra["grid"], n)
    start = VS.sung_start(notes, extra["grid"], mute)

    hard = sorted(h for h in hard if 0 < h < n)

    def crosses(p, L):
        return any(p < h < p + L for h in hard)

    def _pass_hard(block, thr, claimed, max_blocks=20):
        runs, cursor = [], start
        while cursor + block <= n and len(runs) < max_blocks:
            if claimed[cursor:cursor + block].any():
                cursor += VS.UNIT
                continue
            if crosses(cursor, block):                  # (1) l'ancre saute au pic
                nxt = next((h for h in hard if h > cursor), None)
                if nxt is None:
                    break
                cursor = nxt
                continue
            cm, ch = VS._slide(M, cursor, block, n), VS._slide(S, cursor, block, n)
            sc = VS.block_score(cm, ch, cursor, n, mute=mute, block=block)
            occ = [p for p in VS._peaks(sc, cursor, n, thr, block, claimed)
                   if not crosses(p, block)]            # (2) reprise rejetée
            if not occ and block == VS.BLOCK:
                occ = [p for p in VS._peaks(sc, cursor, n, thr + VS.ODD_BONUS,
                                            block, claimed, par=1)
                       if not crosses(p, block)]
            if occ:
                runs.append({"b0": cursor, "occ": occ, "block": block})
                for c in [cursor] + occ:
                    claimed[c:min(n, c + block)] = True
            cursor += block
        return runs

    claimed = np.zeros(n, bool)
    runs = _pass_hard(VS.BLOCK, VS.THR8, claimed)
    runs += _pass_hard(VS.FILL, VS.THR4, claimed)

    owner, occid, k = np.full(n, -1), np.full(n, -1), 0
    for i, r in enumerate(runs):
        for c in [r["b0"]] + r["occ"]:
            owner[c:min(n, c + r["block"])] = i
            occid[c:min(n, c + r["block"])] = k
            k += 1

    cut = set(hard)
    out, b = [], 0
    if start > 0:
        out.append({"b0": 0, "b1": start - 1, "label": "intro"})
        b = start
    while b < n:
        z = b
        while (z + 1 < n and owner[z + 1] == owner[b] and occid[z + 1] == occid[b]
               and (z + 1) not in cut):                 # (3) on coupe sur le pic
            z += 1
        out.append({"b0": b, "b1": z, "label": owner[b]})
        b = z + 1

    ren, k = {}, 0
    for s in out:
        if isinstance(s["label"], (int, np.integer)):
            if s["label"] not in ren:
                ren[s["label"]] = LETTERS[k]; k += 1
            s["label"] = ren[s["label"]]
    VS.merge_letters(S, out, V=V)
    return out


def sections_from_peaks(stem: str) -> dict:
    """Tout ce que la page montre : les pics durs, les trois découpages.

    `pics durs` = la recherche de la prod CONTRAINTE par les pics ; `prod` = la
    même fonction avec zéro pic, donc l'algorithme qui tourne aujourd'hui, sur la
    même grille et le même code — la comparaison est à un seul facteur près.
    """
    P, kept, _rest, lines, n, extra = fused_profile(stem)
    m = len(P)
    S_acc = next(d["S"] for d in lines if d["nom"] == "accords")
    raw = [h for h in sorted({int(round(c * n / max(1, m))) for c in kept}) if 0 < h < n]
    # CALÉS SUR LA GRILLE DE 2 MESURES, et ce n'est pas cosmétique : un pic à une
    # mesure près devient, sous une contrainte DURE, une section coupée au mauvais
    # endroit. Mesuré sur les douze : brut 0,704 de médiane, calé 0,727, et Bein'
    # Green passe de 0,718 à 1,000 (ses pics tombaient en 19 et 35 au lieu de 20
    # et 36). Les sections de Louis commencent sur des mesures paires — c'est le
    # même prior que `voice_sections.UNIT`.
    hard = sorted({2 * int(round(h / 2)) for h in raw if 0 < 2 * int(round(h / 2)) < n})

    L = _block_len(S_acc, n)
    triad = extra["triad"]
    ours = voice_with_hard(stem, hard, n, extra, triad=triad)
    prod = voice_with_hard(stem, [], n, extra, triad=triad)
    zones = list(zip([0] + hard, hard + [n]))
    blocks = _tile(zones, 8)
    tiling = [{"b0": a, "b1": b - 1, "label": lab} for (a, b), lab
              in zip(blocks, _letters(S_acc, blocks, n, extra.get("mute")))]
    return {"sections": ours, "prod": prod, "tiling": tiling, "hard": hard,
            "raw": raw, "L": L, "n": n, "grid": extra["grid"],
            "mute": extra.get("mute")}


# ── la comparaison ──────────────────────────────────────────────────────────

def prod_sections(stem: str, use_cache=True) -> list[dict] | None:
    """Ce que la prod écrit aujourd'hui (`HARMONIA_SECTIONS=voice`), mis en
    cache : c'est 20 à 50 s par morceau."""
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    if use_cache and stem in cache:
        return cache[stem]
    from harmonia_min import pipeline as _pl
    out = None
    for kind, model in _pl.analyze_steps(AUDIO / f"{stem}.m4a", title="x",
                                         file_key="x", audio_url=""):
        if kind == "final":
            out = [{"b0": s["barRanges"][0][0], "b1": s["barRanges"][0][1],
                    "label": s["label"]} for s in model["sections"]]
    cache[stem] = out
    CACHE.write_text(json.dumps(cache))
    return out


def score(ours, theirs, n) -> float:
    from section_metric import compare
    try:
        return float(compare(ours, theirs, n)["score"])
    except Exception:
        return float("nan")


def _strip(ax, secs, n, colours, y=0.0, h=1.0):
    for s in secs:
        w = s["b1"] - s["b0"] + 1
        ax.add_patch(plt.Rectangle((s["b0"], y), w, h, facecolor=colours(s["label"]),
                                   edgecolor="#fffdf6", lw=1.4))
        if w >= max(2, n * 0.03):
            ax.text(s["b0"] + w / 2, y + h / 2, str(s["label"]), ha="center",
                    va="center", fontsize=9.5, color="#1c1c1c")


PALETTE = ["#a8c8dc", "#e0c9a6", "#c4d8bf", "#dcc0c8", "#cdc6e0", "#e6d9a8",
           "#bcd8d8", "#e2c4b0"]


def colourmap():
    seen: dict = {}

    def get(lab):
        lab = str(lab)
        if lab == "solo":
            return "#c9a227"
        if lab.lower().startswith("intro") or lab.lower().startswith("outro"):
            return "#ddd6c4"
        if lab not in seen:
            seen[lab] = PALETTE[len(seen) % len(PALETTE)]
        return seen[lab]
    return get


def song_page(stem: str, title: str, with_prod: bool = True) -> str:
    r = sections_from_peaks(stem)
    n, grid = r["n"], r["grid"]
    gt = gt_sections(stem)
    theirs = gt["sections"] if gt else None

    strips = [("prod + pics durs", r["sections"]),
              ("la prod seule", r["prod"]),
              ("pavage depuis les pics", r["tiling"])]
    if theirs:
        strips.append(("toi", theirs))

    fig, axs = plt.subplots(len(strips), 1, figsize=(12.6, 0.62 * len(strips) + 0.5),
                            facecolor="#fffdf6", gridspec_kw={"hspace": 0.45})
    cm = colourmap()
    for ax, (lab, secs) in zip(np.atleast_1d(axs), strips):
        _strip(ax, secs, n, cm)
        for h in r["hard"]:                    # les pics durs, sur les trois
            ax.axvline(h, color="#0d2437", lw=2.0)
        ax.set_xlim(0, n); ax.set_ylim(0, 1)
        ax.set_yticks([]); ax.set_xticks([])
        ax.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=10)
        for s in ax.spines.values():
            s.set_visible(False)
    img = fig2b64(fig)

    btns = "".join(
        f'<button class=blk data-p="[{s["b0"]},{s["b1"]+1}]">{s["label"]}'
        f'<small>mes. {s["b0"]+1}–{s["b1"]+1}</small></button>'
        for s in r["sections"])

    lines = [f"<b>{len(r['hard'])} pics durs</b> aux mesures "
             f"{' · '.join(str(h + 1) for h in r['hard']) or '—'} "
             f"<span class=hint>(calés sur la grille de 2 ; bruts : "
             f"{' · '.join(str(h + 1) for h in r['raw']) or '—'})</span>"]
    if theirs:
        a = score(r["sections"], theirs, n)
        b = score(r["prod"], theirs, n)
        c = score(r["tiling"], theirs, n)
        arrow = "gagne" if a > b + 0.005 else ("perd" if a < b - 0.005 else "égalité")
        lines.append(f"contre ton découpage : <b>prod + pics {a:.3f}</b> · "
                     f"prod seule {b:.3f} · pavage {c:.3f} — la contrainte "
                     f"<b>{arrow}</b> ici")
    verdict = "<br>".join(lines)

    body = f"""<section><h2>{title} <span class=sub>{n} mesures</span></h2>
<div class=plot><img src="data:image/png;base64,{img}">
<div class=cur></div><div class=hit></div></div>
<div class=bar><button class=pp>▶</button><span class=pos>mes. 1 · 0:00</span>
<span class=hint>touche le graphique pour te déplacer</span></div>
<div class=lane><span class=lab>écouter</span>{btns}</div>
<div class=verdict>{verdict}</div></section>
<audio id=au preload=metadata playsinline src="/audio/{stem}.m4a"></audio>
<script>window.GRID={[round(t, 3) for t in grid]};
window.PLOT=[{PLOT_L},{PLOT_R}]; window.U=1;</script>"""
    return page(f"{title} — sections à partir des pics durs", body, back=True,
                lede_html=LEDE, back_href="hard_sections.html")


LEDE = """<div class=lede>Les traits noirs verticaux sont les <b>pics durs</b> :
aucune section ne les traverse, c'est la règle que tu as posée. Quatre bandes, de
haut en bas — <b>prod + pics durs</b> (la recherche qui tourne aujourd'hui, à qui
on interdit d'enjamber un pic, et qui coupe sur chaque pic : c'est le « un A de
seize s'écrit deux A de huit »), <b>la prod seule</b> (le même code, zéro pic —
la comparaison ne change qu'une chose), <b>pavage depuis les pics</b> (ma
première version : blocs réguliers de 8 depuis chaque pic, sans recherche de
reprises), et <b>toi</b>.<br><br>
<b>Ce que ça donne, et c'est un résultat négatif</b> : sur les douze morceaux la
contrainte fait <b>0,727</b> de médiane contre <b>0,781</b> pour la prod seule
(le pavage seul, 0,600). Mais le détail compte plus que la médiane — elle
<b>gagne là où la prod échoue</b> (The Walk 0,430 → 0,539 ; Blue Lights 0,543 →
0,611) et <b>perd là où la prod réussit</b> (Sunny 0,808 → 0,672 ; She Will Be
Loved 0,754 → 0,584). Un pic juste apporte quelque chose ; un pic faux, sous une
règle DURE, casse une solution déjà bonne.<br><br>
<b>Touche une section pour l'écouter</b>, ou le graphique pour te déplacer — le
jugement qui compte est à l'oreille, pas sur ces nombres.</div>"""


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    with_prod = "--prod" in sys.argv
    todo = [(s, t) for s, t in SONGS if s in argv] or (
        [(s, s) for s in argv] if argv else list(SONGS))
    rows = []
    for stem, title in todo:
        if not (AUDIO / f"{stem}.m4a").exists():
            continue
        (OUTDIR / f"hard_{stem}.html").write_text(song_page(stem, title, with_prod))
        rows.append(f'<tr><td><a href="hard_{stem}.html">{title}</a></td></tr>')
        print(f"  ok {title}")
    (OUTDIR / "hard_sections.html").write_text(page(
        "Sections à partir des pics durs",
        "<section><table class=idx>" + "".join(rows) + "</table></section>",
        lede=False))
    print(f"wrote docs/plots/hard_sections.html + {len(rows)} pages")


if __name__ == "__main__":
    main()
