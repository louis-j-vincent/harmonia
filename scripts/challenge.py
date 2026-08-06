"""Quand le chant contredit le pavage, il a le droit de le casser.

    python scripts/challenge.py [<stem> ...]  ->  /reports/challenge.html

Louis, 2026-08-06 :

  « Il faut soit relaxer la contrainte de taille minimale de bloc, soit laisser
    les queues s'adhérer à droite des sections. Dans This Love, à la mesure 36,
    il y a un clair pic du glissement depuis la matrice SSM vocale qui indique
    un nouveau B, et il n'est pas pris en compte parce qu'on a fixé un bloc C
    rigide de 4 mesures — alors que ce C, c'est la queue du A (2 mesures) + le
    début du B (2 autres mesures). Il faudrait un moyen de challenger des
    hypothèses sur des sections quand un pic clair va à l'encontre de la théorie
    actuelle. »

CE QUE FAIT LA PAGE. Jusqu'ici le chant ne pouvait que dire oui : on lui
demandait « la reprise que j'attends ici, tu la vois ? ». Il ne pouvait pas
lever la main ailleurs. Maintenant il le peut.

1. On glisse le bloc chanté de chaque lettre sur TOUTE la chanson et on garde
   tous ses pics FRANCS, pas seulement ceux qu'on attendait.
2. Un pic franc qui tombe là où le pavage ne prévoit aucun début de bloc est une
   CONTESTATION : le chant dit qu'un B commence ici, la théorie dit non.
3. On ne discute pas : on rouvre le découpage à cet endroit. Les frontières
   deviennent l'union des débuts de blocs et des positions contestées, et les
   sections sont les intervalles entre ces frontières — donc un bloc rigide de
   quatre mesures peut se casser en deux, la queue de l'un et le début de
   l'autre, exactement ce que décrit Louis.
4. Les lettres sont refaites sur les nouveaux morceaux, harmonie et chant
   confondus : deux morceaux partagent une lettre s'ils se ressemblent sur l'une
   OU l'autre matrice, sur leur longueur commune.

Aucune taille minimale n'est imposée aux sections issues d'une contestation :
c'est la deuxième branche de sa phrase — relaxer la taille minimale — et c'est
ce qui permet à une queue de deux mesures d'exister par elle-même.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
from scipy.signal import find_peaks      # noqa: E402

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
sys.path.insert(0, str(HERE / "scratchpad"))
from pattern_lanes import load, fig2b64_fixed, COLS, INK, PLOT_L, PLOT_R  # noqa: E402
import harmonia_min.harmonic_sections as HS                               # noqa: E402
import vocal_anchor as VA                                                 # noqa: E402
import blocks8 as B8                                                      # noqa: E402
import blocks_flex as BF                                                  # noqa: E402
import melody_ssm as MS                                                   # noqa: E402
import melody_check as MC                                                 # noqa: E402
import hypo_sizes as HY                                                   # noqa: E402
import vocal_melody as VM                                                 # noqa: E402
import channels as CN                                                     # noqa: E402

CLEAR = 0.62       # un pic « franc » du glissement chanté
NEAR = 1           # à une mesure près, on considère que le pavage le prévoyait
SAME = 0.90        # deux morceaux partagent une lettre à partir d'ici
DEFAULT = B8.DEFAULT


LENGTHS = (2, 4, 6, 8, 10, 12, 16)     # aucune longueur minimale au-dessus de 2
UNIT = 2


def mini_sections(S, M, n, start, same=SAME, lengths=LENGTHS, unit=UNIT):
    """Les mini-sections, SANS a priori de longueur. Louis, 2026-08-06 :

      « Relaxe complètement l'a priori qu'une répétition doit faire 4 mesures
        minimum, car elle loupe aussi le changement de section — subtil
        harmoniquement, on substitue un A pour un C#min — à la mesure 24 de The
        Walk. »
      « Sur Let It Be tu as fait des sections deux fois trop longues, donc ça
        fausse tout. Il faut aussi pouvoir challenger l'hypothèse des longueurs
        des sections. Qu'en est-il de l'argument qui prend la plus petite
        longueur de répétition quand plusieurs se challengent pour une
        mini-section ? »

    Les deux se règlent au même endroit. À chaque ancre on essaie les longueurs
    dans l'ordre CROISSANT, de deux mesures à seize, et **on garde la première
    qui revient quelque part** : la plus petite gagne dès qu'elle tient. C'est
    ce qui empêche Let It Be de sortir des sections deux fois trop longues — un
    bloc de 8 qui revient contient deux blocs de 4 qui reviennent aussi, et sans
    cette règle c'est le grand qui est retenu.

    Une reprise peut être harmonique OU chantée : sur The Walk le changement de
    la mesure 24 substitue un A à un C#m, presque invisible sur l'harmonie, mais
    la voix, elle, change. Il suffit qu'une des deux matrices le voie.
    """
    cells, claimed, cursor = [], np.zeros(n, bool), start
    while cursor < n:
        if claimed[cursor] or (cursor - start) % unit:
            cursor += 1
            continue
        b0, taken = cursor, None
        for L in lengths:
            if b0 + L > n:
                break
            best_c, best_v = None, 0.0
            for c in range(start % unit, n - L + 1, unit):
                if abs(c - b0) < L or claimed[c:c + L].any():
                    continue
                h = HS.diag_match(S, b0, c, L)
                m = float(np.mean([M[b0 + k, c + k] for k in range(L)]))
                v = max(h, m)
                if v > best_v:
                    best_c, best_v = c, v
            if best_v >= same:
                taken = (L, best_c, best_v)
                break                      # la PLUS PETITE qui tient l'emporte
        if taken is None:
            cursor += unit
            continue
        L, _, _ = taken
        occ = []
        for c in range(start % unit, n - L + 1, unit):
            if claimed[c:c + L].any() or any(abs(c - o) < L for o in occ + [b0]):
                continue
            h = HS.diag_match(S, b0, c, L)
            m = float(np.mean([M[b0 + k, c + k] for k in range(L)]))
            if max(h, m) >= same:
                occ.append(c)
        cells.append({"b0": b0, "L": L, "occ": sorted(occ)})
        for c in [b0] + occ:
            claimed[c:min(n, c + L)] = True
        cursor = b0 + unit
    return cells


def mini_to_sections(cells, n, S=None, M=None, same=SAME, vs=None):
    """Les placements deviennent des sections contiguës, puis on RENOMME.

    Mesuré, et c'est la réponse à la question de Louis (« qu'en est-il de
    l'argument qui prend la plus petite longueur de répétition ? ») : prise au
    pied de la lettre, elle rend TOUTES les mini-sections longues de deux
    mesures, parce qu'à deux mesures presque tout revient quelque part dans une
    chanson pop. Elle est **juste pour trouver les frontières** — elle attrape
    enfin le changement de la mesure 24 de The Walk — et **fausse pour nommer**,
    puisque tout finit sous la même lettre.

    On sépare donc les deux rôles. La cellule la plus courte donne les
    FRONTIÈRES ; la lettre, elle, est décidée sur la SECTION ASSEMBLÉE — la
    suite maximale de mesures portant la même cellule — comparée aux autres
    sections de longueur voisine. Deux sections de 24 et de 4 mesures ne peuvent
    plus partager une lettre juste parce que leurs deux premières mesures se
    ressemblent.
    """
    owner = np.full(n, -1)
    for i, e in enumerate(cells):
        for c in [e["b0"]] + e["occ"]:
            owner[c:min(n, c + e["L"])] = i
    out, b = [], 0
    while b < n:
        z = b
        while z + 1 < n and owner[z + 1] == owner[b]:
            z += 1
        out.append({"b0": b, "b1": z, "kind": "bloc" if owner[b] >= 0 else "reste",
                    "letter": "?", "rep": 2})
        b = z + 1
    if S is None:
        return out
    idx = [i for i, s in enumerate(out) if s["kind"] == "bloc"]
    parent = {i: i for i in idx}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for ai in range(len(idx)):
        for bi in range(ai + 1, len(idx)):
            a, b2 = out[idx[ai]], out[idx[bi]]
            La = a["b1"] - a["b0"] + 1
            Lb = b2["b1"] - b2["b0"] + 1
            if min(La, Lb) < 2 or max(La, Lb) > 1.35 * min(La, Lb):
                continue                    # longueurs trop différentes
            L = min(La, Lb)
            # Le seuil est celui du morceau, et une voie qui ne discrimine pas
            # ne vote pas — voir `channels.py`. Ici aussi c'était `max(h, m)`
            # contre 0.90 absolu, alors que sur Let It Be la médiane des paires
            # vaut 0.906 : plus d'une paire sur deux passait.
            if (CN.same_pair(vs, a["b0"], b2["b0"], L) if vs else
                    max(HS.diag_match(S, a["b0"], b2["b0"], L),
                        float(np.mean([M[a["b0"] + k, b2["b0"] + k]
                                       for k in range(L)]))) >= same):
                parent[find(idx[ai])] = find(idx[bi])
    letters = {}
    for i in idx:
        r = find(i)
        if r not in letters:
            letters[r] = chr(ord("A") + len(letters))
        out[i]["letter"] = letters[r]
    return out


def clear_peaks(cur, clear=CLEAR):
    """Les pics francs de la courbe de glissement chanté."""
    if cur.max() <= 0:
        return []
    lo = max(clear, float(np.median(cur[cur > 0])) + .10)
    idx, _ = find_peaks(cur, height=lo, distance=4)
    return [int(i) for i in idx]


def challenges(M, secs, n):
    """Les pics francs que le pavage ne prévoit pas -> [(lettre, mesure, score)]."""
    first, starts = {}, set()
    for s in secs:
        if s["kind"] != "bloc":
            continue
        base = s["letter"].rstrip("′")
        first.setdefault(base, s)
        starts.add(s["b0"])
    out, curves = [], {}
    for base, a in first.items():
        L = a["b1"] - a["b0"] + 1
        cur = MC.slide_on(M, a["b0"], L, n)
        curves[base] = cur
        for p in clear_peaks(cur):
            if any(abs(p - st) <= NEAR for st in starts):
                continue                     # le pavage le prévoyait déjà
            # UNE CONTESTATION DOIT COUPER QUELQUE CHOSE DE GROS. Sans ça, une
            # fois les mini-sections regroupées il ne reste que peu de débuts de
            # bloc, presque chaque pic devient une contestation, et le
            # regroupement qu'on vient de gagner est rehaché en morceaux de deux
            # mesures. Un pic n'a le droit de casser une section que si elle
            # dépasse deux blocs et s'il tombe franchement en son milieu.
            host = next((x for x in secs if x["b0"] <= p <= x["b1"]), None)
            if host is None or host["b1"] - host["b0"] + 1 <= 8:
                continue
            if p - host["b0"] < 4 or host["b1"] - p < 4:
                continue
            out.append({"base": base, "bar": p, "val": float(cur[p])})
    return out, curves


def resplit(S, M, secs, n, chal, same=SAME, vs=None):
    """Rouvre le découpage aux positions contestées, et refait les lettres.

    Aucune taille minimale : une queue de deux mesures a le droit d'exister.
    Deux morceaux partagent une lettre s'ils se ressemblent sur l'harmonie OU
    sur le chant — le chant a contesté, il a voix au chapitre pour nommer.
    """
    cuts = sorted({0, n} | {s["b0"] for s in secs if s["kind"] == "bloc"}
                  | {s["b1"] + 1 for s in secs if s["kind"] == "bloc"}
                  | {c["bar"] for c in chal})
    cuts = [c for c in cuts if 0 <= c <= n]
    spans = [(cuts[i], cuts[i + 1] - 1) for i in range(len(cuts) - 1)
             if cuts[i + 1] > cuts[i]]
    parent = list(range(len(spans)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(spans)):
        for j in range(i + 1, len(spans)):
            L = min(spans[i][1] - spans[i][0], spans[j][1] - spans[j][0]) + 1
            if L < 2:
                continue
            if (CN.same_pair(vs, spans[i][0], spans[j][0], L) if vs else
                    max(HS.diag_match(S, spans[i][0], spans[j][0], L),
                        float(np.mean([M[spans[i][0] + k, spans[j][0] + k]
                                       for k in range(L)]))) >= same):
                parent[find(i)] = find(j)
    letters, out = {}, []
    for i, (a, z) in enumerate(spans):
        r = find(i)
        if r not in letters:
            letters[r] = chr(ord("A") + len(letters))
        out.append({"b0": a, "b1": z, "letter": letters[r], "kind": "bloc",
                    "rep": 2})
    return out


def song(stem):
    S, n, grid = load(stem)
    voc = VA.separate_vocals(HERE / f"docs/audio/{stem}.m4a")
    onset, *_ = B8.sing_onset(voc)
    vstart = MS.voice_start(VA.bar_of(grid, onset), n) or 0
    tt, ff, vv, rr = VM.track_f0(voc)
    notes, _ = VM.melody_notes(tt, ff, vv, rr)
    notes, _ = VM.clean(notes)
    M, mute = MS.melody_bars(notes, grid, n)

    vs = CN.voices(S, M, n, 4, mute)
    cells = mini_sections(S, M, n, vstart)
    secs = mini_to_sections(cells, n, S, M, vs=vs)

    # LES ÉTAPES DE FUSION, montrées avant le résultat. Louis, 2026-08-06 :
    # « je voulais aussi voir le merging avec hypothèses que tu me proposais
    #   avant, juste avant l'explication des sections derrière, pour
    #   comprendre ». Ce sont les trois règles bâties plus tôt dans la journée,
    #   appliquées à la suite sur la chaîne de mini-sections, chacune dessinée
    #   sur sa propre bande : on voit ce que chaque règle colle, et ce qu'elle
    #   laisse.
    chain0 = HY.chain_of([{"L": e["L"], "b0": e["b0"], "occ": e["occ"]}
                          for e in cells], n)
    stages = [("mini-sections", chain0)]
    ch4, _ = HY.merge_two_to_four(chain0)
    stages.append(("2 → 4 mesures", ch4))
    ch5, _ = HY.merge_always(ch4)
    stages.append(("« toujours suivi de »", ch5))
    ch6, _ = HY.merge_runs(ch5)
    stages.append(("successions répétées", ch6))

    # LES SECTIONS FINALES SORTENT DE LA DERNIÈRE FUSION, pas des fragments
    # bruts. Louis, 2026-08-06 : « tu m'as regroupé aucune section, c'était bien
    # mieux tout à l'heure quand tu regroupais ». Je branchais le résultat sur
    # l'assemblage direct des mini-sections, donc sur des bouts de deux mesures,
    # alors que l'étape des successions les regroupe déjà — sur This Love elle
    # passe de 40 passages à 11. Les contestations du chant s'appliquent
    # ensuite, par-dessus le regroupement et non à sa place.
    merged, _ = HY.sections_of(ch6)
    secs = [{**x, "kind": "bloc" if x["letter"] != "·" else "reste", "rep": 2}
            for x in merged]
    chal, curves = challenges(M, secs, n)
    after = resplit(S, M, secs, n, chal, vs=vs)

    heights = ([2.2] + [0.42] * len(cells) + [0.26]
               + [0.48] * len(stages) + [0.26]
               + [0.80] * len(curves) + [0.55, 0.55])
    Hh = sum(heights) + 1.4
    fig, axs = plt.subplots(len(heights), 1, sharex=True, figsize=(12.6, Hh),
                            gridspec_kw={"height_ratios": heights, "hspace": .28})
    fig.subplots_adjust(left=PLOT_L, right=PLOT_R, top=1 - .34 / Hh, bottom=.60 / Hh)
    axs[0].imshow(M, origin="lower", extent=(0, n, 0, n), cmap="Purples",
                  vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    axs[0].set_ylabel("mesure", fontsize=7.4); axs[0].tick_params(labelsize=6.2)
    axs[0].set_title("matrice du CHANT", fontsize=8, color="#7c3aed", loc="left", pad=3)
    # une ligne par MINI-SECTION, avant tout merge — Louis, 2026-08-06 :
    # « la visualisation en plusieurs lignes de chaque mini-section détectée
    #   avant tout merge »
    for i, e in enumerate(cells):
        ax, col = axs[1 + i], COLS[i % len(COLS)]
        for c in [e["b0"]] + e["occ"]:
            ax.add_patch(plt.Rectangle((c, .16), e["L"], .68, facecolor=col,
                                       edgecolor=INK, lw=.7))
        ax.set_xlim(0, n); ax.set_ylim(0, 1); ax.set_yticks([])
        ax.set_ylabel(f"{chr(ord('A')+i)} · {e['L']} mes. · "
                      f"×{1+len(e['occ'])}", fontsize=6, rotation=0,
                      ha="right", va="center", color=col)
        for sp in ax.spines.values():
            sp.set_color("#e5dcc6")
    axs[1 + len(cells)].axis("off")
    off = 2 + len(cells)
    for i, (name, ch) in enumerate(stages):
        secs_i, _ = HY.sections_of(ch)
        B8.strip(axs[off + i], [{**x, "kind": "bloc" if x["letter"] != "·" else "reste",
                                 "rep": 2} for x in secs_i], n,
                 f"{name}\n{len(ch)} passages")
    axs[off + len(stages)].axis("off")
    off += len(stages) + 1

    starts = {s["b0"] for s in secs if s["kind"] == "bloc"}
    for i, (base, cur) in enumerate(curves.items()):
        ax = axs[off + i]
        col = COLS[i % len(COLS)]
        ax.fill_between(np.arange(n) + .5, cur, color=col, alpha=.28, lw=0)
        ax.plot(np.arange(n) + .5, cur, color=col, lw=1.1)
        for st in starts:
            ax.axvline(st + .5, color="#c9c1ab", lw=.9)
        for c in chal:
            if c["base"] != base:
                continue
            ax.axvline(c["bar"] + .5, color="#b3261e", lw=1.8)
            ax.plot([c["bar"] + .5], [c["val"]], "o", ms=5, color="#b3261e")
            ax.text(c["bar"] + 1, c["val"], f"  mes. {c['bar']+1}", fontsize=6.4,
                    color="#b3261e", va="center", fontweight="bold")
        ax.set_ylim(0, 1.12); ax.set_yticks([])
        ax.set_ylabel(f"{base} glissé", fontsize=6.6, rotation=0, ha="right",
                      va="center", color=col)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
    B8.strip(axs[-2], secs, n, "mini-sections\nassemblées")
    B8.strip(axs[-1], after, n, f"APRÈS\n{len(chal)} contestation(s)")
    axs[-1].set_xticks(range(0, n + 1, 4)); axs[-1].tick_params(labelsize=6.2)
    axs[-1].set_xlabel("mesure", fontsize=8)
    img = fig2b64_fixed(fig)

    fmt = lambda x: " ".join(
        ("intro" if s["kind"] == "intro" else "?" if s["kind"] == "reste"
         else s["letter"]) + f"[{s['b0']+1}-{s['b1']+1}]" for s in x)
    rows = "".join(
        f"<tr><td><b>{c['base']}</b></td><td>mesure {c['bar']+1}</td>"
        f"<td>{c['val']:.2f}</td></tr>" for c in sorted(chal, key=lambda c: c["bar"]))
    gridjs = "[" + ",".join(f"{x:.3f}" for x in grid) + "]"
    btns = "".join(
        f"<button class=blk data-p='[{max(0,c['bar']-2)},{min(n,c['bar']+6)}]'>"
        f"{c['base']}?<small>{c['bar']+1}</small></button>"
        for c in sorted(chal, key=lambda c: c["bar"])[:8])
    return f"""<section data-grid='{gridjs}' data-audio="/audio/{stem}.m4a">
<h2>{stem.replace('_',' ').title()}<span class=sub>{n} mesures ·
départ mes. {vstart+1} · {len(cells)} mini-sections ({', '.join(str(e['L']) for e in cells)} mes.) ·
<b>{len(chal)} contestation(s) du chant</b></span></h2>
<div class=plot><img src="data:image/png;base64,{img}">
<div class=cur></div><div class=hit></div></div>
<div class=bar><button class=pp>▶</button><span class=pos>mes. 1 · 0:00</span>
<span class=hint>écoute les endroits contestés</span>{btns}</div>
<table><tr><th>lettre</th><th>le chant dit qu'elle commence ici</th>
<th>score</th></tr>{rows or '<tr><td colspan=3>aucune contestation</td></tr>'}</table>
<p class=verdict><b>avant</b> : {fmt(secs)}<br><b>après</b> : {fmt(after)}</p>
</section>"""


def main():
    stems = sys.argv[1:] or DEFAULT
    body = ""
    for st in stems:
        if not (HERE / f"docs/audio/{st}.m4a").exists():
            print(f"  !! {st} introuvable")
            continue
        try:
            body += song(st)
            print(f"  ok {st}")
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"  !! {st} — {type(exc).__name__}: {exc}")
    out = HERE / "harmonia_min/state/reports/challenge.html"
    out.write_text(f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Quand le chant conteste</title><style>
*{{box-sizing:border-box;-webkit-tap-highlight-color:transparent}}
body{{margin:0;background:#e7e0d0;font:16px/1.55 -apple-system,system-ui,sans-serif;color:{INK}}}
.wrap{{max-width:1150px;margin:0 auto;padding:20px 12px calc(40px + env(safe-area-inset-bottom))}}
h1{{font:italic 600 24px Georgia,serif;margin:0 0 3px}}
.lede{{color:#6f6858;font-size:13.5px;line-height:1.55;margin-bottom:18px}}
.lede b{{color:{INK}}} ol{{margin:8px 0 0;padding-left:20px}} ol li{{margin-bottom:5px}}
section{{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;padding:14px 13px;margin-bottom:16px}}
h2{{font:700 18px system-ui;margin:0 0 6px;color:#8a2b2b}}
.sub{{font:500 12px system-ui;color:#8a8371;margin-left:8px}}
img{{width:100%;border-radius:8px;display:block}}
table{{border-collapse:collapse;font-size:12.5px;width:auto;margin-top:8px}}
th,td{{border:1px solid #e5dcc6;padding:4px 9px;text-align:left}}
th{{background:#f7f3e9;font-size:11px}}
.verdict{{font:500 11.5px ui-monospace,monospace;background:#f7f3e9;
  border-radius:8px;padding:9px 11px;margin:10px 0 0;line-height:1.8}}
.plot{{position:relative;margin-bottom:8px}} .plot img{{margin:0}}
.cur{{position:absolute;top:0;bottom:0;width:2px;background:#111;opacity:.8;
  display:none;pointer-events:none;box-shadow:0 0 0 1px rgba(255,255,255,.55)}}
.hit{{position:absolute;top:0;bottom:0;cursor:crosshair}}
.bar{{display:flex;flex-wrap:wrap;align-items:center;gap:4px;margin:0 0 6px}}
.pp{{width:36px;height:36px;border-radius:50%;border:1px solid #d8cfb4;
  background:#f7f3e9;font-size:13px;cursor:pointer;flex:none}}
.pos{{font:600 12px ui-monospace,monospace;min-width:88px}}
.hint{{font:500 11px system-ui;color:#a89f8c;margin-right:6px}}
button.blk{{border:1px solid #d8cfb4;background:#f7f3e9;border-radius:6px;
  padding:3px 7px;cursor:pointer;font:700 11.5px ui-monospace,monospace}}
button.blk small{{display:block;font:500 8.5px ui-monospace,monospace;color:#a89f8c}}
button.blk.on{{color:#fff !important;background:#8a2b2b !important}}
</style></head><body><div class=wrap>
<h1>Quand le chant conteste</h1>
<div class=lede>Jusqu'ici le chant ne pouvait que dire oui : on lui demandait
« la reprise que j'attends ici, tu la vois ? ». Il ne pouvait pas lever la main
ailleurs. Maintenant il le peut.
<ol>
<li>On glisse le bloc chanté de chaque lettre sur toute la chanson et on garde
<b>tous ses pics francs</b>, pas seulement ceux qu'on attendait.</li>
<li>Un pic franc là où le pavage ne prévoit aucun début de bloc est une
<b>contestation</b> : le chant dit qu'un B commence ici, la théorie dit non.
Trait rouge sur les courbes.</li>
<li>On ne discute pas, <b>on rouvre le découpage à cet endroit</b>. Les
frontières deviennent l'union des débuts de blocs et des positions contestées —
donc un bloc rigide de quatre mesures peut se casser en deux, la queue de l'un et
le début de l'autre.</li>
<li>Les lettres sont refaites sur les nouveaux morceaux, <b>harmonie et chant
confondus</b> : deux morceaux partagent une lettre s'ils se ressemblent sur l'une
ou l'autre matrice.</li>
</ol>
<br>Aucune taille minimale n'est imposée aux sections nées d'une contestation :
c'est ce qui permet à une queue de deux mesures d'exister par elle-même.</div>
{body}</div>
<audio id=au preload=metadata playsinline></audio>
<script>
const au=document.getElementById("au");
const L0=%L0%, W=%W%;
let stopAt=null,onBtn=null,live=null,raf=null;
const fmt=s=>Math.floor(s/60)+":"+String(Math.floor(s%60)).padStart(2,"0");
function clr(){{ if(onBtn){{onBtn.classList.remove("on");onBtn=null;}} }}
function draw(){{
  if(!live) return;
  const G=live.G, n=G.length-1;
  let t=au.currentTime, f;
  if(t<=G[0]) f=0; else if(t>=G[n]) f=n; else {{
    let lo=0,hi=n; while(hi-lo>1){{const m=(lo+hi)>>1; G[m]<=t?lo=m:hi=m;}}
    f=lo+(t-G[lo])/(G[lo+1]-G[lo]); }}
  live.cur.style.display="block";
  live.cur.style.left="calc("+((L0+W*f/n)*100)+"% - 1px)";
  live.pos.textContent="mes. "+(Math.floor(f)+1)+" · "+fmt(au.currentTime);
}}
function tick(){{ draw();
  if(stopAt!=null&&au.currentTime>=stopAt){{au.pause();stopAt=null;clr();}}
  if(!au.paused) raf=requestAnimationFrame(tick); }}
au.addEventListener("play",()=>{{ if(live) live.pp.textContent="❚❚"; tick(); }});
au.addEventListener("pause",()=>{{ if(live) live.pp.textContent="▶";
  cancelAnimationFrame(raf); draw(); }});
function go(sec,t0,t1,btn){{
  if(live && live.sec!==sec){{ live.pp.textContent="▶"; live.cur.style.display="none"; }}
  live=sec._p; clr(); stopAt=t1;
  if(btn){{onBtn=btn;btn.classList.add("on");}}
  if(au.getAttribute("src")!==sec.dataset.audio){{
    au.setAttribute("src",sec.dataset.audio);au.load();}}
  const seek=()=>{{try{{au.currentTime=t0;}}catch(e){{}} draw();}};
  if(au.readyState>=1) seek(); else au.addEventListener("loadedmetadata",seek,{{once:true}});
  au.play().catch(()=>clr());
}}
document.querySelectorAll("section[data-grid]").forEach(sec=>{{
  const G=JSON.parse(sec.dataset.grid), n=G.length-1;
  const hit=sec.querySelector(".hit");
  sec._p={{G:G,pos:sec.querySelector(".pos"),cur:sec.querySelector(".cur"),
          pp:sec.querySelector(".pp"),sec:sec}};
  hit.style.left=(L0*100)+"%"; hit.style.width=(W*100)+"%";
  hit.onclick=e=>{{ const r=hit.getBoundingClientRect();
    const f=n*(e.clientX-r.left)/r.width;
    const i=Math.max(0,Math.min(n-1,Math.floor(f)));
    go(sec, G[i]+(f-i)*(G[i+1]-G[i]), null, null); }};
  sec.querySelector(".pp").onclick=()=>{{
    if(au.paused||live!==sec._p) go(sec,G[0],null,null); else au.pause(); }};
  sec.querySelectorAll("[data-p]").forEach(b=>{{
    const d=JSON.parse(b.dataset.p);
    b.onclick=()=>go(sec,G[d[0]],G[Math.min(n,d[1])],b); }});
}});
</script></body></html>""".replace("%L0%", str(PLOT_L)).replace("%W%", str(round(PLOT_R - PLOT_L, 6))))
    print(f"wrote {out.relative_to(HERE)} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
