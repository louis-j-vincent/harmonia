"""Les pics de la voix, bloc par bloc — la page pour DÉBUGGER une détection.

    python scripts/voice_peaks_report.py [stem ...]   ->  /reports/voice_peaks_<stem>.html

Louis, 2026-08-08 : « j'aimerais bien que tu me montres le plot des pics voix de
la section A sur She Will Be Loved pour que je puisse debug, car on loupe pas mal
de A ce qui est bizarre ».

CE QU'ON MONTRE, et pourquoi ces courbes-là. `voice_sections._pass` pose une
ancre, glisse le bloc sur toute la chanson et garde les pics : trois filtres
successifs décident du sort de chaque pic, et jusqu'ici seuls les SURVIVANTS
étaient visibles. Un A manquant pouvait donc l'être pour quatre raisons très
différentes — trop faible, hors grille, déjà réclamé, ou masqué par un pic plus
fort — indiscernables sur le résultat. On dessine donc les REFUSÉS avec leur
motif de refus, en pâle, plutôt que de les cacher : c'est tout l'intérêt.

  * la courbe de glissement sur la voie CHANT (`_slide(M, b0, L, n)`) ;
  * la même sur la voie HARMONIE (`_slide(S, b0, L, n)`) ;
  * le score réellement utilisé (`block_score`), qui mélange les deux et pondère
    le chant par ce qu'il ENTEND — sur une mesure muette il s'abstient ;
  * le seuil, les pics retenus, les pics refusés et leur motif ;
  * la bande de vérité de Louis et la nôtre, sur le même axe ;
  * les mesures muettes en fond, parce qu'un A « loupé » peut n'être qu'un
    passage où personne ne chante.

Chaque pic est un BOUTON qui joue exactement ses mesures : la question « est-ce
que la mesure 33 est vraiment un A ? » se tranche à l'oreille, pas au chiffre.

CONVENTION DE MESURES : celle de `pattern_lanes` — la mesure i occupe [i, i+1],
une frontière se pose en `edge(i) = i`, une valeur qui appartient à la mesure i
en `mid(i) = i + 0.5`. Les marges du tracé sont FIXES (`PLOT_L`/`PLOT_R` +
`fig2b64_fixed`, jamais `bbox_inches="tight"`) sinon le curseur de lecture, qui
est un div posé sur l'image, se décale.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402
import numpy as np                                                   # noqa: E402
from scipy.signal import find_peaks                                  # noqa: E402

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))

from pattern_lanes import (COLS, INK, PLOT_L, PLOT_R,                # noqa: E402
                           edge, fig2b64_fixed, mid)
import section_bench as SB                                           # noqa: E402
import section_metric as SM                                          # noqa: E402
from harmonia_min import voice_sections as VS                        # noqa: E402

DEFAULT = "maroon_5_she_will_be_loved_official_music_video"

C_VOICE = COLS[2]      # bleu   — la voie CHANT
C_HARM = COLS[1]       # vert   — la voie HARMONIE
C_THR = "#8a8371"      # le seuil : un seuil ABSOLU, couleur partagée (THR_COLS)
C_KEPT = "#b3261e"     # ce qui survit

# Un motif de refus = une FORME, pas seulement une couleur : la raison doit
# rester lisible en noir et blanc et pour un daltonien.
WHY = {
    "retenu":   ("o", C_KEPT, "retenu"),
    "impaire":  ("X", "#7c3aed", "refusé : mesure impaire (hors grille de 2)"),
    "pris":     ("s", "#c58a2e", "refusé : mesures déjà réclamées"),
    "seuil":    ("v", "#8a8371", "refusé : sous le seuil"),
    "proche":   (".", "#b9b09a", "ignoré : trop près de l'ancre"),
    "masque":   ("D", "#0f766e", "refusé : chevauche un pic plus fort"),
}
NEAR = 0.12       # un pic à plus de ça sous le seuil n'a jamais été en jeu


def shown(c, blk, hisbars):
    """Ce qu'on dessine et ce qu'on tait.

    Une chanson de cent mesures a ~50 maxima locaux par bloc et les montrer tous
    noie le seul fait qui compte — quels pics ÉTAIENT en jeu. On garde donc ce
    qui a franchi le seuil (le refus y est alors une décision, pas une évidence),
    ce qui en approche à `NEAR`, et TOUJOURS les mesures que Louis a annotées,
    même très basses : « on ne l'entend pas du tout » est aussi une réponse.
    """
    if c["why"] == "proche":
        return False
    return (c["p"] in hisbars or c["why"] != "seuil"
            or c["sc"] >= blk["thr"] - NEAR)


# ── la détection, instrumentée ──────────────────────────────────────────────

def classify(sc, b0, n, thr, block, claimed, occ):
    """Chaque maximum local de la courbe, et POURQUOI il a vécu ou non.

    Miroir exact de `voice_sections._peaks`, FILTRES DANS LE MÊME ORDRE — et
    l'ordre compte pour la lecture : là-bas le seuil est appliqué le premier
    (`find_peaks(height=thr)`), donc un pic faible ET sur une mesure impaire est
    mort du seuil, pas de la grille. L'annoncer « hors grille » ferait accuser
    la grille de refus qu'elle n'a pas prononcés. `find_peaks` sans `height` ici
    pour voir aussi ce qui tombe sous le seuil — c'est la moitié de la question.
    """
    idx, _ = find_peaks(sc, distance=VS.UNIT)
    out = []
    for p in (int(x) for x in idx):
        if p + block > n:
            continue
        if sc[p] < thr:
            why = "seuil"
        elif abs(p - b0) < block:
            why = "proche"
        elif (p - b0) % VS.UNIT:
            why = "impaire"
        elif claimed[p:p + block].any():
            why = "pris"
        elif p in occ:
            why = "retenu"
        else:
            why = "masque"
        out.append({"p": p, "sc": float(sc[p]), "why": why})
    return out


def detect(F):
    """Rejoue `voice_sections.detect_sections` en gardant tout ce qu'elle jette."""
    n, S, M, mute = F["n"], F["S"], F["M"], F["mute"]
    start = VS.sung_start(F["notes"], F["grid"], F["mute"])
    claimed = np.zeros(n, bool)
    runs, blocks = [], []
    for block, thr in ((VS.BLOCK, VS.THR8), (VS.FILL, VS.THR4)):
        cursor = start
        while cursor + block <= n and len(runs) < 20:
            if claimed[cursor:cursor + block].any():
                cursor += VS.UNIT
                continue
            cm = VS._slide(M, cursor, block, n)
            ch = VS._slide(S, cursor, block, n)
            sc = VS.block_score(cm, ch, cursor, n, mute=mute, block=block)
            occ = VS._peaks(sc, cursor, n, thr, block, claimed)
            blocks.append({"b0": cursor, "block": block, "thr": thr,
                           "cm": cm, "ch": ch, "sc": sc, "occ": occ,
                           "cand": classify(sc, cursor, n, thr, block,
                                            claimed, set(occ)),
                           "claimed": claimed.copy(), "used": bool(occ)})
            if occ:
                runs.append({"b0": cursor, "occ": occ, "block": block})
                for c in [cursor] + occ:
                    claimed[c:min(n, c + block)] = True
            cursor += block
    ours = VS.merge_letters(F["S"], SB.assemble(runs, n, start), V=F["V"])
    return start, blocks, ours


# ── la figure ───────────────────────────────────────────────────────────────

def strip(ax, secs, n, label, cols):
    for s in secs:
        lab = str(s["label"])
        w = s["b1"] - s["b0"] + 1
        if lab in ("intro", "outro"):
            ax.add_patch(plt.Rectangle((edge(s["b0"]), 0), w, 1, facecolor="#cfc7b2",
                                       hatch="///", edgecolor="#a89f8c", lw=.6))
        else:
            cols.setdefault(lab, COLS[len(cols) % len(COLS)])
            ax.add_patch(plt.Rectangle((edge(s["b0"]), 0), w, 1,
                                       facecolor=cols[lab], edgecolor="#fff", lw=1.1))
        ax.text(edge(s["b0"]) + w / 2, .5, lab[:5], ha="center", va="center",
                fontsize=6.2, fontweight="bold",
                color="#4a4438" if lab in ("intro", "outro") else "#fff")
    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel(label, fontsize=6.6, rotation=0, ha="right", va="center", color=INK)


def figure(F, blk, ours, his, hisbars, n):
    b0, L, thr = blk["b0"], blk["block"], blk["thr"]
    mute = F["mute"]
    heights = [2.5, 1.15, 1.15, 0.5, 0.5]
    H = sum(heights) + 1.55
    fig, axs = plt.subplots(len(heights), 1, sharex=True, figsize=(12.6, H),
                            gridspec_kw={"height_ratios": heights, "hspace": .18})
    fig.subplots_adjust(left=PLOT_L, right=PLOT_R, top=1 - .78 / H, bottom=.62 / H)

    # les mesures muettes, en fond : un A « loupé » peut n'être qu'un silence
    for i in range(n):
        if mute[i]:
            for ax in axs[:3]:
                ax.axvspan(edge(i), edge(i + 1), color="#d9d2be", alpha=.55,
                           lw=0, zorder=0)

    x = mid(np.arange(n))

    # ── le score réellement utilisé ─────────────────────────────────────────
    ax = axs[0]
    ax.axvspan(edge(b0), edge(b0 + L), color=C_KEPT, alpha=.09, lw=0, zorder=1)
    ax.plot(x, blk["sc"], color=INK, lw=1.6, zorder=4,
            label="score utilisé (block_score)")
    ax.axhline(thr, color=C_THR, lw=1.3, ls=(0, (5, 3)), zorder=3,
               label=f"seuil {thr:.2f}")
    seen = set()
    for c in blk["cand"]:
        if not shown(c, blk, hisbars):
            continue
        m, col, lab = WHY[c["why"]]
        kept = c["why"] == "retenu"
        ax.plot(mid(c["p"]), c["sc"], m, ms=9 if kept else 7, zorder=6,
                color=col, alpha=1.0 if kept else .5,
                mec="#fffdf6" if kept else col, mew=1.2 if kept else .9,
                label=lab if lab not in seen else None)
        seen.add(lab)
        if kept:
            ax.add_patch(plt.Rectangle((edge(c["p"]), .02), L, .055,
                                       transform=ax.get_xaxis_transform(),
                                       facecolor=col, alpha=.85, lw=0, zorder=5))
            ax.text(edge(c["p"]) + L / 2, .085, f"{c['sc']:.2f}",
                    transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=5.8, color=col, fontweight="bold", zorder=6)
    # ses occurrences À LUI de la lettre qui couvre l'ancre : le vrai sujet
    for b in hisbars:
        ax.plot(mid(b), blk["sc"][b], "^", ms=11, mfc="none", mec="#111", mew=1.3,
                zorder=7, label="départ d'une occurrence chez Louis"
                if "louis" not in seen else None)
        seen.add("louis")
        ax.text(mid(b), blk["sc"][b] + .045, f"{blk['sc'][b]:.2f}", ha="center",
                va="bottom", fontsize=6.2, color="#111", zorder=7)
    ax.set_ylim(-0.02, max(1.08, float(blk["sc"].max()) * 1.14))
    ax.set_ylabel("score\nutilisé", fontsize=6.8, rotation=0, ha="right", va="center")
    ax.legend(fontsize=6.1, ncol=4, loc="upper center", frameon=False,
              bbox_to_anchor=(.5, 1.28), handletextpad=.3, columnspacing=1.1)
    ax.grid(axis="y", color="#e2dac6", lw=.6, zorder=0)

    # ── les deux voies brutes ───────────────────────────────────────────────
    for ax, cur, col, name in ((axs[1], blk["cm"], C_VOICE, "voie CHANT\n_slide(M)"),
                               (axs[2], blk["ch"], C_HARM, "voie HARMONIE\n_slide(S)")):
        ax.axvspan(edge(b0), edge(b0 + L), color=C_KEPT, alpha=.09, lw=0, zorder=1)
        ax.plot(x, cur, color=col, lw=1.4, zorder=3)
        ax.fill_between(x, 0, cur, color=col, alpha=.13, zorder=2)
        for b in hisbars:
            ax.plot(mid(b), cur[b], "^", ms=8, mfc="none", mec="#111", mew=1.1, zorder=5)
        ax.set_ylim(0, 1.03)
        ax.set_yticks([0, .5, 1])
        ax.tick_params(labelsize=5.8)
        ax.set_ylabel(name, fontsize=6.4, rotation=0, ha="right", va="center", color=col)
        ax.grid(axis="y", color="#e2dac6", lw=.6, zorder=0)

    cols = {}
    strip(axs[3], his, n, "TOI", cols)
    strip(axs[4], ours, n, "NOUS", cols)
    axs[4].set_xticks(range(0, n + 1, 4))
    axs[4].tick_params(labelsize=6.2)
    axs[4].set_xlabel("mesure (l'axe est en indices ; la mesure i s'affiche i+1 "
                      "dans le lecteur)", fontsize=7.4)
    for ax in axs:
        ax.set_xlim(0, n)
        for sp in ax.spines.values():
            sp.set_color("#ddd5c0")
    return fig2b64_fixed(fig)


# ── la page ─────────────────────────────────────────────────────────────────

def fmt(secs):
    return " ".join(f"{s['label']}[{s['b0'] + 1}-{s['b1'] + 1}]" for s in secs)


def card(F, blk, ours, his, n, i):
    b0, L = blk["b0"], blk["block"]
    lab = next((str(s["label"]) for s in his if s["b0"] <= b0 <= s["b1"]), None)
    hisbars = [s["b0"] for s in his if str(s["label"]) == lab] if lab else []
    img = figure(F, blk, ours, his, hisbars, n)
    btn = [f"<button class='blk kept' style='border-color:{INK}' "
           f"data-p='[{b0},{min(n, b0 + L)}]' title=\"l'ancre\">"
           f"mes. {b0 + 1}<small>ancre</small></button>"]
    for c in sorted(blk["cand"], key=lambda c: c["p"]):
        if c["p"] == b0 or not shown(c, blk, hisbars):
            continue
        _, col, why = WHY[c["why"]]
        cls = "kept" if c["why"] == "retenu" else "rej"
        btn.append(f"<button class='blk {cls}' style='border-color:{col}' "
                   f"data-p='[{c['p']},{min(n, c['p'] + L)}]' title='{why}'>"
                   f"mes. {c['p'] + 1}<small>{c['sc']:.2f} · "
                   f"{why.split(':')[-1].strip()}</small></button>")
    lost = [c for c in blk["cand"] if c["p"] in hisbars and c["why"] != "retenu"
            and c["p"] != b0]
    note = ""
    if lost:
        note = ("<p class=lost><b>Ses occurrences perdues ici</b> : " + " · ".join(
            f"mes. {c['p'] + 1} à {c['sc']:.3f} — "
            + (WHY[c["why"]][2].split(":")[-1].strip() if c["why"] != "seuil"
               else "sous le seuil")
            + (" <b>alors qu'elle passait le seuil</b>" if c["sc"] >= blk["thr"]
               else "") for c in lost) + "</p>")
    head = (f"ancre mesure <b>{b0 + 1}</b> · bloc de <b>{L}</b> mesures · seuil "
            f"{blk['thr']:.2f}" + (f" · chez Louis c'est un <b>{lab}</b>" if lab else "")
            + (" · <i>aucun pic retenu, l'ancre est abandonnée</i>" if not blk["used"] else ""))
    return f"""<section data-grid='%GRID%' data-audio="/audio/%STEM%.m4a">
<h2>Bloc {i + 1}<span class=sub>{head}</span></h2>
<div class=plot><img src="data:image/png;base64,{img}">
<div class=cur></div><div class=hit></div></div>
{note}
<div class=bar><button class=pp>▶</button><span class=pos>mes. 1 · 0:00</span>
<span class=hint>écoute un pic</span>{''.join(btn)}</div>
</section>"""


def page(stem):
    F = SB.features(stem)
    n = F["n"]
    T = SB.truth(stem).get(stem)
    start, blocks, ours = detect(F)
    his = T["sections"] if T else []
    m = SM.compare(ours, his, min(n, T["n"])) if T else None

    body = "".join(card(F, b, ours, his, n, i) for i, b in enumerate(blocks))
    grid = "[" + ",".join(f"{x:.3f}" for x in F["grid"]) + "]"
    body = body.replace("%GRID%", grid).replace("%STEM%", stem)

    tot = ""
    if m:
        tot = (f"<p class=tot>SCORE <b>{m['score']:.3f}</b> &nbsp;·&nbsp; découpage "
               f"<b>{m['spans']:.2f}</b> · noms <b>{m['letters']:.2f}</b> "
               f"&nbsp;·&nbsp; <b>{m['n_pred']}</b> sections contre "
               f"<b>{m['n_ref']}</b> chez lui &nbsp;·&nbsp; intro {start} mesures</p>"
               f"<p class=verdict><b>toi</b> &nbsp;: {fmt(his)}<br>"
               f"<b>nous</b> : {fmt(ours)}</p>")

    out = HERE / f"harmonia_min/state/reports/voice_peaks_{short(stem)}.html"
    out.write_text(TPL.replace("%TITLE%", stem.replace("_", " ").title())
                   .replace("%TOT%", tot).replace("%BODY%", body)
                   .replace("%INK%", INK).replace("%L0%", str(PLOT_L))
                   .replace("%W%", str(round(PLOT_R - PLOT_L, 6))))
    print(f"wrote {out.relative_to(HERE)} ({out.stat().st_size // 1024} KB)")
    return out


def short(stem):
    """Un nom de fichier court et stable, et celui que Louis attend pour SWBL."""
    return {DEFAULT: "she_will_be_loved"}.get(stem, stem)


TPL = """<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Les pics de la voix — %TITLE%</title><style>
*{box-sizing:border-box;-webkit-tap-highlight-color:transparent}
body{margin:0;background:#e7e0d0;font:16px/1.55 -apple-system,system-ui,sans-serif;color:%INK%}
.wrap{max-width:1150px;margin:0 auto;padding:20px 12px calc(40px + env(safe-area-inset-bottom))}
h1{font:italic 600 24px Georgia,serif;margin:0 0 3px}
.lede{color:#6f6858;font-size:13.5px;line-height:1.6;margin-bottom:10px}
.lede b{color:%INK%}
.key{background:#fff4e2;border:1px solid #e0cfa8;border-radius:10px;padding:10px 12px;
  font-size:13.5px;line-height:1.6;margin:0 0 14px}
.key b{color:#8a2b2b}
.tot{background:#fffdf6;border:1px solid #e5dcc6;border-radius:10px;padding:9px 12px;
  font:600 13.5px system-ui;margin:0 0 10px}
section{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;
  padding:14px 13px;margin-bottom:16px}
h2{font:700 17px system-ui;margin:0 0 8px;color:#8a2b2b}
.sub{display:block;font:500 12px system-ui;color:#6f6858;margin-top:2px}
img{width:100%;border-radius:8px;display:block}
.plot{position:relative;margin-bottom:8px}.plot img{margin:0}
.cur{position:absolute;top:0;bottom:0;width:2px;background:#111;opacity:.8;
  display:none;pointer-events:none;box-shadow:0 0 0 1px rgba(255,255,255,.55)}
.hit{position:absolute;top:0;bottom:0;cursor:crosshair}
.bar{display:flex;flex-wrap:wrap;align-items:center;gap:4px;margin:0}
.pp{width:36px;height:36px;border-radius:50%;border:1px solid #d8cfb4;
  background:#f7f3e9;font-size:13px;cursor:pointer;flex:none}
.pos{font:600 12px ui-monospace,monospace;min-width:88px}
.hint{font:500 11px system-ui;color:#a89f8c;margin-right:6px}
button.blk{border:1px solid #d8cfb4;background:#f7f3e9;border-radius:6px;
  padding:3px 7px;cursor:pointer;font:700 11.5px ui-monospace,monospace;
  border-left-width:4px}
button.blk small{display:block;font:500 8.5px ui-monospace,monospace;color:#a89f8c}
button.blk.rej{opacity:.62}
button.blk.on{color:#fff !important;background:#8a2b2b !important}
button.blk.on small{color:#f2dcdc}
.lost{background:#f7dede;border-radius:8px;padding:8px 11px;margin:0 0 8px;
  font:500 12.5px system-ui;line-height:1.6}
.verdict{font:500 11.5px ui-monospace,monospace;background:#f7f3e9;
  border-radius:8px;padding:9px 11px;margin:0 0 16px;line-height:1.9}
</style></head><body><div class=wrap>
<h1>Les pics de la voix — %TITLE%</h1>
<div class=lede>Une carte par <b>ancre</b>. Le détecteur pose un bloc, le glisse
sur toute la chanson, et garde ses pics. Trois filtres décident, dans cet ordre :
les mesures <b>déjà réclamées</b>, la <b>grille de 2</b> (un pic ne compte que
s'il tombe à un nombre PAIR de mesures de l'ancre), puis le <b>seuil</b>. Les
refusés sont dessinés en pâle avec leur motif — c'est tout l'intérêt de la page.
<br><br>Le triangle noir ▵ marque le début de chacune des occurrences que
<b>Louis</b> a annotées pour la lettre qui couvre l'ancre : s'il tombe sur un pic
qu'on a refusé, on tient l'explication. Le fond gris est une mesure
<b>muette</b> — personne n'y chante, donc la voie chant s'y abstient.
<br><br>Chaque bouton joue exactement les mesures du pic et s'arrête.</div>
%TOT%%BODY%</div>
<audio id=au preload=metadata playsinline></audio>
<script>
const au=document.getElementById("au");
const L0=%L0%, W=%W%;
let stopAt=null,onBtn=null,live=null,raf=null;
const fmt=s=>Math.floor(s/60)+":"+String(Math.floor(s%60)).padStart(2,"0");
function clr(){ if(onBtn){onBtn.classList.remove("on");onBtn=null;} }
function draw(){
  if(!live) return;
  const G=live.G, n=G.length-1;
  let t=au.currentTime, f;
  if(t<=G[0]) f=0; else if(t>=G[n]) f=n; else {
    let lo=0,hi=n; while(hi-lo>1){const m=(lo+hi)>>1; G[m]<=t?lo=m:hi=m;}
    f=lo+(t-G[lo])/(G[lo+1]-G[lo]); }
  live.cur.style.display="block";
  live.cur.style.left="calc("+((L0+W*f/n)*100)+"% - 1px)";
  live.pos.textContent="mes. "+(Math.floor(f)+1)+" · "+fmt(au.currentTime);
}
function tick(){ draw();
  if(stopAt!=null&&au.currentTime>=stopAt){au.pause();stopAt=null;clr();}
  if(!au.paused) raf=requestAnimationFrame(tick); }
au.addEventListener("play",()=>{ if(live) live.pp.textContent="❚❚"; tick(); });
au.addEventListener("pause",()=>{ if(live) live.pp.textContent="▶";
  cancelAnimationFrame(raf); draw(); });
function go(sec,t0,t1,btn){
  if(live && live.sec!==sec){ live.pp.textContent="▶"; live.cur.style.display="none"; }
  live=sec._p; clr(); stopAt=t1;
  if(btn){onBtn=btn;btn.classList.add("on");}
  if(au.getAttribute("src")!==sec.dataset.audio){
    au.setAttribute("src",sec.dataset.audio);au.load();}
  const seek=()=>{try{au.currentTime=t0;}catch(e){} draw();};
  if(au.readyState>=1) seek(); else au.addEventListener("loadedmetadata",seek,{once:true});
  au.play().catch(()=>clr());
}
document.querySelectorAll("section[data-grid]").forEach(sec=>{
  const G=JSON.parse(sec.dataset.grid), n=G.length-1;
  const hit=sec.querySelector(".hit");
  sec._p={G:G,pos:sec.querySelector(".pos"),cur:sec.querySelector(".cur"),
          pp:sec.querySelector(".pp"),sec:sec};
  hit.style.left=(L0*100)+"%"; hit.style.width=(W*100)+"%";
  hit.onclick=e=>{ const r=hit.getBoundingClientRect();
    const f=n*(e.clientX-r.left)/r.width;
    const i=Math.max(0,Math.min(n-1,Math.floor(f)));
    go(sec, G[i]+(f-i)*(G[i+1]-G[i]), null, null); };
  sec.querySelector(".pp").onclick=()=>{
    if(au.paused||live!==sec._p) go(sec,G[0],null,null); else au.pause(); };
  sec.querySelectorAll("[data-p]").forEach(b=>{
    const d=JSON.parse(b.dataset.p);
    b.onclick=()=>go(sec,G[d[0]],G[Math.min(n,d[1])],b); });
});
</script></body></html>"""


def main():
    for stem in (sys.argv[1:] or [DEFAULT]):
        page(stem)


if __name__ == "__main__":
    main()
