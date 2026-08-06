"""Où commence le A ? Le chant comme indice, et plusieurs hypothèses au lieu d'une.

    python scripts/vocal_anchor.py [<stem> ...]  ->  /reports/vocal_anchor.html

Louis, 2026-08-06 :

  « Ce qui est hyper important, c'est de créer plusieurs hypothèses sur où
    commence la section A. Parce qu'il y a souvent une intro en début de
    chanson, et du coup ça brouille pas mal de pistes. Les sections sont
    toujours à un multiple d'au moins quatre ou huit mesures — on regarde la
    granularité quatre pour commencer, avec des jointures à deux mesures. Et ce
    qui pourrait aider à trouver les débuts de section, ce serait d'extraire la
    piste vocale et de détecter quand la personne commence à CHANTER, pas à
    parler. En général la personne commence à chanter juste avant ou sur le
    premier temps de la section A, parfois un poil après. »

CE QUE FAIT LA PAGE

1. On sépare la voix (demucs, deux pistes) et on cherche **le premier chant** —
   pas le premier bruit. Le critère est la HAUTEUR TENUE : on suit la fréquence
   fondamentale de la piste vocale, et on ne retient que les instants où elle
   est à la fois présente, assez forte, et STABLE sur un demi-temps. Parler
   fait bouger la hauteur sans arrêt et sans la tenir ; chanter la pose. Le
   premier endroit où cela dure au moins une demi-seconde est « le chant
   commence ici ».

2. De là, plusieurs **hypothèses de départ du A**, jamais une seule. Le chant
   arrive « juste avant, sur, ou un poil après » le premier temps du A : on
   propose donc les mesures paires autour de lui, de −4 à +2 mesures, plus le
   début du cœur harmonique comme témoin.

3. Chaque hypothèse **cale la grille de 2 mesures** et refait tout le découpage
   par-dessus. Une intro de longueur impaire décale sinon la phase de tous les
   motifs, et c'est exactement le brouillage décrit.

La page montre l'énergie de la voix, la hauteur suivie, le moment retenu comme
début du chant, et le découpage obtenu sous CHAQUE hypothèse, sur le même axe de
mesures, avec la tête de lecture pour vérifier à l'oreille.
"""
from __future__ import annotations

import os
import subprocess
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
from pattern_lanes import load, fig2b64_fixed, COLS, INK, PLOT_L, PLOT_R  # noqa: E402
from rhythm_ssm import DEFAULT_STEM_CACHE                                 # noqa: E402
import harmonia_min.harmonic_sections as HS                               # noqa: E402
import hypo_sizes as HY                                                   # noqa: E402
import spectral_sections as SP                                            # noqa: E402
import successor_cut as SC                                                # noqa: E402
import fusion_lens as FL                                                  # noqa: E402

OFFSETS = [-4, -2, 0, 2]      # en mesures, autour du premier chant
STABLE_CENTS = 150            # une hauteur « tenue » ne bouge pas plus que ça
STABLE_WIN = 0.12             # …sur cette fenêtre, en secondes
MIN_SING = 0.3                # …et il en faut au moins autant d'affilée
# Réglés en mesurant, pas au jugé. À 90 centièmes sur 0,5 s — mon premier
# choix — le détecteur attendait la première NOTE TENUE du morceau : Let It Be
# ne « chantait » qu'à 165 s, This Love à 80 s, Grenade à 82 s, tous faux. Une
# phrase chantée normale a du vibrato et des mélismes ; elle ne tient pas sa
# hauteur à un demi-ton près pendant une demi-seconde. À 150 / 0,3 s les cinq
# morceaux mesurés tombent juste (Let It Be 13,2 s, This Love 22,5 s, Bein
# Green 14,9 s, Norah 10,2 s, Grenade 7,6 s) et le critère de stabilité sert
# encore : sur Grenade il écarte 2,8 s de voisé non chanté avant l'entrée.
DEFAULT = ["norah_jones_don_t_know_why", "maroon_5_this_love",
           "mayer_hawthorne_the_walk", "bein_green",
           "let_it_be_remastered_2009",
           "bruno_mars_grenade_official_music_video",
           "maroon_5_she_will_be_loved_official_music_video"]


def separate_vocals(audio_path, cache_dir=DEFAULT_STEM_CACHE, device="mps"):
    """demucs deux-pistes sur la voix, mis en cache."""
    audio_path = Path(audio_path)
    out = cache_dir / "htdemucs" / audio_path.stem / "vocals.wav"
    if out.exists():
        return out
    cache_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "demucs", "--two-stems=vocals", "-n", "htdemucs",
           "-d", device, "-o", str(cache_dir), str(audio_path)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if not out.exists():
        raise RuntimeError(f"demucs n'a pas produit {out}: {r.stderr[-500:]}")
    return out


def sing_onset(vocal_path, sr=22050):
    """Le premier CHANT : hauteur présente, forte, et TENUE.

    Parler fait glisser la hauteur en permanence et ne la tient jamais ; chanter
    la pose sur une note. On suit donc la fondamentale, on mesure de combien
    elle bouge sur un demi-temps, et on ne garde que les instants où elle bouge
    moins de STABLE_CENTS centièmes de demi-ton tout en étant assez forte. Le
    premier endroit où cela dure MIN_SING secondes est le début du chant.

    Retourne (temps du premier chant, temps des trames, énergie, hauteur, masque
    « ça chante ») — tout est dessiné, donc le critère est contestable à l'œil.
    """
    import librosa
    y, sr = librosa.load(str(vocal_path), sr=sr, mono=True)
    hop = 256
    f0, voiced, vprob = librosa.pyin(y, sr=sr, hop_length=hop,
                                     fmin=librosa.note_to_hz("C2"),
                                     fmax=librosa.note_to_hz("C6"))
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    t = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    f0 = f0[:len(rms)]
    voiced = voiced[:len(rms)]
    cents = 1200 * np.log2(np.where(np.isfinite(f0), f0, 1.0) / 55.0)
    w = max(2, int(round(STABLE_WIN * sr / hop)))
    stable = np.zeros(len(rms), bool)
    for i in range(len(rms)):
        a, b = max(0, i - w), min(len(rms), i + w + 1)
        seg = cents[a:b][voiced[a:b] & np.isfinite(cents[a:b])]
        stable[i] = len(seg) >= w and (seg.max() - seg.min()) <= STABLE_CENTS
    loud = rms > max(1e-4, .12 * float(np.percentile(rms, 95)))
    sing = stable & voiced & loud
    need = int(round(MIN_SING * sr / hop))
    run, onset = 0, None
    for i, s in enumerate(sing):
        run = run + 1 if s else 0
        if run >= need:
            onset = float(t[i - run + 1])
            break
    return onset, t, rms, np.where(voiced, f0, np.nan), sing


def bar_of(grid, tsec):
    if tsec is None:
        return None
    g = np.asarray(grid, float)
    if tsec <= g[0]:
        return 0
    return int(np.searchsorted(g, tsec) - 1)


def cut_with_start(S, n, start):
    """Le découpage complet sous l'hypothèse « le A commence à la mesure `start` »."""
    cells = HY.build_hypo(S, n, unit=2, start=start)
    chain = HY.merge_two_to_four(HY.chain_of(cells, n))[0]
    syms = [c["sym"] if not c["hole"] else "·" for c in chain]
    var, _ = SC.variety(syms)
    bounds = [chain[a]["b0"] for a, _ in SC.cut_on_rise(syms, var)][1:]
    bounds = [start] + [b for b in bounds if b > start]
    return FL.group(S, FL.spans_from(bounds, n), "diagonale glissée", 0.85)


def song(stem):
    S, n, grid = load(stem)
    voc = separate_vocals(HERE / f"docs/audio/{stem}.m4a")
    onset, t, rms, f0, sing = sing_onset(voc)
    b_sing = bar_of(grid, onset)
    c0, _ = SP.core_range(S)

    hyps = []
    if b_sing is not None:
        base = b_sing - (b_sing % 2)
        for d in OFFSETS:
            h = base + d
            if 0 <= h < n - 8 and h % 2 == 0 and h not in [x for x, _ in hyps]:
                hyps.append((h, f"chant {d:+d} mes."))
    if c0 % 2 == 0 and c0 not in [x for x, _ in hyps]:
        hyps.append((c0, "cœur harmonique"))
    if 0 not in [x for x, _ in hyps]:
        hyps.append((0, "mesure 1 (témoin)"))
    hyps.sort()

    cuts = [(h, lab, cut_with_start(S, n, h)) for h, lab in hyps]

    heights = [2.8, 0.9, 0.9] + [0.55] * len(cuts)
    H = sum(heights) + 1.3
    fig, axs = plt.subplots(len(heights), 1, sharex=True, figsize=(12.6, H),
                            gridspec_kw={"height_ratios": heights, "hspace": .24})
    fig.subplots_adjust(left=PLOT_L, right=PLOT_R, top=1 - .32 / H, bottom=.58 / H)
    axs[0].imshow(S, origin="lower", extent=(0, n, 0, n), cmap="RdYlBu_r",
                  vmin=float(np.percentile(S, 5)), vmax=float(np.percentile(S, 99)),
                  aspect="auto", interpolation="nearest")
    axs[0].set_ylabel("mesure", fontsize=7.5); axs[0].tick_params(labelsize=6.4)

    # l'énergie de la voix et la hauteur, sur l'axe des mesures
    g = np.asarray(grid, float)
    tb = np.interp(t, g, np.arange(len(g)))       # temps -> mesure (flottant)
    ax = axs[1]
    ax.fill_between(tb, rms / (rms.max() or 1), color="#8a2b2b", alpha=.30, lw=0)
    ax.set_ylim(0, 1.05); ax.set_yticks([])
    ax.set_ylabel("voix\n(énergie)", fontsize=6.8, rotation=0, ha="right",
                  va="center", color="#8a2b2b")
    ax = axs[2]
    ax.plot(tb, f0, ".", ms=1.1, color="#8a8371")
    ax.plot(tb[sing], f0[sing], ".", ms=1.6, color="#0f766e")
    ax.set_yscale("log"); ax.set_yticks([])
    ax.set_ylabel("hauteur\nvert = tenue", fontsize=6.8, rotation=0, ha="right",
                  va="center", color="#0f766e")
    for a in (axs[1], axs[2]):
        for sp in ("top", "right", "left"):
            a.spines[sp].set_visible(False)
        if b_sing is not None:
            a.axvline(b_sing, color="#0f766e", lw=1.6)
    if b_sing is not None:
        axs[1].text(b_sing, 1.02, f"  chant : mes. {b_sing+1} ({onset:.1f} s)",
                    fontsize=7, color="#0f766e", va="top")

    for i, (h, lab, secs) in enumerate(cuts):
        FL.BV.strip(axs[3 + i], secs, n, f"A en mes. {h+1}\n{lab}")
        axs[3 + i].axvline(h, color="#111", lw=1.6)
    axs[-1].set_xticks(range(0, n + 1, 4)); axs[-1].tick_params(labelsize=6.4)
    axs[-1].set_xlabel("mesure", fontsize=8)
    img = fig2b64_fixed(fig)

    fmt = lambda x: " ".join(f"{s['letter']}[{s['b0']+1}-{s['b1']+1}]" for s in x)
    rows = "".join(
        f"<tr><td><b>mesure {h+1}</b></td><td>{lab}</td>"
        f"<td>{len({s['letter'] for s in secs})}</td><td>{len(secs)}</td>"
        f"<td class=f>{fmt(secs)}</td></tr>" for h, lab, secs in cuts)
    gridjs = "[" + ",".join(f"{x:.3f}" for x in grid) + "]"
    btns = "".join(
        f"<button class=blk data-p='[{max(0, h-2)},{min(n, h+4)}]'>mes. {h+1}"
        f"<small>{lab[:9]}</small></button>" for h, lab, _ in cuts)
    return f"""<section data-grid='{gridjs}' data-audio="/audio/{stem}.m4a">
<h2>{stem.replace('_',' ').title()}<span class=sub>{n} mesures ·
chant à {f'{onset:.1f} s (mesure {b_sing+1})' if onset else 'non détecté'}</span></h2>
<div class=plot><img src="data:image/png;base64,{img}">
<div class=cur></div><div class=hit></div></div>
<div class=bar><button class=pp>▶</button><span class=pos>mes. 1 · 0:00</span>
<span class=hint>chaque bouton joue les mesures autour d'une hypothèse</span>
{btns}</div>
<table><tr><th>le A commence</th><th>hypothèse</th><th>lettres</th>
<th>sections</th><th>découpage</th></tr>{rows}</table></section>"""


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
    out = HERE / "harmonia_min/state/reports/vocal_anchor.html"
    out.write_text(f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Où commence le A ?</title><style>
*{{box-sizing:border-box;-webkit-tap-highlight-color:transparent}}
body{{margin:0;background:#e7e0d0;font:16px/1.55 -apple-system,system-ui,sans-serif;color:{INK}}}
.wrap{{max-width:1150px;margin:0 auto;padding:20px 12px calc(40px + env(safe-area-inset-bottom))}}
h1{{font:italic 600 24px Georgia,serif;margin:0 0 3px}}
.lede{{color:#6f6858;font-size:13.5px;line-height:1.55;margin-bottom:18px}}
.lede b{{color:{INK}}}
section{{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;padding:14px 13px;margin-bottom:16px}}
h2{{font:700 18px system-ui;margin:0 0 6px;color:#8a2b2b}}
.sub{{font:500 12px system-ui;color:#8a8371;margin-left:8px}}
img{{width:100%;border-radius:8px;display:block}}
table{{border-collapse:collapse;font-size:12.5px;width:100%;margin-top:8px}}
th,td{{border:1px solid #e5dcc6;padding:4px 8px;text-align:left;vertical-align:top}}
th{{background:#f7f3e9;font-size:11px}} td.f{{font:500 11px ui-monospace,monospace}}
.plot{{position:relative;margin-bottom:8px}} .plot img{{margin:0}}
.cur{{position:absolute;top:0;bottom:0;width:2px;background:#111;opacity:.8;
  display:none;pointer-events:none;box-shadow:0 0 0 1px rgba(255,255,255,.55)}}
.hit{{position:absolute;top:0;bottom:0;cursor:crosshair}}
.bar{{display:flex;flex-wrap:wrap;align-items:center;gap:5px;margin:0 0 6px}}
.pp{{width:36px;height:36px;border-radius:50%;border:1px solid #d8cfb4;
  background:#f7f3e9;font-size:13px;cursor:pointer;flex:none}}
.pos{{font:600 12px ui-monospace,monospace;min-width:88px}}
.hint{{font:500 11px system-ui;color:#a89f8c;margin-right:6px}}
button.blk{{border:1px solid #d8cfb4;background:#f7f3e9;border-radius:6px;
  padding:3px 7px;cursor:pointer;font:700 11.5px ui-monospace,monospace}}
button.blk small{{display:block;font:500 8.5px ui-monospace,monospace;color:#a89f8c}}
button.blk.on{{color:#fff !important;background:#8a2b2b !important}}
</style></head><body><div class=wrap>
<h1>Où commence le A ?</h1>
<div class=lede>Il y a presque toujours une intro, et c'est elle qui brouille
tout : si la grille de 2 mesures part de la mesure 1 alors que l'intro fait un
nombre impair de mesures, la phase de tous les motifs est décalée pour le reste
du morceau.<br><br>
<b>Le chant comme indice.</b> On sépare la voix, puis on cherche le premier
<b>chant</b> — pas le premier bruit. Le critère est la <b>hauteur tenue</b> :
parler fait glisser la hauteur en permanence, chanter la pose sur une note. On
suit donc la fondamentale de la piste vocale et on ne garde que les instants où
elle est présente, assez forte, et stable à moins d'un demi-ton sur un
demi-temps. Le premier endroit où ça dure une demi-seconde est retenu. Les deux
bandes sous la matrice montrent l'énergie de la voix et la hauteur suivie — en
vert les instants jugés « chantés » — donc le critère se conteste à l'œil.
<br><br>
<b>Plusieurs hypothèses, jamais une seule.</b> Le chant arrive juste avant, sur,
ou un peu après le premier temps du A : on essaie donc les mesures paires autour
de lui, de −4 à +2, plus le début du cœur harmonique et la mesure 1 comme
témoins. <b>Chaque hypothèse cale la grille de 2 mesures et refait tout le
découpage</b>, dessiné sur le même axe. Les boutons jouent les mesures autour de
chaque hypothèse pour trancher à l'oreille.</div>
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
