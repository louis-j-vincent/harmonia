"""La page de diagnostic des régions de rythme sur The APL Song (2026-10-13).

    python -m tools.page_regions_rythme --avant DIR --apres DIR

`--avant` / `--apres` : deux sorties du rapport d'or (même code, brique
`SETTINGS.regions_rythme` éteinte / allumée). Écrit
`docs/plots/apl_song_tempo_2026-10-13.html` et sa figure `.png`.
Ne lit que des caches chauds (battues, postérieures musx) : aucun modèle.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np

from harmonia import cache
from harmonia.musx import FRAME_DT
from harmonia.rythme import changements_accords
from harmonia.settings import SETTINGS

KEY = "u3wUZw9S2PM"
OUT = SETTINGS.repo / "docs" / "plots" / "apl_song_tempo_2026-10-13"
FENETRES = [(30.0, 46.5), (113.0, 128.0)]          # autour des deux frontières


def _bars(chart: dict) -> list[dict]:
    """Les mesures dans l'ordre de l'audio (APL n'est pas replié : reps = 1)."""
    g = chart["barGrid"]
    out = []
    for s in chart["sections"]:
        for (b0, b1) in s["barRanges"]:
            for i, b in enumerate(range(b0, b1 + 1)):
                items = s["bars"][i] if i < len(s["bars"]) else []
                out.append({"n": b + 1, "t0": g[b], "t1": g[b + 1],
                            "ch": [{"r": c["root"], "q": c["q"], "b": c.get("bass", -1),
                                    "nc": c["nc"], "beat": c["beat"],
                                    "carry": bool(c.get("carry")),
                                    "t0": c["t0"], "t1": c["t1"]} for c in items]})
    return out


def _figure(avant: dict, apres: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    au = SETTINGS.audio_dir / f"{KEY}.m4a"
    bj = cache.load_json("beats", au)
    tri = cache.load_npz("musx_probs", au)["triad"]
    bt = np.asarray(apres["beatTimes"])
    ons = changements_accords(tri, FRAME_DT, float(np.median(np.diff(bt))), sans_nc=True)
    fig, axs = plt.subplots(len(FENETRES), 1, figsize=(9, 2.3 * len(FENETRES)))
    for ax, (a, z) in zip(axs, FENETRES):
        for t in bt[(bt >= a) & (bt <= z)]:
            ax.axvline(t, 0.0, 1.0, color="#ddd", lw=0.8)
        for t in [x for x in bj["downbeats"] if a <= x <= z]:
            ax.plot([t, t], [0.72, 1.0], color="#c0392b", lw=2.2)
        for t in ons[(ons >= a) & (ons <= z)]:
            ax.plot([t, t], [0.38, 0.66], color="#8e44ad", lw=3)
        for t in [x for x in avant["barGrid"] if a <= x <= z]:
            ax.plot([t, t], [0.0, 0.32], color="#999", lw=2, ls=(0, (2, 1.5)))
        for t in [x for x in apres["barGrid"] if a <= x <= z]:
            ax.plot([t, t], [0.0, 0.32], color="#1d6fdb", lw=2.4)
        ax.set_xlim(a, z); ax.set_ylim(0, 1); ax.set_yticks([0.16, 0.52, 0.86])
        ax.set_yticklabels(["barres du chart\n(gris = avant, bleu = après)",
                            "accords musx", "downbeats\nBeat This!"], fontsize=8)
        ax.tick_params(axis="x", labelsize=8)
        ax.set_xlabel("secondes", fontsize=8)
    axs[0].set_title("The APL Song — les accords (violet) et le traqueur (rouge) "
                     "sautent d'un temps à 38,5 s ; retour à 121,4 s", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def _decalage(r: dict, r0: dict) -> str:
    d = (r["phase"] - r0["phase"]) % r["beatsPerBar"]
    return "celle de l'intro" if d == 0 else f"un temps 1 décalé de {d} temps"


def build(avant_dir: Path, apres_dir: Path) -> Path:
    f = f"min_{KEY}.json"
    avant = json.loads((avant_dir / f).read_text())
    apres = json.loads((apres_dir / f).read_text())
    _figure(avant, apres, OUT.with_suffix(".png"))
    data = {"audio": apres["audio_url"], "fen": FENETRES,
            "avant": _bars(avant), "apres": _bars(apres),
            "regions": apres.get("tempoRegions", [])}
    regs = "".join(
        f"<tr><td>{i + 1}</td><td>{r['t0']:.1f} → {r['t1']:.1f} s</td>"
        f"<td>{r['bpm']:.0f}</td><td>{_decalage(r, data['regions'][0])}</td></tr>"
        for i, r in enumerate(data["regions"]))
    page = TEMPLATE.replace("__DATA__", json.dumps(data)).replace(
        "__REGIONS__", regs).replace("__PNG__", OUT.with_suffix(".png").name)
    OUT.with_suffix(".html").write_text(page, encoding="utf-8")
    return OUT.with_suffix(".html")


TEMPLATE = """<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>APL Song, les barres de mesure</title>
<style>
:root{color-scheme:light;--bg:#f6f2e7;--fg:#222;--mut:#8a8371;--line:#c9c0a6;--line2:#e3dcc6;--acc:#8a2b2b;--son:#cfe3ff;--head:#2a6fdb}
*{box-sizing:border-box} html,body{margin:0;overflow-x:hidden}
body{max-width:520px;margin:0 auto;padding:14px 16px 70px;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
h1{font-size:18px;margin:0 0 6px} h2{font-size:16px;margin:24px 0 4px;color:var(--acc)}
.lede{background:#fff7df;border:1px solid #e8d9a8;border-radius:10px;padding:9px 11px;font-size:13.5px;margin:8px 0}
.lede p{margin:4px 0} .n{color:var(--mut);font-size:12px}
img{width:100%;height:auto;border:1px solid var(--line2);border-radius:6px;background:#fff}
table{border-collapse:collapse;font-size:12.5px;width:100%} td,th{border-bottom:1px solid var(--line2);padding:3px 4px;text-align:left}
.pl{font:600 12.5px system-ui;padding:5px 11px;border-radius:14px;border:1px solid var(--acc);background:#fff;color:var(--acc);cursor:pointer;margin:6px 0}
.pl.on{background:var(--acc);color:#fff}
.cap{font:600 10px system-ui;letter-spacing:.04em;text-transform:uppercase;color:var(--mut);margin:6px 0 1px}
.row{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));border-left:1px solid var(--line)}
.bar{position:relative;height:58px;padding:12px 2px 0 4px;border-right:1px solid var(--line);border-bottom:1px solid var(--line2);display:grid;align-items:center;overflow:hidden}
.bar.court{background:#fde7c2}
.bar .no{position:absolute;top:1px;left:3px;font:10px system-ui;color:var(--mut)}
.bar .tps{position:absolute;top:1px;right:3px;font:600 9px system-ui;color:#a35a00}
.ch{height:40px;display:flex;align-items:center;border-radius:6px;cursor:pointer;min-width:0;-webkit-tap-highlight-color:transparent}
.ch.carry{opacity:.45}
.ch.son{background:var(--son);box-shadow:inset 0 0 0 2px var(--head)}
.ch .w{display:inline-block;white-space:nowrap;transform-origin:left center}
.vu{position:sticky;bottom:0;margin-top:12px;background:#fff;border:1px solid var(--line);border-radius:10px;padding:6px 10px;font-size:12.5px;min-height:30px}
</style></head><body>
<h1>The APL Song : l'intro n'a pas la même barre de mesure que le reste</h1>
<div class="lede">
<p><b>Ce que j'ai mesuré.</b> Le tempo ne change pas : 89 BPM dans l'intro comme dans le corps (le traqueur bat à 0,674 s du début de l'intro à la fin). Ce qui change, c'est <b>où tombe le temps 1</b>.</p>
<p>L'intro (l'échantillon, sans batterie, 16 → 38,5 s) dure <b>8 mesures et 1 temps</b>. Quand la batterie entre à 38,5 s, la mesure repart un temps plus tard. Les accords le disent, et Beat This! aussi : ses temps 1 sautent d'un temps au même endroit. Vers 121 s, le morceau revient sur la barre de l'intro (une mesure de 3 temps).</p>
<p><b>Ce que faisait le chart.</b> Une seule barre de mesure pour tout le morceau, celle de l'intro (la mesure 1 détectée à 16,5 s). Résultat : de la mes. 17 à la mes. 48, <b>chaque accord tombait sur le temps 2</b> — « Em Bm | Bm Em | … » au lieu de « Em | Bm | … ».</p>
<p><b>La brique.</b> Une portion d'au moins une phrase (4 mesures) reçoit sa propre barre de mesure quand ses accords <b>et</b> les temps 1 du traqueur tombent, nettement plus souvent que le hasard, sur une autre barre que la portion voisine. La mesure de jonction est courte (ici 1 temps, puis 3 temps).</p>
</div>
<img src="__PNG__" alt="accords, downbeats et barres autour des deux frontières">
<p class="n">Violet : changements d'accords entendus par musx (ils ignorent les battues). Rouge : temps 1 de Beat This!. En bas : barres du chart, gris pointillé avant, bleu après.</p>
<table><tr><th>portion</th><th>où</th><th>BPM</th><th>barre de mesure</th></tr>__REGIONS__</table>
<div id="fen"></div>
<h2>Ce que ça ne règle pas</h2>
<p class="n">Les 16 premières secondes (bruit d'ambiance, sans pulsation) restent 6 mesures vides au tempo du corps. La mesure courte s'affiche comme une case pleine (rien n'écrit « 1/4 »). Un vrai passage en 3/4, un rubato, un accelerando : hors de portée de cette brique.</p>
<div class="vu" id="vu">Touche un accord ou ▶ écouter.</div>
<script id="data" type="application/json">__DATA__</script>
<script type="module">
await import("/static/screens/chart.js");
const { glyphEpure } = await import("/static/ui/kit.js");
const D = JSON.parse(document.getElementById("data").textContent);
const SZ=[0,26,18,14,14];
const vu=document.getElementById("vu"), root=document.getElementById("fen");
const au=document.createElement("audio"); au.preload="auto"; au.setAttribute("playsinline",""); document.body.appendChild(au);
fetch(D.audio).then(r=>r.blob()).then(b=>{au.src=URL.createObjectURL(b);}).catch(()=>{au.src=D.audio;});   // iPhone : fetch()+blob
const CH=[]; let cur=null, timer=null;
function glyph(c,n){ if(c.nc){const s=document.createElement("span");s.textContent="N.C.";s.style.font="italic 13px Georgia,serif";return s;}
  return glyphEpure(c.r,c.q,SZ[Math.min(n,4)],"exact","#1c1c1c",c.b); }
D.fen.forEach(([a,z],w)=>{
  const h=document.createElement("h2"); h.textContent=w===0?"Autour de 38,5 s : la batterie entre":"Autour de 121 s : retour sur la barre de l'intro"; root.appendChild(h);
  const pl=document.createElement("button"); pl.className="pl"; pl.textContent=`▶ écouter ${a.toFixed(0)} → ${z.toFixed(0)} s`;
  pl.onclick=()=>play(a,z,pl,null,"le passage"); root.appendChild(pl);
  for(const ver of ["avant","apres"]){
    const cap=document.createElement("div"); cap.className="cap"; cap.textContent=ver==="avant"?"aujourd'hui (une seule barre)":"avec les régions"; root.appendChild(cap);
    const bars=D[ver].filter(b=>b.t1>a+0.05&&b.t0<z-0.05);
    for(let i=0;i<bars.length;i+=4){
      const row=document.createElement("div"); row.className="row"; root.appendChild(row);
      bars.slice(i,i+4).forEach(b=>{
        const el=document.createElement("div"); el.className="bar"; row.appendChild(el);
        const tps=Math.round((b.t1-b.t0)/0.6743);
        if(tps<4){el.classList.add("court"); const t=document.createElement("span"); t.className="tps"; t.textContent=tps+(tps>1?" temps":" temps"); el.appendChild(t);}
        const no=document.createElement("span"); no.className="no"; no.textContent=b.n; el.appendChild(no);
        el.style.gridTemplateColumns=`repeat(${Math.max(1,b.ch.length)},minmax(0,1fr))`;
        b.ch.forEach(c=>{
          const e=document.createElement("div"); e.className="ch"+(c.carry?" carry":"");
          const sp=document.createElement("span"); sp.className="w"; sp.appendChild(glyph(c,b.ch.length)); e.appendChild(sp); el.appendChild(e);
          const t0=Math.max(c.t0,b.t0), t1=Math.min(c.t1,b.t1);
          e.onclick=()=>play(t0,t1,null,e,`mes. ${b.n}`);
          CH.push({e,t0,t1});
        });
      });
    }
  }
});
function play(t0,t1,btn,el,txt){
  if(cur&&((btn&&cur.btn===btn)||(el&&cur.el===el))&&!au.paused){au.pause();return;}
  if(cur) finish();
  cur={t0,t1,btn,el}; if(btn) btn.classList.add("on");
  vu.textContent=`▶ ${txt} — de ${t0.toFixed(2)} à ${t1.toFixed(2)} s`;
  const go=()=>{ try{au.currentTime=t0;}catch(e){} const p=au.play(); if(p&&p.catch) p.catch(()=>finish()); };
  if(!au.src) setTimeout(go,300); else go();
}
function arm(){ clearTimeout(timer); if(!cur) return; const dt=(cur.t1-au.currentTime)/(au.playbackRate||1);
  if(dt<=0.005){au.pause();return;} timer=setTimeout(arm,Math.max(5,Math.min(dt*1000,250))); }
function finish(){ clearTimeout(timer); if(!cur) return; if(cur.btn) cur.btn.classList.remove("on");
  vu.textContent=vu.textContent.replace("▶","■"); cur=null; CH.forEach(c=>c.e.classList.remove("son")); }
au.addEventListener("playing",arm);
au.addEventListener("pause",()=>finish());
au.addEventListener("timeupdate",()=>{ const t=au.currentTime; if(cur&&t>=cur.t1-0.005){au.pause();return;}
  CH.forEach(c=>c.e.classList.toggle("son",!au.paused&&t>=c.t0&&t<c.t1)); });
</script>
</body></html>
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--avant", type=Path, required=True)
    ap.add_argument("--apres", type=Path, required=True)
    a = ap.parse_args(argv)
    print("→", build(a.avant, a.apres))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
