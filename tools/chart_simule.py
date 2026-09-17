"""Le banc d'essai : annoter une brique, et voir les deux stratégies vivre.

Louis, 2026-09-17 : « fais-moi une simulation du chart d'annotation avec les
2 stratégies (notre algo reformulé vs SongFormer) sur une chanson, pour que je
puisse tester les 2 STRATS INTERACTIVEMENT SUIVANT CE QUE J'ANNOTE ».

On touche une mesure, puis une autre : c'est une brique. On lui donne un nom,
on l'ajoute. Le serveur cherche alors ses reprises dans le morceau, puis
remplit les trous de deux façons — l'agglomération des quatre mots confinée
aux trous, ou ce que SongFormer avait trouvé. Les deux s'affichent au même
endroit, on bascule d'un bouton, et chaque bloc s'écoute au doigt.

RIEN N'EST ÉCRIT. La route `/api/sections/simuler/` ne touche ni au chart, ni
aux brouillons, ni aux sections de vérité : c'est un banc, pas l'outil
d'annotation. La production reste `sections_inferer`, qui ignore ce chemin.

    python -m tools.chart_simule --chart min_B6AHb9W_LkM
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from harmonia.settings import SETTINGS
from tools.annotation_degats import nom_accord


def donnees(cle: str) -> dict:
    from harmonia.soudure import accords_par_mesure
    chart = json.loads((SETTINGS.charts_dir / f"{cle}.json")
                       .read_text(encoding="utf-8"))
    n = int(chart.get("nBars") or 0)
    stem = Path(chart.get("audio_url") or "").stem
    return {
        "cle": cle, "titre": chart.get("title") or stem, "n": n,
        "audio": chart.get("audio_url") or f"/audio/{stem}.m4a",
        "grid": [round(float(t), 3) for t in (chart.get("barGrid") or [])],
        "accords": [" ".join(nom_accord(c) for c in (b or [])) or "·"
                    for b in accords_par_mesure(chart)][:n],
    }


CSS = """
*{box-sizing:border-box}
body{margin:0 auto;padding:12px 12px 40px;background:#faf6ec;color:#2c2820;
 font:15px/1.45 system-ui,-apple-system,sans-serif;max-width:860px}
h1{font-size:18px;margin:0 0 4px}
.note{color:#8a8371;font-size:12.5px}
.lede{background:#fff7df;border:1px solid #e8d9a8;border-radius:10px;
 padding:9px 11px;font-size:13.5px;margin:8px 0}
.barre{position:sticky;top:0;background:#faf6ec;padding:8px 0 6px;z-index:6;
 border-bottom:1px solid #eee3c8}
.row{display:flex;gap:6px;flex-wrap:wrap;align-items:center}
button{border:1px solid #d8cfb4;border-radius:9px;background:#fff;padding:7px 10px;
 font:600 12.5px system-ui;cursor:pointer;min-height:40px;color:#2c2820;
 -webkit-tap-highlight-color:transparent}
button.on{background:#2c2820;border-color:#2c2820;color:#fff}
button.go{background:#8a2b2b;border-color:#8a2b2b;color:#fff}
button:disabled{opacity:.45}
button.lettre{min-width:44px}
.etat{font:600 13px system-ui;margin:6px 0 2px}
.grille{display:grid;grid-template-columns:repeat(4,1fr);gap:3px;margin-top:8px}
.mes{position:relative;min-height:54px;border-radius:5px;padding:14px 5px 5px;
 background:#f3edda;cursor:pointer;-webkit-tap-highlight-color:transparent}
.mes .no{position:absolute;top:2px;left:5px;font:600 10px system-ui;
 color:#8a8371;font-variant-numeric:tabular-nums}
.mes .ac{font:600 13px ui-monospace,monospace;line-height:1.3;word-break:break-word}
.mes.sel{outline:3px solid #8a2b2b;outline-offset:-3px}
.badge{position:absolute;top:-1px;right:3px;font:700 10px system-ui;color:#fff;
 padding:1px 5px;border-radius:0 0 4px 4px}
.badge.brique{box-shadow:0 0 0 2px #2c2820}
.briques{display:flex;gap:5px;flex-wrap:wrap;margin:6px 0}
.pastille{display:flex;align-items:center;gap:5px;border:1px solid #d8cfb4;
 border-radius:20px;padding:3px 5px 3px 10px;font:600 12px system-ui;
 background:#fff}
.pastille b{font-weight:700}
.pastille span{color:#8a8371;font-weight:500}
.pastille button{min-height:26px;padding:0 7px;border-radius:14px;font-size:14px}
"""

JS = r"""
const au=document.getElementById('au');let stop=null;
au.addEventListener('timeupdate',()=>{if(stop!=null&&au.currentTime>=stop){au.pause();stop=null;}});
function jouer(t0,t1){
  const go=()=>{try{au.currentTime=t0;}catch(e){}stop=t1;au.play().catch(()=>{});};
  if(au.getAttribute('src')!==D.audio){au.setAttribute('src',D.audio);
    au.addEventListener('loadedmetadata',go,{once:true});au.load();}
  else go();
}
function teinte(l){
  const T=["#8a2b2b","#2f5fa8","#1f7a6b","#b06a1f","#7b4ea3","#4a7c3f",
           "#a8336a","#556b2f","#8a6d3b","#3b6f8a"];
  const s=String(l||"?").toLowerCase().replace("′","");
  if(s==="intro"||s==="outro"||s==="silence"||s==="?") return "#8a8371";
  let h=0; for(const c of s) h+=c.charCodeAt(0);
  return T[h%T.length];
}
let briques=[], sel=null, lettre="A", vue="songformer", res=null, occupe=false;
const LETTRES=["intro","A","B","C","D","E","pont","outro"];

function cellules(){ return document.querySelectorAll('.mes'); }

function toucher(i){
  if(sel===null){ sel={a:i,b:i}; }
  else if(sel.a===sel.b && i!==sel.a){ sel={a:Math.min(sel.a,i), b:Math.max(sel.a,i)}; }
  else { sel={a:i,b:i}; }
  peindre();
}
async function ajouter(){
  if(sel===null || occupe) return;
  briques.push({label:lettre, mesure_debut:sel.a+1, mesure_fin:sel.b+1});
  sel=null; await calculer();
}
async function retirer(k){ briques.splice(k,1); await calculer(); }

async function calculer(){
  occupe=true; peindre();
  document.getElementById('etat').textContent="je cherche les reprises…";
  try{
    const r=await fetch("/api/sections/simuler/"+D.cle,{method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({humain:briques})});
    const d=await r.json();
    // JAMAIS en silence : un banc qui rend une liste vide ferait croire à un
    // découpage vide plutôt qu'à une panne.
    if(d.error){ res=null; document.getElementById('etat').textContent="⚠ "+d.error; }
    else { res=d; document.getElementById('etat').textContent=""; }
  }catch(e){ res=null; document.getElementById('etat').textContent="⚠ "+e; }
  occupe=false; peindre();
}

function blocs(){
  if(!res) return [];
  if(vue==="auto") return res.auto||[];
  if(vue==="briques") return (res.briques||[]).map(
      o=>({m0:o.m0,m1:o.m1,label:o.label,source:"brique",mot:o.score}));
  return res[vue]||[];
}

function peindre(){
  const cs=cellules();
  for(const c of cs){ c.style.background="#f3edda"; c.style.boxShadow="";
    c.classList.remove('sel');
    const b=c.querySelector('.badge'); if(b) b.remove(); }
  for(const u of blocs()){
    const coul=teinte(u.label), brique=(u.source==="brique");
    for(let m=u.m0;m<=u.m1 && m<cs.length;m++){
      cs[m].style.background=coul+(brique?"3a":"1c");
      cs[m].style.boxShadow="inset 4px 0 0 "+(m===u.m0?coul:"transparent")
                            +(brique?", 0 0 0 2px #2c2820":"");
    }
    const t=cs[u.m0];
    if(t){ const b=document.createElement('div');
      b.className='badge'+(brique?' brique':''); b.style.background=coul;
      b.textContent=(brique?"toi · ":"")+u.label+(u.mot?" · "+u.mot:"");
      t.appendChild(b); }
  }
  if(sel!==null) for(let m=sel.a;m<=sel.b;m++) cs[m] && cs[m].classList.add('sel');

  for(const b of document.querySelectorAll('.vue')) b.classList.toggle('on', b.dataset.v===vue);
  for(const b of document.querySelectorAll('.lettre')) b.classList.toggle('on', b.dataset.l===lettre);
  document.getElementById('ajouter').disabled = (sel===null || occupe);
  document.getElementById('ajouter').textContent = sel===null
    ? "touche une mesure, puis une autre"
    : "ajouter « "+lettre+" » sur les mesures "+(sel.a+1)+"-"+(sel.b+1);

  const bq=document.getElementById('briques'); bq.innerHTML="";
  briques.forEach((b,k)=>{
    const p=document.createElement('div'); p.className='pastille';
    p.innerHTML="<b style='color:"+teinte(b.label)+"'>"+b.label+"</b>"+
                "<span>"+b.mesure_debut+"-"+b.mesure_fin+"</span>";
    const x=document.createElement('button'); x.textContent="✕";
    x.onclick=()=>retirer(k); p.appendChild(x); bq.appendChild(p);
  });
  if(!briques.length) bq.innerHTML="<span class=note>aucune brique pour l'instant</span>";
}

window.addEventListener('DOMContentLoaded',()=>{
  cellules().forEach((c,i)=>{
    c.onclick=()=>{
      const u=blocs().find(x=>i>=x.m0&&i<=x.m1);
      // un appui long joue le bloc, un appui court sélectionne
      if(window.__jouerAuClic && u) jouer(D.grid[u.m0], D.grid[Math.min(u.m1+1,D.grid.length-1)]);
      else toucher(i);
    };
  });
  for(const b of document.querySelectorAll('.vue')) b.onclick=()=>{vue=b.dataset.v;peindre();};
  for(const b of document.querySelectorAll('.lettre')) b.onclick=()=>{lettre=b.dataset.l;peindre();};
  document.getElementById('ajouter').onclick=ajouter;
  document.getElementById('ecouter').onclick=()=>{
    window.__jouerAuClic=!window.__jouerAuClic;
    document.getElementById('ecouter').classList.toggle('on', !!window.__jouerAuClic);
  };
  calculer();
});
"""


def page(d: dict) -> str:
    cells = "".join(
        f"<div class=mes><div class=no>{i+1}</div>"
        f"<div class=ac>{html.escape(d['accords'][i] if i < len(d['accords']) else '·')}</div></div>"
        for i in range(d["n"]))
    lettres = "".join(
        f"<button class=lettre data-l=\"{l}\">{l}</button>"
        for l in ("intro", "A", "B", "C", "D", "E", "pont", "outro"))
    vues = "".join(
        f"<button class=vue data-v=\"{v}\">{t}</button>"
        for v, t in (("songformer", "SongFormer"), ("algo", "notre algo"),
                     ("briques", "tes briques seules"),
                     ("auto", "la machine sans toi")))
    corps = (
        f"<h1>{html.escape(d['titre'])} — banc d'essai</h1>"
        "<div class=lede><b>Touche une mesure, puis une autre</b> : c'est une "
        "brique. Nomme-la, ajoute-la. Le serveur cherche ses reprises, puis "
        "remplit les trous des deux façons — bascule entre elles d'un bouton."
        "<br>Le bouton <b>écouter</b> change le doigt de rôle : il joue le bloc "
        "touché au lieu de sélectionner.<br>"
        "<b>Rien n'est enregistré</b> : c'est un banc, pas l'outil d'annotation."
        "</div>"
        "<div class=barre>"
        f"<div class=row>{vues}<button id=ecouter>▶ écouter</button></div>"
        f"<div class=row style='margin-top:6px'>{lettres}</div>"
        "<div class=row style='margin-top:6px'>"
        "<button class=go id=ajouter style='flex:1'></button></div>"
        "<div class=briques id=briques></div>"
        "<div class=etat id=etat></div>"
        "</div>"
        f"<div class=grille>{cells}</div>"
        f"<p class=note>{d['n']} mesures · le cadre noir marque une brique à "
        "toi ou une de ses reprises trouvées</p>")
    data = json.dumps({"cle": d["cle"], "audio": d["audio"], "grid": d["grid"]},
                      ensure_ascii=False)
    return ("<!-- tools/chart_simule.py -->"
            "<meta charset=utf-8>"
            "<meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>{html.escape(d['titre'])} — banc d'essai</title>"
            "<style>" + CSS + "</style><body>" + corps
            + "<audio id=au preload=auto playsinline></audio>"
            "<script>const D=" + data + ";</script><script>" + JS + "</script>")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chart", default="min_B6AHb9W_LkM")
    ap.add_argument("--titre", default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    d = donnees(a.chart)
    if a.titre:
        d["titre"] = a.titre
    out = a.out or (SETTINGS.reports_dir / f"chart_simule_{a.chart}.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page(d), encoding="utf-8")
    print(f"→ {out}\n   {d['n']} mesures · la page appelle "
          f"/api/sections/simuler/{a.chart}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
