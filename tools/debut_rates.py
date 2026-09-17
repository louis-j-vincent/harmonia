"""Les morceaux où la règle du début se trompe, avec ce qu'elle a regardé.

Louis, 2026-09-17 : « montre-moi ceux où on se trompe, je vais te dire
pourquoi on se trompe, et mets sur chacun en dessous les métriques dont tu te
sers (intensité de la basse, de la batterie, des accords, + une autre
métrique de bruit ambiant parce que des fois dans l'intro il y a ça) ».

La règle est `harmonia.debut` : la première note de basse, calée sur la ligne
de mesure la plus proche. Elle désigne la même mesure que lui sur 35 morceaux
sur 42. Cette page montre les 7 autres, chacun avec les quatre courbes au
même axe de temps que l'onde, pour qu'il puisse dire ce qui manque.

Une seule des quatre sert la règle d'aujourd'hui (la basse). La batterie est
la piste qu'il a nommée en premier et la seule qui ne vienne pas de musx ;
le bruit ambiant est là pour les intros de clip, où il se passe du son sans
qu'il se passe de la musique.

    python -m tools.debut_rates
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np

from harmonia import musx as _musx
from harmonia.debut import (PAS_PISTE, cale_sur_grille, enveloppe, pistes,
                            premier_son, premiere_basse)
from harmonia.settings import SETTINGS

#: les quatre courbes, dans l'ordre d'affichage : (clé, nom lisible, couleur)
COURBES = (("basse", "basse", "#7b4ea3"),
           ("batterie", "batterie", "#1f7a6b"),
           ("accords", "accords", "#b06a1f"),
           ("bruit", "bruit ambiant", "#8a8371"))


def _index(t, grille) -> int | None:
    if t is None or not len(grille):
        return None
    return int(np.abs(np.asarray(grille, dtype=float) - t).argmin())


def _renormalise(v: list[float], n: int) -> list[float]:
    """Remet la courbe à l'échelle de la FENÊTRE MONTRÉE.

    `harmonia.debut` normalise sur 90 s ; une page qui n'en montre que 30
    écraserait l'intro sous le refrain. Ici on veut justement lire l'intro.
    """
    a = np.asarray(v[:n], dtype=float)
    if not len(a):
        return []
    haut = float(np.percentile(a, 97)) or float(a.max()) or 1.0
    return [round(float(min(1.0, x / haut)), 3) for x in a]


def collecte(verite: dict) -> list[dict]:
    out = []
    for stem, v in sorted(verite.items()):
        cp = SETTINGS.charts_dir / f"min_{stem}.json"
        audio = SETTINGS.audio_dir / f"{stem}.m4a"
        if not cp.exists() or not audio.exists():
            continue
        chart = json.loads(cp.read_text(encoding="utf-8"))
        grid = [float(t) for t in (chart.get("barGrid") or [])]
        if len(grid) < 3:
            continue
        env = enveloppe(audio)
        if env is None:
            continue
        probs = _musx.frame_posteriors(audio)
        son = premier_son(env)
        t_regle, _sur = cale_sur_grille(premiere_basse(probs[1], apres=son), grid)
        m_regle, m_vrai = _index(t_regle, grid), _index(v["t"], grid)
        if m_regle == m_vrai:
            continue                       # la règle tombe juste : rien à dire

        fin = min(90.0, max(v["t"], t_regle or 0.0) + 14.0, len(env) * 0.25)
        fin = max(fin, 20.0)
        n_env = int(fin / 0.25)
        n_pis = int(fin / PAS_PISTE)
        cr = pistes(audio, probs=probs)
        crete = max(env[:n_env]) or 1.0
        out.append({
            "stem": stem, "cle": cp.stem,
            "titre": chart.get("title") or stem,
            "audio": chart.get("audio_url") or f"/audio/{stem}.m4a",
            "fin": round(fin, 2),
            "env": [round(e / crete, 3) for e in env[:n_env]],
            "pas_env": 0.25, "pas_piste": PAS_PISTE,
            "courbes": {k: _renormalise(cr[k], n_pis) for k, _n, _c in COURBES},
            "grille": [round(t, 3) for t in grid if t < fin],
            "vrai": round(v["t"], 2), "mesure_vraie": m_vrai,
            "regle": None if t_regle is None else round(t_regle, 2),
            "mesure_regle": m_regle,
            "ecart": (m_regle - m_vrai) if m_regle is not None else None,
            "son": round(son, 2),
            "traqueur": round(grid[0], 2),
        })
    out.sort(key=lambda s: -abs(s["ecart"] or 0))
    return out


CSS = """
*{box-sizing:border-box}
body{margin:0 auto;padding:14px 12px 40px;background:#faf6ec;color:#2c2820;
 font:15px/1.45 system-ui,-apple-system,sans-serif;max-width:880px}
h1{font-size:19px;margin:0 0 8px}
h2{font-size:16px;margin:26px 0 6px}
.lede{background:#fff7df;border:1px solid #e8d9a8;border-radius:10px;
 padding:10px 12px;font-size:14px;margin:10px 0}
.note{color:#8a8371;font-size:12.5px}
.card{border:1px solid #ddd3b8;border-radius:12px;background:#fffdf7;
 padding:11px 13px;margin:16px 0}
.tt{font:600 15px system-ui}
.verdict{font-size:13px;margin:3px 0 7px}
.env,.pistes{position:relative;background:#f3edda;border-radius:8px;
 overflow:hidden;cursor:crosshair;touch-action:manipulation}
.env{height:78px;margin:8px 0 3px}
.pistes{height:150px;margin:3px 0}
canvas{display:block;width:100%}
.env canvas{height:78px}
.pistes canvas{height:150px}
.mk{position:absolute;top:0;bottom:0;width:2px;pointer-events:none}
.mk b{position:absolute;font:700 9.5px system-ui;white-space:nowrap;
 background:#fffdf7cc;padding:0 2px;border-radius:2px}
.cur{position:absolute;top:0;bottom:0;width:3px;background:#2c2820;pointer-events:none}
.cur b{position:absolute;bottom:3px;font:700 11px system-ui;color:#fff;
 background:#2c2820;padding:1px 4px;border-radius:3px;white-space:nowrap}
.leg{display:flex;gap:10px;flex-wrap:wrap;font:600 11.5px system-ui;margin:2px 0 6px}
.leg i{font-style:normal;display:inline-block;width:9px;height:9px;border-radius:2px;
 margin-right:4px;vertical-align:middle}
.row{display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin-top:7px}
button{border:1px solid #d8cfb4;border-radius:9px;background:#fff;padding:8px 11px;
 font:600 13px system-ui;cursor:pointer;min-height:42px;color:#2c2820}
button.play{background:#8a2b2b;border-color:#8a2b2b;color:#fff}
button.saut{font-size:12.5px;min-height:36px;padding:5px 9px}
input.pourquoi{flex:1 1 100%;min-height:42px;border:1px solid #d8cfb4;border-radius:9px;
 padding:8px 10px;font:14px system-ui;background:#fff;color:#2c2820}
textarea{width:100%;min-height:150px;font:12.5px/1.45 ui-monospace,monospace;
 border:1px solid #d8cfb4;border-radius:10px;padding:8px;background:#fff}
"""

JS = r"""
const au=document.getElementById('au');let stop=null;
au.addEventListener('timeupdate',()=>{if(stop!=null&&au.currentTime>=stop){au.pause();stop=null;}});
function jouer(src,t0,dur){                 // PILE au curseur, aucun élan
  const d0=Math.max(0,t0);
  const go=()=>{try{au.currentTime=d0;}catch(e){}stop=d0+(dur||8);au.play().catch(()=>{});};
  if(au.getAttribute('src')!==src){au.setAttribute('src',src);
    au.addEventListener('loadedmetadata',go,{once:true});au.load();}
  else go();
}
function ici(c){return parseFloat(c.dataset.choix)}
const KEY='harmonia_debut_rates_v1';
let P={};try{P=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){P={}}

function pose(c,t){
  const fin=parseFloat(c.dataset.fin);
  t=Math.max(0,Math.min(fin,t));c.dataset.choix=t.toFixed(2);
  for(const cur of c.querySelectorAll('.cur')){
    const pc=100*t/fin;cur.style.left=pc+'%';
    const et=cur.querySelector('b');if(!et)continue;
    et.textContent=t.toFixed(2)+'s';
    if(pc>62){et.style.left='auto';et.style.right='5px';}
    else{et.style.right='auto';et.style.left='5px';}
  }
  const g=JSON.parse(c.dataset.grille);
  let k=0;for(let i=1;i<g.length;i++) if(Math.abs(g[i]-t)<Math.abs(g[k]-t)) k=i;
  c.querySelector('.ou').textContent='mesure '+(k+1)+' de la grille';
}
function noter(inp){
  const c=inp.closest('.card');
  if(inp.value.trim()) P[c.dataset.stem]=inp.value.trim(); else delete P[c.dataset.stem];
  try{localStorage.setItem(KEY,JSON.stringify(P))}catch(e){}
  rendre();
}
function rendre(){
  const L=[];
  for(const st of Object.keys(P)) L.push(st+' : '+P[st]);
  document.getElementById('out').value=L.sort().join('\n')
    ||"(écris sous chaque morceau pourquoi on se trompe)";
}
function presse(txt){
  const replier=()=>{try{document.execCommand('copy')}catch(e){}};
  try{const pr=navigator.clipboard&&navigator.clipboard.writeText(txt);
      if(pr&&pr.catch) pr.catch(replier); else replier();}catch(e){replier();}
}
function copier(){rendre();const t=document.getElementById('out');
  t.select();t.setSelectionRange(0,99999);presse(t.value);}

function fond(x,w,h,g,fin){                  // les lignes de mesure
  x.fillStyle='#f3edda';x.fillRect(0,0,w,h);
  x.strokeStyle='#e0d6bd';x.lineWidth=2;
  for(const t of g){const px=Math.round(w*t/fin)+0.5;
    x.beginPath();x.moveTo(px,0);x.lineTo(px,h);x.stroke();}
}
function dessiner(c){
  const g=JSON.parse(c.dataset.grille), fin=parseFloat(c.dataset.fin);
  const cvE=c.querySelector('.env canvas'), env=JSON.parse(c.dataset.env);
  let w=cvE.width=Math.round(cvE.clientWidth*2), h=cvE.height=156;
  let x=cvE.getContext('2d');fond(x,w,h,g,fin);
  x.fillStyle='#b9a877';
  for(let i=0;i<env.length;i++){const bh=Math.max(2,env[i]*h*0.9);
    x.fillRect(i*w/env.length,h-bh,Math.max(1.5,w/env.length-0.5),bh);}

  const cvP=c.querySelector('.pistes canvas');
  const C=JSON.parse(c.dataset.courbes), noms=JSON.parse(c.dataset.noms);
  w=cvP.width=Math.round(cvP.clientWidth*2);h=cvP.height=300;
  x=cvP.getContext('2d');fond(x,w,h,g,fin);
  const rang=h/noms.length, etiq=[];
  noms.forEach((nc,r)=>{
    const v=C[nc[0]]||[], base=(r+1)*rang-4, haut=rang-9;
    x.strokeStyle='#d9cfb4';x.lineWidth=1;                 // la ligne de zéro
    x.beginPath();x.moveTo(0,base+0.5);x.lineTo(w,base+0.5);x.stroke();
    x.fillStyle=nc[1];
    for(let i=0;i<v.length;i++){const bh=Math.max(1,v[i]*haut);
      x.fillRect(i*w/v.length,base-bh,Math.max(1.5,w/v.length-0.3),bh);}
    etiq.push([nc[2], nc[1], base-haut+16]);
  });
  // Les noms par-dessus TOUTES les barres, sur un fond : dessinés dans la
  // boucle, la courbe suivante les recouvrait — « bruit ambiant » était gris
  // sur gris et illisible.
  x.font='600 17px system-ui';
  for(const [txt,coul,y] of etiq){
    const l=x.measureText(txt).width;
    x.fillStyle='rgba(255,253,247,0.82)';x.fillRect(4,y-14,l+8,19);
    x.fillStyle=coul;x.fillText(txt, 8, y);
  }
}
window.addEventListener('DOMContentLoaded',()=>{
  for(const c of document.querySelectorAll('.card')){
    dessiner(c);
    const fin=parseFloat(c.dataset.fin);
    for(const z of c.querySelectorAll('.env,.pistes'))
      z.addEventListener('pointerdown',e=>{
        const r=z.getBoundingClientRect();
        pose(c,(e.clientX-r.left)/r.width*fin);e.preventDefault();});
    pose(c, parseFloat(c.dataset.vrai));
    const inp=c.querySelector('input.pourquoi');
    if(P[c.dataset.stem]) inp.value=P[c.dataset.stem];
  }
  rendre();
  window.addEventListener('resize',()=>{
    for(const c of document.querySelectorAll('.card')) dessiner(c);});
});
"""


def page(songs: list[dict]) -> str:
    noms = json.dumps([[k, c, n] for k, n, c in COURBES])
    B = ["<h1>Là où la règle du début se trompe</h1>",
         "<div class=lede>La règle, c'est <b>la première note de basse</b>, "
         "calée sur la ligne de mesure la plus proche. Elle désigne la même "
         "mesure que toi sur 35 morceaux sur 42. Voici les "
         f"{len(songs)} autres.<br><br>"
         "<b style='color:#4a7c3f'>Vert</b> = ce que tu as dit. "
         "<b style='color:#7b4ea3'>Violet</b> = ce que la règle trouve. "
         "<b style='color:#8a2b2b'>Rouge</b> = la mesure 1 du traqueur.<br>"
         "Sous l'onde, les quatre courbes que je regarde, au même axe de temps. "
         "Une seule sert la règle aujourd'hui, la basse. Touche n'importe où "
         "pour déplacer le curseur, écoute, et dis-moi sous chaque morceau "
         "pourquoi on se trompe.</div>"]
    for s in songs:
        g = s["grille"]

        def repere(t, coul, texte, haut):
            pc = max(0.0, min(100.0, 100 * t / s["fin"]))
            cote = "right:4px" if pc > 62 else "left:4px"
            return (f"<div class=mk style=\"left:{pc:.3f}%;background:{coul}\">"
                    f"<b style='color:{coul};top:{haut}px;{cote}'>{texte}</b></div>")

        B.append(
            f"<div class=card data-stem=\"{html.escape(s['stem'])}\" "
            f"data-env='{json.dumps(s['env'])}' "
            f"data-courbes='{json.dumps(s['courbes'])}' data-noms='{noms}' "
            f"data-grille='{json.dumps(g)}' data-fin=\"{s['fin']}\" "
            f"data-vrai=\"{s['vrai']}\" data-choix=\"{s['vrai']}\">")
        B.append(f"<div class=tt>{html.escape(s['titre'])}</div>")
        sens = ("trop tard" if (s["ecart"] or 0) > 0 else "trop tôt")
        n = abs(s["ecart"] or 0)
        B.append(f"<div class=verdict>La règle dit <b style='color:#7b4ea3'>"
                 f"{s['regle']:.2f}s</b>, tu dis <b style='color:#4a7c3f'>"
                 f"{s['vrai']:.2f}s</b> — <b>{n} mesure{'s' if n > 1 else ''} "
                 f"{sens}</b>.</div>")
        B.append(f"<div class=note>mesure 1 du traqueur {s['traqueur']:.2f}s · "
                 f"premier son du fichier {s['son']:.2f}s</div>")
        B.append("<div class=env><canvas></canvas>")
        B.append(repere(g[0], "#8a2b2b", "traqueur", 2))
        if s["regle"] is not None and s["regle"] < s["fin"]:
            B.append(repere(s["regle"], "#7b4ea3", "la règle", 16))
        if s["vrai"] < s["fin"]:
            B.append(repere(s["vrai"], "#4a7c3f", "toi", 30))
        B.append("<div class=cur><b></b></div></div>")
        B.append("<div class=pistes><canvas></canvas>"
                 "<div class=cur></div></div>")
        B.append("<div class=leg>" + "".join(
            f"<span style='color:{c}'><i style='background:{c}'></i>{n}</span>"
            for _k, n, c in COURBES) + "</div>")
        B.append("<div class=row>"
                 f"<button class=play onclick=\"jouer('{s['audio']}',"
                 "ici(this.closest('.card')),8)\">▶ écouter d'ici</button>"
                 f"<button class=saut onclick=\"pose(this.closest('.card'),"
                 f"{s['vrai']:.2f})\">→ ton début</button>"
                 + (f"<button class=saut onclick=\"pose(this.closest('.card'),"
                    f"{s['regle']:.2f})\">→ la règle</button>"
                    if s["regle"] is not None else "")
                 + f"<button class=saut onclick=\"pose(this.closest('.card'),"
                 f"{g[0]:.2f})\">→ le traqueur</button>"
                 "<span class='note ou'></span></div>")
        B.append("<div class=row><input class=pourquoi oninput='noter(this)' "
                 "placeholder=\"pourquoi on se trompe ici ?\"></div>")
        B.append("</div>")
    B.append("<h2>Ce que tu m'expliques</h2>"
             "<textarea id=out readonly></textarea>"
             "<div class=row><button onclick=copier()>Copier le bloc</button></div>")
    return ("<!-- tools/debut_rates.py -->"
            "<meta charset=utf-8>"
            "<meta name=viewport content='width=device-width,initial-scale=1'>"
            "<title>Là où le début se trompe</title><style>" + CSS + "</style>"
            "<body>" + "".join(B)
            + "<audio id=au preload=auto playsinline></audio>"
            "<script>" + JS + "</script>")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                    default=SETTINGS.reports_dir / "debut_rates.html")
    a = ap.parse_args(argv)
    verite = json.loads((SETTINGS.repo / "state" / "human" / "debuts.json")
                        .read_text(encoding="utf-8"))["debuts"]
    songs = collecte(verite)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(page(songs), encoding="utf-8")
    print(f"→ {a.out}\n   {len(songs)} ratés sur {len(verite)} :")
    for s in songs:
        print(f"   {s['titre'][:38]:38s} toi {s['vrai']:7.2f}s · "
              f"règle {s['regle']:7.2f}s · {s['ecart']:+d} mesure(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
