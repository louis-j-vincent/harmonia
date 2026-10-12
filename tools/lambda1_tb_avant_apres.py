"""λ1 t+b en AVANT / APRÈS, sur le chart lui-même (Louis, 2026-10-12 :
« Je ne sais pas quel accord j'écoute avec ta page html. Fais que j'entende
juste l'accord joué, et refais la même interface qu'avant, avec les accords
qui se suivent en chart normal et avec la nouvelle version proposée. »).

    .venv/bin/python -m tools.lambda1_tb_avant_apres

Écrit docs/plots/lambda1_tb_avant_apres_2026-10-12.html (servie par
/plots/…). Même calcul que tools/lambda1_tb_autres_morceaux.py (le cœur
`core` de tools/demi_mesures_delta_decodeur.py, aucun réglage changé).

CE QUE « PROPOSÉ » VEUT DIRE. Le chart d'aujourd'hui, mesure par mesure dans
l'ordre de l'audio (déplié, via `soudure.accords_par_mesure`), où l'on ne
remplace QUE les demi-mesures où λ1 t+b change d'avis par rapport à λ0 (la
postérieure seule). Une demi-mesure où λ1 retombe sur le texte du chart ne
change pas à l'écran (comptée en note). Ce que la page ne montre PAS : les
endroits où λ0 et λ1 sont d'accord entre eux mais pas avec le chart (le
repli des sections a pu y réécrire l'accord) — ce n'est pas ce que λ1 apporte.

Rendu : glyphes réels de l'app (`glyphEpure`, kit.js, 27/19/15 px comme le
chart au téléphone) ; audio fetch()+blob, tête de lecture et accord allumé
pilotés par `timeupdate` (pas de boucle rAF — piège iPhone), arrêt précis à
t1 par un minuteur armé au démarrage + `timeupdate` en filet.
"""
from __future__ import annotations

import html
import json

from harmonia import soudure
from harmonia.settings import SETTINGS
from tools.demi_mesures_delta_decodeur import chart_name, core, lab_name
from tools.lambda1_tb_autres_morceaux import IV, ROOT, TAGS

OUT = SETTINGS.repo / "docs/plots/lambda1_tb_avant_apres_2026-10-12.html"
SONGS = [("CHVWPIFAT5A", "Evening Star"),
         ("ujgPfUZ7Whc", "La Jupe En Laine"),
         ("maroon_5_this_love", "This Love (Maroon 5)"),
         ("9zEl-FQLI4A", "San Francisco Street")]
SHOW_ALL_UP_TO = 48          # mesures : en dessous, tout le morceau est montré
#: Harte (sortie du décodeur) → notation de l'app (celle de chart.js / kit.js)
Q_APP = {"maj": "", "min": "-", "7": "7", "maj7": "^7", "min7": "-7", "dim": "o", "dim7": "o7",
         "hdim7": "h7", "aug": "+", "sus4": "sus4", "sus2": "sus2", "sus4(b7)": "7sus4", "9": "9",
         "maj9": "^9", "min9": "-9", "11": "11", "13": "13", "maj(6)": "6", "min(6)": "-6"}


def harte_to_app(lab: str) -> dict:
    if lab == "N":
        return dict(r=-1, q="", b=-1, nc=True, txt="N.C.")
    r, _, q = lab.partition(":")
    q, _, b = q.partition("/")
    root = ROOT[r]
    return dict(r=root, q=Q_APP[q], b=(root + IV[b]) % 12 if b else -1, nc=False, txt=lab_name(lab))


def app_chord(c: dict) -> dict:
    if c.get("nc"):
        return dict(r=-1, q="", b=-1, nc=True, txt="N.C.")
    b = int(c.get("bass", -1))
    return dict(r=int(c["root"]) % 12, q=c["q"], b=b if b >= 0 and b != int(c["root"]) else -1,
                nc=False, txt=chart_name(c).replace("h7", "ø7"))


def bar_items(chords, beats):
    """[(beat, accord)] → éléments avec leur durée réelle, temps de beatthis."""
    out = []
    for k, (beat, ch) in enumerate(chords):
        nxt = chords[k + 1][0] if k + 1 < len(chords) else 4
        out.append(dict(ch, beat=beat, t0=round(beats[beat], 3), t1=round(beats[nxt], 3)))
    return out


def song_data(stem: str, title: str) -> dict:
    c = core(stem, verbose=False)
    chart, D, H, n, beats_of = c["chart"], c["DEC"], c["H"], c["n"], c["beats_of"]
    bars = soudure.accords_par_mesure(chart)
    assert len(bars) == n, (stem, len(bars), n)

    changed = {i: D["l1tb"][i] for i in range(H) if D["l1tb"][i] != D["l0"][i]}
    today, prop, notes, rejoint = [], [], [], []
    for b in range(n):
        tb = [(min(3, int(x.get("beat", 0))), app_chord(x)) for x in bars[b]]
        if not tb or tb[0][0] != 0:                       # mesure sans accord au temps 1 : celui qui sonne
            prev = bars[b - 1][-1] if b and bars[b - 1] else None
            if prev is not None:
                tb.insert(0, (0, app_chord(prev)))
        today.append(bar_items(tb, beats_of[b]))
        halves = [2 * b, 2 * b + 1]
        if not any(i in changed for i in halves):
            prop.append([dict(x, ch=False) for x in today[-1]])
            continue
        items = []
        for half, i in enumerate(halves):
            lo, hi = 2 * half, 2 * half + 2
            if i in changed:
                new = harte_to_app(changed[i])
                show = new["txt"] != c["chart_h"][i].replace("h7", "ø7")
                items.append((lo, dict(new, ch=show)))
                d = dict(bar=b + 1, half=half, chart=c["chart_h"][i].replace("h7", "ø7"), new=new["txt"],
                         l0=lab_name(D["l0"][i]), t0=round(c["halves"][i][0], 3), t1=round(c["halves"][i][1], 3))
                cls, txt = TAGS.get((stem, i), ("q", ""))
                d.update(cls=cls, txt=txt)
                (notes if show else rejoint).append(d)
            else:
                part = [(bt, dict(ch_, ch=False)) for bt, ch_ in tb if lo <= bt < hi]
                if not part or part[0][0] != lo:
                    sounding = [ch_ for bt, ch_ in tb if bt < lo]
                    if sounding:
                        part.insert(0, (lo, dict(sounding[-1], ch=False)))
                items += part
        merged = []                                       # deux fois le même accord de suite = un seul
        for bt, it in items:
            if merged and merged[-1][1]["txt"] == it["txt"]:
                merged[-1][1]["ch"] = merged[-1][1]["ch"] or it["ch"]
                continue
            merged.append((bt, it))
        prop.append(bar_items(merged, beats_of[b]))

    # lignes de 4 mesures comme l'app ; seulement celles qui changent si le morceau est long
    nlines = (n + 3) // 4
    lines = list(range(nlines)) if n <= SHOW_ALL_UP_TO else sorted({(d["bar"] - 1) // 4 for d in notes})
    print(f"[{stem}] {len(changed)} demi-mesures où λ1 ≠ λ0 ; {len(notes)} visibles, "
          f"{len(rejoint)} retombent sur le chart ; lignes montrées {len(lines)}/{nlines}")
    for d in notes:
        print(f"   mes {d['bar']}{'ab'[d['half']]} chart={d['chart']} → {d['new']}  [{d['t0']}–{d['t1']}]")
    return dict(stem=stem, title=title, audio=chart["audio_url"], n=n, grid=[round(x, 3) for x in c["grid"]],
                today=today, prop=prop, lines=lines, all=n <= SHOW_ALL_UP_TO, notes=notes,
                rejoint=[f"mes. {d['bar']} ({'1re' if d['half'] == 0 else '2e'} moitié)" for d in rejoint])


CSS = """
:root{color-scheme:light;--bg:#f6f2e7;--fg:#222;--mut:#8a8371;--line:#c9c0a6;--line2:#e3dcc6;--acc:#8a2b2b;
 --new:#a35a00;--newbg:#fde7c2;--oldbg:#f6dcd6;--son:#cfe3ff;--head:#2a6fdb}
*{box-sizing:border-box} html,body{margin:0;overflow-x:hidden}
body{max-width:520px;margin:0 auto;padding:14px 12px 60px;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
h1{font-size:18px;margin:0 0 6px} h2{font-size:16px;margin:26px 0 4px;color:var(--acc)}
.lede{background:#fff7df;border:1px solid #e8d9a8;border-radius:10px;padding:9px 11px;font-size:13.5px;margin:8px 0}
.lede p{margin:4px 0} .n{color:var(--mut);font-size:12px}
.leg{display:inline-block;padding:0 5px;border-radius:4px}
.line{margin:12px 0 4px}
.lh{display:flex;align-items:center;gap:8px;margin-bottom:2px}
.lh .m{font:600 12px system-ui;color:var(--mut)}
.pl{margin-left:auto;font:600 12.5px system-ui;padding:5px 11px;border-radius:14px;border:1px solid var(--acc);background:#fff;color:var(--acc);cursor:pointer}
.pl.on{background:var(--acc);color:#fff}
.cap{font:600 10px system-ui;letter-spacing:.04em;text-transform:uppercase;color:var(--mut);margin:3px 0 1px}
.cap.p{color:var(--new)}
.row{position:relative;display:grid;grid-template-columns:repeat(4,minmax(0,1fr));border-left:1px solid var(--line)}
.row.p{background:#fffaf0}
.bar{position:relative;height:60px;padding-top:12px !important;border-right:1px solid var(--line);border-bottom:1px solid var(--line2);display:grid;grid-template-columns:repeat(4,minmax(0,1fr));align-items:center;padding:0 2px 0 4px;overflow:hidden}
.bar .no{position:absolute;z-index:2;top:1px;left:3px;font:10px system-ui;color:var(--mut)}
.ch{position:relative;height:44px;display:flex;align-items:center;border-radius:6px;cursor:pointer;padding:0 1px;min-width:0;-webkit-tap-highlight-color:transparent}
.ch .w{display:inline-block;padding-right:.25em;transform-origin:left center;white-space:nowrap}
.ch.new{background:var(--newbg);box-shadow:inset 0 0 0 1.5px var(--new)}
.ch.old{background:var(--oldbg)}
.ch.son{background:var(--son);box-shadow:inset 0 0 0 2px var(--head)}
.ch.tap{box-shadow:inset 0 0 0 2.5px var(--head)}
.ph{position:absolute;top:0;bottom:0;width:2px;background:var(--head);pointer-events:none;display:none;z-index:3}
.note{font-size:12.5px;margin:4px 0 0;padding-left:8px;border-left:3px solid var(--line)}
.note.vok{border-color:#1d8a4e} .note.vq{border-color:#b7791f} .note.vx{border-color:#c0392b}
.note b.ok{color:#1d8a4e} .note b.q{color:#b7791f} .note b.x{color:#c0392b}
details{font-size:12.5px;color:#4a463c} summary{cursor:pointer;color:var(--mut)}
.gap{text-align:center;color:var(--mut);font-size:12px;margin:6px 0}
.vu{position:sticky;bottom:0;margin-top:12px;background:#fff;border:1px solid var(--line);border-radius:10px;padding:6px 10px;font-size:12.5px;min-height:30px}
"""

JS = r"""
await import("/static/screens/chart.js");          // kit.js et chart.js s'importent l'un l'autre : chart.js d'abord
const { glyphEpure } = await import("/static/ui/kit.js");
const DATA = JSON.parse(document.getElementById("data").textContent);
const SZ=[0,27,19,15,15], INK="#1c1c1c", NEW="#7a4200";
const VERD={ok:"plausible",q:"à écouter",x:"artefact probable"};
const root=document.getElementById("songs");
const AU=[], ROWS=[];           // ROWS[k] = [{el, bar0, ph, chords:[{el,t0,t1}]}]
let cur=null;                   // {k, t1, btn, el}
const FIT=[];                   // mesures à plusieurs accords, comprimées après le chargement des polices
const vu=document.getElementById("vu");

function glyph(c,n,ink){
  if(c.nc){const s=document.createElement("span");s.textContent="N.C.";s.style.cssText=`font:italic ${SZ[n]-4}px Georgia,serif;color:${ink}`;return s;}
  return glyphEpure(c.r,c.q,SZ[Math.min(n,4)],"exact",ink,c.b);
}
function fmt(t){return t.toFixed(2)+" s";}

DATA.forEach((s,k)=>{
  const h=document.createElement("h2"); h.textContent=s.title; root.appendChild(h);
  const p=document.createElement("p"); p.className="n";
  p.textContent=(s.all?`Tout le morceau (${s.n} mesures).`:`${s.lines.length} lignes sur ${Math.ceil(s.n/4)} : seulement celles où quelque chose change.`)
    +` ${s.notes.length} demi-mesure${s.notes.length>1?"s":""} changent à l'écran.`
    +(s.rejoint.length?` Ailleurs λ1 change d'avis mais retombe sur le chart (rien ne bouge) : ${s.rejoint.join(", ")}.`:"");
  root.appendChild(p);
  const au=document.createElement("audio"); au.preload="auto"; au.setAttribute("playsinline",""); au.setAttribute("webkit-playsinline","");
  document.body.appendChild(au); AU.push(au); ROWS.push([]);
  fetch(s.audio).then(r=>r.blob()).then(b=>{au.src=URL.createObjectURL(b);}).catch(()=>{au.src=s.audio;});   // iPhone : fetch()+blob
  au.addEventListener("timeupdate",()=>tick(k));
  au.addEventListener("pause",()=>{ if(cur&&cur.k===k) finish(); });
  let prevL=null;
  s.lines.forEach(L=>{
    if(prevL!==null && L!==prevL+1){const g=document.createElement("div");g.className="gap";g.textContent="⋯";root.appendChild(g);}
    prevL=L;
    const b0=4*L, b1=Math.min(s.n,b0+4);
    const box=document.createElement("div"); box.className="line"; root.appendChild(box);
    const lh=document.createElement("div"); lh.className="lh";
    lh.innerHTML=`<span class="m">mes. ${b0+1}–${b1}</span>`;
    const pl=document.createElement("button"); pl.className="pl"; pl.textContent="▶ la ligne";
    pl.onclick=e=>{e.stopPropagation(); play(k,s.grid[b0],s.grid[b1],pl,null);};
    lh.appendChild(pl); box.appendChild(lh);
    const changedBars=new Set(s.notes.map(d=>d.bar-1));
    for(const ver of ["today","prop"]){
      const cap=document.createElement("div"); cap.className="cap"+(ver==="prop"?" p":"");
      cap.textContent= ver==="today"?"aujourd'hui":"proposé λ1 t+b"; box.appendChild(cap);
      const row=document.createElement("div"); row.className="row"+(ver==="prop"?" p":""); box.appendChild(row);
      const ph=document.createElement("div"); ph.className="ph"; row.appendChild(ph);
      const R={el:row,bar0:b0,bar1:b1,ph,chords:[]}; ROWS[k].push(R);
      for(let b=b0;b<b0+4;b++){
        const bar=document.createElement("div"); bar.className="bar"; row.appendChild(bar);
        if(b>=s.n) continue;
        const no=document.createElement("span"); no.className="no"; no.textContent=b+1; bar.appendChild(no);
        const items=s[ver][b], n=items.length;
        bar.style.gridTemplateColumns=`repeat(${Math.max(1,n)},minmax(0,1fr))`;
        // halves changées de cette mesure (pour marquer l'ancien accord en rose)
        const hs=s.notes.filter(d=>d.bar-1===b);
        const ws=[];
        items.forEach((c,j)=>{
          // placement de l'app (comme chart.js) : n accords = n cases égales, quel que soit le temps
          // où ils tombent ; la durée réelle sert à l'écoute, pas au dessin
          const el=document.createElement("div"); el.className="ch";
          if(ver==="prop"&&c.ch) el.classList.add("new");
          if(ver==="today"&&hs.some(d=>c.t0<d.t1-0.01&&c.t1>d.t0+0.01)) el.classList.add("old");
          const w=document.createElement("span"); w.className="w";
          w.appendChild(glyph(c,n,(ver==="prop"&&c.ch)?NEW:INK)); el.appendChild(w); bar.appendChild(el); ws.push(w);
          el.title=`${c.txt} · ${fmt(c.t0)} → ${fmt(c.t1)}`;
          el.onclick=e=>{e.stopPropagation(); play(k,c.t0,c.t1,null,el,c.txt);};
          R.chords.push({el,t0:c.t0,t1:c.t1});
        });
        if(ws.length>1) fitLater(bar,ws);
      }
    }
    s.notes.filter(d=>d.bar-1>=b0&&d.bar-1<b1).forEach(d=>{
      const nt=document.createElement("div"); nt.className="note v"+d.cls;
      nt.innerHTML=`mes. ${d.bar}, temps ${d.half?"3-4":"1-2"} : ${esc(d.chart.split(" → ").join(" puis "))} → <b style="color:${NEW}">${esc(d.new)}</b>`
        +(d.txt?` · <b class="${d.cls}">${VERD[d.cls]}</b>`:"");
      box.appendChild(nt);
      if(d.txt){const det=document.createElement("details");det.innerHTML=`<summary>pourquoi</summary>${esc(d.txt)}`;box.appendChild(det);}
    });
  });
});
function esc(t){const d=document.createElement("div");d.textContent=t;return d.innerHTML;}

// même règle que chart.js : plusieurs accords dans une mesure → on comprime la LARGEUR, jamais la hauteur
function fitLater(bar,ws){FIT.push([bar,ws]);}
function fitAll(){
  FIT.forEach(([bar,ws])=>{
    ws.forEach(w=>w.style.transform="");
    const right=bar.getBoundingClientRect().right-3; let k=1;
    ws.forEach((w,i)=>{const r=w.getBoundingClientRect();
      const lim=i+1<ws.length? ws[i+1].getBoundingClientRect().left-3 : right; if(r.width) k=Math.min(k,(lim-r.left)/r.width);});
    k=Math.max(.45,k); if(k<1) ws.forEach(w=>w.style.transform=`scaleX(${k.toFixed(3)})`);
  });
}
(document.fonts?document.fonts.ready:Promise.resolve()).then(()=>setTimeout(fitAll,0));
addEventListener("resize",()=>setTimeout(fitAll,50));

// ── lecture : exactement [t0, t1], puis stop ─────────────────────────────
let timer=null;
function play(k,t0,t1,btn,el,txt){
  const au=AU[k];
  if(cur && ((btn&&cur.btn===btn)||(el&&cur.el===el)) && !au.paused){au.pause();return;}
  AU.forEach((a,j)=>{ if(j!==k && !a.paused) a.pause(); });
  if(cur) finish();
  cur={k,t0,t1,btn,el};
  if(btn) btn.classList.add("on"); if(el) el.classList.add("tap");
  vu.textContent=(txt?`▶ ${txt}`:"▶ la ligne")+` — de ${fmt(t0)} à ${fmt(t1)}`;
  const go=()=>{
    try{au.currentTime=t0;}catch(e){}
    const pr=au.play(); if(pr&&pr.catch) pr.catch(()=>finish());
  };
  if(!au.src){ setTimeout(go,300); } else go();
}
function arm(k){                 // minuteur précis jusqu'à t1 (timeupdate ne tombe que ~4×/s)
  clearTimeout(timer); if(!cur||cur.k!==k) return;
  const au=AU[k], dt=(cur.t1-au.currentTime)/(au.playbackRate||1);
  if(dt<=0.005){ au.pause(); return; }
  timer=setTimeout(()=>arm(k), Math.max(5,Math.min(dt*1000,250)));
}
AU.forEach((au,k)=>au.addEventListener("playing",()=>arm(k)));
function finish(){
  clearTimeout(timer);
  if(!cur) return;
  const k=cur.k;
  if(cur.btn) cur.btn.classList.remove("on"); if(cur.el) cur.el.classList.remove("tap");
  if(window.__log) window.__log.push({t1:cur.t1,stop:AU[k].currentTime});
  vu.textContent=vu.textContent.replace("▶","■");
  cur=null;
  ROWS[k].forEach(R=>{R.ph.style.display="none"; R.chords.forEach(c=>c.el.classList.remove("son"));});
}
function tick(k){
  const au=AU[k], t=au.currentTime;
  if(cur&&cur.k===k&&t>=cur.t1-0.005){au.pause();return;}
  const g=DATA[k].grid;
  ROWS[k].forEach(R=>{
    const on=t>=g[R.bar0]&&t<g[R.bar1]&&!au.paused;
    R.chords.forEach(c=>c.el.classList.toggle("son",on&&t>=c.t0&&t<c.t1));
    if(!on){R.ph.style.display="none";return;}
    let b=R.bar0; while(b+1<R.bar1&&t>=g[b+1]) b++;
    const f=(t-g[b])/(g[b+1]-g[b]);
    R.ph.style.display="block"; R.ph.style.left=`${((b-R.bar0+f)/4*100).toFixed(2)}%`;
  });
}
"""


def write_page(songs):
    data = json.dumps(songs, ensure_ascii=False).replace("</", "<\\/")
    page = f"""<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>λ1 t+b, avant / après</title>
<style>{CSS}</style></head><body>
<h1>λ1 t+b : le chart d'aujourd'hui, et le chart proposé</h1>
<div class="lede">
<p><b>Touche un accord</b> (en haut ou en bas) : tu entends <b>exactement</b> sa durée, puis ça s'arrête. L'accord qui sonne s'allume en bleu dans les deux versions ; une barre bleue suit la musique.</p>
<p><b>▶ la ligne</b> joue les 4 mesures.</p>
<p>Ligne du haut : le chart tel que l'app l'écrit, mesure par mesure dans l'ordre de l'audio. Ligne du bas : le même, où λ1 t+b a remplacé un accord — <span class="leg" style="background:var(--newbg);color:var(--new)">en orange</span> le nouveau, <span class="leg" style="background:var(--oldbg)">en rose</span> celui qu'il remplace.</p>
</div>
<p class="n">λ1 t+b = le décodeur nourri, à chaque demi-mesure, de la postérieure musx plus sa différence avec les deux demi-mesures voisines (triade et basse seulement). Le « proposé » ne change que les demi-mesures où λ1 t+b change d'avis par rapport à la postérieure seule — c'est ce que λ1 apporte. Numéros de mesure comme dans l'app. Les explications « pourquoi » viennent de la page d'analyse du même jour (lecture du chroma brut) ; Evening Star n'en a pas.</p>
<div id="songs"></div>
<div class="vu" id="vu">Touche un accord.</div>
<p class="n">Script : tools/lambda1_tb_avant_apres.py (calcul : tools/demi_mesures_delta_decodeur.py, inchangé). Analyse détaillée : <a href="lambda1_tb_autres_morceaux_2026-10-12.html">λ1 t+b, autres morceaux</a>.</p>
<script id="data" type="application/json">{data}</script>
<script type="module">{JS}</script>
</body></html>"""
    OUT.write_text(page, encoding="utf-8")
    print("écrit", OUT)


def main():
    write_page([song_data(s, t) for s, t in SONGS])


if __name__ == "__main__":
    main()
