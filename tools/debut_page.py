"""La page où Louis tranche le VRAI début de chaque morceau.

Louis, 2026-09-16, après Sam Smith (40 s d'intro de clip dans le fichier, une
mesure 1 posée une mesure trop tôt, et toute son annotation de sections décalée
derrière) : « fais moi une petite page html où je peux confirmer ou non quel
est le vrai début du morceau, ou le slider moi-même, afin que tu aies de quoi
arbitrer et inférer une règle ».

La question n'est pas « à quelle seconde », c'est **à quelle ligne de mesure**.
Le traqueur de battues pose une grille dont les lignes sont bonnes ; ce qu'il
ne sait pas, c'est laquelle est la première. La page montre donc l'énergie des
premières secondes avec les lignes de la grille dessus, et Louis désigne la
sienne — d'un doigt, ou au pas de mesure. Sa réponse est un nombre de mesures
de décalage, qui est exactement ce qu'une règle devra prédire.

Ma proposition (la ligne bleue) vient de `marche_energie` ci-dessous. Ce n'est
PAS une règle en place : aujourd'hui la mesure 1 est celle du traqueur, sans
aucune correction. C'est la candidate que ses arbitrages valideront ou non.

CE QUE LA PAGE NE FAIT PAS : écrire dans `state/human/marks/`. Elle ne touche
à rien — les verdicts vivent dans le navigateur et se recopient en bloc. Caler
une mesure 1 pour de vrai reste un geste à lui, dans l'app (Outils → « Caler la
mesure 1 »).

    python -m tools.debut_page
"""
from __future__ import annotations

import argparse
import html
import json
import subprocess
from pathlib import Path

import numpy as np

from harmonia.debut import cale_sur_grille, premier_son, premiere_basse
from harmonia.settings import SETTINGS

FENETRE = 90.0          # secondes d'audio analysées
PAS = 0.25              # résolution de l'enveloppe
LISSAGE = 8             # 2 s de moyenne glissante, pour l'affichage seulement
MARCHE_FEN = 4.0        # la fenêtre comparée avant/après une ligne de mesure
MARCHE_NBARS = 40       # on ne cherche un début que dans les 40 premières mesures


def enveloppe(audio: Path) -> tuple[list[float], list[float]] | None:
    """(brute, lissée) — l'énergie par tranches de 0,25 s, en mono 8 kHz."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(audio), "-t", str(FENETRE),
         "-ac", "1", "-ar", "8000", "-f", "f32le", "-"],
        capture_output=True).stdout
    if not raw:
        return None
    x = np.frombuffer(raw, dtype=np.float32)
    n = int(PAS * 8000)
    if len(x) < n * 4:
        return None
    r = np.array([float(np.sqrt(np.mean(x[i:i + n] ** 2)))
                  for i in range(0, len(x) - n, n)])
    liss = np.convolve(r, np.ones(LISSAGE) / LISSAGE, mode="same")
    return [float(v) for v in r], [float(v) for v in liss]


def marche_energie(brut: list[float], grille: list[float]) -> tuple[int, float] | None:
    """La ligne de mesure où l'énergie fait son plus gros pas VERS LE HAUT.

    Pour chaque ligne, le rapport entre les 4 secondes qui suivent et les 4
    secondes qui précèdent. Un morceau qui commence tout de suite a du silence
    devant sa première ligne : le rapport y explose (des centaines). Un morceau
    précédé de paroles, d'un fondu ou d'une intro de clip a un rapport modeste
    (Sam Smith : 4,0) mais net, à la ligne où le groupe entre. Un morceau sans
    marche du tout donne un rapport autour de 2, trouvé au milieu d'un refrain
    — c'est le cas à ne PAS suivre, et c'est pour fixer cette frontière que
    cette page existe.

    Rend `(index de mesure, rapport)`, ou None si la grille est vide.

    Ce que ça ne résout PAS : une grille dont les lignes sont elles-mêmes mal
    placées (mauvais tempo, mauvaise phase de battue). Aucune ligne n'est alors
    bonne, et le rapport ne le dit pas — c'est le troisième verdict de la page.
    """
    e = np.array(brut)
    n = len(e)

    def moy(a: float, b: float) -> float:
        i, j = max(0, int(a / PAS)), min(n, int(b / PAS))
        return float(e[i:j].mean()) if j > i else 0.0

    best, score = None, 0.0
    for k, t in enumerate(grille[:MARCHE_NBARS]):
        rapport = moy(t, t + MARCHE_FEN) / (moy(t - MARCHE_FEN, t) + 1e-4)
        if rapport > score:
            best, score = k, rapport
    return None if best is None else (best, score)


def _basse_de(stem: str, brut: list[float]) -> float | None:
    """La 1re note de basse du morceau — `harmonia.debut`, pas une copie.

    La règle, ses réglages et leur mesure vivent dans ce module ; la page ne
    fait que la montrer. C'est l'indice qui gagne sur les 42 réponses de
    Louis (35/42 contre 28/42 pour le traqueur seul).
    """
    from harmonia import musx as _musx
    audio = SETTINGS.audio_dir / f"{stem}.m4a"
    if not audio.exists():
        return None
    try:
        probs = _musx.frame_posteriors(audio)[1]
    except Exception:                                    # noqa: BLE001
        print(f"   (postérieures indisponibles pour {stem})")
        return None
    return premiere_basse(probs, apres=premier_son(brut))


def premier_accord(chart: dict) -> float | None:
    """Le début du premier accord que musx ne dit pas « silence ».

    Piste de Louis : « regarder si musx ou Beat This chope déjà tout seul le
    bon début ». C'est la réponse de musx, lue dans le chart déjà cuit.
    """
    for c in (chart.get("prompter") or {}).get("chords") or []:
        if not c.get("nc"):
            return float(c.get("t0", 0.0))
    return None


def _sur_la_grille(t: float | None, grille: list[float]) -> int | None:
    """L'index de la ligne de mesure la plus proche de `t`.

    POUR L'ANALYSE SEULEMENT — jamais pour dessiner. Convertir une détection
    en numéro de mesure est ce que je fais de mon côté pour comparer les
    indices entre eux ; un trait montré à Louis doit rester à l'instant
    détecté, sans quoi il montre une ligne de mesure et pas la note.
    """
    if t is None or not grille:
        return None
    return min(range(len(grille)), key=lambda i: abs(grille[i] - t))


def collecte() -> list[dict]:
    out, vus = [], set()
    for p in sorted(SETTINGS.charts_dir.glob("min_*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        stem = Path(d.get("audio_url") or "").stem or p.stem.removeprefix("min_")
        audio = SETTINGS.audio_dir / f"{stem}.m4a"
        grid = [float(t) for t in (d.get("barGrid") or [])]
        if not audio.exists() or not grid:
            continue
        env = enveloppe(audio)
        if env is None:
            continue
        brut, liss = env
        # Deux charts sur le même fichier (une pile, un doublon) : une seule
        # question à poser. La signature est l'audio, pas le nom du chart.
        sig = (round(audio.stat().st_size / 1024), round(grid[0], 2))
        if sig in vus:
            continue
        vus.add(sig)

        m = marche_energie(brut, [t for t in grid if t < FENETRE])
        # La fenêtre montrée doit contenir ma proposition, même quand elle
        # tombe loin : un « non, c'est n'importe quoi » à 78 s est exactement
        # ce qui fixera le plancher du rapport. La cacher serait me donner
        # raison d'avance.
        fin = min(FENETRE, max(30.0, grid[0] + 22.0,
                               (grid[m[0]] + 14.0) if m else 0.0))
        nfen = int(fin / PAS)
        mark = None
        mp = SETTINGS.marks_dir / f"{stem}.json"
        if mp.exists():
            try:
                mark = json.loads(mp.read_text(encoding="utf-8")).get("bar1")
            except (OSError, ValueError):
                mark = None
        crete = max(liss[:nfen]) or 1.0
        gr = [t for t in grid if t < fin]
        t_basse, sur_basse = cale_sur_grille(_basse_de(stem, brut), gr)
        out.append({
            "stem": stem, "cle": p.stem, "titre": d.get("title") or stem,
            # Le TEMPS BRUT de la détection, jamais ramené sur une ligne de
            # mesure. Louis, 2026-09-17 : « des fois tu poses le début basse
            # accord au mauvais endroit » — il avait raison, ces deux traits
            # étaient collés à la ligne la plus proche, donc jusqu'à une demi-
            # mesure à côté de la note qu'ils prétendaient montrer. Un trait
            # qui dit « la basse est ici » doit être là où elle est.
            **dict(zip(("basse", "basse_sur_ligne"),
                       (t_basse, sur_basse))),
            **dict(zip(("accord", "accord_sur_ligne"),
                       cale_sur_grille(premier_accord(d), gr))),
            "audio": d.get("audio_url") or f"/audio/{stem}.m4a",
            "env": [round(v / crete, 3) for v in liss[:nfen]],
            "pas": PAS, "fin": round(fin, 2),
            "grille": [round(t, 3) for t in grid if t < fin],
            "marque": None if mark is None else round(float(mark), 2),
            # LA PROPOSITION, c'est la règle qui gagne — la 1re note de basse
            # calée sur la grille (35/42 contre 28/42 pour le traqueur seul,
            # mesuré sur les 42 réponses de Louis). La marche d'énergie reste
            # affichée comme indice, mais elle ne propose plus rien : c'est
            # le plus mauvais des quatre (27/42).
            "propose": _sur_la_grille(t_basse, gr),
            "energie": None if m is None else m[0],
            "rapport": None if m is None else round(m[1], 1),
            "bpm": round(float((d.get("meta") or {}).get("bpm") or 0)) or None,
        })
    # Les cas où je propose de bouger d'abord, le plus net en tête ; puis les
    # cas où je ne propose rien, le moins évident en tête (c'est là qu'une
    # règle trop gourmande casserait ce qui marche).
    out.sort(key=lambda s: (0 if (s["propose"] or 0) > 0 else 1,
                            -(s["propose"] or 0)))
    return out


CSS = """
*{box-sizing:border-box}
body{margin:0 auto;padding:14px 12px 40px;background:#faf6ec;color:#2c2820;
 font:15px/1.45 system-ui,-apple-system,sans-serif;max-width:860px}
h1{font-size:19px;margin:0 0 8px}
h2{font-size:16px;margin:26px 0 6px}
.lede{background:#fff7df;border:1px solid #e8d9a8;border-radius:10px;
 padding:10px 12px;font-size:14px;margin:10px 0}
.note{color:#8a8371;font-size:12.5px}
.card{border:1px solid #ddd3b8;border-radius:12px;background:#fffdf7;
 padding:11px 13px;margin:16px 0}
.card.bouge{border-color:#c9a86a;background:#fffcf0}
.tt{font:600 15px system-ui}
.env{position:relative;height:88px;margin:9px 0 3px;background:#f3edda;
 border-radius:8px;overflow:hidden;cursor:crosshair;touch-action:manipulation}
.env canvas{display:block;width:100%;height:88px}
.mk{position:absolute;top:0;bottom:0;width:2px;pointer-events:none}
.mk b{position:absolute;left:3px;font:700 9.5px system-ui;white-space:nowrap;
 background:#fffdf7cc;padding:0 2px;border-radius:2px}
.cur{position:absolute;top:0;bottom:0;width:3px;background:#2c2820;pointer-events:none}
.cur b{position:absolute;bottom:3px;font:700 11px system-ui;color:#fff;
 background:#2c2820;padding:1px 4px;border-radius:3px;white-space:nowrap}
.row{display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin-top:7px}
button{border:1px solid #d8cfb4;border-radius:9px;background:#fff;padding:8px 11px;
 font:600 13px system-ui;cursor:pointer;min-height:42px;color:#2c2820}
button:active{background:#f0e9d5}
button.on{background:#2c2820;border-color:#2c2820;color:#fff}
button.play{background:#8a2b2b;border-color:#8a2b2b;color:#fff}
button.pas{min-width:52px;font-size:15px}
button.saut{font-size:12.5px;min-height:36px;padding:5px 9px}
button.caler{border-color:#c9a86a;background:#fdf3d8;font-size:12.5px;min-height:38px;padding:6px 10px}
button.caler.on{background:#4a7c3f;border-color:#4a7c3f;color:#fff}
button:disabled{opacity:.5}
.ou{color:#8a8371;font-size:12.5px;margin:0 2px}
textarea{width:100%;min-height:150px;font:12.5px/1.45 ui-monospace,monospace;
 border:1px solid #d8cfb4;border-radius:10px;padding:8px;background:#fff}
.compte{font:600 13px system-ui;color:#8a2b2b}
"""

JS = r"""
const au=document.getElementById('au');let stop=null;
au.addEventListener('timeupdate',()=>{if(stop!=null&&au.currentTime>=stop){au.pause();stop=null;}});
function jouer(src,t0,dur){
  // PILE au curseur, aucun élan. Louis, 2026-09-17 : « j'espère que t'as pas
  // mis de temps de latence quand on clique sur écouter ici qui fait que ça
  // joue avant le curseur, car sinon ça fausse tout ». Il avait raison : il y
  // avait 1,5 s d'élan, et caler à l'oreille poussait donc la marque 1,5 s
  // trop tard. Le bouton doit dire la vérité sur ce qu'il joue.
  const d0=Math.max(0,t0);
  const go=()=>{try{au.currentTime=d0;}catch(e){}stop=d0+(dur||7.5);au.play().catch(()=>{});};
  if(au.getAttribute('src')!==src){au.setAttribute('src',src);
    au.addEventListener('loadedmetadata',go,{once:true});au.load();}
  else go();
}
function ici(c){return parseFloat(c.dataset.choix)}
const KEY='harmonia_debut_v1';
let V={};try{V=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){V={}}

function lignes(c){return JSON.parse(c.dataset.grille)}
function proche(c,t){                       // la ligne de mesure la plus proche
  const g=lignes(c);let k=0;
  for(let i=1;i<g.length;i++) if(Math.abs(g[i]-t)<Math.abs(g[k]-t)) k=i;
  return {k:k, t:g[k], d:t-g[k]};
}
function pose(c,t,efface){                  // t = un TEMPS, libre
  const fin=parseFloat(c.dataset.fin);
  t=Math.max(0,Math.min(fin,t));
  c.dataset.choix=t.toFixed(2);
  const cur=c.querySelector('.cur'), pc=100*t/fin;
  cur.style.left=pc+'%';
  const et=cur.querySelector('b');
  et.textContent=t.toFixed(2)+'s';
  // Passé les deux tiers, l'étiquette déborderait : on la pose à gauche du
  // trait. `right` sur un trait de 3 px fait pousser le texte vers la gauche,
  // sans bouger le trait.
  if(pc>62){et.style.left='auto';et.style.right='5px';}
  else{et.style.right='auto';et.style.left='5px';}
  // Ce que je lis, moi : la ligne de mesure la plus proche et le décalage en
  // mesures. C'est la grandeur qu'une règle devra prédire — le clic de Louis
  // est libre, la conversion est mon travail.
  const pr=proche(c,t), dm=pr.k-parseInt(c.dataset.actuel,10);
  c.querySelector('.ecart').textContent =
    (Math.abs(pr.d)<0.35 ? 'pile sur la ligne de la mesure '+(pr.k+1)
                         : 'entre deux lignes, la plus proche est la mesure '+(pr.k+1))
    + (dm===0 ? ' — la mesure 1 actuelle'
              : ' — '+(dm>0?'+':'')+dm+' mesure'+(Math.abs(dm)>1?'s':''));
  if(efface) marque(c,null);
}
function pas(c,d){                          // ◀ ▶ : d'une ligne de mesure
  const g=lignes(c), pr=proche(c,ici(c));
  let k=pr.k+d;
  if(Math.abs(pr.d)>0.05) k=(d>0 && pr.d<0)||(d<0 && pr.d>0) ? pr.k : pr.k+d;
  pose(c,g[Math.max(0,Math.min(g.length-1,k))],true);
}
function marque(c,quoi){                    // quoi=null → on efface le verdict
  const st=c.dataset.stem;
  if(quoi===null){delete V[st];}
  else{const t=ici(c), pr=proche(c,t);
       V[st]={v:quoi,t:t,mes:pr.k+1,tmes:pr.t,
              ecart:pr.k-parseInt(c.dataset.actuel,10),sur:Math.abs(pr.d)<0.35};}
  for(const b of c.querySelectorAll('.vd button'))
    b.classList.toggle('on', quoi!==null && b.dataset.v===quoi);
  try{localStorage.setItem(KEY,JSON.stringify(V))}catch(e){}
  rendre();
}
function rendre(){
  // On liste TOUT ce que le navigateur a gardé, pas seulement ce qui trouve
  // une carte à l'écran, et on tolère l'ancien schéma (`verdict` au lieu de
  // `v`). Louis, 2026-09-17 : « y'a seulement les arbitrages où j'ai dit que
  // c'était bon que je peux coller, le reste tu ne les récupères pas ? ».
  // Le format du stockage a changé en cours de route sous la même clé : une
  // entrée de l'ancien format se lisait « undefined » ou pas du tout. Un
  // relevé qu'on ne peut pas recopier en entier ne vaut rien.
  const L=[];let n=0;
  for(const st of Object.keys(V)){
    const v=V[st]||{};const quoi=v.v||v.verdict||'?';n++;
    let l=st+' → '+quoi;
    const t=(typeof v.t==='number')?v.t:null;
    if(t!==null) l+=' : '+t.toFixed(2)+'s';
    if(typeof v.mes==='number')
      l+=' [mesure '+v.mes+(typeof v.tmes==='number'?' à '+v.tmes.toFixed(2)+'s':'')
        +(typeof v.ecart==='number'
          ? ', '+(v.ecart>0?'+':'')+v.ecart+' mesure'+(Math.abs(v.ecart)>1?'s':'') : '')
        +(v.sur===false?', pas sur une ligne':'')+']';
    L.push(l);
  }
  document.getElementById('out').value=L.sort().join('\n')
      ||'(aucun verdict pour l\'instant)';
  document.getElementById('n').textContent=n+' / '+document.querySelectorAll('.card').length;
}
function copierBrut(){
  // Le filet : le contenu EXACT du stockage, sans mise en forme ni tri, pour
  // que rien ne puisse se perdre dans ma façon de le relire.
  const t=document.getElementById('out');
  let brut='';try{brut=localStorage.getItem(KEY)||'';}catch(e){brut='(stockage illisible)';}
  t.value=brut||'(rien de stocké dans ce navigateur)';
  t.select();t.setSelectionRange(0,99999);presse(t.value);
}
function presse(txt){
  const replier=()=>{try{document.execCommand('copy');}catch(e){}};
  try{const pr=navigator.clipboard&&navigator.clipboard.writeText(txt);
      if(pr&&pr.catch) pr.catch(replier); else replier();}catch(e){replier();}
}
function dessiner(c){
  const cv=c.querySelector('canvas'), env=JSON.parse(c.dataset.env);
  const g=lignes(c), fin=parseFloat(c.dataset.fin);
  const w=cv.width=Math.round(cv.clientWidth*2), h=cv.height=176, x=cv.getContext('2d');
  x.fillStyle='#f3edda';x.fillRect(0,0,w,h);
  x.strokeStyle='#e0d6bd';x.lineWidth=2;                 // les lignes de mesure
  for(const t of g){const px=Math.round(w*t/fin)+0.5;
    x.beginPath();x.moveTo(px,0);x.lineTo(px,h);x.stroke();}
  x.fillStyle='#b9a877';                                  // l'énergie
  const n=env.length;
  for(let i=0;i<n;i++){const bh=Math.max(2,env[i]*h*0.9);
    x.fillRect(i*w/n,h-bh,Math.max(1.5,w/n-0.5),bh);}
}
function tempsDe(e,el,fin){
  const r=el.getBoundingClientRect();
  return (e.clientX-r.left)/r.width*fin;
}
window.addEventListener('DOMContentLoaded',()=>{
  for(const c of document.querySelectorAll('.card')){
    dessiner(c);
    const onde=c.querySelector('.env'), fin=parseFloat(c.dataset.fin);
    let tire=false;
    onde.addEventListener('pointerdown',e=>{
      tire=true;onde.setPointerCapture(e.pointerId);
      pose(c,tempsDe(e,onde,fin),true);e.preventDefault();});
    onde.addEventListener('pointermove',e=>{
      if(tire) pose(c,tempsDe(e,onde,fin),false);});
    for(const ev of ['pointerup','pointercancel'])
      onde.addEventListener(ev,()=>{tire=false;});
    const v=V[c.dataset.stem];
    pose(c, v ? v.t : parseFloat(c.dataset.depart), false);
    if(v) for(const b of c.querySelectorAll('.vd button'))
      b.classList.toggle('on',b.dataset.v===v.v);
  }
  rendre();
  window.addEventListener('resize',()=>{
    for(const c of document.querySelectorAll('.card')) dessiner(c);});
});
function caler(btn,cle){
  const c=btn.closest('.card'), t=ici(c), etat=c.querySelector('.etat');
  if(!confirm('Caler la mesure 1 de « '+c.querySelector('.tt').textContent
      +' » à '+t.toFixed(2)+'s ?\n\nÇa réécrit le chart du morceau.')) return;
  btn.disabled=true;etat.textContent='on refait le chart…';
  fetch('/api/bar1/'+encodeURIComponent(cle),{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify({t:t})})
   .then(r=>r.json()).then(d=>{
      if(!d.job_id){throw new Error(d.error||'refusé');}
      let n=0;
      const voir=()=>fetch('/api/job/'+d.job_id).then(r=>r.json()).then(j=>{
        if(j.status==='done'){          // jobs.py n'en rend que trois : running, done, error
          etat.textContent='calé à '+t.toFixed(2)+'s, chart refait';
          btn.textContent='⚑ calé';btn.classList.add('on');return;}
        if(j.status==='error'){throw new Error(j.error||'échec du chart');}
        if(++n>60){throw new Error('trop long, va voir dans l\'app');}
        setTimeout(voir,1500);});
      return voir();})
   // Pas de repli muet : si ça rate, ça se lit sur la carte, pas dans la
   // console. Le bouton redevient cliquable pour réessayer.
   .catch(e=>{etat.textContent='raté : '+e.message;btn.disabled=false;});
}
function copier(){rendre();const t=document.getElementById('out');
  t.select();t.setSelectionRange(0,99999);presse(t.value);}
"""


def page(songs: list[dict]) -> str:
    B = ["<h1>Le vrai début du morceau</h1>",
         "<div class=lede><b>Touche l'onde là où le morceau commence</b> "
         "(ou glisse le doigt), puis écoute pour vérifier. Le curseur va où tu "
         "le mets, il n'est collé à rien.<br>"
         "<b style='color:#8a2b2b'>Rouge</b> = la mesure 1 d'aujourd'hui. "
         "Les autres traits sont les trois pistes que tu m'as données : ""<b style='color:#2f5fa8'>l'énergie</b> qui s'installe, ""<b style='color:#7b4ea3'>la 1re note de basse</b>, ""<b style='color:#b06a1f'>le 1er accord</b> de musx. Les "
         "fins sont les lignes de mesure du traqueur : elles sont justes, il ne "
         "sait pas laquelle est la première.<br><b>Les « c'est bon » me servent "
         "autant que les « c'est décalé ».</b><br>Les verdicts ne changent rien au chart : ils sont là pour que j'en tire une règle. Le seul bouton qui écrit est « ⚑ caler la mesure 1 ici pour de vrai », en bas de chaque carte — il refait le chart du morceau.</div>"]
    bouge = sum(1 for s in songs if (s["propose"] or 0) > 0)
    B.append(f"<p class=note>{len(songs)} morceaux. Je proposerais de déplacer "
             f"la mesure 1 sur {bouge} d'entre eux — ils sont en tête. "
             "Répondu : <span class=compte id=n>0</span></p>")
    for s in songs:
        g = s["grille"]
        depart = g[s["propose"]] if (s["propose"] or 0) > 0 else g[0]
        cls = "card bouge" if (s["propose"] or 0) > 0 else "card"
        B.append(
            f"<div class=\"{cls}\" data-stem=\"{html.escape(s['stem'])}\" "
            f"data-env='{json.dumps(s['env'])}' data-grille='{json.dumps(g)}' "
            f"data-pas=\"{s['pas']}\" data-fin=\"{s['fin']}\" "
            f"data-actuel=\"0\" data-depart=\"{depart:.2f}\" "
            f"data-choix=\"{depart:.2f}\">")
        B.append(f"<div class=tt>{html.escape(s['titre'])}</div>")
        bits = [f"mesure 1 à {g[0]:.2f}s"]
        if s["bpm"]:
            bits.append(f"{s['bpm']} bpm")
        if s["marque"] is not None:
            bits.append(f"tu avais calé {s['marque']:.2f}s")
        if (s["propose"] or 0) > 0:
            bits.append(f"je proposerais la mesure {s['propose'] + 1}")
        hors = [nom for nom, ok, t in (("basse", s["basse_sur_ligne"], s["basse"]),
                                       ("accord", s["accord_sur_ligne"], s["accord"]))
                if t is not None and not ok]
        if hors:
            bits.append(" et ".join(hors)
                        + (" ne tombe" if len(hors) == 1 else " ne tombent")
                        + " sur aucune ligne")
        B.append(f"<div class=note>{' · '.join(bits)}</div>")
        def repere(t, couleur, texte, haut):
            pc = 100 * t / s["fin"]
            cote = "right:4px" if pc > 62 else "left:4px"
            return (f"<div class=mk style=\"left:{pc:.3f}%;background:{couleur}\">"
                    f"<b style='color:{couleur};top:{haut}px;{cote}'>{texte}</b></div>")

        B.append("<div class=env><canvas></canvas>")
        B.append(repere(g[0], "#8a2b2b", "mes.1", 2))
        if s["energie"]:
            B.append(repere(g[s["energie"]], "#2f5fa8", "énergie", 16))
        # `is not None`, pas la vérité booléenne : un temps de 0,0 s est une
        # détection valide et se dessinait pas du tout (14 cartes sur 45
        # portaient un trait basse alors que 45 en avaient un).
        if s["basse"] is not None and s["basse"] < s["fin"]:
            B.append(repere(s["basse"], "#7b4ea3", "basse", 30))
        if s["accord"] is not None and s["accord"] < s["fin"]:
            B.append(repere(s["accord"], "#b06a1f", "accord", 44))
        if s["marque"] is not None and s["marque"] < s["fin"]:
            B.append(repere(s["marque"], "#4a7c3f", "ta marque", 58))
        B.append("<div class=cur><b></b></div></div>")
        sauts = []
        for t_cand, nom, coul in (
                (g[s["energie"]] if s["energie"] else None,
                 "énergie", "#2f5fa8"),
                (s["basse"], "basse", "#7b4ea3"),
                (s["accord"], "accord", "#b06a1f")):
            if t_cand is not None and t_cand < s["fin"]:
                sauts.append(f"<button class=saut style=\"color:{coul};"
                             f"border-color:{coul}55\" onclick=\"pose("
                             f"this.closest('.card'),{t_cand:.2f},true)\">"
                             f"→ {nom}</button>")
        if sauts:
            B.append("<div class=row><span class=note>sauter à&nbsp;:</span>"
                     + "".join(sauts) + "</div>")
        B.append("<div class=row>"
                 "<button class=pas onclick=\"pas(this.closest('.card'),-1)\">◀</button>"
                 "<button class=pas onclick=\"pas(this.closest('.card'),1)\">▶</button>"
                 f"<button class=play onclick=\"jouer('{s['audio']}',"
                 "ici(this.closest('.card')),8)\">"
                 "▶ écouter à partir d'ici</button>"
                 f"<button onclick=\"jouer('{s['audio']}',{g[0]:.2f},7.5)\">"
                 "▶ la mesure 1 actuelle</button>"
                 "<span class='note ecart'></span></div>")
        st = html.escape(s["stem"])
        B.append("<div class='row vd'>"
                 f"<button data-v=bon onclick=\"marque(this.closest('.card'),'bon')\">"
                 "la mesure 1 est bonne</button>"
                 f"<button data-v=ailleurs onclick=\"marque(this.closest('.card'),'ailleurs')\">"
                 "le vrai début est au curseur</button>"
                 f"<button data-v='grille fausse' "
                 "onclick=\"marque(this.closest('.card'),'grille fausse')\">"
                 "aucune ligne ne tombe juste</button>"
                 f"<button data-v='sais pas' onclick=\"marque(this.closest('.card'),'sais pas')\">"
                 "je ne sais pas</button></div>")
        # Le seul bouton de la page qui ÉCRIT. Louis, 2026-09-17 : « c'est lui
        # qui définit où commence la chanson » — son clic n'a donc aucune raison
        # de faire un détour par l'app. Il refait le chart du morceau (caches
        # chauds, quelques secondes) et rien d'autre.
        B.append("<div class=row><button class=caler "
                 f"onclick=\"caler(this,'{html.escape(s['cle'])}')\">"
                 "⚑ caler la mesure 1 ici pour de vrai</button>"
                 "<span class='note etat'></span></div>")
        B.append("</div>")
    B.append("<h2>Tes arbitrages</h2>"
             "<textarea id=out readonly></textarea>"
             "<div class=row><button onclick=copier()>Copier le bloc</button>"
             "<button onclick=copierBrut()>Copier tout, brut</button>"
             "<button onclick=rendre()>Relire</button></div>"
             "<p class=note>Colle-le dans la conversation. Rien n'est envoyé "
             "d'ici ; tes réponses restent dans ce navigateur, tu peux t'y "
             "reprendre en plusieurs fois. « Copier tout, brut » vide le "
             "stockage tel quel, au cas où ma mise en forme laisserait "
             "tomber quelque chose.</p>")
    return ("<!-- tools/debut_page.py -->"
            "<meta charset=utf-8>"
            "<meta name=viewport content='width=device-width,initial-scale=1'>"
            "<title>Le vrai début du morceau</title><style>" + CSS + "</style>"
            "<body>" + "".join(B)
            + "<audio id=au preload=auto playsinline></audio>"
            "<script>" + JS + "</script>")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                    default=SETTINGS.reports_dir / "debut_morceaux.html")
    a = ap.parse_args(argv)
    songs = collecte()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(page(songs), encoding="utf-8")
    bouge = [s for s in songs if (s["propose"] or 0) > 0]
    print(f"→ {a.out}\n   {len(songs)} morceaux, {len(bouge)} où je proposerais "
          f"de déplacer la mesure 1 :")
    for s in bouge:
        print(f"   {s['titre'][:38]:38s} {s['grille'][0]:7.2f}s → "
              f"{s['grille'][s['propose']]:7.2f}s  "
              f"(+{s['propose']} mes., 1re note de basse)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
