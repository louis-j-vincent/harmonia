"""docs/plots/folds.html — un fold, trois folds, cinq folds, mesure par mesure.

La question de Louis : « pk 5 réseaux musx ? ». Les chiffres disent qu'un fold
seul change 26 à 41 % des mesures ; ils ne disent pas SI c'est pire. Cette page
met les trois versions l'une sous l'autre, chaque mesure cliquable, pour qu'il
tranche à l'oreille.
"""
import html, json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0,"/Users/vincente/Documents/Projets Perso/Code/harmonia")
from harmonia_min import musx as M, beats as B
from harmonia_min.labels import to_chord

SHIFT=18; SPEC=252
dev=M._device()
with M._InMusxDir():
    from mir.nn.train import NetworkInterface
    from chordnet_ismir_naive import ChordNet
    import chordnet_ismir_naive as _cn
    M._patch_init_hidden(_cn)
    NETS=[NetworkInterface(ChordNet(None), n, load_checkpoint=False).net.eval().to(dev)
          for n in M.MODEL_NAMES]

N=["C","D♭","D","E♭","E","F","G♭","G","A♭","A","B♭","B"]
# `to_chord` rend deja la qualite dans la notation iReal de l'app (C-7, E^7,
# Dh7, Bo) : on la garde telle quelle, c'est celle que Louis lit dans le chart.
def nom(lab):
    ch=to_chord(lab)
    if ch is None: return "N.C."
    s=f"{N[ch['root']]}{ch['q']}"
    if ch.get("bass",-1)>=0 and ch["bass"]!=ch["root"]: s+=f"/{N[ch['bass']]}"
    return s

def version(cqt, bd, bpb, idx):
    x=torch.tensor(np.asarray(cqt),dtype=torch.float32).to(dev)
    acc=None
    t=time.time()
    with torch.no_grad():
        for i in idx:
            o=NETS[i].inference(x)
            acc=list(o) if acc is None else [a+b for a,b in zip(acc,o)]
    pr=[(a/len(idx)).astype(np.float32) for a in acc]; dt=time.time()-t
    lab,_=M.redecode(bd["beats"],pr,downbeat_times=bd["downbeats"],
                     beats_per_bar=bpb,quarter_beats="all")
    bt=np.asarray(bd["beats"])
    bars={}
    for t0,t1,l in lab:
        bi=int(np.abs(bt-t0).argmin())
        bars.setdefault(bi//bpb,[]).append(nom(l))
    nb=(max(bars)+1) if bars else 0
    return {"t":dt, "bars":[" ".join(bars.get(b,["%"])) for b in range(nb)]}

songs=[]
for stem, titre in [("maroon_5_this_love","Maroon 5 — This Love"),
                    ("ben_e_king_stand_by_me_audio","Ben E. King — Stand By Me"),
                    ("autumn_leaves","Autumn Leaves")]:
    p=Path("docs/audio")/f"{stem}.m4a"
    cqt=M.song_cqt(p); bd=B.track(p)
    bt=np.asarray(bd["beats"])
    bpb=int(round(np.median(np.diff(bd["downbeats"]))/np.median(np.diff(bt)))) or 4
    v5=version(cqt,bd,bpb,[0,1,2,3,4])
    v3=version(cqt,bd,bpb,[0,1,2])
    v1=version(cqt,bd,bpb,[0])
    nb=min(len(v5["bars"]),len(v3["bars"]),len(v1["bars"]))
    temps=[float(bt[min(b*bpb,len(bt)-1)]) for b in range(nb)]
    songs.append({"stem":stem,"titre":titre,"bpb":bpb,"nb":nb,"temps":temps,
                  "v5":v5,"v3":v3,"v1":v1})
    print(f"{stem}: {nb} mesures, 5f {v5['t']:.2f}s / 3f {v3['t']:.2f}s / 1f {v1['t']:.2f}s")

def cellules(s, key, ref):
    out=[]
    for b in range(s["nb"]):
        txt=s[key]["bars"][b]; diff = txt!=s["v5"]["bars"][b] and key!="v5"
        out.append(f'<div class="m{" d" if diff else ""}" data-t="{s["temps"][b]:.2f}">'
                   f'<span class="n">{b+1}</span>{html.escape(txt)}</div>')
    return "".join(out)

blocs=[]
for s in songs:
    def pct(key):
        n=s["nb"]; same=sum(1 for b in range(n) if s[key]["bars"][b]==s["v5"]["bars"][b])
        return round(100*same/max(n,1))
    blocs.append(f"""
<section>
  <h2>{html.escape(s['titre'])}</h2>
  <audio controls preload="none" src="/audio/{s['stem']}.m4a"></audio>
  <p class="lead">{s['nb']} mesures. Clique une mesure pour t'y placer dans l'audio.
     Les mesures <b class="dd">colorées</b> sont celles où la version diffère des 5 folds.</p>
  <h3>5 folds <small>l'ensemble d'aujourd'hui — {s['v5']['t']:.2f} s</small></h3>
  <div class="g">{cellules(s,'v5',None)}</div>
  <h3>3 folds <small>{pct('v3')} % des mesures identiques — {s['v3']['t']:.2f} s</small></h3>
  <div class="g">{cellules(s,'v3',s['v5'])}</div>
  <h3>1 fold <small>{pct('v1')} % des mesures identiques — {s['v1']['t']:.2f} s</small></h3>
  <div class="g">{cellules(s,'v1',s['v5'])}</div>
</section>""")

page=f"""<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Un fold ou cinq ?</title><style>
:root{{--pap:#f7f3ea;--ink:#1c1c1c;--f:#8a7f6d;--l:#ded5c4;--acc:#8a2b2b;--d:#f2dcc4}}
body{{margin:0;padding:18px 14px 60px;background:var(--pap);color:var(--ink);
 font:15px/1.55 -apple-system,system-ui,sans-serif;max-width:900px;margin:auto}}
h1{{font:700 22px/1.3 Georgia,serif;margin:0 0 4px}}
h2{{font:700 18px/1.3 Georgia,serif;margin:34px 0 8px;border-top:1px solid var(--l);padding-top:18px}}
h3{{font:600 13px/1.4 -apple-system,sans-serif;margin:16px 0 6px;text-transform:uppercase;
 letter-spacing:.06em;color:var(--f)}}
h3 small{{text-transform:none;letter-spacing:0;font-weight:500}}
p.lead{{color:var(--f);font-size:13.5px;margin:6px 0 10px}}
audio{{width:100%;margin:2px 0 6px}}
.g{{display:grid;grid-template-columns:repeat(4,1fr);gap:3px}}
.m{{position:relative;background:#fff;border:1px solid var(--l);border-radius:6px;
 padding:12px 4px 7px;text-align:center;font:600 15px Georgia,serif;cursor:pointer;
 min-height:20px}}
.m .n{{position:absolute;top:2px;left:4px;font:500 8px -apple-system,sans-serif;color:#c3b8a4}}
.m.d{{background:var(--d);border-color:#e0bf95}}
.m:active{{outline:2px solid var(--acc)}}
b.dd{{background:var(--d);padding:1px 5px;border-radius:4px;font-weight:600}}
.intro{{background:#fff;border:1px solid var(--l);border-radius:10px;padding:14px 16px;margin:14px 0}}
.intro b{{color:var(--acc)}}
@media(min-width:620px){{.g{{grid-template-columns:repeat(8,1fr)}}}}
</style></head><body>
<h1>Un fold ou cinq ?</h1>
<div class="intro">
<p>musx est livré en <b>5 réseaux</b> : la même architecture entraînée cinq fois sur
cinq découpages du jeu d'entraînement. On moyenne leurs sorties depuis le début,
sans jamais avoir vérifié ce que ça achète.</p>
<p>Un fold seul coûte <b>5× moins cher</b>. Sur les trois morceaux ci-dessous il
change <b>1 mesure sur 3</b> (This Love, Autumn Leaves) ou <b>1 sur 10</b>
(Stand By Me). Les chiffres ne disent pas si c'est <i>pire</i> — d'où cette page :
écoute les mesures colorées et dis-moi si l'ensemble mérite son prix.</p>
</div>
{"".join(blocs)}
<script>
document.querySelectorAll('section').forEach(sec=>{{
  const a=sec.querySelector('audio');
  sec.querySelectorAll('.m').forEach(m=>m.addEventListener('click',()=>{{
    a.currentTime=parseFloat(m.dataset.t)||0; a.play();
  }}));
}});
</script></body></html>"""
Path("docs/plots/folds.html").write_text(page, encoding="utf-8")
print("→ docs/plots/folds.html")
