"""Build /reports/beats_player.html — play any cached song WITH its detected
beats: sample-accurate clicks (WebAudio shared clock, downbeats accented),
waveform with the beats drawn on top, zoom window following playback.

Louis's ask (2026-08-05): « ce qu'il me faut absolument c'est pouvoir jouer le
morceau avec au dessus les beats détectés ». The whole point is judging the
tracker BY EAR, so playback goes through WebAudio (one clock for music and
clicks), not an <audio> element (whose currentTime is too coarse to trust).
Audio is fetched to a blob first — the iOS-PWA lesson (memory:
reference_ios_standalone_audio_bug).

Data: harmonia_min/state/beats/*.json ∩ docs/audio/*.m4a (23 songs), plus the
grid-guard numbers per song (consistency, phase instability) from the
grid_debug session so the suspect grids are labeled in the picker.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
BEATS = HERE / "harmonia_min" / "state" / "beats"
AUDIO = HERE / "docs" / "audio"
OUT = HERE / "harmonia_min" / "state" / "reports" / "beats_player.html"
DRIFT = Path("/private/tmp/claude-501/-Users-vincente-Documents-Projets-Perso-"
             "Code-harmonia/96b7806b-3278-401d-b520-1914f33ab075/scratchpad/"
             "drift_corpus.json")

NICE = {
    "michael_jackson_beat_it_official_4k_video": "Beat It",
    "bobby_hebb_sunny_official_audio": "Sunny",
    "maroon_5_this_love": "This Love",
    "norah_jones_don_t_know_why": "Don't Know Why",
    "carpenters_close_to_you": "Close to You",
    "ray_charles_georgia_on_my_mind_official_video": "Georgia On My Mind",
    "michael_jackson_billie_jean_official_video": "Billie Jean",
    "aretha_franklin_chain_of_fools_official_lyric_video": "Chain of Fools",
    "let_it_be_remastered_2009": "Let It Be",
    "muppets_kermit_its_not_easy_being_green_original": "Kermit — Bein' Green",
    "blue_bossa_150bpm_backing_track": "Blue Bossa 150 (backing)",
    "blue_bossa": "Blue Bossa",
    "yesterday_remastered_2009": "Yesterday",
    "katy_perry_hot_n_cold_official_music_video": "Hot N Cold",
    "nina_simone_feeling_good_lyric_video": "Feeling Good",
    "ben_e_king_stand_by_me_audio": "Stand By Me",
    "the_ronettes_be_my_baby_music_video": "Be My Baby",
    "the_police_every_breath_you_take_official_music_video": "Every Breath You Take",
    "maroon_5_she_will_be_loved_official_music_video": "She Will Be Loved",
    "bein_green": "Bein' Green",
    "rwc_rwc_p001": "RWC 001",
}

drift = {}
if DRIFT.exists():
    for r in json.loads(DRIFT.read_text()):
        drift[r["stem"]] = r

songs = {}
for p in sorted(BEATS.glob("*.json")):
    if not (AUDIO / f"{p.stem}.m4a").exists():
        continue
    d = json.loads(p.read_text())
    g = drift.get(p.stem, {})
    refused = bool((g.get("n_bars") or 0) >= 30 and
                   (g.get("metre") != 4 or (g.get("consistency") or 0) < 0.80))
    songs[p.stem] = {
        "t": NICE.get(p.stem, p.stem),
        "b": d["beats"], "d": d["downbeats"], "bpm": d["bpm"],
        "coh": g.get("consistency"), "ph": g.get("offmodal_downbeat_share"),
        "ref": refused,
    }

order = sorted(songs, key=lambda s: songs[s]["t"].lower())
DATA = json.dumps({"order": order, "songs": songs}, separators=(",", ":"))

html = """<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Les beats détectés, par-dessus le morceau</title><style>
body{margin:0;background:#e7e0d0;font:15px/1.55 -apple-system,system-ui,sans-serif;color:#1c1c1c}
.wrap{max-width:1100px;margin:0 auto;padding:22px 15px 60px}
h1{font:italic 600 25px Georgia,serif;margin:0 0 4px}
.lede{color:#8a8371;font-size:13px;margin-bottom:18px;max-width:860px}
section{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;padding:16px 18px;margin-bottom:16px}
canvas{width:100%;border-radius:8px;background:#f7f3e9;display:block;touch-action:none}
#zoom{height:150px;margin-bottom:10px}#full{height:64px}
.bar{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:10px 0 12px}
select,button{font:600 14px system-ui;padding:7px 12px;border-radius:9px;border:1px solid #e5dcc6;background:#fffdf6;color:#1c1c1c}
button{cursor:pointer;min-width:88px}button:active{background:#f7f3e9}
#play{background:#8a2b2b;color:#fff;border-color:#8a2b2b}
label{font:500 12.5px system-ui;color:#8a8371;display:flex;align-items:center;gap:5px}
input[type=range]{width:110px}
#time{font:600 13px ui-monospace,monospace;color:#1c1c1c;min-width:118px}
#status{font:500 12px system-ui;color:#8a8371}
.legend{font-size:12px;color:#8a8371;margin-top:8px}
.legend b{color:#8a2b2b}.legend i{color:#1c1c1c;font-style:normal;font-weight:600}
</style></head><body><div class=wrap>
<h1>Les beats détectés, par-dessus le morceau</h1>
<div class=lede>Le morceau et les clics passent par la même horloge audio : les clics
tombent exactement sur les temps que Beat This! a détectés — c'est le tracker que tu
juges à l'oreille, pas la latence du navigateur. Clic grave = beat, clic aigu =
downbeat (le « 1 »). Clique dans une bande pour naviguer, espace pour lecture/pause.</div>
<section>
<div class=bar>
 <select id=song></select>
 <button id=play disabled>▶ Lire</button>
 <span id=time>0:00.0 / 0:00</span>
 <span id=status>choisis un morceau…</span>
</div>
<div class=bar>
 <label><input type=checkbox id=ckb checked> clics beats</label>
 <label><input type=checkbox id=ckd checked> clics downbeats (aigus)</label>
 <label>volume clics <input type=range id=vol min=0 max=1 step=0.05 value=0.6></label>
 <label>fenêtre <select id=win><option value=4>4 s</option><option value=8 selected>8 s</option><option value=16>16 s</option></select></label>
</div>
<canvas id=zoom></canvas>
<canvas id=full></canvas>
<div class=legend>Bande du haut : la fenêtre qui suit la lecture — forme d'onde,
<i>traits courts</i> = beats, <b>traits pleins rouges</b> = downbeats (numérotés).
Bande du bas : tout le morceau, downbeats seuls. Dans le menu : cohérence de la
grille et instabilité de phase (session grid_debug) — ✗ = refusé par le garde-fou.</div>
</section>
<script>
const DATA=__DATA__;
const S=DATA.songs, sel=document.getElementById('song');
for(const k of DATA.order){
  const s=S[k], o=document.createElement('option');o.value=k;
  const bits=[`${Math.round(s.bpm)} bpm`];
  if(s.coh!=null)bits.push(`coh ${s.coh.toFixed(2)}`);
  if(s.ph!=null)bits.push(`phase ${s.ph.toFixed(2)}`);
  o.textContent=(s.ref?'✗ ':'')+s.t+' — '+bits.join(' · ');sel.appendChild(o);
}
const play=document.getElementById('play'),time=document.getElementById('time'),
 status=document.getElementById('status'),zoom=document.getElementById('zoom'),
 full=document.getElementById('full'),ckb=document.getElementById('ckb'),
 ckd=document.getElementById('ckd'),vol=document.getElementById('vol'),
 winSel=document.getElementById('win');
let ctx=null,buf=null,src=null,gain=null,clickGain=null;
let playing=false,startCtx=0,startOff=0,dur=0,env=null,ENV_DT=0.005;
let sched=new Set(),schedTimer=null,cur=null;

function ac(){if(!ctx){ctx=new (window.AudioContext||window.webkitAudioContext)();
 gain=ctx.createGain();gain.connect(ctx.destination);
 clickGain=ctx.createGain();clickGain.gain.value=+vol.value;clickGain.connect(ctx.destination);}return ctx;}
vol.oninput=()=>{if(clickGain)clickGain.gain.value=+vol.value;};

async function load(stem){
  stop();play.disabled=true;status.textContent='chargement…';cur=stem;
  try{
    const r=await fetch('/audio/'+stem+'.m4a');            // blob d'abord (leçon iOS)
    const ab=await r.arrayBuffer();
    buf=await ac().decodeAudioData(ab);
    dur=buf.duration;
    const ch=buf.getChannelData(0),n=Math.ceil(dur/ENV_DT),per=Math.floor(ch.length/n);
    env=new Float32Array(n);
    for(let i=0;i<n;i++){let m=0;const a=i*per,z=Math.min(ch.length,a+per);
      for(let j=a;j<z;j+=4){const v=Math.abs(ch[j]);if(v>m)m=v;}env[i]=m;}
    startOff=0;play.disabled=false;status.textContent=S[stem].t+' — prêt';
  }catch(e){status.textContent='échec du chargement : '+e;}
}
function pos(){return playing?Math.min(dur,startOff+ctx.currentTime-startCtx):startOff;}
function stop(){if(src){try{src.stop()}catch(e){}src=null;}playing=false;
 if(schedTimer){clearInterval(schedTimer);schedTimer=null;}sched.clear();
 play.textContent='▶ Lire';}
function begin(){if(!buf)return;ac().resume();
 src=ctx.createBufferSource();src.buffer=buf;src.connect(gain);
 startCtx=ctx.currentTime+0.05;src.start(startCtx,startOff);
 src.onended=()=>{if(playing&&pos()>=dur-0.05){stop();startOff=0;}};
 playing=true;play.textContent='⏸ Pause';
 schedTimer=setInterval(schedule,90);}
function toggle(){if(!buf)return;if(playing){startOff=pos();stop();}else begin();}
play.onclick=toggle;
document.addEventListener('keydown',e=>{if(e.code==='Space'){e.preventDefault();toggle();}});
function seek(t){const was=playing;if(playing){stop();}startOff=Math.max(0,Math.min(dur,t));
 sched.clear();if(was)begin();}
sel.onchange=()=>load(sel.value);

function blip(when,db){
  const o=ctx.createOscillator(),g=ctx.createGain();
  o.frequency.value=db?1568:784;g.gain.setValueAtTime(db?0.9:0.45,when);
  g.gain.exponentialRampToValueAtTime(1e-4,when+0.05);
  o.connect(g);g.connect(clickGain);o.start(when);o.stop(when+0.06);}
function events(s){                       // [t, estDownbeat] fusionnés, triés —
  if(s._ev)return s._ev;                  // un downbeat absent des beats garde
  const ev=s.d.map(t=>[t,true]);          // quand même son clic et son trait
  for(const bt of s.b){if(!s.d.some(dt=>Math.abs(dt-bt)<2e-3))ev.push([bt,false]);}
  return s._ev=ev.sort((a,b)=>a[0]-b[0]);}
function schedule(){
  if(!playing)return;const s=S[cur],t=pos(),horizon=t+0.55,ev=events(s);
  for(let i=0;i<ev.length;i++){const[bt,db]=ev[i];
    if(bt<t+0.02||bt>horizon||sched.has(i))continue;
    if(db?(!ckd.checked):(!ckb.checked))continue;
    blip(ctx.currentTime+(bt-t),db);sched.add(i);}
  for(const i of sched){if(ev[i][0]<t-2)sched.delete(i);}}

function sized(c){const r=c.getBoundingClientRect(),d=window.devicePixelRatio||1;
 if(c.width!==r.width*d){c.width=r.width*d;c.height=r.height*d;}
 return [c.getContext('2d'),c.width,c.height];}
function drawEnv(g,W,H,t0,t1,color){if(!env)return;
 g.strokeStyle=color;g.lineWidth=1;g.beginPath();
 for(let x=0;x<W;x++){const t=t0+(t1-t0)*x/W,i=Math.floor(t/ENV_DT);
  const v=(i>=0&&i<env.length)?env[i]:0;
  g.moveTo(x,H/2*(1-v*0.92));g.lineTo(x,H/2*(1+v*0.92));}g.stroke();}
function draw(){
  requestAnimationFrame(draw);if(!cur)return;const s=S[cur],t=pos();
  const W2=+winSel.value/2;
  { const[g,W,H]=sized(zoom);g.clearRect(0,0,W,H);
    const t0=t-W2,t1=t+W2;
    drawEnv(g,W,H,t0,t1,'#c9bfa5');
    g.font=`${11*(window.devicePixelRatio||1)}px system-ui`;
    for(const[bt,db]of events(s)){if(bt<t0||bt>t1)continue;
      const x=(bt-t0)/(t1-t0)*W;
      g.strokeStyle=db?'#8a2b2b':'#1c1c1c';g.lineWidth=db?2.5:1.2;
      g.beginPath();g.moveTo(x,db?0:H*0.28);g.lineTo(x,db?H:H*0.72);g.stroke();}
    g.fillStyle='#8a2b2b';
    for(let i=0;i<s.d.length;i++){const bt=s.d[i];
      if(bt>=t0&&bt<=t1)g.fillText(String(i+1),(bt-t0)/(t1-t0)*W+3,14*(window.devicePixelRatio||1));}
    g.strokeStyle='#1f8a5b';g.lineWidth=2.5;
    g.beginPath();g.moveTo(W/2,0);g.lineTo(W/2,H);g.stroke();}
  { const[g,W,H]=sized(full);g.clearRect(0,0,W,H);
    drawEnv(g,W,H,0,dur,'#c9bfa5');
    g.strokeStyle='#8a2b2b';g.lineWidth=1;
    for(const bt of s.d){const x=bt/dur*W;g.beginPath();g.moveTo(x,0);g.lineTo(x,H);g.stroke();}
    g.strokeStyle='#1f8a5b';g.lineWidth=2.5;const x=t/dur*W;
    g.beginPath();g.moveTo(x,0);g.lineTo(x,H);g.stroke();}
  const f=x=>{const m=Math.floor(x/60),s2=(x-60*m).toFixed(1);return m+':'+(s2<10?'0':'')+s2;};
  time.textContent=f(t)+' / '+f(dur||0);
}
function canvasSeek(c,ev,t0f,t1f){const r=c.getBoundingClientRect();
 const fx=(ev.clientX-r.left)/r.width;const[t0,t1]=[t0f(),t1f()];seek(t0+fx*(t1-t0));}
zoom.addEventListener('pointerdown',e=>canvasSeek(zoom,e,()=>pos()-(+winSel.value/2),()=>pos()+(+winSel.value/2)));
full.addEventListener('pointerdown',e=>canvasSeek(full,e,()=>0,()=>dur));
draw();
sel.value=DATA.order.includes('michael_jackson_beat_it_official_4k_video')?
 'michael_jackson_beat_it_official_4k_video':DATA.order[0];
load(sel.value);
</script></div></body></html>"""

OUT.write_text(html.replace("__DATA__", DATA), encoding="utf-8")
print(f"écrit {OUT} ({len(html) // 1024 + len(DATA) // 1024} ko)")
