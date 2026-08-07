"""Raw bar-by-bar chart pages — no section folding (Louis, 2026-08-07).

« montre-moi le chart brut barre à barre, sers-toi juste de l'outil qui
détecte l'intro grâce à la matrice SSM voix pour commencer sur le bon temps »

Per song:
  * fresh harmonia_min analyze() under the NEW unrestricted-granularity
    default — chords taken from the PROMPTER capture (the beat-grid redecode
    BEFORE sections / folding / template rewrite touch anything);
  * bar 1 of the chart = the voice rule (scripts/melody_ssm.voice_start):
    demucs vocals → first sung onset (pitch MOVES: B8.sing_onset) → its bar,
    rounded UP onto the 2-bar grid; earlier bars are drawn dimmed as INTRO;
  * lead-sheet rendering: 4 bars per row, chords at their beat position,
    real audio + the current bar highlighted while it plays, click a bar to
    jump there. Light/dark, palette from the dataviz skill (validated).

Outputs docs/plots/chart_raw_<stem>.html.
Run: .venv/bin/python scratchpad/raw_chart_pages.py [stem ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scratchpad"))

SONGS = [
    ("let_it_be_remastered_2009", "Let It Be"),
    ("bein_green", "Bein Green"),
    ("ray_charles_georgia_on_my_mind_official_video", "Georgia On My Mind"),
]

SHARP = "C C# D D# E F F# G G# A A# B".split()
FLAT = "C Db D Eb E F Gb G Ab A Bb B".split()


def note_names(model):
    k = model.get("key") or {}
    tonic, mode = int(k.get("tonic", 0)), k.get("mode", "major")
    maj = tonic if mode == "major" else (tonic + 3) % 12
    return SHARP if maj in (7, 2, 9, 4, 11) else FLAT


def voice_form_start(stem: str, bar_grid, unit: int = 2):
    """First sung bar, rounded UP onto the `unit`-bar grid (melody_ssm rule)."""
    import vocal_anchor as VA
    import blocks8 as B8
    voc = VA.separate_vocals(REPO / f"docs/audio/{stem}.m4a")
    onset = B8.sing_onset(voc)[0]
    if onset is None:
        return 0, None
    b = int(np.searchsorted(np.asarray(bar_grid), onset + 1e-9) - 1)
    b = max(0, b)
    return min(len(bar_grid) - 2, b + (-b) % unit), float(onset)


def build_song(stem: str, title: str):
    from harmonia_min import pipeline as _pl
    model = _pl.analyze(REPO / f"docs/audio/{stem}.m4a", title=title,
                        file_key=f"raw_{stem}", audio_url=f"/audio/{stem}.m4a")
    grid = model["barGrid"]
    bt = np.asarray(model["beatTimes"], float)
    bpb = int(model["bpb"])
    step = float(np.median(np.diff(bt)))
    names = note_names(model)
    form_start, onset = voice_form_start(stem, grid)

    bars = [[] for _ in range(len(grid) - 1)]
    for c in model["prompter"]["chords"]:
        i = int(np.argmin(np.abs(bt - c["t0"])))
        tb = float(bt[i]) if abs(bt[i] - c["t0"]) < 0.35 * step else c["t0"]
        b = int(np.searchsorted(grid, tb + 1e-6) - 1)
        if not (0 <= b < len(bars)):
            continue
        beat = max(0, min(bpb - 1, int(round((tb - grid[b]) / step))))
        if c["nc"]:
            lab, bass = "N.C.", ""
        else:
            lab = names[c["root"]] + c["q"]
            bass = ("/" + names[c["bass"]]
                    if c["bass"] not in (-1, c["root"]) else "")
        bars[b].append({"beat": beat, "lab": lab + bass, "c": c["c"],
                        "t0": c["t0"], "t1": c["t1"], "nc": c["nc"]})
    # Louis, 2026-08-07 (« ça me rend un peu fou ») : TOUJOURS écrire le
    # premier accord d'une section — si la barre 1 d'une section commence
    # au-delà du temps 0, l'accord tenu venu d'avant y est écrit (estompé),
    # même règle que le `carry` du pipeline de l'app.
    for sec_start in {0, form_start}:
        cur = bars[sec_start]
        if cur and cur[0]["beat"] == 0:
            continue
        held = None
        for pb in range(sec_start - 1, -1, -1):
            if bars[pb]:
                held = bars[pb][-1]
                break
        if held is not None and not held["nc"]:
            cur.insert(0, {"beat": 0, "lab": held["lab"], "c": held["c"],
                           "t0": grid[sec_start], "t1": held["t1"],
                           "nc": False, "carry": True})
    return {
        "stem": stem, "title": title, "audio": f"../audio/{stem}.m4a",
        "keyName": model.get("keyName", "?"), "bpb": bpb,
        "bpm": round(float(model.get("meta", {}).get("bpm", 0))),
        "grid": [round(float(t), 3) for t in grid],
        "formStart": form_start, "singOnset": onset,
        "bars": bars,
    }


TEMPLATE = """<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__ — chart brut</title>
<style>
  :root { color-scheme: light dark; }
  .viz-root { color-scheme: light;
    --surface-1:#fcfcfb; --surface-2:#f0efec; --border:#d8d6d0;
    --ink:#0b0b0b; --ink-2:#52514e; --accent:#2a78d6; --play:#e34948;
    --mic:#1baf7a; }
  @media (prefers-color-scheme: dark) {
    :root:where(:not([data-theme="light"])) .viz-root { color-scheme: dark;
      --surface-1:#1a1a19; --surface-2:#242423; --border:#3a3936;
      --ink:#ffffff; --ink-2:#c3c2b7; --accent:#3987e5; --play:#e66767;
      --mic:#199e70; } }
  :root[data-theme="dark"] .viz-root { color-scheme: dark;
    --surface-1:#1a1a19; --surface-2:#242423; --border:#3a3936;
    --ink:#ffffff; --ink-2:#c3c2b7; --accent:#3987e5; --play:#e66767;
    --mic:#199e70; }
  body { margin:0; }
  .viz-root { background:var(--surface-1); color:var(--ink);
    font:15px/1.4 -apple-system,"Segoe UI",sans-serif; min-height:100vh;
    padding:18px; box-sizing:border-box; }
  h1 { font-size:20px; margin:0 0 2px; }
  .sub { color:var(--ink-2); margin:0 0 10px; }
  .bar-ctl { position:sticky; top:0; z-index:10; background:var(--surface-1);
    padding:8px 0; border-bottom:1px solid var(--border); margin-bottom:14px; }
  audio { height:32px; width:min(480px,100%); vertical-align:middle; }
  #sheet { display:grid; grid-template-columns:repeat(4, minmax(150px,1fr));
    gap:0; max-width:1080px; }
  @media (max-width:700px){ #sheet { grid-template-columns:repeat(2,1fr); } }
  .cell { border:1px solid var(--border); border-left-width:2px;
    min-height:58px; position:relative; padding:20px 4px 4px 6px;
    box-sizing:border-box; cursor:pointer; }
  .cell.intro { background:var(--surface-2); }
  .cell.intro .chords { opacity:.55; }
  .cell.now { outline:2.5px solid var(--play); outline-offset:-2px; }
  .bno { position:absolute; top:3px; left:6px; font-size:10px;
    color:var(--ink-2); }
  .tag { position:absolute; top:3px; right:6px; font-size:10px;
    color:var(--ink-2); letter-spacing:.05em; }
  .tag.mic { color:var(--mic); font-weight:700; }
  .chords { position:relative; height:30px; }
  .ch { position:absolute; bottom:0; font-weight:650; font-size:17px;
    white-space:nowrap; }
  .ch.lo { color:var(--ink-2); font-weight:500; }
  .ch.nc { color:var(--ink-2); font-weight:400; font-style:italic; }
  .legend { color:var(--ink-2); font-size:13px; margin:12px 0 0; max-width:70em; }
</style></head><body><div class="viz-root">
  <h1>__TITLE__ — chart brut (sans repliement)</h1>
  <p class="sub">__KEY__ · __BPM__ BPM · __BPB__ temps/barre · barre 1 posée
  par la règle de la voix (premier chant à __ONSET__, arrondi sur la grille de
  2 barres) · granularité libre (un accord par temps possible)</p>
  <div class="bar-ctl"><audio id="au" controls preload="metadata"
    src="__AUDIO__"></audio>
    <label style="color:var(--ink-2)"><input type="checkbox" id="follow"
    checked> suivre</label></div>
  <div id="sheet"></div>
  <p class="legend">Gris = intro (avant le premier chant). <b>Gras</b> =
  confiance ≥ 0.5, fin = en dessous. 🎤 = première barre chantée. Clique une
  barre pour y sauter ; la barre en cours d'écoute est encadrée en rouge.</p>
</div><script>
const D=__DATA__;
const au=document.getElementById('au');
const sheet=document.getElementById('sheet');
const cells=[];
D.bars.forEach((bar,i)=>{
  const d=document.createElement('div');
  const intro=i<D.formStart;
  d.className='cell'+(intro?' intro':'');
  const no=document.createElement('span'); no.className='bno';
  no.textContent=intro?'intro':String(i-D.formStart+1);
  d.appendChild(no);
  if(i===D.formStart){const m=document.createElement('span');
    m.className='tag mic'; m.textContent='🎤'; d.appendChild(m);}
  const ch=document.createElement('div'); ch.className='chords';
  bar.forEach(c=>{
    const s=document.createElement('span');
    s.className='ch'+(c.nc?' nc':(c.carry?' lo':(c.c<0.5?' lo':'')));
    s.style.left=(c.beat/D.bpb*100)+'%';
    s.textContent=c.nc?'N.C.':(c.carry?'('+c.lab+')':c.lab);
    s.title=c.lab+'  '+c.t0.toFixed(2)+'→'+c.t1.toFixed(2)+'s  conf '+c.c;
    ch.appendChild(s);
  });
  d.appendChild(ch);
  d.onclick=()=>{au.currentTime=D.grid[i]+0.01; au.play();};
  sheet.appendChild(d); cells.push(d);
});
let cur=-1;
(function loop(){
  const t=au.currentTime;
  let b=D.grid.findIndex((g,i)=>i+1<D.grid.length&&t>=g&&t<D.grid[i+1]);
  if(b!==cur){
    if(cur>=0&&cells[cur])cells[cur].classList.remove('now');
    if(b>=0&&cells[b]){cells[b].classList.add('now');
      if(document.getElementById('follow').checked&&!au.paused)
        cells[b].scrollIntoView({block:'center',behavior:'smooth'});}
    cur=b;}
  requestAnimationFrame(loop);
})();
</script></body></html>
"""


def fmt_t(t):
    return "—" if t is None else f"{int(t // 60)}:{int(t % 60):02d}"


def main():
    only = sys.argv[1:]
    for stem, title in SONGS:
        if only and stem not in only:
            continue
        d = build_song(stem, title)
        html = (TEMPLATE
                .replace("__TITLE__", title)
                .replace("__KEY__", d["keyName"])
                .replace("__BPM__", str(d["bpm"]))
                .replace("__BPB__", str(d["bpb"]))
                .replace("__ONSET__", fmt_t(d["singOnset"]))
                .replace("__AUDIO__", d["audio"])
                .replace("__DATA__", json.dumps(d, ensure_ascii=False)))
        out = REPO / "docs" / "plots" / f"chart_raw_{stem}.html"
        out.write_text(html, encoding="utf-8")
        print(f"wrote {out.relative_to(REPO)}  formStart(0-idx)={d['formStart']} "
              f"sing={fmt_t(d['singOnset'])} bars={len(d['bars'])}")


if __name__ == "__main__":
    main()
