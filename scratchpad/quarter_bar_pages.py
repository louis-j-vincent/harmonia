"""Generate the 4 listening-diagnostic pages (feat/quarter-bar, 2026-08-07).

    docs/plots/diag_quarter_let_it_be.html   demi-barre vs quart, F récupérés
    docs/plots/diag_quarter_bein_green.html  walkdown que PERSONNE n'attrape
    docs/plots/diag_grid_blue_bossa.html     phase des downbeats tracker vs GT
    docs/plots/diag_grid_georgia_on_my_mind.html  octave métrique (bpb 2 vs 4)

Each page: real-audio <audio> (../audio/*.m4a), chord lanes (GT / decode arms),
beat + downbeat tick rows (detected vs GT), a moving playhead that follows the
music, click-to-seek, zoom, points of interest. Colors follow the dataviz skill
palette (validated 3-slot light+dark): blue = detected grid, orange = GT,
aqua = quarter-only chords, red = playhead (utility, not a series).

Data: golden/brick0 GT, harmonia_min/state/beats, scratchpad/quarter_bar_pred.
Run: .venv/bin/python scratchpad/quarter_bar_pages.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BRICK0 = REPO / "golden" / "brick0"
BEATS = REPO / "harmonia_min" / "state" / "beats"
PRED = REPO / "scratchpad" / "quarter_bar_pred"
OUT = REPO / "docs" / "plots"


def song_data(song_id: str, arms: list[str]):
    gt = json.loads((BRICK0 / f"{song_id}.gt.json").read_text())
    stem = Path(gt["audio_path"]).stem
    bd = json.loads((BEATS / f"{stem}.json").read_text())
    lanes = []
    lanes.append({
        "name": "GT" + ("" if gt["verified"] else " (non vérifié)"),
        "kind": "gt",
        "chords": [{"t0": c["t0"], "t1": c["t1"], "label": c["label"]}
                   for c in gt["gt_chords"]],
    })
    for arm, disp in arms:
        d = json.loads((PRED / arm / f"{song_id}.json").read_text())
        lanes.append({"name": disp, "kind": arm,
                      "chords": [{"t0": c["start_s"], "t1": c["end_s"],
                                  "label": c["label"]}
                                 for c in d["chords"] if c["label"] != "N"]})
    # mark quarter-only blocks (present in the last arm, absent from the first)
    if len(lanes) == 3:
        base = {(round(c["t0"], 1), c["label"]) for c in lanes[1]["chords"]}
        for c in lanes[2]["chords"]:
            if (round(c["t0"], 1), c["label"]) not in base:
                c["new"] = True
    return {
        "songId": song_id,
        "audio": f"../audio/{stem}.m4a",
        "beats": [round(t, 3) for t in bd["beats"]],
        "downbeats": [round(t, 3) for t in bd["downbeats"]],
        "gtDownbeats": [round(t, 3) for t in gt.get("downbeat_times", [])],
        "lanes": lanes,
    }


PAGES = [
    {
        "file": "diag_quarter_let_it_be.html",
        "title": "Let It Be — demi-barre vs quart de barre",
        "blurb": ("La piste QUART autorise un changement d'accord sur chaque "
                  "temps (pénalité forte). Les blocs <b>verts</b> n'existent "
                  "que dans QUART : trois F d'un seul temps — le « whisper "
                  "words of wisdom » — que la demi-barre avale. Le GT iReal "
                  "n'est PAS vérifié : indicatif seulement."),
        "song": "let_it_be",
        "arms": [("none", "DEMI-BARRE (livré)"), ("all", "QUART de barre")],
        "pois": [(109.0, "F récupéré (1:49)"), (116.0, "F récupéré (1:56)"),
                 (234.8, "F récupéré (3:55)"), (62.0, "frontière C avancée")],
    },
    {
        "file": "diag_quarter_bein_green.html",
        "title": "Bein Green — le walkdown que personne n'attrape",
        "blurb": ("GT vérifié main. À 37.6–40 s le GT descend "
                  "G♯/D♯ → F♯/C♯ → F:7/C (un accord par temps) ; les deux "
                  "décodages entendent un seul accord de 3 s — même porte "
                  "quart-de-barre grande ouverte. Le goulot n'est pas la "
                  "grille : les postérieurs musx lissent ces renversements. "
                  "Deuxième walkdown identique à 2:20."),
        "song": "bein_green",
        "arms": [("none", "DEMI-BARRE (livré)"), ("all", "QUART de barre")],
        "pois": [(36.5, "walkdown 1 (0:37)"), (139.5, "walkdown 2 (2:20)"),
                 (157.5, "F7sus4→F7 (2:38)")],
    },
    {
        "file": "diag_grid_blue_bossa.html",
        "title": "Blue Bossa — la phase des downbeats est fausse",
        "blurb": ("Ticks <b>bleus</b> = les « 1 » du tracker (Beat This!) ; "
                  "ticks <b>oranges</b> = les « 1 » du GT vérifié main. "
                  "Écoute en suivant les ticks : le « 1 » que tu entends "
                  "tombe sur l'orange, le bleu est ailleurs — et l'écart "
                  "change au fil du morceau (dérive). Le décodage ne peut "
                  "poser ses accords QUE sur la grille bleue : root 0.58, le "
                  "pire des 7, sans aucun rapport avec la granularité."),
        "song": "blue_bossa",
        "arms": [("none", "DÉCODÉ (demi-barre, livré)")],
        "pois": [(15.0, "début (0:15)"), (120.0, "milieu (2:00)"),
                 (209.0, "turnaround (3:29)"), (420.0, "fin (7:00)")],
    },
    {
        "file": "diag_grid_georgia_on_my_mind.html",
        "title": "Georgia — l'octave métrique : des barres deux fois trop courtes",
        "blurb": ("Le tracker pose un « 1 » <b>bleu</b> tous les 2 temps "
                  "(1.9 s) ; le GT vérifié n'en a qu'un tous les 4 temps "
                  "(3.8 s, orange). Un tick bleu sur deux est un faux "
                  "downbeat : compte « 1-2-3-4 » avec la chanson et regarde "
                  "quel tick tombe sur ton « 1 ». Conséquence : la « barre » "
                  "du décodeur est une demi-barre réelle, et tout ce qui "
                  "compte en barres (sections, repli, demi-barre elle-même) "
                  "travaille à la mauvaise échelle."),
        "song": "georgia_on_my_mind",
        "arms": [("none", "DÉCODÉ (demi-barre, livré)")],
        "pois": [(10.0, "compte avec l'intro (0:10)"),
                 (60.0, "couplet (1:00)"), (100.0, "pont (1:40)")],
    },
]

TEMPLATE = """<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { color-scheme: light dark; }
  .viz-root {
    color-scheme: light;
    --surface-1:#fcfcfb; --surface-2:#f0efec; --border:#d8d6d0;
    --text-primary:#0b0b0b; --text-secondary:#52514e;
    --det:#2a78d6; --gt:#eb6834; --new:#1baf7a; --play:#e34948;
    --block:#e7e5e0; --block-ink:#0b0b0b;
  }
  @media (prefers-color-scheme: dark) {
    :root:where(:not([data-theme="light"])) .viz-root {
      color-scheme: dark;
      --surface-1:#1a1a19; --surface-2:#242423; --border:#3a3936;
      --text-primary:#ffffff; --text-secondary:#c3c2b7;
      --det:#3987e5; --gt:#d95926; --new:#199e70; --play:#e66767;
      --block:#2e2e2c; --block-ink:#ffffff;
    }
  }
  :root[data-theme="dark"] .viz-root {
    color-scheme: dark;
    --surface-1:#1a1a19; --surface-2:#242423; --border:#3a3936;
    --text-primary:#ffffff; --text-secondary:#c3c2b7;
    --det:#3987e5; --gt:#d95926; --new:#199e70; --play:#e66767;
    --block:#2e2e2c; --block-ink:#ffffff;
  }
  body { margin:0; }
  .viz-root { background:var(--surface-1); color:var(--text-primary);
    font:14px/1.45 -apple-system, "Segoe UI", sans-serif; min-height:100vh;
    padding:16px; box-sizing:border-box; }
  h1 { font-size:19px; margin:0 0 6px; }
  .blurb { color:var(--text-secondary); max-width:72em; margin:0 0 12px; }
  .bar { display:flex; flex-wrap:wrap; gap:10px; align-items:center;
    position:sticky; top:0; background:var(--surface-1); padding:8px 0;
    z-index:30; border-bottom:1px solid var(--border); }
  audio { height:32px; max-width:min(420px, 100%); }
  .bar button { background:var(--surface-2); color:var(--text-primary);
    border:1px solid var(--border); border-radius:6px; padding:4px 10px;
    cursor:pointer; font:inherit; }
  .bar button:hover { border-color:var(--text-secondary); }
  label.follow { color:var(--text-secondary); user-select:none; }
  .legend { display:flex; gap:16px; flex-wrap:wrap; margin:10px 0 6px;
    color:var(--text-secondary); }
  .legend span { display:inline-flex; align-items:center; gap:6px; }
  .chip { width:12px; height:12px; border-radius:3px; display:inline-block; }
  #tl { overflow-x:auto; border:1px solid var(--border); border-radius:8px;
    background:var(--surface-1); }
  #canvas { position:relative; }
  .lane { position:absolute; left:0; right:0; }
  .lane-name { position:sticky; left:6px; top:2px; font-size:11px;
    color:var(--text-secondary); z-index:5; display:inline-block;
    background:color-mix(in srgb, var(--surface-1) 82%, transparent);
    padding:1px 5px; border-radius:4px; }
  .blk { position:absolute; height:34px; background:var(--block);
    color:var(--block-ink); border-radius:4px; font-size:12px;
    overflow:hidden; white-space:nowrap; padding:8px 0 0 5px;
    box-sizing:border-box; box-shadow: inset 0 0 0 1px var(--surface-1); }
  .blk.gt { box-shadow: inset 0 0 0 1.5px var(--gt); background:transparent; }
  .blk.new { background:var(--new); color:#fff; }
  .tick { position:absolute; width:1px; }
  .tick.beat { background:var(--border); }
  .tick.db { width:2px; background:var(--det); }
  .tick.gtdb { width:2px; background:var(--gt); }
  #playhead { position:absolute; top:0; bottom:0; width:2px;
    background:var(--play); z-index:20; pointer-events:none; }
  #tip { position:fixed; display:none; background:var(--surface-2);
    color:var(--text-primary); border:1px solid var(--border);
    border-radius:6px; padding:5px 9px; font-size:12px; z-index:50;
    pointer-events:none; max-width:300px; }
  details { margin-top:14px; color:var(--text-secondary); }
  table { border-collapse:collapse; font-size:12px; }
  td, th { border:1px solid var(--border); padding:2px 8px; text-align:left; }
  .axis { position:absolute; font-size:10px; color:var(--text-secondary); }
</style></head><body>
<div class="viz-root">
  <h1>__TITLE__</h1>
  <p class="blurb">__BLURB__</p>
  <div class="bar">
    <audio id="au" controls preload="metadata" src="__AUDIO__"></audio>
    <label class="follow"><input type="checkbox" id="follow" checked> suivre</label>
    <button id="zin">zoom +</button><button id="zout">zoom −</button>
    __POIS__
  </div>
  <div class="legend">
    <span><span class="chip" style="background:var(--det)"></span>downbeats détectés (tracker)</span>
    <span><span class="chip" style="background:var(--gt)"></span>downbeats GT (vérifiés main)</span>
    <span><span class="chip" style="background:var(--border)"></span>temps détectés</span>
    __LEGEND_EXTRA__
  </div>
  <div id="tl"><div id="canvas"><div id="playhead"></div></div></div>
  <p class="blurb" style="margin-top:8px">Clique dans la timeline pour te
  déplacer ; la ligne rouge suit l'écoute. Survole un accord pour ses temps
  exacts. Audio introuvable&nbsp;? La page attend <code>__AUDIO__</code>
  relatif à <code>docs/plots/</code>.</p>
  <details><summary>table des accords (accessibilité)</summary>
    <div id="tbl"></div></details>
  <div id="tip"></div>
</div>
<script>
const D = __DATA__;
let pps = __PPS__;                       // pixels par seconde
const LANE_H = 40, TICK_H = 26, TOP = 8;
const au = document.getElementById('au');
const tl = document.getElementById('tl');
const cv = document.getElementById('canvas');
const tip = document.getElementById('tip');
const ph = document.getElementById('playhead');
const dur = Math.max(D.beats[D.beats.length-1]||0,
  ...D.lanes.map(l => l.chords.length ? l.chords[l.chords.length-1].t1 : 0)) + 4;

function render() {
  cv.querySelectorAll('.lane,.tick,.axis').forEach(e => e.remove());
  const W = Math.ceil(dur * pps);
  const nl = D.lanes.length;
  cv.style.width = W + 'px';
  cv.style.height = (TOP + nl * LANE_H + TICK_H + 18) + 'px';
  D.lanes.forEach((ln, i) => {
    const lane = document.createElement('div');
    lane.className = 'lane';
    lane.style.top = (TOP + i * LANE_H) + 'px';
    lane.style.height = LANE_H + 'px';
    const nm = document.createElement('span');
    nm.className = 'lane-name'; nm.textContent = ln.name;
    lane.appendChild(nm);
    ln.chords.forEach(c => {
      const b = document.createElement('div');
      b.className = 'blk' + (ln.kind === 'gt' ? ' gt' : '') + (c.new ? ' new' : '');
      b.style.left = (c.t0 * pps) + 'px';
      b.style.width = Math.max(2, (c.t1 - c.t0) * pps - 1) + 'px';
      b.style.top = '14px';
      b.textContent = c.label;
      b.dataset.info = c.label + '  ' + c.t0.toFixed(2) + ' → ' +
        c.t1.toFixed(2) + ' s  (' + (c.t1 - c.t0).toFixed(2) + ' s)' +
        (c.new ? '  — QUART seulement' : '');
      lane.appendChild(b);
    });
    cv.appendChild(lane);
  });
  const ty = TOP + nl * LANE_H;
  const mk = (t, cls, h, y) => {
    const e = document.createElement('div');
    e.className = 'tick ' + cls;
    e.style.left = (t * pps) + 'px';
    e.style.top = y + 'px'; e.style.height = h + 'px';
    cv.appendChild(e);
  };
  if (pps > 18) D.beats.forEach(t => mk(t, 'beat', 10, ty + 12));
  D.downbeats.forEach(t => mk(t, 'db', 12, ty));
  D.gtDownbeats.forEach(t => mk(t, 'gtdb', 12, ty + 12));
  for (let s = 0; s <= dur; s += (pps < 12 ? 30 : 10)) {
    const a = document.createElement('div');
    a.className = 'axis'; a.textContent = fmt(s);
    a.style.left = (s * pps + 3) + 'px'; a.style.top = (ty + TICK_H) + 'px';
    cv.appendChild(a);
  }
}
const fmt = s => Math.floor(s/60) + ':' + String(Math.floor(s%60)).padStart(2,'0');
render();

cv.addEventListener('click', e => {
  const r = cv.getBoundingClientRect();
  au.currentTime = (e.clientX - r.left) / pps;
});
cv.addEventListener('mouseover', e => {
  if (!e.target.dataset.info) return;
  tip.textContent = e.target.dataset.info; tip.style.display = 'block';
});
cv.addEventListener('mousemove', e => {
  tip.style.left = Math.min(e.clientX + 12, innerWidth - 310) + 'px';
  tip.style.top = (e.clientY + 14) + 'px';
});
cv.addEventListener('mouseout', () => tip.style.display = 'none');
document.getElementById('zin').onclick = () => { pps *= 1.5; render(); };
document.getElementById('zout').onclick = () => { pps /= 1.5; render(); };
document.querySelectorAll('[data-seek]').forEach(b =>
  b.onclick = () => { au.currentTime = +b.dataset.seek; au.play(); });
(function loop() {
  const x = au.currentTime * pps;
  ph.style.transform = 'translateX(' + x + 'px)';
  if (document.getElementById('follow').checked && !au.paused)
    tl.scrollLeft = x - tl.clientWidth / 2;
  requestAnimationFrame(loop);
})();
au.addEventListener('error', () => {
  document.querySelector('.blurb').insertAdjacentHTML('beforeend',
    ' <b style="color:var(--play)">⚠ audio introuvable — ouvre cette page '
    + 'depuis docs/plots/ (l’audio est cherché en ../audio/).</b>');
});
const tbl = document.getElementById('tbl');
tbl.innerHTML = D.lanes.map(l => '<h3>' + l.name + '</h3><table><tr>'
  + '<th>début</th><th>fin</th><th>accord</th></tr>'
  + l.chords.map(c => '<tr><td>' + c.t0.toFixed(2) + '</td><td>'
     + c.t1.toFixed(2) + '</td><td>' + c.label
     + (c.new ? ' (quart seulement)' : '') + '</td></tr>').join('')
  + '</table>').join('');
</script></body></html>
"""


def build(page):
    d = song_data(page["song"], page["arms"])
    pois = "".join(
        f'<button data-seek="{t}">{lab}</button>' for t, lab in page["pois"])
    extra = ('<span><span class="chip" style="background:var(--new)"></span>'
             'accord présent seulement en QUART</span>'
             if len(page["arms"]) == 2 else "")
    html = (TEMPLATE
            .replace("__TITLE__", page["title"])
            .replace("__BLURB__", page["blurb"])
            .replace("__AUDIO__", d["audio"])
            .replace("__POIS__", pois)
            .replace("__LEGEND_EXTRA__", extra)
            .replace("__PPS__", "40")
            .replace("__DATA__", json.dumps(d, ensure_ascii=False)))
    out = OUT / page["file"]
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out.relative_to(REPO)}  "
          f"({len(html) // 1024} kB, {len(d['lanes'])} lanes)")


if __name__ == "__main__":
    for p in PAGES:
        build(p)
