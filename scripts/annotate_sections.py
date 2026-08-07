"""L'outil pour que Louis annote lui-même les sections. Sa vérité, pas la nôtre.

    python scripts/annotate_sections.py [<stem> ...]  ->  /reports/annotate.html

Louis, 2026-08-07 :

  « Crée-moi un outil d'annotation facile de sections — j'ai juste à cliquer sur
    le bouton intro, A, B, C, D, outro, bridge, queue de section, et j'annote la
    chanson en cliquant sur la mesure de début et de fin de la partie que je
    décide de mettre, ça snap to grid à la double barre la plus proche. Comme ça
    je peux te donner des vrais ground truths des sections comme je les vois moi.
    Mets les découpages melody_check en suggestion pour que j'aie une base de
    départ, avec option "copy this", et après je peux drag les limites des
    sections, les supprimer… quelque chose de user friendly. »

POURQUOI ÇA PASSE AVANT LE RESTE. Les trois chantiers ouverts — le seuil des
double-mesures, le seuil du score des pics, et le score global à maximiser qu'il
décrit ensuite — se règlent tous contre une vérité qu'on n'a pas. Aujourd'hui
elle tient en quatre lignes dans `docs/section_truth.json`, dont une qu'il a dû
rétracter. Régler des seuils sans elle, c'est deviner.

CE QUE L'OUTIL FAIT.

  * Un ruban par morceau, gradué en mesures, avec la lecture audio et un curseur.
  * Une palette de libellés ; on tire à la souris ou au doigt sur le ruban pour
    poser une section. Tout **aimante sur la double-mesure** la plus proche.
  * Les bords se tirent, le milieu se déplace, la croix supprime. Un appui long
    ou le bouton ▶ joue la section.
  * La proposition de `melody_check` est dessinée au-dessus, en sourdine, avec un
    bouton pour la recopier et partir de là.

LA SAUVEGARDE. `/api/annotations/<file>` ne sait porter que des accords — il
jette tout sauf `chords` et `merges` — donc une route à part a été ajoutée au
serveur : `POST /api/sections/<stem>`, un fichier JSON par morceau dans
`harmonia_min/state/sections/`. L'outil y écrit à chaque geste et relit au
chargement, donc l'annotation survit à un rafraîchissement et à un changement
d'appareil. Le navigateur garde une copie de secours au cas où le serveur ne
répondrait pas, et les boutons copier/télécharger restent là pour sortir le tout
d'un bloc.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
sys.path.insert(0, str(HERE / "scratchpad"))
from pattern_lanes import load, INK                                      # noqa: E402
import vocal_anchor as VA                                                # noqa: E402
import blocks8 as B8                                                     # noqa: E402
import blocks_flex as BF                                                 # noqa: E402
import melody_ssm as MS                                                  # noqa: E402
import vocal_melody as VM                                                # noqa: E402
import melody_check as MC                                                # noqa: E402
import channels as CN                                                    # noqa: E402

DEFAULT = B8.DEFAULT
UNIT = 2           # « ça snap to grid à la double barre la plus proche »

LABELS = [("intro", "#8a8371"), ("A", "#b3261e"), ("B", "#1f8a5b"),
          ("C", "#2a6fb0"), ("D", "#c58a2e"), ("E", "#7c3aed"),
          ("bridge", "#0f766e"), ("queue", "#a0522d"), ("outro", "#6b7280")]


def suggestion(stem):
    """Ce que `melody_check` propose — la base de départ, pas la vérité."""
    S, n, grid = load(stem)
    voc = VA.separate_vocals(HERE / f"docs/audio/{stem}.m4a")
    onset, *_ = B8.sing_onset(voc)
    vstart = MS.voice_start(VA.bar_of(grid, onset), n) or 0
    tt, ff, vv, rr = VM.track_f0(voc)
    notes, _ = VM.melody_notes(tt, ff, vv, rr)
    notes, _ = VM.clean(notes)
    M, mute = MS.melody_bars(notes, grid, n)
    vs = CN.voices(S, M, n, BF.HEAD, mute)
    H = BF.head_matrix(S, n)
    b = BF.best_tiling(S, H, n, vstart, vs)
    secs = BF.to_sections(n, vstart, b[1], b[2], b[3], b[4])
    out = []
    for s in secs:
        lab = ("intro" if s["kind"] == "intro"
               else "outro" if s["kind"] == "reste"
               else s["letter"].rstrip("′"))
        out.append({"label": lab, "b0": s["b0"], "b1": s["b1"]})
    return n, grid, out


def main():
    stems = sys.argv[1:] or DEFAULT
    songs = []
    for st in stems:
        if not (HERE / f"docs/audio/{st}.m4a").exists():
            print(f"  !! {st} introuvable")
            continue
        try:
            n, grid, sug = suggestion(st)
            songs.append({"stem": st, "n": n,
                          "grid": [round(float(x), 3) for x in grid],
                          "sug": sug})
            print(f"  ok {st}  ({n} mesures, {len(sug)} sections suggérées)")
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"  !! {st} — {type(exc).__name__}: {exc}")

    data = json.dumps(songs).replace("</", "<\\/")
    labels = json.dumps(LABELS)
    pal = "".join(
        f'<button class=lab data-l="{l}" style="--c:{c}">{l}</button>'
        for l, c in LABELS)
    cards = "".join(f"""
<section data-stem="{s['stem']}" data-n="{s['n']}">
  <h2>{s['stem'].replace('_',' ').title()}
      <span class=sub>{s['n']} mesures</span></h2>
  <div class=sugwrap>
    <div class=sugrow><span class=tag>suggestion</span><div class=sug></div></div>
    <button class=copy>↓ partir de cette base</button>
    <button class=clear>tout effacer</button>
  </div>
  <div class=rulerwrap>
    <div class=ruler></div>
    <div class=scrub><span class=knob></span></div>
    <div class=track></div>
    <div class=play></div>
  </div>
  <div class=bar>
    <button class=pp>▶</button><span class=pos>mes. 1 · 0:00</span>
    <span class=hint>bande grise = défilement · ruban = sections</span>
    <button class=ok>✓ valider ce morceau</button>
  </div>
  <ul class=list></ul>
</section>""" for s in songs)

    out = HERE / "harmonia_min/state/reports/annotate.html"
    out.write_text(f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Annoter les sections</title><style>
*{{box-sizing:border-box;-webkit-tap-highlight-color:transparent}}
body{{margin:0;background:#e7e0d0;font:16px/1.5 -apple-system,system-ui,sans-serif;
  color:{INK};-webkit-user-select:none;user-select:none}}
.wrap{{max-width:1150px;margin:0 auto;padding:16px 12px calc(120px + env(safe-area-inset-bottom))}}
h1{{font:italic 600 24px Georgia,serif;margin:0 0 3px}}
.lede{{color:#6f6858;font-size:13.5px;margin-bottom:14px}} .lede b{{color:{INK}}}
section{{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;
  padding:12px 12px 10px;margin-bottom:14px}}
h2{{font:700 17px system-ui;margin:0 0 8px;color:#8a2b2b}}
.sub{{font:500 12px system-ui;color:#8a8371;margin-left:8px}}
.sugwrap{{display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin-bottom:8px}}
.sugrow{{display:flex;align-items:center;gap:6px;flex:1;min-width:220px}}
.tag{{font:600 9.5px system-ui;color:#a89f8c;text-transform:uppercase;flex:none}}
.sug{{position:relative;height:16px;flex:1;background:#f2ece0;border-radius:4px;
  overflow:hidden}}
.sug i{{position:absolute;top:0;bottom:0;opacity:.45;border-right:1px solid #fff;
  font:700 8.5px ui-monospace,monospace;color:#fff;text-align:center;
  line-height:16px;font-style:normal}}
.copy,.clear{{border:1px solid #d8cfb4;background:#f7f3e9;border-radius:7px;
  padding:5px 9px;font:600 11.5px system-ui;cursor:pointer;flex:none}}
.clear{{color:#8a2b2b}}
.rulerwrap{{position:relative;margin:2px 0 8px;touch-action:pan-y}}
.ruler{{position:relative;height:15px;border-bottom:1px solid #e0d8c4}}
.ruler b{{position:absolute;top:0;font:600 9px ui-monospace,monospace;
  color:#a89f8c;font-weight:600;transform:translateX(-50%)}}
.ruler u{{position:absolute;bottom:0;width:1px;height:5px;background:#ded5bf}}
.scrub{{position:relative;height:22px;background:#ece5d5;border-radius:5px;
  margin-bottom:3px;cursor:ew-resize;touch-action:none;
  background-image:repeating-linear-gradient(90deg,transparent 0,transparent calc(var(--w) * 4 - 1px),#ddd4bd calc(var(--w) * 4 - 1px),#ddd4bd calc(var(--w) * 4))}}
.knob{{position:absolute;top:1px;bottom:1px;width:10px;margin-left:-5px;left:0;
  background:#1c1c1c;border-radius:3px;box-shadow:0 0 0 2px #fffdf6}}
.track{{position:relative;height:54px;background:
  repeating-linear-gradient(90deg,#f7f3e9 0,#f7f3e9 var(--w),#efe8d8 var(--w),#efe8d8 calc(2*var(--w)));
  border-radius:6px;cursor:crosshair;overflow:hidden}}
.seg{{position:absolute;top:3px;bottom:3px;border-radius:5px;color:#fff;
  display:flex;align-items:center;justify-content:center;
  font:800 13px ui-monospace,monospace;box-shadow:inset 0 0 0 1px rgba(255,255,255,.5)}}
.seg.sel{{box-shadow:inset 0 0 0 2px #111,0 2px 8px rgba(0,0,0,.25);z-index:3}}
.seg .gL,.seg .gR{{position:absolute;top:0;bottom:0;width:14px;cursor:ew-resize}}
.seg .gL{{left:-2px}} .seg .gR{{right:-2px}}
.seg .x{{position:absolute;top:-1px;right:2px;font:700 11px system-ui;
  opacity:.85;padding:0 3px;cursor:pointer}}
.play{{position:absolute;top:15px;bottom:0;width:2px;background:#111;display:none;
  pointer-events:none;box-shadow:0 0 0 1px rgba(255,255,255,.6)}}
.bar{{display:flex;align-items:center;gap:8px;flex-wrap:wrap}}
.pp{{width:34px;height:34px;border-radius:50%;border:1px solid #d8cfb4;
  background:#f7f3e9;font-size:12px;cursor:pointer;flex:none}}
.pos{{font:600 12px ui-monospace,monospace;min-width:96px}}
.hint{{font:500 11px system-ui;color:#a89f8c}}
.ok{{border:1.5px solid #1f8a5b;background:#fff;color:#1f8a5b;border-radius:8px;
  padding:5px 10px;font:700 11.5px system-ui;cursor:pointer;margin-left:auto}}
section.done{{border-color:#1f8a5b;box-shadow:0 0 0 2px #1f8a5b22}}
section.done .ok{{background:#1f8a5b;color:#fff}}
section.done h2::after{{content:" ✓";color:#1f8a5b}}
ul.list{{list-style:none;margin:8px 0 0;padding:0;display:flex;flex-wrap:wrap;gap:5px}}
ul.list li{{display:flex;align-items:center;gap:5px;border:1px solid #e5dcc6;
  border-radius:7px;padding:2px 4px 2px 7px;font:600 11.5px ui-monospace,monospace;
  background:#fff}}
ul.list li span.d{{width:9px;height:9px;border-radius:2px;flex:none}}
ul.list li button{{border:0;background:none;cursor:pointer;font-size:12px;padding:0 2px}}
#dock{{position:fixed;left:0;right:0;bottom:0;background:#fffdf6cc;
  backdrop-filter:blur(10px);border-top:1px solid #e5dcc6;padding:8px 10px
  calc(8px + env(safe-area-inset-bottom));display:flex;gap:5px;flex-wrap:wrap;
  align-items:center;z-index:9}}
button.lab{{border:1.5px solid var(--c);background:#fff;color:var(--c);
  border-radius:8px;padding:6px 11px;font:800 13px ui-monospace,monospace;cursor:pointer}}
button.lab.on{{background:var(--c);color:#fff}}
#exp,#dl{{margin-left:auto;border:1px solid #d8cfb4;background:#f7f3e9;
  border-radius:8px;padding:6px 10px;font:700 12px system-ui;cursor:pointer}}
#toast{{position:fixed;left:50%;bottom:96px;transform:translateX(-50%);
  background:#1c1c1c;color:#fff;padding:8px 14px;border-radius:20px;
  font:600 12.5px system-ui;opacity:0;transition:opacity .2s;pointer-events:none;z-index:20}}
</style></head><body><div class=wrap>
<h1>Annoter les sections</h1>
<div class=lede>Choisis un libellé en bas, puis <b>tire sur le ruban</b> pour
poser la section. Tout aimante sur la <b>double-mesure</b>. Les bords se tirent,
le milieu se déplace, la croix supprime. <b>▶ sur une section</b> pour l'écouter.
La <b>bande grise juste au-dessus du ruban</b> sert à balayer la chanson sans
rien modifier. <b>✓ valider</b> quand un morceau est fini.
Tout est Tout part <b>sur le serveur</b> à chaque geste, donc rien
ne se perd si tu rafraîchis ou changes d'appareil.</div>
{cards}</div>
<div id=dock>{pal}<button id=exp>copier le JSON</button><button id=dl>télécharger</button></div>
<div id=toast></div>
<audio id=au preload=metadata playsinline></audio>
<script>
const SONGS = {data};
const LABELS = {labels};
const UNIT = {UNIT};
const COL = Object.fromEntries(LABELS);
const KEY = "harmonia.sections.v1";     // filet de secours si le serveur tombe
const au = document.getElementById("au");
const toastEl = document.getElementById("toast");

let active = "A";
let store = {{}};
let done = {{}};
try {{ store = JSON.parse(localStorage.getItem(KEY) || "{{}}"); }} catch (e) {{ store = {{}}; }}

localStorage.setItem(KEY, JSON.stringify(store));
const pending = {{}};
function push(stem) {{
  clearTimeout(pending[stem]);
  pending[stem] = setTimeout(() => {{
    const meta = SONGS.find(s => s.stem === stem);
    fetch("/api/sections/" + encodeURIComponent(stem), {{
      method: "POST", headers: {{ "Content-Type": "application/json" }},
      body: JSON.stringify({{ n: meta ? meta.n : null,
                             sections: store[stem] || [],
                             validated: !!done[stem] }}),
    }}).then(r => {{ if (!r.ok) toast("⚠ pas enregistré sur le serveur"); }})
      .catch(() => toast("⚠ pas enregistré sur le serveur"));
  }}, 400);
}}
function save(stem) {{
  localStorage.setItem(KEY, JSON.stringify(store));
  if (stem) push(stem);
}}
const fmt = s => Math.floor(s / 60) + ":" + String(Math.floor(s % 60)).padStart(2, "0");
const snap = (v, n) => Math.max(0, Math.min(n, Math.round(v / UNIT) * UNIT));
let toastT = null;
function toast(msg) {{
  toastEl.textContent = msg; toastEl.style.opacity = 1;
  clearTimeout(toastT); toastT = setTimeout(() => toastEl.style.opacity = 0, 1400);
}}

document.querySelectorAll("button.lab").forEach(b => {{
  if (b.dataset.l === active) b.classList.add("on");
  b.onclick = () => {{
    document.querySelectorAll("button.lab").forEach(x => x.classList.remove("on"));
    b.classList.add("on"); active = b.dataset.l;
  }};
}});

let live = null, stopAt = null, raf = null;

function tick() {{
  if (live) {{
    const G = live.grid, n = G.length - 1;
    let t = au.currentTime, f;
    if (t <= G[0]) f = 0; else if (t >= G[n]) f = n; else {{
      let lo = 0, hi = n;
      while (hi - lo > 1) {{ const m = (lo + hi) >> 1; G[m] <= t ? lo = m : hi = m; }}
      f = lo + (t - G[lo]) / (G[lo + 1] - G[lo]);
    }}
    live.playEl.style.display = "block";
    live.playEl.style.left = (100 * f / live.n) + "%";
    if (live.knobEl) live.knobEl.style.left = (100 * f / live.n) + "%";
    live.posEl.textContent = "mes. " + (Math.floor(f) + 1) + " · " + fmt(au.currentTime);
  }}
  if (stopAt != null && au.currentTime >= stopAt) {{ au.pause(); stopAt = null; }}
  if (!au.paused) raf = requestAnimationFrame(tick);
}}
au.addEventListener("play", () => {{ if (live) live.ppEl.textContent = "❚❚"; tick(); }});
au.addEventListener("pause", () => {{
  if (live) live.ppEl.textContent = "▶";
  cancelAnimationFrame(raf); tick();
}});

function play(S, b0, b1) {{
  if (live && live !== S) {{ live.ppEl.textContent = "▶"; live.playEl.style.display = "none"; }}
  live = S;
  const src = "/audio/" + S.stem + ".m4a";
  if (au.getAttribute("src") !== src) {{ au.setAttribute("src", src); au.load(); }}
  stopAt = b1 == null ? null : S.grid[Math.min(S.grid.length - 1, b1)];
  const t0 = S.grid[Math.max(0, Math.min(S.grid.length - 1, b0))];
  const seek = () => {{ try {{ au.currentTime = t0; }} catch (e) {{}} }};
  if (au.readyState >= 1) seek();
  else au.addEventListener("loadedmetadata", seek, {{ once: true }});
  au.play().catch(() => {{}});
}}

function build(S) {{
  const el = S.el, n = S.n;
  el.querySelector(".track").style.setProperty("--w", (100 / n) + "%");

  const ruler = el.querySelector(".ruler");
  let rh = "";
  const step = n > 120 ? 8 : 4;
  for (let b = 0; b <= n; b += step) {{
    rh += `<u style="left:${{100 * b / n}}%"></u>`;
    rh += `<b style="left:${{100 * b / n}}%">${{b + 1}}</b>`;
  }}
  ruler.innerHTML = rh;

  const sug = el.querySelector(".sug");
  sug.innerHTML = S.sug.map(s =>
    `<i style="left:${{100 * s.b0 / n}}%;width:${{100 * (s.b1 - s.b0 + 1) / n}}%;
      background:${{COL[s.label] || "#8a8371"}}">${{s.label}}</i>`).join("");

  el.querySelector(".copy").onclick = () => {{
    store[S.stem] = S.sug.map(s => ({{ ...s }}));
    save(S.stem); render(S); toast("base copiée");
  }};
  el.querySelector(".clear").onclick = () => {{
    store[S.stem] = []; save(S.stem); render(S); toast("effacé");
  }};
  el.querySelector(".pp").onclick = () => {{
    if (au.paused || live !== S) play(S, 0, null); else au.pause();
  }};

  // LA BANDE DE DÉFILEMENT. Louis, 2026-08-07 : « il faut que je puisse
  // rapidement balayer dans la chanson avec une tête de défilement qui bouge
  // au-dessus ; là quand je clique ça me modifie les sections ». Le ruban des
  // sections est un outil d'édition, pas un outil de lecture : cliquer dedans
  // pour écouter posait une section. Le défilement a donc sa propre bande, et
  // les deux gestes ne se marchent plus dessus.
  const scrub = el.querySelector(".scrub");
  scrub.style.setProperty("--w", (100 / S.n) + "%");
  const seekTo = ev => {{
    const r = scrub.getBoundingClientRect();
    const f = Math.max(0, Math.min(1, (ev.clientX - r.left) / r.width));
    const b = f * S.n, i = Math.max(0, Math.min(S.grid.length - 2, Math.floor(b)));
    const t = S.grid[i] + (b - i) * (S.grid[i + 1] - S.grid[i]);
    if (live !== S || au.getAttribute("src") !== "/audio/" + S.stem + ".m4a") {{
      play(S, 0, null);
    }}
    stopAt = null;
    try {{ au.currentTime = t; }} catch (e) {{}}
    tick();
  }};
  let scrubbing = false;
  scrub.addEventListener("pointerdown", ev => {{
    scrubbing = true; scrub.setPointerCapture(ev.pointerId); seekTo(ev);
  }});
  scrub.addEventListener("pointermove", ev => {{ if (scrubbing) seekTo(ev); }});
  const stopScrub = () => {{ scrubbing = false; }};
  scrub.addEventListener("pointerup", stopScrub);
  scrub.addEventListener("pointercancel", stopScrub);
  S.knobEl = el.querySelector(".knob");

  el.querySelector(".ok").onclick = () => {{
    done[S.stem] = !done[S.stem];
    el.classList.toggle("done", !!done[S.stem]);
    save(S.stem);
    toast(done[S.stem] ? "morceau validé" : "validation retirée");
  }};

  wire(S);
  render(S);
}}

function segs(S) {{ return store[S.stem] || (store[S.stem] = []); }}

function render(S) {{
  const track = S.el.querySelector(".track"), n = S.n;
  track.querySelectorAll(".seg").forEach(x => x.remove());
  const list = segs(S).slice().sort((a, b) => a.b0 - b.b0);
  store[S.stem] = list;
  list.forEach((s, i) => {{
    const d = document.createElement("div");
    d.className = "seg" + (S.sel === i ? " sel" : "");
    d.style.left = (100 * s.b0 / n) + "%";
    d.style.width = (100 * (s.b1 - s.b0 + 1) / n) + "%";
    d.style.background = COL[s.label] || "#8a8371";
    d.dataset.i = i;
    d.innerHTML = `<span class=gL></span>${{s.label}}<span class=gR></span>`
                + `<span class=x data-x="${{i}}">✕</span>`;
    track.appendChild(d);
  }});
  const ul = S.el.querySelector(".list");
  ul.innerHTML = list.map((s, i) =>
    `<li><span class=d style="background:${{COL[s.label] || "#8a8371"}}"></span>`
    + `${{s.label}} ${{s.b0 + 1}}–${{s.b1 + 1}}`
    + `<button data-p="${{i}}">▶</button><button data-k="${{i}}">✕</button></li>`).join("");
  ul.querySelectorAll("[data-p]").forEach(b => b.onclick = () => {{
    const s = list[+b.dataset.p]; play(S, s.b0, s.b1 + 1);
  }});
  ul.querySelectorAll("[data-k]").forEach(b => b.onclick = () => {{
    list.splice(+b.dataset.k, 1); S.sel = null; save(S.stem); render(S);
  }});
}}

function wire(S) {{
  const track = S.el.querySelector(".track"), n = S.n;
  const barAt = ev => {{
    const r = track.getBoundingClientRect();
    return snap(n * (ev.clientX - r.left) / r.width, n);
  }};
  let mode = null, idx = -1, anchor = 0, off = 0;

  track.addEventListener("pointerdown", ev => {{
    const x = ev.target.closest("[data-x]");
    if (x) {{ segs(S).splice(+x.dataset.x, 1); S.sel = null; save(S.stem); render(S); return; }}
    track.setPointerCapture(ev.pointerId);
    const seg = ev.target.closest(".seg");
    const b = barAt(ev);
    if (seg) {{
      idx = +seg.dataset.i; S.sel = idx;
      if (ev.target.classList.contains("gL")) mode = "L";
      else if (ev.target.classList.contains("gR")) mode = "R";
      else {{ mode = "move"; off = b - segs(S)[idx].b0; }}
      render(S);
      return;
    }}
    mode = "new"; anchor = b;
    segs(S).push({{ label: active, b0: b, b1: Math.min(n - 1, b + UNIT - 1) }});
    idx = segs(S).length - 1; S.sel = idx; render(S);
  }});

  track.addEventListener("pointermove", ev => {{
    if (!mode) return;
    const list = segs(S), s = list[idx];
    if (!s) return;
    const b = barAt(ev);
    if (mode === "new") {{
      s.b0 = Math.min(anchor, b); s.b1 = Math.max(anchor, b + UNIT) - 1;
    }} else if (mode === "L") {{
      s.b0 = Math.min(b, s.b1 - UNIT + 1);
    }} else if (mode === "R") {{
      s.b1 = Math.max(b + UNIT, s.b0 + UNIT) - 1;
    }} else {{
      const len = s.b1 - s.b0;
      s.b0 = Math.max(0, Math.min(n - 1 - len, b - off)); s.b1 = s.b0 + len;
    }}
    s.b0 = Math.max(0, s.b0); s.b1 = Math.min(n - 1, s.b1);
    // on garde l'indice sous la main : `render` retrie la liste
    const me = s;
    render(S);
    idx = segs(S).indexOf(me); S.sel = idx;
  }});

  const end = () => {{ if (mode) {{ mode = null; save(S.stem); render(S); }} }};
  track.addEventListener("pointerup", end);
  track.addEventListener("pointercancel", end);
}}

document.querySelectorAll("section[data-stem]").forEach(el => {{
  const meta = SONGS.find(s => s.stem === el.dataset.stem);
  if (!meta) return;
  const S = {{
    stem: meta.stem, n: meta.n, grid: meta.grid, sug: meta.sug, el: el, sel: null,
    playEl: el.querySelector(".play"), posEl: el.querySelector(".pos"),
    ppEl: el.querySelector(".pp"),
  }};
  build(S);
  // LE SERVEUR FAIT FOI. Le navigateur n'est qu'un filet : si une annotation
  // existe côté serveur elle écrase la copie locale, sinon on garde le local.
  fetch("/api/sections/" + encodeURIComponent(S.stem))
    .then(r => r.json())
    .then(d => {{
      if (d && Array.isArray(d.sections) && d.sections.length) {{
        store[S.stem] = d.sections; render(S);
      }}
      if (d && d.validated) {{ done[S.stem] = true; el.classList.add("done"); }}
    }}).catch(() => {{}});
}});

const payload = () => JSON.stringify(
  Object.fromEntries(SONGS.map(s => [s.stem,
    {{ n: s.n, validated: !!done[s.stem], sections: (store[s.stem] || []) }}])), null, 1);

document.getElementById("exp").onclick = async () => {{
  const t = payload();
  try {{ await navigator.clipboard.writeText(t); toast("JSON copié"); }}
  catch (e) {{
    const ta = document.createElement("textarea");
    ta.value = t; document.body.appendChild(ta); ta.select();
    document.execCommand("copy"); ta.remove(); toast("JSON copié");
  }}
}};
document.getElementById("dl").onclick = () => {{
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([payload()], {{ type: "application/json" }}));
  a.download = "harmonia_sections.json"; a.click();
  URL.revokeObjectURL(a.href); toast("téléchargé");
}};
</script></body></html>""")
    print(f"wrote {out.relative_to(HERE)} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
