"""Hear every block the dictionary found.

« Pour ces deux chansons montre-moi la répartition en block avec le morceau
jouable pour que je puisse entendre chaque section. »

One row per dictionary entry, one button per occurrence: it plays exactly that
block's bars and stops at its end. Plus the bars no entry claimed, so the gaps
are audible too. Audio mechanics copied verbatim from fold_label_deck.html,
which is verified working on :7772.

    python scripts/dictionary_audio.py   ->  /reports/dictionary_audio.html
"""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
from harmonia_min import sections as hs                  # noqa: E402
import harmonic_method as HM                             # noqa: E402
from dictionary_harmonic import bar_grid, build, ENTRY_COLS   # noqa: E402

OUT = HERE / "harmonia_min/state/reports/dictionary_audio.html"
SONGS = [("maroon_5_this_love", "This Love"),
         ("norah_jones_don_t_know_why", "Don't Know Why")]


def _songs_from_argv(default):
    """`python scripts/<page>.py <stem> [...]` inspects any song in docs/audio.
    Without arguments the page keeps its validated pair."""
    import sys as _s
    if len(_s.argv) <= 1:
        return default, ""
    stems = _s.argv[1:]
    return ([(st, st.replace("_", " ").title()) for st in stems],
            "_" + "_".join(st[:24] for st in stems))
PC = ["C", "D♭", "D", "E♭", "E", "F", "G♭", "G", "A♭", "A", "B♭", "B"]


def chord_text(chart_bars, a, b):
    out = []
    for i in range(a, min(b + 1, len(chart_bars))):
        bar = chart_bars[i]
        if not bar:
            out.append("%")
        else:
            out.append(" ".join("N.C." if c["nc"] else PC[c["root"]] + c["q"]
                                for c in bar))
    return " | ".join(out[:8]) + (" …" if b - a + 1 > 8 else "")


def song_data(stem, title):
    cap = bar_grid(stem)
    grid = cap["grid"]
    n = len(grid) - 1
    S = HM.ssm(HERE / f"docs/audio/{stem}.m4a", grid)
    entries, _boxed = build(S, n)

    # the per-bar chords the pipeline decoded, for the label under each block
    from harmonia_min import pipeline as _pl
    real = hs.detect_sections
    cap2 = {}

    def spy(g, a, t, bars=None, **_kw):
        cap2["bars"] = copy.deepcopy(bars)
        return real(g, a, t, bars, **_kw)

    hs.detect_sections = spy
    try:
        _pl.analyze(HERE / f"docs/audio/{stem}.m4a", title="x", file_key="x",
                    audio_url="")
    finally:
        hs.detect_sections = real
    bars = cap2.get("bars") or [[] for _ in range(n)]

    t = lambda b: float(grid[min(b, len(grid) - 1)])
    ents, owner = [], -np.ones(n, int)
    for ei, e in enumerate(entries):
        L = e["L"]
        blocks = sorted(set([e["b0"]] + list(e["occ"])))
        for b in blocks:
            owner[b:min(n, b + L)] = ei
        ents.append({
            "i": ei + 1, "L": L, "colour": ENTRY_COLS[ei % len(ENTRY_COLS)],
            "blocks": [{"b0": b, "b1": min(n - 1, b + L - 1),
                        "t0": t(b), "t1": t(min(n, b + L)),
                        "chords": chord_text(bars, b, min(n - 1, b + L - 1))}
                       for b in blocks]})
    gaps, b = [], 0
    while b < n:
        if owner[b] < 0:
            a = b
            while b < n and owner[b] < 0:
                b += 1
            gaps.append({"b0": a, "b1": b - 1, "t0": t(a), "t1": t(b),
                         "chords": chord_text(bars, a, b - 1)})
        else:
            b += 1
    return {"stem": stem, "title": title, "n": n,
            "audio": f"/audio/{stem}.m4a", "entries": ents, "gaps": gaps,
            "covered": int((owner >= 0).sum())}


PAGE = """<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Entendre les blocs</title><style>
*{box-sizing:border-box;-webkit-tap-highlight-color:transparent}
body{margin:0;background:#e7e0d0;font:15px/1.5 -apple-system,system-ui,sans-serif;color:#1c1c1c}
.wrap{max-width:900px;margin:0 auto;padding:20px 14px calc(40px + env(safe-area-inset-bottom))}
h1{font:italic 600 24px Georgia,serif;margin:0 0 4px}
.lede{color:#8a8371;font-size:13px;margin-bottom:20px}
section{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;padding:15px 16px;margin-bottom:14px}
h2{font:700 17px system-ui;margin:0 0 4px;color:#8a2b2b}
.sub{font:500 12px system-ui;color:#8a8371;margin-bottom:12px}
.entry{margin:14px 0 4px;font:700 12px system-ui;letter-spacing:.03em}
.blocks{display:flex;flex-wrap:wrap;gap:7px}
button.blk{border:1.5px solid;background:#f7f3e9;border-radius:11px;padding:8px 11px;
  cursor:pointer;text-align:left;font:600 12.5px system-ui;min-width:150px}
button.blk small{display:block;font:500 10.5px ui-monospace,monospace;color:#8a8371;margin-top:3px}
button.blk.on{color:#fff}
.gap button.blk{border-color:#b9b09a;color:#8a8371}
</style></head><body><div class=wrap>
<h1>Entendre les blocs</h1>
<div class=lede>Un bouton par bloc trouvé : il joue exactement ses mesures et
s'arrête à la fin. Les blocs gris sont les mesures qu'aucune entrée n'a
revendiquées. Sous chaque bouton, les accords que le pipeline y a décodés.</div>
__BODY__
</div><audio id=au preload=auto playsinline></audio>
<script>
const au=document.getElementById("au");
let stopAt=null, cur=null;
au.addEventListener("timeupdate",()=>{
  if(stopAt!=null && au.currentTime>=stopAt){ au.pause(); stopAt=null;
    if(cur){ cur.classList.remove("on"); cur.style.background="#f7f3e9"; cur=null; } }
});
function play(btn,src,t0,t1,col){
  if(au.getAttribute("src")!==src){ au.setAttribute("src",src); au.load(); }
  if(cur){ cur.classList.remove("on"); cur.style.background="#f7f3e9"; }
  cur=btn; btn.classList.add("on"); btn.style.background=col;
  stopAt=t1;
  const seek=()=>{ try{ au.currentTime=t0; }catch(e){} };
  if(au.readyState>=1) seek(); else au.addEventListener("loadedmetadata",seek,{once:true});
  au.play().catch(e=>{ btn.classList.remove("on"); btn.style.background="#f7f3e9"; });
}
document.querySelectorAll("[data-b]").forEach(b=>{
  const d=JSON.parse(b.dataset.b);
  b.onclick=()=>play(b,d[0],d[1],d[2],d[3]);
});
</script></body></html>"""


def main():
    songs, tag = _songs_from_argv(SONGS)
    out = OUT.with_name(OUT.stem + tag + OUT.suffix) if tag else OUT
    body = ""
    for stem, title in songs:
        d = song_data(stem, title)
        rows = ""
        for e in d["entries"]:
            btns = "".join(
                f'<button class=blk style="border-color:{e["colour"]}" '
                f"data-b='{json.dumps([d['audio'], round(b['t0'], 3), round(b['t1'], 3), e['colour']])}'>"
                f"▶ mes. {b['b0']+1}–{b['b1']+1}<small>{b['chords']}</small></button>"
                for b in e["blocks"])
            rows += (f'<div class=entry style="color:{e["colour"]}">'
                     f'entrée {e["i"]} — motif de {e["L"]} mesures, '
                     f'{len(e["blocks"])} blocs</div><div class=blocks>{btns}</div>')
        if d["gaps"]:
            btns = "".join(
                f'<button class=blk '
                f"data-b='{json.dumps([d['audio'], round(g['t0'], 3), round(g['t1'], 3), '#b9b09a'])}'>"
                f"▶ mes. {g['b0']+1}–{g['b1']+1}<small>{g['chords']}</small></button>"
                for g in d["gaps"])
            rows += ('<div class=entry style="color:#8a8371">non couvert</div>'
                     f'<div class="blocks gap">{btns}</div>')
        body += (f'<section><h2>{d["title"]}</h2><div class=sub>{d["n"]} mesures · '
                 f'{len(d["entries"])} entrées · {d["covered"]}/{d["n"]} mesures '
                 f'couvertes</div>{rows}</section>')
        print(f"  ok {title}")
    out.write_text(PAGE.replace("__BODY__", body))
    print(f"wrote {out.relative_to(HERE)} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
