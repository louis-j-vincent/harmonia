"""PROTOTYPE — symbolic repeat placer for section STARTS (lit-review candidate #1).

Louis, by ear: « tous les blocs sont bons, le souci c'est que des fois on ne
commence pas les blocs au début. »  Measured (docs/research_sessions/
structure_literature_2026-08-04.md §1.2): checkerboard novelty on chroma — what
`harmonia_min/sections.py` runs — places 34.7% of section starts on the exact
right bar; matching the CHORD STRING against its own repeats places 75.2%.

This script does NOT change the pipeline. It:
  1. spies on `detect_sections` during a real `pipeline.analyze()` run to
     capture the true `grid` / `bars` / segments in flight (same trick as
     `scripts/sections_explainer.py`; deepcopy because the fold stage later
     mutates those bar dicts);
  2. writes the song as ONE CHORD SYMBOL PER BAR;
  3. for every section, searches a shift of -4..+4 bars and picks the phase at
     which all occurrences of that letter agree on the SAME chord string —
     the cross-occurrence constraint nothing since RefraiD (2006) enforces;
  4. emits an audition page where Louis hears CURRENT start vs PROPOSED start
     back to back.

    python scripts/repeat_placer_prototype.py
    -> /reports/placer_prototype.html on :7772
"""
from __future__ import annotations

import copy
import html
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))

from harmonia_min import sections as hs                      # noqa: E402

SONGS = [
    ("norah_jones_don_t_know_why", "Norah Jones — Don't Know Why"),
    ("maroon_5_this_love", "Maroon 5 — This Love"),
]
OUT = HERE / "harmonia_min/state/reports/placer_prototype.html"

W_MAX = 8            # the lit review's window: "does this 8-bar string repeat?"
W_MIN = 4            # never score on fewer bars than this
SHIFTS = range(-4, 5)
MIN_SEG = 2          # a section keeps at least this many bars
PC = ["C", "D♭", "D", "E♭", "E", "F", "G♭", "G", "A♭", "A", "B♭", "B"]


# ── one chord symbol per bar ────────────────────────────────────────────────
def bar_sym(bars, b, n_bars):
    """(root, family) per chord written in the bar. Same rule as
    `sections._sig` — held bars carry a written copy of what SOUNDS in them,
    so 'held G' signs like 'attacked G'. N.C./empty signs as ('%',)."""
    if bars is None or not (0 <= b < n_bars) or not bars[b]:
        return ("%",)
    out = []
    for c in bars[b]:
        if c.get("nc"):
            return ("%",)
        q = c["q"]
        fam = ("m" if q.startswith("-") else
               "d" if q[:1] in ("h", "o") else
               "s" if "sus" in q else
               "a" if q.startswith("+") else "M")
        out.append((c["root"], fam))
    return tuple(out) or ("%",)


def sym_text(s):
    if s == ("%",):
        return "·"
    return " ".join(PC[r] + ("m" if f == "m" else "ø" if f == "d" else
                             "sus" if f == "s" else "+" if f == "a" else "")
                    for r, f in s)


def match(x, y) -> float:
    """Graded per-bar agreement. Exact bar = 1; same first chord = 0.5."""
    if x == ("%",) or y == ("%",):
        return 1.0 if x == y else 0.0
    if x == y:
        return 1.0
    return 0.5 if x[0] == y[0] else 0.0


def window(S, b, w, n_bars):
    return [S[b + i] if 0 <= b + i < n_bars else None for i in range(w)]


def agree(A, B) -> float:
    ok = [(x, y) for x, y in zip(A, B) if x is not None and y is not None]
    if not ok:
        return 0.0
    return sum(match(x, y) for x, y in ok) / len(ok)


def repeat_score(S, b, n_bars, w=W_MAX, gap=4) -> float:
    """The lit review's cue, verbatim: does the w-bar chord string starting at
    bar b occur AGAIN somewhere else in the song? Best match elsewhere."""
    if b + w > n_bars:
        w = n_bars - b
    if w < W_MIN:
        return 0.0
    A = window(S, b, w, n_bars)
    best = 0.0
    for p in range(0, n_bars - w + 1):
        if abs(p - b) < gap:
            continue
        best = max(best, agree(A, window(S, p, w, n_bars)))
    return best


# ── the placer ──────────────────────────────────────────────────────────────
def place(segs, S, n_bars):
    """Return {seg_index: delta_bars}. All occurrences of a letter are pushed
    onto ONE shared phase: we try every (occurrence, shift) pair as the anchor
    template, greedily fit the others to it, and keep the anchor whose
    consensus is strongest. That is the cross-occurrence constraint."""
    by_letter = {}
    for i, s in enumerate(segs):
        by_letter.setdefault(s["label"], []).append(i)

    prop, detail = {}, {}
    for lab, g in by_letter.items():
        if len(g) < 2:
            detail[lab] = {"n": len(g), "note": "jouée une seule fois — pas de contrainte de répétition"}
            continue
        lens = [segs[i]["b1"] - segs[i]["b0"] + 1 for i in g]
        w = max(W_MIN, min(W_MAX, min(lens)))

        def valid(i, d):
            b = segs[i]["b0"] + d
            return 0 <= b <= n_bars - MIN_SEG and (segs[i]["b0"] != 0 or d == 0)

        best = None
        for ai in g:
            for da in SHIFTS:
                if not valid(ai, da):
                    continue
                T = window(S, segs[ai]["b0"] + da, w, n_bars)
                if all(t is None for t in T):
                    continue
                tot, assign = 1.0, {ai: (da, 1.0)}
                for j in g:
                    if j == ai:
                        continue
                    cand = []
                    for dj in SHIFTS:
                        if not valid(j, dj):
                            continue
                        cand.append((agree(T, window(S, segs[j]["b0"] + dj, w, n_bars)),
                                     -abs(dj), dj))
                    if not cand:
                        continue
                    sc, _, dj = max(cand)
                    assign[j] = (dj, sc)
                    tot += sc
                key = (tot / len(g), -sum(abs(d) for d, _ in assign.values()))
                if best is None or key > best[0]:
                    best = (key, ai, da, assign)
        if best is None:
            continue
        (cons, _), ai, da, assign = best
        # SAME normalisation as `cons` (anchor's own 1.0 included), otherwise
        # the two numbers are not comparable and a no-op looks like a gain.
        T0 = window(S, segs[ai]["b0"], w, n_bars)
        base = (1.0 + sum(agree(T0, window(S, segs[j]["b0"], w, n_bars))
                          for j in g if j != ai)) / len(g)
        detail[lab] = {"n": len(g), "w": w, "consensus": round(cons, 3),
                       "before": round(float(base), 3),
                       "lens": sorted(lens),
                       "anchor": f"{lab}@{segs[ai]['b0']}{da:+d}"}
        for j, (dj, _) in assign.items():
            if dj:
                prop[j] = dj

    # keep the section list legal: strictly increasing starts, >= MIN_SEG bars,
    # section 0 pinned at bar 0 (sections must cover the song).
    new = [s["b0"] + prop.get(i, 0) for i, s in enumerate(segs)]
    new[0] = 0
    dropped = []
    for i in range(1, len(segs)):
        lo = new[i - 1] + MIN_SEG
        hi = (new[i + 1] if i + 1 < len(new) else n_bars) - MIN_SEG
        if not (lo <= new[i] <= hi) and i in prop:
            dropped.append((i, prop.pop(i)))
            new[i] = segs[i]["b0"]
    return prop, detail, dropped


# ── run one song ────────────────────────────────────────────────────────────
def run(stem, title):
    from harmonia_min import pipeline as _pl
    cap = {}
    _real = hs.detect_sections

    def _spy(grid, arr, times, bars=None, **_kw):
        out = _real(grid, arr, times, bars, **_kw)
        cap.update(grid=grid, bars=copy.deepcopy(bars),
                   segs=copy.deepcopy(out))
        return out

    hs.detect_sections = _spy
    try:
        _pl.analyze(HERE / f"docs/audio/{stem}.m4a", title=title,
                    file_key=f"min_{stem}", audio_url=f"/audio/{stem}.m4a")
    finally:
        hs.detect_sections = _real

    grid, bars, segs = cap["grid"], cap["bars"], cap["segs"]
    n_bars = len(grid) - 1
    S = [bar_sym(bars, b, n_bars) for b in range(n_bars)]

    prop, detail, dropped = place(segs, S, n_bars)

    new_b0 = [s["b0"] + prop.get(i, 0) for i, s in enumerate(segs)]
    new_b0[0] = 0
    new_b1 = [(new_b0[i + 1] - 1 if i + 1 < len(segs) else n_bars - 1)
              for i in range(len(segs))]

    def t(b):
        return float(grid[max(0, min(len(grid) - 1, b))])

    rows = []
    for i, s in enumerate(segs):
        nb0, nb1 = new_b0[i], new_b1[i]
        play_n = max(2, min(8, nb1 - nb0 + 1))
        old_n = max(2, min(8, s["b1"] - s["b0"] + 1))
        rows.append({
            "i": i, "lab": s["label"],
            "cur": [s["b0"], s["b1"]], "new": [nb0, nb1],
            "shift": nb0 - s["b0"],
            "cur_len": s["b1"] - s["b0"] + 1, "new_len": nb1 - nb0 + 1,
            "cur_t": [round(t(s["b0"]), 2), round(t(s["b0"] + old_n), 2)],
            "new_t": [round(t(nb0), 2), round(t(nb0 + play_n), 2)],
            "cur_r": round(repeat_score(S, s["b0"], n_bars), 3),
            "new_r": round(repeat_score(S, nb0, n_bars), 3),
            "strip_cur": [sym_text(S[b]) if 0 <= b < n_bars else ""
                          for b in range(s["b0"] - 2, s["b0"] + 6)],
            "strip_new": [sym_text(S[b]) if 0 <= b < n_bars else ""
                          for b in range(nb0 - 2, nb0 + 6)],
        })

    return {"stem": stem, "title": title, "audio": f"/audio/{stem}.m4a",
            "n_bars": n_bars, "rows": rows, "detail": detail,
            "dropped": [[i, d] for i, d in dropped],
            "chords": [sym_text(x) for x in S]}


# ── page ────────────────────────────────────────────────────────────────────
CSS = """
*{box-sizing:border-box;-webkit-tap-highlight-color:transparent}
body{margin:0;background:#e7e0d0;font:16px/1.45 -apple-system,BlinkMacSystemFont,system-ui,sans-serif;color:#1c1c1c}
.wrap{max-width:760px;margin:0 auto;padding:16px 14px calc(40px + env(safe-area-inset-bottom))}
h1{font:italic 600 24px Georgia,serif;margin:0 0 4px}
.lede{color:#8a8371;font-size:13px;margin-bottom:20px}
h2{font:italic 600 20px Georgia,serif;margin:26px 0 8px}
.card{background:#fffdf6;border:1px solid #e5dcc6;border-radius:16px;padding:14px;margin-bottom:11px;
      box-shadow:0 10px 26px -20px rgba(50,35,20,.5)}
.card.moved{border-color:#8a2b2b}
.hd{display:flex;align-items:baseline;gap:9px;margin-bottom:9px;flex-wrap:wrap}
.let{font:700 20px system-ui;color:#8a2b2b}
.meta{font:500 12px system-ui;color:#8a8371}
.badge{font:700 10px system-ui;letter-spacing:.06em;text-transform:uppercase;
       border-radius:5px;padding:2px 7px;border:1.5px solid #8a2b2b;color:#8a2b2b}
.badge.same{border-color:#b9b09a;color:#8a8371}
.pass{display:flex;gap:9px;margin-bottom:8px}
.pass button{flex:1;min-width:0;border:1.5px solid #b9b09a;background:#f7f3e9;border-radius:12px;
  padding:12px 9px;font:600 14px system-ui;color:#1c1c1c;cursor:pointer;text-align:left}
.pass button.prop{border-color:#1f8a5b}
.pass button.on{border-color:#8a2b2b;background:#8a2b2b;color:#fff}
.pass small{display:block;font:500 11px system-ui;opacity:.72;margin-top:3px}
.chain{width:100%;border:1.5px dashed #b9b09a;background:transparent;border-radius:12px;
  padding:10px;font:600 13px system-ui;color:#8a8371;cursor:pointer}
.strip{display:flex;gap:3px;margin-top:10px;overflow-x:auto;padding-bottom:3px}
.strip div{flex:0 0 auto;min-width:48px;text-align:center;font:600 11px system-ui;
  background:#f7f3e9;border:1px solid #e5dcc6;border-radius:6px;padding:5px 4px;color:#8a8371}
.strip div.start{background:#1f8a5b;color:#fff;border-color:#1f8a5b}
.strip div.pre{opacity:.45}
.striplab{font:500 11px system-ui;color:#8a8371;margin-top:7px}
table{border-collapse:collapse;font:12px system-ui;width:100%;margin:8px 0 4px}
th,td{border:1px solid #e5dcc6;padding:4px 8px;text-align:left}
th{background:#f7f3e9;font-weight:700}
td.mv{color:#8a2b2b;font-weight:700}
.note{background:#f7f3e9;border-left:3px solid #1f8a5b;padding:9px 12px;font-size:13px;
      border-radius:0 8px 8px 0;margin:10px 0}
audio{display:none}
"""

JS = """
const au = document.getElementById("au");
let stopAt = null, chainNext = null;
au.addEventListener("timeupdate", () => {
  if (stopAt != null && au.currentTime >= stopAt) {
    au.pause(); stopAt = null;
    document.querySelectorAll(".pass button").forEach(b => b.classList.remove("on"));
    if (chainNext) { const f = chainNext; chainNext = null; setTimeout(f, 300); }
  }
});
function play(src, t0, t1, btn, then) {
  if (au.getAttribute("src") !== src) { au.setAttribute("src", src); au.load(); }
  document.querySelectorAll(".pass button").forEach(b => b.classList.remove("on"));
  if (btn) btn.classList.add("on");
  chainNext = then || null;
  // play() must be called SYNCHRONOUSLY inside the click: user activation
  // does not survive an await/event gap, so calling it from a later
  // `loadedmetadata` callback gets the promise rejected by the autoplay
  // policy — and the old empty .catch() swallowed that silently (the page
  // seeked to t0 and sat there paused; verified headlessly 2026-08-04).
  // Start playback now, seek as soon as metadata allows.
  stopAt = t1;
  const seek = () => { try { au.currentTime = t0; } catch(e){} };
  if (au.readyState >= 1) seek(); else au.addEventListener("loadedmetadata", seek, {once:true});
  au.play().then(() => {
    if (au.readyState >= 1 && Math.abs(au.currentTime - t0) > 1.5) seek();
  }).catch(e => {
    const w = document.getElementById("playerr");
    if (w) { w.style.display = "block"; w.textContent = "Lecture bloquée : " + e; }
  });
}
window.__play = play;
document.querySelectorAll("[data-play]").forEach(b => {
  const d = JSON.parse(b.getAttribute("data-play"));
  b.onclick = e => play(d[0], d[1], d[2], e.currentTarget);
});
document.querySelectorAll("[data-chain]").forEach(b => {
  const d = JSON.parse(b.getAttribute("data-chain"));
  b.onclick = () => play(d[0], d[1], d[2], document.getElementById(d[5]),
                   () => play(d[0], d[3], d[4], document.getElementById(d[6])));
});
"""


def strip_html(cells, start_idx=2):
    out = []
    for k, c in enumerate(cells):
        cls = "start" if k == start_idx else ("pre" if k < start_idx else "")
        out.append(f'<div class="{cls}">{html.escape(c)}</div>')
    return '<div class="strip">' + "".join(out) + "</div>"


def song_html(d):
    a = d["audio"]
    moved = [r for r in d["rows"] if r["shift"]]
    tbl = "".join(
        "<tr>"
        f'<td><b>{r["lab"]}</b></td>'
        f'<td>{r["cur"][0]+1}–{r["cur"][1]+1}</td><td>{r["cur_len"]}</td>'
        f'<td class="{"mv" if r["shift"] else ""}">{r["new"][0]+1}–{r["new"][1]+1}</td>'
        f'<td>{r["new_len"]}</td>'
        f'<td class="{"mv" if r["shift"] else ""}">'
        f'{("%+d" % r["shift"]) if r["shift"] else "·"}</td>'
        f'<td>{r["cur_r"]:.2f} → {r["new_r"]:.2f}</td></tr>'
        for r in d["rows"])

    det = "".join(
        f'<tr><td><b>{k}</b></td><td>{v["n"]}</td>'
        f'<td>{v.get("w","–")}</td>'
        f'<td>{v.get("before","–")}</td><td>{v.get("consensus","–")}</td>'
        f'<td>{"/".join(str(x) for x in v["lens"]) if "lens" in v else "–"}</td>'
        f'<td>{html.escape(str(v.get("note","")))}</td></tr>'
        for k, v in sorted(d["detail"].items()))
    bad_len = [k for k, v in d["detail"].items()
               if "lens" in v and len(set(v["lens"])) > 1]
    lennote = ""
    if bad_len:
        txt = "; ".join(
            f'<b>{k}</b> se joue ' +
            " / ".join(f"{x} mes." for x in d["detail"][k]["lens"])
            for k in sorted(bad_len))
        lennote = (f'<div class="note"><b>Longueurs qui ne s\'accordent pas :</b> '
                   f'{txt}. Le placeur ne touche <b>que la phase</b> (où ça '
                   f'commence). Réécrire toutes les occurrences à la même '
                   f'longueur reviendrait à sur-replier — ta règle dit '
                   f'l\'inverse : chaque occurrence s\'écrit à la longueur '
                   f'qu\'elle joue vraiment.</div>')

    cards = []
    for r in d["rows"]:
        i, mv = r["i"], r["shift"]
        idc, idp = f'{d["stem"]}_c{i}', f'{d["stem"]}_p{i}'
        cur = json.dumps([a, r["cur_t"][0], r["cur_t"][1]])
        new = json.dumps([a, r["new_t"][0], r["new_t"][1]])
        chain = json.dumps([a, r["cur_t"][0], r["cur_t"][1],
                            r["new_t"][0], r["new_t"][1], idc, idp])
        badge = (f'<span class="badge">décalé de {mv:+d} mesure'
                 f'{"s" if abs(mv) > 1 else ""}</span>' if mv else
                 '<span class="badge same">inchangé</span>')
        body = f"""
<div class="card {'moved' if mv else ''}">
  <div class="hd"><span class="let">{r['lab']}</span>{badge}
    <span class="meta">actuel mes. {r['cur'][0]+1}–{r['cur'][1]+1} ({r['cur_len']} mes.)
      · proposé mes. {r['new'][0]+1}–{r['new'][1]+1} ({r['new_len']} mes.)
      · score de répétition {r['cur_r']:.2f} → {r['new_r']:.2f}</span></div>
  <div class="pass">
    <button id="{idc}" data-play='{cur}'>▶ début ACTUEL
      <small>mes. {r['cur'][0]+1} · {r['cur_t'][0]:.1f}s</small></button>
    <button id="{idp}" class="prop" data-play='{new}'>▶ début PROPOSÉ
      <small>mes. {r['new'][0]+1} · {r['new_t'][0]:.1f}s</small></button>
  </div>
  <button class="chain" data-chain='{chain}'>▶ actuel puis proposé</button>
  <div class="striplab">accords écrits — mesure de départ actuelle en vert
    (2 mesures de contexte avant)</div>
  {strip_html(r['strip_cur'])}
  {'<div class="striplab">accords écrits — mesure de départ PROPOSÉE en vert</div>'
   + strip_html(r['strip_new']) if mv else ''}
</div>"""
        cards.append(body)

    drop = ""
    if d["dropped"]:
        drop = ('<div class="note">Décalages refusés (ils auraient chevauché la '
                'section voisine) : ' +
                ", ".join(f"section #{i} ({v:+d})" for i, v in d["dropped"]) +
                "</div>")

    return f"""
<h2>{html.escape(d['title'])}</h2>
<div class="meta">{d['n_bars']} mesures · {len(d['rows'])} sections ·
  <b>{len(moved)} départs déplacés</b></div>
<table><tr><th>lettre</th><th>mesures actuelles</th><th>lg</th>
  <th>mesures proposées</th><th>lg</th><th>décalage</th>
  <th>répétition du début (avant → après)</th></tr>{tbl}</table>
<table><tr><th>lettre</th><th>occurrences</th><th>fenêtre</th>
  <th>accord entre sœurs AVANT</th><th>APRÈS</th><th>longueurs jouées</th>
  <th></th></tr>{det}</table>
{lennote}
{drop}
{''.join(cards)}"""


def main():
    data = []
    for stem, title in SONGS:
        print(f"— {stem}")
        d = run(stem, title)
        nm = sum(1 for r in d["rows"] if r["shift"])
        print(f"  {d['n_bars']} bars, {len(d['rows'])} sections, {nm} moved: "
              + ", ".join(f"{r['lab']}@{r['cur'][0]}{r['shift']:+d}"
                          for r in d["rows"] if r["shift"]))
        data.append(d)
        if os.environ.get("PLACER_DUMP"):
            Path(os.environ["PLACER_DUMP"], f"{stem}.json").write_text(
                json.dumps(d, indent=1))

    body = "".join(song_html(d) for d in data)
    OUT.write_text(f"""<!DOCTYPE html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Placeur par répétition — prototype</title>
<style>{CSS}</style></head><body><div class="wrap">
<h1>Où commencent vraiment les blocs ?</h1>
<div class="lede">Prototype. Pour chaque section : le début que le pipeline
choisit aujourd'hui, et le début que propose un placeur qui ne regarde que la
<b>suite d'accords écrite mesure par mesure</b> — il cherche la phase à laquelle
toutes les occurrences d'une même lettre jouent la même chaîne. Écoute les deux,
c'est ton oreille qui tranche. Rien n'est modifié dans le pipeline.</div>
<div class="note"><b>Comment lire.</b> « début ACTUEL » joue 8 mesures à partir
de la mesure choisie aujourd'hui. « début PROPOSÉ » joue 8 mesures à partir de
la mesure proposée. La question est simple : <b>lequel des deux commence sur
un vrai début de phrase ?</b></div>
{body}
<div id="playerr" style="display:none;position:fixed;left:8px;right:8px;bottom:8px;z-index:99;background:#8a2b2b;color:#fff;font:600 12px system-ui;padding:9px 12px;border-radius:10px"></div></div><audio id="au" preload="auto" playsinline></audio>
<script>{JS}</script></body></html>""")
    print(f"\nwrote {OUT.relative_to(HERE)} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
