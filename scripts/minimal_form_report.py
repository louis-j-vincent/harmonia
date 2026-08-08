"""La version minimale de chaque morceau, à côté de sa version dépliée.

    python scripts/minimal_form_report.py   ->  /reports/forme_minimale.html

Louis, 2026-08-08 : « affiche-moi les chansons dans leur version minimale : pour
chaque section, quelle est la forme minimale — 4 mesures qui bouclent, 4 mesures
qui bouclent avec les 2 dernières qui varient, 8 mesures qui bouclent avec les 2
dernières qui varient… La règle d'or : le plus compact possible, MAIS toutes les
variations doivent être représentées. »

La page montre, morceau par morceau : la forme minimale (`harmonia_min/
minimal_form.py`), la grille dépliée en regard, et le taux de compression
(mesures écrites ÷ mesures jouées). Chaque boucle et chaque variante est
jouable — on clique, l'audio saute au passage. C'est le seul moyen de vérifier
qu'une compression est juste : la lire ne suffit pas, il faut l'entendre.

Aucun PNG ici, donc pas de marge fixe `PLOT_L`/`PLOT_R` à tenir (la convention de
`pattern_lanes.py` existe pour poser un curseur de lecture sur une image ; les
grilles sont du HTML, elles se placent toutes seules et restent lisibles à
390 px). Le reste suit la maison : fond parchemin, cartes crème, rouge de titre.

Les données viennent de `scripts/minimal_form_capture.py`, qui espionne l'appel à
`minimal_fold` dans le pipeline : le chart servi par l'app ne contient plus la
suite d'accords dépliée, seulement le bloc déjà replié.

CE SCRIPT NE TOUCHE PAS AU CHART DE L'APP. Il ne fait que mesurer et montrer.
"""
from __future__ import annotations

import html
import os
import re
import sys
from pathlib import Path

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))

from minimal_form_capture import capture, stems                      # noqa: E402
from harmonia_min import minimal_form as MF                          # noqa: E402

OUT = HERE / "harmonia_min/state/reports/forme_minimale.html"

# Les morceaux qu'il connaît le mieux passent en premier.
FIRST = ["bein_green", "maroon_5_this_love", "norah_jones_don_t_know_why",
         "let_it_be_remastered_2009", "ben_e_king_stand_by_me_audio"]

_NOISE = re.compile(
    r"\b(official|music|lyric|lyrics|video|audio|hd|4k|remastered|"
    r"explicit|feat|ft|original|remaster)\b|\b(19|20)\d\d\b", re.I)


def pretty(stem: str) -> str:
    """« let_it_be_remastered_2009 » -> « Let It Be ». Un identifiant reste tel."""
    if "_" not in stem:
        return stem
    t = _NOISE.sub(" ", stem.replace("_", " "))
    t = " ".join(t.split()).title()
    return re.sub(r"\b([A-Za-z]{2,}) T\b", r"\1't", t) or stem


NOTE = "C D♭ D E♭ E F G♭ G A♭ A B♭ B".split()
SUP = {"": "", "-": "m", "7": "7", "^7": "Δ", "-7": "m7", "h7": "ø7",
       "o": "°", "o7": "°7", "9": "9", "-9": "m9", "^9": "Δ9", "13": "13",
       "+": "+", "sus2": "sus2", "sus4": "sus4", "7sus4": "7sus4",
       "^": "Δ", "-^7": "mΔ7", "6": "6", "-6": "m6"}
INK = "#1c1c1c"


# ── rendu d'une mesure (signature -> HTML) ──────────────────────────────────

def chord(c) -> str:
    _, root, q, bass, nc, carry = c
    if nc:
        return "<i class='nc'>N.C.</i>"
    s = f"<b>{NOTE[root]}</b><sup>{html.escape(SUP.get(q, q))}</sup>"
    if bass >= 0 and bass != root:
        s += f"<span class='bs'>/{NOTE[bass]}</span>"
    return f"<span class='{'cy' if carry else ''}'>{s}</span>"


def barcell(sig, extra: str = "") -> str:
    if not sig:
        return f"<div class='bar {extra}'></div>"
    return (f"<div class='bar {extra}'>"
            + " ".join(chord(c) for c in sig) + "</div>")


def gridof(sigs, extra=None, span=None) -> str:
    """Une grille de mesures, 4 par ligne, cliquable si `span` donne les temps."""
    out = []
    for i, s in enumerate(sigs):
        cls = (extra or {}).get(i, "")
        t = (span or {}).get(i)
        cell = barcell(s, cls)
        if t:
            cell = cell.replace("<div class='bar",
                                f"<div data-t=\"{t[0]:.2f},{t[1]:.2f}\" "
                                f"class='bar clik", 1)
        out.append(cell)
    # une grille de moins de 4 mesures n'étale pas 4 colonnes vides
    cols = min(4, max(1, len(sigs)))
    return (f"<div class='grid' style='grid-template-columns:"
            f"repeat({cols},1fr)'>" + "".join(out) + "</div>")


# ── une section, en forme minimale ──────────────────────────────────────────

def block_html(lf, i, blk, ranges, grid) -> str:
    """Un bloc : sa boucle, ses variantes, ses mesures écrites au long."""
    nm = lf.name(i)

    def times(occ, off, n):
        b0 = ranges[occ][0] + off
        return grid[b0], grid[min(len(grid) - 1, b0 + n)]

    # tous les passages où la boucle se joue telle quelle -> bouton de lecture
    plays = [times(r.occ, r.b0, blk.period) for r in blk.renditions
             if not r.patches]
    head = f"<span class='lb'>{html.escape(nm)}</span>"
    if blk.alias_of:
        head += (f"<span class='meta'>même boucle que "
                 f"<b>{html.escape(blk.alias_of)}</b> — seules ses fins "
                 f"changent</span>")
    elif blk.period:
        head += (f"<span class='meta'>boucle de <b>{blk.period}</b> mesure"
                 f"{'s' if blk.period > 1 else ''} × <b>"
                 f"{len(blk.renditions)}</b></span>")
    else:
        head += "<span class='meta'>écrite au long — aucune boucle</span>"
    if plays:
        t0, t1 = plays[0]
        head += (f"<button class='pl' data-t='{t0:.2f},{t1:.2f}'>▶</button>")

    body = ""
    if blk.period and not blk.alias_of:
        body += gridof(blk.cell)
    for pi, p in enumerate(blk.patches):
        who = [j + 1 for j, r in enumerate(blk.renditions) if pi in r.patches]
        where = ("fin" if p.is_ending(blk.period)
                 else f"mesure {p.pos + 1}"
                 + (f"–{p.pos + len(p.bars)}" if len(p.bars) > 1 else ""))
        r0 = next(r for r in blk.renditions if pi in r.patches)
        tt = times(r0.occ, r0.b0 + p.pos, len(p.bars))
        body += (f"<div class='var'><div class='vh'>{where} · reprise"
                 f"{'s' if len(who) > 1 else ''} {', '.join(map(str, who))}"
                 f"<button class='pl' data-t='{tt[0]:.2f},{tt[1]:.2f}'>▶</button>"
                 f"</div>{gridof(p.bars)}</div>")
    for lit in blk.literals:
        tt = times(lit.occ, lit.b0, len(lit.bars))
        body += (f"<div class='var lit'><div class='vh'>au long · "
                 f"{len(lit.bars)} mesure{'s' if len(lit.bars) > 1 else ''}"
                 f"<button class='pl' data-t='{tt[0]:.2f},{tt[1]:.2f}'>▶</button>"
                 f"</div>{gridof(lit.bars)}</div>")
    return f"<div class='blk'><div class='bh'>{head}</div>{body}</div>"


def song_html(stem, d) -> tuple[str, dict]:
    form = MF.compress_song(d["sections"], d["bars"])
    loose = MF.compress_song(d["sections"], d["bars"], loose=True)
    ranges = MF.occurrence_ranges(d["sections"])
    grid, sigs = d["grid"], MF.signatures(d["bars"])
    n = len(d["bars"])

    mini = "".join(block_html(lf, i, b, ranges[lf.label], grid)
                   for lf in form.letters for i, b in enumerate(lf.blocks))

    # la grille dépliée, avec la lettre en marge à chaque changement
    owner = [""] * n
    for s in d["sections"]:
        for b0, b1 in s["barRanges"]:
            for b in range(b0, min(b1 + 1, n)):
                owner[b] = s["label"]
    rows, b = [], 0
    while b < n:
        z = b
        while z + 1 < n and owner[z + 1] == owner[b]:
            z += 1
        span = {i - b: (grid[i], grid[min(len(grid) - 1, i + 1)])
                for i in range(b, z + 1)}
        rows.append(f"<div class='occ'><span class='ol'>"
                    f"{html.escape(owner[b] or '·')}</span>"
                    f"{gridof(sigs[b:z + 1], span=span)}</div>")
        b = z + 1

    boucle = sum(bl.played for lf in form.letters for bl in lf.blocks
                 if bl.period)
    stat = {"stem": stem, "title": pretty(stem), "written": form.written,
            "played": form.played, "ratio": form.ratio,
            "loose": loose.ratio, "loop_share": boucle / max(1, form.played),
            "blocks": sum(len(lf.blocks) for lf in form.letters),
            "flat": sum(bl.played for lf in form.letters for bl in lf.blocks
                        if not bl.period)}
    title = html.escape(pretty(stem))
    x = form.played / max(1, form.written)
    body = f"""
<section id="s-{html.escape(stem)}">
  <div class="sh">
    <h2>{title}</h2>
    <span class="ratio"><b>{form.written}</b> écrites / <b>{form.played}</b>
      jouées · <b>÷{x:.1f}</b></span>
  </div>
  <audio controls preload="none" src="/audio/{html.escape(stem)}.m4a"></audio>
  <div class="cols">
    <div class="col"><h3>Forme minimale</h3>{mini}</div>
    <div class="col"><h3>Déplié · {n} mesures</h3>
      <div class="flat">{''.join(rows)}</div></div>
  </div>
</section>"""
    return body, stat


# ── la page ─────────────────────────────────────────────────────────────────

CSS = """
*{box-sizing:border-box;-webkit-tap-highlight-color:transparent}
body{margin:0;background:#e7e0d0;color:#1c1c1c;
     font:16px/1.55 -apple-system,system-ui,sans-serif}
.wrap{max-width:1150px;margin:0 auto;
      padding:20px 12px calc(40px + env(safe-area-inset-bottom))}
h1{font:italic 600 26px Georgia,serif;margin:0 0 4px}
.lede{color:#6b6455;margin:0 0 16px;max-width:62ch}
.lede b{color:#8a2b2b}
section{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;
        padding:14px 13px;margin-bottom:16px}
h2{font:italic 600 20px Georgia,serif;margin:0;color:#8a2b2b}
h3{font:600 12px/1 -apple-system,sans-serif;letter-spacing:.08em;
   text-transform:uppercase;color:#8a8371;margin:0 0 8px}
.sh{display:flex;flex-wrap:wrap;align-items:baseline;gap:10px;
    justify-content:space-between}
.ratio{font:13px ui-monospace,monospace;color:#6b6455}
audio{width:100%;height:34px;margin:10px 0 4px}
.cols{display:grid;grid-template-columns:1fr;gap:14px}
@media(min-width:860px){.cols{grid-template-columns:1fr 1fr;gap:18px}}
.grid{display:grid;grid-template-columns:repeat(4,1fr);
      border-left:2px solid #1c1c1c;border-top:1px solid #d8cfb8}
.bar{border-right:1px solid #b9b09a;border-bottom:1px solid #e5dcc6;
     padding:6px 4px;min-height:30px;font-size:13px;line-height:1.25;
     display:flex;flex-wrap:wrap;gap:4px;align-items:center;
     overflow-wrap:anywhere}
.bar b{font:italic 700 15px Georgia,serif}
.bar sup{font-size:9px;font-weight:700}
.bs{opacity:.6;font-size:11px}
.nc{color:#a89f88;font-style:italic;font-size:11px}
.cy{opacity:.5}
.clik{cursor:pointer}
.clik:active{background:#f5edd8}
.blk{margin-bottom:12px;border-left:3px solid #e5dcc6;padding-left:9px}
.bh{display:flex;flex-wrap:wrap;align-items:center;gap:7px;margin-bottom:5px}
.lb{font:800 12px -apple-system,sans-serif;color:#8a2b2b;
    border:1.5px solid #8a2b2b;border-radius:5px;padding:1px 7px;
    background:#fffdf6}
.meta{font-size:12px;color:#6b6455}
.meta b{color:#1c1c1c}
.pl{border:1px solid #c9bfa5;background:#fffdf6;color:#8a2b2b;border-radius:20px;
    font-size:11px;padding:1px 8px;cursor:pointer;line-height:1.5}
.var{margin:5px 0 0 12px;border-left:2px dotted #c9bfa5;padding-left:8px}
.var .vh{font-size:11px;color:#8a8371;margin:3px 0 2px;
         display:flex;align-items:center;gap:6px}
.lit .vh{color:#9c6a2b}
.flat{max-height:520px;overflow:auto;border:1px solid #eee4cd;border-radius:8px;
      padding:6px}
.occ{display:flex;gap:6px;align-items:flex-start;margin-bottom:5px}
.ol{font:700 10px -apple-system,sans-serif;color:#8a2b2b;min-width:26px;
    padding-top:6px}
.occ .grid{flex:1;min-width:0}
.tw{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;width:100%;font:13px ui-monospace,monospace}
td,th{padding:3px 6px;text-align:right;border-bottom:1px solid #eee4cd;
      white-space:nowrap}
th:first-child,td:first-child{text-align:left;white-space:normal;
      min-width:11em}
.bb{display:inline-block;height:9px;background:#8a2b2b;border-radius:2px;
    vertical-align:middle;opacity:.75}
a{color:#8a2b2b}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(128px,1fr));
       gap:9px;margin:12px 0}
.tile{background:#fffdf6;border:1px solid #e5dcc6;border-radius:11px;
      padding:9px 11px}
.tile .v{font:700 22px Georgia,serif;color:#8a2b2b}
.tile .k{font-size:11px;color:#8a8371}
"""


def page(sections_html: str, stats: list[dict]) -> str:
    W = sum(s["written"] for s in stats)
    P = sum(s["played"] for s in stats)
    L = sum(s["loose"] * s["played"] for s in stats) / max(1, P)
    worst = sorted(stats, key=lambda s: -s["ratio"])[:6]
    rows = "".join(
        f"<tr><td><a href='#s-{s['stem']}'>{html.escape(s['title'][:38])}</a></td>"
        f"<td>{s['written']}</td><td>{s['played']}</td>"
        f"<td>{s['ratio']:.2f}</td>"
        f"<td><span class='bb' style='width:{s['ratio'] * 88:.0f}px'></span></td>"
        f"</tr>"
        for s in sorted(stats, key=lambda s: s["ratio"]))
    wl = "".join(
        f"<li><b>{html.escape(s['title'][:40])}</b> — {s['ratio']:.2f} ; "
        f"{s['flat']} de ses {s['played']} mesures ne bouclent pas "
        f"({100 * (1 - s['loop_share']):.0f} %)</li>" for s in worst)
    return f"""<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Forme minimale</title>
<style>{CSS}</style>
<div class="wrap">
<h1>Forme minimale</h1>
<p class="lede">Pour chaque section : la plus petite boucle qui l'explique, plus
la liste de ses variantes. <b>La règle est double</b> — écrire le moins de
mesures possible, mais pouvoir rejouer la section <b>exactement</b>. Une boucle
n'est retenue que si elle rend à elle seule plus de la moitié de ce qu'on
entend ; sinon la section est écrite au long. Les deux propriétés sont des
tests (<code>tests/test_minimal_form.py</code>) : aucune suite de mesures n'est
dessinée deux fois, et la forme rejoue la grille d'origine mesure pour mesure.</p>
<p class="lede">Cliquez une mesure, une boucle ▶ ou une variante ▶ : l'audio
saute au passage. <b>Rien de ceci n'est branché sur le chart de l'app</b> —
c'est une mesure, pas un changement.</p>

<div class="tiles">
  <div class="tile"><div class="v">{P}</div><div class="k">mesures jouées</div></div>
  <div class="tile"><div class="v">{W}</div><div class="k">mesures écrites</div></div>
  <div class="tile"><div class="v">{W / P:.2f}</div><div class="k">taux moyen (écrit ÷ joué)</div></div>
  <div class="tile"><div class="v">÷{P / W:.1f}</div><div class="k">compression</div></div>
  <div class="tile"><div class="v">{L:.2f}</div><div class="k">taux si le décodage ne tremblait pas</div></div>
</div>

<section>
<h3>Où la règle d'or force à rester long</h3>
<p class="lede" style="margin:0 0 8px">Ces morceaux ne se compriment pas parce
qu'ils ne bouclent pas — soit la musique est vraiment à travers-composée, soit
le découpage en sections ou le décodage d'accords tremble assez pour qu'aucune
boucle ne survive. Les écrire court serait les écrire faux.</p>
<ul style="font-size:14px;color:#4a4437;margin:0;padding-left:20px">{wl}</ul>
</section>

<section>
<h3>Taux par morceau</h3>
<div class="tw"><table>
<tr><th>morceau</th><th>écrit</th><th>joué</th><th>taux</th><th></th></tr>
{rows}</table></div>
</section>

{sections_html}
</div>
<script>
let cur=null,tid=null;
function stop(){{if(tid)clearTimeout(tid);tid=null;if(cur)cur.pause();}}
document.addEventListener('click',e=>{{
  const el=e.target.closest('[data-t]');
  if(!el)return;
  const sec=el.closest('section'),a=sec&&sec.querySelector('audio');
  if(!a)return;
  const [t0,t1]=el.dataset.t.split(',').map(Number);
  stop();cur=a;a.currentTime=t0;a.play();
  tid=setTimeout(()=>a.pause(),Math.max(300,(t1-t0)*1000));
}});
</script>"""


def main() -> None:
    order = FIRST + [s for s in stems() if s not in FIRST]
    parts, stats = [], []
    for st in order:
        try:
            d = capture(st)
        except Exception as e:                       # noqa: BLE001
            print(f"  {st}: {type(e).__name__}: {e}", flush=True)
            continue
        body, stat = song_html(st, d)
        parts.append(body)
        stats.append(stat)
        print(f"  {st[:46]:48} {stat['written']:3d}/{stat['played']:3d} "
              f"= {stat['ratio']:.2f}", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page("".join(parts), stats), encoding="utf-8")
    W = sum(s["written"] for s in stats)
    P = sum(s["played"] for s in stats)
    print(f"\n{len(stats)} morceaux · {W}/{P} = {W / P:.3f}\n-> {OUT}")


if __name__ == "__main__":
    main()
