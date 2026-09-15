"""La page d'arbitrage du repli : les mesures où le consensus a réécrit un
accord que la première passe entendait avec confiance.

Le repli empile les passages d'une section et écrit, à chaque position,
l'accord du consensus. Quand un passage joue autre chose — une substitution,
un D-7 là où les autres jouent G7 — le consensus l'efface, en silence. Cette
page liste ces mesures (première passe à confiance ≥ SEUIL, fondamentale
absente de ce que le chart écrit), et donne à Louis de quoi trancher :

  * ce que le modèle a entendu ICI (accords + confiance) et ce que le chart
    écrit à la place ;
  * ce que les autres passages jouent à la même position de la boucle ;
  * un bouton d'écoute pour chacun (la mesure, la mesure avec son contexte,
    chaque passage frère), l'audio servi par l'app ;
  * trois verdicts par mesure (le passage a raison / le consensus a raison /
    ni l'un ni l'autre), gardés dans le navigateur et copiables en bloc.

Ce que la page ne fait PAS : décider. Un verdict de Louis devient une loi
dans `folding.py` (mémoire « corrections → automation »), pas une rustine
par morceau.

    python -m tools.arbitrage_repli              # → <reports>/arbitrage_repli.html
    python -m tools.arbitrage_repli --seuil 0.6
"""
from __future__ import annotations

import argparse
import html
import json
import logging
from pathlib import Path

from harmonia.settings import SETTINGS

NOTES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
SEUIL = 0.70


def chord_txt(c, conf=True) -> str:
    if c.get("nc"):
        return "N.C."
    t = NOTES[c["root"]] + c["q"]
    if c.get("bass", -1) >= 0 and c["bass"] != c["root"]:
        t += "/" + NOTES[c["bass"]]
    if c.get("carry"):
        t = "(" + t + ")"
    if conf and c.get("c") is not None and not c.get("carry"):
        t += f" {c['c']:.2f}"
    return t


def _onsets(bar):
    return [c for c in bar if not c.get("carry") and not c.get("nc")]


def raw_bars(audio: Path, chart: dict) -> list:
    """Les mesures de la PREMIÈRE passe (avant sections et repli)."""
    from harmonia.pipeline import analyze_steps
    from harmonia.server.jobs import bar1_for
    gen = analyze_steps(audio, title=chart.get("title") or "", file_key=chart["file"],
                        audio_url=chart.get("audio_url") or "",
                        bar1_time=bar1_for(chart["file"]))
    try:
        for kind, model in gen:
            if kind == "raw":
                return json.loads(json.dumps(model))["sections"][0]["bars"]
    finally:
        gen.close()
    return []


def served_bars(chart: dict) -> dict[int, tuple[str, list]]:
    """{mesure de chanson: (lettre, accords écrits)} — le premier passage
    qui couvre la mesure fait foi (c'est celui que l'app affiche)."""
    out = {}
    for s in chart.get("sections") or []:
        for b0, b1 in s.get("barRanges") or []:
            for i, bar in enumerate(s.get("bars") or []):
                if b0 + i <= b1 and (b0 + i) not in out:
                    out[b0 + i] = (s.get("label") or "?", bar)
    return out


def _lettre(label: str) -> str:
    return label.rstrip("′″‴'")


def freres(chart: dict, b: int, label: str, P: int) -> list[int]:
    """Les mesures à la même position de boucle, dans les autres passages."""
    ranges = [tuple(r) for s in chart.get("sections") or []
              if _lettre(s.get("label") or "") == _lettre(label)
              for r in (s.get("barRanges") or [])]
    home = next(((c0, c1) for c0, c1 in ranges if c0 <= b <= c1), None)
    if home is None or not P:
        return []
    pos = (b - home[0]) % P
    out = []
    for c0, c1 in ranges:
        for bb in range(c0, c1 + 1):
            if bb != b and (bb - c0) % P == pos:
                out.append(bb)
    # Six suffisent pour entendre ce que « les autres » jouent ; les plus
    # proches dans le temps d'abord (même son, même mixage), en ordre de
    # lecture ensuite.
    return sorted(sorted(set(out), key=lambda bb: abs(bb - b))[:6])


def collecter(seuil: float) -> list[dict]:
    items = []
    for p in sorted(SETTINGS.charts_dir.glob("min_*.json")):
        chart = json.loads(p.read_text(encoding="utf-8"))
        stem = Path(chart.get("audio_url") or "").stem or p.stem.removeprefix("min_")
        audio = SETTINGS.audio_dir / f"{stem}.m4a"
        fold = chart.get("fold") or {}
        changed = {b: L for L, rep in fold.items() if isinstance(rep, dict)
                   and not rep.get("reason") for b in (rep.get("changed") or [])}
        if not changed or not audio.exists():
            continue
        try:
            raw = raw_bars(audio, chart)
        except Exception as exc:                           # noqa: BLE001
            logging.warning("%s : première passe indisponible (%s)", p.stem, exc)
            continue
        served = served_bars(chart)
        grid = chart["barGrid"]
        for b, L in sorted(changed.items()):
            if b >= len(raw) or b not in served:
                continue
            label, fin = served[b]
            surs = [c for c in _onsets(raw[b]) if c.get("c", 0) >= seuil]
            # Une tenue compte : « (Gb) » à la place de « Gb », c'est le même
            # accord pour qui lit — l'attaque a disparu, pas l'harmonie. On ne
            # garde que les mesures où la FONDAMENTALE entendue n'est plus là.
            roots_fin = {c["root"] for c in fin if not c.get("nc")}
            if not surs or not ({c["root"] for c in surs} - roots_fin):
                continue
            rep = fold.get(L) or {}
            P = rep.get("period") or 0
            sib = freres(chart, b, label, P)
            variants = set(rep.get("variants") or [])
            items.append({
                "key": p.stem, "stem": stem, "title": chart.get("title") or stem,
                "audio": chart.get("audio_url") or f"/audio/{audio.name}",
                "bar": b, "label": label, "period": P,
                "t0": grid[b], "t1": grid[b + 1] if b + 1 < len(grid) else grid[b] + 2,
                "ctx0": grid[max(0, b - 1)],
                "ctx1": grid[min(len(grid) - 1, b + 2)],
                "entendu": " ".join(chord_txt(c) for c in raw[b]),
                "ecrit": " ".join(chord_txt(c, conf=False) for c in fin) or "·",
                "freres": [{"bar": bb, "t0": grid[bb],
                            "t1": grid[bb + 1] if bb + 1 < len(grid) else grid[bb] + 2,
                            "entendu": " ".join(chord_txt(c) for c in raw[bb]) if bb < len(raw) else "?",
                            "variante": bb in variants} for bb in sib],
            })
    return items


CSS = """
*{box-sizing:border-box}
body{margin:0;padding:14px 12px 90px;background:#faf6ec;color:#2c2820;
 font:15px/1.45 system-ui,sans-serif;max-width:760px;margin-inline:auto}
h1{font-size:19px;margin:0 0 6px} h2{font-size:16px;margin:26px 0 2px}
.lede{background:#fff7df;border:1px solid #e8d9a8;border-radius:10px;
 padding:10px 12px;font-size:14px;margin:10px 0}
.note{color:#8a8371;font-size:12.5px}
.card{border:1px solid #ddd3b8;border-radius:12px;background:#fffdf7;padding:10px 12px;margin:14px 0}
.row{display:flex;flex-wrap:wrap;gap:6px;margin:6px 0;align-items:center}
.lab{font:600 11px system-ui;color:#8a8371;min-width:92px;text-transform:uppercase;letter-spacing:.04em}
.chip{border:1px solid #d8cfb4;border-radius:8px;background:#fff;padding:5px 9px;cursor:pointer;
 font-variant-numeric:tabular-nums}
.chip.on{background:#8a2b2b;border-color:#8a2b2b;color:#fff}
.chip.heard{border-color:#a9c488;background:#eef4e6}
.chip.written{border-color:#c9a26b;background:#f7ecd6}
.chip.var{opacity:.6}
.open a{text-decoration:none;font:600 13px system-ui;border-radius:10px;padding:8px 12px;
 border:1px solid #d8cfb4;background:#fff;color:#2c2820}
.verdict{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px}
.verdict button{border:1px solid #d8cfb4;border-radius:10px;background:#fff;padding:8px 10px;
 font:600 13px system-ui;color:#2c2820;cursor:pointer}
.verdict button.sel{background:#2c2820;border-color:#2c2820;color:#fff}
textarea{width:100%;min-height:120px;font:13px/1.4 ui-monospace,monospace;border:1px solid #d8cfb4;
 border-radius:10px;padding:8px;background:#fff}
.copy{border:1px solid #8a2b2b;border-radius:10px;background:#8a2b2b;color:#fff;padding:9px 13px;
 font:600 13px system-ui;cursor:pointer;margin:8px 0}
"""

JS = """
const au=document.getElementById('au');let stopAt=null,cur=null;
au.addEventListener('timeupdate',()=>{if(stopAt!=null&&au.currentTime>=stopAt){
 au.pause();stopAt=null;if(cur){cur.classList.remove('on');cur=null;}}});
function play(el,src,t0,t1){
 if(cur===el&&!au.paused){au.pause();return;}
 if(cur)cur.classList.remove('on');cur=el;el.classList.add('on');
 const go=()=>{try{au.currentTime=t0;}catch(e){}stopAt=t1;
  au.play().catch(()=>{el.classList.remove('on');cur=null;});};
 if(au.getAttribute('src')!==src){au.setAttribute('src',src);
  au.addEventListener('loadedmetadata',go,{once:true});au.load();}else go();}
const KEY='arbitrage_repli';let V={};
try{V=JSON.parse(localStorage.getItem(KEY)||'{}');}catch(e){V={};}
function verdict(id,val,btn){V[id]=val;try{localStorage.setItem(KEY,JSON.stringify(V));}catch(e){}
 for(const b of btn.parentNode.querySelectorAll('button'))b.classList.toggle('sel',b===btn);render();}
function render(){const lines=[];for(const c of document.querySelectorAll('.card')){
 const id=c.dataset.id;const v=V[id];if(v)lines.push(id+' → '+v);}
 document.getElementById('out').value=lines.join('\\n')||'(aucun verdict encore)';}
function copier(){const t=document.getElementById('out');t.select();
 try{navigator.clipboard.writeText(t.value);}catch(e){document.execCommand('copy');}}
window.addEventListener('DOMContentLoaded',()=>{for(const c of document.querySelectorAll('.card')){
 const v=V[c.dataset.id];if(v)for(const b of c.querySelectorAll('.verdict button'))
  b.classList.toggle('sel',b.dataset.v===v);}render();});
"""


def page(items: list[dict], seuil: float) -> str:
    B = ["<h1>Arbitrage du repli — l'accord entendu contre l'accord écrit</h1>",
         f"<div class=lede>{len(items)} mesure(s) où la première passe a entendu un accord "
         f"avec une confiance ≥ {seuil:.2f} et où le consensus du repli a écrit une autre "
         "fondamentale. Pour chacune : ce que le modèle a entendu <b>ici</b>, ce que le chart "
         "<b>écrit</b>, et ce que les <b>autres passages</b> jouent à la même position. "
         "Clique pour écouter, puis tranche. Les verdicts restent dans ce navigateur ; "
         "le bloc en bas se copie d'un geste.</div>",
         "<p class=note>Lecture : « (C) » = accord tenu depuis la mesure d'avant ; le nombre "
         "après un accord = confiance du modèle ; « variante » = passage que la pile a rejeté.</p>"]
    for it in items:
        sid = f"{it['stem']} mes.{it['bar'] + 1}"
        a = it["audio"]
        B.append(f"<div class=card data-id=\"{html.escape(sid)}\">")
        B.append(f"<h2>{html.escape(it['title'])} — {html.escape(it['label'])}, mesure "
                 f"{it['bar'] + 1}</h2>")
        B.append(f"<p class=note>boucle de {it['period']} mesures · "
                 f"{it['t0']:.1f} s</p>")
        B.append(f"<div class=row><span class=lab>entendu ici</span>"
                 f"<div class='chip heard' onclick=\"play(this,'{a}',{it['t0']:.2f},{it['t1']:.2f})\">"
                 f"▶ {html.escape(it['entendu'])}</div>"
                 f"<div class=chip onclick=\"play(this,'{a}',{it['ctx0']:.2f},{it['ctx1']:.2f})\">"
                 f"▶ avec contexte</div></div>")
        B.append(f"<div class=row><span class=lab>écrit</span>"
                 f"<div class='chip written' onclick=\"play(this,'{a}',{it['t0']:.2f},{it['t1']:.2f})\">"
                 f"{html.escape(it['ecrit'])}</div></div>")
        if it["freres"]:
            B.append("<div class=row><span class=lab>autres passages</span>")
            for f in it["freres"]:
                cls = "chip var" if f["variante"] else "chip"
                B.append(f"<div class='{cls}' onclick=\"play(this,'{a}',{f['t0']:.2f},{f['t1']:.2f})\">"
                         f"▶ mes. {f['bar'] + 1} : {html.escape(f['entendu'])}"
                         f"{' · variante' if f['variante'] else ''}</div>")
            B.append("</div>")
        B.append(f"<div class=open><a href='/?open={html.escape(it['key'])}'>ouvrir dans l'app</a></div>")
        B.append("<div class=verdict>"
                 "<button data-v='entendu' onclick=\"verdict(this.closest('.card').dataset.id,'entendu',this)\">"
                 "ce passage a raison</button>"
                 "<button data-v='écrit' onclick=\"verdict(this.closest('.card').dataset.id,'écrit',this)\">"
                 "le consensus a raison</button>"
                 "<button data-v='ni' onclick=\"verdict(this.closest('.card').dataset.id,'ni',this)\">"
                 "ni l'un ni l'autre</button></div>")
        B.append("</div>")
    B.append("<h2>Tes verdicts</h2><textarea id=out readonly></textarea>"
             "<button class=copy onclick='copier()'>Copier le bloc</button>"
             "<p class=note>Colle-le dans la conversation : un verdict devient une loi du repli, "
             "pas une rustine par morceau.</p>")
    return ("<meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
            "<title>Arbitrage du repli</title><style>" + CSS + "</style><body>" + "".join(B) +
            "<audio id=au preload=auto playsinline></audio><script>" + JS + "</script>")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seuil", type=float, default=SEUIL)
    ap.add_argument("--reports", type=Path, default=SETTINGS.reports_dir)
    a = ap.parse_args(argv)
    logging.disable(logging.INFO)
    items = collecter(a.seuil)
    a.reports.mkdir(parents=True, exist_ok=True)
    out = a.reports / "arbitrage_repli.html"
    out.write_text(page(items, a.seuil), encoding="utf-8")
    (a.reports / "arbitrage_repli.json").write_text(
        json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ {out} · {len(items)} mesure(s)")
    for it in items:
        print(f"  {it['title'][:30]:30s} {it['label']} mes. {it['bar'] + 1:3d} | "
              f"{it['entendu']} → {it['ecrit']} | {len(it['freres'])} frère(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
