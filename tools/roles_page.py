"""La page qui montre, terme à terme, ce que l'équivalence de rôle (`harmonia.roles`)
replierait dans les piles du repli — sur des morceaux que Louis connaît.

Pour chaque lettre jouée plusieurs fois, position par position : les
mesures que la première passe a entendues différemment d'un passage à
l'autre, regroupées d'abord à l'identique (ce que le repli voit
aujourd'hui), puis par rôle (ce que la brique proposerait de replier).
Chaque accord distinct s'écoute — l'oreille tranche si G7 et Db7 tiennent
bien le même rôle ICI.

Rien n'est branché dans le repli : c'est la page de la règle 2 du projet
(« screen the premise cheaply ») pour la brique de Louis du 2026-09-15.

    python -m tools.roles_page                     # 4 morceaux par défaut
    python -m tools.roles_page min_4JkIs37a2JE …
"""
from __future__ import annotations

import html
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

from harmonia.roles import NOTES, equivalent, parse_key_name, role, role_name
from harmonia.settings import SETTINGS
from tools.arbitrage_repli import raw_bars, CSS, JS

DEFAUT = ["min_4JkIs37a2JE", "min_0DdCoNbbRvQ", "min_mayer_hawthorne_the_walk",
          "min_DksSPZTZES0", "min_T64BgKEL-Sw"]


def chord_txt(c) -> str:
    t = NOTES[c["root"]] + (c.get("q") or "")
    if c.get("bass", -1) >= 0 and c["bass"] != c["root"]:
        t += "/" + NOTES[c["bass"]]
    return t


def _onsets(bar):
    return [c for c in bar if not c.get("carry") and not c.get("nc")]


def bar_sig(bar) -> tuple:
    return tuple(chord_txt(c) for c in _onsets(bar))


def bars_equivalent(ba, bb, key) -> bool:
    oa, ob = _onsets(ba), _onsets(bb)
    return len(oa) == len(ob) and all(equivalent(x, y, key) for x, y in zip(oa, ob))


def positions(chart: dict, letter: str, rep: dict) -> list[list[int]]:
    """Les membres de chaque position de la boucle, comme le repli les
    empile (sans les fins gardées hors pile)."""
    P = rep.get("period") or 0
    if not P:
        return []
    out = [[] for _ in range(P)]
    for s in chart["sections"]:
        if s["label"].rstrip("′″‴'") != letter:
            continue
        for b0, b1 in s["barRanges"]:
            for b in range(b0, b1 + 1):
                if P < (b1 - b0 + 1) and b == b1:
                    continue                          # fin hors pile
                out[(b - b0) % P].append(b)
    return out


def analyser(key_chart: str) -> dict | None:
    p = SETTINGS.charts_dir / f"{key_chart}.json"
    chart = json.loads(p.read_text(encoding="utf-8"))
    stem = Path(chart.get("audio_url") or "").stem or key_chart.removeprefix("min_")
    audio = SETTINGS.audio_dir / f"{stem}.m4a"
    if not audio.exists():
        return None
    raw = raw_bars(audio, chart)
    key = parse_key_name(chart.get("keyName"))
    grid = chart["barGrid"]
    lettres = []
    for letter, rep in (chart.get("fold") or {}).items():
        if not isinstance(rep, dict) or rep.get("reason"):
            continue
        pos = positions(chart, letter, rep)
        cards = []
        for k, members in enumerate(pos):
            members = [b for b in members if b < len(raw) and _onsets(raw[b])]
            if len(members) < 2:
                continue
            strict = Counter(bar_sig(raw[b]) for b in members)
            if len(strict) < 2:
                continue                                # pas de variation
            # classes de rôle : union des signatures strictes équivalentes
            reps_ = list(strict)                        # une signature par classe stricte
            cls = []                                    # [[sig, …], …]
            for sig in reps_:
                b_sig = next(b for b in members if bar_sig(raw[b]) == sig)
                for c in cls:
                    b_c = next(b for b in members if bar_sig(raw[b]) == c[0])
                    if bars_equivalent(raw[b_sig], raw[b_c], key):
                        c.append(sig)
                        break
                else:
                    cls.append([sig])
            cards.append({"k": k, "n": len(members), "strict": strict, "classes": cls,
                          "exemple": {sig: next(b for b in members if bar_sig(raw[b]) == sig)
                                      for sig in strict},
                          "roles": {sig: " · ".join(role_name(role(c, key), key)
                                                    for c in _onsets(raw[next(b for b in members if bar_sig(raw[b]) == sig)]))
                                    for sig in strict}})
        if cards:
            lettres.append({"letter": letter, "P": rep.get("period"),
                            "reps": sum(1 for s in chart["sections"]
                                        if s["label"].rstrip("′″‴'") == letter
                                        for _ in s["barRanges"]),
                            "cards": cards})
    return {"key": key_chart, "title": chart.get("title") or stem, "keyName": chart.get("keyName"),
            "audio": chart.get("audio_url") or f"/audio/{audio.name}", "grid": grid, "lettres": lettres}


def page(songs: list[dict]) -> str:
    B = ["<h1>Rôles dans la cadence — ce que la brique replierait</h1>",
         "<div class=lede>Pour chaque section jouée plusieurs fois, position par position, "
         "les mesures que le modèle a entendues <b>différemment</b> d'un passage à l'autre. "
         "D'abord telles quelles (ce que le repli voit aujourd'hui), puis regroupées par "
         "<b>rôle</b> : même fondamentale et même famille, basse étrangère = sus, même triton = "
         "même dominante, V majeur = dominante. Une classe = ce qu'on pourrait replier. "
         "Clique un accord pour entendre sa mesure.</div>",
         "<p class=note>Le nombre après un accord = passages où on l'entend. « ✓ » = la brique "
         "réunit tout en un seul rôle ; « ✗ » = elle ne réunit rien de plus que l'identique.</p>"]
    for s in songs:
        B.append(f"<h2>{html.escape(s['title'])} <span class=note>· {html.escape(s['keyName'] or '?')}</span></h2>")
        if not s["lettres"]:
            B.append("<p class=note>aucune variation entre passages</p>")
        for L in s["lettres"]:
            for c in L["cards"]:
                merged = len(c["classes"]) < len(c["strict"])
                tout = len(c["classes"]) == 1
                flag = "✓ un seul rôle" if tout else ("≈ rapproche" if merged else "✗ rôles distincts")
                B.append(f"<div class=card data-id=\"{html.escape(s['key'])} {html.escape(L['letter'])} pos{c['k'] + 1}\">")
                B.append(f"<h3 style='margin:2px 0 4px;font-size:15px'>{html.escape(L['letter'])} ×{L['reps']}, "
                         f"boucle de {L['P']}, position {c['k'] + 1} <span class=note>· {c['n']} passages · {flag}</span></h3>")
                for ci, cl in enumerate(c["classes"]):
                    B.append(f"<div class=row><span class=lab>{'rôle ' + str(ci + 1) if len(c['classes']) > 1 else 'rôle'}</span>")
                    for sig in cl:
                        b = c["exemple"][sig]
                        t0, t1 = s["grid"][b], s["grid"][b + 1] if b + 1 < len(s["grid"]) else s["grid"][b] + 2
                        B.append(f"<div class='chip{' heard' if len(cl) > 1 else ''}' "
                                 f"onclick=\"play(this,'{s['audio']}',{t0:.2f},{t1:.2f})\">"
                                 f"▶ {html.escape(' '.join(sig) or '·')} ×{c['strict'][sig]} "
                                 f"<span class=note>mes. {b + 1}</span></div>")
                    B.append(f"<span class=note>{html.escape(c['roles'][cl[0]])}</span></div>")
                B.append("<div class=verdict>"
                         "<button data-v='même rôle' onclick=\"verdict(this.closest('.card').dataset.id,'même rôle',this)\">"
                         "même rôle, à replier</button>"
                         "<button data-v='distincts' onclick=\"verdict(this.closest('.card').dataset.id,'distincts',this)\">"
                         "rôles distincts</button>"
                         "<button data-v='bruit' onclick=\"verdict(this.closest('.card').dataset.id,'bruit',this)\">"
                         "détection fausse</button></div>")
                B.append("</div>")
    B.append("<h2>Tes verdicts</h2><textarea id=out readonly></textarea>"
             "<button class=copy onclick='copier()'>Copier le bloc</button>"
             "<p class=note>Colle-le dans la conversation : c'est ce qui décide quelles règles "
             "entrent dans le repli.</p>")
    return ("<meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
            "<title>Rôles dans la cadence</title><style>" + CSS + "</style><body>" + "".join(B) +
            "<audio id=au preload=auto playsinline></audio><script>"
            + JS.replace("const KEY='arbitrage_repli'", "const KEY='roles_cadence'") + "</script>")


def main(argv=None) -> int:
    keys = (argv if argv is not None else sys.argv[1:]) or DEFAUT
    logging.disable(logging.INFO)
    songs = [s for s in (analyser(k) for k in keys) if s]
    out = SETTINGS.reports_dir / "roles_cadence.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page(songs), encoding="utf-8")
    print(f"→ {out}")
    for s in songs:
        for L in s["lettres"]:
            for c in L["cards"]:
                print(f"  {s['title'][:22]:22s} {L['letter']}×{L['reps']} P={L['P']} pos {c['k'] + 1}: "
                      f"{len(c['strict'])} écritures → {len(c['classes'])} rôle(s) | "
                      + " ; ".join(" | ".join(" ".join(sig) or "·" for sig in cl) for cl in c["classes"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
