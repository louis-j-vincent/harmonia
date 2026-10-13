"""La liste du corpus : quels charts afficheraient une pastille de changement
de gamme (2026-10-13). Lecture seule de state/cache/charts.

    .venv/bin/python -m tools.page_zones_tonalite [--out docs/plots/…html]

Montre, morceau par morceau, TOUS les segments de `keySegments` (gardés ou
rendus à la zone d'avant) avec leur durée en mesures et leur `x3`, et les
pastilles qui en sortent — pour que Louis juge les faux positifs.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from harmonia.key_zones import PHRASE_BARS, _bar_starts, _n_bars, _signature, key_zones_for_model
from harmonia.settings import SETTINGS

FLAT = "C D♭ D E♭ E F G♭ G A♭ A B♭ B".split()
SHARP = "C C♯ D D♯ E F F♯ G G♯ A A♯ B".split()
FLAT_MAJ = {0, 1, 3, 5, 6, 8, 10}


def nom(tonic, mode):
    names = FLAT if _signature(tonic, mode) in FLAT_MAJ else SHARP
    return f"{names[tonic % 12]} {'mineur' if mode == 'minor' else 'majeur'}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/plots/zones_tonalite_2026-10-13.html")
    a = ap.parse_args()
    charts = Path(SETTINGS.charts_dir)
    meta_p = Path("state/human/chart_meta.json")
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    rows, n_tot, n_multi, n_pill = [], 0, 0, 0
    for p in sorted(charts.glob("*.json")):
        stem = p.stem
        if stem.startswith(("old_", "new_")) or stem.endswith("_essai"):
            continue                       # copies avant/après : même morceau
        try:
            m = json.loads(p.read_text())
        except Exception:
            continue
        n_tot += 1
        segs = m.get("keySegments") or []
        if len(segs) < 2:
            continue
        n_multi += 1
        zones = key_zones_for_model(m)
        n_pill += len(zones) > 1
        starts = _bar_starts(m.get("barGrid"), m.get("beatTimes"), m.get("bpb"))
        title = (meta.get(stem) or {}).get("title") or m.get("title") or stem
        seg_html = []
        for s in segs:
            nb = _n_bars(starts, s["t0"], s["t1"])
            ok = nb >= PHRASE_BARS
            seg_html.append(
                f"<li class='{'' if ok else 'court'}'>{html.escape(nom(s['tonic'], s['mode']))}"
                f" · {s['t0']:.0f}–{s['t1']:.0f} s · <b>{nb}</b> mesures · x3 {s.get('x3')}"
                f"{'' if ok else ' — <i>moins d’une phrase : rendu à la zone d’avant</i>'}</li>")
        pills = [("" if i == 0 else "→ ") + nom(z["tonic"], z["mode"]) + f" ({z['t0']:.0f} s, {z['bars']} mes.)"
                 for i, z in enumerate(zones)]
        rows.append(
            f"<tr><td><a href='http://100.89.209.63:7772/?open={html.escape(stem)}'>{html.escape(title)}</a>"
            f"<div class='s'>{html.escape(stem)} · ton du titre : {html.escape(nom(m['key']['tonic'], m['key']['mode']))}</div></td>"
            f"<td><ul>{''.join(seg_html)}</ul></td>"
            f"<td>{'<br>'.join(html.escape(x) for x in pills[1:]) or '<span class=s>aucune pastille</span>'}</td></tr>")
    page = f"""<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Changements de gamme</title>
<style>body{{font:15px -apple-system,system-ui,sans-serif;background:#f7f3e9;color:#1c1c1c;margin:16px;max-width:1000px}}
table{{border-collapse:collapse;width:100%}}td{{border-top:1px solid #b9b09a;padding:8px 6px;vertical-align:top}}
ul{{margin:0;padding-left:16px}}li.court{{color:#8a8371}}.s{{color:#8a8371;font-size:12px}}a{{color:#8a2b2b}}</style>
<h1>Changements de gamme — corpus</h1>
<p><b>Règle</b> : une pastille « → A♭ majeur » s'écrit au début d'une zone qui change la gamme
(l'armure : sol majeur → mi mineur n'en est pas une) et dure au moins une phrase ({PHRASE_BARS} mesures du chart).
Un segment plus court est rendu à la zone d'avant ; dans chaque zone, les accords s'écrivent avec ses ♭ ou ses ♯.</p>
<p>{n_tot} charts (copies avant/après exclues) · {n_multi} ont plus d'un segment de tonalité · <b>{n_pill}</b> afficheraient une pastille.</p>
<table><tr><th>morceau</th><th>segments détectés</th><th>pastilles</th></tr>{''.join(rows)}</table>"""
    Path(a.out).write_text(page, encoding="utf-8")
    print(a.out, n_tot, n_multi, n_pill)


if __name__ == "__main__":
    main()
