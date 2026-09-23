"""T5-prep — diff mesure par mesure entre le chart baseline et un levier.

Lit les JSON produits par `run_lever.py` (bars/barSpans/barGrid par section)
et rend, par morceau : le texte de chaque mesure AVANT/APRÈS, la liste des
mesures qui diffèrent, et le temps de la mesure (pour le lien audio).

Reprend la logique de `tools/avant_apres.py::song_bars` (bars→{mesure:
texte} via barSpans, pas via les blocs repliés) sans importer ce module
(qui republie dans la bibliothèque — pas voulu ici, comparaison hors-ligne).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

NOTES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]


def chord_txt(c):
    if c.get("nc"):
        return "N"
    t = NOTES[c["root"]] + c["q"]
    if c.get("bass", -1) >= 0 and c["bass"] != c["root"]:
        t += "/" + NOTES[c["bass"]]
    return ("(" + t + ")") if c.get("carry") else t


def song_bars(entry):
    grid = entry.get("barGrid") or []
    n = max(0, len(grid) - 1)
    out = {}
    for sec in entry.get("sections") or []:
        for r, bar in enumerate(sec.get("bars") or []):
            spans = sec.get("barSpans") or []
            cand = spans[r] if r < len(spans) else []
            for sp in (cand or []):
                if not sp:
                    continue
                lo = None
                for k in range(n):
                    if grid[k] <= sp[0] + 1e-4:
                        lo = k
                if lo is not None and lo not in out:
                    out[lo] = " ".join(chord_txt(c) for c in bar) or "·"
    return out, grid


def diff_song(baseline_entry, lever_entry):
    a, grid_a = song_bars(baseline_entry)
    b, grid_b = song_bars(lever_entry)
    rows = []
    for k in sorted(set(a) & set(b)):
        if a[k] != b[k]:
            t0 = grid_a[k] if k < len(grid_a) else None
            rows.append({"bar": k + 1, "before": a[k], "after": b[k], "t0": t0})
    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))
    return rows, only_a, only_b


def main():
    scratch = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    baseline = json.loads((scratch / "baseline.json").read_text())
    summary = {}
    for lever in ("small0", "fold1", "fold3"):
        p = scratch / f"{lever}.json"
        if not p.exists():
            continue
        data = json.loads(p.read_text())
        summary[lever] = {}
        for stem, base_entry in baseline.items():
            lev_entry = data.get(stem)
            if lev_entry is None:
                continue
            rows, only_a, only_b = diff_song(base_entry, lev_entry)
            summary[lever][stem] = {
                "nBars_before": base_entry.get("nBars"),
                "nBars_after": lev_entry.get("nBars"),
                "n_beats_before": base_entry.get("n_beats"),
                "n_beats_after": lev_entry.get("n_beats"),
                "changed_bars": rows,
                "n_changed": len(rows),
                "bars_only_before": only_a,
                "bars_only_after": only_b,
                "total_s_before": base_entry.get("total_s"),
                "total_s_after": lev_entry.get("total_s"),
                "backend_timings_before": base_entry.get("backend_timings"),
                "backend_timings_after": lev_entry.get("backend_timings"),
            }
            n = summary[lever][stem]["n_changed"]
            print(f"{lever}/{stem}: {n} mesures changées "
                  f"(nBars {base_entry.get('nBars')}->{lev_entry.get('nBars')}, "
                  f"beats {base_entry.get('n_beats')}->{lev_entry.get('n_beats')})")
    out_path.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
