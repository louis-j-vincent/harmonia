"""La file des morceaux à annoter en sections, classée par ce qu'ils apportent.

Louis, 2026-08-12 : « donne-moi d'autres morceaux à annoter ». Ses 18
annotations à la main (`harmonia_min/state/sections/`) sont la SEULE
référence de sections du projet : c'est contre elles que se mesurent le
détecteur (0,782) et tout ce qui en dépend. Agrandir cette référence est
donc du travail de fond, pas de la saisie.

CLASSEMENT — on met devant ce qui rapporte le plus, c'est-à-dire les
morceaux où notre détection est la plus FRAGILE, parce que c'est là qu'une
vérité change quelque chose :

  1. les passages très courts (< 4 mesures) : une section d'une ou deux
     mesures est presque toujours une frontière ratée, pas une section ;
  2. beaucoup de lettres pour peu de musique : le détecteur a fragmenté ;
  3. les lettres que le repli refuse : sans frontières justes il ne peut
     pas empiler, et c'est exactement le levier qu'on vient de livrer.

Chaque ligne ouvre le chart dans l'app, outil sections prêt à l'emploi.

    python scripts/a_annoter.py           -> /reports/a_annoter.html
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CHARTS = REPO / "harmonia_min" / "state" / "charts"
GT = REPO / "harmonia_min" / "state" / "sections"
DRAFT = REPO / "harmonia_min" / "state" / "sections_draft"
REPORTS = REPO / "harmonia_min" / "state" / "reports"


def score(row):
    """Plus c'est haut, plus l'annotation rapporte."""
    frag = row["short"] * 3                      # passages d'moins de 4 mesures
    dense = max(0, row["letters"] - row["bars"] // 20)   # trop de lettres
    refus = row["n_fold"] - row["n_acc"]
    return frag + 2 * dense + refus


def collect():
    done = {p.stem for p in GT.glob("*.json")}
    drafts = {p.stem for p in DRAFT.glob("*.json")} if DRAFT.exists() else set()
    rows = []
    for p in sorted(CHARTS.glob("min_*.json")):
        m = json.loads(p.read_text(encoding="utf-8"))
        stem = m["file"].removeprefix("min_")
        if stem in done:
            continue
        secs = m.get("sections") or []
        occ = [(s.get("label"), r) for s in secs
               for r in (s.get("barRanges") or [])]
        lens = [r[1] - r[0] + 1 for _, r in occ]
        fold = m.get("fold") or {}
        acc = sum(1 for v in fold.values()
                  if isinstance(v, dict) and "n_obs" in v)
        rows.append({"stem": stem, "title": m.get("title") or stem,
                     "bars": m.get("nBars") or 0, "n_occ": len(occ),
                     "letters": len({l for l, _ in occ}),
                     "short": sum(1 for L in lens if L < 4),
                     "shortest": min(lens) if lens else 0,
                     "n_acc": acc, "n_fold": len(fold),
                     "draft": stem in drafts})
    rows.sort(key=lambda r: -score(r))
    return rows, len(done)


def why(r):
    bits = []
    if r["short"]:
        bits.append(f"{r['short']} passage(s) de moins de 4 mesures "
                    f"(le plus court : {r['shortest']})")
    if r["letters"] > max(2, r["bars"] // 20):
        bits.append(f"{r['letters']} lettres pour {r['bars']} mesures — "
                    "probablement fragmenté")
    if r["n_fold"] - r["n_acc"] > 0:
        bits.append(f"le repli refuse {r['n_fold'] - r['n_acc']} lettre(s) "
                    f"sur {r['n_fold']}")
    return " · ".join(bits) or "structure d'apparence saine — utile comme "\
                               "contre-exemple"


CSS = """
*{box-sizing:border-box}
body{margin:0;padding:14px 10px 80px;background:#faf6ec;color:#2c2820;
 font:15px/1.45 system-ui,sans-serif;max-width:760px;margin-inline:auto}
h1{font-size:19px;margin:0 0 8px}
.lede{background:#fff7df;border:1px solid #e8d9a8;border-radius:10px;
 padding:10px 12px;font-size:14px;margin:10px 0}
.card{border:1px solid #e0d8c2;border-radius:12px;background:#fff;
 padding:11px 12px;margin:9px 0}
.t{font:700 15px system-ui;margin-bottom:2px}
.w{font:italic 12.5px Georgia,serif;color:#8a8371;margin-bottom:8px}
a.go{display:inline-block;text-decoration:none;background:#8a2b2b;color:#fff;
 border-radius:10px;padding:9px 14px;font:600 13.5px system-ui}
.badge{display:inline-block;font:600 11px system-ui;color:#6b654f;
 background:#f1ead6;border-radius:5px;padding:2px 7px;margin-left:6px}
.note{color:#8a8371;font-size:12.5px}
"""


def main(argv):
    rows, n_done = collect()
    B = ["<h1>À annoter — la file</h1>",
         f"<div class=lede>Tes annotations de sections sont la <b>seule</b> "
         f"référence du projet : le détecteur (0,782) et le repli se mesurent "
         f"contre elles. Tu en as <b>{n_done}</b> ; voici les "
         f"<b>{len(rows)}</b> morceaux restants, les plus utiles d'abord. "
         "Chaque bouton ouvre le chart : mets-toi en <b>Annoter</b>, "
         "« Marquer les sections », choisis une lettre, appuie un instant sur "
         "la première mesure et glisse jusqu'à la dernière.</div>",
         "<p class=note>Le classement met devant ce qui est le plus "
         "probablement FAUX aujourd'hui — passages d'une ou deux mesures, "
         "trop de lettres pour la durée, lettres que le repli refuse. C'est "
         "là qu'une vérité change quelque chose.</p>"]
    for r in rows:
        d = "<span class=badge>brouillon en cours</span>" if r["draft"] else ""
        B.append(
            f"<div class=card><div class=t>{html.escape(r['title'][:44])}{d}"
            f"</div><div class=w>{html.escape(why(r))}</div>"
            f"<div class=note>{r['bars']} mesures · {r['n_occ']} passages · "
            f"{r['letters']} lettres · repli {r['n_acc']}/{r['n_fold']}</div>"
            f"<div style='margin-top:9px'><a class=go href='/?open=min_"
            f"{html.escape(r['stem'])}'>ouvrir et annoter</a></div></div>")
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / "a_annoter.html"
    out.write_text("<meta charset=utf-8><meta name=viewport "
                   "content='width=device-width,initial-scale=1'>"
                   "<title>À annoter</title><style>" + CSS + "</style><body>"
                   + "".join(B), encoding="utf-8")
    print(f"→ {out} · {n_done} annotés, {len(rows)} à faire")
    for r in rows[:6]:
        print(f"   {r['title'][:34]:34s} — {why(r)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
