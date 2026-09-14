#!/usr/bin/env python3
"""scripts/page_songformer.py — ton découpage contre celui du modèle, à l'oreille.

Louis, 2026-08-18 : « lance le 1 » — faire tourner un modèle pré-entraîné de
structure sur ses morceaux et arbitrer À L'OREILLE, sans un tableau de scores.

Une bande par morceau, sur le même axe de temps :
  * en haut, SES sections à lui (celles qu'il a annotées) ;
  * en dessous, celles de SongFormer, avec leurs noms — intro, couplet,
    refrain, pont ;
  * en dessous encore, ce que le détecteur maison écrit dans le chart.
Chaque bloc se joue d'un tap. Aucun chiffre nulle part : c'est lui qui tranche.

    .venv/bin/python scripts/page_songformer.py
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SORTIE = REPO / "docs" / "plots" / "songformer.html"
ETAT = REPO / "harmonia_min" / "state"
SF = REPO / "docs" / "research_sessions" / "songformer.json"

COUL = {"intro": "#8a8371", "verse": "#3f6f8f", "chorus": "#a4462b",
        "bridge": "#3d6b47", "outro": "#8a6d1f", "inst": "#6b5b8a",
        "pre-chorus": "#b07a3a", "silence": "#c9c2b2"}
FR = {"intro": "intro", "verse": "couplet", "chorus": "refrain",
      "bridge": "pont", "outro": "outro", "inst": "instru",
      "pre-chorus": "pré-refrain", "silence": "silence"}
LET = ["#3f6f8f", "#a4462b", "#3d6b47", "#8a6d1f", "#6b5b8a", "#b07a3a",
       "#4a7c8c", "#8a4a6d"]


def bande(blocs, dur, hauteur=30, noms=True):
    """Une bande de blocs colorés sur l'axe du temps, cliquable."""
    out = []
    for b in blocs:
        x = b["t0"] / max(dur, 1e-6) * 100
        w = max((b["t1"] - b["t0"]) / max(dur, 1e-6) * 100, 0.4)
        out.append(
            f'<div class="bl" style="left:{x:.3f}%;width:{w:.3f}%;'
            f'background:{b["c"]}" data-t="{b["t0"]:.2f}" '
            f'title="{html.escape(b["nom"])} — {b["t0"]:.0f}s">'
            + (f'<span>{html.escape(b["nom"])}</span>' if noms else "") + "</div>")
    return f'<div class="bande" style="height:{hauteur}px">{"".join(out)}</div>'


def main() -> None:
    sf = json.loads(SF.read_text(encoding="utf-8"))
    blocs_html = []
    for stem, v in sf.items():
        p = ETAT / "charts" / f"min_{stem}.json"
        if not p.exists():
            continue
        m = json.loads(p.read_text(encoding="utf-8"))
        g = m.get("barGrid") or []
        if len(g) < 2:
            continue
        dur = float(g[-1])

        # 1. SES sections
        sien = None
        for d in ("sections", "sections_draft"):
            f = ETAT / d / f"{stem}.json"
            if f.exists():
                a = json.loads(f.read_text(encoding="utf-8"))
                if a.get("n") == m.get("nBars"):
                    sien = a["sections"]
        b_sien = []
        if sien:
            lettres = sorted({s["label"] for s in sien})
            for s in sien:
                b_sien.append({"t0": g[s["b0"]], "t1": g[min(s["b1"] + 1, len(g) - 1)],
                               "nom": s["label"],
                               "c": LET[lettres.index(s["label"]) % len(LET)]})

        # 2. SongFormer
        b_sf = [{"t0": s["start"], "t1": s["end"],
                 "nom": FR.get(s["label"], s["label"]),
                 "c": COUL.get(s["label"], "#999")} for s in v["brut"]]

        # 3. le detecteur maison
        b_det = []
        lettres = sorted({s["label"] for s in m.get("sections") or []})
        for s in m.get("sections") or []:
            for b0, b1 in s.get("barRanges") or []:
                b_det.append({"t0": g[b0], "t1": g[min(b1 + 1, len(g) - 1)],
                              "nom": s["label"],
                              "c": LET[lettres.index(s["label"]) % len(LET)]})

        rangs = ""
        if b_sien:
            rangs += f'<div class="lg">toi</div>{bande(b_sien, dur)}'
        rangs += f'<div class="lg">SongFormer</div>{bande(b_sf, dur)}'
        rangs += f'<div class="lg">le détecteur du chart</div>{bande(b_det, dur, 22, False)}'
        blocs_html.append(f'''
  <section data-audio="{m['audio_url']}" data-dur="{dur:.2f}">
    <h2>{html.escape(v.get("titre") or stem)}</h2>
    <audio controls preload="none"></audio>
    {rangs}
    <div class="regle"></div>
  </section>''')

    doc = f"""<!doctype html><html lang="fr"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SongFormer contre toi</title>
<style>
:root{{--fg:#1b1b19;--fond:#fbf7ec;--trait:#ddd6c7;--doux:#6b6558}}
*{{box-sizing:border-box}}
body{{margin:0 auto;padding:18px 14px 60px;background:var(--fond);color:var(--fg);
 font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,sans-serif;
 max-width:900px}}
h1{{font-size:20px;margin:0 0 6px}}
h2{{font-size:15px;margin:0 0 7px}}
.chapo{{color:var(--doux);margin:0 0 14px;font-size:13.5px}}
section{{background:#fff;border:1px solid var(--trait);border-radius:9px;
 padding:11px 13px;margin:0 0 12px}}
audio{{width:100%;height:32px;margin:0 0 8px;display:block}}
.lg{{font:600 9.5px -apple-system,sans-serif;color:var(--doux);
 text-transform:uppercase;letter-spacing:.05em;margin:5px 0 2px}}
.bande{{position:relative;width:100%;border-radius:5px;overflow:hidden;
 background:#f4efe3}}
.bl{{position:absolute;top:0;bottom:0;cursor:pointer;overflow:hidden;
 border-right:1px solid #fdfbf5;display:flex;align-items:center}}
.bl span{{font:600 10px -apple-system,sans-serif;color:#fff;padding:0 4px;
 white-space:nowrap;text-shadow:0 1px 1px rgba(0,0,0,.25)}}
.bl:active{{filter:brightness(1.15)}}
.regle{{position:relative;height:3px;margin-top:5px;background:#efe9db;
 border-radius:2px}}
.regle i{{position:absolute;top:-2px;width:2px;height:7px;background:#a4462b;
 border-radius:1px;left:0;opacity:0}}
</style>
<h1>SongFormer contre toi</h1>
<p class="chapo">Un modèle pré-entraîné qui sort les sections directement de
l'audio, sans rien apprendre de nous. Tape n'importe quel bloc pour l'écouter.</p>
{''.join(blocs_html)}
<script>
document.querySelectorAll("section[data-audio]").forEach(function(sec){{
  var el = sec.querySelector("audio"), dur = parseFloat(sec.dataset.dur);
  var tete = document.createElement("i");
  sec.querySelector(".regle").appendChild(tete);
  el.src = sec.dataset.audio;
  if (window.fetch) fetch(sec.dataset.audio)
    .then(function(r){{ return r.ok ? r.blob() : null; }})
    .then(function(b){{ if (b) el.src = URL.createObjectURL(b); }})
    .catch(function(){{}});
  sec.addEventListener("click", function(ev){{
    var c = ev.target.closest("[data-t]"); if (!c) return;
    try {{ el.currentTime = parseFloat(c.dataset.t); el.play(); }} catch(e){{}}
  }});
  /* timeupdate, jamais requestAnimationFrame (Safari iOS). */
  el.addEventListener("timeupdate", function(){{
    tete.style.left = (el.currentTime/dur*100) + "%";
    tete.style.opacity = 1;
  }});
}});
</script>
</html>
"""
    SORTIE.write_text(doc, encoding="utf-8")
    print(f"  http://100.89.209.63:7772/plots/{SORTIE.name}")


if __name__ == "__main__":
    main()
