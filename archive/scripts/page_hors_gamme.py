#!/usr/bin/env python3
"""scripts/page_hors_gamme.py — les accords qui sortent de la gamme.

Louis, 2026-08-19, devant le `Db/F` et le `Gb+` que la pile a ecrits sur Bora
Bora : « en effet musx se trompe sur ceux la, comment est-ce qu'on pourrait
faire pour avoir des accords + carres ? »

CE QUE CETTE PAGE VERIFIE, avant d'ecrire une ligne de correcteur. L'hypothese
est que les accords qui sonnent faux sont, presque toujours, ceux qui SORTENT
DE LA GAMME du morceau. Si elle tient, le levier est evident : demander plus de
preuves a un accord hors gamme qu'a un accord dedans. Si elle ne tient pas, ce
levier ne sert a rien et il vaut mieux le savoir maintenant.

La page ecrit donc tout le chart, mesure par mesure, et marque chaque accord
dont une note sort de la gamme. Chaque mesure se joue d'un tap : c'est en
ecoutant les mesures marquees qu'on voit si elles sont bien les fautives.

    .venv/bin/python scripts/page_hors_gamme.py [--morceaux N]
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

ETAT = REPO / "harmonia_min" / "state"
AUD = REPO / "docs" / "audio"
SORTIE = REPO / "docs" / "plots" / "hors_gamme.html"

NOTES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

# Les degres d'une gamme, en demi-tons depuis la tonique. En mineur on admet
# la 7e haussee (mineur harmonique) et la 6e haussee (melodique) : ce sont des
# notes de la tonalite, pas des emprunts — sans elles, tout V7 d'un morceau
# mineur serait declare hors gamme, ce qui viderait la mesure de son sens.
GAMME = {
    "major": {0, 2, 4, 5, 7, 9, 11},
    "minor": {0, 2, 3, 5, 7, 8, 9, 10, 11},
}


def _diatoniques(tonic: int, mode: str) -> set[int]:
    """Les degres sur lesquels la tonalite pose ses accords."""
    return {(tonic + d) % 12 for d in GAMME.get(mode or "major", GAMME["major"])}


def statut(c, tonic: int, mode: str) -> tuple[str, list[str]]:
    """(« dedans » | « fonction » | « etranger », notes hors gamme).

    TROIS NIVEAUX, PAS DEUX — c'est le resultat du premier essai. Marquer
    simplement « hors gamme » sonnait l'alarme sur tous les E7 et F7 de Bora
    Bora, qui sont des DOMINANTES SECONDAIRES : hors de la gamme par leur
    tierce haussee, mais au coeur du langage de la tonalite (V de V, V de IV).
    Les confondre avec un `Gb+` ou un `Db/F` viderait la mesure de son sens.

      dedans     toutes les notes sont dans la gamme ;
      fonction   hors gamme, mais l'accord est une dominante (7 ou majeur) qui
                 resout une quinte plus bas sur un degre de la tonalite ;
      etranger   ni l'un ni l'autre — c'est la que sont les fautes de musx.
    """
    if c.get("nc"):
        return "dedans", []
    from harmonia_min.labels import chord_pcs
    pcs = set(chord_pcs(int(c["root"]), c.get("q") or ""))
    if c.get("bass", -1) >= 0:
        pcs.add(int(c["bass"]) % 12)
    ok = GAMME.get(mode or "major", GAMME["major"])
    dehors = [NOTES[p] for p in sorted(pcs) if (p - tonic) % 12 not in ok]
    if not dehors:
        return "dedans", []
    q = (c.get("q") or "")
    cible = (int(c["root"]) + 5) % 12          # la quinte en dessous
    dominante = (q in ("", "7", "9", "13", "7sus4", "sus4")
                 or q.startswith("7"))
    if dominante and cible in _diatoniques(tonic, mode):
        return "fonction", dehors
    return "etranger", dehors


def txt(c) -> str:
    if c.get("nc"):
        return "N.C."
    t = NOTES[c["root"] % 12] + (c.get("q") or "")
    if c.get("bass", -1) >= 0 and c["bass"] != c["root"]:
        t += "/" + NOTES[c["bass"] % 12]
    return t


def _tonique_par_mesure(chart, bars, grid, audio, defaut: int) -> list[int]:
    """La tonique locale, mesure par mesure. Se rabat sur celle du chart."""
    ch = ((chart.get("prompter") or {}).get("chords")) or []
    if not ch:
        return [defaut] * len(bars)
    try:
        from harmonia_min.harmonic_key import tonic_track
        from harmonia_min.nnls_features import extract_bothchroma
        arr, times = extract_bothchroma(audio)
        segs = tonic_track(ch, arr, times)
    except Exception as exc:
        print(f"    tonique locale indisponible ({type(exc).__name__}), "
              "on garde celle du chart")
        return [defaut] * len(bars)
    out = []
    for b in range(len(bars)):
        t = 0.5 * (grid[b] + grid[min(b + 1, len(grid) - 1)])
        ton = defaut
        for sg in segs:
            i0, i1 = sg["i0"], min(sg["i1"], len(ch)) - 1
            if i1 < i0:
                continue
            if ch[i0]["t0"] <= t <= ch[i1]["t1"]:
                ton = int(sg["tonic"])
                break
        out.append(ton)
    return out


def main() -> None:
    from harmonia_min.soudure import accords_par_mesure

    vise = int(sys.argv[sys.argv.index("--morceaux") + 1]) \
        if "--morceaux" in sys.argv else 4
    voulus = sys.argv[sys.argv.index("--stems") + 1].split(",") \
        if "--stems" in sys.argv else None

    blocs, faits = [], 0
    for p in sorted((ETAT / "charts").glob("min_*.json")):
        if faits >= vise:
            break
        stem = p.stem[len("min_"):]
        if voulus and stem not in voulus:
            continue
        try:
            chart = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        grid = chart.get("barGrid") or []
        audio = AUD / Path(chart.get("audio_url") or "x").name
        k = chart.get("key") or {}
        if not audio.exists() or len(grid) < 3 or "tonic" not in k:
            continue
        mode = str(k.get("mode") or "major")
        bars = accords_par_mesure(chart)
        # LA TONIQUE LOCALE, PAS CELLE DU CHART. Bora Bora monte d'un demi-ton
        # au deux tiers : mesuré contre la seule tonique de départ, la moitié
        # du morceau ressortait « hors gamme » et la page ne disait plus rien.
        # `tonic_track` suit le centre tonal accord par accord — c'est la même
        # question que la transposition du repli, posée un cran plus haut.
        tonique = _tonique_par_mesure(chart, bars, grid, audio,
                                      int(k["tonic"]))
        cases, n_out = [], 0
        for b, bar in enumerate(bars):
            if b + 1 >= len(grid):
                break
            morceaux_, pire = [], "dedans"
            for c in bar or []:
                st, dehors = statut(c, tonique[b], mode)
                if st == "dedans":
                    morceaux_.append(html.escape(txt(c)))
                    continue
                if st == "etranger":
                    pire = "etranger"
                elif pire == "dedans":
                    pire = "fonction"
                morceaux_.append(
                    f'<em class="{st}" title="hors gamme : '
                    f'{", ".join(dehors)}">{html.escape(txt(c))}</em>')
            n_out += 1 if pire == "etranger" else 0
            cases.append(
                f'<span class="m {pire}" data-t="{grid[b]:.2f}">'
                f"<i>{b + 1}</i>" + (" ".join(morceaux_) or "·") + "</span>")
        faits += 1
        blocs.append(
            f'<section data-audio="{chart["audio_url"]}" '
            f'data-dur="{float(grid[-1]):.2f}">'
            f'<h2>{html.escape(chart.get("title") or stem)} '
            f'<small>{html.escape(chart.get("keyName") or "")}</small></h2>'
            f'<audio controls preload="none"></audio>'
            f'<div class="grille">{"".join(cases)}</div>'
            f'<div class="regle"><i></i></div></section>')
        print(f"  {chart.get('title')} — {n_out} mesure(s) étrangère(s) "
              f"sur {len(cases)}")

    doc = f"""<!doctype html><html lang="fr"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Les accords qui sortent de la gamme</title>
<style>
:root{{--fg:#1b1b19;--fond:#fbf7ec;--trait:#ddd6c7;--doux:#6b6558;--out:#a4462b}}
*{{box-sizing:border-box}}
body{{margin:0 auto;padding:18px 14px 60px;background:var(--fond);color:var(--fg);
 font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,sans-serif;
 max-width:900px}}
h1{{font-size:20px;margin:0 0 6px}}
h2{{font-size:15px;margin:0 0 8px}}
h2 small{{font-weight:400;color:var(--doux);font-size:12.5px;margin-left:6px}}
.chapo{{color:var(--doux);margin:0 0 14px;font-size:13.5px}}
section{{background:#fff;border:1px solid var(--trait);border-radius:9px;
 padding:11px 13px;margin:0 0 12px}}
audio{{width:100%;height:32px;margin:0 0 10px;display:block}}
.grille{{display:flex;flex-wrap:wrap;gap:4px}}
.m{{flex:0 0 auto;min-width:74px;border:1px solid #ece5d6;border-radius:5px;
 padding:4px 7px 3px;font-size:12.5px;cursor:pointer;background:#fdfbf5;
 position:relative}}
.m i{{display:block;font-style:normal;font-size:9px;color:#b6ae9d;
 line-height:1.1}}
.m:active{{background:#f3ead8}}
.m.fonction{{background:#fbf6ea;border-color:#e6d9b8}}
.m.etranger{{background:#fdeee7;border-color:#e8bda8}}
.m em{{font-style:normal;font-weight:700}}
.m em.fonction{{color:#8a6d1f;border-bottom:2px dotted #d8c187}}
.m em.etranger{{color:var(--out);border-bottom:2px solid #e0a58c}}
.regle{{position:relative;height:3px;margin-top:9px;background:#efe9db;
 border-radius:2px}}
.regle i{{position:absolute;top:-2px;width:2px;height:7px;background:var(--out);
 border-radius:1px;left:0;opacity:0}}
</style>
<h1>Les accords qui sortent de la gamme</h1>
<p class="chapo">Une hypothèse à vérifier à l'oreille avant d'écrire un
correcteur : <b>les accords qui sonnent faux sont ceux qui sortent du langage
de la tonalité</b>. Tout le chart est écrit ici, mesure par mesure, avec la
tonique <i>locale</i> — celle qui suit les modulations.</p>
<p class="chapo"><b style="color:#8a6d1f">En jaune</b>, hors de la gamme mais
<b>dans le langage</b> : une dominante qui résout une quinte plus bas sur un
degré de la tonalité (le E7 de Bora Bora, V de V). Ces accords-là sont normaux.
<b style="color:#a4462b">En rouge</b>, les <b>étrangers</b> : ni dans la gamme,
ni une dominante fonctionnelle. C'est là que devraient être les fautes.
Tape une mesure pour l'entendre.</p>
{''.join(blocs)}
<script>
document.querySelectorAll("section[data-audio]").forEach(function(sec){{
  var el = sec.querySelector("audio"), dur = parseFloat(sec.dataset.dur);
  var tete = sec.querySelector(".regle i");
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
    tete.style.left = (el.currentTime/dur*100) + "%"; tete.style.opacity = 1;
  }});
}});
</script>
</html>
"""
    SORTIE.write_text(doc, encoding="utf-8")
    print(f"\n  http://100.89.209.63:7772/plots/{SORTIE.name}")


if __name__ == "__main__":
    main()
