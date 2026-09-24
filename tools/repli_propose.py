"""tools/repli_propose.py — le repliement proposé, calculé, pas dessiné à la
main.

Né du morceau San Francisco Street (2026-09-24) : quatre fois de suite, une
version dessinée à la main (`docs/plots/san_francisco_street_repli_propose_
2026-09-24.html`) s'est trompée, et les quatre fois pour la MÊME raison —
elle lisait le texte affiché par le chart (le gabarit d'une lettre, les
frontières de section) au lieu de calculer la structure depuis l'audio.
Ce script fait ce que la main a fini par faire, mécaniquement :

  1. décode CHAQUE mesure indépendamment (`musx.decode_window`, jamais le
     gabarit d'une lettre répétée — voir sa docstring pour le bug que ça
     évite : `soudure.sections_pour_chart` copie les accords d'UN SEUL
     passage sur toutes les occurrences d'une lettre quand le repli interne
     a été refusé) ;
  2. cherche les cellules qui se répètent, SANS frontière de lettre — une
     cellule peut se retrouver n'importe où dans le morceau, pas seulement
     dans les autres occurrences de sa propre lettre. C'est exactement ce
     qui manque à `remplir_par_algo` / `nommer` (known_issues.md, « un
     brique gelée ne reconnaît plus ses reprises »).
     ESSAYÉ D'ABORD `section_tool.find_repeats` (l'outil de PRODUCTION,
     finger-swipe + « find repeats ») et ABANDONNÉ pour cet usage : sa
     recherche transposée, calibrée pour un humain qui a DÉJÀ désigné une
     occurrence réelle, chaîne des correspondances tournées parasites quand
     on l'amorce à l'aveugle (mesuré sur ce morceau : tout le morceau
     fusionne en UNE famille de 27 mesures). `transpose=False` réduisait le
     dégât sans le faire disparaître, et son score n'est pas une cosinus
     bornée (des « trouvailles » à 1,7 / 4,9). À la place : fenêtres
     glissantes de P mesures, vecteurs chord-tone CONCATÉNÉS (pas moyennés
     — moyenner perd l'ordre, et confond « Dmaj Dmaj Em Em » avec
     « Dmaj Gmaj Em A », mesuré aussi), cosinus entre fenêtres non
     chevauchantes, seuil CELL_MATCH_THRESH (0,90 — les vraies répétitions
     de ce morceau mesurent 0,97–1,0, les quasi-manqués 0,5–0,8, marge
     large). Groupement glouton : à chaque tour, la fenêtre encore libre qui
     a le PLUS de partenaires devient une famille, jusqu'à épuisement.
  3. repère les « fins alternatives » : dans une cellule qui revient N fois,
     une occurrence dont seule la DERNIÈRE position diverge (le reste colle)
     n'est pas une nouvelle lettre — c'est un 1st/2nd ending. Le seuil
     réutilise `folding.PERIOD_MIN_SCORE` (0,80), jamais un nouveau chiffre
     inventé pour l'occasion.

Ce que ça NE fait PAS : ça ne réécrit aucun chart, aucune section — c'est un
diagnostic, comme `arbitrage_repli` et `chart_simule`. Une lecture proposée,
à arbitrer par Louis avant de toucher `soudure.py` / `folding.py`. Et ce que
ça ne résout PAS encore : la recherche essaie des cellules de 4 mesures
d'abord, puis 2 sur ce qui reste — jamais 8 ; un pont qui serait un vrai
bloc irréductible de 8 mesures (pas 2×4, contrairement à celui de San
Francisco Street) resterait NON détecté comme cellule, et retomberait en
mesures « uniques ». Validé sur UN morceau (San Francisco Street) : les 4
familles et les 2 fins alternatives retrouvées reproduisent exactement la
lecture à l'oreille de Louis — pas encore mesuré sur le reste du corpus.

    python -m tools.repli_propose --chart min_9zEl-FQLI4A
    python -m tools.repli_propose --chart min_9zEl-FQLI4A --out /tmp/x.html
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from harmonia import musx as _musx
from harmonia.sections.similarity import harmonic_vectors
from harmonia.settings import SETTINGS

#: reused from folding.py, never re-tuned here (project rule: one calibrated
#: threshold per question, not a second one for a second tool) -- used for
#: the alternate-ending check only (a per-BAR-position question, the same
#: grain folding.py's own stacking already answers).
from harmonia.folding import PERIOD_MIN_SCORE

#: cell-vs-cell whole-window cosine threshold (concatenated chord-tone
#: vectors, NOT the same scale as PERIOD_MIN_SCORE). Measured on San
#: Francisco Street: true repeats of the same cell score 0.97-1.0, the
#: closest false positives (different cell, overlapping vocabulary) score
#: 0.5-0.8 -- wide margin, 0.90 sits in the middle of it.
CELL_MATCH_THRESH = 0.90
CELL_LENS = (4, 2)              # tried in this order on what's still unclaimed
LOWCONF = 0.45                  # raw-decode confidence flag, same spirit as
                                 # the hand page's cue, not a folding.py constant
FAMILY_COLORS = [
    ("#e8734a", "#fff"), ("#3f7cac", "#fff"), ("#4a9a6a", "#fff"),
    ("#8a5fb0", "#fff"), ("#c9a23f", "#111"), ("#5aa7a1", "#fff"),
    ("#b0558a", "#fff"), ("#7d8a3f", "#fff"),
]
UNIQUE_COLOR = ("#3a3f4a", "#cfd3da")

_SHORT_QUAL = {"maj": "", "min": "m", "sus4": "sus4", "sus2": "sus2", "dim": "dim", "aug": "aug"}


def _short_chord(label: str) -> str:
    if label == "N":
        return "N.C."
    root, _, qual = label.partition(":")
    return root + _SHORT_QUAL.get(qual, qual)


def _load(chart_key: str) -> dict:
    p = SETTINGS.charts_dir / f"{chart_key}.json"
    return json.loads(p.read_text(encoding="utf-8"))


def _raw_decode(triad: np.ndarray, grid: list) -> list[tuple[str, float]]:
    n = len(grid) - 1
    return [_musx.decode_window(triad, grid[i], grid[i + 1]) for i in range(n)]


def _cluster_at_period(V: np.ndarray, unclaimed: set[int], P: int) -> list[list[int]]:
    """[[bar_start, ...], ...] -- one list of non-overlapping window starts
    per family found at exactly this period, among still-unclaimed bars.

    Greedy max-matches: each round, the still-fully-free window with the
    MOST same-period partners (cosine >= CELL_MATCH_THRESH, non-overlapping)
    becomes a family; repeat until no window has any partner left. This is
    what replaced `find_repeats`-based seeding -- see the module docstring
    for why (phase and transposition both broke it there)."""
    n = V.shape[0]

    def window_free(s: int) -> bool:
        return all((s + k) in unclaimed for k in range(P))

    def vec(s: int) -> np.ndarray:
        w = V[s:s + P].reshape(-1)
        return w / max(float(np.linalg.norm(w)), 1e-9)

    groups = []
    while True:
        candidates = [s for s in range(0, n - P + 1) if window_free(s)]
        if not candidates:
            break
        vecs = {s: vec(s) for s in candidates}
        best_s, best_matches = None, []
        for s in candidates:
            matches = [s2 for s2 in candidates
                      if abs(s2 - s) >= P and float(vecs[s] @ vecs[s2]) >= CELL_MATCH_THRESH]
            if len(matches) > len(best_matches):
                best_s, best_matches = s, matches
        if best_s is None or not best_matches:
            break
        group = sorted({best_s, *best_matches})
        resolved, taken = [], set()
        for g in group:                                    # drop any that now overlap
            if any((g + k) in taken for k in range(P)):
                continue
            resolved.append(g)
            taken.update(range(g, g + P))
        for g in resolved:
            for k in range(P):
                unclaimed.discard(g + k)
        groups.append(resolved)
    return groups


def _discover_families(V: np.ndarray, n: int) -> list[dict]:
    """[{cell_bars, members: [(b0,b1), ...], unique}] -- `_cluster_at_period`
    run at each of CELL_LENS in turn (biggest first is wrong here: a 4-bar
    cell inside an 8-bar span would get grabbed whole and never split, so
    smallest-explains-most-first, matching section_tool's own "le plus petit
    qui tient" rule), then whatever bars are still unclaimed become their
    own one-off ("unique") blocks -- contiguous runs kept together."""
    unclaimed = set(range(n))
    families = []
    for P in CELL_LENS:
        for group in _cluster_at_period(V, unclaimed, P):
            members = [(s, s + P - 1) for s in group]
            families.append({"cell_bars": P, "members": members, "unique": False})

    leftover = sorted(unclaimed)
    i = 0
    while i < len(leftover):
        j = i
        while j + 1 < len(leftover) and leftover[j + 1] == leftover[j] + 1:
            j += 1
        b0, b1 = leftover[i], leftover[j]
        families.append({"cell_bars": b1 - b0 + 1, "members": [(b0, b1)], "unique": True})
        i = j + 1
    return families


def _flag_alt_endings(fam: dict, V: np.ndarray) -> dict:
    """Marks members whose LAST bar-position alone diverges from the
    family's own median at that position -- a 1st/2nd-ending variant, not a
    new family. Needs >=3 members to have a meaningful median."""
    members = fam["members"]
    P = fam["cell_bars"]
    fam["alt_ending"] = [False] * len(members)
    if fam["unique"] or len(members) < 3:
        return fam
    # median vector at each within-cell position, across all members
    pos_vecs = [[] for _ in range(P)]
    for (b0, b1) in members:
        for k in range(P):
            bi = b0 + k
            if bi <= b1:
                pos_vecs[k].append(V[bi])
    medians = [np.median(np.stack(vs), axis=0) if vs else None for vs in pos_vecs]
    medians = [v / np.maximum(np.linalg.norm(v), 1e-9) if v is not None else None
              for v in medians]
    for i, (b0, b1) in enumerate(members):
        sims = []
        for k in range(P):
            bi = b0 + k
            if bi > b1 or medians[k] is None:
                sims.append(None)
                continue
            sims.append(float(V[bi] @ medians[k]))
        early = [s for s in sims[:-1] if s is not None]
        last = sims[-1]
        if last is not None and early and last < PERIOD_MIN_SCORE and min(early) >= PERIOD_MIN_SCORE:
            fam["alt_ending"][i] = True
    return fam


def build(chart_key: str) -> dict:
    chart = _load(chart_key)
    grid = chart["barGrid"]
    n = len(grid) - 1
    stem = Path(chart.get("audio_url") or "").stem
    audio = SETTINGS.audio_dir / f"{stem}.m4a"
    if not audio.exists():
        raise SystemExit(f"audio introuvable: {audio}")

    triad = _musx.frame_posteriors(audio)[0]
    raw = _raw_decode(triad, grid)
    V = harmonic_vectors(triad, grid)
    S = V @ V.T

    families = _discover_families(V, n)
    families = [_flag_alt_endings(f, V) for f in families]
    families.sort(key=lambda f: f["members"][0][0])

    return {
        "chart": chart, "grid": grid, "n": n, "stem": stem,
        "audio_url": chart.get("audio_url") or f"/audio/{stem}.m4a",
        "raw": raw, "V": V, "S": S, "families": families,
    }


#: one stable color per production section-letter (cycles), so repeated
#: occurrences of the same letter read as the same color -- the point of
#: this view is to compare the app's OWN letters/chords against the raw
#: independent decode, not to show a family clustering of my own.
LETTER_COLORS = FAMILY_COLORS


def _page_brut(data: dict) -> str:
    """Le chart brut : chords tels que l'app les affiche AUJOURD'HUI (gabarit
    de lettre compris, avec son bug de copie s'il y en a un), le decoupage de
    sections tel qu'il existe EN PROD, et par-dessus le decode brut
    independant + sa confiance -- pour voir d'un coup d'oeil si le texte
    affiche colle a ce que musx a vraiment entendu, mesure par mesure."""
    chart, grid, n = data["chart"], data["grid"], data["n"]
    raw = data["raw"]
    title = chart.get("title") or data["stem"]

    NOTE = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

    def chord_name(c):
        s = NOTE[c["root"] % 12] + (c["q"] or "")
        bass = c.get("bass", -1)
        if bass is not None and bass >= 0 and bass != c["root"]:
            s += "/" + NOTE[bass % 12]
        return s

    full = {}
    for s in chart.get("sections") or []:
        for b0, b1 in s.get("barRanges") or []:
            for i in range(b0, b1 + 1):
                src_bar = s["bars"][(i - b0) % len(s["bars"])]
                full[i] = " ".join(chord_name(c) for c in src_bar) if src_bar else "·"

    letter_color = {}
    blocks = []  # (b0, b1, label, occurrence_index)
    for s in chart.get("sections") or []:
        label = s.get("label") or "?"
        if label not in letter_color:
            letter_color[label] = LETTER_COLORS[len(letter_color) % len(LETTER_COLORS)]
        for oi, (b0, b1) in enumerate(s.get("barRanges") or []):
            blocks.append((b0, b1, label, oi + 1, len(s.get("barRanges") or [])))
    blocks.sort(key=lambda b: b[0])

    rows = []
    for b0, b1, label, oi, n_occ in blocks:
        bg, fg = letter_color[label]
        cells = []
        for i in range(b0, min(b1, n - 1) + 1):
            t0, t1 = grid[i], grid[i + 1]
            rlab, rconf = raw[i]
            lowconf = " lowconf" if rconf < LOWCONF else ""
            cells.append(
                f'<div class="bar{lowconf}" data-t0="{t0:.3f}" data-t1="{t1:.3f}" '
                f'title="chart: {full.get(i, "?")} | raw decode: {rlab} (conf {rconf:.2f})" '
                f'tabindex="0" role="button">'
                f'<span class="idx">{i+1}</span><span class="chord">{full.get(i, "?")}</span>'
                f'<span class="lab">{rconf:.2f}</span></div>'
            )
        occ_txt = f" -- occurrence {oi}/{n_occ}" if n_occ > 1 else ""
        rows.append(
            f'<div class="seg" style="background:{bg};color:{fg}">'
            f'<div class="seg-note">mes. {b0+1}-{b1+1} &middot; lettre {label}{occ_txt}</div>'
            f'<div class="bars">{"".join(cells)}</div></div>'
        )

    legend_html = "".join(
        f'<span class="chip" style="background:{bg};color:{fg}">{lab}</span>'
        for lab, (bg, fg) in letter_color.items()
    )
    n_low = sum(1 for _, c in raw if c < LOWCONF)

    return f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>{title} — chart brut</title>
<style>
  body {{ font-family: -apple-system, Helvetica, Arial, sans-serif; background:#111318; color:#eee;
         margin:0; padding:24px 16px 80px; }}
  h1 {{ font-size:1.3rem; margin:0 0 4px; }}
  .sub {{ color:#9aa3af; margin:0 0 8px; font-size:0.95rem; }}
  .legend {{ margin:16px 0 12px; display:flex; flex-wrap:wrap; gap:8px; }}
  .chip {{ padding:4px 10px; border-radius:6px; font-size:0.85rem; }}
  #audiostatus {{ font-size:0.85rem; color:#9aa3af; margin:0 0 20px; min-height:1.2em; }}
  #audiostatus.ready {{ color:#7fd0a0; }}
  #audiostatus.err {{ color:#e08a7a; }}
  .seg {{ border-radius:10px; padding:10px 12px; margin-bottom:10px; }}
  .seg-note {{ font-size:0.82rem; opacity:0.9; margin-bottom:6px; }}
  .bars {{ display:flex; flex-wrap:wrap; gap:4px; }}
  .bar {{ background:rgba(0,0,0,0.18); border-radius:6px; padding:4px 8px; min-width:48px;
         display:flex; flex-direction:column; align-items:center; cursor:pointer;
         border:1.5px solid transparent; transition:transform .08s, border-color .08s; }}
  .bar.lowconf {{ border:1.5px dashed rgba(255,255,255,0.55); }}
  .bar:hover {{ border-color:rgba(255,255,255,0.75); }}
  .bar:active {{ transform:scale(0.95); }}
  .bar.playing {{ border-color:#fff; background:rgba(255,255,255,0.3); }}
  .idx {{ font-size:0.62rem; opacity:0.75; }}
  .chord {{ font-weight:600; font-size:0.88rem; }}
  .lab {{ font-size:0.6rem; opacity:0.7; font-variant-numeric:tabular-nums; }}
</style>
</head>
<body>
<h1>{title} — chart brut (tel qu'affiche aujourd'hui + decode independant)</h1>
<p class="sub">{n} mesures &middot; accord = ce que le chart affiche (gabarit de la lettre, copie
  sur chaque occurrence) &middot; petit chiffre sous chaque case = confiance du decode brut
  independant (musx, sur l'audio reel de CETTE mesure) &middot; {n_low} mesure(s) sous {LOWCONF}
  (pointilles). Survole une case pour comparer les deux. Clique pour ecouter.</p>
<p id="audiostatus">clique une mesure pour charger l'audio</p>

<div class="legend">{legend_html}</div>

{"".join(rows)}

<script>
(function(){{
  var AUDIO_URL = {json.dumps(data["audio_url"])};
  var statusEl = document.getElementById('audiostatus');
  var audioEl = null, blobPromise = null, stopTimer = null, curBar = null;
  function setStatus(txt, cls){{ statusEl.textContent = txt; statusEl.className = cls || ''; }}
  function ensureAudio(){{
    if(!blobPromise){{
      setStatus('chargement de l\\'audio…');
      audioEl = new Audio();
      blobPromise = fetch(AUDIO_URL)
        .then(function(r){{ if(!r.ok) throw new Error('HTTP ' + r.status); return r.blob(); }})
        .then(function(b){{
          return new Promise(function(resolve){{
            audioEl.src = URL.createObjectURL(b);
            var ready = function(){{ setStatus('audio pret', 'ready'); resolve(); }};
            if(audioEl.readyState >= 1) ready();
            else audioEl.addEventListener('loadedmetadata', ready, {{once:true}});
          }});
        }})
        .catch(function(e){{ setStatus('audio indisponible : ' + e, 'err'); throw e; }});
    }}
    return blobPromise;
  }}
  function playBar(t0, t1, cellEl){{
    if(curBar) curBar.classList.remove('playing');
    curBar = cellEl;
    cellEl.classList.add('playing');
    clearTimeout(stopTimer);
    ensureAudio().then(function(){{
      try{{ audioEl.currentTime = t0; }}catch(e){{}}
      audioEl.play().catch(function(e){{ setStatus('lecture refusee : ' + e, 'err'); }});
      var durMs = Math.max(150, (t1 - t0) * 1000);
      stopTimer = setTimeout(function(){{ audioEl.pause(); cellEl.classList.remove('playing'); }}, durMs);
    }});
  }}
  document.querySelectorAll('.bar').forEach(function(cell){{
    var t0 = parseFloat(cell.dataset.t0), t1 = parseFloat(cell.dataset.t1);
    cell.addEventListener('click', function(){{ playBar(t0, t1, cell); }});
    cell.addEventListener('keydown', function(ev){{
      if(ev.key === 'Enter' || ev.key === ' '){{ ev.preventDefault(); playBar(t0, t1, cell); }}
    }});
  }});
}})();
</script>
</body>
</html>
"""


def _page_folded(data: dict) -> str:
    """Le chart REPLIÉ : chaque cellule écrite une fois, en ordre de
    lecture -- pas la vue diagnostic groupée par famille. Un badge x N sur
    le premier passage ; les passages suivants se réduisent à une pastille
    cliquable (le contenu ne change pas, pourquoi le réécrire) ; une fin
    alternative reste écrite en entier, comme un 1st/2nd ending -- jamais
    remplacée par une moyenne, toujours le vrai contenu de CE passage-là."""
    chart, grid, n = data["chart"], data["grid"], data["n"]
    raw, families = data["raw"], data["families"]
    title = chart.get("title") or data["stem"]

    fam_color = {}
    for i, fam in enumerate(f for f in families if not f["unique"]):
        fam_color[id(fam)] = FAMILY_COLORS[i % len(FAMILY_COLORS)]

    entries = []
    for f in families:
        alt = f.get("alt_ending", [False] * len(f["members"]))
        for mi, (b0, b1) in enumerate(f["members"]):
            entries.append((b0, b1, f, mi, alt[mi] if mi < len(alt) else False))
    entries.sort(key=lambda e: e[0])

    seen = set()
    blocks = []
    for b0, b1, f, mi, is_alt in entries:
        n_mem = len(f["members"])
        first = (id(f) not in seen)
        if not f["unique"]:
            seen.add(id(f))
        blocks.append((b0, b1, f, mi, n_mem, first, is_alt))

    html_blocks = []
    for b0, b1, f, mi, n_mem, first, is_alt in blocks:
        color = UNIQUE_COLOR if f["unique"] else fam_color[id(f)]
        bg, fg = color
        t0, t1 = grid[b0], grid[b1 + 1] if b1 + 1 < len(grid) else grid[b0] + 2
        if f["unique"] or first or is_alt:
            cells = []
            for i in range(b0, b1 + 1):
                rlab, rconf = raw[i]
                lowconf = " lowconf" if rconf < LOWCONF else ""
                cells.append(
                    f'<div class="bar{lowconf}" data-t0="{grid[i]:.3f}" data-t1="{grid[i+1]:.3f}" '
                    f'title="raw decode: {rlab} (conf {rconf:.2f})">'
                    f'<span class="chord">{_short_chord(rlab)}</span></div>'
                )
            badge = ""
            if f["unique"]:
                badge = f'<span class="tag">unique</span>'
            elif is_alt:
                badge = f'<span class="tag alt">fin alternative -- passage {mi+1}/{n_mem}</span>'
            else:
                badge = f'<span class="tag">&times;{n_mem}</span>'
            html_blocks.append(
                f'<div class="cell" style="background:{bg};color:{fg}" '
                f'data-t0="{t0:.3f}" data-t1="{t1:.3f}" tabindex="0" role="button">'
                f'<div class="cell-head">mes.{b0+1}-{b1+1} {badge}</div>'
                f'<div class="cell-bars">{"".join(cells)}</div></div>'
            )
        else:
            html_blocks.append(
                f'<div class="pill" style="background:{bg};color:{fg}" '
                f'data-t0="{t0:.3f}" data-t1="{t1:.3f}" tabindex="0" role="button">'
                f'passage {mi+1}/{n_mem} &middot; mes.{b0+1}-{b1+1}</div>'
            )

    n_fam = sum(1 for f in families if not f["unique"])
    n_written = sum(1 for _, _, f, _, _, first, is_alt in blocks if f["unique"] or first or is_alt)
    n_collapsed = len(blocks) - n_written

    return f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>{title} — chart replie</title>
<style>
  body {{ font-family: -apple-system, Helvetica, Arial, sans-serif; background:#111318; color:#eee;
         margin:0; padding:24px 16px 80px; }}
  h1 {{ font-size:1.3rem; margin:0 0 4px; }}
  .sub {{ color:#9aa3af; margin:0 0 8px; font-size:0.95rem; }}
  #audiostatus {{ font-size:0.85rem; color:#9aa3af; margin:0 0 20px; min-height:1.2em; }}
  #audiostatus.ready {{ color:#7fd0a0; }}
  #audiostatus.err {{ color:#e08a7a; }}
  .form {{ display:flex; flex-wrap:wrap; gap:8px; align-items:flex-start; }}
  .cell {{ border-radius:10px; padding:8px 10px; cursor:pointer; min-width:120px;
          border:1.5px solid transparent; }}
  .cell:hover {{ border-color:rgba(255,255,255,0.65); }}
  .cell.playing {{ border-color:#fff; }}
  .cell-head {{ font-size:0.72rem; opacity:0.85; margin-bottom:6px; display:flex; gap:6px; align-items:center; }}
  .tag {{ background:rgba(0,0,0,0.28); border-radius:5px; padding:1px 6px; font-weight:700; }}
  .tag.alt {{ background:#7a5a00; color:#ffe9b0; }}
  .cell-bars {{ display:flex; gap:3px; flex-wrap:wrap; }}
  .cell-bars .bar {{ background:rgba(0,0,0,0.2); border-radius:5px; padding:3px 6px; }}
  .bar.lowconf {{ border:1px dashed rgba(255,255,255,0.5); }}
  .chord {{ font-weight:600; font-size:0.85rem; }}
  .pill {{ border-radius:20px; padding:6px 12px; font-size:0.78rem; opacity:0.85; cursor:pointer;
          border:1.5px dashed transparent; align-self:center; }}
  .pill:hover {{ border-color:rgba(255,255,255,0.65); opacity:1; }}
  .pill.playing {{ border-color:#fff; opacity:1; }}
</style>
</head>
<body>
<h1>{title} — chart replie</h1>
<p class="sub">{n} mesures &middot; {n_fam} cellule(s) repliee(s) &middot; {n_written} bloc(s) ecrit(s)
  en entier, {n_collapsed} passage(s) reduits a une pastille (meme contenu, deja ecrit plus haut).
  Ordre de lecture du morceau, de gauche a droite puis de haut en bas. Clique pour ecouter CE passage.</p>
<p id="audiostatus">clique un bloc pour charger l'audio</p>

<div class="form">{"".join(html_blocks)}</div>

<script>
(function(){{
  var AUDIO_URL = {json.dumps(data["audio_url"])};
  var statusEl = document.getElementById('audiostatus');
  var audioEl = null, blobPromise = null, stopTimer = null, curEl = null;
  function setStatus(txt, cls){{ statusEl.textContent = txt; statusEl.className = cls || ''; }}
  function ensureAudio(){{
    if(!blobPromise){{
      setStatus('chargement de l\\'audio…');
      audioEl = new Audio();
      blobPromise = fetch(AUDIO_URL)
        .then(function(r){{ if(!r.ok) throw new Error('HTTP ' + r.status); return r.blob(); }})
        .then(function(b){{
          return new Promise(function(resolve){{
            audioEl.src = URL.createObjectURL(b);
            var ready = function(){{ setStatus('audio pret', 'ready'); resolve(); }};
            if(audioEl.readyState >= 1) ready();
            else audioEl.addEventListener('loadedmetadata', ready, {{once:true}});
          }});
        }})
        .catch(function(e){{ setStatus('audio indisponible : ' + e, 'err'); throw e; }});
    }}
    return blobPromise;
  }}
  function playBlock(t0, t1, el){{
    if(curEl) curEl.classList.remove('playing');
    curEl = el;
    el.classList.add('playing');
    clearTimeout(stopTimer);
    ensureAudio().then(function(){{
      try{{ audioEl.currentTime = t0; }}catch(e){{}}
      audioEl.play().catch(function(e){{ setStatus('lecture refusee : ' + e, 'err'); }});
      var durMs = Math.max(150, (t1 - t0) * 1000);
      stopTimer = setTimeout(function(){{ audioEl.pause(); el.classList.remove('playing'); }}, durMs);
    }});
  }}
  document.querySelectorAll('.cell, .pill').forEach(function(el){{
    var t0 = parseFloat(el.dataset.t0), t1 = parseFloat(el.dataset.t1);
    el.addEventListener('click', function(){{ playBlock(t0, t1, el); }});
    el.addEventListener('keydown', function(ev){{
      if(ev.key === 'Enter' || ev.key === ' '){{ ev.preventDefault(); playBlock(t0, t1, el); }}
    }});
  }});
}})();
</script>
</body>
</html>
"""


def _render_ssm_png(S: np.ndarray, boundaries: list[int], out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = S.shape[0]
    fig, ax = plt.subplots(figsize=(7.2, 6.4), dpi=150)
    im = ax.imshow(S, cmap="magma", vmin=0.2, vmax=1.0, origin="upper")
    ax.set_title("SSM chord-tone (mesure x mesure)", fontsize=10, color="white")
    ticks = list(range(0, n, 8))
    ax.set_xticks(ticks); ax.set_xticklabels([t + 1 for t in ticks], fontsize=6, rotation=90, color="white")
    ax.set_yticks(ticks); ax.set_yticklabels([t + 1 for t in ticks], fontsize=6, color="white")
    for b in boundaries:
        ax.axhline(b - 0.5, color="#7fd0a0", linewidth=0.6, alpha=0.7)
        ax.axvline(b - 0.5, color="#7fd0a0", linewidth=0.6, alpha=0.7)
    fig.patch.set_facecolor("#111318"); ax.set_facecolor("#111318")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=6, colors="white")
    fig.tight_layout()
    fig.savefig(out_png, facecolor=fig.get_facecolor())
    plt.close(fig)


def _page(data: dict, ssm_png_name: str) -> str:
    chart, grid, n = data["chart"], data["grid"], data["n"]
    raw, V, families = data["raw"], data["V"], data["families"]
    title = chart.get("title") or data["stem"]

    letters_for_bar = {}
    for s in chart.get("sections") or []:
        for b0, b1 in s.get("barRanges") or []:
            for i in range(b0, b1 + 1):
                letters_for_bar[i] = s.get("label") or "?"

    fam_color = {}
    for i, fam in enumerate(f for f in families if not f["unique"]):
        fam_color[id(fam)] = FAMILY_COLORS[i % len(FAMILY_COLORS)]

    rows = []
    legend_seen = {}
    for fam in families:
        color = UNIQUE_COLOR if fam["unique"] else fam_color[id(fam)]
        bg, fg = color
        cell_bars = fam["cell_bars"]
        n_members = len(fam["members"])
        for mi, (b0, b1) in enumerate(fam["members"]):
            is_alt = fam.get("alt_ending", [False] * n_members)[mi]
            if fam["unique"]:
                note = f"unique -- ne se repete nulle part ailleurs ({cell_bars} mes.)"
            elif is_alt:
                note = f"passage {mi+1}/{n_members} -- FIN ALTERNATIVE (derniere position divergente)"
            else:
                note = f"passage {mi+1}/{n_members} (cellule {cell_bars} mes., {n_members} occurrences trouvees)"
            cells = []
            for i in range(b0, b1 + 1):
                lab = letters_for_bar.get(i, "?")
                t0, t1 = grid[i], grid[i + 1]
                rlab, rconf = raw[i]
                lowconf = " lowconf" if rconf < LOWCONF else ""
                altcls = " altend" if is_alt and i == b1 else ""
                cells.append(
                    f'<div class="bar{lowconf}{altcls}" data-t0="{t0:.3f}" data-t1="{t1:.3f}" '
                    f'title="raw decode: {rlab} (conf {rconf:.2f})" tabindex="0" role="button">'
                    f'<span class="idx">{i+1}</span><span class="chord">{_short_chord(rlab)}</span>'
                    f'<span class="lab">{lab}</span></div>'
                )
            rows.append(
                f'<div class="seg" style="background:{bg};color:{fg}">'
                f'<div class="seg-note">mes. {b0+1}-{b1+1} &middot; {note}</div>'
                f'<div class="bars">{"".join(cells)}</div></div>'
            )
            if not fam["unique"] and id(fam) not in legend_seen:
                legend_seen[id(fam)] = (bg, fg, cell_bars, n_members)

    legend_html = "".join(
        f'<span class="chip" style="background:{bg};color:{fg}">cellule {cb} mes. x{nm}</span>'
        for bg, fg, cb, nm in legend_seen.values()
    )

    n_alt = sum(sum(f.get("alt_ending", [])) for f in families)
    n_fam = sum(1 for f in families if not f["unique"])
    n_unique_bars = sum(f["cell_bars"] for f in families if f["unique"])

    return f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>{title} — repliement propose (auto)</title>
<style>
  body {{ font-family: -apple-system, Helvetica, Arial, sans-serif; background:#111318; color:#eee;
         margin:0; padding:24px 16px 80px; }}
  h1 {{ font-size:1.3rem; margin:0 0 4px; }}
  h2 {{ font-size:1.05rem; }}
  .sub {{ color:#9aa3af; margin:0 0 8px; font-size:0.95rem; }}
  a {{ color:#7fb0e0; }}
  .legend {{ margin:16px 0 12px; display:flex; flex-wrap:wrap; gap:8px; }}
  .chip {{ padding:4px 10px; border-radius:6px; font-size:0.85rem; }}
  #audiostatus {{ font-size:0.85rem; color:#9aa3af; margin:0 0 20px; min-height:1.2em; }}
  #audiostatus.ready {{ color:#7fd0a0; }}
  #audiostatus.err {{ color:#e08a7a; }}
  .seg {{ border-radius:10px; padding:10px 12px; margin-bottom:10px; }}
  .seg-note {{ font-size:0.82rem; opacity:0.9; margin-bottom:6px; }}
  .bars {{ display:flex; flex-wrap:wrap; gap:4px; }}
  .bar {{ background:rgba(0,0,0,0.18); border-radius:6px; padding:4px 8px; min-width:48px;
         display:flex; flex-direction:column; align-items:center; cursor:pointer;
         border:1.5px solid transparent; transition:transform .08s, border-color .08s; }}
  .bar.lowconf {{ border:1.5px dashed rgba(255,255,255,0.45); }}
  .bar.altend {{ box-shadow: 0 0 0 2px #ffd36b inset; }}
  .bar:hover {{ border-color:rgba(255,255,255,0.75); }}
  .bar:active {{ transform:scale(0.95); }}
  .bar.playing {{ border-color:#fff; background:rgba(255,255,255,0.3); }}
  .idx {{ font-size:0.62rem; opacity:0.75; }}
  .chord {{ font-weight:600; font-size:0.88rem; }}
  .lab {{ font-size:0.62rem; opacity:0.75; }}
  .verdict, .ssmbox {{ background:#1b1e26; border-radius:10px; padding:16px 18px; margin-top:20px;
             line-height:1.55; }}
  .ssmbox img {{ width:100%; max-width:520px; display:block; margin:0 auto; border-radius:8px; }}
  code {{ background:#262a33; padding:1px 5px; border-radius:4px; }}
</style>
</head>
<body>
<h1>{title} — repliement propose (calcule, pas dessine)</h1>
<p class="sub">{n} mesures &middot; {n_fam} cellule(s) repetee(s) trouvees sans frontiere de lettre
  (fenetres glissantes, cosinus chord-tone), {n_alt} fin(s) alternative(s) detectee(s),
  {n_unique_bars} mesure(s) uniques (ne se repetent nulle part).
  Bordure jaune pleine = derniere position d'une fin alternative. Pointilles = decode brut peu
  confiant (musx, conf &lt; {LOWCONF}). Clique une mesure pour l'ecouter.</p>
<p id="audiostatus">clique une mesure pour charger l'audio</p>

<div class="legend">{legend_html}</div>

{"".join(rows)}

<div class="ssmbox">
<h2>La matrice de similarite (chord-tone SSM)</h2>
<img src="{ssm_png_name}" alt="SSM chord-tone mesure x mesure">
<p>harmonia.sections.similarity.harmonic_vectors -- la meme brique que Soudure et l'outil sections,
jamais fondamentale-seule. Lignes vertes = les frontieres de cellule que l'algo a posees.</p>
</div>

<div class="verdict">
<h2>Comment lire cette page</h2>
<p>Chaque bloc colore est une cellule retrouvee ailleurs dans le morceau, PAS confinee a une
lettre de section -- c'est la difference avec ce que le chart affiche.
Une case a bordure jaune est une fin alternative : la cellule colle partout sauf sa derniere
position, comme un 1st/2nd ending -- pas une nouvelle lettre. Les blocs gris sont des mesures qui
ne se repetent nulle part (a l'oeil ou a l'oreille, verifie si elles devraient).</p>
<p>C'est un calcul, pas une lecture a l'oreille -- arbitrage necessaire avant de toucher
<code>soudure.py</code> / <code>folding.py</code>. Reutilise les seuils deja calibres
(<code>section_tool.TOOL_THR</code>, <code>folding.PERIOD_MIN_SCORE</code>), n'en invente pas.</p>
</div>

<script>
(function(){{
  var AUDIO_URL = {json.dumps(data["audio_url"])};
  var statusEl = document.getElementById('audiostatus');
  var audioEl = null, blobPromise = null, stopTimer = null, curBar = null;

  function setStatus(txt, cls){{ statusEl.textContent = txt; statusEl.className = cls || ''; }}

  function ensureAudio(){{
    if(!blobPromise){{
      setStatus('chargement de l\\'audio…');
      audioEl = new Audio();
      blobPromise = fetch(AUDIO_URL)
        .then(function(r){{ if(!r.ok) throw new Error('HTTP ' + r.status); return r.blob(); }})
        .then(function(b){{
          return new Promise(function(resolve){{
            audioEl.src = URL.createObjectURL(b);
            var ready = function(){{ setStatus('audio pret', 'ready'); resolve(); }};
            if(audioEl.readyState >= 1) ready();
            else audioEl.addEventListener('loadedmetadata', ready, {{once:true}});
          }});
        }})
        .catch(function(e){{ setStatus('audio indisponible : ' + e, 'err'); throw e; }});
    }}
    return blobPromise;
  }}

  function playBar(t0, t1, cellEl){{
    if(curBar) curBar.classList.remove('playing');
    curBar = cellEl;
    cellEl.classList.add('playing');
    clearTimeout(stopTimer);
    ensureAudio().then(function(){{
      try{{ audioEl.currentTime = t0; }}catch(e){{}}
      audioEl.play().catch(function(e){{ setStatus('lecture refusee : ' + e, 'err'); }});
      var durMs = Math.max(150, (t1 - t0) * 1000);
      stopTimer = setTimeout(function(){{ audioEl.pause(); cellEl.classList.remove('playing'); }}, durMs);
    }});
  }}

  document.querySelectorAll('.bar').forEach(function(cell){{
    var t0 = parseFloat(cell.dataset.t0), t1 = parseFloat(cell.dataset.t1);
    cell.addEventListener('click', function(){{ playBar(t0, t1, cell); }});
    cell.addEventListener('keydown', function(ev){{
      if(ev.key === 'Enter' || ev.key === ' '){{ ev.preventDefault(); playBar(t0, t1, cell); }}
    }});
  }});
}})();
</script>
</body>
</html>
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chart", required=True, help="ex: min_9zEl-FQLI4A")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--brut", action="store_true",
                    help="le chart tel qu'affiche en prod (gabarits de lettre compris) "
                         "+ decode independant en regard, PAS les cellules auto-decouvertes")
    ap.add_argument("--folded", action="store_true",
                    help="le resultat du repliement : chaque cellule ecrite une fois, "
                         "en ordre de lecture, avec ses x N et ses fins alternatives")
    a = ap.parse_args(argv)

    data = build(a.chart)

    if a.brut:
        out = a.out or (SETTINGS.reports_dir / f"chart_brut_{a.chart}.html")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_page_brut(data), encoding="utf-8")
        n_low = sum(1 for _, c in data["raw"] if c < LOWCONF)
        print(f"→ {out}")
        print(f"   {data['n']} mesures, {n_low} sous confiance {LOWCONF}")
        return 0

    if a.folded:
        out = a.out or (SETTINGS.reports_dir / f"chart_replie_{a.chart}.html")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_page_folded(data), encoding="utf-8")
        n_fam = sum(1 for f in data["families"] if not f["unique"])
        n_alt = sum(sum(f.get("alt_ending", [])) for f in data["families"])
        print(f"→ {out}")
        print(f"   {data['n']} mesures -> {n_fam} cellules repliees, {n_alt} fins alternatives")
        return 0

    out = a.out or (SETTINGS.reports_dir / f"repli_propose_{a.chart}.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    png_name = f"repli_propose_{a.chart}_ssm.png"
    boundaries = sorted({b0 for f in data["families"] for b0, _ in f["members"]})
    _render_ssm_png(data["S"], boundaries, out.parent / png_name)
    out.write_text(_page(data, png_name), encoding="utf-8")

    n_fam = sum(1 for f in data["families"] if not f["unique"])
    n_alt = sum(sum(f.get("alt_ending", [])) for f in data["families"])
    print(f"→ {out}")
    print(f"   {data['n']} mesures, {n_fam} cellules repetees, {n_alt} fins alternatives")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
