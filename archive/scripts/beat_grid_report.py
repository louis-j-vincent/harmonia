"""scripts/beat_grid_report.py — le garde-fou de grille, avant/après, à l'oreille.

    python scripts/beat_grid_report.py  ->  harmonia_min/state/reports/beat_grid.html
    (ouvrir via http://localhost:7772/reports/beat_grid.html — l'audio est
     servi par harmonia_min/server.py sur ce port, jamais 7771)

CONTEXTE (voir docs/known_issues.md, "RÉSOLU — le garde de grille refusait la
MESURE, pas la musique", 2026-08-07). `harmonia_min/beats.py` refuse un
morceau quand sa grille de mesures n'est pas fiable, plutôt que de produire une
charte posée sur des mesures qui n'en sont pas. L'ANCIEN garde mesurait le
nombre de temps entre deux débuts de mesure consécutifs (les "downbeats" du
traceur Beat This!) et refusait si ce compte n'était pas régulier — il a
refusé 19 morceaux sur les 63 de la bibliothèque.

En écoutant les 19 (jamais un seul morceau — règle n°5 du projet), le vrai
défaut est apparu : Beat This! marque SOUVENT le 3ᵉ temps comme un début de
mesure EN PLUS du vrai. Une mesure à 4 temps ressort alors comme deux mesures
de 2 temps ("Close to You" : {4 temps: 63 mesures, 2 temps: 35 mesures}).
L'histogramme des écarts devient bimodal, la "cohérence" (part du mode le
plus fréquent) s'effondre à ~0,6 — alors que les temps eux-mêmes sont
métronomiques.

Le NOUVEAU garde (`beats.repair_grid`) ne compte plus les écarts : il essaie
de PAVER l'axe des temps avec des mesures de longueur exacte, en n'utilisant
que les débuts de mesure que le traceur a déjà posés (jamais un temps
inventé). `coverage` = part du morceau ainsi pavée (seuil 0,85). `direct` =
part des écarts bruts du traceur qui valent déjà la métrique retenue (seuil
0,15) — ce filtre existe pour qu'un vrai verrou demi-tempo (deux fois plus
lent que la réalité) reste refusé même s'il se pave "parfaitement" à 4.

Résultat mesuré sur les 63 morceaux : refus 19 -> 6, zéro régression (aucun
des 44 déjà acceptés n'a changé de métrique ni de phase), 0/44 exactement
plus précisément.

Cette page :
  1. le tableau des 63 morceaux, verdict avant/après, colorié ;
  2. pour les 19 qui changent de camp (13 nouvellement acceptés + 6 toujours
     refusés) : un extrait audio de 40 s avec le tracé de l'énergie d'attaque
     et les trois grilles (temps, downbeats bruts, mesures réparées) + son
     histogramme d'écarts ;
  3. l'audio réel de ces 19 morceaux, pour trancher à l'oreille.

Données : scripts/_beat_grid_data.json (copie figée de l'audit — stats
résumées, une ligne par morceau). Les temps de battement/downbeat viennent de
harmonia_min/state/beats/<stem>.json (cache Beat This!) ; la grille réparée
est RECALCULÉE ici avec le vrai `harmonia_min.beats.repair_grid`, pas
réimplémentée, pour ne jamais diverger du code qui tourne en prod.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.lines import Line2D      # noqa: E402
import numpy as np                       # noqa: E402

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))

from harmonia_min.beats import repair_grid, GRID_MIN_COVERAGE, GRID_MIN_DIRECT  # noqa: E402

DATA_PATH = HERE / "scripts" / "_beat_grid_data.json"
AUDIO_DIR = HERE / "docs" / "audio"
BEATS_DIR = HERE / "harmonia_min" / "state" / "beats"
OUT = HERE / "harmonia_min" / "state" / "reports" / "beat_grid.html"

# Onset envelopes are slow to compute (ffmpeg decode + librosa STFT per song).
# Cache them locally so a rerun is instant; on a fresh checkout this directory
# is empty and the script falls back to computing every one from audio — no
# hard dependency on the scratch path where they were first computed.
ONSET_CACHE = HERE / "harmonia_min" / "state" / "reports" / "_onset_cache"
_LEGACY_ONSET_CACHE = Path(
    "/private/tmp/claude-501/-Users-vincente-Documents-Projets-Perso-Code-harmonia/"
    "29e6c8ff-69c3-4685-aae3-2c46f129bcde/scratchpad/onsets")

SR = 22050
HOP = 512
WINDOW_S = 40.0     # excerpt length for the diagnostic plots

INK = "#1c1c1c"
GREEN = "#1f8a5b"    # accepté / mesure réparée
BLUE = "#2a6fb0"     # neutre / courbe d'attaque
RED = "#8a2b2b"      # toujours refusé
ORANGE = "#c58a2e"   # downbeat brut du traceur
GREY = "#8a8371"     # temps bruts


# ── data plumbing ────────────────────────────────────────────────────────────

def load_data() -> list[dict]:
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


def classify(old: str, new: str) -> str:
    """green = accepté avant et après · blue = nouvellement accepté ·
    red = toujours refusé · orange = régression (ne devrait jamais arriver —
    mesuré 0/63, voir docs/known_issues.md)."""
    a, b = old.startswith("accept"), new.startswith("accept")
    if a and b:
        return "green"
    if not a and b:
        return "blue"
    if not a and not b:
        return "red"
    return "orange"


_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,14}$")


def pretty(stem: str) -> str:
    """Titre lisible — sauf pour les stems qui SONT un id YouTube opaque
    (mélange de casse, pas de mots séparés par des underscores) : les
    "Titre-Iser" abîmerait l'id, donc on le garde tel quel."""
    if stem.startswith("rwc_"):
        return "RWC " + stem[4:].replace("rwc_", "").upper()
    if _ID_RE.match(stem) and any(c.isupper() for c in stem) and "_" not in stem:
        return stem
    return stem.replace("_", " ").strip().title()


def mmss(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}:{s:02d}"


def pct(x: float) -> str:
    return f"{x:.0%}"


# ── audio / onset envelope ──────────────────────────────────────────────────

def _load_audio_mono(stem: str) -> np.ndarray:
    p = AUDIO_DIR / f"{stem}.m4a"
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(p), "-f", "f32le",
         "-ac", "1", "-ar", str(SR), "-"],
        capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def onset_env(stem: str) -> np.ndarray:
    """Onset-strength envelope (librosa used ONLY as a spectral feature
    extractor here — never librosa.beat.*, that tracker is banned in this
    project). Cached locally; reuses the scratch-dir cache from the audit run
    if present, else recomputes from audio."""
    local = ONSET_CACHE / f"{stem}.npy"
    if local.exists():
        return np.load(local)
    legacy = _LEGACY_ONSET_CACHE / f"{stem}.npy"
    if legacy.exists():
        env = np.load(legacy).astype(np.float32)
    else:
        import librosa
        y = _load_audio_mono(stem)
        env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP).astype(np.float32)
    ONSET_CACHE.mkdir(parents=True, exist_ok=True)
    np.save(local, env)
    return env


def load_beats(stem: str) -> tuple[np.ndarray, np.ndarray]:
    d = json.loads((BEATS_DIR / f"{stem}.json").read_text(encoding="utf-8"))
    return np.asarray(d["beats"], float), np.asarray(d["downbeats"], float)


# ── plotting ─────────────────────────────────────────────────────────────────

def fig2b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=124, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def song_figure(rec: dict) -> str:
    stem = rec["stem"]
    beats, downbeats_raw = load_beats(stem)
    rep = repair_grid(beats, downbeats_raw)
    repaired = np.asarray(rep["downbeats"], float)

    env = onset_env(stem)
    t = np.arange(len(env)) * HOP / SR
    dur = float(rec["duration"])
    t1 = min(dur, max(WINDOW_S, dur / 2.0 + WINDOW_S / 2.0))
    t0 = max(0.0, t1 - WINDOW_S)

    m_env = (t >= t0) & (t <= t1)
    m_b = (beats >= t0) & (beats <= t1)
    m_db = (downbeats_raw >= t0) & (downbeats_raw <= t1)
    m_rep = (repaired >= t0) & (repaired <= t1)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.8, 2.55), gridspec_kw={"width_ratios": [3.1, 1]})

    # -- left: onset envelope + the three grids --------------------------------
    if m_env.any():
        ax1.fill_between(t[m_env], env[m_env], color=BLUE, alpha=.16, lw=0)
        ax1.plot(t[m_env], env[m_env], color=BLUE, lw=.9)
        ytop = max(float(env[m_env].max()) * 1.18, 1e-3)
    else:
        ytop = 1.0
    for x in beats[m_b]:
        ax1.axvline(x, color=GREY, lw=.6, alpha=.55, zorder=1)
    for x in downbeats_raw[m_db]:
        ax1.axvline(x, color=ORANGE, lw=1.15, alpha=.9, zorder=2)
    for x in repaired[m_rep]:
        ax1.axvline(x, color=GREEN, lw=2.1, alpha=.95, zorder=3)
    ax1.set_xlim(t0, t1)
    ax1.set_ylim(0, ytop)
    ax1.set_xlabel("temps (s)", fontsize=8)
    ax1.set_ylabel("attaque (u.a.)", fontsize=8)
    ax1.tick_params(labelsize=7)
    for sp in ("top", "right"):
        ax1.spines[sp].set_visible(False)
    handles = [
        Line2D([0], [0], color=GREY, lw=1.4, label="temps (tous)"),
        Line2D([0], [0], color=ORANGE, lw=1.6, label="downbeats bruts (traceur)"),
        Line2D([0], [0], color=GREEN, lw=2.2, label="mesures réparées (retenues)"),
    ]
    ax1.legend(handles=handles, loc="upper right", fontsize=6.3, frameon=False,
               handlelength=1.6, labelspacing=.25)

    # -- right: gap histogram (temps par mesure) --------------------------------
    gh = rec["gap_hist"]
    xs = sorted(int(k) for k in gh.keys())
    ys = [gh[str(x)] for x in xs]
    colors = [GREEN if x == rec["metre"] else "#c9beA0" for x in xs]
    ax2.bar(xs, ys, color=colors, width=.72)
    ax2.set_xlabel("temps / mesure", fontsize=8)
    ax2.set_ylabel("nb. mesures (brut)", fontsize=8)
    ax2.set_xticks(xs)
    ax2.tick_params(labelsize=7)
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)

    fig.suptitle(f"{pretty(stem)} — couverture {pct(rec['coverage'])}"
                 f" (extrait {mmss(t0)}–{mmss(t1)})",
                 fontsize=9.3, x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return fig2b64(fig)


# ── HTML assembly ────────────────────────────────────────────────────────────

CHIP = {"green": ("OK", GREEN), "blue": ("OK", GREEN),
        "red": ("REFUS", RED), "orange": ("REFUS", RED)}


def verdict_chip(status: str, full: str) -> str:
    label, color = ("OK", GREEN) if status.startswith("accept") else ("REFUS", RED)
    safe = full.replace('"', "&quot;")
    return (f'<span class=chip style="background:{color}22;color:{color};" '
            f'title="{safe}">{label}</span>')


def table_row(rec: dict) -> str:
    stem = rec["stem"]
    cat = rec["_cat"]
    anchor = f' id="row-{stem}"'
    link = (f' <a class=jump href="#plot-{stem}">voir ↓</a>'
            if cat in ("blue", "red") else "")
    return (
        f'<tr class="row-{cat}"{anchor}>'
        f'<td>{pretty(stem)}{link}</td>'
        f'<td>{mmss(rec["duration"])}</td>'
        f'<td>{rec["bpm"]:.0f}</td>'
        f'<td>{rec["raw_metre"]}</td>'
        f'<td>{pct(rec["raw_cons"])}</td>'
        f'<td>{rec["metre"]}</td>'
        f'<td>{pct(rec["coverage"])}</td>'
        f'<td>{pct(rec["direct"])}</td>'
        f'<td>{pct(rec["kept"])}</td>'
        f'<td>{verdict_chip(rec["old"], rec["old"])}</td>'
        f'<td>{verdict_chip(rec["new"], rec["new"])}</td>'
        f'</tr>'
    )


def phase_line(rec: dict) -> str:
    ps = rec.get("phase_scores") or []
    if not ps:
        return ""
    best = max(range(len(ps)), key=lambda i: ps[i])
    agree = best == 0
    mark = "✓ la phase retenue gagne" if agree else "✗ le validateur préfère un autre décalage"
    color = GREEN if agree else ORANGE
    cells = " · ".join(
        (f"<b>{v:+.2f}</b>" if i == 0 else f"{v:+.2f}") for i, v in enumerate(ps))
    return (f'<p class=phase>validateur harmonique (nouveauté d\'accords par '
            f'temps), score par décalage candidat — <b>retenu en gras</b> : '
            f'{cells} &nbsp;<span style="color:{color}">{mark}</span></p>')


def song_card(rec: dict) -> str:
    stem = rec["stem"]
    img = song_figure(rec)
    audio = f'/audio/{stem}.m4a'
    steady = rec.get("local_steady")
    steady_txt = f"{steady:.2f}" if steady is not None else "?"
    return f"""<section class=card id="plot-{stem}">
<h3>{pretty(stem)} <a class=jump href="#row-{stem}">↑ table</a></h3>
<p class=meta>métrique retenue <b>{rec['metre']}</b> · couverture
<b>{pct(rec['coverage'])}</b> (seuil {pct(GRID_MIN_COVERAGE)}) · direct
<b>{pct(rec['direct'])}</b> (seuil {pct(GRID_MIN_DIRECT)}) · stabilité locale
des temps <b>{steady_txt}</b> · {rec['n_bars']} mesures brutes, {rec['rep_bars']}
retenues</p>
<img src="data:image/png;base64,{img}" alt="grille de {pretty(stem)}">
{phase_line(rec)}
<audio controls preload="none" src="{audio}"></audio>
</section>"""


STYLE = """
*{box-sizing:border-box;-webkit-tap-highlight-color:transparent}
body{margin:0;background:#e7e0d0;font:16px/1.55 -apple-system,system-ui,sans-serif;color:#1c1c1c}
.wrap{max-width:1180px;margin:0 auto;padding:22px 14px calc(48px + env(safe-area-inset-bottom))}
h1{font:italic 600 25px Georgia,serif;margin:0 0 4px}
h2{font:700 19px system-ui;margin:26px 0 10px;color:#8a2b2b}
h3{font:700 15px system-ui;margin:0 0 4px;display:flex;justify-content:space-between;align-items:baseline}
.lede{color:#4a4436;font-size:13.8px;line-height:1.6;margin-bottom:16px}
.lede b{color:#1c1c1c}
.stats{display:flex;gap:10px;flex-wrap:wrap;margin:14px 0 18px}
.stat{background:#fffdf6;border:1px solid #e5dcc6;border-radius:10px;padding:10px 16px;min-width:130px}
.stat b{display:block;font:700 22px ui-monospace,monospace;color:#8a2b2b}
.stat span{font-size:11.5px;color:#6f6858}
.blame{background:#fffdf6;border:1px solid #e5dcc6;border-radius:12px;padding:14px 16px;margin:6px 0 22px;font-size:13.3px;line-height:1.65}
.blame ul{margin:8px 0;padding-left:20px}
.blame li{margin-bottom:7px}
.blame code{background:#f2ece0;padding:1px 5px;border-radius:4px;font-size:12px}
table{border-collapse:collapse;font-size:12.6px;width:100%;margin-bottom:8px;background:#fffdf6}
th,td{border:1px solid #e5dcc6;padding:5px 8px;text-align:left}
th{background:#f2ece0;font-size:11px;position:sticky;top:0;cursor:pointer;user-select:none}
th:hover{background:#e9e0cc}
tr.row-green td{background:#eaf3ee}
tr.row-blue td{background:#e9f0f8}
tr.row-red td{background:#f9ece9}
.chip{display:inline-block;padding:1px 7px;border-radius:5px;font:700 10.5px ui-monospace,monospace;cursor:help}
.jump{font:600 11px system-ui;color:#2a6fb0;text-decoration:none;margin-left:8px}
.note{color:#6f6858;font-size:12.3px;margin:6px 0 18px}
.card{background:#fffdf6;border:1px solid #e5dcc6;border-radius:12px;padding:13px 14px;margin-bottom:14px}
.card img{width:100%;display:block;border-radius:8px;margin:4px 0}
.card .meta{font-size:11.8px;color:#6f6858;margin:2px 0 4px}
.card audio{width:100%;margin-top:6px}
.phase{font-size:11.3px;color:#4a4436;margin:4px 0 2px}
.phase b{color:#1c1c1c}
"""

SORT_JS = """
document.querySelectorAll('table.sortable th').forEach((th,i)=>{
  th.addEventListener('click',()=>{
    const table=th.closest('table');
    const tbody=table.querySelector('tbody');
    const rows=[...tbody.querySelectorAll('tr')];
    const asc=th.dataset.asc!=='1';
    table.querySelectorAll('th').forEach(h=>delete h.dataset.asc);
    th.dataset.asc=asc?'1':'0';
    rows.sort((a,b)=>{
      const av=a.children[i].innerText.trim(), bv=b.children[i].innerText.trim();
      const an=parseFloat(av.replace('%','').replace(':','.')), bn=parseFloat(bv.replace('%','').replace(':','.'));
      const bothNum=!isNaN(an)&&!isNaN(bn);
      const cmp=bothNum?(an-bn):av.localeCompare(bv,'fr');
      return asc?cmp:-cmp;
    });
    rows.forEach(r=>tbody.appendChild(r));
  });
});
"""


def main() -> None:
    data = load_data()
    for x in data:
        x["_cat"] = classify(x["old"], x["new"])
    n_total = len(data)
    n_red = sum(1 for x in data if x["_cat"] == "red")
    n_blue = sum(1 for x in data if x["_cat"] == "blue")
    n_green = sum(1 for x in data if x["_cat"] == "green")
    n_orange = sum(1 for x in data if x["_cat"] == "orange")
    n_refused_before = n_red + n_blue
    if n_orange:
        print(f"  !! {n_orange} régression(s) trouvée(s) dans les données — "
              f"la page les marque en orange, à investiguer.")

    order = {"red": 0, "blue": 1, "green": 2, "orange": 3}
    rows = sorted(data, key=lambda x: (order[x["_cat"]], pretty(x["stem"])))
    table_html = "\n".join(table_row(r) for r in rows)

    plot_targets = [x for x in data if x["_cat"] in ("red", "blue")]
    still_refused = sorted((x for x in plot_targets if x["_cat"] == "red"),
                            key=lambda x: pretty(x["stem"]))
    newly_accepted = sorted((x for x in plot_targets if x["_cat"] == "blue"),
                            key=lambda x: pretty(x["stem"]))

    print(f"rendering {len(plot_targets)} diagnostic plots "
          f"({len(still_refused)} refused + {len(newly_accepted)} newly accepted)…")
    cards_refused = ""
    for i, rec in enumerate(still_refused, 1):
        cards_refused += song_card(rec)
        print(f"  [{i}/{len(still_refused)}] refused  {rec['stem']}")
    cards_accepted = ""
    for i, rec in enumerate(newly_accepted, 1):
        cards_accepted += song_card(rec)
        print(f"  [{i}/{len(newly_accepted)}] accepted {rec['stem']}")

    blame = f"""<div class=blame>
<b>À qui la faute, sur les {n_refused_before} refus de l'ancien garde ?</b>
<ul>
<li><b>13/{n_refused_before}</b> — <b>la mesure, pas la musique</b> : les temps
sont métronomiques (stabilité locale 0,90–1,00), seuls les <i>downbeats</i>
étaient sur-marqués (le traceur pose un début de mesure en plus, sur le
3ᵉ temps).</li>
<li><b>5/{n_refused_before}</b> — <b>beats vraiment instables</b>
(A-DuOmA75lI, Georgia, Chiquitita, Alessi Brothers, Commodores) : les mesures
hors-grille sont <b>dispersées</b> (Chiquitita 28 zones séparées, Autumn
Leaves 26 alors qu'elle est désormais acceptée), donc ce n'est pas une
irrégularité locale mais du bruit de suivi réparti sur tout le morceau.</li>
<li><b>1/{n_refused_before}</b> — <b>vraie métrique non supportée</b> :
kwUtA8bUS30 est en 6, refusé exprès (5/6/7 restent hors du garde, aucun autre
morceau du corpus ne les exerce).</li>
<li>Hypothèse écartée : l'ambiguïté de <code>argmin</code> — les downbeats de
Beat This! sont exactement des beats (100 % à moins de 1e-6 s, sur les 63
morceaux).</li>
<li>Hypothèse écartée : une irrégularité métrique <b>locale</b> — les zones
hors-grille sont dispersées, pas groupées.</li>
</ul>
Note : les 5 morceaux "beats instables" ont pourtant une grille réparée que le
validateur harmonique (nouveauté d'accords par temps) approuve fortement —
Georgia z=7,13, Commodores z=8,32, Alessi z=7,79 (voir le score sous chaque
graphe). Le garde reste volontairement conservateur : rien ici ne répare une
insertion/suppression de battement isolée, seulement le sur-marquage
systématique des downbeats.
</div>"""

    stats = f"""<div class=stats>
<div class=stat><b>19 → 6</b><span>refus (ancien → nouveau garde)</span></div>
<div class=stat><b>13</b><span>nouvellement acceptés</span></div>
<div class=stat><b>0</b><span>régression sur les {n_green} déjà acceptés</span></div>
<div class=stat><b>{n_total}</b><span>morceaux dans la bibliothèque</span></div>
</div>"""

    html = f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Grille de mesures — garde-fou avant/après</title>
<style>{STYLE}</style></head><body><div class=wrap>
<h1>La grille de mesures — le garde-fou corrigé</h1>
<div class=lede>
Avant d'écrire un accord dans une mesure, le pipeline vérifie que la grille
posée par le traceur de temps (Beat This!) tient debout — sinon mieux vaut
refuser que produire une charte fausse. L'<b>ancien</b> garde comptait les
temps entre deux débuts de mesure et refusait si ce compte n'était pas
régulier ; Beat This! marque souvent le 3ᵉ temps comme un second début de
mesure, donc une vraie mesure à 4 temps ressortait comme deux mesures de 2 —
et le garde refusait le comptage, pas la musique. Le <b>nouveau</b> garde
(<code>beats.repair_grid</code>) essaie plutôt de PAVER l'axe des temps avec
des mesures exactes, en n'utilisant que les débuts que le traceur a déjà
posés — il ne peut donc jamais inventer une mesure.
</div>
{stats}
<h2>Les {n_total} morceaux</h2>
<p class=note>Trié pour mettre en premier ce qui a bougé — les toujours
refusés (rouge), puis les nouvellement acceptés (bleu) ; le reste (accepté
avant et après, vert) suit en ordre alphabétique. Cliquer un en-tête trie la
colonne.</p>
<table class=sortable><thead><tr>
<th>morceau</th><th>durée</th><th>BPM</th><th>métrique brute</th>
<th>cohérence brute (ancien)</th><th>métrique retenue</th>
<th>couverture (nouveau)</th><th>direct</th><th>% downbeats gardés</th>
<th>verdict avant</th><th>verdict après</th>
</tr></thead><tbody>
{table_html}
</tbody></table>
{blame}
<h2>Toujours refusés — {len(still_refused)} morceaux</h2>
<p class=note>Extrait audio de 40 s pris au milieu du morceau. Gris = tous
les temps · orange = downbeats bruts du traceur · vert épais = mesures
retenues par la réparation. À droite, l'histogramme des écarts entre
downbeats bruts (en temps) — la métrique retenue est en vert.</p>
{cards_refused}
<h2>Nouvellement acceptés — {len(newly_accepted)} morceaux</h2>
<p class=note>Mêmes 13 morceaux que la ligne "13" ci-dessus : refusés par
l'ancien garde, acceptés par le nouveau. Écouter si le grillage vert tombe
bien sur le temps 1.</p>
{cards_accepted}
<p class=note>Cette page doit être ouverte via
<code>http://localhost:7772/reports/beat_grid.html</code> (ou l'adresse
Tailscale équivalente) pour que l'audio joue — <code>file://</code> ne sert
pas <code>/audio/</code>.</p>
</div>
<script>{SORT_JS}</script>
</body></html>"""

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    n_audio = html.count("<audio ")
    n_imgs = html.count("data:image/png;base64,")
    print(f"wrote {OUT.relative_to(HERE)} "
          f"({OUT.stat().st_size / 1024:.0f} KB, {n_imgs} plots, {n_audio} audio tags)")


if __name__ == "__main__":
    main()
