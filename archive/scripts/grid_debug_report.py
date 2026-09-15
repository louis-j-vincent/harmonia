"""Build /reports/grid_debug.html — why the SSM is bad on some songs, upstream
of the LOCKED method (scripts/harmonic_method.py, imported never re-implemented).

Three songs, three measured causes, re-examined:
  Beat It      — the published −11.4 % "drift" is an artifact of 8 intro beats;
                 the real defect is 5 half-bar-shifted bars.  Grid repaired by
                 downbeat re-anchoring; the SSM judges.
  Sunny        — modulation; transposition-invariant SSM (max over chroma
                 rotations), raw 12-rotation vs constrained {−1,0,+1}, with
                 This Love and Norah as must-not-break controls.
  corpus (83)  — drift-detector candidates for the grid guard, distributions
                 BEFORE any threshold (data from the corpus agent run).

Inputs are the scratchpad npz/json computed by the session (paths below).
Output is self-contained (base64 PNGs), hard data, served by /reports/.
"""
from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

SCRATCH = Path("/private/tmp/claude-501/-Users-vincente-Documents-Projets-Perso-"
               "Code-harmonia/96b7806b-3278-401d-b520-1914f33ab075/scratchpad")
OUT = HERE / "harmonia_min" / "state" / "reports" / "grid_debug.html"

CREAM, CARD, INK, MUT, RED = "#e7e0d0", "#fffdf6", "#1c1c1c", "#8a8371", "#8a2b2b"
GREEN = "#1f8a5b"


def b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, facecolor=CARD,
                bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def style(ax):
    ax.set_facecolor(CARD)
    for s in ax.spines.values():
        s.set_color("#e5dcc6")
    ax.tick_params(colors=MUT, labelsize=8)


def show_ssm(ax, S, title):
    ax.imshow(S, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=9, color=INK)
    style(ax)


def show_curve(ax, curve, L, b0, pk, title):
    ref = float(curve[b0])
    ax.plot(curve, lw=1.1, color=INK)
    ax.axhline(0.9 * ref, ls="--", lw=0.8, color=MUT)
    ax.plot(b0, curve[b0], "s", ms=6, color=GREEN, zorder=5)
    if len(pk):
        ax.plot(pk, curve[pk], "o", ms=5, color=RED, zorder=5)
    ax.set_title(title, fontsize=8.5, color=INK)
    ax.set_xlim(0, len(curve) - 1)
    style(ax)


# ═══ §1 Beat It: the drift that wasn't ══════════════════════════════════════
bd = json.loads((HERE / "harmonia_min/state/beats/"
                 "michael_jackson_beat_it_official_4k_video.json").read_text())
b = np.array(bd["beats"]); db = np.array(bd["downbeats"])
ibi = np.diff(b); t = b[:-1]
out_mask = np.abs(ibi - np.median(ibi)) > 0.05
A_naif = np.polyfit(t, ibi, 1)
A_rob = np.polyfit(t[~out_mask], ibi[~out_mask], 1)
d_naif = A_naif[0] * (t[-1] - t[0]) / ibi.mean() * 100
d_rob = A_rob[0] * (t[-1] - t[0]) / ibi[~out_mask].mean() * 100

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 3.6),
                             gridspec_kw={"width_ratios": [3, 2]})
fig.patch.set_facecolor(CARD)
a1.plot(t[~out_mask], ibi[~out_mask], ".", ms=3, color=INK, alpha=.5,
        label="intervalle entre beats")
a1.plot(t[out_mask], ibi[out_mask], "o", ms=7, mfc="none", mec=RED, mew=1.6,
        label="les 8 beats d'intro (21–30 s)")
xx = np.array([t[0], t[-1]])
a1.plot(xx, A_naif[1] + A_naif[0] * xx, color=RED, lw=1.6,
        label=f"régression brute : « dérive » {d_naif:+.1f} %")
a1.plot(xx, A_rob[1] + A_rob[0] * xx, color=GREEN, lw=1.6,
        label=f"sans ces 8 beats : {d_rob:+.2f} %")
a1.set_xlabel("temps (s)", fontsize=8, color=MUT)
a1.set_ylabel("s entre deux beats", fontsize=8, color=MUT)
a1.legend(fontsize=7.5, loc="upper right", framealpha=.9)
a1.set_title("Beat It — l'intervalle entre beats, sur tout le morceau",
             fontsize=9.5, color=INK)
style(a1)

idx = np.array([int(np.abs(b - x).argmin()) for x in db])
from collections import Counter
off = Counter(i % 4 for i in idx).most_common(1)[0][0]
ph = (idx - off) % 4
a2.step(db, ph, where="post", color=INK, lw=1.2)
a2.axvspan(21.5, 30.2, color=RED, alpha=.12)
a2.axvspan(50.1, 59.7, color=RED, alpha=.25)
a2.text(55, 3.4, "6 mesures\ndécalées d'½ mesure", fontsize=7.5, color=RED,
        ha="center")
a2.set_yticks([0, 1, 2, 3])
a2.set_xlabel("temps (s)", fontsize=8, color=MUT)
a2.set_ylabel("phase du downbeat\n(en beats, vs grille rigide)", fontsize=8,
              color=MUT)
a2.set_title("La phase des downbeats du tracker", fontsize=9.5, color=INK)
style(a2)
fig.tight_layout()
F1 = b64(fig)

# ═══ §2 corpus distributions ════════════════════════════════════════════════
rows = json.loads((SCRATCH / "drift_corpus.json").read_text())
real = [r for r in rows if not r["guitarset"]]
gset = [r for r in rows if r["guitarset"]]


def guard_refuses(r):
    if (r.get("n_bars") or 0) < 30:
        return False
    return r.get("metre") != 4 or (r.get("consistency") or 0) < 0.80


NICE = {
    "michael_jackson_beat_it_official_4k_video": "Beat It",
    "bobby_hebb_sunny_official_audio": "Sunny",
    "maroon_5_this_love": "This Love",
    "norah_jones_don_t_know_why": "Don't Know Why",
    "carpenters_close_to_you": "Close to You",
    "ray_charles_georgia_on_my_mind_official_video": "Georgia",
    "michael_jackson_billie_jean_official_video": "Billie Jean",
    "aretha_franklin_chain_of_fools_official_lyric_video": "Chain of Fools",
    "let_it_be_remastered_2009": "Let It Be",
    "muppets_kermit_its_not_easy_being_green_original": "Kermit",
    "blue_bossa_150bpm_backing_track": "Blue Bossa 150",
    "blue_bossa": "Blue Bossa",
    "yesterday_remastered_2009": "Yesterday",
    "katy_perry_hot_n_cold_official_music_video": "Hot N Cold",
    "nina_simone_feeling_good_lyric_video": "Feeling Good",
    "ben_e_king_stand_by_me_audio": "Stand By Me",
    "the_ronettes_be_my_baby_music_video": "Be My Baby",
    "the_police_every_breath_you_take_official_music_video": "Every Breath",
    "maroon_5_she_will_be_loved_official_music_video": "She Will Be Loved",
    "bein_green": "Bein' Green",
    "rwc_rwc_p001": "RWC 001",
}


def nice(stem):
    return NICE.get(stem, stem[:14])


fig, (a1, a2) = plt.subplots(2, 1, figsize=(12.5, 7.2))
fig.patch.set_facecolor(CARD)

# panel 1: robust drift, strip by group
rng = np.random.default_rng(7)
gy = rng.uniform(-.16, .16, len(gset))
a1.plot([r["drift_robust_pct"] for r in gset], 1 + gy, ".", ms=5,
        color=MUT, alpha=.55, label="GuitarSet (extraits 10–16 mes.)")
lab_i = 0
for r in sorted(real, key=lambda r: r["drift_robust_pct"]):
    x = r["drift_robust_pct"]
    refused = guard_refuses(r)
    a1.plot(x, 0, "x" if refused else "o", ms=7 if refused else 6,
            color=RED if refused else INK, mfc="none", mew=1.4)
    if abs(x) >= 2.0:
        a1.annotate(nice(r["stem"]), (x, 0), textcoords="offset points",
                    xytext=(0, 9 + 14 * (lab_i % 3)), fontsize=7, rotation=30,
                    color=INK)
        lab_i += 1
for thr in (2, 3, 5):
    a1.axvline(thr, ls=":", lw=.8, color=MUT)
    a1.axvline(-thr, ls=":", lw=.8, color=MUT)
a1.set_yticks([0, 1]); a1.set_yticklabels(["vraies chansons (23)",
                                           "GuitarSet (60)"], fontsize=8)
a1.set_ylim(-.5, 1.6)
a1.set_xlabel("dérive robuste sur le morceau (%) — pente OLS des intervalles "
              "hors outliers ±50 ms, × durée / intervalle moyen",
              fontsize=8, color=MUT)
a1.set_title("Dérive de tempo ROBUSTE — les 83 morceaux du cache "
             "(× rouge = déjà refusé par le garde-fou actuel)",
             fontsize=9.5, color=INK)
a1.legend(fontsize=7.5, loc="upper left")
style(a1)

# panel 2: off-modal downbeat share
a2.plot([r["offmodal_downbeat_share"] for r in gset
         if r.get("offmodal_downbeat_share") is not None],
        1 + rng.uniform(-.16, .16, sum(1 for r in gset
            if r.get("offmodal_downbeat_share") is not None)),
        ".", ms=5, color=MUT, alpha=.55)
for r in real:
    x = r.get("offmodal_downbeat_share")
    if x is None:
        continue
    refused = guard_refuses(r)
    a2.plot(x, 0, "x" if refused else "o", ms=7 if refused else 6,
            color=RED if refused else INK, mfc="none", mew=1.4)
    if x >= 0.02:
        a2.annotate(nice(r["stem"]), (x, 0), textcoords="offset points",
                    xytext=(0, 9), fontsize=7, rotation=45, color=INK)
a2.axvspan(0.088, 0.296, color=GREEN, alpha=.10)
a2.text(0.19, 1.45, "le trou : aucune VRAIE chanson\nentre 0,088 et 0,296",
        fontsize=8, color=GREEN, ha="center")
a2.set_yticks([0, 1]); a2.set_yticklabels(["vraies chansons",
                                           "GuitarSet"], fontsize=8)
a2.set_ylim(-.5, 1.75)
a2.set_xlabel("part des downbeats HORS de la phase modale (après le 8e "
              "downbeat) — 0 = toutes les mesures d'accord sur où tombe le 1",
              fontsize=8, color=MUT)
a2.set_title("L'instabilité de PHASE — la mesure qui sépare vraiment",
             fontsize=9.5, color=INK)
style(a2)
fig.tight_layout()
F2 = b64(fig)

# ═══ §3 Beat It repaired ════════════════════════════════════════════════════
BZ = {k: np.load(SCRATCH / f"beatit_{k}.npz") for k in ("live", "db", "dbT")}
titles = {"live": "AVANT — grille rigide off + b×4 (156 mes.)",
          "db": "APRÈS — ré-ancrage sur chaque downbeat (160 mes.)",
          "dbT": "APRÈS + intro purgée (155 mes.)"}
fig, axes = plt.subplots(2, 3, figsize=(13.5, 9),
                         gridspec_kw={"height_ratios": [3, 1]})
fig.patch.set_facecolor(CARD)
for j, k in enumerate(("live", "db", "dbT")):
    z = BZ[k]
    show_ssm(axes[0, j], z["S"], titles[k])
    show_curve(axes[1, j], z["curve"], int(z["L"]), int(z["b0"]), z["peaks"],
               f"bloc glissé — L={int(z['L'])}, phase {int(z['b0'])}, "
               f"{len(z['peaks'])} pics retenus")
fig.tight_layout()
F3 = b64(fig)

# zoom on the only healed zone — TIME-aligned windows (the two grids number
# their bars differently, so comparing equal INDICES compares different music)
T_ROWS, T_COLS = (42.0, 68.0), (30.0, 130.0)
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
fig.patch.set_facecolor(CARD)
for ax, k, ti in [(axes[0], "live", "AVANT — grille rigide, fenêtre 42–68 s "
                   "(la zone décalée d'½ mesure)"),
                  (axes[1], "db", "APRÈS ré-ancrage — la même fenêtre en temps")]:
    z = BZ[k]
    g = z["grid"] if "grid" in z.files else None
    if g is None:  # live grid: reload from capture
        g = np.array(json.loads((SCRATCH / "grids/michael_jackson_beat_it_"
                                 "official_4k_video.json").read_text())["grid"])
    r0 = int(np.searchsorted(g, T_ROWS[0])) - 1
    r1 = int(np.searchsorted(g, T_ROWS[1]))
    c0 = int(np.searchsorted(g, T_COLS[0])) - 1
    c1 = int(np.searchsorted(g, T_COLS[1]))
    Sm = z["S"][r0:r1, c0:c1]
    ax.imshow(Sm, cmap="magma", vmin=0, vmax=1, interpolation="nearest",
              extent=[g[c0], g[min(c1, len(g) - 1)],
                      g[min(r1, len(g) - 1)], g[r0]], aspect="auto")
    ax.set_xlabel("temps (s)", fontsize=8, color=MUT)
    ax.set_ylabel("temps (s)", fontsize=8, color=MUT)
    ax.set_title(ti, fontsize=9, color=INK)
    style(ax)
fig.tight_layout()
F3z = b64(fig)

# ═══ §4 Sunny TI ════════════════════════════════════════════════════════════
def ti_fig(key, label):
    z = np.load(SCRATCH / f"ti_{key}.npz")
    z3 = np.load(SCRATCH / f"ti3_{key}.npz")
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.6),
                             gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor(CARD)
    show_ssm(axes[0, 0], z["S0"], f"{label} — AVANT (dans la tonalité)")
    show_ssm(axes[0, 1], z["S1"], "max sur les 12 rotations")
    show_ssm(axes[0, 2], z3["S3"], "max sur {−1, 0, +1} demi-ton")
    show_curve(axes[1, 0], z["c0"], int(z["L0"]), int(z["b00"]), z["p0"],
               f"L={int(z['L0'])}, phase {int(z['b00'])}, {len(z['p0'])} pics")
    show_curve(axes[1, 1], z["c1"], int(z["L1"]), int(z["b01"]), z["p1"],
               f"L={int(z['L1'])}, phase {int(z['b01'])}, {len(z['p1'])} pics")
    show_curve(axes[1, 2], z3["c3"], int(z3["L3"]), int(z3["b03"]), z3["p3"],
               f"L={int(z3['L3'])}, phase {int(z3['b03'])}, {len(z3['p3'])} pics")
    fig.tight_layout()
    return b64(fig)


F4 = ti_fig("sunny", "Sunny")
F6a = ti_fig("this_love", "This Love")
F6b = ti_fig("norah", "Don't Know Why")

# ΔS maps: what allowing ±1 semitone ADDS
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
fig.patch.set_facecolor(CARD)
for ax, key, label in zip(axes, ("sunny", "this_love", "norah"),
                          ("Sunny", "This Love", "Don't Know Why")):
    z = np.load(SCRATCH / f"ti_{key}.npz")
    z3 = np.load(SCRATCH / f"ti3_{key}.npz")
    d = z3["S3"] - z["S0"]
    im = ax.imshow(d, cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(f"{label} — gain S(±1) − S(0)", fontsize=9, color=INK)
    style(ax)
fig.colorbar(im, ax=axes, fraction=.03, pad=.02)
F5 = b64(fig)

# ═══ numbers for the tables ═════════════════════════════════════════════════
def fmt_row(r):
    return (r["drift_robust_pct"], r.get("offmodal_downbeat_share"),
            guard_refuses(r))


tbl_share = sorted(
    [r for r in real if r.get("offmodal_downbeat_share") is not None],
    key=lambda r: -r["offmodal_downbeat_share"])[:8]

share_rows = "".join(
    f"<tr><td>{nice(r['stem'])}</td>"
    f"<td>{r['offmodal_downbeat_share']:.3f}</td>"
    f"<td>{r['phase_slips']}</td>"
    f"<td>{(r.get('consistency') or 0):.2f}</td>"
    f"<td>{'déjà refusé' if guard_refuses(r) else '<b>PASSE</b>'}</td></tr>"
    for r in tbl_share)

drift_rows = "".join(
    f"<tr><td>{nice(r['stem'])}</td><td>{r['drift_robust_pct']:+.2f} %</td>"
    f"<td>{(r.get('offmodal_downbeat_share') if r.get('offmodal_downbeat_share') is not None else 0):.3f}</td>"
    f"<td>{'déjà refusé' if guard_refuses(r) else 'passe'}</td></tr>"
    for r in sorted(real, key=lambda r: -abs(r["drift_robust_pct"]))[:6])

Z = {k: np.load(SCRATCH / f"ti_{k}.npz") for k in ("sunny", "this_love", "norah")}
Z3 = {k: np.load(SCRATCH / f"ti3_{k}.npz") for k in ("sunny", "this_love", "norah")}
LOCKED = {"sunny": 2, "this_love": 7, "norah": 10}
ti_rows = ""
for k, label in (("sunny", "Sunny"), ("this_love", "This Love"),
                 ("norah", "Don't Know Why")):
    z, z3 = Z[k], Z3[k]
    ti_rows += (
        f"<tr><td>{label}</td>"
        f"<td>L={int(z['L0'])} · {len(z['p0'])} pics (verrouillé : {LOCKED[k]})</td>"
        f"<td>L={int(z['L1'])} · {len(z['p1'])} pics</td>"
        f"<td>L={int(z3['L3'])} · {len(z3['p3'])} pics</td></tr>")

infl_rows = ""
for k, label in (("sunny", "Sunny"), ("this_love", "This Love"),
                 ("norah", "Don't Know Why")):
    def st(S):
        i = np.arange(len(S))
        off = S[np.abs(i[:, None] - i[None, :]) >= 2]
        return f"{off.mean():.2f}"
    infl_rows += (f"<tr><td>{label}</td><td>{st(Z[k]['S0'])}</td>"
                  f"<td>{st(Z3[k]['S3'])}</td><td>{st(Z[k]['S1'])}</td></tr>")

# ═══ HTML ═══════════════════════════════════════════════════════════════════
html = f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Pourquoi la matrice est mauvaise — grille, dérive, modulation</title><style>
body{{margin:0;background:{CREAM};font:15px/1.55 -apple-system,system-ui,sans-serif;color:{INK}}}
.wrap{{max-width:1400px;margin:0 auto;padding:22px 15px 60px}}
h1{{font:italic 600 25px Georgia,serif;margin:0 0 4px}}
.lede{{color:{MUT};font-size:13px;margin-bottom:22px;max-width:920px}}
section{{background:{CARD};border:1px solid #e5dcc6;border-radius:14px;padding:16px 18px;margin-bottom:16px}}
h2{{font:700 17px system-ui;margin:0 0 10px;color:{RED}}}
.sub{{font:500 12px system-ui;color:{MUT}}}
h3{{font:700 11px system-ui;margin:18px 0 6px;color:{MUT};text-transform:uppercase;letter-spacing:.05em}}
img{{max-width:100%;border-radius:8px;margin-bottom:6px}}
.cap{{font-size:12px;color:{MUT};margin:0 0 4px;max-width:900px}}
table{{border-collapse:collapse;font-size:12.5px;margin-top:8px}}
th,td{{border:1px solid #e5dcc6;padding:3px 10px;text-align:left}}
th{{background:#f7f3e9}}
.verdict{{background:#f7f3e9;border-left:3px solid {RED};padding:8px 12px;
font-size:13.5px;margin:10px 0;max-width:900px}}
</style></head><body><div class=wrap>
<h1>Pourquoi la matrice est mauvaise — grille, dérive, modulation</h1>
<div class=lede>La méthode est verrouillée (<code>scripts/harmonic_method.py</code>,
importée telle quelle — les sorties verrouillées This Love 7 pics, Don't Know Why 10,
Sunny 2 sont reproduites exactement ici). Cette page traite l'AMONT : le garde-fou,
la grille de Beat It, la modulation de Sunny. Les matrices d'abord, les chiffres après.</div>

<section><h2>1 · Beat It — la « dérive de −11,4 % » n'existe pas</h2>
<p class=cap>Le chiffre publié vient d'une régression sur TOUS les intervalles entre
beats. Or les 8 premiers (21–30 s : l'intro au gong, avant que le tracker n'accroche
le groove) valent 0,74–1,74 s au lieu de 0,44 s. Placés en tête de régression, ils
fabriquent toute la pente. Sans eux : <b>−0,27 %</b>. Le tempo est métronomique —
l'intervalle médian vaut 0,4400 s dans chacune des 20 tranches du morceau.</p>
<img src="data:image/png;base64,{F1}">
<p class=cap>À droite : le seul vrai litige, invisible du garde-fou actuel. Entre
50 et 60 s, le tracker décale ses downbeats d'une demi-mesure pendant 6 mesures,
puis revient ; partout ailleurs, 151 des 160 downbeats sont en phase 0. Qui a
tort sur ces 6 mesures — le tracker ou la grille rigide ? Le §3 tranche par
l'image : c'est le tracker.</p>
<div class=verdict>La théorie « c'est la dérive » est morte : erreur type n°1 du
projet (un chiffre plausible, jamais confronté à son tracé). Beat It passe le
garde-fou À RAISON — sa grille est bonne à 97 %.</div></section>

<section><h2>2 · Le trou du garde-fou, mesuré sur les 83 morceaux du cache</h2>
<p class=cap>Deux détecteurs candidats, distributions AVANT tout seuil.
<b>Dérive robuste</b> = pente OLS des intervalles entre beats (outliers ±50 ms
exclus) × durée / intervalle moyen. <b>Instabilité de phase</b> = part des
downbeats du tracker qui ne tombent pas sur la phase majoritaire de la grille à
4 beats (calculée après le 8e downbeat pour ignorer l'accrochage).</p>
<img src="data:image/png;base64,{F2}">
<h3>Dérive robuste — les 6 plus grosses vraies chansons</h3>
<table><tr><th>morceau</th><th>dérive robuste</th><th>instab. phase</th><th>garde-fou actuel</th></tr>
{drift_rows}</table>
<p class=cap>Un seuil de dérive à |3 %| ne refuserait que Let It Be (+7,4 %) et
Kermit (+6,5 %) — deux morceaux qui accélèrent VRAIMENT (interprétation humaine,
pas un défaut du tracker). Or une vraie accélération ne casse pas la grille : les
mesures suivent les beats réels, chaque mesure fait toujours 4 vrais beats. La
dérive n'est pas le bon signal.</p>
<h3>Instabilité de phase — les 8 plus grosses vraies chansons</h3>
<table><tr><th>morceau</th><th>instab. phase</th><th>glissements</th><th>cohérence (garde-fou)</th><th>statut</th></tr>
{share_rows}</table>
<div class=verdict>La mesure qui sépare est l'instabilité de phase : les vraies
chansons saines vivent sous 0,09, les suspectes au-dessus de 0,296 — aucune vraie
chanson entre les deux (des extraits GuitarSet y vivent, mais ils font 10–16
mesures : le garde-fou ne les juge pas, seuil n_bars ≥ 30). <b>Proposition :
refuser au-dessus de 0,15</b> (le milieu du trou). Trois prises nouvelles, qui
passent toutes le garde-fou actuel : <b>Chain of Fools</b> (0,486 : UN glissement
précoce qui déphase la moitié du morceau — cohérence 0,99, le garde-fou actuel ne
voit RIEN), le backing track <b>Blue Bossa 150</b> (0,317) et <b>Kermit</b>
(0,296, qui cumule la plus grosse dérive robuste réelle, +6,5 % — du rubato
plausible). Refus à tort : aucun cas clair, mais ces trois-là sont à vérifier à
l'oreille avant de coder le seuil. Ce que ça ne résout PAS : un tracker faux en
BLOC (Close to You) reste l'affaire de la cohérence existante, et une demi-mesure
ponctuelle comme les 6 mesures de Beat It (0,039) reste sous tout seuil
raisonnable — c'est un défaut local, pas un défaut de grille.</div></section>

<section><h2>3 · Beat It — la grille réparée&nbsp;: ré-ancrage sur les downbeats</h2>
<p class=cap>Réparation testée : au lieu d'une phase unique <code>off + b×4</code>
sur les indices de beats, chaque downbeat du tracker RÉ-ANCRE la mesure (les
frontières de mesures sont les downbeats eux-mêmes) ; variante avec l'intro purgée
(départ au premier downbeat suivi de 4 écarts pleins). Les deux grilles ne
diffèrent de la rigide que sur <b>5 mesures (14–18)</b> — la zone 50–60 s — plus
l'intro. Échelle commune 0–1, même méthode verrouillée en dessous.</p>
<img src="data:image/png;base64,{F3}">
<p class=cap>Le zoom sur la seule zone réellement en litige, aligné EN TEMPS
(42–68 s en lignes, 30–130 s en colonnes — les deux grilles numérotent leurs
mesures différemment, comparer des indices comparerait deux musiques) :</p>
<img src="data:image/png;base64,{F3z}">
<div class=verdict>Le ré-ancrage AGGRAVE. Le bloc clair uniforme qui apparaît à
droite (42–60 s) est la signature exacte du mélange demi-mesure : chaque
« mesure » ré-ancrée contient moitié mi, moitié ré, donc toutes se ressemblent —
alors que la grille rigide garde son damier net au même endroit. Lecture : dans
la zone 50–60 s c'est le TRACKER qui a vacillé d'une demi-mesure, pas la musique ;
la grille rigide <code>off + b×4</code> avait raison de l'ignorer, et suivre les
downbeats du tracker propage son erreur (30 pics → 2). Il n'y avait RIEN à
réparer : la grille de Beat It est bonne. Son vrai problème est ailleurs — UN
riff de 2 mesures partout (L=2 détecté, le motif colle sur la moitié du morceau,
30 « pics » tous vrais musicalement, aucun utile pour une forme) plus un bloc
uniforme (~78–95, le breakdown). Morceau harmoniquement homogène : hors du
domaine d'un détecteur de répétition HARMONIQUE. Le bon outil ici serait un
substrat rythmique/timbral.</div></section>

<section><h2>4 · Sunny — la similarité invariante par transposition</h2>
<p class=cap>Sunny module : ses sections F et G sont la même musique un demi-ton
plus haut (mesuré : ratio inter-blocs 0,797 brut → 0,973 après rotation). Test
demandé : S[i,j] = max sur les rotations de chroma du cosinus entre les vecteurs
12-notes des mesures i et j. Deux variantes : les 12 rotations, et {{−1, 0, +1}}
demi-ton (les modulations réelles du corpus sont à un demi-ton). Le max sur R
rotations gonfle mécaniquement TOUTES les similarités — d'où les deux contrôles
This Love et Don't Know Why plus bas, qui ne doivent rien casser.</p>
<img src="data:image/png;base64,{F4}">
<p class=cap>Où le gain de rotation se loge — S(±1) − S(0), c'est-à-dire ce que
la tolérance au demi-ton AJOUTE à chaque case :</p>
<img src="data:image/png;base64,{F5}">
<h3>Les contrôles — This Love, Don't Know Why</h3>
<img src="data:image/png;base64,{F6a}">
<img src="data:image/png;base64,{F6b}">
<h3>Les chiffres (après les images)</h3>
<table><tr><th>morceau</th><th>avant</th><th>max 12 rotations</th><th>max ±1 demi-ton</th></tr>
{ti_rows}</table>
<p class=cap>Inflation mécanique — moyenne hors diagonale (|i−j| ≥ 2) :</p>
<table><tr><th>morceau</th><th>avant</th><th>max ±1</th><th>max 12</th></tr>
{infl_rows}</table>
<div class=verdict>Le max sur 12 rotations DÉTRUIT : moyenne hors-diagonale 0,37 →
0,86 sur Sunny, matrices délavées, contrôles cassés (This Love perd sa phase et
gagne 4 faux pics). Le max sur ±1 demi-ton est la version défendable : les
contrôles gardent période, phase, et leurs pics verrouillés (This Love : les 7
exacts + 2 parasites en fin ; Norah : les 10 exacts + 1) ; Sunny passe de 2 pics
muets à une lecture L=16 avec 6 pics — et sa carte de gain montre des BLOCS
cohérents (les répétitions modulées) là où les contrôles ne montrent que du bruit
diffus. Deux prix payés : l'inflation (0,37 → 0,64) avec 1 à 2 pics parasites par
contrôle ; et surtout, les 6 pics de Sunny vivent tous dans la PREMIÈRE moitié —
Sunny module en chaîne, donc les sections à ≥ 2 demi-tons cumulés de distance
restent invisibles à un « max ±1 » qui ne suit qu'une marche (la courbe retombe
après la mesure 35). En l'état c'est un DIAGNOSTIC de modulation solide, pas
encore un substrat de remplacement. Aucun chiffre de placement n'est produit ici, donc pas
de repère « ne bouge jamais » à opposer.</div></section>

<div class=lede>Prototypes : <code>scripts/grid_debug_core.py</code>,
<code>scripts/grid_debug_report.py</code> · données : cache beats (83 fichiers),
grilles live capturées via le spy de <code>harmonic_method.__main__</code> ·
session 2026-08-05.</div>
</div></body></html>"""

OUT.write_text(html, encoding="utf-8")
print(f"écrit {OUT} ({len(html)//1024} ko)")
