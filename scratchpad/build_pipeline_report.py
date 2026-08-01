"""Génère un rapport HTML interactif expliquant CHAQUE étape du pipeline
harmonia_min pour une chanson : audio → BPM/barres → accords 1ʳᵉ passe →
répétitions (SSM/runs) → sections → empilement + ré-inférence → chart.

Usage: .venv/bin/python scratchpad/build_pipeline_report.py <stem> "<Titre>"
Sortie: harmonia_min/state/reports/<stem>.html  (servi sur
        http://localhost:7772/reports/<stem>.html — l'audio est écoutable
        dedans, chaque décision a ses boutons ▶)
"""
import base64
import copy
import io
import json
import subprocess
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore")
import logging

logging.disable(logging.INFO)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/Users/vincente/Documents/Projets Perso/Code/harmonia")
from harmonia_min import beats as hb
from harmonia_min import musx as hm
from harmonia_min import folding as hf
from harmonia_min import sections as hs
from harmonia_min.nnls_features import extract_bothchroma
import harmonia_min.pipeline as hp

STEM = sys.argv[1]
TITLE = sys.argv[2]
AUDIO = f"docs/audio/{STEM}.m4a"
OUT = f"harmonia_min/state/reports/{STEM}.html"
NOTE = "C Db D Eb E F Gb G Ab A Bb B".split()
INK, ACC, GRN, RED, PAP = "#1c1c1c", "#8a2b2b", "#1f8a5b", "#a8281f", "#f7f3e9"

def pretty(bar):
    return " ".join(("(" + NOTE[c["root"]] + c["q"] + ")" if c.get("carry")
                     else NOTE[c["root"]] + c["q"]) if not c["nc"] else "N.C."
                    for c in bar) or "?"

def fig2b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=105, bbox_inches="tight",
                facecolor="#fcfcfb")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

# ── capture du pipeline complet (pré-fold, stacks, décision par décision) ────
cap = {"redecode": None, "stacks": []}
_orig_rd = hm.redecode
def spy_rd(bt, probs, **kw):
    out = _orig_rd(bt, probs, **kw)
    if cap["redecode"] is None:
        cap["redecode"] = out
    return out
hm.redecode = spy_rd
_orig_tc = hf._template_chords
def spy_tc(pm, bp, npb, Lf, bpb, P):
    out = _orig_tc(pm, bp, npb, Lf, bpb, P)
    avg = []
    for k in range(P):
        mems = [bp(b) for b in pm[k]]
        avg.append(np.mean([hf._resample(m[0], Lf) for m in mems], axis=0))
    cap["stacks"].append({"P": P, "members": [list(g) for g in pm],
                          "avg": avg, "chords": out})
    return out
hf._template_chords = spy_tc
_orig_fold = hf.fold_letter_groups
def spy_fold(sections, bars, grid, probs, bpb, arr=None, times=None):
    cap["prefold"] = copy.deepcopy(bars)
    cap["fold_report"] = _orig_fold(sections, bars, grid, probs, bpb,
                                    arr=arr, times=times)
    return cap["fold_report"]
hf.fold_letter_groups = spy_fold

M = hp.analyze(AUDIO, title=TITLE, file_key=f"min_{STEM}",
               audio_url=f"/audio/{STEM}.m4a")
grid = M["barGrid"]
n_bars = len(grid) - 1
bd = hb.track(AUDIO)
bt, db = np.array(bd["beats"]), np.array(bd["downbeats"])
arr, times = extract_bothchroma(AUDIO)
probs = hm.frame_posteriors(AUDIO)
allbars = {}
for s in M["sections"]:
    for r_i, (b0, b1) in enumerate(s["barRanges"]):
        if r_i == 0:
            for k, bar in enumerate(s["bars"]):
                if b0 + k <= b1:
                    allbars[b0 + k] = bar
bars_l = [allbars.get(b, []) for b in range(n_bars)]
F = hs.halfbar_features(grid, arr, times)
Vb = hf._bar_vecs(F, n_bars)
S = F @ F.T
Sb = hs._blur(S, hs.BLUR_SIGMA)
runs = hs.tiling_runs(Vb, n_bars)
coverage = sum(r["b1"] - r["b0"] + 1 for r in runs) / max(1, n_bars)

figs, aud = {}, []

# F1 — forme d'onde
with tempfile.TemporaryDirectory() as td:
    wav = f"{td}/a.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", AUDIO, "-ac", "1",
                    "-ar", "8000", wav], check=True)
    import wave as _w
    w = _w.open(wav)
    sig = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(float)
    w.close()
sig = sig[:len(sig) - len(sig) % 800]
env = np.abs(sig).reshape(-1, 800).mean(1)
tx = np.linspace(0, len(sig) / 8000, len(env))
fig, ax = plt.subplots(figsize=(12, 2.2))
ax.fill_between(tx, -env, env, color="#8a8371", lw=0)
for t in grid[::8]:
    ax.axvline(t, color=ACC, lw=0.5, alpha=0.5)
ax.set_yticks([]); ax.set_xlabel("secondes")
ax.set_title("L'audio (m4a → décodé) — traits rouges: une barre sur 8 de la grille finale", loc="left", fontsize=10)
figs["wave"] = fig2b64(fig)

# F2 — tempo & barres
fig, axes = plt.subplots(1, 2, figsize=(12, 3))
axes[0].plot(bt[1:], 60 / np.diff(bt), color=INK, lw=1)
axes[0].axhline(bd["bpm"], color=ACC, ls="--", lw=1)
axes[0].set_title(f"tempo instantané (60/écart entre beats) — médiane {bd['bpm']} BPM", loc="left", fontsize=10)
axes[0].set_xlabel("s"); axes[0].set_ylabel("BPM")
dd = np.diff(db)
axes[1].plot(db[1:], dd, color=INK, lw=1)
axes[1].set_title(f"durée de chaque barre (downbeats Beat This!) — méd. {np.median(dd):.2f}s, {M['bpb']} temps/barre", loc="left", fontsize=10)
axes[1].set_xlabel("s"); axes[1].set_ylabel("s/barre")
for a in axes: a.grid(color="#e1e0d9", lw=0.5)
figs["tempo"] = fig2b64(fig)

# F3 — chroma + grille de barres (0-40 s)
sel = times <= 40
fig, ax = plt.subplots(figsize=(12, 3))
ax.imshow(arr[sel, 12:].T, aspect="auto", origin="lower", cmap="YlOrRd",
          extent=[0, 40, 0, 12])
ax.set_yticks(range(12)); ax.set_yticklabels(["A","Bb","B","C","Db","D","Eb","E","F","Gb","G","Ab"], fontsize=7)
for b, t in enumerate(grid):
    if t > 40: break
    ax.axvline(t, color=INK, lw=1.2 if b % 4 == 0 else 0.4, alpha=0.8)
ax.set_title("chroma NNLS (aigus) + grille de barres posée (traits épais = 1 barre sur 4)", loc="left", fontsize=10)
ax.set_xlabel("s")
figs["grid"] = fig2b64(fig)

# F4 — posteriors musx + accords 1ʳᵉ passe (0-40 s)
segs1, lat = cap["redecode"]
tri = probs[0]
fsel = int(40 / hm.FRAME_DT)
top = np.argsort(tri[:fsel].sum(0))[::-1][:8]
TYPES = ["", "m", "sus4", "sus2", "dim", "aug"]
fig, ax = plt.subplots(figsize=(12, 3))
ax.imshow(tri[:fsel, top].T, aspect="auto", origin="upper", cmap="YlOrRd",
          extent=[0, 40, len(top), 0])
ax.set_yticks(np.arange(len(top)) + 0.5)
ax.set_yticklabels(["N" if i == 0 else NOTE[(i-1) % 12] + TYPES[(i-1)//12] for i in top], fontsize=8)
for t0, t1, lab in segs1:
    if t0 > 40: break
    ax.axvline(t0, color=INK, lw=0.8)
    ax.text(t0 + 0.1, -0.4, lab.replace(":", ""), fontsize=6.5, rotation=45, color=INK)
ax.set_title(f"posteriors musx (top-8 accords) + 1ʳᵉ passe décodée (latence {lat*1000:.0f} ms, changements aux demi-barres)", loc="left", fontsize=10)
ax.set_xlabel("s")
figs["musx"] = fig2b64(fig)

# F5 — SSM + runs + coupes
fig, ax = plt.subplots(figsize=(7.5, 7))
ax.imshow(Sb, cmap="RdYlBu_r", origin="lower",
          vmin=np.percentile(Sb, 5), vmax=np.percentile(Sb, 99))
for r in runs:
    ax.add_patch(plt.Rectangle((2*r["b0"], 2*r["b0"]), 2*(r["b1"]-r["b0"]+1),
                 2*(r["b1"]-r["b0"]+1), fill=False, ec=GRN, lw=2))
    ax.text(2*r["b0"]+1, 2*r["b1"], f"P{r['period']}", color=GRN, fontsize=9, fontweight="bold")
for s_ in M["sections"]:
    for b0, _ in s_["barRanges"]:
        ax.axvline(2*b0, color=INK, lw=0.8, alpha=0.6)
ax.set_title(f"SSM floutée (demi-barres) — cadres verts: les RUNS de tuilage (couverture {coverage*100:.0f}%)"
             + ("" if coverage >= hs.RUN_COVERAGE_MIN else " → repli nouveauté"), loc="left", fontsize=10)
tk = list(range(0, 2*n_bars, 16))
ax.set_xticks(tk); ax.set_xticklabels([str(t//2) for t in tk], fontsize=7)
ax.set_yticks(tk); ax.set_yticklabels([str(t//2) for t in tk], fontsize=7)
figs["ssm"] = fig2b64(fig)

# F6 — courbes de tuilage
fig, ax = plt.subplots(figsize=(12, 3))
for P, col in ((2, ACC), (4, "#2a6fb0")):
    xs = range(n_bars - P)
    ax.plot(list(xs), [float(Vb[b] @ Vb[b+P]) for b in xs], color=col, lw=1.2,
            label=f"ressemblance barre ↔ barre+{P}")
ax.axhline(hs.TILE_MIN, color=INK, ls="--", lw=1, label=f"seuil tuilage {hs.TILE_MIN}")
for s_ in M["sections"]:
    ax.axvline(s_["barRanges"][0][0], color=INK, lw=0.8, alpha=0.5)
ax.legend(fontsize=8); ax.set_xlabel("barre"); ax.set_ylim(0, 1.05)
ax.grid(color="#e1e0d9", lw=0.5)
ax.set_title("le tuilage, barre par barre — un « run » = plage au-dessus du seuil ; les chutes = frontières", loc="left", fontsize=10)
figs["tile"] = fig2b64(fig)

# F7 — matrice des lettres
secs = M["sections"]
k = len(secs)
Mm = np.zeros((k, k))
for i in range(k):
    for j in range(k):
        ri = slice(2*secs[i]["barRanges"][0][0], 2*(secs[i]["barRanges"][0][1]+1))
        rj = slice(2*secs[j]["barRanges"][0][0], 2*(secs[j]["barRanges"][0][1]+1))
        Mm[i, j] = Sb[ri, rj].mean()
R = Mm / np.sqrt(np.outer(np.diag(Mm), np.diag(Mm)))
fig, ax = plt.subplots(figsize=(5.5, 5))
im = ax.imshow(R, cmap="RdYlBu_r", vmin=0.6, vmax=1.0)
labels = [f"{s['label']}{i}" for i, s in enumerate(secs)]
ax.set_xticks(range(k)); ax.set_xticklabels(labels, fontsize=8)
ax.set_yticks(range(k)); ax.set_yticklabels(labels, fontsize=8)
for i in range(k):
    for j in range(k):
        ax.text(j, i, f"{R[i,j]:.2f}", ha="center", va="center", fontsize=7,
                color="white" if R[i,j] > 0.93 else INK)
ax.set_title("qui ressemble à qui (ratio de blocs SSM)\n≥0.96 = même lettre", loc="left", fontsize=10)
plt.colorbar(im, fraction=0.046)
figs["letters"] = fig2b64(fig)

# F8 — templates moyennés (jusqu'à 4 positions du plus gros stack)
if cap["stacks"]:
    big = max(cap["stacks"], key=lambda st: sum(len(m) for m in st["members"]))
    P = big["P"]
    fig, axes = plt.subplots(1, min(P, 4), figsize=(3.2*min(P, 4), 3.2), squeeze=False)
    for kk in range(min(P, 4)):
        ax = axes[0][kk]
        A = big["avg"][kk]
        topc = np.argsort(A.sum(0))[::-1][:5]
        ax.imshow(A[:, topc].T, aspect="auto", origin="upper", cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_yticks(range(len(topc)))
        ax.set_yticklabels(["N" if i == 0 else NOTE[(i-1) % 12] + TYPES[(i-1)//12] for i in topc], fontsize=8)
        dec = " ".join((("(" + NOTE[e["root"]] + e["q"] + ")") if e.get("carry")
                        else NOTE[e["root"]] + e["q"]) for e in big["chords"][kk])
        ax.set_title(f"pos {kk+1} — {len(big['members'][kk])} obs\n→ {dec}", fontsize=9, loc="left")
        ax.set_xticks([])
    fig.suptitle("posteriors EMPILÉS (moyennés) par position du motif → consensus décodé", fontsize=10, y=1.04)
    figs["stack"] = fig2b64(fig)

# ── HTML ─────────────────────────────────────────────────────────────────────
def sec_audio_btns():
    rows = []
    for i, s_ in enumerate(secs):
        for pi, (b0, b1) in enumerate(s_["barRanges"]):
            t0, t1 = grid[b0], grid[min(b0+4, b1+1)]
            rows.append(f"<button onclick=\"play({t0:.2f},{t1:.2f})\">▶ {s_['label']}{'' if s_['reps']==1 else f'·passe{pi+1}'} ({mmss(t0)})</button>")
    return " ".join(rows)

def mmss(t): return f"{int(t//60)}:{int(t%60):02d}"

boundaries = []
for s_ in secs[1:]:
    b0 = s_["barRanges"][0][0]
    boundaries.append((s_["label"], b0, grid[max(0, b0-2)], grid[min(n_bars, b0+2)]))
bnd_btns = " ".join(f"<button onclick=\"play({a:.2f},{b:.2f})\">▶ frontière →{L} (barre {bb}, {mmss(grid[bb])})</button>"
                    for L, bb, a, b in boundaries)

cell_btns = ""
if cap["stacks"]:
    big = max(cap["stacks"], key=lambda st: sum(len(m) for m in st["members"]))
    mem0 = big["members"][0][:6]
    cell_btns = " ".join(f"<button onclick=\"play({grid[b]:.2f},{grid[min(n_bars,b+big['P'])]:.2f})\">▶ occurrence {i+1} (barre {b})</button>"
                         for i, b in enumerate(mem0))

fr = M.get("fold") or {}
fold_rows = ""
for L, v in fr.items():
    if v.get("n_obs"):
        fold_rows += f"<tr><td>{L}</td><td>P{v['period']}</td><td>{v['n_obs']}</td><td>{len(v['variants'])}</td><td>{len(v['changed'])}</td></tr>"
    else:
        fold_rows += f"<tr><td>{L}</td><td colspan=4>pas de pli — {v.get('reason','')}</td></tr>"

changed_rows = ""
pre = cap.get("prefold", [])
for L, v in fr.items():
    for b in (v.get("changed") or [])[:14]:
        if b < len(pre):
            changed_rows += (f"<tr><td>{b}</td><td>{mmss(grid[b])}</td><td>{pretty(pre[b])}</td>"
                             f"<td>{pretty(bars_l[b])}</td>"
                             f"<td><button onclick=\"play({grid[b]:.2f},{grid[min(n_bars,b+1)]:.2f})\">▶</button></td></tr>")

form = " ".join(f"{s_['label']}{'×'+str(s_['reps']) if s_['reps']>1 else ''}" for s_ in secs)
sections_rows = "".join(
    f"<tr><td>{s_['label']}×{s_['reps']}</td><td>{s_['barRanges'][0][0]}–{s_['barRanges'][0][1]}</td>"
    f"<td>{mmss(s_['spans'][0][0])}</td><td>{s_['barRanges'][0][1]-s_['barRanges'][0][0]+1}b</td>"
    f"<td>{pretty(s_['bars'][0])} | {pretty(s_['bars'][1]) if len(s_['bars'])>1 else ''}</td></tr>"
    for s_ in secs)
runs_rows = "".join(f"<tr><td>{r['b0']}–{r['b1']}</td><td>P{r['period']}</td>"
                    f"<td>{mmss(grid[r['b0']])}–{mmss(grid[r['b1']+1])}</td>"
                    f"<td><button onclick=\"play({grid[r['b0']]:.2f},{grid[min(n_bars,r['b0']+2*r['period'])]:.2f})\">▶ 2 cellules</button></td></tr>"
                    for r in runs)

html = f"""<!DOCTYPE html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pipeline — {TITLE}</title><style>
body{{font-family:-apple-system,Georgia,serif;background:#e7e0d0;color:{INK};margin:0;padding:20px;line-height:1.55}}
main{{max-width:960px;margin:0 auto}}
h1{{font-size:26px}} h2{{color:{ACC};border-bottom:2px solid {ACC};padding-bottom:4px;margin-top:44px}}
.card{{background:{PAP};border-radius:12px;padding:16px 20px;margin:14px 0;box-shadow:0 4px 14px -8px rgba(50,35,20,.4)}}
img{{max-width:100%;border-radius:8px}}
button{{background:{ACC};color:#fff;border:none;border-radius:8px;padding:6px 10px;margin:3px;cursor:pointer;font-size:13px}}
table{{border-collapse:collapse;width:100%;font-size:13.5px}} td,th{{border-bottom:1px solid #cdc4ad;padding:5px 8px;text-align:left}}
.met{{display:inline-block;background:#fffdf6;border:1px solid #cdc4ad;border-radius:8px;padding:6px 12px;margin:4px;font-size:14px}}
.met b{{font-size:18px;color:{ACC}}}
nav{{position:sticky;top:0;background:#e7e0d0ee;padding:8px 0;z-index:5;font-size:13px}}
nav a{{color:{ACC};margin-right:12px;text-decoration:none;font-weight:600}}
audio{{width:100%}}</style></head><body><main>
<h1>{TITLE} — le pipeline, étape par étape</h1>
<p>Chaque section montre <b>la décision prise</b>, <b>les métriques qui l'ont menée</b>, et <b>l'audio à écouter</b> pour la juger.</p>
<audio id="aud" controls src="/audio/{STEM}.m4a"></audio>
<nav><a href="#s1">1·Audio</a><a href="#s2">2·BPM/Barres</a><a href="#s3">3·Accords 1ʳᵉ passe</a>
<a href="#s4">4·Répétitions</a><a href="#s5">5·Sections</a><a href="#s6">6·Empilement</a><a href="#s7">7·Chart</a></nav>

<h2 id="s1">1 · L'audio — la source de tout (règle 0)</h2>
<div class="card"><img src="data:image/png;base64,{figs['wave']}"></div>

<h2 id="s2">2 · BPM et placement des barres (Beat This!)</h2>
<div class="card">
<span class="met">BPM médian <b>{bd['bpm']}</b></span>
<span class="met">beats détectés <b>{len(bt)}</b></span>
<span class="met">downbeats <b>{len(db)}</b></span>
<span class="met">temps/barre <b>{M['bpb']}</b></span>
<span class="met">barres posées <b>{n_bars}</b></span>
<p>Beat This! donne les beats ET les premiers temps. La grille est ancrée sur eux
(arithmétique d'indices de beats, jamais de contenance temporelle). À partir d'ici,
<b>la barre est l'unité de vérité</b> — tout le reste se dit en barres/demi-barres.</p>
<img src="data:image/png;base64,{figs['tempo']}">
<img src="data:image/png;base64,{figs['grid']}">
<p>Écoute les premières barres pour vérifier la grille :
<button onclick="play({grid[0]:.2f},{grid[4]:.2f})">▶ barres 1–4</button>
<button onclick="play({grid[8]:.2f},{grid[12]:.2f})">▶ barres 9–12</button></p></div>

<h2 id="s3">3 · Accords, première passe (musx)</h2>
<div class="card">
<span class="met">segments décodés <b>{len(segs1)}</b></span>
<span class="met">latence choisie <b>{lat*1000:.0f} ms</b></span>
<span class="met">changements <b>demi-barre uniquement</b></span>
<p>musx lit ses posteriors de frames (23 ms) ; on les re-décode en n'autorisant les
changements d'accord qu'aux barres et demi-barres (ta règle « premier niveau de
fiabilité »). Pas d'accord → le dernier se propage (copies marquées, plus claires).</p>
<img src="data:image/png;base64,{figs['musx']}"></div>

<h2 id="s4">4 · Détection des répétitions (SSM + runs de tuilage)</h2>
<div class="card">
<span class="met">runs trouvés <b>{len(runs)}</b></span>
<span class="met">couverture <b>{coverage*100:.0f}%</b></span>
<span class="met">mode <b>{"runs (option A)" if coverage >= hs.RUN_COVERAGE_MIN else "repli nouveauté"}</b></span>
<p>Un <b>run</b> = plage où chaque barre ressemble à la barre ±P (la cellule tuile).
Les frontières de sections = les bords des runs — chaque section démarre donc sur
la première barre de sa cellule, et « multiples de 2 » se compte depuis là.</p>
<img src="data:image/png;base64,{figs['tile']}">
<img src="data:image/png;base64,{figs['ssm']}">
<table><tr><th>run (barres)</th><th>période</th><th>temps</th><th>écouter</th></tr>{runs_rows}</table></div>

<h2 id="s5">5 · Sections : coupes, lettres, forme</h2>
<div class="card">
<span class="met">forme <b>{form}</b></span>
<p>Les lettres viennent des blocs croisés de la SSM (deux sections partagent une
lettre quand leur ressemblance croisée ≈ leur ressemblance interne, seuil 0.96).
Les queues de cadence (attaque + tenue) roulent dans la section qui se ferme.</p>
<img src="data:image/png;base64,{figs['letters']}">
<table><tr><th>section</th><th>barres</th><th>début</th><th>long.</th><th>ouverture</th></tr>{sections_rows}</table>
<p><b>Écouter chaque frontière</b> (2 barres avant → 2 barres après) — le test décisif :</p>
<p>{bnd_btns}</p>
<p><b>Écouter chaque section</b> (ses 4 premières barres, chaque passe) :</p>
<p>{sec_audio_btns()}</p></div>

<h2 id="s6">6 · Empilement des répétitions + ré-inférence (2ᵉ passe musx)</h2>
<div class="card">
<table><tr><th>lettre</th><th>période</th><th>observations/position</th><th>variantes</th><th>barres changées</th></tr>{fold_rows}</table>
<p>Les barres de même position s'empilent, leurs posteriors se moyennent (bruit ÷√n),
musx re-décode le motif une seconde fois, et le consensus se redistribue. Les
<b>variantes</b> (écart individuel anormal vs l'écart collectif, z&gt;3) gardent leur
propre lecture — typiquement la barre de transition en fin de section.</p>
{f'<img src="data:image/png;base64,{figs["stack"]}">' if "stack" in figs else ""}
<p><b>Écouter les occurrences empilées de la cellule principale</b> (elles doivent être « la même chose ») :</p>
<p>{cell_btns}</p>
<p><b>Ce que la 2ᵉ passe a corrigé</b> (avant → après, à écouter) :</p>
<table><tr><th>barre</th><th>temps</th><th>1ʳᵉ passe</th><th>consensus</th><th></th></tr>{changed_rows or "<tr><td colspan=5>aucun changement</td></tr>"}</table></div>

<h2 id="s7">7 · Le chart final</h2>
<div class="card"><p>Forme <b>{form}</b>, clé <b>{M['keyName']}</b>.
<a href="/">Ouvrir dans l'app</a> — le chart rendu est exactement ce que les étapes
ci-dessus ont produit, sans post-traitement.</p></div>

<script>
let _st=null;
function play(t0,t1){{const a=document.getElementById('aud');a.currentTime=t0;a.play();
if(_st)clearTimeout(_st);_st=setTimeout(()=>a.pause(),(t1-t0)*1000);}}
</script></main></body></html>"""

import pathlib
pathlib.Path(OUT).write_text(html, encoding="utf-8")
print(f"wrote {OUT} ({len(html)//1024} KB)")
