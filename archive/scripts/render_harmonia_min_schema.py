"""Regenerate docs/harmonia_min_schema.png — the harmonia_min pipeline map.

The 2026-07-31 version was a one-off untracked script; this one is committed
so the schema can be kept in sync with the code. Heights and y-positions are
computed from line counts (no hand-tuned overlaps).
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BG = "#f6f1e4"
BOX = "#fffdf6"
EDGE = "#8a7a5c"
INK = "#3e3428"
SUB = "#6b5d45"
WARN = "#a04b2f"
ACC = {"in": "#7a6a3f", "ana": "#4a6f8a", "disk": "#7d6a8a", "serve": "#5f7d54"}

F = "Georgia"
LINE = 1.32         # vertical units per body line (fits 8.0pt x 1.3 spacing)
HEAD = 2.9          # title slot
PAD_BOT = 0.8
fig, ax = plt.subplots(figsize=(16.5, 11.5), dpi=170)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")


def box_h(lines, warn):
    return HEAD + LINE * len(lines) + (LINE if warn else 0) + PAD_BOT


def draw_box(x, y, w, h, title, lines, acc, size, warn):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35",
                                fc=BOX, ec=EDGE, lw=1.1, zorder=2))
    ax.add_patch(FancyBboxPatch((x, y), 0.55, h, boxstyle="round,pad=0.35",
                                fc=acc, ec=acc, lw=0, zorder=3))
    ax.text(x + 1.4, y + h - 1.5, title, fontsize=10.5, family=F,
            color=INK, weight="bold", va="top", zorder=4)
    ax.text(x + 1.4, y + h - HEAD, "\n".join(lines), fontsize=size, family=F,
            color=SUB, va="top", zorder=4, linespacing=1.3)
    if warn:
        ax.text(x + 1.4, y + 0.5, warn, fontsize=7.4, family=F, color=WARN,
                va="bottom", zorder=4, style="italic")


def stack(x, w, y_top, items, gap=1.1, arrows=True, size=8.0):
    """items = [(title, lines, acc, warn)]; returns list of (y, h)."""
    pos = []
    y = y_top
    for title, lines, acc, warn in items:
        h = box_h(lines, warn)
        y -= h
        draw_box(x, y, w, h, title, lines, acc, size, warn)
        pos.append((y, h))
        y -= gap
    if arrows:
        for (y0, _), (y1, h1) in zip(pos, pos[1:]):
            arrow(x + w / 2, y0, x + w / 2, y1 + h1 + 0.05)
    return pos


def arrow(x0, y0, x1, y1, label=None, color=SUB):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=13, color=color, lw=1.4,
                                 zorder=1))
    if label:
        ax.text((x0 + x1) / 2 + 0.8, (y0 + y1) / 2, label, fontsize=7.6,
                family=F, color=color, va="center", zorder=4, style="italic")


ax.text(2, 97.8, "HARMONIA MIN — la pipeline actuelle", fontsize=19,
        family=F, color=INK, weight="bold")
ax.text(2, 95.6, "branche feat/chord-lm @ 3c58773 · 2026-08-05 · "
        "harmonia_min/ = ~4 000 lignes de Python + app_shell.html (client)",
        fontsize=9.5, family=F, color=SUB)

# ---- colonne gauche : analyse (empilée, hauteurs calculées) -----------------
left = [
    ("0 · AUDIO — recherche & acquisition", [
        "POST /api/yt-search : locaux docs/audio/*.m4a d'abord, puis yt-dlp (vraie recherche YouTube).",
        "POST /api/analyze {url} -> job asynchrone ; le client sonde GET /api/job/<id> (stage 0 -> 5).",
    ], ACC["in"], None),
    ("1 · BEATS — beats.py (99 l.)", [
        "Beat This! uniquement (librosa banni : octave 2x) ; m4a -> ffmpeg wav si le décodage échoue.",
        "Sortie : {beats:[s], downbeats:[s], bpm} ; cache state/beats/<stem>.json.",
    ], ACC["ana"], "(!) pas de fallback : si Beat This! tombe, l'analyse s'arrête (voulu)"),
    ("2 · ACCORDS bruts — musx.py (338 l.)", [
        "music-x-lab ISMIR2019 (le SOTA), ensemble 5-fold vendored ; posteriors par frame (43,07 fps) :",
        "triad(T,73) + bass(T,13) + 7e/9e/11e/13e ; cache data/cache/musx_probs/<stem>.npz.",
        "redecode() : Viterbi qui force les changements SUR NOS beats (coûts gradués downbeat / demi-",
        "mesure), latence auto-compensée (grille 0-280 ms choisie par log-vraisemblance du chemin).",
    ], ACC["ana"], None),
    ("3 · MESURES — pipeline.py (bar layout)", [
        "Onsets snappés à l'INDEX de beat (pas au temps) ; vote de phase : si >=55 % des accords tombent",
        "sur le même beat != 0, les accords gagnent contre le downbeat du tracker ; accord tenu = carry.",
    ], ACC["ana"], None),
    ("4 · SECTIONS — sections.py (490 l.)", [
        "Substrat : chroma NNLS brut par demi-mesure (PAS les accords décodés) -> SSM non floutée.",
        "Frontières = UNION(runs de tuilage P dans {2,4,8} ; pics de nouveauté checkerboard) :",
        "41,8 % exact-bar sur 285 Billboard (runs seuls 35,8 ; pics seuls 30,1) ; lettres : ratio 0,96.",
    ], ACC["ana"], "(!) lettres non invariantes à la transposition (Sunny : une modulation = nouvelle lettre)"),
    ("5 · FOLDING — folding.py (555 l.)", [
        "Par lettre : période interne (2/4/8), empilement des occurrences, filtrage outliers (z=3),",
        "MOYENNE des posteriors musx du stack, re-décodage du template -> accords consensus réécrits.",
        "minimal_fold : 1 section affichée par lettre, reps + barSpans (le contrat du playhead).",
    ], ACC["ana"], "(!) un fold REFUSÉ garde son champ period — toujours vérifier reason (bug corrigé le 02/08)"),
    ("6 · TONALITÉ & COULEURS — harmonic_key.py (v7c)", [
        "infer_key : Krumhansl-Schmuckler bayésien sur le chroma brut (clé globale).",
        "Tonique causale (CUSUM sur la masse interdite b2+#4, tenue jusqu'à preuve), audit du mode,",
        "HMM collant 4 états (naturel/harmonique/dorien/mélodique) -> colour/inflect/flag/sug par accord.",
    ], ACC["ana"], None),
    ("7 · CHARTMODEL — state/charts/<file>.json", [
        "{key, keySegments, sections, fold, bars[[{root,q,bass,c,t0,t1,carry,colour...}]], barGrid, barSpans, meta}",
    ], ACC["in"], None),
]
lpos = stack(2, 55, 94.6, left)

# ---- colonne droite : disque / annexes --------------------------------------
right = [
    ("DISQUE — caches & modèles", [
        "docs/audio/<video_id>.m4a (yt-dlp) · state/beats/<stem>.json",
        "data/cache/musx_probs/<stem>.npz (3-9 Mo/chanson, partagé avec l'ancien)",
        "data/cache/nnls_infer/<stem>.npz (chroma VAMP, partagé)",
        "state/charts/*.json · state/annotations/*.json",
        "Modèles : harmonia/third_party/ISMIR2019.../ (poids musx, vendored),",
        "data/models/chord_lm_*.pt (13 Mo), data/cache/chord_context_prior*.npz,",
        "harmonia/models/nnls24_heads.npz (fallback uniquement).",
    ], ACC["disk"], None),
    ("LIENS RÉSIDUELS avec l'ancien harmonia/", [
        "Imports Python : 2, jamais sur le chemin live —",
        "   chord_context_prior -> fine_to_q5 (ré-entraînement du prior) ;",
        "   chord_lm/corpus -> ireal_corpus (corpus d'entraînement du LM).",
        "Fichiers : poids musx sous harmonia/third_party/, nnls24_heads.npz,",
        "caches data/cache/ partagés.",
        "-> couper le cordon = déplacer ces 3 chemins + copier 2 loaders.",
    ], ACC["disk"], None),
    ("OFF PAR DÉFAUT — chord_lm/ (1 944 l.)", [
        "Transformer 256d x 4 couches, RoPE, cloze masqué, vocab 90 tokens",
        "(7 familles d'accords, grille par demi-mesure), entraîné sur 2 401 grilles iReal.",
        "HARMONIA_CHORD_LM_SUGGEST=1 -> suggestions seulement (jamais de réécriture) :",
        "gate deux côtés LM>=0,80 & pipeline<=0,60 ; net 0 sur GuitarSet -> resté OFF.",
    ], ACC["disk"], None),
]
stack(61, 37, 94.2, right, gap=2.0, arrows=False, size=8.0)

# ---- bande basse : servir + interaction -------------------------------------
serve_lines = [
    "GET / -> app_shell.html (client complet, 4 113 l., jamais modifié par le serveur).",
    "GET /api/chart-model/<f> -> chart JSON + overlay des annotations côté serveur.",
    "GET /api/library -> liste + capabilities (\"reinfer\" seulement si le prior est entraîné).",
    "GET /min/<f> -> vue minimale HTML ; POST/GET /api/annotations/<f> (écriture atomique).",
    "Route inconnue -> 404 honnête \"pas dans le milestone 1\" (l'UI dégrade proprement).",
]
sh = box_h(serve_lines, None)
inter_warn = "(!) un lock ne deplace jamais une frontiere · la 7e est perdue sur un span deplace (QUAL5)"
inter_lines = [
    "Lock d'un accord -> POST /api/context_rescore/<f> (alias /api/reinfer) :",
    "re-decodage DIFFERENTIEL (baseline sans lock vs lock clampe, memes",
    "params) -> seuls les changements CAUSES par le lock sont appliques.",
    "Evidence : musx_probs (cache) ; prior trigramme relatif-a-la-cible",
    "(lambda=2, delta=0,5, K=6).",
]
ih = box_h(inter_lines, inter_warn)
top_band = lpos[-1][0] - 2.2
y_serve = top_band - sh
y_inter = top_band - ih
draw_box(2, y_serve, 55, sh, "SERVIR — harmonia_min/server.py (Flask, port 7772)",
         serve_lines, ACC["serve"], 8.0, None)
draw_box(61, y_inter, 37, ih, "INTERACTION — lock & propagation", inter_lines,
         ACC["serve"], 8.0, inter_warn)
arrow(29.5, lpos[-1][0], 29.5, y_serve + sh + 0.1)
arrow(57, y_serve + sh / 2, 61, y_inter + ih / 2, color=ACC["serve"])

y_foot = min(y_serve, y_inter) - 2.0
ax.text(2, y_foot, "Reglages payes cher (ne pas reapprendre) : librosa banni · latence musx auto ·"
        " vote de phase des onsets · SSM non floutee · union runs+pics ·"
        " verifier fold.reason · diff differentiel pour le lock",
        fontsize=8.6, family=F, color=WARN, style="italic")
ax.text(2, y_foot - 1.7, "Genere par scripts/render_harmonia_min_schema.py — a regenerer quand la pipeline bouge.",
        fontsize=7.6, family=F, color=SUB)
ax.set_ylim(y_foot - 3.2, 100)

fig.savefig("docs/harmonia_min_schema.png", bbox_inches="tight", facecolor=BG)
print("ok")
