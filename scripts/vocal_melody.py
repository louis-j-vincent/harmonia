"""La mélodie chantée, extraite de la voix et rejouée au piano — façon tutoriel.

    python scripts/vocal_melody.py [<stem> ...]  ->  /reports/vocal_melody.html

Louis, 2026-08-06 :

  « Vu que tu peux extraire la voix, on peut extraire la mélodie chantée et
    choper les notes, et faire un petit tutoriel de comment jouer ces notes-là.
    Pour l'affichage, ce qui serait cool ce serait le défilement type : tu as le
    clavier en bas et les notes qui tombent dessus, comme dans les tutoriels
    YouTube. »

LA CHAÎNE, EN QUATRE ÉTAPES

1. **La voix seule** — demucs deux-pistes (`vocal_anchor.separate_vocals`), déjà
   en cache pour les sept morceaux de la liste.
2. **La hauteur** — pyin sur la piste vocale (même appel que
   `blocks8.sing_onset`), 11,6 ms par trame. On ne garde que les trames voisées
   ET assez fortes : un stem de voix n'est jamais silencieux, il respire, il
   souffle, et pyin donne volontiers une hauteur à un souffle.
3. **Les notes** — la hauteur continue devient un ESCALIER : on corrige d'abord
   le désaccord global du morceau (les disques ne sont pas tous à 440), on
   arrondit au demi-ton, on passe un filtre médian, et une note est un palier de
   l'escalier. Un trou de moins de 80 ms au milieu d'un palier ne le coupe pas
   (les consonnes coupent la voix, pas la note) ; un palier de moins de 100 ms
   est jeté. C'est exactement « une suite de trames dont la hauteur médiane ne
   bouge pas de plus d'un demi-demi-ton » : arrondir au demi-ton, c'est rester à
   ±0,5 d'un entier.
4. **La grille** — les mesures viennent du vrai pipeline (`pattern_lanes.load`,
   downbeats Beat This!), quatre temps par mesure, et on cale une copie des
   notes sur les demi-temps. La page affiche les DEUX et un bouton bascule.

CE QUI EST AFFICHÉ PAR DÉFAUT : **le brut**, pas le calé. Le test honnête est le
bouton « les deux ensemble » — si la mélodie extraite colle au disque, ça sonne
verrouillé ; le calage sur la grille, lui, corrige le rythme mais peut décoller
de l'enregistrement quand la grille se trompe. Le brut ne ment pas.

L'ERREUR CLASSIQUE DE PYIN, c'est l'octave : un unisson mal résolu et la note
part une octave trop bas. On la corrige après coup (une note isolée à ≥ 8
demi-tons de ses voisines qui rentre dans le rang en la décalant de ±12), et le
compte des corrections est écrit sur la page, morceau par morceau.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
sys.path.insert(0, str(HERE / "scratchpad"))
from pattern_lanes import load, fig2b64_fixed, PLOT_L, PLOT_R    # noqa: E402
from vocal_anchor import separate_vocals                         # noqa: E402
from rhythm_ssm import DEFAULT_STEM_CACHE                        # noqa: E402

INK, ACC = "#1c1c1c", "#8a2b2b"
SR, HOP = 22050, 256          # 11,6 ms par trame
LOUD_Q = 0.12                 # seuil d'énergie, en fraction du 95e centile
SMOOTH = 5                    # médiane sur la hauteur continue (~58 ms)
STAIR = 7                     # …puis sur l'escalier arrondi (~81 ms)
MAX_GAP = 0.080               # un trou plus court que ça ne coupe pas la note
MIN_DUR = 0.130               # …et une note plus courte que ça n'existe pas. À 100
                              # à la noire, une double croche fait 150 ms : en
                              # dessous de 130 ms on ne jette pas de musique, on
                              # jette des transitions. Mesuré : +1 à +2,5 points
                              # de « dans la gamme » sur les sept morceaux.
GLISS = 0.18                  # un passage court ENTRE deux notes, dans le sens du
                              # saut, est un portamento, pas une note
OCT_FAR = 8.0                 # note « aberrante » : à ce point loin de ses voisines
OCT_WIN = 3.0                 # …mesuré sur ses voisines à ± ce nombre de secondes
MERGE_GAP = 0.12              # deux notes de même hauteur si proches = une seule
BLIP = 0.14                   # …et un hoquet plus court que ça entre deux fois la
                              # même note est un hoquet, pas une note
OUT_FAR = 12.0                # ce qui reste à une octave de ses voisines est jeté
SUB = 2                       # calage : 2 = au demi-temps (croche)
BEATS_PER_BAR = 4

DEFAULT = ["norah_jones_don_t_know_why", "bein_green", "maroon_5_this_love",
           "mayer_hawthorne_the_walk", "let_it_be_remastered_2009",
           "bruno_mars_grenade_official_music_video",
           "maroon_5_she_will_be_loved_official_music_video"]

NAMES = {
    "norah_jones_don_t_know_why": "Norah Jones — Don't Know Why",
    "bein_green": "Kermit — Bein' Green",
    "maroon_5_this_love": "Maroon 5 — This Love",
    "mayer_hawthorne_the_walk": "Mayer Hawthorne — The Walk",
    "let_it_be_remastered_2009": "The Beatles — Let It Be",
    "bruno_mars_grenade_official_music_video": "Bruno Mars — Grenade",
    "maroon_5_she_will_be_loved_official_music_video": "Maroon 5 — She Will Be Loved",
}

# noms en BÉMOLS : un musicien lit Si♭ et Mi♭, pas La♯ et Ré♯
PC = ["Do", "Ré♭", "Ré", "Mi♭", "Mi", "Fa", "Sol♭", "Sol", "La♭", "La", "Si♭", "Si"]
BLACK = {1, 3, 6, 8, 10}


def note_name(m):
    return f"{PC[int(m) % 12]}{int(m) // 12 - 1}"


# ── 1. la hauteur, mise en cache (pyin coûte ~30 s par morceau) ─────────────
def track_f0(vocal_path, cache_dir=DEFAULT_STEM_CACHE):
    """pyin sur la piste vocale. Retourne (t, f0 Hz, voisé, énergie)."""
    cache = Path(cache_dir) / "f0" / f"{Path(vocal_path).parent.name}.npz"
    if cache.exists():
        d = np.load(cache)
        return d["t"], d["f0"], d["voiced"], d["rms"]
    import librosa
    y, sr = librosa.load(str(vocal_path), sr=SR, mono=True)
    f0, voiced, _ = librosa.pyin(y, sr=sr, hop_length=HOP,
                                 fmin=librosa.note_to_hz("C2"),
                                 fmax=librosa.note_to_hz("C6"))
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    n = min(len(f0), len(rms))
    f0, voiced, rms = f0[:n], voiced[:n], rms[:n]
    t = librosa.frames_to_time(np.arange(n), sr=sr, hop_length=HOP)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, t=t, f0=np.where(np.isfinite(f0), f0, 0.0),
                        voiced=voiced, rms=rms)
    return t, np.where(np.isfinite(f0), f0, 0.0), voiced, rms


def first_sing(vocal_path, cache_dir=DEFAULT_STEM_CACHE):
    """Le premier instant CHANTÉ, par le critère de `blocks8` (la hauteur bouge).

    Sert de témoin indépendant : tout ce que la mélodie place AVANT n'est pas du
    chant, c'est de la fuite dans le stem — un souffle, une nappe, une guitare
    que demucs a laissée passer. Mis en cache : le critère refait un pyin.
    """
    cache = Path(cache_dir) / "f0" / f"{Path(vocal_path).parent.name}.sing"
    if cache.exists():
        try:
            return float(cache.read_text())
        except ValueError:
            return None
    import blocks8
    onset = blocks8.sing_onset(vocal_path)[0]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("" if onset is None else f"{onset:.4f}")
    return onset


def tuning_offset(midi):
    """Le désaccord global du morceau, en demi-tons (≈ ±0,5 max).

    Un disque n'est pas forcément à 440 Hz, et un chanteur pas forcément juste :
    si tout le morceau est 30 centièmes sous le demi-ton, l'arrondi hésite entre
    deux notes sur TOUTE la chanson et l'escalier tremble. On mesure donc la
    phase moyenne de la partie fractionnaire (moyenne circulaire, la seule
    correcte ici : 0,49 et −0,49 sont voisins) et on la retire avant d'arrondir.
    """
    frac = midi - np.round(midi)
    if len(frac) < 50:
        return 0.0
    z = np.exp(2j * np.pi * frac).mean()
    return float(np.angle(z) / (2 * np.pi))


def _medfilt(x, k):
    from scipy.ndimage import median_filter
    return median_filter(x, size=k, mode="nearest")


def melody_notes(t, f0, voiced, rms):
    """L'escalier : hauteur continue -> liste de notes (début, durée, midi).

    Retourne (notes, diag) où diag porte tout ce qui sert à juger l'extraction :
    le masque « ça chante », l'escalier lui-même, le désaccord corrigé.
    """
    ok = voiced & (f0 > 0) & (rms > max(1e-4, LOUD_Q * float(np.percentile(rms, 95))))
    midi = np.full(len(t), np.nan)
    midi[ok] = 69 + 12 * np.log2(f0[ok] / 440.0)

    off = tuning_offset(midi[ok])
    midi = midi - off

    # lisser la hauteur CONTINUE avant d'arrondir : sinon le vibrato (±0,4
    # demi-ton, tout à fait normal) fait clignoter l'escalier entre deux notes
    filled = np.copy(midi)
    idx = np.arange(len(filled))
    if ok.any():
        filled = np.interp(idx, idx[ok], midi[ok])
    smooth = _medfilt(filled, SMOOTH)
    stair = _medfilt(np.round(smooth), STAIR)
    stair[~ok] = np.nan

    # les paliers, avec pontage des trous courts de MÊME hauteur
    notes = []
    i, n = 0, len(t)
    dt = float(np.median(np.diff(t))) if n > 1 else 0.0116
    while i < n:
        if not ok[i]:
            i += 1
            continue
        p, i0, last = stair[i], i, i
        j = i + 1
        while j < n:
            if ok[j] and stair[j] == p:
                last, j = j, j + 1
            elif t[j] - t[last] <= MAX_GAP:      # trou court : on regarde après
                j += 1
            else:
                break
        dur = float(t[last] - t[i0] + dt)
        if dur >= MIN_DUR:
            notes.append([float(t[i0]), dur, int(p)])
        i = last + 1
    return notes, {"ok": ok, "stair": stair, "tuning": off, "midi": midi}


def fix_octaves(notes):
    """La panne classique de pyin : une note isolée une octave en dessous.

    Une note qui est à ≥ OCT_FAR demi-tons de la MÉDIANE de ses voisines (± 3 s)
    et qui rentre dans le rang si on la décale d'une octave est décalée. On rend
    le nombre de corrections — la page l'affiche, parce que beaucoup de
    corrections veut dire une extraction fragile même après réparation.
    """
    if len(notes) < 3:
        return notes, 0
    a = np.array(notes, float)
    fixed, k = a.copy(), 0
    for i in range(len(a)):
        m = (np.abs(a[:, 0] - a[i, 0]) <= OCT_WIN)
        m[i] = False
        if m.sum() < 2:
            continue
        med = float(np.median(a[m, 2]))
        d = fixed[i, 2] - med
        if abs(d) >= OCT_FAR:
            cand = fixed[i, 2] - 12 * np.sign(d) * round(abs(d) / 12)
            if abs(cand - med) < abs(d) - 2:      # il faut un vrai gain
                fixed[i, 2], k = cand, k + 1
    return [[float(x[0]), float(x[1]), int(x[2])] for x in fixed], k


def leap_rate(notes):
    """Part des intervalles enchaînés d'au moins 8 demi-tons.

    Une mélodie chantée avance par petits pas : la médiane des intervalles est
    un ton sur les sept morceaux. Un taux de grands sauts qui monte, c'est
    l'octave mal résolue. C'est aussi le seul juge disponible pour la CORRECTION
    d'octave — un décalage d'octave ne change pas la note, donc « dans la
    gamme » y est aveugle.
    """
    if len(notes) < 3:
        return 0.0
    d = np.abs(np.diff([n[2] for n in notes]))
    return float(np.mean(d >= 8))


def clean(notes):
    """Le ménage : hoquets recollés, aberrations jetées.

    Trois passes, dans cet ordre — chacune rend la suivante possible :

    1. deux notes de MÊME hauteur séparées par moins de MERGE_GAP sont une seule
       note (une consonne, une respiration, un mot recommencé) ;
    2. un hoquet — une note courte encadrée par deux fois la même hauteur, du
       genre La-Si-La en 90 ms — est retiré, et les deux La fusionnent. C'est
       la « salade de notes » du mélisme : la voix passe par une hauteur sans
       jamais l'avoir chantée ;
    3. un PORTAMENTO : une note courte qui tombe pile entre ses deux voisines,
       dans le sens du saut (La → Si → Do♯ pour un La→Do♯), n'est pas chantée,
       c'est la voix qui glisse. On la retire et on rallonge la note d'avant ;
    4. ce qui reste à une octave entière de ses voisines est jeté. Après la
       correction d'octave, ce n'est plus une octave mal résolue, c'est du
       souffle, une consonne voisée ou une deuxième voix qui traverse le stem.

    Retourne (notes, comptes) — tous les comptes sont sur la page, parce qu'un
    grand nombre de hoquets veut dire une extraction fragile même réparée.
    """
    if not notes:
        return notes, {"merged": 0, "blips": 0, "gliss": 0, "dropped": 0}
    out, merged = [], 0
    for nt in sorted(notes):
        if out and out[-1][2] == nt[2] and nt[0] - (out[-1][0] + out[-1][1]) <= MERGE_GAP:
            out[-1][1] = max(out[-1][1], nt[0] + nt[1] - out[-1][0])
            merged += 1
        else:
            out.append(list(nt))
    blips = 0
    i = 1
    while i < len(out) - 1:
        if out[i][1] < BLIP and out[i - 1][2] == out[i + 1][2] and \
                out[i + 1][0] - (out[i - 1][0] + out[i - 1][1]) <= 2 * BLIP:
            out[i - 1][1] = out[i + 1][0] + out[i + 1][1] - out[i - 1][0]
            del out[i:i + 2]
            blips += 1
            i = max(1, i - 1)
        else:
            i += 1
    gl, i = 0, 1
    while i < len(out) - 1:
        a_, b_, c_ = out[i - 1], out[i], out[i + 1]
        between = (a_[2] - b_[2]) * (b_[2] - c_[2]) > 0
        if b_[1] < GLISS and between and abs(c_[2] - a_[2]) >= 2 \
                and b_[0] - (a_[0] + a_[1]) <= 0.10 \
                and c_[0] - (b_[0] + b_[1]) <= 0.10:
            a_[1] = max(a_[1], b_[0] + b_[1] * .5 - a_[0])
            del out[i]
            gl += 1
        else:
            i += 1
    a = np.array(out, float)
    keep = np.ones(len(a), bool)
    for i in range(len(a)):
        m = (np.abs(a[:, 0] - a[i, 0]) <= OCT_WIN)
        m[i] = False
        if m.sum() < 4:
            continue
        if abs(a[i, 2] - float(np.median(a[m, 2]))) >= OUT_FAR:
            keep[i] = False
    return ([o for o, k in zip(out, keep) if k],
            {"merged": merged, "blips": blips, "gliss": gl,
             "dropped": int((~keep).sum())})


# ── « dans la gamme » : le seul contrôle objectif qu'on ait ici ─────────────
MAJ = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
MIN = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1])   # mineur, 7e naturelle ET
                                                       # sensible : en pop comme
                                                       # en jazz les deux servent
PCN = PC
# Profils de Krumhansl–Kessler. Un simple masque « sept notes sur douze » ne
# sait pas distinguer Do majeur de Fa majeur quand la mélodie évite la sensible
# — et c'est le cas de Let It Be. Les profils, eux, pèsent la tonique et la
# dominante : vérifié, ils retrouvent Do majeur pour Let It Be, Ré mineur pour
# Grenade, Do mineur pour This Love, Si♭ majeur pour Don't Know Why.
KK_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def in_scale(notes):
    """(part du TEMPS chanté dans la gamme de la tonalité, nom de la tonalité).

    Sans partition on ne peut pas dire « cette note-là est fausse ». Mais une
    mélodie de chanson tient à ~95 % dans sa gamme : quand l'extraction tombe à
    80 %, ce ne sont pas des altérations, ce sont des transitions et des octaves
    qu'on a prises pour des notes. C'est un plancher, pas une preuve, et la
    tonalité est estimée sur la MÉLODIE SEULE — donc à contester aussi.
    """
    w = np.zeros(12)
    for _, d, m in notes:
        w[int(m) % 12] += d
    if w.sum() <= 0:
        return 0.0, "—"
    w = w / w.sum()
    best = (-9.0, 0, "maj")
    for r in range(12):
        for name, prof in (("maj", KK_MAJ), ("min", KK_MIN)):
            c = float(np.corrcoef(np.roll(prof, r), w)[0, 1])
            if c > best[0]:
                best = (c, r, name)
    _, r, name = best
    sc = np.roll(MAJ if name == "maj" else MIN, r)
    return (float((w * sc).sum()),
            f"{PCN[r]} {'majeur' if name == 'maj' else 'mineur'}")


# ── 2. la grille : mesures -> temps -> demi-temps ───────────────────────────
def beat_grid(grid, sub=SUB):
    """Les mesures du pipeline, subdivisées en temps puis en demi-temps."""
    g = [float(x) for x in grid]
    out = []
    for a, b in zip(g[:-1], g[1:]):
        k = BEATS_PER_BAR * sub
        out += [a + (b - a) * j / k for j in range(k)]
    return np.array(out + [g[-1]])


def quantise(notes, sgrid):
    """Chaque début et chaque fin au demi-temps le plus proche (≥ 1 demi-temps)."""
    if not notes or len(sgrid) < 3:
        return [list(x) for x in notes], 0.0
    snap = lambda x: int(np.argmin(np.abs(sgrid - x)))
    out, shifts = [], []
    for t0, dur, m in notes:
        i = snap(t0)
        j = max(i + 1, snap(t0 + dur))
        j = min(j, len(sgrid) - 1)
        if j <= i:
            continue
        shifts.append(abs(sgrid[i] - t0))
        out.append([float(sgrid[i]), float(sgrid[j] - sgrid[i]), int(m)])
    # deux notes identiques collées après calage : on les fond
    merged = []
    for nt in out:
        if merged and merged[-1][2] == nt[2] and \
                abs(merged[-1][0] + merged[-1][1] - nt[0]) < 1e-6:
            merged[-1][1] += nt[1]
        else:
            merged.append(nt)
    return merged, float(np.median(shifts)) if shifts else 0.0


# ── 3. la vue d'ensemble (piano-roll sur l'axe des mesures) ─────────────────
def overview(notes, grid, stem):
    n = len(grid) - 1
    g = np.asarray(grid, float)
    b = lambda x: float(np.interp(x, g, np.arange(len(g))))
    lo = min(nt[2] for nt in notes) if notes else 60
    hi = max(nt[2] for nt in notes) if notes else 72
    lo, hi = lo - 1, hi + 1
    fig, ax = plt.subplots(figsize=(12.6, 2.15))
    H = 2.15
    fig.subplots_adjust(left=PLOT_L, right=PLOT_R, top=1 - .12 / H, bottom=.52 / H)
    for m in range(int(lo), int(hi) + 1):
        if m % 12 in BLACK:
            ax.axhspan(m - .5, m + .5, color="#efe8d6", lw=0, zorder=0)
        if m % 12 == 0:
            ax.axhline(m - .5, color="#d8cfb4", lw=.6, zorder=1)
    for x in range(0, n + 1, 4):
        ax.axvline(x, color="#e0d8c2", lw=.6, zorder=1)
    for t0, dur, m in notes:
        x0, x1 = b(t0), b(t0 + dur)
        ax.add_patch(plt.Rectangle((x0, m - .40), max(x1 - x0, .04), .80,
                                   facecolor=ACC, edgecolor="none", zorder=3))
    ax.set_xlim(0, n)
    ax.set_ylim(lo - .5, hi + .5)
    yt = [m for m in range(int(lo), int(hi) + 1) if m % 12 == 0]
    ax.set_yticks(yt)
    ax.set_yticklabels([note_name(m) for m in yt], fontsize=6.2)
    ax.set_xticks(range(0, n + 1, 4))
    ax.tick_params(labelsize=6.4)
    ax.set_xlabel("mesure", fontsize=8)
    ax.set_ylabel("mélodie", fontsize=7, color=ACC)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    return fig2b64_fixed(fig)


# ── 4. une chanson ─────────────────────────────────────────────────────────
def song(stem):
    S, n, grid = load(stem)
    voc = separate_vocals(HERE / f"docs/audio/{stem}.m4a")
    t, f0, voiced, rms = track_f0(voc)
    notes, diag = melody_notes(t, f0, voiced, rms)

    # La correction d'octave est PROPOSÉE, puis jugée : on ne la garde que si
    # elle fait baisser le taux de grands sauts. Sur Norah Jones elle le fait
    # MONTER (3,6 % -> 6,6 %) — ses sauts d'octave sont vrais, et la correction
    # les rabotait. Un réglage qui aide six morceaux sur sept n'a pas le droit
    # d'abîmer le septième en silence : la page dit ce qui a été décidé.
    fixed, n_prop = fix_octaves(notes)
    before, after = leap_rate(notes), leap_rate(fixed)
    oct_used = after < before
    n_oct = n_prop if oct_used else 0
    if oct_used:
        notes = fixed
    notes, cl = clean(notes)
    sgrid = beat_grid(grid)
    qnotes, shift = quantise(notes, sgrid)

    sc_f, sc_n = in_scale(notes)
    sing = first_sing(voc)
    dur_song = float(grid[-1] - grid[0])
    tot = sum(d for _, d, _ in notes) or 1e-9
    early = (sum(min(d, max(0.0, sing - t0)) for t0, d, _ in notes) / tot
             if sing else 0.0)
    covered = sum(d for _, d, _ in notes)
    durs = np.array([d for _, d, _ in notes]) if notes else np.array([0.0])
    pitches = [m for _, _, m in notes] or [60]
    beat = dur_song / max(1, (len(grid) - 1) * BEATS_PER_BAR)
    st = {
        "n_notes": len(notes), "n_bars": n,
        "median_dur": float(np.median(durs)),
        "short": float(np.mean(durs < 0.15)),
        "cover": covered / max(dur_song, 1e-6),
        "lo": min(pitches), "hi": max(pitches),
        # l'ambitus UTILE : deux notes égarées ne définissent pas une tessiture
        "p2": int(round(np.percentile(pitches, 2))),
        "p98": int(round(np.percentile(pitches, 98))),
        "ambitus": max(pitches) - min(pitches),
        "oct": n_oct, "tuning": diag["tuning"] * 100,
        "merged": cl["merged"], "blips": cl["blips"], "gliss": cl["gliss"],
        "dropped": cl["dropped"], "scale": sc_f, "scale_name": sc_n,
        "oct_used": oct_used, "oct_prop": n_prop,
        "leap_before": before, "leap_after": after, "leap": leap_rate(notes),
        "sing": sing, "early": early,
        "shift_ms": shift * 1000, "shift_beat": shift / max(beat, 1e-6),
        "notes_per_bar": len(notes) / max(n, 1),
    }
    img = overview(notes, grid, stem)
    data = {"stem": stem, "audio": f"/audio/{stem}.m4a",
            "grid": [round(float(x), 3) for x in grid],
            "raw": [[round(a, 3), round(b, 3), c] for a, b, c in notes],
            "q": [[round(a, 3), round(b, 3), c] for a, b, c in qnotes],
            "lo": st["lo"], "hi": st["hi"]}
    return img, st, data


# ── 5. la page ─────────────────────────────────────────────────────────────
def fr(s):
    """Chiffres à la française : 3.9 % -> 3,9 %. Louis lit du français."""
    import re as _re
    return _re.sub(r"(\d)\.(\d)", r"\1,\2", s)


def plur(n, word):
    return f"{n} {word}{'s' if n > 1 else ''}"


def section_html(stem, img, st, data, verdict):
    title = NAMES.get(stem, stem.replace("_", " ").title())
    rows = [
        ("notes trouvées", f"{st['n_notes']} · {st['notes_per_bar']:.1f} par mesure"),
        ("durée médiane", f"{st['median_dur']*1000:.0f} ms · "
                          f"{st['short']:.0%} sous 150 ms"),
        ("le chant occupe", f"{st['cover']:.0%} du morceau"),
        ("avant le premier chant",
         (f"{st['early']:.1%} de la mélodie est posée avant {st['sing']:.1f} s, "
          f"où le détecteur de chant dit que ça commence"
          if st["sing"] else "premier chant non détecté")),
        ("tessiture", f"{note_name(st['p2'])} → {note_name(st['p98'])} "
                      f"(98 % du temps) · extrêmes {note_name(st['lo'])}–"
                      f"{note_name(st['hi'])}"),
        ("octaves corrigées",
         (f"{st['oct']} notes ({st['oct']/max(1, st['n_notes']):.0%}) — "
          f"grands sauts {st['leap_before']:.1%} → {st['leap_after']:.1%}"
          if st["oct_used"] else
          f"<b>correction refusée</b> — les {st['oct_prop']} propositions "
          f"faisaient MONTER les grands sauts ({st['leap_before']:.1%} → "
          f"{st['leap_after']:.1%}) : ses sauts d'octave sont vrais")),
        ("grands sauts restants", f"{st['leap']:.1%} des enchaînements ≥ 8 demi-tons"),
        ("ménage", f"{plur(st['merged'], 'fusion')} · "
                   f"{plur(st['blips'], 'hoquet')} · "
                   f"{plur(st['gliss'], 'portamento')} · "
                   f"{st['dropped']} jetée{'s' if st['dropped'] > 1 else ''}"),
        ("dans la gamme", f"<b>{st['scale']:.0%}</b> du temps chanté "
                          f"(tonalité estimée sur la mélodie seule : "
                          f"{st['scale_name']})"),
        ("désaccord du disque", f"{st['tuning']:+.0f} centièmes (retiré avant l'arrondi)"),
        ("décalage au calage", f"{st['shift_ms']:.0f} ms médian = "
                               f"{st['shift_beat']:.2f} temps"),
    ]
    tbl = "".join(f"<tr><td>{a}</td><td>{fr(b)}</td></tr>" for a, b in rows)
    return f"""<section data-song='{json.dumps(data)}'>
<h2>{title}<span class=sub>{st['n_notes']} notes · {st['n_bars']} mesures</span></h2>
<div class=plot><img src="data:image/png;base64,{img}">
<div class=cur></div><div class=hit></div></div>
<div class=bar>
  <button class="tr t-orig">▶ original</button>
  <button class="tr t-piano">▶ piano seul</button>
  <button class="tr t-both">▶ les deux</button>
  <button class="tr t-stop">■</button>
  <span class=pos>mes. 1 · 0:00</span>
  <button class=qt data-q=0>brut</button>
  <span class=hint>touche le piano-roll pour te déplacer</span>
</div>
<canvas class=fall></canvas>
<p class=cap>Les notes tombent et touchent leur touche <b>pile quand elles
sonnent</b> ; les traits horizontaux sont les barres de mesure, numérotées.
La touche s'allume tant que la note dure.</p>
<table>{tbl}</table>
<p class=verdict>{verdict}</p></section>"""


LEDE = """<div class=lede>On sépare la voix (demucs), on suit sa <b>hauteur</b>
(pyin, une mesure toutes les 11 ms), et on transforme cette courbe en
<b>notes</b> : on corrige le désaccord du disque, on arrondit au demi-ton, on
lisse, et une note est un palier de l'escalier obtenu. Un trou de moins de 80 ms
ne coupe pas une note — les consonnes coupent la voix, pas la note ; un palier de
moins de 100 ms est jeté.<br><br>
<b>Trois boutons pour juger.</b> « original » = le disque. « piano seul » = la
mélodie extraite, jouée par une synthèse maison dans le navigateur (aucun fichier
de son, la page est autonome). <b>« les deux ensemble » est le vrai test</b> : si
l'extraction est bonne, le piano se colle au chant et disparaît dedans ; sinon
ça flotte, et ça s'entend en deux secondes.<br><br>
<b>Brut ou calé.</b> Le bouton <b>brut</b> / <b>calé</b> bascule entre les notes
telles qu'elles ont été mesurées et les mêmes notes posées sur les demi-temps de
la grille de mesures du pipeline (downbeats Beat This!). <b>C'est le brut qui est
affiché par défaut</b> : le calé est plus lisible en partition mais hérite des
erreurs de la grille, alors que le brut ne peut mentir que sur lui-même. Le
tableau donne de combien le calage déplace les notes — au-delà d'un quart de
temps, c'est la grille et la voix qui ne sont pas d'accord.<br><br>
Le <b>piano-roll</b> sert aussi de barre de lecture : touche-le pour te
déplacer. En dessous, les notes tombent sur le clavier, et arrivent dessus
<b>exactement</b> quand elles sonnent.</div>"""


def page(body, notes_html):
    return f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Tutoriel de mélodie</title><style>
*{{box-sizing:border-box;-webkit-tap-highlight-color:transparent}}
body{{margin:0;background:#e7e0d0;font:16px/1.55 -apple-system,system-ui,sans-serif;color:{INK}}}
.wrap{{max-width:1150px;margin:0 auto;padding:20px 12px calc(40px + env(safe-area-inset-bottom))}}
h1{{font:italic 600 24px Georgia,serif;margin:0 0 3px}}
.lede{{color:#6f6858;font-size:13.5px;line-height:1.55;margin-bottom:18px}}
.lede b{{color:{INK}}}
section{{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;padding:14px 13px;margin-bottom:16px}}
h2{{font:700 18px system-ui;margin:0 0 6px;color:{ACC}}}
.sub{{font:500 12px system-ui;color:#8a8371;margin-left:8px}}
img{{width:100%;border-radius:8px;display:block}}
table{{border-collapse:collapse;font-size:12.5px;width:100%;margin-top:10px}}
td{{border:1px solid #e5dcc6;padding:3px 8px;text-align:left}}
td:first-child{{color:#8a8371;width:38%;font-size:11.5px}}
.plot{{position:relative;margin-bottom:8px}} .plot img{{margin:0}}
.cur{{position:absolute;top:0;bottom:0;width:2px;background:#111;opacity:.8;
  display:none;pointer-events:none;box-shadow:0 0 0 1px rgba(255,255,255,.55)}}
.hit{{position:absolute;top:0;bottom:0;cursor:crosshair}}
.bar{{display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin:0 0 8px}}
button.tr{{border:1px solid #d8cfb4;background:#f7f3e9;border-radius:8px;
  padding:7px 11px;cursor:pointer;font:600 12.5px system-ui;color:#6f6858}}
button.tr.on{{background:{ACC};border-color:{ACC};color:#fff}}
button.qt{{border:1px solid #d8cfb4;background:#f7f3e9;border-radius:8px;
  padding:7px 11px;cursor:pointer;font:600 12.5px ui-monospace,monospace;color:#6f6858}}
button.qt.on{{background:#0f766e;border-color:#0f766e;color:#fff}}
.pos{{font:600 12px ui-monospace,monospace;min-width:96px}}
.hint{{font:500 11px system-ui;color:#a89f8c}}
canvas.fall{{width:100%;height:300px;display:block;border-radius:10px;
  background:#1b1a17;touch-action:manipulation}}
.cap{{font:500 11.5px system-ui;color:#a89f8c;margin:6px 0 0}}
.cap b{{color:#8a8371}}
.verdict{{font-size:13px;background:#f7f3e9;border-radius:8px;padding:9px 11px;margin:10px 0 0}}
.brk{{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;padding:14px 13px}}
.brk h2{{color:{ACC}}} .brk li{{font-size:13.5px;margin-bottom:7px}}
.brk b{{color:{ACC}}}
</style></head><body><div class=wrap>
<h1>Tutoriel de mélodie</h1>
{LEDE}
{body}
{notes_html}
</div>
<audio id=au preload=metadata playsinline></audio>
<script>
const au=document.getElementById("au");
const L0={PLOT_L}, W={round(PLOT_R - PLOT_L, 6)};
const PCB=[0,1,0,1,0,0,1,0,1,0,1,0];        // 1 = touche noire
// LOOK est déclaré ICI et pas à côté de `drawFall` : la boucle d'installation
// des sections appelle `drawFall` pour dessiner la première phrase à l'arrêt,
// donc avant que le corps du script ait fini de s'exécuter. Un `const` déclaré
// plus bas est encore dans sa zone morte à ce moment-là et lève
// « Cannot access 'LOOK' before initialization », ce qui tuait toute
// l'installation : aucun bouton câblé, aucun piano-roll dessiné.
const LOOK=2.6;                       // secondes visibles au-dessus du clavier
let AC=null, master=null, live=null, raf=null, base=null;

function ctx(){{
  if(!AC){{ AC=new (window.AudioContext||window.webkitAudioContext)();
    master=AC.createGain(); master.gain.value=0.22; master.connect(AC.destination); }}
  if(AC.state==="suspended") AC.resume();
  return AC;
}}
// Couper le son TOUT DE SUITE : les notes déjà programmées vivent dans le
// graphe audio et continueraient à sonner après un pause ou un saut. On jette
// le noeud de sortie, ce qui les rend muettes instantanément.
function kill(){{
  if(!master) return;
  try{{ master.disconnect(); }}catch(e){{}}
  master=AC.createGain(); master.gain.value=0.22; master.connect(AC.destination);
}}
// Un piano de synthèse : deux dents de scie légèrement désaccordées, un
// passe-bas qui se referme, une enveloppe percussive. Aucun fichier externe.
function pluck(midi,at,dur){{
  const c=ctx(), f=440*Math.pow(2,(midi-69)/12);
  const g=c.createGain(), lp=c.createBiquadFilter();
  lp.type="lowpass";
  lp.frequency.setValueAtTime(Math.min(7000,Math.max(900,f*6)),at);
  lp.frequency.exponentialRampToValueAtTime(Math.min(7000,Math.max(700,f*2.2)),at+0.35);
  lp.Q.value=0.6;
  const end=at+Math.max(0.12,dur);
  g.gain.setValueAtTime(0.0001,at);
  g.gain.exponentialRampToValueAtTime(0.9,at+0.008);
  g.gain.exponentialRampToValueAtTime(0.34,at+0.28);
  g.gain.setTargetAtTime(0.0001,end,0.055);
  [0,-6,7].forEach((cts,i)=>{{
    const o=c.createOscillator();
    o.type=i===2?"triangle":"sawtooth";
    o.frequency.value=f*Math.pow(2,cts/1200);
    const og=c.createGain(); og.gain.value=i===2?0.55:0.32;
    o.connect(og); og.connect(lp); o.start(at); o.stop(end+0.4);
  }});
  lp.connect(g); g.connect(master);
}}

function fmt(s){{return Math.floor(s/60)+":"+String(Math.floor(s%60)).padStart(2,"0");}}

document.querySelectorAll("section[data-song]").forEach(sec=>{{
  const D=JSON.parse(sec.dataset.song), G=D.grid, n=G.length-1;
  const cv=sec.querySelector("canvas.fall"), hit=sec.querySelector(".hit"),
        cur=sec.querySelector(".cur"), pos=sec.querySelector(".pos"),
        qt=sec.querySelector(".qt");
  const S={{sec:sec,D:D,G:G,n:n,cv:cv,cur:cur,pos:pos,q:0,mode:null,fired:0}};
  sec._s=S;
  S.notes=()=>S.q?D.q:D.raw;
  hit.style.left=(L0*100)+"%"; hit.style.width=(W*100)+"%";

  const t2b=t=>{{ if(t<=G[0])return 0; if(t>=G[n])return n;
    let lo=0,hi=n; while(hi-lo>1){{const m=(lo+hi)>>1; G[m]<=t?lo=m:hi=m;}}
    return lo+(t-G[lo])/(G[lo+1]-G[lo]); }};
  S.t2b=t2b;
  hit.onclick=e=>{{ const r=hit.getBoundingClientRect();
    const f=n*(e.clientX-r.left)/r.width;
    const i=Math.max(0,Math.min(n-1,Math.floor(f)));
    seek(S, G[i]+(f-i)*(G[i+1]-G[i])); }};
  qt.onclick=()=>{{ S.q^=1; qt.classList.toggle("on",!!S.q);
    qt.textContent=S.q?"calé":"brut"; S.fired=0;
    drawFall(S, live===S?au.currentTime:0); }};
  sec.querySelector(".t-orig").onclick =()=>start(S,"orig");
  sec.querySelector(".t-piano").onclick=()=>start(S,"piano");
  sec.querySelector(".t-both").onclick =()=>start(S,"both");
  sec.querySelector(".t-stop").onclick =()=>{{ au.pause(); }};
  // au repos : on montre la première phrase figée, 1,2 s avant qu'elle tombe.
  // Un canvas vide au chargement ne dit rien de ce qui va se passer.
  drawFall(S, D.raw.length ? D.raw[0][0]-1.2 : 0);
}});

function setBtns(S){{
  S.sec.querySelectorAll("button.tr").forEach(b=>b.classList.remove("on"));
  if(S.mode && live===S && !au.paused)
    S.sec.querySelector(".t-"+S.mode).classList.add("on");
}}
function start(S,mode){{
  ctx();
  if(live===S && S.mode===mode && !au.paused){{ au.pause(); return; }}   // re-clic = pause
  if(live && live!==S){{ live.cur.style.display="none"; setBtns(live); }}
  const t0 = (live===S) ? au.currentTime : 0;
  live=S; S.mode=mode; S.fired=0; base=null; kill();
  au.muted = (mode==="piano");
  if(au.getAttribute("src")!==S.D.audio){{ au.setAttribute("src",S.D.audio); au.load(); }}
  const go=()=>{{ try{{ au.currentTime=t0; }}catch(e){{}}
                  au.play().catch(()=>{{}}); }};
  if(au.readyState>=1) go(); else au.addEventListener("loadedmetadata",go,{{once:true}});
  setBtns(S);
}}
function seek(S,t){{
  ctx();
  if(live!==S){{ if(live) live.cur.style.display="none"; live=S;
    if(au.getAttribute("src")!==S.D.audio){{ au.setAttribute("src",S.D.audio); au.load(); }}
    S.mode=S.mode||"both"; au.muted=(S.mode==="piano"); }}
  const go=()=>{{ try{{ au.currentTime=t; }}catch(e){{}}
    S.fired=0; base=null; kill(); drawAll(); }};
  if(au.readyState>=1) go(); else au.addEventListener("loadedmetadata",go,{{once:true}});
  if(au.paused) au.play().catch(()=>{{}});
  setBtns(S);
}}

au.addEventListener("play",()=>{{ ctx(); base=null; if(live) setBtns(live); loop(); }});
au.addEventListener("pause",()=>{{ if(live) setBtns(live);
  cancelAnimationFrame(raf); kill(); drawAll(); }});
au.addEventListener("seeking",()=>{{ if(live){{ live.fired=0; base=null; kill(); }} }});

// ── l'ordonnanceur ────────────────────────────────────────────────────────
// L'horloge, c'est `au.currentTime` : un saut dans la barre de lecture ne peut
// donc pas désynchroniser le piano. Mais cette horloge-là avance par paliers
// sur iOS, et lire l'écart à chaque image ferait trembler les attaques de
// quelques dizaines de ms. On VERROUILLE donc le décalage entre l'horloge audio
// et l'horloge WebAudio, et on ne le refait qu'en cas de vraie dérive (> 60 ms).
const AHEAD=0.25, DRIFT=0.06;
function schedule(){{
  if(!live || au.paused || live.mode==="orig") return;
  const now=au.currentTime;
  if(base===null || Math.abs((AC.currentTime-now)-base) > DRIFT) base=AC.currentTime-now;
  const N=live.notes();
  while(live.fired<N.length && N[live.fired][0] < now-0.05) live.fired++;
  while(live.fired<N.length && N[live.fired][0] < now+AHEAD){{
    const nt=N[live.fired++];
    pluck(nt[2], Math.max(AC.currentTime+0.005, base+nt[0]), nt[1]);
  }}
}}

function loop(){{ drawAll(); schedule();
  if(!au.paused) raf=requestAnimationFrame(loop); }}

function drawAll(){{
  if(!live) return;
  const t=au.currentTime, f=live.t2b(t);
  live.cur.style.display="block";
  live.cur.style.left="calc("+((L0+W*f/live.n)*100)+"% - 1px)";
  live.pos.textContent="mes. "+(Math.floor(f)+1)+" · "+fmt(t);
  drawFall(live,t);
}}

// ── le clavier et les notes qui tombent ────────────────────────────────────
function drawFall(S,now){{
  const cv=S.cv, dpr=window.devicePixelRatio||1;
  const w=cv.clientWidth, h=cv.clientHeight;
  if(cv.width!==Math.round(w*dpr)||cv.height!==Math.round(h*dpr)){{
    cv.width=Math.round(w*dpr); cv.height=Math.round(h*dpr); }}
  const g=cv.getContext("2d");
  g.setTransform(dpr,0,0,dpr,0,0);
  g.clearRect(0,0,w,h);
  g.fillStyle="#1b1a17"; g.fillRect(0,0,w,h);

  // L'étendue du clavier : juste la voix, arrondie à une touche BLANCHE de
  // chaque côté. Arrondir à l'octave entière donnerait cinq touches mortes à
  // gauche sur un téléphone — sur Grenade, 21 blanches au lieu de 13.
  let lo=S.D.lo-1, hi=S.D.hi+1;
  while(PCB[lo%12]) lo--;
  while(PCB[hi%12]) hi++;
  const whites=[]; for(let m=lo;m<=hi;m++) if(!PCB[m%12]) whites.push(m);
  const kw=w/whites.length, KH=Math.min(62,h*0.22), top=h-KH;
  const px=top/LOOK;
  const xOf=m=>{{ if(!PCB[m%12]) return {{x:whites.indexOf(m)*kw, w:kw}};
    const l=whites.filter(v=>v<m).length;   // noire : à cheval sur deux blanches
    return {{x:l*kw-kw*0.30, w:kw*0.60}}; }};

  // les barres de mesure qui tombent avec les notes
  g.font="9px ui-monospace,monospace";
  for(let i=0;i<S.G.length;i++){{
    const y=top-(S.G[i]-now)*px;
    if(y<-20||y>top+2) continue;
    g.strokeStyle="rgba(255,255,255,.16)"; g.lineWidth=1;
    g.beginPath(); g.moveTo(0,y); g.lineTo(w,y); g.stroke();
    g.fillStyle="rgba(255,255,255,.34)"; g.fillText(i+1,3,y-3);
  }}

  // les notes
  const N=S.notes(), on={{}};
  for(let i=0;i<N.length;i++){{
    const t0=N[i][0], d=N[i][1], m=N[i][2];
    if(t0-now>LOOK+.2) break;
    if(t0+d<now-0.35) continue;
    const y1=top-(t0-now)*px, y0=top-(t0+d-now)*px;
    const p=xOf(m), black=!!PCB[m%12];
    const hitNow=(now>=t0&&now<t0+d);
    if(hitNow) on[m]=1;
    const rr=Math.min(4,p.w*0.3);
    g.fillStyle= hitNow ? "#ffd9a0" : (black?"#b3452f":"#e07a3f");
    g.beginPath();
    if(g.roundRect) g.roundRect(p.x+1.2,y0,p.w-2.4,Math.max(3,y1-y0),rr);
    else g.rect(p.x+1.2,y0,p.w-2.4,Math.max(3,y1-y0));
    g.fill();
  }}

  // le clavier
  for(const m of whites){{
    const p=xOf(m);
    g.fillStyle=on[m]?"#ffb861":"#f4efe3";
    g.fillRect(p.x,top,p.w-1,KH);
    if(m%12===0){{ g.fillStyle="#8a8371"; g.font="8px ui-monospace,monospace";
      g.fillText("C"+(m/12-1),p.x+2,h-4); }}
  }}
  for(let m=lo;m<=hi;m++) if(PCB[m%12]){{
    const p=xOf(m);
    g.fillStyle=on[m]?"#e08a2a":"#2a2825";
    g.fillRect(p.x,top,p.w,KH*0.62);
  }}
  g.strokeStyle="rgba(255,255,255,.35)"; g.lineWidth=1;
  g.beginPath(); g.moveTo(0,top+.5); g.lineTo(w,top+.5); g.stroke();
}}
window.addEventListener("resize",()=>{{ if(live) drawFall(live,au.currentTime);
  document.querySelectorAll("section[data-song]").forEach(s=>{{
    if(s._s!==live) drawFall(s._s,0); }}); }});
</script></body></html>"""


# ── ce qui casse : écrit après avoir regardé les extractions, pas avant ────
# Chaque verdict est adossé à un chiffre du tableau de la même carte. Rien ici
# n'a été écrit avant d'avoir vu le piano-roll et les compteurs.
VERDICTS = {
 "maroon_5_she_will_be_loved_official_music_video":
   "<b>Le meilleur du lot.</b> 97 % du temps chanté dans la gamme, et sur le "
   "piano-roll le refrain redessine exactement le même contour aux mesures 28, "
   "56 et 88 — une extraction bruitée ne se répète pas comme ça. Réserve : "
   "sept notes, 3,4 % du temps, sont plus d'une octave au-dessus du reste "
   "(les falsettos) ; c'est le seul endroit à vérifier à l'oreille.",
 "bruno_mars_grenade_official_music_video":
   "<b>Très bon.</b> 1,1 % de grands sauts, le plus bas des sept : la mélodie "
   "avance par pas de un ou deux demi-tons, comme une vraie ligne de chant. "
   "Grenade est très syllabique — une syllabe, une note — et c'est exactement "
   "le cas facile pour cette méthode.",
 "norah_jones_don_t_know_why":
   "<b>Bon, et instructif.</b> C'est le seul morceau où la correction d'octave "
   "a été <b>refusée</b> : ses grands sauts sont vrais, et les « corriger » les "
   "faisait passer de 3,9 % à 6,1 %. Mélodie aérée (39 % du morceau), ce qui "
   "est juste pour une ballade lente — le reste, c'est le piano de Norah.",
 "let_it_be_remastered_2009":
   "<b>Bon.</b> La ligne de couplet est là, et la tonalité estimée sur la seule "
   "mélodie extraite tombe sur Do majeur, ce qui est la bonne réponse. C'est "
   "aussi le morceau où le calage sur la grille déplace le plus (0,14 temps, "
   "114 ms) : les Beatles ne jouent pas sur un clic, la grille et la voix "
   "négocient.",
 "bein_green":
   "<b>Correct, avec une réserve de timbre.</b> Les phrases sont nettes sur le "
   "piano-roll, mais c'est la voix de Kermit : timbre serré, nasal, sur un "
   "enregistrement ancien. La tessiture utile (La2–Si♭3) est basse et deux "
   "notes descendent jusqu'à Mi2, ce qui n'existe pas — un résidu d'octave que "
   "la correction n'a pas attrapé.",
 "maroon_5_this_love":
   "<b>Moyen.</b> 33 octaves corrigées (11 % des notes) et 88,8 % dans la "
   "gamme, le deuxième plus mauvais. Le chant est rapide et serré, les "
   "syllabes s'enchaînent plus vite que le découpage ne sait suivre ; sur le "
   "piano-roll ça se voit tout de suite — des traits courts partout, peu de "
   "notes tenues, et la phrase du refrain ne se redessine pas franchement.",
 "mayer_hawthorne_the_walk":
   "<b>Le cas raté — gardé exprès.</b> 98 octaves corrigées sur 335 notes "
   "(29 %), et il reste malgré tout 12 % de grands sauts, trois fois plus que "
   "partout ailleurs. Mayer Hawthorne alterne voix de poitrine et fausset sur "
   "des voix superposées : pyin choisit tantôt la fondamentale, tantôt "
   "l'harmonique, et la médiane locale n'arrive plus à trancher. Écoute « les "
   "deux ensemble » : c'est ici que ça décroche.",
}

BREAKS = """<div class=brk><h2>Ce qui casse</h2>
<ol>
<li><b>L'octave, de loin le pire défaut.</b> Sur <i>The Walk</i>, 29 % des notes
ont dû être décalées d'une octave, et il reste ensuite 12 % de grands sauts
contre 1 à 5 % ailleurs. La cause est connue : pyin suit une harmonique au lieu
de la fondamentale, et il le fait surtout sur les voix superposées et les
falsettos. Réparé en partie, jamais résolu.</li>
<li><b>La réparation elle-même peut se tromper.</b> Sur <i>Don't Know Why</i>
elle faisait <i>monter</i> les grands sauts de 3,9 % à 6,1 % : Norah Jones fait
de vrais sauts d'octave, et le correcteur les rabotait. C'est pour ça qu'elle
est maintenant jugée avant d'être appliquée, morceau par morceau — mais cela
veut dire qu'un morceau qui a <i>à la fois</i> de vraies octaves et de fausses
octaves sera mal servi dans les deux sens.</li>
<li><b>Le mélisme fait de la salade.</b> Avant nettoyage, 27 à 30 % des notes
duraient moins de 150 ms. On en recolle une partie (hoquets, portamentos) et on
jette le reste sous 130 ms, ce qui ramène à 4–7 %. Mais une syllabe réellement
chantée sur cinq notes reste cinq notes : sans partition, rien ne distingue un
vrai mélisme d'un artefact.</li>
<li><b>La durée des notes est celle de la VOIX, pas celle d'un piano.</b> Un
chanteur coupe sur les consonnes et respire ; un pianiste tient. Les notes
rendues sont donc plus courtes et plus hachées qu'un tutoriel ne le voudrait —
c'est audible en « piano seul ».</li>
<li><b>On ne distingue pas « instrumental » de « extraction en panne ».</b> La
mélodie couvre 38 à 69 % du morceau. Le reste, ce sont des passages sans chant
<i>et</i> des passages où l'extraction a renoncé, et rien ici ne dit lequel est
lequel.</li>
<li><b>La tonalité affichée est estimée sur la mélodie extraite seule</b>
(profils de Krumhansl–Kessler), pas sur les accords du pipeline. Elle tombe
juste sur les quatre morceaux dont je connais la réponse (Do majeur pour
<i>Let It Be</i>, Ré mineur pour <i>Grenade</i>, Do mineur pour <i>This
Love</i>, Si♭ majeur pour <i>Don't Know Why</i>) — ce qui est un contrôle, pas
une garantie.</li>
<li><b>Ce que je croyais casser et qui ne casse pas :</b> la fuite de l'intro.
Je m'attendais à voir des nappes et des guitares prises pour du chant avant
l'entrée de la voix. Mesuré sur les sept morceaux, avec le détecteur de chant de
<code>blocks8</code> comme témoin indépendant : <b>moins de 1 % du temps de
mélodie est posé avant le premier chant</b>, et 0,0 % sur quatre d'entre eux.
demucs fait mieux que ce que je supposais.</li>
</ol></div>"""


def main():
    stems = [a for a in sys.argv[1:] if not a.startswith("--")] or DEFAULT
    body, table = "", []
    for st in stems:
        if not (HERE / f"docs/audio/{st}.m4a").exists():
            print(f"  !! {st} introuvable")
            continue
        try:
            img, stats, data = song(st)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"  !! {st} — {type(exc).__name__}: {exc}")
            continue
        body += section_html(st, img, stats, data,
                             VERDICTS.get(st, "— pas encore jugé."))
        table.append((st, stats))
        print(f"  ok {st}: {stats['n_notes']} notes, {stats['oct']} octaves "
              f"corrigées, {stats['cover']:.0%} couvert, "
              f"calage {stats['shift_ms']:.0f} ms")
    out = HERE / "harmonia_min/state/reports/vocal_melody.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page(body, BREAKS))
    print(f"\nwrote {out.relative_to(HERE)} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
