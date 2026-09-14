"""L'intro contre le reste de la chanson, cherchée DANS LA VOIX.

    python scripts/intro_vocal.py           ->  /reports/intro_vocal.html

Louis, 2026-08-07 :

  « Je pense que les indices sont dans le vocal. Parce que des fois, entre
    l'intro et le début de la chanson, c'est la même harmonie, mais c'est le
    vocal qui change. Donc cherche plein, plein, plein de critères sur le vocal
    — les différences de timbre, les différences de je ne sais pas quoi. Tu
    regardes vraiment la matrice SSM vocale et tu te sers de ça pour essayer de
    détecter l'intro versus le reste de la chanson. **Et des fois il n'y a pas
    d'intro, prends bien ça en compte.** »

CE QUE CE SCRIPT MESURE. Pour chaque mesure du morceau, une trentaine de
descripteurs pris sur la PISTE VOCALE seule (demucs) : timbre, énergie, hauteur,
doublage, et la matrice de similarité de la voix. Puis, pour chacun tout seul,
combien de morceaux sur dix il place la frontière intro/chanson au bon endroit.

LES DEUX QUESTIONS SONT SÉPARÉES, comme il l'a demandé :

  PRÉSENCE  y a-t-il une intro, oui ou non ?
  POSITION  si oui, à quelle mesure finit-elle ?

Un descripteur qui répond bien à la seconde et jamais à la première n'est pas
utilisable : il inventera une intro sur les morceaux qui n'en ont pas.

LE PLANCHER À BATTRE, et il est haut. La vérité de Louis sur les dix morceaux
vaut 1, 4, 1, 0, 4, 4, 8, 8, 4, 2 mesures. Donc **« toujours 4 mesures » fait
déjà 4/10** sans rien écouter, et la règle livrée (`voice_sections.sung_start` :
première mesure chantée, plus une si la note tombe après 40 % de la mesure) fait
9/10. Tout chiffre en dessous de 4/10 est moins bon que de ne pas regarder.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))

API = "http://127.0.0.1:7772/api/sections"
CACHE = HERE / "harmonia_min/state/intro_vocal"
SR = 22050
HOP = 512                 # 23 ms — assez fin pour une mesure de 2 s
KMAX = 16                 # une intro fait au plus seize mesures
WIN = 8                   # la fenêtre de comparaison avant/après, en mesures


# ── la vérité ───────────────────────────────────────────────────────────────

def truth(refresh=False):
    """Les annotations de Louis → {morceau: mesure où finit l'intro}, 0 = aucune.

    Recopiée sur le disque au passage. Le serveur :7772 est mono-fil et sans
    rechargeur ; il a déjà expiré en plein milieu d'un balayage, ce qui fait
    perdre le calcul et non la mesure. La copie n'est là que pour ça — elle est
    rafraîchie dès que le serveur répond.
    """
    snap = CACHE / "truth.json"
    try:
        with urllib.request.urlopen(API, timeout=20) as r:
            T = json.load(r)
        snap.parent.mkdir(parents=True, exist_ok=True)
        snap.write_text(json.dumps(T))
    except Exception:
        if refresh or not snap.exists():
            raise
        T = json.loads(snap.read_text())
    out = {}
    for k, v in T.items():
        secs = v.get("sections")
        if not secs:
            continue
        intro = [s for s in secs if s["label"] == "intro"]
        out[k] = (intro[0]["b1"] + 1) if intro else 0
    return out


# ── les descripteurs, mesure par mesure ─────────────────────────────────────

def _bar_agg(t, x, grid, n, how="mean"):
    """Agrège une courbe de trames sur les mesures."""
    out = np.zeros(n)
    idx = np.searchsorted(grid, t) - 1
    for b in range(n):
        v = x[idx == b]
        v = v[np.isfinite(v)]
        if v.size:
            out[b] = {"mean": np.mean, "std": np.std, "max": np.max,
                      "p90": lambda z: np.percentile(z, 90)}[how](v)
    return out


def _ssm_rest(V, n, gap=4, top=None):
    """« À quel point cette mesure ressemble au RESTE du morceau. »

    C'est l'idée centrale de Louis, écrite littéralement : on exclut le
    voisinage immédiat (`gap` mesures), sinon toute mesure ressemble d'abord à
    ses voisines et la courbe ne mesure plus la reprise mais la continuité.
    """
    nrm = np.clip(np.linalg.norm(V, axis=1, keepdims=True), 1e-9, None)
    U = V / nrm
    M = np.clip(U @ U.T, -1, 1)
    dead = (np.linalg.norm(V, axis=1) <= 1e-9)
    M[dead, :] = np.nan
    M[:, dead] = np.nan
    r = np.zeros(n)
    for b in range(n):
        c = np.array([M[b, j] for j in range(n) if abs(j - b) >= gap])
        c = c[np.isfinite(c)]
        if not c.size:
            continue
        if top:
            c = np.sort(c)[-top:]
        r[b] = float(np.mean(c))
    return r, M


def _foote(M, L=8):
    """La nouveauté en damier de Foote, TRONQUÉE AUX BORDS.

    Écrite comme dans l'article, elle exige `L` mesures de chaque côté, donc
    elle ne rend rien avant la mesure L. C'est rédhibitoire ici et ça m'a coûté
    une première mesure entièrement fausse : **neuf de nos dix frontières
    tombent à la mesure 8 ou avant**, donc un noyau de huit mesures est aveugle
    exactement là où se trouve la réponse. Le détecteur « trouvait » la mesure 8
    sur les dix morceaux — pas parce qu'il l'avait vue, parce que c'était la
    première qu'il avait le droit de regarder.

    On tronque donc le noyau au bord et on renormalise par le poids réellement
    utilisé, ce qui rend la courbe définie dès la mesure 1 au prix d'une
    variance plus grande à gauche.
    """
    n = M.shape[0]
    K = np.zeros((2 * L, 2 * L))
    K[:L, :L] = K[L:, L:] = 1.0
    K[:L, L:] = K[L:, :L] = -1.0
    g = np.exp(-((np.arange(2 * L) - L + .5) / (L / 1.5)) ** 2)
    K *= np.outer(g, g)
    out = np.zeros(n)
    A = np.where(np.isfinite(M), M, 0.0)
    for b in range(1, n):
        a0, a1 = max(0, b - L), min(n, b + L)
        k = K[a0 - b + L:a1 - b + L, a0 - b + L:a1 - b + L]
        w = float(np.abs(k).sum())
        if w > 0:
            out[b] = float(np.sum(A[a0:a1, a0:a1] * k)) / w
    return out


def _comb(t, f0, voiced, S, freqs, n_frames):
    """Part de l'énergie expliquée par UN seul peigne harmonique.

    Une voix seule est bien décrite par les harmoniques d'une seule fondamentale.
    Une voix doublée à l'octave, harmonisée à la tierce ou empilée en chœur ne
    l'est pas : il reste de l'énergie ailleurs. C'est le descripteur de DOUBLAGE
    que Louis demande, et il ne coûte qu'une STFT qu'on a déjà.
    """
    out = np.zeros(n_frames)
    tot = S.sum(0) + 1e-9
    for i in range(n_frames):
        if not (i < len(f0) and voiced[i] and f0[i] > 0):
            continue
        hit = np.zeros(len(freqs), bool)
        for k in range(1, 9):
            fk = f0[i] * k
            if fk > freqs[-1]:
                break
            hit |= np.abs(1200 * np.log2(np.clip(freqs, 1e-6, None) / fk)) < 60
        out[i] = float(S[hit, i].sum() / tot[i])
    return out


def features(stem, grid, n):
    """Tous les descripteurs vocaux, (n mesures) chacun. Mis en cache."""
    cache = CACHE / f"{stem}.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        return {k: d[k] for k in d.files}

    import librosa
    import vocal_anchor as VA
    import vocal_melody as VM
    import melody_ssm as MS

    voc = VA.separate_vocals(HERE / f"docs/audio/{stem}.m4a")
    y, _ = librosa.load(str(voc), sr=SR, mono=True)
    ys, _ = librosa.load(str(voc), sr=SR, mono=False)
    if ys.ndim == 1:
        ys = np.stack([ys, ys])

    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=2048)
    tf = librosa.frames_to_time(np.arange(S.shape[1]), sr=SR, hop_length=HOP)
    F = {}

    # — timbre —
    mf = librosa.feature.mfcc(S=librosa.power_to_db(S ** 2), n_mfcc=20)
    dm = librosa.feature.delta(mf)
    F["_mfcc"] = mf                                     # gardé pour la SSM
    for j in (1, 2, 3, 4):
        F[f"mfcc{j}"] = _bar_agg(tf, mf[j], grid, n)
    F["mfcc_delta"] = _bar_agg(tf, np.linalg.norm(dm[1:], axis=0), grid, n)
    F["centroid"] = _bar_agg(tf, librosa.feature.spectral_centroid(S=S)[0], grid, n)
    F["bandwidth"] = _bar_agg(tf, librosa.feature.spectral_bandwidth(S=S)[0], grid, n)
    F["flatness"] = _bar_agg(tf, librosa.feature.spectral_flatness(S=S)[0], grid, n)
    F["rolloff85"] = _bar_agg(tf, librosa.feature.spectral_rolloff(S=S, roll_percent=.85)[0], grid, n)
    ct = librosa.feature.spectral_contrast(S=S, sr=SR)
    F["contrast"] = _bar_agg(tf, ct.mean(0), grid, n)
    F["contrast_hi"] = _bar_agg(tf, ct[-1], grid, n)
    F["zcr"] = _bar_agg(librosa.frames_to_time(np.arange(len(y) // HOP + 1), sr=SR, hop_length=HOP),
                        librosa.feature.zero_crossing_rate(y, hop_length=HOP)[0][:len(y) // HOP + 1],
                        grid, n)

    # — énergie —
    rms = librosa.feature.rms(S=S)[0]
    F["rms"] = _bar_agg(tf, rms, grid, n)
    F["rms_p90"] = _bar_agg(tf, rms, grid, n, "p90")
    F["rms_std"] = _bar_agg(tf, rms, grid, n, "std")
    F["rms_dyn"] = F["rms_std"] / np.clip(F["rms"], 1e-9, None)

    # — doublage : la largeur stéréo et le peigne harmonique —
    mid = (ys[0] + ys[1]) / 2
    side = (ys[0] - ys[1]) / 2
    rm = librosa.feature.rms(y=mid, hop_length=HOP)[0]
    rs = librosa.feature.rms(y=side, hop_length=HOP)[0]
    k = min(len(rm), len(rs))
    tw = librosa.frames_to_time(np.arange(k), sr=SR, hop_length=HOP)
    F["stereo_side"] = _bar_agg(tw, rs[:k] / np.clip(rm[:k], 1e-9, None), grid, n)

    # — hauteur —
    t0, f0, vo, r0 = VM.track_f0(voc)
    loud = r0 > max(1e-4, .12 * float(np.percentile(r0, 95)))
    ok = vo & (f0 > 0) & loud
    semi = np.where(ok, 69 + 12 * np.log2(np.clip(f0, 1e-6, None) / 440.0), np.nan)
    F["voiced"] = _bar_agg(t0, ok.astype(float), grid, n)
    F["f0_med"] = _bar_agg(t0, semi, grid, n)
    F["f0_std"] = _bar_agg(t0, semi, grid, n, "std")
    F["f0_max"] = _bar_agg(t0, semi, grid, n, "max")
    cents = np.where(ok, semi, np.nan)
    dv = np.abs(np.diff(cents, prepend=cents[0])) * 100
    F["vibrato"] = _bar_agg(t0, np.where(np.isfinite(dv) & (dv < 200), dv, np.nan), grid, n)

    f0i = np.interp(tf, t0, np.where(ok, f0, 0.0))
    voi = np.interp(tf, t0, ok.astype(float)) > .5
    F["comb"] = _bar_agg(tf, _comb(tf, f0i, voi, S, freqs, S.shape[1]), grid, n)

    # — les notes extraites —
    notes, _ = VM.melody_notes(t0, f0, vo, r0)
    notes, _ = VM.clean(notes)
    dens = np.zeros(n)
    cov = np.zeros(n)
    for a, d, m in notes:
        b = int(np.searchsorted(grid, a) - 1)
        if 0 <= b < n:
            dens[b] += 1
            cov[b] += min(d, grid[b + 1] - grid[b])
    F["note_dens"] = dens
    F["note_cov"] = cov / np.clip(np.diff(grid[:n + 1]), 1e-9, None)

    # — les matrices de similarité de la voix —
    P = np.zeros((n, 12))
    for a, d, m in notes:
        b = int(np.searchsorted(grid, a) - 1)
        if 0 <= b < n:
            P[b, int(m) % 12] += d
    for tag, V in (("pitch", P), ("timbre", np.stack([
            _bar_agg(tf, mf[j], grid, n) for j in range(1, 14)], 1))):
        if tag == "timbre":
            V = V - V.mean(0)
        r, M = _ssm_rest(V, n, top=None)
        rt, _ = _ssm_rest(V, n, top=6)
        F[f"sim_{tag}"] = r
        F[f"simtop_{tag}"] = rt
        F[f"rank_{tag}"] = np.argsort(np.argsort(r)) / max(1, n - 1)
        F[f"foote_{tag}"] = _foote(M)
        F[f"_M_{tag}"] = np.where(np.isfinite(M), M, 0.0)

    F["_notes"] = np.array(notes, float) if notes else np.zeros((0, 3))
    F["_grid"] = np.asarray(grid, float)
    F["_mute"] = (P.sum(1) <= 0).astype(float)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **F)
    return F


# ── les détecteurs ──────────────────────────────────────────────────────────

def step_curve(x, kmax=KMAX, win=WIN):
    """La force de la MARCHE à la mesure k : « avant k » contre « après k ».

    Une frontière intro/chanson est une marche dans la courbe, pas un pic. On
    prend donc, pour chaque k, l'écart des moyennes de part et d'autre, divisé
    par l'écart-type groupé — un t de Welch, sans sa loi. Le signe est gardé :
    +1 si le descripteur MONTE en entrant dans la chanson, −1 s'il descend.
    """
    n = len(x)
    out = np.full(kmax + 1, -np.inf)
    for k in range(1, min(kmax, n - win) + 1):
        a, b = x[max(0, k - win):k], x[k:k + win]
        if len(a) < 1 or len(b) < 2:
            continue
        sd = np.sqrt((np.var(a) + np.var(b)) / 2) + 1e-6
        out[k] = (np.mean(b) - np.mean(a)) / sd
    return out


def detect_step(x, sign, kmax=KMAX, win=WIN):
    """(mesure prédite, force de la marche) pour un descripteur donné."""
    s = sign * step_curve(x, kmax, win)
    s[0] = -np.inf
    k = int(np.argmax(s))
    return k, float(s[k])


def detect_thresh(x, q, kmax=KMAX):
    """La 1re mesure où le descripteur passe le q-quantile du morceau."""
    n = len(x)
    lo, hi = np.nanmin(x), np.nanmax(x)
    thr = lo + q * (hi - lo)
    for k in range(min(kmax, n - 1) + 1):
        if x[k] >= thr:
            return k, float(x[k] - thr)
    return 0, 0.0


# les descripteurs, avec le SENS attendu en entrant dans la chanson.
# +1 = ça monte quand la chanson commence. Le sens est posé A PRIORI (une voix
# entre : ça chante plus, plus fort, plus haut, et ça se met à ressembler au
# reste) et non choisi après coup ; les deux sens sont quand même mesurés et la
# colonne « sens » du rapport dit lequel a servi.
SIGNS = {
    "voiced": +1, "note_cov": +1, "note_dens": +1, "rms": +1, "rms_p90": +1,
    "rms_std": +1, "rms_dyn": +1, "f0_med": +1, "f0_std": +1, "f0_max": +1,
    "vibrato": +1, "centroid": +1, "bandwidth": +1, "flatness": -1,
    "rolloff85": +1, "contrast": +1, "contrast_hi": +1, "zcr": +1,
    "mfcc1": +1, "mfcc2": +1, "mfcc3": +1, "mfcc4": +1, "mfcc_delta": +1,
    "stereo_side": +1, "comb": -1,
    "sim_pitch": +1, "simtop_pitch": +1, "rank_pitch": +1, "foote_pitch": +1,
    "sim_timbre": +1, "simtop_timbre": +1, "rank_timbre": +1, "foote_timbre": +1,
}
CURVES = list(SIGNS)

FR = {
    "voiced": "voix présente (part de la mesure)",
    "note_cov": "mesure couverte par des notes chantées",
    "note_dens": "densité de notes",
    "rms": "énergie moyenne", "rms_p90": "énergie du 9e décile",
    "rms_std": "dynamique dans la mesure", "rms_dyn": "dynamique relative",
    "f0_med": "hauteur médiane", "f0_std": "écart-type de la hauteur",
    "f0_max": "hauteur maximale (tessiture haute)",
    "vibrato": "agitation de la hauteur (vibrato)",
    "centroid": "centroïde spectral", "bandwidth": "largeur de bande",
    "flatness": "aplatissement spectral", "rolloff85": "rolloff 85 %",
    "contrast": "contraste spectral", "contrast_hi": "contraste, bande haute",
    "zcr": "passages par zéro",
    "mfcc1": "MFCC 1", "mfcc2": "MFCC 2", "mfcc3": "MFCC 3", "mfcc4": "MFCC 4",
    "mfcc_delta": "vitesse du timbre (Δ MFCC)",
    "stereo_side": "largeur stéréo (doublage)",
    "comb": "énergie hors peigne harmonique (harmonisation)",
    "sim_pitch": "SSM chant : ressemblance au reste",
    "simtop_pitch": "SSM chant : ressemblance aux 6 meilleures",
    "rank_pitch": "SSM chant : rang de la mesure",
    "foote_pitch": "SSM chant : nouveauté de Foote",
    "sim_timbre": "SSM timbre : ressemblance au reste",
    "simtop_timbre": "SSM timbre : ressemblance aux 6 meilleures",
    "rank_timbre": "SSM timbre : rang de la mesure",
    "foote_timbre": "SSM timbre : nouveauté de Foote",
}


def shipped(stem, F):
    """La règle LIVRÉE : `voice_sections.sung_start`. C'est elle qu'il faut battre."""
    from harmonia_min import voice_sections as VS
    notes = [tuple(x) for x in F["_notes"]]
    return int(VS.sung_start(notes, list(F["_grid"]), F["_mute"] > .5))


def collect(stems=None):
    T = truth()
    stems = stems or sorted(T)
    from pattern_lanes import load
    data = {}
    for st in stems:
        S, n, g = load(st)
        F = features(st, g, n)
        data[st] = {"n": n, "grid": F["_grid"], "truth": T[st], "F": F,
                    "shipped": shipped(st, F), "S": S}
    return data


if __name__ == "__main__":
    import intro_vocal_report as R
    R.main(sys.argv[1:])
