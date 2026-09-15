"""Le morceau réduit à une ligne de texte par mesure — ce qu'un LLM peut lire.

    .venv/bin/python scripts/resume_texte.py <stem>
        -> l'affiche, et l'écrit dans scratchpad/ia_sections/<stem>.txt

Louis, 2026-08-14 : « et si un quick appel à une IA peut le faire aussi ».

L'IDÉE. Un modèle de langue n'entend pas l'audio. Mais il a lu des milliers de
descriptions de formes de chansons, et la forme pop est très contrainte
(couplet 8, refrain 8, pont 8, la reprise est identique au premier passage). Si
on lui donne un résumé HONNÊTE et compact — une ligne par mesure : l'accord, si
ça chante, le niveau sonore, la batterie — la question « où sont les frontières »
devient un problème de LECTURE DE MOTIF, ce qu'un LLM fait bien et vite.

CE QU'IL VOIT, mesure par mesure :

    mes  accord   chant  niveau  batterie
    12   Am       ..##   -2      ###

  accord   l'argmax du postérieur musx sur la mesure (plan de triades) — donc
           la même source que la matrice « accords » de la page des critères,
           pas une transcription humaine.
  chant    quatre cases : le chant est-il présent sur chaque quart de mesure.
  niveau   le RMS de la mesure en dB, RELATIF à la médiane du morceau (0 = le
           niveau ordinaire, −6 = deux fois moins fort).
  batterie l'énergie d'attaques de la piste batterie séparée, en quartiles du
           morceau (0 à 4). `-` = pas de piste batterie exploitable.

CE QUE CE RÉSUMÉ NE DIT PAS, et c'est volontaire : aucune frontière, aucune
lettre de section, aucun compte de mesures « rond ». Le fichier ne contient rien
qui vienne des annotations de Louis — sinon le test serait truqué.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
sys.path.insert(0, str(HERE / "scratchpad"))

OUT = HERE / "scratchpad" / "ia_sections"
OUT.mkdir(exist_ok=True)

NOMS = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
TYPES = {1: "", 2: "m", 3: "sus4", 4: "sus2", 5: "dim", 6: "aug"}


def accord_par_mesure(triad, grid):
    """L'argmax du postérieur musx, mesure par mesure — un nom d'accord."""
    from harmonia_min.harmonic_sections import FRAME_DT
    out = []
    for b in range(len(grid) - 1):
        a = int(grid[b] / FRAME_DT)
        z = max(a + 1, int(grid[b + 1] / FRAME_DT))
        seg = triad[a:min(z, len(triad))]
        if not len(seg):
            out.append("?")
            continue
        i = int(np.argmax(seg.mean(0)))
        if i == 0:
            out.append("N")
        else:
            out.append(NOMS[(i - 1) % 12] + TYPES.get((i - 1) // 12 + 1, "?"))
    return out


def resume(stem):
    import librosa
    import order_bundle
    from ssm_zoo import capture
    from licks import notes_de
    from criteres_sections import cases, pool, AUDIO

    b = order_bundle.get(stem)
    n, grid = b["n"], np.asarray(b["grid"], float)
    cap = capture(stem)
    ch = accord_par_mesure(cap["triad"], grid)
    notes = notes_de(stem)

    y, sr = librosa.load(str(AUDIO / f"{stem}.m4a"), sr=22050, mono=True)
    hop = 512
    tf = librosa.frames_to_time(np.arange(1 + len(y) // hop), sr=sr, hop_length=hop)
    rms = librosa.amplitude_to_db(np.clip(
        librosa.feature.rms(y=y, hop_length=hop)[0], 1e-6, None))
    e1 = cases(grid, n, 1)
    db = pool(rms[:, None], tf[:len(rms)], e1)[:, 0]
    db = db - np.median(db)

    bat = None
    try:
        from rhythm_ssm import separate_drums, _drum_onset_bands
        import soundfile as sf
        d = separate_drums(AUDIO / f"{stem}.m4a")
        if d is not None:
            dy, dsr = sf.read(str(d))
            dy = dy.mean(1) if dy.ndim > 1 else dy
            if dsr != 22050:
                dy = librosa.resample(dy, orig_sr=dsr, target_sr=22050)
            env, tenv = _drum_onset_bands(dy.astype(np.float32), 22050)
            v = pool(env.mean(0)[:, None], tenv, e1)[:, 0]
            q = np.quantile(v, [.2, .4, .6, .8])
            bat = np.searchsorted(q, v)
    except Exception:
        bat = None

    # le chant, quart de mesure par quart de mesure
    e4 = cases(grid, n, 4)
    on = np.zeros(len(e4) - 1, bool)
    for t0, dur, _m in notes:
        i0 = int(np.searchsorted(e4, t0) - 1)
        i1 = int(np.searchsorted(e4, t0 + dur) - 1)
        for i in range(max(0, i0), min(len(on), i1 + 1)):
            on[i] = True

    L = [f"{stem} · {n} mesures · une ligne par mesure",
         "mes  accord  chant  niveau  batterie"]
    for i in range(n):
        c = "".join("#" if on[4 * i + k] else "." for k in range(4))
        bb = "-" if bat is None else "#" * int(bat[i])
        L.append(f"{i + 1:>3}  {ch[i]:<6}  {c}   {db[i]:>+5.1f}  {bb}")
    return "\n".join(L)


def main():
    for stem in sys.argv[1:]:
        txt = resume(stem)
        (OUT / f"{stem}.txt").write_text(txt)
        print(txt)
        print()


if __name__ == "__main__":
    main()
