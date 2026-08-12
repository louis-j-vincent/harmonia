"""Labo d'agrégation des bi-mesures similaires (demande Louis 2026-08-08).

« Ton rôle est de savoir COMMENT exploiter les répétitions pour avoir des
prédictions plus fiables […] en agrégeant les bi-barres détectées comme
similaires : empiler les audio et CQT en amont du modèle ? empiler les
priors en aval ? Ne score rien, montre-moi juste ce que ça donne à chaque
fois pour que je juge à l'oreille. »

Pour chaque chanson : les bi-mesures (grille paire) sont groupées par
similarité (même substrat que la détection de sections), et pour chaque
groupe on montre QUATRE décodages, sans aucun score :

  seul   chaque bi-mesure décodée seule (la référence « 1 observation ») ;
  audio  les signaux audio des membres superposés (moyennés, alignés par
         rééchantillonnage) → CQT → musx → décodage ; l'audio superposé
         est ÉCOUTABLE sur la page (le déphasage attendu s'entend) ;
  cqt    le CQT de chaque membre, moyennés (pas de déphasage possible :
         domaine magnitude) → musx → décodage ;
  probs  les 6 flux de probabilités musx des membres, moyennés → décodage
         (= ce que fait le repli actuel, à la granularité bi-mesure).

Tout est décodé par LE MÊME chemin (template 2 mesures pavé ×3, Viterbi
beat-grid, latence 0) pour que seule l'agrégation change.

    HARMONIA_MUSX_DIR=<clone de l'arbre principal> \
    python scripts/bibar_stack_lab.py <stem> [--thr 0.92] [--max 8]

Sort : harmonia_min/state/occmerge/lab_<stem>.json (résultats),
       <LIVE>/state/reports/occmerge_audio/<stem>/c<i>.wav (superposés).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from harmonia_min import musx as _musx                      # noqa: E402
from harmonia_min.folding import _bar_vecs, _resample       # noqa: E402
from harmonia_min.sections import halfbar_features          # noqa: E402

LIVE = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia")
AUDIO_DIR = LIVE / "docs" / "audio"
WAV_DIR = LIVE / "harmonia_min" / "state" / "reports" / "occmerge_audio"
SNAP = REPO / "harmonia_min" / "state" / "occmerge"

SR = _musx.MUSX_SR

# Seuil du check de cohérence (Louis 2026-08-08 : « regarder le score de
# prédiction final de musx et voir s'il est sous un certain seuil »).
# 0.60 = point de départ à arbitrer sur les pages, pas une calibration.
CHECK_THR = 0.60


def bibar_clusters(grid, arr, times, thr=0.92):
    """Groupes de bi-mesures similaires sur la grille paire (lien complet)."""
    n_bars = len(grid) - 1
    F = halfbar_features(grid, arr, times)
    n_bi = n_bars // 2
    W = np.array([F[4 * j:4 * j + 4].reshape(-1) for j in range(n_bi)])
    W = W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-9)
    S = W @ W.T
    used, clusters = set(), []
    order = sorted(range(n_bi),
                   key=lambda j: -float((S[j] >= thr).sum()))
    for j in order:
        if j in used:
            continue
        mem = [j]
        for k in range(n_bi):
            if k == j or k in used:
                continue
            if all(S[k, m] >= thr for m in mem):
                mem.append(k)
        if len(mem) >= 3:
            clusters.append(sorted(mem))
            used.update(mem)
    clusters.sort(key=lambda m: m[0])
    return clusters


def decode_block(block, dur, bpb=4):
    """Décodage commun : template 2 mesures pavé ×3, latence 0.

    Le décodage étant périodique, un accord TENU à cheval sur la frontière
    de copie n'a pas de départ dans la fenêtre du milieu — il est réémis en
    t0=0 avec `carry` (sans ça, une bi-mesure à accord unique rendait une
    liste VIDE, et l'adhésion du check retombait sur le 0.5 neutre)."""
    cat = [np.concatenate([np.asarray(p, dtype=np.float64)] * 3) for p in block]
    step = dur / (2 * bpb)
    beats = [i * step for i in range(3 * 2 * bpb + 1)]
    lab, _ = _musx.redecode(beats, cat, downbeat_times=beats[::bpb],
                            beat_trans_penalty=(15.0, 15.0, 100.0),
                            quarter_beats="all", latency_grid=(0.0,))
    out, spanning = [], None
    for t0, t1, s in lab:
        if t0 < dur - 1e-6:
            if t1 > dur + 1e-6:
                spanning = (s, min(t1, 2 * dur))
            continue
        if t0 >= 2 * dur - 1e-6:
            continue
        conf = _musx.label_confidence(cat[0], t0, min(t1, 2 * dur), s)
        out.append({"t0": round(t0 - dur, 3), "t1": round(min(t1, 2 * dur) - dur, 3),
                    "label": s, "c": round(conf, 3)})
    if spanning and (not out or out[0]["t0"] > 1e-3):
        s, t_end = spanning
        conf = _musx.label_confidence(cat[0], dur, t_end, s)
        out.insert(0, {"t0": 0.0, "t1": round(t_end - dur, 3), "label": s,
                       "c": round(conf, 3), "carry": True})
    return out


def slice_probs(probs, t0, t1):
    a = max(0, int(round(t0 / _musx.FRAME_DT)))
    z = min(probs[0].shape[0], int(round(t1 / _musx.FRAME_DT)))
    return [p[a:z] for p in probs]


def _cqt_of_wav(path):
    with _musx._InMusxDir():
        from mir import io, DataEntry
        from extractors.cqt import CQTV2
        entry = DataEntry()
        entry.prop.set('sr', SR)
        entry.prop.set('hop_length', _musx.MUSX_HOP)
        entry.append_file(str(path), io.MusicIO, 'music')
        entry.append_extractor(CQTV2, 'cqt')
        return np.asarray(entry.cqt)


def _posteriors_from_cqt(cqt):
    with _musx._InMusxDir():
        from mir.nn.train import NetworkInterface
        from chordnet_ismir_naive import ChordNet
        acc = None
        for name in _musx.MODEL_NAMES:
            net = NetworkInterface(ChordNet(None), name, load_checkpoint=False)
            out = net.inference(cqt)
            acc = list(out) if acc is None else [a + b for a, b in zip(acc, out)]
            del net
    return [(a / len(_musx.MODEL_NAMES)).astype(np.float64) for a in acc]


def run(stem, thr=0.92, max_clusters=8):
    import librosa
    import soundfile as sf
    snap = json.loads((SNAP / f"{stem}.json").read_text())
    grid, bpb = snap["grid"], snap["bpb"]
    audio = AUDIO_DIR / f"{stem}.m4a"
    probs = _musx.frame_posteriors(audio)
    from harmonia_min.nnls_features import extract_bothchroma
    arr, times = extract_bothchroma(audio)
    clusters = bibar_clusters(grid, arr, times, thr)
    n_all = len(clusters)
    clusters = sorted(clusters, key=lambda m: -len(m))[:max_clusters]
    clusters.sort(key=lambda m: m[0])
    wdir = WAV_DIR / stem
    wdir.mkdir(parents=True, exist_ok=True)
    y_full, _ = librosa.load(str(audio), sr=SR, mono=True)

    res = {"stem": stem, "title": snap.get("title", stem), "bpb": bpb,
           "grid": grid, "audio": snap["audio"], "thr": thr,
           "n_clusters_total": n_all, "clusters": []}
    for ci, mem in enumerate(clusters):
        spans = [(grid[2 * j], grid[2 * j + 2]) for j in mem]
        durs = [t1 - t0 for t0, t1 in spans]
        dur = float(np.median(durs))
        Lf = int(round(dur / _musx.FRAME_DT))

        # seul : chaque membre décodé sur SES probabilités
        solo = []
        for (t0, t1) in spans:
            blk = [_resample(p, Lf) for p in slice_probs(probs, t0, t1)]
            solo.append(decode_block(blk, dur, bpb))

        # probs : moyenne des 6 flux
        stacks = [[_resample(p, Lf) for p in slice_probs(probs, t0, t1)]
                  for t0, t1 in spans]
        blk_probs = [np.mean([s[i] for s in stacks], axis=0)
                     for i in range(len(probs))]
        agg_probs = decode_block(blk_probs, dur, bpb)

        # audio : superposition des signaux (rééchantillonnés à dur commune)
        n_out = int(round(dur * SR))
        ys = []
        for (t0, t1) in spans:
            seg = y_full[int(round(t0 * SR)):int(round(t1 * SR))]
            if len(seg) < 2:
                continue
            xs = np.linspace(0, len(seg) - 1, n_out)
            ys.append(np.interp(xs, np.arange(len(seg)), seg))
        y_stack = np.mean(ys, axis=0)
        peak = float(np.max(np.abs(y_stack)) or 1.0)
        y_stack = 0.9 * y_stack / peak
        wav = wdir / f"c{ci}.wav"
        sf.write(wav, y_stack, SR)
        p_audio = _musx.frame_posteriors(wav, use_cache=False)
        blk_audio = [_resample(p, Lf) for p in p_audio]
        agg_audio = decode_block(blk_audio, dur, bpb)

        # cqt : moyenne des CQT membres (magnitude, pas de déphasage)
        cqts = []
        for k, (t0, t1) in enumerate(spans):
            seg = y_full[int(round(t0 * SR)):int(round(t1 * SR))]
            tmp = wdir / f"_m{ci}_{k}.wav"
            sf.write(tmp, seg, SR)
            cqts.append(_cqt_of_wav(tmp))
            tmp.unlink()
        Tm = int(np.median([c.shape[0] for c in cqts]))
        cqts = [_resample(c, Tm) for c in cqts]
        p_cqt = _posteriors_from_cqt(np.mean(cqts, axis=0))
        blk_cqt = [_resample(p, Lf) for p in p_cqt]
        agg_cqt = decode_block(blk_cqt, dur, bpb)

        # check de cohérence (Louis 2026-08-08) : le score FINAL de musx,
        # sous un seuil, dans les deux placements.
        #   avant : « adhésion » d'une répétition = score musx des accords
        #   du consensus, mesuré sur SES frames à elle ; sous le seuil elle
        #   est écartée et le CQT est re-moyenné sans elle ;
        #   après : un accord du consensus sous le seuil reste marqué sur
        #   la page (le flag est posé au rendu, ici on stocke les scores).
        # L'adhésion n'est mesurée que sur les accords du consensus EUX-MÊMES
        # confiants (≥ seuil) : punir une répétition de ne pas coller à un
        # accord que le consensus lui-même ne soutient pas (This Love : un
        # Ab:7 à 0.148) excluait à tort. Aucun accord confiant → pas de
        # verdict (None), jamais d'exclusion.
        anchors = [e for e in agg_cqt if e["c"] >= CHECK_THR]
        adhesion = []
        for s in stacks:
            scs = [_musx.label_confidence(s[0], e["t0"], e["t1"], e["label"])
                   for e in anchors]
            adhesion.append(round(float(np.mean(scs)), 3) if scs else None)
        excl = [k for k, a in enumerate(adhesion)
                if a is not None and a < CHECK_THR]
        agg_cqt2 = None
        if excl and len(spans) - len(excl) >= 2:
            keep = [c for k, c in enumerate(cqts) if k not in excl]
            p2 = _posteriors_from_cqt(np.mean(keep, axis=0))
            agg_cqt2 = decode_block([_resample(p, Lf) for p in p2], dur, bpb)

        res["clusters"].append({
            "bibars": mem, "spans": [[round(a, 3), round(b, 3)]
                                     for a, b in spans],
            "dur": round(dur, 3),
            "solo": solo, "probs": agg_probs, "audio": agg_audio,
            "cqt": agg_cqt, "wav": f"occmerge_audio/{stem}/c{ci}.wav",
            "adhesion": adhesion, "check_thr": CHECK_THR,
            "excluded": excl, "cqt_sans_ecartees": agg_cqt2})
        print(f"  cluster {ci}: {len(mem)} bi-mesures "
              f"(bars {[2 * j for j in mem]})", flush=True)
    out = SNAP / f"lab_{stem}.json"
    out.write_text(json.dumps(res))
    print(f"→ {out} ({len(res['clusters'])}/{n_all} groupes)")


if __name__ == "__main__":
    stem = sys.argv[1]
    thr = float(sys.argv[sys.argv.index("--thr") + 1]) \
        if "--thr" in sys.argv else 0.92
    mx = int(sys.argv[sys.argv.index("--max") + 1]) \
        if "--max" in sys.argv else 8
    run(stem, thr, mx)
