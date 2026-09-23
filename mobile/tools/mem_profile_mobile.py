"""Pic de mémoire (RSS) du chemin de calcul mobile, étape par étape.

Pourquoi un sous-processus par étape : ``ru_maxrss`` est un HIGH WATERMARK
monotone du processus — il ne redescend jamais. Mesurer deux étapes dans le
même processus donnerait au second le pic du premier. Chaque étape tourne
donc dans un python neuf, et on lit ``ru_maxrss`` à la fin.

Sur macOS ``ru_maxrss`` est en OCTETS (pas en Ko comme sur Linux) — vérifié
empiriquement (tableau float64 de 200 Mo touché -> 229 Mo rapportés).

    ~/harmonia/.venv-mobile/bin/python mobile/tools/mem_profile_mobile.py
    ~/harmonia/.venv-mobile/bin/python mobile/tools/mem_profile_mobile.py \
        --stage musx --song maroon_5_this_love --layers
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
AUDIO = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia/docs/audio")
CQT_REF = Path(__file__).resolve().parent / "cqt_ref"
SONGS = ["maroon_5_this_love", "ben_e_king_stand_by_me_audio",
         "maroon_5_she_will_be_loved_official_music_video"]
STAGES = ["baseline", "decode", "cqt", "musx", "nnls"]


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def _cqt_path(song: str) -> Path:
    return CQT_REF / f"{song}.npz"


def _get_cqt(song: str) -> np.ndarray:
    """CQT (T, 288) : soit le dump P2b, soit recalculé depuis l'audio."""
    p = _cqt_path(song)
    if p.exists():
        with np.load(p) as z:
            return z["cqt"].astype(np.float32)
    cache = Path(os.environ.get("TMPDIR", "/tmp")) / f"mem_cqt_{song}.npy"
    if cache.exists():
        return np.load(cache).astype(np.float32)
    sys.path.insert(0, str(REPO))
    from mobile.harmonia_mobile import backends_mobile as bm
    from mobile.harmonia_mobile.cqt_musx import song_cqt_288
    y = bm.audio_ffmpeg(AUDIO / f"{song}.m4a", 22050)
    cqt = song_cqt_288(y, sr=22050).astype(np.float32)
    np.save(cache, cqt)
    return cqt


def run_stage(stage: str, song: str, layers: bool = False, chunk: int | None = None) -> dict:
    """Deux chiffres, et ils ne disent pas la même chose.

    ``pic_rss_mb`` = ``ru_maxrss``, le pic de mémoire RÉSIDENTE du processus.
    C'est ce que l'OS a dû fournir — mais sur macOS l'allocateur garde les
    gros blocs libérés dans ses caches, donc ce pic surestime ce qui est
    vivant en même temps (mesuré : 2611 Mo de watermark pour 1862 Mo de
    tableaux réellement simultanés).

    ``pic_alloc_mb`` = ``tracemalloc``, domaine numpy inclus : la somme des
    tableaux VIVANTS au pire instant. C'est le chiffre à comparer au budget
    du téléphone, parce qu'il ne dépend pas de l'allocateur du Mac.
    """
    import tracemalloc
    sys.path.insert(0, str(REPO))
    info: dict = {}
    tracemalloc.start()
    if stage == "baseline":
        import numpy  # noqa: F401
        from mobile.harmonia_mobile import musx_net  # noqa: F401
    elif stage == "decode":
        from mobile.harmonia_mobile import backends_mobile as bm
        y = bm.audio_ffmpeg(AUDIO / f"{song}.m4a", 22050)
        info["n"] = int(y.size)
        info["duree_s"] = round(y.size / 22050, 1)
    elif stage == "cqt":
        from mobile.harmonia_mobile import backends_mobile as bm
        from mobile.harmonia_mobile.cqt_musx import song_cqt_288
        y = bm.audio_ffmpeg(AUDIO / f"{song}.m4a", 22050)
        base = rss_mb()
        cqt = song_cqt_288(y, sr=22050)
        info["apres_decodage_mb"] = round(base, 1)
        info["T"] = int(cqt.shape[0])
    elif stage == "musx":
        from mobile.harmonia_mobile import musx_net
        cqt = _get_cqt(song)
        w = [musx_net.load_weights_npz(REPO / "mobile" / "models" / f"musx_fold{i}_weights.npz")
             for i in range(5)]
        base = rss_mb()
        info["avant_reseau_mb"] = round(base, 1)
        info["T"] = int(cqt.shape[0])
        if layers:
            info["layers"] = _layer_trace(musx_net, cqt, w[0])
        else:
            kw = {} if chunk is None else {"chunk_frames": chunk}
            musx_net.predict_ensemble(cqt, w, **kw)
    elif stage == "nnls":
        from mobile.harmonia_mobile import backends_mobile as bm
        from mobile.harmonia_mobile.nnls_chroma import extract_bothchroma_numpy
        y = bm.audio_ffmpeg(AUDIO / f"{song}.m4a", 44100)
        base = rss_mb()
        info["apres_decodage_mb"] = round(base, 1)
        extract_bothchroma_numpy(y, 44100)
    else:
        raise SystemExit(f"stage inconnu: {stage}")
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    info["pic_alloc_mb"] = round(peak / 1e6, 1)
    info["pic_rss_mb"] = round(rss_mb(), 1)
    return info


def _layer_trace(musx_net, cqt_288: np.ndarray, w: dict) -> list:
    """Un fold, couche par couche : plus gros tableau temporaire + pic RSS."""
    trace = []
    orig_conv, orig_norm = musx_net._conv2d, musx_net._instance_norm
    state = {"name": "?"}

    def conv(x, weight, bias, pad_t, pad_f):
        cin, t, f = x.shape
        cout, _, kt, kf = weight.shape
        tout, fout = t + 2 * pad_t - kt + 1, f + 2 * pad_f - kf + 1
        taps_mb = kt * kf * cin * tout * fout * 4 / 1e6
        out_mb = tout * fout * cout * 4 / 1e6
        before = rss_mb()
        r = orig_conv(x, weight, bias, pad_t, pad_f)
        trace.append(dict(couche=state["name"], op="conv2d", entree=list(x.shape),
                          sortie=list(r.shape), im2col_mb=round(taps_mb, 1),
                          sortie_mb=round(out_mb, 1),
                          rss_avant_mb=round(before, 1), rss_apres_mb=round(rss_mb(), 1)))
        return r

    def norm(x, eps=1e-5):
        before = rss_mb()
        r = orig_norm(x, eps)
        trace.append(dict(couche=state["name"], op="instance_norm", entree=list(x.shape),
                          f32_mb=round(x.nbytes / 1e6, 1), f64_mb=round(x.nbytes * 2 / 1e6, 1),
                          rss_avant_mb=round(before, 1), rss_apres_mb=round(rss_mb(), 1)))
        return r

    musx_net._conv2d, musx_net._instance_norm = conv, norm
    # nommer la couche courante
    orig_cnn = musx_net._cnn_forward

    def cnn(cqt_window, ww):
        x = cqt_window[None, :, :]
        for name, pad_t, pad_f in musx_net._CONV_LAYERS:
            if name == "POOL":
                x = musx_net._maxpool_freq(x, pad_t)
                trace.append(dict(couche="POOL", op="maxpool", sortie=list(x.shape),
                                  rss_apres_mb=round(rss_mb(), 1)))
                continue
            state["name"] = name
            p = f"audio_feature_block.{name}"
            x = musx_net._conv2d(x, ww[f"{p}.weight"], ww[f"{p}.bias"], pad_t, pad_f)
            x = musx_net._instance_norm(x)
            x = musx_net._selu(x)
            trace.append(dict(couche=name, op="selu", rss_apres_mb=round(rss_mb(), 1)))
        c, t, f = x.shape
        return x.transpose(1, 0, 2).reshape(t, c * f)

    musx_net._cnn_forward = cnn
    try:
        musx_net.predict_fold(cqt_288, w)
    finally:
        musx_net._conv2d, musx_net._instance_norm = orig_conv, orig_norm
        musx_net._cnn_forward = orig_cnn
    trace.append(dict(couche="TOTAL_fold", rss_apres_mb=round(rss_mb(), 1)))
    return trace


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage")
    ap.add_argument("--song")
    ap.add_argument("--layers", action="store_true")
    ap.add_argument("--chunk", type=int, default=None)
    ap.add_argument("--child", action="store_true")
    a = ap.parse_args()

    if a.child:
        print("__JSON__" + json.dumps(run_stage(a.stage, a.song, a.layers, a.chunk)))
        return

    if a.stage:
        stages, songs = [a.stage], [a.song] if a.song else SONGS
    else:
        stages, songs = STAGES, SONGS

    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[k] = "1"
    results: dict = {}
    for song in songs:
        for stage in stages:
            cmd = [sys.executable, __file__, "--child", "--stage", stage, "--song", song]
            if a.layers:
                cmd.append("--layers")
            if a.chunk is not None:
                cmd += ["--chunk", str(a.chunk)]
            out = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(REPO))
            line = [l for l in out.stdout.splitlines() if l.startswith("__JSON__")]
            if not line:
                print(f"ECHEC {song}/{stage}:\n{out.stdout[-2000:]}\n{out.stderr[-2000:]}")
                continue
            info = json.loads(line[0][len("__JSON__"):])
            results.setdefault(song, {})[stage] = info
            if a.layers and "layers" in info:
                for row in info["layers"]:
                    print("   ", json.dumps(row))
            print(f"{song:<48} {stage:<9} vivant={info.get('pic_alloc_mb', -1):>8.1f} "
                  f"rss={info['pic_rss_mb']:>8.1f} Mo   "
                  f"{ {k: v for k, v in info.items() if k not in ('pic_rss_mb', 'pic_alloc_mb', 'layers')} }",
                  flush=True)
    outp = Path(__file__).resolve().parent / "parity_out" / "mem_profile.json"
    outp.parent.mkdir(exist_ok=True)
    prev = json.loads(outp.read_text()) if outp.exists() else {}
    tag = "chunk%s" % a.chunk if a.chunk else ("layers" if a.layers else "plein")
    prev[tag] = results
    outp.write_text(json.dumps(prev, indent=1))
    print("\n->", outp)


if __name__ == "__main__":
    main()
