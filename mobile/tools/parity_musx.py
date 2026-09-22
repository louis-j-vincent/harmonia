"""Parity + speed: numpy ChordNet ensemble (musx_net.py) vs the real torch
5-fold ensemble, on the CQT references already dumped by P2b
(``mobile/tools/cqt_ref/*.npz``, from ``cqt_ref_extract.py`` — the vetted
librosa CQT path via the real clone, chdir included).

Torch reference = ``harmonia.musx._run_nets`` equivalent (5 folds, simple
mean) called directly here (not through the pipeline, no cache dependency).

Run with the MOBILE venv:
    ~/harmonia/.venv-mobile/bin/python mobile/tools/parity_musx.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
MUSX_DIR = REPO / "third_party" / "musx_ismir2019"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
CQT_REF_DIR = Path(__file__).resolve().parent / "cqt_ref"

sys.path.insert(0, str(REPO / "mobile"))
from harmonia_mobile import musx_net  # noqa: E402

MODEL_NAMES = ['joint_chord_net_ismir_naive_v1.0_reweight(0.0,10.0)_s%d.best' % i
               for i in range(5)]

SONGS = ["maroon_5_this_love", "ben_e_king_stand_by_me_audio", "autumn_leaves"]


class InMusxDir:
    def __enter__(self):
        import os
        self._cwd = Path.cwd()
        os.chdir(MUSX_DIR)
        if str(MUSX_DIR) not in sys.path:
            sys.path.insert(0, str(MUSX_DIR))
        return self

    def __exit__(self, *exc):
        import os
        os.chdir(self._cwd)
        return False


def torch_ensemble_probs(cqt_288: np.ndarray) -> list[np.ndarray]:
    import torch
    with InMusxDir():
        from mir.nn.train import NetworkInterface
        from chordnet_ismir_naive import ChordNet
        nets = []
        for name in MODEL_NAMES:
            ni = NetworkInterface(ChordNet(None), name, load_checkpoint=False)
            nets.append(ni.net.eval())
    x = torch.tensor(np.asarray(cqt_288), dtype=torch.float32)
    acc = None
    with torch.no_grad():
        for net in nets:
            out = net.inference(x)  # tuple of 6 numpy arrays already (softmaxed, .cpu().numpy())
            acc = list(out) if acc is None else [a + b for a, b in zip(acc, out)]
    return [(a / len(nets)).astype(np.float32) for a in acc]


def main():
    weights_by_fold = [musx_net.load_weights_npz(MODELS_DIR / f"musx_fold{i}_weights.npz")
                        for i in range(5)]

    rows = []
    for song in SONGS:
        ref_path = CQT_REF_DIR / f"{song}.npz"
        if not ref_path.exists():
            print(f"SKIP {song}: no {ref_path}")
            continue
        z = np.load(ref_path)
        cqt288 = z["cqt"].astype(np.float32)
        n_frame = cqt288.shape[0]
        duration_s = n_frame * 512 / 22050

        # torch ensemble (ground truth)
        t0 = time.time()
        torch_probs = torch_ensemble_probs(cqt288)
        t_torch = time.time() - t0

        # numpy ensemble
        t0 = time.time()
        np_probs = musx_net.predict_ensemble(cqt288, weights_by_fold)
        t_numpy = time.time() - t0

        names = ("triad", "bass", "s7", "s9", "s11", "s13")
        max_abs_diff = max(float(np.abs(a - b).max()) for a, b in zip(torch_probs, np_probs))
        triad_argmax_t = torch_probs[0].argmax(axis=1)
        triad_argmax_n = np_probs[0].argmax(axis=1)
        argmax_agree = float((triad_argmax_t == triad_argmax_n).mean())

        per_head = {n: float(np.abs(a - b).max()) for n, a, b in zip(names, torch_probs, np_probs)}

        norm_4min = t_numpy * 240.0 / duration_s

        print(f"\n== {song} == duration={duration_s:.1f}s n_frame={n_frame}")
        print(f"  torch ensemble (5 folds): {t_torch:.3f}s")
        print(f"  numpy ensemble (5 folds): {t_numpy:.3f}s  (normalized/4min: {norm_4min:.3f}s)")
        print(f"  max abs diff (all heads): {max_abs_diff:.6f}  (target 1e-3)")
        print(f"  per-head max abs diff: {per_head}")
        print(f"  triad argmax agreement: {argmax_agree*100:.4f}%  (target >=99.9%)")

        rows.append(dict(song=song, duration_s=duration_s, t_torch=t_torch, t_numpy=t_numpy,
                          norm_4min=norm_4min, max_abs_diff=max_abs_diff,
                          argmax_agree=argmax_agree, per_head=per_head))

    print("\n\n=== SUMMARY TABLE ===")
    print(f"{'song':<32}{'dur(s)':>8}{'torch(s)':>10}{'numpy(s)':>10}{'norm4min(s)':>12}"
          f"{'maxdiff':>10}{'argmax%':>10}")
    for r in rows:
        print(f"{r['song']:<32}{r['duration_s']:>8.1f}{r['t_torch']:>10.3f}{r['t_numpy']:>10.3f}"
              f"{r['norm_4min']:>12.3f}{r['max_abs_diff']:>10.6f}{r['argmax_agree']*100:>10.4f}")

    out = MODELS_DIR / "parity_musx_numpy_result.npz"
    np.savez(out, rows=rows)
    print("\nsaved raw ->", out)


if __name__ == "__main__":
    main()
