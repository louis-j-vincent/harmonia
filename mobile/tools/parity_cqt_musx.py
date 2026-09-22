"""P2b — parity between the numpy CQT port and the reference musx CQT,
measured the way that actually matters: argmax chord posteriors out of the
REAL musx torch nets, not just feature-space distance.

Usage: .venv/bin/python mobile/tools/parity_cqt_musx.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "mobile"))

from harmonia import musx  # noqa: E402
from harmonia_mobile import cqt_musx as cm  # noqa: E402

REF_DIR = Path(__file__).resolve().parent / "cqt_ref"
SONGS = ["maroon_5_this_love", "ben_e_king_stand_by_me_audio", "autumn_leaves"]


def triad_argmax_agreement(probs_a, probs_b) -> tuple[float, float]:
    triad_a, triad_b = probs_a[0], probs_b[0]
    n = min(len(triad_a), len(triad_b))
    argmax_a = np.argmax(triad_a[:n], axis=1)
    argmax_b = np.argmax(triad_b[:n], axis=1)
    agree = float(np.mean(argmax_a == argmax_b))
    max_abs_diff = float(np.max(np.abs(triad_a[:n] - triad_b[:n])))
    return agree, max_abs_diff


def main() -> None:
    results = []
    for stem in SONGS:
        ref_path = REF_DIR / f"{stem}.npz"
        d = np.load(ref_path)
        y = d["y"].astype(np.float64)
        ref_cqt = d["cqt"]
        tuning_ref = float(d["tuning"])

        t0 = time.time()
        mine_cqt = cm.song_cqt_288(y, tuning=tuning_ref)
        t_cqt_reftuning = time.time() - t0

        t0 = time.time()
        mine_cqt_own_tuning = cm.song_cqt_288(y, tuning=None)
        t_cqt_own = time.time() - t0
        my_tuning = None  # estimate_tuning is internal; recompute for logging
        my_tuning = cm.estimate_tuning(y, sr=cm.MUSX_SR, bins_per_octave=cm.BINS_PER_OCTAVE)

        n = min(mine_cqt.shape[0], ref_cqt.shape[0])
        feat_absdiff = float(np.max(np.abs(mine_cqt[:n] - ref_cqt[:n])))
        feat_reldiff = feat_absdiff / float(np.max(np.abs(ref_cqt[:n])))

        with musx._InMusxDir():
            probs_ref = musx.posteriors_from_cqt(ref_cqt)
            probs_mine = musx.posteriors_from_cqt(mine_cqt)
            probs_mine_owntuning = musx.posteriors_from_cqt(mine_cqt_own_tuning)

        agree, post_absdiff = triad_argmax_agreement(probs_ref, probs_mine)
        agree_ot, post_absdiff_ot = triad_argmax_agreement(probs_ref, probs_mine_owntuning)

        results.append(dict(
            song=stem, tuning_ref=tuning_ref, tuning_mine=my_tuning,
            feat_absdiff=feat_absdiff, feat_reldiff=feat_reldiff,
            argmax_agree_reftuning=agree, post_absdiff_reftuning=post_absdiff,
            argmax_agree_owntuning=agree_ot, post_absdiff_owntuning=post_absdiff_ot,
            t_cqt_reftuning=t_cqt_reftuning, t_cqt_own=t_cqt_own,
            dur_s=len(y) / cm.MUSX_SR,
        ))
        print(results[-1])

    print("\n--- summary ---")
    for r in results:
        print(
            f"{r['song']}: feat_reldiff={r['feat_reldiff']:.2e} "
            f"argmax_agree(ref tuning)={r['argmax_agree_reftuning']*100:.2f}% "
            f"argmax_agree(own tuning)={r['argmax_agree_owntuning']*100:.2f}% "
            f"post_absdiff={r['post_absdiff_reftuning']:.4f} "
            f"speed(4min-equiv)={r['t_cqt_reftuning']/r['dur_s']*240:.2f}s"
        )


if __name__ == "__main__":
    main()
