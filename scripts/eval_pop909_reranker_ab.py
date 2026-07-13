"""POP909 cross-check for issue #25: progression reranker ON vs OFF, real path.

The jazz1460 real-path 2x2 (scripts/eval_two_pass_801d.py, 2026-07-13) showed
the #21 progression reranker's default-ON REVERSES on the production path
(−3.6pp majmin on 684d). Before flipping `use_progression_prior` to False,
CLAUDE.md rule #6 requires checking the other corpus: this runs the real
`infer_chords_v1` on the 5 rendered POP909 songs (v005_musescoregeneral,
the current default renders) with the reranker ON vs OFF and reports MIREX
root/majmin/7ths per arm.

Usage: .venv/bin/python scripts/eval_pop909_reranker_ab.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from harmonia.data.pop909_parser import POP909Parser  # noqa: E402
from harmonia.eval.mirex_eval import evaluate_song  # noqa: E402
from harmonia.models.chord_pipeline_v1 import infer_chords_v1  # noqa: E402

DATA_ROOT = REPO / "data"
SONGS = ["001", "002", "003", "004", "005"]


def main() -> None:
    parser = POP909Parser(DATA_ROOT / "pop909" / "POP909")
    agg = {arm: {"root": [], "majmin": [], "7ths": []} for arm in ("on", "off")}
    for song_id in SONGS:
        wav = (DATA_ROOT / "renders" / "pop909" / song_id
               / f"{song_id}_v005_musescoregeneral.wav")
        if not wav.exists():
            print(f"  {song_id}: render missing, skipped")
            continue
        gt = parser.parse_song(song_id)
        ref_int = np.array([[ev.start_beat, ev.end_beat] for ev in gt.chord_events])
        ref_lab = [ev.label for ev in gt.chord_events]
        for arm, flag in (("on", True), ("off", False)):
            chart = infer_chords_v1(
                wav, cache_dir=DATA_ROOT / "cache",
                use_progression_prior=flag,
            )
            sco = evaluate_song(chart.chords, ref_int, ref_lab)
            agg[arm]["root"].append(sco.root)
            agg[arm]["majmin"].append(sco.majmin)
            agg[arm]["7ths"].append(sco.sevenths)
            print(f"  {song_id} reranker={arm:>3}: root={sco.root:.1%} "
                  f"majmin={sco.majmin:.1%} 7ths={sco.sevenths:.1%}", flush=True)

    print("\n=== POP909 (5 songs, v005_musescoregeneral) — reranker A/B ===")
    print(f"{'arm':<5} {'root':>7} {'majmin':>7} {'7ths':>7} {'n':>3}")
    for arm in ("off", "on"):
        a = agg[arm]
        print(f"{arm:<5} {np.mean(a['root']):>7.1%} {np.mean(a['majmin']):>7.1%} "
              f"{np.mean(a['7ths']):>7.1%} {len(a['root']):>3}")


if __name__ == "__main__":
    main()
