"""Fit the jazz1460 chord-duration prior from SYMBOLIC charts (no audio).

Produces {"pooled": (D,), "per_q5": (5, D)} PMFs over chord duration in beats,
cached to data/cache/duration_prior_jazz1460.npz for semi_markov_decode.

Excludes the held-out gate split (idx 70..95) to avoid any eval leakage, even
though corpus-level duration statistics are extremely stable.
"""
import json, sys
from pathlib import Path
from collections import defaultdict
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from analyze_accomp_emission import song_chord_spans
from build_audio_chord_features import BUCKET_FAMILY
from eval_two_pass_801d import MMA_TO_Q5

D = 32
Q5 = ["maj", "min", "dom", "hdim", "dim"]
GATE = range(70, 95)  # held-out eval split — excluded from the fit
OUT = REPO / "data" / "cache" / "duration_prior_jazz1460.npz"


def main():
    recs = [json.loads(l) for l in open(REPO / "data" / "accomp_db" / "db.jsonl")]
    jz = [r for r in recs if r.get("corpus") == "jazz1460"
          and r["beats_per_bar"] == 4 and (REPO / r["midi_path"]).exists()]
    pooled = np.zeros(D); per = np.zeros((5, D))
    n = 0
    for i, rec in enumerate(jz):
        if i in GATE:
            continue
        spb = 60.0 / rec["tempo"]
        for t0, t1, r, q in song_chord_spans(rec):
            if q not in BUCKET_FAMILY:
                continue
            d = int(round((t1 - t0) / spb))
            if d < 1:
                continue
            d = min(d, D)
            pooled[d - 1] += 1; n += 1
            fam = MMA_TO_Q5.get(q)
            if fam in Q5:
                per[Q5.index(fam), d - 1] += 1
    pooled_pmf = pooled / pooled.sum()
    per_pmf = per / np.clip(per.sum(1, keepdims=True), 1, None)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, pooled=pooled_pmf, per_q5=per_pmf, q5_names=np.array(Q5))
    print(f"fit {n} chord events from {len(jz)} jazz songs (excl gate) -> {OUT}")
    print("pooled d=1..8:", pooled_pmf[:8].round(3).tolist())
    for i, name in enumerate(Q5):
        print(f"  {name:5s} d=1..8:", per_pmf[i, :8].round(3).tolist())


if __name__ == "__main__":
    main()
