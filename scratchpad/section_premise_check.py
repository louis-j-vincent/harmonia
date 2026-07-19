"""section_premise_check.py — 2026-07-18, section-level suggestion tool,
task 1 (premise check, cheapest possible falsification test, per CLAUDE.md
rule #2 / research-loop doctrine).

User's worked example: on Autumn Leaves (AABC form), the A section repeats,
so "bar 2" and "bar 10" (their own 1-indexed framing) should be flagged as
"the same" 8-bar-aligned position across the two A occurrences.

Verify against data/ireal/jazz1460.txt "Autumn Leaves" (key G-):
  bar1=Cm7 bar2=F7 ... bar8=Gm6 | bar9=Cm7 bar10=F7 ... bar15=Gm6 | bar16=B section starts
So bars 1-8 = A occurrence 1, bars 9-15 = A occurrence 2 (only 7 bars — the
chart's two A's are NOT both exactly 8 bars; occurrence 2 is missing its
final bar, section B starts one bar early at 16). 1-indexed bar2/bar10 =
0-indexed bar index 1 and 9. nuclear_spans(n, 8) gives blocks [0,8),[8,16)...
so index1 lands in block0 at within-block position 1, index9 lands in
block1 at within-block position 1 — SAME relative position, confirming the
user's own indexing intuition once translated to 0-indexed blocks. This
script computes the grain=8 block0-vs-block1 similarity two ways:
  (a) SYMBOLIC — iReal chart bass/treble proxy vectors (same feature as
      tau_auto_search.py), block_sim via hierarchy_shortcut's derive_grain
      logic (reused directly, not reimplemented).
  (b) REAL AUDIO — rawchroma.py bt_concat/bass_only/treble_only on
      docs/audio/autumn_leaves.m4a, same derive_grain machinery.
If both come back high-confidence, the premise holds and grain=8 is a
reasonable starting granularity for the section-suggestion tool. If audio
disagrees with symbolic, that's the real-audio transfer risk flagged in the
brief and must be dealt with (joint gate) before shipping audio-only.
"""
from __future__ import annotations
import sys, json, io
from pathlib import Path
from contextlib import redirect_stdout
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np

from hierarchy_shortcut import diagonal_prefix_sums, diag_sum
from chord_distance_eval import nuclear_spans
from rawchroma import per_bar_rawchroma
from harmonia.data.ireal_corpus import load_playlist, tune_to_mma, chord_root_pc
from noise_calibrate import root_onehot
from chord_distance import chord_vector_binary
from symstruct import qbucket

REPO = Path(__file__).resolve().parent.parent


def block_sim_from_gram(G, sq, i0, j0, L):
    prefix = diagonal_prefix_sums(G)
    n = G.shape[0]
    d = j0 - i0
    num = diag_sum(prefix, n, d, i0, L)
    na = np.sqrt(float(np.sum(sq[i0:i0 + L])))
    nb = np.sqrt(float(np.sum(sq[j0:j0 + L])))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(num / (na * nb))


def symbolic_check():
    buf = io.StringIO()
    with redirect_stdout(buf):
        tunes = load_playlist(Path("data/ireal/jazz1460.txt"))
    tune = next(t for t in tunes if t.title.lower() == "autumn leaves")
    mc = tune_to_mma(tune)
    shift = 0
    if mc.key:
        pc = chord_root_pc(mc.key.rstrip("-"))
        shift = (-pc % 12) if pc is not None else 0
    bass_vecs, treb_vecs, labels = [], [], []
    for bar_no, section, slots in mc.timeline:
        bass_accum, treb_accum = None, None
        lab = []
        for (_, _, mma) in slots:
            pc = chord_root_pc(mma)
            if pc is None:
                continue
            rpc = (pc + shift) % 12
            q = qbucket(mma)
            bv = root_onehot(rpc, q)
            tv = chord_vector_binary(rpc, q)
            bass_accum = bv if bass_accum is None else bass_accum + bv
            treb_accum = tv if treb_accum is None else treb_accum + tv
            lab.append(mma)
        bass_vecs.append(bass_accum if bass_accum is not None else np.zeros(12))
        treb_vecs.append(treb_accum if treb_accum is not None else np.zeros(12))
        labels.append((bar_no, section, lab))

    def _l2(vecs):
        out = []
        for v in vecs:
            n = np.linalg.norm(v)
            out.append(v / n if n > 1e-9 else v)
        return np.array(out)

    bass = _l2(bass_vecs)
    treb = _l2(treb_vecs)
    n = len(bass)
    print("Autumn Leaves (iReal): n_bars=%d, key=%s, shift=%d" % (n, mc.key, shift))
    for i in range(16):
        print("  bar_idx(0)=%2d 1-indexed=%2d section=%s chord=%s" %
          (i, labels[i][0], labels[i][1], labels[i][2]))

    spans = nuclear_spans(n, 8)
    print("nuclear_spans(n=%d, size=8):" % n, spans[:4])
    # block0 = spans[0], block1 = spans[1]
    s0, e0 = spans[0]
    s1, e1 = spans[1]
    L = min(e0 - s0, e1 - s1)
    results = {}
    for reg, vecs in (("bass", bass), ("treble", treb)):
        G = vecs @ vecs.T
        sq = np.diag(G).copy()
        sim = block_sim_from_gram(G, sq, s0, s1, L)
        results[reg] = sim
        print("  [symbolic %s] block0(%d-%d) vs block1(%d-%d), L=%d -> sim=%.4f" %
              (reg, s0, e0, s1, e1, L, sim))
    avg = 0.5 * (results["bass"] + results["treble"])
    results["avg"] = avg
    print("  [symbolic avg bass/treble] sim=%.4f" % avg)
    return {"spans": spans[:4], "block0": [s0, e0], "block1": [s1, e1], "L": L,
            "sim_bass": results["bass"], "sim_treble": results["treble"], "sim_avg": avg}


def audio_check():
    audio = REPO / "docs" / "audio" / "autumn_leaves.m4a"
    variants, bar_times, tempo, tonic = per_bar_rawchroma(audio)
    print("\nReal audio autumn_leaves.m4a: tempo=%.1f n_bars=%d tonic_pc=%d" %
          (tempo, len(variants["bt_concat"]), tonic))
    n = len(variants["bt_concat"])
    spans = nuclear_spans(n, 8)
    s0, e0 = spans[0]
    s1, e1 = spans[1]
    L = min(e0 - s0, e1 - s1)
    results = {}
    for reg in ("bt_concat", "bass_only", "treble_only"):
        v = variants[reg]
        G = v @ v.T
        sq = np.diag(G).copy()
        sim = block_sim_from_gram(G, sq, s0, s1, L)
        results[reg] = sim
        print("  [audio %s] block0(%d-%d) vs block1(%d-%d), L=%d -> sim=%.4f" %
              (reg, s0, e0, s1, e1, L, sim))
    return {"n_bars": n, "block0": [s0, e0], "block1": [s1, e1], "L": L, **results}


def main():
    print("=== SYMBOLIC (iReal chart) ===")
    sym = symbolic_check()
    print("\n=== REAL AUDIO ===")
    aud = audio_check()
    out = {"symbolic": sym, "audio": aud}
    (Path(__file__).resolve().parent / "section_premise_check_results.json").write_text(
        json.dumps(out, indent=2))
    print("\nwrote section_premise_check_results.json")


if __name__ == "__main__":
    main()
