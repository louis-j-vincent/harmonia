"""Build clipped-pooled vs beat-grid JAAH feature subsets for the boundary-bleed
audit. Uses only confidently-paired songs (chroma-fit assignment margin>0.05,
best>0.4) PLUS a clipped-feature root-agreement guard, so we never stake numbers
on an ambiguous (acts<->slug) pairing (CLAUDE.md rule 1).

Beat-grid features come verbatim from the shipped jaah_bp48.npz (the buggy
whole-beat-snap pooling). Clipped features re-pool frames exactly in [t0,t1)
from the retained bp_cache activations (zero bleed, no beat grid, no re-download).
"""
from __future__ import annotations
import sys, json, pickle
from pathlib import Path
import numpy as np

REPO = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from harmonia.data.yt_chord_corpus import _reg_raw, _norm_blocks, QUALITY_IDX, QUALITIES
from harmonia.data.corpus_schema import load_corpus, save_corpus
from scripts.build_jaah_corpus import load_lab, parse_jaah

CACHE = REPO / "data/cache/jaah"


def clipped_feats(ft, onset_f, note_f, t0, t1, root):
    m = (ft >= t0) & (ft < t1)
    if not m.any():
        return None, None
    seg_on = onset_f[m].sum(0); seg_nt = note_f[m].sum(0)
    ch_on = _reg_raw(seg_on); ch_nt = _reg_raw(seg_nt)
    bs = _reg_raw(seg_on, 0, 52); tr = _reg_raw(seg_on, 60, 200)
    rr = lambda c: np.roll(c, -root)
    f48 = _norm_blocks(np.hstack([rr(ch_on), rr(ch_nt), rr(bs), rr(tr)])).astype(np.float32)
    f48a = _norm_blocks(np.hstack([ch_on, ch_nt, bs, tr])).astype(np.float32)
    return f48, f48a


def main():
    pr = pickle.load(open(REPO / "scratchpad/jaah_pairing.pkl", "rb"))
    M = np.load(REPO / "scratchpad/jaah_pair_M.npy")
    slugs, hashes = pr["slugs"], pr["hashes"]
    ri, ci = pr["ri"], pr["ci"]
    # confident: best>0.4 and margin>0.05
    conf = {}
    for i, j in zip(ri, ci):
        best = M[i, j]; second = np.partition(M[i], -2)[-2]
        if best > 0.4 and (best - second) > 0.05:
            conf[slugs[i]] = hashes[j]
    print(f"{len(conf)} confidently-paired songs")

    d = load_corpus(CACHE / "jaah_bp48.npz")
    sid = d["song_id"]

    acts_cache = {}
    def get_acts(h):
        if h not in acts_cache:
            z = np.load((CACHE / "bp_cache" / f"{h}.npz"))
            acts_cache[h] = (z["frame_times"], z["onset_probs"], z["note_probs"])
        return acts_cache[h]

    rows_out = {"clip": [], "grid": []}
    kept_songs = []
    for slug, h in conf.items():
        song_id = f"jaah_{slug}"
        idx = np.where(sid == song_id)[0]
        if len(idx) == 0:
            continue
        ft, onf, ntf = get_acts(h)
        # recompute clipped for each record; guard on root agreement
        clip_recs = []; grid_recs = []; agree = 0; tot = 0
        for i in idx:
            t0 = float(d["t0"][i]); t1 = float(d["t1"][i]); root = int(d["root"][i])
            f48, f48a = clipped_feats(ft, onf, ntf, t0, t1, root)
            if f48 is None:
                continue
            # root agreement of clipped abs feature: argmax over first 12-dim block
            pred_root = int(np.argmax(f48a[:12]))
            agree += (pred_root == root); tot += 1
            clip_recs.append((f48, f48a, i))
            grid_recs.append((d["feat48"][i], d["feat48_abs"][i], i))
        if tot == 0:
            continue
        root_agree = agree / tot
        if root_agree < 0.30:   # guard: suspect pairing -> drop
            print(f"  DROP {slug}: clipped root-agree {root_agree:.2f} < 0.30 (suspect pairing)")
            continue
        kept_songs.append((slug, root_agree, tot))
        for (f48, f48a, i) in clip_recs:
            rows_out["clip"].append((f48, f48a, i))
        for (f48, f48a, i) in grid_recs:
            rows_out["grid"].append((f48, f48a, i))

    print(f"\n{len(kept_songs)} songs pass root-agree guard")
    for s, a, n in sorted(kept_songs, key=lambda x: -x[1]):
        print(f"  {s:30s} clipped-root-agree={a:.2f}  n={n}")

    for tag in ("clip", "grid"):
        recs = rows_out[tag]
        idxs = np.array([r[2] for r in recs])
        out = {
            "feat48": np.stack([r[0] for r in recs]).astype(np.float32),
            "feat48_abs": np.stack([r[1] for r in recs]).astype(np.float32),
            "root": d["root"][idxs].astype(np.int32),
            "quality_idx": d["quality_idx"][idxs].astype(np.int32),
            "quality": d["quality"][idxs],
            "labels": d["labels"][idxs],
            "match": d["match"][idxs],
            "t0": d["t0"][idxs], "t1": d["t1"][idxs],
            "song_id": d["song_id"][idxs],
            "qualities": np.array(QUALITIES),
        }
        outp = REPO / f"scratchpad/jaah_subset_{tag}.npz"
        save_corpus(outp, **out)
        print(f"wrote {outp.name}: {len(idxs)} records, {len(set(out['song_id'].tolist()))} songs")


if __name__ == "__main__":
    main()
