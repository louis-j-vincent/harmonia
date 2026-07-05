"""EM mechanism test: does certainty-weighted agreement tell TRUE structure from FALSE?

The heart of the joint structure↔chord loop (docs/joint_structure_chord_design_
2026-07-05.md): a structure hypothesis is confirmed if the slots it groups agree
(weighted by certainty) and refuted if they confidently disagree. This tests that
discriminator directly, independent of repeat variation:

  For each song, take the audio model's per-chord FAMILY distribution + certainty.
  - TRUE grouping   : slots grouped by (section-label, position-in-section) — the
                      real repeat structure (ground truth).
  - RANDOM grouping : same group sizes, random membership (a wrong hypothesis).
  Score each grouping by certainty-weighted within-group agreement. If TRUE ≫ RANDOM,
  the loop can confirm real structure and refute wrong hypotheses. Also compares
  certainty-weighted vs uniform to show certainty sharpens the discrimination.

Usage: .venv/bin/python scripts/experiment_em_structure.py
"""

from __future__ import annotations

import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from analyze_accomp_emission import parse_chord, song_chord_spans  # noqa: E402
from analyze_accomp_priors import parse_key  # noqa: E402
from build_audio_chord_features import (BUCKET_FAMILY, FAM_IDX, full_chroma,  # noqa: E402
                                        reg_chroma)
from learn_stage1_mapping import pool_beats  # noqa: E402
from harmonia.models.stage1_pitch import PitchExtractor  # noqa: E402

DB = REPO / "data" / "accomp_db" / "db.jsonl"
MANIFEST = REPO / "data" / "accomp_db" / "audio_hard" / "manifest_hard_varied.jsonl"
CLEAN_FEAT = REPO / "data" / "cache" / "audio_chord_features.npz"


def section_pos(labels):
    sec, start = {}, {}
    i = 0
    while i < len(labels):
        j = i
        while j < len(labels) and labels[j] == labels[i]:
            j += 1
        for b in range(i, j):
            sec[b], start[b] = labels[i], i
        i = j
    return sec, start


def agreement(dists, certs, weighted):
    """Within-group agreement = concentration of the (weighted) mean distribution."""
    w = certs / (certs.sum() + 1e-9) if weighted else np.ones(len(certs)) / len(certs)
    m = (dists * w[:, None]).sum(0)
    return float(m.max())          # 1.0 = all agree on one family


def main():
    d = np.load(CLEAN_FEAT, allow_pickle=True)
    Xc = np.hstack([d["onset"], d["note"], d["bass"], d["treble"]])
    sc = StandardScaler().fit(Xc)
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(Xc), d["family"].astype(int))
    nc = int(d["family"].max()) + 1
    ex = PitchExtractor(cache_dir=REPO / "data" / "cache" / "accomp_varied")
    records = {r["song_id"]: r for r in map(json.loads, open(DB))}
    rng = np.random.default_rng(0)

    true_w, true_u, rand_w, share_true, share_rand = [], [], [], [], []
    for m in map(json.loads, open(MANIFEST)):
        wav = REPO / m["wav"]
        if not wav.exists():
            continue
        rec = records[m["song_id"]]
        if parse_key(rec["key"]) is None or len(set(rec["section_per_bar"])) < 2:
            continue
        spb = 60.0 / m["tempo"]; bpb = m["beats_per_bar"]; nb = m["n_bars"] * bpb
        try:
            acts = ex.extract(wav)
        except Exception:
            continue
        onset = pool_beats(acts.frame_times, acts.onset_probs, nb, spb)
        note = pool_beats(acts.frame_times, acts.note_probs, nb, spb)
        sec, start = section_pos(rec["section_per_bar"])
        chord_at = {(ev["bar"] - 1) * bpb + ev["beat"]: ev["mma"] for ev in rec["chord_timeline"]}
        feats, gtf, slots = [], [], []
        for t0, t1, root, _q in song_chord_spans(rec):
            b0, b1 = int(round(t0 / spb)), min(int(round(t1 / spb)), nb)
            mma = chord_at.get(b0)
            p = parse_chord(mma) if mma else None
            if p is None or p[1] not in BUCKET_FAMILY or b1 <= b0:
                continue
            rt = root % 12
            rr = lambda c: np.roll(c, -rt)
            on_c = rr(full_chroma(onset[b0:b1].sum(axis=0)))
            if on_c.sum() < 1e-9:
                continue
            feats.append(np.hstack([on_c, rr(full_chroma(note[b0:b1].sum(axis=0))),
                                    rr(reg_chroma(onset[b0:b1], 0, 52)),
                                    rr(reg_chroma(onset[b0:b1], 60, 200))]))
            gtf.append(FAM_IDX[BUCKET_FAMILY[p[1]]])
            bar = b0 // bpb
            slots.append((sec.get(bar, "?"), b0 - start.get(bar, bar) * bpb))
        if len(feats) < 8:
            continue
        P = np.full((len(feats), nc), 1e-9)
        P[:, clf.classes_] = clf.predict_proba(sc.transform(np.stack(feats)))
        P /= P.sum(1, keepdims=True)
        cert = P.max(1)
        gtf = np.array(gtf)

        # TRUE grouping by (section,pos)
        groups = defaultdict(list)
        for i, s in enumerate(slots):
            groups[s].append(i)
        groups = [g for g in groups.values() if len(g) >= 2]
        if not groups:
            continue
        tw = np.mean([agreement(P[g], cert[g], True) for g in groups])
        tu = np.mean([agreement(P[g], cert[g], False) for g in groups])
        sh_t = np.mean([len(set(gtf[g].tolist())) == 1 for g in groups])
        # RANDOM grouping with the same group sizes
        perm = rng.permutation(len(feats))
        rgroups, off = [], 0
        for g in groups:
            rgroups.append(perm[off:off + len(g)]); off += len(g)
        rw = np.mean([agreement(P[g], cert[g], True) for g in rgroups])
        sh_r = np.mean([len(set(gtf[g].tolist())) == 1 for g in rgroups])

        true_w.append(tw); true_u.append(tu); rand_w.append(rw)
        share_true.append(sh_t); share_rand.append(sh_r)

    print(f"{len(true_w)} multi-section varied-jazz songs\n")
    print("Certainty-weighted within-group AGREEMENT (1.0 = all repeats agree):")
    print(f"    TRUE structure (section,pos) : {np.mean(true_w):.3f}")
    print(f"    RANDOM grouping (wrong hyp.) : {np.mean(rand_w):.3f}"
          f"   → margin {np.mean(true_w)-np.mean(rand_w):+.3f}")
    print(f"    TRUE, uniform (no certainty) : {np.mean(true_u):.3f}"
          f"   (certainty adds {np.mean(true_w)-np.mean(true_u):+.3f})")
    print("\nDo the grouped slots ACTUALLY share the GT family? (validates agreement):")
    print(f"    TRUE grouping   : {np.mean(share_true):.1%} of groups are one family")
    print(f"    RANDOM grouping : {np.mean(share_rand):.1%}")
    frac = np.mean(np.array(true_w) > np.array(rand_w))
    print(f"\nTRUE beats RANDOM on {frac:.0%} of songs → the certainty-weighted "
          "agreement\ncan confirm real structure and refute wrong hypotheses "
          "(the EM discriminator works).")


if __name__ == "__main__":
    main()
