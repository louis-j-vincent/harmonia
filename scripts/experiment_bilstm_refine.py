"""BiLSTM refinement: feed the SOFT audio distributions of surrounding chords
(before + after) and output a refined chord + a calibrated certainty.

User's idea (2026-07-05): don't feed the LSTM hard detected chords — feed the
probability table for each chord, use both past and future context (a chord is
constrained by what follows), use a neutral element where there is no
neighbour (or it's masked), and output a certainty for the inferred chord.

Design:
  input per position = [ audio quality distribution (14, soft, out-of-fold),
                         scale-degree one-hot (12) ]        → 26-d
  a bidirectional LSTM reads the whole sequence (subsumes "4 before + after")
  → per-position logits → softmax = a normalised distribution over qualities.
  certainty = max softmax prob. Neutral element = a zero vector (used at
  boundaries and for random masking during training, so the model is robust to
  a missing/uncertain neighbour).

No leakage: audio distributions are out-of-fold; the BiLSTM is evaluated by
song-grouped CV. Compares audio-alone vs refined, and reports certainty
calibration.

Usage: .venv/bin/python scripts/experiment_bilstm_refine.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from analyze_accomp_emission import parse_chord  # noqa: E402
from analyze_accomp_priors import merged_events, parse_key  # noqa: E402
from build_audio_chord_features import BUCKET_BASE7, BUCKET_FAMILY, full_chroma, reg_chroma  # noqa: E402
from learn_stage1_mapping import pool_beats  # noqa: E402
from harmonia.models.stage1_pitch import PitchExtractor  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)
DB = REPO / "data" / "accomp_db" / "db.jsonl"
MANIFEST = REPO / "data" / "accomp_db" / "audio" / "manifest.jsonl"
B7 = sorted(set(BUCKET_BASE7.values()))
B7I = {b: i for i, b in enumerate(B7)}
FAM = ["major", "minor", "diminished", "augmented", "suspended"]
FAMI = {f: i for i, f in enumerate(FAM)}
B7_FAM = np.array([FAMI[BUCKET_FAMILY[b]] for b in BUCKET_FAMILY
                   for bb in [BUCKET_BASE7[b]] if False])  # placeholder
# base7 label -> family index
_b7_to_fam = {}
for bucket, fam in BUCKET_FAMILY.items():
    _b7_to_fam[BUCKET_BASE7[bucket]] = FAMI[fam]
B7_FAM = np.array([_b7_to_fam[b] for b in B7])


def build_song_sequences():
    """Per rendered song: ordered lists of (audio_feat_48, degree, base7_idx)."""
    records = {r["song_id"]: r for r in map(json.loads, open(DB))}
    ex = PitchExtractor(cache_dir=REPO / "data" / "cache" / "accomp")
    songs = []
    for m in map(json.loads, open(MANIFEST)):
        wav = REPO / m["wav"]
        if not wav.exists():
            continue
        rec = records[m["song_id"]]
        k = parse_key(rec["key"])
        if k is None:
            continue
        tonic = k[0]
        spb = 60.0 / m["tempo"]; bpb = m["beats_per_bar"]; nb = m["n_bars"] * bpb
        try:
            acts = ex.extract(wav)
        except Exception:
            continue
        onset = pool_beats(acts.frame_times, acts.onset_probs, nb, spb)
        note = pool_beats(acts.frame_times, acts.note_probs, nb, spb)
        feats, degs, labs = [], [], []
        for chord, sb, dur in merged_events(rec):
            p = parse_chord(chord)
            b0, b1 = sb, min(sb + dur, nb)
            if p is None or p[1] not in BUCKET_FAMILY or b1 <= b0:
                continue
            root_t = (p[0] + m["transpose"]) % 12
            rr = lambda c: np.roll(c, -root_t)
            on_c = rr(full_chroma(onset[b0:b1].sum(axis=0)))
            if on_c.sum() < 1e-9:
                continue
            feats.append(np.hstack([on_c, rr(full_chroma(note[b0:b1].sum(axis=0))),
                                    rr(reg_chroma(onset[b0:b1], 0, 52)),
                                    rr(reg_chroma(onset[b0:b1], 60, 200))]))
            degs.append((p[0] - tonic) % 12)
            labs.append(B7I[BUCKET_BASE7[p[1]]])
        if len(feats) >= 4:
            songs.append({"id": m["song_id"], "wav": m["wav"],
                          "feat": np.array(feats), "deg": np.array(degs),
                          "lab": np.array(labs)})
    return songs


class BiLSTMRefine(nn.Module):
    def __init__(self, in_dim=26, hid=64, n_out=14):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hid, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(2 * hid, 64), nn.ReLU(), nn.Linear(64, n_out))

    def forward(self, x):
        return self.head(self.lstm(x)[0])


def make_input(audio_dist, deg, mask_p=0.0):
    """(L,26): soft audio quality dist + degree one-hot, with optional masking."""
    L = len(deg)
    deg_oh = np.zeros((L, 12))
    deg_oh[np.arange(L), deg] = 1.0
    ad = audio_dist.copy()
    if mask_p > 0:
        m = np.random.rand(L) < mask_p
        ad[m] = 0.0            # neutral element where masked
        deg_oh[m] = 0.0
    return np.hstack([ad, deg_oh]).astype(np.float32)


def main():
    songs = build_song_sequences()
    ids = np.array([s["id"] for s in songs])
    print(f"{len(songs)} rendered songs, {sum(len(s['lab']) for s in songs)} chords\n")

    # ── out-of-fold audio quality distributions (LR, 5-fold by song) ──────────
    allfeat = np.vstack([s["feat"] for s in songs])
    alllab = np.concatenate([s["lab"] for s in songs])
    allgrp = np.concatenate([[s["id"]] * len(s["lab"]) for s in songs])
    oof = np.zeros((len(alllab), len(B7)))
    gkf = GroupKFold(5)
    for tr, te in gkf.split(allfeat, alllab, allgrp):
        sc = StandardScaler().fit(allfeat[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(allfeat[tr]), alllab[tr])
        p = np.full((len(te), len(B7)), 1e-6)
        p[:, clf.classes_] = clf.predict_proba(sc.transform(allfeat[te]))
        oof[te] = p / p.sum(1, keepdims=True)
    # scatter OOF back per song
    off = 0
    for s in songs:
        s["audio"] = oof[off:off + len(s["lab"])]
        off += len(s["lab"])

    # ── BiLSTM refinement, song-grouped CV ────────────────────────────────────
    conf_all, correct_ref, correct_aud, fam_ref, fam_aud = [], [], [], [], []
    gkf2 = GroupKFold(5)
    song_idx = np.arange(len(songs))
    grp = ids
    for tr, te in gkf2.split(song_idx, song_idx, grp):
        net = BiLSTMRefine()
        opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
        lossf = nn.CrossEntropyLoss()
        net.train()
        for _ in range(25):
            for i in np.random.permutation(tr):
                s = songs[i]
                x = torch.tensor(make_input(s["audio"], s["deg"], mask_p=0.15))[None]
                y = torch.tensor(s["lab"])
                opt.zero_grad()
                loss = lossf(net(x)[0], y)
                loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            for i in te:
                s = songs[i]
                x = torch.tensor(make_input(s["audio"], s["deg"]))[None]
                probs = torch.softmax(net(x)[0], -1).numpy()
                ref = probs.argmax(1)
                aud = s["audio"].argmax(1)
                conf_all.append(probs.max(1))
                correct_ref.append(ref == s["lab"])
                correct_aud.append(aud == s["lab"])
                fam_ref.append(B7_FAM[ref] == B7_FAM[s["lab"]])
                fam_aud.append(B7_FAM[aud] == B7_FAM[s["lab"]])

    conf = np.concatenate(conf_all)
    cref = np.concatenate(correct_ref); caud = np.concatenate(correct_aud)
    fref = np.concatenate(fam_ref); faud = np.concatenate(fam_aud)
    print("Seventh-level accuracy:")
    print(f"    audio alone (argmax of soft dist) : {caud.mean():.1%}")
    print(f"    BiLSTM refined (before+after)     : {cref.mean():.1%}   (Δ {cref.mean()-caud.mean():+.1%})")
    print("Family-level accuracy:")
    print(f"    audio alone : {faud.mean():.1%}")
    print(f"    BiLSTM      : {fref.mean():.1%}   (Δ {fref.mean()-faud.mean():+.1%})")

    # ── certainty calibration + coverage/accuracy tradeoff ────────────────────
    print("\nCertainty calibration (is a high-confidence guess actually right?):")
    print(f"    {'confidence bin':<18}{'n':>7}{'accuracy':>11}")
    for lo, hi in [(0.0, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 0.95), (0.95, 1.01)]:
        m = (conf >= lo) & (conf < hi)
        if m.sum():
            print(f"    {f'{lo:.2f}-{hi:.2f}':<18}{m.sum():>7}{cref[m].mean():>11.1%}")
    print("\nAnswer-only-if-confident (report exact seventh only above a threshold,")
    print("else back off to family — how the certainty drives the tree reporting):")
    print(f"    {'threshold':<12}{'coverage':>10}{'acc@covered':>13}")
    for th in (0.0, 0.5, 0.7, 0.85):
        m = conf >= th
        print(f"    {th:<12.2f}{m.mean():>10.1%}{cref[m].mean() if m.sum() else 0:>13.1%}")


if __name__ == "__main__":
    main()
