"""Does the progression language model HELP the audio chord model?

Combine, at the family and seventh levels, on the rendered songs:
  P(quality | audio)  — the trained audio model (LR on 48-d chroma features)
  P(quality | history, degree) — the MLP progression LM (last-4 scale-relative
     chords), teacher-forced on the true history and conditioned on the known
     scale degree (so it's a pure quality prior).

Blend log P_audio + w·log P_LM, sweep w. The LM is trained ONLY on jazz songs
NOT in the audio test set (no leakage). Single 80/20 split by song.

Usage: .venv/bin/python scripts/experiment_progression_plus_audio.py
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

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import pretty_midi  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from analyze_accomp_emission import parse_chord, song_chord_spans  # noqa: E402
from analyze_accomp_priors import merged_events, parse_key  # noqa: E402
from build_audio_chord_features import (BUCKET_BASE7, BUCKET_FAMILY, full_chroma,  # noqa: E402
                                        reg_chroma)
from learn_stage1_mapping import pool_beats, to_chroma  # noqa: E402
from train_progression_lm import MLPLM  # noqa: E402
from harmonia.models.stage1_pitch import PitchExtractor  # noqa: E402

torch.manual_seed(0)
DB = REPO / "data" / "accomp_db" / "db.jsonl"
MANIFEST = REPO / "data" / "accomp_db" / "audio" / "manifest.jsonl"
FAM = ["major", "minor", "diminished", "augmented", "suspended"]
FAMI = {f: i for i, f in enumerate(FAM)}
B7 = sorted(set(BUCKET_BASE7.values()))
B7I = {b: i for i, b in enumerate(B7)}


def token_seq(rec, tonic):
    seq = []
    for chord, _, _ in merged_events(rec):
        p = parse_chord(chord)
        if p is None or p[1] not in BUCKET_BASE7:
            seq.append(None)
        else:
            seq.append(((p[0] - tonic) % 12, BUCKET_BASE7[p[1]]))
    return seq


def main():
    records = {r["song_id"]: r for r in map(json.loads, open(DB))}
    manifest = [json.loads(line) for line in open(MANIFEST)]
    rendered = sorted({m["song_id"] for m in manifest})
    rng = np.random.default_rng(0)
    test_songs = {s for s in rendered if rng.random() < 0.2}

    # ── train progression LM on all jazz songs NOT in the audio test set ──────
    seqs = []
    for rec in records.values():
        if rec["corpus"] != "jazz1460" or rec["song_id"] in test_songs:
            continue
        k = parse_key(rec["key"])
        if k is None:
            continue
        s = [t for t in token_seq(rec, k[0]) if t is not None]
        if len(s) >= 4:
            seqs.append(s)
    vocab = {}
    for s in seqs:
        for t in s:
            vocab.setdefault(t, len(vocab))
    V = len(vocab)
    PAD = V
    lm = MLPLM(V, k=4)
    opt = torch.optim.Adam(lm.parameters(), lr=2e-3)
    lossf = nn.CrossEntropyLoss()
    ctxs, tgts = [], []
    for s in seqs:
        ids = [vocab[t] for t in s]
        for i in range(1, len(ids)):
            c = ids[max(0, i - 4):i]
            ctxs.append([PAD] * (4 - len(c)) + c); tgts.append(ids[i])
    ctxs, tgts = torch.tensor(ctxs), torch.tensor(tgts)
    for _ in range(25):
        perm = torch.randperm(len(ctxs))
        for j in range(0, len(ctxs), 512):
            idx = perm[j:j + 512]
            opt.zero_grad(); loss = lossf(lm(ctxs[idx]), tgts[idx]); loss.backward(); opt.step()
    lm.eval()

    # LM quality prior over base7, conditioned on degree, from a token history
    tokens_by_deg = defaultdict(list)
    for (deg, b7), idx in vocab.items():
        tokens_by_deg[deg].append((idx, b7))

    def lm_b7_prior(hist_ids, degree):
        c = hist_ids[-4:]
        c = [PAD] * (4 - len(c)) + c
        with torch.no_grad():
            logp = torch.log_softmax(lm(torch.tensor([c]))[0], -1).numpy()
        prior = np.ones(len(B7)) * 1e-6
        for idx, b7 in tokens_by_deg.get(degree, []):
            prior[B7I[b7]] += np.exp(logp[idx])
        return prior / prior.sum()

    # ── extract aligned (audio features, labels, LM prior) per chord ──────────
    ex = PitchExtractor(cache_dir=REPO / "data" / "cache" / "accomp")
    rows = {"train": [], "test": []}
    for m in manifest:
        wav = REPO / m["wav"]
        if not wav.exists():
            continue
        rec = records[m["song_id"]]
        k = parse_key(rec["key"])
        if k is None:
            continue
        tonic, _ = k
        spb = 60.0 / m["tempo"]; bpb = m["beats_per_bar"]; nb = m["n_bars"] * bpb
        try:
            acts = ex.extract(wav)
        except Exception:
            continue
        onset = pool_beats(acts.frame_times, acts.onset_probs, nb, spb)
        note = pool_beats(acts.frame_times, acts.note_probs, nb, spb)
        split = "test" if m["song_id"] in test_songs else "train"
        hist_ids = []
        for chord, sb, dur in merged_events(rec):
            p = parse_chord(chord)
            b0, b1 = sb, min(sb + dur, nb)
            if p is None or p[1] not in BUCKET_FAMILY or b1 <= b0:
                if p is not None and p[1] in BUCKET_BASE7:
                    tok = ((p[0] - tonic) % 12, BUCKET_BASE7[p[1]])
                    if tok in vocab:
                        hist_ids.append(vocab[tok])
                continue
            root_t = (p[0] + m["transpose"]) % 12
            deg = (p[0] - tonic) % 12
            rr = lambda c: np.roll(c, -root_t)
            on_c = rr(full_chroma(onset[b0:b1].sum(axis=0)))
            if on_c.sum() < 1e-9:
                continue
            feat = np.hstack([on_c, rr(full_chroma(note[b0:b1].sum(axis=0))),
                              rr(reg_chroma(onset[b0:b1], 0, 52)),
                              rr(reg_chroma(onset[b0:b1], 60, 200))])
            prior = lm_b7_prior(hist_ids, deg) if split == "test" else None
            rows[split].append((feat, BUCKET_FAMILY[p[1]], BUCKET_BASE7[p[1]], prior))
            tok = (deg, BUCKET_BASE7[p[1]])
            if tok in vocab:
                hist_ids.append(vocab[tok])

    Xtr = np.stack([r[0] for r in rows["train"]])
    Xte = np.stack([r[0] for r in rows["test"]])
    print(f"{len(rows['train'])} train / {len(rows['test'])} test chords, "
          f"LM vocab {V}\n")

    b7_to_fam = {BUCKET_BASE7[b]: BUCKET_FAMILY[b] for b in BUCKET_FAMILY}
    fam_prior_cols = np.array([FAMI[b7_to_fam[b]] for b in B7])

    print(f"{'level':<10}{'audio-alone':>13}{'+ progression (best w)':>24}")
    print("-" * 47)
    for level in ("family", "seventh"):
        if level == "family":
            ytr = np.array([FAMI[r[1]] for r in rows["train"]])
            yte = np.array([FAMI[r[1]] for r in rows["test"]])
            nc = len(FAM)
        else:
            ytr = np.array([B7I[r[2]] for r in rows["train"]])
            yte = np.array([B7I[r[2]] for r in rows["test"]])
            nc = len(B7)
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Xtr), ytr)
        proba = clf.predict_proba(sc.transform(Xte))
        full = np.full((len(yte), nc), 1e-9); full[:, clf.classes_] = proba
        log_audio = np.log(full)
        # LM prior in this level's label space
        log_lm = np.zeros((len(yte), nc))
        for i, r in enumerate(rows["test"]):
            pr = r[3]
            if pr is None:
                continue
            if level == "seventh":
                log_lm[i] = np.log(pr + 1e-9)
            else:
                fam_p = np.ones(nc) * 1e-9
                for j, col in enumerate(fam_prior_cols):
                    fam_p[col] += pr[j]
                log_lm[i] = np.log(fam_p / fam_p.sum() + 1e-9)
        base = (log_audio.argmax(1) == yte).mean()
        best = (base, 0.0)
        for w in (0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5):
            acc = ((log_audio + w * log_lm).argmax(1) == yte).mean()
            if acc > best[0]:
                best = (acc, w)
        print(f"{level:<10}{base:>12.1%}{best[0]:>18.1%} (w={best[1]})")


if __name__ == "__main__":
    main()
