"""exp_leakage_robust_realaudio.py — REAL-AUDIO transfer check for the
leakage-robust chord-identity screen (companion to
exp_leakage_robust_identity.py).

The MIDI screen injects SYNTHETIC contamination. This asks the transfer
question with ZERO synthetic contamination: on POP909 chroma derived from the
REAL rendered audio (Mauch/Dixon NNLS `bothchroma`, which carries natural
sustain / reverb / harmonic bleed), does the directional-delta feature (V2)
still beat mean-pool (V0) at short pooling windows near the boundary?

Reuses the cached NNLS features already on disk (data/cache/nnls_infer/*.npz,
keyed by song stem) — NO audio rendering, NO disk cost. GT chord boundaries and
labels come from the POP909 parser (functional root; /bass discarded).

Same three variants and the same cosine-kNN screen as the MIDI experiment,
imported directly from exp_leakage_robust_identity. The only change is the
front-end: real NNLS bothchroma frames (index0=A -> rolled to C-first) instead
of a MIDI piano-roll, and no `contaminate()` call.

Usage:
    .venv/bin/python scripts/exp_leakage_robust_realaudio.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import exp_leakage_robust_identity as E  # noqa: E402

CACHE = REPO / "data" / "cache" / "nnls_infer"
_ROLL_TO_C = 9  # NNLS index0=A -> roll by 9 puts C at index 0 (nnls_features)


def load_real_frames(stem: str):
    """Cached NNLS bothchroma -> (frames (T,24)=[bassC|trebC], times (T,))."""
    z = np.load(CACHE / f"{stem}.npz")
    arr, times = z["arr"].astype(np.float32), z["times"]
    bass = np.roll(arr[:, :12], _ROLL_TO_C, axis=1)
    treb = np.roll(arr[:, 12:], _ROLL_TO_C, axis=1)
    return np.concatenate([bass, treb], axis=1), times


def main():
    parser = E.POP909Parser(E.POP909_DIR)
    stems = sorted(p.stem for p in CACHE.glob("*.npz")
                   if re.fullmatch(r"\d{3}", p.stem))
    songs, frames = [], {}
    for st in stems:
        sg = parser.parse_song(st)
        if sg is None or len(sg.chord_events) < 5:
            continue
        f, t = load_real_frames(st)
        songs.append(sg)
        frames[st] = (f, t)
    print(f"Real-audio transfer: {len(songs)} POP909 songs "
          f"(NNLS bothchroma, ~{1/np.median(np.diff(frames[songs[0].song_id][1])):.0f} Hz)")

    windows = [0.1, 0.2, 0.4, 1.0]
    k, seeds = 25, 5
    sids = [s.song_id for s in songs]

    # build records once (no synthetic contamination; real audio is the leak)
    recs = []
    for sg in songs:
        f, t = frames[sg.song_id]
        med = np.median(f.sum(1)) + 1e-9
        recs += E.extract_records(sg, f, t, med)
    print(f"  {len(recs)} chord transitions.")

    def knn(Xtr, ytr, Xte, yte, nc):
        return E.knn_acc(Xtr, ytr, Xte, yte, k, nc)

    accI = {"v0": np.zeros((len(windows), seeds)),
            "v1": np.zeros((len(windows), seeds)),
            "v2": np.zeros(seeds)}
    accQ = {"v0": np.zeros((len(windows), seeds)),
            "v1": np.zeros((len(windows), seeds)),
            "v2": np.zeros(seeds)}

    for seed in range(seeds):
        rs = np.random.default_rng(seed)
        perm = list(sids); rs.shuffle(perm)
        n_test = max(1, int(round(0.25 * len(perm))))
        test = set(perm[:n_test])
        tr = [r for r in recs if r.song not in test]
        te = [r for r in recs if r.song in test]
        yI_tr = np.array([r.interval for r in tr]); yI_te = np.array([r.interval for r in te])
        yQ_tr = np.array([r.quality for r in tr]); yQ_te = np.array([r.quality for r in te])
        nQ = len(E.QUALITIES)

        X2tr = np.array([r.v2 for r in tr], np.float32)
        X2te = np.array([r.v2 for r in te], np.float32)
        accI["v2"][seed] = knn(X2tr, yI_tr, X2te, yI_te, 12)
        accQ["v2"][seed] = knn(X2tr, yQ_tr, X2te, yQ_te, nQ)

        for wi, W in enumerate(windows):
            for name, fn in (("v0", E.feat_v0), ("v1", E.feat_v1)):
                Xtr, it = [], []
                for j, r in enumerate(tr):
                    f, t = frames[r.song]; fv = fn(r, f, t, W)
                    if fv is not None:
                        Xtr.append(fv); it.append(j)
                Xte, ie = [], []
                for j, r in enumerate(te):
                    f, t = frames[r.song]; fv = fn(r, f, t, W)
                    if fv is not None:
                        Xte.append(fv); ie.append(j)
                accI[name][wi, seed] = knn(np.array(Xtr, np.float32), yI_tr[it],
                                           np.array(Xte, np.float32), yI_te[ie], 12)
                accQ[name][wi, seed] = knn(np.array(Xtr, np.float32), yQ_tr[it],
                                           np.array(Xte, np.float32), yQ_te[ie], nQ)

    # V2 delta-window sweep: is there ANY window where the real-audio delta
    # beats mean-pool? (small window = its whole premise; large = not a delta)
    v2_wins = [0.10, 0.15, 0.25, 0.40, 0.60, 1.00]
    I_v2_bywin = np.zeros((len(v2_wins), seeds))
    for wi, dw in enumerate(v2_wins):
        rec_dw = []
        for sg in songs:
            f, t = frames[sg.song_id]
            med = np.median(f.sum(1)) + 1e-9
            rec_dw += E.extract_records(sg, f, t, med, win=dw)
        for seed in range(seeds):
            rs = np.random.default_rng(seed)
            perm = list(sids); rs.shuffle(perm)
            n_test = max(1, int(round(0.25 * len(perm))))
            test = set(perm[:n_test])
            tr = [r for r in rec_dw if r.song not in test]
            te = [r for r in rec_dw if r.song in test]
            yI_tr = np.array([r.interval for r in tr]); yI_te = np.array([r.interval for r in te])
            X2tr = np.array([r.v2 for r in tr], np.float32)
            X2te = np.array([r.v2 for r in te], np.float32)
            I_v2_bywin[wi, seed] = knn(X2tr, yI_tr, X2te, yI_te, 12)

    print("\n=== REAL-AUDIO next-chord INTERVAL acc (mean +/- std over "
          f"{seeds} song-splits) ===")
    print("window:  " + "   ".join(f"{w:.2f}" for w in windows))
    for name in ("v0", "v1"):
        mu, sd = accI[name].mean(1), accI[name].std(1)
        print(f"  {name}:   " + "  ".join(f"{m:.3f}" for m in mu))
    print(f"  v2 (window-free): {accI['v2'].mean():.3f} +/- {accI['v2'].std():.3f}")
    print("\n=== REAL-AUDIO QUALITY acc ===")
    for name in ("v0", "v1"):
        print(f"  {name}:   " + "  ".join(f"{m:.3f}" for m in accQ[name].mean(1)))
    print(f"  v2: {accQ['v2'].mean():.3f}")

    print("\n=== REAL-AUDIO V2 delta-window sweep (interval acc) ===")
    print("  win:  " + "  ".join(f"{w:.2f}" for w in v2_wins))
    print("  v2:   " + "  ".join(f"{m:.3f}" for m in I_v2_bywin.mean(1)))

    out = E.OUT_DIR / "leakage_robust_realaudio.npz"
    np.savez(out, windows=np.array(windows),
             I_v0=accI["v0"], I_v1=accI["v1"], I_v2=accI["v2"],
             Q_v0=accQ["v0"], Q_v1=accQ["v1"], Q_v2=accQ["v2"],
             v2_wins=np.array(v2_wins), I_v2_bywin=I_v2_bywin,
             n_songs=np.array([len(songs)]), n_transitions=np.array([len(recs)]))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
