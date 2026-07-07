"""Train a family classifier on root-shifted LTAS-CQT chroma.

Features (per oracle segment, root-shifted so root → index 0):
  - 12d  : mean LTAS-normalised CQT chroma (root-shifted)
  - 5d   : max log-likelihood over 12 keys per family (from Gaussian dists)
  - 17d  : both concatenated (primary model)

Labels: chord family (major / minor / diminished / augmented / suspended)

Evaluation: stratified 5-fold CV on all segments, then full retrain.
Saved model: harmonia/models/family_ltas_model.npz

Usage:
    .venv/bin/python scripts/train_family_ltas_model.py
    .venv/bin/python scripts/train_family_ltas_model.py --n-songs 80
"""
from __future__ import annotations
import argparse, json, sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

from analyze_accomp_emission import parse_chord, song_chord_spans
from build_accomp_audio_hard import render_to_array, stem_midi, SOUNDFONTS
from build_audio_chord_features import BUCKET_FAMILY
from harmonia.data.midi_renderer import MIDIRenderer

DB       = REPO / "data" / "accomp_db" / "db.jsonl"
MANIFEST = REPO / "data" / "accomp_db" / "audio" / "manifest.jsonl"
DIST_CACHE = REPO / "data" / "cache" / "ltas_family_dist.npz"
MODEL_OUT  = REPO / "harmonia" / "models" / "family_ltas_model.npz"
PLOT_OUT   = REPO / "docs" / "plots" / "family_ltas_model_eval.png"

FAMILIES = ["major", "minor", "diminished", "augmented", "suspended"]
NOTE     = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


# ── shared helpers (duplicated from plot_family_likelihood to keep standalone) ─

def _render_chord_only(midi_path, sf_name, renderer):
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    chord_pm = stem_midi(pm, lambda i: (not i.is_drum) and "bass" not in i.name.lower())
    if chord_pm is None or not chord_pm.instruments:
        return None, None
    audio, sr = render_to_array(renderer, chord_pm, sf_name, reverb=False)
    return audio.astype(float), sr


def _ltas_cqt(audio, sr, hop=512):
    raw = librosa.feature.chroma_cqt(y=audio, sr=sr,
                                      bins_per_octave=36, hop_length=hop)
    ltas = raw.mean(axis=1, keepdims=True)
    ltas = np.where(ltas < 1e-9, 1.0, ltas)
    chroma = raw / ltas
    ct = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop)
    return chroma, ct


def diag_gaussian_ll(x, mu, std):
    return float(-0.5 * np.sum(((x - mu) / std) ** 2) - np.sum(np.log(std)))


def max_ll_over_keys(x, mu, std):
    best_ll, best_r = -np.inf, 0
    for r in range(12):
        ll = diag_gaussian_ll(np.roll(x, -r), mu, std)
        if ll > best_ll:
            best_ll, best_r = ll, r
    return best_ll, best_r


def load_dist():
    if not DIST_CACHE.exists():
        raise FileNotFoundError(
            f"{DIST_CACHE} not found — run plot_family_likelihood.py --rebuild-cache first")
    d = np.load(DIST_CACHE)
    return {k: d[k] for k in d.files}


# ── data collection ───────────────────────────────────────────────────────────

def collect_features(n_songs: int, dist: dict) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Returns X (N, 17), y (N,), song_ids (N,)."""
    recs = {json.loads(l)["song_id"]: json.loads(l) for l in open(DB)}
    man: dict = {}
    for m in map(json.loads, open(MANIFEST)):
        if m["song_id"] not in man or m.get("transpose", 0) == 0:
            man[m["song_id"]] = m

    avail = sorted([s for s in recs if s in man], key=lambda s: recs[s]["title"])
    chosen = avail[:n_songs]
    renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")
    sf_name  = SOUNDFONTS[0]

    Xlist, ylist, slist = [], [], []

    for i, sid in enumerate(chosen):
        rec = recs[sid]; m = man[sid]
        bpb = m["beats_per_bar"]; spb = 60.0 / m["tempo"]
        print(f"\r  [{i+1}/{len(chosen)}] {rec['title'][:40]:40s}", end="", flush=True)
        try:
            audio, sr = _render_chord_only(REPO / m["midi_path"], sf_name, renderer)
            if audio is None: continue
        except Exception:
            continue
        chroma, ct = _ltas_cqt(audio, sr)
        chord_at = {(e["bar"]-1)*bpb+e["beat"]: e for e in rec["chord_timeline"]}

        for t0, t1, root_gt, _ in song_chord_spans(rec):
            b0 = int(round(t0 / spb))
            mma = chord_at.get(b0, {}).get("mma")
            p = parse_chord(mma) if mma else None
            if p is None or p[1] not in BUCKET_FAMILY: continue
            fam = BUCKET_FAMILY[p[1]]
            if fam not in FAMILIES: continue

            i0 = int(np.searchsorted(ct, t0))
            i1 = int(np.searchsorted(ct, t1))
            if i1 <= i0: i1 = i0 + 1
            seg = chroma[:, i0:i1].mean(axis=1)
            shifted = np.roll(seg, -int(root_gt % 12))   # root-shifted (GT root)
            n = np.linalg.norm(shifted)
            if n < 1e-9: continue
            x12 = shifted / n

            # 5d LL features
            ll5 = np.array([
                max_ll_over_keys(x12, dist[f"{f}_mu"], dist[f"{f}_std"])[0]
                for f in FAMILIES
            ])

            feat = np.concatenate([x12, ll5])   # 17d
            Xlist.append(feat)
            ylist.append(FAMILIES.index(fam))
            slist.append(sid)

    print()
    return np.array(Xlist), np.array(ylist), slist


# ── evaluation helpers ────────────────────────────────────────────────────────

def cv_accuracy(X, y, sc, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs = {"12d": [], "5d": [], "17d": []}
    cms  = {"12d": np.zeros((5,5),int), "5d": np.zeros((5,5),int),
            "17d": np.zeros((5,5),int)}
    slices = {"12d": slice(0,12), "5d": slice(12,17), "17d": slice(0,17)}

    for train_idx, val_idx in skf.split(X, y):
        Xtr, Xva = X[train_idx], X[val_idx]
        ytr, yva = y[train_idx], y[val_idx]
        for key, sl in slices.items():
            Xtr_s = sc.fit_transform(Xtr[:, sl])
            Xva_s = sc.transform(Xva[:, sl])
            clf = LogisticRegression(max_iter=1000, solver="lbfgs",
                                     class_weight="balanced", C=1.0)
            clf.fit(Xtr_s, ytr)
            pred = clf.predict(Xva_s)
            accs[key].append((pred == yva).mean())
            cms[key] += confusion_matrix(yva, pred, labels=list(range(5)))

    return {k: (np.mean(v), np.std(v)) for k, v in accs.items()}, cms


def train_final(X, y, sc, feature_slice):
    Xs = sc.fit_transform(X[:, feature_slice])
    clf = LogisticRegression(max_iter=1000, solver="lbfgs",
                             class_weight="balanced", C=1.0)
    clf.fit(Xs, y)
    return clf, sc


# ── plotting ──────────────────────────────────────────────────────────────────

FAM_COLORS = ["#58d4ff","#a65fd4","#e34948","#e0a03b","#1baf7a"]

def plot_results(accs, cms, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor="#0d1520")
    fig.suptitle("Family classifier — LTAS-CQT chroma features  (5-fold CV)",
                 color="#e2e8f0", fontsize=13, y=1.01)

    labels_short = ["maj","min","dim","aug","sus"]
    titles = {
        "12d": "12d — root-shifted chroma only",
        "5d":  "5d  — Gaussian LL per family",
        "17d": "17d — chroma + LL (combined)",
    }

    for ax, key in zip(axes, ["12d","5d","17d"]):
        cm = cms[key].astype(float)
        cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-9)
        im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues",
                       aspect="auto")
        ax.set_facecolor("#0d1520")
        ax.set_xticks(range(5)); ax.set_xticklabels(labels_short, fontsize=9, color="#88aacc")
        ax.set_yticks(range(5)); ax.set_yticklabels(labels_short, fontsize=9, color="#88aacc")
        ax.set_xlabel("predicted", color="#5a6a7e", fontsize=9)
        ax.set_ylabel("true",      color="#5a6a7e", fontsize=9)
        ax.spines[:].set_color("#253447")
        ax.tick_params(colors="#5a6a7e")
        # cell text
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{cm_norm[i,j]:.2f}", ha="center", va="center",
                        fontsize=8, color="#ffffff" if cm_norm[i,j] > 0.5 else "#5a6a7e")
        mu, sd = accs[key]
        ax.set_title(f"{titles[key]}\nacc = {mu:.1%} ± {sd:.1%}",
                     color="#e2e8f0", fontsize=9, pad=6)

    plt.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"→ {out}")


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-songs", type=int, default=80)
    args = ap.parse_args()

    dist = load_dist()
    print(f"Collecting features from {args.n_songs} songs...")
    X, y, sids = collect_features(args.n_songs, dist)
    print(f"  {len(X)} segments  |  class counts: "
          + "  ".join(f"{FAMILIES[i]}={int((y==i).sum())}" for i in range(5)))

    sc = StandardScaler()
    print("Running 5-fold CV (three feature sets)...")
    accs, cms = cv_accuracy(X, y, sc)
    print()
    print(f"  {'Feature':30s}  {'Acc':>8s}  {'±':>6s}")
    print("  " + "-"*46)
    for key, (mu, sd) in accs.items():
        print(f"  {key:30s}  {mu:8.1%}  {sd:6.1%}")

    print("\nPer-family recall (17d model, from CV confusion matrix):")
    cm17 = cms["17d"].astype(float)
    for i, fam in enumerate(FAMILIES):
        total = cm17[i].sum()
        recall = cm17[i,i] / total if total > 0 else 0
        print(f"  {fam:12s}: {recall:.1%}  (n={int(total)})")

    plot_results(accs, cms, PLOT_OUT)

    # Final model on all data (17d)
    clf_final, sc_final = train_final(X, y, StandardScaler(), slice(0,17))
    np.savez(MODEL_OUT,
             coef=clf_final.coef_,
             intercept=clf_final.intercept_,
             classes=clf_final.classes_,
             mean=sc_final.mean_,
             scale=sc_final.scale_,
             families=np.array(FAMILIES),
             feature_dim=np.array([17]),
             note="17d = 12d root-shifted LTAS chroma + 5d max-LL per family")
    print(f"\nModel saved → {MODEL_OUT}")
