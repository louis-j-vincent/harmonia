"""Train family classifier on hard audio with oracle boundaries.

Same 17d features as train_family_ltas_model.py but audio is the full
degraded multi-stem mix (chords + bass + drums + melody + pink noise),
matching what the blind pipeline actually sees.

Oracle = GT boundaries used for segment extraction; GT root used for
root-shifting (so we isolate the audio-quality effect from root errors).

Comparison printed at end:
  clean chord-only  vs  hard multi-stem
to quantify how much the audio degradation alone costs.

Usage:
    .venv/bin/python scripts/train_family_hard_oracle.py
    .venv/bin/python scripts/train_family_hard_oracle.py --n-songs 80
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
from build_accomp_audio_hard import (
    SCENARIOS, SOUNDFONTS, LEAD_PROGRAMS,
    make_melody, render_to_array, stem_midi, time_varying_degrade,
)
from build_audio_chord_features import BUCKET_FAMILY
from harmonia.data.midi_renderer import MIDIRenderer

DB         = REPO / "data" / "accomp_db" / "db.jsonl"
MANIFEST   = REPO / "data" / "accomp_db" / "audio" / "manifest.jsonl"
DIST_CACHE = REPO / "data" / "cache" / "ltas_family_dist.npz"
MODEL_OUT  = REPO / "harmonia" / "models" / "family_hard_oracle_model.npz"
PLOT_OUT   = REPO / "docs" / "plots" / "family_hard_oracle_eval.png"

FAMILIES  = ["major", "minor", "diminished", "augmented", "suspended"]
FAM_COLORS = ["#58d4ff","#a65fd4","#e34948","#e0a03b","#1baf7a"]
NOTE      = ["C","Db","D","Eb","E","F","F#","G","Ab","A","Bb","B"]


def _render_hard(midi_path, man_entry, rng):
    """Full degraded mix: chords + bass + drums + melody + pink noise."""
    import pretty_midi
    renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    scen = str(rng.choice(list(SCENARIOS)))
    gains = {k: v * float(rng.uniform(0.8, 1.2)) for k, v in SCENARIOS[scen].items()}
    sf_name = SOUNDFONTS[int(rng.integers(0, len(SOUNDFONTS)))]
    stems = {
        "chords": stem_midi(pm, lambda i: (not i.is_drum) and "bass" not in i.name.lower()),
        "bass":   stem_midi(pm, lambda i: "bass" in i.name.lower()),
        "drums":  stem_midi(pm, lambda i: i.is_drum),
    }
    if gains.get("melody", 0) > 0.01:
        mel_pm = pretty_midi.PrettyMIDI()
        m = make_melody(pm, int(rng.choice(LEAD_PROGRAMS)), rng)
        if m: mel_pm.instruments.append(m); stems["melody"] = mel_pm
    waves, sr = {}, 44100
    for name, s in stems.items():
        if s and s.instruments:
            w, sr2 = render_to_array(renderer, s, sf_name, reverb=False)
            waves[name] = w; sr = sr2
    L = max(len(w) for w in waves.values())
    mix = np.zeros(L, np.float32)
    for name, w in waves.items():
        mix[:len(w)] += gains.get(name, 0.5) * w
    mix = time_varying_degrade(mix, sr, rng)
    peak = np.abs(mix).max()
    if peak > 0.99: mix *= 0.99 / peak
    return mix.astype(float), sr


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


def collect_features(n_songs: int, dist: dict, rng) -> tuple[np.ndarray, np.ndarray, list]:
    recs = {json.loads(l)["song_id"]: json.loads(l) for l in open(DB)}
    man: dict = {}
    for m in map(json.loads, open(MANIFEST)):
        if m["song_id"] not in man or m.get("transpose", 0) == 0:
            man[m["song_id"]] = m

    avail = sorted([s for s in recs if s in man], key=lambda s: recs[s]["title"])
    chosen = avail[:n_songs]

    Xlist, ylist, slist = [], [], []
    for i, sid in enumerate(chosen):
        rec = recs[sid]; m = man[sid]
        bpb = m["beats_per_bar"]; spb = 60.0 / m["tempo"]
        print(f"\r  [{i+1}/{len(chosen)}] {rec['title'][:40]:40s}", end="", flush=True)
        try:
            audio, sr = _render_hard(REPO / m["midi_path"], m, rng)
        except Exception as e:
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
            shifted = np.roll(seg, -int(root_gt % 12))
            n = np.linalg.norm(shifted)
            if n < 1e-9: continue
            x12 = shifted / n

            ll5 = np.array([
                max_ll_over_keys(x12, dist[f"{f}_mu"], dist[f"{f}_std"])[0]
                for f in FAMILIES
            ])
            Xlist.append(np.concatenate([x12, ll5]))
            ylist.append(FAMILIES.index(fam))
            slist.append(sid)

    print()
    return np.array(Xlist), np.array(ylist), slist


def cv_accuracy(X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs = []
    cm_total = np.zeros((5, 5), int)
    sc = StandardScaler()
    for train_idx, val_idx in skf.split(X, y):
        Xtr_s = sc.fit_transform(X[train_idx])
        Xva_s = sc.transform(X[val_idx])
        clf = LogisticRegression(max_iter=1000, solver="lbfgs",
                                 class_weight="balanced", C=1.0)
        clf.fit(Xtr_s, y[train_idx])
        pred = clf.predict(Xva_s)
        accs.append((pred == y[val_idx]).mean())
        cm_total += confusion_matrix(y[val_idx], pred, labels=list(range(5)))
    return float(np.mean(accs)), float(np.std(accs)), cm_total


def plot_comparison(cm_clean, acc_clean, cm_hard, acc_hard, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), facecolor="#0d1520")
    fig.suptitle("Family classifier — clean chord-only vs hard multi-stem  (oracle boundaries, 17d, 5-fold CV)",
                 color="#e2e8f0", fontsize=12, y=1.01)
    labels_short = ["maj","min","dim","aug","sus"]
    pairs = [
        (axes[0], cm_clean, acc_clean, "Clean audio (chord-only stem)"),
        (axes[1], cm_hard,  acc_hard,  "Hard audio (full mix + noise)"),
    ]
    for ax, cm, (mu, sd), title in pairs:
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
        ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        ax.set_facecolor("#0d1520")
        ax.set_xticks(range(5)); ax.set_xticklabels(labels_short, fontsize=9, color="#88aacc")
        ax.set_yticks(range(5)); ax.set_yticklabels(labels_short, fontsize=9, color="#88aacc")
        ax.set_xlabel("predicted", color="#5a6a7e", fontsize=9)
        ax.set_ylabel("true",      color="#5a6a7e", fontsize=9)
        ax.spines[:].set_color("#253447")
        ax.tick_params(colors="#5a6a7e")
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{cm_norm[i,j]:.2f}", ha="center", va="center",
                        fontsize=8.5,
                        color="#ffffff" if cm_norm[i,j] > 0.5 else "#5a6a7e")
        ax.set_title(f"{title}\nacc = {mu:.1%} ± {sd:.1%}",
                     color="#e2e8f0", fontsize=9, pad=6)
    plt.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"→ {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-songs", type=int, default=80)
    ap.add_argument("--seed",    type=int, default=42)
    args = ap.parse_args()

    if not DIST_CACHE.exists():
        print("ERROR: run plot_family_likelihood.py --rebuild-cache first"); sys.exit(1)
    d = np.load(DIST_CACHE)
    dist = {k: d[k] for k in d.files}

    rng = np.random.default_rng(args.seed)
    print(f"Collecting features — hard audio, oracle boundaries, {args.n_songs} songs...")
    X, y, sids = collect_features(args.n_songs, dist, rng)
    print(f"  {len(X)} segments  |  class counts: "
          + "  ".join(f"{FAMILIES[i]}={int((y==i).sum())}" for i in range(5)))

    print("5-fold CV on hard audio...")
    mu_h, sd_h, cm_hard = cv_accuracy(X, y)
    print(f"  hard audio  acc = {mu_h:.1%} ± {sd_h:.1%}")

    print("\nPer-family recall (hard audio):")
    for i, fam in enumerate(FAMILIES):
        total = cm_hard[i].sum()
        recall = cm_hard[i,i] / total if total > 0 else 0
        print(f"  {fam:12s}: {recall:.1%}  (n={int(total)})")

    # Load clean-audio confusion matrix from existing eval for comparison
    clean_model = REPO / "harmonia" / "models" / "family_ltas_model.npz"
    if clean_model.exists():
        # Re-run CV on clean features so confusion matrices are comparable
        from train_family_ltas_model import collect_features as collect_clean
        print("\nRe-collecting clean features for comparison plot...")
        Xc, yc, _ = collect_clean(args.n_songs, dist)
        mu_c, sd_c, cm_clean = cv_accuracy(Xc, yc)
        print(f"  clean audio acc = {mu_c:.1%} ± {sd_c:.1%}")
        plot_comparison(cm_clean, (mu_c, sd_c), cm_hard, (mu_h, sd_h), PLOT_OUT)
    else:
        print("(clean model not found — skipping comparison plot)")

    # Save hard-audio model
    sc_final = StandardScaler()
    Xs = sc_final.fit_transform(X)
    clf_final = LogisticRegression(max_iter=1000, solver="lbfgs",
                                   class_weight="balanced", C=1.0)
    clf_final.fit(Xs, y)
    np.savez(MODEL_OUT,
             coef=clf_final.coef_,
             intercept=clf_final.intercept_,
             classes=clf_final.classes_,
             mean=sc_final.mean_,
             scale=sc_final.scale_,
             families=np.array(FAMILIES),
             feature_dim=np.array([17]),
             note="17d LTAS chroma + LL, trained on hard multi-stem audio, oracle boundaries")
    print(f"\nModel saved → {MODEL_OUT}")
