"""Experiment: per-frame LL voting vs mean-chroma baseline.

Compares four aggregation strategies for family classification on hard audio,
oracle boundaries, GT root shift:

  baseline    : LL(mean(frames))                  — current approach
  sum_ll      : Σ_t LL(x_t)                       — product of frame likelihoods
  logsumexp   : log Σ_t exp(LL(x_t))              — soft-max over frames
  softmax_att : Σ_t w_t · LL(x_t), w = softmax   — attention-weighted LL

For each segment, the "query" for logsumexp/sum_ll/softmax_att is the
per-frame score vector rather than a single chroma mean.
Root is still found by sweeping 12 keys, but now per-frame (cheaply: we
only do the root sweep on the aggregate frame scores, not per-frame ×12).

Outputs:
  - accuracy table per strategy (5-fold CV, balanced LR, hard audio)
  - per-family breakdown
  - docs/plots/frame_voting_comparison.png  (confusion matrix grid)

Usage:
    .venv/bin/python scripts/experiment_frame_voting.py
    .venv/bin/python scripts/experiment_frame_voting.py --n-songs 50
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
from scipy.special import logsumexp as scipy_logsumexp
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
PLOT_OUT   = REPO / "docs" / "plots" / "frame_voting_comparison.png"

FAMILIES   = ["major", "minor", "diminished", "augmented", "suspended"]
FAM_COLORS = ["#58d4ff","#a65fd4","#e34948","#e0a03b","#1baf7a"]
NOTE       = ["C","Db","D","Eb","E","F","F#","G","Ab","A","Bb","B"]

HOP = 512   # ~11.6ms per frame at 44100 Hz


# ── audio / chroma helpers ────────────────────────────────────────────────────

def _render_hard(midi_path, rng):
    import pretty_midi
    renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")
    pm   = pretty_midi.PrettyMIDI(str(midi_path))
    scen = str(rng.choice(list(SCENARIOS)))
    gains = {k: v * float(rng.uniform(0.8, 1.2)) for k, v in SCENARIOS[scen].items()}
    sf    = SOUNDFONTS[int(rng.integers(0, len(SOUNDFONTS)))]
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
            w, sr2 = render_to_array(renderer, s, sf, reverb=False)
            waves[name] = w; sr = sr2
    L = max(len(w) for w in waves.values())
    mix = np.zeros(L, np.float32)
    for name, w in waves.items(): mix[:len(w)] += gains.get(name, 0.5) * w
    mix = time_varying_degrade(mix, sr, rng)
    peak = np.abs(mix).max()
    if peak > 0.99: mix *= 0.99 / peak
    return mix.astype(float), sr


def _ltas_chroma_frames(audio, sr):
    """Return (12, T) LTAS-normalised CQT chroma + frame times."""
    raw  = librosa.feature.chroma_cqt(y=audio, sr=sr, bins_per_octave=36, hop_length=HOP)
    ltas = raw.mean(axis=1, keepdims=True)
    ltas = np.where(ltas < 1e-9, 1.0, ltas)
    ct   = librosa.frames_to_time(np.arange(raw.shape[1]), sr=sr, hop_length=HOP)
    return raw / ltas, ct


# ── per-family LL computation ─────────────────────────────────────────────────

def _diag_ll(x, mu, std):
    return float(-0.5 * np.sum(((x - mu) / std)**2) - np.sum(np.log(std)))


def _max_ll_over_keys(x, mu, std):
    best_ll, best_r = -np.inf, 0
    for r in range(12):
        ll = _diag_ll(np.roll(x, -r), mu, std)
        if ll > best_ll: best_ll, best_r = ll, r
    return best_ll, best_r


def _frame_ll_matrix(frames_shifted: np.ndarray, dist: dict) -> np.ndarray:
    """
    frames_shifted : (12, T) root-shifted LTAS chroma frames
    Returns        : (T, 5) per-frame max-LL over 12 keys for each family.

    Naive: O(T × 5 × 12) LL calls. For typical segments (T≈20-80) this is fast.
    """
    T = frames_shifted.shape[1]
    ll_mat = np.zeros((T, len(FAMILIES)), dtype=np.float32)
    for t in range(T):
        x = frames_shifted[:, t]
        n = np.linalg.norm(x)
        if n < 1e-9: continue
        x = x / n
        for fi, fam in enumerate(FAMILIES):
            ll, _ = _max_ll_over_keys(x, dist[f"{fam}_mu"], dist[f"{fam}_std"])
            ll_mat[t, fi] = ll
    return ll_mat    # (T, 5)


# ── aggregation strategies ────────────────────────────────────────────────────

def agg_baseline(frames_shifted: np.ndarray, dist: dict) -> np.ndarray:
    """LL on the mean chroma (current approach). Returns (5,) LL vector."""
    mean_x = frames_shifted.mean(axis=1)
    n = np.linalg.norm(mean_x)
    if n < 1e-9: return np.zeros(5)
    x = mean_x / n
    return np.array([_max_ll_over_keys(x, dist[f"{fam}_mu"], dist[f"{fam}_std"])[0]
                     for fam in FAMILIES], dtype=np.float32)


def agg_sum_ll(ll_mat: np.ndarray) -> np.ndarray:
    """Σ_t LL(x_t) — product of frame likelihoods. Returns (5,)."""
    return ll_mat.sum(axis=0)


def agg_logsumexp(ll_mat: np.ndarray) -> np.ndarray:
    """log Σ_t exp(LL(x_t)) — soft-max over frames. Returns (5,)."""
    return scipy_logsumexp(ll_mat, axis=0).astype(np.float32)


def agg_softmax_att(ll_mat: np.ndarray) -> np.ndarray:
    """Attention: w_t = softmax(max_f LL(t,f)), score_f = Σ_t w_t · LL(t,f). Returns (5,)."""
    confidence = ll_mat.max(axis=1)          # (T,) — how confident is each frame?
    w = np.exp(confidence - confidence.max())
    w = w / (w.sum() + 1e-12)               # (T,) normalised
    return (w[:, None] * ll_mat).sum(axis=0).astype(np.float32)


# ── data collection ───────────────────────────────────────────────────────────

def collect(n_songs, dist, rng):
    """Returns list of feature dicts, one per segment, all four strategies."""
    recs = {json.loads(l)["song_id"]: json.loads(l) for l in open(DB)}
    man: dict = {}
    for m in map(json.loads, open(MANIFEST)):
        if m["song_id"] not in man or m.get("transpose", 0) == 0:
            man[m["song_id"]] = m
    avail  = sorted([s for s in recs if s in man], key=lambda s: recs[s]["title"])
    chosen = avail[:n_songs]

    records = []
    for i, sid in enumerate(chosen):
        rec = recs[sid]; m = man[sid]
        bpb = m["beats_per_bar"]; spb = 60.0 / m["tempo"]
        print(f"\r  [{i+1}/{n_songs}] {rec['title'][:40]:40s}", end="", flush=True)
        try:
            audio, sr = _render_hard(REPO / m["midi_path"], rng)
        except Exception:
            continue
        chroma, ct = _ltas_chroma_frames(audio, sr)
        chord_at = {(e["bar"]-1)*bpb+e["beat"]: e for e in rec["chord_timeline"]}

        for t0, t1, root_gt, _ in song_chord_spans(rec):
            b0  = int(round(t0 / spb))
            mma = chord_at.get(b0, {}).get("mma")
            p   = parse_chord(mma) if mma else None
            if p is None or p[1] not in BUCKET_FAMILY: continue
            fam = BUCKET_FAMILY[p[1]]
            if fam not in FAMILIES: continue
            root = int(root_gt % 12)

            i0 = int(np.searchsorted(ct, t0)); i1 = int(np.searchsorted(ct, t1))
            if i1 <= i0: i1 = i0 + 1
            frames_abs = chroma[:, i0:i1]                # (12, T) absolute pitch
            frames_shifted = np.roll(frames_abs, -root, axis=0)  # (12, T) root-shifted

            # chroma mean (12d, L2-normed) — used in the 17d LogReg strategies
            mean_x = frames_shifted.mean(axis=1)
            n = np.linalg.norm(mean_x)
            chroma_mean = (mean_x / n).astype(np.float32) if n > 1e-9 else np.zeros(12, np.float32)

            # baseline LL on the mean chroma (same as the 80% setup)
            ll_base = agg_baseline(frames_shifted, dist)

            # frame-level LL matrix — shared by frame-voting strategies
            ll_mat = _frame_ll_matrix(frames_shifted, dist)
            if ll_mat.shape[0] == 0: continue

            ll_lse  = agg_logsumexp(ll_mat)
            ll_att  = agg_softmax_att(ll_mat)

            y = FAMILIES.index(fam)
            records.append({
                "y": y,
                "chroma_mean": chroma_mean,
                "ll_base": ll_base,
                "ll_lse":  ll_lse,
                "ll_att":  ll_att,
                "n_frames": ll_mat.shape[0],
                "dur": t1 - t0,
            })
    print()
    return records


# ── evaluation ────────────────────────────────────────────────────────────────

def cv_eval(X, y, n_splits=5):
    skf  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs = []; cm_tot = np.zeros((5,5), int)
    sc   = StandardScaler()
    for tr, va in skf.split(X, y):
        Xtr = sc.fit_transform(X[tr]); Xva = sc.transform(X[va])
        clf = LogisticRegression(max_iter=1000, solver="lbfgs",
                                 class_weight="balanced", C=1.0)
        clf.fit(Xtr, y[tr]); pred = clf.predict(Xva)
        accs.append((pred == y[va]).mean())
        cm_tot += confusion_matrix(y[va], pred, labels=list(range(5)))
    return float(np.mean(accs)), float(np.std(accs)), cm_tot


CONTEXT_K = 4   # ±4 segments of context

STRATEGIES = {
    # pure LL-vote strategies (5d feature, no chroma mean)
    "baseline_ll":     lambda r: r["ll_base"],
    "logsumexp_ll":    lambda r: r["ll_lse"],
    "softmax_att_ll":  lambda r: r["ll_att"],
    # 17d LogReg strategies: 12d chroma mean + 5d LL (matching the 80% setup)
    "logreg_base":     lambda r: np.concatenate([r["chroma_mean"], r["ll_base"]]),
    "logreg_lse":      lambda r: np.concatenate([r["chroma_mean"], r["ll_lse"]]),
    "logreg_att":      lambda r: np.concatenate([r["chroma_mean"], r["ll_att"]]),
    "logreg_att+ctx":  None,   # built specially below
}


def build_context_features(records, k: int = CONTEXT_K) -> np.ndarray:
    """
    For each segment i: 12d chroma_mean + softmax_att LL for positions
    [i-k, ..., i, ..., i+k].
    Returns (N, 12 + (2k+1)*5).
    """
    chroma = np.stack([r["chroma_mean"] for r in records])  # (N, 12)
    att    = np.stack([r["ll_att"]      for r in records])  # (N, 5)
    N, F = att.shape
    window = 2 * k + 1
    ctx = np.zeros((N, window * F), dtype=np.float32)
    for i in range(N):
        for j, offset in enumerate(range(-k, k + 1)):
            ni = i + offset
            if 0 <= ni < N:
                ctx[i, j*F:(j+1)*F] = att[ni]
    return np.concatenate([chroma, ctx], axis=1)  # (N, 12 + 45)


def run_all(records):
    y = np.array([r["y"] for r in records])
    results = {}
    for name, feat_fn in STRATEGIES.items():
        if feat_fn is None:   # context strategy built separately
            X = build_context_features(records, CONTEXT_K)
        else:
            X = np.stack([feat_fn(r) for r in records])
        mu, sd, cm = cv_eval(X, y)
        results[name] = (mu, sd, cm)
    return results


# ── plotting ──────────────────────────────────────────────────────────────────

def plot(results, out: Path):
    strategies = list(results.keys())
    n = len(strategies)
    fig, axes = plt.subplots(1, n, figsize=(n*4.2, 5), facecolor="#0d1520")
    fig.suptitle(
        "Per-frame LL voting vs mean-chroma baseline\n"
        "(hard audio, oracle boundaries, GT root shift, 5-fold CV)",
        color="#e2e8f0", fontsize=11, y=1.02)

    labels = ["maj","min","dim","aug","sus"]
    for ax, name in zip(axes, strategies):
        mu, sd, cm = results[name]
        cm_n = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
        ax.imshow(cm_n, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        ax.set_facecolor("#0d1520")
        ax.set_xticks(range(5)); ax.set_xticklabels(labels, fontsize=8, color="#88aacc")
        ax.set_yticks(range(5)); ax.set_yticklabels(labels, fontsize=8, color="#88aacc")
        ax.set_xlabel("predicted", color="#5a6a7e", fontsize=8)
        ax.set_ylabel("true",      color="#5a6a7e", fontsize=8)
        ax.spines[:].set_color("#253447")
        ax.tick_params(colors="#5a6a7e")
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{cm_n[i,j]:.2f}", ha="center", va="center",
                        fontsize=8, color="#fff" if cm_n[i,j] > 0.5 else "#5a6a7e")
        ax.set_title(f"{name}\n{mu:.1%} ± {sd:.1%}", color="#e2e8f0", fontsize=9, pad=6)

    # bar chart of accuracies on the right
    fig2, ax2 = plt.subplots(figsize=(5, 3), facecolor="#0d1520")
    ax2.set_facecolor("#0d1520")
    mus = [results[s][0] for s in strategies]
    sds = [results[s][1] for s in strategies]
    cols = ["#4a6080","#3a7a60","#2a6a80",
            "#58d4ff","#a65fd4","#1baf7a","#e0a03b"]
    bars = ax2.bar(strategies, mus, yerr=sds, color=cols, edgecolor="#253447",
                   linewidth=0.8, capsize=4, error_kw=dict(ecolor="#ffffff55", lw=1.2))
    ax2.set_ylim(0, 1.0)
    base_idx = list(results.keys()).index("logreg_base")
    ax2.axhline(mus[base_idx], color="#4a6080", lw=1, linestyle="--", alpha=0.5)
    for i, (mu, sd) in enumerate(zip(mus, sds)):
        delta = mu - mus[0]
        sign  = "+" if delta >= 0 else ""
        ax2.text(i, mu + sd + 0.015, f"{mu:.1%}\n({sign}{delta*100:.1f}pp)",
                 ha="center", va="bottom", fontsize=8, color="#e2e8f0")
    ax2.set_xticklabels(strategies, fontsize=8, color="#88aacc", rotation=10)
    ax2.tick_params(axis="y", colors="#5a6a7e", labelsize=7)
    ax2.spines[:].set_color("#253447")
    ax2.set_title("Accuracy vs baseline (dashed)", color="#e2e8f0", fontsize=9)
    fig2.tight_layout()

    out2 = out.parent / (out.stem + "_bar.png")
    fig2.savefig(out2, dpi=150, bbox_inches="tight", facecolor=fig2.get_facecolor())
    plt.close(fig2)
    print(f"→ {out2}")

    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"→ {out}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-songs", type=int, default=40)
    ap.add_argument("--seed",    type=int, default=42)
    args = ap.parse_args()

    if not DIST_CACHE.exists():
        print("ERROR: run plot_family_likelihood.py --rebuild-cache first"); sys.exit(1)
    d = np.load(DIST_CACHE)
    dist = {k: d[k] for k in d.files}

    rng = np.random.default_rng(args.seed)
    print(f"Collecting segments ({args.n_songs} songs, hard audio, oracle bounds)...")
    records = collect(args.n_songs, dist, rng)
    print(f"  {len(records)} segments  |  avg {np.mean([r['n_frames'] for r in records]):.1f} frames/seg"
          f"  |  avg dur {np.mean([r['dur'] for r in records]):.2f}s")

    print("\nRunning 5-fold CV for all strategies...")
    results = run_all(records)

    print(f"\n  {'Strategy':14s}  {'Acc':>7s}  {'±':>5s}  {'Δ baseline':>10s}")
    print("  " + "-"*42)
    base_acc = results["logreg_base"][0]
    for name, (mu, sd, _) in results.items():
        delta = mu - base_acc
        sign  = "+" if delta >= 0 else ""
        print(f"  {name:14s}  {mu:7.1%}  {sd:5.1%}  {sign}{delta*100:+.1f}pp")

    print()
    print("Per-family recall  (logreg_base vs logreg_att vs logreg_att+ctx):")
    labels = ["major","minor","diminished","augmented","suspended"]
    for i, fam in enumerate(labels):
        cm_b  = results["logreg_base"][2]
        cm_a  = results["logreg_att"][2]
        cm_ac = results["logreg_att+ctx"][2]
        r_b  = cm_b[i,i] / max(cm_b[i].sum(), 1)
        r_a  = cm_a[i,i] / max(cm_a[i].sum(), 1)
        r_ac = cm_ac[i,i] / max(cm_ac[i].sum(), 1)
        print(f"  {fam:12s}  base={r_b:.1%}  att={r_a:.1%}  att+ctx={r_ac:.1%}"
              f"  n={int(cm_b[i].sum())}")

    plot(results, PLOT_OUT)
