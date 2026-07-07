"""Sweep degradation levels and report family accuracy at each level.

Levels (each non-uniform in time):
  clean    — straight stereo mix, no processing
  mild     — drifting gain + low pink noise (SNR 15–30 dB), no band-limit
  moderate — time_varying_degrade(phone=False): gain + noise, no phone filter
  current  — time_varying_degrade(phone=True): adds 150–6 kHz band-limit + hum + soft clip
  hard     — strong_nonuniform_degrade: wow/flutter, dropout, overdrive, bitcrush

For each level reports:
  logreg_base  (17d: chroma + 5d LL) — the existing baseline
  gate_mlp     (entropy gate over MLP, key-unified context tensor) — best model

Usage:
    .venv/bin/python scripts/experiment_degrade_sweep.py
    .venv/bin/python scripts/experiment_degrade_sweep.py --n-songs 40 --epochs 60
    .venv/bin/python scripts/experiment_degrade_sweep.py --levels clean mild current
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
from scipy.special import logsumexp as sp_logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

from analyze_accomp_emission import parse_chord, song_chord_spans
from build_accomp_audio_hard import (
    SCENARIOS, SOUNDFONTS, LEAD_PROGRAMS,
    make_melody, render_to_array, stem_midi,
    time_varying_degrade, strong_nonuniform_degrade, pink,
)
from build_audio_chord_features import BUCKET_FAMILY
from harmonia.data.midi_renderer import MIDIRenderer

DB         = REPO / "data" / "accomp_db" / "db.jsonl"
MANIFEST   = REPO / "data" / "accomp_db" / "audio" / "manifest.jsonl"
DIST_CACHE = REPO / "data" / "cache" / "ltas_family_dist.npz"
OUT_PNG    = REPO / "docs" / "plots" / "degrade_sweep.png"

FAMILIES   = ["major", "minor", "diminished", "augmented", "suspended"]
FAM_COLORS = ["#58d4ff", "#a65fd4", "#e34948", "#e0a03b", "#1baf7a"]
HOP        = 512
CTX_K      = 4

LEVEL_LABELS = {
    "clean":    "Clean\n(no processing)",
    "mild":     "Mild\n(gain drift + low noise)",
    "moderate": "Moderate\n(gain + noise, no phone)",
    "current":  "Current\n(phone band-limit + noise)",
    "hard":     "Hard\n(wow/flutter + dropout + clip)",
}
LEVEL_COLORS = {
    "clean":    "#1baf7a",
    "mild":     "#58d4ff",
    "moderate": "#a65fd4",
    "current":  "#e0a03b",
    "hard":     "#e34948",
}


# ── degradation functions ─────────────────────────────────────────────────────

def degrade(mix: np.ndarray, sr: int, level: str, rng) -> np.ndarray:
    if level == "clean":
        return mix.astype(np.float32)

    if level == "mild":
        n = len(mix)
        K = max(5, n // (sr * 2))
        ctrl = np.linspace(0, n - 1, K)
        idx  = np.arange(n)
        gain = np.interp(idx, ctrl, rng.uniform(0.85, 1.05, K)).astype(np.float32)
        snr  = np.interp(idx, ctrl, rng.uniform(15.0, 30.0, K)).astype(np.float32)
        p    = float(np.mean(mix ** 2)) + 1e-9
        noise = pink(n, rng) * np.sqrt(p / (10 ** (snr / 10)))
        y = mix * gain + noise
        peak = np.abs(y).max()
        return (y * 0.99 / peak).astype(np.float32) if peak > 0.99 else y.astype(np.float32)

    if level == "moderate":
        return time_varying_degrade(mix, sr, rng, phone=False)

    if level == "current":
        return time_varying_degrade(mix, sr, rng, phone=True)

    if level == "hard":
        return strong_nonuniform_degrade(mix, sr, rng)

    raise ValueError(f"Unknown level: {level!r}")


# ── rendering ─────────────────────────────────────────────────────────────────

def _render_mix(midi_path, rng):
    """Render full mix (chords + bass + drums + optional melody) without any degradation."""
    import pretty_midi
    renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")
    pm    = pretty_midi.PrettyMIDI(str(midi_path))
    scen  = str(rng.choice(list(SCENARIOS)))
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
    peak = np.abs(mix).max()
    if peak > 0.99: mix *= 0.99 / peak
    return mix.astype(np.float32), sr


# ── LL helpers ────────────────────────────────────────────────────────────────

def _diag_ll(x, mu, std):
    return float(-0.5 * np.sum(((x - mu) / std)**2) - np.sum(np.log(std)))


def _compute_ll_mat(chroma_abs, dist):
    n = np.linalg.norm(chroma_abs)
    if n < 1e-9: return np.zeros((5, 12), np.float32)
    x = chroma_abs / n
    ll = np.zeros((5, 12), np.float32)
    for fi, fam in enumerate(FAMILIES):
        mu  = dist[f"{fam}_mu"]
        std = dist[f"{fam}_std"]
        for r in range(12):
            ll[fi, r] = _diag_ll(np.roll(x, -r), mu, std)
    return ll


def _softmax_att_ll(frames_shifted, dist):
    T = frames_shifted.shape[1]
    if T == 0: return np.zeros((5, 12), np.float32)
    frame_mats = np.zeros((T, 5, 12), np.float32)
    for t in range(T):
        x = frames_shifted[:, t]
        nn = np.linalg.norm(x)
        if nn < 1e-9: continue
        frame_mats[t] = _compute_ll_mat(x, dist)
    conf = frame_mats.max(axis=(1, 2))
    w    = np.exp(conf - conf.max()); w /= w.sum() + 1e-12
    return (w[:, None, None] * frame_mats).sum(axis=0)


# ── collect segments for one degradation level ────────────────────────────────

def collect(n_songs, dist, rng, level):
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
        print(f"\r    [{i+1:2d}/{n_songs}] {rec['title'][:38]:38s}", end="", flush=True)
        try:
            mix, sr = _render_mix(REPO / m["midi_path"], rng)
        except Exception:
            continue
        audio = degrade(mix, sr, level, rng).astype(float)

        raw  = librosa.feature.chroma_cqt(y=audio, sr=sr, bins_per_octave=36, hop_length=HOP)
        ltas = raw.mean(axis=1, keepdims=True)
        ltas = np.where(ltas < 1e-9, 1.0, ltas)
        chroma = raw / ltas
        ct = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=HOP)

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
            frames_shifted = np.roll(chroma[:, i0:i1], -root, axis=0)

            mean_s = frames_shifted.mean(axis=1)
            nn = np.linalg.norm(mean_s)
            if nn < 1e-9: continue
            chroma_mean = (mean_s / nn).astype(np.float32)

            ll_mat = _softmax_att_ll(frames_shifted, dist)   # (5, 12)

            records.append({
                "y":          FAMILIES.index(fam),
                "chroma_mean": chroma_mean,
                "ll_mat":     ll_mat,
                "root_pc":    root,
            })
    print()
    return records


# ── feature building ──────────────────────────────────────────────────────────

def _ctx_tensor(records, k=CTX_K):
    N = len(records); W = 2 * k + 1
    out = np.zeros((N, W, 5, 12), np.float32)
    for i in range(N):
        root_i = records[i]["root_pc"]
        for j, offset in enumerate(range(-k, k + 1)):
            ni = i + offset
            if 0 <= ni < N:
                delta = (records[ni]["root_pc"] - root_i) % 12
                out[i, j] = np.roll(records[ni]["ll_mat"], -delta, axis=1)
    return out


def build_features(records):
    y        = np.array([r["y"] for r in records])
    X_chroma = np.stack([r["chroma_mean"] for r in records])
    X_ll5    = np.stack([r["ll_mat"].max(axis=1) for r in records])
    X_base   = np.concatenate([X_chroma, X_ll5], axis=1)          # (N, 17)
    ctx      = _ctx_tensor(records)                                 # (N, 9, 5, 12)
    ctx_flat = ctx.reshape(len(records), -1)
    X_ctx    = np.concatenate([X_chroma, ctx_flat], axis=1)        # (N, 552)
    return y, X_base, X_ctx, ctx


# ── CV helpers (imported logic from experiment_ctx_model) ────────────────────

def _cv_logreg_logits(X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    sc  = StandardScaler()
    accs = []; cm_tot = np.zeros((5, 5), int)
    oof  = np.zeros((len(y), 5), np.float32)
    for tr, va in skf.split(X, y):
        Xtr = sc.fit_transform(X[tr]); Xva = sc.transform(X[va])
        clf = LogisticRegression(max_iter=2000, solver="lbfgs",
                                 class_weight="balanced", C=1.0)
        clf.fit(Xtr, y[tr])
        oof[va] = clf.decision_function(Xva)
        pred = clf.predict(Xva)
        accs.append((pred == y[va]).mean())
        cm_tot += confusion_matrix(y[va], pred, labels=list(range(5)))
    return float(np.mean(accs)), float(np.std(accs)), cm_tot, oof


def _cv_torch_logits(model_fn, X_np, y, n_splits=5, epochs=60, lr=3e-3, batch=64):
    import torch, torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs = []; cm_tot = np.zeros((5, 5), int)
    oof  = np.zeros((len(y), 5), np.float32)
    for fold, (tr, va) in enumerate(skf.split(X_np, y)):
        Xtr = torch.tensor(X_np[tr], dtype=torch.float32)
        Xva = torch.tensor(X_np[va], dtype=torch.float32)
        ytr = torch.tensor(y[tr], dtype=torch.long)
        counts = np.bincount(y[tr], minlength=5).astype(float)
        wts = torch.tensor(1.0 / (counts + 1e-9), dtype=torch.float32)
        wts = wts / wts.sum() * 5
        model   = model_fn(X_np.shape[1:])
        opt     = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        loss_fn = nn.CrossEntropyLoss(weight=wts)
        dl      = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch, shuffle=True)
        model.train()
        for ep in range(epochs):
            for xb, yb in dl:
                opt.zero_grad(); loss_fn(model(xb), yb).backward(); opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            logits = model(Xva).cpu().numpy()
        oof[va] = logits
        pred = logits.argmax(1)
        accs.append((pred == y[va]).mean())
        cm_tot += confusion_matrix(y[va], pred, labels=list(range(5)))
        print(f"      fold {fold+1}/5  val={accs[-1]:.1%}", flush=True)
    return float(np.mean(accs)), float(np.std(accs)), cm_tot, oof


def _fit_gate(lb, lc, y):
    import torch, torch.nn as nn
    def _sm(x):
        x = x - x.max(1, keepdims=True); e = np.exp(x); return e / e.sum(1, keepdims=True)
    pb = torch.tensor(_sm(lb), dtype=torch.float32)
    pc = torch.tensor(_sm(lc), dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    H  = -(pb * (pb + 1e-12).log()).sum(1, keepdim=True)
    w  = torch.nn.Parameter(torch.tensor([1.0]))
    b  = torch.nn.Parameter(torch.tensor([0.0]))
    opt = torch.optim.LBFGS([w, b], lr=0.5, max_iter=200)
    def closure():
        opt.zero_grad()
        alpha = torch.sigmoid(w * H + b)
        loss  = nn.NLLLoss()((alpha * pb + (1 - alpha) * pc).log(), yt)
        loss.backward(); return loss
    opt.step(closure)
    return float(w.item()), float(b.item())


def _apply_gate(lb, lc, w, b):
    import torch
    def _sm(x):
        x = x - x.max(1, keepdims=True); e = np.exp(x); return e / e.sum(1, keepdims=True)
    pb = torch.tensor(_sm(lb), dtype=torch.float32)
    pc = torch.tensor(_sm(lc), dtype=torch.float32)
    H  = -(pb * (pb + 1e-12).log()).sum(1, keepdim=True)
    alpha = torch.sigmoid(torch.tensor(w) * H + torch.tensor(b))
    return (alpha * pb + (1 - alpha) * pc).argmax(1).numpy()


def _cv_gate(lb_all, lc_all, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs = []; cm_tot = np.zeros((5, 5), int)
    for tr, va in skf.split(lb_all, y):
        w, b = _fit_gate(lb_all[tr], lc_all[tr], y[tr])
        pred = _apply_gate(lb_all[va], lc_all[va], w, b)
        accs.append((pred == y[va]).mean())
        cm_tot += confusion_matrix(y[va], pred, labels=list(range(5)))
    return float(np.mean(accs)), float(np.std(accs)), cm_tot


def _make_mlp(flat_dim):
    import torch.nn as nn
    d = int(np.prod(flat_dim))
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(d, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.3),
        nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.2),
        nn.Linear(128, 5),
    )


# ── per-family recall helper ───────────────────────────────────────────────────

def per_family_recall(cm):
    return [cm[i, i] / max(cm[i].sum(), 1) for i in range(5)]


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-songs", type=int, default=40)
    ap.add_argument("--epochs",  type=int, default=60)
    ap.add_argument("--seed",    type=int, default=42)
    ap.add_argument("--levels",  nargs="+",
                    default=["clean", "mild", "moderate", "current", "hard"],
                    choices=list(LEVEL_LABELS))
    args = ap.parse_args()

    if not DIST_CACHE.exists():
        print("ERROR: run plot_family_likelihood.py --rebuild-cache first"); sys.exit(1)
    d = np.load(DIST_CACHE)
    dist = {k: d[k] for k in d.files}

    try:
        import torch
        has_torch = True
    except ImportError:
        has_torch = False
        print("(PyTorch not found — will only run logreg_base)")

    results = {}   # level → {model: (acc, std, cm)}

    for level in args.levels:
        print(f"\n{'='*60}")
        print(f"  Level: {level.upper()}")
        print(f"{'='*60}")
        rng = np.random.default_rng(args.seed)   # same seed per level for reproducibility
        records = collect(args.n_songs, dist, rng, level)
        print(f"  {len(records)} segments")

        y, X_base, X_ctx, ctx = build_features(records)
        level_res = {}

        print("  logreg_base...")
        mu, sd, cm, oof_base = _cv_logreg_logits(X_base, y)
        level_res["logreg_base"] = (mu, sd, cm)
        print(f"    {mu:.1%} ± {sd:.1%}")

        if has_torch:
            print("  MLP...")
            mu, sd, cm, oof_mlp = _cv_torch_logits(_make_mlp, X_ctx, y, epochs=args.epochs)
            level_res["mlp"] = (mu, sd, cm)
            print(f"    {mu:.1%} ± {sd:.1%}")

            print("  gate (base ↔ MLP)...")
            mu, sd, cm = _cv_gate(oof_base, oof_mlp, y)
            level_res["gate"] = (mu, sd, cm)
            print(f"    {mu:.1%} ± {sd:.1%}")

        results[level] = level_res

    # ── summary table ─────────────────────────────────────────────────────────
    print(f"\n\n{'─'*72}")
    print(f"  {'Level':12s}  {'logreg_base':>12s}  {'mlp':>9s}  {'gate':>9s}")
    print(f"{'─'*72}")
    for level in args.levels:
        r = results[level]
        base_s = f"{r['logreg_base'][0]:.1%} ±{r['logreg_base'][1]:.1%}"
        mlp_s  = f"{r['mlp'][0]:.1%}" if "mlp" in r else "—"
        gate_s = f"{r['gate'][0]:.1%}" if "gate" in r else "—"
        print(f"  {level:12s}  {base_s:>12s}  {mlp_s:>9s}  {gate_s:>9s}")
    print(f"{'─'*72}")

    # per-family at each level for gate model
    if has_torch:
        print(f"\nPer-family recall — gate model:")
        fam_short = [f[:3] for f in FAMILIES]
        header = f"  {'Level':12s}" + "".join(f"  {f:>6s}" for f in fam_short)
        print(header)
        for level in args.levels:
            cm = results[level].get("gate", results[level]["logreg_base"])[2]
            row = f"  {level:12s}" + "".join(f"  {cm[i,i]/max(cm[i].sum(),1):6.1%}"
                                              for i in range(5))
            print(row)

    # ── plot ──────────────────────────────────────────────────────────────────
    levels = args.levels
    models = [m for m in ["logreg_base", "mlp", "gate"] if m in results[levels[0]]]
    model_labels = {"logreg_base": "LogReg base", "mlp": "MLP (ctx)",
                    "gate": "Entropy gate"}
    model_cols   = {"logreg_base": "#4a7090", "mlp": "#58d4ff", "gate": "#1baf7a"}

    x = np.arange(len(levels))
    width = 0.22
    offsets = np.linspace(-(len(models)-1)/2, (len(models)-1)/2, len(models)) * width

    fig, (ax_main, ax_gap) = plt.subplots(2, 1, figsize=(11, 7),
                                           facecolor="#0d1520",
                                           gridspec_kw={"height_ratios": [3, 1],
                                                        "hspace": 0.4})
    for ax in (ax_main, ax_gap):
        ax.set_facecolor("#0d1520"); ax.spines[:].set_color("#253447")
        ax.tick_params(colors="#5a6a7e")

    for mi, model in enumerate(models):
        mus = [results[lv][model][0] for lv in levels]
        sds = [results[lv][model][1] for lv in levels]
        col = model_cols[model]
        bars = ax_main.bar(x + offsets[mi], mus, width, label=model_labels[model],
                           color=col, alpha=0.85, edgecolor="#253447", linewidth=0.5)
        ax_main.errorbar(x + offsets[mi], mus, yerr=sds, fmt="none",
                         ecolor="#ffffff55", capsize=3, elinewidth=1)
        for xi, mu in zip(x + offsets[mi], mus):
            ax_main.text(xi, mu + 0.008, f"{mu:.0%}", ha="center", va="bottom",
                         fontsize=7, color="#c8d8e8")

    ax_main.set_xticks(x)
    ax_main.set_xticklabels([LEVEL_LABELS[lv] for lv in levels],
                             fontsize=8.5, color="#88aacc")
    ax_main.set_ylim(0.5, 1.02)
    ax_main.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax_main.set_yticklabels(["50%","60%","70%","80%","90%","100%"], fontsize=7, color="#5a6a7e")
    ax_main.axhline(0.798, color="#4a6080", linewidth=0.8, linestyle="--", alpha=0.6)
    ax_main.text(len(levels) - 0.5, 0.802, "baseline (current)", color="#4a6080",
                 fontsize=7, ha="right")
    ax_main.legend(framealpha=0.3, facecolor="#111e2e", edgecolor="#253447",
                   labelcolor="#e2e8f0", fontsize=8.5, loc="lower left")
    ax_main.set_ylabel("Family accuracy (5-fold CV)", color="#5a7a9a", fontsize=9)
    ax_main.set_title("Family classifier across degradation levels\n"
                      "(hard audio + oracle bounds + GT root, 40 songs)",
                      color="#c8d8e8", fontsize=11, pad=6)

    # bottom panel: gate gain over logreg_base per level
    if "gate" in models:
        gate_gains = [results[lv]["gate"][0] - results[lv]["logreg_base"][0]
                      for lv in levels]
        cols = [LEVEL_COLORS[lv] for lv in levels]
        ax_gap.bar(x, gate_gains, color=cols, alpha=0.85, edgecolor="#253447",
                   linewidth=0.5, width=0.5)
        ax_gap.axhline(0, color="#3a5060", linewidth=0.8)
        for xi, g in zip(x, gate_gains):
            ax_gap.text(xi, g + 0.002 if g >= 0 else g - 0.005,
                        f"{g:+.1%}", ha="center",
                        va="bottom" if g >= 0 else "top",
                        fontsize=7, color="#c8d8e8")
        ax_gap.set_xticks(x)
        ax_gap.set_xticklabels([lv for lv in levels], fontsize=8, color="#88aacc")
        ax_gap.set_ylabel("Gate gain\nover base", color="#5a7a9a", fontsize=8)
        ax_gap.set_title("Context gate benefit by degradation level",
                         color="#8899aa", fontsize=9, pad=4)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n→ {OUT_PNG}")
