"""Diagnostic grid for family classifier failures on hard audio / oracle boundaries.

For each misclassified segment renders:
  row 1 — LTAS CQT chroma through time (root-shifted, B at top → root at bottom)
  row 2 — mean aggregated chroma (root-shifted bar chart)
  row 3 — max-LL per family (best key labelled), model vote vs GT highlighted

HTML output additionally embeds playable audio snippets alongside the charts.

Usage:
    .venv/bin/python scripts/diagnose_family_failures.py
    .venv/bin/python scripts/diagnose_family_failures.py --n-songs 40 --max-failures 24
"""
from __future__ import annotations
import argparse, base64, io, json, sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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
OUT        = REPO / "docs" / "plots" / "family_failures_diagnostic.png"
OUT_HTML   = REPO / "docs" / "plots" / "family_failures_diagnostic.html"

FAMILIES   = ["major", "minor", "diminished", "augmented", "suspended"]
FAM_SUFFIX = {"major":"^7","minor":"-7","diminished":"o7","augmented":"+","suspended":"sus"}
FAM_COLORS = {"major":"#58d4ff","minor":"#a65fd4","diminished":"#e34948",
              "augmented":"#e0a03b","suspended":"#1baf7a"}
NOTE       = ["C","Db","D","Eb","E","F","F#","G","Ab","A","Bb","B"]
DEGREE     = ["R","b2","2","b3","3","4","b5","5","#5","6","b7","7"]


def _render_hard(midi_path, man_entry, rng):
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
    for name, w in waves.items(): mix[:len(w)] += gains.get(name, 0.5) * w
    mix = time_varying_degrade(mix, sr, rng)
    peak = np.abs(mix).max()
    if peak > 0.99: mix *= 0.99 / peak
    return mix.astype(float), sr


def _ltas_cqt(audio, sr, hop=512):
    raw = librosa.feature.chroma_cqt(y=audio, sr=sr, bins_per_octave=36, hop_length=hop)
    ltas = raw.mean(axis=1, keepdims=True)
    ltas = np.where(ltas < 1e-9, 1.0, ltas)
    chroma = raw / ltas
    ct = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop)
    return chroma, ct


def diag_ll(x, mu, std):
    return float(-0.5 * np.sum(((x - mu) / std) ** 2) - np.sum(np.log(std)))


def max_ll_over_keys(x, mu, std):
    best_ll, best_r = -np.inf, 0
    for r in range(12):
        ll = diag_ll(np.roll(x, -r), mu, std)
        if ll > best_ll: best_ll, best_r = ll, r
    return best_ll, best_r


def collect(n_songs, dist, rng):
    """Returns list of dicts, one per segment, including 2D chroma slice and audio snippet."""
    recs = {json.loads(l)["song_id"]: json.loads(l) for l in open(DB)}
    man: dict = {}
    for m in map(json.loads, open(MANIFEST)):
        if m["song_id"] not in man or m.get("transpose", 0) == 0:
            man[m["song_id"]] = m
    avail = sorted([s for s in recs if s in man], key=lambda s: recs[s]["title"])
    chosen = avail[:n_songs]
    segments = []
    for i, sid in enumerate(chosen):
        rec = recs[sid]; m = man[sid]
        bpb = m["beats_per_bar"]; spb = 60.0 / m["tempo"]
        print(f"\r  [{i+1}/{len(chosen)}] {rec['title'][:38]:38s}", end="", flush=True)
        try:
            audio, sr = _render_hard(REPO / m["midi_path"], m, rng)
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
            seg_raw = chroma[:, i0:i1]           # (12, T) LTAS, absolute pitch
            # root-shift everything
            root = int(root_gt % 12)
            seg_shifted = np.roll(seg_raw, -root, axis=0)   # (12, T) root-shifted
            mean_shifted = seg_shifted.mean(axis=1)
            n = np.linalg.norm(mean_shifted)
            if n < 1e-9: continue
            x12 = mean_shifted / n
            ll5 = []
            keys5 = []
            for f in FAMILIES:
                ll, kr = max_ll_over_keys(x12, dist[f"{f}_mu"], dist[f"{f}_std"])
                ll5.append(ll); keys5.append(kr)
            feat = np.concatenate([x12, ll5])
            # extract raw audio snippet for t0..t1
            a0 = int(t0 * sr)
            a1 = int(t1 * sr)
            audio_snippet = audio[a0:a1].astype(np.float32)
            segments.append({
                "title": rec["title"], "sid": sid,
                "root": root, "gt_fam": fam, "gt_fam_i": FAMILIES.index(fam),
                "x12": x12, "ll5": np.array(ll5), "keys5": keys5,
                "feat": feat,
                "chroma2d": seg_shifted,   # (12, T) root-shifted LTAS
                "t0": t0, "t1": t1,
                "audio_snippet": audio_snippet,
                "sr": sr,
            })
    print()
    return segments


def train_predict(segments):
    X = np.stack([s["feat"] for s in segments])
    y = np.array([s["gt_fam_i"] for s in segments])
    sc  = StandardScaler()
    Xs  = sc.fit_transform(X)
    clf = LogisticRegression(max_iter=1000, solver="lbfgs",
                             class_weight="balanced", C=1.0)
    clf.fit(Xs, y)
    pred = clf.predict(Xs)     # in-sample — good enough to find failure modes
    proba = clf.predict_proba(Xs)
    for s, p, pb in zip(segments, pred, proba):
        s["pred_fam_i"] = int(p)
        s["pred_fam"]   = FAMILIES[int(p)]
        s["pred_proba"] = pb
    acc = (pred == y).mean()
    print(f"  in-sample acc = {acc:.1%}  "
          f"(CV acc from training run: ~80.4% — in-sample is optimistic)")
    failures = [s for s in segments if s["pred_fam_i"] != s["gt_fam_i"]]
    correct  = [s for s in segments if s["pred_fam_i"] == s["gt_fam_i"]]
    return failures, correct


def normalise_ll(ll5_all):
    """Softmax within each segment so bars are within-card comparable (argmax preserved)."""
    ll_arr = np.stack(ll5_all).astype(float)   # (N, 5)
    ll_arr -= ll_arr.max(axis=1, keepdims=True)
    exp = np.exp(ll_arr)
    return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)


def plot_failures(failures, ll_norm_all, out: Path, n_cols=3):
    if not failures:
        print("No failures found."); return

    n = len(failures)
    n_rows = (n + n_cols - 1) // n_cols
    # Each case: 3 sub-rows (heatmap, mean chroma, LL bars) + title
    sub_rows = 3
    fig_h = n_rows * (sub_rows * 1.4 + 0.5)
    fig_w = n_cols * 4.8

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#0d1520")
    fig.suptitle(
        f"Family classifier failures — hard audio, oracle boundaries, GT root shift  "
        f"({n} shown)",
        color="#e2e8f0", fontsize=12, y=1.002)

    outer = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                              hspace=0.55, wspace=0.35)

    for idx, seg in enumerate(failures):
        row, col = divmod(idx, n_cols)
        inner = gridspec.GridSpecFromSubplotSpec(
            sub_rows, 1, subplot_spec=outer[row, col],
            hspace=0.05, height_ratios=[2.5, 1.2, 1.8])

        gt_col   = FAM_COLORS[seg["gt_fam"]]
        pred_col = FAM_COLORS[seg["pred_fam"]]
        gt_tok   = NOTE[seg["root"]] + FAM_SUFFIX[seg["gt_fam"]]
        pred_tok = NOTE[seg["root"]] + FAM_SUFFIX[seg["pred_fam"]]
        dur = seg["t1"] - seg["t0"]

        # ── title ────────────────────────────────────────────────────────────
        ax_title = fig.add_subplot(inner[0])
        ax_title.set_facecolor("#0d1520")
        ax_title.axis("off")

        # chroma heatmap (root-shifted, root at bottom)
        c2d = seg["chroma2d"]
        # downsample time to ≤60 cols
        T = c2d.shape[1]
        if T > 60:
            step = T / 60
            c2d = np.stack([c2d[:, int(j*step):max(int(j*step)+1,
                            int((j+1)*step))].mean(1) for j in range(60)], axis=1)
        # flip so root (index 0) is at BOTTOM, degree 11 at top
        c2d_plot = c2d[::-1]          # row 0 = degree 11, row 11 = root (0)
        vmax = c2d_plot.max() * 0.85 or 1.0
        ax_title.imshow(c2d_plot, aspect="auto", origin="upper",
                        cmap="YlOrRd", vmin=0, vmax=vmax,
                        extent=[-0.5, c2d_plot.shape[1]-0.5, -0.5, 11.5])
        ax_title.set_yticks(range(12))
        # bottom row = degree 0 (root), top row = degree 11
        ax_title.set_yticklabels(DEGREE[::-1], fontsize=6, color="#88aacc")
        ax_title.tick_params(axis="x", labelbottom=False, length=0)

        # ── mean chroma bars ──────────────────────────────────────────────────
        ax_chroma = fig.add_subplot(inner[1])
        ax_chroma.set_facecolor("#0d1520")
        x12 = seg["x12"]
        bar_cols = [gt_col if i in [0] else "#253447" for i in range(12)]
        ax_chroma.bar(range(12), x12, color=bar_cols, width=0.8,
                      edgecolor="#1a2535", linewidth=0.5)
        ax_chroma.set_xticks(range(12))
        ax_chroma.set_xticklabels(DEGREE, fontsize=6, color="#88aacc")
        ax_chroma.tick_params(axis="y", colors="#5a6a7e", labelsize=5)
        ax_chroma.spines[:].set_color("#253447")
        ax_chroma.set_xlim(-0.6, 11.6)
        ax_chroma.set_ylabel("mean\nLTAS", color="#5a6a7e", fontsize=5.5,
                             rotation=0, labelpad=24, va="center")

        # ── LL bars per family ────────────────────────────────────────────────
        ax_ll = fig.add_subplot(inner[2])
        ax_ll.set_facecolor("#0d1520")
        ll_norm = ll_norm_all[idx]      # (5,) normalised
        raw_keys = seg["keys5"]
        for fi, fam in enumerate(FAMILIES):
            col = FAM_COLORS[fam]
            is_gt   = fam == seg["gt_fam"]
            is_pred = fam == seg["pred_fam"]
            alpha = 0.9 if (is_gt or is_pred) else 0.45
            edge_col = "#ffffff" if is_pred else (col if is_gt else "#1a2535")
            edge_w   = 1.5 if is_pred else (1.0 if is_gt else 0.4)
            ax_ll.bar(fi, ll_norm[fi], color=col, alpha=alpha,
                      edgecolor=edge_col, linewidth=edge_w, width=0.7)
            # best-key label
            key_str = NOTE[(seg["root"] + raw_keys[fi]) % 12]
            ax_ll.text(fi, ll_norm[fi] + 0.04, key_str,
                       ha="center", va="bottom", fontsize=6.5,
                       color=col, fontweight="bold")
            # GT / pred markers
            if is_gt:
                ax_ll.text(fi, -0.18, "GT", ha="center", va="top",
                           fontsize=5.5, color=gt_col, fontweight="bold")
            if is_pred:
                ax_ll.text(fi, -0.32, "pred", ha="center", va="top",
                           fontsize=5.5, color=pred_col)

        fam_short = [f[:3] for f in FAMILIES]
        ax_ll.set_xticks(range(5))
        ax_ll.set_xticklabels(fam_short, fontsize=6.5, color="#88aacc")
        ax_ll.tick_params(axis="y", colors="#5a6a7e", labelsize=5)
        ax_ll.spines[:].set_color("#253447")
        ax_ll.set_xlim(-0.6, 4.6); ax_ll.set_ylim(-0.45, 1.45)
        ax_ll.set_yticks([0, 0.5, 1.0])
        ax_ll.set_yticklabels(["lo","","hi"], fontsize=5, color="#5a6a7e")
        ax_ll.set_ylabel("norm\nLL", color="#5a6a7e", fontsize=5.5,
                         rotation=0, labelpad=18, va="center")

        # Compound title drawn on the heatmap axis
        ax_title.set_title(
            f"GT: {gt_tok}  →  pred: {pred_tok}   [{dur:.1f}s]   {seg['title'][:28]}",
            color="#e2e8f0", fontsize=7.5, pad=3)

    plt.tight_layout(rect=[0, 0, 1, 0.998])
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"→ {out}")


# ─────────────────────────────────────────────────────────────────────────────
# HTML output helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fig_to_b64(fig, w_px: int, h_px: int, dpi: int = 100) -> str:
    """Render a matplotlib Figure into a base64 PNG string at the given pixel size."""
    fig.set_size_inches(w_px / dpi, h_px / dpi)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _audio_to_b64_wav(audio: np.ndarray, sr: int) -> str:
    """Encode a float32 audio array as a base64 WAV string (PCM 16-bit)."""
    import scipy.io.wavfile as wavfile
    # Convert float32 to int16 for WAV
    pcm = np.clip(audio, -1.0, 1.0)
    pcm_i16 = (pcm * 32767).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, sr, pcm_i16)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _render_heatmap_b64(seg: dict, gt_col: str) -> str:
    """Render the LTAS chroma heatmap panel as a base64 PNG (300×80px)."""
    fig, ax = plt.subplots(facecolor="#0d1520")
    ax.set_facecolor("#0d1520")
    c2d = seg["chroma2d"].copy()
    T = c2d.shape[1]
    if T > 60:
        step = T / 60
        c2d = np.stack([c2d[:, int(j*step):max(int(j*step)+1,
                        int((j+1)*step))].mean(1) for j in range(60)], axis=1)
    c2d_plot = c2d[::-1]
    vmax = c2d_plot.max() * 0.85 or 1.0
    ax.imshow(c2d_plot, aspect="auto", origin="upper",
              cmap="YlOrRd", vmin=0, vmax=vmax,
              extent=[-0.5, c2d_plot.shape[1]-0.5, -0.5, 11.5])
    ax.set_yticks(range(12))
    ax.set_yticklabels(DEGREE[::-1], fontsize=5, color="#88aacc")
    ax.tick_params(axis="x", labelbottom=False, length=0)
    ax.tick_params(axis="y", length=2, color="#253447")
    for spine in ax.spines.values():
        spine.set_color("#253447")
    fig.tight_layout(pad=0.3)
    return _fig_to_b64(fig, 300, 80)


def _render_chroma_b64(seg: dict, gt_col: str) -> str:
    """Render the mean chroma bar chart as a base64 PNG (300×100px)."""
    fig, ax = plt.subplots(facecolor="#0d1520")
    ax.set_facecolor("#0d1520")
    x12 = seg["x12"]
    bar_cols = [gt_col if i == 0 else "#253447" for i in range(12)]
    ax.bar(range(12), x12, color=bar_cols, width=0.8,
           edgecolor="#1a2535", linewidth=0.5)
    ax.set_xticks(range(12))
    ax.set_xticklabels(DEGREE, fontsize=5, color="#88aacc")
    ax.tick_params(axis="y", colors="#5a6a7e", labelsize=4)
    ax.spines[:].set_color("#253447")
    ax.set_xlim(-0.6, 11.6)
    ax.set_ylabel("LTAS", color="#5a6a7e", fontsize=5)
    fig.tight_layout(pad=0.3)
    return _fig_to_b64(fig, 300, 100)


def _render_ll_b64(seg: dict, ll_norm: np.ndarray) -> str:
    """Two-row chart: top = softmax(LL) within segment, bottom = LogReg pred_proba."""
    gt_col   = FAM_COLORS[seg["gt_fam"]]
    pred_col = FAM_COLORS[seg["pred_fam"]]
    fig, (ax_ll, ax_pr) = plt.subplots(2, 1, facecolor="#0d1520",
                                        gridspec_kw={"hspace": 0.55})
    for ax in (ax_ll, ax_pr):
        ax.set_facecolor("#0d1520")
        ax.spines[:].set_color("#253447")
        ax.tick_params(axis="y", colors="#5a6a7e", labelsize=4)

    raw_keys = seg["keys5"]
    ll_argmax = int(np.argmax(ll_norm))
    pr_argmax = int(np.argmax(seg["pred_proba"]))

    # ── top: softmax(LL) ──────────────────────────────────────────────────────
    for fi, fam in enumerate(FAMILIES):
        col = FAM_COLORS[fam]
        is_gt   = fam == seg["gt_fam"]
        is_pred = fam == seg["pred_fam"]
        is_ll_win = fi == ll_argmax
        alpha = 1.0 if (is_gt or is_ll_win) else 0.4
        edge_col = "#ffffff" if is_ll_win else (col if is_gt else "#1a2535")
        edge_w   = 1.5 if is_ll_win else (0.8 if is_gt else 0.3)
        ax_ll.bar(fi, ll_norm[fi], color=col, alpha=alpha,
                  edgecolor=edge_col, linewidth=edge_w, width=0.7)
        key_str = NOTE[(seg["root"] + raw_keys[fi]) % 12]
        ax_ll.text(fi, ll_norm[fi] + 0.02, key_str,
                   ha="center", va="bottom", fontsize=5.5,
                   color=col, fontweight="bold")
        if is_gt:
            ax_ll.text(fi, -0.08, "GT", ha="center", va="top",
                       fontsize=4.5, color=gt_col, fontweight="bold")
    ax_ll.set_xlim(-0.6, 4.6); ax_ll.set_ylim(-0.18, 0.82)
    ax_ll.set_xticks(range(5))
    ax_ll.set_xticklabels([f[:3] for f in FAMILIES], fontsize=5.5, color="#88aacc")
    ax_ll.set_yticks([0, 0.5]); ax_ll.set_yticklabels(["0","0.5"], fontsize=4, color="#5a6a7e")
    ax_ll.set_ylabel("softmax\n(LL)", color="#5a6a7e", fontsize=4.5, rotation=0,
                     labelpad=22, va="center")
    ax_ll.set_title("↑ LL signal     ↓ LogReg decision", color="#4a607a",
                    fontsize=5, pad=2)

    # ── bottom: pred_proba from LogReg ────────────────────────────────────────
    proba = seg["pred_proba"]
    for fi, fam in enumerate(FAMILIES):
        col = FAM_COLORS[fam]
        is_gt   = fam == seg["gt_fam"]
        is_pred = fam == seg["pred_fam"]
        alpha = 1.0 if (is_gt or is_pred) else 0.4
        edge_col = "#ffffff" if is_pred else (col if is_gt else "#1a2535")
        edge_w   = 1.5 if is_pred else (0.8 if is_gt else 0.3)
        ax_pr.bar(fi, proba[fi], color=col, alpha=alpha,
                  edgecolor=edge_col, linewidth=edge_w, width=0.7)
        if is_gt:
            ax_pr.text(fi, -0.06, "GT", ha="center", va="top",
                       fontsize=4.5, color=gt_col, fontweight="bold")
        if is_pred:
            ax_pr.text(fi, -0.12, "pred", ha="center", va="top",
                       fontsize=4.5, color=pred_col)
    ax_pr.set_xlim(-0.6, 4.6); ax_pr.set_ylim(-0.22, 1.05)
    ax_pr.set_xticks(range(5))
    ax_pr.set_xticklabels([f[:3] for f in FAMILIES], fontsize=5.5, color="#88aacc")
    ax_pr.set_yticks([0, 0.5, 1.0]); ax_pr.set_yticklabels(["0","","1"], fontsize=4, color="#5a6a7e")
    ax_pr.set_ylabel("pred\nproba", color="#5a6a7e", fontsize=4.5, rotation=0,
                     labelpad=22, va="center")

    fig.tight_layout(pad=0.3)
    return _fig_to_b64(fig, 300, 140)


def _make_card(seg: dict, ll_norm: np.ndarray, is_failure: bool) -> str:
    gt_col    = FAM_COLORS[seg["gt_fam"]]
    pred_col  = FAM_COLORS[seg["pred_fam"]]
    # GT root is always seg["root"]; predicted root is the best-key offset for pred family
    pred_root = (seg["root"] + seg["keys5"][seg["pred_fam_i"]]) % 12
    gt_tok    = NOTE[seg["root"]]  + FAM_SUFFIX[seg["gt_fam"]]
    pred_tok  = NOTE[pred_root]    + FAM_SUFFIX[seg["pred_fam"]]
    dur      = seg["t1"] - seg["t0"]

    hm_b64     = _render_heatmap_b64(seg, gt_col)
    chroma_b64 = _render_chroma_b64(seg, gt_col)
    ll_b64     = _render_ll_b64(seg, ll_norm)
    wav_b64    = _audio_to_b64_wav(seg["audio_snippet"], seg["sr"])

    border_col = "#3a1515" if is_failure else "#0f2b1a"
    badge_bg   = "#5a1a1a" if is_failure else "#0e2e1a"
    badge_txt  = "FAIL" if is_failure else "OK"
    badge_col  = "#ff5555" if is_failure else "#33cc77"

    arrow_html = (
        f'<span style="color:{gt_col};font-weight:700">{gt_tok}</span>'
        f'<span style="color:#8899aa"> → </span>'
        f'<span style="color:{pred_col};font-weight:700">{pred_tok}</span>'
    ) if is_failure else (
        f'<span style="color:{gt_col};font-weight:700">{gt_tok}</span>'
        f'<span style="color:#22aa55"> ✓</span>'
    )

    return f"""
<div class="card" style="border-color:{border_col}">
  <div class="card-title">
    <span class="badge" style="background:{badge_bg};color:{badge_col}">{badge_txt}</span>
    {arrow_html}
    <span class="dur"> [{dur:.1f}s]</span>
    <span class="songname"> {seg['title'][:38]}</span>
  </div>
  <div class="panel-label">LTAS chroma (root at bottom)</div>
  <img src="data:image/png;base64,{hm_b64}" width="300" height="80" alt="heatmap">
  <div class="panel-label">Mean chroma</div>
  <img src="data:image/png;base64,{chroma_b64}" width="300" height="100" alt="chroma">
  <div class="panel-label">LL softmax (top) + LogReg pred proba (bottom, label = absolute key)</div>
  <img src="data:image/png;base64,{ll_b64}" width="300" height="140" alt="ll">
  <div class="panel-label">Audio snippet</div>
  <audio controls preload="auto" style="width:300px;margin-top:4px">
    <source src="data:audio/wav;base64,{wav_b64}" type="audio/wav">
  </audio>
</div>"""


def plot_failures_html(failures: list, correct: list,
                       ll_norm_failures: np.ndarray, ll_norm_correct: np.ndarray,
                       out_html: Path, n_correct: int = 6) -> None:
    """HTML: failures first, then a comparison sample of correct predictions."""
    if not failures:
        print("No failures — skipping HTML output.")
        return

    cards_html = []
    # failures section
    for idx, seg in enumerate(failures):
        cards_html.append(_make_card(seg, ll_norm_failures[idx], is_failure=True))

    # correct comparison section
    from collections import defaultdict
    by_fam: dict = defaultdict(list)
    for s in correct:
        by_fam[s["gt_fam"]].append(s)
    correct_sample = []
    per_fam = max(1, n_correct // max(len(by_fam), 1))
    for fam in FAMILIES:
        correct_sample.extend(by_fam[fam][:per_fam])
    correct_sample = correct_sample[:n_correct]

    if correct_sample:
        ll_norm_cs = normalise_ll([s["ll5"] for s in correct_sample])
        for idx, seg in enumerate(correct_sample):
            cards_html.append(_make_card(seg, ll_norm_cs[idx], is_failure=False))

    all_cards = "\n".join(cards_html)
    n_fail = len(failures); n_corr = len(correct_sample)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Family Classifier Failures — Harmonia</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 16px;
    background: #0d1520; color: #e2e8f0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-size: 13px;
  }}
  h1 {{
    color: #c8d8e8; font-size: 15px; font-weight: 600;
    margin: 0 0 4px; padding-bottom: 8px;
    border-bottom: 1px solid #253447;
  }}
  p.note {{ color: #5a7a9a; font-size: 11px; margin: 0 0 14px; }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(3, 320px);
    gap: 16px;
  }}
  .card {{
    background: #111e2e;
    border: 1px solid #1e3050;
    border-radius: 6px;
    padding: 10px 10px 8px;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }}
  .card-title {{
    font-size: 12px;
    margin-bottom: 6px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}
  .badge {{
    display: inline-block;
    padding: 1px 5px;
    border-radius: 3px;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.05em;
    margin-right: 5px;
    vertical-align: middle;
  }}
  .dur {{ color: #6a80a0; }}
  .songname {{ color: #8899aa; font-style: italic; }}
  .panel-label {{
    color: #4a607a;
    font-size: 10px;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }}
  img {{
    display: block;
    border-radius: 3px;
    max-width: 100%;
  }}
  audio {{
    filter: invert(0.85) hue-rotate(180deg);
  }}
</style>
</head>
<body>
<h1>Family classifier — {n_fail} failures + {n_corr} correct comparisons
  &nbsp;(hard audio, oracle boundaries, GT root shift)</h1>
<p class="note">Key labels on LL bars = absolute pitch class (= root + best_roll mod 12).
  Failures first (red border), correct predictions below (green border).</p>
<div class="grid">
{all_cards}
</div>
</body>
</html>"""

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    size_kb = out_html.stat().st_size / 1024
    print(f"→ {out_html}  ({size_kb:.0f} KB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-songs",      type=int, default=30)
    ap.add_argument("--max-failures", type=int, default=21)
    ap.add_argument("--seed",         type=int, default=42)
    ap.add_argument("--n-cols",       type=int, default=3)
    ap.add_argument("--n-correct",    type=int, default=6,
                    help="correct prediction examples to include for comparison")
    ap.add_argument("--out",          default=None)
    ap.add_argument("--out-html",     default=None)
    args = ap.parse_args()

    out      = Path(args.out)      if args.out      else OUT
    out_html = Path(args.out_html) if args.out_html else OUT_HTML

    if not DIST_CACHE.exists():
        print("ERROR: run plot_family_likelihood.py --rebuild-cache first"); exit(1)
    d = np.load(DIST_CACHE)
    dist = {k: d[k] for k in d.files}

    rng = np.random.default_rng(args.seed)
    print(f"Collecting hard-audio oracle segments ({args.n_songs} songs)...")
    segs = collect(args.n_songs, dist, rng)
    print(f"  {len(segs)} segments collected")

    print("Training classifier + finding failures...")
    failures, correct = train_predict(segs)
    print(f"  {len(failures)} failures  ({len(failures)/len(segs):.1%} error rate)")

    # Sample up to max_failures, stratified by (gt_fam, pred_fam) pair
    from collections import defaultdict
    by_pair = defaultdict(list)
    for f in failures:
        by_pair[(f["gt_fam"], f["pred_fam"])].append(f)
    sampled = []
    pairs_sorted = sorted(by_pair.items(), key=lambda x: -len(x[1]))
    per_pair = max(1, args.max_failures // max(len(by_pair), 1))
    for (gt, pred), cases in pairs_sorted:
        sampled.extend(cases[:per_pair])
        if len(sampled) >= args.max_failures: break
    sampled = sampled[:args.max_failures]

    print(f"  Plotting {len(sampled)} failures ({len(by_pair)} distinct error types)")
    for (gt, pred), cases in pairs_sorted[:8]:
        print(f"    {gt:12s} → {pred:12s}: {len(cases):3d}")

    ll_norm_all = normalise_ll([s["ll5"] for s in sampled])
    plot_failures(sampled, ll_norm_all, out, n_cols=args.n_cols)
    plot_failures_html(sampled, correct, ll_norm_all,
                       normalise_ll([s["ll5"] for s in sampled]),
                       out_html, n_correct=args.n_correct)
