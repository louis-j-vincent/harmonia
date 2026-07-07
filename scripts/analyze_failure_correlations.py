"""Correlate per-segment features with classifier success/failure.

For each segment we compute:
  - chroma_entropy      : H of softmax(chroma) — low = clean harmonic, high = noisy/spread
  - chroma_peakedness   : max(chroma) / mean(chroma) — high = dominant pitch class
  - ll_margin           : best_LL - 2nd_best_LL (narrow = ambiguous)
  - ll_best_normed      : best_LL / abs(mean_LL) — relative confidence
  - duration            : segment length in seconds
  - n_frames            : number of LTAS frames
  - context_diversity   : fraction of ±4 neighbours with different family (GT)
  - context_same_fam    : fraction of ±4 neighbours same family (GT)
  - prev_interval       : semitone distance to previous chord root (0–6)
  - next_interval       : semitone distance to next chord root (0–6)
  - chroma_temporal_var : std of per-frame chroma — high = changing texture
  - audio_rms           : RMS loudness proxy
  - gt_fam_freq         : how common this family is in the corpus (rarity proxy)

Then for each metric:
  - point-biserial correlation with correct/fail label
  - mean±std split by correct vs fail
  - per-family breakdown

Outputs:
  - console table ranked by |correlation|
  - docs/plots/failure_correlation.png — sorted bar + per-metric violin plots
  - docs/plots/failure_correlation.html — interactive version

Usage:
    .venv/bin/python scripts/analyze_failure_correlations.py
    .venv/bin/python scripts/analyze_failure_correlations.py --n-songs 40
"""
from __future__ import annotations
import argparse, json, sys, warnings
from pathlib import Path
from collections import defaultdict

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
from scipy import stats
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
OUT_PNG    = REPO / "docs" / "plots" / "failure_correlation.png"
OUT_HTML   = REPO / "docs" / "plots" / "failure_correlation.html"

FAMILIES   = ["major", "minor", "diminished", "augmented", "suspended"]
FAM_COLORS = {"major":"#58d4ff","minor":"#a65fd4","diminished":"#e34948",
              "augmented":"#e0a03b","suspended":"#1baf7a"}
HOP        = 512


def _render_hard(midi_path, rng):
    import pretty_midi
    renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")
    pm = pretty_midi.PrettyMIDI(str(midi_path))
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
    mix = time_varying_degrade(mix, sr, rng)
    peak = np.abs(mix).max()
    if peak > 0.99: mix *= 0.99 / peak
    return mix.astype(float), sr


def _diag_ll(x, mu, std):
    return float(-0.5 * np.sum(((x - mu) / std)**2) - np.sum(np.log(std)))


def _ll5(x12, dist):
    lls, keys = [], []
    for fam in FAMILIES:
        best_ll, best_r = -np.inf, 0
        for r in range(12):
            ll = _diag_ll(np.roll(x12, -r), dist[f"{fam}_mu"], dist[f"{fam}_std"])
            if ll > best_ll: best_ll, best_r = ll, r
        lls.append(best_ll); keys.append(best_r)
    return np.array(lls), keys


def collect(n_songs, dist, rng):
    recs = {json.loads(l)["song_id"]: json.loads(l) for l in open(DB)}
    man: dict = {}
    for m in map(json.loads, open(MANIFEST)):
        if m["song_id"] not in man or m.get("transpose", 0) == 0:
            man[m["song_id"]] = m
    avail  = sorted([s for s in recs if s in man], key=lambda s: recs[s]["title"])
    chosen = avail[:n_songs]

    all_segs = []
    for i, sid in enumerate(chosen):
        rec = recs[sid]; m = man[sid]
        bpb = m["beats_per_bar"]; spb = 60.0 / m["tempo"]
        print(f"\r  [{i+1}/{n_songs}] {rec['title'][:40]:40s}", end="", flush=True)
        try:
            audio, sr = _render_hard(REPO / m["midi_path"], rng)
        except Exception:
            continue

        raw  = librosa.feature.chroma_cqt(y=audio, sr=sr, bins_per_octave=36, hop_length=HOP)
        ltas = raw.mean(axis=1, keepdims=True)
        ltas = np.where(ltas < 1e-9, 1.0, ltas)
        chroma = raw / ltas
        ct = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=HOP)

        # RMS of the full mix — we'll slice per segment below
        rms_full = np.sqrt(np.mean(audio**2))

        chord_at = {(e["bar"]-1)*bpb+e["beat"]: e for e in rec["chord_timeline"]}
        song_segs = []

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
            frames_abs     = chroma[:, i0:i1]
            frames_shifted = np.roll(frames_abs, -root, axis=0)

            mean_s = frames_shifted.mean(axis=1)
            nn = np.linalg.norm(mean_s)
            if nn < 1e-9: continue
            x12 = (mean_s / nn).astype(np.float32)

            lls, keys = _ll5(x12, dist)

            # audio RMS for this segment
            a0 = max(0, int(t0 * sr)); a1 = min(len(audio), int(t1 * sr))
            seg_audio = audio[a0:a1]
            audio_rms = float(np.sqrt(np.mean(seg_audio**2) + 1e-12))

            # chroma entropy (within-segment softmax then H)
            c_sm = np.exp(x12 - x12.max()); c_sm /= c_sm.sum()
            chroma_entropy = float(-(c_sm * np.log(c_sm + 1e-12)).sum())

            # chroma peakedness
            chroma_peakedness = float(x12.max() / (x12.mean() + 1e-12))

            # temporal chroma variance
            chroma_temporal_var = float(frames_shifted.std(axis=1).mean())

            # ll margin
            sorted_ll = np.sort(lls)[::-1]
            ll_margin = float(sorted_ll[0] - sorted_ll[1])
            ll_best_normed = float(sorted_ll[0] / (abs(np.mean(lls)) + 1e-9))

            song_segs.append({
                "sid": sid, "title": rec["title"],
                "gt_fam": fam, "gt_fam_i": FAMILIES.index(fam),
                "root": root,
                "x12": x12, "ll5": lls, "keys5": keys,
                "feat": np.concatenate([x12, lls]),
                "duration": float(t1 - t0),
                "n_frames": int(i1 - i0),
                "audio_rms": audio_rms,
                "chroma_entropy": chroma_entropy,
                "chroma_peakedness": chroma_peakedness,
                "chroma_temporal_var": chroma_temporal_var,
                "ll_margin": ll_margin,
                "ll_best_normed": ll_best_normed,
                "t0": t0, "t1": t1,
                "chord_str": p[1],
            })
        all_segs.extend(song_segs)

    print()
    # add context features (need all segs loaded first)
    for idx, seg in enumerate(all_segs):
        neighbours = [all_segs[j] for j in range(max(0,idx-4), min(len(all_segs),idx+5))
                      if j != idx]
        same = sum(1 for n in neighbours if n["gt_fam"] == seg["gt_fam"])
        seg["context_same_fam"]  = same / max(len(neighbours), 1)
        seg["context_diversity"] = 1.0 - seg["context_same_fam"]

        # interval to prev/next chord root (fold to 0–6)
        prev_seg = all_segs[idx-1] if idx > 0 else None
        next_seg = all_segs[idx+1] if idx < len(all_segs)-1 else None
        seg["prev_interval"] = int(min((seg["root"] - prev_seg["root"]) % 12,
                                       (prev_seg["root"] - seg["root"]) % 12)) if prev_seg else 0
        seg["next_interval"] = int(min((seg["root"] - next_seg["root"]) % 12,
                                       (next_seg["root"] - seg["root"]) % 12)) if next_seg else 0

    # family frequency as rarity proxy
    fam_counts = defaultdict(int)
    for s in all_segs: fam_counts[s["gt_fam"]] += 1
    total = len(all_segs)
    for s in all_segs:
        s["gt_fam_freq"] = fam_counts[s["gt_fam"]] / total

    return all_segs


METRICS = [
    ("chroma_entropy",      "Chroma entropy",        "→ 0=clean, high=spread"),
    ("chroma_peakedness",   "Chroma peakedness",     "max/mean: high=dominant pitch"),
    ("ll_margin",           "LL margin (1st-2nd)",   "narrow=ambiguous"),
    ("ll_best_normed",      "LL best (normed)",      "relative to mean LL"),
    ("duration",            "Duration (s)",          "segment length"),
    ("n_frames",            "N frames",              "LTAS frame count"),
    ("audio_rms",           "Audio RMS",             "loudness proxy"),
    ("chroma_temporal_var", "Chroma temporal var",   "within-seg pitch stability"),
    ("context_same_fam",    "Context same-fam frac", "isolation in ±4 context"),
    ("context_diversity",   "Context diversity",     "fraction different fam"),
    ("prev_interval",       "Prev chord interval",   "semitones (0-6)"),
    ("next_interval",       "Next chord interval",   "semitones (0-6)"),
    ("gt_fam_freq",         "Family base rate",      "corpus frequency of GT fam"),
]


def train_predict(segs):
    X = np.stack([s["feat"] for s in segs])
    y = np.array([s["gt_fam_i"] for s in segs])
    sc  = StandardScaler(); Xs = sc.fit_transform(X)
    clf = LogisticRegression(max_iter=2000, solver="lbfgs",
                             class_weight="balanced", C=1.0)
    clf.fit(Xs, y)
    pred = clf.predict(Xs)
    for s, p in zip(segs, pred):
        s["correct"] = int(p == s["gt_fam_i"])
        s["pred_fam_i"] = int(p)
        s["pred_fam"] = FAMILIES[int(p)]
    return float((pred == y).mean())


def correlate(segs):
    correct = np.array([s["correct"] for s in segs], dtype=float)
    rows = []
    for key, label, desc in METRICS:
        vals = np.array([s[key] for s in segs], dtype=float)
        r, p = stats.pointbiserialr(correct, vals)
        mean_corr = vals[correct == 1].mean()
        mean_fail = vals[correct == 0].mean()
        std_corr  = vals[correct == 1].std()
        std_fail  = vals[correct == 0].std()
        rows.append({
            "key": key, "label": label, "desc": desc,
            "r": r, "p": p,
            "mean_corr": mean_corr, "mean_fail": mean_fail,
            "std_corr": std_corr, "std_fail": std_fail,
            "vals": vals,
        })
    rows.sort(key=lambda x: -abs(x["r"]))
    return rows


def per_family_breakdown(segs, key):
    """For each family: mean of metric split by correct/fail."""
    out = {}
    for fam in FAMILIES:
        sub = [s for s in segs if s["gt_fam"] == fam]
        if not sub: out[fam] = (0, 0, 0); continue
        vals  = np.array([s[key] for s in sub])
        corr  = np.array([s["correct"] for s in sub])
        out[fam] = (
            vals[corr == 1].mean() if (corr == 1).any() else float("nan"),
            vals[corr == 0].mean() if (corr == 0).any() else float("nan"),
            len(sub),
        )
    return out


def print_table(rows, segs):
    correct = np.array([s["correct"] for s in segs])
    n_corr = correct.sum(); n_fail = len(segs) - n_corr
    print(f"\n  N={len(segs)}  correct={n_corr} ({n_corr/len(segs):.1%})  "
          f"fail={n_fail} ({n_fail/len(segs):.1%})")
    print(f"\n  {'Metric':26s}  {'r':>7s}  {'p':>7s}  {'mean_corr':>10s}  {'mean_fail':>10s}  desc")
    print("  " + "-"*90)
    for row in rows:
        sig = "***" if row["p"] < 0.001 else "** " if row["p"] < 0.01 else "*  " if row["p"] < 0.05 else "   "
        print(f"  {row['label']:26s}  {row['r']:+7.3f}  {row['p']:7.4f}{sig}"
              f"  {row['mean_corr']:10.3f}  {row['mean_fail']:10.3f}  {row['desc']}")

    print("\nPer-family accuracy:")
    fam_segs = defaultdict(list)
    for s in segs: fam_segs[s["gt_fam"]].append(s)
    for fam in FAMILIES:
        sub = fam_segs[fam]
        if not sub: continue
        acc = np.mean([s["correct"] for s in sub])
        print(f"  {fam:12s}: {acc:.1%}  n={len(sub)}")


def plot_png(rows, segs, out: Path):
    n_metrics = len(rows)
    # top: sorted correlation bar; bottom: violin grid for top-8 metrics
    top8 = rows[:8]

    fig = plt.figure(figsize=(16, 14), facecolor="#0d1520")
    gs  = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1, 2.5], hspace=0.45)

    # ── top: correlation bar chart ────────────────────────────────────────────
    ax_bar = fig.add_subplot(gs[0])
    ax_bar.set_facecolor("#0d1520")
    labels = [r["label"] for r in rows]
    rs     = [r["r"]     for r in rows]
    ps     = [r["p"]     for r in rows]
    cols   = ["#58d4ff" if r > 0 else "#e34948" for r in rs]
    alphas = [1.0 if p < 0.05 else 0.4 for p in ps]
    bars = ax_bar.bar(range(n_metrics), rs, color=cols, alpha=0.8, width=0.7,
                      edgecolor="#253447", linewidth=0.5)
    for i, (bar, alpha, p) in enumerate(zip(bars, alphas, ps)):
        bar.set_alpha(alpha)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        if sig:
            ax_bar.text(i, rs[i] + (0.01 if rs[i] >= 0 else -0.015), sig,
                        ha="center", va="bottom" if rs[i] >= 0 else "top",
                        fontsize=7, color="#e2e8f0")
    ax_bar.axhline(0, color="#3a5060", linewidth=0.8)
    ax_bar.set_xticks(range(n_metrics))
    ax_bar.set_xticklabels(labels, rotation=30, ha="right", fontsize=8, color="#88aacc")
    ax_bar.tick_params(axis="y", colors="#5a6a7e", labelsize=7)
    ax_bar.spines[:].set_color("#253447")
    ax_bar.set_ylabel("Point-biserial r\n(positive = corr w/ correct)", color="#5a7a9a", fontsize=8)
    ax_bar.set_title("Feature correlations with correct classification\n"
                     "(filled = p<0.05, faded = not significant)",
                     color="#c8d8e8", fontsize=10, pad=6)

    # ── bottom: violin plots for top-8 ────────────────────────────────────────
    gs2 = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=gs[1],
                                           hspace=0.55, wspace=0.4)
    correct_flag = np.array([s["correct"] for s in segs])

    for idx, row in enumerate(top8):
        row_idx, col_idx = divmod(idx, 4)
        ax = fig.add_subplot(gs2[row_idx, col_idx])
        ax.set_facecolor("#0d1520")
        ax.spines[:].set_color("#253447")
        ax.tick_params(colors="#5a6a7e", labelsize=6)

        vals = row["vals"]
        v_corr = vals[correct_flag == 1]
        v_fail = vals[correct_flag == 0]

        # violin
        parts = ax.violinplot([v_fail, v_corr], positions=[0, 1],
                               showmedians=True, showextrema=False)
        for pc, col in zip(parts["bodies"], ["#e34948", "#58d4ff"]):
            pc.set_facecolor(col); pc.set_alpha(0.55); pc.set_edgecolor("#253447")
        parts["cmedians"].set_color("#ffffff"); parts["cmedians"].set_linewidth(1.0)

        # scatter jitter
        rng_j = np.random.default_rng(0)
        for xi, (v, col) in enumerate([(v_fail, "#e34948"), (v_corr, "#58d4ff")]):
            jitter = rng_j.uniform(-0.12, 0.12, len(v))
            ax.scatter(xi + jitter, v, s=2, alpha=0.3, color=col, linewidths=0)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["fail", "corr"], fontsize=7, color="#88aacc")
        ax.set_title(f"{row['label']}\nr={row['r']:+.3f}{'*' if row['p']<0.05 else ''}",
                     color="#c8d8e8", fontsize=7.5, pad=3)
        ax.set_ylabel(row["key"][:12], color="#5a7a9a", fontsize=6)

    fig.suptitle("Failure correlation analysis — hard audio, oracle bounds, GT root\n"
                 "in-sample LogReg predictions (17d: 12d chroma + 5d LL)",
                 color="#e2e8f0", fontsize=11, y=1.01)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"→ {out}")


def plot_html(rows, segs, out: Path):
    """HTML: per-family breakdown table + per-metric sorted scatter."""
    correct_flag = np.array([s["correct"] for s in segs])
    fam_segs = defaultdict(list)
    for s in segs: fam_segs[s["gt_fam"]].append(s)

    # family summary table
    fam_rows_html = ""
    for fam in FAMILIES:
        sub = fam_segs[fam]
        if not sub: continue
        acc = np.mean([s["correct"] for s in sub])
        col = FAM_COLORS[fam]
        metrics_html = ""
        for key, label, _ in METRICS[:6]:   # top 6 metrics per family
            mc = np.mean([s[key] for s in sub if s["correct"]])
            mf = np.mean([s[key] for s in sub if not s["correct"]])
            diff = mc - mf
            arrow = "▲" if diff > 0 else "▼"
            diff_col = "#33cc77" if diff > 0 else "#ff5555"
            metrics_html += (f'<td style="color:{diff_col}">{arrow}{abs(diff):.2f}</td>')
        fam_rows_html += (
            f'<tr><td style="color:{col};font-weight:700">{fam}</td>'
            f'<td>{len(sub)}</td><td>{acc:.1%}</td>{metrics_html}</tr>\n'
        )

    metric_header = "".join(f'<th>{label[:14]}</th>' for label, _, _ in
                             [(r["label"], r["key"], r["desc"]) for r in rows[:6]])

    # correlation table rows
    corr_rows_html = ""
    for row in rows:
        sig = "***" if row["p"] < 0.001 else "**" if row["p"] < 0.01 else "*" if row["p"] < 0.05 else ""
        sig_col = "#33cc77" if row["p"] < 0.05 else "#5a6a7e"
        r_col   = "#58d4ff" if row["r"] > 0 else "#e34948"
        bar_w   = int(abs(row["r"]) * 80)
        bar_col = "#58d4ff44" if row["r"] > 0 else "#e3494844"
        corr_rows_html += (
            f'<tr>'
            f'<td style="font-weight:500">{row["label"]}</td>'
            f'<td style="color:{r_col};font-weight:700">{row["r"]:+.3f}'
            f'  <span style="color:{sig_col}">{sig}</span>'
            f'  <span style="display:inline-block;width:{bar_w}px;height:8px;'
            f'background:{bar_col};vertical-align:middle;border-radius:2px"></span>'
            f'</td>'
            f'<td style="color:#8899aa">{row["p"]:.4f}</td>'
            f'<td>{row["mean_corr"]:.3f} ± {row["std_corr"]:.3f}</td>'
            f'<td>{row["mean_fail"]:.3f} ± {row["std_fail"]:.3f}</td>'
            f'<td style="color:#4a607a;font-size:11px">{row["desc"]}</td>'
            f'</tr>\n'
        )

    # top confusions
    confusion_counts = defaultdict(int)
    for s in segs:
        if not s["correct"]:
            confusion_counts[(s["gt_fam"], s["pred_fam"])] += 1
    top_conf = sorted(confusion_counts.items(), key=lambda x: -x[1])[:10]
    conf_rows = ""
    for (gt, pred), cnt in top_conf:
        gc = FAM_COLORS[gt]; pc = FAM_COLORS[pred]
        conf_rows += (
            f'<tr><td style="color:{gc}">{gt}</td>'
            f'<td style="color:{pc}">{pred}</td>'
            f'<td>{cnt}</td></tr>\n'
        )

    # per-metric fail-vs-correct detail for top metrics
    detail_html = ""
    for row in rows[:8]:
        fam_breakdown = ""
        for fam in FAMILIES:
            sub = fam_segs[fam]
            if not sub: continue
            v_c = [s[row["key"]] for s in sub if s["correct"]]
            v_f = [s[row["key"]] for s in sub if not s["correct"]]
            mc = f"{np.mean(v_c):.3f}" if v_c else "—"
            mf = f"{np.mean(v_f):.3f}" if v_f else "—"
            col = FAM_COLORS[fam]
            fam_breakdown += (f'<span style="color:{col}">{fam[:3]}</span>: '
                              f'✓{mc} ✗{mf}  ')
        sig = "***" if row["p"] < 0.001 else "**" if row["p"] < 0.01 else "*" if row["p"] < 0.05 else "(ns)"
        r_col = "#58d4ff" if row["r"] > 0 else "#e34948"
        detail_html += (
            f'<div class="metric-block">'
            f'<div class="metric-name">{row["label"]} '
            f'<span style="color:{r_col}">r={row["r"]:+.3f} {sig}</span></div>'
            f'<div class="metric-desc">{row["desc"]}</div>'
            f'<div class="fam-breakdown">{fam_breakdown}</div>'
            f'</div>\n'
        )

    n_corr = int(correct_flag.sum()); n_fail = len(segs) - n_corr

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Failure Correlation Analysis — Harmonia</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin:0; padding:16px; background:#0d1520; color:#e2e8f0;
         font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; font-size:13px; }}
  h1 {{ color:#c8d8e8; font-size:15px; font-weight:600; margin:0 0 4px;
        padding-bottom:8px; border-bottom:1px solid #253447; }}
  h2 {{ color:#a8b8c8; font-size:13px; font-weight:600; margin:20px 0 8px; }}
  p.note {{ color:#5a7a9a; font-size:11px; margin:0 0 14px; }}
  table {{ border-collapse:collapse; width:100%; margin-bottom:20px; }}
  th {{ background:#111e2e; color:#5a7a9a; font-size:11px; padding:5px 8px;
        text-align:left; border-bottom:1px solid #253447; font-weight:500; }}
  td {{ padding:5px 8px; border-bottom:1px solid #1a2535; font-size:12px; }}
  tr:hover td {{ background:#111e2e; }}
  .metric-block {{ background:#111e2e; border:1px solid #1e3050; border-radius:5px;
                   padding:10px 12px; margin-bottom:8px; }}
  .metric-name {{ font-size:13px; font-weight:600; margin-bottom:3px; }}
  .metric-desc {{ color:#5a7a9a; font-size:11px; margin-bottom:4px; }}
  .fam-breakdown {{ font-size:11px; color:#8899aa; line-height:1.7; }}
  .two-col {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; }}
</style>
</head>
<body>
<h1>Failure Correlation Analysis — {n_corr} correct  {n_fail} fail  (N={len(segs)})</h1>
<p class="note">In-sample LogReg (17d: 12d chroma + 5d LL). Metrics ranked by |point-biserial r| with correct/fail label.</p>

<div class="two-col">
<div>
<h2>Correlation with correct classification (ranked)</h2>
<table>
<tr><th>Metric</th><th>r (+ = correlated with correct)</th><th>p-value</th>
    <th>mean ± std (correct)</th><th>mean ± std (fail)</th><th>Description</th></tr>
{corr_rows_html}
</table>
</div>
<div>
<h2>Top confusions</h2>
<table><tr><th>GT family</th><th>Predicted</th><th>Count</th></tr>
{conf_rows}
</table>

<h2>Per-family accuracy</h2>
<table><tr><th>Family</th><th>N</th><th>Acc</th>
{metric_header}
</tr>
{fam_rows_html}
</table>
<p class="note">Δ columns: correct-mean minus fail-mean for top metrics (▲ = correct segments have higher value)</p>
</div>
</div>

<h2>Per-metric breakdown by family (top 8)</h2>
{detail_html}
</body>
</html>"""

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"→ {out}  ({out.stat().st_size/1024:.0f} KB)")


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
    print(f"Collecting ({args.n_songs} songs, hard audio, oracle bounds)...")
    segs = collect(args.n_songs, dist, rng)
    print(f"  {len(segs)} segments")

    print("Training in-sample classifier...")
    acc = train_predict(segs)
    print(f"  in-sample acc = {acc:.1%}")

    print("Computing correlations...")
    rows = correlate(segs)
    print_table(rows, segs)

    print("\nPlotting...")
    plot_png(rows, segs, OUT_PNG)
    plot_html(rows, segs, OUT_HTML)
