"""Exact-quality tree-search diagnostic HTML.

Greedy top-down tree search (FAMILY → BASE7 → EXACT):
  At each node, pick the child with the highest max-LL over 12 keys.
  The winning key found at L0 is treated as the predicted root — no separate
  root detector needed. If the GT root != predicted root, that's a root error,
  and we report it separately from the quality error.

Targets EXACT quality (18 classes), but reports errors at all 3 levels.
Outputs an HTML file with failures + correct comparisons, sorted by gt_exact,
each card showing:
  - chroma heatmap (root-shifted to GT root)
  - mean chroma bar chart
  - tree path taken (which child won at each level, highlighted)
  - audio snippet

Requires: data/cache/chord_tree_ltas.npz  (build with chord_tree_ltas.py --n-songs 60)

Usage:
    .venv/bin/python scripts/diagnose_exact_tree.py
    .venv/bin/python scripts/diagnose_exact_tree.py --n-songs 30 --max-failures 30
"""
from __future__ import annotations
import argparse, base64, io, json, sys, warnings
from collections import defaultdict
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

from analyze_accomp_emission import parse_chord, song_chord_spans
from build_accomp_audio_hard import (
    SCENARIOS, SOUNDFONTS, LEAD_PROGRAMS,
    make_melody, render_to_array, stem_midi, time_varying_degrade,
)
from build_audio_chord_features import BUCKET_FAMILY, BUCKET_BASE7
from harmonia.data.midi_renderer import MIDIRenderer

DB         = REPO / "data" / "accomp_db" / "db.jsonl"
MANIFEST   = REPO / "data" / "accomp_db" / "audio" / "manifest.jsonl"
TREE_CACHE = REPO / "data" / "cache" / "chord_tree_ltas.npz"
OUT_HTML   = REPO / "docs" / "plots" / "exact_tree_diagnostic.html"

NOTE   = ["C","Db","D","Eb","E","F","F#","G","Ab","A","Bb","B"]
DEGREE = ["R","b2","2","b3","3","4","b5","5","#5","6","b7","7"]

TREE = {
    "major":      {"majT":["maj","6"], "maj7":["maj7"], "dom7":["dom7","dom7alt"]},
    "minor":      {"minT":["min","m6"], "min7":["min7"], "minmaj7":["minmaj7"]},
    "diminished": {"dimT":["dim"], "dim7":["dim7"], "m7b5":["m7b5"]},
    "augmented":  {"augT":["aug"], "aug7":["aug7"], "augmaj7":["augmaj7"]},
    "suspended":  {"susT":["sus2","sus4"], "7sus4":["7sus4"]},
}
FAM_COLORS = {
    "major":"#58d4ff","minor":"#a65fd4","diminished":"#e34948",
    "augmented":"#e0a03b","suspended":"#1baf7a",
}
BASE7_FAM = {b7:fam for fam,ch in TREE.items() for b7 in ch}
EXACT_B7  = {ex:b7 for fam,ch in TREE.items() for b7,exs in ch.items() for ex in exs}
EXACT_FAM = {ex:fam for fam,ch in TREE.items()
             for b7,exs in ch.items() for ex in exs}

EXACT_DISPLAY = {
    "maj":"maj", "6":"6", "maj7":"maj7", "dom7":"7", "dom7alt":"7alt",
    "min":"min", "m6":"m6", "min7":"min7", "minmaj7":"mΔ7",
    "dim":"dim", "dim7":"dim7", "m7b5":"ø7",
    "aug":"aug", "aug7":"aug7", "augmaj7":"augΔ7",
    "sus2":"sus2", "sus4":"sus4", "7sus4":"7sus4",
}


# ── helpers ──────────────────────────────────────────────────────────────────

def _ltas_cqt(audio, sr, hop=512):
    raw = librosa.feature.chroma_cqt(y=audio, sr=sr, bins_per_octave=36, hop_length=hop)
    ltas = raw.mean(axis=1, keepdims=True)
    ltas = np.where(ltas < 1e-9, 1.0, ltas)
    chroma = raw / ltas
    ct = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop)
    return chroma, ct


def _diag_ll(x, mu, std):
    return float(-0.5 * np.sum(((x - mu) / std) ** 2) - np.sum(np.log(std)))


def _max_ll_over_keys(x, mu, std):
    best_ll, best_r = -np.inf, 0
    for r in range(12):
        ll = _diag_ll(np.roll(x, -r), mu, std)
        if ll > best_ll: best_ll, best_r = ll, r
    return best_ll, best_r


def greedy_tree_search(x12, dist):
    """Returns (pred_fam, pred_b7, pred_exact, pred_root_pc, path_scores)."""
    # L0
    fam_scores = {}; fam_roots = {}
    for fam in TREE:
        ll, r = _max_ll_over_keys(x12, dist[f"fam_{fam}_mu"], dist[f"fam_{fam}_std"])
        fam_scores[fam] = ll; fam_roots[fam] = r
    best_fam  = max(fam_scores, key=fam_scores.__getitem__)
    pred_root = fam_roots[best_fam]

    # L1
    b7_scores = {}
    for b7 in TREE[best_fam]:
        ll, _ = _max_ll_over_keys(x12, dist[f"b7_{b7}_mu"], dist[f"b7_{b7}_std"])
        b7_scores[b7] = ll
    best_b7 = max(b7_scores, key=b7_scores.__getitem__)

    # L2
    ex_scores = {}
    for ex in TREE[best_fam][best_b7]:
        ll, _ = _max_ll_over_keys(x12, dist[f"exact_{ex}_mu"], dist[f"exact_{ex}_std"])
        ex_scores[ex] = ll
    best_ex = max(ex_scores, key=ex_scores.__getitem__)

    return best_fam, best_b7, best_ex, pred_root, {
        "fam": fam_scores, "b7": b7_scores, "ex": ex_scores,
    }


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


# ── data collection ──────────────────────────────────────────────────────────

def collect(n_songs, dist, rng):
    recs = {json.loads(l)["song_id"]: json.loads(l) for l in open(DB)}
    man: dict = {}
    for m in map(json.loads, open(MANIFEST)):
        if m["song_id"] not in man or m.get("transpose", 0) == 0:
            man[m["song_id"]] = m
    avail = sorted([s for s in recs if s in man], key=lambda s: recs[s]["title"])
    chosen = avail[:n_songs]

    segs = []
    for i, sid in enumerate(chosen):
        rec = recs[sid]; m = man[sid]
        bpb = m["beats_per_bar"]; spb = 60.0 / m["tempo"]
        print(f"\r  [{i+1}/{len(chosen)}] {rec['title'][:38]:38s}", end="", flush=True)
        try:
            audio, sr = _render_hard(REPO / m["midi_path"], rng)
        except Exception:
            continue
        chroma, ct = _ltas_cqt(audio, sr)
        chord_at = {(e["bar"]-1)*bpb+e["beat"]: e for e in rec["chord_timeline"]}
        for t0, t1, root_gt, _ in song_chord_spans(rec):
            b0  = int(round(t0 / spb))
            mma = chord_at.get(b0, {}).get("mma")
            p   = parse_chord(mma) if mma else None
            if p is None or p[1] not in EXACT_FAM: continue
            gt_exact = p[1]; gt_b7 = EXACT_B7[gt_exact]; gt_fam = EXACT_FAM[gt_exact]
            root = int(root_gt % 12)
            i0 = int(np.searchsorted(ct, t0)); i1 = int(np.searchsorted(ct, t1))
            if i1 <= i0: i1 = i0 + 1
            seg_raw = chroma[:, i0:i1]
            seg_shifted = np.roll(seg_raw, -root, axis=0)
            mean_s = seg_shifted.mean(axis=1)
            n = np.linalg.norm(mean_s)
            if n < 1e-9: continue
            x12 = mean_s / n
            pred_fam, pred_b7, pred_ex, pred_root, path_scores = greedy_tree_search(x12, dist)
            audio_snippet = audio[int(t0*sr):int(t1*sr)].astype(np.float32)
            segs.append({
                "title": rec["title"],
                "root": root,
                "gt_fam": gt_fam, "gt_b7": gt_b7, "gt_exact": gt_exact,
                "pred_fam": pred_fam, "pred_b7": pred_b7, "pred_exact": pred_ex,
                "pred_root": pred_root,
                "x12": x12, "chroma2d": seg_shifted,
                "t0": t0, "t1": t1,
                "path_scores": path_scores,
                "audio_snippet": audio_snippet, "sr": sr,
            })
    print()
    return segs


# ── rendering helpers ─────────────────────────────────────────────────────────

def _fig_to_b64(fig, w_px, h_px, dpi=100):
    fig.set_size_inches(w_px / dpi, h_px / dpi)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _audio_b64(audio, sr):
    import scipy.io.wavfile as wf
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    buf = io.BytesIO(); wf.write(buf, sr, pcm); buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _render_heatmap_b64(seg):
    fig, ax = plt.subplots(facecolor="#0d1520")
    ax.set_facecolor("#0d1520")
    c2d = seg["chroma2d"].copy()
    if c2d.shape[1] > 60:
        step = c2d.shape[1] / 60
        c2d = np.stack([c2d[:, int(j*step):max(int(j*step)+1,
                        int((j+1)*step))].mean(1) for j in range(60)], axis=1)
    c2d_p = c2d[::-1]
    ax.imshow(c2d_p, aspect="auto", origin="upper", cmap="YlOrRd",
              vmin=0, vmax=max(c2d_p.max()*0.85, 1e-3),
              extent=[-0.5, c2d_p.shape[1]-0.5, -0.5, 11.5])
    ax.set_yticks(range(12))
    ax.set_yticklabels(DEGREE[::-1], fontsize=5, color="#88aacc")
    ax.tick_params(axis="x", labelbottom=False, length=0)
    for sp in ax.spines.values(): sp.set_color("#253447")
    fig.tight_layout(pad=0.3)
    return _fig_to_b64(fig, 300, 80)


def _render_chroma_b64(seg, col):
    fig, ax = plt.subplots(facecolor="#0d1520")
    ax.set_facecolor("#0d1520")
    ax.bar(range(12), seg["x12"],
           color=[col if i == 0 else "#253447" for i in range(12)],
           width=0.8, edgecolor="#1a2535", linewidth=0.5)
    ax.set_xticks(range(12))
    ax.set_xticklabels(DEGREE, fontsize=5, color="#88aacc")
    ax.tick_params(axis="y", colors="#5a6a7e", labelsize=4)
    ax.spines[:].set_color("#253447"); ax.set_xlim(-0.6, 11.6)
    ax.set_ylabel("LTAS", color="#5a6a7e", fontsize=5)
    fig.tight_layout(pad=0.3)
    return _fig_to_b64(fig, 300, 90)


def _render_tree_path_b64(seg):
    """Visualise the greedy tree path with scores at each level."""
    ps = seg["path_scores"]
    pred_fam = seg["pred_fam"]; pred_b7 = seg["pred_b7"]; pred_ex = seg["pred_exact"]
    gt_fam = seg["gt_fam"]; gt_b7 = seg["gt_b7"]; gt_ex = seg["gt_exact"]

    fig, axes = plt.subplots(1, 3, figsize=(9, 2.2), facecolor="#0d1520")
    fig.subplots_adjust(wspace=0.4)

    # L0 family
    fams = list(ps["fam"].keys())
    ll_f = np.array([ps["fam"][f] for f in fams])
    ll_fn = (ll_f - ll_f.min()) / (ll_f.max() - ll_f.min() + 1e-12)
    ax = axes[0]; ax.set_facecolor("#0d1520")
    for i, fam in enumerate(fams):
        col = FAM_COLORS[fam]
        is_pred = fam == pred_fam; is_gt = fam == gt_fam
        alpha = 0.9 if is_pred else 0.3
        ew = 2.0 if is_pred else (1.0 if is_gt else 0.3)
        ec = "#ffffff" if is_pred else (col if is_gt else "#1a2535")
        ax.bar(i, ll_fn[i], color=col, alpha=alpha, edgecolor=ec, linewidth=ew, width=0.7)
        if is_pred: ax.text(i, ll_fn[i]+0.05, "▲", ha="center", fontsize=6, color=col)
        if is_gt and not is_pred: ax.text(i, -0.22, "GT", ha="center",
                                          fontsize=5, color=col, fontweight="bold")
    ax.set_xticks(range(len(fams)))
    ax.set_xticklabels([f[:3] for f in fams], fontsize=5.5, color="#88aacc")
    ax.set_ylim(-0.35, 1.5); ax.tick_params(axis="y", colors="#5a6a7e", labelsize=4)
    ax.spines[:].set_color("#253447")
    ax.set_title("L0 family", color="#8899aa", fontsize=7, pad=2)

    # L1 base7 (only children of pred_fam)
    b7s = list(ps["b7"].keys())
    ll_b = np.array([ps["b7"][b] for b in b7s])
    ll_bn = (ll_b - ll_b.min()) / (ll_b.max() - ll_b.min() + 1e-12)
    ax = axes[1]; ax.set_facecolor("#0d1520")
    col = FAM_COLORS[pred_fam]
    for i, b7 in enumerate(b7s):
        is_pred = b7 == pred_b7; is_gt = b7 == gt_b7 and gt_fam == pred_fam
        alpha = 0.9 if is_pred else 0.3
        ew = 2.0 if is_pred else (1.0 if is_gt else 0.3)
        ec = "#ffffff" if is_pred else (col if is_gt else "#1a2535")
        ax.bar(i, ll_bn[i], color=col, alpha=alpha, edgecolor=ec, linewidth=ew, width=0.7)
        if is_pred: ax.text(i, ll_bn[i]+0.05, "▲", ha="center", fontsize=6, color=col)
        if is_gt and not is_pred: ax.text(i, -0.22, "GT", ha="center",
                                          fontsize=5, color=col, fontweight="bold")
    ax.set_xticks(range(len(b7s)))
    ax.set_xticklabels(b7s, fontsize=5.5, color="#88aacc", rotation=15)
    ax.set_ylim(-0.35, 1.5); ax.tick_params(axis="y", colors="#5a6a7e", labelsize=4)
    ax.spines[:].set_color("#253447")
    ax.set_title(f"L1 base7 (within {pred_fam[:3]})", color="#8899aa", fontsize=7, pad=2)

    # L2 exact
    exs = list(ps["ex"].keys())
    ll_e = np.array([ps["ex"][e] for e in exs])
    ll_en = (ll_e - ll_e.min()) / (ll_e.max() - ll_e.min() + 1e-12)
    ax = axes[2]; ax.set_facecolor("#0d1520")
    for i, ex in enumerate(exs):
        is_pred = ex == pred_ex; is_gt = ex == gt_ex and gt_b7 == pred_b7 and gt_fam == pred_fam
        alpha = 0.9 if is_pred else 0.3
        ew = 2.0 if is_pred else (1.0 if is_gt else 0.3)
        ec = "#ffffff" if is_pred else (col if is_gt else "#1a2535")
        ax.bar(i, ll_en[i], color=col, alpha=alpha, edgecolor=ec, linewidth=ew, width=0.7)
        if is_pred: ax.text(i, ll_en[i]+0.05, "▲", ha="center", fontsize=6, color=col)
        if is_gt and not is_pred: ax.text(i, -0.22, "GT", ha="center",
                                          fontsize=5, color=col, fontweight="bold")
    ax.set_xticks(range(len(exs)))
    ax.set_xticklabels([EXACT_DISPLAY.get(e, e) for e in exs],
                       fontsize=5.5, color="#88aacc")
    ax.set_ylim(-0.35, 1.5); ax.tick_params(axis="y", colors="#5a6a7e", labelsize=4)
    ax.spines[:].set_color("#253447")
    ax.set_title(f"L2 exact (within {pred_b7})", color="#8899aa", fontsize=7, pad=2)

    return _fig_to_b64(fig, 560, 120)


# ── HTML output ──────────────────────────────────────────────────────────────

def _make_card(seg, is_failure):
    root     = seg["root"]
    gt_fam   = seg["gt_fam"]; gt_b7 = seg["gt_b7"]; gt_ex = seg["gt_exact"]
    pf       = seg["pred_fam"]; pb7 = seg["pred_b7"]; pex = seg["pred_exact"]
    col      = FAM_COLORS[gt_fam]
    pred_col = FAM_COLORS[pf]
    gt_tok   = NOTE[root] + EXACT_DISPLAY.get(gt_ex, gt_ex)
    pred_tok = NOTE[(root + seg["pred_root"]) % 12] + EXACT_DISPLAY.get(pex, pex)
    dur      = seg["t1"] - seg["t0"]

    # error level label
    if is_failure:
        if pf != gt_fam:   err_level = "FAMILY ERR"
        elif pb7 != gt_b7: err_level = "BASE7 ERR"
        else:              err_level = "EXACT ERR"
        border_col = "#3a1515"; badge_bg = "#5a1a1a"
        badge_col = "#ff5555"
    else:
        err_level = "OK"
        border_col = "#0f2b1a"; badge_bg = "#0e2e1a"; badge_col = "#33cc77"

    hm_b64   = _render_heatmap_b64(seg)
    ch_b64   = _render_chroma_b64(seg, col)
    path_b64 = _render_tree_path_b64(seg)
    wav_b64  = _audio_b64(seg["audio_snippet"], seg["sr"])

    if is_failure:
        title_html = (
            f'<span style="color:{col};font-weight:700">{gt_tok}</span>'
            f'<span style="color:#8899aa"> → </span>'
            f'<span style="color:{pred_col};font-weight:700">{pred_tok}</span>'
        )
    else:
        title_html = f'<span style="color:{col};font-weight:700">{gt_tok}</span> <span style="color:#22aa55">✓</span>'

    return f"""
<div class="card" style="border-color:{border_col}">
  <div class="card-title">
    <span class="badge" style="background:{badge_bg};color:{badge_col}">{err_level}</span>
    {title_html}
    <span class="dur"> [{dur:.1f}s]</span>
    <span class="songname"> {seg['title'][:35]}</span>
  </div>
  <div class="panel-label">LTAS chroma (GT root-shifted)</div>
  <img src="data:image/png;base64,{hm_b64}" width="300" height="80">
  <div class="panel-label">Mean chroma</div>
  <img src="data:image/png;base64,{ch_b64}" width="300" height="90">
  <div class="panel-label">Tree path (▲ = chosen, GT = annotated if different)</div>
  <img src="data:image/png;base64,{path_b64}" width="560" height="120">
  <div class="panel-label">Audio</div>
  <audio controls preload="auto" style="width:300px;margin-top:4px">
    <source src="data:audio/wav;base64,{wav_b64}" type="audio/wav">
  </audio>
</div>"""


def write_html(failures, correct_sample, out_html):
    cards = []
    for s in failures:      cards.append(_make_card(s, True))
    for s in correct_sample: cards.append(_make_card(s, False))

    n_f = len(failures); n_c = len(correct_sample)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Exact-quality tree-search diagnostic</title>
<style>
  *{{ box-sizing:border-box }}
  body{{ margin:0;padding:16px;background:#0d1520;color:#e2e8f0;
        font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;font-size:13px }}
  h1{{ color:#c8d8e8;font-size:15px;font-weight:600;margin:0 0 4px;
       padding-bottom:8px;border-bottom:1px solid #253447 }}
  p.note{{ color:#5a7a9a;font-size:11px;margin:0 0 14px }}
  .grid{{ display:grid;grid-template-columns:repeat(auto-fill,580px);gap:16px }}
  .card{{ background:#111e2e;border:1px solid #1e3050;border-radius:6px;
          padding:10px 10px 8px;display:flex;flex-direction:column;gap:2px }}
  .card-title{{ font-size:12px;margin-bottom:6px;white-space:nowrap;
                overflow:hidden;text-overflow:ellipsis }}
  .badge{{ display:inline-block;padding:1px 5px;border-radius:3px;
           font-size:9px;font-weight:700;letter-spacing:.05em;
           margin-right:5px;vertical-align:middle }}
  .dur{{ color:#6a80a0 }} .songname{{ color:#8899aa;font-style:italic }}
  .panel-label{{ color:#4a607a;font-size:10px;margin-top:4px;
                 text-transform:uppercase;letter-spacing:.04em }}
  img{{ display:block;border-radius:3px;max-width:100% }}
  audio{{ filter:invert(0.85) hue-rotate(180deg) }}
</style>
</head>
<body>
<h1>Exact-quality greedy tree search — {n_f} failures + {n_c} correct
  &nbsp;(hard audio, oracle boundaries, GT root shift)</h1>
<p class="note">Tree path: L0 scores all 5 families → winner → L1 scores its base7 children →
  winner → L2 scores its exact children → final prediction. ▲ = chosen at each level.
  Key labels on LL bars are NOT shown here (absorbed into tree scores). Root in card title
  uses predicted root from L0 argmax key.</p>
<div class="grid">
{"".join(cards)}
</div>
</body>
</html>"""

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    print(f"→ {out_html}  ({out_html.stat().st_size/1024:.0f} KB)")


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-songs",      type=int, default=25)
    ap.add_argument("--max-failures", type=int, default=24)
    ap.add_argument("--n-correct",    type=int, default=9)
    ap.add_argument("--seed",         type=int, default=42)
    ap.add_argument("--out-html",     default=None)
    args = ap.parse_args()

    out_html = Path(args.out_html) if args.out_html else OUT_HTML

    if not TREE_CACHE.exists():
        print(f"ERROR: {TREE_CACHE} not found — run chord_tree_ltas.py first"); sys.exit(1)
    d = np.load(TREE_CACHE)
    dist = {k: d[k] for k in d.files}

    rng = np.random.default_rng(args.seed)
    print(f"Collecting segments ({args.n_songs} songs, hard audio, oracle boundaries)...")
    segs = collect(args.n_songs, dist, rng)
    print(f"  {len(segs)} segments")

    failures = [s for s in segs if s["pred_exact"] != s["gt_exact"]]
    correct  = [s for s in segs if s["pred_exact"] == s["gt_exact"]]

    fam_err  = sum(1 for s in failures if s["pred_fam"]  != s["gt_fam"])
    b7_err   = sum(1 for s in failures if s["pred_fam"]  == s["gt_fam"]
                                       and s["pred_b7"]  != s["gt_b7"])
    ex_err   = sum(1 for s in failures if s["pred_fam"]  == s["gt_fam"]
                                       and s["pred_b7"]  == s["gt_b7"]
                                       and s["pred_exact"]!= s["gt_exact"])

    print(f"\n  {len(segs)} total  |  {len(failures)} failures ({len(failures)/len(segs):.1%})")
    print(f"    family-level errors : {fam_err}")
    print(f"    base7-level errors  : {b7_err}")
    print(f"    exact-level errors  : {ex_err}")
    print(f"  {len(correct)} correct ({len(correct)/len(segs):.1%})")

    # Sample failures stratified by (gt_exact, pred_exact)
    by_pair: dict = defaultdict(list)
    for s in failures: by_pair[(s["gt_exact"], s["pred_exact"])].append(s)
    pairs_sorted = sorted(by_pair.items(), key=lambda x: -len(x[1]))
    per_pair = max(1, args.max_failures // max(len(by_pair), 1))
    sampled_fail = []
    for (gt, pred), cases in pairs_sorted:
        sampled_fail.extend(cases[:per_pair])
        if len(sampled_fail) >= args.max_failures: break
    sampled_fail = sampled_fail[:args.max_failures]

    # Sample correct stratified by gt_exact
    by_exact: dict = defaultdict(list)
    for s in correct: by_exact[s["gt_exact"]].append(s)
    per_ex = max(1, args.n_correct // max(len(by_exact), 1))
    sampled_corr = []
    for ex in sorted(by_exact.keys()):
        sampled_corr.extend(by_exact[ex][:per_ex])
        if len(sampled_corr) >= args.n_correct: break
    sampled_corr = sampled_corr[:args.n_correct]

    print(f"\n  Rendering {len(sampled_fail)} failure cards + {len(sampled_corr)} correct cards...")
    print("  Top error pairs:")
    for (gt, pred), cases in pairs_sorted[:10]:
        print(f"    {gt:10s} → {pred:10s}: {len(cases)}")

    write_html(sampled_fail, sampled_corr, out_html)
