"""Visualise the exact input tensor seen by the context MLP / gate for one segment.

Produces a single figure showing:
  - Left column  : 12d root-shifted chroma mean (the segment itself)
  - Right 9 panels: the (5, 12) LL matrix for each context position
                    key-unified so col 0 = target chord's root
                    rows = 5 families, cols = 12 keys relative to target root

Usage:
    .venv/bin/python scripts/illustrate_ctx_input.py
    .venv/bin/python scripts/illustrate_ctx_input.py --song-idx 3 --seg-idx 10
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
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.special import logsumexp as sp_logsumexp

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
OUT        = REPO / "docs" / "plots" / "ctx_input_illustration.png"

FAMILIES   = ["major", "minor", "dim", "aug", "sus"]
FAM_COLORS = ["#58d4ff", "#a65fd4", "#e34948", "#e0a03b", "#1baf7a"]
NOTE       = ["C","Db","D","Eb","E","F","F#","G","Ab","A","Bb","B"]
DEGREE     = ["R","b2","2","b3","3","4","b5","5","#5","6","b7","7"]
HOP        = 512
CTX_K      = 4


def _render_hard(midi_path, rng):
    import pretty_midi
    renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    sf = SOUNDFONTS[0]
    stems = {
        "chords": stem_midi(pm, lambda i: (not i.is_drum) and "bass" not in i.name.lower()),
        "bass":   stem_midi(pm, lambda i: "bass" in i.name.lower()),
        "drums":  stem_midi(pm, lambda i: i.is_drum),
    }
    waves, sr = {}, 44100
    for name, s in stems.items():
        if s and s.instruments:
            w, sr2 = render_to_array(renderer, s, sf, reverb=False)
            waves[name] = w; sr = sr2
    L = max(len(w) for w in waves.values())
    mix = np.zeros(L, np.float32)
    gains = {"chords": 0.7, "bass": 0.5, "drums": 0.3}
    for name, w in waves.items(): mix[:len(w)] += gains.get(name, 0.5) * w
    peak = np.abs(mix).max()
    if peak > 0.99: mix *= 0.99 / peak
    return mix.astype(float), sr


def _diag_ll(x, mu, std):
    return float(-0.5 * np.sum(((x - mu) / std)**2) - np.sum(np.log(std)))


def _compute_ll_mat(chroma_mean, dist):
    n = np.linalg.norm(chroma_mean)
    if n < 1e-9: return np.zeros((5, 12), np.float32)
    x = chroma_mean / n
    fam_names = ["major", "minor", "diminished", "augmented", "suspended"]
    ll = np.zeros((5, 12), np.float32)
    for fi, fam in enumerate(fam_names):
        mu  = dist[f"{fam}_mu"]
        std_= dist[f"{fam}_std"]
        for r in range(12):
            ll[fi, r] = _diag_ll(np.roll(x, -r), mu, std_)
    return ll


def collect_one_song(song_idx, dist, rng):
    recs = {json.loads(l)["song_id"]: json.loads(l) for l in open(DB)}
    man: dict = {}
    for m in map(json.loads, open(MANIFEST)):
        if m["song_id"] not in man or m.get("transpose", 0) == 0:
            man[m["song_id"]] = m
    avail  = sorted([s for s in recs if s in man], key=lambda s: recs[s]["title"])
    sid    = avail[song_idx % len(avail)]
    rec    = recs[sid]; m = man[sid]
    bpb    = m["beats_per_bar"]; spb = 60.0 / m["tempo"]
    print(f"  Song: {rec['title']}")

    audio, sr = _render_hard(REPO / m["midi_path"], rng)
    raw = librosa.feature.chroma_cqt(y=audio, sr=sr, bins_per_octave=36, hop_length=HOP)
    ltas = raw.mean(axis=1, keepdims=True)
    ltas = np.where(ltas < 1e-9, 1.0, ltas)
    chroma = raw / ltas
    ct = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=HOP)

    chord_at = {(e["bar"]-1)*bpb+e["beat"]: e for e in rec["chord_timeline"]}
    records = []
    for t0, t1, root_gt, _ in song_chord_spans(rec):
        b0  = int(round(t0 / spb))
        mma = chord_at.get(b0, {}).get("mma")
        p   = parse_chord(mma) if mma else None
        if p is None or p[1] not in BUCKET_FAMILY: continue
        fam_names = ["major", "minor", "diminished", "augmented", "suspended"]
        fam = BUCKET_FAMILY[p[1]]
        if fam not in fam_names: continue
        root = int(root_gt % 12)

        i0 = int(np.searchsorted(ct, t0)); i1 = int(np.searchsorted(ct, t1))
        if i1 <= i0: i1 = i0 + 1
        frames_abs     = chroma[:, i0:i1]
        frames_shifted = np.roll(frames_abs, -root, axis=0)

        mean_s = frames_shifted.mean(axis=1)
        nn = np.linalg.norm(mean_s)
        chroma_mean = (mean_s / nn).astype(np.float32) if nn > 1e-9 else np.zeros(12, np.float32)

        # compute ll_mat in absolute-pitch space (root at col 0 after roll by -root)
        mean_abs = frames_abs.mean(axis=1)
        ll_mat = _compute_ll_mat(mean_abs, dist)   # (5, 12) absolute keys

        records.append({
            "gt_fam":     fam,
            "root_pc":    root,
            "chord_str":  NOTE[root] + p[1],
            "chroma_mean": chroma_mean,    # (12,) root-shifted, root at 0
            "ll_mat":     ll_mat,          # (5, 12) absolute keys
            "t0": t0, "t1": t1,
        })
    return records, rec["title"]


def build_ctx_for_seg(records, i, k=CTX_K):
    """
    Build the (2k+1, 5, 12) key-unified context tensor for segment i.
    col 0 of every context position = target chord's root.
    Returns the tensor plus metadata for each context position.
    """
    N = len(records)
    W = 2 * k + 1
    root_i = records[i]["root_pc"]
    tensor = np.zeros((W, 5, 12), np.float32)
    ctx_meta = []
    for j, offset in enumerate(range(-k, k + 1)):
        ni = i + offset
        if 0 <= ni < N:
            root_j = records[ni]["root_pc"]
            delta  = (root_j - root_i) % 12
            # roll so col 0 = root_i in j's frame
            tensor[j] = np.roll(records[ni]["ll_mat"], -delta, axis=1)
            interval = (root_j - root_i) % 12
            ctx_meta.append({
                "offset": offset,
                "chord": records[ni]["chord_str"],
                "interval": interval,
                "interval_name": DEGREE[interval],
                "present": True,
            })
        else:
            ctx_meta.append({"offset": offset, "chord": "(pad)", "interval": None,
                             "interval_name": "—", "present": False})
    return tensor, ctx_meta


def plot(records, seg_idx, song_title, out: Path):
    seg = records[seg_idx]
    tensor, ctx_meta = build_ctx_for_seg(records, seg_idx)
    chroma = seg["chroma_mean"]   # (12,) root-shifted
    W = 2 * CTX_K + 1            # 9 context positions

    # ── layout ────────────────────────────────────────────────────────────────
    # Row 0: title + chroma bar (spans 3 cols) | 9 ll_mat heatmaps (3×3 grid)
    fig = plt.figure(figsize=(18, 9), facecolor="#0d1520")
    fig.suptitle(
        f"MLP input tensor -- '{seg['chord_str']}' from '{song_title}'\n"
        f"Left: 12d chroma mean (root-shifted, root=R).  "
        f"Right: (9 × 5 × 12) key-unified LL matrix — each panel = one context position;\n"
        f"rows = 5 families, cols = 12 keys relative to target root (col 0 = same root, col 7 = a fifth up, etc.)",
        color="#c8d8e8", fontsize=10.5, y=1.01)

    outer = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 4], wspace=0.08)

    # ── left: chroma bar chart ────────────────────────────────────────────────
    ax_ch = fig.add_subplot(outer[0])
    ax_ch.set_facecolor("#0d1520")
    bar_cols = [FAM_COLORS[["major","minor","diminished","augmented","suspended"].index(seg["gt_fam"])]
                if i == 0 else "#2a3a50" for i in range(12)]
    ax_ch.barh(range(12), chroma, color=bar_cols, height=0.75,
               edgecolor="#1a2535", linewidth=0.4)
    ax_ch.set_yticks(range(12))
    ax_ch.set_yticklabels(DEGREE, fontsize=8, color="#88aacc")
    ax_ch.set_xlabel("LTAS chroma\n(root-shifted mean)", color="#5a7a9a", fontsize=8)
    ax_ch.tick_params(axis="x", colors="#5a6a7e", labelsize=7)
    ax_ch.spines[:].set_color("#253447")
    ax_ch.set_title(f"12d chroma\n{seg['chord_str']} (R = root)",
                    color="#e2e8f0", fontsize=9, pad=6)
    ax_ch.invert_yaxis()

    # ── right: 3×3 grid of LL heatmaps ───────────────────────────────────────
    right = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=outer[1],
                                             hspace=0.55, wspace=0.35)

    fam_names_full = ["major","minor","diminished","augmented","suspended"]
    # global vmin/vmax across all 9 panels for consistent colour scale
    vmin = tensor[tensor != 0].min() if (tensor != 0).any() else -100
    vmax = tensor.max()

    for j in range(W):
        row, col = divmod(j, 3)
        ax = fig.add_subplot(right[row, col])
        ax.set_facecolor("#0d1520")
        meta = ctx_meta[j]

        if meta["present"]:
            im = ax.imshow(tensor[j], aspect="auto", origin="upper",
                           cmap="RdYlGn", vmin=vmin, vmax=vmax,
                           interpolation="nearest")
            # highlight col 0 (= shared root) with a thin white line
            ax.axvline(0, color="#ffffff", linewidth=0.8, alpha=0.5)
        else:
            ax.set_facecolor("#080f18")
            ax.text(0.5, 0.5, "padding\n(zero)", ha="center", va="center",
                    transform=ax.transAxes, color="#3a5070", fontsize=8)

        # family labels on y-axis
        ax.set_yticks(range(5))
        ax.set_yticklabels(FAMILIES, fontsize=6, color="#88aacc")
        # key degree labels on x-axis (relative to target root)
        ax.set_xticks(range(0, 12, 2))
        ax.set_xticklabels(DEGREE[::2], fontsize=5.5, color="#7a9ab8")
        ax.tick_params(length=2, color="#253447")
        ax.spines[:].set_color("#253447")

        # panel title
        offset_str = f"i{meta['offset']:+d}" if meta["offset"] != 0 else "i (target)"
        if meta["present"]:
            interval_str = f"Δ={meta['interval_name']}" if meta["offset"] != 0 else ""
            title_str = f"{offset_str}: {meta['chord']}  {interval_str}"
            title_col = "#58d4ff" if meta["offset"] == 0 else "#8899aa"
            box_col   = "#1a3a5a" if meta["offset"] == 0 else None
        else:
            title_str = f"{offset_str}: (pad)"
            title_col = "#3a5070"
            box_col   = None

        ax.set_title(title_str, color=title_col, fontsize=7, pad=3,
                     bbox=dict(boxstyle="round,pad=0.2",
                               facecolor=box_col or "#0d1520",
                               edgecolor="#253447", linewidth=0.6) if box_col else None)

    # ── colour bar ────────────────────────────────────────────────────────────
    cax = fig.add_axes([0.92, 0.15, 0.008, 0.65])
    sm  = plt.cm.ScalarMappable(cmap="RdYlGn",
                                 norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("log-likelihood", color="#5a7a9a", fontsize=8)
    cb.ax.tick_params(labelcolor="#5a7a9a", labelsize=6)
    cb.outline.set_edgecolor("#253447")

    # ── annotation: what the gate does ───────────────────────────────────────
    ll_max_per_fam = tensor[CTX_K].max(axis=1)   # (5,) for target position
    softmax_t = np.exp(ll_max_per_fam - ll_max_per_fam.max())
    softmax_t /= softmax_t.sum()
    H = -(softmax_t * np.log(softmax_t + 1e-12)).sum()
    H_max = np.log(5)
    anno = (f"LL argmax (this segment): {FAMILIES[int(np.argmax(ll_max_per_fam))].upper()}  "
            f"(H={H:.2f}, H_max={H_max:.2f})\n"
            f"Gate α ≈ sigmoid(−0.49·H − 0.29): "
            f"α ≈ {1/(1+np.exp(0.49*H+0.29)):.2f}  "
            f"→ {'trust context (low α)' if H > 0.8 else 'trust point estimate (high α)'}")
    fig.text(0.5, -0.01, anno, ha="center", va="top",
             color="#7a9ab8", fontsize=8.5,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#0a1520",
                       edgecolor="#253447", linewidth=0.8))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"→ {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--song-idx", type=int, default=7,
                    help="index into sorted song list (default=7: Alfie's Theme)")
    ap.add_argument("--seg-idx",  type=int, default=8,
                    help="segment index within that song")
    ap.add_argument("--seed",     type=int, default=42)
    ap.add_argument("--out",      default=None)
    args = ap.parse_args()

    out = Path(args.out) if args.out else OUT

    if not DIST_CACHE.exists():
        print("ERROR: run plot_family_likelihood.py --rebuild-cache first"); sys.exit(1)
    d = np.load(DIST_CACHE)
    dist = {k: d[k] for k in d.files}

    rng = np.random.default_rng(args.seed)
    print(f"Collecting song {args.song_idx}...")
    records, song_title = collect_one_song(args.song_idx, dist, rng)
    print(f"  {len(records)} segments")

    seg_idx = args.seg_idx % len(records)
    seg = records[seg_idx]
    print(f"  Segment {seg_idx}: {seg['chord_str']} ({seg['gt_fam']})  "
          f"t={seg['t0']:.1f}–{seg['t1']:.1f}s")
    print(f"  Context: ", end="")
    _, ctx_meta = build_ctx_for_seg(records, seg_idx)
    for m in ctx_meta:
        marker = "[TGT]" if m["offset"] == 0 else f"({m['offset']:+d})"
        print(f"{marker}{m['chord']} ", end="")
    print()

    plot(records, seg_idx, song_title, out)
