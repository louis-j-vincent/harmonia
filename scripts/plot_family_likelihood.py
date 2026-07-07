"""Per-half-bar family log-likelihood diagnostic for a single song.

For each half-bar chunk:
  1. Extract mean LTAS-normalised CQT chroma (12,), L2-normalise it.
  2. For each of 5 chord families, try all 12 root shifts (roll(x, -r));
     compute the diagonal-Gaussian log-likelihood under the family's
     (μ, σ) distribution (learned from corpus renders).
  3. Take max over the 12 roots → max LL + best key label.

Top panel : LTAS CQT chroma heatmap with GT and half-bar grid.
Bottom 5  : one bar chart per family — bar height = max-LL over 12 keys,
             best key label printed on each bar.

Family distributions are cached to data/cache/ltas_family_dist.npz so
the 30-song render runs only once.

Usage:
    .venv/bin/python scripts/plot_family_likelihood.py
    .venv/bin/python scripts/plot_family_likelihood.py --song "Bye Bye Blackbird" --bars 8
    .venv/bin/python scripts/plot_family_likelihood.py --rebuild-cache
"""
from __future__ import annotations
import argparse, json, sys, tempfile, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pretty_midi
import soundfile as sf

from analyze_accomp_emission import parse_chord, song_chord_spans
from build_accomp_audio_hard import render_to_array, stem_midi, SOUNDFONTS
from build_audio_chord_features import BUCKET_FAMILY
from harmonia.data.midi_renderer import MIDIRenderer
from harmonia.models.stage1_pitch import PitchExtractor

DB       = REPO / "data" / "accomp_db" / "db.jsonl"
MANIFEST = REPO / "data" / "accomp_db" / "audio" / "manifest.jsonl"
DIST_CACHE = REPO / "data" / "cache" / "ltas_family_dist.npz"

NOTE     = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
FAMILIES = ["major", "minor", "diminished", "augmented", "suspended"]
FAM_COLORS = {
    "major":      "#58d4ff",
    "minor":      "#a65fd4",
    "diminished": "#e34948",
    "augmented":  "#e0a03b",
    "suspended":  "#1baf7a",
}
FAM_SUFFIX = {"major": "^7", "minor": "-7", "diminished": "o7",
              "augmented": "+", "suspended": "sus"}


# ── family distribution ───────────────────────────────────────────────────────

def _render_chord_only(midi_path, sf_name, renderer):
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    chord_pm = stem_midi(pm, lambda i: (not i.is_drum)
                                        and "bass" not in i.name.lower())
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


def build_family_distributions(n_songs: int = 30) -> dict:
    """Render n_songs chord-only, collect root-shifted LTAS chroma per family."""
    recs = {json.loads(l)["song_id"]: json.loads(l) for l in open(DB)}
    man: dict = {}
    for m in map(json.loads, open(MANIFEST)):
        if m["song_id"] not in man or m.get("transpose", 0) == 0:
            man[m["song_id"]] = m

    avail = sorted([s for s in recs if s in man], key=lambda s: recs[s]["title"])
    chosen = avail[:n_songs]
    renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")
    sf_name  = SOUNDFONTS[0]

    by_fam: dict[str, list[np.ndarray]] = {f: [] for f in FAMILIES}

    for i, sid in enumerate(chosen):
        rec = recs[sid]; m = man[sid]
        bpb = m["beats_per_bar"]; spb = 60.0 / m["tempo"]
        print(f"\r  [{i+1}/{len(chosen)}] {rec['title'][:40]:40s}", end="", flush=True)
        try:
            audio, sr = _render_chord_only(REPO / m["midi_path"], sf_name, renderer)
            if audio is None: continue
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
            i0 = int(np.searchsorted(ct, t0))
            i1 = int(np.searchsorted(ct, t1))
            if i1 <= i0: i1 = i0 + 1
            seg = chroma[:, i0:i1].mean(axis=1)
            shifted = np.roll(seg, -int(root_gt % 12))
            n = np.linalg.norm(shifted)
            if n > 1e-9:
                by_fam[fam].append(shifted / n)
    print()

    result = {}
    for fam in FAMILIES:
        vecs = np.stack(by_fam[fam]) if by_fam[fam] else np.zeros((2, 12))
        result[f"{fam}_mu"]  = vecs.mean(axis=0)
        result[f"{fam}_std"] = vecs.std(axis=0) + 1e-4   # floor to avoid 0-div
        result[f"{fam}_n"]   = np.array([len(by_fam[fam])])
    return result


def load_or_build_distributions(n_songs: int = 30, rebuild: bool = False) -> dict:
    if not rebuild and DIST_CACHE.exists():
        print(f"Loading cached family distributions from {DIST_CACHE.name}")
        d = np.load(DIST_CACHE)
        return {k: d[k] for k in d.files}
    print(f"Building family distributions from {n_songs} songs...")
    dist = build_family_distributions(n_songs)
    np.savez(DIST_CACHE, **dist)
    print(f"  Saved to {DIST_CACHE}")
    return dist


# ── log-likelihood ────────────────────────────────────────────────────────────

def diag_gaussian_ll(x: np.ndarray, mu: np.ndarray, std: np.ndarray) -> float:
    """Diagonal Gaussian log-likelihood (up to constant)."""
    return float(-0.5 * np.sum(((x - mu) / std) ** 2) - np.sum(np.log(std)))


def max_ll_over_keys(x: np.ndarray, mu: np.ndarray, std: np.ndarray
                     ) -> tuple[float, int]:
    """Try all 12 root shifts; return (max_ll, best_root_pc)."""
    best_ll, best_r = -np.inf, 0
    for r in range(12):
        ll = diag_gaussian_ll(np.roll(x, -r), mu, std)
        if ll > best_ll:
            best_ll, best_r = ll, r
    return best_ll, best_r


# ── main ─────────────────────────────────────────────────────────────────────

def run(song_q: str, n_bars: int, seed: int, rebuild_cache: bool, out_path: Path):
    dist = load_or_build_distributions(rebuild=rebuild_cache)

    recs = {json.loads(l)["song_id"]: json.loads(l) for l in open(DB)}
    man: dict = {}
    for m in map(json.loads, open(MANIFEST)):
        if m["song_id"] not in man or m.get("transpose", 0) == 0:
            man[m["song_id"]] = m

    sid = next((s for s in recs if song_q.lower() in recs[s]["title"].lower()), None)
    if sid is None:
        print(f"Song '{song_q}' not found"); sys.exit(1)
    rec = recs[sid]; m = man[sid]
    bpb = m["beats_per_bar"]; spb = 60.0 / m["tempo"]
    print(f"Song: {rec['title']}  tempo={m['tempo']}  bpb={bpb}")

    renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")
    audio, sr = _render_chord_only(REPO / m["midi_path"], SOUNDFONTS[0], renderer)
    chroma, ct = _ltas_cqt(audio, sr)
    hop = 512

    # Perfect beat grid
    _, bf = librosa.beat.beat_track(y=audio, sr=sr, bpm=float(m["tempo"]), units="frames")
    raw_bt = librosa.frames_to_time(bf, sr=sr)
    t0_phase = float(raw_bt[0])
    n_beats_total = int((len(audio) / sr - t0_phase) / spb) + 1
    bt = np.array([t0_phase + i * spb for i in range(n_beats_total)])

    t_end = t0_phase + n_bars * bpb * spb
    mask = ct <= t_end
    ct_plot = ct[mask]

    # GT chord boundaries
    chord_at = {(e["bar"]-1)*bpb+e["beat"]: e for e in rec["chord_timeline"]}
    gt_cuts = []
    for t0, t1, root, _ in song_chord_spans(rec):
        if t0 > t_end + spb: continue
        b0 = int(round(t0 / spb))
        mma = chord_at.get(b0, {}).get("mma")
        p = parse_chord(mma) if mma else None
        fam = BUCKET_FAMILY.get(p[1]) if p else None
        suf = FAM_SUFFIX.get(fam, "")
        gt_cuts.append((t0, NOTE[root % 12] + suf))

    # Half-bar chunk grid
    half_step = max(1, bpb // 2)
    chunk_bounds = []   # (t_start, t_end, beat_idx_start)
    b = 0
    while True:
        t_a = t0_phase + b * spb
        t_b = t0_phase + (b + half_step) * spb
        if t_a >= t_end: break
        t_b = min(t_b, t_end)
        chunk_bounds.append((t_a, t_b, b))
        b += half_step

    # Compute max-LL per chunk per family
    results = []   # list of {ta, tb, fam → (max_ll, best_root)}
    for ta, tb, _ in chunk_bounds:
        i0 = int(np.searchsorted(ct, ta))
        i1 = int(np.searchsorted(ct, tb))
        if i1 <= i0: i1 = i0 + 1
        seg = chroma[:, i0:i1].mean(axis=1)
        n = np.linalg.norm(seg)
        x = seg / n if n > 1e-9 else seg

        chunk_res = {}
        for fam in FAMILIES:
            mu  = dist[f"{fam}_mu"]
            std = dist[f"{fam}_std"]
            ll, best_r = max_ll_over_keys(x, mu, std)
            chunk_res[fam] = (ll, best_r)
        results.append((ta, tb, chunk_res))

    # Normalise LL to [0, 1] per family for display
    # (raw values are very negative; we want relative difference across chunks)
    ll_by_fam = {fam: np.array([r[2][fam][0] for r in results]) for fam in FAMILIES}
    ll_norm = {}
    for fam in FAMILIES:
        v = ll_by_fam[fam]
        vmin, vmax = v.min(), v.max()
        ll_norm[fam] = (v - vmin) / (vmax - vmin + 1e-12)

    # ── plot ──────────────────────────────────────────────────────────────────
    n_fam = len(FAMILIES)
    fig = plt.figure(figsize=(16, 12), facecolor="#0d1520")
    fig.suptitle(f"{rec['title']} — bars 1–{n_bars}  ·  half-bar family log-likelihood",
                 color="#e2e8f0", fontsize=13, y=0.995)

    gs = fig.add_gridspec(1 + n_fam, 1,
                          height_ratios=[3.5] + [1.0] * n_fam,
                          hspace=0.08)

    # ── top: LTAS heatmap ─────────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0])
    ax0.set_facecolor("#0d1520")
    cp = chroma[:, mask]
    ax0.imshow(cp, aspect="auto", origin="lower",
               extent=[ct_plot[0], ct_plot[-1], -0.5, 11.5],
               cmap="YlOrRd", vmin=0, vmax=cp.max() * 0.85)
    for bt_i in bt:
        if bt_i > t_end: break
        ax0.axvline(bt_i, color="#ffffff22", lw=0.5)
    for bar in range(n_bars + 1):
        t_bar = t0_phase + bar * bpb * spb
        ax0.axvline(t_bar, color="#ffffff55", lw=1.2)
        if bar < n_bars:
            ax0.text(t_bar + 0.01, 11.6, f"bar {bar+1}",
                     color="#88aacc", fontsize=7.5, va="bottom")
    # half-bar grid
    for ta, tb, _ in chunk_bounds[1:]:
        ax0.axvline(ta, color="#ffffff33", lw=0.8, linestyle=":")
    # GT labels
    for t0, label in gt_cuts:
        ax0.axvline(t0, color="#1baf7a", lw=1.2, linestyle="--", alpha=0.85)
        ax0.text(t0 + 0.01, -0.45, label, color="#1baf7a",
                 fontsize=6.5, va="top", rotation=0)
    ax0.set_xlim(ct_plot[0], t_end)
    ax0.set_ylim(-0.5, 11.5)
    ax0.set_yticks(range(12))
    ax0.set_yticklabels(NOTE, fontsize=8, color="#88aacc")
    ax0.tick_params(axis="x", colors="#5a6a7e", labelsize=7)
    ax0.set_title("LTAS-normalised CQT chroma  ·  green dashed = GT boundary  ·  dotted = half-bar grid",
                  color="#88aacc", fontsize=8, pad=3)

    # ── family panels ─────────────────────────────────────────────────────────
    chunk_mids = [(ta + tb) / 2 for ta, tb, _ in results]
    chunk_widths = [(tb - ta) * 0.82 for ta, tb, _ in results]
    x_pos = np.array(chunk_mids)

    axes_fam = [fig.add_subplot(gs[i + 1]) for i in range(n_fam)]

    for ax_f, fam in zip(axes_fam, FAMILIES):
        col = FAM_COLORS[fam]
        norms = ll_norm[fam]
        raw_lls = ll_by_fam[fam]

        # bar heights = normalised LL; colour intensity = same value
        bar_colors = [matplotlib.colors.to_rgba(col, alpha=0.35 + 0.6 * v)
                      for v in norms]
        bars = ax_f.bar(x_pos, norms, width=chunk_widths,
                        color=bar_colors, edgecolor=col, linewidth=0.7)

        # Best-key label on each bar
        for ci, (ta, tb, chunk_res) in enumerate(results):
            ll_val, best_r = chunk_res[fam]
            key_label = NOTE[best_r]
            bar_top = norms[ci]
            ax_f.text(chunk_mids[ci], bar_top + 0.04, key_label,
                      ha="center", va="bottom", fontsize=7.5,
                      color=col, fontweight="bold", fontfamily="monospace")

        # GT boundary lines
        for t0, _ in gt_cuts:
            ax_f.axvline(t0, color="#1baf7a", lw=0.9, linestyle="--", alpha=0.55)
        # half-bar lines
        for ta, tb, _ in chunk_bounds[1:]:
            ax_f.axvline(ta, color="#ffffff18", lw=0.6, linestyle=":")

        ax_f.set_xlim(ct_plot[0], t_end)
        ax_f.set_ylim(-0.08, 1.45)
        ax_f.set_yticks([0, 0.5, 1.0])
        ax_f.set_yticklabels(["lo", "", "hi"], fontsize=6.5, color="#5a6a7e")
        ax_f.tick_params(axis="x", colors="#5a6a7e", labelsize=7)
        ax_f.set_facecolor("#0d1520")
        ax_f.spines[:].set_color("#253447")
        n_segs = int(dist[f"{fam}_n"][0])
        ax_f.set_ylabel(f"{fam}\n(n={n_segs})", color=col,
                        fontsize=8, rotation=0, labelpad=52, va="center")
        if ax_f is not axes_fam[-1]:
            ax_f.set_xticklabels([])

    axes_fam[-1].set_xlabel("time (s)", color="#5a6a7e", fontsize=8)

    # shared x sync
    for ax_f in axes_fam:
        ax_f.set_xlim(ax0.get_xlim())

    fig.text(0.5, -0.005,
             "Bar height = max log-likelihood over 12 root shifts (normalised within family)"
             "  ·  label = best key for that chunk  ·  green dashed = GT boundary",
             ha="center", color="#5a6a7e", fontsize=8)

    plt.tight_layout(rect=[0, 0.01, 1, 0.995])
    fig.savefig(out_path, dpi=160, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"→ {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--song",          default="Anthropology")
    ap.add_argument("--bars",          type=int, default=4)
    ap.add_argument("--seed",          type=int, default=42)
    ap.add_argument("--n-dist-songs",  type=int, default=30,
                    help="songs used to build family distributions (ignored if cache exists)")
    ap.add_argument("--rebuild-cache", action="store_true",
                    help="force re-render of family distribution cache")
    ap.add_argument("--out",           default=None)
    args = ap.parse_args()

    out = Path(args.out) if args.out else \
          REPO / "docs" / "plots" / \
          f"family_ll_{args.song.replace(' ','_').lower()}_b{args.bars}.png"

    run(args.song, args.bars, args.seed, args.rebuild_cache, out)
