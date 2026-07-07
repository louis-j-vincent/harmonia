"""3D UMAP/t-SNE scatter of all chord-quality LTAS distributions.

Embeds the 37 distribution means (5 family + 14 base7 + 18 exact, each 12-dim
root-shifted chroma) into 3D and colour-codes them by family, with marker shape
encoding tree level (family=sphere/big, base7=square/medium, exact=star/small).

Also embeds individual segment vectors coloured by GT family so you can see
the actual cloud vs the node means.

Outputs:
  docs/plots/chord_embedding_umap.html  — interactive Plotly 3D scatter
  docs/plots/chord_embedding_umap.png   — static PNG

Requires:
  pip install umap-learn plotly
  data/cache/chord_tree_ltas.npz  (chord_tree_ltas.py)

Usage:
    .venv/bin/python scripts/plot_chord_embedding.py
    .venv/bin/python scripts/plot_chord_embedding.py --method tsne --n-songs 40
    .venv/bin/python scripts/plot_chord_embedding.py --method umap --n-songs 60 --3d
"""
from __future__ import annotations
import argparse, json, sys, warnings
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
from build_accomp_audio_hard import render_to_array, stem_midi, SOUNDFONTS
from build_audio_chord_features import BUCKET_FAMILY, BUCKET_BASE7
from harmonia.data.midi_renderer import MIDIRenderer

DB         = REPO / "data" / "accomp_db" / "db.jsonl"
MANIFEST   = REPO / "data" / "accomp_db" / "audio" / "manifest.jsonl"
TREE_CACHE = REPO / "data" / "cache" / "chord_tree_ltas.npz"
OUT_HTML   = REPO / "docs" / "plots" / "chord_embedding_umap.html"
OUT_PNG    = REPO / "docs" / "plots" / "chord_embedding_umap.png"

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
EXACT_FAM = {ex:fam for fam,ch in TREE.items() for b7,exs in ch.items() for ex in exs}
EXACT_B7  = {ex:b7  for fam,ch in TREE.items() for b7,exs in ch.items() for ex in exs}

EXACT_DISPLAY = {
    "maj":"maj","6":"6","maj7":"Δ7","dom7":"7","dom7alt":"7alt",
    "min":"min","m6":"m6","min7":"min7","minmaj7":"mΔ7",
    "dim":"dim","dim7":"°7","m7b5":"ø7",
    "aug":"aug","aug7":"aug7","augmaj7":"augΔ7",
    "sus2":"sus2","sus4":"sus4","7sus4":"7sus4",
}


def _ltas_cqt(audio, sr, hop=512):
    raw = librosa.feature.chroma_cqt(y=audio, sr=sr, bins_per_octave=36, hop_length=hop)
    ltas = raw.mean(axis=1, keepdims=True)
    ltas = np.where(ltas < 1e-9, 1.0, ltas)
    chroma = raw / ltas
    ct = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop)
    return chroma, ct


def collect_segment_vectors(n_songs: int) -> tuple[np.ndarray, list[dict]]:
    """Collect root-shifted LTAS chroma vectors from clean chord-only audio."""
    recs = {json.loads(l)["song_id"]: json.loads(l) for l in open(DB)}
    man: dict = {}
    for m in map(json.loads, open(MANIFEST)):
        if m["song_id"] not in man or m.get("transpose", 0) == 0:
            man[m["song_id"]] = m
    avail = sorted([s for s in recs if s in man], key=lambda s: recs[s]["title"])
    chosen = avail[:n_songs]
    renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")
    sf_name  = SOUNDFONTS[0]

    vecs, meta = [], []
    for i, sid in enumerate(chosen):
        rec = recs[sid]; m = man[sid]
        bpb = m["beats_per_bar"]; spb = 60.0 / m["tempo"]
        print(f"\r  [{i+1}/{len(chosen)}] {rec['title'][:42]:42s}", end="", flush=True)
        try:
            import pretty_midi
            pm = pretty_midi.PrettyMIDI(str(REPO / m["midi_path"]))
            cp = stem_midi(pm, lambda i: (not i.is_drum) and "bass" not in i.name.lower())
            if cp is None or not cp.instruments: continue
            audio, sr = render_to_array(renderer, cp, sf_name, reverb=False)
            audio = audio.astype(float)
        except Exception:
            continue
        chroma, ct = _ltas_cqt(audio, sr)
        chord_at = {(e["bar"]-1)*bpb+e["beat"]: e for e in rec["chord_timeline"]}
        for t0, t1, root_gt, _ in song_chord_spans(rec):
            b0  = int(round(t0 / spb))
            mma = chord_at.get(b0, {}).get("mma")
            p   = parse_chord(mma) if mma else None
            if p is None or p[1] not in EXACT_FAM: continue
            gt_ex = p[1]; root = int(root_gt % 12)
            i0 = int(np.searchsorted(ct, t0)); i1 = int(np.searchsorted(ct, t1))
            if i1 <= i0: i1 = i0 + 1
            seg = chroma[:, i0:i1].mean(axis=1)
            shifted = np.roll(seg, -root)
            n = np.linalg.norm(shifted)
            if n < 1e-9: continue
            x12 = shifted / n
            vecs.append(x12)
            meta.append({"exact": gt_ex, "b7": EXACT_B7[gt_ex],
                         "fam": EXACT_FAM[gt_ex], "root": root,
                         "title": rec["title"]})
    print()
    return np.stack(vecs) if vecs else np.zeros((0,12)), meta


def node_vectors(dist: dict):
    """Extract distribution means for all 37 nodes: (vecs, meta_list)."""
    vecs, meta = [], []
    for fam in TREE:
        mu = dist[f"fam_{fam}_mu"]
        vecs.append(mu / (np.linalg.norm(mu) + 1e-12))
        meta.append({"level": "family", "name": fam, "fam": fam})
        for b7 in TREE[fam]:
            mu = dist[f"b7_{b7}_mu"]
            vecs.append(mu / (np.linalg.norm(mu) + 1e-12))
            meta.append({"level": "base7", "name": b7, "fam": fam})
            for ex in TREE[fam][b7]:
                mu = dist[f"exact_{ex}_mu"]
                vecs.append(mu / (np.linalg.norm(mu) + 1e-12))
                meta.append({"level": "exact", "name": ex,
                             "display": EXACT_DISPLAY.get(ex, ex), "fam": fam})
    return np.stack(vecs), meta


def embed_3d(all_vecs: np.ndarray, method: str, seed: int) -> np.ndarray:
    if method == "umap":
        try:
            import umap as umap_mod
            reducer = umap_mod.UMAP(n_components=3, n_neighbors=12, min_dist=0.15,
                                    metric="cosine", random_state=seed)
            return reducer.fit_transform(all_vecs)
        except ImportError:
            print("umap-learn not installed, falling back to t-SNE")
            method = "tsne"
    if method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=3, metric="cosine", perplexity=min(30, len(all_vecs)//4),
                       random_state=seed, max_iter=1000)
        return reducer.fit_transform(all_vecs)
    raise ValueError(f"Unknown method: {method}")


def write_html(coords_segs, meta_segs, coords_nodes, meta_nodes, out: Path):
    """Interactive Plotly 3D scatter."""
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except ImportError:
        print("plotly not installed — skipping HTML output")
        return

    fig = go.Figure()

    # background segment cloud (tiny, transparent)
    by_fam: dict = defaultdict(lambda: {"x":[],"y":[],"z":[],"text":[]})
    for (x,y,z), m in zip(coords_segs, meta_segs):
        by_fam[m["fam"]]["x"].append(float(x)); by_fam[m["fam"]]["y"].append(float(y))
        by_fam[m["fam"]]["z"].append(float(z))
        by_fam[m["fam"]]["text"].append(f"{m['title'][:20]} — {m['exact']}")
    for fam, d in by_fam.items():
        fig.add_trace(go.Scatter3d(
            x=d["x"], y=d["y"], z=d["z"], text=d["text"],
            mode="markers", name=f"{fam} (segs)",
            marker=dict(size=2.5, color=FAM_COLORS[fam], opacity=0.25),
            hovertemplate="%{text}<extra></extra>",
        ))

    # node means — sized by level
    level_size  = {"family": 18, "base7": 11, "exact": 7}
    level_sym   = {"family": "circle",    "base7": "square", "exact": "diamond"}
    level_opac  = {"family": 1.0,         "base7": 0.85,     "exact": 0.75}
    for level in ("family", "base7", "exact"):
        xs, ys, zs, texts, cols = [], [], [], [], []
        for (x,y,z), m in zip(coords_nodes, meta_nodes):
            if m["level"] != level: continue
            label = m.get("display", m["name"])
            xs.append(float(x)); ys.append(float(y)); zs.append(float(z))
            texts.append(label); cols.append(FAM_COLORS[m["fam"]])
        if not xs: continue
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs, text=texts,
            mode="markers+text" if level == "family" else "markers",
            name=f"μ ({level})",
            textfont=dict(size=9, color="white"),
            textposition="top center",
            marker=dict(size=level_size[level], symbol=level_sym[level],
                        color=cols, opacity=level_opac[level],
                        line=dict(width=1 if level=="exact" else 2, color="white")),
            hovertemplate="%{text}<extra></extra>",
        ))

    fig.update_layout(
        title="Chord-quality LTAS distribution embedding (3D)",
        paper_bgcolor="#0d1520", plot_bgcolor="#0d1520",
        font=dict(color="#e2e8f0", size=11),
        scene=dict(
            bgcolor="#0d1520",
            xaxis=dict(showgrid=False, showticklabels=False, title=""),
            yaxis=dict(showgrid=False, showticklabels=False, title=""),
            zaxis=dict(showgrid=False, showticklabels=False, title=""),
        ),
        legend=dict(bgcolor="#111e2e", bordercolor="#253447", borderwidth=1),
        margin=dict(l=0, r=0, t=40, b=0),
        width=1100, height=750,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    pio.write_html(fig, str(out), include_plotlyjs="cdn", full_html=True)
    print(f"→ {out}  ({out.stat().st_size/1024:.0f} KB)")


def write_png(coords_segs, meta_segs, coords_nodes, meta_nodes, out: Path):
    """Static matplotlib 3D scatter."""
    fig = plt.figure(figsize=(13, 9), facecolor="#0d1520")
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0d1520")
    fig.patch.set_facecolor("#0d1520")
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False

    # segment cloud
    for fam in TREE:
        mask = [m["fam"] == fam for m in meta_segs]
        if not any(mask): continue
        xs = coords_segs[mask, 0]; ys = coords_segs[mask, 1]; zs = coords_segs[mask, 2]
        ax.scatter(xs, ys, zs, c=FAM_COLORS[fam], s=3, alpha=0.15, linewidths=0)

    # node means
    level_size = {"family": 160, "base7": 70, "exact": 25}
    level_mark = {"family": "o",   "base7": "s", "exact": "D"}
    for level in ("exact", "base7", "family"):
        for m, (x,y,z) in zip(meta_nodes, coords_nodes):
            if m["level"] != level: continue
            ax.scatter([x],[y],[z], c=FAM_COLORS[m["fam"]],
                       s=level_size[level], marker=level_mark[level],
                       edgecolors="white", linewidths=0.5 if level=="exact" else 1.0,
                       alpha=0.9, zorder=5)
            if level in ("family", "base7"):
                label = m.get("display", m["name"])
                ax.text(x, y, z, f" {label}", fontsize=6 if level=="base7" else 8,
                        color=FAM_COLORS[m["fam"]], fontweight="bold")

    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.set_edgecolor("#1e2c3a")

    # legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=FAM_COLORS[f], label=f) for f in TREE]
    ax.legend(handles=handles, loc="upper left", framealpha=0.3,
              facecolor="#111e2e", edgecolor="#253447",
              labelcolor="#e2e8f0", fontsize=8)
    ax.set_title("Chord-quality LTAS distribution embedding (3D)\n"
                 "○ family  □ base7  ◇ exact  |  cloud = individual segments",
                 color="#e2e8f0", fontsize=11, pad=8)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"→ {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method",   default="umap", choices=["umap","tsne"])
    ap.add_argument("--n-songs",  type=int, default=40)
    ap.add_argument("--seed",     type=int, default=42)
    ap.add_argument("--out-html", default=None)
    ap.add_argument("--out-png",  default=None)
    args = ap.parse_args()

    out_html = Path(args.out_html) if args.out_html else OUT_HTML
    out_png  = Path(args.out_png)  if args.out_png  else OUT_PNG

    if not TREE_CACHE.exists():
        print(f"ERROR: {TREE_CACHE} not found — run chord_tree_ltas.py first"); sys.exit(1)
    d = np.load(TREE_CACHE)
    dist = {k: d[k] for k in d.files}

    print(f"Collecting segment vectors ({args.n_songs} songs, clean chord-only)...")
    seg_vecs, seg_meta = collect_segment_vectors(args.n_songs)
    print(f"  {len(seg_vecs)} segment vectors")

    node_vecs, node_meta = node_vectors(dist)
    print(f"  {len(node_vecs)} node means (5 family + 14 base7 + 18 exact)")

    # embed everything jointly so node means sit inside their clouds
    all_vecs = np.vstack([seg_vecs, node_vecs])
    print(f"Embedding {len(all_vecs)} points with {args.method.upper()} (3D)...")
    coords = embed_3d(all_vecs, args.method, args.seed)

    coords_segs  = coords[:len(seg_vecs)]
    coords_nodes = coords[len(seg_vecs):]

    print("Writing outputs...")
    write_html(coords_segs, seg_meta, coords_nodes, node_meta, out_html)
    write_png (coords_segs, seg_meta, coords_nodes, node_meta, out_png)
