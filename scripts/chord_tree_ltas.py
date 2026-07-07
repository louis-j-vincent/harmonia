"""Chord quality tree — LTAS chroma distributions + greedy top-down search.

Tree structure (3 levels):
  L0  family  (5)  : major / minor / diminished / augmented / suspended
  L1  base7   (14) : majT / maj7 / dom7 / minT / min7 / minmaj7 / dimT / dim7 /
                     m7b5 / augT / aug7 / augmaj7 / susT / 7sus4
  L2  exact   (18) : maj / 6 / maj7 / dom7 / dom7alt / min / m6 / min7 /
                     minmaj7 / dim / dim7 / m7b5 / aug / aug7 / augmaj7 /
                     sus2 / sus4 / 7sus4

Each node's distribution = corpus mean/std of root-shifted LTAS-CQT chroma
pooled over all segments whose exact quality maps to that node.

Greedy top-down search (given a query chroma vector x):
  1. At L0: score all 5 families → take argmax → descend into winner's children.
  2. At L1: score only the children of the winning family → take argmax → descend.
  3. At L2: score only the children of the winning base7 → take argmax.
  Never backtracks, O(5 + 3 + 2) = O(10) LL evaluations worst-case vs O(18×12)=O(216)
  for exhaustive search.

Score function: max diagonal-Gaussian log-likelihood over 12 root shifts.

Usage:
    .venv/bin/python scripts/chord_tree_ltas.py               # build + plot + eval
    .venv/bin/python scripts/chord_tree_ltas.py --n-songs 80
    .venv/bin/python scripts/chord_tree_ltas.py --rebuild      # force re-render
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

from analyze_accomp_emission import parse_chord, song_chord_spans
from build_accomp_audio_hard import render_to_array, stem_midi, SOUNDFONTS
from build_audio_chord_features import BUCKET_FAMILY, BUCKET_BASE7
from harmonia.data.midi_renderer import MIDIRenderer

DB         = REPO / "data" / "accomp_db" / "db.jsonl"
MANIFEST   = REPO / "data" / "accomp_db" / "audio" / "manifest.jsonl"
TREE_CACHE = REPO / "data" / "cache" / "chord_tree_ltas.npz"
PLOT_OUT   = REPO / "docs" / "plots" / "chord_tree_ltas.png"

NOTE   = ["C","Db","D","Eb","E","F","F#","G","Ab","A","Bb","B"]
DEGREE = ["R","b2","2","b3","3","4","b5","5","#5","6","b7","7"]

# ── tree definition ──────────────────────────────────────────────────────────
# L0 → L1 → L2  (exact quality)
TREE = {
    "major":      {
        "majT":    ["maj", "6"],
        "maj7":    ["maj7"],
        "dom7":    ["dom7", "dom7alt"],
    },
    "minor":      {
        "minT":    ["min", "m6"],
        "min7":    ["min7"],
        "minmaj7": ["minmaj7"],
    },
    "diminished": {
        "dimT":    ["dim"],
        "dim7":    ["dim7"],
        "m7b5":    ["m7b5"],
    },
    "augmented":  {
        "augT":    ["aug"],
        "aug7":    ["aug7"],
        "augmaj7": ["augmaj7"],
    },
    "suspended":  {
        "susT":    ["sus2", "sus4"],
        "7sus4":   ["7sus4"],
    },
}

# Theoretical chord tones per exact quality (for overlay markers, relative to root)
CHORD_TONES = {
    "maj":     [0,4,7],       "6":      [0,4,7,9],
    "maj7":    [0,4,7,11],    "dom7":   [0,4,7,10],   "dom7alt": [0,4,7,10],
    "min":     [0,3,7],       "m6":     [0,3,7,9],
    "min7":    [0,3,7,10],    "minmaj7":[0,3,7,11],
    "dim":     [0,3,6],       "dim7":   [0,3,6,9],    "m7b5":    [0,3,6,10],
    "aug":     [0,4,8],       "aug7":   [0,4,8,10],   "augmaj7": [0,4,8,11],
    "sus2":    [0,2,7],       "sus4":   [0,5,7],      "7sus4":   [0,5,7,10],
}

FAM_COLORS = {
    "major":"#58d4ff","minor":"#a65fd4","diminished":"#e34948",
    "augmented":"#e0a03b","suspended":"#1baf7a",
}
BASE7_FAM = {b7: fam for fam, children in TREE.items() for b7 in children}
EXACT_FAM = {ex: fam for fam, children in TREE.items()
             for b7, exacts in children.items() for ex in exacts}


# ── helpers ──────────────────────────────────────────────────────────────────

def _render_chord_only(midi_path, sf_name, renderer):
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    chord_pm = stem_midi(pm, lambda i: (not i.is_drum) and "bass" not in i.name.lower())
    if chord_pm is None or not chord_pm.instruments:
        return None, None
    audio, sr = render_to_array(renderer, chord_pm, sf_name, reverb=False)
    return audio.astype(float), sr


def _ltas_cqt(audio, sr, hop=512):
    raw = librosa.feature.chroma_cqt(y=audio, sr=sr, bins_per_octave=36, hop_length=hop)
    ltas = raw.mean(axis=1, keepdims=True)
    ltas = np.where(ltas < 1e-9, 1.0, ltas)
    return raw / ltas, librosa.frames_to_time(np.arange(raw.shape[1]), sr=sr, hop_length=hop)


def _diag_ll(x, mu, std):
    return float(-0.5 * np.sum(((x - mu) / std) ** 2) - np.sum(np.log(std)))


def _max_ll_over_keys(x, mu, std):
    best_ll, best_r = -np.inf, 0
    for r in range(12):
        ll = _diag_ll(np.roll(x, -r), mu, std)
        if ll > best_ll: best_ll, best_r = ll, r
    return best_ll, best_r


# ── data collection ──────────────────────────────────────────────────────────

def build_distributions(n_songs: int) -> dict:
    """Collect root-shifted LTAS chroma per exact quality, then aggregate up."""
    recs = {json.loads(l)["song_id"]: json.loads(l) for l in open(DB)}
    man: dict = {}
    for m in map(json.loads, open(MANIFEST)):
        if m["song_id"] not in man or m.get("transpose", 0) == 0:
            man[m["song_id"]] = m
    avail = sorted([s for s in recs if s in man], key=lambda s: recs[s]["title"])
    chosen = avail[:n_songs]
    renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")
    sf_name  = SOUNDFONTS[0]

    by_exact: dict[str, list[np.ndarray]] = defaultdict(list)

    for i, sid in enumerate(chosen):
        rec = recs[sid]; m = man[sid]
        bpb = m["beats_per_bar"]; spb = 60.0 / m["tempo"]
        print(f"\r  [{i+1}/{len(chosen)}] {rec['title'][:42]:42s}", end="", flush=True)
        try:
            audio, sr = _render_chord_only(REPO / m["midi_path"], sf_name, renderer)
            if audio is None: continue
        except Exception:
            continue
        chroma, ct = _ltas_cqt(audio, sr)
        chord_at = {(e["bar"]-1)*bpb+e["beat"]: e for e in rec["chord_timeline"]}
        for t0, t1, root_gt, _ in song_chord_spans(rec):
            b0  = int(round(t0 / spb))
            mma = chord_at.get(b0, {}).get("mma")
            p   = parse_chord(mma) if mma else None
            if p is None or p[1] not in BUCKET_FAMILY: continue
            exact = p[1]
            i0 = int(np.searchsorted(ct, t0))
            i1 = int(np.searchsorted(ct, t1))
            if i1 <= i0: i1 = i0 + 1
            seg = chroma[:, i0:i1].mean(axis=1)
            shifted = np.roll(seg, -int(root_gt % 12))
            n = np.linalg.norm(shifted)
            if n > 1e-9:
                by_exact[exact].append(shifted / n)
    print()

    # aggregate: L2 exact, L1 base7 (union of children), L0 family (union)
    out = {}
    for fam, b7_dict in TREE.items():
        fam_vecs = []
        for b7, exacts in b7_dict.items():
            b7_vecs = []
            for ex in exacts:
                vecs = by_exact.get(ex, [])
                if len(vecs) < 2:
                    vecs = [np.zeros(12), np.ones(12)/np.sqrt(12)]  # fallback
                arr = np.stack(vecs)
                out[f"exact_{ex}_mu"]  = arr.mean(0)
                out[f"exact_{ex}_std"] = arr.std(0) + 1e-4
                out[f"exact_{ex}_n"]   = np.array([len(vecs)])
                b7_vecs.extend(vecs)
            arr = np.stack(b7_vecs)
            out[f"b7_{b7}_mu"]  = arr.mean(0)
            out[f"b7_{b7}_std"] = arr.std(0) + 1e-4
            out[f"b7_{b7}_n"]   = np.array([len(b7_vecs)])
            fam_vecs.extend(b7_vecs)
        arr = np.stack(fam_vecs)
        out[f"fam_{fam}_mu"]  = arr.mean(0)
        out[f"fam_{fam}_std"] = arr.std(0) + 1e-4
        out[f"fam_{fam}_n"]   = np.array([len(fam_vecs)])
    return out


def load_or_build(n_songs: int, rebuild: bool) -> dict:
    if not rebuild and TREE_CACHE.exists():
        print(f"Loading {TREE_CACHE.name}")
        d = np.load(TREE_CACHE)
        return {k: d[k] for k in d.files}
    print(f"Building chord-tree distributions from {n_songs} songs...")
    dist = build_distributions(n_songs)
    np.savez(TREE_CACHE, **dist)
    print(f"  Saved → {TREE_CACHE}")
    return dist


# ── greedy top-down search ────────────────────────────────────────────────────

def greedy_tree_search(x12: np.ndarray, dist: dict) -> tuple[str, str, str, int]:
    """
    x12 : L2-normalised root-shifted LTAS chroma (12,)
    Returns (family, base7, exact, best_root_pc).

    Algorithm:
      L0: score 5 families   → argmax → winning_family
      L1: score children of winning_family (2-3 base7s) → argmax → winning_b7
      L2: score children of winning_b7 (1-2 exacts) → argmax → winning_exact
    Root is the argmax key from the L0 winner (consistent root across levels).
    """
    # L0 — family
    fam_scores = {}
    fam_roots  = {}
    for fam in TREE:
        ll, r = _max_ll_over_keys(x12, dist[f"fam_{fam}_mu"], dist[f"fam_{fam}_std"])
        fam_scores[fam] = ll
        fam_roots[fam]  = r
    best_fam = max(fam_scores, key=fam_scores.__getitem__)
    best_root = fam_roots[best_fam]

    # L1 — base7 within winning family
    b7_scores = {}
    for b7 in TREE[best_fam]:
        ll, _ = _max_ll_over_keys(x12, dist[f"b7_{b7}_mu"], dist[f"b7_{b7}_std"])
        b7_scores[b7] = ll
    best_b7 = max(b7_scores, key=b7_scores.__getitem__)

    # L2 — exact within winning base7
    ex_scores = {}
    for ex in TREE[best_fam][best_b7]:
        ll, _ = _max_ll_over_keys(x12, dist[f"exact_{ex}_mu"], dist[f"exact_{ex}_std"])
        ex_scores[ex] = ll
    best_ex = max(ex_scores, key=ex_scores.__getitem__)

    return best_fam, best_b7, best_ex, best_root


# ── plotting ─────────────────────────────────────────────────────────────────

def plot_tree(dist: dict, out: Path):
    """One panel per node — arranged as a tree grid.

    Layout: 3 level-rows. Within each level, nodes are grouped by family.
    Each panel: mean chroma bars + ±std band + chord-tone markers.
    """
    families = list(TREE.keys())

    # Count total panels per level
    l0_nodes = families                                   # 5
    l1_nodes = [(fam, b7) for fam in families            # 14
                for b7 in TREE[fam]]
    l2_nodes = [(fam, b7, ex) for fam in families        # 18
                for b7, exacts in TREE[fam].items()
                for ex in exacts]

    col_counts = [len(l0_nodes), len(l1_nodes), len(l2_nodes)]
    n_cols = max(col_counts)   # 18
    n_rows = 3

    fig_w = n_cols * 1.55
    fig_h = n_rows * 2.6
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#0d1520")
    fig.suptitle(
        "LTAS-CQT chroma distributions — full chord quality tree  (root-shifted, clean audio)",
        color="#e2e8f0", fontsize=11, y=1.002)

    def _panel(ax, mu, std, n, tones, col, label):
        ax.set_facecolor("#0a0e14")
        xs = np.arange(12)
        ax.bar(xs, mu + std, color=col+"22", width=0.8)
        ax.bar(xs, np.maximum(0, mu - std), color="#0a0e14", width=0.8)
        ax.bar(xs, mu, color=col+"99", width=0.8, edgecolor=col, linewidth=0.6)
        for t in tones:
            ax.axvline(t, color=col, lw=1.2, alpha=0.85)
        ax.set_xticks(xs)
        ax.set_xticklabels(DEGREE, fontsize=5.5, color="#6a7a8e")
        ax.set_xlim(-0.6, 11.6)
        ax.set_ylim(-0.02, max(mu.max() + std.max() + 0.05, 0.5))
        ax.tick_params(axis="y", colors="#3a4a5e", labelsize=4.5)
        ax.spines[:].set_color("#1e2c3a")
        ax.set_title(f"{label}  n={int(n)}", color=col, fontsize=6.5, pad=2)

    # ── L0 family row ─────────────────────────────────────────────────────────
    # Centre each family's block over its L1 children
    # Build column-index map: for each L1 node, assign a column (0..13)
    l1_col = {}
    c = 0
    fam_l1_cols = {}      # fam → list of L1 column indices
    for fam in families:
        cols_this = []
        for b7 in TREE[fam]:
            l1_col[(fam, b7)] = c
            cols_this.append(c)
            c += 1
        fam_l1_cols[fam] = cols_this

    # L0 panels centred over their L1 children using gridspec with merged cells
    gs_l0 = gridspec.GridSpec(1, len(l1_nodes), figure=fig,
                               left=0.02, right=0.98, top=0.95, bottom=0.69,
                               wspace=0.35)
    for fam in families:
        cols = fam_l1_cols[fam]
        col_start, col_end = cols[0], cols[-1]
        ax = fig.add_subplot(gs_l0[0, col_start:col_end+1])
        col = FAM_COLORS[fam]
        mu  = dist[f"fam_{fam}_mu"]
        std = dist[f"fam_{fam}_std"]
        n   = int(dist[f"fam_{fam}_n"][0])
        # aggregate chord tones: union of all exact tones in family
        tones_set = set()
        for b7, exacts in TREE[fam].items():
            for ex in exacts:
                tones_set.update(CHORD_TONES.get(ex, []))
        _panel(ax, mu, std, n, sorted(tones_set), col, fam)

    # ── L1 base7 row ──────────────────────────────────────────────────────────
    gs_l1 = gridspec.GridSpec(1, len(l1_nodes), figure=fig,
                               left=0.02, right=0.98, top=0.665, bottom=0.365,
                               wspace=0.35)
    for fam in families:
        col = FAM_COLORS[fam]
        for b7 in TREE[fam]:
            ci = l1_col[(fam, b7)]
            ax = fig.add_subplot(gs_l1[0, ci])
            mu  = dist[f"b7_{b7}_mu"]
            std = dist[f"b7_{b7}_std"]
            n   = int(dist[f"b7_{b7}_n"][0])
            tones_set = set()
            for ex in TREE[fam][b7]:
                tones_set.update(CHORD_TONES.get(ex, []))
            _panel(ax, mu, std, n, sorted(tones_set), col, b7)

    # ── L2 exact row ──────────────────────────────────────────────────────────
    # assign exact columns left-to-right, grouped by family then base7
    l2_col = {}
    c = 0
    b7_l2_cols = {}
    for fam in families:
        for b7, exacts in TREE[fam].items():
            b7_l2_cols[(fam, b7)] = []
            for ex in exacts:
                l2_col[(fam, b7, ex)] = c
                b7_l2_cols[(fam, b7)].append(c)
                c += 1
    # pad to n_cols with empty axes
    gs_l2 = gridspec.GridSpec(1, n_cols, figure=fig,
                               left=0.02, right=0.98, top=0.34, bottom=0.04,
                               wspace=0.35)
    for fam in families:
        col = FAM_COLORS[fam]
        for b7, exacts in TREE[fam].items():
            for ex in exacts:
                ci = l2_col[(fam, b7, ex)]
                ax = fig.add_subplot(gs_l2[0, ci])
                mu  = dist[f"exact_{ex}_mu"]
                std = dist[f"exact_{ex}_std"]
                n   = int(dist[f"exact_{ex}_n"][0])
                _panel(ax, mu, std, n, CHORD_TONES.get(ex, []), col, ex)

    # level labels on the left
    for y_mid, label in [(0.82, "L0\nfamily"), (0.52, "L1\nbase7"), (0.19, "L2\nexact")]:
        fig.text(0.001, y_mid, label, color="#5a6a7e", fontsize=8,
                 va="center", ha="left", style="italic")

    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"→ {out}")


# ── evaluation: greedy search vs exhaustive search ────────────────────────────

def evaluate(n_songs: int, dist: dict, rng) -> None:
    """Compare greedy-tree accuracy vs flat family classifier on hard audio."""
    from build_accomp_audio_hard import (
        SCENARIOS, LEAD_PROGRAMS, make_melody, time_varying_degrade,
    )
    import pretty_midi

    recs = {json.loads(l)["song_id"]: json.loads(l) for l in open(DB)}
    man: dict = {}
    for m in map(json.loads, open(MANIFEST)):
        if m["song_id"] not in man or m.get("transpose", 0) == 0:
            man[m["song_id"]] = m
    avail = sorted([s for s in recs if s in man], key=lambda s: recs[s]["title"])
    chosen = avail[:n_songs]
    renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")

    results = []
    for i, sid in enumerate(chosen):
        rec = recs[sid]; m = man[sid]
        bpb = m["beats_per_bar"]; spb = 60.0 / m["tempo"]
        print(f"\r  eval [{i+1}/{len(chosen)}] {rec['title'][:38]:38s}", end="", flush=True)

        scen    = str(rng.choice(list(SCENARIOS)))
        gains   = {k: v * float(rng.uniform(0.8, 1.2)) for k, v in SCENARIOS[scen].items()}
        sf_name = SOUNDFONTS[int(rng.integers(0, len(SOUNDFONTS)))]
        try:
            pm = pretty_midi.PrettyMIDI(str(REPO / m["midi_path"]))
            stems = {
                "chords": stem_midi(pm, lambda i: (not i.is_drum) and "bass" not in i.name.lower()),
                "bass":   stem_midi(pm, lambda i: "bass" in i.name.lower()),
                "drums":  stem_midi(pm, lambda i: i.is_drum),
            }
            if gains.get("melody", 0) > 0.01:
                mel_pm = pretty_midi.PrettyMIDI()
                mel = make_melody(pm, int(rng.choice(LEAD_PROGRAMS)), rng)
                if mel: mel_pm.instruments.append(mel); stems["melody"] = mel_pm
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
            audio = mix.astype(float)
        except Exception:
            continue

        chroma, ct = _ltas_cqt(audio, sr)
        chord_at = {(e["bar"]-1)*bpb+e["beat"]: e for e in rec["chord_timeline"]}
        for t0, t1, root_gt, _ in song_chord_spans(rec):
            b0  = int(round(t0 / spb))
            mma = chord_at.get(b0, {}).get("mma")
            p   = parse_chord(mma) if mma else None
            if p is None or p[1] not in BUCKET_FAMILY: continue
            gt_exact = p[1]
            gt_b7    = BUCKET_BASE7.get(gt_exact, "?")
            gt_fam   = BUCKET_FAMILY.get(gt_exact, "?")
            i0 = int(np.searchsorted(ct, t0)); i1 = int(np.searchsorted(ct, t1))
            if i1 <= i0: i1 = i0 + 1
            seg = chroma[:, i0:i1].mean(axis=1)
            shifted = np.roll(seg, -int(root_gt % 12))
            n = np.linalg.norm(shifted)
            if n < 1e-9: continue
            x12 = shifted / n

            pred_fam, pred_b7, pred_ex, _ = greedy_tree_search(x12, dist)
            results.append({
                "gt_fam": gt_fam, "gt_b7": gt_b7, "gt_ex": gt_exact,
                "pred_fam": pred_fam, "pred_b7": pred_b7, "pred_ex": pred_ex,
            })
    print()

    if not results:
        print("No results."); return

    fam_acc = np.mean([r["gt_fam"] == r["pred_fam"] for r in results])
    b7_acc  = np.mean([r["gt_b7"]  == r["pred_b7"]  for r in results])
    ex_acc  = np.mean([r["gt_ex"]  == r["pred_ex"]  for r in results])
    print(f"\n  Greedy tree search  (N={len(results)} segments, hard audio, oracle boundaries)")
    print(f"  {'Level':12s}  {'Accuracy':>9s}")
    print(f"  {'-'*24}")
    print(f"  {'L0 family':12s}  {fam_acc:9.1%}")
    print(f"  {'L1 base7':12s}  {b7_acc:9.1%}")
    print(f"  {'L2 exact':12s}  {ex_acc:9.1%}")
    print()
    print("  Per-family breakdown (L0):")
    for fam in TREE:
        sub = [r for r in results if r["gt_fam"] == fam]
        if not sub: continue
        acc = np.mean([r["gt_fam"] == r["pred_fam"] for r in sub])
        print(f"    {fam:12s}: {acc:.1%}  (n={len(sub)})")


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-songs",  type=int,  default=60)
    ap.add_argument("--rebuild",  action="store_true")
    ap.add_argument("--eval-songs", type=int, default=20,
                    help="additional songs for greedy-search eval (different from dist)")
    ap.add_argument("--seed",     type=int,  default=42)
    args = ap.parse_args()

    dist = load_or_build(args.n_songs, args.rebuild)

    print("Plotting chord tree distributions...")
    plot_tree(dist, PLOT_OUT)

    # print node sample counts
    print("\nNode sample counts:")
    for fam in TREE:
        n_fam = int(dist[f"fam_{fam}_n"][0])
        print(f"  {fam:12s}  n={n_fam}")
        for b7 in TREE[fam]:
            n_b7 = int(dist[f"b7_{b7}_n"][0])
            print(f"    {b7:12s}  n={n_b7}")
            for ex in TREE[fam][b7]:
                n_ex = int(dist[f"exact_{ex}_n"][0])
                print(f"      {ex:12s}  n={n_ex}")

    print(f"\nRunning greedy-tree eval on {args.eval_songs} held-out songs...")
    rng = np.random.default_rng(args.seed)
    evaluate(args.eval_songs, dist, rng)
