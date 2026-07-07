"""Plot LTAS-normalised CQT chroma distributions per chord family.

Renders clean chord-only audio (no bass/drums/noise) for N songs, extracts
oracle GT segment chroma via librosa CQT, LTAS-normalises, root-shifts
(root → index 0), groups by family, and plots mean ± std as bar charts.

Usage:
    .venv/bin/python scripts/plot_ltas_chroma_by_family.py
    .venv/bin/python scripts/plot_ltas_chroma_by_family.py --n-songs 50
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
import numpy as np
import pretty_midi
import soundfile as sf

from analyze_accomp_emission import parse_chord, song_chord_spans
from build_accomp_audio_hard import render_to_array, stem_midi, SOUNDFONTS
from build_audio_chord_features import BUCKET_FAMILY, FAM_IDX
from harmonia.data.midi_renderer import MIDIRenderer

DB       = REPO / "data" / "accomp_db" / "db.jsonl"
MANIFEST = REPO / "data" / "accomp_db" / "audio" / "manifest.jsonl"
OUT      = REPO / "docs" / "plots" / "ltas_chroma_by_family.png"

FAMILIES  = ["major", "minor", "diminished", "augmented", "suspended"]
# Degree names relative to root (0=R, 1=b2, 2=2, 3=b3, 4=3, 5=4, 6=b5, 7=5, 8=#5, 9=6, 10=b7, 11=7)
DEGREE_NAMES = ["R", "b2", "2", "b3", "3", "4", "b5", "5", "#5", "6", "b7", "7"]

# Theoretical templates for reference overlay
_FAM_TONES = {
    "major":      [0, 4, 7],
    "minor":      [0, 3, 7],
    "diminished": [0, 3, 6],
    "augmented":  [0, 4, 8],
    "suspended":  [0, 5, 7],
}
# Jazz extension tones (7th) commonly added
_FAM_7TH = {
    "major": [11],       # maj7
    "minor": [10],       # b7
    "diminished": [9],   # dim7 / b7
    "augmented": [],
    "suspended": [10],   # b7
}


def collect_chroma(n_songs: int, seed: int) -> dict[str, list[np.ndarray]]:
    """Render chord-only audio and extract LTAS-CQT chroma per oracle segment."""
    rng = np.random.default_rng(seed)
    recs = {json.loads(l)["song_id"]: json.loads(l) for l in open(DB)}
    man: dict = {}
    for m in map(json.loads, open(MANIFEST)):
        if m["song_id"] not in man or m.get("transpose", 0) == 0:
            man[m["song_id"]] = m

    avail = sorted([s for s in recs if s in man], key=lambda s: recs[s]["title"])
    chosen = avail[:n_songs]

    renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")
    sf_name  = SOUNDFONTS[0]  # consistent soundfont

    by_family: dict[str, list[np.ndarray]] = {f: [] for f in FAMILIES}

    for i, sid in enumerate(chosen):
        rec = recs[sid]; m = man[sid]
        print(f"\r[{i+1}/{len(chosen)}] {rec['title'][:40]:40s}", end="", flush=True)

        bpb = m["beats_per_bar"]
        spb = 60.0 / m["tempo"]
        hop = 512

        try:
            pm = pretty_midi.PrettyMIDI(str(REPO / m["midi_path"]))
            # chord-only stem (no bass, no drums)
            chord_pm = stem_midi(pm, lambda inst: (not inst.is_drum)
                                                   and "bass" not in inst.name.lower())
            if chord_pm is None or not chord_pm.instruments:
                continue
            audio, sr = render_to_array(renderer, chord_pm, sf_name, reverb=False)
        except Exception as e:
            print(f"\n  skip {rec['title']}: {e}"); continue

        audio = audio.astype(float)

        # LTAS-normalised CQT
        chroma_raw = librosa.feature.chroma_cqt(
            y=audio, sr=sr, bins_per_octave=36, hop_length=hop)
        ltas = chroma_raw.mean(axis=1, keepdims=True)
        ltas = np.where(ltas < 1e-9, 1.0, ltas)
        chroma = chroma_raw / ltas      # (12, T), mean≈1 per row
        ct = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop)

        chord_at = {(e["bar"] - 1) * bpb + e["beat"]: e
                    for e in rec["chord_timeline"]}

        for t0, t1, root_gt, _ in song_chord_spans(rec):
            b0 = int(round(t0 / spb))
            mma = chord_at.get(b0, {}).get("mma")
            p = parse_chord(mma) if mma else None
            if p is None or p[1] not in BUCKET_FAMILY:
                continue
            fam = BUCKET_FAMILY[p[1]]
            if fam not in by_family:
                continue

            i0 = int(np.searchsorted(ct, t0))
            i1 = int(np.searchsorted(ct, t1))
            if i1 <= i0:
                i1 = i0 + 1
            seg = chroma[:, i0:i1].mean(axis=1)   # (12,)

            # root-shift: roll so root_gt → index 0
            shifted = np.roll(seg, -int(root_gt % 12))
            # L2-normalise for shape comparison (not magnitude)
            n = np.linalg.norm(shifted)
            if n > 1e-9:
                by_family[fam].append(shifted / n)

    print()
    return by_family


def plot(by_family: dict[str, list[np.ndarray]], out: Path) -> None:
    n_fam = len(FAMILIES)
    fig, axes = plt.subplots(1, n_fam, figsize=(18, 4.5),
                             facecolor="#0d1520", sharey=False)
    fig.suptitle(
        "LTAS-normalised CQT chroma by chord family  (root-shifted: index 0 = root)",
        color="#e2e8f0", fontsize=13, y=1.01)

    fam_colors = {
        "major":      "#58d4ff",
        "minor":      "#a65fd4",
        "diminished": "#e34948",
        "augmented":  "#e0a03b",
        "suspended":  "#1baf7a",
    }

    for ax, fam in zip(axes, FAMILIES):
        vecs = np.stack(by_family[fam]) if by_family[fam] else np.zeros((1, 12))
        mu  = vecs.mean(axis=0)
        std = vecs.std(axis=0)
        n   = len(by_family[fam])

        col = fam_colors[fam]
        xs  = np.arange(12)

        # std band
        ax.bar(xs, mu + std, color=col + "22", width=0.7)
        ax.bar(xs, np.maximum(0, mu - std), color="#0d1520", width=0.7)  # subtract band
        # mean bars
        ax.bar(xs, mu, color=col + "aa", width=0.7, edgecolor=col, linewidth=0.8)

        # theoretical chord tone markers
        for tone in _FAM_TONES[fam]:
            ax.axvline(tone, color=col, lw=1.5, alpha=0.9, linestyle="-")
        for tone in _FAM_7TH[fam]:
            ax.axvline(tone, color=col, lw=1.0, alpha=0.55, linestyle="--")

        ax.set_facecolor("#0d1520")
        ax.set_xticks(xs)
        ax.set_xticklabels(DEGREE_NAMES, fontsize=8.5, color="#88aacc")
        ax.tick_params(axis="y", colors="#5a6a7e", labelsize=7)
        ax.spines[:].set_color("#253447")
        ax.set_title(f"{fam}\n(n={n})", color=col, fontsize=11, pad=5)
        ax.set_xlim(-0.6, 11.6)
        ax.set_xlabel("degree (root-shifted)", color="#5a6a7e", fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel("mean LTAS chroma (L2-normed)", color="#5a6a7e", fontsize=8)

    fig.text(0.5, -0.03,
             "Bars = mean  ·  shaded band = ±1 std  ·  solid vertical = triad tone  ·  dashed = 7th",
             ha="center", color="#5a6a7e", fontsize=8.5)

    plt.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"→ {out}")
    for fam in FAMILIES:
        print(f"  {fam:12s}: {len(by_family[fam]):4d} segments")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-songs", type=int, default=30)
    ap.add_argument("--seed",    type=int, default=42)
    ap.add_argument("--out",     default=None)
    args = ap.parse_args()
    out = Path(args.out) if args.out else OUT

    by_family = collect_chroma(args.n_songs, args.seed)
    plot(by_family, out)
