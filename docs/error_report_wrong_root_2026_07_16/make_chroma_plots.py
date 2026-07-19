"""Render TEMPORAL (frame-by-frame) BP48 chroma heatmaps for the 18 selected
wrong-root examples -- REUSES the exact plotting method from
docs/bleed_verification_2026_07_16 (scratchpad/bleed_verify.py) instead of
the earlier single-pooled-snapshot heatmap this file used to produce.

Per user complaint (2026-07-16): the first version of this tool reinvented a
simpler 4x12 "pooled snapshot" chroma plot instead of reusing the
bleed_verification tool's frame-by-frame (86.13 Hz) temporal chroma with the
exact pooled-window box overlay. This version imports
pooled_window()/frame_chroma() directly from scratchpad/bleed_verify.py
(kept importable, not copy-pasted) and reuses its make_plot() rendering
verbatim, only swapping the overlay semantics: GT root (green) / PREDICTED
root (red dashed) instead of GT/next-chord-root, since this is an
error-diagnosis tool -- plus the bass-argmax triangle diagnostic this tool
already had.

Re-runs PitchExtractor per song (12 distinct RWC songs across the 18
examples) via the same RemoteZip mechanism as bleed_verify.py / fetch_clips.py
-- read-only against the corpus zip, WAVs deleted after each song, BP cache
isolated to scratchpad.
"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

REPO = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scratchpad"))
from bleed_verify import pooled_window, frame_chroma, PC  # exact reuse
from scripts.build_rwc_corpus import fetch_chords, ZIP_URL
from harmonia.models.stage1_pitch import PitchExtractor
from remotezip import RemoteZip

OUT_DIR = REPO / "docs/error_report_wrong_root_2026_07_16"
CHROMA_DIR = OUT_DIR / "chroma"
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

SCRATCH = Path("/private/tmp/claude-501/-Users-vincente-Documents-Projets-Perso-Code-harmonia/"
               "a6a4757d-a729-450a-a5be-b9970a43c412/scratchpad")
BP_CACHE = SCRATCH / "bp_cache_errortool"
WAV_DIR = SCRATCH / "wav_errortool"
BP_CACHE.mkdir(parents=True, exist_ok=True)
WAV_DIR.mkdir(parents=True, exist_ok=True)

manifest = json.loads((OUT_DIR / "examples_manifest.json").read_text())
examples = manifest["examples"]

by_song = defaultdict(list)
for i, ex in enumerate(examples):
    by_song[ex["song_id"]].append((i, ex))


def make_plot(ex, ft, on, nt, next_t, path):
    """Same rendering as bleed_verify.make_plot: 4 BP48 blocks, per-frame
    (86.13 Hz) temporal chroma, exact pooled-window box, wider display
    context on both sides. Overlay swapped to GT (green) / PRED (red dashed)
    root lines + bass-argmax triangle (this tool's existing diagnostic)."""
    t0, t1 = ex["t0"], ex["t1"]
    i0, i1, floored = pooled_window(ft, t0, t1)
    # Display window == the audio clip window EXACTLY [t0,t1). The paired audio
    # is trimmed to sample-exact [t0,t1) (fetch_clips.py); the chroma MUST span
    # the same time so the two line up. Previously this reused bleed_verify's
    # asymmetric wider context [t0-0.20, t1+0.45] -- correct for a bleed proof,
    # but here it made the chroma show a DIFFERENT (wider, right-shifted) span
    # than the audio (user report 2026-07-16).
    d0 = int(np.searchsorted(ft, t0, side="left"))
    d1 = int(np.searchsorted(ft, t1, side="left"))
    d0 = max(0, d0); d1 = min(len(ft), d1)
    fr = np.arange(d0, d1)
    times = ft[fr]
    blocks = [
        ("onset (all reg)", on, 0, 200),
        ("note (all reg)", nt, 0, 200),
        ("bass (MIDI<52)", on, 0, 52),
        ("treble (MIDI>=60)", on, 60, 200),
    ]
    fig, axes = plt.subplots(4, 1, figsize=(7.6, 7.4), sharex=True)
    gt, pred, bass_argmax = ex["gt_root"], ex["pred_root"], ex["bass_argmax_pc"]
    wx0 = ft[i0]; wx1 = ft[i1 - 1]
    for ax, (name, src, lo, hi) in zip(axes, blocks):
        M = np.stack([frame_chroma(src[j], lo, hi) for j in fr], axis=1)
        ax.imshow(M, aspect="auto", origin="lower", cmap="magma",
                  extent=[times[0], times[-1], -0.5, 11.5], vmin=0, vmax=1)
        ax.add_patch(Rectangle((wx0, -0.5), wx1 - wx0, 12,
                     fill=False, edgecolor="#39d353", lw=2.0))
        ax.axvline(t1, color="#ff4d4d", lw=1.6, ls="-")
        if next_t is not None:
            ax.axvline(next_t, color="#ffd166", lw=1.2, ls="--")
        ax.axhline(gt, color="#22c55e", lw=1.0, ls=":", alpha=0.9)
        ax.axhline(pred, color="#ef4444", lw=1.0, ls=":", alpha=0.9)
        if name.startswith("bass"):
            ax.plot([wx0 + (wx1 - wx0) / 2], [bass_argmax], marker="v",
                    color="#38bdf8", markersize=9, markeredgecolor="white",
                    markeredgewidth=0.7, zorder=5)
        ax.set_yticks(range(12)); ax.set_yticklabels(PC, fontsize=6)
        ax.set_ylabel(name, fontsize=7)
        ax.tick_params(labelsize=6)
    axes[-1].set_xlabel("time (s)  |  x-axis span == the audio clip [t0,t1) EXACTLY  |  "
                        "green box = frames model pools  |  red = span end t1  |  "
                        "blue triangle = bass-argmax", fontsize=6.2)
    axes[0].set_title(f"{ex['song_id']}  [{t0:.2f},{t1:.2f})  GT={ex['label']} (green)  "
                       f"pred={ex['pred_root_name']}:{ex['pred_quality']} (red)", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


n_ok = 0
for song_id, items in by_song.items():
    rwcid = song_id.replace("rwc_", "")
    print(f"[{rwcid}] fetching chords + WAV ({len(items)} examples)...", flush=True)
    rows = fetch_chords(rwcid)
    with RemoteZip(ZIP_URL) as z:
        names = {Path(inf.filename).stem: inf.filename for inf in z.infolist()
                 if inf.filename.endswith(".wav")}
        zname = names.get(rwcid)
        if zname is None:
            print(f"  !! {rwcid} not in zip, skipping")
            continue
        z.extract(zname, path=str(WAV_DIR))
        wav = WAV_DIR / zname

    extractor = PitchExtractor(cache_dir=BP_CACHE)
    acts = extractor.extract(wav)
    ft, on, nt = acts.frame_times, acts.onset_probs, acts.note_probs

    for i, ex in items:
        next_t = None
        if rows is not None:
            for ridx, (r0, r1, lab) in enumerate(rows):
                if abs(r0 - ex["t0"]) < 0.01:
                    if ridx + 1 < len(rows):
                        next_t = rows[ridx + 1][0]
                    break
        out = CHROMA_DIR / f"ex{i:02d}.png"
        make_plot(ex, ft, on, nt, next_t, out)
        n_ok += 1
        print(f"  ex{i:02d} {ex['label']:18s} GT={ex['gt_root_name']} pred={ex['pred_root_name']}", flush=True)

    wav.unlink(missing_ok=True)
    for f in BP_CACHE.glob("*"):
        f.unlink(missing_ok=True)

print(f"\nwrote {n_ok}/{len(examples)} temporal chroma plots -> {CHROMA_DIR}")
