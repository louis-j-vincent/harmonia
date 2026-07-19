"""Boundary-bleed VERIFICATION tool (2026-07-16).

Proves-or-disproves the "0.0ms contamination" claim for the FIXED RWC feature
extraction (scripts/build_rwc_corpus.py::build_song, frame-clip pooling).

For a set of chord spans it:
  1. Re-runs the EXACT fixed frame-selection logic (searchsorted 'left' + the
     MIN_FRAMES=4 floor) so we know precisely which BasicPitch frames the model
     pools for each example.
  2. Trims audio to EXACTLY [t0,t1) (sample-accurate WAV, ZERO padding) and
     verifies duration == t1-t0 to <1 sample via ffprobe.
  3. Renders the per-frame (86.13 Hz) temporal chroma for all 4 BP48 blocks,
     with a WIDER context window drawn but the EXACT pooled [i0,i1) frames boxed
     and t1 / next-chord onset marked, so any next-chord bleed is visible.

Isolated cache (scratchpad), WAVs deleted after trimming, npz caches read-only.
"""
from __future__ import annotations
import sys, json, subprocess, shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

REPO = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia")
sys.path.insert(0, str(REPO))
from scripts.build_rwc_corpus import fetch_chords, ZIP_URL, MIN_FRAMES
from harmonia.models.stage1_pitch import PitchExtractor
from harmonia.models.chord_pipeline_v1 import _reg_raw
from scripts.build_jaah_corpus import parse_jaah as parse_harte
from remotezip import RemoteZip

SCRATCH = Path("/private/tmp/claude-501/-Users-vincente-Documents-Projets-Perso-Code-harmonia/"
               "a6a4757d-a729-450a-a5be-b9970a43c412/scratchpad")
BP_CACHE = SCRATCH / "bp_cache_bleedverify"
WAV_DIR = SCRATCH / "wav_bleedverify"
OUT = REPO / "docs/bleed_verification_2026_07_16"
CLIPS = OUT / "clips"
CHROMA = OUT / "chroma"
for d in (BP_CACHE, WAV_DIR, CLIPS, CHROMA):
    d.mkdir(parents=True, exist_ok=True)

PC = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# (song, chord-row index, human note). Mix of short chords + longer controls,
# continuity with the earlier root_error_analysis set (P091 C:7, P071 Bb:sus4b7).
SEL = [
    ("RWC_P091", 21, "C:7 -> F:min  (THE smoking gun: old pred=F = next root)"),
    ("RWC_P091", 23, "Eb:maj/2 -> F:min  (short, inversion)"),
    ("RWC_P091", 111, "Eb:maj -> F:min  (short)"),
    ("RWC_P091", 18 if False else 20, "control long span P091"),  # placeholder replaced below
    ("RWC_P071", 29, "Bb:sus4(b7) -> Eb:maj  (0.61s, P4 up)"),
    ("RWC_P071", 68, "Bb:7 -> Eb:maj  (0.57s, dominant resolution P4)"),
    ("RWC_P071", 81, "Eb:maj/3 -> F:min7  (0.43s SHORTEST, MIN_FRAMES stress)"),
    ("RWC_P071", 103, "Bb:sus4(b7) -> Ab:maj  (1.74s control, earlier example)"),
    ("RWC_P071", 110, "D:min7/b3 -> E:7  (0.61s)"),
    ("RWC_P071", 18, "Bb:maj -> C:min  (1.86s clean long control)"),
]
# fix the P091 long control -> use idx 18 area? P091 idx list; use a real long one:
SEL[3] = ("RWC_P091", 22, "F:min -> Eb:maj/2  (1.95s control)")


def root_pc(lab):
    r, fam, _ = parse_harte(lab)
    return (int(r) % 12) if r is not None else None


def pooled_window(ft, t0, t1):
    """EXACT copy of build_song frame selection (fixed pooling)."""
    nfr = len(ft)
    i0 = int(np.searchsorted(ft, t0, side="left"))
    i1 = int(np.searchsorted(ft, t1, side="left"))
    floored = False
    if i1 - i0 < MIN_FRAMES:
        floored = True
        mid = 0.5 * (t0 + t1)
        c = int(np.searchsorted(ft, mid, side="left"))
        i0 = max(0, c - MIN_FRAMES // 2)
        i1 = min(nfr, i0 + MIN_FRAMES)
        i0 = max(0, i1 - MIN_FRAMES)
    i0 = max(i0, 0); i1 = min(i1, nfr)
    return i0, i1, floored


def frame_chroma(v88, lo, hi):
    c = _reg_raw(v88, lo, hi)
    n = np.linalg.norm(c)
    return c / n if n > 1e-9 else c


def make_plot(ex, ft, on, nt, path):
    t0, t1 = ex["t0"], ex["t1"]
    i0, i1 = ex["i0"], ex["i1"]
    # wider display context to prove nothing outside [i0,i1) is used
    d0 = int(np.searchsorted(ft, t0 - 0.20, side="left"))
    d1 = int(np.searchsorted(ft, t1 + 0.45, side="left"))
    d0 = max(0, d0); d1 = min(len(ft), d1)
    fr = np.arange(d0, d1)
    times = ft[fr]
    blocks = [
        ("onset (all reg)", on, 0, 200),
        ("note (all reg)", nt, 0, 200),
        ("bass (MIDI<52)", on, 0, 52),
        ("treble (MIDI>=60)", on, 60, 200),
    ]
    fig, axes = plt.subplots(4, 1, figsize=(7.6, 7.2), sharex=True)
    gt = ex["gt_root"]; nxt = ex["next_root"]
    for ax, (name, src, lo, hi) in zip(axes, blocks):
        M = np.stack([frame_chroma(src[j], lo, hi) for j in fr], axis=1)  # (12, nframes)
        ax.imshow(M, aspect="auto", origin="lower", cmap="magma",
                  extent=[times[0], times[-1], -0.5, 11.5], vmin=0, vmax=1)
        # exact pooled window box
        wx0 = ft[i0]; wx1 = ft[i1 - 1]
        ax.add_patch(Rectangle((wx0, -0.5), wx1 - wx0, 12,
                     fill=False, edgecolor="#39d353", lw=2.0))
        ax.axvline(t1, color="#ff4d4d", lw=1.6, ls="-")   # true span end
        if ex["next_t"] is not None:
            ax.axvline(ex["next_t"], color="#ffd166", lw=1.2, ls="--")
        if gt is not None:
            ax.axhline(gt, color="#39d353", lw=0.8, ls=":", alpha=0.8)
        if nxt is not None and nxt != gt:
            ax.axhline(nxt, color="#ffd166", lw=0.8, ls=":", alpha=0.8)
        ax.set_yticks(range(12)); ax.set_yticklabels(PC, fontsize=6)
        ax.set_ylabel(name, fontsize=7)
        ax.tick_params(labelsize=6)
    axes[-1].set_xlabel("time (s)  |  green box = EXACT frames the model pools  |  "
                        "red = span end t1  |  dashed yellow = next-chord onset", fontsize=6.5)
    axes[0].set_title(ex["title"], fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=96)
    plt.close(fig)


def trim_clip(wav, t0, t1, path):
    dur = t1 - t0
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav),
                    "-ss", f"{t0:.6f}", "-t", f"{dur:.6f}",
                    "-ac", "1", "-ar", "44100", str(path)], check=True)
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                        "format=duration", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


def main():
    examples = []
    songs = {}
    for sid, idx, note in SEL:
        songs.setdefault(sid, []).append((idx, note))

    with RemoteZip(ZIP_URL) as z:
        names = {Path(i.filename).stem: i.filename for i in z.infolist()
                 if i.filename.endswith(".wav")}
        for sid, items in songs.items():
            rows = fetch_chords(sid)
            print(f"[{sid}] extracting WAV...", flush=True)
            z.extract(names[sid], path=str(WAV_DIR))
            wav = WAV_DIR / names[sid]
            ex = PitchExtractor(cache_dir=BP_CACHE)
            acts = ex.extract(wav)
            ft, on, nt = acts.frame_times, acts.onset_probs, acts.note_probs
            fr_period = float(np.median(np.diff(ft)))
            for idx, note in items:
                t0, t1, lab = rows[idx]
                nxt = rows[idx + 1] if idx + 1 < len(rows) else None
                i0, i1, floored = pooled_window(ft, t0, t1)
                w0, w1 = float(ft[i0]), float(ft[i1 - 1])
                pre_bleed = max(0.0, t0 - w0)
                post_bleed = max(0.0, w1 - t1)
                cid = f"{sid}_{idx}"
                clip = CLIPS / f"{cid}.wav"
                meas = trim_clip(wav, t0, t1, clip)
                png = CHROMA / f"{cid}.png"
                exd = {
                    "id": cid, "song": sid, "idx": idx, "note": note,
                    "label": lab, "title": f"{sid} #{idx}  {lab}   span={t1-t0:.3f}s",
                    "t0": float(t0), "t1": float(t1), "span": float(t1 - t0),
                    "i0": i0, "i1": i1, "n_frames": int(i1 - i0), "floored": floored,
                    "w0": w0, "w1": w1, "pre_bleed_ms": pre_bleed * 1e3,
                    "post_bleed_ms": post_bleed * 1e3,
                    "gt_root": root_pc(lab),
                    "next_label": nxt[2] if nxt else None,
                    "next_root": root_pc(nxt[2]) if nxt else None,
                    "next_t": float(nxt[0]) if nxt else None,
                    "clip_dur": meas, "clip_err_ms": (meas - (t1 - t0)) * 1e3,
                    "fr_period_ms": fr_period * 1e3,
                }
                make_plot(exd, ft, on, nt, png)
                examples.append(exd)
                print(f"  {cid} {lab:14s} span={t1-t0:.3f} nfr={i1-i0} "
                      f"pre={pre_bleed*1e3:.1f} post={post_bleed*1e3:.1f} "
                      f"clip={meas:.4f} err={ (meas-(t1-t0))*1e3:+.3f}ms floor={floored}",
                      flush=True)
            wav.unlink(missing_ok=True)
            for f in BP_CACHE.glob("*"):
                f.unlink(missing_ok=True)

    (OUT / "examples.json").write_text(json.dumps(examples, indent=2))
    print(f"\nWrote {len(examples)} examples -> {OUT/'examples.json'}")
    # cleanup wav dir
    shutil.rmtree(WAV_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
