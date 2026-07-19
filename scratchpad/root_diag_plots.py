"""Part 1 diagnostics: chroma-on-waveform + inferred-vs-GT root for Billboard BP48.

Re-downloads audio for a handful of chosen songs, re-extracts per-beat BP
activations, and produces two PNGs per song:
  A) waveform + per-beat bass/note chroma heatmaps + GT chord boundaries
  B) waveform + GT-root vs model-predicted-root step plot, color-coded by
     root-position vs inversion, mismatches highlighted.

Model predictions come from the READ-ONLY shipped checkpoint
data/models/billboard_bp48_60_rollaug_v1.pt (not retrained here).
WAVs deleted immediately after extraction (disk discipline).
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import soundfile as sf
import librosa
import torch

REPO = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia")
sys.path.insert(0, str(REPO))
from harmonia.models.chord_pipeline_v1 import extract_beat_features, _reg_raw
from harmonia.data.yt_chord_corpus import seg_feature_abs, download_audio

PC_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
CACHE = REPO / "data/cache/billboard_60"
AUDIO = CACHE / "audio"; BP = CACHE / "bp_cache"
AUDIO.mkdir(parents=True, exist_ok=True); BP.mkdir(parents=True, exist_ok=True)
PLOTS = REPO / "docs/plots"

SONGS = {  # tid -> (video_id, artist-title, note)
    "362":  ("JRiAMe1zsQ0", "Wednesday - Last Kiss",            "root acc 0.10, 0% inv - pure muddy/fifth-confusion"),
    "1111": ("3joI5VtuNV0", "Chris Kenner - Land of 1000 Dances","root acc 0.99, 0% inv - clean control"),
    "1027": ("qNHWpi7CKgU", "Greg Kihn - Lucky",                "root acc 0.32, 49% inv - inv-heavy + low acc"),
    "887":  ("NerII_Zgd5I", "De La Soul - Me Myself and I",     "root acc 0.70, 44% inv - inversion story"),
}

# ---- load corpus + checkpoint (read-only) ----
d = np.load(REPO/"data/cache/billboard_bp48_60_fixed_beatgrid.npz", allow_pickle=True)
C = {k: d[k] for k in d.keys()}
ck = torch.load(REPO/"data/models/billboard_bp48_60_rollaug_v1.pt", map_location="cpu", weights_only=False)
rm = ck["root_model"]; rm.eval(); r_mean = ck["root_mean"]; r_std = ck["root_std"]

def predict_root(feat_abs):
    Xn = ((feat_abs - r_mean)/r_std).astype(np.float32)
    with torch.no_grad():
        return rm(torch.tensor(Xn)).argmax(1).numpy()

def beat_chroma_strip(vec_b, lo, hi):
    """(n_beats,88) -> (12, n_beats) L2-normed-per-beat register chroma."""
    out = np.zeros((12, len(vec_b)))
    for b in range(len(vec_b)):
        c = _reg_raw(vec_b[b], lo, hi)
        n = np.linalg.norm(c)
        out[:, b] = c/n if n > 1e-9 else c
    return out

def process(tid):
    vid, title, desc = SONGS[tid]
    sid = f"bb_{tid}"
    print(f"\n=== {sid} {title} ===", flush=True)
    wav = download_audio(vid, AUDIO)
    try:
        bf = extract_beat_features(wav, cache_dir=BP)
        y, sr = sf.read(wav)
        y = (y.mean(1) if y.ndim > 1 else y).astype("float32")
    finally:
        wav.unlink(missing_ok=True)  # disk discipline

    bt = bf.beat_times                    # (n_beats+1,)
    bc = 0.5*(bt[:-1]+bt[1:])             # beat centers
    bass = beat_chroma_strip(bf.onset_b, 0, 52)
    note = beat_chroma_strip(bf.note_b, 0, 200)

    # song records (already have feat48_abs, GT root, labels, t0/t1)
    m = C["song_id"] == sid
    t0 = C["t0"][m]; t1 = C["t1"][m]; gt = C["root"][m].astype(int)
    labels = C["labels"][m]
    pred = predict_root(C["feat48_abs"][m])
    order = np.argsort(t0); t0,t1,gt,labels,pred = t0[order],t1[order],gt[order],labels[order],pred[order]
    inv = np.array(["/" in str(l) for l in labels])
    # sounding bass pc of the slash (for inversion story)
    def bass_pc(l):
        l=str(l)
        if "/" not in l: return None
        b=l.split("/")[1]
        M={"C":0,"C#":1,"Db":1,"D":2,"D#":3,"Eb":3,"E":4,"F":5,"F#":6,"Gb":6,"G":7,"G#":8,"Ab":8,"A":9,"A#":10,"Bb":10,"B":11}
        return M.get(b, None)
    slash_bass = [bass_pc(l) for l in labels]

    tmax = min(bt[-1], float(t1.max()))
    acc = (pred==gt).mean()
    print(f"  n_records={m.sum()} beat n={len(bc)} root acc={acc:.2f} inv={inv.mean():.2f}", flush=True)

    # ---------- FIGURE A: chroma-on-waveform ----------
    fig, ax = plt.subplots(3, 1, figsize=(15, 8), sharex=True,
                           gridspec_kw={"height_ratios":[1,2,2]})
    # waveform envelope
    tt = np.arange(len(y))/sr
    ds = max(1, len(y)//4000)
    ax[0].plot(tt[::ds], y[::ds], lw=0.4, color="#3b6ea5")
    ax[0].set_ylabel("wave"); ax[0].set_yticks([])
    ax[0].set_title(f"{sid}  {title}  |  {desc}  |  model root acc {acc:.0%} (TRAIN song)", fontsize=11)
    # bass chroma heatmap + GT root overlay
    ext = [bc[0], bc[-1], -0.5, 11.5]
    ax[1].imshow(bass, aspect="auto", origin="lower", extent=ext, cmap="magma", vmin=0, vmax=0.9)
    ax[1].set_ylabel("bass chroma (0-52 MIDI)\npitch class")
    ax[1].set_yticks(range(12)); ax[1].set_yticklabels(PC_NAMES, fontsize=6)
    # note chroma heatmap
    ax[2].imshow(note, aspect="auto", origin="lower", extent=ext, cmap="magma", vmin=0, vmax=0.9)
    ax[2].set_ylabel("note chroma (full)\npitch class")
    ax[2].set_yticks(range(12)); ax[2].set_yticklabels(PC_NAMES, fontsize=6)
    ax[2].set_xlabel("time (s)")
    # GT chord roots as horizontal segments on both heatmaps; boundaries as lines
    for a in (ax[1], ax[2]):
        for s,e,g,iv in zip(t0,t1,gt,inv):
            a.hlines(g, s, e, color="#00e0c0" if not iv else "#ff5588", lw=2.0, alpha=0.9)
            a.axvline(s, color="white", lw=0.3, alpha=0.25)
    ax[0].set_xlim(0, tmax)
    from matplotlib.lines import Line2D
    ax[1].legend([Line2D([0],[0],color="#00e0c0",lw=2), Line2D([0],[0],color="#ff5588",lw=2)],
                 ["GT root (root-pos)","GT root (inversion)"], loc="upper right", fontsize=7, framealpha=0.6)
    plt.tight_layout()
    fa = PLOTS/f"root_diag_A_chroma_{sid}.png"
    fig.savefig(fa, dpi=110); plt.close(fig)

    # ---------- FIGURE B: predicted vs GT root ----------
    fig, ax = plt.subplots(2, 1, figsize=(15, 6), sharex=True,
                           gridspec_kw={"height_ratios":[1,3]})
    ax[0].plot(tt[::ds], y[::ds], lw=0.4, color="#3b6ea5")
    ax[0].set_yticks([]); ax[0].set_ylabel("wave")
    ax[0].set_title(f"{sid}  {title}  |  predicted root vs GT root  (acc {acc:.0%})", fontsize=11)
    b = ax[1]
    for s,e,g,p,iv,sb in zip(t0,t1,gt,pred,inv,slash_bass):
        ok = (g==p)
        # GT root: thick, colored by inversion
        b.hlines(g, s, e, color="#00b3a0" if not iv else "#d94f8a", lw=4, alpha=0.85)
        # predicted root: thin black, dashed if mismatch
        b.hlines(p, s, e, color="black", lw=1.6, ls="-" if ok else "--", alpha=0.9)
        if not ok:
            interval = (p-g) % 12
            col = "#e8a33d" if interval in (5,7) else "#888"  # highlight P4/P5
            b.vlines((s+e)/2, min(g,p), max(g,p), color=col, lw=1.2, alpha=0.7)
        # mark where prediction equals the sounding slash-bass (inversion story)
        if iv and sb is not None and p==sb and not ok:
            b.plot((s+e)/2, p, "v", color="#d94f8a", ms=6, alpha=0.9)
    b.set_yticks(range(12)); b.set_yticklabels(PC_NAMES, fontsize=7)
    b.set_ylabel("pitch class"); b.set_xlabel("time (s)"); b.set_ylim(-0.7, 11.7)
    b.grid(True, axis="y", alpha=0.15)
    from matplotlib.lines import Line2D
    b.legend([Line2D([0],[0],color="#00b3a0",lw=4), Line2D([0],[0],color="#d94f8a",lw=4),
              Line2D([0],[0],color="black",lw=1.6), Line2D([0],[0],color="black",lw=1.6,ls="--"),
              Line2D([0],[0],color="#e8a33d",lw=1.2),
              Line2D([0],[0],marker="v",color="#d94f8a",lw=0)],
             ["GT root (root-pos)","GT root (inversion)","pred (correct)","pred (wrong)",
              "P4/P5 error span","pred=sounding bass"], loc="upper right", fontsize=7, ncol=2, framealpha=0.6)
    ax[0].set_xlim(0, tmax)
    plt.tight_layout()
    fb = PLOTS/f"root_diag_B_rootvsgt_{sid}.png"
    fig.savefig(fb, dpi=110); plt.close(fig)

    # error stats
    err = pred!=gt
    p4p5 = np.array([( (p-g)%12 in (5,7)) for g,p in zip(gt,pred)])
    inv_bass_hit = np.array([iv and sb is not None and p==sb for iv,sb,p in zip(inv,slash_bass,pred)])
    stats = dict(sid=sid, title=title, n=int(m.sum()), acc=float(acc), inv_frac=float(inv.mean()),
                 err_frac=float(err.mean()),
                 p4p5_share_of_err=float(p4p5[err].mean()) if err.any() else 0.0,
                 inv_err_acc=float((~err[inv]).mean()) if inv.any() else None,
                 rootpos_acc=float((~err[~inv]).mean()) if (~inv).any() else None,
                 pred_eq_bass_share_of_inv_err=float(inv_bass_hit[inv&err].mean()) if (inv&err).any() else 0.0)
    print("  stats:", json.dumps(stats), flush=True)
    return stats

if __name__ == "__main__":
    allstats = []
    for tid in ["362","1111","1027","887"]:
        try:
            allstats.append(process(tid))
        except Exception as e:
            import traceback; traceback.print_exc()
            print("  FAILED", tid, e)
    (Path(__file__).parent/"root_diag_stats.json").write_text(json.dumps(allstats, indent=2))
    print("\nDONE. stats written.")
