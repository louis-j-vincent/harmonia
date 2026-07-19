"""Verify the +6.87s GT offset saved for track 341 (Commodores - Easy) is
actually correct, whole-song, not just the auto-guess heuristic's own
first-onset check. Also re-scores root accuracy with vs without the
correction using the shipped billboard_bp48_60_rollaug_v1.pt checkpoint.

Audio already exists locally at docs/audio/the_commodores_easy_1977.m4a --
no download needed, no disk risk.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa
import torch

REPO = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scratchpad"))

from harmonia.models.chord_pipeline_v1 import extract_beat_features
from harmonia.data.yt_chord_corpus import seg_feature_abs
from rebuild_billboard_fixed import parse_harte  # noqa: E402
import mirdata

AUDIO = REPO / "docs/audio/the_commodores_easy_1977.m4a"
BP_CACHE = REPO / "data/cache/billboard_60/bp_cache"
PLOTS = REPO / "docs/plots"
TID = "341"
OFFSET = 6.87

ds = mirdata.initialize("billboard")
cf = ds.track(TID).chords_full
labels = [str(l) for l in cf.labels]
intervals = [(float(a), float(b)) for a, b in cf.intervals]
real_changes = [(intervals[i][0], labels[i]) for i in range(len(labels))
                 if labels[i] not in ("N", "X") and (i == 0 or labels[i] != labels[i - 1])]

print(f"GT: {len(intervals)} intervals, {len(real_changes)} real chord changes, "
      f"gt_end={intervals[-1][1]:.1f}s")

# ---------------- whole-song alignment check ----------------
y, sr = librosa.load(str(AUDIO), sr=22050, mono=True)
audio_dur = len(y) / sr
hop = 512
oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
onsets = librosa.onset.onset_detect(onset_envelope=oenv, sr=sr, hop_length=hop,
                                     units="time", backtrack=True)
print(f"audio_dur={audio_dur:.1f}s  n_onsets={len(onsets)}")

def residuals(offset, tol=0.35):
    out = []
    for t, lab in real_changes:
        tc = t + offset
        if tc > audio_dur:
            continue
        j = np.argmin(np.abs(onsets - tc))
        d = onsets[j] - tc
        out.append((t, d if abs(d) < tol else np.nan))
    return out

res0 = residuals(0.0)
res_corr = residuals(OFFSET)

def summarize(res, label):
    ts = np.array([r[0] for r in res])
    ds_ = np.array([r[1] for r in res])
    matched = ~np.isnan(ds_)
    frac = matched.mean()
    if matched.sum() >= 2:
        # linear drift fit vs time (only matched points)
        p = np.polyfit(ts[matched], ds_[matched], 1)
        drift_per_min = p[0] * 60
    else:
        drift_per_min = float("nan")
    med = np.nanmedian(ds_)
    print(f"[{label}] matched(<0.35s)={frac:.0%}  median_resid={med*1000:+.0f}ms  "
          f"drift={drift_per_min*1000:+.0f}ms/min  n={len(res)}")
    return dict(frac=float(frac), median_ms=float(med * 1000) if not np.isnan(med) else None,
                drift_ms_per_min=float(drift_per_min) if not np.isnan(drift_per_min) else None)

print("\n--- Whole-song residual-to-nearest-onset, RAW vs +6.87s corrected ---")
s0 = summarize(res0, "RAW (no offset)")
s1 = summarize(res_corr, "CORRECTED (+6.87s)")

# thirds-of-song breakdown for the corrected version (does it drift internally?)
ts = np.array([r[0] for r in res_corr])
ds_ = np.array([r[1] for r in res_corr])
matched = ~np.isnan(ds_)
tmax = ts.max()
print("\n--- Corrected-offset residual by song-third (checks for internal edit / drift) ---")
for lo, hi, name in [(0, tmax/3, "first third"), (tmax/3, 2*tmax/3, "middle third"), (2*tmax/3, tmax, "last third")]:
    sel = matched & (ts >= lo) & (ts < hi)
    if sel.sum():
        print(f"  {name:12s} n={sel.sum():3d} matched={sel.sum()}/{((ts>=lo)&(ts<hi)).sum()} "
              f"median={np.median(ds_[sel])*1000:+.0f}ms  range=[{ds_[sel].min()*1000:+.0f},{ds_[sel].max()*1000:+.0f}]ms")
    else:
        print(f"  {name:12s} no matches")

# ---------------- plot ----------------
fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=False)
ds_show = max(1, len(y)//6000)
tt = np.arange(len(y)) / sr

# Panel 1: first 25s zoom, raw vs corrected GT overlay
ax = axes[0]
seg = int(25 * sr)
ax.plot(tt[:seg:ds_show], y[:seg:ds_show] / (np.max(np.abs(y[:seg]))+1e-9) * 0.9, color="#b9b09a", lw=0.4)
for t, l in real_changes:
    if t < 25: ax.axvline(t, color="#1f8a5b", lw=1.2, alpha=0.8)
for t, l in real_changes:
    tc = t + OFFSET
    if tc < 25: ax.axvline(tc, color="#8a2b2b", ls="--", lw=1.1, alpha=0.8)
for o in onsets[onsets < 25]:
    ax.axvline(o, color="#2b5f8a", ls=":", lw=0.6, alpha=0.5, ymin=0, ymax=0.15)
ax.set_title(f"Commodores - Easy (tid 341): first 25s -- GT raw (green) vs GT+{OFFSET}s (red dashed) vs onsets (blue)")
ax.set_yticks([])

# Panel 2: last 30s of audio (song end) zoom
ax = axes[1]
lo_t = max(0, audio_dur - 30)
seg0, seg1 = int(lo_t*sr), len(y)
ax.plot(tt[seg0:seg1:ds_show], y[seg0:seg1:ds_show] / (np.max(np.abs(y[seg0:seg1]))+1e-9) * 0.9, color="#b9b09a", lw=0.4)
for t, l in real_changes:
    if lo_t <= t <= audio_dur: ax.axvline(t, color="#1f8a5b", lw=1.2, alpha=0.8)
for t, l in real_changes:
    tc = t + OFFSET
    if lo_t <= tc <= audio_dur: ax.axvline(tc, color="#8a2b2b", ls="--", lw=1.1, alpha=0.8)
for o in onsets[(onsets >= lo_t) & (onsets <= audio_dur)]:
    ax.axvline(o, color="#2b5f8a", ls=":", lw=0.6, alpha=0.5, ymin=0, ymax=0.15)
ax.axvline(audio_dur, color="black", lw=1.5, alpha=0.6)
ax.set_title(f"Last 30s of audio (dur={audio_dur:.1f}s) -- black line = audio end. GT_end(raw)={intervals[-1][1]:.1f}s GT_end+offset={intervals[-1][1]+OFFSET:.1f}s")
ax.set_yticks([])

# Panel 3: residual vs song time, corrected offset
ax = axes[2]
ax.scatter(ts[matched], ds_[matched]*1000, s=10, color="#8a2b2b", alpha=0.7)
ax.axhline(0, color="gray", lw=0.8)
if matched.sum() >= 2:
    p = np.polyfit(ts[matched], ds_[matched], 1)
    xx = np.linspace(0, tmax, 50)
    ax.plot(xx, np.polyval(p, xx)*1000, color="black", lw=1.5, ls="--",
            label=f"fit: {p[0]*60*1000:+.0f} ms/min")
    ax.legend()
ax.set_xlabel("GT time (s, uncorrected)")
ax.set_ylabel("residual to nearest onset (ms)")
ax.set_title("Residual (GT+offset vs nearest onset) across whole song")

plt.tight_layout()
outp = PLOTS / "bridge_commodores_offset_verification.png"
plt.savefig(outp, dpi=110)
print(f"\nSaved plot -> {outp}")

# ---------------- root accuracy WITH vs WITHOUT correction ----------------
print("\n--- Root accuracy: model billboard_bp48_60_rollaug_v1.pt ---")
bf = extract_beat_features(AUDIO, cache_dir=BP_CACHE)
beat_times = bf.beat_times
onset_b, note_b = bf.onset_b, bf.note_b

ck = torch.load(REPO / "data/models/billboard_bp48_60_rollaug_v1.pt", map_location="cpu", weights_only=False)
rm = ck["root_model"]; rm.eval(); r_mean = ck["root_mean"]; r_std = ck["root_std"]

def predict_root(feat_abs):
    Xn = ((feat_abs - r_mean) / r_std).astype(np.float32)
    with torch.no_grad():
        return rm(torch.tensor(Xn)).argmax(1).numpy()

def build_records(offset_s):
    recs = []
    for (t0, t1), label in zip(intervals, labels):
        t0c, t1c = t0 + offset_s, t1 + offset_s
        pc, fam = parse_harte(label)
        if pc is None:
            continue
        b0 = int(np.searchsorted(beat_times, t0c, side="right")) - 1
        b1 = int(np.searchsorted(beat_times, t1c, side="right"))
        b0 = max(b0, 0); b1 = min(b1, len(onset_b))
        if b1 - b0 < 1:
            continue
        feat_abs = seg_feature_abs(onset_b, note_b, b0, b1)
        recs.append((feat_abs, pc % 12, label, t0, t1))
    return recs

for offset_s, name in [(0.0, "UNCORRECTED (raw GT timestamps)"), (OFFSET, f"CORRECTED (+{OFFSET}s)")]:
    recs = build_records(offset_s)
    if not recs:
        print(f"[{name}] no records!")
        continue
    feats = np.stack([r[0] for r in recs])
    gt = np.array([r[1] for r in recs])
    pred = predict_root(feats)
    acc = (pred == gt).mean()
    print(f"[{name}] n={len(recs)}  root_acc={acc:.3f}")
