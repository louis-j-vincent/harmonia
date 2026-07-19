"""Chord-CHANGE-POINT diagnostic: deployed pipeline's own segmentation vs Billboard GT.

Runs the REAL deployed inference path (infer_chords_billboard_v1 — same function
scripts/harmonia_server.py._run_analysis calls) on real Billboard audio, extracts
the inferred chord-change timestamps (where the merged-beat chord label changes),
and compares them to Billboard chords_full GT change timestamps. This measures
boundary-placement quality at actual inference time — separate from whether the
label at each segment is correct (that's the #31 labeling question).

READ-ONLY on harmonia/models/chord_pipeline_v1.py and scripts/harmonia_server.py —
imports and calls infer_chords_billboard_v1 directly, does not touch the server.

Disk discipline: one song at a time, WAV deleted immediately after feature
extraction (extract_beat_features/infer_chords_billboard_v1 read the file then
we delete it).
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import soundfile as sf
import mirdata

REPO = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia")
sys.path.insert(0, str(REPO))
from harmonia.models.chord_pipeline_v1 import infer_chords_billboard_v1
from harmonia.data.yt_chord_corpus import download_audio

PLOTS = REPO / "docs/plots"
CACHE = REPO / "data/cache/billboard_60"
AUDIO = CACHE / "audio"; BP = CACHE / "bp_cache"
AUDIO.mkdir(parents=True, exist_ok=True); BP.mkdir(parents=True, exist_ok=True)

# Reuse exact video-id sourcing from scratchpad/root_diag_plots.py (same songs,
# for continuity with the root-diagnostics plots) — spans easy -> hard.
SONGS = {
    "1111": ("3joI5VtuNV0", "Chris Kenner - Land of 1000 Dances", "root acc 0.99 (clean/easy)"),
    "887":  ("NerII_Zgd5I", "De La Soul - Me Myself and I",       "root acc 0.70 (inversion-heavy)"),
    "1027": ("qNHWpi7CKgU", "Greg Kihn - Lucky",                  "root acc 0.32 (inv-heavy, low acc)"),
    "362":  ("JRiAMe1zsQ0", "Wednesday - Last Kiss",               "root acc 0.10 (hard/muddy)"),
}

bb = mirdata.initialize("billboard")


def gt_changes(tid):
    t = bb.track(tid)
    cd = t.chords_full
    ivs, labs = np.asarray(cd.intervals), np.asarray(cd.labels)
    # keep 'N'/silence too — Billboard uses N for no-chord; still a real GT span
    order = np.argsort(ivs[:, 0])
    ivs, labs = ivs[order], labs[order]
    # change points = every interval start except possibly-duplicate merges of
    # identical adjacent labels (Billboard chords_full doesn't usually repeat,
    # but merge defensively)
    starts, ends, lbls = [], [], []
    for (s, e), l in zip(ivs, labs):
        if starts and lbls[-1] == l and abs(s - ends[-1]) < 1e-6:
            ends[-1] = e
            continue
        starts.append(s); ends.append(e); lbls.append(l)
    starts, ends, lbls = np.array(starts), np.array(ends), np.array(lbls)
    changes = starts[1:]  # interior boundaries only (exclude t=0 song start)
    return starts, ends, lbls, changes


def match_boundaries(gt_b, pred_b, tol):
    """One-to-one greedy matching within tolerance. Returns (tp_gt_idx, tp_pred_idx, fn_idx, fp_idx)."""
    used_pred = set()
    tp_gt, tp_pred = [], []
    for i, g in enumerate(gt_b):
        # nearest unused pred boundary within tol
        cands = [(abs(g - p), j) for j, p in enumerate(pred_b) if j not in used_pred and abs(g - p) <= tol]
        if cands:
            cands.sort()
            j = cands[0][1]
            used_pred.add(j)
            tp_gt.append(i); tp_pred.append(j)
    fn = [i for i in range(len(gt_b)) if i not in tp_gt]
    fp = [j for j in range(len(pred_b)) if j not in used_pred]
    return tp_gt, tp_pred, fn, fp


def classify_fn(gt_starts, gt_ends, gt_labels, missed_gt_idx, pred_starts, pred_ends):
    """For each missed GT boundary (GT changes chord but no nearby inferred boundary),
    check: is the inferred segmentation MERGING across it (one inferred segment spans
    both GT chords either side)? That's the failure mode most disruptive to the Wheel
    editor (can't select the sub-span to correct)."""
    merged = 0
    for i in missed_gt_idx:
        t_change = gt_starts[i]  # GT boundary time (== gt_starts[i], boundary before this chord)
        # find inferred segment covering t_change - eps and t_change + eps
        for ps, pe in zip(pred_starts, pred_ends):
            if ps <= t_change - 0.05 and pe >= t_change + 0.05:
                merged += 1
                break
    return merged


def process(tid):
    vid, title, desc = SONGS[tid]
    sid = f"bb_{tid}"
    print(f"\n=== {sid} {title} ===", flush=True)

    gt_starts, gt_ends, gt_labels, gt_change = gt_changes(tid)

    wav = download_audio(vid, AUDIO)
    try:
        chart = infer_chords_billboard_v1(wav, cache_dir=BP)
        y, sr = sf.read(wav)
        y = (y.mean(1) if y.ndim > 1 else y).astype("float32")
    finally:
        wav.unlink(missing_ok=True)

    pred_chords = chart.chords
    pred_starts = np.array([c["start_s"] for c in pred_chords])
    pred_ends = np.array([c["end_s"] for c in pred_chords])
    pred_labels = np.array([c["label"] for c in pred_chords])
    pred_change = pred_starts[1:]  # interior inferred boundaries

    tmax = min(float(gt_ends[-1]), float(pred_ends[-1]), len(y) / sr)
    beat_dur = 60.0 / max(chart.tempo_bpm, 1.0)

    results = {}
    for tol_name, tol in [("0.5s", 0.5), ("1beat", beat_dur)]:
        tp_gt, tp_pred, fn, fp = match_boundaries(gt_change, pred_change, tol)
        n_gt, n_pred = len(gt_change), len(pred_change)
        prec = len(tp_pred) / n_pred if n_pred else 0.0
        rec = len(tp_gt) / n_gt if n_gt else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        merged = classify_fn(gt_starts, gt_ends, gt_labels, fn, pred_starts, pred_ends)
        results[tol_name] = dict(
            tol_s=round(tol, 3), n_gt=n_gt, n_pred=n_pred,
            tp=len(tp_gt), fn=len(fn), fp=len(fp),
            precision=round(prec, 3), recall=round(rec, 3), f1=round(f1, 3),
            fn_that_are_merges=merged,
            fn_merge_frac=round(merged / len(fn), 3) if fn else None,
        )
        print(f"  tol={tol_name} ({tol:.3f}s): n_gt={n_gt} n_pred={n_pred} "
              f"P={prec:.2f} R={rec:.2f} F1={f1:.2f}  FN-that-are-merges={merged}/{len(fn)}", flush=True)

    # ---------- plot ----------
    fig, ax = plt.subplots(3, 1, figsize=(16, 7), sharex=True,
                            gridspec_kw={"height_ratios": [1, 1.3, 1.3]})
    tt = np.arange(len(y)) / sr
    ds = max(1, len(y) // 6000)
    ax[0].plot(tt[::ds], y[::ds], lw=0.4, color="#3b6ea5")
    ax[0].set_yticks([]); ax[0].set_ylabel("wave")
    ax[0].set_title(f"{sid}  {title}  |  {desc}  |  boundary F1(0.5s)={results['0.5s']['f1']:.2f}  "
                     f"F1(1beat)={results['1beat']['f1']:.2f}", fontsize=11)

    # GT chord grid row
    axg = ax[1]
    for i, (s, e, l) in enumerate(zip(gt_starts, gt_ends, gt_labels)):
        axg.axvspan(s, e, color="#00b3a0" if i % 2 == 0 else "#00857a", alpha=0.35)
        axg.axvline(s, color="black", lw=0.5, alpha=0.6)
        if e - s > 0.6:
            axg.text((s + e) / 2, 0.5, l, ha="center", va="center", fontsize=6, rotation=0)
    axg.set_ylim(0, 1); axg.set_yticks([]); axg.set_ylabel("GT (Billboard)")

    # inferred chord grid row + change ticks
    axp = ax[2]
    for i, (s, e, l) in enumerate(zip(pred_starts, pred_ends, pred_labels)):
        axp.axvspan(s, e, color="#e8a33d" if i % 2 == 0 else "#c77f1f", alpha=0.35)
        if e - s > 0.6:
            axp.text((s + e) / 2, 0.5, l, ha="center", va="center", fontsize=6, rotation=0)
    axp.set_ylim(0, 1); axp.set_yticks([]); axp.set_ylabel("inferred\n(deployed pipeline)")
    axp.set_xlabel("time (s)")

    # overlay matched/missed/spurious boundary markers on both rows using 0.5s tol result
    r = results["0.5s"]
    tp_gt, tp_pred, fn, fp = match_boundaries(gt_change, pred_change, 0.5)
    for i in tp_gt:
        axg.axvline(gt_change[i], color="#2ecc71", lw=1.4, ymin=0, ymax=1)
    for i in fn:
        axg.axvline(gt_change[i], color="red", lw=1.6, ls="--", ymin=0, ymax=1)
    for j in tp_pred:
        axp.axvline(pred_change[j], color="#2ecc71", lw=1.4, ymin=0, ymax=1)
    for j in fp:
        axp.axvline(pred_change[j], color="magenta", lw=1.6, ls=":", ymin=0, ymax=1)

    from matplotlib.lines import Line2D
    ax[0].legend([Line2D([0], [0], color="#2ecc71", lw=1.4),
                  Line2D([0], [0], color="red", lw=1.6, ls="--"),
                  Line2D([0], [0], color="magenta", lw=1.6, ls=":")],
                 ["matched boundary (<=0.5s)", "GT boundary MISSED (FN)", "spurious inferred boundary (FP)"],
                 loc="upper right", fontsize=7, framealpha=0.6)
    ax[0].set_xlim(0, tmax)
    plt.tight_layout()
    fp_path = PLOTS / f"chord_boundary_diag_{sid}.png"
    fig.savefig(fp_path, dpi=110)
    plt.close(fig)
    print(f"  saved {fp_path}", flush=True)

    return dict(sid=sid, title=title, desc=desc, tempo_bpm=chart.tempo_bpm,
                beat_dur_s=round(beat_dur, 3), n_gt_chords=len(gt_starts),
                n_pred_chords=len(pred_chords), results=results)


if __name__ == "__main__":
    order = ["1111", "887", "1027", "362"]
    allres = []
    for tid in order:
        try:
            allres.append(process(tid))
        except Exception as e:
            import traceback; traceback.print_exc()
            print("FAILED", tid, e)
    outp = Path(__file__).parent / "boundary_diag_results.json"
    outp.write_text(json.dumps(allres, indent=2))
    print("\nDONE ->", outp)
