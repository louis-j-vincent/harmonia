"""GT triage: per-REGION confidence from evidence independent of our own decode.

Signals (none of them our pipeline's chord decode):
  1. chroma chord-tone fit      - raw librosa CQT chroma (LTAS) vs GT chord template
  2. best-alternative margin    - best chord in a 12x12 vocab minus the GT chord
  3. music-x-lab posteriors     - third-party ISMIR2019 net (root/triad + bass)
  4. cross-repeat consistency   - same GT label sequence elsewhere in the song
  5. time-shift search          - does shifting the GT window recover the fit (drift)
  6. energy                     - region RMS vs song median

Writes regions.json to the scratchpad. READ-ONLY on the repo.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia")
SCRATCH = Path("/private/tmp/claude-501/-Users-vincente-Documents-Projets-Perso-Code-harmonia/"
               "22c6b747-cea4-410f-85ae-f6ed58a9d2eb/scratchpad")
sys.path.insert(0, str(REPO))

from harmonia.core.chroma import chroma_cqt_ltas          # noqa: E402
from harmonia.models.musx_redecode import frame_posteriors, FRAME_DT  # noqa: E402

# quality -> semitone intervals (same table as scripts/brick0_propose.py)
QUALITY_INTERVALS: dict[str, list[int]] = {
    "maj": [0, 4, 7], "min": [0, 3, 7], "7": [0, 4, 7, 10],
    "maj7": [0, 4, 7, 11], "min7": [0, 3, 7, 10], "dim": [0, 3, 6],
    "dim7": [0, 3, 6, 9], "hdim7": [0, 3, 6, 10], "aug": [0, 4, 8],
    "sus2": [0, 2, 7], "sus4": [0, 5, 7], "7sus4": [0, 5, 7, 10],
    "6": [0, 4, 7, 9], "maj6": [0, 4, 7, 9], "min6": [0, 3, 7, 9],
    "9": [0, 4, 7, 10, 2], "maj9": [0, 4, 7, 11, 2], "min9": [0, 3, 7, 10, 2],
    "11": [0, 4, 7, 10, 2, 5], "13": [0, 4, 7, 10, 2, 9], "minmaj7": [0, 3, 7, 11],
}
PC = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

VOCAB_Q = ("maj", "min", "7", "maj7", "min7", "dim", "dim7", "hdim7",
           "6", "min6", "aug", "sus4", "minmaj7", "9", "min9")

# maj/min family bucket, the MIREX majmin convention used by the scorer
_MIN = {"min", "min7", "min6", "min9", "minmaj7", "dim", "dim7", "hdim7"}


def famclass(q: str) -> str:
    if q in _MIN:
        return "min"
    if q in ("sus4", "sus2", "7sus4"):
        return "sus"
    return "maj"


def triad_of(q: str) -> str:
    """The bare 3-note core: what survives if you strip every extension."""
    if q in ("min", "min7", "min6", "min9", "minmaj7"):
        return "min"
    if q in ("dim", "dim7", "hdim7"):
        return "dim"
    if q in ("aug",):
        return "aug"
    if q in ("sus4", "7sus4"):
        return "sus4"
    if q in ("sus2",):
        return "sus2"
    return "maj"


def centre_norm(mat: np.ndarray) -> np.ndarray:
    c = mat - mat.mean(axis=-1, keepdims=True)
    n = np.linalg.norm(c, axis=-1, keepdims=True)
    n = np.where(n == 0, 1.0, n)
    return c / n


def template(root: int, q: str) -> np.ndarray:
    v = np.zeros(12)
    for iv in QUALITY_INTERVALS.get(q, [0, 4, 7]):
        v[(root + iv) % 12] = 1.0
    return v


# the written triad plus exactly one colour tone (7th / 6th / 9th) - the set a
# reading has to fall inside for the disagreement to be "extension-level only"
_EXT_OF = {
    "maj": ("maj", "maj7", "7", "6", "9", "maj9"),
    "min": ("min", "min7", "min6", "min9", "minmaj7"),
    "dim": ("dim", "dim7", "hdim7"),
    "aug": ("aug",),
    "sus4": ("sus4", "7sus4"),
    "sus2": ("sus2",),
}

VOCAB = [(r, q) for r in range(12) for q in VOCAB_Q]
VOCAB_CN = centre_norm(np.array([template(r, q) for r, q in VOCAB]))

# music-x-lab triad-index decoding: col 0 = N, col i>=1 -> root (i-1)%12,
# triad type (i-1)//12 in {maj,min,sus4,sus2,dim,aug}
MUSX_TRIADS = ["maj", "min", "sus4", "sus2", "dim", "aug"]

SONGS = ["bein_green", "blue_bossa", "blue_bossa_backing", "close_to_you",
         "every_breath_you_take", "georgia_on_my_mind", "stand_by_me"]

TAU_GRID = np.round(np.arange(-1.60, 1.601, 0.05), 3)


def mean_chroma(ch: np.ndarray, times: np.ndarray, t0: float, t1: float):
    lo = int(np.searchsorted(times, t0))
    hi = int(np.searchsorted(times, t1))
    if hi <= lo:
        mid = min(max(lo, 0), len(times) - 1)
        return ch[:, mid]
    return ch[:, lo:hi].mean(axis=1)


def analyse(song_id: str) -> dict:
    gt = json.loads((REPO / "golden" / "brick0" / f"{song_id}.gt.json").read_text())
    audio = REPO / gt["audio_path"]
    import librosa
    y, sr = librosa.load(str(audio), sr=22050, mono=True)
    chroma, ftimes = chroma_cqt_ltas(y, sr, hop_length=512)     # (12, T)
    ch_cn_frames = centre_norm(chroma.T)                        # (T, 12)
    dur = len(y) / sr

    # frame RMS on the same grid
    rms = librosa.feature.rms(y=y, hop_length=512)[0]
    rms_t = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)

    probs = frame_posteriors(audio)
    triad_p, bass_p = probs[0], probs[1]                        # (F,73) (F,13)
    n_f = triad_p.shape[0]
    mtimes = np.arange(n_f) * FRAME_DT
    # root/triad marginals
    tp = triad_p[:, 1:].reshape(n_f, 6, 12)                     # (F, type, root)
    musx_root_p = tp.sum(axis=1)                                # (F,12)
    musx_nc = triad_p[:, 0]
    musx_arg = triad_p.argmax(axis=1)
    musx_arg_root = np.where(musx_arg == 0, -1, (musx_arg - 1) % 12)
    musx_arg_type = np.where(musx_arg == 0, -1, (musx_arg - 1) // 12)
    musx_bass_arg = bass_p.argmax(axis=1) - 1                   # -1 = N

    gts = gt["gt_chords"]

    # ---- per-GT-chord evidence -------------------------------------------
    per_chord = []
    for c in gts:
        t0, t1 = float(c["t0"]), float(c["t1"])
        root, q = c.get("root_pc"), c.get("quality") or "N"
        rec = {"t0": t0, "t1": t1, "label": c.get("label"), "root_pc": root,
               "quality": q, "bass_pc": c.get("bass_pc")}
        mc = mean_chroma(chroma, ftimes, t0, t1)
        mcn = centre_norm(mc[None])[0]
        if root is None:
            rec.update(gt_fit=None, best_fit=None, best=None)
        else:
            rec["gt_fit"] = float(centre_norm(template(root, q)[None])[0] @ mcn)
            fits = VOCAB_CN @ mcn
            bi = int(fits.argmax())
            rec["best_fit"] = float(fits[bi])
            rec["best"] = [VOCAB[bi][0], VOCAB[bi][1]]
            # best alternative that keeps the GT root (isolates the extension axis)
            same_root = [i for i, (r, _q) in enumerate(VOCAB) if r == root]
            j = same_root[int(np.argmax(fits[same_root]))]
            rec["best_same_root"] = [VOCAB[j][0], VOCAB[j][1]]
            rec["best_same_root_fit"] = float(fits[j])
            # best alternative that keeps the GT root AND the GT triad -> only the
            # 7th / 6th / 9th is allowed to move (the "Georgia" axis)
            tri = triad_of(q)
            ext = [i for i, (r, qq) in enumerate(VOCAB)
                   if r == root and triad_of(qq) == tri]
            if ext:
                k = ext[int(np.argmax(fits[ext]))]
                rec["ext_best"] = [VOCAB[k][0], VOCAB[k][1]]
                rec["ext_fit"] = float(fits[k])
            else:
                rec["ext_best"], rec["ext_fit"] = None, rec["gt_fit"]
            rec["ext_gap"] = rec["ext_fit"] - rec["gt_fit"]
            # does the GT root / bass pitch class actually sound?
            z = (mc - mc.mean()) / (mc.std() + 1e-9)
            rec["root_z"] = float(z[root])
            rec["bass_z"] = (float(z[c["bass_pc"]])
                             if c.get("bass_pc") is not None else None)

        lo = int(np.searchsorted(mtimes, t0))
        hi = max(int(np.searchsorted(mtimes, t1)), lo + 1)
        hi = min(hi, n_f)
        if hi > lo:
            rec["musx_nc"] = float(musx_nc[lo:hi].mean())
            if root is not None:
                rec["musx_p_root"] = float(musx_root_p[lo:hi, root].mean())
                rec["musx_root_agree"] = float((musx_arg_root[lo:hi] == root).mean())
                gt_ty = triad_of(q)
                ti = MUSX_TRIADS.index(gt_ty) if gt_ty in MUSX_TRIADS else -1
                rec["musx_triad_agree"] = float(
                    ((musx_arg_root[lo:hi] == root)
                     & (musx_arg_type[lo:hi] == ti)).mean())
                rec["musx_bass_agree"] = (
                    float((musx_bass_arg[lo:hi] == c["bass_pc"]).mean())
                    if c.get("bass_pc") is not None else None)
                seg_root = musx_arg_root[lo:hi]
                vals, cnt = np.unique(seg_root[seg_root >= 0], return_counts=True)
                rec["musx_top_root"] = int(vals[cnt.argmax()]) if len(vals) else None
                seg_ty = musx_arg_type[lo:hi]
                vt, ct = np.unique(seg_ty[seg_ty >= 0], return_counts=True)
                rec["musx_top_triad"] = MUSX_TRIADS[int(vt[ct.argmax()])] if len(vt) else None
                # Is the third-party reading really a DIFFERENT chord, or the same
                # chord voiced without its root (Em7/D read as G major)?  If its
                # note set fits inside the written chord plus one colour tone, the
                # disagreement is on the 7th/extension axis, not the root axis.
                rec["ext_consistent"], rec["ext_consistent_as"] = False, None
                if rec.get("musx_top_root") is not None and rec["musx_top_triad"]:
                    mset = {(rec["musx_top_root"] + iv) % 12
                            for iv in QUALITY_INTERVALS[rec["musx_top_triad"]]}
                    for q2 in _EXT_OF.get(triad_of(q), ()):
                        gset = {(root + iv) % 12 for iv in QUALITY_INTERVALS[q2]}
                        if mset <= gset:
                            rec["ext_consistent"] = True
                            rec["ext_consistent_as"] = q2
                            break
                # how much of the region's duration is spent on frames where the
                # third-party reading is one of those rootless voicings
                rec["ext_frames"] = None
                if rec["ext_consistent"]:
                    rec["ext_frames"] = 1.0
        rl, rh = int(np.searchsorted(rms_t, t0)), int(np.searchsorted(rms_t, t1))
        rec["rms"] = float(rms[rl:max(rh, rl + 1)].mean()) if rh > rl else float(rms[min(rl, len(rms) - 1)])
        per_chord.append(rec)

    song_rms = float(np.median([r["rms"] for r in per_chord]))

    # ---- regions: consecutive GT chords grouped to ~8 s -------------------
    TARGET = 8.0
    regions, cur = [], []
    for i, c in enumerate(gts):
        cur.append(i)
        span = gts[cur[-1]]["t1"] - gts[cur[0]]["t0"]
        nxt = gts[i + 1]["t1"] - gts[cur[0]]["t0"] if i + 1 < len(gts) else None
        if span >= TARGET or len(cur) >= 8 or i + 1 == len(gts) or (
                nxt is not None and nxt > TARGET * 1.8):
            regions.append(cur)
            cur = []
    if cur:
        regions[-1].extend(cur) if regions else regions.append(cur)

    out_regions = []
    for ri, idxs in enumerate(regions):
        cs = [per_chord[i] for i in idxs]
        t0, t1 = cs[0]["t0"], cs[-1]["t1"]
        w = np.array([c["t1"] - c["t0"] for c in cs])
        wn = w / w.sum()
        has = np.array([c["root_pc"] is not None for c in cs])

        def wavg(key, mask=None):
            m = has if mask is None else mask
            vals = [c.get(key) for c in cs]
            ok = [i for i in range(len(cs)) if m[i] and vals[i] is not None]
            if not ok:
                return None
            ww = w[ok] / w[ok].sum()
            return float(np.sum(ww * np.array([vals[i] for i in ok])))

        # ---- time-shift search on this region (drift) --------------------
        taus, fits = [], []
        for tau in TAU_GRID:
            tot, wsum = 0.0, 0.0
            for k, c in enumerate(cs):
                if c["root_pc"] is None:
                    continue
                mc = mean_chroma(chroma, ftimes, c["t0"] + tau, c["t1"] + tau)
                mcn = centre_norm(mc[None])[0]
                f = centre_norm(template(c["root_pc"], c["quality"])[None])[0] @ mcn
                tot += f * w[k]
                wsum += w[k]
            taus.append(tau)
            fits.append(tot / wsum if wsum else np.nan)
        fits = np.array(fits)
        if np.all(np.isnan(fits)):
            tau_star, tau_gain, fit0 = 0.0, 0.0, None
        else:
            bi = int(np.nanargmax(fits))
            tau_star = float(taus[bi])
            fit0 = float(fits[np.argmin(np.abs(np.array(taus)))])
            tau_gain = float(fits[bi] - fit0)

        # duration share of the region spent on chords whose ONLY problem is the
        # colour tone, and on chords the third-party model reads as a rootless
        # voicing of the written chord
        wtot = w[has].sum() if has.any() else 1.0
        ext_gap_frac = float(sum(w[i] for i, c in enumerate(cs)
                                 if has[i] and (c.get("ext_gap") or 0) >= 0.25) / wtot)
        ext_cons_frac = float(sum(w[i] for i, c in enumerate(cs)
                                  if has[i] and c.get("ext_consistent")) / wtot)
        seq = tuple((c["root_pc"], c["quality"]) for c in cs)
        rec = {
            "song_id": song_id, "region_idx": ri, "t0": t0, "t1": t1,
            "chord_idx": idxs,
            "labels": [c["label"] for c in cs],
            "seq": ["%s|%s" % (c["root_pc"], c["quality"]) for c in cs],
            "seq_key": "-".join("%s|%s" % (r, q) for r, q in seq),
            "gt_fit": wavg("gt_fit"),
            "best_fit": wavg("best_fit"),
            "best_same_root_fit": wavg("best_same_root_fit"),
            "ext_fit": wavg("ext_fit"),
            "root_z": wavg("root_z"),
            "bass_z": wavg("bass_z"),
            "musx_p_root": wavg("musx_p_root"),
            "musx_root_agree": wavg("musx_root_agree"),
            "musx_triad_agree": wavg("musx_triad_agree"),
            "musx_bass_agree": wavg("musx_bass_agree"),
            "musx_nc": wavg("musx_nc", np.ones(len(cs), bool)),
            "ext_gap_frac": ext_gap_frac, "ext_cons_frac": ext_cons_frac,
            "tau_star": tau_star, "tau_gain": tau_gain, "fit_at_0": fit0,
            "rms_db": float(20 * np.log10(max(wavg("rms", np.ones(len(cs), bool)), 1e-9)
                                          / max(song_rms, 1e-9))),
            "chords": cs,
        }
        rec["margin"] = (None if rec["gt_fit"] is None or rec["best_fit"] is None
                         else rec["best_fit"] - rec["gt_fit"])
        # how much of that margin survives once the 7th/extension may move
        rec["margin_ext"] = (None if rec["ext_fit"] is None or rec["best_fit"] is None
                             else rec["best_fit"] - rec["ext_fit"])
        # how much BETTER a different 7th on the same root+triad would fit:
        # this, not `margin`, is the "the colour tone is wrong" signal
        rec["ext_gap"] = (None if rec["ext_fit"] is None or rec["gt_fit"] is None
                          else rec["ext_fit"] - rec["gt_fit"])
        # ... and once ANY chord on the GT root is allowed
        rec["margin_root"] = (None if rec["best_same_root_fit"] is None
                              or rec["best_fit"] is None
                              else rec["best_fit"] - rec["best_same_root_fit"])
        out_regions.append(rec)

    # ---- cross-repeat consistency ----------------------------------------
    # Slide the region's chart content over the WHOLE song and find every other
    # place the same chord sequence is charted; compare how well the audio backs
    # the chart there vs here.  Region boundaries need not line up with the form.
    key_all = [(c["root_pc"], c["quality"]) for c in gts]

    def fit_run(j: int, L: int) -> float | None:
        ws, tot = 0.0, 0.0
        for k in range(j, j + L):
            c = per_chord[k]
            if c.get("gt_fit") is None:
                continue
            ww = c["t1"] - c["t0"]
            tot += c["gt_fit"] * ww
            ws += ww
        return tot / ws if ws else None

    for r in out_regions:
        idxs = r["chord_idx"]
        L = len(idxs)
        pat = key_all[idxs[0]:idxs[0] + L]
        sib = []
        if L >= 2:
            for j in range(0, len(key_all) - L + 1):
                if j == idxs[0]:
                    continue
                if key_all[j:j + L] == pat:
                    f = fit_run(j, L)
                    if f is not None:
                        sib.append({"t0": gts[j]["t0"], "t1": gts[j + L - 1]["t1"],
                                    "fit": f})
        r["sib_n"] = len(sib)
        r["sib"] = sib[:6]
        if sib and r["gt_fit"] is not None:
            med = float(np.median([s["fit"] for s in sib]))
            r["sib_med"] = med
            r["sib_gap"] = float(med - r["gt_fit"])
        else:
            r["sib_med"], r["sib_gap"] = None, None

    return {
        "song_id": song_id, "title": gt.get("title"), "audio_path": gt["audio_path"],
        "duration_s": dur, "verified": gt.get("verified"),
        "form": gt.get("form"), "n_gt_chords": len(gts),
        "gt_span": [gts[0]["t0"], gts[-1]["t1"]],
        "regions": out_regions,
    }


if __name__ == "__main__":
    only = sys.argv[1:] or SONGS
    out = {}
    for s in only:
        print("...", s, flush=True)
        out[s] = analyse(s)
        print("   regions:", len(out[s]["regions"]), flush=True)
    (SCRATCH / "regions.json").write_text(json.dumps(out))
    print("wrote", SCRATCH / "regions.json")
