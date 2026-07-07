"""State-of-the-art demo: infer one song's chords WITH a certainty per chord.

Pipeline (best of what we built):
  audio  → per-chord features (register-split chroma)
         → trained model (family / seventh / exact probabilities)         [the likelihood]
         → certainty-weighted structure folding of repeated sections       [structure]
         → hierarchical report: family by default, deeper when confident   [chord tree + certainty]

Output: a plot of the inferred chart — per chord, the reported label at the depth
the model is confident about, a confidence bar, and the ground-truth chord, so you
can see where it's sure and where it backs off. → docs/plots/demo_inference.png

Usage: .venv/bin/python scripts/demo_infer_song.py [--title "All The Things You Are"]
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from analyze_accomp_emission import parse_chord, song_chord_spans  # noqa: E402
from analyze_accomp_priors import parse_key  # noqa: E402
from build_audio_chord_features import (BASE7, BASE7_IDX, BUCKET_BASE7, BUCKET_FAMILY,  # noqa: E402
                                        EXACT, EXACT_IDX, FAM_IDX, full_chroma, reg_chroma)
import tempfile  # noqa: E402

from learn_stage1_mapping import pool_beats  # noqa: E402
from harmonia.data.midi_renderer import MIDIRenderer, RenderConfig  # noqa: E402
from harmonia.models.stage1_pitch import PitchExtractor  # noqa: E402


def get_activations(ex, wav, midi_path, phone=False):
    """Extract BP activations; re-render this one song from MIDI if the WAV is gone.
    With phone=True, degrade the audio to a grubby phone recording first."""
    if Path(wav).exists() and not phone:
        return ex.extract(Path(wav))
    import soundfile as sf
    from build_accomp_audio_hard import time_varying_degrade
    renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
        tmp = Path(wf.name)
    renderer.render(REPO / midi_path, tmp,
                    RenderConfig(soundfont_path=renderer._find_soundfont("MuseScore_General.sf2")))
    if phone:
        a, sr = sf.read(tmp)
        a = a.mean(1) if a.ndim > 1 else a
        # NON-uniform degradation: each repeat corrupted differently → folding can help
        sf.write(tmp, time_varying_degrade(a.astype("float32"), sr, np.random.default_rng(0)), sr)
    acts = ex.extract(tmp, use_cache=False)
    tmp.unlink(missing_ok=True)
    return acts

DB = REPO / "data" / "accomp_db" / "db.jsonl"
MANIFEST = REPO / "data" / "accomp_db" / "audio" / "manifest.jsonl"
CLEAN_FEAT = REPO / "data" / "cache" / "audio_chord_features.npz"
FAMILIES = ["major", "minor", "diminished", "augmented", "suspended"]
NOTE = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
FAM_SUFFIX = {"major": "", "minor": "m", "diminished": "dim", "augmented": "aug", "suspended": "sus"}


def label(root, level, fam_i, b7_i, ex_i):
    r = NOTE[root % 12]
    if level == 1:
        return r + FAM_SUFFIX[FAMILIES[fam_i]]
    if level == 2:
        return r + BASE7[b7_i].replace("majT", "").replace("minT", "m").replace("dimT", "dim").replace("augT", "aug").replace("susT", "sus")
    return r + EXACT[ex_i].replace("maj", "M").replace("min", "m") if EXACT[ex_i] not in ("maj",) else r


# ── iReal-glyph tokens for chart_render (△/ø/° etc.) ────────────────────────────
_FAM_IREAL = {"major": "", "minor": "-", "diminished": "o", "augmented": "+", "suspended": "sus"}
_B7_IREAL = {"7sus4": "7sus", "aug7": "+7", "augT": "+", "augmaj7": "+^7", "dim7": "o7",
             "dimT": "o", "dom7": "7", "m7b5": "h7", "maj7": "^7", "majT": "", "min7": "-7",
             "minT": "-", "minmaj7": "-^7", "susT": "sus"}
_EX_IREAL = {"6": "6", "7sus4": "7sus", "aug": "+", "aug7": "+7", "augmaj7": "+^7", "dim": "o",
             "dim7": "o7", "dom7": "7", "dom7alt": "7alt", "m6": "-6", "m7b5": "h7", "maj": "",
             "maj7": "^7", "min": "-", "min7": "-7", "minmaj7": "-^7", "sus2": "sus2", "sus4": "sus"}


def ireal_label(root, level, fam_i, b7_i, ex_i):
    """Same prediction as label() but as an iReal token so chart_render draws the
    proper jazz glyphs (C-7, Ab^7, F#h7 → ø, Bo7 → °)."""
    r = NOTE[root % 12]
    if level == 1:
        return r + _FAM_IREAL[FAMILIES[fam_i]]
    if level == 2:
        return r + _B7_IREAL[BASE7[b7_i]]
    return r + _EX_IREAL[EXACT[ex_i]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", default="All The Things You Are")
    ap.add_argument("--conf", type=float, default=0.6, help="confidence to descend the tree")
    ap.add_argument("--phone", action="store_true", help="infer from a grubby phone recording")
    args = ap.parse_args()

    infer_song(args.title, conf_thresh=args.conf, phone=args.phone)


def infer_song(title, conf_thresh=0.6, phone=False):
    """Run the full inference for one song and return (rec, chords, bpb, stats).

    Each entry in ``chords`` carries the model's prediction ready for display:
    ``pred`` (label at the confident tree depth), ``pred_ireal`` (same, as an
    iReal-glyph token for chart_render), ``conf``, ``bar``/``beat``,
    ``gt_label`` and ``correct_fam``. ``stats`` has the stagewise accuracies.
    """
    records = [json.loads(l) for l in open(DB)]
    man = {}
    for mm in map(json.loads, open(MANIFEST)):        # prefer the clean transpose-0 render
        if mm["song_id"] not in man or mm.get("transpose", 0) == 0:
            man[mm["song_id"]] = mm
    cand = [r for r in records if title.lower() in r["title"].lower() and r["song_id"] in man]
    rec = cand[0]
    m = man[rec["song_id"]]
    print(f"Inferring: {rec['title']}  ({rec['form']})")

    # train models on all OTHER songs (no leakage), predict this one
    d = np.load(CLEAN_FEAT, allow_pickle=True)
    keep = d["song"] != rec["song_id"]
    Xtr = np.hstack([d["onset"], d["note"], d["bass"], d["treble"]])[keep]
    sc = StandardScaler().fit(Xtr)
    clf = {lv: LogisticRegression(max_iter=2000).fit(sc.transform(Xtr), d[k].astype(int)[keep])
           for lv, k in [("fam", "family"), ("b7", "base7"), ("ex", "exact")]}
    ncl = {"fam": 5, "b7": len(BASE7), "ex": len(EXACT)}

    # extract this song's per-chord audio + gt
    ex = PitchExtractor(cache_dir=REPO / "data" / "cache" / "accomp")
    acts = get_activations(ex, REPO / m["wav"], m["midi_path"], phone=phone)
    spb = 60.0 / m["tempo"]; bpb = m["beats_per_bar"]; nb = m["n_bars"] * bpb
    onset = pool_beats(acts.frame_times, acts.onset_probs, nb, spb)
    note = pool_beats(acts.frame_times, acts.note_probs, nb, spb)
    tonic = parse_key(rec["key"])[0]
    chord_at = {(e["bar"] - 1) * bpb + e["beat"]: e["mma"] for e in rec["chord_timeline"]}
    sec = {}
    i = 0
    lab = rec["section_per_bar"]
    while i < len(lab):
        j = i
        while j < len(lab) and lab[j] == lab[i]:
            j += 1
        for b in range(i, j):
            sec[b] = (lab[i], i)
        i = j

    chords = []
    for t0, t1, root, _q in song_chord_spans(rec):
        b0, b1 = int(round(t0 / spb)), min(int(round(t1 / spb)), nb)
        mma = chord_at.get(b0)
        p = parse_chord(mma) if mma else None
        if p is None or p[1] not in BUCKET_FAMILY or b1 <= b0:
            continue
        rt = root % 12
        rr = lambda c: np.roll(c, -rt)
        on_c = rr(full_chroma(onset[b0:b1].sum(axis=0)))
        if on_c.sum() < 1e-9:
            continue
        feat = np.hstack([on_c, rr(full_chroma(note[b0:b1].sum(axis=0))),
                          rr(reg_chroma(onset[b0:b1], 0, 52)), rr(reg_chroma(onset[b0:b1], 60, 200))])
        bar = b0 // bpb
        seclab, secstart = sec.get(bar, ("?", bar))
        pos_in_sec = b0 - secstart * bpb          # position within the section (beats)
        chords.append({"root": root, "b0": b0, "gt": p[1],
                       "gt_fam": FAM_IDX[BUCKET_FAMILY[p[1]]],
                       "gt_b7": BASE7_IDX[BUCKET_BASE7[p[1]]], "gt_ex": EXACT_IDX[p[1]],
                       # H1 fix: group by (section LABEL, position-in-section, root) so ALL
                       # occurrences of a section fold together — not by section start bar.
                       "slot": (seclab, pos_in_sec, root), "deg": (root - tonic) % 12,
                       "feat": feat})

    X = sc.transform(np.stack([c["feat"] for c in chords]))
    prob = {}
    for lv in ("fam", "b7", "ex"):
        p = np.full((len(chords), ncl[lv]), 1e-9)
        p[:, clf[lv].classes_] = clf[lv].predict_proba(X)
        prob[lv] = p / p.sum(1, keepdims=True)

    def fam_acc():
        return np.mean([prob["fam"][i].argmax() == chords[i]["gt_fam"] for i in range(len(chords))])
    acc_audio = fam_acc()

    # ── H1 solution: certainty-weighted structure fold, grouped by (label, pos, root) ──
    groups = defaultdict(list)
    for i, c in enumerate(chords):
        groups[c["slot"]].append(i)
    n_multi = sum(1 for g in groups.values() if len(g) >= 2)
    for lv in ("fam", "b7", "ex"):
        cert = prob[lv].max(1)
        for g in groups.values():
            if len(g) >= 2:
                g = np.array(g)
                w = cert[g] / (cert[g].sum() + 1e-9)
                prob[lv][g] = (prob[lv][g] * w[:, None]).sum(0)
    acc_fold = fam_acc()

    # ── H2 solution (a): KEY prior P(family|degree) — fixes diatonic errors (E→minor) ──
    mode = 0 if parse_key(rec["key"])[1] == "major" else 1
    kf, kb = defaultdict(lambda: np.ones(5) * 0.5), defaultdict(lambda: np.ones(ncl["b7"]) * 0.5)
    for dg, md, fm, b7v in zip(d["degree"], d["mode"], d["family"], d["base7"]):
        kf[(int(md), int(dg))][int(fm)] += 1
        kb[(int(md), int(dg))][int(b7v)] += 1
    W_KEY = 0.3
    for i, c in enumerate(chords):
        for lv, tbl in (("fam", kf), ("b7", kb)):
            pr = tbl[(mode, c["deg"])]; pr = pr / pr.sum()
            prob[lv][i] *= pr ** W_KEY
            prob[lv][i] /= prob[lv][i].sum()
    acc_key = fam_acc()

    # ── H2 solution (b): PROGRESSION prior P(quality | prev-chord, root-motion) —
    # captures ii-V-i / secondary dominants (A resolving down a fifth is a dominant) ──
    pf_tab = defaultdict(lambda: np.ones(5) * 0.5)
    pb_tab = defaultdict(lambda: np.ones(ncl["b7"]) * 0.5)
    for pb7, ri, fm, b7v in zip(d["prev_b7"], d["root_interval"], d["family"], d["base7"]):
        pf_tab[(int(pb7), int(ri))][int(fm)] += 1
        pb_tab[(int(pb7), int(ri))][int(b7v)] += 1
    W_PROG = 0.4
    prev_b7 = -1
    for i, c in enumerate(chords):
        ri = (c["root"] - chords[i - 1]["root"]) % 12 if i > 0 else 12
        for lv, tbl in (("fam", pf_tab), ("b7", pb_tab)):
            pr = tbl[(prev_b7, ri)]; pr = pr / pr.sum()
            prob[lv][i] *= pr ** W_PROG
            prob[lv][i] /= prob[lv][i].sum()
        prev_b7 = prob["b7"][i].argmax()      # inferred previous (not teacher-forced)
    acc_prog = fam_acc()

    print(f"family accuracy by stage: audio {acc_audio:.0%} → +fold {acc_fold:.0%} "
          f"({n_multi} multi-repeat slots) → +key {acc_key:.0%} → +progression {acc_prog:.0%}")

    # hierarchical report: descend while confident
    for c, pf, pb, pe in zip(chords, prob["fam"], prob["b7"], prob["ex"]):
        fam_i, b7_i, ex_i = pf.argmax(), pb.argmax(), pe.argmax()
        cf, cb, ce = pf.max(), pb.max(), pe.max()
        if ce >= conf_thresh and cb >= conf_thresh:
            lvl, cval = 3, ce
        elif cb >= conf_thresh:
            lvl, cval = 2, cb
        else:
            lvl, cval = 1, cf
        c.update(pred_fam=fam_i, pred_lvl=lvl, conf=cval, bar=c["b0"] // bpb, beat=c["b0"] % bpb,
                 pred=label(c["root"], lvl, fam_i, b7_i, ex_i),
                 pred_ireal=ireal_label(c["root"], lvl, fam_i, b7_i, ex_i),
                 gt_label=NOTE[c["root"] % 12] + c["gt"],
                 correct_fam=(fam_i == c["gt_fam"]),
                 # per-depth prediction + certainty, so a renderer can pin a fixed
                 # level or reproduce the confidence-gated descent itself.
                 levels={
                     "family": {"ireal": ireal_label(c["root"], 1, fam_i, b7_i, ex_i),
                                "conf": float(cf)},
                     "seventh": {"ireal": ireal_label(c["root"], 2, fam_i, b7_i, ex_i),
                                 "conf": float(cb)},
                     "exact": {"ireal": ireal_label(c["root"], 3, fam_i, b7_i, ex_i),
                               "conf": float(ce)},
                 })
    fam_acc = np.mean([c["correct_fam"] for c in chords])

    # ── iReal-style lead sheet ─────────────────────────────────────────────────
    MPR = 4                                  # measures per row
    n_bars = rec["n_bars"]
    bar_chords = defaultdict(list)
    for c in chords:
        bar_chords[c["bar"]].append(c)
    # section run starts (for the A/B/C markers)
    sec_start = {}
    lab = rec["section_per_bar"]
    i = 0
    while i < len(lab):
        j = i
        while j < len(lab) and lab[j] == lab[i]:
            j += 1
        sec_start[i] = lab[i]; i = j

    from matplotlib.colors import LinearSegmentedColormap
    # continuous confidence gradient: red (unsure) → amber → near-black (sure)
    conf_cmap = LinearSegmentedColormap.from_list(
        "conf", ["#C0392B", "#C77B1E", "#5A6B22", "#1a1a1a"])

    def conf_color(cf):
        return conf_cmap(float(np.clip((cf - 0.35) / 0.6, 0, 1)))

    n_rows = (n_bars + MPR - 1) // MPR
    fig, ax = plt.subplots(figsize=(MPR * 2.6 + 1, n_rows * 0.92 + 2.4))
    Wm, Hm = 1.0, 1.0
    ax.text(0, 1.45, f"{rec['title']}", fontsize=15, fontweight="bold")
    ax.text(0, 1.05, f"key {rec['key']}    ·    form [{rec['form']}]    ·    "
            "inferred by Harmonia", fontsize=10, color="#555")
    for bar in range(n_bars):
        row, col = bar // MPR, bar % MPR
        x, y = col * Wm, -row * Hm
        # measure box (iReal look)
        ax.add_patch(Rectangle((x, y), Wm, Hm * 0.7, fill=False, edgecolor="#333", lw=1.3))
        if bar in sec_start:
            ax.add_patch(Rectangle((x - 0.02, y + Hm * 0.7), 0.22, 0.22,
                                   facecolor="#333"))
            ax.text(x + 0.09, y + Hm * 0.7 + 0.11, sec_start[bar], ha="center",
                    va="center", color="white", fontsize=10, fontweight="bold")
        bc = sorted(bar_chords.get(bar, []), key=lambda c: c["beat"])
        fscale = 1.0 if len(bc) <= 2 else (0.72 if len(bc) == 3 else 0.58)
        for c in bc:
            bx = x + 0.06 + (c["beat"] / bpb) * (Wm - 0.12)
            t = float(np.clip((c["conf"] - 0.35) / 0.6, 0, 1))   # continuous confidence
            ax.text(bx, y + Hm * 0.34, c["pred"], ha="left", va="center",
                    fontsize=(10.5 + 3.5 * t) * fscale, fontweight="bold",
                    color=conf_color(c["conf"]))
            ax.text(bx, y + 0.08, c["gt_label"], ha="left", va="center",
                    fontsize=6.5 * (fscale + 0.15),
                    color="#3a7d3a" if c["correct_fam"] else "#C44E52", alpha=0.8)

    # confidence gradient legend (a colour strip)
    ybar = -n_rows * Hm - 0.45
    grad = np.linspace(0, 1, 100).reshape(1, -1)
    ax.imshow(grad, extent=[0, 1.4, ybar, ybar + 0.14], aspect="auto", cmap=conf_cmap)
    ax.text(-0.05, ybar + 0.07, "unsure", ha="right", va="center", fontsize=8, color="#C0392B")
    ax.text(1.5, ybar + 0.07, "sure", ha="left", va="center", fontsize=8, color="#1a1a1a")
    ax.text(0, ybar - 0.28,
            f"chord colour = model confidence (see gradient); where unsure it backs off up "
            f"the chord tree (e.g. 'Bb' instead of 'Bbm7').  tiny GT below each "
            f"(green = family correct).   Family accuracy {fam_acc:.0%}, "
            f"mean confidence {np.mean([c['conf'] for c in chords]):.2f}",
            fontsize=8.5, color="#555")
    ax.set_xlim(-0.15, MPR * Wm + 0.15); ax.set_ylim(ybar - 0.5, 1.7)
    ax.axis("off")
    plt.tight_layout()
    out = REPO / "docs" / "plots" / "demo_leadsheet.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"family accuracy {fam_acc:.0%}; mean confidence "
          f"{np.mean([c['conf'] for c in chords]):.2f}")
    print(f"→ {out}")

    stats = {"fam_acc": fam_acc, "mean_conf": float(np.mean([c["conf"] for c in chords]))}
    return rec, chords, bpb, stats


if __name__ == "__main__":
    main()
