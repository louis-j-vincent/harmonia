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


def get_activations(ex, wav, midi_path):
    """Extract BP activations; if the WAV was deleted (disk cleanup), re-render this
    one song from its MIDI to a temp file (small, removed after)."""
    if Path(wav).exists():
        return ex.extract(Path(wav))
    renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
        tmp = Path(wf.name)
    renderer.render(REPO / midi_path, tmp,
                    RenderConfig(soundfont_path=renderer._find_soundfont("MuseScore_General.sf2")))
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", default="All The Things You Are")
    ap.add_argument("--conf", type=float, default=0.6, help="confidence to descend the tree")
    args = ap.parse_args()

    records = [json.loads(l) for l in open(DB)]
    man = {}
    for mm in map(json.loads, open(MANIFEST)):        # prefer the clean transpose-0 render
        if mm["song_id"] not in man or mm.get("transpose", 0) == 0:
            man[mm["song_id"]] = mm
    cand = [r for r in records if args.title.lower() in r["title"].lower() and r["song_id"] in man]
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
    acts = get_activations(ex, REPO / m["wav"], m["midi_path"])
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
        chords.append({"root": root, "b0": b0, "gt": p[1],
                       "gt_fam": FAM_IDX[BUCKET_FAMILY[p[1]]],
                       "gt_b7": BASE7_IDX[BUCKET_BASE7[p[1]]], "gt_ex": EXACT_IDX[p[1]],
                       "slot": (sec.get(bar, ("?", bar)), root),
                       "feat": feat})

    X = sc.transform(np.stack([c["feat"] for c in chords]))
    prob = {}
    for lv in ("fam", "b7", "ex"):
        p = np.full((len(chords), ncl[lv]), 1e-9)
        p[:, clf[lv].classes_] = clf[lv].predict_proba(X)
        prob[lv] = p / p.sum(1, keepdims=True)

    # certainty-weighted structure fold (per repeated slot)
    groups = defaultdict(list)
    for i, c in enumerate(chords):
        groups[c["slot"]].append(i)
    for lv in ("fam", "b7", "ex"):
        cert = prob[lv].max(1)
        for g in groups.values():
            if len(g) >= 2:
                g = np.array(g)
                w = cert[g] / (cert[g].sum() + 1e-9)
                prob[lv][g] = (prob[lv][g] * w[:, None]).sum(0)

    # hierarchical report: descend while confident
    for c, pf, pb, pe in zip(chords, prob["fam"], prob["b7"], prob["ex"]):
        fam_i, b7_i, ex_i = pf.argmax(), pb.argmax(), pe.argmax()
        cf, cb, ce = pf.max(), pb.max(), pe.max()
        if ce >= args.conf and cb >= args.conf:
            lvl, conf = 3, ce
        elif cb >= args.conf:
            lvl, conf = 2, cb
        else:
            lvl, conf = 1, cf
        c.update(pred_fam=fam_i, pred_lvl=lvl, conf=conf,
                 pred=label(c["root"], lvl, fam_i, b7_i, ex_i),
                 gt_label=NOTE[c["root"] % 12] + c["gt"],
                 correct_fam=(fam_i == c["gt_fam"]))

    # ── plot ──────────────────────────────────────────────────────────────────
    n = len(chords)
    fig, ax = plt.subplots(figsize=(min(0.5 * n + 2, 26), 5))
    for i, c in enumerate(chords):
        col = "#55A868" if c["correct_fam"] else "#C44E52"
        ax.add_patch(Rectangle((i, 1.2), 0.9, 0.9 * c["conf"], facecolor=col, alpha=0.85))
        ax.text(i + 0.45, 1.15, c["pred"], ha="center", va="top", fontsize=8,
                rotation=90, color=col, fontweight="bold")
        ax.text(i + 0.45, 0.5, c["gt_label"], ha="center", va="top", fontsize=7,
                rotation=90, color="#333")
        depth = {1: "fam", 2: "7th", 3: "exact"}[c["pred_lvl"]]
        ax.text(i + 0.45, 2.2, depth, ha="center", fontsize=6, color="#888", rotation=90)
    ax.axhline(1.15, color="#ccc", lw=0.5)
    fam_acc = np.mean([c["correct_fam"] for c in chords])
    ax.text(0, 2.75, f"predicted (bar height = confidence; green=family correct)   ·   "
            f"family accuracy {fam_acc:.0%}", fontsize=10, fontweight="bold")
    ax.text(0, 0.7, "ground truth", fontsize=9, color="#333")
    ax.text(0, 2.35, "reported depth", fontsize=8, color="#888")
    ax.set_xlim(-0.5, n); ax.set_ylim(0, 3)
    ax.axis("off")
    ax.set_title(f"Inferred chart with per-chord certainty — {rec['title']} ({rec['form']})\n"
                 "best model + certainty-weighted structure folding + confidence-gated reporting",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = REPO / "docs" / "plots" / "demo_inference.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"family accuracy {fam_acc:.0%}; mean confidence {np.mean([c['conf'] for c in chords]):.2f}")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
