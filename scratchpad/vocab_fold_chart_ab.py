#!/usr/bin/env python3
"""vocab_fold_chart_ab.py — the CHART, with and without the vocabulary fold.

The in-pipeline fold defers on This Love because it builds its own cheap
provisional chord chain and that chain is too noisy to yield a vocabulary. The
general fix (and the one demoed here) is TWO-PASS: decode once, build the
chart-grade vocabulary from the real decode, fold the observations with it, decode
again. Same rule for every song; no per-song tuning.

Run: .venv/bin/python scratchpad/vocab_fold_chart_ab.py [audio-stem]
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
CACHE = REPO / "data" / "cache"

_PC = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#d8d7d2"
ACCENT = "#8a2b2b"


def to_rq(label: str) -> tuple[int, str]:
    """`infer_chords_v1` label -> (root_pc, iReal-ish quality tail)."""
    from harmonia.theory.local_key import parse_token
    lab = (label or "").strip()
    if not lab or lab.upper() in ("N", "NC", "N.C."):
        return 0, "N"
    # labels arrive as mir_eval style "G:maj7" / "C:min" — map to iReal tails
    if ":" in lab:
        root, _, q = lab.partition(":")
        m = {"maj": "", "min": "-", "7": "7", "maj7": "^7", "min7": "-7",
             "dim": "o", "dim7": "o7", "hdim7": "h7", "aug": "+",
             "sus2": "sus2", "sus4": "sus4", "maj6": "6", "min6": "-6",
             "minmaj7": "-^7"}
        r, _q2, _b = parse_token(root)
        return r, m.get(q.split("/")[0], "")
    r, q, _b = parse_token(lab)
    return r, q


# `infer_chords_v1`'s DEFAULTS ARE NOT PRODUCTION: feature_frontend=bp48,
# quality/bass=nnls24. Calling it bare decodes a different (much worse) pipeline
# than the app — that is why an earlier version of this demo produced a jazz
# standard instead of This Love. The shipped config is mirrored in
# harmonia.eval.accuracy_score.SHIPPED_CONFIG; always pass one explicitly.
from harmonia.eval.accuracy_score import SHIPPED_CONFIG  # noqa: E402

PURE_NNLS = {**SHIPPED_CONFIG, "bass_frontend": "nnls24",
             "quality_frontend": "nnls24", "segment_source": "nnls"}


def decode(wav: Path, fold: bool, provisional=None, cfg=None):
    """One live decode. `provisional`, when given, replaces the pipeline's noisy
    provisional chord chain with a real one (the two-pass rule)."""
    import harmonia.models.chord_pipeline_v1 as P
    prev_flag = os.environ.get("HARMONIA_VOCAB_FOLD")
    prev_fn = P._provisional_chords
    os.environ["HARMONIA_VOCAB_FOLD"] = "1" if fold else "0"
    if provisional is not None:
        P._provisional_chords = lambda beat_proba, bt, min_beats=2: provisional
    try:
        chart = P.infer_chords_v1(wav, cache_dir=CACHE, **(cfg or SHIPPED_CONFIG))
    finally:
        P._provisional_chords = prev_fn
        if prev_flag is None:
            os.environ.pop("HARMONIA_VOCAB_FOLD", None)
        else:
            os.environ["HARMONIA_VOCAB_FOLD"] = prev_flag
    ch = getattr(chart, "chords", None)
    return list(ch if ch is not None else chart.get("chords", []))


def as_fold_input(chords: list[dict]) -> list[dict]:
    """Live chords -> the {root,q,t0,t1} shape the fold's grid/vocab chain wants."""
    out = []
    for c in chords:
        r, q = to_rq(c.get("label", ""))
        if q == "N":
            continue
        out.append({"root": r, "q": q,
                    "t0": float(c.get("start_s", 0.0)),
                    "t1": float(c.get("end_s", 0.0))})
    return out


def build_chart(chords: list[dict]):
    """Live chords -> the production display sections (the chart the app draws)."""
    from harmonia.models.rigid_grid import apply_rigid_grid, rigid_grid_for
    from harmonia.output.chart_display import regrid_display_sections
    payload = []
    for c in chords:
        r, q = to_rq(c.get("label", ""))
        payload.append({"root": r, "t0": float(c.get("start_s", 0.0)),
                        "t1": float(c.get("end_s", 0.0)),
                        "lv": {"exact": {"q": q, "c": float(c.get("confidence", 0.5))}},
                        "nc": q == "N"})
    grid = rigid_grid_for(payload, tonic_pc=0)
    if grid is None:
        return None, None, "rigid grid deferred"
    rc, n_bars = apply_rigid_grid(payload, grid, beats_per_bar=4,
                                 drop_before_grid=True)
    bars = [[] for _ in range(n_bars)]
    for c in rc:
        b = c.get("bar", 0)
        if 0 <= b < n_bars:
            lv = (c.get("lv") or {}).get("exact") or {}
            e = {"root": c["root"] % 12, "q": lv.get("q", ""),
                 "c": float(lv.get("c", 0.5)), "beat": c.get("beat", 0),
                 "t0": c.get("t0", 0.0), "t1": c.get("t1", 0.0)}
            if c.get("nc"):
                e["q"] = "N"
                e["nc"] = True
            bars[b].append(e)
    for bar in bars:
        bar.sort(key=lambda e: e["beat"])
    out = regrid_display_sections(bars, n_bars, tonic_pc=0, bpb=4)
    if out is None:
        return None, None, "display detector deferred"
    return out[0], out[1], None


def cell(bar: list[dict]) -> str:
    real = [c for c in bar if c.get("q") != "N" and not c.get("nc")]
    if not real:
        return "·"
    return " ".join(_PC[c["root"] % 12] + (c.get("q") or "") for c in real[:2])


def draw(ax, sections, form, title, note):
    ax.set_facecolor(SURFACE)
    ax.axis("off")
    ax.set_title(title, fontsize=12, color=INK, loc="left", fontweight="bold")
    if sections is None:
        ax.text(0.02, 0.9, note or "no chart", fontsize=11, color=ACCENT,
                transform=ax.transAxes, va="top")
        return
    y = 0.96
    for s in sections:
        occ = ", ".join(f"{a+1}-{b+1}" for a, b in s.get("barRanges", []))
        head = f"{s['label']}"
        if s.get("reps", 1) > 1:
            head += f"  ×{s['reps']}"
        ax.text(0.02, y, head, fontsize=12, color=ACCENT, fontweight="bold",
                transform=ax.transAxes, va="top")
        ax.text(0.14, y, f"bars {occ}", fontsize=8, color=INK2,
                transform=ax.transAxes, va="top")
        y -= 0.045
        cells = [cell(b) for b in s["bars"]]
        for r0 in range(0, len(cells), 4):
            row = cells[r0:r0 + 4]
            ax.text(0.04, y, " │ ".join(f"{c:<12s}" for c in row),
                    fontsize=10.5, color=INK, family="monospace",
                    transform=ax.transAxes, va="top")
            y -= 0.040
        for v in s.get("endings", {}).get("variants", []):
            tail = " │ ".join(cell(b) for b in v["bars"])
            ax.text(0.06, y, f"{v['label']}.  {tail}", fontsize=9.5,
                    color="#1f8a6b", family="monospace",
                    transform=ax.transAxes, va="top")
            y -= 0.038
        y -= 0.018
    ax.text(0.02, max(y, 0.03), f"FORM  {form}", fontsize=9.5, color=INK2,
            transform=ax.transAxes, va="top", family="monospace")


def main():
    stem = sys.argv[1] if len(sys.argv) > 1 else "maroon_5_this_love"
    audio = REPO / "docs" / "audio" / f"{stem}.m4a"
    if not audio.exists():
        raise SystemExit(f"no audio at {audio}")

    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "a.wav"
        subprocess.run(["ffmpeg", "-y", "-i", str(audio), "-ar", "44100", str(wav)],
                       check=True, capture_output=True)

        cfgname = sys.argv[2] if len(sys.argv) > 2 else "shipped"
        cfg = SHIPPED_CONFIG if cfgname == "shipped" else PURE_NNLS
        print(f"config: {cfgname}  {cfg}")
        print("pass 1 — decode with the fold OFF")
        off = decode(wav, fold=False, cfg=cfg)
        print(f"  {len(off)} chords; sample label {off[0].get('label')!r}")

        prov = as_fold_input(off)
        print(f"pass 2 — decode with the fold ON, vocabulary built from pass 1 "
              f"({len(prov)} chords)")
        on = decode(wav, fold=True, provisional=prov, cfg=cfg)
        print(f"  {len(on)} chords")

    changed = sum(1 for a, b in zip(off, on)
                  if a.get("label") != b.get("label")) if len(off) == len(on) else -1
    print(f"\nchord count OFF {len(off)}  ON {len(on)}   "
          + (f"labels changed: {changed}" if changed >= 0
             else "(different lengths — compare by time)"))

    s_off, f_off, n_off = build_chart(off)
    s_on, f_on, n_on = build_chart(on)
    print(f"\nFOLD OFF  form: {f_off or n_off}")
    print(f"FOLD ON   form: {f_on or n_on}")

    for tag, secs, form, note in (("OFF", s_off, f_off, n_off),
                                  ("ON", s_on, f_on, n_on)):
        print(f"\n--- chart, fold {tag} ---")
        if secs is None:
            print(f"  {note}")
            continue
        for s in secs:
            occ = ", ".join(f"{a+1}-{b+1}" for a, b in s.get("barRanges", []))
            print(f"  {s['label']:4s} ×{s.get('reps',1)}  bars {occ}")
            print("        " + " | ".join(cell(b) for b in s["bars"]))
            for v in s.get("endings", {}).get("variants", []):
                print(f"        {v['label']}. " + " | ".join(cell(b) for b in v["bars"]))

    fig, ax = plt.subplots(1, 2, figsize=(15, 8.5), facecolor=SURFACE)
    draw(ax[0], s_off, f_off, f"{stem} [{cfgname}] — fold OFF", n_off)
    draw(ax[1], s_on, f_on, f"{stem} [{cfgname}] — fold ON (two-pass vocab)", n_on)
    plt.tight_layout()
    out = REPO / "scratchpad" / f"vocab_fold_chart_ab_{stem}_{cfgname}.png"
    plt.savefig(out, dpi=125, facecolor=SURFACE)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
