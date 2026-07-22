"""harmonia/serving/loaders.py — pure read-only data-loaders (serving refactor).

Behavior-preserving MOVE out of ``scripts/harmonia_server.py``: these helpers
load a chart/annotation from disk and return plain data. The function bodies
below are BYTE-FOR-BYTE the originals; only their home changed. Every
dependency is a leaf-module constant (``ANNOT_DIR`` / ``PLOTS_DIR`` from
``harmonia.serving.config``) or the stdlib (``json`` / ``re`` / ``Path``), so
this is a true leaf module — it imports nothing from the server and creates no
import cycle. ``scripts/harmonia_server.py`` imports all three names back, so
every existing call site (including the annotation write helpers and the
grid-lane routes that read iReal alignments) resolves unchanged.

Deliberately LEFT in the server this round: ``_gt_chords_for_video`` — despite
loading GT, it is NOT a pure leaf. It delegates to ``_gt_chords_for_video_raw``,
which reads/writes the stateful module globals ``_billboard_ds`` (reassigned
under ``global``) and the mutable ``_billboard_gt_cache`` (cleared by
``_save_gt_offset``), and the corpus reader ``_billboard_video_to_track_id``.
Moving only the thin wrapper would fragment that billboard cluster across two
modules and need a lazy back-import — not a clean move — so it stays put.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from harmonia.serving.config import ANNOT_DIR, PLOTS_DIR


def _annot_path(filename: str) -> Path:
    return ANNOT_DIR / f"{filename}.json"


def _load_annotation(filename: str) -> dict:
    try:
        return json.loads(_annot_path(filename).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"schema": 1, "chart": filename, "annotator": "", "modified": None,
                "chords": [], "merges": []}


def _load_ireal_alignment(slug: str):
    """Parse docs/plots/irealb_<slug>.html → (chords, tempo).

    Each chord: {i, bar, beat, section, label, t0, t1}. `beat` is the 0-based
    ordinal within its bar (the irealb payload has no beat-in-bar offset), so
    (bar, beat) is a unique per-song key — the sidecar's chord address (§3).
    t0/t1 are the DTW-aligned starting suggestion the user corrects from.
    """
    p = PLOTS_DIR / f"irealb_{slug}.html"
    if not p.exists():
        return None, None
    m = re.search(r"window\.P\s*=\s*(\{.*?\})\s*;", p.read_text(encoding="utf-8"), re.S)
    if not m:
        return None, None
    try:
        payload = json.loads(m.group(1))
    except ValueError:
        return None, None
    tempo = float(payload.get("tempo") or 120)
    chords, bar_counts = [], {}
    for idx, c in enumerate(payload.get("chords", [])):
        bar = int(c.get("bar", 0))
        beat = bar_counts.get(bar, 0)
        bar_counts[bar] = beat + 1
        # irealb payloads carry no per-chord posterior; the DTW `match` field
        # (exact|mismatch vs the acoustic reading) is the only confidence-like
        # signal available, so the waveform UI colours bars from it:
        # exact -> high (green), mismatch -> low (red), unknown -> mid (amber).
        match = c.get("match", "")
        conf = 0.9 if match == "exact" else (0.25 if match == "mismatch" else 0.6)
        chords.append({
            "i": idx, "bar": bar, "beat": beat,
            "section": c.get("section", ""), "label": c.get("label", ""),
            "t0": float(c.get("t0", 0.0)), "t1": float(c.get("t1", 0.0)),
            "match": match, "conf": conf,
        })
    return chords, tempo
