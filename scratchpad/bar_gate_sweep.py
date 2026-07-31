#!/usr/bin/env python3
"""Measure the per-bar fold gate on Brick-0. 2026-07-31.

The gate (harmonia/models/musx_posterior_fold.bar_agreement / _bar_weights) was
written but never scored — the agent that built it hit a session limit first.
Baseline to beat, fold ON, gate OFF: root 0.768 / partial 0.716 / strict 0.532.

Run: .venv/bin/python scratchpad/bar_gate_sweep.py
"""
from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "golden" / "brick0"

ARMS = [
    ("gate OFF (baseline)", {"HARMONIA_FOLD_BAR_GATE": "off"}),
    ("hard, amaj >= 0.60",  {"HARMONIA_FOLD_BAR_GATE": "hard", "HARMONIA_FOLD_BAR_STAT": "amaj", "HARMONIA_FOLD_BAR_MIN": "0.60"}),
    ("hard, amaj >= 0.75",  {"HARMONIA_FOLD_BAR_GATE": "hard", "HARMONIA_FOLD_BAR_STAT": "amaj", "HARMONIA_FOLD_BAR_MIN": "0.75"}),
    ("hard, amaj == 1.00",  {"HARMONIA_FOLD_BAR_GATE": "hard", "HARMONIA_FOLD_BAR_STAT": "amaj", "HARMONIA_FOLD_BAR_MIN": "1.0"}),
    ("soft, amaj t=0.50",   {"HARMONIA_FOLD_BAR_GATE": "soft", "HARMONIA_FOLD_BAR_STAT": "amaj", "HARMONIA_FOLD_BAR_MIN": "0.50"}),
    ("soft, cos  t=0.50",   {"HARMONIA_FOLD_BAR_GATE": "soft", "HARMONIA_FOLD_BAR_STAT": "cos",  "HARMONIA_FOLD_BAR_MIN": "0.50"}),
]

gts = sorted(str(p) for p in GOLDEN.glob("*.gt.json")
             if json.loads(p.read_text()).get("verified"))
print(f"{len(gts)} verified songs\n")
rows = []
for name, env in ARMS:
    e = {**os.environ, **env}
    out = subprocess.run([str(REPO / ".venv/bin/python"), "-m",
                          "harmonia.eval.accuracy_score", *gts],
                         cwd=REPO, env=e, capture_output=True, text=True)
    tail = out.stdout.strip().splitlines()
    import re as _re
    got, per = {}, {}
    for ln in tail:
        if "POOLED" in ln:
            got = {k: float(v) for k, v in _re.findall(r"(\w+)=([0-9.]+)", ln)}
        elif "root=" in ln and "|" in ln:
            song = ln.strip().split()[0]
            per[song] = {k: float(v) for k, v in _re.findall(r"(\w+)=([0-9.]+)", ln)}
    got["_per"] = per
    rows.append((name, got, "\n".join(tail[-14:])))
    print(f"{name:24s} root={got.get('root',0):.3f} partial={got.get('partial',0):.3f} strict={got.get('strict',0):.3f}")
    if not got:
        print("  --- raw tail ---"); print("\n".join(tail[-18:])); print("  --- stderr ---"); print(out.stderr[-800:])
        break
json.dump([(n, g) for n, g, _ in rows], open(REPO/"scratchpad/bar_gate_sweep.json","w"), indent=1)
