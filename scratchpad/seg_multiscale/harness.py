"""Shared harness: baked chart HTML -> the `bars` the section detector sees.

The baked `docs/plots/inferred_*.html` files embed `const P = {...}` — the chart
payload AFTER the rigid regrid (Don't Know Why: nBars 67, 2.72 s bars), so the
bar/beat fields are already on the true grid and no audio is needed. That makes a
59-chart sweep cost seconds instead of hours.

`bars_from_payload` mirrors `chart_model.to_chart_model`'s chords->bars step
(minus the annotation sidecar, which is display-only): one list of chord dicts
per bar, sorted by beat, capped at 2 per bar by acoustic score.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

PLOTS = REPO / "docs" / "plots"
_DISPLAY_LEVEL = "exact"


def load_payload(path: Path) -> dict | None:
    s = path.read_text(errors="replace")
    key = "const P = "
    i = s.find(key)
    if i < 0:
        return None
    j = s.index("\n", i)
    try:
        return json.loads(s[i + len(key): j].rstrip().rstrip(";"))
    except Exception:
        return None


def bars_from_payload(P: dict, max_per_bar: int = 2):
    n_bars = P.get("nBars") or 0
    bars: list[list[dict]] = [[] for _ in range(n_bars)]
    for c in P.get("chords", []):
        b = c.get("bar", 0)
        if not 0 <= b < n_bars:
            continue
        lv = (c.get("lv") or {}).get(_DISPLAY_LEVEL) or {}
        entry = {
            "root": c.get("root", 0) % 12,
            "q": lv.get("q", ""),
            "cAcoustic": float(lv.get("c", 0.0)),
            "bar": b, "beat": c.get("beat", 0),
            "t0": float(c.get("t0", 0.0)), "t1": float(c.get("t1", 0.0)),
        }
        if c.get("nc"):
            entry["nc"] = True
            entry["q"] = "N"
        bars[b].append(entry)
    for i, bar in enumerate(bars):
        bar.sort(key=lambda e: e["beat"])
        if len(bar) > max_per_bar:
            keep = sorted(sorted(bar, key=lambda e: -e["cAcoustic"])[:max_per_bar],
                          key=lambda e: e["beat"])
            bars[i] = keep
    return bars, n_bars


# Experiment variants of a base chart: same song, re-baked under a one-off flag.
# Keeping them would weight those songs 2-3x in every corpus mean and, worse, some
# are STALE (chiquitita's `_npattern` still has 225 bars where the current bake has
# 58 — a pre-octave-fix grid). Dropped when the base file exists.
_VARIANT_SUFFIXES = ("_npattern", "_readable", "_loopdemo", "_missedchords", "_nfix",
                     "_bestfit", "_phone", "_barlocked", "_barlocked_anchored")
_DROP_ALWAYS = {"inferred_mayer_just_ain_t_flux_anchored"}


def _canonical(paths: list[Path]) -> list[Path]:
    stems = {p.stem for p in paths}
    out = []
    for p in paths:
        if p.stem in _DROP_ALWAYS:
            continue
        base = p.stem
        for suf in sorted(_VARIANT_SUFFIXES, key=len, reverse=True):
            if base.endswith(suf):
                base = base[: -len(suf)]
                break
        if base != p.stem and base in stems:
            continue
        out.append(p)
    return out


def all_charts(pattern: str = "inferred_*.html", canonical: bool = True):
    """[(slug, payload, bars, n_bars, bpb, tonic_pc)] for every baked chart."""
    out = []
    paths = sorted(PLOTS.glob(pattern))
    if canonical:
        paths = _canonical(paths)
    for p in paths:
        P = load_payload(p)
        if not P:
            continue
        bars, n_bars = bars_from_payload(P)
        if n_bars < 8:
            continue
        out.append({
            "slug": P.get("slug") or p.stem,
            "file": p,
            "payload": P,
            "bars": bars,
            "n_bars": n_bars,
            "bpb": P.get("bpb") or 4,
            "tonic_pc": int((P.get("home") or {}).get("tonic", 0)),
        })
    return out


if __name__ == "__main__":
    cs = all_charts()
    print(f"{len(cs)} charts")
    for c in cs[:5]:
        print(f"  {c['slug'][:50]:52s} n_bars={c['n_bars']:4d} bpb={c['bpb']}")
