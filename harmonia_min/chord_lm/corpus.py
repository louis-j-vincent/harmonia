"""harmonia_min/chord_lm/corpus.py — iReal playlists -> gridded training corpus.

`data/ireal/*.txt` holds 2,401 charts across 7 playlists (jazz1460, pop400,
brazilian220, blues50, latin_salsa50, country, dixieland1). Those are the
project's highest-trust chord source (CLAUDE.md: iReal > UG > tabs > model), and
the only symbolic source we have with *bar structure*, which is what a metrical
chord LM needs. Chordonomicon (666k songs) has no bar timing at all — its chord
field is a flat symbol sequence — so it cannot train this grid, only an
event-level LM.

Splits are by SONG and deterministic (hash of the title), so a tune's A section
can never leak from train into test through its own repeats.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import re
from dataclasses import dataclass
from pathlib import Path

from harmonia.data.ireal_corpus import load_playlist, sectionized_measures, split_chords

from .grid import GriddedChart, chart_to_grid

DEFAULT_DIR = Path("data/ireal")


def _quiet_load(path: Path):
    """pyRealParser prints a line per tune to stdout — swallow it."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            return load_playlist(path)
        except Exception:
            return []


def load_charts(ireal_dir: Path | str = DEFAULT_DIR, *, slots_per_bar: int = 2,
                only_4_4: bool = True, min_bars: int = 8,
                max_unmapped_frac: float = 0.05) -> list[GriddedChart]:
    """Every parsable iReal chart as a GriddedChart.

    `only_4_4` keeps the 91.6% of tunes in 4/4 (plus 2/2, which is 4/4 in cut
    time). Mixing 3/4 in would put two different meanings on the same slot index
    — a 3/4 "half bar" is 1.5 beats — and the grid's metrical embedding assumes
    one meaning per slot.
    """
    ireal_dir = Path(ireal_dir)
    charts: list[GriddedChart] = []
    seen: set[str] = set()
    for path in sorted(ireal_dir.glob("*.txt")):
        for tune in _quiet_load(path):
            ts = tune.time_signature or (4, 4)
            if only_4_4 and ts not in ((4, 4), (2, 2)):
                continue
            beats = 4 if ts == (2, 2) else ts[0]
            key = f"{tune.title}|{tune.composer}"
            if key in seen:          # same standard in two playlists
                continue
            seen.add(key)
            try:
                measures = [(lab, split_chords(m))
                            for lab, m in sectionized_measures(tune)]
            except Exception:
                continue
            measures = [(lab, toks) for lab, toks in measures if toks]
            if len(measures) < min_bars:
                continue
            g = chart_to_grid(measures, title=tune.title,
                              slots_per_bar=slots_per_bar, beats_per_bar=beats,
                              key=tune.key, style=tune.style)
            if g.n_symbols == 0:
                continue
            if g.n_unmapped / g.n_symbols > max_unmapped_frac:
                continue
            charts.append(g)
    return charts


def split_of(title: str, *, val_frac: float = 0.1, test_frac: float = 0.1) -> str:
    """Deterministic song-level split from a hash of the title."""
    h = int(hashlib.sha1(title.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < test_frac:
        return "test"
    if h < test_frac + val_frac:
        return "val"
    return "train"


@dataclass
class CorpusSplits:
    train: list[GriddedChart]
    val: list[GriddedChart]
    test: list[GriddedChart]

    def summary(self) -> str:
        def n(cs):
            return f"{len(cs)} songs / {sum(len(c.tokens) for c in cs)} slots"
        return f"train {n(self.train)} | val {n(self.val)} | test {n(self.test)}"


def load_splits(ireal_dir: Path | str = DEFAULT_DIR, **kw) -> CorpusSplits:
    charts = load_charts(ireal_dir, **kw)
    buckets: dict[str, list[GriddedChart]] = {"train": [], "val": [], "test": []}
    for c in charts:
        buckets[split_of(c.title)].append(c)
    return CorpusSplits(**buckets)
