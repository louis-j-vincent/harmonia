"""Clean same-process profiling run: 1st call pays JIT/import warmup (discard
its number, that's a one-time server-lifetime cost), 2nd+3rd calls (DIFFERENT
songs, so genuinely cold nnls/musx caches) give the real per-song cold cost in
a warm process -- the number that matters for "how long does MY analyze take
on an already-running server".
"""
import cProfile
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
from harmonia.models.chord_pipeline_v1 import infer_chords_v1

KW = dict(feature_frontend="nnls24", bass_frontend="musx",
          quality_frontend="musx", segment_source="nnls",
          beat_period_mode="bestfit", cache_dir=Path("data/cache"))

SONGS = [
    ("warmup(discard)", "/tmp/warmup_dummy.wav"),
    ("song_a(commodores,260s)", "/tmp/song_a.wav"),
    ("song_b(billiejean,296s)", "/tmp/song_b.wav"),
]

for label, path in SONGS:
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    chart = infer_chords_v1(Path(path), **KW)
    pr.disable()
    t1 = time.perf_counter()
    print(f"[{label}] TOTAL wall: {t1-t0:.2f}s  chords={len(chart.chords)} "
          f"sections={len(chart.sections)} key={chart.global_key}")
    pr.dump_stats(f"scratchpad/profile_v3_{label.split('(')[0]}.pstats")
