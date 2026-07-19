"""Same-process cold+warm profiling (v2 — v1 accidentally spawned two
separate processes, so 'warm' still paid full numba/librosa JIT+import
warmup). This version calls infer_chords_v1 twice in ONE process: first call
pays JIT/import warmup + cold nnls/musx caches; second call (same audio, same
process) isolates true per-song steady-state cost.
"""
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
from harmonia.models.chord_pipeline_v1 import infer_chords_v1

AUDIO = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/henny_prof.wav")

KW = dict(feature_frontend="nnls24", bass_frontend="musx",
          quality_frontend="musx", segment_source="nnls",
          beat_period_mode="bestfit", cache_dir=Path("data/cache"))


def run(label, prof=False):
    pr = cProfile.Profile() if prof else None
    t0 = time.perf_counter()
    if prof:
        pr.enable()
    chart = infer_chords_v1(AUDIO, **KW)
    if prof:
        pr.disable()
    t1 = time.perf_counter()
    print(f"[{label}] TOTAL wall: {t1-t0:.2f}s  chords={len(chart.chords)} "
          f"sections={len(chart.sections)}")
    if pr:
        pr.dump_stats(f"scratchpad/profile_{label}_sameproc.pstats")
    return t1 - t0


t_cold = run("cold(1st-call-this-process)", prof=True)
t_warm = run("warm(2nd-call-same-process)", prof=True)
print(f"\nDELTA (per-process warmup, not per-song): {t_cold - t_warm:.2f}s")

if len(sys.argv) > 2:
    AUDIO2 = Path(sys.argv[2])
    KW2 = dict(KW)
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    chart2 = infer_chords_v1(AUDIO2, **KW2)
    pr.disable()
    t1 = time.perf_counter()
    print(f"[NEW SONG, warm process, cold caches] TOTAL wall: {t1-t0:.2f}s "
          f"chords={len(chart2.chords)} sections={len(chart2.sections)}")
    pr.dump_stats("scratchpad/profile_newsong_warmproc.pstats")
