"""Re-run the 3-song section-fallback V-measure check after fixing the
librosa_laplacian_sections timelag_filter bug (2026-07-17 adversarial review).

Symbolic (§10b) side is UNCHANGED by the bug (reused from the cached
scratchpad/sec_{song}.json dumped by the original session's sec_val.py); only
the librosa half is re-run against the corrected filter and re-scored vs iReal
GT with mir_eval.segment.nce (V-measure), same metric as the original table.
"""
import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import mir_eval

from harmonia.data.ireal_corpus import load_playlist, tune_to_mma
from harmonia.models.section_structure import librosa_laplacian_sections

SONGS = {
    "chain_of_fools": {
        "audio": "docs/audio/aretha_franklin_chain_of_fools_official_lyric_video.m4a",
        "playlist": "pop400", "title": "Chain Of Fools",
    },
    "autumn_leaves": {
        "audio": "docs/audio/autumn_leaves_easy_jazz_piano_piano_cover_sheets.m4a",
        "playlist": "jazz1460", "title": "Autumn Leaves",
    },
    "goodbye_ybr": {
        "audio": "docs/audio/elton_john_goodbye_yellow_brick_road_lyrics.m4a",
        "playlist": "pop400", "title": "Goodbye Yellow Brick Road",
    },
}


def ireal_section_per_bar(playlist: str, title: str) -> list[str]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        tunes = load_playlist(Path(f"data/ireal/{playlist}.txt"))
        match = next(t for t in tunes if t.title == title)
        mma = tune_to_mma(match)
    return mma.section_per_bar


def to_intervals_labels(sections: list[dict], duration_s: float):
    """[{start_s,end_s,label}] -> (intervals Nx2, labels) covering [0, duration_s]."""
    if not sections:
        return [[0.0, duration_s]], ["A"]
    ivs, labs = [], []
    for s in sections:
        ivs.append([s["start_s"], s["end_s"]])
        labs.append(s["label"])
    ivs[0][0] = 0.0
    ivs[-1][1] = duration_s
    for i in range(len(ivs) - 1):
        ivs[i + 1][0] = ivs[i][1]
    return ivs, labs


def gt_intervals_labels(section_per_bar: list[str], duration_s: float):
    n_bars = len(section_per_bar)
    bar_dur = duration_s / n_bars
    ivs, labs = [], []
    for i, lbl in enumerate(section_per_bar):
        t0, t1 = i * bar_dur, (i + 1) * bar_dur
        if labs and labs[-1] == lbl:
            ivs[-1][1] = t1
        else:
            ivs.append([t0, t1])
            labs.append(lbl)
    ivs[-1][1] = duration_s
    return ivs, labs


def vmeasure(ref_iv, ref_lab, est_iv, est_lab) -> float:
    import numpy as np
    _, _, v = mir_eval.segment.nce(np.array(ref_iv), ref_lab, np.array(est_iv), est_lab)
    return v


results = {}
for name, cfg in SONGS.items():
    cached = json.loads(Path(f"scratchpad/sec_{name}.json").read_text())
    sym_secs = cached["symbolic"]["secs"]
    audio_path = Path(cfg["audio"])
    duration_s = max(s["e"] for s in sym_secs) if sym_secs else 0.0
    # librosa_laplacian_sections doesn't report total duration; get it from ffprobe-free librosa load once, cheaply, via soundfile info if possible.
    import librosa
    duration_s = librosa.get_duration(path=str(audio_path))

    lap_secs_fixed = librosa_laplacian_sections(audio_path)
    lap_fixed_fmt = [{"start_s": s["start_s"], "end_s": s["end_s"], "label": s["label"]} for s in lap_secs_fixed]
    sym_fmt = [{"start_s": s["s"], "end_s": s["e"], "label": s["L"]} for s in sym_secs]
    lap_old_fmt = [{"start_s": s["s"], "end_s": s["e"], "label": s["L"]} for s in cached["librosa"]["secs"]]

    gt_per_bar = ireal_section_per_bar(cfg["playlist"], cfg["title"])
    ref_iv, ref_lab = gt_intervals_labels(gt_per_bar, duration_s)

    sym_iv, sym_lab = to_intervals_labels(sym_fmt, duration_s)
    lap_fixed_iv, lap_fixed_lab = to_intervals_labels(lap_fixed_fmt, duration_s)
    lap_old_iv, lap_old_lab = to_intervals_labels(lap_old_fmt, duration_s)

    v_sym = vmeasure(ref_iv, ref_lab, sym_iv, sym_lab)
    v_lap_fixed = vmeasure(ref_iv, ref_lab, lap_fixed_iv, lap_fixed_lab)
    v_lap_old = vmeasure(ref_iv, ref_lab, lap_old_iv, lap_old_lab)

    results[name] = {
        "duration_s": round(duration_s, 1),
        "n_bars_gt": len(gt_per_bar),
        "symbolic_labels": "".join(sym_lab),
        "librosa_OLD_labels": "".join(lap_old_lab),
        "librosa_FIXED_labels": "".join(lap_fixed_lab),
        "V_F_symbolic": round(v_sym, 3),
        "V_F_librosa_OLD_buggy": round(v_lap_old, 3),
        "V_F_librosa_FIXED": round(v_lap_fixed, 3),
    }
    print(name, json.dumps(results[name], indent=1))

Path("scratchpad/revalidation_results.json").write_text(json.dumps(results, indent=2))
means = {k: round(sum(r[k] for r in results.values()) / len(results), 3)
         for k in ("V_F_symbolic", "V_F_librosa_OLD_buggy", "V_F_librosa_FIXED")}
print("MEANS:", means)
