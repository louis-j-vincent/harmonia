"""harmonia_min/pipeline.py — the thin orchestration. Audio in → ChartModel out.

    audio → Beat This! beats/downbeats → musx frame posteriors →
    beat-grid re-decode (boundaries ON our beats) → key from decoded chords →
    bars on REAL downbeats → ChartModel dict for app_shell.html

Deliberate divergences from the old pipeline (feat/minimal-pipeline, 2026-07-31):

* **The bar grid is Beat This!'s real downbeats**, not a synthetic constant-
  tempo lattice and not chord-onset anchors interpolated per bar
  (known_issues "BOTH CLOCKS ARE SYNTHETIC"). Synthetic bar lines exist only
  where the tracker gave none (leading pickup / trailing bars), extrapolated
  at the median bar length.
* **No repeat folding, no section detection**: one section spanning the whole
  song, every bar written out. Sections/folding are a later milestone, scoped
  with Louis.
* **No post-passes**: no Occam, no vocabulary fold, no section arbiter. What
  the re-decode says is what renders.
* **Confidence is acoustic**: mean musx triad posterior of the decoded label
  over the segment's frames — not the old repetition-count heuristic.

ChartModel contract (mirrors chart_model.py's docstring + what loadModel in
app_shell.html actually reads): {file,title,video_id,audio_url,key,keyName,
bpb,nBars,barGrid,beatTimes,form,sections:[{id,label,tag,reps,spans,barRanges,
bars,barSpans}]}, Bar=[Chord×0..2], Chord={root,q,c,bass,nc,bar,beat,t0,t1}.
barSpans[r]=[[t0,t1]] is the playhead's map (server-built, one pass each).
"""
from __future__ import annotations

import logging

import numpy as np

from harmonia_min import beats as _beats
from harmonia_min import musx as _musx
from harmonia_min.key_profiles import infer_key
from harmonia_min.labels import chord_pcs, to_chord

logger = logging.getLogger(__name__)

# musx triad-plane family index (1-based; see musx.frame_posteriors docstring:
# column i>=1 is root (i-1)%12, triad type (i-1)//12+1 in {maj,min,sus4,sus2,
# dim,aug}) for each label quality we can meet.
_TRIAD_FAMILY = {
    "maj": 1, "maj7": 1, "7": 1, "9": 1, "maj9": 1, "13": 1,
    "maj/3": 1, "maj/5": 1, "maj/b7": 1, "maj/2": 1,
    "min": 2, "min7": 2, "min9": 2, "min/b3": 2, "min/5": 2,
    "min/b7": 2, "min/2": 2,
    "sus4": 3, "sus4(b7)": 3, "11": 3,
    "sus2": 4,
    "dim": 5, "dim7": 5, "hdim7": 5,
    "aug": 6,
}


def _segment_confidence(triad: np.ndarray, t0: float, t1: float,
                        label: str) -> float:
    """Mean triad-plane posterior of the decoded label over its own frames."""
    a = max(0, int(round(t0 / _musx.FRAME_DT)))
    b = min(triad.shape[0], int(round(t1 / _musx.FRAME_DT)))
    if b <= a:
        return 0.5
    if label == "N":
        col = 0
    else:
        root_s, _, qual = label.partition(":")
        fam = _TRIAD_FAMILY.get(qual)
        if fam is None:
            return 0.5
        from harmonia_min.labels import parse_root
        col = 1 + (fam - 1) * 12 + parse_root(root_s)
    return float(triad[a:b, col].mean())


def _bar_grid(downbeats: list[float], beat_times: list[float],
              t_start: float, t_end: float) -> list[float]:
    """Bar-line times covering [t_start, t_end]: REAL downbeats everywhere the
    tracker spoke, median-length extrapolation only before the first / after
    the last (so a chord sounding before the first detected downbeat — Let It
    Be's opening C — still gets a bar instead of being dropped)."""
    db = [float(t) for t in downbeats]
    if len(db) < 3:
        # No usable downbeats: fall back to every-4-beats from the first beat.
        bt = [float(t) for t in beat_times]
        db = bt[::4]
        logger.warning("bar grid: <3 downbeats from tracker, using every 4th beat")
    bar = float(np.median(np.diff(db)))
    grid = list(db)
    # Cover music before the first detected downbeat: full bars while they
    # fit, then ONE partial (pickup) bar clamped at t_start — Let It Be's
    # opening C sounds 1.8 s (~half a bar) before the first downbeat and must
    # get a cell, not be dropped. A sliver gap (<0.2 bar) just stretches bar 0.
    t0c = max(0.0, t_start)
    gap = grid[0] - t0c
    if gap > 0.2 * bar:
        while gap > 1.25 * bar:
            grid.insert(0, grid[0] - bar)
            gap -= bar
        grid.insert(0, t0c)
    elif gap > 0:
        # sliver (Let It Be: C onsets 20 ms before the first downbeat):
        # stretch bar 0 back rather than dropping the opening chord
        grid[0] = t0c
    while grid[-1] < t_end - 0.05:                # cover the tail
        grid.append(grid[-1] + bar)
    return [round(t, 4) for t in grid]


def analyze(audio_path, *, title: str = "", file_key: str = "",
            audio_url: str = "", progress=None) -> dict:
    """Full thin pipeline for one audio file → ChartModel dict.

    ``progress(stage:int, **fields)`` if given is called at each stage with
    the /api/job fields the analysing screen shows.
    """
    def report(stage, **kw):
        if progress:
            progress(stage, **kw)

    # 1 ── beats (hard error if Beat This! fails; librosa is banned)
    bd = _beats.track(audio_path)
    beat_times, downbeats = bd["beats"], bd["downbeats"]
    report(1, tempo_bpm=bd["bpm"], time_signature="4/4")

    # 2 ── musx frame posteriors (cache-hit for library songs; ~minutes fresh)
    probs = _musx.frame_posteriors(audio_path)
    triad = probs[0]

    # 3 ── beat-grid re-decode: boundaries land exactly on our beats
    segments, latency = _musx.redecode(beat_times, probs)
    report(3, draft_chords=[s for _, _, s in segments if s != "N"])

    # 4 ── key from duration-weighted pitch classes of the decoded chords
    pc_mass = np.zeros(12)
    for t0, t1, lab in segments:
        ch = to_chord(lab)
        if ch is None:
            continue
        for pc in chord_pcs(ch["root"], ch["q"]):
            pc_mass[pc] += (t1 - t0)
    kp = infer_key(pc_mass)
    key = {"tonic": kp.tonic, "mode": kp.mode}
    report(2, key_name=kp.key_name)

    # 5 ── bars on the real downbeat grid
    t_end = max(t1 for _, t1, _ in segments) if segments else beat_times[-1]
    # First onset that is real music: any chord, or an N segment long enough
    # (>half a bar) to be an actual played no-chord intro (Stand By Me's bass
    # riff) rather than a moment of leading silence.
    rough_bar = 4 * float(np.median(np.diff(beat_times)))
    t_start = next((t0 for t0, t1, lab in segments
                    if lab != "N" or (t1 - t0) > 0.5 * rough_bar),
                   beat_times[0])
    grid = _bar_grid(downbeats, beat_times, t_start, t_end)
    n_bars = len(grid) - 1
    beats_per_bar = int(round(np.median(np.diff(downbeats)) /
                              np.median(np.diff(beat_times)))) if len(downbeats) >= 3 else 4
    bpb = beats_per_bar if 2 <= beats_per_bar <= 7 else 4
    bt_arr = np.asarray(beat_times)

    # 6 ── chords into bars: a chord lives in the bar containing its onset;
    # a bar with no onset is a held bar ([]); ≤2 chords per bar (UI contract)
    bars: list[list[dict]] = [[] for _ in range(n_bars)]
    n_dropped = 0
    for t0, t1, lab in segments:
        ch = to_chord(lab)
        b = int(np.searchsorted(grid, t0 + 1e-6) - 1)
        if b < 0 or b >= n_bars:
            continue
        beat_in_bar = int(round((np.abs(bt_arr - t0).argmin() -
                                 np.abs(bt_arr - grid[b]).argmin())))
        entry = {
            "root": 0 if ch is None else ch["root"],
            "q": "" if ch is None else ch["q"],
            "bass": -1 if ch is None else ch["bass"],
            "nc": ch is None,
            "c": round(_segment_confidence(triad, t0, t1, lab), 3),
            "bar": b, "beat": max(0, min(bpb - 1, beat_in_bar)),
            "t0": round(float(t0), 3), "t1": round(float(t1), 3),
        }
        bars[b].append(entry)
    for b in range(n_bars):
        if len(bars[b]) > 2:                     # keep the two longest
            bars[b].sort(key=lambda c: c["t1"] - c["t0"], reverse=True)
            n_dropped += len(bars[b]) - 2
            bars[b] = sorted(bars[b][:2], key=lambda c: c["t0"])
    if n_dropped:
        logger.info("pipeline: dropped %d chords beyond 2-per-bar cap", n_dropped)

    # repetition count n (feeds the UI's "played N times" wording)
    from collections import Counter
    fam = Counter()
    for bar in bars:
        for c in bar:
            if not c["nc"]:
                fam[(c["root"], c["q"][:1])] += 1
    for bar in bars:
        for c in bar:
            c["n"] = 0 if c["nc"] else fam[(c["root"], c["q"][:1])]

    # 7 ── ChartModel: one unfolded section; barSpans IS the playhead map
    section = {
        "id": "A0", "label": "A", "tag": "", "reps": 1,
        "spans": [[grid[0], grid[-1]]],
        "barRanges": [[0, n_bars - 1]],
        "bars": bars,
        "barSpans": [[[grid[b], grid[b + 1]]] for b in range(n_bars)],
    }
    model = {
        "file": file_key, "title": title or "Untitled", "video_id": "",
        "audio_url": audio_url,
        "key": key, "keyName": kp.key_name,
        "bpb": bpb, "nBars": n_bars,
        "barGrid": grid, "beatTimes": beat_times,
        "form": None,
        "sections": [section],
        "meta": {"bpm": bd["bpm"], "musx_latency_ms": round(latency * 1000),
                 "n_segments": len(segments), "engine": "harmonia_min"},
    }
    report(4, final_chords=[s for _, _, s in segments if s != "N"],
           n_sections=1)
    return model
