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


# ── bar layout by BEAT-INDEX arithmetic (the live app's method) ──────────────
# First harmonia_min version assigned chords to bars by raw time containment
# (which bar's [t0,t1) contains the onset). Louis rejected that chart
# (2026-07-31, This Love): redecode boundaries sit on beats only up to the
# 23.22 ms musx frame grid, so a chord changing ON a bar line lands a few ms
# either side and gets the WRONG bar — every bar-opening chord rendered as the
# tail of the previous bar plus a held "%". The live app never does time
# containment: chords carry a detected-beat INDEX and bar/beat are integer
# arithmetic (scripts/render_youtube_chart.py::chart_to_interactive_inputs).
# This is the minimal version of exactly that.

# Harmonic bar-phase re-anchor thresholds — copied from the live app
# (render_youtube_chart.py, justified there on This Love: 75/120 chords on
# "beat 3", 1/120 on beat 0). If ≥55% of chords agree on ONE non-zero
# beat-in-bar residue and ≤15% sit on beat 0, the chords out-vote the
# tracker's downbeat phase: on a chart, the chord changes ARE the bar lines.
PHASE_CONSENSUS_MIN = 0.55
PHASE_BEAT0_MAX = 0.15


def _phase_correction(residues: list[int], bpb: int) -> int:
    """Signed beat shift to add to the bar-phase offset (0 = no-op)."""
    if not residues or bpb <= 1:
        return 0
    from collections import Counter
    res = Counter(r % bpb for r in residues)
    n = sum(res.values())
    modal, cnt = max(res.items(), key=lambda kv: kv[1])
    if modal == 0 or cnt / n < PHASE_CONSENSUS_MIN or res.get(0, 0) / n > PHASE_BEAT0_MAX:
        return 0
    return modal - bpb if modal > bpb / 2 else modal


def _bar_time(bt: np.ndarray, beat_idx: int, step: float) -> float:
    """Time of (possibly out-of-range) beat index, extrapolating at the edges."""
    n = len(bt)
    if beat_idx < 0:
        return float(bt[0] + beat_idx * step)
    if beat_idx >= n:
        return float(bt[-1] + (beat_idx - (n - 1)) * step)
    return float(bt[beat_idx])


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

    # 3 ── beat-grid re-decode: boundaries land exactly on our beats.
    # downbeat_times wired IN (2026-07-31, Louis's This Love report): at
    # phrase turns the frame evidence goes ambiguous for ~a beat and a FLAT
    # penalty is indifferent between last-beat and next-downbeat — two chords
    # landed one beat early (116.6s, 172.1s). Downbeat-graded costs (change on
    # downbeat cheap, elsewhere expensive) resolve the ambiguity the way a
    # lead sheet writes it. Old accuracy study: −0.17 pp (a wash) on label
    # overlap; placement is what the chart lives on.
    segments, latency = _musx.redecode(beat_times, probs,
                                       downbeat_times=downbeats)
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

    # 5 ── bar layout by beat-index arithmetic (the live app's method — see
    # the block comment above _phase_correction)
    bt_arr = np.asarray(beat_times, dtype=float)
    # drop a SHORT leading N (leading silence, not music); a long leading N
    # (Stand By Me's bass-riff intro) stays and gets its N.C. bar
    if segments and segments[0][2] == "N" and \
            (segments[0][1] - segments[0][0]) < 2.0 * float(np.median(np.diff(bt_arr))):
        segments = segments[1:]
    step = float(np.median(np.diff(bt_arr)))
    beats_per_bar = int(round(np.median(np.diff(downbeats)) / step)) \
        if len(downbeats) >= 3 else 4
    bpb = beats_per_bar if 2 <= beats_per_bar <= 7 else 4

    # every chord onset → nearest detected beat INDEX (boundaries already
    # land on beats up to the 23 ms musx frame grid, so nearest is exact)
    seg_bidx = [int(np.abs(bt_arr - t0).argmin()) for t0, _, _ in segments]

    # bar phase: the tracker's downbeats vote first (modal beat-index residue)…
    if len(downbeats) >= 3:
        db_idx = [int(np.abs(bt_arr - t).argmin()) for t in downbeats]
        from collections import Counter
        off = Counter(i % bpb for i in db_idx).most_common(1)[0][0]
    else:
        off = 0
    # …then the chords may out-vote them (harmonic re-anchor, live thresholds)
    chord_residues = [bi - off for bi, (_, _, lab) in zip(seg_bidx, segments)
                      if lab != "N"]
    corr = _phase_correction(chord_residues, bpb)
    if corr:
        logger.info("pipeline: harmonic re-anchor fired, shifting bar phase "
                    "by %+d beat(s)", corr)
        off += corr

    # bar/beat per segment: pure integer arithmetic, no time containment
    last_beat = int(np.abs(bt_arr - max(t1 for _, t1, _ in segments)).argmin()) \
        if segments else len(bt_arr) - 1
    n_bars = max(1, -((off - last_beat) // bpb))     # ceil((last_beat-off)/bpb)
    bars: list[list[dict]] = [[] for _ in range(n_bars)]
    n_dropped = 0
    for (t0, t1, lab), bi in zip(segments, seg_bidx):
        ch = to_chord(lab)
        eff = bi - off
        b = max(0, eff // bpb)                       # pickups clamp into bar 0
        if b >= n_bars:
            continue
        entry = {
            "root": 0 if ch is None else ch["root"],
            "q": "" if ch is None else ch["q"],
            "bass": -1 if ch is None else ch["bass"],
            "nc": ch is None,
            "c": round(_segment_confidence(triad, t0, t1, lab), 3),
            # pickups (eff<0) clamp into bar 0 AND display at beat 0 — the
            # modulo residue is meaningless once the bar is clamped, and the
            # beat now drives the visual quarter position
            "bar": b, "beat": (eff % bpb) if eff >= 0 else 0,
            "pickup": eff < 0,
            "t0": round(float(t0), 3), "t1": round(float(t1), 3),
        }
        bars[b].append(entry)
    # Overflow (> one chord per beat slot) is only possible in bar 0, where
    # pickups are CLAMPED in — an artifact of our own transformation. Splitter
    # lesson (2026-07-31, docs/postmusx_segment_loss.md): a floor/filter may
    # only ever discard artifacts its own cut created, NEVER received content.
    # So: shed clamped pickups (shortest first); if a bar still overflows,
    # something upstream broke its one-onset-per-beat invariant — fail loudly.
    for b in range(n_bars):
        if len(bars[b]) <= bpb:
            continue
        picks = sorted((c for c in bars[b] if c.get("pickup")),
                       key=lambda c: c["t1"] - c["t0"])
        while len(bars[b]) > bpb and picks:
            bars[b].remove(picks.pop(0))
            n_dropped += 1
        if len(bars[b]) > bpb:
            raise RuntimeError(
                f"bar {b} holds {len(bars[b])} chords for {bpb} beat slots "
                "with no pickups to shed — beat-aligned re-decode invariant "
                "broken upstream, refusing to silently drop real chords")
    if n_dropped:
        logger.info("pipeline: shed %d clamped pickup chords from bar 0",
                    n_dropped)

    # Louis's bar rule (2026-07-31): a bar lists ALL chords sounding in it.
    # If a bar's first onset is mid-bar, the chord carried over the bar line
    # is written again at beat 0 ("| C | C G |", never "| C | G |").
    # NO "%" any more (Louis, 2026-07-31 evening): a bar with no onset WRITES
    # its sounding chord too (carry-marked, so repetition counts ignore it) —
    # simile marks come back later as a pure rendering overlay. Killing the
    # empty-bar state removes a whole class of held-bar special cases.
    prev = None
    for b in range(n_bars):
        first = bars[b][0] if bars[b] else None
        if first is None and prev is not None:
            bars[b].append({**prev, "carry": True, "bar": b, "beat": 0,
                            "t0": round(_bar_time(bt_arr, off + b * bpb, step), 3),
                            "t1": round(_bar_time(bt_arr, off + (b + 1) * bpb, step), 3)})
        elif first is not None and first["beat"] > 0 and prev is not None \
                and not prev["nc"]:
            bars[b].insert(0, {**prev, "carry": True, "bar": b, "beat": 0,
                               "t0": round(_bar_time(bt_arr, off + b * bpb, step), 3),
                               "t1": first["t0"]})
        if bars[b]:
            prev = bars[b][-1]

    # repetition count n (feeds the UI's "played N times" wording)
    from collections import Counter
    fam = Counter()
    for bar in bars:
        for c in bar:
            if not c["nc"] and not c.get("carry"):
                fam[(c["root"], c["q"][:1])] += 1
    for bar in bars:
        for c in bar:
            c["n"] = 0 if c["nc"] else fam[(c["root"], c["q"][:1])]

    # 7 ── ChartModel sections: detected boundaries + A/B/C labels, UNFOLDED
    # (reps=1 each — the label strip milestone; folding is scoped separately).
    # Bar b's time span = the REAL beat times at its boundary indices
    # (extrapolated by the median beat only off the tracked range).
    grid = [round(_bar_time(bt_arr, off + b * bpb, step), 4)
            for b in range(n_bars + 1)]
    from harmonia_min.sections import detect_sections
    from harmonia_min.nnls_features import extract_bothchroma as _ebc
    _arr, _times = _ebc(audio_path)
    sections = []
    for si, sg in enumerate(detect_sections(grid, _arr, _times, bars)):
        b0, b1 = sg["b0"], sg["b1"]
        sections.append({
            "id": f"S{si}", "label": sg["label"], "tag": "", "reps": 1,
            "spans": [[grid[b0], grid[b1 + 1]]],
            "barRanges": [[b0, b1]],
            "bars": bars[b0:b1 + 1],
            "barSpans": [[[grid[b], grid[b + 1]]] for b in range(b0, b1 + 1)],
        })
    # 7b ── REPLI phase 1 (Louis, 2026-07-31): detect each section's internal
    # loop, stack same-position bars across all occurrences of a letter,
    # average their musx posteriors, decode the template a second time and
    # write its chords back on every contributing bar (variants excluded —
    # they keep the first-pass decode). Display folding comes later.
    from harmonia_min.folding import fold_letter_groups
    fold_report = fold_letter_groups(sections, bars, grid, probs, bpb,
                                     arr=_arr, times=_times)
    # repetition counts recomputed on the folded chords
    from collections import Counter as _C2
    fam2 = _C2()
    for bar in bars:
        for c in bar:
            if not c["nc"] and not c.get("carry"):
                fam2[(c["root"], c["q"][:1])] += 1
    for bar in bars:
        for c in bar:
            c["n"] = 0 if c["nc"] else fam2[(c["root"], c["q"][:1])]

    # 8 ── harmonic key analysis (harmonic_key.py: tonic track → mode →
    # colours → feedback). FAILS LOUDLY on any error — no silent fallback.
    # Splitter lesson #2 (2026-07-31, docs/known_issues.md): the old
    # pipeline's Occam gate silently disabled itself when musx_redecode
    # failed, and two "same config" runs differed by 6 chords. A stage that
    # fails must fail where everyone can see it (same doctrine as beats.py).
    flat = [c for bar in bars for c in bar]
    from harmonia_min.harmonic_key import analyze_harmony
    from harmonia_min.nnls_features import extract_bothchroma
    arr, times = extract_bothchroma(audio_path)
    H = analyze_harmony(arr, times, flat)
    for i, c in enumerate(flat):
        c["colour"] = H["colours"][i]
        if i in H["inflections"]:
            c["inflect"] = H["inflections"][i]
        if i in H["challenges"]:
            d = H["challenges"][i]
            c["flag"] = d["kind"]
            c["sug"] = [{"root": r, "q": q, "c": sc}
                        for r, q, sc in d["alts"]]
    key_segments = H["segments"]
    main = max(H["segments"], key=lambda s: s["t1"] - s["t0"])
    key = {"tonic": main["tonic"], "mode": main["mode"]}
    maj = main["tonic"] if main["mode"] == "major" else (main["tonic"] + 3) % 12
    names = ("C C# D Eb E F F# G G# A Bb B" if maj in (7, 2, 9, 4, 11)
             else "C Db D Eb E F Gb G Ab A Bb B").split()
    key_name = f"{names[main['tonic']]} {main['mode']}"
    report(2, key_name=key_name)

    model = {
        "file": file_key, "title": title or "Untitled", "video_id": "",
        "audio_url": audio_url,
        "key": key, "keyName": key_name, "keySegments": key_segments,
        "bpb": bpb, "nBars": n_bars,
        "barGrid": grid, "beatTimes": beat_times,
        "form": None,
        "fold": fold_report,
        "sections": sections,
        "meta": {"bpm": bd["bpm"], "musx_latency_ms": round(latency * 1000),
                 "n_segments": len(segments), "engine": "harmonia_min"},
    }
    report(4, final_chords=[s for _, _, s in segments if s != "N"],
           n_sections=len(sections))
    return model
