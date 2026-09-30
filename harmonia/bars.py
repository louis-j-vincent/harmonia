"""Bar layout: redecoded chord segments -> a bar grid, by beat-index arithmetic.

Assigns each chord segment from the beat-grid re-decode to a (bar, beat)
slot by INTEGER arithmetic on beat indices, never by time containment.
Extracted 2026-09-14 from harmonia_min/pipeline.py's bar-layout block
(sprint 6 of the refactor); the comments below are the reasoning trail
that block carried, kept verbatim with their dates.

Why beat-index arithmetic and not time containment: redecode boundaries sit
on beats only up to the 23.22 ms musx frame grid, so a chord changing ON a
bar line lands a few ms either side of it under time containment and gets
the WRONG bar — every bar-opening chord rendered as the tail of the previous
bar plus a held "%" (Louis rejected exactly this chart, 2026-07-31, This
Love). The live app never does time containment: chords carry a detected
beat INDEX and bar/beat are integer arithmetic
(scripts/render_youtube_chart.py::chart_to_interactive_inputs). This module
is the minimal version of exactly that.

Rules this module encodes, each paid for by a real chart bug:
* **Louis's bar rule** (2026-07-31): a bar lists ALL chords sounding in it —
  a chord carried over the bar line is written again at beat 0
  ("| C | C G |", never "| C | G |").
* **No "%"** (2026-07-31 evening): a bar with no onset writes its sounding
  chord too, `carry`-marked so repetition counts ignore it; simile marks are
  a display overlay applied later, not a state here.
* **Carry never an N.C.** (2026-08-10, "on propage les accords, pas les
  NC !"): only a real chord propagates into an onset-free bar. An N.C. means
  "I recognised nothing HERE", not "silence for the next N bars" — measured
  before the fix, Stand By Me rendered 93% N.C. against 8% in the raw
  detection because one N.C. at bar 0 was copied over eleven onset-free bars.
* **N.C. tail crossing the barline** (2026-08-09, "l'accord devrait
  commencer au début de la barre"): an N.C. that STARTED in an earlier bar
  and crosses into a chord's bar is folded to that chord's bar at beat 0 — a
  chart writes the harmony from the barline, not from the moment the band
  comes in. An N.C. that starts inside the bar (a mid-bar stop) stays where
  it happens.
* **Pickups clamp into bar 0** and display at beat 0 — the modulo residue is
  meaningless once the bar is clamped, the beat now drives the visual
  quarter position.
* **Overflow rule** (docs/postmusx_segment_loss.md, 2026-07-31 splitter
  lesson): a floor/filter may only ever discard artifacts ITS OWN cut
  created, never received content. So an overflowing bar 0 — the only bar
  where overflow is possible, since pickups are clamped in by this module's
  own transformation — sheds its own clamped pickups first (shortest first),
  and only if it still overflows does it fail loudly: something upstream
  broke the one-onset-per-beat invariant.

Ce que ce module ne fait PAS : pas de sections, pas de repli (folding), pas
de clé. Il pose des accords sur une grille de mesures, rien de plus — la
détection de sections et le repli des répétitions (harmonia_min/pipeline.py)
travaillent ENSUITE sur la liste `bars` qu'il renvoie, en la mutant sur
place.
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass

import numpy as np

from harmonia.labels import to_chord
from harmonia.musx import label_confidence

logger = logging.getLogger(__name__)


def _segment_confidence(triad: np.ndarray, t0: float, t1: float,
                        label: str) -> float:
    """Mean triad-plane posterior of the decoded label over its own frames.

    Delegates to `musx.label_confidence` so that folding — which computes the
    same thing on its AVERAGED template — cannot drift onto a different scale.
    See that function for why this matters (measured: 2 points of AUC).
    """
    return label_confidence(triad, t0, t1, label)


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


@dataclass(frozen=True)
class BarLayout:
    """The output of `layout_bars`: chords laid out on a bar grid.

    `bars` is the mutable payload: callers (folding, harmonic key) mutate its
    dicts IN PLACE, so this dataclass holds a reference, not a copy — freezing
    the dataclass only prevents rebinding the `bars` attribute itself, not
    mutating its contents. The raw chart is serialised from `bars` before any
    of that mutation happens (harmonia_min/pipeline.py's "DEUX RENDUS" split).
    """
    bpb: int
    off: int
    n_bars: int
    bar1_bar: int | None
    grid: list[float]
    bars: list[list[dict]]
    step: float
    #: The chord segments layout_bars actually placed — NOT always the
    #: `segments` the caller passed in: a short leading N (silence, not
    #: music) is dropped here, and callers that keep reading `segments`
    #: afterwards (the prompter, the final chord count) must read THIS list,
    #: not their own, or they disagree with what got laid on the grid.
    segments: list[tuple[float, float, str]]


def layout_bars(segments, beat_times, downbeats, triad: np.ndarray, *,
                bar1_time: float | None = None) -> BarLayout:
    """Lay redecoded chord segments onto a bar grid by beat-index arithmetic.

    See the module docstring for the rules encoded below and their dates.
    """
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
        off = Counter(i % bpb for i in db_idx).most_common(1)[0][0]
    else:
        off = 0
    # …then the chords may out-vote them (harmonic re-anchor, live thresholds)
    chord_residues = [bi - off for bi, (_, _, lab) in zip(seg_bidx, segments)
                      if lab != "N"]
    corr = _phase_correction(chord_residues, bpb)
    if corr:
        logger.info("bars: harmonic re-anchor fired, shifting bar phase "
                    "by %+d beat(s)", corr)
        off += corr
    _bar1_k = None
    if bar1_time is not None:
        # Set bar 1 (Louis, 2026-08-08): the user's own marked downbeat
        # OUT-VOTES both the tracker's phase and the harmonic re-anchor.
        # Snapped to the nearest tracked beat; the PHASE is taken (k mod bpb)
        # so the music before the mark stays as leading bars — nothing is
        # cut — and the marked bar itself becomes a HARD section boundary
        # further down (_force_bar1_sections), so the first section really
        # starts on the mark (same-night report on Sam Smith: the phase moved
        # but A still began elsewhere).
        _bar1_k = int(np.abs(bt_arr - float(bar1_time)).argmin())
        off = _bar1_k % bpb
        logger.info("bars: bar1 override at %.3fs -> beat %d, phase %d",
                    float(bar1_time), _bar1_k, off)

    # bar/beat per segment: pure integer arithmetic, no time containment
    last_beat = int(np.abs(bt_arr - max(t1 for _, t1, _ in segments)).argmin()) \
        if segments else len(bt_arr) - 1
    n_bars = max(1, -((off - last_beat) // bpb))     # ceil((last_beat-off)/bpb)
    # the marked beat's BAR index — (k - off) is a multiple of bpb by
    # construction (off = k mod bpb)
    bar1_bar = None if _bar1_k is None else \
        max(0, min(n_bars - 1, (_bar1_k - off) // bpb))
    bars: list[list[dict]] = [[] for _ in range(n_bars)]
    n_dropped = 0
    for _si, ((t0, t1, lab), bi) in enumerate(zip(segments, seg_bidx)):
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
        # An N.C. tail crossing the barline belongs to the chord's bar: a
        # chart writes the harmony from the barline, not from the moment the
        # band comes in (Louis, 2026-08-09: « l'accord devrait commencer au
        # début de la barre » — D entered on beat 2 of his marked bar 1, the
        # two beats before it being the intro silence's tail). Only when the
        # N.C. STARTED in an earlier bar — a stop inside the bar stays
        # written where it happens, N.C. intro bars stay N.C. (Stand By Me).
        if (not entry["nc"] and entry["beat"] > 0 and not entry["pickup"]
                and _si > 0 and segments[_si - 1][2] == "N"
                and (seg_bidx[_si - 1] - off) // bpb < b):
            entry["beat"] = 0
            entry["t0"] = round(_bar_time(bt_arr, off + b * bpb, step), 3)
        bars[b].append(entry)
    # Q4 (Louis, 2026-08-01): when an N.C. and a chord land on the SAME
    # (bar, beat) slot — leading silence snapped onto a real onset, or a
    # trailing N.C. on the last detected beat — merge by DROPPING the N.C.
    for b in range(n_bars):
        slots = {}
        for c in bars[b]:
            slots.setdefault(c["beat"], []).append(c)
        for beat, group in slots.items():
            if len(group) > 1 and any(c["nc"] for c in group)                     and any(not c["nc"] for c in group):
                for c in [c for c in group if c["nc"]]:
                    bars[b].remove(c)

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
        logger.info("bars: shed %d clamped pickup chords from bar 0",
                    n_dropped)

    # Louis's bar rule (2026-07-31): a bar lists ALL chords sounding in it.
    # If a bar's first onset is mid-bar, the chord carried over the bar line
    # is written again at beat 0 ("| C | C G |", never "| C | G |").
    # NO "%" any more (Louis, 2026-07-31 evening): a bar with no onset WRITES
    # its sounding chord too (carry-marked, so repetition counts ignore it) —
    # simile marks come back later as a pure rendering overlay. Killing the
    # empty-bar state removes a whole class of held-bar special cases.
    # …but ONLY a real chord is carried, never an N.C. (Louis, 2026-08-10:
    # « on propage les accords, pas les NC ! »). An N.C. means "I recognised
    # nothing HERE", not "nothing sounds for the next 25 seconds" — carrying
    # it turns one uncertain bar into a wall of silence. Measured before the
    # fix: Stand By Me displayed 93% N.C. against 8% in the raw detection,
    # because a single N.C. at bar 0 was copied over the eleven onset-free
    # bars that followed (docs/known_issues.md, 2026-08-09). A bar left empty
    # renders blank — honest — instead of asserting a silence nobody heard.
    prev = None
    for b in range(n_bars):
        first = bars[b][0] if bars[b] else None
        if first is None and prev is not None and not prev["nc"]:
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

    return BarLayout(bpb=bpb, off=off, n_bars=n_bars, bar1_bar=bar1_bar,
                     grid=grid, bars=bars, step=step, segments=segments)
