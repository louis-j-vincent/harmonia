"""harmonia_min/harmonic_key.py — tonic track → mode → scale colours → feedback.

Minimal port (2026-07-31) of the v7c pipe in scratchpad/colour_hmm_song.py
(``decode_segments``, feat/harmonic-key). Stages, each consuming only what the
previous stage validated:

  1. per-chord chroma        NNLS bothchroma (cache), Hann centre-weighted,
                             TREBLE ONLY (BASS_MODE="none", Louis 2026-07-30)
  2. tonic track (v7)        CUSUM on "forbidden mass" (b2+#4 of each rival
                             tonic), hold-until-forced; segment named v7b-style:
                             duration candidates (roots the segment sits on)
                             decided by the restricted Krumhansl posterior
  3. mode audit (v6)         raised-3rd share on tonic-rooted chords
                             (fallback: all 3rd-voicing chords when mass < 1.5)
  4. colours (v3–v5)         gated 6th/7th-degree evidence → sticky 4-state
                             HMM with relax-to-home prior, per segment
  5. feedback                inflections (momentary borrowing) + challenges
                             (chord non-diatonic to every colour → alternatives)

Deliberate cuts from the scratchpad version (minimal, and the crap is gone):
  * NO structure fold — it needs rigid_grid/vocab_sections (cut section
    machinery); the script itself treats no-fold as a graceful fallback.
    Revisit with the sections milestone.
  * NO module-global MODE — mode is a parameter everywhere. The scratchpad
    mutates a global inside the segment loop and resets it after; one missed
    reset away from cross-segment contamination.
  * ONE quality vocabulary — templates come from labels.py, not a third
    duplicate table.
  * NO plotting / printing / payload parsing — chords come in as the
    ChartModel dicts pipeline.py already builds ({root,q,bass,nc,t0,t1}).

Calibration constants are byte-identical to v4.1 (do not retune here).
"""
from __future__ import annotations

import logging

import numpy as np

from harmonia_min.key_profiles import infer_key
from harmonia_min.labels import _TAIL_PCS, _TRIAD_PCS

logger = logging.getLogger(__name__)

# state -> (raised 6th?, raised 7th?) — tonic-relative
STATES = {"natural": (0, 0), "harmonic": (0, 1), "dorian": (1, 0), "melodic": (1, 1)}
MODE_HOME = {"minor": "natural", "major": "melodic"}
# what each minor-machinery state MEANS when the segment is major
MAJOR_LABEL = {"melodic": "major", "dorian": "mixolydian",
               "harmonic": "harmonic maj", "natural": "mixo b6"}

# ── calibration, EXACTLY v4.1 (colour_hmm_this_love.py) ──────────────────────
Q = 0.85          # emission: expected raised-share when the state says "raised"
GAIN = 25.0       # evidence weight per unit of contrast-pc mass share
LAMBDA = 0.25     # per-chord prior cost of each raised degree
P_N_STAY, P_N_TO_R = 0.94, 0.02
P_R_STAY, P_R_TO_N, P_R_TO_R = 0.85, 0.12, 0.015
INFL_MIN_W = 1.0
INFL_MIN_MARGIN = 0.15
CHALLENGE_MARGIN = 1.25
TT_EPS = 0.02     # tonic track: per-chord noise margin
TT_PEN = 0.8      # tonic track: accumulated advantage to force a change
TT_INIT_N = 12    # tonic track: chords self-anchoring the opening tonic
MODE_MASS_MIN = 1.5   # mode audit: min tonic-chord mass before broad fallback

# challenge candidate vocabulary — kept to the exact This Love seven
_CAND_QUALITIES = {"": (0, 4, 7), "-": (0, 3, 7), "7": (0, 4, 7, 10),
                   "-7": (0, 3, 7, 10), "^7": (0, 4, 7, 11),
                   "h7": (0, 3, 6, 10), "o": (0, 3, 6)}


def template(q: str) -> tuple[int, ...] | None:
    """Interval template of a chart quality tail, or None if unscoreable."""
    return _TAIL_PCS.get(q) or _TRIAD_PCS.get(q)


# ── 1. per-chord chroma ──────────────────────────────────────────────────────

def chord_chroma(arr, times, t0, t1, tonic: int) -> np.ndarray:
    """L1-normalised TONIC-first 12-d treble chroma for one chord span.

    Hann centre weighting (Louis's transition rule: the middle of the span is
    the chord; edges are transition smear). arr is nnls bothchroma, index0=A.
    """
    roll = (9 - tonic) % 12                      # A-first -> tonic-first
    sel = (times >= t0) & (times < t1)
    if not sel.any():
        sel = np.array([np.argmin(np.abs(times - 0.5 * (t0 + t1)))])
    if np.count_nonzero(sel) >= 3:
        w = np.hanning(np.count_nonzero(sel))
        seg = (arr[sel] * w[:, None]).sum(0) / w.sum()
    else:
        seg = arr[sel].mean(0)
    m = np.roll(seg[12:], roll)                  # treble only
    s = m.sum()
    return m / s if s > 1e-9 else m


# ── 2. tonic track ───────────────────────────────────────────────────────────

def _forbidden_mass(m_abs: np.ndarray, tonic: int) -> float:
    """Mass on the two pcs outside EVERY colour scale of `tonic` (b2, #4)."""
    return float(m_abs[(tonic + 1) % 12] + m_abs[(tonic + 6) % 12])


def tonic_track(chords, arr, times) -> list[dict]:
    """Causal hold-until-forced tonic segments. [{tonic, i0, i1}] over chords."""
    ms = [chord_chroma(arr, times, ch["t0"], ch["t1"], 0) for ch in chords]
    fm = np.array([[_forbidden_mass(m, T) for T in range(12)] for m in ms])
    cur = int(np.argmin(fm[:TT_INIT_N].sum(0)))
    S = np.zeros(12)
    run_start = [0] * 12
    segs = [{"tonic": cur, "i0": 0}]
    for i in range(len(chords)):
        adv = (fm[i, cur] - fm[i]) - TT_EPS
        for T in range(12):
            if T == cur:
                continue
            if S[T] <= 0 and adv[T] > 0:
                run_start[T] = i
            S[T] = max(0.0, S[T] + adv[T])
        if S.max() > TT_PEN:
            T = int(S.argmax())
            segs[-1]["i1"] = run_start[T]
            segs.append({"tonic": T, "i0": run_start[T]})
            cur = T
            S[:] = 0.0
    segs[-1]["i1"] = len(chords)

    # v7b naming: the tonal CENTRE is a root the segment SITS on (>=60% of the
    # longest-held root), decided among candidates by the Krumhansl posterior
    # (maj+min summed). Krumhansl alone names the collection's F#-vs-F
    # neighbour; duration alone lets IV out-sit I. Krumhansl-only fallback for
    # segments with no usable roots. Merge neighbours that name identically.
    out: list[dict] = []
    for s in segs:
        dur: dict[int, float] = {}
        for ch in chords[s["i0"]:s["i1"]]:
            if not ch.get("nc"):
                dur[ch["root"]] = dur.get(ch["root"], 0.0) + ch["t1"] - ch["t0"]
        t0 = chords[s["i0"]]["t0"]
        t1 = chords[s["i1"] - 1]["t1"]
        sel = (times >= t0) & (times < t1)
        if not sel.any():                        # degenerate short segment
            out and out.append({**s, "tonic": out[-1]["tonic"]})
            continue
        if dur:
            # v7b.1 (19d7841): duration decides when DECISIVE (>=1.25x the
            # runner-up); Krumhansl only breaks near-ties, in log domain.
            # Measured there: LL margins cannot arbitrate (the wrong Ab
            # margin on Close beats the right C margin on This Love), but
            # duration decisiveness separates the cases (1.06x vs 1.55x).
            # Threshold calibrated on two songs = hypothesis (rule #5).
            ranked = sorted(dur, key=dur.get, reverse=True)
            if len(ranked) == 1 or dur[ranked[0]] >= 1.25 * dur[ranked[1]]:
                tonic = ranked[0]
            else:
                k = infer_key(np.roll(arr[sel, 12:].sum(0), 9))
                cands = [r for r in ranked if dur[r] >= 0.6 * dur[ranked[0]]]
                tonic = max(cands,
                            key=lambda r: float(np.logaddexp(
                                k.log_probs[r], k.log_probs[12 + r])))
        else:
            k = infer_key(np.roll(arr[sel, 12:].sum(0), 9))
            tonic = int(k.tonic)
        s = {**s, "tonic": int(tonic)}
        if out and out[-1]["tonic"] == s["tonic"]:
            out[-1]["i1"] = s["i1"]
        else:
            out.append(s)
    return out


# ── 3. mode audit ────────────────────────────────────────────────────────────

def mode_audit(chords, arr, times, tonic: int) -> tuple[str, float, float]:
    """(mode, raised-3rd share, evidence mass). share > 0.5 -> major."""
    def _pass(tonic_only):
        num = den = 0.0
        for ch in chords:
            if ch.get("nc"):
                continue
            ivs = template(ch["q"])
            if ivs is None:
                continue
            rr = (ch["root"] - tonic) % 12
            if tonic_only:
                if rr != 0:
                    continue
            elif not {(rr + iv) % 12 for iv in ivs} & {3, 4}:
                continue
            m = chord_chroma(arr, times, ch["t0"], ch["t1"], tonic)
            num += m[4]
            den += m[3] + m[4]
        return num, den

    num, den = _pass(True)
    if den < MODE_MASS_MIN:                      # song rarely sits on its tonic
        num, den = _pass(False)
    x3 = num / den if den > 1e-9 else 0.5
    return ("major" if x3 > 0.5 else "minor"), x3, den


# ── 4. colour decode (gated evidence → sticky HMM) ───────────────────────────

def _gates(ch, tonic: int) -> tuple[bool, bool]:
    """(chord voices a 6th degree?, a 7th degree?) — tonic-relative."""
    if ch.get("nc"):
        return False, False
    ivs = template(ch["q"]) or (0, 4, 7)
    pcs = {((ch["root"] - tonic) % 12 + iv) % 12 for iv in ivs}
    return bool(pcs & {8, 9}), bool(pcs & {10, 11})


def _emission(t6, x6, t7, x7, home: str) -> dict:
    ll = {}
    h6, h7 = STATES[home]
    for s, (r6, r7) in STATES.items():
        q6 = Q if r6 else 1 - Q
        q7 = Q if r7 else 1 - Q
        v = GAIN * t6 * (x6 * np.log(q6) + (1 - x6) * np.log(1 - q6))
        v += GAIN * t7 * (x7 * np.log(q7) + (1 - x7) * np.log(1 - q7))
        v -= LAMBDA * (abs(r6 - h6) + abs(r7 - h7))
        ll[s] = v
    return ll


def _viterbi(rows: list[dict], home: str) -> list[str]:
    names = list(STATES)
    T = np.empty((4, 4))
    for i, si in enumerate(names):
        for j, sj in enumerate(names):
            if si == home:
                p = P_N_STAY if i == j else P_N_TO_R
            else:
                p = P_R_STAY if i == j else (P_R_TO_N if sj == home else P_R_TO_R)
            T[i, j] = np.log(p)
    delta = np.array([rows[0][s] for s in names])
    back = []
    for row in rows[1:]:
        tr = delta[:, None] + T
        back.append(tr.argmax(0))
        delta = tr.max(0) + np.array([row[s] for s in names])
    path = [int(delta.argmax())]
    for bp in reversed(back):
        path.append(int(bp[path[-1]]))
    return [names[i] for i in reversed(path)]


def _decode(arr, times, chords, tonic: int, mode: str):
    home = MODE_HOME[mode]
    rows, ev6, ev7, chromas = [], [], [], []
    for ch in chords:
        m = chord_chroma(arr, times, ch["t0"], ch["t1"], tonic)
        g6, g7 = _gates(ch, tonic)
        t6 = m[8] + m[9] if g6 else 0.0
        t7 = m[10] + m[11] if g7 else 0.0
        x6 = m[9] / t6 if t6 > 1e-9 else 0.5
        x7 = m[11] / t7 if t7 > 1e-9 else 0.5
        rows.append(_emission(t6, x6, t7, x7, home))
        ev6.append((t6, x6))
        ev7.append((t7, x7))
        chromas.append(m)
    return _viterbi(rows, home), rows, ev6, ev7, chromas


# ── 5. feedback on the chords ────────────────────────────────────────────────

def _inflections(path, rows, ev6, ev7) -> dict[int, str]:
    """Chords whose own evidence says a different colour than the prevailing
    path — momentary borrowing, flagged only above the evidence floor."""
    flags = {}
    for i, row in enumerate(rows):
        own = max(row, key=row.get)
        if own == path[i]:
            continue
        (t6, x6), (t7, x7) = ev6[i], ev7[i]
        r6p, r7p = STATES[path[i]]
        r6o, r7o = STATES[own]
        ok = False
        if r6o != r6p:
            ok |= GAIN * t6 >= INFL_MIN_W and abs(x6 - 0.5) >= INFL_MIN_MARGIN
        if r7o != r7p:
            ok |= GAIN * t7 >= INFL_MIN_W and abs(x7 - 0.5) >= INFL_MIN_MARGIN
        if ok:
            flags[i] = own
    return flags


def _scale_pcs(colour: str, mode: str) -> frozenset[int]:
    r6, r7 = STATES[colour]
    third = 4 if mode == "major" else 3
    return frozenset({0, 2, third, 5, 7, 9 if r6 else 8, 11 if r7 else 10})


def _challenges(chords, chromas, path, flags, tonic: int, mode: str) -> dict:
    """Chords non-diatonic to EVERY colour → best-supported alternatives.
    {idx: {"kind": "challenge"|"suspect", "alts": [(root_abs, q, score)]}}"""
    all_scales = [_scale_pcs(c, mode) for c in STATES]
    out = {}
    for i, ch in enumerate(chords):
        if ch.get("nc"):
            continue
        ivs = template(ch["q"])
        if ivs is None:
            continue
        rr = (ch["root"] - tonic) % 12
        tpl = frozenset((rr + iv) % 12 for iv in ivs)
        if any(tpl <= s for s in all_scales):
            continue
        scale = _scale_pcs(flags.get(i, path[i]), mode)

        def score(pcs):
            fit = len(pcs & scale) / len(pcs)
            sup = float(np.mean([chromas[i][p] for p in sorted(pcs)]))
            return sup * (0.7 + 0.3 * fit)

        written = score(tpl)
        cands = []
        for r in range(12):
            for qq, qivs in _CAND_QUALITIES.items():
                t2 = frozenset((r + iv) % 12 for iv in qivs)
                if t2 != tpl:
                    cands.append((score(t2), (r + tonic) % 12, qq))
        cands.sort(key=lambda c: -c[0])
        if cands[0][0] >= CHALLENGE_MARGIN * written:
            kind = "challenge"
        elif cands[0][0] > written:
            kind = "suspect"
        else:
            continue
        out[i] = {"kind": kind, "written_score": round(written, 3),
                  "alts": [(r, q, round(sc, 3)) for sc, r, q in cands[:3]]}
    return out


# ── the packaged pipe (decode_segments equivalent) ───────────────────────────

def analyze_harmony(arr, times, chords) -> dict:
    """Full v7c pipe on a time-ordered chord list (ChartModel chord dicts).

    Returns {"segments": [{i0,i1,t0,t1,tonic,mode,x3,colours}],
             "colours": [colour-or-major-label per chord],
             "inflections": {idx: colour}, "challenges": {idx: {...}}}.
    """
    segs = tonic_track(chords, arr, times)
    colours: list[str] = []
    inflections: dict[int, str] = {}
    challenges: dict[int, dict] = {}
    seg_out = []
    for s in segs:
        sub = chords[s["i0"]:s["i1"]]
        mode, x3, _mass = mode_audit(sub, arr, times, s["tonic"])
        path, rows, ev6, ev7, chromas = _decode(arr, times, sub, s["tonic"], mode)
        sflags = _inflections(path, rows, ev6, ev7)
        for i, own in sflags.items():
            inflections[s["i0"] + i] = own
        for i, d in _challenges(sub, chromas, path, sflags,
                                s["tonic"], mode).items():
            challenges[s["i0"] + i] = d
        labelled = [MAJOR_LABEL[c] if mode == "major" else c for c in path]
        colours.extend(labelled)
        seg_out.append({
            "i0": s["i0"], "i1": s["i1"],
            "t0": round(float(sub[0]["t0"]), 2),
            "t1": round(float(sub[-1]["t1"]), 2),
            "tonic": int(s["tonic"]), "mode": mode, "x3": round(x3, 2),
            "colours": {c: labelled.count(c) for c in set(labelled)},
        })
    return {"segments": seg_out, "colours": colours,
            "inflections": inflections, "challenges": challenges}
