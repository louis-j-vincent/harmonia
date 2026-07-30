"""Per-chord scale-colour tracking, generalised beyond This Love (v4.1 model).

Generic version of ``colour_hmm_this_love.py`` (which is frozen while the
main session works on it — do not merge back blindly): slug comes from
argv, tonic/mode/bpb come from the baked chart payload, and everything
tonic-specific is computed in TONIC-RELATIVE space:

  - chroma is rotated so the payload tonic sits at index 0;
  - chord roots are transposed to tonic-relative pcs for gates, colour
    scales and challenge templates;
  - output chord names are spelled back in absolute pitch names (sharp
    spelling for sharp-side minor tonics, flats otherwise).

The colour states (natural / harmonic / dorian / melodic minor) and every
calibration constant are byte-identical to the This Love v4.1 script —
this script exists to measure how This-Love-shaped those constants are
(error-pattern #5), NOT to retune them.

Outputs
-------
scratchpad/colour_hmm_<slug>.png   decoded band + per-degree evidence
stdout                             decode summary + ablations + flags + audits

Usage:  .venv/bin/python scratchpad/colour_hmm_song.py <slug> [--tonic N]
        --tonic N  overrides the payload tonic (diagnostic; loudly labelled)
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from harmonia.models import nnls_features as nf  # noqa: E402
from harmonia.models.beat_grid import bestfit_beat_period  # noqa: E402
from harmonia.theory.key_profiles import infer_key  # noqa: E402

PC_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
PC_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
SHARP_TONICS = {9, 4, 11, 6, 1, 8}  # Am Em Bm F#m C#m G#m -> sharp spelling
ROLL_TO_C = 9  # A-first -> C-first (nnls_features._ROLL_TO_C)

# state -> (raised 6th?, raised 7th?)  — tonic-relative, unchanged from v4.1
STATES = {
    "natural": (0, 0),
    "harmonic": (0, 1),
    "dorian": (1, 0),
    "melodic": (1, 1),
}
STATE_COLORS = {
    "natural": "#2a78d6",
    "harmonic": "#eb6834",
    "melodic": "#1baf7a",
    "dorian": "#eda100",
}
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"

# ── v6: mode layer ──────────────────────────────────────────────────────────
# The 3rd-degree axis (b3 vs 3) decides the HOME the relax prior pulls
# toward: minor home = "natural" (b6,b7); major home = "melodic" (6,7) —
# the major scale is the melodic state wearing a major third (measured on
# Close to You: the whole C-major body decoded "melodic" under minor
# assumptions). Set ONCE per song by mode_audit() BEFORE any decode.
# Module-global on purpose: scratchpad script, one song per run; keeps the
# v4.1 signatures byte-stable for the This Love reproduction guard.
MODE = "minor"
MODE_HOME = {"minor": "natural", "major": "melodic"}
MODE_STATE_LABEL = {
    "minor": {s: s for s in STATES},
    "major": {"melodic": "major", "dorian": "mixolydian",
              "harmonic": "harmonic maj", "natural": "mixo b6"},
}

# ── calibration constants, EXACTLY as in colour_hmm_this_love.py v4.1 ──────
BASS_MODE = "none"  # Louis 2026-07-30 directive — do not change
BASS_W = 0.3
BASS_MARGIN = 1.5
Q = 0.85  # emission: expected raised-share when the state says "raised"
GAIN = 25.0  # evidence weight per unit of contrast-pc mass share
HELD_W = 0.5  # below this total evidence weight, the chord is "held"
LAMBDA = 0.25  # per-chord prior cost of each raised degree
P_N_STAY, P_N_TO_R = 0.94, 0.02  # from natural
P_R_STAY, P_R_TO_N, P_R_TO_R = 0.85, 0.12, 0.015  # from a raised state
INFL_MIN_W = 1.0  # inflection: min gated evidence weight
INFL_MIN_MARGIN = 0.15  # inflection: min |raised share - 0.5|
CHALLENGE_MARGIN = 1.25  # candidate must beat the written chord by this factor
CENTRE_POOL = True  # Hann-centred span pooling (Louis's transition rule)

# chord-quality templates, intervals from the root (chart vocabulary).
# QUALITY_PCS is ALSO the challenge candidate vocabulary — kept to the exact
# This Love seven so the reproduction check is byte-identical.
QUALITY_PCS = {
    "": (0, 4, 7),
    "-": (0, 3, 7),
    "7": (0, 4, 7, 10),
    "-7": (0, 3, 7, 10),
    "^7": (0, 4, 7, 11),
    "h7": (0, 3, 6, 10),
    "o": (0, 3, 6),
}
# qualities that appear in other songs' charts: used to template/gate a
# WRITTEN chord, never proposed as a challenge candidate.
EXTRA_QUALITY_PCS = {
    "sus": (0, 5, 7),
    "+": (0, 4, 8),
    "-6": (0, 3, 7, 9),
    "6": (0, 4, 7, 9),
    "o7": (0, 3, 6, 9),
}


def quality_ivs(q: str) -> tuple[int, ...] | None:
    return QUALITY_PCS.get(q) or EXTRA_QUALITY_PCS.get(q)


# ── song loading ────────────────────────────────────────────────────────────


@dataclass
class Song:
    slug: str
    chords: list[dict]
    tonic: int  # C-first pc of the home tonic
    mode: str
    key_name: str
    bpb: int
    tonic_overridden: bool = False
    pcn: list[str] = field(default_factory=lambda: PC_FLAT)

    @property
    def roll(self) -> int:
        """np.roll amount taking an A-first 12-d vector to TONIC-first."""
        return (ROLL_TO_C - self.tonic) % 12

    def name_pc(self, pc_abs: int) -> str:
        return self.pcn[pc_abs % 12]


def load_payload(slug: str) -> dict:
    txt = (REPO / "docs/plots" / f"inferred_{slug}.html").read_text(errors="ignore")
    m = re.search(r"const\s+P\s*=\s*", txt)
    i = txt.index("{", m.end())
    depth, j, instr, esc = 0, i, False, False
    while j < len(txt):
        c = txt[j]
        if instr:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                instr = False
        elif c == '"':
            instr = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        j += 1
    return json.loads(txt[i : j + 1])


def load_song(slug: str, tonic_override: int | None = None) -> Song:
    P = load_payload(slug)
    home = P.get("home") or {}
    tonic = home.get("tonic", 0) if tonic_override is None else tonic_override
    pcn = PC_SHARP if tonic in SHARP_TONICS else PC_FLAT
    return Song(
        slug=slug,
        chords=P["chords"],
        tonic=tonic,
        mode=home.get("mode", "?"),
        key_name=P.get("keyName", "?"),
        bpb=int(P.get("bpb") or 4),
        tonic_overridden=tonic_override is not None,
        pcn=pcn,
    )


def chord_name(ch: dict, song: Song) -> str:
    if ch.get("nc"):
        return "NC"
    name = song.name_pc(ch["root"]) + ch["lv"]["exact"]["q"]
    if ch.get("bass", -1) >= 0 and ch["bass"] != ch["root"]:
        name += "/" + song.name_pc(ch["bass"])
    return name


def chord_gates(ch: dict, tonic: int) -> tuple[bool, bool]:
    """(voices a 6th degree?, voices a 7th degree?) — tonic-relative."""
    if ch.get("nc"):
        return False, False
    q = ch["lv"]["exact"]["q"]
    ivs = quality_ivs(q) or (0, 4, 7)
    root_rel = (ch["root"] - tonic) % 12
    pcs = {(root_rel + iv) % 12 for iv in ivs}
    return bool(pcs & {8, 9}), bool(pcs & {10, 11})


# ── chroma + emissions (tonic-relative once rolled) ─────────────────────────


def chord_chroma(arr, times, t0, t1, roll: int, bass_mode=None) -> np.ndarray:
    """L1-normalised TONIC-first 12-d chroma for one chord span."""
    bass_mode = BASS_MODE if bass_mode is None else bass_mode
    sel = (times >= t0) & (times < t1)
    if not sel.any():
        sel = np.array([np.argmin(np.abs(times - 0.5 * (t0 + t1)))])
    if CENTRE_POOL and np.count_nonzero(sel) >= 3:
        w = np.hanning(np.count_nonzero(sel))
        seg = (arr[sel] * w[:, None]).sum(0) / w.sum()
    else:
        seg = arr[sel].mean(0)
    bass = np.roll(seg[:12], roll)
    treb = np.roll(seg[12:], roll)

    def l1(v):
        s = v.sum()
        return v / s if s > 1e-9 else v

    m = l1(treb).copy()
    if bass_mode == "l1":
        m = m + 1.0 * l1(bass)
    elif bass_mode == "argmax" and bass.sum() > 1e-9:
        order = np.argsort(bass)[::-1]
        if bass[order[0]] >= BASS_MARGIN * max(bass[order[1]], 1e-9):
            m[int(order[0])] += BASS_W
    return l1(m)


def emission_from_ev(t6, x6, t7, x7, *, prior=True):
    ll = {}
    for s, (r6, r7) in STATES.items():
        q6 = Q if r6 else 1 - Q
        q7 = Q if r7 else 1 - Q
        v = GAIN * t6 * (x6 * np.log(q6) + (1 - x6) * np.log(1 - q6))
        v += GAIN * t7 * (x7 * np.log(q7) + (1 - x7) * np.log(1 - q7))
        if prior:
            h6, h7 = STATES[MODE_HOME[MODE]]
            v -= LAMBDA * (abs(r6 - h6) + abs(r7 - h7))
        ll[s] = v
    return ll


def emissions(chroma, gates, *, prior=True):
    g6, g7 = gates
    t6 = chroma[8] + chroma[9] if g6 else 0.0
    t7 = chroma[10] + chroma[11] if g7 else 0.0
    x6 = chroma[9] / t6 if t6 > 1e-9 else 0.5
    x7 = chroma[11] / t7 if t7 > 1e-9 else 0.5
    return emission_from_ev(t6, x6, t7, x7, prior=prior), (t6, x6), (t7, x7)


def _log_trans() -> np.ndarray:
    names = list(STATES)
    home = MODE_HOME[MODE]
    T = np.empty((len(names), len(names)))
    for i, si in enumerate(names):
        for j, sj in enumerate(names):
            if si == home:
                p = P_N_STAY if i == j else P_N_TO_R
            else:
                p = P_R_STAY if i == j else (P_R_TO_N if sj == home else P_R_TO_R)
            T[i, j] = np.log(p)
    return T


def viterbi(ll_rows: list[dict]) -> list[str]:
    names = list(STATES)
    T = _log_trans()
    delta = np.array([ll_rows[0][s] for s in names])
    back = []
    for row in ll_rows[1:]:
        trans = delta[:, None] + T
        back.append(trans.argmax(0))
        delta = trans.max(0) + np.array([row[s] for s in names])
    path = [int(delta.argmax())]
    for bp in reversed(back):
        path.append(int(bp[path[-1]]))
    return [names[i] for i in reversed(path)]


def decode(arr, times, chords, song: Song, bass_mode=None, *, gate=True, prior=True):
    rows, ev6, ev7, chromas = [], [], [], []
    for ch in chords:
        m = chord_chroma(arr, times, ch["t0"], ch["t1"], song.roll, bass_mode)
        gates = chord_gates(ch, song.tonic) if gate else ((not ch.get("nc"),) * 2)
        ll, e6, e7 = emissions(m, gates, prior=prior)
        rows.append(ll)
        ev6.append(e6)
        ev7.append(e7)
        chromas.append(m)
    return viterbi(rows), rows, ev6, ev7, chromas


def inflections(path, rows, ev6, ev7):
    flags = []
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
            flags.append((i, own))
    return flags


def scale_pcs_of(colour: str) -> frozenset[int]:
    """The 7-pc scale of a colour (TONIC-first pcs); 3rd degree from MODE."""
    r6, r7 = STATES[colour]
    third = 4 if MODE == "major" else 3
    return frozenset({0, 2, third, 5, 7, 9 if r6 else 8, 11 if r7 else 10})


def mode_audit(chords, arr, times, song: Song):
    """Decide minor vs major BEFORE the chord-level audit (v6).

    Aggregates the 3rd-degree contrast (b3 vs 3) over chords whose symbol
    voices a third-degree pc, weighted by the pair's chroma mass.
    Returns (mode, raised_share, evidence_mass); share > 0.5 -> major.
    Deliberately independent of infer_key (its confidence is pinned at
    1.0000 — known_issues 2026-07-30) and of the payload mode (wrong on
    2 of the 3 tested real-audio minor charts).
    """
    # Two passes: tonic-rooted chords first (their third IS the mode; a
    # modulation section can't poison them — Close to You's Db-land floods
    # rel-pc 3 with Eb=2nd-of-Db and dragged the broad measure to 0.46),
    # broad 3rd-degree-voicing gate as fallback when the song rarely sits
    # on its tonic chord.
    def _pass(tonic_only):
        num = den = 0.0
        for ch in chords:
            if ch.get("nc"):
                continue
            ivs = quality_ivs(ch["lv"]["exact"]["q"])
            if ivs is None:
                continue
            root_rel = (ch["root"] - song.tonic) % 12
            if tonic_only:
                if root_rel != 0:
                    continue
            elif not {(root_rel + iv) % 12 for iv in ivs} & {3, 4}:
                continue
            m = chord_chroma(arr, times, ch["t0"], ch["t1"], song.roll)
            num += m[4]
            den += m[3] + m[4]
        return num, den

    num, den = _pass(tonic_only=True)
    basis = "tonic-rooted chords"
    # Fallback bar measured, not guessed: Close to You's 12 tonic chords
    # carry mass 2.47 with a clean x3=0.75 (major); ~0.2 mass/chord is
    # real signal. 1.5 ≈ seven tonic chords' worth.
    if den < 1.5:
        num, den = _pass(tonic_only=False)
        basis = "all 3rd-voicing chords (fallback)"
    x3 = num / den if den > 1e-9 else 0.5
    return ("major" if x3 > 0.5 else "minor"), x3, den, basis


# ── v7: tonic track (hold-until-forced, chroma-driven) ──────────────────────
TT_EPS = 0.02  # per-chord noise margin on the forbidden-mass advantage
TT_PEN = 0.8  # accumulated advantage needed to force a tonic change
TT_INIT_N = 12  # chords used to self-anchor the opening tonic


def _forbidden_mass(m_abs: np.ndarray, tonic: int) -> float:
    """Chroma mass on the two pcs outside EVERY colour scale of `tonic`
    (b2 and #4) — the sharpest cheap evidence against a tonic."""
    return float(m_abs[(tonic + 1) % 12] + m_abs[(tonic + 6) % 12])


def tonic_track(chords, arr, times):
    """Causal hold-until-forced tonic segments from chroma (v7).

    Adapts the continuity doctrine of local_key.continuity_scale_track_v2
    (hold the collection until forced) — but that function needs
    trustworthy chord TOKENS, and the chart symbols are exactly what's
    under audit here, so the evidence is chroma. CUSUM per rival tonic:
    switch only after a sustained forbidden-mass advantage > TT_PEN; the
    boundary is where the winning rival's run began. The opening tonic is
    self-anchored on the first TT_INIT_N chords (the payload tonic is
    wrong on 2 of 3 tested charts — do not trust it).
    """
    ms = [chord_chroma(arr, times, ch["t0"], ch["t1"], ROLL_TO_C)
          for ch in chords]
    fm = np.array([[_forbidden_mass(m, T) for T in range(12)] for m in ms])
    cur = int(np.argmin(fm[:TT_INIT_N].sum(0)))
    start = cur
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

    # The 2-pc forbidden-mass contrast finds BOUNDARIES sharply (Close's
    # 98s modulation lands exactly) but is too thin to name the tonic (it
    # anchored This Love on Eb). Naming (v7b): the tonal CENTRE is the
    # chord root the segment sits on longest — Krumhansl names the
    # *collection* and picks the F#-vs-F neighbour on Close (B7/Bm flood
    # F#); the ear names the centre, and pop sits on its tonic chord.
    # Krumhansl stays as fallback for segments with no usable roots.
    # Documented risk: a IV-heavy vamp could out-sit the tonic chord.
    relabeled = []
    for s in segs:
        dur: dict[int, float] = {}
        for ch in chords[s["i0"]:s["i1"]]:
            if not ch.get("nc"):
                dur[ch["root"]] = dur.get(ch["root"], 0.0) + ch["t1"] - ch["t0"]
        t0 = chords[s["i0"]]["t0"]
        t1 = chords[s["i1"] - 1]["t1"]
        sel = (times >= t0) & (times < t1)
        k = infer_key(np.roll(arr[sel, 12:].sum(0), ROLL_TO_C))
        if dur:
            # candidates = roots the segment sits on (>=60% of the longest);
            # among them, the Krumhansl posterior (maj+min summed) decides.
            # Duration alone picked F on This Love (IV out-sits I there);
            # Krumhansl alone picked the F#-collection neighbour on Close.
            dmax = max(dur.values())
            cands = [r for r, d in dur.items() if d >= 0.6 * dmax]
            tonic = max(cands, key=lambda r: k.probs[r] + k.probs[12 + r])
        else:
            tonic = int(k.tonic)
        s = {**s, "tonic": int(tonic)}
        if relabeled and relabeled[-1]["tonic"] == s["tonic"]:
            relabeled[-1]["i1"] = s["i1"]
        else:
            relabeled.append(s)
    return relabeled, start


def _count_nondiatonic(chords, song: Song) -> int:
    """Chords whose template fits NO colour scale under the current MODE."""
    scales = [scale_pcs_of(c) for c in STATES]
    n = 0
    for ch in chords:
        if ch.get("nc"):
            continue
        ivs = quality_ivs(ch["lv"]["exact"]["q"])
        if ivs is None:
            continue
        rr = (ch["root"] - song.tonic) % 12
        tpl = frozenset((rr + iv) % 12 for iv in ivs)
        n += not any(tpl <= s for s in scales)
    return n


def _support(chroma: np.ndarray, pcs: frozenset[int]) -> float:
    return float(np.mean([chroma[p] for p in sorted(pcs)]))


def challenge_chords(chords, chromas, path, flags, song: Song):
    """Colour prior pushes back on chords non-diatonic to every colour.

    All template arithmetic is tonic-relative; proposals are spelled back
    in absolute pitch names. Written chords with a quality outside both
    template dicts are skipped (can't be scored)."""
    tonic = song.tonic
    all_colour_pcs = [scale_pcs_of(c) for c in STATES]
    flag_map = dict(flags)
    out = []
    for i, ch in enumerate(chords):
        if ch.get("nc"):
            continue
        q = ch["lv"]["exact"]["q"]
        ivs = quality_ivs(q)
        if ivs is None:
            continue
        root_rel = (ch["root"] - tonic) % 12
        tpl = frozenset((root_rel + iv) % 12 for iv in ivs)
        if any(tpl <= s for s in all_colour_pcs):
            continue  # diatonic to some colour: the flag layer handles it
        scale = scale_pcs_of(flag_map.get(i, path[i]))

        def score(pcs):
            fit = len(pcs & scale) / len(pcs)
            return _support(chromas[i], pcs) * (0.7 + 0.3 * fit)

        written = score(tpl)
        cands = []
        for r in range(12):
            for qq in QUALITY_PCS:
                t2 = frozenset((r + iv) % 12 for iv in QUALITY_PCS[qq])
                if t2 != tpl:
                    cands.append((score(t2), song.name_pc(r + tonic) + qq))
        cands.sort(reverse=True)
        if cands[0][0] >= CHALLENGE_MARGIN * written:
            out.append((i, cands[0][1], written, cands[:3], "challenge"))
        elif cands[0][0] > written:
            out.append((i, cands[0][1], written, cands[:3], "suspect"))
    return out


# ── beat grid (for the chart script; generic version of key_scale_dotprod) ──


def beat_grid(slug: str, duration_s: float) -> np.ndarray:
    raw = np.asarray(
        json.load(open(REPO / "data/cache/raw_beat_times_v2" / f"{slug}.json")), float
    )
    tempo_bpm = 60.0 / float(np.median(np.diff(raw)))
    period = bestfit_beat_period(raw, 60.0 / tempo_bpm)
    ang = 2 * np.pi * (raw % period) / period
    phase = (np.angle(np.mean(np.exp(1j * ang))) % (2 * np.pi)) * period / (2 * np.pi)
    return np.unique(
        np.concatenate(
            [[0.0], np.arange(phase, duration_s + period, period), [duration_s]]
        )
    )


def bar_anchor_offset(bt: np.ndarray, chords: list[dict], bpb: int) -> tuple[int, int]:
    """Beat offset whose bar lines catch the most chord onsets (<120 ms)."""
    onsets = np.array([c["t0"] for c in chords if not c.get("nc")])
    best_off, best_hits = 0, -1
    for off in range(bpb):
        starts = bt[off::bpb]
        hits = int(sum(np.abs(starts - t).min() < 0.12 for t in onsets))
        if hits > best_hits:
            best_off, best_hits = off, hits
    return best_off, best_hits


# ── structure fold (v4) ─────────────────────────────────────────────────────


def structure_slots(chords, song: Song):
    """chord index -> structural slot key, via the vocab-section machinery.

    Falls back gracefully when rigid_grid_for or vocab_sections declines
    (both return None on songs they don't fit — This Love never exercised
    that path): every chord becomes its own solo slot (fold = no-op)."""
    from harmonia.models.rigid_grid import apply_rigid_grid, rigid_grid_for
    from harmonia.models.section_vocab import form_string, vocab_sections

    tonic, bpb = song.tonic, song.bpb
    grid = rigid_grid_for(chords, tonic_pc=tonic)
    if grid is None:
        return {i: ("_solo", i, 0) for i in range(len(chords))}, "(rigid grid declined)"
    gch, n_bars = apply_rigid_grid(chords, grid, beats_per_bar=bpb,
                                   drop_before_grid=True)
    bars: list[list[dict]] = [[] for _ in range(n_bars)]
    for c in gch:
        b = c.get("bar", 0)
        if 0 <= b < n_bars:
            bars[b].append(
                {**c, "q": ((c.get("lv") or {}).get("exact") or {}).get("q", "")}
            )
    secs = vocab_sections(bars, n_bars, tonic_pc=tonic, bpb=bpb)
    if not secs:
        return {i: ("_solo", i, 0) for i in range(len(chords))}, "(no vocab sections)"
    t0_to_slot = {}
    for c in gch:
        b = c.get("bar", 0)
        sec = next((s for s in secs if s["bar0"] <= b < s["bar1"]), None)
        if sec is None:
            continue
        bi = (b - sec["bar0"]) % sec["d_bars"]
        half = min(int(c.get("beat", 0) / (bpb / 2)), 1)
        t0_to_slot[round(c["t0"], 3)] = (sec["label"], bi, half)
    slotmap = {}
    for i, c in enumerate(chords):
        slotmap[i] = t0_to_slot.get(round(c["t0"], 3), ("_solo", i, 0))
    return slotmap, form_string(secs)


def fold_evidence(ev6, ev7, chromas, slotmap):
    from collections import defaultdict

    groups = defaultdict(list)
    for i, k in slotmap.items():
        groups[k].append(i)
    pe6, pe7 = list(ev6), list(ev7)
    pch = [c.copy() for c in chromas]
    for idxs in groups.values():
        if len(idxs) < 2:
            continue
        for ev, pe in ((ev6, pe6), (ev7, pe7)):
            ts = np.array([ev[j][0] for j in idxs])
            xs = np.array([ev[j][1] for j in idxs])
            tbar = float(ts.mean())
            xbar = float((ts * xs).sum() / ts.sum()) if ts.sum() > 1e-9 else 0.5
            for j in idxs:
                pe[j] = (tbar, xbar)
        m = np.mean([chromas[j] for j in idxs], axis=0)
        for j in idxs:
            pch[j] = m
    return pe6, pe7, pch


def decode_folded(arr, times, chords, song: Song):
    _, _, ev6, ev7, chromas = decode(arr, times, chords, song)
    slotmap, form = structure_slots(chords, song)
    pe6, pe7, pch = fold_evidence(ev6, ev7, chromas, slotmap)
    rows = [emission_from_ev(*pe6[i], *pe7[i]) for i in range(len(chords))]
    return viterbi(rows), rows, pe6, pe7, pch, slotmap, form


# ── main ────────────────────────────────────────────────────────────────────


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        sys.exit("usage: colour_hmm_song.py <slug> [--tonic N]")
    slug = args[0]
    tonic_override = None
    if "--tonic" in sys.argv:
        tonic_override = int(sys.argv[sys.argv.index("--tonic") + 1])

    song = load_song(slug, tonic_override)
    audio = REPO / "docs" / "audio" / f"{slug}.m4a"
    arr, times = nf.extract_bothchroma(audio)

    key = infer_key(np.roll(arr[:, 12:].sum(0), ROLL_TO_C))
    tname = song.name_pc(song.tonic)
    print(f"song: {slug}")
    print(f"payload key: {song.key_name} (tonic={song.tonic} mode={song.mode})   "
          f"infer_key check: {key.key_name}  conf={key.confidence:.4f}")
    if song.tonic_overridden:
        print(f"*** TONIC OVERRIDDEN to {tname} (diagnostic run, not the payload) ***")

    chords = song.chords

    # ── v7: tonic track BEFORE everything ──────────────────────────────────
    global MODE
    MODE = "minor"
    segs, tt_start = tonic_track(chords, arr, times)
    seg_desc = "  ".join(
        f"{song.name_pc(s['tonic'])}[{chords[s['i0']]['t0']:.0f}-"
        f"{chords[s['i1'] - 1]['t1']:.0f}s]" for s in segs)
    print(f"\ntonic track (v7): {seg_desc}   (payload tonic "
          f"{song.name_pc(song.tonic)}; centres = duration candidates "
          f"decided by restricted Krumhansl, v7b)")
    if len(segs) > 1:
        # v7c: FULL per-segment re-decode — colours, flags and audits in
        # each segment's local tonic and mode (previously only the
        # eligibility count was local). The plot below still shows the
        # global decode; per-segment plotting is an open item.
        total = 0
        for s in segs:
            sub = chords[s["i0"]:s["i1"]]
            loc = replace(song, tonic=s["tonic"], tonic_overridden=True)
            m_l, x3_l, _, _ = mode_audit(sub, arr, times, loc)
            MODE = m_l
            # spelling follows the segment's MODE: SHARP_TONICS is a
            # minor-key table (Db=C# sits sharp-side as C#m); major
            # sharp keys are G D A E B F#
            sharp = (s["tonic"] in SHARP_TONICS if m_l == "minor"
                     else s["tonic"] in {7, 2, 9, 4, 11, 6})
            loc = replace(loc, pcn=PC_SHARP if sharp else PC_FLAT)
            nd = _count_nondiatonic(sub, loc)
            total += nd
            spath, srows, se6, se7, sch, _slots, sform = decode_folded(
                arr, times, sub, loc)
            sflags = inflections(spath, srows, se6, se7)
            sauds = challenge_chords(sub, sch, spath, sflags, loc)
            lbl = MODE_STATE_LABEL[m_l]
            counts = "  ".join(f"{lbl[c]}={spath.count(c)}" for c in STATES)
            print(f"  segment {song.name_pc(s['tonic'])} {m_l} "
                  f"(x3={x3_l:.2f})  form {sform}")
            print(f"    colours: {counts}   non-diatonic {nd}/{len(sub)}")
            for i, own in sflags:
                ch = sub[i]
                print(f"    flag  {chord_name(ch, loc):7s} {ch['t0']:6.1f}s  "
                      f"prevailing={lbl[spath[i]]} chord says {lbl[own]}")
            for i, best, wscore, top3, kind in sauds:
                ch = sub[i]
                alts = "  ".join(f"{nm} {sc:.3f}" for sc, nm in top3)
                print(f"    {kind:9s} {chord_name(ch, loc):7s} "
                      f"{ch['t0']:6.1f}s  written={wscore:.3f} -> {alts}")
            MODE = "minor"
        print(f"  audit eligibility with LOCAL tonics: {total} "
              f"(vs {_count_nondiatonic(chords, song)} under the global tonic)")

    # ── v6: mode audit BEFORE any chord-level decision ─────────────────────
    elig_minor = _count_nondiatonic(chords, song)
    mode, x3, mass, basis = mode_audit(chords, arr, times, song)
    MODE = mode
    print(f"\nmode audit: **{mode}**  (raised-3rd share {x3:.2f}, evidence "
          f"mass {mass:.1f}, basis: {basis}; payload said {song.mode!r})")
    if mode == "major":
        print("  major-mode state labels: " +
              ", ".join(f"{k}={v}" for k, v in MODE_STATE_LABEL["major"].items()))
        print(f"  audit eligibility: {elig_minor} chords non-diatonic under "
              f"minor -> {_count_nondiatonic(chords, song)} under major")
    labels = [chord_name(c, song) for c in chords]
    n = len(chords)
    p6r, p6f = song.name_pc(song.tonic + 9), song.name_pc(song.tonic + 8)
    p7r, p7f = song.name_pc(song.tonic + 11), song.name_pc(song.tonic + 10)

    path, rows, ev6, ev7, chromas = decode(arr, times, chords, song)
    path_nogate, *_ = decode(arr, times, chords, song, gate=False)
    path_noprior, *_ = decode(arr, times, chords, song, prior=False)

    w_tot = np.array([GAIN * (t6 + t7) for (t6, _), (t7, _) in zip(ev6, ev7)])
    held = w_tot < HELD_W

    print(f"\n{n} chords   Q={Q} GAIN={GAIN} LAMBDA={LAMBDA} "
          f"trans N:{P_N_STAY}/{P_N_TO_R} R:{P_R_STAY}/{P_R_TO_N}/{P_R_TO_R}")
    print(f"{'colour':9s} {'v3':>4s} {'no-gate':>8s} {'no-prior':>9s}")
    for s in STATES:
        print(f"{s:9s} {path.count(s):4d} {path_nogate.count(s):8d} "
              f"{path_noprior.count(s):9d}")
    n_switch = sum(a != b for a, b in zip(path, path[1:]))
    print(f"switches: {n_switch}   no gated evidence (held): {held.sum()}")

    flags = inflections(path, rows, ev6, ev7)
    print(f"\ninflection flags ({len(flags)}):")
    for i, own in flags:
        print(f"  #{i:3d} {labels[i]:6s} {chords[i]['t0']:6.1f}s  "
              f"prevailing={path[i]:9s} chord says {own}")

    audits = challenge_chords(chords, chromas, path, flags, song)
    print(f"\nchord audits ({len(audits)}) — non-diatonic to every colour:")
    for i, best, wscore, top3, kind in audits:
        alts = "  ".join(f"{nm} {sc:.3f}" for sc, nm in top3)
        conf = chords[i]["lv"]["exact"]["c"]
        print(f"  {kind:9s} #{i:3d} {labels[i]:6s} {chords[i]['t0']:6.1f}s  "
              f"conf={conf:.2f}  written={wscore:.3f}  ->  {alts}")

    iv_major = [i for i in range(n)
                if not chords[i].get("nc")
                and (chords[i]["root"] - song.tonic) % 12 == 5
                and chords[i]["lv"]["exact"]["q"] in ("", "7", "^7")]
    print(f"\nIV-degree major-family chords ({song.name_pc(song.tonic + 5)}*) "
          f"— the dorian markers — prevailing / flag:")
    flag_map = dict(flags)
    for i in iv_major:
        print(f"  #{i:3d} {labels[i]:6s} {chords[i]['t0']:6.1f}s  "
              f"{path[i]:9s} / {flag_map.get(i, '-')}")

    # ── v4: structure-folded decode ─────────────────────────────────────────
    fpath, frows, fe6, fe7, fch, slotmap, form = decode_folded(arr, times, chords, song)
    n_solo = sum(1 for k in slotmap.values() if k[0] == "_solo")
    print(f"\n=== v4 structure fold ===\nform: {form}   "
          f"({n - n_solo}/{n} chords slotted)")
    print("folded colour counts: " +
          "  ".join(f"{s}={fpath.count(s)}" for s in STATES))
    fswitch = sum(a != b for a, b in zip(fpath, fpath[1:]))
    print(f"folded switches: {fswitch}")
    fflags = inflections(fpath, frows, fe6, fe7)
    print(f"folded flags ({len(fflags)}):")
    for i, own in fflags:
        print(f"  #{i:3d} {labels[i]:6s} {chords[i]['t0']:6.1f}s  "
              f"prevailing={fpath[i]:9s} chord says {own}")
    faudits = challenge_chords(chords, fch, fpath, fflags, song)
    print(f"folded audits ({len(faudits)}):")
    for i, best, wscore, top3, kind in faudits:
        alts = "  ".join(f"{nm} {sc:.3f}" for sc, nm in top3)
        conf = chords[i]["lv"]["exact"]["c"]
        print(f"  {kind:9s} #{i:3d} {labels[i]:6s} {chords[i]['t0']:6.1f}s  "
              f"conf={conf:.2f}  written={wscore:.3f}  ->  {alts}")
    print("folded IV-degree major-family chords:")
    fflag_map = dict(fflags)
    for i in iv_major:
        print(f"  #{i:3d} {labels[i]:6s} {chords[i]['t0']:6.1f}s  "
              f"{fpath[i]:9s} / {fflag_map.get(i, '-')}")
    d = [i for i in range(n) if fpath[i] != path[i]]
    print(f"fold changed {len(d)} prevailing colours; "
          f"flags {len(flags)} -> {len(fflags)}")

    # ── plot ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        3, 1, figsize=(20, 8), sharex=True, height_ratios=[0.9, 1, 1]
    )
    fig.patch.set_facecolor(SURFACE)
    x = np.arange(n)

    ax = axes[0]
    for i, s in enumerate(path):
        ax.barh(0, 1, left=i, height=1, color=STATE_COLORS[s],
                alpha=0.35 if held[i] else 0.95, linewidth=0)
    for i, own in flags:
        ax.barh(-0.42, 1, left=i, height=0.16, color=STATE_COLORS[own], linewidth=0)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    handles = [plt.Rectangle((0, 0), 1, 1, color=STATE_COLORS[s]) for s in STATES]
    ax.legend(handles, [MODE_STATE_LABEL[MODE][s] for s in STATES],
              loc="upper right", ncol=4, frameon=False, fontsize=9)
    ax.set_title(
        f"{slug} — prevailing {tname} {MODE} colour (band) + inflection flags; "
        "pale = held   [v6: mode audit before chord audit]",
        color=INK, loc="left", fontsize=13,
    )

    for ax, ev, lab in ((axes[1], ev7, f"{p7r} share of 7th degree"),
                        (axes[2], ev6, f"{p6r} share of 6th degree")):
        t = np.array([tt for tt, _ in ev])
        v = np.array([vv for _, vv in ev])
        ax.axhline(0.5, color=MUTED, lw=0.8, ls="--")
        ax.scatter(x, v, s=8 + 700 * t, c=INK2, alpha=0.75, linewidths=0)
        ax.set_ylabel(lab, color=INK2)
        ax.set_ylim(-0.05, 1.05)
    axes[1].set_title(f"7th degree: {p7r} / ({p7r} + {p7f}) — gated; "
                      "dot size = evidence mass", color=INK2, loc="left", fontsize=11)
    axes[2].set_title(f"6th degree: {p6r} / ({p6r} + {p6f}) — gated; "
                      "dot size = evidence mass", color=INK2, loc="left", fontsize=11)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=90, fontsize=6, color=INK2)
    axes[2].set_xlabel("chord (from the baked chart)", color=INK2)

    for ax in axes:
        ax.set_facecolor(SURFACE)
        if ax is not axes[0]:
            ax.grid(True, axis="y", color=GRID, lw=0.6)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.tick_params(colors=INK2)
        ax.margins(x=0.005)

    suffix = f"_tonic{song.tonic}" if song.tonic_overridden else ""
    out = REPO / "scratchpad" / f"colour_hmm_{slug}{suffix}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
