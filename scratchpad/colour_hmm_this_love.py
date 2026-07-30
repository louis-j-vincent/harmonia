"""Per-chord scale-colour tracking on This Love — sticky HMM (v3).

Louis's design calls (2026-07-30): key stays C minor; track the scale
*colour* per chord; bass enters as the sounding-bass note (argmax) only.

v3 adds, in Louis's priority order:
  Fix 2 — chord-gated evidence: a degree's A/Ab (or B/Bb) chroma is only
    counted when the current chord actually VOICES that degree (F chords
    voice a 6th, G/Bb/Eb chords voice a 7th, ...). On a plain C- triad,
    any A natural is overtones or melody, not the scale.
  Fix 1 — relax-to-natural prior: raised notes are guests that belong to
    the chord that brings them. Each raised degree costs LAMBDA per chord
    (emission prior), and leaving a raised state toward natural is cheaper
    than acquiring one — so with no evidence the colour decays home
    instead of freezing (kills the melodic latch after Ab->G7).
  Fix 3 — inflection layer: the sticky track gives the PREVAILING colour;
    single chords whose own gated evidence clearly contradicts it get a
    borrowed-colour flag (like an accidental on a chart) instead of
    forcing the whole track to flip.

States = the four C-minor colours = joint setting of the contrast degrees:
natural (Ab,Bb), harmonic (Ab,B), dorian (A,Bb), melodic (A,B).

Outputs
-------
scratchpad/colour_hmm_this_love.png   decoded band + per-degree evidence
stdout                                 decode summary + ablations + flags

Usage:  .venv/bin/python scratchpad/colour_hmm_this_love.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from harmonia.models import nnls_features as nf  # noqa: E402
from harmonia.theory.key_profiles import infer_key  # noqa: E402

SLUG = "maroon_5_this_love"
AUDIO = REPO / "docs" / "audio" / f"{SLUG}.m4a"

PC_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
ROLL = 9  # A-first -> C-first (nnls_features._ROLL_TO_C)

# state -> (raised 6th?, raised 7th?)
STATES = {
    "natural": (0, 0),
    "harmonic": (0, 1),
    "dorian": (1, 0),
    "melodic": (1, 1),
}
STATE_COLORS = {  # same semantics as v1 plot
    "natural": "#2a78d6",
    "harmonic": "#eb6834",
    "melodic": "#1baf7a",
    "dorian": "#eda100",
}
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"

BASS_MODE = "argmax"  # sounding-bass one-hot; "l1" leaks partials (v2 bug), "none"
BASS_W = 0.3  # mass added to the sounding-bass pc in argmax mode
BASS_MARGIN = 1.5  # trust the bass argmax only if top pc >= this x runner-up
#                    (on This Love's F- the bass half reads A/E/F almost tied
#                    and argmax picks A — the v2 fake-A bug through a pinhole)
Q = 0.85  # emission: expected raised-share when the state says "raised"
GAIN = 25.0  # evidence weight per unit of contrast-pc mass share
HELD_W = 0.5  # below this total evidence weight, mark the chord as "held"

LAMBDA = 0.25  # per-chord prior cost of each raised degree (fix 1)
# asymmetric stickiness (fix 1): relaxing to natural is cheaper than
# acquiring a raised degree
P_N_STAY, P_N_TO_R = 0.94, 0.02  # from natural
P_R_STAY, P_R_TO_N, P_R_TO_R = 0.85, 0.12, 0.015  # from a raised state

# chord-quality templates, intervals from the root (chart vocabulary)
QUALITY_PCS = {
    "": (0, 4, 7),
    "-": (0, 3, 7),
    "7": (0, 4, 7, 10),
    "-7": (0, 3, 7, 10),
    "^7": (0, 4, 7, 11),
    "h7": (0, 3, 6, 10),
    "o": (0, 3, 6),
}

# inflection thresholds (fix 3)
INFL_MIN_W = 1.0  # min gated evidence weight on the disagreeing degree
INFL_MIN_MARGIN = 0.15  # min |raised share - 0.5|


def load_chart_chords() -> list[dict]:
    txt = (REPO / "docs/plots" / f"inferred_{SLUG}.html").read_text(errors="ignore")
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
    return json.loads(txt[i : j + 1])["chords"]


def chord_name(ch: dict) -> str:
    if ch.get("nc"):
        return "NC"
    name = PC_FLAT[ch["root"]] + ch["lv"]["exact"]["q"]
    if ch.get("bass", -1) >= 0 and ch["bass"] != ch["root"]:
        name += "/" + PC_FLAT[ch["bass"]]
    return name


def chord_gates(ch: dict) -> tuple[bool, bool]:
    """(voices a 6th degree?, voices a 7th degree?) for this chord symbol."""
    if ch.get("nc"):
        return False, False
    q = ch["lv"]["exact"]["q"]
    pcs = {(ch["root"] + iv) % 12 for iv in QUALITY_PCS.get(q, (0, 4, 7))}
    return bool(pcs & {8, 9}), bool(pcs & {10, 11})


def chord_chroma(arr, times, t0, t1, bass_mode=None) -> np.ndarray:
    """L1-normalised C-first 12-d chroma for one chord span."""
    bass_mode = BASS_MODE if bass_mode is None else bass_mode
    sel = (times >= t0) & (times < t1)
    if not sel.any():
        sel = np.array([np.argmin(np.abs(times - 0.5 * (t0 + t1)))])
    seg = arr[sel].mean(0)
    bass = np.roll(seg[:12], ROLL)
    treb = np.roll(seg[12:], ROLL)

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


def emissions(chroma, gates, *, prior=True):
    """Per-state emission LLs + per-degree evidence (t, x); gated degrees only."""
    g6, g7 = gates
    t6 = chroma[8] + chroma[9] if g6 else 0.0
    t7 = chroma[10] + chroma[11] if g7 else 0.0
    x6 = chroma[9] / t6 if t6 > 1e-9 else 0.5
    x7 = chroma[11] / t7 if t7 > 1e-9 else 0.5
    ll = {}
    for s, (r6, r7) in STATES.items():
        q6 = Q if r6 else 1 - Q
        q7 = Q if r7 else 1 - Q
        v = GAIN * t6 * (x6 * np.log(q6) + (1 - x6) * np.log(1 - q6))
        v += GAIN * t7 * (x7 * np.log(q7) + (1 - x7) * np.log(1 - q7))
        if prior:
            v -= LAMBDA * (r6 + r7)
        ll[s] = v
    return ll, (t6, x6), (t7, x7)


def _log_trans() -> np.ndarray:
    names = list(STATES)
    T = np.empty((len(names), len(names)))
    for i, si in enumerate(names):
        for j, sj in enumerate(names):
            if si == "natural":
                p = P_N_STAY if i == j else P_N_TO_R
            else:
                p = P_R_STAY if i == j else (P_R_TO_N if sj == "natural" else P_R_TO_R)
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


def decode(arr, times, chords, bass_mode=None, *, gate=True, prior=True):
    rows, ev6, ev7 = [], [], []
    for ch in chords:
        m = chord_chroma(arr, times, ch["t0"], ch["t1"], bass_mode)
        gates = chord_gates(ch) if gate else ((not ch.get("nc"),) * 2)
        ll, e6, e7 = emissions(m, gates, prior=prior)
        rows.append(ll)
        ev6.append(e6)
        ev7.append(e7)
    return viterbi(rows), rows, ev6, ev7


def inflections(path, rows, ev6, ev7):
    """Chords whose own gated evidence clearly contradicts the prevailing
    colour -> (index, borrowed colour) flags (fix 3)."""
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


def main() -> None:
    arr, times = nf.extract_bothchroma(AUDIO)
    key = infer_key(np.roll(arr[:, 12:].sum(0), ROLL))
    print(f"key check: {key.key_name}  conf={key.confidence:.4f}")

    chords = load_chart_chords()
    labels = [chord_name(c) for c in chords]
    n = len(chords)

    path, rows, ev6, ev7 = decode(arr, times, chords)
    path_nogate, *_ = decode(arr, times, chords, gate=False)
    path_noprior, *_ = decode(arr, times, chords, prior=False)

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

    fmaj = [i for i in range(n)
            if chords[i].get("root") == 5 and not chords[i].get("nc")
            and chords[i]["lv"]["exact"]["q"] in ("", "7", "^7")]
    print("\nchart F-major chords — prevailing / flag:")
    flag_map = dict(flags)
    for i in fmaj:
        print(f"  #{i:3d} {labels[i]:5s} {chords[i]['t0']:6.1f}s  "
              f"{path[i]:9s} / {flag_map.get(i, '-')}")

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
    ax.legend(handles, list(STATES), loc="upper right", ncol=4, frameon=False,
              fontsize=9)
    ax.set_title(
        "This Love — prevailing colour (band) + inflection flags (bottom ticks); "
        "pale = held, not evidence   [v3: chord-gated evidence, relax-to-natural]",
        color=INK, loc="left", fontsize=13,
    )

    for ax, ev, lab in ((axes[1], ev7, "B share of 7th degree"),
                        (axes[2], ev6, "A share of 6th degree")):
        t = np.array([tt for tt, _ in ev])
        v = np.array([vv for _, vv in ev])
        ax.axhline(0.5, color=MUTED, lw=0.8, ls="--")
        ax.scatter(x, v, s=8 + 700 * t, c=INK2, alpha=0.75, linewidths=0)
        ax.set_ylabel(lab, color=INK2)
        ax.set_ylim(-0.05, 1.05)
    axes[1].set_title("7th degree: B / (B + Bb) — gated; dot size = evidence mass",
                      color=INK2, loc="left", fontsize=11)
    axes[2].set_title("6th degree: A / (A + Ab) — gated; dot size = evidence mass",
                      color=INK2, loc="left", fontsize=11)
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

    out = REPO / "scratchpad" / "colour_hmm_this_love.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
