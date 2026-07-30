"""Per-chord scale-colour tracking on This Love — sticky HMM (v2).

Louis's design calls (2026-07-30): key stays C minor; track the scale
*colour* per chord; include the bass half of the NNLS chroma; sticky HMM
now, section folding later.

States = the four C-minor colours, i.e. the joint setting of the two
contrast degrees:  6th (Ab vs A) x 7th (Bb vs B).

    natural  = (Ab, Bb)      harmonic = (Ab, B)
    dorian   = (A,  Bb)      melodic  = (A,  B)

Evidence per chord span = chroma mass on the contrast pcs only (v1 showed
the full 7-pc mask dilutes the signal).  A chord with no 6th/7th-degree
mass contributes ~zero evidence and the stickiness carries the state
("hold until forced", same philosophy as local key).

Outputs
-------
scratchpad/colour_hmm_this_love.png   decoded band + per-degree evidence
stdout                                 decode summary + bass-half ablation

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

BASS_MODE = "argmax"  # "argmax": one-hot sounding-bass pc (pipeline precedent);
#                        "l1": add the whole normalised bass half (v2a — leaks
#                        the 5th partial: an F bass injects A natural);
#                        "none": treble only
BASS_W = 0.3  # mass added to the sounding-bass pc in argmax mode (1.0 in l1 mode)
Q = 0.85  # emission: expected raised-share when the state says "raised"
GAIN = 25.0  # evidence weight per unit of contrast-pc mass share
STAY = 0.92  # sticky HMM self-transition probability
HELD_W = 0.5  # below this total evidence weight, mark the chord as "held"


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
        m[int(bass.argmax())] += BASS_W
    return l1(m)


def emissions(chroma: np.ndarray, nc: bool):
    """Per-state emission log-likelihoods + per-degree evidence (t, x)."""
    if nc:
        return {s: 0.0 for s in STATES}, (0.0, 0.5), (0.0, 0.5)
    t6 = chroma[8] + chroma[9]
    t7 = chroma[10] + chroma[11]
    x6 = chroma[9] / t6 if t6 > 1e-9 else 0.5
    x7 = chroma[11] / t7 if t7 > 1e-9 else 0.5
    ll = {}
    for s, (r6, r7) in STATES.items():
        q6 = Q if r6 else 1 - Q
        q7 = Q if r7 else 1 - Q
        ll[s] = GAIN * t6 * (x6 * np.log(q6) + (1 - x6) * np.log(1 - q6)) + GAIN * t7 * (
            x7 * np.log(q7) + (1 - x7) * np.log(1 - q7)
        )
    return ll, (t6, x6), (t7, x7)


def viterbi(ll_rows: list[dict]) -> list[str]:
    names = list(STATES)
    n = len(names)
    log_stay = np.log(STAY)
    log_switch = np.log((1 - STAY) / (n - 1))
    delta = np.array([ll_rows[0][s] for s in names])
    back = []
    for row in ll_rows[1:]:
        trans = delta[:, None] + np.where(np.eye(n, dtype=bool), log_stay, log_switch)
        back.append(trans.argmax(0))
        delta = trans.max(0) + np.array([row[s] for s in names])
    path = [int(delta.argmax())]
    for bp in reversed(back):
        path.append(int(bp[path[-1]]))
    return [names[i] for i in reversed(path)]


def decode(arr, times, chords, bass_mode=None):
    rows, ev6, ev7 = [], [], []
    for ch in chords:
        m = chord_chroma(arr, times, ch["t0"], ch["t1"], bass_mode)
        ll, e6, e7 = emissions(m, bool(ch.get("nc")))
        rows.append(ll)
        ev6.append(e6)
        ev7.append(e7)
    return viterbi(rows), ev6, ev7


def main() -> None:
    arr, times = nf.extract_bothchroma(AUDIO)
    key = infer_key(np.roll(arr[:, 12:].sum(0), ROLL))
    print(f"key check: {key.key_name}  conf={key.confidence:.4f}")

    chords = load_chart_chords()
    labels = [chord_name(c) for c in chords]
    n = len(chords)

    path, ev6, ev7 = decode(arr, times, chords)  # BASS_MODE default
    path_l1, _, _ = decode(arr, times, chords, bass_mode="l1")
    path_none, _, _ = decode(arr, times, chords, bass_mode="none")

    w_tot = np.array([GAIN * (t6 + t7) for (t6, _), (t7, _) in zip(ev6, ev7)])
    held = w_tot < HELD_W

    print(f"\n{n} chords   STAY={STAY}  GAIN={GAIN}  Q={Q}  "
          f"BASS_MODE={BASS_MODE}  BASS_W={BASS_W}")
    print(f"{'colour':9s} {'argmax':>7s} {'l1':>5s} {'none':>5s}")
    for s in STATES:
        print(f"{s:9s} {path.count(s):7d} {path_l1.count(s):5d} {path_none.count(s):5d}")
    n_switch = sum(a != b for a, b in zip(path, path[1:]))
    print(f"switches: {n_switch}   held (evidence < {HELD_W}): {held.sum()}")

    # bleed diagnostic on the early chorus (Louis: F- there, ear-checked)
    print("\nearly chorus (#20-23) — where does the fake A come from?")
    for i in range(20, 24):
        ch = chords[i]
        sel = (times >= ch["t0"]) & (times < ch["t1"])
        seg = arr[sel].mean(0)
        bass = np.roll(seg[:12], ROLL)
        treb = np.roll(seg[12:], ROLL)
        bass, treb = bass / max(bass.sum(), 1e-9), treb / max(treb.sum(), 1e-9)
        top3 = np.argsort(bass)[::-1][:3]
        print(
            f"  #{i} {labels[i]:4s} treble A={treb[9]:.3f} Ab={treb[8]:.3f}   "
            f"bass A={bass[9]:.3f} Ab={bass[8]:.3f}   "
            f"bass top3: {', '.join(f'{PC_FLAT[p]} {bass[p]:.2f}' for p in top3)}"
        )

    g_chords = [i for i in range(n) if labels[i].startswith("G")]
    g_harm = sum(path[i] in ("harmonic", "melodic") for i in g_chords)
    print(f"\nG-root chords decoded with raised 7th (B): {g_harm}/{len(g_chords)}")

    # ear GT (Louis 2026-07-30): B section is F- (natural) except its last
    # 4 bars, which have F major (dorian). Where does the chart write F major?
    fmaj = [i for i in range(n)
            if chords[i].get("root") == 5 and not chords[i].get("nc")
            and chords[i]["lv"]["exact"]["q"] in ("", "7", "^7")]
    print("\nchart F-major chords and their decoded colour:")
    for i in fmaj:
        print(f"  #{i:3d} {labels[i]:5s} {chords[i]['t0']:6.1f}s  -> {path[i]}")

    # ── plot ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        3, 1, figsize=(20, 8), sharex=True, height_ratios=[0.9, 1, 1]
    )
    fig.patch.set_facecolor(SURFACE)
    x = np.arange(n)

    ax = axes[0]
    for i, s in enumerate(path):
        ax.barh(
            0, 1, left=i, height=1, color=STATE_COLORS[s],
            alpha=0.35 if held[i] else 0.95, linewidth=0,
        )
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    handles = [plt.Rectangle((0, 0), 1, 1, color=STATE_COLORS[s]) for s in STATES]
    ax.legend(
        handles, list(STATES), loc="upper right", ncol=4, frameon=False, fontsize=9,
    )
    ax.set_title(
        f"This Love — per-chord scale colour, sticky HMM (STAY={STAY}); "
        "pale = held on stickiness, not evidence",
        color=INK, loc="left", fontsize=13,
    )

    ax = axes[1]
    t7 = np.array([t for t, _ in ev7])
    x7 = np.array([v for _, v in ev7])
    ax.axhline(0.5, color=MUTED, lw=0.8, ls="--")
    ax.scatter(x, x7, s=8 + 700 * t7, c=INK2, alpha=0.75, linewidths=0)
    ax.set_ylabel("B share of 7th degree", color=INK2)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("7th degree: B / (B + Bb) — dot size = evidence mass", color=INK2,
                 loc="left", fontsize=11)

    ax = axes[2]
    t6 = np.array([t for t, _ in ev6])
    x6 = np.array([v for _, v in ev6])
    ax.axhline(0.5, color=MUTED, lw=0.8, ls="--")
    ax.scatter(x, x6, s=8 + 700 * t6, c=INK2, alpha=0.75, linewidths=0)
    ax.set_ylabel("A share of 6th degree", color=INK2)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("6th degree: A / (A + Ab) — dot size = evidence mass", color=INK2,
                 loc="left", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=6, color=INK2)
    ax.set_xlabel("chord (from the baked chart)", color=INK2)

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
