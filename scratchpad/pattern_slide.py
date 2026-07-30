#!/usr/bin/env python3
"""pattern_slide.py — Louis's pattern-slide segmentation, 2026-07-30.

NOT Foote novelty. The difference matters:

  * Foote (scratchpad/ssm_block_segment.py, harmonia/models/ssm_block_sections.py)
    slides a checkerboard kernel ALONG THE DIAGONAL and asks "does the texture
    change here?" -> detects BOUNDARIES.
  * This module pins the rows to one block and slides ACROSS THE X AXIS only,
    asking "where else in the song does THIS pattern come back?" -> detects
    OCCURRENCES, with exact start slots, for free, over the whole song.

Granularity is 2 slots per bar (half-bar), which is what lets a 2-chords-per-bar
chorus show its true 2-bar loop while a 1-chord-per-bar verse shows a 4-bar one.

The algorithm (Louis, 2026-07-30):
  1. Build the half-bar SSM on the RIGID grid.
  2. At the current block start t0, find the MINIMAL REPEATING PATTERN length d:
     walk right along row t0, skip the held run, take the distance to the next
     high cell. (Verified against a diagonal-band period score; disagreements
     are logged, never silently resolved.)
  3. Cut the d x d square K = S[t0:t0+d, t0:t0+d] off the diagonal.
  4. Slide K horizontally: corr(t) = <K, S[t0:t0+d, t:t+d]> / <K, K>.
     corr(t0) == 1 by construction -> that is the ceiling, as Louis predicted.
  5. Peaks of corr = every occurrence of the pattern. Split strong-vs-medium by
     the largest gap in the sorted peak values (no hand-tuned threshold).
  6. Walk the expected grid t0, t0+d, t0+2d, ... while corr stays strong -> the
     block repeats that many times; the first slot that drops out is the next
     block's start. Recurse from there.
  7. A new block inherits an earlier block's letter when the earlier block's own
     corr curve peaks at the new block's start.

Assumes (per Louis, for now): a correct grid and a correct first chord.

Run:  .venv/bin/python scratchpad/pattern_slide.py [song-substring]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scratchpad"))

from harmonia.models.section_structure import build_chord_ssm
from section_merge_declined import _load_payload, _PC

HI = 0.9            # an SSM cell this high counts as "same chord"
SLOTS_PER_BAR = 2   # half-bar granularity

# ── thresholds, MEASURED not guessed ─────────────────────────────────────────
# Calibrated 2026-07-30 by scoring the slide against spans whose identity is known
# from docs/this_love_target_spec.md (Louis's lead sheet), on the `chord_ssm` scale:
#
#   same section, clean repeat        0.893 - 1.000
#   same section, a chord mis-decoded 0.763 - 0.843
#   DIFFERENT section                 0.168 - 0.697   <- max is chorus vs bridge,
#                                                        which genuinely share Fm/Eb
#   1st/2nd ending                    whole 0.51-0.71, but opening 0.82-1.00
#                                                          and tail    0.20-0.41
#
# So there is a real empty band between 0.697 and 0.763, and CONTINUE sits in it.
# An ending is NEVER identified by its whole-window score (0.71 overlaps the
# different-section range) — only by the opening/tail SPLIT.
PERIOD_MARGIN = 0.30  # accept the SMALLEST period scoring within this of the best
PERIOD_FLOOR = 0.75   # ...but never below this. This Love's bridge tops out at 0.706
                      # (its two halves diverge at the tail), so it correctly finds
                      # NO period and falls through to the ending test.
REP_STRICT = 0.85     # a clean repeat of the pattern
CONTINUE = 0.75       # ...still the same section, with a chord mis-decoded
ENDING_PREFIX = 0.75  # shared-opening match that makes a variant an ENDING, not a section
LABEL_CUT = 0.75      # a block inherits an earlier block's letter at this match

# coarse quality families (same convention as ssm_block_segment._QMAP)
_QMAP = {"": 0, "6": 0, "^7": 0, "maj7": 0, "add9": 0,
         "-": 1, "-7": 1, "-6": 1, "-^7": 1, "m": 1, "min7": 1,
         "7": 2, "9": 2, "13": 2, "7b9": 2,
         "h7": 3, "-7b5": 3,
         "o": 4, "o7": 4}


# ── step 1: rigid grid -> half-bar slots ──────────────────────────────────────

def rigid_slots(P: dict, bpb: int = 4):
    """Regrid the payload onto the rigid bar grid, then lay it out at
    SLOTS_PER_BAR slots/bar with forward-fill (a held chord occupies its slots).

    Returns ``(seq, names, n_bars, bar_sec)``; ``seq`` is the
    ``(root_rel_to_tonic, quality_family)`` list build_chord_ssm wants.
    """
    from harmonia.models.rigid_grid import rigid_grid_for, apply_rigid_grid
    tonic = int((P.get("home") or {}).get("tonic", 0)) % 12
    grid = rigid_grid_for(P.get("chords", []), tonic_pc=tonic)
    if grid is None:
        raise SystemExit("rigid_grid_for deferred (returned None) — no grid to stand on")
    chords, n_bars = apply_rigid_grid(P.get("chords", []), grid,
                                     beats_per_bar=bpb, drop_before_grid=True)
    bar_sec = float(np.median(np.diff(grid)))

    n = n_bars * SLOTS_PER_BAR
    slot: list[dict | None] = [None] * n
    for c in sorted(chords, key=lambda c: (c["bar"], c.get("beat", 0))):
        b, beat = c.get("bar", 0), c.get("beat", 0)
        if not 0 <= b < n_bars:
            continue
        s = b * SLOTS_PER_BAR + min(int(beat / (bpb / SLOTS_PER_BAR)), SLOTS_PER_BAR - 1)
        if slot[s] is None:          # earliest chord owns the slot
            slot[s] = c
    cur = None
    for s in range(n):               # forward-fill: a held chord is still sounding
        if slot[s] is not None:
            cur = slot[s]
        else:
            slot[s] = cur

    def _q(c):
        return _QMAP.get(((c.get("lv") or {}).get("exact") or {}).get("q", ""), 0)

    # ``names`` is what the SSM is built from (see `chord_ssm`). ``seq`` carries the
    # tonic-relative root purely so `minimal_period` can reject a candidate window
    # that holds only ONE distinct chord — it is not used for similarity.
    seq = [(((c["root"] - tonic) % 12), 0) if c else (-1, 0) for c in slot]
    names = [_PC[c["root"] % 12] + (((c.get("lv") or {}).get("exact") or {}).get("q", ""))
             if c else "·" for c in slot]
    return seq, names, n_bars, bar_sec


def chord_ssm(names: list[str]) -> np.ndarray:
    """THE SSM for this project (Louis, 2026-07-30, standing rule): cosine on
    weighted CHORD-TONE content, so **Bb is closer to Gm than to F**.

    Chords become weighted pitch-class vectors via
    ``harmonia.theory.local_key.chord_pcs`` (root 2.0 / 3rd 1.5 / 5th 1.0 / 7th 0.8)
    and two slots' similarity is the cosine between them. Verified on This Love:

        Bb <-> Gm  0.701     <- relatives are CLOSE (they share Bb and D)
        Bb <-> F   0.276     <- a fifth apart is FAR (they share only F)
        G  <-> G7  0.959     } the same chord with its quality wobbling by the
        Do <-> Dh7 0.959     } decoder still reads as the same chord
        G  <-> Cm  0.264     <- verse and chorus openings stay far apart
        Cm <-> Bb  0.212

    Two alternatives were tried and are WRONG here, recorded so they are not
    retried: ``build_chord_ssm`` (root one-hot | quality one-hot, cosine) scores a
    root-match/quality-mismatch at only 0.5, which dragged This Love's true 4-bar
    verse period down to 0.69 and made it undetectable; forcing its quality slot
    constant to get "root only" leaves a 0.5 FLOOR under every cell, halving the
    dynamic range of every curve (error pattern #1 — a silent scale bug that still
    produced plausible numbers). This function is a true 0..1 scale and every
    threshold in this module is calibrated against it.
    """
    from harmonia.theory.local_key import chord_pcs
    n = len(names)
    F = np.zeros((n, 12))
    for i, nm in enumerate(names):
        if nm == "·":
            continue
        for pc, w in chord_pcs(nm).items():
            F[i, pc % 12] += w
    F /= np.clip(np.linalg.norm(F, axis=1, keepdims=True), 1e-9, None)
    return np.clip(F @ F.T, 0.0, 1.0).astype(np.float32)


# ── step 2: the minimal repeating pattern at a block start ────────────────────

def minimal_period(S: np.ndarray, seq: list[tuple], t0: int, *, min_d: int = 2,
                   max_d: int | None = None, log=print):
    """Louis's rule, literally: from slot t0 walk right along ROW t0, skip the
    contiguous high run (the chord being held), then the distance to the next
    high cell is the minimal repeating pattern.

    Cross-checked against a diagonal-band period score
    ``band(d) = mean_i S[t0+i, t0+i+d]`` over the pattern's own span — a true
    period scores ~1. Both are reported; a disagreement is logged, not hidden.
    """
    n = S.shape[0]
    max_d = max_d or min(n - t0, 64)

    j = t0 + 1
    while j < n and S[t0, j] >= HI:          # the held run
        j += 1
    held = j - t0
    while j < n and S[t0, j] < HI:           # the next occurrence of this chord
        j += 1
    d_row = (j - t0) if j < n else None

    # Candidate loop lengths must be:
    #  * a WHOLE number of BARS — a pop loop is; odd slot counts gave junk 3-slot
    #    "patterns" like `C- F^7 Ab G7 G7 G7`;
    #  * LONGER than the held run — a chord held h slots trivially self-matches at
    #    every lag <= h, which is how `C C` and `G7 G7` got accepted as 1-bar
    #    "patterns" with a perfect band score of 1.00;
    #  * at least 2 DISTINCT chords wide — one chord is not a pattern.
    band, rejected = {}, []
    lo = max(min_d, SLOTS_PER_BAR, (held // SLOTS_PER_BAR + 1) * SLOTS_PER_BAR)
    for d in range(lo, max_d + 1, SLOTS_PER_BAR):
        if t0 + 2 * d > n:
            break
        if len({r for r, _ in seq[t0:t0 + d]}) < 2:
            rejected.append(d)
            continue
        band[d] = float(np.mean([S[t0 + i, t0 + i + d] for i in range(d)]))

    # A MULTIPLE of the true period always scores at least as well as the period
    # itself, so an absolute threshold picks the wrong (longer) one — and a strict
    # threshold rejects every period on a song with any decode wobble. Instead:
    # take the SMALLEST d that scores within PERIOD_MARGIN of the best. That is
    # "minimal repeating pattern" stated as a decision rule.
    d_band, best, cut = None, 0.0, 0.0
    if band:
        best = max(band.values())
        cut = max(PERIOD_FLOOR, best - PERIOD_MARGIN)
        d_band = next((d for d in sorted(band) if band[d] >= cut), None)

    log(f"    row-walk: chord held {held} slot(s), next same chord at +{d_row} "
        f"-> d_row={d_row}"
        + (f"   [d<={held} rejected as held-run; d={rejected} rejected as single-chord]"
           if rejected else ""))
    if band:
        top = sorted(band.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
        log(f"    band scores (best 5): " + ", ".join(f"{d}:{v:.2f}" for d, v in top)
            + f"   best={best:.2f} cut={cut:.2f} -> d_band={d_band}")
    if d_band is None:
        # No lag reproduces the span. That is the SIGNATURE of a 1st/2nd-ending
        # section (This Love's bridge: `Fm Eb G7 Cm` then `Fm Eb G G7` — the two
        # halves share their opening and diverge at the tail, so the lag-8 band
        # scores only ~0.5 and no period passes). Fall back to the row-walk length
        # and let the ENDING test downstream recognise the tail-only divergence.
        log(f"    no lag reproduces this span (best {best:.2f} < cut {cut:.2f}) — the "
            f"signature of a 1st/2nd-ending section; falling back to d_row={d_row}")
    elif d_row != d_band:
        log(f"    (row-walk said {d_row}, band says {d_band} — trusting the band: "
            f"one cell can match by accident, a whole diagonal cannot)")
    return d_band or d_row, {"d_row": d_row, "d_band": d_band, "held": held,
                             "band": band}


# ── step 3+4: cut the square, slide it across X ───────────────────────────────

def slide_across_x(S: np.ndarray, t0: int, d: int):
    """Pin the ROWS to [t0, t0+d) and slide the COLUMNS — only the column index
    moves. That is the whole difference from Foote, where both indices move
    together down the diagonal.

    Returns ``(diag, square)``, two readings of the same slid window:

    * ``square[t]`` = <K, S[rows, t:t+d]> / <K, K>, the full d x d dot product
      Louis described. It is PHASE-INVARIANT, and that turns out to be the
      problem: it scores This Love's chorus 0.81 against the verse pattern,
      because both regions are internally dense in the same way. It measures
      "same kind of texture", not "same chords".
    * ``diag[t]``  = mean_i S[t0+i, t+i], the DIAGONAL of the slid stripe. For an
      exact repeat every term is 1. It is phase-LOCKED, so it answers the actual
      question — "do these d slots reproduce my pattern, slot for slot?" — and it
      separates verse from chorus cleanly (0.125 where the square said 0.81).
      This is the visible off-diagonal STRIPE in the SSM picture.

    ``diag`` drives every decision; ``square`` is kept and printed so the
    difference stays inspectable.
    """
    n = S.shape[0]
    K = S[t0:t0 + d, t0:t0 + d]
    denom = float((K * K).sum()) or 1.0
    diag = np.full(n, np.nan)
    square = np.full(n, np.nan)
    for t in range(0, n - d + 1):
        W = S[t0:t0 + d, t:t + d]
        diag[t] = float(np.mean([S[t0 + i, t + i] for i in range(d)]))
        square[t] = float((K * W).sum()) / denom
    return diag, square


# ── step 5: peaks, and the strong/medium divide ───────────────────────────────

def peaks_of(corr: np.ndarray, d: int, floor: float = 0.5) -> list[int]:
    """Local maxima of the slide curve, non-max-suppressed by half a pattern."""
    v = np.nan_to_num(corr, nan=-1.0)
    n = len(v)
    cand = [i for i in range(n)
            if v[i] >= floor
            and v[i] >= (v[i - 1] if i else -1)
            and v[i] >= (v[i + 1] if i + 1 < n else -1)]
    cand.sort(key=lambda i: -v[i])
    kept: list[int] = []
    for i in cand:
        if all(abs(i - k) >= max(2, d // 2) for k in kept):
            kept.append(i)
    return sorted(kept)


def strong_medium_split(vals: list[float], min_gap: float = 0.08):
    """Largest-gap split of the sorted peak heights — Louis's "is there a clear
    divide between strong and medium peaks?". Returns (threshold, gap). A gap
    below ``min_gap`` means the peaks form one population: no split, keep all."""
    if len(vals) < 2:
        return (0.0, 0.0)
    s = sorted(vals, reverse=True)
    gaps = [(s[i] - s[i + 1], i) for i in range(len(s) - 1)]
    gap, i = max(gaps)
    if gap < min_gap:
        return (0.0, gap)
    return ((s[i] + s[i + 1]) / 2.0, gap)


# ── step 6+7: walk the blocks ─────────────────────────────────────────────────

def segment(S: np.ndarray, seq: list[tuple], names: list[str], log=print):
    """Walk left to right: find the pattern, slide it, count how many times it
    repeats contiguously, jump to where it stops, repeat. Then letter the blocks
    using the slide curves we already have."""
    n = S.shape[0]
    blocks: list[dict] = []
    known: list[dict] = []      # canonical patterns already learned, in order
    letters = "ABCDEFGH"
    t0 = 0
    guard = 0
    while t0 < n - 1 and guard < 64:
        guard += 1
        log(f"\n  block @slot {t0} (bar {t0/SLOTS_PER_BAR:g}), first chord {names[t0]}")

        # ── Do I already know this pattern? ──────────────────────────────────
        # Louis's original framing: "compare each new block to the ones I've
        # already seen". Ask FIRST, before deriving anything. Deriving a fresh
        # period at a spot where a known section merely plays ONCE is what
        # produced the bogus 16-bar "pattern" that swallowed the C bridge: the
        # single A at bar 44 has no internal repeat, so the period search reached
        # forward into the bridge and scored a meaningless 0.72.
        adopt = next((k for k in known
                      if not np.isnan(k["corr"][t0]) and k["corr"][t0] >= REP_STRICT),
                     None)
        if adopt is not None:
            d, corr, sq, row0 = adopt["d"], adopt["corr"], adopt["sq"], adopt["row0"]
            label, strong = adopt["label"], adopt["strong"]
            log(f"    already known: this is {label} (matches the {label} pattern "
                f"learned at bar {row0//SLOTS_PER_BAR} with diag={corr[t0]:.2f}) — "
                f"reusing its {d//SLOTS_PER_BAR}-bar pattern, no new period derived")
        else:
            d, info = minimal_period(S, seq, t0, log=log)
            if not d:
                log("    no period found — absorbing the tail into the previous block")
                if blocks:
                    blocks[-1]["end"] = n
                break
            corr, sq = slide_across_x(S, t0, d)
            row0, label = t0, None
            pk = peaks_of(corr, d)
            thresh, gap = strong_medium_split([float(corr[p]) for p in pk])
            strong = [p for p in pk if corr[p] >= REP_STRICT]
            log(f"    pattern = {d} slots = {d/SLOTS_PER_BAR:g} bars: "
                f"{' '.join(names[t0:t0+d])}")
            log(f"    peaks (slot:diag/square): "
                + ", ".join(f"{p}:{corr[p]:.2f}/{sq[p]:.2f}" for p in pk))
            log(f"    peak distribution: largest gap {gap:.2f} at {thresh:.2f}"
                + ("  [one population]" if thresh == 0 else "")
                + f";  ALL occurrences of this pattern in the song (>= {REP_STRICT}): "
                + f"slots {strong} = bars {[p//SLOTS_PER_BAR for p in strong]}")

        # ── walk the expected grid t0, t0+d, t0+2d, ... ──────────────────────
        # Three outcomes per slot, in priority order:
        #   >= REP_STRICT           -> a clean repeat, keep going
        #   shared prefix, new tail -> a 1st/2nd ENDING of THIS section, absorb+stop
        #   >= CONTINUE             -> same section, chord decode wobbled, keep going
        #   else                    -> a genuinely different pattern, section ends
        reps, k, ending = 0, 0, None
        while True:
            t = t0 + k * d
            if t + d > n:
                if t < n and reps:
                    log(f"      grid slot {t}: truncated tail -> counted")
                    reps += 1
                break
            c = float(corr[t])
            if c >= REP_STRICT:
                log(f"      grid slot {t} (bar {t//SLOTS_PER_BAR}): diag={c:.2f} REPEAT")
                reps, k = reps + 1, k + 1
                continue
            # 1st/2nd ENDING: This Love's chorus is `Cm Fm | Bb Eb` x3 then
            # `C F | Ab G` — same opening, different tail. Same shared-prefix /
            # tail-only-divergence test the shipped `_detect_endings` uses.
            pre = max(1, d // 2)
            pfx = float(np.mean([S[row0 + i, t + i] for i in range(pre)]))
            tail = float(np.mean([S[row0 + i, t + i] for i in range(pre, d)])) if d > pre else 0.0
            if reps and pfx >= ENDING_PREFIX and tail < REP_STRICT:
                log(f"      grid slot {t} (bar {t//SLOTS_PER_BAR}): diag={c:.2f} -> "
                    f"1st/2nd ENDING of this section (opening matches {pfx:.2f}, "
                    f"tail differs {tail:.2f}) — absorbing, section ends after it")
                ending = (t, t + d)
                k += 1
                break
            if reps and c >= CONTINUE:
                log(f"      grid slot {t} (bar {t//SLOTS_PER_BAR}): diag={c:.2f} -> "
                    f"below {REP_STRICT} but above {CONTINUE}: same section, chord "
                    f"decode wobbled — keep going")
                reps, k = reps + 1, k + 1
                continue
            log(f"      grid slot {t} (bar {t//SLOTS_PER_BAR}): diag={c:.2f} <- a "
                f"different pattern, section ends here")
            break
        reps = max(reps, 1)
        end = min(t0 + max(k, reps) * d, n)

        # a genuinely new pattern earns the next letter and is remembered
        if label is None:
            weak = next((k for k in known
                         if not np.isnan(k["corr"][t0]) and k["corr"][t0] >= LABEL_CUT),
                        None)
            if weak is not None:
                label = weak["label"]
                log(f"    -> letter {label} (weaker match {weak['corr'][t0]:.2f} to the "
                    f"{label} learned at bar {weak['row0']//SLOTS_PER_BAR})")
            else:
                label = letters[len(known)]
                log(f"    -> NEW pattern, letter {label}")
                known.append({"label": label, "d": d, "corr": corr, "sq": sq,
                              "row0": t0, "strong": strong})

        blocks.append({"start": t0, "end": end, "d": d, "reps": reps, "passes": 1,
                       "corr": corr, "square": sq, "peaks": strong, "label": label,
                       "ending": ending, "row0": row0,
                       "pattern": names[row0:row0 + d]})
        t0 = end

    # ── merge adjacent blocks of the same section ────────────────────────────
    # Two flavours, both "this was never a section change":
    #  * identical shape (same letter, same loop length, same rep count, both with
    #    or both without an ending) -> another PASS of the same section. This is
    #    what turns the three 8-bar choruses at the end into B x3.
    #  * same letter but one pass got split by a mis-decoded chord -> extend.
    merged: list[dict] = []
    for b in blocks:
        p = merged[-1] if merged else None
        if p and p["label"] == b["label"] and p["end"] == b["start"] and p["d"] == b["d"]:
            same_shape = (p["reps"] == b["reps"]
                          and (p["ending"] is not None) == (b["ending"] is not None))
            if same_shape:
                log(f"  merge: {b['label']} @bar {b['start']//SLOTS_PER_BAR} is another "
                    f"PASS of the {b['label']} at bar {p['start']//SLOTS_PER_BAR} "
                    f"(same {b['d']//SLOTS_PER_BAR}-bar loop x{b['reps']}"
                    f"{' + ending' if b['ending'] else ''}) -> pass "
                    f"{p['passes'] + 1}")
                p["passes"] += 1
            else:
                log(f"  merge: {b['label']} @bar {b['start']//SLOTS_PER_BAR} continues the "
                    f"{b['label']} at bar {p['start']//SLOTS_PER_BAR} (same letter, "
                    f"adjacent — a mis-decoded repetition split it)")
                p["reps"] += b["reps"]
                p["ending"] = p["ending"] or b["ending"]
            p["end"] = b["end"]
        else:
            merged.append(b)

    # The displayed UNIT is one pass: the loop reps plus the 1st/2nd ending when
    # there is one, else the bare loop. `xN` counts units, which is why This Love
    # reads `A x4` (a 4-bar loop four times) but `B x3` (an 8-bar chorus, itself
    # loop x3 + ending, three times).
    for b in merged:
        b["mult"] = b["passes"] if b["ending"] else b["reps"]
    return merged


def main():
    sub = sys.argv[1] if len(sys.argv) > 1 else "this_love"
    f = next(Path(REPO / "docs" / "plots").glob(f"inferred_*{sub}*.html"))
    slug = f.stem.replace("inferred_", "")
    P = _load_payload(f)
    seq, names, n_bars, bar_sec = rigid_slots(P)
    S = chord_ssm(names)

    print(f"\n=== {slug} ===")
    print(f"rigid grid: {n_bars} bars @ {bar_sec:.3f}s  "
          f"-> {len(seq)} half-bar slots, SSM {S.shape}")
    print(f"first chord: {names[0]}  (assumed correct, per the brief)")
    print("\nslots (2 per bar):")
    for i in range(0, len(names), 16):
        print(f"  slot {i:3d} (bar {i//2:2d}): " + " ".join(f"{x:>5s}" for x in names[i:i + 16]))

    print("\n--- pattern-slide walk ---")
    blocks = segment(S, seq, names)

    print("\n--- result ---")
    for b in blocks:
        end = f" + 1st/2nd ending (bars {b['ending'][0]//2}-{(b['ending'][1]-1)//2})" \
            if b["ending"] else ""
        passes = f"   [played x{b['passes']}]" if b["passes"] > 1 else ""
        print(f"  {b['label']}  bars {b['start']//2:2d}-{(b['end']-1)//2:2d}  "
              f"{b['d']//2}-bar loop x{b['reps']}{end}{passes}")
        print(f"        {' '.join(b['pattern'])}")
    form = " ".join(f"{b['label']}x{b['mult']}" if b["mult"] > 1 else b["label"]
                    for b in blocks)
    print(f"\n  form: {form}")
    return S, names, blocks, slug, bar_sec


if __name__ == "__main__":
    main()
