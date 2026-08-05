"""THE METHOD — sections from a repetition dictionary. Locked 2026-08-05.

Moved here from `scripts/harmonic_method.py` + `scripts/dictionary_harmonic.py`
+ `scripts/sections_from_dict.py` on 2026-08-05 when Louis put it in production
(« push en prod et remplacer la pipeline actuelle par cette version améliorée »).
Those three scripts now RE-EXPORT from this module — this file is the single
definition. The method had already drifted twice while spread across five
scripts (a margin using MAD instead of σ, a peak floor taken against the curve
maximum instead of the motif's own value), which is why there is exactly one
copy now.

    bar grid (Beat This! downbeats, guarded by beats.check_grid)
      → musx chord posteriors, averaged per bar
      → projected onto the 12 PITCH CLASSES through the chord-tone matrix, so
        the dot product IS harmonic overlap: B♭ major {D F B♭} and G minor
        {D G B♭} share two notes, A♭ major {C E♭ A♭} shares none
      → SSM = cosine between those 12-d bar vectors
      → dictionary: find a motif, slide it along X reading the DIAGONAL, keep
        the peaks, the claimed bars leave the game, repeat
      → sections: a maximal chain of consecutive occurrences of ONE entry

Validated by Louis on the plots, song by song. Perfect on This Love and Don't
Know Why. It FAILS on Sunny (modulates — our similarity is not transposition
invariant), Beat It (−11.4 % tempo drift that the grid guard still passes at
96 %) and Close to You (66 % downbeat alignment, refused upstream). Those are
SSM/grid problems upstream, not method problems — see `docs/known_issues.md`,
"THE HARMONIC METHOD IS LOCKED", and
`docs/handoff_2026-08-05_harmonic_sections.md` for the open list.

What this does NOT solve, stated (rule #4):
  * no transposition invariance — a modulating repeat gets a fresh letter;
  * leftover gaps can be 2 bars long, which is not a section on a chart;
  * the thresholds are quantiles of the song's own matrix and the separation
    they cut is under one percent wide (q90 ≈ 0.986–0.990). A study with an
    ear-validated positive set is what settles them; `TILE`-style corpus
    numbers cannot.
"""
from __future__ import annotations

import logging

import numpy as np
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)

# ── the constants, all of them ──────────────────────────────────────────────
FRAME_DT = 23.22e-3      # musx posterior frame step
LAG_MIN, LAG_MAX = 2, 16  # candidate periods, in bars
PHASE_QUANTILE = 0.90    # a bar is "strong" at lag L when it ranks here
CONT_QUANTILE = 0.80     # …and a started run CONTINUES down to here
MARGIN_SIGMA = 0.5       # a peak must clear the local median by this × σ
MARGIN_WIN = 3           # the local window is ± MARGIN_WIN × L bars
INITIAL_PEAK_FRAC = 0.90  # …AND reach this fraction of the initial peak.
                         # Chosen by Louis on the 70/75/80/85/90/95 sweep, by
                         # eye, over five songs — the corpus metric cannot
                         # separate these thresholds at all.
GAP_MATCH = 0.90         # two leftover gaps share a letter when their
                         # bar-to-bar diagonal reaches this (same scale, same
                         # 0.90, as the peak rule: 1.0 = identical bar for bar)
CLEAR_MARGIN = 0.15      # report-only: "clearly above" for the separate box
MAX_ENTRIES = 6

# musx triad plane: col 0 = N; col i≥1 is root (i−1)%12, type (i−1)//12+1 in
# {maj, min, sus4, sus2, dim, aug}. Chord tones as semitone offsets:
TRIAD_TONES = {1: (0, 4, 7), 2: (0, 3, 7), 3: (0, 5, 7),
               4: (0, 2, 7), 5: (0, 3, 6), 6: (0, 4, 8)}


# ── substrate ───────────────────────────────────────────────────────────────
def chord_tone_matrix(n_cols: int) -> np.ndarray:
    """(n_cols, 12) — each chord class as its pitch-class indicator."""
    M = np.zeros((n_cols, 12))
    for i in range(1, n_cols):
        root, typ = (i - 1) % 12, (i - 1) // 12 + 1
        for s in TRIAD_TONES.get(typ, (0, 4, 7)):
            M[i, (root + s) % 12] = 1.0
    return M


def _unit(V):
    return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)


def harmonic_vectors(triad: np.ndarray, grid) -> np.ndarray:
    """(n_bars, 12) — each bar's chord posterior, as pitch classes.

    `triad` is musx's triad-plane frame posterior array — the pipeline already
    holds it (`probs[0]`), so nothing is decoded or re-loaded here.
    """
    M = chord_tone_matrix(triad.shape[1])
    n = len(grid) - 1
    P = []
    for b in range(n):
        a = int(grid[b] / FRAME_DT)
        z = max(a + 1, int(grid[b + 1] / FRAME_DT))
        seg = triad[a:min(z, len(triad))]
        P.append(seg.mean(0) if len(seg) else np.zeros(triad.shape[1]))
    return _unit(np.array(P) @ M)


def ssm(triad: np.ndarray, grid) -> np.ndarray:
    V = harmonic_vectors(triad, grid)
    return V @ V.T


# ── the locked reading rules ────────────────────────────────────────────────
def period_and_phase(S: np.ndarray) -> tuple[int, int]:
    """(L, b0) — the dominant repetition period, and the first bar where it is
    strong, which is the phase the motif is read from."""
    n = len(S)
    best, L = -1.0, LAG_MIN
    for Lx in range(LAG_MIN, min(LAG_MAX, n - 2) + 1):
        m = float(np.mean([S[b, b + Lx] for b in range(n - Lx)]))
        if m > best:
            best, L = m, Lx
    thr = float(np.quantile(off_diagonal(S), PHASE_QUANTILE))
    b0 = next((b for b in range(n - L) if S[b, b + L] >= thr), 0)
    return L, b0


def off_diagonal(S: np.ndarray) -> np.ndarray:
    """Every |lag| ≥ LAG_MIN cell — the ONE distribution all the quantiles come
    from. Cells nearer the diagonal are trivially self-similar and would drag
    every threshold up."""
    i = np.arange(len(S))
    return S[np.abs(i[:, None] - i[None, :]) >= LAG_MIN]


def diag_match(S, a: int, b: int, L: int) -> float:
    """Bar-to-bar similarity of two blocks, aligned: mean of S[a+i, b+i].

    This is NOT the sliding square. Measured on Don't Know Why's B section
    (2026-08-05): bars 22–29 and 38–45 are the same music, and their bar-to-bar
    diagonal is 0.984 — while the full 8×8 block average is 0.529. The gap is
    not noise: B is a THROUGH-COMPOSED 8 bars (Gm7 | C7 | F7 | Dm7 …), so its
    own internal block averages 0.533 — every bar differs from every other. A
    sliding square therefore scores a perfect match of a non-repetitive block
    barely above chance, and the dictionary misses it. The diagonal finds it
    instantly.

    Rule of thumb this establishes: the SQUARE finds material that repeats
    INSIDE itself (loops, cells); the DIAGONAL finds material that repeats
    ELSEWHERE regardless of its internal structure.
    """
    L = min(L, S.shape[0] - max(a, b))
    if L <= 0:
        return 0.0
    return float(np.mean([S[a + i, b + i] for i in range(L)]))


def slide(S: np.ndarray, L: int, b0: int) -> np.ndarray:
    """DIAGONAL reading: f(c) = mean over i of S[b0+i, c+i].

    "Is the material at c the same music as the motif, bar for bar."

    This replaced the sliding SQUARE everywhere on 2026-08-05 (Louis: « la
    diagonale est clairement mieux que le carré, on switch pour celui-là
    partout »). Head-to-head on the same blocks with the same peak rule
    (/reports/diag_vs_square.html):

        This Love, 4-bar cycle    square 7 occurrences   diagonal 7   identical
        This Love, 8-bar chorus   square 12              diagonal 4
        Don't Know Why, 4-bar     square 10              diagonal 10  identical
        Don't Know Why, 8-bar B   square 2               diagonal 2   identical

    The square's 12 on the chorus are 14, 16, 34, 36, 56, 58, 60, 62, 64, 66,
    68, 70 — overlapping junk every two bars, impossible for an 8-bar motif.
    The diagonal returns 16, 36, 56, 64. Same signal, cleaner reading; the two
    only diverge on long motifs, where the square's off-diagonal cells swamp it.
    """
    n = len(S)
    return np.array([diag_match(S, b0, c, L) for c in range(0, n - L + 1)])


def square_slide(S: np.ndarray, L: int, b0: int, *,
                 i_understand_this_is_deprecated: bool = False) -> np.ndarray:
    """DEPRECATED 2026-08-05 — DO NOT USE. Kept only so the historical
    comparison page can still draw it.

    The sliding square. On long motifs it returns overlapping occurrences that
    cannot exist (an 8-bar motif recurring every 2 bars), because its
    off-diagonal cells dominate the product. Use `slide()`.
    """
    if not i_understand_this_is_deprecated:
        raise RuntimeError(
            "square_slide() is DEPRECATED and must not be reconnected. It "
            "returns impossible overlapping occurrences on long motifs (This "
            "Love's 8-bar chorus: 12 hits, four of them 2 bars apart) — see "
            "/reports/diag_vs_square.html and docs/known_issues.md. Use "
            "harmonic_sections.slide(), which is the DIAGONAL reading. If you "
            "genuinely need the old behaviour for a historical figure, pass "
            "i_understand_this_is_deprecated=True and say so in your report.")
    n = len(S)
    P = S[b0:b0 + L, b0:b0 + L]
    return np.array([float((P * S[b0:b0 + L, c:c + L]).sum()) / (L * L)
                     for c in range(0, n - L + 1)])


def peaks(curve: np.ndarray, L: int, b0: int) -> np.ndarray:
    """THE RULE: local-baseline margin AND ≥ INITIAL_PEAK_FRAC of the initial
    peak — the curve where the motif sits on itself."""
    ref = float(curve[b0])
    win = max(4, MARGIN_WIN * L)
    cand, _ = find_peaks(curve)
    sd = float(np.std(curve))
    out = []
    for k in cand:
        a, z = max(0, k - win), min(len(curve), k + win + 1)
        if (curve[k] - float(np.median(curve[a:z]))) < MARGIN_SIGMA * sd:
            continue
        if curve[k] < INITIAL_PEAK_FRAC * ref:
            continue
        out.append(int(k))
    return np.array(out, int)


# ── the dictionary ──────────────────────────────────────────────────────────
def build_dictionary(S, n, max_entries=MAX_ENTRIES, criterion="hybrid"):
    """Louis's loop. Returns (entries, boxed); an entry is {L, b0, curve, occ}.

    `boxed` scores every occurrence with every entry — a report-only view of
    the arbitration, kept because the dictionary page draws it.
    """
    entries, claimed = [], np.zeros(n, bool)
    for _ in range(max_entries):
        free = np.flatnonzero(~claimed)
        if len(free) < 2 * LAG_MIN + 2:
            break
        # Motif search on the free bars — by the LONGEST RUN of consecutive
        # strong bar-matches at some lag, not by the mean and not by a total.
        #
        # Two dead ends recorded so they are not retried (2026-08-05):
        #   * the MEAN over a lag: a lag where 8 bars agree at 0.98 loses to one
        #     where 20 bars average 0.5 — Don't Know Why's B block sat at lag 16
        #     (0.894) and lost to lag 4 (0.981), so it was never proposed;
        #   * the TOTAL evidence above a threshold: biased toward SHORT lags,
        #     which simply have more pairs. It turned This Love's 8-bar chorus
        #     into a 2-bar motif with 15 occurrences and cost 6 bars of coverage.
        #
        # A real block repeat shows as a CONSECUTIVE run of strong matches along
        # one diagonal. That run's LENGTH is the motif length and the lag is the
        # distance to its copy — the two are different quantities, which the old
        # "L = lag" formulation conflated.
        off_all = off_diagonal(S)
        strong = float(np.quantile(off_all, PHASE_QUANTILE))
        use_mean = (criterion == "mean") or (criterion == "hybrid" and not entries)
        if use_mean:
            # ROUND 1 (hybrid) or always (mean): the mean-based period + phase.
            L, b0 = period_and_phase(S)
            best = (L, b0, L)
        elif criterion == "total":
            bt, bL = -1.0, None
            for Lx in range(LAG_MIN, min(LAG_MAX, n - 2) + 1):
                v = [S[b, b + Lx] for b in free
                     if b + Lx < n and not claimed[b + Lx]]
                if len(v) < 2:
                    continue
                sc = float(sum(max(0.0, x - strong) for x in v))
                if sc > bt:
                    bt, bL = sc, Lx
            if bL is None or bt <= 0:
                break
            b0m = next((b for b in free if b + bL < n and S[b, b + bL] >= strong), None)
            if b0m is None:
                break
            best = (bL, b0m, bL)
        else:
            best = (0, None, None)                  # (run length, start, lag)
            # HYSTERESIS. A run used to break on a single dip below `strong`,
            # and `strong` is this song's q90 — 0.986 on This Love, so a bar
            # matching its neighbour at 0.978 counted as a break and cut the
            # 8-bar chorus to 7 (Louis spotted it: "This Love est un motif de 8
            # mesures pas 7"). A run now STARTS above q90 and CONTINUES above
            # q80, the standard two-level rule, so 0.8 % of noise no longer
            # ends a section.
            cont = float(np.quantile(off_all, CONT_QUANTILE))
            for d in range(LAG_MIN, n - LAG_MIN):
                run, start = 0, None
                for b in range(n - d):
                    is_free = not claimed[b] and not claimed[b + d]
                    v = S[b, b + d]
                    if is_free and (v >= strong or (run > 0 and v >= cont)):
                        if run == 0:
                            start = b
                        run += 1
                        if run > best[0]:
                            best = (run, start, d)
                    else:
                        run = 0
        run_len, b0, lag = best
        if run_len < LAG_MIN or b0 is None:
            break
        # The RUN is the extent of the repeating region; the MOTIF is the lag
        # when that region holds more than one copy. This Love's chorus gives a
        # 16-bar run at lag 8 — that is 8 bars played twice, not a 16-bar motif.
        # Norah's B gives an 8-bar run at lag 16 — there the run IS the motif.
        # Both then come out at 8, which is what Louis said they were.
        L = int(min(run_len, lag)) if lag else int(run_len)
        curve = slide(S, L, b0)
        # Occurrences must not overlap each other either: a motif of L bars
        # cannot start again L/2 bars later (This Love's 8-bar entry was
        # returning 56, 58, 60, 62 … as separate occurrences). Take them
        # strongest-first and drop anything that collides with a kept one or
        # with a block an earlier entry already owns.
        cand = sorted((int(o) for o in peaks(curve, L, b0)),
                      key=lambda o: -curve[o])
        occ, taken = [], claimed.copy()
        for o in cand:
            if taken[o:min(n, o + L)].any():
                continue
            occ.append(o)
            taken[o:min(n, o + L)] = True
        occ.sort()
        if not occ:
            break
        entries.append({"L": L, "b0": int(b0), "curve": curve, "occ": occ})
        # A detected block leaves the game (Louis, 2026-08-05, revising his
        # earlier "keep them as candidates"): « lorsqu'on a détecté un block, il
        # ne devrait plus être considéré par les autres blocks, il est maintenant
        # affecté à une section et mis dans le dictionnaire donc il n'apparaît
        # plus ». The motif's OWN block is claimed too — without that, b0 stays
        # free and every round rediscovers the same motif (measured: 6 identical
        # entries per song).
        claimed[b0:min(n, b0 + L)] = True
        for o in occ:
            claimed[o:min(n, o + L)] = True
        if claimed.all():
            break

    # every occurrence, scored by EVERY entry — this is the separate box
    boxed = []
    for ei, e in enumerate(entries):
        for o in e["occ"]:
            scores = []
            for f in entries:
                c = f["curve"]
                k = min(max(o, 0), len(c) - 1)
                scores.append(float(c[k]) / max(float(c[f["b0"]]), 1e-9))
            order = np.argsort(scores)[::-1]
            win, run = int(order[0]), (int(order[1]) if len(order) > 1 else None)
            margin = scores[win] - (scores[run] if run is not None else 0.0)
            boxed.append({"bar": o, "found_by": ei, "scores": scores,
                          "winner": win if (run is None or margin >= CLEAR_MARGIN) else None,
                          "margin": margin})
    return entries, boxed


# ── dictionary → sections ───────────────────────────────────────────────────
def sections_from(S, n, entries):
    """[{b0, b1, letter, why}] over bar indices — contiguous and covering.

      1. RUNS   — a section is a maximal chain of CONSECUTIVE occurrences of
                  one entry, not one occurrence. This Love's 4-bar cycle played
                  at 0, 4, 8, 12 is ONE 16-bar section, not four.
      2. GAPS   — what no entry claims becomes a section of its own.
      3. TAILS  — a gap shorter than the motif around it joins the run before.
      4. RESIDUAL PASS — the remaining gaps are compared to each other by the
                  DIAGONAL reading and those that match share a letter.

    Letters come from the dictionary ENTRY, never from a similarity threshold,
    so two passages of incompatible length can no longer land under one letter.
    """
    # 1 — runs of consecutive occurrences
    secs = []
    for ei, e in enumerate(entries):
        L = e["L"]
        pos = sorted(set([e["b0"]] + list(e["occ"])))
        run = [pos[0]]
        for p in pos[1:]:
            # Chain when the next occurrence starts where this one ends, give or
            # take a bar. NOT `p == run[-1] + L`: a motif measures the run of
            # strong agreement, which can stop a bar short of the musical block
            # (This Love's chorus is a 7-bar motif recurring every 8), and the
            # strict test then refused to chain 56, 64, 72 into one section.
            if 0 <= p - (run[-1] + L) <= 1:
                run.append(p)
            else:
                secs.append({"b0": run[0], "b1": run[-1] + L - 1, "entry": ei,
                             "why": f"{len(run)}× le motif de l'entrée {ei+1}"})
                run = [p]
        secs.append({"b0": run[0], "b1": run[-1] + L - 1, "entry": ei,
                     "why": f"{len(run)}× le motif de l'entrée {ei+1}"})
    secs.sort(key=lambda s: s["b0"])
    for s in secs:
        s["b1"] = min(s["b1"], n - 1)

    # 2 — gaps
    filled, b = [], 0
    for s in secs:
        if s["b0"] > b:
            filled.append({"b0": b, "b1": s["b0"] - 1, "entry": None,
                           "why": "aucune entrée ne le revendique"})
        if s["b1"] >= b:
            s = dict(s, b0=max(s["b0"], b))
            filled.append(s)
            b = s["b1"] + 1
    if b < n:
        filled.append({"b0": b, "b1": n - 1, "entry": None,
                       "why": "aucune entrée ne le revendique"})

    # 3 — a gap shorter than the motif around it is a tail: join it left
    motif_of = {i: e["L"] for i, e in enumerate(entries)}
    out = []
    for s in filled:
        prev = out[-1] if out else None
        if (s["entry"] is None and prev is not None and prev["entry"] is not None
                and (s["b1"] - s["b0"] + 1) <= motif_of[prev["entry"]]):
            prev["b1"] = s["b1"]
            prev["why"] += " + sa queue"
            continue
        out.append(dict(s))

    # 4 — residual pass: gaps matching each other by the DIAGONAL share a letter
    gaps = [i for i, s in enumerate(out) if s["entry"] is None]
    group, gid = {}, 0
    for a, ia in enumerate(gaps):
        if ia in group:
            continue
        group[ia] = gid
        for ic in gaps[a + 1:]:
            if ic in group:
                continue
            L = min(out[ia]["b1"] - out[ia]["b0"] + 1,
                    out[ic]["b1"] - out[ic]["b0"] + 1)
            if L >= 2 and diag_match(S, out[ia]["b0"], out[ic]["b0"], L) >= GAP_MATCH:
                group[ic] = gid
                out[ic]["why"] = ("apparié au trou de la mesure "
                                  f"{out[ia]['b0']+1} par la diagonale")
        gid += 1

    # letters: one per entry, then one per gap group
    letters, nxt = {}, 0
    for i, s in enumerate(out):
        key = ("e", s["entry"]) if s["entry"] is not None else ("g", group[i])
        if key not in letters:
            letters[key] = chr(ord("A") + nxt) if nxt < 26 else f"S{nxt}"
            nxt += 1
        s["letter"] = letters[key]
    return out


# ── the pipeline entry point ────────────────────────────────────────────────
def detect_sections(grid, triad: np.ndarray, bars=None) -> list[dict]:
    """[{b0, b1, label}] over BAR indices — contiguous, covering, unfolded.

    Same contract as the chroma detector it replaces. `bars` is accepted and
    unused: none of the chart-level rules (cells, held bars, even lengths) of
    the old detector survive here — the dictionary decides boundaries from the
    harmony alone, which is the whole point of the change.
    """
    n = len(grid) - 1
    if n < 2 * LAG_MIN + 2:
        return [{"b0": 0, "b1": max(0, n - 1), "label": "A"}]
    S = ssm(triad, grid)
    entries, _ = build_dictionary(S, n)
    secs = sections_from(S, n, entries)
    out = [{"b0": s["b0"], "b1": s["b1"], "label": s["letter"]} for s in secs]

    # contract check — contiguous, covering, non-empty. A section list that
    # silently loses bars would show up as a chart missing music, which is the
    # kind of thing that goes unnoticed for weeks (splitter lesson, 2026-07-31).
    if not out or out[0]["b0"] != 0 or out[-1]["b1"] != n - 1 or \
            any(a["b1"] + 1 != b["b0"] for a, b in zip(out, out[1:])) or \
            any(s["b1"] < s["b0"] for s in out):
        raise RuntimeError(
            f"harmonic sections are not a partition of bars 0..{n-1}: "
            f"{[(s['b0'], s['b1'], s['label']) for s in out]} — refusing to "
            "render a chart that would silently drop or duplicate bars")
    logger.info("sections (harmonic dictionary): %d entries, %d sections — %s",
                len(entries), len(out),
                " ".join(f"{s['label']}[{s['b0']+1}-{s['b1']+1}]" for s in out))
    return out
