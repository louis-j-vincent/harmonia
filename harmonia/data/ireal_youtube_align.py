"""iReal ↔ YouTube alignment via discrete downbeat anchoring (2026-07-21).

Mission 1 (docs/mission_1_phase1b_results.md, 2026-07-13) tried continuous
subsequence-DTW of a synthetic iReal chroma template against full-mix CQT
chroma and FAILED its own ±150ms gate on all 3 pilots (mean 1169-1504ms):
full-mix chroma SNR is too low for template matching to discriminate the
right alignment from a wrong one over a long, FREE continuous search.

This module tries a different decomposition, made possible by this
session's new downbeat tracker (``harmonia.models.downbeat_anchor``, Beat
This!, ISMIR 2024): collapse "continuous warp over the whole file" into "a
SMALL discrete number of candidate starting downbeats" — walk real detected
downbeats as bar anchors (bar 1 = downbeats[d0], bar 2 = downbeats[d0+1], …),
score each candidate d0 with Mission 1's own DTW-independent change-point
validator (``measure_alignment_error`` — reused unchanged from
``scripts/mission_1_build_benchmark.py``), and accept the best-scoring d0
only if it clears a quality gate.

Honest result on a 10-pair spot-check (7 pop + 3 jazz, real library audio ×
matched pop400/jazz1460 charts, scratchpad/mission1_phase2_pop.py): the
BEST case ("She Will Be Loved") aligns to a median 207ms — a real ~6×
improvement over Mission 1's best DTW case — but only ~1-2 of 10 pairs clear
a reasonable acceptance gate. The other ~80% fail for a genuinely DIFFERENT
reason than Mission 1's SNR problem: the real recording's actual bar-by-bar
structure (extra chorus repeats, bridges, ad-libs) does not match the iReal
chart's linear bar count, so a RIGID anchor walk (no skips/insertions
allowed, unlike DTW) drifts by however many bars the structures diverge —
confirmed directly on Billie Jean: even brute-force scoring EVERY possible
starting downbeat (146 candidates) never got below 787ms mean.

This is real, measured, partial progress — not a solved problem. Use
``align_tune_to_audio`` and CHECK its ``accepted`` flag; never treat a
rejected alignment as usable GT.

**2026-07-21, same day, user's own follow-up idea**: rather than rescue the
structural-mismatch cases with bounded repeat/skip hypotheses, decompose
the problem instead — align each of the chart's own SECTIONS (iReal's own
"i"/"A"/"B" markers) independently (``align_tune_sections_to_audio``), so
one divergent bridge/extra-verse can't sink every other section. This is
NOT a smaller version of the same idea — it changes the yield dramatically:
on a 6-song spot-check (Billie Jean, She Will Be Loved, Let It Be, Every
Breath You Take, Goodbye Yellow Brick Road, Easy — Chain Of Fools dropped,
it's a near-static one-chord vamp with too few chord CHANGES to validate
against regardless of framing, a different failure mode entirely), **22/42
individual sections (52%) clear the acceptance gate**, most at 20-130ms —
vs the whole-song method's ~10-20% of whole SONGS. 5/6 songs now have AT
LEAST ONE usable aligned section (only Billie Jean has zero) — a much
stronger foundation for corpus-scale GT than "reject 80-90% of songs
outright." Confirmed directly on Billie Jean (0/6, whole-song was also bad):
sections still individually reject there (376-1451ms) — it's a genuinely
harder recording, not a framing artifact letting a bad case through.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Median (not mean) acceptance gate — mean is dominated by the occasional
# genuine structural-mismatch point even in an otherwise well-aligned song
# (e.g. "She Will Be Loved"'s max=1540ms despite a 207ms median); median is
# the honest "is most of this song aligned" signal. 250ms chosen from the
# observed gap between the 2 accepted pairs (80-207ms) and the next-best
# rejected one (942ms) in the spot-check — a wide, non-arbitrary margin.
DEFAULT_ACCEPT_MEDIAN_MS = 250.0


def _downbeat_anchored_warp(chords: list[dict], downbeats: np.ndarray, d0: int,
                            bpb: int, spb_nominal: float):
    """Piecewise-linear warp(t_template_s) -> t_audio_s hypothesising iReal
    bar 1 = downbeats[d0], bar 2 = downbeats[d0+1], etc. Chord positions
    WITHIN a bar are placed by fractional bar position (computed from the
    chart's own beat_abs, recovered losslessly from t_template_s/spb_nominal
    — see module docstring; this is NOT a tempo assumption, spb_nominal
    exactly cancels since that's how mma_chart_to_chords built start_s in
    the first place), interpolated between the two REAL bar anchors —
    so within-bar timing follows the actual recording's local tempo, not
    the chart's nominal one, even though the math routes through it."""
    n_bars_needed = max(c["bar"] for c in chords)
    anchors = [(k + 1, float(downbeats[d0 + k])) for k in range(n_bars_needed + 2)
              if 0 <= d0 + k < len(downbeats)]
    if len(anchors) < 2:
        return None
    bar_nums = np.array([a[0] for a in anchors], dtype=float)
    bar_times = np.array([a[1] for a in anchors], dtype=float)

    def warp(t_template_s: float) -> float:
        beat_abs = t_template_s / spb_nominal
        bar_frac = beat_abs / bpb + 1.0
        return float(np.interp(bar_frac, bar_nums, bar_times))
    return warp


def _section_runs(chart) -> "list[tuple[str, int, int]]":
    """Contiguous (label, bar0, bar1) runs from ``chart.timeline`` — iReal's
    own per-bar section markers (e.g. "i" intro, "A", "B"). A label that
    recurs non-contiguously (a second chorus later in the tune) yields a
    SEPARATE run each time it appears, deliberately — see
    ``align_tune_sections_to_audio``'s docstring for why each physical
    occurrence needs its own independent alignment, not a shared one."""
    runs: list[tuple[str, int, int]] = []
    cur_label = cur_bar0 = prev_barno = None
    for barno, label, _slots in chart.timeline:
        if label != cur_label:
            if cur_label is not None:
                runs.append((cur_label, cur_bar0, prev_barno))
            cur_label, cur_bar0 = label, barno
        prev_barno = barno
    if cur_label is not None:
        runs.append((cur_label, cur_bar0, prev_barno))
    return runs


def _mma_chart_to_chords_for_bars(chart, bar_lo: int, bar_hi: int) -> list[dict]:
    """Same flattening as Mission 1's ``mma_chart_to_chords``, restricted to
    ``[bar_lo, bar_hi]`` (inclusive) and REBASED so that bar ``bar_lo``
    becomes local bar 1 — i.e. this section can be aligned to audio as if it
    were its own mini-chart, independent of every other section's placement.
    'p' (repeat-previous-chord) tokens still resolve correctly across the
    bar_lo boundary by tracking prev_mma/prev_tok through the skipped bars
    first. Carries the RAW iReal ``tok`` (e.g. "Ah7", not the MMA-converted
    "Am7b5") alongside ``mma`` — a training-corpus consumer needs ``tok``:
    it's what ``harmonia.irealb_fetcher._parse_ireal_chord_token`` expects
    (verified 2026-07-21 that the two notations disagree on the SAME chord —
    ``_parse_ireal_chord_token`` on the MMA string "Am7b5" wrongly resolves
    to dominant "7", not half-diminished; on the raw token "Ah7" it correctly
    gives "hdim7" — the MMA string uses a different convention entirely)."""
    from harmonia.data.ireal_corpus import chord_root_pc

    bpb = chart.beats_per_bar
    spb = 60.0 / chart.tempo
    events: list[dict] = []
    prev_mma, prev_tok = "z", "z"
    for barno, _label, slots in chart.timeline:
        if barno < bar_lo:
            for _beat_off, tok, mma in slots:
                if not (mma == "z" and tok.strip() == "p"):
                    prev_mma, prev_tok = mma, tok
            continue
        if barno > bar_hi:
            break
        local_barno = barno - bar_lo + 1
        for k, (beat_off, tok, mma) in enumerate(slots):
            if mma == "z" and tok.strip() in ("p",):
                mma, tok = prev_mma, prev_tok
            beat_abs = (local_barno - 1) * bpb + beat_off
            if k + 1 < len(slots):
                beat_end = (local_barno - 1) * bpb + slots[k + 1][0]
            else:
                beat_end = local_barno * bpb
            events.append({
                "mma": mma, "tok": tok, "root": chord_root_pc(mma), "bar": local_barno,
                "beat_abs": float(beat_abs), "start_s": beat_abs * spb, "end_s": beat_end * spb,
            })
            if mma != "z":
                prev_mma, prev_tok = mma, tok
    return events


# Minimum fraction of consecutive chord-to-chord root INTERVALS that must
# agree between the chart and the model's independent decode (see
# _model_root_shape_agreement) for a candidate to be accepted.
#
# 2026-07-21, user's own requested A/B (real 6-song comparison, WITH vs
# WITHOUT this gate, min_shape_agreement=0.6): the safeguard REDUCED yield
# from 22/42 to 9/42 accepted sections — and, more importantly, it rejected
# "She Will Be Loved"'s bar-5-20 section (shape_agreement=0.33) despite that
# section aligning to a 47ms median on the independent chroma change-point
# score — one of the cleanest results in the whole study. The model's own
# root head just isn't reliable enough on short (3-10 chord) windows for a
# consecutive-PAIR interval check: a single wrong root corrupts BOTH
# intervals touching it, so error compounds fast on short sequences even
# when most individual chords are probably right. This is a validated
# NEGATIVE RESULT, not a tuning problem — no threshold in [0.33, 1.0] would
# have kept that case, and anything below 0.33 rejects nothing. Default is
# 0.0 (the check still RUNS and its score is logged/returned on every result
# for visual inspection — see /debug/section-align — just never used to
# reject) until a form of this safeguard that survives real validation is
# found (e.g. aggregating across many sections/songs rather than judging one
# short section in isolation).
DEFAULT_MIN_SHAPE_AGREEMENT = 0.0


def _model_root_shape_agreement(
    chords: list[dict], warp, arr: np.ndarray, times: np.ndarray, heads,
) -> "float | None":
    """User's own safeguard, 2026-07-21: the chroma change-point validator
    alone can be fooled by weak harmonic contrast (confirmed: many sections
    get "no valid candidate" simply because the chart itself doesn't change
    chord often enough to validate against). Add the chord-recognition
    model's OWN independent decode as a second check — but carefully, so it
    can't reintroduce the exact circularity (docs/mission_1_real_audio_
    benchmark_design.md) this whole method exists to avoid.

    The model NEVER supplies a label and NEVER defines "correct" here: it
    only says whether the SHAPE of its own blind root guesses (transposition
    -invariant root-to-root intervals, not absolute pitch classes — the
    model's own dom->maj or hdim->min confusions, known real weaknesses per
    known_issues.md, don't matter to an interval comparison) matches the
    shape of the chart's OWN declared root sequence, at the candidate
    offset under test. A candidate where the model's blind guesses move up
    a fourth exactly where the chart also moves up a fourth is evidence for
    that offset being right; the model is never asked "which chord is this"
    in a way that could feed back into the label.

    Returns the fraction of consecutive interval agreements, or None if
    there aren't enough chord spans to judge (< 3).
    """
    from harmonia.models.nnls_features import _l2, _ROLL_TO_C

    chart_roots: list[int] = []
    model_roots: list[int] = []
    for c in chords:
        t0, t1 = warp(c["start_s"]), warp(c["end_s"])
        if t1 <= t0:
            continue
        mask = (times >= t0) & (times < t1)
        if not mask.any():
            continue
        seg = arr[mask].mean(0)
        # MUST match the trained heads' feature convention exactly (roll to
        # C-frame, L2-normalise each half) — confirmed the hard way, 2026-
        # 07-21: feeding raw un-normalised bothchroma gave near-random root
        # guesses (0-33% shape agreement even on sections independently
        # verified correct via the chroma change-point score), which is a
        # feature bug, not evidence the shape-check idea itself is unsound.
        bass = _l2(np.roll(seg[:12], _ROLL_TO_C))
        treb = _l2(np.roll(seg[12:], _ROLL_TO_C))
        feat = np.concatenate([bass, treb])[None, :]
        model_root = int(heads.root_proba(feat).argmax(1)[0])
        chart_roots.append(c["root"])
        model_roots.append(model_root)
    if len(chart_roots) < 3:
        return None
    hits = sum(
        1 for i in range(1, len(chart_roots))
        if (chart_roots[i] - chart_roots[i - 1]) % 12 == (model_roots[i] - model_roots[i - 1]) % 12
    )
    return hits / (len(chart_roots) - 1)


def align_tune_sections_to_audio(
    tune, audio_path: "Path | str", *, min_run_bars: int = 4,
    accept_median_ms: float = DEFAULT_ACCEPT_MEDIAN_MS,
    min_shape_agreement: float = DEFAULT_MIN_SHAPE_AGREEMENT,
) -> "list[dict]":
    """Per-SECTION version of ``align_tune_to_audio`` (2026-07-21, user's own
    idea after seeing the whole-song version's ~10-20% yield): instead of one
    rigid bar-by-bar walk across the ENTIRE tune — which drifts the moment
    the real recording's structure (extra chorus, bridge, ad-libs) diverges
    from the chart's linear bar count, confirmed directly on Billie Jean —
    align each of the chart's own sections (iReal's "i"/"A"/"B" markers)
    INDEPENDENTLY, searching the whole song (not just the first
    ``max_intro_s``, since a second verse/chorus can start anywhere) for
    where THAT section's chord pattern best matches.

    This can't rescue a section that's simply absent from a given recording,
    but it stops one divergent section from sinking every other section's
    alignment — and it directly answers the question that motivated it: "do
    we detect chords well on each independent section," not just "does the
    whole song align as one rigid brick."

    Returns one dict per chart section run, each shaped like
    ``align_tune_to_audio``'s return (plus ``label``/``bar0``/``bar1``) —
    check ``accepted`` per section, same as the whole-song version.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
    from mission_1_build_benchmark import measure_alignment_error
    from harmonia.data.ireal_corpus import tune_to_mma
    from harmonia.models.downbeat_anchor import beat_this_downbeats
    from harmonia.models import nnls_features as nf
    import librosa

    chart = tune_to_mma(tune)
    bpb = chart.beats_per_bar
    spb = 60.0 / chart.tempo

    downbeats, conf = beat_this_downbeats(str(audio_path))
    results: list[dict] = []
    if len(downbeats) < 5:
        logger.info("align_tune_sections_to_audio(%s): too few downbeats (%d)",
                   tune.title, len(downbeats))
        return results

    y, sr = librosa.load(str(audio_path), sr=22050)
    chroma_v = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512).T
    val_t = librosa.frames_to_time(np.arange(chroma_v.shape[0]), sr=sr, hop_length=512)
    # Loaded once, shared across every section/candidate below — the model's
    # OWN independent root decode used only for the shape safeguard (never
    # to supply a label). None if the trained heads aren't available; the
    # shape check then degrades to "skipped", not "reject everything".
    heads = nf.get_heads()
    nnls_arr = nnls_times = None
    if heads is not None:
        try:
            nnls_arr, nnls_times = nf.extract_bothchroma(audio_path, use_cache=False)
        except Exception as exc:  # noqa: BLE001 — shape check is a safeguard, not required
            logger.warning("align_tune_sections_to_audio(%s): NNLS extraction for the "
                          "model shape-safeguard failed (%s) — skipping that check",
                          tune.title, exc)
            heads = None

    for label, bar0, bar1 in _section_runs(chart):
        base = {"label": label, "bar0": bar0, "bar1": bar1, "beat_this_confidence": conf}
        if bar1 - bar0 + 1 < min_run_bars:
            results.append({**base, "accepted": False, "reason": "run too short"})
            continue
        chords = _mma_chart_to_chords_for_bars(chart, bar0, bar1)
        best = None
        for d0 in range(len(downbeats)):
            warp = _downbeat_anchored_warp(chords, downbeats, d0, bpb, spb)
            if warp is None:
                continue
            err = measure_alignment_error(chords, warp, chroma_v, val_t, n_points=12)
            if err["n_used"] < 3 or np.isnan(err["mean_ms"]):
                continue
            if best is None or err["median_ms"] < best[1]["median_ms"]:
                best = (d0, err, warp)
        if best is None:
            results.append({**base, "accepted": False, "reason": "no valid candidate"})
            continue
        d0, err, warp = best
        accepted = err["median_ms"] <= accept_median_ms
        shape_agreement = None
        if accepted and heads is not None:
            shape_agreement = _model_root_shape_agreement(chords, warp, nnls_arr, nnls_times, heads)
            if shape_agreement is not None and shape_agreement < min_shape_agreement:
                accepted = False
        logger.info("align_tune_sections_to_audio(%s) section %s [bar %d-%d]: "
                   "d0=%d median=%.0fms n=%d shape_agree=%s -> %s", tune.title, label,
                   bar0, bar1, d0, err["median_ms"], err["n_used"],
                   f"{shape_agreement:.2f}" if shape_agreement is not None else "n/a",
                   "ACCEPTED" if accepted else "rejected")
        results.append({**base, "accepted": accepted, "warp": warp, "d0": d0,
                        "offset_s": warp(0.0), "error": err, "chords": chords,
                        "shape_agreement": shape_agreement})
    return results


def align_tune_to_audio(
    tune, audio_path: "Path | str", *, max_intro_s: float = 60.0,
    accept_median_ms: float = DEFAULT_ACCEPT_MEDIAN_MS,
) -> "dict | None":
    """Align one iReal ``Tune`` to a matched audio file. Returns None if
    Beat This!'s own downbeat confidence is too low to even try (see
    ``downbeat_anchor.sota_downbeat_phase``'s gate), or a dict:

        {"accepted": bool, "warp": callable, "d0": int, "offset_s": float,
         "error": {...},  # measure_alignment_error's full breakdown
         "chords": [...]}  # mma_chart_to_chords output, for the caller to warp

    ``accepted`` is the ONLY field callers should gate on before treating
    this as usable ground truth — an unaccepted result is still returned
    (with its warp/error) so callers can inspect *why* it failed, but must
    not be used as GT.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
    from mission_1_build_benchmark import mma_chart_to_chords, measure_alignment_error
    from harmonia.data.ireal_corpus import tune_to_mma
    from harmonia.models.downbeat_anchor import beat_this_downbeats
    import librosa

    chart = tune_to_mma(tune)
    chords = mma_chart_to_chords(chart)
    bpb = chart.beats_per_bar
    spb = 60.0 / chart.tempo

    downbeats, conf = beat_this_downbeats(str(audio_path))
    if len(downbeats) < 5:
        logger.info("align_tune_to_audio(%s): too few downbeats (%d)", tune.title, len(downbeats))
        return None

    y, sr = librosa.load(str(audio_path), sr=22050)
    chroma_v = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512).T
    val_t = librosa.frames_to_time(np.arange(chroma_v.shape[0]), sr=sr, hop_length=512)

    cand_d0 = [i for i, t in enumerate(downbeats) if t <= max_intro_s]
    best = None
    for d0 in cand_d0:
        warp = _downbeat_anchored_warp(chords, downbeats, d0, bpb, spb)
        if warp is None:
            continue
        err = measure_alignment_error(chords, warp, chroma_v, val_t, n_points=20)
        if err["n_used"] < 4 or np.isnan(err["mean_ms"]):
            continue
        if best is None or err["median_ms"] < best[1]["median_ms"]:
            best = (d0, err, warp)
    if best is None:
        logger.info("align_tune_to_audio(%s): no valid candidate offset", tune.title)
        return None

    d0, err, warp = best
    accepted = err["median_ms"] <= accept_median_ms
    logger.info("align_tune_to_audio(%s): d0=%d median=%.0fms mean=%.0fms n=%d beat_this_conf=%.2f -> %s",
               tune.title, d0, err["median_ms"], err["mean_ms"], err["n_used"], conf,
               "ACCEPTED" if accepted else "rejected")
    return {"accepted": accepted, "warp": warp, "d0": d0, "offset_s": warp(0.0),
           "error": err, "chords": chords, "beat_this_confidence": conf}
