#!/usr/bin/env python3
"""Section-wise rigid-tempo alignment via inferred-chord proxy matching.

The single-global-tempo grid (`fit_beat_grid.py`) breaks the moment a song has
per-section tempo changes or vamps between sections. Autumn Leaves is the
canonical failure: bars 0-7 (A) sit at a rigid 181 BPM, then there is a vamp,
then A repeats — a single constant tempo scores `all_resid_rms ~= 39 s`.

This tool treats each chart *section instance* (A, A, B, C, ...) as its own
rigid tempo block and uses the model's *inferred* chord sequence from the audio
as a proxy to locate where each section actually starts in the recording,
rather than assuming the chart's continuous bar numbering maps linearly to time.

Core steps
----------
1. Parse the gt-align chart JSON into ordered section instances, each a chord
   sequence with in-section bar/beat metric positions.
2. Load the model's inferred per-unit chords (root pitch-class, quality family,
   confidence, t0/t1 seconds) from the `inferred_<slug>.html` `const P` blob.
3. Estimate a single GLOBAL transposition offset between chart roots and
   inferred roots (Autumn Leaves' model output is a whole tone flat: keyName
   "G# major" vs true Bb major, a -2 semitone shift that is constant across the
   whole recording). Matching is done AFTER applying this offset.
4. For each section instance, grid-search (anchor_t, slope_s_per_bar) to
   maximise a confidence-weighted chord-proximity score against the inferred
   sequence. Sections are matched in recording order (each instance must start
   at/after the previous instance's end, minus a small tolerance), which
   disambiguates the repeated A theme; tempo is fit independently per instance.
5. Flag vamps: gaps > 0.5 bar between consecutive matched sections, and
   sections whose own residual is high.
6. Assemble a final grid: every chart chord gets a `t0_perfect` from its
   section's fitted tempo; vamp regions are flagged and not marked as clean
   training data.

Proximity (per spec): exact root+quality = 1.0, root+family = 0.8,
root only = 0.4, no root match = 0.0 — roots primary, quality a tiebreak.

CLI:
    python scripts/align_by_sections.py \
        --chart docs/plots/annotations/irealb_autumn_leaves.html.json \
        --inferred docs/plots/inferred_autumn_leaves.html \
        --bpm-prior 181 \
        --out docs/plots/annotations/irealb_autumn_leaves_sectionwise.json \
        --diag-html docs/plots/autumn_leaves_section_alignment.html
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

_NOTE_TO_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


# ─────────────────────────────────────────────────────────────────────────────
# Chord label parsing
# ─────────────────────────────────────────────────────────────────────────────
def label_root_pc(label: str) -> int | None:
    """Pitch class of a chart chord label's root (iReal notation), or None."""
    if not label or label in ("z", "N", "NC"):
        return None
    m = re.match(r"^([A-G])([#b]?)", label)
    if not m:
        return None
    pc = _NOTE_TO_PC[m.group(1)]
    if m.group(2) == "#":
        pc += 1
    elif m.group(2) == "b":
        pc -= 1
    return pc % 12


def quality_family(label_or_qual: str) -> str:
    """Coarse quality family in {maj, min, dom, dim} from a label or a quality
    suffix. Roots are stripped first so this works on both 'A#^7' and '^7'."""
    s = label_or_qual or ""
    # strip a leading root note if present (label form)
    s = re.sub(r"^[A-G][#b]?", "", s)
    sl = s.lower()
    # order matters: half-dim/dim markers first, then explicit min, then maj/dom
    if "h" in sl or "ø" in s or "dim" in sl or (sl.startswith("o")):
        return "dim"
    if s.startswith("-") or sl.startswith("min") or sl.startswith("m") and not sl.startswith("ma"):
        return "min"
    if "^" in s or "maj" in sl or "ma" in sl:
        return "maj"
    if "7" in s or "9" in s or "13" in s or "11" in s:
        return "dom"
    return "maj"


def quality_family_strict(label: str) -> str:
    """Family from a full chart label, disambiguating minor cleanly."""
    s = re.sub(r"^[A-G][#b]?", "", label or "")
    if s.startswith("-"):
        return "min"
    if "h" in s.lower() or "ø" in s or ("o" in s.lower() and "^" not in s):
        return "dim"
    if "^" in s or "maj" in s.lower():
        return "maj"
    if "7" in s or "9" in s or "13" in s or "11" in s:
        return "dom"
    return "maj"


# ─────────────────────────────────────────────────────────────────────────────
# Load chart sections
# ─────────────────────────────────────────────────────────────────────────────
def load_chart(chart_json: Path, beats_per_bar: int = 4):
    """Return (all_chords, section_instances).

    Each chord dict gains: root_pc, fam, m (in-song metric bar position).
    section_instances: list of {label, bar_lo, bar_hi, chords, m0}.
    """
    data = json.loads(Path(chart_json).read_text())
    chords = data["chords"]

    # in-bar metric offset: spread k chords in a bar evenly
    from collections import Counter

    per_bar = Counter(c["bar"] for c in chords)
    seen: dict[int, int] = {}
    for c in chords:
        bar = c["bar"]
        j = seen.get(bar, 0)
        seen[bar] = j + 1
        k = per_bar[bar]
        c["m"] = bar + j / k
        c["root_pc"] = label_root_pc(c["label"])
        c["fam"] = quality_family_strict(c["label"])

    # Split into section instances. Group by contiguous runs of the same
    # section label, then subdivide any run longer than `form_len` bars into
    # `form_len`-bar instances (AABC form = 8-bar sections).
    form_len = 8
    instances = []
    run = []
    run_label = None

    def flush(run, label):
        if not run:
            return
        bar_lo = min(c["bar"] for c in run)
        bar_hi = max(c["bar"] for c in run)
        span = bar_hi - bar_lo + 1
        if span > form_len:
            # subdivide into form_len chunks aligned to bar_lo
            n_chunks = int(np.ceil(span / form_len))
            for ci in range(n_chunks):
                lo = bar_lo + ci * form_len
                hi = lo + form_len - 1
                sub = [c for c in run if lo <= c["bar"] <= hi]
                if sub:
                    instances.append(_make_instance(label, sub))
        else:
            instances.append(_make_instance(label, run))

    for c in chords:
        lab = c["section"]
        if lab != run_label and run:
            flush(run, run_label)
            run = []
        run_label = lab
        run.append(c)
    flush(run, run_label)
    return chords, instances


def _make_instance(label, sub):
    bar_lo = min(c["bar"] for c in sub)
    bar_hi = max(c["bar"] for c in sub)
    m0 = min(c["m"] for c in sub)
    return {
        "label": label,
        "bar_lo": bar_lo,
        "bar_hi": bar_hi,
        "chords": sub,
        "m0": m0,
        "n_bars": bar_hi - bar_lo + 1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Load inferred chords from the `const P = {...}` blob in inferred_<slug>.html
# ─────────────────────────────────────────────────────────────────────────────
def load_inferred(inferred_html: Path):
    html = Path(inferred_html).read_text()
    i = html.find("const P")
    if i < 0:
        raise ValueError("no `const P` blob in inferred html")
    eq = html.find("=", i)
    s = html.find("{", eq)
    depth = 0
    j = s
    instr = False
    esc = False
    while j < len(html):
        ch = html[j]
        if instr:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                instr = False
        else:
            if ch == '"':
                instr = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
        j += 1
    P = json.loads(html[s : j + 1])
    units = []
    for u in P["chords"]:
        q = u["lv"]["exact"]["q"]
        conf = u["lv"]["exact"]["c"]
        units.append(
            {
                "t0": u["t0"],
                "t1": u["t1"],
                "root": u["root"],
                "fam": quality_family(q),
                "q": q,
                "conf": conf,
            }
        )
    return P, units


# ─────────────────────────────────────────────────────────────────────────────
# Rasterise inferred sequence onto a fine time grid for fast overlap scoring
# ─────────────────────────────────────────────────────────────────────────────
class InferredRaster:
    def __init__(self, units, hop=0.1):
        self.hop = hop
        t_end = max(u["t1"] for u in units)
        n = int(np.ceil(t_end / hop)) + 1
        self.n = n
        self.root = np.full(n, -1, dtype=int)
        self.fam = np.empty(n, dtype=object)
        self.fam[:] = None
        self.conf = np.zeros(n, dtype=float)
        for u in units:
            a = int(round(u["t0"] / hop))
            b = int(round(u["t1"] / hop))
            b = max(b, a + 1)
            self.root[a:b] = u["root"]
            self.fam[a:b] = u["fam"]
            self.conf[a:b] = u["conf"]

    def slice(self, ta, tb):
        a = int(round(ta / self.hop))
        b = int(round(tb / self.hop))
        a = max(a, 0)
        b = min(max(b, a + 1), self.n)
        return a, b


def chord_prox(chart_root, chart_fam, inf_root, inf_fam):
    """0..1 proximity. Roots primary; quality a tiebreak."""
    if chart_root is None or inf_root < 0:
        return 0.0
    if inf_root != chart_root:
        return 0.0
    if inf_fam == chart_fam:
        return 1.0
    # family-adjacent partial credit (e.g. maj vs dom share the major third)
    adj = {("maj", "dom"), ("dom", "maj"), ("min", "dim"), ("dim", "min")}
    if (inf_fam, chart_fam) in adj:
        return 0.8
    return 0.4


def score_placement(inst, raster, anchor, slope, offset):
    """Confidence-weighted proximity of a section placed at (anchor, slope).

    Each chart chord occupies [t, t+slope*width]; we integrate overlap*conf*prox
    over the inferred raster and normalise by overlap*conf. Returns
    (score in 0..1, coverage 0..1, per-chord scores)."""
    chords = inst["chords"]
    m0 = inst["m0"]
    # chord widths in bars: gap to next chord, last chord = 1 bar
    ms = [c["m"] for c in chords]
    widths = []
    for idx in range(len(ms)):
        if idx + 1 < len(ms):
            widths.append(max(ms[idx + 1] - ms[idx], 0.01))
        else:
            widths.append(1.0)
    num = 0.0
    den = 0.0
    per_chord = []
    for c, w in zip(chords, widths):
        ta = anchor + slope * (c["m"] - m0)
        tb = ta + slope * w
        a, b = raster.slice(ta, tb)
        seg_conf = raster.conf[a:b]
        seg_root = raster.root[a:b]
        seg_fam = raster.fam[a:b]
        if len(seg_conf) == 0:
            per_chord.append(0.0)
            continue
        cnum = 0.0
        cden = 0.0
        for k in range(len(seg_conf)):
            ir = seg_root[k]
            if ir < 0:
                continue
            ir_shift = (ir + offset) % 12
            p = chord_prox(c["root_pc"], c["fam"], ir_shift, seg_fam[k])
            wgt = seg_conf[k]
            cnum += wgt * p
            cden += wgt
        cs = (cnum / cden) if cden > 0 else 0.0
        per_chord.append(cs)
        num += cnum
        den += cden
    score = (num / den) if den > 0 else 0.0
    coverage = np.mean([1.0 if p > 0 else 0.0 for p in per_chord]) if per_chord else 0.0
    return score, coverage, per_chord


# ─────────────────────────────────────────────────────────────────────────────
# Global transposition offset from a trusted anchor section
# ─────────────────────────────────────────────────────────────────────────────
def estimate_offset(inst, raster, anchor, slope):
    best = (0, -1.0)
    scores = {}
    for off in range(12):
        s, _, _ = score_placement(inst, raster, anchor, slope, off)
        scores[off] = s
        if s > best[1]:
            best = (off, s)
    return best[0], scores


# ─────────────────────────────────────────────────────────────────────────────
# Per-section grid search
# ─────────────────────────────────────────────────────────────────────────────
def search_section(inst, raster, offset, bpm_prior, beats_per_bar,
                   win_lo, win_hi, prior=None, prior_sigma_bars=4.0):
    """Grid-search (anchor, slope) for one section within a LOCAL audio window
    [win_lo, win_hi] (the anchor of the section's first chord). slope in s/bar
    = beats_per_bar*60/bpm. Coarse pass then local refine. Returns best fit.

    The window is a regional prior (from the section's gt-align onset): it keeps
    the search near where the section actually plays, so the repeated A theme and
    the solo section don't pull the fit to a spurious global match."""
    span_bars = inst["n_bars"]
    # per-section tempo may differ (e.g. a half-time A); search 0.45..1.5x prior
    bpms = np.linspace(bpm_prior * 0.45, bpm_prior * 1.5, 48)
    slopes = beats_per_bar * 60.0 / bpms
    audio_end = raster.n * raster.hop
    win_lo = max(0.0, win_lo)
    win_hi = max(win_lo + 0.25, win_hi)
    anchors = np.arange(win_lo, win_hi, 0.25)
    sig_s = prior_sigma_bars * (beats_per_bar * 60.0 / bpm_prior)

    def onset_prior(anchor):
        # soft MAP prior toward the section's trusted gt-align onset; resolves
        # the plateau ambiguity when the same theme repeats back-to-back.
        if prior is None:
            return 1.0
        return float(np.exp(-0.5 * ((anchor - prior) / sig_s) ** 2))

    # fallback placement so `best` is never None
    best = {"obj": -1.0, "anchor": float(win_lo),
            "slope": float(beats_per_bar * 60.0 / bpm_prior),
            "score": 0.0, "coverage": 0.0}
    for slope in slopes:
        max_anchor = audio_end - slope * (span_bars - 1) - 0.5
        for anchor in anchors:
            if anchor > max_anchor:
                continue
            s, cov, _ = score_placement(inst, raster, anchor, slope, offset)
            obj = s * (0.5 + 0.5 * cov) * onset_prior(anchor)
            if obj > best["obj"]:
                best = {"obj": obj, "anchor": float(anchor), "slope": float(slope),
                        "score": float(s), "coverage": float(cov)}
    # local refine around best
    if best is not None:
        a0 = best["anchor"]
        sl0 = best["slope"]
        for anchor in np.arange(a0 - 0.3, a0 + 0.3, 0.05):
            for slope in np.linspace(sl0 * 0.92, sl0 * 1.08, 17):
                if anchor < 0:
                    continue
                s, cov, per = score_placement(inst, raster, anchor, slope, offset)
                obj = s * (0.5 + 0.5 * cov) * onset_prior(anchor)
                if obj > best["obj"]:
                    best = {"obj": obj, "anchor": float(anchor), "slope": float(slope),
                            "score": float(s), "coverage": float(cov)}
    # If chord-proxy is too weak to trust the tempo, the fitted slope is noise
    # (short-span overfit). Fall back to the prior tempo at the fitted onset so
    # a flagged section still yields a sane grid; the low score keeps it flagged.
    CLEAN = 0.45
    if best["score"] < CLEAN:
        best["slope"] = float(beats_per_bar * 60.0 / bpm_prior)
        best["tempo_fallback"] = True
    else:
        best["tempo_fallback"] = False
    # final per-chord + residual
    s, cov, per = score_placement(inst, raster, best["anchor"], best["slope"], offset)
    best["per_chord"] = per
    best["bpm"] = float(beats_per_bar * 60.0 / best["slope"])
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Main alignment
# ─────────────────────────────────────────────────────────────────────────────
def align_sections_to_audio(chart_json, inferred_html, bpm_prior,
                            beats_per_bar=4, anchor_bar0=None, anchor_slope=None):
    chords, instances = load_chart(chart_json, beats_per_bar)
    P, units = load_inferred(inferred_html)
    raster = InferredRaster(units, hop=0.1)
    audio_end = raster.n * raster.hop

    # --- global offset from the first (trusted, hand-corrected) instance ---
    a0 = instances[0]
    # anchor/slope for offset estimation come from the trusted A-section GT if
    # provided, else from the instance's own gt-align t0.
    if anchor_bar0 is None:
        anchor_bar0 = a0["chords"][0]["t0"]
    if anchor_slope is None:
        anchor_slope = beats_per_bar * 60.0 / bpm_prior
    offset, off_scores = estimate_offset(a0, raster, anchor_bar0, anchor_slope)

    # --- per-section search in a LOCAL window around each section's gt-align
    # onset. The chart maps to a bounded region of the audio (the head); the
    # gt-align t0 of each section's first chord is a reliable regional prior,
    # and chord-proxy matching refines onset + fits tempo within +/- WIN_BARS.
    WIN_BARS = 5.0  # search radius around the regional prior, in bars
    slope_guess = beats_per_bar * 60.0 / bpm_prior
    results = []
    prev_end = 0.0
    for idx, inst in enumerate(instances):
        prior = inst["chords"][0].get("t0")
        if prior is None:
            prior = prev_end
        win_lo = max(prev_end - 1.0 * slope_guess, prior - WIN_BARS * slope_guess)
        win_hi = prior + WIN_BARS * slope_guess
        # NOTE: the Gaussian onset prior is intentionally disabled (prior=None).
        # The whole point of chord-proxy matching is to override the unreliable
        # DTW bar-ordering; the local window (centered on the gt-align onset)
        # already prevents runaway to a different chorus, so we let chord-proxy
        # pick the onset freely inside it. Set `prior=prior` to re-enable a soft
        # MAP pull toward the gt-align onset for songs with trustworthy anchors.
        fit = search_section(inst, raster, offset, bpm_prior, beats_per_bar,
                             win_lo, win_hi, prior=None)
        # section audio window
        m_span = inst["bar_hi"] - inst["m0"] + 1.0  # bars from m0 to end of last bar
        t_start = fit["anchor"]
        t_end = fit["anchor"] + fit["slope"] * (inst["bar_hi"] - inst["m0"] + 1.0)
        fit.update({
            "idx": idx,
            "label": inst["label"],
            "bar_lo": inst["bar_lo"],
            "bar_hi": inst["bar_hi"],
            "t_start": float(t_start),
            "t_end": float(t_end),
            "n_bars": inst["n_bars"],
        })
        results.append(fit)
        prev_end = t_end

    # --- vamp detection between sections ---
    vamps = []
    for i in range(len(results) - 1):
        gap = results[i + 1]["t_start"] - results[i]["t_end"]
        # bar length at this boundary (use following section's tempo)
        bar_s = results[i + 1]["slope"]
        if gap > 0.5 * bar_s:
            vamps.append({
                "after_idx": i,
                "t_start": float(results[i]["t_end"]),
                "t_end": float(results[i + 1]["t_start"]),
                "dur_s": float(gap),
                "dur_bars": float(gap / bar_s),
                "kind": "inter-section vamp",
            })
    # leading vamp (before first section) and trailing
    if results and results[0]["t_start"] > 0.5 * results[0]["slope"]:
        vamps.insert(0, {
            "after_idx": -1,
            "t_start": 0.0,
            "t_end": float(results[0]["t_start"]),
            "dur_s": float(results[0]["t_start"]),
            "dur_bars": float(results[0]["t_start"] / results[0]["slope"]),
            "kind": "intro",
        })
    if results and audio_end - results[-1]["t_end"] > 2 * results[-1]["slope"]:
        vamps.append({
            "after_idx": len(results) - 1,
            "t_start": float(results[-1]["t_end"]),
            "t_end": float(audio_end),
            "dur_s": float(audio_end - results[-1]["t_end"]),
            "dur_bars": float((audio_end - results[-1]["t_end"]) / results[-1]["slope"]),
            "kind": "outro/coda",
        })

    # --- assemble final grid: per-chord t0_perfect from its section fit ---
    # map each chart chord to the instance that owns its bar
    bar_to_fit = {}
    for fit, inst in zip(results, instances):
        for c in inst["chords"]:
            bar_to_fit[id(c)] = (fit, inst)
    SCORE_CLEAN = 0.45  # section score above which we treat data as trainable
    out_chords = []
    for c in chords:
        fit, inst = bar_to_fit[id(c)]
        t0p = fit["anchor"] + fit["slope"] * (c["m"] - inst["m0"])
        # width to next chord
        t1p = t0p + fit["slope"] * 1.0
        is_vamp = fit["score"] < SCORE_CLEAN or fit["coverage"] < 0.4
        out_chords.append({
            "bar": c["bar"],
            "beat": c["beat"],
            "section": c["section"],
            "label": c["label"],
            "root_pc": c["root_pc"],
            "family": c["fam"],
            "m": c["m"],
            "t0_orig": c.get("t0"),
            "t0_perfect": round(t0p, 4),
            "t1_perfect": round(t1p, 4),
            "section_id": fit["idx"],
            "tempo_fit": round(fit["bpm"], 2),
            "match_score": round(fit["score"], 3),
            "is_vamp": bool(is_vamp),
        })

    payload = {
        "source_chart": str(chart_json),
        "source_inferred": str(inferred_html),
        "bpm_prior": bpm_prior,
        "beats_per_bar": beats_per_bar,
        "global_transpose_offset": offset,
        "offset_scores": {str(k): round(v, 3) for k, v in off_scores.items()},
        "audio_end_s": round(audio_end, 2),
        "sections": [
            {
                "idx": r["idx"], "label": r["label"],
                "bar_lo": r["bar_lo"], "bar_hi": r["bar_hi"],
                "t_start": round(r["t_start"], 3), "t_end": round(r["t_end"], 3),
                "bpm_fit": round(r["bpm"], 2),
                "slope_s_per_bar": round(r["slope"], 4),
                "match_score": round(r["score"], 3),
                "coverage": round(r["coverage"], 3),
                "per_chord_score": [round(x, 2) for x in r["per_chord"]],
                "is_vamp_flagged": bool(r["score"] < SCORE_CLEAN or r["coverage"] < 0.4),
            }
            for r in results
        ],
        "vamps": vamps,
        "chords": out_chords,
    }
    return payload, results, instances, units, offset, off_scores, raster


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic HTML
# ─────────────────────────────────────────────────────────────────────────────
def render_html(payload, results, instances, units, offset, out_path):
    audio_end = payload["audio_end_s"]
    W = 1180
    PADL = 60
    scale = (W - PADL - 20) / audio_end

    def x(t):
        return PADL + t * scale

    SECFILL = {"A": "#4c8dff", "B": "#ff9f43", "C": "#2dd4a8", "D": "#c678dd"}

    # inferred lane: colour by root match to nearest section (after offset)
    inf_rects = []
    for u in units:
        x0 = x(u["t0"])
        w = max(x(u["t1"]) - x0, 1.0)
        shade = 0.25 + 0.75 * min(u["conf"], 1.0)
        inf_rects.append(
            f'<rect x="{x0:.1f}" y="0" width="{w:.1f}" height="26" '
            f'fill="rgba(150,160,180,{shade:.2f})" stroke="#0b0f16" stroke-width="0.3">'
            f'<title>{u["t0"]:.1f}s  {NAMES[(u["root"]+offset)%12]}{u["q"]}  (raw {NAMES[u["root"]]}, c={u["conf"]:.2f})</title></rect>'
        )

    # section blocks + per-chord onset ticks
    sec_rects = []
    chord_ticks = []
    for r, inst in zip(results, instances):
        x0 = x(r["t_start"])
        x1 = x(r["t_end"])
        fill = SECFILL.get(r["label"], "#888")
        flagged = r["score"] < 0.45 or r["coverage"] < 0.4
        op = 0.30 if not flagged else 0.12
        stroke = fill if not flagged else "#ff5c5c"
        sec_rects.append(
            f'<rect x="{x0:.1f}" y="0" width="{max(x1-x0,2):.1f}" height="34" '
            f'rx="4" fill="{fill}" fill-opacity="{op}" stroke="{stroke}" stroke-width="1.5">'
            f'<title>{r["label"]} bars {r["bar_lo"]}-{r["bar_hi"]}  '
            f'{r["t_start"]:.1f}-{r["t_end"]:.1f}s  {r["bpm"]:.0f} BPM  '
            f'score={r["score"]:.2f} cov={r["coverage"]:.2f}</title></rect>'
        )
        sec_rects.append(
            f'<text x="{x0+4:.1f}" y="15" fill="{fill}" font-size="12" '
            f'font-weight="700">{r["label"]}</text>'
            f'<text x="{x0+4:.1f}" y="29" fill="#9aa4b2" font-size="9">'
            f'{r["bpm"]:.0f}bpm · {r["score"]:.2f}</text>'
        )
        m0 = inst["m0"]
        for c, ps in zip(inst["chords"], r["per_chord"]):
            t = r["anchor"] + r["slope"] * (c["m"] - m0)
            col = "#2dd4a8" if ps >= 0.8 else ("#ffd166" if ps >= 0.4 else "#ff5c5c")
            chord_ticks.append(
                f'<line x1="{x(t):.1f}" y1="0" x2="{x(t):.1f}" y2="34" '
                f'stroke="{col}" stroke-width="1.4" stroke-opacity="0.9">'
                f'<title>{c["label"]} @bar{c["bar"]}  t={t:.1f}s  prox={ps:.2f}</title></line>'
            )

    # vamp bands
    vamp_rects = []
    for v in payload["vamps"]:
        x0 = x(v["t_start"])
        w = max(x(v["t_end"]) - x0, 2)
        vamp_rects.append(
            f'<rect x="{x0:.1f}" y="-4" width="{w:.1f}" height="46" '
            f'fill="url(#vamp)" stroke="#ff5c5c" stroke-dasharray="3 3" stroke-width="1">'
            f'<title>{v["kind"]}: {v["t_start"]:.1f}-{v["t_end"]:.1f}s '
            f'({v["dur_bars"]:.1f} bars)</title></rect>'
        )
        vamp_rects.append(
            f'<text x="{x0+3:.1f}" y="-8" fill="#ff8080" font-size="9">vamp {v["dur_bars"]:.1f}b</text>'
        )

    # time axis ticks
    axis = []
    for t in range(0, int(audio_end) + 1, 30):
        axis.append(
            f'<line x1="{x(t):.1f}" y1="0" x2="{x(t):.1f}" y2="6" stroke="#556" stroke-width="1"/>'
            f'<text x="{x(t):.1f}" y="18" fill="#667" font-size="9" text-anchor="middle">{t}s</text>'
        )

    # section table
    rows = []
    for s in payload["sections"]:
        flag = "⚠ vamp/uncertain" if s["is_vamp_flagged"] else "clean"
        cls = "flag" if s["is_vamp_flagged"] else "ok"
        rows.append(
            f'<tr class="{cls}"><td>{s["idx"]}</td><td><b>{s["label"]}</b></td>'
            f'<td>{s["bar_lo"]}–{s["bar_hi"]}</td>'
            f'<td>{s["t_start"]:.1f}–{s["t_end"]:.1f}s</td>'
            f'<td>{s["bpm_fit"]:.0f}</td><td>{s["match_score"]:.2f}</td>'
            f'<td>{s["coverage"]:.2f}</td><td>{flag}</td></tr>'
        )
    off_str = ", ".join(f'{NAMES[int(k)]}:{v:.2f}' for k, v in
                        sorted(payload["offset_scores"].items(), key=lambda kv: -kv[1])[:4])

    html = f"""<div class="wrap">
<h1>Autumn Leaves — section-wise rigid-tempo alignment</h1>
<p class="sub">Each chart section (A/B/C) is fit as its own constant-tempo block and located in
the recording by matching the model's <b>inferred</b> chord sequence (chord-proxy matching).
Global transposition offset detected: <b>+{offset} semitones</b> (inferred output is a whole
tone flat; top offsets by anchor match: {off_str}). Audio length {audio_end:.0f}s.</p>

<div class="legend">
<span><i style="background:#2dd4a8"></i> exact/near chord match</span>
<span><i style="background:#ffd166"></i> root-only match</span>
<span><i style="background:#ff5c5c"></i> no match / flagged</span>
<span><i style="background:rgba(150,160,180,.7)"></i> inferred chord (opacity=confidence)</span>
<span><i style="background:repeating-linear-gradient(45deg,#ff5c5c33,#ff5c5c33 3px,transparent 3px,transparent 6px)"></i> vamp region</span>
</div>

<svg width="{W}" height="150" style="overflow:visible">
  <defs>
    <pattern id="vamp" width="6" height="6" patternTransform="rotate(45)" patternUnits="userSpaceOnUse">
      <rect width="6" height="6" fill="rgba(255,92,92,0.05)"/>
      <line x1="0" y1="0" x2="0" y2="6" stroke="rgba(255,92,92,0.25)" stroke-width="2"/>
    </pattern>
  </defs>
  <text x="0" y="20" fill="#9aa4b2" font-size="10">chart</text>
  <g transform="translate(0,10)">{''.join(vamp_rects)}{''.join(sec_rects)}{''.join(chord_ticks)}</g>
  <text x="0" y="78" fill="#9aa4b2" font-size="10">audio</text>
  <g transform="translate(0,60)">{''.join(inf_rects)}</g>
  <g transform="translate(0,100)">{''.join(axis)}</g>
</svg>

<h2>Section fits</h2>
<table>
<tr><th>#</th><th>sec</th><th>bars</th><th>audio window</th><th>BPM</th><th>score</th><th>cov</th><th>status</th></tr>
{''.join(rows)}
</table>

<h2>Vamp / unaligned regions</h2>
<table>
<tr><th>kind</th><th>window</th><th>bars</th></tr>
{''.join(f'<tr><td>{v["kind"]}</td><td>{v["t_start"]:.1f}–{v["t_end"]:.1f}s</td><td>{v["dur_bars"]:.1f}</td></tr>' for v in payload["vamps"]) or '<tr><td colspan=3>none</td></tr>'}
</table>
</div>

<style>
  .wrap {{ font-family: -apple-system, system-ui, sans-serif; max-width: 1220px;
    margin: 0 auto; padding: 24px; color: #d6dbe4; background: #0b0f16; }}
  h1 {{ font-size: 20px; margin: 0 0 4px; }}
  h2 {{ font-size: 15px; margin: 28px 0 8px; color:#aab4c4; }}
  .sub {{ color:#8b95a7; font-size: 12.5px; line-height:1.5; max-width: 900px; }}
  .legend {{ display:flex; gap:16px; flex-wrap:wrap; font-size:11px; color:#9aa4b2; margin:14px 0; }}
  .legend i {{ display:inline-block; width:11px; height:11px; border-radius:2px; vertical-align:-1px; margin-right:4px; }}
  svg {{ margin: 20px 0; max-width:100%; }}
  table {{ border-collapse: collapse; font-size: 12px; margin-top:6px; }}
  th, td {{ text-align:left; padding: 4px 12px; border-bottom: 1px solid #1c2430; }}
  th {{ color:#7f8ba0; font-weight:600; }}
  tr.flag td {{ color:#ff9d9d; }}
  tr.ok td {{ color:#bfe8d6; }}
</style>
"""
    Path(out_path).write_text(html)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chart", required=True)
    ap.add_argument("--inferred", required=True)
    ap.add_argument("--bpm-prior", type=float, default=181.0)
    ap.add_argument("--beats-per-bar", type=int, default=4)
    ap.add_argument("--anchor-bar0", type=float, default=None,
                    help="known audio onset (s) of the first chord, for offset seed")
    ap.add_argument("--anchor-bpm", type=float, default=None,
                    help="known BPM of the trusted first section, for offset seed")
    ap.add_argument("--out", required=True)
    ap.add_argument("--diag-html", required=True)
    args = ap.parse_args()

    anchor_slope = None
    if args.anchor_bpm:
        anchor_slope = args.beats_per_bar * 60.0 / args.anchor_bpm

    payload, results, instances, units, offset, off_scores, raster = \
        align_sections_to_audio(
            args.chart, args.inferred, args.bpm_prior,
            beats_per_bar=args.beats_per_bar,
            anchor_bar0=args.anchor_bar0, anchor_slope=anchor_slope,
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    render_html(payload, results, instances, units, offset, args.diag_html)

    # console summary
    print(f"global transpose offset: +{offset} semitones")
    print(f"{'sec':>3} {'bars':>7} {'window':>16} {'bpm':>6} {'score':>6} {'cov':>5}  status")
    for s in payload["sections"]:
        st = "VAMP?" if s["is_vamp_flagged"] else "clean"
        print(f'{s["label"]:>3} {s["bar_lo"]:>3}-{s["bar_hi"]:<3} '
              f'{s["t_start"]:>6.1f}-{s["t_end"]:<6.1f} {s["bpm_fit"]:>6.0f} '
              f'{s["match_score"]:>6.2f} {s["coverage"]:>5.2f}  {st}')
    print(f'vamps: {len(payload["vamps"])}')
    for v in payload["vamps"]:
        print(f'  {v["kind"]:>18}: {v["t_start"]:.1f}-{v["t_end"]:.1f}s ({v["dur_bars"]:.1f} bars)')
    print(f"\nwrote {args.out}\nwrote {args.diag_html}")


if __name__ == "__main__":
    main()
