#!/usr/bin/env python3
"""Mission 5 — LLM-based priors for the Bayesian chord decoder.

WHAT THIS IS
------------
An LLM "analyst" reads a song (its iReal chart + metadata, optionally a compact
audio-derived summary) and emits *priors* for the existing joint / semi-Markov
decoder (``harmonia/models/joint_decode.py``, ``semi_markov_decode.py``):

    { structure, chord_priors P(q|root,pos), transition_priors P(root|prev_root),
      confidence }

The LLM supplies *intuition* (it has memorised thousands of standards and their
functional harmony); the Bayesian decoder keeps *control* (exact MAP Viterbi,
quantified uncertainty). The priors enter through the SAME factor interface the
user-input constraints already use (``harmonia/models/user_constraints.py``) —
the LLM is treated as an automated annotator with LOWER authority than a human
(smaller emission bonus, gated by its own self-reported confidence). See
``docs/mission_5_bayesian_integration.md``.

HONESTY (per CLAUDE.md rules #2, #3)
------------------------------------
* The LLM does NOT hear the waveform. Claude has no audio modality here; its
  input is the SYMBOLIC iReal chart + title/style + (optionally) a compact
  numeric summary of our own audio front-end (detected key, tempo, per-beat
  root posterior). Autumn-Leaves-style standards are the strong case — the
  model recognises the tune. A blind unknown-audio case is weaker and the
  confidence it returns should (and does) reflect that.
* Priors derived from the iReal chart are only a fair *end-to-end audio* test
  when the audio is a DIFFERENT source than the chart (trust hierarchy: iReal >
  tabs > model). Scoring an audio decode that was seeded from the same tune's
  chart against that chart is partly circular — ``eval_llm_priors.py`` guards
  this by measuring convergence/robustness, not chart-agreement.

RUNTIME
-------
* With ``anthropic`` installed + credentials resolvable (``ANTHROPIC_API_KEY``
  or ``ant auth login``): calls ``claude-opus-4-8`` with adaptive thinking and a
  strict JSON schema (structured outputs). This is the real LLM path.
* Otherwise: a deterministic OFFLINE analyst parses the chart with music-theory
  rules and emits the identical schema. This makes the prototype runnable and
  unit-testable in this repo with no network, and doubles as a cheap baseline
  the LLM must beat.

Usage:
    python scripts/llm_chord_priors.py --song "Autumn Leaves"
    python scripts/llm_chord_priors.py --song "Autumn Leaves" --offline
    python scripts/llm_chord_priors.py --song "Autumn Leaves" --out priors.json
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

MODEL_ID = "claude-opus-4-8"  # per the claude-api skill: default, do not downgrade

# ── pitch-class + quality vocabulary (kept local; mirrors ireal_corpus) ────────
_NOTE_TO_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
Q5_NAMES = ("maj", "min", "dom", "hdim", "dim")  # matches semi_markov_decode.Q5_NAMES
PC_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]


def note_to_pc(note: str) -> int | None:
    if not note:
        return None
    pc = _NOTE_TO_PC.get(note[0].upper())
    if pc is None:
        return None
    for acc in note[1:]:
        if acc == "#":
            pc += 1
        elif acc == "b":
            pc -= 1
        else:
            break
    return pc % 12


def ireal_quality_to_q5(qual: str) -> str:
    """Coarse (maj/min/dom/hdim/dim) family of an iReal quality suffix.

    Mirrors the fam_of_bucket collapse used to fit the progression bigram: dom7
    is a 'major-family' *degree* but its own q5 is dom; ø/m7b5 is hdim.
    """
    q = qual.strip()
    if q in ("h", "h7", "h9") or "b5" in q and q.startswith(("-7", "m7", "h")):
        return "hdim"
    if q.startswith(("o", "dim")) and "^" not in q:  # o, o7, dim, dim7
        return "dim"
    if q.startswith(("-", "m")):
        # minor family (incl. m7, m6, m9, mM7). m7b5 already caught above.
        return "min"
    if q.startswith(("^", "6", "69", "M", "add", "2", "5")) or q == "":
        return "maj"
    # 7, 9, 11, 13, 7alt, 7b9, 7sus, +, aug7 … → dominant
    return "dom"


def parse_chord_token(tok: str) -> tuple[int, str] | None:
    """'C-7' -> (pc=0, q5='min'); 'D7b13' -> (2,'dom'). Returns None for 'x'/rests."""
    tok = tok.strip()
    if not tok or tok in ("x", "n", "N", "|", "p"):
        return None
    if tok[0] not in _NOTE_TO_PC:
        return None
    i = 1
    if i < len(tok) and tok[i] in "#b":
        i += 1
    pc = note_to_pc(tok[:i])
    if pc is None:
        return None
    qual = tok[i:].split("/")[0]  # drop slash-bass
    return pc, ireal_quality_to_q5(qual)


# ── chart loading ──────────────────────────────────────────────────────────────
@dataclass
class SongChart:
    title: str
    style: str
    key: str
    sections: list[tuple[str, list[tuple[int, str]]]]  # (label, [(pc,q5),...]) per bar
    raw_measures: list[tuple[str, str]]                 # (label, raw_token) per bar


def chart_from_tune(tune) -> SongChart:
    """Build a SongChart from an already-parsed ireal_corpus Tune.

    Factored out of :func:`load_chart` so callers that already hold a Tune (e.g.
    the cross-source eval, which loads two versions of the same title from
    different playlists) can build a chart without re-globbing a playlist.
    """
    from harmonia.data.ireal_corpus import sectionized_measures

    secm = sectionized_measures(tune)
    bars: list[tuple[str, list[tuple[int, str]]]] = []
    raw: list[tuple[str, str]] = []
    for label, measure in secm:
        raw.append((label, measure))
        # a measure may hold >1 chord, e.g. 'G-7Gb7' — split on note-starts
        toks, cur = [], ""
        for ch in measure:
            if ch in _NOTE_TO_PC and cur and cur[-1] not in "#b":
                toks.append(cur)
                cur = ch
            else:
                cur += ch
        if cur:
            toks.append(cur)
        parsed = [p for p in (parse_chord_token(t) for t in toks) if p]
        bars.append((label, parsed))
    return SongChart(tune.title, tune.style or "", tune.key or "", bars, raw)


def load_chart(title: str, playlist: Path) -> SongChart:
    from harmonia.data.ireal_corpus import load_playlist

    with contextlib.redirect_stdout(io.StringIO()):
        tunes = load_playlist(playlist)
    matches = [t for t in tunes if title.lower() in t.title.lower()]
    if not matches:
        raise SystemExit(f"'{title}' not found in {playlist}")
    return chart_from_tune(matches[0])


# ── the analysis SCHEMA both paths emit ────────────────────────────────────────
# structure.sections: list of {label, start_bar, end_bar} (bars 1-indexed)
# structure.repeats:   list of lists of section-indices asserted identical
# chord_priors:        {"<pc>": {"maj":p, "min":p, "dom":p, "hdim":p, "dim":p}}
#                      per-root marginal quality prior (position-agnostic form v1)
# transition_priors:   {"<prev_pc>": {"<next_pc>": p}}  (row-normalised)
# confidence:          float 0..1 (LLM's self-assessed reliability)

ANALYSIS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "key": {"type": "string"},
        "mode": {"type": "string", "enum": ["major", "minor"]},
        "tonic_pc": {"type": "integer"},
        "structure": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "form": {"type": "string"},
                "sections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "label": {"type": "string"},
                            "start_bar": {"type": "integer"},
                            "end_bar": {"type": "integer"},
                        },
                        "required": ["label", "start_bar", "end_bar"],
                    },
                },
                "repeats": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}},
            },
            "required": ["form", "sections", "repeats"],
        },
        "chord_priors": {"type": "object"},
        "transition_priors": {"type": "object"},
        "confidence": {"type": "number"},
        "notes": {"type": "string"},
    },
    "required": ["key", "mode", "tonic_pc", "structure", "chord_priors",
                 "transition_priors", "confidence"],
}


# ── OFFLINE analyst (rule-based; runnable with no network) ──────────────────────
def _key_to_tonic(key: str) -> tuple[int, str]:
    key = key.strip()
    minor = key.endswith("-") or key.lower().endswith("m")
    root = key.rstrip("-mM ").strip() or "C"
    pc = note_to_pc(root)
    return (pc if pc is not None else 0), ("minor" if minor else "major")


def offline_analyze(chart: SongChart) -> dict:
    tonic_pc, mode = _key_to_tonic(chart.key)

    # sections (contiguous runs of the same label) + repeat groups (same label set)
    sections: list[dict] = []
    bar = 1
    runs: list[tuple[str, int, int]] = []
    for label, _ in chart.sections:
        if runs and runs[-1][0] == label:
            runs[-1] = (label, runs[-1][1], bar)
        else:
            runs.append((label, bar, bar))
        bar += 1
    for i, (label, s, e) in enumerate(runs):
        sections.append({"label": label, "start_bar": s, "end_bar": e})
    # Parallelism detection (P3 pooling premise). Two signals, both conservative:
    #   (a) exact adjacent block repeat — the largest L with bars[i:i+L] ==
    #       bars[i+L:i+2L] token-for-token (catches an 8-bar strain played twice,
    #       which iReal labels with one contiguous 'A' run and run-merging hides).
    #   (b) two whole sections with an identical bar-token sequence.
    # Emitted as sub-section spans so the decoder can pool matching slots.
    raw_tokens = [m for _, m in chart.raw_measures]
    span_groups = _detect_repeat_spans(raw_tokens)  # [[(s,e),(s,e)], ...] 1-indexed
    repeats: list[list[int]] = []
    for group in span_groups:
        idxs = []
        for s, e in group:
            sections.append({"label": f"rep{len(sections)}", "start_bar": s, "end_bar": e})
            idxs.append(len(sections) - 1)
        repeats.append(idxs)

    # P(q | root): observed quality distribution per root, Laplace-smoothed.
    per_root = defaultdict(Counter)
    flat = [c for _, bar_chords in chart.sections for c in bar_chords]
    for pc, q5 in flat:
        per_root[pc][q5] += 1
    chord_priors: dict[str, dict[str, float]] = {}
    for pc in range(12):
        cnt = per_root.get(pc)
        if not cnt:
            continue
        tot = sum(cnt.values()) + 0.5 * len(Q5_NAMES)
        chord_priors[str(pc)] = {q: (cnt.get(q, 0) + 0.5) / tot for q in Q5_NAMES}

    # P(root | prev_root): observed adjacent-root transitions, Laplace-smoothed.
    trans = defaultdict(Counter)
    prev = None
    for pc, _ in flat:
        if prev is not None and pc != prev:
            trans[prev][pc] += 1
        prev = pc
    transition_priors: dict[str, dict[str, float]] = {}
    for pr, cnt in trans.items():
        tot = sum(cnt.values())
        if tot:
            transition_priors[str(pr)] = {str(k): v / tot for k, v in cnt.items()}

    # Confidence: a rule-based read of a clean, fully-diatonic standard chart is
    # trustworthy but not omniscient. Scale it down when the chart is chromatic
    # (many non-diatonic roots) — that's where the offline heuristic is weakest.
    scale = _diatonic_pcs(tonic_pc, mode)
    roots = [pc for pc, _ in flat]
    diat_frac = sum(1 for pc in roots if pc in scale) / max(len(roots), 1)
    confidence = round(0.55 + 0.30 * diat_frac, 3)  # 0.55..0.85

    form = " ".join(
        f"{lab}{e - s + 1}" for lab, s, e in runs
    )
    return {
        "key": chart.key or PC_NAMES[tonic_pc] + (" minor" if mode == "minor" else " major"),
        "mode": mode,
        "tonic_pc": tonic_pc,
        "structure": {"form": form, "sections": sections, "repeats": repeats},
        "chord_priors": chord_priors,
        "transition_priors": transition_priors,
        "confidence": confidence,
        "notes": "offline rule-based analyst (no LLM); diatonic-fraction gated confidence",
    }


def _detect_repeat_spans(bars: list[str]) -> list[list[tuple[int, int]]]:
    """Find adjacent, token-identical bar blocks -> parallel spans (1-indexed).

    Greedy, left-to-right: at each position take the LARGEST block length L such
    that bars[i:i+L] == bars[i+L:i+2L], emit both as a repeat group, and skip
    past them. Deliberately conservative (exact match only) — pooling on a
    non-identical repeat was Gen-1 Candidate C and it hurt (known_issues #1).
    """
    n = len(bars)
    # Expand 'x'/empty repeat-bars by carrying the previous bar forward, so a
    # strain ending 'G-6 | x' matches an identical strain regardless of which
    # copy iReal happened to expand.
    norm: list[str] = []
    for b in bars:
        c = b.replace(" ", "")
        if c in ("", "x", "n", "p") and norm:
            c = norm[-1]
        norm.append(c)
    groups: list[list[tuple[int, int]]] = []
    i = 0
    while i < n:
        best = 0
        max_l = (n - i) // 2
        for L in range(max_l, 1, -1):
            if norm[i:i + L] == norm[i + L:i + 2 * L] and any(norm[i:i + L]):
                best = L
                break
        if best:
            groups.append([(i + 1, i + best), (i + best + 1, i + 2 * best)])
            i += 2 * best
        else:
            i += 1
    return groups


def _diatonic_pcs(tonic: int, mode: str) -> set[int]:
    steps = [0, 2, 3, 5, 7, 8, 10] if mode == "minor" else [0, 2, 4, 5, 7, 9, 11]
    return {(tonic + s) % 12 for s in steps}


# ── LLM analyst (real path; claude-opus-4-8, structured output) ─────────────────
def _build_brief(chart: SongChart, audio_summary: dict | None) -> str:
    lines = [
        f"Song: {chart.title}",
        f"Style: {chart.style}",
        f"iReal key signature: {chart.key}",
        "",
        "iReal chart (one bar per line, 'label: token(s)'; 'x' = repeat previous bar):",
    ]
    for label, measure in chart.raw_measures:
        lines.append(f"  {label}: {measure}")
    if audio_summary:
        lines += ["", "Audio front-end summary (our pipeline, may be noisy):",
                  json.dumps(audio_summary, indent=1)]
    lines += [
        "",
        "You are a jazz-harmony analyst producing PRIORS for a Bayesian chord",
        "decoder. Return, as strict JSON matching the provided schema:",
        "  - tonic_pc (0=C..11=B) and mode; key as a string.",
        "  - structure.sections + structure.repeats: group section indices that",
        "    are harmonically the SAME (so the decoder can pool their evidence).",
        "    Only assert a repeat you are confident is identical.",
        "  - chord_priors: for each root pitch-class present, P(quality) over",
        "    (maj,min,dom,hdim,dim) — your expectation of the quality at that root",
        "    in THIS key (e.g. V is likely dom7, vi likely min).",
        "  - transition_priors: P(next_root | prev_root) over pitch-classes,",
        "    reflecting the tune's functional motion (ii->V->I etc.).",
        "  - confidence in [0,1]: be honest. High if you recognise the tune and",
        "    the harmony is unambiguous; low for an unfamiliar/ambiguous chart.",
        "Probabilities in each distribution must sum to ~1.",
    ]
    return "\n".join(lines)


def llm_analyze(chart: SongChart, audio_summary: dict | None) -> dict:
    import anthropic  # raises ImportError → caller falls back to offline

    client = anthropic.Anthropic()  # resolves ANTHROPIC_API_KEY or an ant profile
    brief = _build_brief(chart, audio_summary)
    resp = client.messages.create(
        model=MODEL_ID,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        output_config={"effort": "high",
                       "format": {"type": "json_schema", "schema": ANALYSIS_SCHEMA}},
        system=("You are an expert jazz theorist and Bayesian-model designer. "
                "You output only valid JSON priors; you are calibrated and honest "
                "about uncertainty."),
        messages=[{"role": "user", "content": brief}],
    )
    text = next(b.text for b in resp.content if b.type == "text")
    return json.loads(text)


def analyze(chart: SongChart, *, offline: bool = False,
            audio_summary: dict | None = None) -> tuple[dict, str]:
    """Return (analysis_dict, path_used)."""
    if not offline:
        try:
            return llm_analyze(chart, audio_summary), "llm"
        except ImportError:
            print("[llm_chord_priors] anthropic SDK not installed — offline analyst",
                  file=sys.stderr)
        except Exception as exc:  # noqa: BLE001 — any API/auth failure → offline
            print(f"[llm_chord_priors] LLM path failed ({type(exc).__name__}: {exc}) "
                  "— offline analyst", file=sys.stderr)
    return offline_analyze(chart), "offline"


# ── translation to Bayesian decoder factors ────────────────────────────────────
@dataclass
class BayesianFactors:
    """Priors mapped onto the joint/semi-Markov decoder's factor interface.

    These are consumed by ``docs/mission_5_bayesian_integration.md``'s glue; the
    field names mirror ``joint_decode.joint_decode`` / ``infer_chords_v1``.
    """
    tonic: int                                   # -> joint_decode(tonic=...)
    mode: str
    pool_group_bars: list[list[tuple[int, int]]]  # bar-spans asserted parallel
    quality_bonus: dict[int, dict[int, float]]    # root -> {q5_idx: +nats}
    root_transition_bias: dict[int, dict[int, float]]  # prev_root -> {root:+nats}
    strength: float                               # global prior strength in nats
    confidence: float


def to_bayesian_factors(analysis: dict, *, max_nats: float = 8.0) -> BayesianFactors:
    """LLM confidence -> prior STRENGTH (nats). Honest-about-what-it-knows.

    A real user confirm is CLAMP_NATS≈40 (user_constraints.py). The LLM is an
    automated annotator, so its ceiling is far lower (default 8 nats) and it is
    further scaled by the LLM's self-reported confidence — a low-confidence
    analysis barely tilts the decoder; a high-confidence one meaningfully shapes
    it but never pins it (the acoustic evidence + transition factor can still
    overrule).
    """
    conf = float(analysis.get("confidence", 0.5))
    strength = max_nats * conf

    q_bonus: dict[int, dict[int, float]] = {}
    for pc_s, dist in analysis.get("chord_priors", {}).items():
        pc = int(pc_s)
        row = {}
        for q5_idx, qn in enumerate(Q5_NAMES):
            p = float(dist.get(qn, 0.0))
            # additive log-bonus proportional to how much prob mass the prior puts
            # on this quality, centred so the mean quality gets ~0.
            row[q5_idx] = strength * (p - 1.0 / len(Q5_NAMES))
        q_bonus[pc] = row

    r_bias: dict[int, dict[int, float]] = {}
    for pr_s, dist in analysis.get("transition_priors", {}).items():
        pr = int(pr_s)
        row = {}
        for nx_s, p in dist.items():
            row[int(nx_s)] = strength * float(p)  # log-boost toward expected motion
        r_bias[pr] = row

    pools: list[list[tuple[int, int]]] = []
    sects = analysis.get("structure", {}).get("sections", [])
    for group in analysis.get("structure", {}).get("repeats", []):
        spans = []
        for idx in group:
            if 0 <= idx < len(sects):
                spans.append((int(sects[idx]["start_bar"]), int(sects[idx]["end_bar"])))
        if len(spans) > 1:
            pools.append(spans)

    return BayesianFactors(
        tonic=int(analysis.get("tonic_pc", 0)),
        mode=analysis.get("mode", "major"),
        pool_group_bars=pools,
        quality_bonus=q_bonus,
        root_transition_bias=r_bias,
        strength=strength,
        confidence=conf,
    )


def _factors_summary(f: BayesianFactors) -> dict:
    return {
        "tonic": f.tonic, "tonic_name": PC_NAMES[f.tonic], "mode": f.mode,
        "confidence": f.confidence, "strength_nats": round(f.strength, 2),
        "n_pool_groups": len(f.pool_group_bars),
        "pool_group_bars": f.pool_group_bars,
        "n_roots_with_quality_prior": len(f.quality_bonus),
        "n_roots_with_transition_bias": len(f.root_transition_bias),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--song", default="Autumn Leaves")
    ap.add_argument("--playlist", default=str(REPO / "data" / "ireal" / "jazz1460.txt"))
    ap.add_argument("--offline", action="store_true",
                    help="force the rule-based analyst (skip the LLM path)")
    ap.add_argument("--out", default=None, help="write analysis JSON here")
    args = ap.parse_args()

    chart = load_chart(args.song, Path(args.playlist))
    analysis, path = analyze(chart, offline=args.offline)
    factors = to_bayesian_factors(analysis)

    print(f"# LLM chord priors — {chart.title}  (analyst path: {path})")
    print(json.dumps(analysis, indent=1))
    print("\n# → Bayesian decoder factors")
    print(json.dumps(_factors_summary(factors), indent=1))

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"analysis": analysis, "factors_summary": _factors_summary(factors)},
            indent=1))
        print(f"\nwrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
