#!/usr/bin/env python3
"""Error-analysis over human chord corrections logged by the annotation UI.

Reads every ``*.json`` under ``data/training_logs/<song>/`` — each file pairs the
model's original reading with the human fix and the ``/api/reinfer`` propagation
diff (written by ``/api/correction-log`` in ``scripts/harmonia_server.py``) — and
aggregates the corrections to surface systematic model weaknesses.

Aggregations
------------
- **Quality confusion**  predicted chord-quality -> corrected quality (which
  qualities does the model get wrong most, and what does it confuse them with).
- **Root confusion**     predicted root -> corrected root (are some roots harder;
  is there a consistent transposition error, e.g. a semitone / 5th slip).
- **Context**            the neighbouring chords that changed in the reinfer diff
  (best-effort: the log stores one correction per file, so the diff spans are the
  only local-context signal available).
- **Confidence bias**    corrections the model made while *confident*
  (``original_prediction.confidence > FALSE_CONF_THRESHOLD``) — false confidence.
- **Propagation**        ``benefit.propagation_count`` (a.k.a. reinfer
  ``n_chords_changed``): high-impact errors corrupt several downstream chords.

Outputs
-------
- ``docs/error_analysis_<timestamp>.md``  — versioned, one per run.
- ``docs/error_analysis_report.md``        — canonical "latest" copy.
- ``docs/error_analysis_dashboard.html``   — self-contained visual dashboard.

Usage
-----
    python scripts/analyze_corrections.py [--output-format md|html|both]
                                          [--logs-dir DIR] [--docs-dir DIR]
                                          [--false-conf-threshold F]
                                          [--high-impact-threshold N]

Idempotent: each run writes a fresh timestamped report; re-running never mutates
prior runs. Empty / missing / malformed log files are skipped with a warning.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_LOGS_DIR = REPO / "data" / "training_logs"
DEFAULT_DOCS_DIR = REPO / "docs"

FALSE_CONF_THRESHOLD = 0.70   # confidence above which "wrong" == false confidence
HIGH_IMPACT_THRESHOLD = 3     # propagation_count >= this == high-impact error

SEMITONE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTE_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


# --------------------------------------------------------------------------- #
# Chord-label parsing. Two notations coexist in the logs:                      #
#   * harmonia labels ("G:hdim7", "A#:maj", "C:7")  -> predictions & diffs     #
#   * iReal labels    ("Ah7", "A#^7", "C-7", "G-7b5") -> human corrections     #
# parse_chord() handles both and returns (root_pc, normalized_quality).        #
# --------------------------------------------------------------------------- #

def _normalize_quality(tail: str) -> str:
    """Map a raw quality tail (harmonia or iReal) onto a canonical family."""
    t = (tail or "").strip()
    if t == "":
        return "maj"
    low = t.lower()

    # half-diminished must be tested before plain minor / dim.
    if low in ("hdim7", "hdim", "m7b5", "min7b5", "-7b5", "h7", "h", "ø", "ø7"):
        return "hdim7"
    if low in ("dim7", "o7", "°7"):
        return "dim7"
    if low in ("dim", "o", "°"):
        return "dim"
    if low in ("maj7", "major7", "ma7", "m7maj", "^7", "^", "maj", "major"):
        # ^ alone in iReal is a major triad; ^7 is maj7.
        return "maj7" if ("7" in low) else "maj"
    if low in ("min7", "minor7", "-7", "m7"):
        return "min7"
    if low in ("min", "minor", "-", "m"):
        return "min"
    if low in ("7", "dom7", "dominant7"):
        return "7"
    if low in ("aug", "+", "aug7", "+7"):
        return "aug"
    if low.startswith("sus"):
        return "sus"
    if low in ("6", "maj6"):
        return "maj6"
    if low in ("m6", "min6", "-6"):
        return "min6"
    if low in ("9", "11", "13"):
        return "7"          # extended dominants collapse to the dominant family
    if low.startswith("^") or low.startswith("maj"):
        return "maj7"
    if low.startswith("-") or low.startswith("m"):
        return "min7"
    return low              # unknown tail: keep verbatim so it stays visible


def parse_chord(label):
    """Return (root_pc:int|None, quality:str, ok:bool) for a chord label.

    Accepts both "ROOT:quality" (harmonia) and "ROOT<qual>" (iReal) forms.
    Returns ok=False when the string is empty / unparseable.
    """
    if not label:
        return None, "", False
    s = str(label).strip()
    if not s:
        return None, "", False

    # harmonia "ROOT:quality"
    if ":" in s:
        root_str, _, tail = s.partition(":")
        root_str = root_str.strip()
    else:
        # iReal: letter + optional accidental, remainder is the quality tail.
        m = re.match(r"^([A-Ga-g])([#b]?)(.*)$", s)
        if not m:
            return None, "", False
        root_str = m.group(1).upper() + m.group(2)
        tail = m.group(3)

    m = re.match(r"^([A-Ga-g])([#b]?)$", root_str)
    if not m:
        return None, _normalize_quality(tail), False
    pc = NOTE_PC[m.group(1).upper()]
    if m.group(2) == "#":
        pc = (pc + 1) % 12
    elif m.group(2) == "b":
        pc = (pc + 11) % 12
    return pc, _normalize_quality(tail), True


def pc_name(pc):
    return SEMITONE_NAMES[pc % 12] if pc is not None else "?"


def interval_name(delta):
    """Human-readable name for a root-motion interval (0..11 semitones)."""
    names = {
        0: "same root", 1: "+1 semitone", 2: "+2 (whole tone)", 3: "+3 (min 3rd)",
        4: "+4 (maj 3rd)", 5: "+5 (perfect 4th)", 6: "+6 (tritone)",
        7: "+7 (perfect 5th)", 8: "-4 (min 6th)", 9: "-3 (maj 6th)",
        10: "-2 (whole tone down)", 11: "-1 semitone",
    }
    return names.get(delta % 12, f"+{delta % 12}")


# --------------------------------------------------------------------------- #
# Loading                                                                      #
# --------------------------------------------------------------------------- #

class Correction:
    """A flattened, parsed view of one correction log record."""

    __slots__ = ("song", "timestamp", "session", "bar", "beat",
                 "pred_label", "pred_root", "pred_q", "confidence",
                 "corr_label", "corr_root", "corr_q",
                 "propagation", "n_changed", "diff", "reinfer_error",
                 "self_corrected", "path")

    @property
    def root_changed(self):
        return (self.pred_root is not None and self.corr_root is not None
                and self.pred_root != self.corr_root)

    @property
    def quality_changed(self):
        return bool(self.pred_q) and bool(self.corr_q) and self.pred_q != self.corr_q

    @property
    def false_confidence(self):
        return self.confidence is not None and self.confidence > FALSE_CONF_THRESHOLD

    @property
    def high_impact(self):
        p = self.propagation if self.propagation is not None else (self.n_changed or 0)
        return (p or 0) >= HIGH_IMPACT_THRESHOLD


def load_corrections(logs_dir: Path):
    """Load & parse every correction log. Returns (corrections, warnings)."""
    corrections, warnings = [], []
    if not logs_dir.exists():
        warnings.append(f"logs dir does not exist: {logs_dir}")
        return corrections, warnings

    files = sorted(logs_dir.rglob("*.json"))
    if not files:
        warnings.append(f"no .json correction logs found under {logs_dir}")
        return corrections, warnings

    for path in files:
        try:
            raw = path.read_text(encoding="utf-8").strip()
            if not raw:
                warnings.append(f"empty file skipped: {path.name}")
                continue
            rec = json.loads(raw)
        except (json.JSONDecodeError, OSError) as exc:
            warnings.append(f"unreadable/invalid JSON skipped: {path.name} ({exc})")
            continue

        if not isinstance(rec, dict):
            warnings.append(f"non-object record skipped: {path.name}")
            continue

        pred = rec.get("original_prediction") or {}
        corr = rec.get("human_correction") or {}
        rr = rec.get("reinfer_result") or {}
        ben = rec.get("benefit") or {}

        pred_label = pred.get("chord")
        corr_label = corr.get("chord")
        if not pred_label or not corr_label:
            warnings.append(f"missing chord field skipped: {path.name}")
            continue

        p_root, p_q, _ = parse_chord(pred_label)
        c_root, c_q, _ = parse_chord(corr_label)

        # root/q may be provided explicitly (spec format); prefer them if present.
        if isinstance(pred.get("root"), int):
            p_root = pred["root"] % 12
        if pred.get("q"):
            p_q = _normalize_quality(pred["q"])
        if isinstance(corr.get("root"), int):
            c_root = corr["root"] % 12
        if corr.get("q"):
            c_q = _normalize_quality(corr["q"])

        conf = pred.get("confidence")
        conf = float(conf) if isinstance(conf, (int, float)) else None

        c = Correction()
        c.song = rec.get("song") or path.parent.name
        c.timestamp = rec.get("timestamp") or ""
        c.session = rec.get("human_session") or ""
        c.bar = pred.get("bar")
        c.beat = pred.get("beat")
        c.pred_label, c.pred_root, c.pred_q = pred_label, p_root, p_q
        c.corr_label, c.corr_root, c.corr_q = corr_label, c_root, c_q
        c.confidence = conf
        prop = ben.get("propagation_count")
        c.propagation = int(prop) if isinstance(prop, (int, float)) else None
        nch = rr.get("n_chords_changed")
        c.n_changed = int(nch) if isinstance(nch, (int, float)) else None
        c.diff = rr.get("diff") or []
        c.reinfer_error = rr.get("error")
        c.self_corrected = bool(ben.get("self_corrected"))
        c.path = path
        corrections.append(c)

    return corrections, warnings


# --------------------------------------------------------------------------- #
# Analysis                                                                     #
# --------------------------------------------------------------------------- #

def analyze(corrections):
    n = len(corrections)
    a = {
        "n": n,
        "songs": Counter(c.song for c in corrections),
        "pred_quality": Counter(),      # predicted quality that got corrected
        "pred_root": Counter(),         # predicted root that got corrected
        "quality_confusion": Counter(), # (pred_q -> corr_q)
        "root_confusion": Counter(),    # (pred_root -> corr_root) names
        "root_interval": Counter(),     # semitone motion pred->corr
        "false_confidence": [],
        "high_impact": [],
        "context": Counter(),           # diff old->new label pairs (local context)
        "confidences": [],
        "root_error_count": 0,
        "quality_error_count": 0,
    }
    for c in corrections:
        if c.pred_q:
            a["pred_quality"][c.pred_q] += 1
        if c.pred_root is not None:
            a["pred_root"][pc_name(c.pred_root)] += 1
        if c.quality_changed:
            a["quality_error_count"] += 1
            a["quality_confusion"][(c.pred_q, c.corr_q)] += 1
        if c.root_changed:
            a["root_error_count"] += 1
            a["root_confusion"][(pc_name(c.pred_root), pc_name(c.corr_root))] += 1
            a["root_interval"][(c.corr_root - c.pred_root) % 12] += 1
        if c.confidence is not None:
            a["confidences"].append(c.confidence)
        if c.false_confidence:
            a["false_confidence"].append(c)
        if c.high_impact:
            a["high_impact"].append(c)
        for d in c.diff:
            old, new = d.get("old_label"), d.get("new_label")
            if old and new and old != new:
                a["context"][(old, new)] += 1

    a["high_impact"].sort(
        key=lambda c: (c.propagation if c.propagation is not None else (c.n_changed or 0)),
        reverse=True)
    a["false_confidence"].sort(
        key=lambda c: (c.confidence or 0), reverse=True)
    return a


def systematic_patterns(a):
    """Heuristic, human-readable weakness statements from the aggregates."""
    out = []
    n = a["n"]
    if n == 0:
        return out

    # dominant quality confusions
    for (pq, cq), k in a["quality_confusion"].most_common(6):
        pct = 100.0 * k / n
        out.append(f"Model predicts **{pq}** but the true chord is **{cq}** "
                    f"({k} case{'s' if k != 1 else ''}, {pct:.0f}% of corrections).")

    # consistent root-motion error (transposition slip)
    if a["root_interval"]:
        (delta, k) = a["root_interval"].most_common(1)[0]
        if delta != 0 and k >= max(2, 0.3 * max(1, a["root_error_count"])):
            out.append(f"Root errors cluster at **{interval_name(delta)}** "
                        f"({k}/{a['root_error_count']} root corrections) — a "
                        f"systematic transposition slip, not random.")

    # false confidence
    fc = len(a["false_confidence"])
    if fc:
        out.append(f"**False confidence:** {fc} correction(s) had model "
                    f"confidence > {FALSE_CONF_THRESHOLD:.2f} yet were wrong — "
                    f"the confidence signal is not trustworthy above that band.")
    elif a["confidences"]:
        hi = max(a["confidences"])
        out.append(f"No false-confidence cases: every correction sat at "
                    f"confidence <= {hi:.2f} (all below the "
                    f"{FALSE_CONF_THRESHOLD:.2f} band). The model was already "
                    f"uncertain where humans corrected it — a well-calibrated sign.")

    # high impact
    hi = len(a["high_impact"])
    if hi:
        out.append(f"**High-impact errors:** {hi} correction(s) each propagated "
                    f"to >= {HIGH_IMPACT_THRESHOLD} chords — fixing these first "
                    f"yields the largest downstream cleanup.")
    return out


# --------------------------------------------------------------------------- #
# Markdown report                                                              #
# --------------------------------------------------------------------------- #

def render_markdown(a, warnings, ts_iso):
    L = []
    w = L.append
    w(f"# Chord-correction error analysis\n")
    w(f"_Generated {ts_iso} by `scripts/analyze_corrections.py`._\n")
    w(f"- **Total corrections logged:** {a['n']}")
    w(f"- **Songs annotated:** {len(a['songs'])} "
      f"({', '.join(f'{s} ({k})' for s, k in a['songs'].most_common())})")
    w(f"- **Corrections that changed the root:** {a['root_error_count']}")
    w(f"- **Corrections that changed the quality:** {a['quality_error_count']}")
    if a["confidences"]:
        mean = sum(a["confidences"]) / len(a["confidences"])
        w(f"- **Mean model confidence on corrected chords:** {mean:.3f} "
          f"(min {min(a['confidences']):.3f}, max {max(a['confidences']):.3f})")
    w("")

    if a["n"] == 0:
        w("> No corrections found. Annotate some songs in the UI, then re-run.\n")
        if warnings:
            w("### Warnings\n")
            for x in warnings:
                w(f"- {x}")
        return "\n".join(L)

    w("## Top model weaknesses\n")
    for i, p in enumerate(systematic_patterns(a), 1):
        w(f"{i}. {p}")
    w("")

    w("## Top 10 most-corrected chord types (predicted quality)\n")
    w("| # | Predicted quality | Corrections |")
    w("|---|---|---|")
    for i, (q, k) in enumerate(a["pred_quality"].most_common(10), 1):
        w(f"| {i} | `{q}` | {k} |")
    w("")

    w("## Top 10 most-corrected roots (predicted root)\n")
    w("| # | Predicted root | Corrections |")
    w("|---|---|---|")
    for i, (r, k) in enumerate(a["pred_root"].most_common(10), 1):
        w(f"| {i} | {r} | {k} |")
    w("")

    w("## Quality confusion (predicted -> corrected)\n")
    w("| Predicted | Corrected | Count |")
    w("|---|---|---|")
    for (pq, cq), k in a["quality_confusion"].most_common(15):
        w(f"| `{pq}` | `{cq}` | {k} |")
    if not a["quality_confusion"]:
        w("| _(none)_ | | |")
    w("")

    w("## Root confusion (predicted -> corrected)\n")
    w("| Predicted | Corrected | Motion | Count |")
    w("|---|---|---|---|")
    for (pr, cr), k in a["root_confusion"].most_common(15):
        # recover interval for display
        d = (NOTE_PC.get(cr[0], 0) - NOTE_PC.get(pr[0], 0)) % 12
        w(f"| {pr} | {cr} | {interval_name(d)} | {k} |")
    if not a["root_confusion"]:
        w("| _(none)_ | | | |")
    w("")

    w(f"## Confidence bias (false confidence: conf > {FALSE_CONF_THRESHOLD:.2f} yet corrected)\n")
    if a["false_confidence"]:
        w("| Song | Bar | Predicted | Corrected | Confidence |")
        w("|---|---|---|---|---|")
        for c in a["false_confidence"][:20]:
            w(f"| {c.song} | {c.bar} | `{c.pred_label}` | `{c.corr_label}` "
              f"| {c.confidence:.3f} |")
    else:
        w("_No false-confidence cases: no correction had confidence above "
          f"{FALSE_CONF_THRESHOLD:.2f}._")
    w("")

    w(f"## High-impact errors (propagation >= {HIGH_IMPACT_THRESHOLD})\n")
    if a["high_impact"]:
        w("| Song | Bar | Predicted | Corrected | Propagation | Reinfer changed |")
        w("|---|---|---|---|---|---|")
        for c in a["high_impact"][:10]:
            p = c.propagation if c.propagation is not None else c.n_changed
            w(f"| {c.song} | {c.bar} | `{c.pred_label}` | `{c.corr_label}` "
              f"| {p} | {c.n_changed} |")
    else:
        w(f"_No correction propagated to >= {HIGH_IMPACT_THRESHOLD} chords._")
    w("")

    w("## Local context (chords that changed in reinfer diffs)\n")
    if a["context"]:
        w("| Old label | -> New label | Times |")
        w("|---|---|---|")
        for (old, new), k in a["context"].most_common(15):
            w(f"| `{old}` | `{new}` | {k} |")
    else:
        w("_No propagation diffs recorded._")
    w("")

    if warnings:
        w("## Warnings\n")
        for x in warnings:
            w(f"- {x}")
        w("")

    w("---")
    w("_Feeds Mission 2 (retrain quality head — target the top confused "
      "qualities), Mission 3 (calibration — fix false-confidence cases), and "
      "future context-failure work._")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# HTML dashboard                                                               #
# --------------------------------------------------------------------------- #

def _js(obj):
    return json.dumps(obj, ensure_ascii=False)


def render_dashboard(a, corrections, ts_iso):
    quality = a["pred_quality"].most_common(12)
    roots = a["pred_root"].most_common(12)
    scatter = [
        {"conf": c.confidence, "pred": c.pred_label, "corr": c.corr_label,
         "song": c.song, "bar": c.bar, "false": bool(c.false_confidence)}
        for c in corrections if c.confidence is not None
    ]
    high = [
        {"song": c.song, "bar": c.bar, "pred": c.pred_label, "corr": c.corr_label,
         "prop": (c.propagation if c.propagation is not None else c.n_changed) or 0,
         "nch": c.n_changed or 0}
        for c in sorted(
            a["high_impact"],
            key=lambda x: (x.propagation if x.propagation is not None else (x.n_changed or 0)),
            reverse=True)[:10]
    ]
    raw = [
        {"song": c.song, "timestamp": c.timestamp, "bar": c.bar,
         "pred_label": c.pred_label, "pred_root": c.pred_root, "pred_q": c.pred_q,
         "corr_label": c.corr_label, "corr_root": c.corr_root, "corr_q": c.corr_q,
         "confidence": c.confidence,
         "propagation": c.propagation, "n_changed": c.n_changed,
         "root_changed": c.root_changed, "quality_changed": c.quality_changed}
        for c in corrections
    ]
    patterns = systematic_patterns(a)

    data = {
        "generated": ts_iso, "n": a["n"],
        "quality": quality, "roots": roots, "scatter": scatter, "high": high,
        "raw": raw, "patterns": patterns,
        "falseThreshold": FALSE_CONF_THRESHOLD,
        "highThreshold": HIGH_IMPACT_THRESHOLD,
        "songs": a["songs"].most_common(),
    }

    return """<title>Harmonia — chord-correction error analysis</title>
<style>
  :root{
    --bg:#0f1216; --panel:#171b21; --ink:#e7ecf3; --muted:#93a0b0;
    --grid:#2a313b; --accent:#5b8cff; --warn:#ff6b6b; --good:#39c07a; --bar:#5b8cff;
  }
  @media (prefers-color-scheme: light){
    :root{ --bg:#f6f8fb; --panel:#fff; --ink:#141a22; --muted:#5a6675;
           --grid:#e3e8ef; --accent:#2f6bff; --warn:#d93838; --good:#149a5b; --bar:#2f6bff; }
  }
  *{box-sizing:border-box}
  body{margin:0;font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
       background:var(--bg);color:var(--ink);padding:24px;}
  h1{font-size:20px;margin:0 0 2px} h2{font-size:15px;margin:0 0 12px}
  .sub{color:var(--muted);margin:0 0 20px;font-size:13px}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px}
  .card{background:var(--panel);border:1px solid var(--grid);border-radius:12px;padding:16px;overflow:hidden}
  .full{grid-column:1/-1}
  .kpis{display:flex;flex-wrap:wrap;gap:14px;margin-bottom:16px}
  .kpi{background:var(--panel);border:1px solid var(--grid);border-radius:10px;padding:12px 16px;min-width:120px}
  .kpi .v{font-size:24px;font-weight:600} .kpi .l{color:var(--muted);font-size:12px}
  ul.pat{margin:0;padding-left:18px} ul.pat li{margin:4px 0}
  table{width:100%;border-collapse:collapse;font-size:13px}
  th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--grid)}
  th{color:var(--muted);font-weight:600}
  code{background:rgba(127,127,127,.14);padding:1px 5px;border-radius:5px;font-size:12px}
  .barrow{display:grid;grid-template-columns:110px 1fr 40px;align-items:center;gap:8px;margin:5px 0}
  .barlbl{font-size:12px;text-align:right;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .bartrack{background:var(--grid);border-radius:6px;height:18px;overflow:hidden}
  .barfill{background:var(--bar);height:100%;border-radius:6px}
  .barval{font-size:12px;color:var(--muted)}
  .btns{display:flex;gap:8px;margin-bottom:16px;flex-wrap:wrap}
  button{background:var(--accent);color:#fff;border:0;border-radius:8px;padding:8px 14px;font-size:13px;cursor:pointer}
  button.ghost{background:transparent;color:var(--accent);border:1px solid var(--accent)}
  svg text{fill:var(--muted);font-size:11px}
  .dot{stroke-width:1.5}
  .empty{color:var(--muted);font-style:italic}
</style>
<h1>Chord-correction error analysis</h1>
<p class="sub" id="sub"></p>
<div class="btns">
  <button onclick="expCSV()">Export raw CSV</button>
  <button class="ghost" onclick="expJSON()">Export raw JSON</button>
</div>
<div class="kpis" id="kpis"></div>
<div class="grid">
  <div class="card full"><h2>Top model weaknesses</h2><ul class="pat" id="patterns"></ul></div>
  <div class="card"><h2>Most-corrected chord types</h2><div id="qbars"></div></div>
  <div class="card"><h2>Most-corrected roots</h2><div id="rbars"></div></div>
  <div class="card full"><h2>Confidence vs. corrected (false-confidence cluster is right of the line)</h2><div id="scatter"></div></div>
  <div class="card full"><h2>High-impact errors (top by propagation)</h2><div id="hitab"></div></div>
</div>
<script>
const D = __DATA__;
document.getElementById('sub').textContent =
  D.n + ' corrections \\u00b7 generated ' + D.generated +
  ' \\u00b7 songs: ' + D.songs.map(s=>s[0]+' ('+s[1]+')').join(', ');

function kpi(v,l){return '<div class="kpi"><div class="v">'+v+'</div><div class="l">'+l+'</div></div>';}
const nFalse = D.scatter.filter(s=>s.false).length;
const nHigh = D.high.length;
document.getElementById('kpis').innerHTML =
  kpi(D.n,'corrections') +
  kpi(nFalse,'false-confidence (>'+D.falseThreshold+')') +
  kpi(nHigh,'high-impact (\\u2265'+D.highThreshold+' prop.)') +
  kpi(D.quality.length,'distinct wrong qualities');

document.getElementById('patterns').innerHTML =
  D.patterns.length ? D.patterns.map(p=>'<li>'+md(p)+'</li>').join('')
                    : '<li class="empty">Not enough data yet.</li>';
function md(s){return s.replace(/\\*\\*(.+?)\\*\\*/g,'<b>$1</b>').replace(/`(.+?)`/g,'<code>$1</code>');}

function bars(el,rows){
  const max = Math.max(1,...rows.map(r=>r[1]));
  el.innerHTML = rows.length ? rows.map(r=>
    '<div class="barrow"><div class="barlbl">'+r[0]+'</div>'+
    '<div class="bartrack"><div class="barfill" style="width:'+(100*r[1]/max)+'%"></div></div>'+
    '<div class="barval">'+r[1]+'</div></div>').join('')
    : '<p class="empty">No data.</p>';
}
bars(document.getElementById('qbars'), D.quality);
bars(document.getElementById('rbars'), D.roots);

// scatter: x = confidence 0..1, y = jittered "was corrected" (all corrected=1)
(function(){
  const W=680,H=220,PADL=44,PADB=30,PADT=10,PADR=14;
  const iw=W-PADL-PADR, ih=H-PADT-PADB;
  const x=c=>PADL+iw*Math.max(0,Math.min(1,c));
  const thr=x(D.falseThreshold);
  let s='<svg viewBox="0 0 '+W+' '+H+'" width="100%" preserveAspectRatio="xMidYMid meet">';
  // axes
  s+='<line x1="'+PADL+'" y1="'+(H-PADB)+'" x2="'+(W-PADR)+'" y2="'+(H-PADB)+'" stroke="var(--grid)"/>';
  for(let t=0;t<=1.0001;t+=0.2){const xx=x(t);
    s+='<line x1="'+xx+'" y1="'+PADT+'" x2="'+xx+'" y2="'+(H-PADB)+'" stroke="var(--grid)"/>';
    s+='<text x="'+xx+'" y="'+(H-PADB+14)+'" text-anchor="middle">'+t.toFixed(1)+'</text>';}
  // threshold line
  s+='<line x1="'+thr+'" y1="'+PADT+'" x2="'+thr+'" y2="'+(H-PADB)+'" stroke="var(--warn)" stroke-dasharray="4 3"/>';
  s+='<text x="'+(thr+4)+'" y="'+(PADT+12)+'" fill="var(--warn)">false-conf > '+D.falseThreshold+'</text>';
  s+='<text x="'+PADL+'" y="'+(PADT+4)+'" text-anchor="end">corrected</text>';
  // points
  D.scatter.forEach((p,i)=>{
    const jx=x(p.conf);
    const jy=PADT+ih*0.35 + (Math.sin(i*2.399)*0.5+0.5)*ih*0.3; // deterministic jitter
    const col=p.false?'var(--warn)':'var(--accent)';
    s+='<circle class="dot" cx="'+jx.toFixed(1)+'" cy="'+jy.toFixed(1)+'" r="5" fill="'+col+'" fill-opacity="0.65" stroke="'+col+'"><title>'+
       p.pred+' \\u2192 '+p.corr+' (conf '+p.conf.toFixed(3)+', '+p.song+' bar '+p.bar+')</title></circle>';
  });
  if(!D.scatter.length) s+='<text x="'+(W/2)+'" y="'+(H/2)+'" text-anchor="middle" class="empty">No confidence values logged.</text>';
  s+='</svg>';
  document.getElementById('scatter').innerHTML=s;
})();

(function(){
  const rows=D.high;
  if(!rows.length){document.getElementById('hitab').innerHTML='<p class="empty">No error propagated to \\u2265'+D.highThreshold+' chords.</p>';return;}
  let h='<table><tr><th>Song</th><th>Bar</th><th>Predicted</th><th>Corrected</th><th>Propagation</th><th>Reinfer changed</th></tr>';
  rows.forEach(r=>{h+='<tr><td>'+r.song+'</td><td>'+r.bar+'</td><td><code>'+r.pred+'</code></td><td><code>'+r.corr+'</code></td><td>'+r.prop+'</td><td>'+r.nch+'</td></tr>';});
  document.getElementById('hitab').innerHTML=h+'</table>';
})();

function download(name,text,type){
  const b=new Blob([text],{type}); const u=URL.createObjectURL(b);
  const a=document.createElement('a'); a.href=u; a.download=name; a.click();
  setTimeout(()=>URL.revokeObjectURL(u),1000);
}
function expJSON(){download('corrections_raw.json',JSON.stringify(D.raw,null,2),'application/json');}
function expCSV(){
  const cols=['song','timestamp','bar','pred_label','pred_root','pred_q','corr_label','corr_root','corr_q','confidence','propagation','n_changed','root_changed','quality_changed'];
  const esc=v=>{if(v==null)return'';v=String(v);return /[",\\n]/.test(v)?'"'+v.replace(/"/g,'""')+'"':v;};
  const lines=[cols.join(',')].concat(D.raw.map(r=>cols.map(c=>esc(r[c])).join(',')));
  download('corrections_raw.csv',lines.join('\\n'),'text/csv');
}
</script>
""".replace("__DATA__", _js(data))


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main(argv=None):
    global FALSE_CONF_THRESHOLD, HIGH_IMPACT_THRESHOLD
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-format", choices=["md", "html", "both"], default="both")
    ap.add_argument("--logs-dir", type=Path, default=DEFAULT_LOGS_DIR)
    ap.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR)
    ap.add_argument("--false-conf-threshold", type=float, default=FALSE_CONF_THRESHOLD)
    ap.add_argument("--high-impact-threshold", type=int, default=HIGH_IMPACT_THRESHOLD)
    args = ap.parse_args(argv)

    FALSE_CONF_THRESHOLD = args.false_conf_threshold
    HIGH_IMPACT_THRESHOLD = args.high_impact_threshold

    corrections, warnings = load_corrections(args.logs_dir)
    for wmsg in warnings:
        print(f"[warn] {wmsg}", file=sys.stderr)

    a = analyze(corrections)
    now = datetime.now(timezone.utc)
    ts_iso = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    ts_file = now.strftime("%Y-%m-%dT%H-%M-%SZ")

    args.docs_dir.mkdir(parents=True, exist_ok=True)
    written = []

    if args.output_format in ("md", "both"):
        md = render_markdown(a, warnings, ts_iso)
        versioned = args.docs_dir / f"error_analysis_{ts_file}.md"
        canonical = args.docs_dir / "error_analysis_report.md"
        versioned.write_text(md, encoding="utf-8")
        canonical.write_text(md, encoding="utf-8")
        written += [versioned, canonical]

    if args.output_format in ("html", "both"):
        html = render_dashboard(a, corrections, ts_iso)
        dash = args.docs_dir / "error_analysis_dashboard.html"
        dash.write_text(html, encoding="utf-8")
        written.append(dash)

    print(f"Analyzed {a['n']} correction(s) from "
          f"{len(a['songs'])} song(s); {len(warnings)} warning(s).")
    for p in written:
        print(f"  wrote {p}")

    # Brief console summary of top-3 weaknesses.
    pats = systematic_patterns(a)
    if pats:
        print("\nTop weaknesses:")
        for i, p in enumerate(pats[:3], 1):
            clean = re.sub(r"[*`]", "", p)
            print(f"  {i}. {clean}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
