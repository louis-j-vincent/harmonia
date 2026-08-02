"""Annotation sidecars for harmonia_min — persist confirmed chords (P1).

The shell POSTs every chord lock to ``/api/annotations/<file>`` and never
GETs them back, so rehydration must happen server-side inside
``/api/chart-model/<file>``: :func:`overlay` stamps the stored corrections
onto the ChartModel before it is served, and the shell cannot tell a
rehydrated confirmation from a fresh one.

On-disk shape follows ``docs/annotation_sidecar_schema.md`` (schema 1,
one file per chart) with the divergences the live payload forces — the
payload wins:

* the shell sends ``t0``/``t1`` per chord (undocumented, and load-bearing:
  a split's second half has no chord to inherit timing from);
* it sends no ``old``/``ts``;
* it **hardcodes ``bass:-1``**, so ``-1`` here means "no opinion, keep the
  model's bass", NOT "no bass". Treating it as an assertion would delete
  the ``/B`` from every slash chord the moment a user confirms it (This
  Love bar 0 is G/B) — see tests/test_harmonia_min_annotations.py.

What this does NOT solve:

* ``(bar, beat)`` is not unique — folded sections replay one bar identity
  (Norah Jones ``LA``, Stand By Me ``LB``). A lock is applied to EVERY
  rendered copy, deliberately: the copies are the same folded material,
  and applying it to one only would desync the repeats. A user who wants
  two folded passes to differ cannot express that here.
* Last-write-wins, no merge of concurrent writers, no undo history.
* ``merges`` are stored and echoed back opaquely; nothing interprets them.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

PKG = Path(__file__).resolve().parent
ANNOT_DIR = PKG / "state" / "annotations"

SCHEMA = 1
_BEAT_TOL = 3  # decimals; the schema warns `beat` may be float


def _path(file_key: str) -> Path:
    return ANNOT_DIR / f"{Path(file_key).stem}.json"


def _key(bar, beat) -> tuple:
    try:
        return int(bar), round(float(beat or 0), _BEAT_TOL)
    except (TypeError, ValueError):
        return bar, beat


def empty_doc(file_key: str) -> dict:
    return {"schema": SCHEMA, "chart": Path(file_key).stem, "annotator": "",
            "modified": None, "chords": [], "merges": []}


def load_annotation(file_key: str) -> dict:
    try:
        doc = json.loads(_path(file_key).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return empty_doc(file_key)
    doc.setdefault("chords", [])
    doc.setdefault("merges", [])
    return doc


def save_annotation(file_key: str, doc: dict) -> dict:
    """Stamp and write the sidecar atomically; returns the stamped doc.

    Serialisation happens BEFORE the file is touched, and the write is
    tmp+``os.replace``, so a bad payload or a crash mid-write leaves the
    previous good sidecar intact (the old app's plain ``write_text`` could
    truncate it).
    """
    out = dict(doc)
    out["schema"] = SCHEMA
    out["chart"] = Path(file_key).stem
    out["modified"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out.setdefault("annotator", "")
    out.setdefault("chords", [])
    out.setdefault("merges", [])
    payload = json.dumps(out, indent=2)  # raises before any file is opened

    ANNOT_DIR.mkdir(parents=True, exist_ok=True)
    dest = _path(file_key)
    tmp = dest.with_name(dest.name + f".tmp{os.getpid()}")
    try:
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)
    return out


def delete_annotation(file_key: str) -> None:
    _path(file_key).unlink(missing_ok=True)


def _apply(chord: dict, fix: dict) -> None:
    chord["root"] = int(fix["root"]) % 12
    chord["q"] = fix.get("q", "")
    chord["c"] = 1.0
    chord["confirmed"] = True
    chord["nc"] = False
    chord["carry"] = False
    if fix.get("t0") is not None:
        chord["t0"] = fix["t0"]
    if fix.get("t1") is not None:
        chord["t1"] = fix["t1"]
    bass = fix.get("bass", -1)
    if bass is not None and int(bass) >= 0:
        chord["bass"] = int(bass)  # an explicit bass is an assertion
    # bass < 0 means "no opinion": keep whatever the model inferred.


def _synthesize(template: dict, fix: dict) -> dict:
    ch = dict(template)
    ch["beat"] = fix.get("beat", 0)
    ch.pop("sug", None)
    ch["n_obs"] = 1
    ch["folded"] = False
    _apply(ch, fix)
    if fix.get("bass", -1) is None or int(fix.get("bass", -1)) < 0:
        ch["bass"] = -1  # a synthesized chord inherits no slash from its host
    return ch


def overlay(model: dict, ann: dict | None) -> dict:
    """Stamp stored corrections onto a ChartModel, in place.

    Confirmed chords come back with ``confirmed=True`` and ``c=1.0`` so the
    shell renders the ✓ badge exactly as it does right after a lock.
    """
    ann = ann or {}
    model["merges"] = ann.get("merges", []) or []
    fixes = {}
    for fix in ann.get("chords", []) or []:
        if fix.get("root") is None:
            continue
        fixes[_key(fix.get("bar"), fix.get("beat"))] = fix
    if not fixes:
        return model

    consumed: set[tuple] = set()
    for sec in model.get("sections", []) or []:
        for bar in sec.get("bars", []) or []:
            for chord in bar:
                fix = fixes.get(_key(chord.get("bar"), chord.get("beat")))
                if fix is not None:
                    _apply(chord, fix)
                    consumed.add(_key(chord.get("bar"), chord.get("beat")))

    leftover = [f for k, f in fixes.items() if k not in consumed]
    if not leftover:
        return model

    # Split halves: a lock on the second half of a split bar has a beat the
    # model has never seen. Insert it into EVERY rendered copy of that bar
    # (see the folding note in the module docstring).
    for fix in leftover:
        try:
            bar_no = int(fix.get("bar"))
        except (TypeError, ValueError):
            continue
        for sec in model.get("sections", []) or []:
            for bar in sec.get("bars", []) or []:
                host = next((c for c in bar if c.get("bar") == bar_no), None)
                if host is None:
                    continue
                bar.append(_synthesize(host, fix))
                bar.sort(key=lambda c: float(c.get("beat") or 0))
    return model
