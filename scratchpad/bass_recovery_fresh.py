"""bass_recovery_fresh.py — re-run the bass recovery against a FRESH decode.

WHY THIS EXISTS
---------------
``bass_root_recovery.py`` audits ``our_seq`` as stored in the ``ug_score_*.json``,
i.e. the baked ``docs/plots/inferred_*.html``.  ``docs/postmusx_segment_loss.md``
(commit ``957971d``) established that those bakes are STALE: Let It Be's chart is
from 2026-07-21 and predates the music-x-lab ``.lab`` it was compared against by
17 hours; it was rendered by a bare ``infer_chords_v1``, whose defaults then were
``bp48 + nnls24 + segment_source="nnls"`` — **music-x-lab never ran in it**.

So every "the decoder absorbed this chord" claim measured on the baked chart is
really a claim about a *different, older* decoder.  This script re-decodes each
song under ``SHIPPED_CONFIG`` and re-scores, so the recovery is judged against
the pipeline that actually ships.

THE TRAP (documented in postmusx_segment_loss.md, and LIVE in this worktree)
---------------------------------------------------------------------------
``musx_bass.musx_dir()`` accepts a clone that has ``chord_recognition.py`` and
``cache_data/*.sdict``.  But ``musx_redecode._decoder`` additionally needs
``data/<dict>_chord_list.txt``, and this worktree's ``third_party`` copy has no
``data/`` at all.  Result: the re-decode fails to a *warning*,
``_used_redecode`` stays False, the Occam gate flips ON, and Occam compresses
129 spans to 59 — two runs of "the shipped config" differing only by an env var
gave 114 vs 120 chords.  :func:`_resolve_musx_dir` therefore refuses to run
unless it finds a clone with BOTH, and :func:`decode` asserts afterwards that the
re-decode really happened.  A silent fallback here would invalidate the whole
comparison (CLAUDE.md rule #1 / #6).

WHAT THIS DOES NOT SOLVE
------------------------
* The UG side is reused from the stored JSON (``ug_seq``, ``nomusic_spans``,
  ``t_intro``).  That is sound for the tab alignment itself — ``ug_align`` fits
  the TAB to the AUDIO chroma and never reads our chart — but ``t_intro`` is
  computed from both sides, so it is held fixed at the stored value rather than
  recomputed.  A shifted intro boundary would move a handful of errors in or out
  of the unscored zone.
* The stored JSON does not carry the UG side's audio-support value, which
  ``find_anchors`` consults; it is reconstructed as neutral.  The replay is
  checked against the published counts and any song that fails the check is
  reported as unfaithful rather than quietly averaged in.
* Decoding is cached per song under ``scratchpad/fresh_charts/``; delete that
  directory after changing the pipeline or the cache will hide the change.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRATCH = REPO / "scratchpad"
FRESH = SCRATCH / "fresh_charts"
AUDIO = REPO / "docs" / "audio"
sys.path.insert(0, str(REPO))


def _resolve_musx_dir() -> Path:
    """A clone that satisfies BOTH musx_dir() and _decoder(), or die loudly."""
    cands = []
    if os.environ.get("HARMONIA_MUSX_DIR"):
        cands.append(Path(os.environ["HARMONIA_MUSX_DIR"]))
    cands += [
        REPO / "harmonia" / "third_party" / "ISMIR2019-Large-Vocabulary-Chord-Recognition",
        Path("/Users/vincente/Documents/Projets Perso/Code/harmonia/harmonia/"
             "third_party/ISMIR2019-Large-Vocabulary-Chord-Recognition"),
    ]
    for d in cands:
        if ((d / "chord_recognition.py").exists()
                and list((d / "cache_data").glob("*.sdict"))
                and (d / "data" / "submission_chord_list.txt").exists()):
            return d
    raise SystemExit(
        "no COMPLETE music-x-lab clone found (need chord_recognition.py, "
        "cache_data/*.sdict AND data/submission_chord_list.txt). Running "
        "without it silently disables the re-decode and flips the Occam gate — "
        "see docs/postmusx_segment_loss.md.")


os.environ["HARMONIA_MUSX_DIR"] = str(_resolve_musx_dir())

from harmonia.eval.accuracy_score import SHIPPED_CONFIG, _decode_to_wav  # noqa: E402


def load_mod(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


BRR = load_mod("bass_root_recovery", SCRATCH / "bass_root_recovery.py")
UG = BRR.UG


#: Substrings that mean "this pipeline quietly became a different pipeline".
#: EVERY one of these has actually fired in this worktree or is documented as
#: having fired before:
#:   * "heads missing"   -> harmonia/models/nnls24_heads.npz is gitignored, so a
#:                          fresh worktree has NO trained heads and chord_head
#:                          degrades to a ONE-CHORD chart for the whole song
#:                          ("C:maj", 243 s, confidence 0.0).  It looks like a
#:                          decode.  It parses.  It scores.  It is nothing.
#:   * "re-decode unavailable" -> the Occam-gate flip in postmusx_segment_loss.md
#: The lesson is the same each time (CLAUDE.md #1): the failure mode of this
#: pipeline is a WARNING plus a plausible-looking output, so a comparison has to
#: assert its inputs rather than eyeball them.
_DEGRADED = ("missing", "unavailable", "fallback", "failed", "degrade")


class _CatchDegradation(logging.Handler):
    """Trip on any warning that means the shipped pipeline did not really run."""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.hits: list[str] = []

    def emit(self, rec):
        msg = str(rec.getMessage())
        if any(k in msg.lower() for k in _DEGRADED):
            self.hits.append(f"{rec.name}: {msg}")


def decode(slug: str) -> list[dict]:
    """SHIPPED_CONFIG chords for one song, cached to scratchpad/fresh_charts."""
    FRESH.mkdir(parents=True, exist_ok=True)
    cache = FRESH / f"{slug}.json"
    if cache.exists():
        return json.loads(cache.read_text())

    import harmonia.models.chord_pipeline_v1 as P
    src = AUDIO / f"{slug}.m4a"
    if not src.exists():
        raise FileNotFoundError(src)
    wav = _decode_to_wav(src, FRESH / "_wav")
    h = _CatchDegradation()
    logging.getLogger().addHandler(h)
    logging.getLogger().setLevel(logging.INFO)
    try:
        chart = P.infer_chords_v1(wav, cache_dir=REPO / "data" / "cache",
                                  **SHIPPED_CONFIG)
    finally:
        logging.getLogger().removeHandler(h)
    if h.hits:
        raise SystemExit(f"{slug}: pipeline degraded — {h.hits[0]!r}. Refusing "
                         "to score a silently different pipeline.")
    ch = getattr(chart, "chords", None)
    chords = list(ch if ch is not None else chart.get("chords", []))
    # Second, independent guard: a degenerate decode is the observable symptom
    # even if nobody logged anything.  A 240 s song is never one chord.
    if len(chords) < 5:
        raise SystemExit(f"{slug}: decode returned {len(chords)} chord(s) — "
                         "degenerate, refusing to score it.")
    out = [{"t0": float(c.get("start_s", 0.0)), "t1": float(c.get("end_s", 0.0)),
            "label": str(c.get("label", "")),
            "conf": float(c.get("confidence", 0.5))} for c in chords]
    cache.write_text(json.dumps(out, indent=1))
    return out


#: music-x-lab / mir_eval Harte quality -> the iReal token ``ug_score`` speaks.
#: ``ug_score.family``/``triad_core`` read iReal spellings, so a Harte "maj7"
#: passed through untranslated would be parsed as min-maj7 and silently invent
#: QUALITY errors.
_HARTE_TO_IREAL = {
    "maj": "", "min": "-", "7": "7", "maj7": "^7", "min7": "-7",
    "dim": "o", "dim7": "o7", "hdim7": "h7", "aug": "+", "sus2": "sus2",
    "sus4": "sus", "7sus4": "7sus", "maj6": "6", "min6": "-6",
    "minmaj7": "-^7", "9": "9", "min9": "-9", "maj9": "^9",
    "11": "11", "13": "13",
}


def to_rq(lab: str) -> "tuple[int | None, str]":
    """A live chord label -> (root pc, iReal quality token). ``(None, "N")`` for
    no-chord.  Handles both Harte (``C:maj7/E``) and iReal (``C^7``) spellings
    because the pipeline emits Harte on the musx path and iReal elsewhere."""
    from harmonia.models.musx_bass import _parse_root
    lab = (lab or "").strip()
    if lab in ("", "N", "X", "N.C."):
        return None, "N"
    body = lab.split("/", 1)[0]
    if ":" in body:
        root_s, _, q = body.partition(":")
        return _parse_root(root_s), _HARTE_TO_IREAL.get(q, "")
    i = 1 + (1 if len(body) > 1 and body[1] in "#b" else 0)
    return _parse_root(body[:i]), body[i:]


def to_evs(chords: list[dict]) -> list:
    """Fresh chords -> the same Ev shape ``ug_score.our_sequence`` produces."""
    evs = []
    for c in chords:
        r, q = to_rq(c["label"])
        nc = (q == "N") or r is None or r < 0
        evs.append(UG.Ev(c["t0"], c["t1"], int(r or 0), "" if nc else q,
                         c["conf"], bool(nc)))
    return UG._collapse(evs)


KEYS = ("MISSED", "ADDED", "ROOT", "QUALITY")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slugs", nargs="*", default=None)
    ap.add_argument("--late-only", action="store_true", default=True)
    ap.add_argument("--tail-absorb", type=float, default=0.7)
    ap.add_argument("--margin", type=float, default=BRR.BASS_MARGIN)
    ap.add_argument("--out", default=str(SCRATCH / "bass_recovery_fresh.json"))
    a = ap.parse_args()

    slugs = a.slugs or sorted(p.stem[len("ug_score_"):]
                              for p in SCRATCH.glob("ug_score_*.json"))
    rows, detail = [], {}
    for slug in slugs:
        try:
            _ours_baked, ug, nm, t_intro, stored = BRR.rebuild(slug)
        except FileNotFoundError:
            continue
        try:
            chords = decode(slug)
        except (FileNotFoundError, SystemExit) as exc:
            rows.append((slug, None, None, 0, f"decode unavailable: {exc}"))
            continue
        ours = to_evs(chords)
        base = BRR.score_seq(slug, ours, ug, nm, t_intro)
        try:
            bass = BRR.Bass(slug)
        except FileNotFoundError:
            rows.append((slug, base["counts"], None, 0, "no NNLS cache"))
            continue
        new, fired = BRR.recover(ours, bass, margin=a.margin,
                                 late_only=a.late_only,
                                 tail_absorb=a.tail_absorb)
        after = BRR.score_seq(slug, new, ug, nm, t_intro)
        rows.append((slug, base["counts"], after["counts"], len(fired), ""))
        detail[slug] = {
            "n_chords_fresh": len(ours), "n_chords_baked": len(_ours_baked),
            "baked_counts": {k: stored["counts"][k] for k in KEYS},
            "fresh_counts": {k: base["counts"][k] for k in KEYS},
            "after_counts": {k: after["counts"][k] for k in KEYS},
            "agree_fresh": base["agreement_pct"],
            "agree_after": after["agreement_pct"], "fired": fired}

    print(f"\nFRESH decode (SHIPPED_CONFIG), margin {a.margin}x, "
          f"late-only={a.late_only}, tail-absorb={a.tail_absorb}\n")
    print(f"{'song':<46} {'chords':>13} {'fired':>5}  {'MISSED':>13} "
          f"{'ADDED':>13} {'ROOT':>11} {'QUAL':>11}")
    for slug, b, aft, nf, note in rows:
        if b is None or aft is None:
            print(f"{slug:<46} {note}")
            continue
        d = detail[slug]
        def f(k):
            s = f"{b[k]:3d} -> {aft[k]:3d}"
            return s + ("  " if aft[k] == b[k] else (" +" if aft[k] > b[k] else " -"))
        print(f"{slug:<46} {d['n_chords_baked']:5d}->{d['n_chords_fresh']:5d} "
              f"{nf:>5}  {f('MISSED'):>13} {f('ADDED'):>13} {f('ROOT'):>11} "
              f"{f('QUALITY'):>11}")
    good = [r for r in rows if r[1] and r[2]]
    if good:
        tot = {k: [sum(r[1][k] for r in good), sum(r[2][k] for r in good)]
               for k in KEYS}
        print("\n  TOTAL (fresh baseline -> fresh + recovery): " + "  ".join(
            f"{k} {v[0]}->{v[1]}" for k, v in tot.items()))
        print("  baked baseline for reference:              " + "  ".join(
            f"{k} {sum(detail[r[0]]['baked_counts'][k] for r in good)}"
            for k in KEYS))
    Path(a.out).write_text(json.dumps(detail, indent=1))


if __name__ == "__main__":
    main()
