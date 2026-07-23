"""Determinism gate for the v3 nnls24 chord-stage parity net (parity._nnls24_stages).

Covers the three additive golden blocks — ``nnls24_features``,
``nnls24_precoalesce``, ``nnls24_sections`` — that let a future Phase-3 chord
PORT be gated label-for-label (the ``[MED — safety-net GAP]`` known-issue).

They are captured DETERMINISTICALLY from CACHED features (stem-keyed nnls/musx
caches; a MISS records ``__needs_inference__`` instead of paying a cold decode).
This test asserts:

  * the three stages are PRESENT and non-marker for the two check songs;
  * they are byte-identical across two independent IN-PROCESS captures;
  * they are byte-identical across a fresh CROSS-PROCESS capture;
  * (red-first) the check would FAIL if the stages were absent — i.e. this test
    genuinely gates the extension, it is not vacuously green.

Two check songs (both have all caches on disk): ``bein_green`` and
``ben_e_king_stand_by_me_audio`` (task-specified). Each ``capture()`` re-runs the
beats stage (Beat This!, ~5-9 s) + a wav decode, so the captures are cached at
module scope to keep the run bounded.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from harmonia.eval import parity
from harmonia.eval.benchmark_set import LIVE_ORACLE_KWARGS, REPO, parity_songs

NNLS24_STAGES = ("nnls24_features", "nnls24_precoalesce", "nnls24_sections")
CHECK_SONGS = ("bein_green", "ben_e_king_stand_by_me_audio")
CACHE_DIR = REPO / "data" / "cache"
_MARKERS = ("__error__", "__missing__", "__needs_inference__")

# v5 bp48 relaxation (Louis 2026-07-24): the DEPRECATED bp48 head's Basic-Pitch
# (ONNX) feature ARRAYS drift ~1e-5/element across env/runs, so their sha256 +
# numeric stats false-positived the net.  v5 DROPS those float feature-array
# summaries and keeps only the STABLE bp48 labels/decisions + structural ints.
BP48_STAGES = ("bp48_features", "bp48_pooled", "bp48_beat_proba",
               "bp48_segments", "bp48_key")
# float feature-array fields that MUST now be absent (the relaxation)
_BP48_FLOAT_DROPPED = {
    "bp48_features": ("onsets", "activations", "frame_times"),
    "bp48_pooled": ("onset_b", "note_b"),
    "bp48_beat_proba": ("beat_proba", "mean_conf"),
}
# bp48 labels/decisions that MUST still gate byte-exact (the honesty property)
_BP48_LABELS = (
    ("bp48_beat_proba", "root_argmax"),  # per-beat root LABEL (list[int])
    ("bp48_segments", "n_segments"),     # segmentation decision (int)
    ("bp48_key", "key_name"),            # key LABEL (str)
)

# Cross-process runner: decode a FRESH wav + capture in an independent process,
# dump the capture JSON. Stem-keyed caches still hit on the fresh wav; the beats
# stage re-runs Beat This! — a genuine cross-process reproduction.
_SUBPROC_SRC = r"""
import json, sys
from pathlib import Path
from harmonia.eval import parity
from harmonia.eval.benchmark_set import LIVE_ORACLE_KWARGS, REPO, parity_songs
sid, wav_path, out_path = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
entry = next(s for s in parity_songs() if s["song_id"] == sid)
cap = parity.capture(entry, wav_path=wav_path, cache_dir=REPO / "data" / "cache",
                     oracle_kwargs=LIVE_ORACLE_KWARGS)
wav_path.unlink(missing_ok=True)
out_path.write_text(json.dumps(cap))
"""


def _capture_inprocess(entry: dict, wav_path: Path) -> dict:
    cap = parity.capture(entry, wav_path=wav_path, cache_dir=CACHE_DIR,
                         oracle_kwargs=LIVE_ORACLE_KWARGS)
    wav_path.unlink(missing_ok=True)
    return cap


def _capture_subprocess(sid: str, wav_path: Path, out_path: Path) -> dict:
    proc = subprocess.run(
        [sys.executable, "-c", _SUBPROC_SRC, sid, str(wav_path), str(out_path)],
        cwd=str(REPO), capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, f"subprocess capture failed:\n{proc.stderr[-2000:]}"
    return json.loads(out_path.read_text())


def _nnls24_only(cap: dict) -> dict:
    """A sub-capture holding only the three nnls24 stages, so parity.diff scopes
    to exactly the extension (existing stages excluded from this comparison)."""
    stages = cap.get("stages", {})
    return {"song_id": cap.get("song_id"),
            "stages": {k: stages[k] for k in NNLS24_STAGES if k in stages}}


def _assert_real_capture(cap: dict) -> None:
    """Guard so a determinism check can't pass VACUOUSLY on two identical markers:
    require the three stages present and free of any error/missing/needs-inference."""
    stages = cap.get("stages", {})
    for name in NNLS24_STAGES:
        block = stages.get(name)
        assert isinstance(block, dict) and not any(m in block for m in _MARKERS), \
            f"{cap.get('song_id')}: {name} is a marker/absent, not a real capture: {block}"
    assert "__sha256__" in stages["nnls24_features"]["feat"]


@pytest.fixture(scope="module")
def captures(tmp_path_factory) -> dict:
    """Per song: two independent in-process captures (A, B) + one cross-process
    capture (C). Computed once for the whole module (Beat This! is expensive).

    Each wav is named ``{sid}.wav`` (in its own subdir to avoid collision) so its
    STEM equals the song_id — the nnls/musx caches are stem-keyed, exactly as the
    parity CLI names its wavs; a differently-stemmed wav would miss and trip the
    __needs_inference__ guard."""
    tmp = tmp_path_factory.mktemp("nnls24_parity")
    out: dict = {}
    for sid in CHECK_SONGS:
        entry = next(s for s in parity_songs() if s["song_id"] == sid)
        assert entry["capturable"], f"{sid} not capturable (missing audio/pitch cache)"
        for d in ("a", "b", "c"):
            (tmp / d).mkdir(exist_ok=True)
        a = _capture_inprocess(entry, tmp / "a" / f"{sid}.wav")
        b = _capture_inprocess(entry, tmp / "b" / f"{sid}.wav")
        c = _capture_subprocess(sid, tmp / "c" / f"{sid}.wav", tmp / f"{sid}_c.json")
        out[sid] = {"a": a, "b": b, "c": c}
    return out


@pytest.mark.parametrize("sid", CHECK_SONGS)
def test_nnls24_stages_present_and_non_marker(captures, sid):
    """The three stages exist and are real captures (no error/missing/needs-inference
    marker) — the caches are present, so the deterministic path must be reached."""
    stages = captures[sid]["a"]["stages"]
    assert captures[sid]["a"]["schema_version"] == parity.CAPTURE_SCHEMA_VERSION
    for name in NNLS24_STAGES:
        assert name in stages, f"{sid}: stage {name!r} absent from capture"
        block = stages[name]
        assert isinstance(block, dict)
        for m in _MARKERS:
            assert m not in block, f"{sid}: {name} recorded marker {m}: {block.get(m)!r}"
    # load-bearing block has real content
    pc = stages["nnls24_precoalesce"]
    assert pc["n_segs"] >= 1 and len(pc["labels"]) == pc["n_segs"]
    assert pc["n_coalesced"] <= pc["n_segs"]
    assert "__sha256__" in stages["nnls24_features"]["feat"]
    assert isinstance(stages["nnls24_sections"]["barlocked_fired"], bool)


@pytest.mark.parametrize("sid", CHECK_SONGS)
def test_nnls24_stages_deterministic_inprocess(captures, sid):
    """Two independent in-process captures agree byte-for-byte on all three stages."""
    _assert_real_capture(captures[sid]["a"])
    _assert_real_capture(captures[sid]["b"])
    rep = parity.diff(_nnls24_only(captures[sid]["a"]), _nnls24_only(captures[sid]["b"]))
    assert rep["zero_divergence"], parity._diff_report_str(rep)


@pytest.mark.parametrize("sid", CHECK_SONGS)
def test_nnls24_stages_deterministic_crossprocess(captures, sid):
    """A fresh capture from an independent process agrees byte-for-byte."""
    _assert_real_capture(captures[sid]["a"])
    _assert_real_capture(captures[sid]["c"])
    rep = parity.diff(_nnls24_only(captures[sid]["a"]), _nnls24_only(captures[sid]["c"]))
    assert rep["zero_divergence"], parity._diff_report_str(rep)


def test_red_first_absent_capture_would_fail(captures):
    """Red-first guard: a capture WITHOUT the nnls24 stages (the pre-helper state)
    must fail the presence gate — proving these tests actually gate the extension."""
    cap = json.loads(json.dumps(captures[CHECK_SONGS[0]]["a"]))  # deep copy
    for name in NNLS24_STAGES:
        cap["stages"].pop(name, None)
    missing = [n for n in NNLS24_STAGES if n not in cap["stages"]]
    assert missing == list(NNLS24_STAGES), "expected all nnls24 stages absent"
    # and the diff vs the real capture flags them as removed stages (non-zero)
    rep = parity.diff(_nnls24_only(captures[CHECK_SONGS[0]]["a"]), _nnls24_only(cap))
    assert not rep["zero_divergence"]
    assert rep["total_divergences"] >= len(NNLS24_STAGES)


# ── v5 bp48 relaxation: drop drifting floats, keep labels gating ──────────────
def _bp48_only(cap: dict) -> dict:
    stages = cap.get("stages", {})
    return {"song_id": cap.get("song_id"),
            "stages": {k: stages[k] for k in BP48_STAGES if k in stages}}


@pytest.mark.parametrize("sid", CHECK_SONGS)
def test_bp48_feature_floats_dropped(captures, sid):
    """v5 relaxation: the drifting bp48 float feature-array summaries are NOT
    captured, so the net can no longer false-positive on their ~1e-5 ONNX
    env-drift; the stable structural ints + labels ARE kept."""
    stages = captures[sid]["a"]["stages"]
    for stage, dropped in _BP48_FLOAT_DROPPED.items():
        blk = stages[stage]
        for k in dropped:
            assert k not in blk, f"{sid}: {stage}.{k} must be dropped (v5) but is present"
        # no {shape,dtype,__sha256__,stats} array-summary survives in the feature stages
        assert "__sha256__" not in json.dumps(blk), \
            f"{sid}: {stage} still carries an array summary (should be float-free)"
    # kept stable fields (labels + structural ints)
    assert isinstance(stages["bp48_beat_proba"]["root_argmax"], list)
    assert "n_frames" in stages["bp48_features"] and "n_beats" in stages["bp48_pooled"]


@pytest.mark.parametrize("sid", CHECK_SONGS)
def test_bp48_stages_deterministic_inprocess(captures, sid):
    """With the drifting floats dropped, the remaining bp48 stages (labels +
    ints) are byte-identical across two independent captures — i.e. the
    relaxation actually removes the false-positive surface."""
    rep = parity.diff(_bp48_only(captures[sid]["a"]), _bp48_only(captures[sid]["b"]))
    assert rep["zero_divergence"], parity._diff_report_str(rep)


@pytest.mark.parametrize("stage,field", _BP48_LABELS)
def test_red_first_bp48_label_change_still_fails(captures, stage, field):
    """Red-first: dropping the bp48 FLOATS must NOT make bp48 vacuously green.
    A flipped bp48 LABEL/decision (root_argmax / bp48_segments / bp48_key) must
    still be caught by parity.diff — the honesty property the drop preserves."""
    base = _bp48_only(captures[CHECK_SONGS[0]]["a"])
    # sanity: unmutated diff is green (else the red-first proves nothing)
    assert parity.diff(base, json.loads(json.dumps(base)))["zero_divergence"]

    mutated = json.loads(json.dumps(base))  # deep copy
    blk = mutated["stages"][stage]
    val = blk[field]
    if isinstance(val, list):
        blk[field] = [(int(val[0]) + 1) % 12, *val[1:]] if val else [1]
    elif isinstance(val, bool):  # (guard: bool is a subclass of int)
        blk[field] = not val
    elif isinstance(val, int):
        blk[field] = val + 1
    else:  # str
        blk[field] = str(val) + "_X"

    rep = parity.diff(base, mutated)
    assert not rep["zero_divergence"], \
        f"label change on {stage}.{field} was NOT caught — bp48 gating is vacuous"
    assert rep["per_stage"][stage]["diverged"] >= 1
